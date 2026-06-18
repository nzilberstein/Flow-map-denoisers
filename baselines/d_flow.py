import torch
import torch.nn.functional as F
import numpy as np
from time import perf_counter

try:
    from torchdiffeq import odeint_adjoint as odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class _CNF(torch.nn.Module):
    """Wraps AMFPrecond net as a CNF for torchdiffeq ODE integration (backward pass)."""

    def __init__(self, net, device):
        super().__init__()
        self.net = net
        self.device = device

    def forward(self, t, x):
        with torch.no_grad():
            t_4d = t.repeat(x.shape[0]).view(-1, 1, 1, 1)
            return self.net(x, t_4d, t_4d)


class D_FLOW:
    """
    D-Flow method for inverse problems (Ben-Hamu et al., 2024), adapted for AMFPrecond.

    Minimises ||H(T(z)) - y||^2 over the latent z, where T is the forward flow map,
    using L-BFGS with a Gaussian prior regulariser on z.

    Model API:       net(x, t_src, t_tgt) -> velocity  [B, C, H, W]
    Degradation API: degradation(x)           -> y      (callable forward)
                     degradation.transpose(y) -> x      (adjoint)
    """

    def __init__(self, net, device, loss_fn_lpips=None):
        """
        Args:
            net:            AMFPrecond model, already on device
            device:         torch device string, e.g. 'cuda:7'
            loss_fn_lpips:  optional LPIPS callable (e.g. loss_fn_alex from the notebook)
        """
        self.net = net
        self.device = device
        self.loss_fn_lpips = loss_fn_lpips

    def model_forward(self, x, t):
        """
        Wraps net(x, t, t) — no lookahead for this baseline.

        Args:
            x: [B, C, H, W]
            t: scalar float or 1-D tensor [B]
        Returns:
            velocity [B, C, H, W]
        """
        if not isinstance(t, torch.Tensor):
            t = torch.ones(x.shape[0], device=self.device) * t
        t_4d = t.view(-1, 1, 1, 1)
        return self.net(x, t_4d, t_4d)

    def _compute_psnr(self, img1, img2):
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return (20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))).item()

    def compute_norm(self, img):
        return torch.sqrt((img ** 2).sum([1, 2, 3]))

    def forward_flow_matching(self, z, steps=100, start_time=0.0):
        """
        Forward midpoint-Euler integration: maps latent z (t=0) to image x (t=1).
        """
        delta = (1 - start_time) / (steps - 1)
        for i in range(steps - 1):
            t1 = torch.ones(len(z), device=self.device) * (delta * i + start_time)
            z = z + delta * self.model_forward(
                z + delta / 2 * self.model_forward(z, t1),
                t1 + delta / 2,
            )
        return z

    def inverse_flow_matching(self, x):
        """
        Backward ODE integration: maps image x (t=1) to latent z (t=0).
        Requires torchdiffeq (pip install torchdiffeq).
        """
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "torchdiffeq is required for inverse_flow_matching. "
                "Install with: pip install torchdiffeq"
            )
        flow = _CNF(self.net, self.device)
        z_t = odeint(
            flow, x,
            torch.tensor([1.0, 0.0]).to(self.device),
            atol=1e-5, rtol=1e-5,
            method='dopri5',
        )
        return z_t[-1].detach()

    def solve_ip(self, clean_img, noisy_img, degradation,
                 steps_euler=100, start_time=0.0,
                 alpha=1.0, lmbda=1.0,
                 max_iter=10, lbfgs_iter=20,
                 compute_time=False, compute_memory=False):
        """
        Run D-Flow on a single batch of images.

        Args:
            clean_img:    ground-truth images [B, C, H, W]
            noisy_img:    degraded observation [B, C', H', W'], already on device
            degradation:  callable + .transpose (e.g. SuperResolutionOperator)
            steps_euler:  number of midpoint-Euler steps for forward_flow_matching
            start_time:   start time for the forward integration
            alpha:        blend weight for z initialisation (0=pure noise, 1=inverted)
            lmbda:        Gaussian prior regularisation weight
            max_iter:     number of L-BFGS outer iterations
            lbfgs_iter:   max_iter inside each L-BFGS call
            compute_time: if True, measure and return total wall time
            compute_memory: if True, track and return peak GPU memory

        Returns:
            result: dict with keys 'restored', 'psnr', and optionally
                    'lpips', 'time', 'memory'
        """
        torch.cuda.empty_cache()

        H     = degradation
        H_adj = degradation.transpose

        clean_img = clean_img.to(self.device)
        noisy_img = noisy_img.detach()
        d = clean_img.shape[1] * clean_img.shape[2] * clean_img.shape[3]

        # Initialise z by inverting the adjoint-initialised observation
        x = H_adj(noisy_img.clone())
        z = self.inverse_flow_matching(x)

        # Blend initialisation as in the D-Flow paper
        z = np.sqrt(alpha) * z + np.sqrt(1 - alpha) * torch.randn_like(z)
        z = z.detach().requires_grad_(True)

        optim_img = torch.optim.LBFGS(
            [z], max_iter=lbfgs_iter, history_size=100, line_search_fn='strong_wolfe'
        )

        if compute_time:
            torch.cuda.synchronize()
            time_total = 0.0

        if compute_memory:
            torch.cuda.reset_max_memory_allocated(self.device)

        for _ in range(max_iter):
            if compute_time:
                t0 = perf_counter()

            def closure():
                optim_img.zero_grad()
                norm_z = self.compute_norm(z)
                reg = (
                    0.5 * torch.clamp(norm_z ** 2, min=-1e6, max=1e6)
                    - (d - 1) * torch.log(norm_z + 1e-5)
                )
                loss = (
                    torch.sum(
                        (H(self.forward_flow_matching(
                            z, steps=steps_euler, start_time=start_time)
                         ) - noisy_img) ** 2,
                        dim=(1, 2, 3),
                    ) + lmbda * reg
                ).sum()
                loss.backward()
                return loss

            optim_img.step(closure)

            if compute_time:
                torch.cuda.synchronize()
                time_total += perf_counter() - t0

        restored = self.forward_flow_matching(
            z.detach(), steps=steps_euler, start_time=start_time
        )#.clamp(0, 1)

        result = {'restored': restored}
        result['psnr'] = self._compute_psnr(restored, clean_img)
        print(f"[D_FLOW] PSNR={result['psnr']:.2f} dB")

        if self.loss_fn_lpips is not None:
            with torch.no_grad():
                lpips_score = sum(
                    self.loss_fn_lpips(restored[i:i + 1], clean_img[i:i + 1]).item()
                    for i in range(restored.shape[0])
                )
            result['lpips'] = lpips_score / restored.shape[0]
            print(f"[D_FLOW] LPIPS={result['lpips']:.4f}")

        if compute_memory:
            result['memory'] = torch.cuda.max_memory_allocated(self.device)

        if compute_time:
            result['time'] = time_total
            print(f"[D_FLOW] time={time_total:.2f}s")

        torch.cuda.empty_cache()
        return result

    def run_method(self, img, noisy_img, degradation,
                   steps_euler=100, start_time=0.0,
                   alpha=1.0, lmbda=1.0,
                   max_iter=10, lbfgs_iter=10,
                   compute_time=False, compute_memory=False):
        """
        Run D-Flow on a pre-built image batch.

        Args:
            img:        clean images [B, C, H, W] — e.g. the notebook's `img`
            noisy_img:  degraded observation [B, C', H', W'], already on device
            degradation: degradation operator (callable + .transpose)
            (remaining args forwarded to solve_ip)

        Returns:
            result: dict with keys 'restored', 'psnr', and optionally
                    'lpips', 'time', 'memory'
        """
        return self.solve_ip(
            clean_img=img,
            noisy_img=noisy_img,
            degradation=degradation,
            steps_euler=steps_euler,
            start_time=start_time,
            alpha=alpha,
            lmbda=lmbda,
            max_iter=max_iter,
            lbfgs_iter=lbfgs_iter,
            compute_time=compute_time,
            compute_memory=compute_memory,
        )
