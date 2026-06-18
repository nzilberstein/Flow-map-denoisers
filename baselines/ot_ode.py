import torch
import os
from time import perf_counter
from types import SimpleNamespace

try:
    import utils
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


class DegradationWrapper:
    """Wraps a callable degradation operator into the H / H_adj interface.

    For self-adjoint operators (e.g. Gaussian blur) pass the same callable
    for both arguments.  For non-symmetric operators supply the adjoint
    separately.

    Example::

        deg = DegradationWrapper(blur_op)           # self-adjoint
        deg = DegradationWrapper(H_forward, H_adj)  # asymmetric
    """

    def __init__(self, H, H_adj=None):
        self.H = H
        self.H_adj = H_adj if H_adj is not None else H

    def __getattr__(self, name):
        # Forward unknown attribute lookups to the wrapped forward operator.
        # Guard against infinite recursion if H/H_adj are not yet set.
        if name in ('H', 'H_adj'):
            raise AttributeError(name)
        return getattr(self.H, name)


class OT_ODE(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = getattr(args, 'method', 'ot_ode')

    def model_forward(self, x, t):
        """Call net(x, t, t) — no lookahead."""
        if not isinstance(t, torch.Tensor):
            t = torch.ones(x.shape[0], device=self.device) * t
        t_4d = t.view(-1, 1, 1, 1)
        return self.model(x, t_4d, t_4d)

    def initialization(self, x, noisy_img, t0):
        """Return the adjoint initialisation unchanged.
        start_time only controls which loop iterations are skipped."""
        return t0 * x + (1-t0) * torch.randn_like(x)
        # return t0 * noisy_img + (1-t0) * torch.randn_like(x)

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):  
        self.args.sigma_noise = sigma_noise

        H = degradation.H
        H_adj = degradation.H_adj

        loader = iter(test_loader)
        x_last = None
        noisy_img_last = None
        for batch in range(self.args.max_batch):
            (clean_img, _labels) = next(loader)
            self.args.batch = batch

            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * sigma_noise
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to('cpu')

            # initialise the image with the adjoint operator
            x = H_adj(noisy_img.clone()).to(self.device)
            x = self.initialization(H_adj(noisy_img.clone()), noisy_img.clone(), self.args.start_time)

            steps, delta = self.args.steps_ode, 1 / self.args.steps_ode

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            for _count, iteration in enumerate(range(int(steps * self.args.start_time), int(steps))):

                if self.args.compute_time:
                    time_counter_1 = perf_counter()

                with torch.no_grad():
                    t1 = torch.ones(len(x), device=self.device) * delta * iteration
                    vt = self.model_forward(x, t1)
                    if torch.isnan(vt).any():
                        print(f"[OT_ODE] NaN in vt at iter={iteration}, t={t1[0].item():.3f}, x range=[{x.min().item():.3f},{x.max().item():.3f}]")
                    rt_squared = ((1 - t1) ** 2 / ((1 - t1) ** 2 + t1 ** 2)).view(-1, 1, 1, 1)
                    x1_hat = x + (1 - t1.view(-1, 1, 1, 1)) * vt

                    # Solve linear problem Cx = d
                    d = noisy_img - H(x1_hat)
                    if torch.isnan(d).any():
                        print(f"[OT_ODE] NaN in d at iter={iteration}")
                    sol = torch.zeros_like(d)

                    problem = getattr(self.args, 'problem', 'inpainting')

                    if problem in ("inpainting", "random_inpainting", "paintbrush_inpainting"):
                        for i in range(d.shape[0]):
                            sol_tmp = 1 / (H(torch.ones_like(x))[i] * rt_squared[i] + sigma_noise ** 2) * d[i]
                            sol[i] = sol_tmp.reshape(d[i].shape)

                    elif problem == "denoising":
                        for i in range(d.shape[0]):
                            sol_tmp = d[i] / (rt_squared[i] + sigma_noise ** 2)
                            sol[i] = sol_tmp.reshape(d[i].shape)

                    elif problem == "superresolution":
                        rt_squared_scalar = torch.tensor(
                            (1 - delta * iteration) ** 2 / ((1 - delta * iteration) ** 2 + (delta * iteration) ** 2)
                        )
                        s = degradation.scale_factor
                        hht_diag = 1.0 / (s ** 2)
                        sol_tmp = 1 / (hht_diag * rt_squared_scalar + torch.tensor(sigma_noise) ** 2)
                        sol_tmp = sol_tmp * d.flatten(start_dim=2)
                        sol = sol_tmp.reshape(d.shape)

                    elif problem == "gaussian_deblurring_FFT":
                        H_size, W_size = d.shape[-2], d.shape[-1]
                        fft_d = torch.fft.fft2(d)
                        # Wrap kernel to corners for correct circular-convolution FFT
                        k = degradation.filter.to(self.device).float().squeeze()  # [K, K]
                        K = k.shape[0]
                        half = K // 2
                        kernel_padded = torch.zeros(1, 1, H_size, W_size, device=self.device)
                        kernel_padded[0, 0, :half+1, :half+1] = k[half:, half:]
                        kernel_padded[0, 0, -half:,  :half+1] = k[:half, half:]
                        kernel_padded[0, 0, :half+1, -half:]  = k[half:, :half]
                        kernel_padded[0, 0, -half:,  -half:]  = k[:half, :half]
                        fft_kernel = torch.fft.fft2(kernel_padded)           # [1,1,H,W] complex
                        h_fft_sq = (fft_kernel * torch.conj(fft_kernel)).real  # [1,1,H,W] float32
                        inv = (rt_squared * h_fft_sq + sigma_noise ** 2).to(fft_d.dtype)
                        # Compute vec = H† (HH†rt² + σ²I)⁻¹ d directly in Fourier space.
                        # Applying H_adj spatially to a Wiener-filtered sol would reintroduce
                        # the reflection-vs-circular mismatch, amplifying zeroed frequencies.
                        sol = None
                        vec = torch.fft.ifft2(torch.conj(fft_kernel) * fft_d / inv).real
                        if torch.isnan(vec).any():
                            print(f"[OT_ODE] NaN in vec (FFT) at iter={iteration}")

                    else:
                        # General case: GMRES
                        if not HAS_UTILS:
                            raise RuntimeError(
                                f"Problem '{problem}' requires utils.GMRES but the utils module is not available."
                            )
                        for i in range(d.shape[0]):
                            def C_ope(z):
                                z = z.reshape(noisy_img.shape[1:]).unsqueeze(0)
                                return (rt_squared[i].unsqueeze(0) * H(H_adj(z)) + sigma_noise ** 2 * z).reshape(-1)
                            sol_tmp, _ = utils.GMRES(C_ope, d[i].reshape(-1), max_iter=100)
                            sol[i] = sol_tmp.reshape(d[i].shape)

                    if sol is not None:
                        vec = H_adj(sol)
                    if torch.isnan(vec).any():
                        print(f"[OT_ODE] NaN in vec at iter={iteration}")

                # VJP correction
                t = t1.view(-1, 1, 1, 1)
                gamma_mode = getattr(self.args, 'gamma', 'constant')
                if gamma_mode == 'constant':
                    gamma = 1
                elif gamma_mode == 'gamma_t':
                    gamma = torch.sqrt(t / (t ** 2 + (1 - t) ** 2))

                with torch.enable_grad():
                    x_r = x.detach().requires_grad_(True)
                    f_x = self.model_forward(x_r, t1)
                    g = torch.autograd.grad(
                        outputs=f_x, inputs=x_r,
                        grad_outputs=vec.detach(),
                        create_graph=False, allow_unused=True,
                    )[0]
                if g is None:
                    g = torch.zeros_like(x)
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                if torch.isnan(g).any():
                    print(f"[OT_ODE] NaN in VJP g at iter={iteration}")

                with torch.no_grad():
                    g = vec + (1 - t1.view(-1, 1, 1, 1)) * g
                    ratio = (1 - t1.view(-1, 1, 1, 1)) / t1.view(-1, 1, 1, 1).clamp(min=1e-8)
                    v_adapted = vt + ratio * gamma * g
                    x = x + delta * v_adapted
                    if torch.isnan(x).any():
                        print(f"[OT_ODE] NaN in x at iter={iteration}, ratio={ratio.mean().item():.2f}, |g|={g.abs().max().item():.3e}")

                    if self.args.save_results and HAS_UTILS:
                        if iteration % 10 == 0 or self.should_save_image(iteration, steps):
                            restored_img = x.detach().clone()
                            utils.compute_psnr(clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_ssim(clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                            utils.compute_lpips(clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)

                if self.args.compute_time:
                    torch.cuda.synchronize()
                    time_counter_2 = perf_counter()
                    time_per_batch += time_counter_2 - time_counter_1

            x_last = x.detach().clone()
            noisy_img_last = noisy_img

        return x_last, noisy_img_last

    def solve(self, img, degradation, sigma_noise):
        """Notebook-friendly entry point — no DataLoader needed.

        Args:
            img:          clean images tensor [B, C, H, W] (values in [-1, 1])
            degradation:  DegradationWrapper (or any object with .H / .H_adj)
            sigma_noise:  observation noise standard deviation

        Returns:
            (x, noisy_img): restored tensor and degraded observation, both on self.device
        """
        dataset = torch.utils.data.TensorDataset(img.cpu(), torch.zeros(img.shape[0]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=img.shape[0], shuffle=False)
        self.args.max_batch = 1
        return self.solve_ip(loader, degradation, sigma_noise)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise):
        if not HAS_UTILS:
            raise RuntimeError("run_method requires the utils module.")
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)
        os.makedirs(self.args.save_path_ip, exist_ok=True)
        self.solve_ip(data_loaders[self.args.eval_split], degradation, sigma_noise)


def make_args(steps_ode=100, problem='inpainting', gamma='constant', start_time=0.0,
              compute_time=False, compute_memory=False, save_results=False,
              **kwargs):
    """Build a minimal args namespace for OT_ODE.

    Args:
        steps_ode:    number of ODE steps
        problem:      inverse problem type — 'inpainting', 'random_inpainting',
                      'denoising', 'superresolution', 'gaussian_deblurring_FFT',
                      or any string (falls back to GMRES)
        gamma:        VJP scaling — 'constant' (=1) or 'gamma_t'
        start_time:   ODE warm-start time in [0, 1)

    Example::

        args = make_args(steps_ode=100, problem='gaussian_deblurring_FFT')
        solver = OT_ODE(net, device, args)
        x, noisy = solver.solve(img, DegradationWrapper(blur_op), sigma_noise=1e-3)
    """
    ns = SimpleNamespace(
        steps_ode=steps_ode,
        problem=problem,
        gamma=gamma,
        start_time=start_time,
        compute_time=compute_time,
        compute_memory=compute_memory,
        save_results=save_results,
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns
