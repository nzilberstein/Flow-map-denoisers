import torch
import os
from time import perf_counter
from types import SimpleNamespace

from baselines.ot_ode import DegradationWrapper  # re-exported for convenience

try:
    import pnpflow.utils as _pnp_utils
except ImportError:
    _pnp_utils = None


def _hut_estimator(model_fn, x, t, n_samples=1):
    """Hutchinson trace estimator for div(model_fn) at (x, t).

    Returns a [B] tensor of per-sample trace estimates.
    """
    t_tensor = torch.ones(x.shape[0], device=x.device) * t
    trace = torch.zeros(x.shape[0], device=x.device)
    for _ in range(n_samples):
        v = torch.randn_like(x)
        with torch.enable_grad():
            x_r = x.detach().requires_grad_(True)
            f = model_fn(x_r, t_tensor)
            vjp = torch.autograd.grad((f * v).sum(), x_r)[0]
        trace = trace + (vjp * v).sum(dim=(1, 2, 3))
    return trace / n_samples


class FLOW_PRIORS(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = getattr(args, 'method', 'flow_priors')
        self.N = args.N

    def model_forward(self, x, t):
        """Calls net(x, t, t) — no lookahead."""
        if not isinstance(t, torch.Tensor):
            t = torch.ones(x.shape[0], device=self.device) * t
        t_4d = t.view(-1, 1, 1, 1)
        return self.model(x, t_4d, t_4d)

    def solve_ip(self, test_loader, degradation, sigma_noise):
        torch.cuda.empty_cache()
        self.args.sigma_noise = sigma_noise

        N = self.args.N
        K = self.args.K
        lmbda = self.args.lmbda
        eta = self.args.eta
        noise_type = getattr(self.args, 'noise_type', 'gaussian')

        H = degradation.H
        H_adj = degradation.H_adj

        loader = iter(test_loader)
        x_last = None
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch

            if noise_type == 'gaussian':
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif noise_type == 'laplace':
                noisy_img = H(clean_img.clone().to(self.device))
                noise = torch.distributions.Laplace(
                    torch.zeros_like(noisy_img),
                    sigma_noise * torch.ones_like(noisy_img),
                ).sample().to(self.device)
                noisy_img += noise
            else:
                raise ValueError(f'Unsupported noise_type: {noise_type!r}')

            clean_img = clean_img.to('cpu')

            x_init = torch.randn(clean_img.shape).to(self.device)
            x = x_init.clone()
            x.requires_grad_(True)

            start_time = getattr(self.args, 'start_time', 0.0)
            if start_time > 0.0:
                eps = start_time
                dt = (1 - eps) / N
            else:
                dt = 1.0 / N
                eps = 1e-3

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            for iteration in range(N):

                if self.args.compute_time:
                    time_counter_1 = perf_counter()

                num_t = iteration / N * (1 - eps) + eps
                t1 = torch.ones(len(x), device=self.device) * num_t
                t = t1.view(-1, 1, 1, 1)

                x = x.detach().clone()
                x.requires_grad = True
                optim_img = torch.optim.Adam([x], lr=eta)

                if iteration == 0:
                    for k in range(K):
                        x_next = x + self.model_forward(x, t1) * dt
                        y_next = (t + dt) * noisy_img + (1 - (t + dt)) * H(x_init)
                        trace_term = _hut_estimator(self.model_forward, x, num_t)
                        if noise_type == 'gaussian':
                            loss = lmbda * torch.sum(
                                (H(x_next) - y_next) ** 2, dim=(1, 2, 3))
                        else:
                            loss = lmbda * torch.sum(
                                torch.abs(H(x_next) - y_next), dim=(1, 2, 3))
                        loss = (loss + 0.5 * torch.sum(x ** 2, dim=(1, 2, 3))
                                + trace_term * dt).sum()
                        optim_img.zero_grad()
                        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
                        x.grad = grad
                        optim_img.step()
                else:
                    for k in range(K):
                        pred = self.model_forward(x, t1)
                        x_next = x + pred * dt
                        y_next = (t + dt) * noisy_img + (1 - (t + dt)) * H(x_init)
                        trace_term = _hut_estimator(self.model_forward, x, num_t)
                        if noise_type == 'gaussian':
                            loss = lmbda * torch.sum(
                                (H(x_next) - y_next) ** 2, dim=(1, 2, 3))
                        else:
                            loss = lmbda * torch.sum(
                                torch.abs(H(x_next) - y_next), dim=(1, 2, 3))
                        loss = (loss + trace_term * dt).sum()
                        optim_img.zero_grad()
                        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
                        grad_xt_lik = (
                            -1.0 / (1.0 - num_t) * (-x + num_t * pred.detach())
                        ).detach()
                        x.grad = grad + grad_xt_lik
                        optim_img.step()

                x = x + self.model_forward(x, t1) * dt

                if self.args.compute_time:
                    torch.cuda.synchronize()
                    time_counter_2 = perf_counter()
                    time_per_batch += time_counter_2 - time_counter_1

                torch.cuda.empty_cache()

            x_last = x.detach().clone()

            if self.args.compute_memory:
                dict_memory = {'batch': batch,
                               'max_allocated': torch.cuda.max_memory_allocated(self.device)}
                if _pnp_utils is not None:
                    _pnp_utils.save_memory_use(dict_memory, self.args)

            if self.args.compute_time:
                dict_time = {'batch': batch, 'time_per_batch': time_per_batch}
                if _pnp_utils is not None:
                    _pnp_utils.save_time_use(dict_time, self.args)

            if self.args.save_results and _pnp_utils is not None:
                restored_img = x_last.clone()
                _pnp_utils.save_images(clean_img, noisy_img, restored_img,
                                       self.args, H_adj, iter='final')
                _pnp_utils.compute_psnr(clean_img, noisy_img,
                                        restored_img, self.args, H_adj, iter='final')
                _pnp_utils.compute_ssim(clean_img, noisy_img,
                                        restored_img, self.args, H_adj, iter='final')
                _pnp_utils.compute_lpips(clean_img, noisy_img,
                                         restored_img, self.args, H_adj, iter='final')

            del clean_img, x
            torch.cuda.empty_cache()

        if self.args.save_results and _pnp_utils is not None:
            _pnp_utils.compute_average_psnr(self.args)
            _pnp_utils.compute_average_ssim(self.args)
            _pnp_utils.compute_average_lpips(self.args)
        if self.args.compute_memory and _pnp_utils is not None:
            _pnp_utils.compute_average_memory(self.args)
        if self.args.compute_time and _pnp_utils is not None:
            _pnp_utils.compute_average_time(self.args)

        return x_last, noisy_img

    def solve(self, img, degradation, sigma_noise):
        """Notebook-friendly entry point — no DataLoader needed.

        Args:
            img:          clean images tensor [B, C, H, W] (values in [-1, 1])
            degradation:  DegradationWrapper (or any object with .H / .H_adj)
            sigma_noise:  observation noise standard deviation

        Returns:
            x: restored tensor [B, C, H, W] on self.device
        """
        dataset = torch.utils.data.TensorDataset(
            img.cpu(), torch.zeros(img.shape[0]))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=img.shape[0], shuffle=False)
        self.args.max_batch = 1
        return self.solve_ip(loader, degradation, sigma_noise)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise):
        if _pnp_utils is None:
            raise RuntimeError("pnpflow is required for run_method")
        folder = _pnp_utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)
        os.makedirs(self.args.save_path_ip, exist_ok=True)
        self.solve_ip(data_loaders[self.args.eval_split], degradation, sigma_noise)


def make_args(N=100, K=1, lmbda=1.0, eta=1e-2,
              noise_type='gaussian', start_time=0.0,
              compute_time=False, compute_memory=False, save_results=False,
              **kwargs):
    """Build a minimal args namespace for notebook use.

    Args:
        N:            number of ODE steps
        K:            inner Adam optimisation steps per ODE step
        lmbda:        data-fidelity weight
        eta:          Adam learning rate
        noise_type:   'gaussian' or 'laplace'
        start_time:   ODE start time (0 = full trajectory)

    Example::

        args = make_args(N=100, K=3, lmbda=1.0, eta=1e-2)
        solver = FLOW_PRIORS(net, device, args)
        x = solver.solve(img, DegradationWrapper(blur_op), sigma_noise=1e-3)
    """
    ns = SimpleNamespace(
        N=N,
        K=K,
        lmbda=lmbda,
        eta=eta,
        noise_type=noise_type,
        start_time=start_time,
        compute_time=compute_time,
        compute_memory=compute_memory,
        save_results=save_results,
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns
