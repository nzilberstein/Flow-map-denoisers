"""Diffusion Posterior Sampling (DPS) baseline for inverse problems.

Implements DPS (Chung et al., 2023) on top of an AMFPrecond flow-map net.
At each step we predict x_tweedie via the model, evaluate the data-fidelity
loss against the observation, and take a gradient step on the latent x.

Mirrors the API of FLOW_PRIORS / OT_ODE in this package: build an args
namespace via :func:`make_args`, instantiate :class:`DPS`, then call
:meth:`DPS.solve`.
"""

from types import SimpleNamespace

import torch


def _data_fidelity_loss(Hx, y, noise_model):
    """Negative log-likelihood (up to constants) for the chosen noise model.

    Both Hx and y are in [-1, 1].
    """
    if noise_model == "gaussian":
        return ((y - Hx) ** 2).sum()
    if noise_model == "poisson":
        Hx01 = ((Hx + 1) / 2).clamp(min=1e-6)
        y01 = ((y + 1) / 2).clamp(min=0.0)
        return (Hx01 - y01 * torch.log(Hx01)).sum()
    raise ValueError(f"Unknown noise_model: {noise_model!r}")


class DPS:
    """DPS sampler over a flow-map prior."""

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)

    def solve(self, img, noisy_img, degradation):
        """Run DPS given a clean batch and its precomputed observation.

        Args:
            img:        clean images [B, C, H, W] (only used for shape / device)
            noisy_img:  observation y on ``self.device`` (already noised)
            degradation: object with ``.H`` (forward operator)

        Returns:
            x: restored tensor [B, C, H, W] on ``self.device``
        """
        device = self.device
        H = degradation.H
        N = self.args.num_steps
        eta = self.args.eta
        noise_model = self.args.noise_model

        x = torch.randn_like(img).to(device)
        t_values = torch.linspace(0.0, 1.0, N + 1, device=device)

        for i in range(N):
            t = t_values[i].repeat(len(x))
            dt = 1 / N

            x = x.detach().requires_grad_(True)
            prior = self.model(x, t.view(-1, 1, 1, 1), t.view(-1, 1, 1, 1))
            x_tweedie = x + (1 - t.view(-1, 1, 1, 1)) * prior

            log_likelihood = -_data_fidelity_loss(H(x_tweedie), noisy_img, noise_model)
            lr_t = (eta / (2 * torch.sqrt(-log_likelihood))).detach()
            grad = torch.autograd.grad(log_likelihood, x)[0]

            x = x + dt * (lr_t * grad + prior)

        return x


def make_args(num_steps=100, eta=2000.0, noise_model="gaussian", **kwargs):
    """Build a minimal args namespace for :class:`DPS`.

    Args:
        num_steps:   number of ODE steps
        eta:         DPS step-size scale (``args.dps_eta`` upstream)
        noise_model: 'gaussian' or 'poisson'

    Example::

        args = make_args(num_steps=100, eta=2000.0)
        solver = DPS(net, device, args)
        x = solver.solve(img, noisy_img, DegradationWrapper(blur_op))
    """
    ns = SimpleNamespace(
        num_steps=num_steps,
        eta=eta,
        noise_model=noise_model,
    )
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns
