# Copyright (c) 2025. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.

"""Plain flow-matching loss (unconditional generation).

This is exactly the *diagonal* (r = t) term of IsotropicFlowMapLoss, isolated as
its own objective so that "train a flow-matching model" is explicit rather than a
side effect of running the flow-map loss with data_proportion=1:

    z_t      = t * x + (1 - t) * eps           # linear / rectified-flow interpolant
    v_true   = x - eps                         # constant-velocity target
    u_pred   = net(z_t, t, t)                  # network velocity (two-time net on the diagonal)
    loss     = || u_pred - v_true ||^2

The network keeps its two-time flow-map signature net(x, alpha_r, alpha_t); we
simply always evaluate it on the diagonal alpha_r = alpha_t = t, so it learns a
single-time velocity field v(x, t). There is **no** off-diagonal / Lagrangian
self-distillation term — this is standard flow matching.
"""

import torch
from torch_utils import persistence
from training.loss_iflowmap import sample_interpolant


@persistence.persistent_class
class FlowMatchingLoss:
    def __init__(
        self,
        P_mean=-0.4,
        P_std=1.0,
        noise_dist='logit_normal',
        norm_p=1.0,
        norm_eps=1.0,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_dist = noise_dist
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    def _logit_normal_dist(self, shape, device):
        rnd_normal = torch.randn(shape, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def _uniform_dist(self, shape, device):
        return torch.rand(shape, device=device)

    def noise_distribution(self, shape, device):
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist(shape, device)
        elif self.noise_dist == 'uniform':
            return self._uniform_dist(shape, device)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def __call__(self, net, images):
        """
        Args:
            net:    network with signature net(x, alpha_r, alpha_t), evaluated on
                    the diagonal alpha_r = alpha_t = t.
            images: clean images [B, C, H, W]
        """
        x = images
        device = x.device
        shape = (x.shape[0], 1, 1, 1)

        t = self.noise_distribution(shape, device)
        z_t, alpha_t, _, v_true = sample_interpolant(x, t)

        net_module = net.module if hasattr(net, 'module') else net
        u_pred = net_module(z_t, alpha_t, alpha_t)
        loss_fm = (u_pred - v_true).pow(2).sum(dim=[1, 2, 3])

        # Same weighting convention as the diagonal branch of IsotropicFlowMapLoss:
        # learned per-sample logvar weighting if the net exposes calc_weight, else
        # a global adaptive normalisation.
        if hasattr(net_module, 'calc_weight'):
            logvar = net_module.calc_weight(alpha_t, alpha_t).view(x.shape[0])
            logvar = torch.clamp(logvar, -10.0, 10.0)
            loss = torch.exp(-logvar) * loss_fm + logvar
        else:
            unweighted_loss = loss_fm.mean()
            with torch.no_grad():
                adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
            loss = unweighted_loss * adaptive_weight

        return loss.mean()


def flow_matching_sample(net, batch_size, img_channels, img_resolution,
                         num_steps=50, device='cuda'):
    """Multi-step Euler integration of the flow-matching ODE dz/dt = v(z, t),
    from t=0 (pure noise) to t=1 (data), evaluating the velocity on the diagonal
    net(z, t, t).

    NB: this is NOT the flow-map sampler `isotropic_sample` (which queries the net
    off-diagonal at net(z, r, t)). A plain flow-matching model is only trained on
    the diagonal, so it must be integrated with many small Euler steps here.
    """
    z = torch.randn(batch_size, img_channels, img_resolution, img_resolution, device=device)
    t_values = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t  = t_values[i].view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)
        dt = (t_values[i + 1] - t_values[i])
        v  = net(z, t, t)
        z  = z + dt * v

    return z
