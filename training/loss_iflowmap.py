# Copyright (c) 2025, Weijian Luo. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions for Isotropic FlowMap (unconditional generation).

Isotropic version of AnisotropicFlowMapLoss with no spatial masking:
  - alpha_t = t, scalar per sample [B, 1, 1, 1] — never expanded to H×W
  - z_t = t * x + (1 - t) * eps
  - v_true = x - eps
  - Flow map: X_{r,t}(z_r) = z_r + (alpha_t - alpha_r) * v

Compared to the anisotropic version this avoids:
  - The Python loop in generate_random_box_mask
  - Materialising [B, 1, H, W] alpha tensors
  - Redundant (1-mask) multiplications (mask=0 everywhere)
"""

import torch
from torch_utils import persistence


def sample_interpolant(x, t, eps=None):
    """
    Sample from the isotropic interpolant: z_t = t * x + (1 - t) * eps

    Args:
        x: clean data [B, C, H, W]
        t: time [B, 1, 1, 1], scalar per batch element
        eps: noise (optional, sampled if not provided)

    Returns:
        z_t:   interpolated sample [B, C, H, W]
        t:     alpha (same as input, returned for consistency) [B, 1, 1, 1]
        eps:   noise used [B, C, H, W]
        drift: x - eps [B, C, H, W]  (= v_true)
    """
    if eps is None:
        eps = torch.randn_like(x)
    z_t = t * x + (1 - t) * eps
    drift = x - eps
    return z_t, t, eps, drift


@persistence.persistent_class
class IsotropicFlowMapLoss:
    """Isotropic FlowMap loss for unconditional generation.

    Two-term loss mirroring AnisotropicFlowMapLoss:
      1. Diagonal loss:       train net(z_t, alpha_t, alpha_t) to predict drift.
      2. Off-diagonal (LSD):  enforce the flow-map ODE via JVP w.r.t. alpha_t.

    Key difference from the anisotropic version: alpha_t stays as [B, 1, 1, 1]
    (scalar per sample) instead of being expanded to [B, 1, H, W], which saves
    memory and avoids large intermediate tensors inside the JVP.
    """

    def __init__(
        self,
        P_mean=-0.4,
        P_std=1.0,
        noise_dist='logit_normal',
        data_proportion=0.75,
        norm_p=1.0,
        norm_eps=1.0,
        jvp_fp32=False,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.noise_dist = noise_dist
        self.jvp_fp32 = jvp_fp32

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
            net:    network with signature net(x, alpha_r, alpha_t)
                    where alpha_r, alpha_t are scalar [B, 1, 1, 1] tensors
            images: clean images [B, C, H, W]
        """
        x = images
        device = x.device
        batch_size = x.shape[0]

        batch_diag    = int(batch_size * self.data_proportion)
        batch_offdiag = batch_size - batch_diag
        shape_diag    = (batch_diag,    1, 1, 1)
        shape_offdiag = (batch_offdiag, 1, 1, 1)

        net_module = net.module if hasattr(net, 'module') else net

        # --- Diagonal loss ---
        # Sample t ~ noise_dist, build z_t, predict drift at (z_t, alpha_t, alpha_t)
        t_diag = self.noise_distribution(shape_diag, device)
        z_t, alpha_t_diag, _, v_true = sample_interpolant(x[:batch_diag], t_diag)

        u_pred    = net_module(z_t, alpha_t_diag, alpha_t_diag)
        loss_diag = (u_pred - v_true).pow(2).sum(dim=[1, 2, 3])

        # --- Loss weighting (diagonal only, used during warmup or when batch_offdiag==0) ---
        if batch_offdiag == 0:
            if hasattr(net_module, 'calc_weight'):
                logvar_diag = net_module.calc_weight(alpha_t_diag, alpha_t_diag).view(batch_diag)
                logvar_diag = torch.clamp(logvar_diag, -10.0, 10.0)
                loss = torch.exp(-logvar_diag) * loss_diag + logvar_diag
            else:
                unweighted_loss = loss_diag.mean()
                with torch.no_grad():
                    adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
                loss = unweighted_loss * adaptive_weight
            return loss.mean()

        # --- Off-diagonal Lagrangian condition (LSD) loss ---
        # Sample r <= t, build z_r, enforce dX_{r,t}/d(alpha_t) = eta(X_{r,t}, alpha_t)
        t_offdiag = self.noise_distribution(shape_offdiag, device)
        r_offdiag = self.noise_distribution(shape_offdiag, device)
        r_offdiag, t_offdiag = torch.min(r_offdiag, t_offdiag), torch.max(r_offdiag, t_offdiag)

        alpha_r_offdiag = r_offdiag.detach()
        alpha_t_offdiag = t_offdiag.detach()

        z_r, _, _, _ = sample_interpolant(x[batch_diag:], r_offdiag)
        z_r_det = z_r.detach()

        # Tangent direction: d(alpha_t)/dt = 1 everywhere (no mask, fully isotropic)
        dalpha = torch.ones_like(alpha_t_offdiag)  # [B, 1, 1, 1]

        if self.jvp_fp32:
            # Cast off-diagonal batch to fp32 to avoid tangent overflow in fp16.
            # Only 25% of the batch is affected, so memory overhead is modest.
            was_fp16 = getattr(net_module, 'use_fp16', False)
            net_module.use_fp16 = False
            z_r_det      = z_r_det.float()
            alpha_r_offdiag = alpha_r_offdiag.float()
            alpha_t_offdiag = alpha_t_offdiag.float()
            dalpha       = dalpha.float()

        def flow_map_fn(alpha_t_in):
            """Flow map X_{r,t} as a function of TARGET alpha_t (scalar)."""
            v = net_module(z_r_det, alpha_r_offdiag, alpha_t_in)
            return z_r_det + (alpha_t_in - alpha_r_offdiag) * v

        # JVP differentiates the flow map w.r.t. the scalar alpha_t
        X_rt, dX_dalphat = torch.func.jvp(flow_map_fn, (alpha_t_offdiag,), (dalpha,))
        #!! CHECK !! Clamp X_rt to the training data range to prevent OOD inputs to eta_target,
        # which can produce NaN especially early in training (untrained velocity).
        # X_rt = X_rt.clamp(-5.0, 5.0)
        # !!!
        eta_target = net_module(X_rt, alpha_t_offdiag, alpha_t_offdiag)

        if self.jvp_fp32:
            net_module.use_fp16 = was_fp16

        # Lagrangian condition: dX/d(alpha_t) = eta  (dalpha=1, so no extra multiply)
        loss_lsd = (dX_dalphat - eta_target.detach()).pow(2).sum(dim=[1, 2, 3])

        # --- Loss weighting ---
        if hasattr(net_module, 'calc_weight'):
            logvar_diag = net_module.calc_weight(alpha_t_diag,    alpha_t_diag).view(batch_diag)
            logvar_lsd  = net_module.calc_weight(alpha_r_offdiag, alpha_t_offdiag).view(batch_offdiag)
            logvar_diag = torch.clamp(logvar_diag, -10.0, 10.0)
            logvar_lsd  = torch.clamp(logvar_lsd,  -10.0, 10.0)
            loss_diag_weighted = torch.exp(-logvar_diag) * loss_diag + logvar_diag
            loss_lsd_weighted  = torch.exp(-logvar_lsd)  * loss_lsd  + logvar_lsd
            loss = torch.cat([loss_diag_weighted, loss_lsd_weighted], dim=0)
        else:
            unweighted_loss = loss_diag.mean() + loss_lsd.mean()
            with torch.no_grad():
                adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
            loss = unweighted_loss * adaptive_weight

        return loss.mean()


@persistence.persistent_class
class IsotropicPSDLoss:
    """Isotropic Progressive Self-Distillation (PSD) loss.

    Two-term loss following Algorithm 4 (semigroup condition):
      1. Diagonal loss L_b:   train net(z_t, t, t) to predict true drift.
      2. Off-diagonal L_PSD:  enforce the semigroup condition via three network
                              evaluations — no JVP required.

    Semigroup condition (eq 15):
      v_{s,t}(I_s) ≈ (1-γ)·v_{s,u}(I_s) + γ·v_{u,t}(X_{s,u}(I_s))
    where
      u = γ·s + (1-γ)·t          (split point, γ ~ U[0,1])
      X_{s,u}(I_s) = I_s + (u-s)·v_{s,u}(I_s)   (flow-map step)
      γ ~ p_γ = U([0,1])
    """

    def __init__(
        self,
        noise_dist='uniform',
        data_proportion=0.75,
        norm_p=1.0,
        norm_eps=1.0,
        P_mean=-0.4,
        P_std=1.0,
    ):
        self.noise_dist = noise_dist
        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.P_mean = P_mean
        self.P_std = P_std

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
            net:    network with signature net(x, s, t)
                    where s, t are [B, 1, 1, 1] tensors
            images: clean images [B, C, H, W]  (x0 in flow-matching notation)
        """
        x = images
        device = x.device
        batch_size = x.shape[0]

        batch_diag    = int(batch_size * self.data_proportion)
        batch_offdiag = batch_size - batch_diag
        shape_diag    = (batch_diag,    1, 1, 1)
        shape_offdiag = (batch_offdiag, 1, 1, 1)

        net_module = net.module if hasattr(net, 'module') else net

        # --- Diagonal loss L_b ---
        # Sample t ~ noise_dist, build I_t = t*x0 + (1-t)*eps, predict drift x0-eps
        t_diag = self.noise_distribution(shape_diag, device)
        z_t, alpha_t_diag, _, v_true = sample_interpolant(x[:batch_diag], t_diag)

        u_pred    = net_module(z_t, alpha_t_diag, alpha_t_diag)
        loss_diag = (u_pred - v_true).pow(2).sum(dim=[1, 2, 3])

        # --- Warmup / diagonal-only path ---
        if batch_offdiag == 0:
            if hasattr(net_module, 'calc_weight'):
                logvar_diag = net_module.calc_weight(alpha_t_diag, alpha_t_diag).view(batch_diag)
                logvar_diag = torch.clamp(logvar_diag, -10.0, 10.0)
                loss = torch.exp(-logvar_diag) * loss_diag + logvar_diag
            else:
                unweighted = loss_diag.mean()
                with torch.no_grad():
                    w = 1 / (unweighted + self.norm_eps).pow(self.norm_p)
                loss = unweighted * w
            return loss.mean()

        # --- Off-diagonal PSD loss L_PSD ---
        # Sample (s, t) ~ U_od  (uniform over s < t)
        t1 = torch.rand(shape_offdiag, device=device)
        t2 = torch.rand(shape_offdiag, device=device)
        s = torch.min(t1, t2).detach()   # [B_od, 1, 1, 1]
        t = torch.max(t1, t2).detach()   # [B_od, 1, 1, 1]

        # Sample γ ~ U([0,1]), compute split point u = γ·s + (1-γ)·t
        gamma = torch.rand(shape_offdiag, device=device)          # [B_od, 1, 1, 1]
        u = (gamma * s + (1 - gamma) * t).detach()                # [B_od, 1, 1, 1]

        # Build interpolant I_s = s·x0 + (1-s)·eps
        I_s, _, _, _ = sample_interpolant(x[batch_diag:], s)
        I_s = I_s.detach()

        # Teacher — all three evaluations are stop-gradient
        with torch.no_grad():
            v_su = net_module(I_s, s, u)                          # v_{s,u}(I_s)
            X_su = I_s + (u - s) * v_su                          # X_{s,u}(I_s)
            v_ut = net_module(X_su, u, t)                         # v_{u,t}(X_{s,u})
            v_teacher = (1 - gamma) * v_su + gamma * v_ut        # composite teacher

        # Student — v_{s,t}(I_s), gradient flows here only
        v_student = net_module(I_s, s, t)

        # PSD residual and per-sample loss
        r = v_student - v_teacher
        loss_psd = r.pow(2).sum(dim=[1, 2, 3])

        # --- Loss weighting ---
        if hasattr(net_module, 'calc_weight'):
            logvar_diag = net_module.calc_weight(alpha_t_diag, alpha_t_diag).view(batch_diag)
            logvar_psd  = net_module.calc_weight(s, t).view(batch_offdiag)
            logvar_diag = torch.clamp(logvar_diag, -10.0, 10.0)
            logvar_psd  = torch.clamp(logvar_psd,  -10.0, 10.0)
            loss_diag_weighted = torch.exp(-logvar_diag) * loss_diag + logvar_diag
            loss_psd_weighted  = torch.exp(-logvar_psd)  * loss_psd  + logvar_psd
            loss = torch.cat([loss_diag_weighted, loss_psd_weighted], dim=0)
        else:
            unweighted = loss_diag.mean() + loss_psd.mean()
            with torch.no_grad():
                w = 1 / (unweighted + self.norm_eps).pow(self.norm_p)
            loss = unweighted * w

        return loss.mean()


@torch.no_grad()
def isotropic_sample(net, batch_size, img_channels, img_resolution, num_steps=1, device='cuda'):
    """
    Few-step (or one-step) unconditional sampling with the IsotropicFlowMap.

    Integrates from alpha=0 (pure noise) to alpha=1 (clean data) in num_steps
    equal steps using the flow map.

    Args:
        net:            trained network with signature net(x, alpha_r, alpha_t)
        batch_size:     number of samples to generate
        img_channels:   number of image channels
        img_resolution: spatial resolution
        num_steps:      number of integration steps (1 = one-step generation)
        device:         target device

    Returns:
        x_final: generated images [B, C, H, W]
    """
    z = torch.randn(batch_size, img_channels, img_resolution, img_resolution, device=device)
    t_values = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        alpha_r = t_values[i    ].view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)
        alpha_t = t_values[i + 1].view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)
        v_pred  = net(z, alpha_r, alpha_t)
        z = z + (alpha_t - alpha_r) * v_pred

    return z
