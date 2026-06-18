# Copyright (c) 2025, Weijian Luo. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Mean Flows for One-step Generative Modeling"."""

# NOTE: This file is a copy of the file of the EDM project.
#       It has been modified to fit the needs of the project.
#       The original file can be found at:
#       https://github.com/NVlabs/edm

import torch
from training.mask_generation import generate_random_box_mask
from torch_utils import persistence
import torch.nn.functional as F


def compute_alpha_t(t, mask):
    """
    Compute α_t from time and mask.

    Args:
        t: time parameter [B, 1, 1, 1], ranges from 0 (data) to 1 (noise)
        mask: binary mask [B, C, H, W], 1 = observed, 0 = masked

    Returns:
        α_t: [B, C, H, W]
    """
    # α_t = (1 - mask) * t
    # Observed (mask=1): α_t = 0 (always at data)
    # Masked (mask=0): α_t = t (goes from data to noise)
    alpha_t = (1 - mask) * t
    return alpha_t

def sample_interpolant(x, t, mask, eps=None):
    """
    Sample from the interpolant: z_t = (1 - α_t) ⊙ x + α_t ⊙ ε

    Args:
        x: clean data [B, C, H, W]
        t: time [B, 1, 1, 1], ranges from 0 (data) to 1 (noise)
        mask: binary mask [B, C, H, W], 1 = observed, 0 = masked
        eps: noise (optional, will sample if not provided)

    Returns:
        z_t: interpolated sample [B, C, H, W]
        alpha_t: interpolation coefficients [B, C, H, W]
        eps: noise [B, C, H, W]
        v_true: true velocity field [B, C, H, W]
    """
    if eps is None:
        eps = torch.randn_like(x)

    # Compute α_t = (1 - mask) * t
    alpha_t = compute_alpha_t(t, mask)

    # Interpolant: z_t = (1 - α_t) ⊙ x + α_t ⊙ ε
    # At t=0: z_0 = x (all data)
    # At t=1: z_1 = mask⊙x + (1-mask)⊙ε (observed=data, masked=noise)
    z_t = (1 - alpha_t) * x + alpha_t * eps

    # Velocity: v = (ε - x) * dα_t/dt = (ε - x) * (1 - mask)
    v_true = (1 - mask) * (eps - x)

    return z_t, alpha_t, eps, v_true


def compute_jacobian_correction(net, z_t, alpha_t, alpha_r, dalpha_dt, t, h, v_true):
    """
    Compute Jacobian correction terms using JVP.
    
    Args:
        net: neural network
        z_t: noisy sample [B, C, H, W]
        alpha_t: interpolation coefficients [B, C, H, W]
        dalpha_dt: time derivative of alpha [B, C, H, W]
        t: time [B, 1, 1, 1]
        h: auxiliary parameter [B, 1, 1, 1]
        v_true: true velocity [B, C, H, W]
    
    Returns:
        jvp_z: (∇_z u_θ) @ v_true
        jvp_alpha: (∇_{α} u_θ) @ dalpha_dt
    """
    # Take first channel of alpha_t for spatial conditioning
    alpha_spatial_t = alpha_t[:, :1, :, :]  # [B, 1, H, W]
    alpha_spatial_r = alpha_r[:, :1, :, :]  # [B, 1, H, W]

    # JVP w.r.t. z in direction v_true
    def u_wrapper_z(z):
        return net(z, alpha_spatial_t, alpha_spatial_r)
    _, jvp_z = torch.func.jvp(u_wrapper_z, (z_t,), (v_true,))
    
    # JVP w.r.t. alpha in direction dalpha_dt
    dalpha_spatial = dalpha_dt[:, :1, :, :]  # [B, 1, H, W]
    def u_wrapper_alpha(alpha):
        return net(z_t, alpha, alpha_spatial_r)
    _, jvp_alpha = torch.func.jvp(u_wrapper_alpha, (alpha_spatial_t,), (dalpha_spatial,))
    
    return jvp_z, jvp_alpha

@persistence.persistent_class
class MeanFlowLoss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, 
                 noise_dist='logit_normal', detach_tgt=True,
                 data_proportion=0.75, num_classes=None,
                 class_dropout_prob=0.1, norm_p=1.0, norm_eps=1.0,
                 guidance_eq='cfg', omega=1.0, kappa=0.5, t_start = 0.0, t_end = 1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.detach_tgt = detach_tgt
        self.data_proportion = data_proportion
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.guidance_eq = guidance_eq
        self.omega = omega
        self.kappa = kappa
        self.noise_dist = noise_dist
        self.t_start = t_start
        self.t_end = t_end

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

    def _apply_guidance(self, v, v_uncond, v_cond, t):
        omega = torch.where((t >= self.t_start) & (t <= self.t_end), 
                            self.omega, 1.0)
        
        if self.guidance_eq == 'cfg' and self.kappa == 0:
            return v_uncond + omega * (v - v_uncond)
        elif self.guidance_eq == 'cfg' and self.kappa > 0:
            kappa = torch.where((t >= self.t_start) & (t <= self.t_end), 
                                self.kappa, 0.0)
            return omega * v + (1 - omega - kappa) * v_uncond + kappa * v_cond
        else:
            return v

    def _cond_drop(self, labels, v, v_g):
        rand_mask = torch.rand(labels.shape[0], device=labels.device) < self.class_dropout_prob
        labels_drop = labels.clone()
        labels_drop[rand_mask] = self.num_classes
        v_g = torch.where(rand_mask[:, None, None, None], v, v_g)
        return labels_drop, v_g

    def __call__(self, net, images, labels=None, augment_pipe=None):
        x = images
        device = x.device
        batch_size = x.shape[0]
        shape = (batch_size, 1, 1, 1)

        t = self.noise_distribution(shape, device) # Sample t and r from noise distribution
        r = self.noise_distribution(shape, device)
        t, r = torch.max(t, r), torch.min(t, r)

        zero_mask = torch.arange(batch_size, device=device) < int(batch_size * self.data_proportion)
        zero_mask = zero_mask.view(shape)
        r = torch.where(zero_mask, t, r)  # Ensure t >= r and apply data proportion

        # Apply augmentations if needed
        if augment_pipe is not None:
            y, augment_labels = augment_pipe(x)
        else:
            y, augment_labels = x, None

        n = torch.randn_like(y) # Create noise and corrupted image
        z_t = (1 - t) * y + t * n
        v = n - y  # True velocity
        
        # Prepare labels for guidance
        labels_in = labels

        if labels is not None and self.class_dropout_prob > 0:
            with torch.no_grad():
                # Unconditional labels
                labels_null = torch.full_like(labels, self.num_classes)
                
                # Get conditional and unconditional velocities
                v_cond = net.module(z_t, t, class_labels=labels, h=torch.zeros_like(t), augment_labels=augment_labels)
                v_uncond = net.module(z_t, t, class_labels=labels_null, h=torch.zeros_like(t), augment_labels=augment_labels)
                
                # Apply guidance and conditional dropout
                v_g = self._apply_guidance(v, v_uncond, v_cond, t)
                labels_in, v_g = self._cond_drop(labels, v, v_g)
        else:
            v_g = v

        # Compute model output and time derivative
        def u_wrapper(z, t, r):
            return net.module(z, t, class_labels=labels_in, h=t-r, augment_labels=augment_labels)
        
        primals = (z_t, t, r)
        tangents = (v_g, torch.ones_like(t), torch.zeros_like(t))
        u, du_dt = torch.func.jvp(u_wrapper, primals, tangents)

        u_tgt = v_g - torch.clamp(t - r, min=0.0, max=1.0) * du_dt # Compute target velocity
        
        if self.detach_tgt:
            u_tgt = u_tgt.detach()

        unweighted_loss = (u - u_tgt).pow(2).sum(dim=[1, 2, 3]) # Adaptive loss weighting
        with torch.no_grad():
            adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
        
        loss = unweighted_loss * adaptive_weight
        return loss.sum()


@persistence.persistent_class
class MeanFlowLossV2:
    """Mean Flow loss using the IsotropicFlowMapLoss convention.

    Interpolant: z_t = t * x + (1 - t) * eps  (t=0 = noise, t=1 = data)
    True velocity: v = x - eps
    Network signature: net(z, alpha_r, alpha_t)  with alpha_r, alpha_t in [B, 1, 1, 1]

    Two-term loss:
      1. Diagonal (r=t):   train net(z_t, t, t) to predict v = x - eps.
      2. Off-diagonal:     mean-flow self-consistency via JVP w.r.t. alpha_t.

    Mean-flow condition:
      u_tgt = v − (alpha_t − alpha_r) · d/d(alpha_t)[u(z_{alpha_t}, alpha_r, alpha_t)]
    where the total derivative along the trajectory is:
      d/d(alpha_t) = ∂/∂(alpha_t) + (∂/∂z) · v
    """

    def __init__(
        self,
        P_mean=-0.4,
        P_std=1.0,
        noise_dist='logit_normal',
        data_proportion=0.75,
        norm_p=1.0,
        norm_eps=1.0,
        detach_tgt=True,
        jvp_fp32=False,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_dist = noise_dist
        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.detach_tgt = detach_tgt
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
            net:    network with signature net(z, alpha_r, alpha_t)
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
        # Sample t, build z_t = t*x + (1-t)*eps, train net(z_t, t, t) → v = x - eps
        t_diag   = self.noise_distribution(shape_diag, device)
        eps_diag = torch.randn_like(x[:batch_diag])
        z_t_diag = t_diag * x[:batch_diag] + (1 - t_diag) * eps_diag
        v_true   = x[:batch_diag] - eps_diag

        u_pred    = net_module(z_t_diag, t_diag, t_diag)
        loss_diag = (u_pred - v_true).pow(2).sum(dim=[1, 2, 3])

        if batch_offdiag == 0:
            if hasattr(net_module, 'calc_weight'):
                logvar = net_module.calc_weight(t_diag, t_diag).view(batch_diag)
                logvar = torch.clamp(logvar, -10.0, 10.0)
                loss = torch.exp(-logvar) * loss_diag + logvar
            else:
                unweighted_loss = loss_diag.mean()
                with torch.no_grad():
                    adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
                loss = unweighted_loss * adaptive_weight
            return loss.mean()

        # --- Off-diagonal mean-flow loss ---
        # Sample r <= t, build z_t, enforce mean-flow condition via JVP w.r.t. alpha_t
        t_offdiag = self.noise_distribution(shape_offdiag, device)
        r_offdiag = self.noise_distribution(shape_offdiag, device)
        r_offdiag, t_offdiag = torch.min(r_offdiag, t_offdiag), torch.max(r_offdiag, t_offdiag)

        alpha_r = r_offdiag.detach()
        alpha_t = t_offdiag.detach()

        eps_od = torch.randn_like(x[batch_diag:])
        z_t_od = (alpha_t * x[batch_diag:] + (1 - alpha_t) * eps_od).detach()
        v_od   = (x[batch_diag:] - eps_od).detach()

        if self.jvp_fp32:
            was_fp16 = getattr(net_module, 'use_fp16', False)
            net_module.use_fp16 = False
            z_t_od  = z_t_od.float()
            alpha_r = alpha_r.float()
            alpha_t = alpha_t.float()
            v_od    = v_od.float()

        def mean_flow_fn(z, alpha_t_in):
            """u(z_{alpha_t}, alpha_r, alpha_t) as a function of (z, alpha_t)."""
            return net_module(z, alpha_r, alpha_t_in)

        # JVP: total derivative of u w.r.t. alpha_t along the trajectory.
        # Tangents: dz/d(alpha_t) = v,  d(alpha_t)/d(alpha_t) = 1
        u, du_dalphat = torch.func.jvp(
            mean_flow_fn,
            (z_t_od, alpha_t),
            (v_od, torch.ones_like(alpha_t)),
        )

        u_tgt = v_od - (alpha_t - alpha_r) * du_dalphat

        if self.jvp_fp32:
            net_module.use_fp16 = was_fp16

        if self.detach_tgt:
            u_tgt = u_tgt.detach()

        loss_offdiag = (u - u_tgt).pow(2).sum(dim=[1, 2, 3])

        # --- Loss weighting ---
        if hasattr(net_module, 'calc_weight'):
            logvar_diag = net_module.calc_weight(t_diag, t_diag).view(batch_diag)
            logvar_od   = net_module.calc_weight(alpha_r, alpha_t).view(batch_offdiag)
            logvar_diag = torch.clamp(logvar_diag, -10.0, 10.0)
            logvar_od   = torch.clamp(logvar_od,   -10.0, 10.0)
            loss_diag_w = torch.exp(-logvar_diag) * loss_diag + logvar_diag
            loss_od_w   = torch.exp(-logvar_od)   * loss_offdiag + logvar_od
            loss = torch.cat([loss_diag_w, loss_od_w], dim=0)
        else:
            unweighted_loss = loss_diag.mean() + loss_offdiag.mean()
            with torch.no_grad():
                adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
            loss = unweighted_loss * adaptive_weight

        return loss.mean()