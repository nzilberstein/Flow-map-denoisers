# #!/usr/bin/env python
# """
# Estimate the operator-norm Lipschitz constant L_D(t) of the denoiser

#     D_{s,t}(x) = x + (1-s) * net(x, s, t)

# via alternating power iteration on the Jacobian J = dD_{s,t}/dx.

# For each noise level s and a grid of t in [s, 1]:
#   - Generates n_images noisy samples  x_s = s*x_1 + (1-s)*eps
#   - Runs n_steps of power iteration using forward-mode (JVP) and
#     reverse-mode (VJP) AD to estimate sigma_max(J) per image
#   - Reports max over all images as L_D(t)
#   - Plots L_D(t) vs t with the threshold line 1/L_F = 1/s

# Usage:
#     python lipschitz_est.py --model celeba
#     python lipschitz_est.py --model afhq --n_images 50 --n_steps 30
# """

# import argparse
# import os
# import sys

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# sys.path.insert(0, os.path.dirname(__file__))
# from PD_experiment.denoising_exp import MODEL_CONFIGS, load_model, load_dataset


# # ---------------------------------------------------------------------------
# # JVP / VJP primitives  (exact AD, called with B=1 to stay within memory)
# # ---------------------------------------------------------------------------

# def _jvp(f, x, u):
#     """Exact JVP via forward-mode AD (B=1 keeps peak memory to ~2× a single forward pass)."""
#     with torch.enable_grad():
#         _, Ju = torch.func.jvp(f, (x,), (u,))
#     return Ju


# def _vjp(f, x, v):
#     """VJP via standard reverse-mode autograd."""
#     x = x.detach().requires_grad_(True)
#     with torch.enable_grad():
#         out = f(x)
#         JTv = torch.autograd.grad(out, x, grad_outputs=v, create_graph=False)[0]
#     return JTv.detach()


# # ---------------------------------------------------------------------------
# # Power iteration  (B=1 per call to keep GPU memory bounded)
# # ---------------------------------------------------------------------------

# def sigma_max_power_iter(f, x_batch, n_steps: int = 30):
#     """
#     Estimate sigma_max(J) = ||dD/dx|| for each image in x_batch independently.
#     Processes one image at a time — exact forward-mode AD avoids the OOM that
#     occurs when running jvp over the full batch simultaneously.

#     Args:
#         f       : callable (1,C,H,W) -> (1,C,H,W), closed over s / t
#         x_batch : (B, C, H, W) evaluation points, detached
#         n_steps : power-iteration steps

#     Returns:
#         sigmas : list of float, one per image
#     """
#     sigmas = []
#     for xi in x_batch:
#         x = xi.unsqueeze(0).detach()         # (1, C, H, W)

#         u = torch.randn_like(x)
#         u = u / u.view(-1).norm().clamp(min=1e-12)

#         for _ in range(n_steps):
#             # v = J u  (exact JVP), normalise in output space
#             Ju = _jvp(f, x, u)
#             Ju_norm = Ju.view(-1).norm().clamp(min=1e-12)
#             v = Ju / Ju_norm

#             # u = J^T v  (VJP), normalise in input space
#             JTv = _vjp(f, x, v)
#             JTv_norm = JTv.view(-1).norm().clamp(min=1e-12)
#             u = JTv / JTv_norm

#         # Final sigma = ||J u||
#         Ju = _jvp(f, x, u)
#         sigmas.append(Ju.view(-1).norm().item())

#         torch.cuda.empty_cache()

#     return sigmas


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--model',    choices=list(MODEL_CONFIGS), default='celeba')
#     p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
#     p.add_argument('--seed',     type=int, default=42)
#     p.add_argument('--n_images', type=int, default=50,
#                    help='Number of noisy images per (s, t) pair')
#     p.add_argument('--n_steps',  type=int, default=30,
#                    help='Power-iteration steps')
#     p.add_argument('--n_t',      type=int, default=15,
#                    help='Number of t values per noise level')
#     p.add_argument('--output',   type=str, default='lipschitz_plot.pdf')
#     return p.parse_args()


# def main():
#     args = parse_args()
#     device = torch.device(args.device)

#     net, cfg = load_model(args.model, device)
#     net.eval()

#     # Load clean images (one batch is enough)
#     loader = load_dataset(cfg, batch_size=args.n_images)
#     clean, _ = next(loader)
#     clean = clean[:args.n_images].to(device)
#     print(f"Loaded {clean.shape[0]} clean images  [{cfg['data']} {cfg['img_size']}px]")

#     s_values = [0.3, 0.5, 0.7, 0.9]
#     colors   = ['#534AB7', '#0F6E56', '#D85A30', '#185FA5']
#     markers  = ['o', 's', 'D', '^']

#     fig, ax = plt.subplots(figsize=(6.0, 4.5))

#     for ci, s_val in enumerate(s_values):
#         torch.manual_seed(args.seed)
#         eps = torch.randn_like(clean)
#         x_s = (s_val * clean + (1.0 - s_val) * eps).detach()

#         t_grid  = np.linspace(s_val, 1.0, args.n_t)
#         L_curve = []

#         for t_val in t_grid:
#             # Capture s_val and t_val by value to avoid closure bugs
#             def f(x_in, _s=float(s_val), _t=float(t_val)):
#                 B = x_in.shape[0]
#                 s_t = x_in.new_full((B, 1, 1, 1), _s)
#                 t_t = x_in.new_full((B, 1, 1, 1), _t)
#                 return x_in + (1.0 - _s) * net(x_in, s_t, t_t)

#             sigmas = sigma_max_power_iter(f, x_s, n_steps=args.n_steps)
#             L_D    = max(sigmas)
#             L_curve.append(L_D)
#             print(f"  s={s_val:.1f}  t={t_val:.3f}  L_D={L_D:.4f}  "
#                   f"threshold 1/s={1/s_val:.3f}  {'OK' if L_D < 1/s_val else 'VIOLATION'}")

#         t_arr = np.array(t_grid)
#         L_arr = np.array(L_curve)

#         ax.plot(t_arr, L_arr, '-', color=colors[ci], linewidth=1.8, alpha=0.85)
#         step = max(1, len(t_arr) // 8)
#         ax.plot(t_arr[::step], L_arr[::step], markers[ci],
#                 color=colors[ci], markersize=6,
#                 markeredgecolor='k', markeredgewidth=0.5,
#                 label=f'$s={s_val}$')
#         ax.axhline(y=1.0 / s_val, color=colors[ci],
#                    linestyle='--', linewidth=0.9, alpha=0.55)

#     # Annotate threshold lines on the right margin
#     for ci, s_val in enumerate(s_values):
#         ax.annotate(f'$1/s$', xy=(1.0, 1.0 / s_val),
#                     xytext=(4, 0), textcoords='offset points',
#                     fontsize=7, color=colors[ci], va='center')

#     ax.set_xlabel('Lookahead $t$', fontsize=12)
#     ax.set_ylabel('$L_D(t) = \\|\\partial D_{s,t}/\\partial x\\|_2$', fontsize=12)
#     ax.set_title(f'Lipschitz constant of $D_{{s,t}}$  [{cfg["data"]}]', fontsize=12)
#     ax.legend(fontsize=10, loc='upper left')
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(args.output, bbox_inches='tight', dpi=150)
#     plt.close()
#     print(f"\nSaved {args.output}")


# if __name__ == '__main__':
#     main()


# ----------------


#!/usr/bin/env python
"""
Estimate the Lipschitz constant L_D(t) of the average denoiser

    D_{s,t}(x) = x + (1 - s) * net(x, s, t)

via alternating power iteration on the Jacobian J = dD/dx, for the
Gaussian-deblur setting.

Together with L_F = ||I - lambda * H^T H|| (computed analytically for the
Gaussian blur kernel), this lets us check the corrected contraction bound
from Section 4.1:

    L_T <= s * L_D(t) * L_F

The figure plots L_D(t), the threshold 1/(s * L_F), and the implied
composite bound s * L_D(t) * L_F vs the contraction line at 1.

Usage:
    python lipschitz_deblur.py --model celeba
    python lipschitz_deblur.py --model afhq --kernel_size 45 --std 3.0
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from PD_experiment.denoising_exp import MODEL_CONFIGS, load_model, load_dataset


# ---------------------------------------------------------------------------
# Forward operator (Gaussian blur, matches user's BlurKernel)
# ---------------------------------------------------------------------------

class BlurKernel(nn.Module):
    """Self-adjoint Gaussian blur via grouped conv with reflection pad."""
    def __init__(self, kernel_size=45, std=3.0, channels=3, device='cuda'):
        super().__init__()
        self.kernel_size = kernel_size
        n = np.zeros((kernel_size, kernel_size))
        n[kernel_size // 2, kernel_size // 2] = 1.0
        k = scipy.ndimage.gaussian_filter(n, sigma=std)
        k_t = torch.from_numpy(k).float().to(device)

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              stride=1, padding=0, bias=False,
                              groups=channels).to(device)
        with torch.no_grad():
            for f in self.conv.parameters():
                f.copy_(k_t.unsqueeze(0).unsqueeze(0).expand_as(f))
        for p in self.conv.parameters():
            p.requires_grad_(False)

        # Save the 1D Fourier magnitudes of the kernel for analytic L_F bound
        self.std_kernel = std
        self.kernel_size_kernel = kernel_size

    def forward(self, x):
        return self.conv(self.pad(x))

    def transpose(self, x):
        return self.forward(x)


def estimate_HtH_norm(H_op, shape, device, n_steps=30):
    """sigma_max(H^T H) via power iteration on the linear operator."""
    u = torch.randn(shape, device=device)
    u = u / u.view(-1).norm().clamp(min=1e-12)
    sigma = 0.0
    for _ in range(n_steps):
        with torch.no_grad():
            v = H_op.transpose(H_op(u))
        sigma = v.view(-1).norm().item()
        u = v / max(sigma, 1e-12)
    return sigma


# ---------------------------------------------------------------------------
# AD primitives
# ---------------------------------------------------------------------------

def _jvp(f, x, u):
    with torch.enable_grad():
        _, Ju = torch.func.jvp(f, (x,), (u,))
    return Ju


def _vjp(f, x, v):
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        out = f(x)
        JTv = torch.autograd.grad(out, x, grad_outputs=v, create_graph=False)[0]
    return JTv.detach()


# ---------------------------------------------------------------------------
# Power iteration for sigma_max(J) at noisy samples
# ---------------------------------------------------------------------------

def sigma_max_power_iter(f, x_batch, n_steps=30, tol=5e-3):
    """Estimate sigma_max(dD/dx) for each image, with early stopping."""
    sigmas = []
    for xi in x_batch:
        x = xi.unsqueeze(0).detach()

        u = torch.randn_like(x)
        u = u / u.view(-1).norm().clamp(min=1e-12)

        sigma_prev = 0.0
        for k in range(n_steps):
            Ju = _jvp(f, x, u)
            sigma = Ju.view(-1).norm().item()
            v = Ju / max(sigma, 1e-12)

            JTv = _vjp(f, x, v)
            u = JTv / JTv.view(-1).norm().clamp(min=1e-12)

            if k > 5 and abs(sigma - sigma_prev) / max(sigma, 1e-12) < tol:
                break
            sigma_prev = sigma

        Ju = _jvp(f, x, u)
        sigmas.append(Ju.view(-1).norm().item())

    return sigmas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',       choices=list(MODEL_CONFIGS), default='celeba')
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--n_images',    type=int, default=20,
                   help='Number of noisy samples per (s, t)')
    p.add_argument('--n_steps',     type=int, default=25,
                   help='Power iteration steps')
    p.add_argument('--n_t',         type=int, default=10,
                   help='Number of t values per s')
    p.add_argument('--kernel_size', type=int, default=45)
    p.add_argument('--std',         type=float, default=3.0)
    p.add_argument('--obs_noise',   type=float, default=0.05)
    p.add_argument('--lam',         type=float, default=1.0,
                   help='PnP step size lambda (assumes lam <= 1/||H^T H||)')
    p.add_argument('--output',      type=str, default='lipschitz_deblur.pdf')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    net, cfg = load_model(args.model, device)
    net.eval()

    loader = load_dataset(cfg, batch_size=args.n_images)
    clean, _ = next(loader)
    clean = clean[:args.n_images].to(device)
    print(f"Loaded {clean.shape[0]} clean images  [{cfg['data']} {cfg['img_size']}px]")

    # Forward model
    H_op = BlurKernel(kernel_size=args.kernel_size, std=args.std,
                      channels=cfg['channels'], device=device)

    # Compute ||H^T H|| analytically via power iteration on the operator
    HtH_norm = estimate_HtH_norm(H_op, clean.shape, device, n_steps=50)
    # L_F = ||I - lambda H^T H||.  For Gaussian blur, H^T H has eigenvalues in
    # [eig_min, HtH_norm].  For a symmetric kernel with reflection pad,
    # eig_min ~ (sum of kernel)^2 ~ 1 for a normalized Gaussian.  We use the
    # tighter bound L_F = max(|1 - lambda * HtH_norm|, |1 - lambda * eig_min|).
    # In practice eig_min is close to 0 for highly-blurring kernels, so a safe
    # upper bound is L_F = max(1, |1 - lambda * HtH_norm|).
    L_F = max(1.0, abs(1.0 - args.lam * HtH_norm))
    print(f"||H^T H|| ~ {HtH_norm:.4f}  =>  L_F = ||I - {args.lam} * H^T H|| <= {L_F:.4f}")

    s_values = [0.3, 0.5, 0.7, 0.9]
    colors   = ['#534AB7', '#0F6E56', '#D85A30', '#185FA5']
    markers  = ['o', 's', 'D', '^']

    # Two panels:
    #   (a) L_D(t)             -- raw denoiser Lipschitz
    #   (b) s * L_D(t) * L_F   -- composite contraction bound (Theorem 2)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    all_results = {}

    for ci, s_val in enumerate(s_values):
        torch.manual_seed(args.seed)
        eps = torch.randn_like(clean)
        x_s = (s_val * clean + (1.0 - s_val) * eps).detach()

        t_grid  = np.linspace(s_val, 1.0, args.n_t)
        L_curve = []

        for t_val in t_grid:
            def f(x_in, _s=float(s_val), _t=float(t_val)):
                B = x_in.shape[0]
                s_t = x_in.new_full((B, 1, 1, 1), _s)
                t_t = x_in.new_full((B, 1, 1, 1), _t)
                return x_in + (1.0 - _s) * net(x_in, s_t, t_t)

            sigmas = sigma_max_power_iter(f, x_s, n_steps=args.n_steps)
            L_D    = max(sigmas)         # max over images = worst-case L_D
            L_curve.append(L_D)
            sLDF = s_val * L_D * L_F
            ok = "OK " if sLDF < 1.0 else "VIOLATION"
            print(f"  s={s_val:.1f}  t={t_val:.3f}  "
                  f"L_D={L_D:.3f}  s*L_D*L_F={sLDF:.3f}  {ok}")

        all_results[s_val] = (np.array(t_grid), np.array(L_curve))

        t_arr = np.array(t_grid)
        L_arr = np.array(L_curve)

        # ---- Panel (a): L_D(t) and threshold 1/(s * L_F) ----
        threshold_a = 1.0 / (s_val * L_F)
        axes[0].plot(t_arr, L_arr, '-', color=colors[ci], lw=1.8, alpha=0.9)
        step = max(1, len(t_arr) // 8)
        axes[0].plot(t_arr[::step], L_arr[::step], markers[ci],
                     color=colors[ci], ms=6,
                     markeredgecolor='k', markeredgewidth=0.5,
                     label=f'$s={s_val}$')
        axes[0].axhline(threshold_a, color=colors[ci],
                        ls='--', lw=0.9, alpha=0.55)

        # ---- Panel (b): composite bound s * L_D(t) * L_F ----
        axes[1].plot(t_arr, s_val * L_arr * L_F, '-',
                     color=colors[ci], lw=1.8, alpha=0.9)
        axes[1].plot(t_arr[::step], (s_val * L_arr * L_F)[::step], markers[ci],
                     color=colors[ci], ms=6,
                     markeredgecolor='k', markeredgewidth=0.5,
                     label=f'$s={s_val}$')

    # Annotate threshold lines on panel (a)
    for ci, s_val in enumerate(s_values):
        threshold_a = 1.0 / (s_val * L_F)
        axes[0].annotate(r'$1/(s L_F)$', xy=(1.0, threshold_a),
                         xytext=(4, 0), textcoords='offset points',
                         fontsize=7, color=colors[ci], va='center')

    # Reference contraction line on panel (b)
    axes[1].axhline(1.0, color='red', ls='-', lw=1.2, alpha=0.8)
    axes[1].annotate(r'contraction ($=1$)', xy=(0.32, 1.0),
                     xytext=(0, 4), textcoords='offset points',
                     fontsize=8, color='red')

    axes[0].set_xlabel('Lookahead $t$', fontsize=11)
    axes[0].set_ylabel(r'$L_D(t)$', fontsize=11)
    axes[0].set_title(r'(a) Denoiser Lipschitz $L_D(t)$', fontsize=11)
    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel('Lookahead $t$', fontsize=11)
    axes[1].set_ylabel(r'$s \cdot L_D(t) \cdot L_F$', fontsize=11)
    axes[1].set_title(r'(b) Composite contraction bound', fontsize=11)
    axes[1].legend(fontsize=9, loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'Lipschitz analysis on Gaussian deblur — {cfg["data"]} '
                 f'(kernel={args.kernel_size}, std={args.std}, '
                 fr'$L_F$={L_F:.3f})', fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved {args.output}")

    # Save raw numbers
    out_npz = args.output.replace('.pdf', '.npz')
    flat = {'L_F': L_F, 'HtH_norm': HtH_norm, 'lam': args.lam,
            'kernel_size': args.kernel_size, 'std': args.std}
    for s_val, (t_arr, L_arr) in all_results.items():
        flat[f's{s_val:.1f}_t'] = t_arr
        flat[f's{s_val:.1f}_L_D'] = L_arr
    np.savez(out_npz, **flat)
    print(f"Saved {out_npz}")


if __name__ == '__main__':
    main()