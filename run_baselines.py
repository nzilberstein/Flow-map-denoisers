#!/usr/bin/env python
"""
Run baselines for inverse problems using AMFPrecond checkpoints.

Usage:
    python script.py --model celeba --method ot_ode --degradation deblurring --num_batches 5
    python script.py --model celeba --method all --degradation inpainting --num_batches 10 --batch_size 8
"""

import argparse
import os
import pickle
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

import lpips

from baselines.d_flow import D_FLOW
from baselines.dps import DPS
from baselines.dps import make_args as dps_make_args
from baselines.flow_priors import FLOW_PRIORS
from baselines.flow_priors import make_args as fp_make_args
from baselines.ot_ode import DegradationWrapper, OT_ODE
from baselines.ot_ode import make_args as ot_make_args
from utils.degradations import (
    BlurKernel,
    Decolorize,
    DiffJPEGOperator,
    JPEGOperator,
    MotionBlurOperator,
    SuperResolutionOperator,
    PhaseRetrievalOperator,
    generate_box_mask,
    generate_random_mask,
)

# ---------------------------------------------------------------------------
# Checkpoint / dataset configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "cifar10": {
        "checkpoint": (
            "/home/nvidia/flow_maps/logs/aniso_flow_map/"
            "lsd_correct_partition/00024-cifar10-32x32-uncond-ddpmpp-mf-"
            "gpus4-batch512-fp32/network-snapshot-100000.pkl"
        ),
        "img_size": 32,
        "channels": 3,
        "data": "cifar10",
    },
    "celeba": {
        "checkpoint": (
            "/home/nvidia/flow_maps/logs/aniso_flow_map/"
            "lsd/00054-celeba-128x128-uncond-ddpmpp-mf-gpus8-batch128-fp32/"
            "network-snapshot-087584.pkl"
        ),
        "img_size": 128,
        "channels": 3,
        "data": "celeba",
    },
    "afhq": {
        "checkpoint": (
            "/home/nvidia/flow_maps/logs/aniso_flow_map/"
            # "lsd/00055-afhq-cat-192x192-uncond-ddpmpp-mf-gpus8-batch64-fp32/"
            # "lsd/00058-afhq-256x256-uncond-ddpmpp-mf-gpus8-batch32-fp32/"
            "flow_matching_model/00003-afhq-256x256-uncond-ddpmpp-mf-gpus4-batch32-fp32/"
            # "network-snapshot-017505.pkl"
            "network-snapshot-015004.pkl"
        ),
        "img_size": 256,
        "channels": 3,
        "data": "afhq",
    },
    "afhq-cat": {
        "checkpoint": (
            "/home/nvidia/flow_maps/logs/aniso_flow_map/"
            "lsd/00055-afhq-cat-192x192-uncond-ddpmpp-mf-gpus8-batch64-fp32/"
            "network-snapshot-035033.pkl"
        ),
        "img_size": 192,
        "channels": 3,
        "data": "afhq-cat",
    },
}

ALL_METHODS = ["ot_ode", "flow_priors", "d_flow", "dps", "pnp_flow"]
ALL_DEGRADATIONS = ["phase_retrieval", "colorization", "jpeg", "inpainting_box", "inpainting_random", "deblurring_motion", "deblurring_gaussian", "super_resolution"]


# ---------------------------------------------------------------------------
# Utility classes
# ---------------------------------------------------------------------------

class NextDataLoader(torch.utils.data.DataLoader):
    def __next__(self):
        try:
            return next(self.iterator)
        except (StopIteration, AttributeError):
            self.iterator = iter(self)
            return next(self.iterator)


class CelebAZip(torch.utils.data.Dataset):
    def __init__(self, zip_path, partition_file, split="test", transform=None):
        self.zip_path = zip_path
        self.transform = transform
        split_map = {"train": 0, "val": 1, "test": 2}
        split_id = split_map[split]
        with open(partition_file) as f:
            entries = [line.strip().split() for line in f]
        valid = {e[0].replace(".jpg", ".png") for e in entries if int(e[1]) == split_id}
        with zipfile.ZipFile(zip_path) as z:
            self.names = sorted(n for n in z.namelist() if n in valid)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open(self.names[idx]) as f:
                img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def grid(array, ncols=8):
    array = np.pad(array, [(0,0),(1,1),(1,1),(0,0)], 'constant')
    nindex, height, width, intensity = array.shape
    ncols = min(nindex, ncols)
    nrows = (nindex+ncols-1)//ncols
    r = nrows*ncols - nindex
    arr = np.concatenate([array]+[np.zeros([1,height,width,intensity])]*r)
    result = (arr.reshape(nrows, ncols, height, width, intensity)
            .swapaxes(1,2)
            .reshape(height*nrows, width*ncols, intensity))
    return np.pad(result, [(1,1),(1,1),(0,0)], 'constant')

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(img1, img2):
    """Both inputs should be in [0, 1]."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf"), 0.0
    psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
    return psnr.item(), mse.item()


def compute_lpips(loss_fn, x_, img, device):
    score = 0.0
    for i in range(x_.shape[0]):
        score += loss_fn(x_[i:i+1].to(device), img[i:i+1].to(device)).item()
    return score / x_.shape[0]


# ---------------------------------------------------------------------------
# Model / dataset loading
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    cfg = MODEL_CONFIGS[model_name]
    print(f"Loading checkpoint: {cfg['checkpoint']}")
    with open(cfg["checkpoint"], "rb") as f:
        checkpoint = pickle.load(f)
    net = checkpoint["ema"]
    net.eval().to(device)
    return net, cfg


def load_dataset(cfg, batch_size):
    img_size = cfg["img_size"]
    data = cfg["data"]

    if data == "cifar10":
        dataset = CIFAR10(
            "data", train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        )
    elif data == "celeba":
        dataset = CelebAZip(
            zip_path="/home/nvidia/flow_maps/data/celeba/celeba-128x128.zip",
            partition_file="/home/nvidia/flow_maps/data/celeba/list_eval_partition.txt",
            split="test",
            transform=transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        )
    elif data == "afhq":
        from training.dataset import ImageFolderDataset

        base = ImageFolderDataset(
            path="/home/nvidia/flow_maps/data/afhq-256x256.zip",
            split="test",
            train_ratio=0.9,
        )
        class _AFHQWrapper(torch.utils.data.Dataset):
            def __len__(self):
                return len(base)

            def __getitem__(self, idx):
                img_np, label = base[idx]
                return torch.from_numpy(img_np).float() / 127.5 - 1, label

        dataset = _AFHQWrapper()
    elif data == "afhq-cat":
        from training.dataset import ImageFolderDataset

        base = ImageFolderDataset(
            path="/home/nvidia/flow_maps/data/afhq-cat-192x192.zip",
            split="test",
            train_ratio=0.9,
        )
        class _AFHQCatWrapper(torch.utils.data.Dataset):
            def __len__(self):
                return len(base)

            def __getitem__(self, idx):
                img_np, label = base[idx]
                return torch.from_numpy(img_np).float() / 127.5 - 1, label

        dataset = _AFHQCatWrapper()
    else:
        raise ValueError(f"Unknown dataset: {data}")

    loader = NextDataLoader(
        dataset, batch_size, num_workers=1, prefetch_factor=2,
        pin_memory=True, shuffle=False,
    )
    return loader


def build_degradation(args, cfg, batch_size, device):
    img_size = cfg["img_size"]
    channels = cfg["channels"]

    if args.degradation == "inpainting_box":
        box_size = args.box_size
        raw_mask = generate_box_mask(
            batch_size, 1, img_size, img_size, box_size, device
        ).repeat(1, channels, 1, 1)  # [B, C, H, W]
        # For inpainting the degradation operator is elementwise multiplication
        # by the mask.  We wrap it in a DegradationWrapper-compatible object.
        class InpaintingOp:
            def __init__(self, mask):
                self.mask = mask
                self.transpose = self  # self-adjoint

            def __call__(self, x):
                return self.mask * x

        op = InpaintingOp(raw_mask)
        deg = DegradationWrapper(op)
        return deg, raw_mask
    elif args.degradation == "inpainting_random":
        mask_ratio = args.mask_ratio
        raw_mask = generate_random_mask(batch_size, 1, img_size, img_size,
                                         mask_ratio=mask_ratio, device=device).repeat(1, channels, 1, 1)  # [B, C, H, W]
        
        class InpaintingOp:
            def __init__(self, mask):
                self.mask = mask
                self.transpose = self  # self-adjoint

            def __call__(self, x):
                return self.mask * x
            
        op = InpaintingOp(raw_mask)
        deg = DegradationWrapper(op)
        return deg, raw_mask

    elif args.degradation == "deblurring_gaussian":
        blur = BlurKernel(
            blur_type='gaussian',
            kernel_size=args.blur_kernel_size,
            std=args.blur_std,
            channels=channels,
        ).to(device)
        deg = DegradationWrapper(blur, blur)  # self-adjoint
        return deg, None
    elif args.degradation == "deblurring_motion":
        blur = MotionBlurOperator(
            kernel_size=args.blur_kernel_size,
            intensity=args.blur_std,
            device=device,
        ).to(device)
        deg = DegradationWrapper(blur, blur)  # self-adjoint
        return deg, None
    
    elif args.degradation == "super_resolution":
        sr_op = SuperResolutionOperator(
            in_shape=(batch_size, channels, img_size, img_size),
            scale_factor=args.sr_factor,
            device=device,
        )
        deg = DegradationWrapper(sr_op, sr_op.transpose)
        return deg, None
    elif args.degradation == "colorization":
        decolor = Decolorize(channels=channels, device=device)
        deg = DegradationWrapper(decolor, decolor.transpose)
        return deg, None
    elif args.degradation == "phase_retrieval":
        pr_op = PhaseRetrievalOperator(oversample=2.0, device=device, resolution=img_size)
        deg = DegradationWrapper(pr_op, pr_op)
        return deg, None
    elif args.degradation == "jpeg":
        # Mirror PnP-CM: differentiable JPEG inside the optimization loop, plain
        # JPEG_ArtifactRemoval (with .round()) for the measurement y.
        jpeg_diff = DiffJPEGOperator(qf=args.jpeg_qf, ste=args.jpeg_ste, device=device)
        jpeg_meas = JPEGOperator(qf=args.jpeg_qf, device=device)
        deg = DegradationWrapper(jpeg_diff, jpeg_diff)
        deg.H_meas = jpeg_meas
        return deg, None
    else:
        raise ValueError(f"Unknown degradation: {args.degradation}")


# ---------------------------------------------------------------------------
# Individual method runners
# ---------------------------------------------------------------------------

_DEGRADATION_TO_PROBLEM = {
    "inpainting_box": "inpainting",
    "inpainting_random": "inpainting",
    "deblurring_gaussian": "gaussian_deblurring_FFT",
    "deblurring_motion": "gaussian_deblurring_FFT",
    "super_resolution": "superresolution",
    "colorization": "colorization",
    "phase_retrieval": "phase_retrieval",

}

def run_ot_ode(net, img, noisy_img, deg, args, cfg, device, loss_fn):
    problem = _DEGRADATION_TO_PROBLEM.get(args.degradation, args.degradation)
    solver_args = ot_make_args(
        steps_ode=args.num_steps,
        problem=problem,
        gamma=args.ot_gamma,
        start_time=args.ot_start_time,
    )
    solver = OT_ODE(net, device, solver_args)
    x, noisy_img = solver.solve(img, deg, sigma_noise=args.sigma_noise)
    return _metrics(x, img, loss_fn, device), {"restored": x, "noisy": noisy_img}


def run_flow_priors(net, img, noisy_img, deg, args, cfg, device, loss_fn):
    fp_args = fp_make_args(
        N=args.num_steps,
        K=args.fp_K,
        lmbda=args.fp_lmbda,
        eta=args.fp_eta,
    )
    solver = FLOW_PRIORS(net, device, fp_args)
    x, noisy_img = solver.solve(img, deg, sigma_noise=args.sigma_noise)
    return _metrics(x, img, loss_fn, device), {"restored": x, "noisy": noisy_img}


def run_d_flow(net, img, noisy_img, deg, args, cfg, device, loss_fn):
    solver = D_FLOW(net, device, loss_fn_lpips=loss_fn)
    # D_FLOW needs the noisy image directly
    H = deg.H
    with torch.no_grad():
        noisy = H(img.to(device))
        noisy = noisy + args.sigma_noise * torch.randn_like(noisy)
    result = solver.solve_ip(
        clean_img=img,
        noisy_img=noisy,
        degradation=H,
        steps_euler=args.num_steps,
        lmbda=args.df_lmbda,
        max_iter=args.df_max_iter,
        alpha=args.alpha_dflow,
    )
    x = result["restored"]
    x_ = x.clamp(0, 1)
    img_01 = ((img.to(device) + 1) / 2).clamp(0, 1)
    psnr, mse = compute_psnr(x_, img_01)
    lpips_score = compute_lpips(loss_fn, x.detach(), img.detach(), device) if loss_fn else None
    return {"psnr": psnr, "mse": mse, "lpips": lpips_score}, {"restored": x, "noisy": noisy}


def run_dps(net, img, noisy_img, deg, args, cfg, device, loss_fn):
    dps_args = dps_make_args(
        num_steps=args.num_steps,
        eta=args.dps_eta,
        noise_model=args.noise_model,
    )
    solver = DPS(net, device, dps_args)
    x = solver.solve(img, noisy_img, deg)
    return _metrics(x, img, loss_fn, device), {"restored": x, "noisy": noisy_img}

def _apply_observation_noise(y_clean, args):
    """Add observation noise to a clean measurement in [-1, 1]."""
    if args.noise_model == "gaussian":
        return y_clean + args.sigma_noise * torch.randn_like(y_clean)
    if args.noise_model == "poisson":
        y01 = ((y_clean + 1) / 2).clamp(0.0, 1.0)
        return (torch.poisson(args.peak * y01) / args.peak) * 2 - 1
    raise ValueError(f"Unknown noise_model: {args.noise_model}")


def _data_fidelity_loss(Hx, y, args):
    """Negative log-likelihood (up to constants) for the chosen noise model.

    Used by PnP-Flow and DPS so the data-fidelity gradient matches the noise.
    Both Hx and y are in [-1, 1].
    """
    if args.noise_model == "gaussian":
        return ((y - Hx) ** 2).sum()
    if args.noise_model == "poisson":
        Hx01 = ((Hx + 1) / 2).clamp(min=1e-6)
        y01 = ((y + 1) / 2).clamp(min=0.0)
        return (Hx01 - y01 * torch.log(Hx01)).sum()
    raise ValueError(f"Unknown noise_model: {args.noise_model}")


def _metrics(x, img, loss_fn, device):
    """x is the restored image in [-1, 1]."""
    x_ = ((x.detach() + 1) / 2).clamp(0, 1)
    img_01 = ((img.to(device) + 1) / 2).clamp(0, 1)
    return _metrics_from_01(x_, img_01, x.detach(), img.to(device), loss_fn, device)


def _metrics_from_01(x_, img_01, x_unnorm, img_unnorm, loss_fn, device):
    psnr, mse = compute_psnr(x_, img_01)
    lpips_score = compute_lpips(loss_fn, x_unnorm, img_unnorm, device) if loss_fn else None
    return {"psnr": psnr, "mse": mse, "lpips": lpips_score}


def run_pnp_flow(net, img, noisy_img, deg, args, cfg, device, loss_fn):
    """PnP flow map with fixed lookahead (from pnp_samples.ipynb)."""
    channels = cfg["channels"]
    img_size = cfg["img_size"]
    batch_size = img.shape[0]

    H = deg.H
    x = torch.zeros((batch_size, channels, img_size, img_size), device=device)
    N = args.num_steps

    save_every = getattr(args, "pnp_save_every", 0)
    intermediates = {} if save_every > 0 else None

    for i in range(N):
        t = torch.ones(batch_size, device=device) * (i / N)
        t_target = torch.clamp(t + args.lookahead, max=1.0)
        lr_t = (1 - t).view(-1, 1, 1, 1) ** args.alpha

        # Data fidelity step
        if args.pnp_optimizer == "adam":
            lr_scalar = float((1.0 - i / N) ** args.alpha)
            x_var = x.detach().clone()
            x_var.requires_grad = True
            optim_img = torch.optim.Adam([x_var], lr=args.pnp_adam_lr * lr_scalar)
            for _ in range(args.pnp_adam_steps):
                loss = _data_fidelity_loss(H(x_var), noisy_img, args)
                optim_img.zero_grad()
                grad = torch.autograd.grad(loss, x_var, create_graph=False)[0]
                x_var.grad = grad
                optim_img.step()
            z = x_var.detach()
        else:
            x_leaf = x.detach().requires_grad_(True)
            loss = _data_fidelity_loss(H(x_leaf), noisy_img, args)
            grad = torch.autograd.grad(loss, x_leaf)[0]
            z = x_leaf.detach() - args.gain * lr_t * grad.detach()

        # Interpolation + denoising with lookahead
        with torch.no_grad():
            z_tilde = t.view(-1, 1, 1, 1) * z + \
                torch.randn_like(x) * (1 - t.view(-1, 1, 1, 1))
            x = z_tilde + (1 - t.view(-1, 1, 1, 1)) * \
                net(z_tilde, t.view(-1, 1, 1, 1), t_target.view(-1, 1, 1, 1))
            
            # # Define SNR  t / 1-t
            # snr = t / (1 - t + 1e-8)
            # snr = snr.view(-1, 1, 1, 1) ** -1
            # x = snr * x + (1 - snr) * z

        if intermediates is not None and ((i + 1) % save_every == 0 or i == N - 1):
            intermediates[i + 1] = x.detach().cpu().clone()

    out = {"restored": x, "noisy": noisy_img}
    if intermediates is not None:
        out["intermediates"] = intermediates
    return _metrics(x, img, loss_fn, device), out


METHOD_RUNNERS = {
    "ot_ode": run_ot_ode,
    "flow_priors": run_flow_priors,
    "d_flow": run_d_flow,
    "dps": run_dps,
    "pnp_flow": run_pnp_flow,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run inverse-problem baselines")

    # Core
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="celeba",
                   help="Which pretrained model / dataset to use")
    p.add_argument("--method", default="ot_ode",
                   help=f"Baseline method(s). One of {ALL_METHODS} or 'all'")
    p.add_argument("--degradation", choices=ALL_DEGRADATIONS, default="deblurring",
                   help="Inverse problem type")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Batching / averaging
    p.add_argument("--num_batches", type=int, default=20,
                   help="Number of batches to average metrics over")
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # Shared solver params
    p.add_argument("--num_steps", type=int, default=100,
                   help="Number of ODE steps (N)")
    p.add_argument("--lookahead", type=float, default=0.0,
                   help="Lookahead for OT-ODE")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Learning-rate exponent for OT-ODE")
    p.add_argument("--gain", type=float, default=1.0,
                   help="Data-fidelity gradient gain for PnP-flow")
    p.add_argument("--pnp_optimizer", choices=["gd", "adam"], default="gd",
                   help="Optimizer for PnP-flow data fidelity step")
    p.add_argument("--pnp_adam_lr", type=float, default=1e-2,
                   help="Base Adam LR for PnP-flow data step (scaled by (1-t)^alpha)")
    p.add_argument("--pnp_adam_steps", type=int, default=1,
                   help="Inner Adam steps per outer ODE step (PnP-flow)")
    p.add_argument("--pnp_save_every", type=int, default=0,
                   help="Save PnP-flow intermediate images every K steps (0 = disabled). "
                        "Snapshots include the final step.")
    p.add_argument("--sigma_noise", type=float, default=1e-3,
                   help="Observation noise std (Gaussian model)")
    p.add_argument("--noise_model", choices=["gaussian", "poisson"], default="gaussian",
                   help="Observation noise model. `poisson` samples y ~ Poisson(peak * y01)/peak "
                        "and switches PnP/DPS data-fidelity to the Poisson NLL.")
    p.add_argument("--peak", type=float, default=50.0,
                   help="Poisson rate (photon count). Lower = noisier. Used when --noise_model poisson.")

    # Degradation-specific
    p.add_argument("--box_size", type=int, default=50,
                   help="Inpainting box size (pixels)")
    p.add_argument("--mask_ratio", type=float, default=0.9,
                   help="Random inpainting mask ratio (proportion of pixels to mask)")
    p.add_argument("--blur_kernel_size", type=int, default=45)
    p.add_argument("--blur_std", type=float, default=3.0)
    p.add_argument("--sr_factor", type=int, default=4,
                   help="Super-resolution downscale factor")
    p.add_argument("--jpeg_qf", type=int, default=5,
                   help="JPEG quality factor (lower = more compression). Image size must be divisible by 16.")
    p.add_argument("--jpeg_ste", action="store_true",
                   help="Use straight-through-estimator variant of Diff-JPEG instead of polynomial soft rounding.")

    # Method-specific: OT_ODE
    p.add_argument("--ot_gamma", default="constant", choices=["constant", "gamma_t"],
                   help="VJP scaling mode for OT_ODE")
    p.add_argument("--ot_start_time", type=float, default=0.1,
                   help="ODE warm-start time for OT_ODE (paper tunes t0 in {0.1,0.2,0.3,0.4})")

    # Method-specific: FLOW_PRIORS
    p.add_argument("--fp_K", type=int, default=1,
                   help="Inner Adam steps per ODE step (FLOW_PRIORS)")
    p.add_argument("--fp_lmbda", type=float, default=1.0e4,
                   help="Data-fidelity weight (FLOW_PRIORS)")
    p.add_argument("--fp_eta", type=float, default=1e-2,
                   help="Adam LR (FLOW_PRIORS)")

    # Method-specific: D_FLOW
    p.add_argument("--df_lmbda", type=float, default=1.0,
                   help="Prior regularisation weight (D_FLOW)")
    p.add_argument("--df_max_iter", type=int, default=10,
                   help="L-BFGS outer iterations (D_FLOW)")
    p.add_argument("--alpha_dflow", type=float, default=0.1,
                   help="Blending factor for D_FLOW")

    # Method-specific: DPS
    p.add_argument("--dps_eta", type=float, default=2000.0,
                   help="DPS step size")

    # Misc
    p.add_argument("--no_lpips", action="store_true",
                   help="Skip LPIPS computation (faster)")
    p.add_argument("--save_as_grid", action="store_true",
                   help="Save example results as a grid of images instead of individual files")
    p.add_argument("--custom_name_folder", type=str, default=None,
                   help="Directory to save results")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    methods = ALL_METHODS if args.method == "all" else [args.method]

    # Validate method names
    for m in methods:
        if m not in METHOD_RUNNERS:
            raise ValueError(f"Unknown method '{m}'. Choose from {ALL_METHODS} or 'all'.")

    # Load model and dataset
    net, cfg = load_model(args.model, device)
    loader = load_dataset(cfg, args.batch_size)

    # LPIPS
    loss_fn = None
    if not args.no_lpips:
        loss_fn = lpips.LPIPS(net="alex").to(device)

    print(f"\nModel   : {args.model}")
    print(f"Methods : {methods}")
    print(f"Degrad. : {args.degradation}")
    print(f"Batches : {args.num_batches}")
    print(f"Steps   : {args.num_steps}\n")

    # Accumulators: {method: {metric: [values]}}
    accum = {m: {"psnr": [], "mse": [], "lpips": []} for m in methods}

    for batch_idx in range(args.num_batches):
        print(f"--- Batch {batch_idx + 1}/{args.num_batches} ---")

        torch.manual_seed(args.seed + batch_idx)
        np.random.seed(args.seed + batch_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + batch_idx)

        img = next(loader)[0]  # [B, C, H, W]

        # Build the degradation operator (may depend on batch_size)
        deg, _ = build_degradation(args, cfg, img.shape[0], device)

        # Apply degradation to get noisy_img (needed by some methods).
        # If a non-differentiable measurement operator is attached (e.g. JPEG),
        # use it for y so the actual measurement isn't a Diff-JPEG approximation.
        H = deg.H
        H_meas = getattr(deg, "H_meas", H)
        with torch.no_grad():
            noisy_img = H_meas(img.to(device))
        noisy_img = _apply_observation_noise(noisy_img, args)

        for method in methods:
            print(f"  Running {method}...", end=" ", flush=True)
            torch.manual_seed(args.seed + batch_idx)
            np.random.seed(args.seed + batch_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + batch_idx)
            runner = METHOD_RUNNERS[method]
            try:
                with _nullctx():
                    import time
                    start_time = time.time()
                    metrics, results = runner(net, img, noisy_img, deg, args, cfg, device, loss_fn)
                    elapsed = time.time() - start_time
                print(f"done in {elapsed:.1f}s. Metrics: ", end="")
                accum[method]["psnr"].append(metrics["psnr"])
                accum[method]["mse"].append(metrics["mse"])
                if metrics["lpips"] is not None:
                    accum[method]["lpips"].append(metrics["lpips"])
                print(
                    f"PSNR={metrics['psnr']:.2f}dB  "
                    f"MSE={metrics['mse']:.5f}"
                    + (f"  LPIPS={metrics['lpips']:.4f}" if metrics["lpips"] is not None else "")
                )
                # Save example results in generated samples and noisy images, where each folder contains the info of the method, degradation, and batch index.
                if method == "pnp_flow":  # Only save results for pnp_flow to avoid clutter
                    save_path = f"generated_samples_flow_matching/{args.model}_{method}_{args.lookahead}_{args.degradation}_{args.num_steps}_{args.sigma_noise}"
                else:
                    save_path = f"generated_samples_flow_matching/{args.model}_{method}_{args.degradation}"
                if args.custom_name_folder is not None:
                    save_path = save_path + f"_{args.custom_name_folder}"
                    print(f"Custom save path: {save_path}")

                # If save path does not exist, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # for key, img_tensor in results.items():
                if args.save_as_grid == True:
                    img_np = torch.clamp((results['restored'].detach().cpu() + 1)/2, 0, 1).permute(0,2,3,1)
                    if img_np.shape[-1] == 1:
                        img_np = img_np.repeat(1, 1, 1, 3)
                    img_grid = grid(img_np)
                    plt.imshow(img_grid)
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, f"batch_{batch_idx + 1}_restored.pdf"), bbox_inches='tight', pad_inches=0)

                    noisy_img_np = torch.clamp((results['noisy'].detach().cpu() + 1)/2, 0, 1).permute(0,2,3,1)
                    if noisy_img_np.shape[-1] == 1:
                        noisy_img_np = noisy_img_np.repeat(1, 1, 1, 3)
                    noisy_img_grid = grid(noisy_img_np)
                    plt.imshow(noisy_img_grid)
                    plt.axis('off')
                    plt.savefig(os.path.join(save_path, f"batch_{batch_idx + 1}_noisy.pdf"), bbox_inches='tight', pad_inches=0)
                else:
                    for key, img_tensor in results.items():
                        if key == "intermediates":
                            continue
                        img_np = torch.clamp((img_tensor.detach().cpu() + 1)/2, 0, 1).permute(0,2,3,1).numpy()
                        if img_np.shape[-1] == 1:
                            img_np = np.repeat(img_np, 3, axis=-1)
                        for i in range(img_np.shape[0]):
                            plt.imsave(os.path.join(save_path, f"batch_{batch_idx + 1}_{key}_{i}.png"), img_np[i])

                if "intermediates" in results:
                    steps_sorted = sorted(results["intermediates"].keys())
                    snaps = [results["intermediates"][s] for s in steps_sorted]  # each [B,C,H,W]
                    snaps = torch.stack(snaps, dim=1)  # [B, S, C, H, W]
                    B, S, C, Hs, Ws = snaps.shape

                    # Match measurement and ground-truth to snapshot resolution
                    meas = results["noisy"].detach().cpu()
                    gt = img.detach().cpu()
                    if meas.shape[-2:] != (Hs, Ws):
                        meas = torch.nn.functional.interpolate(meas, size=(Hs, Ws), mode="nearest")
                    if gt.shape[-2:] != (Hs, Ws):
                        gt = torch.nn.functional.interpolate(gt, size=(Hs, Ws), mode="nearest")
                    if meas.shape[1] != C:
                        meas = meas.repeat(1, C // meas.shape[1], 1, 1)
                    if gt.shape[1] != C:
                        gt = gt.repeat(1, C // gt.shape[1], 1, 1)

                    snaps_full = torch.cat([meas.unsqueeze(1), snaps, gt.unsqueeze(1)], dim=1)
                    S_full = snaps_full.shape[1]
                    snaps_np = torch.clamp((snaps_full + 1) / 2, 0, 1).permute(0, 1, 3, 4, 2).numpy()
                    panels = snaps_np.reshape(B * S_full, *snaps_np.shape[2:])  # rows=batch, cols=meas|steps|gt
                    grid_img = grid(panels, ncols=S_full)

                    # Native-resolution PNG (no resampling, no labels)
                    plt.imsave(
                        os.path.join(save_path, f"batch_{batch_idx + 1}_progression.png"),
                        np.clip(grid_img, 0, 1),
                    )

                    # Labeled PDF at 1 px = 1 fig px, high dpi
                    dpi = 200
                    gh, gw = grid_img.shape[:2]
                    fig, ax = plt.subplots(figsize=(gw / dpi, gh / dpi), dpi=dpi)
                    ax.imshow(grid_img, interpolation="nearest")
                    ax.set_axis_off()
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    labels = ["meas"] + [f"t={s}" for s in steps_sorted] + ["GT"]
                    cell_w = gw / S_full
                    for s_idx, lab in enumerate(labels):
                        ax.text(cell_w * (s_idx + 0.5), -2, lab,
                                ha="center", va="bottom", fontsize=18)
                    fig.savefig(
                        os.path.join(save_path, f"batch_{batch_idx + 1}_progression.pdf"),
                        bbox_inches="tight", pad_inches=0.05, dpi=dpi,
                    )
                    plt.close(fig)

                # Save the dictonary in a .pt file for later use
                torch.save(results, os.path.join(save_path, f"batch_{batch_idx + 1}_results.pt"))




            except Exception as e:
                print(f"ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY  ({args.num_batches} batches, {args.degradation}, {args.model})")
    print("=" * 60)
    for method in methods:
        psnr_vals = accum[method]["psnr"]
        mse_vals = accum[method]["mse"]
        lpips_vals = accum[method]["lpips"]
        if not psnr_vals:
            print(f"{method:15s}  no results")
            continue
        print(f"{method:15s}  "
              f"PSNR={np.mean(psnr_vals):.2f}±{np.std(psnr_vals):.2f}dB  "
              f"MSE={np.mean(mse_vals):.5f}±{np.std(mse_vals):.5f}"
              + (f"  LPIPS={np.mean(lpips_vals):.4f}±{np.std(lpips_vals):.4f}"
                 if lpips_vals else ""))
    print("=" * 60)


class _nullctx:
    """No-op context manager."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


if __name__ == "__main__":
    main()
