#!/usr/bin/env python
"""
Compute metrics for each folder in the P-D sweep output, then generate plots.

Usage:
    python compute_pd_metrics.py \\
        --sweep_root ./pd_sweep_images/celeba \\
        --ref_path /path/to/celeba/test_images \\
        --dataset celeba \\
        --results_json pd_results.json

    python compute_pd_metrics.py \\
        --sweep_root ./pd_sweep_images/afhq \\
        --ref_path /path/to/afhq/test_images \\
        --dataset afhq \\
        --results_json pd_results_afhq.json

--ref_path should point to a flat directory of PNG/JPG images that are the
held-out test split for that dataset (used as input1 for FID/KID/PRC).
These are the same images saved under sweep_root/gt/ — you can simply pass
sweep_root/gt as --ref_path.
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from torch_fidelity.datasets import ImagesPathDataset
from torch_fidelity.registry import register_dataset
from tqdm import tqdm

torch.set_grad_enabled(False)

DATASET_TAG = None  # set in main() before any calculate_metrics call


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_given_folder(xhat_dir, gt_dir, ref_path):
    lpips_met = piq.LPIPS(reduction='mean').cuda()
    rec_files = sorted([os.path.join(xhat_dir, f) for f in os.listdir(xhat_dir)])
    gt_files  = sorted([os.path.join(gt_dir,   f) for f in os.listdir(gt_dir)])
    assert len(gt_files) == len(rec_files), f"{len(gt_files)} gt vs {len(rec_files)} rec"
    for a, b in zip(gt_files, rec_files):
        assert os.path.basename(a) == os.path.basename(b), f"{a} vs {b}"

    xhat_ds = ImagesPathDataset(rec_files)
    gt_ds   = ImagesPathDataset(gt_files)
    gt_dl   = DataLoader(gt_ds,   batch_size=128, shuffle=False, drop_last=False, num_workers=10)
    rec_dl  = DataLoader(xhat_ds, batch_size=128, shuffle=False, drop_last=False, num_workers=10)

    mse = lpips = psnr = ssim = 0.0
    var_rec_sum = var_gt_sum = 0.0
    for gt, rec in tqdm(zip(gt_dl, rec_dl), total=len(gt_dl), leave=False):
        gt  = gt.cuda().float()
        rec = rec.cuda().float()
        mse   += ((gt - rec) ** 2).mean() * gt.shape[0]
        lpips += lpips_met(gt / 255., rec / 255.) * gt.shape[0]
        psnr  += piq.psnr(gt / 255., rec / 255., data_range=1., reduction='sum')
        ssim  += piq.ssim(gt / 255., rec / 255., data_range=1., reduction='sum')
        # per-image variance (over C,H,W), summed over batch
        var_rec_sum += rec.var(dim=[1, 2, 3]).sum()
        var_gt_sum  += gt.var(dim=[1, 2, 3]).sum()

    n = len(gt_ds)
    var_rec = (var_rec_sum / n).item()
    var_gt  = (var_gt_sum  / n).item()
    results = {
        'mse':       (mse   / n).item(),
        'lpips':     (lpips / n).item(),
        'psnr':      (psnr  / n).item(),
        'ssim':      (ssim  / n).item(),
        'var':       var_rec,
        'var_ratio': var_rec / var_gt,
    }

    fidelity_results = calculate_metrics(
        batch_size=512,
        input1=DATASET_TAG,
        input2=xhat_ds,
        datasets_root=ref_path,
        datasets_download=False,
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        prc=True,
        verbose=True,
        kid_subset_size=min(1000, len(xhat_ds)),
        cache=True,
    )
    return {**results, **fidelity_results}


# ---------------------------------------------------------------------------
# Directory walk
# ---------------------------------------------------------------------------

def collect_sweep_results(sweep_root, gt_dir, ref_path, results_json):
    """Walk sweep_root/alpha_*/t_*/ and compute metrics, caching to JSON."""

    if os.path.exists(results_json):
        with open(results_json) as f:
            cache = json.load(f)
    else:
        cache = {}

    # Choose the dirs every 2 steps to avoid overcrowding the plots (adjust as needed).
    alpha_dirs = sorted(
        d for d in os.listdir(sweep_root)
        if d.startswith("alpha_") and os.path.isdir(os.path.join(sweep_root, d))
    )

    all_results = {}

    for alpha_dir in alpha_dirs:
        alpha = float(alpha_dir.split("_", 1)[1])
        alpha_path = os.path.join(sweep_root, alpha_dir)

        t_dirs = sorted(
            d for d in os.listdir(alpha_path)
            if d.startswith("t_") and os.path.isdir(os.path.join(alpha_path, d))
        )

        alpha_res = {k: [] for k in
                     ['t', 'mse', 'rmse', 'lpips', 'psnr', 'ssim',
                      'var', 'var_ratio',
                      'frechet_inception_distance',
                      'kernel_inception_distance_mean',
                      'inception_score_mean',
                      'precision', 'recall']}

        for t_dir in t_dirs:
            t_val = float(t_dir.split("_", 1)[1])
            xhat_dir = os.path.join(alpha_path, t_dir)
            cache_key = f"{alpha_dir}/{t_dir}"

            _required = {'var', 'var_ratio'}
            if cache_key not in cache or not _required.issubset(cache[cache_key]):
                print(f"\nComputing metrics: {cache_key}")
                metrics = compute_metrics_given_folder(xhat_dir, gt_dir, ref_path)
                cache[cache_key] = metrics
                with open(results_json, "w") as f:
                    json.dump(cache, f, indent=2)
            else:
                print(f"  [cached] {cache_key}")

            m = cache[cache_key]
            alpha_res['t'].append(t_val)
            alpha_res['mse'].append(m['mse'])
            alpha_res['rmse'].append(math.sqrt(m['mse']))
            alpha_res['lpips'].append(m['lpips'])
            alpha_res['psnr'].append(m['psnr'])
            alpha_res['ssim'].append(m['ssim'])
            alpha_res['frechet_inception_distance'].append(m.get('frechet_inception_distance', float('nan')))
            alpha_res['kernel_inception_distance_mean'].append(m.get('kernel_inception_distance_mean', float('nan')))
            alpha_res['inception_score_mean'].append(m.get('inception_score_mean', float('nan')))
            alpha_res['precision'].append(m.get('precision', float('nan')))
            alpha_res['recall'].append(m.get('recall', float('nan')))
            alpha_res['var'].append(m.get('var', float('nan')))
            alpha_res['var_ratio'].append(m.get('var_ratio', float('nan')))

        all_results[alpha] = alpha_res

    return all_results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    alphas  = sorted(all_results.keys())
    colors  = ['#534AB7', '#0F6E56', '#D85A30', '#185FA5']
    markers = ['o', 's', 'D', '^']

    # --------------------------------------------------------
    # 1. Single-panel overlay: FID vs RMSE
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, alpha in enumerate(alphas):
        res = all_results[alpha]
        fid_arr  = np.array(res['frechet_inception_distance'])
        rmse_arr = np.array(res['rmse'])

        ax.plot(fid_arr, rmse_arr, '-', color=colors[i], alpha=0.4, linewidth=1.5)
        step = max(1, len(fid_arr) // 8)
        ax.plot(fid_arr[::step], rmse_arr[::step], markers[i],
                color=colors[i], markersize=6,
                markeredgecolor='k', markeredgewidth=0.5,
                label=f'$\\xi={alpha}$ ($s={1-alpha:.1f}$)')
        ax.plot(fid_arr[0],  rmse_arr[0],  markers[i], color=colors[i],
                markersize=9, markeredgecolor='k', markeredgewidth=1.0, zorder=5)
        ax.plot(fid_arr[-1], rmse_arr[-1], markers[i], color=colors[i],
                markersize=9, markeredgecolor='k', markeredgewidth=1.0, zorder=5)

    ax.set_xlabel('FID $\\rightarrow$ better', fontsize=12)
    ax.set_ylabel('RMSE $\\rightarrow$ better', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(out_dir, 'pd_rmse_vs_fid_overlay.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")

    # --------------------------------------------------------
    # 2. Multi-panel: one per noise level
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4), sharey=False)
    if len(alphas) == 1:
        axes = [axes]
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        res = all_results[alpha]
        fid_arr  = np.array(res['frechet_inception_distance'])
        rmse_arr = np.array(res['rmse'])
        t_arr    = np.array(res['t'])

        span  = max(t_arr[-1] - t_arr[0], 1e-8)
        sizes = 30 + 120 * (t_arr - t_arr[0]) / span
        ax.scatter(fid_arr, rmse_arr, s=sizes, c=colors[i],
                   edgecolors='k', linewidths=0.5, zorder=3)
        ax.plot(fid_arr, rmse_arr, '-', color=colors[i], alpha=0.3, linewidth=1.5)
        ax.annotate('$t{=}s$', (fid_arr[0],  rmse_arr[0]),
                    textcoords='offset points', xytext=(-50, 0), fontsize=18, color='black')
        ax.annotate('$t{=}1$', (fid_arr[-1], rmse_arr[-1]),
                    textcoords='offset points', xytext=(-50, -8), fontsize=18, color='black')
        ax.set_xlabel('FID $\\rightarrow$ better', fontsize=20)
        if i == 0:
            ax.set_ylabel('RMSE $\\rightarrow$ better', fontsize=20)
        # Increase size of values in each axis numbers
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(f'$\\xi={alpha}$', fontsize=22)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)


    plt.tight_layout()
    path = os.path.join(out_dir, 'pd_rmse_vs_fid_panels.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")
        
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, (alpha, res) in enumerate(all_results.items()):
        ax.plot(res['frechet_inception_distance'], res['var_ratio'], '-'+markers[i], color=colors[i],
                markersize=5, label=f'$\\xi={alpha}$')

    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.invert_xaxis()  # lower FID = better perception
    ax.set_xlabel('FID $\\rightarrow$ better', fontsize=20)
    ax.set_ylabel('$\\mathrm{Var}(D_{s,t}) \\,/\\, \\mathrm{Var}(\\mathbf{x}_1)$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.set_title('Variance restoration vs.\\ perception', fontsize=12)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'pd_var_ratio_vs_fid.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")

    # --------------------------------------------------------
    # 3. fig:variance — var_ratio vs t (one curve per alpha)
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, alpha in enumerate(alphas):
        res = all_results[alpha]
        t_arr   = np.array(res['t'])
        vr_arr  = np.array(res['var_ratio'])
        order   = np.argsort(t_arr)
        ax.plot(t_arr[order], vr_arr[order], '-' + markers[i],
                color=colors[i], markersize=5,
                label=f'$\\xi={alpha}$')
        print(vr_arr)

    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='GT')
    ax.set_xlabel('Lookahead $t$', fontsize=12)
    ax.set_ylabel('$\\mathrm{Var}(D_{s,t}) \\,/\\, \\mathrm{Var}(\\mathbf{x}_1)$', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'pd_var_ratio_vs_t.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")

    # --------------------------------------------------------
    # 4. fig:variance_vs_lpips — var_ratio vs LPIPS
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, alpha in enumerate(alphas):
        res = all_results[alpha]
        lp_arr = np.array(res['lpips'])
        vr_arr = np.array(res['var_ratio'])
        order  = np.argsort(lp_arr)[::-1]  # left = better LPIPS (lower)
        ax.plot(lp_arr[order], vr_arr[order], '-' + markers[i],
                color=colors[i], markersize=5, markevery=3,
                label=f'$\\xi={alpha}$')

    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='GT')
    ax.invert_xaxis()  # lower LPIPS = better perception
    ax.set_xlabel('LPIPS $\\leftarrow$ better', fontsize=12)
    ax.set_ylabel('$\\mathrm{Var}(D_{s,t}) \\,/\\, \\mathrm{Var}(\\mathbf{x}_1)$', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'pd_var_ratio_vs_lpips.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {path}")

    # --------------------------------------------------------
    # 5. Summary table
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'alpha':>6} | {'s':>4} | {'RMSE(t=s)':>10} | {'RMSE(t=1)':>10} | "
          f"{'FID(t=s)':>9} | {'FID(t=1)':>8}")
    print("-" * 70)
    for alpha in alphas:
        res = all_results[alpha]
        print(f"{alpha:6.1f} | {1-alpha:4.1f} | {res['rmse'][0]:10.4f} | "
              f"{res['rmse'][-1]:10.4f} | "
              f"{res['frechet_inception_distance'][0]:9.2f} | "
              f"{res['frechet_inception_distance'][-1]:8.2f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_root', type=str, required=True,
                   help='Root of sweep output, e.g. pd_sweep_images/celeba')
    p.add_argument('--dataset', type=str, required=True,
                   choices=['celeba', 'afhq'],
                   help='Which dataset was used (sets the FID reference)')
    p.add_argument('--ref_path', type=str, required=False, default=None,
                   help='Path to reference images for FID/KID/PRC. '
                        'Defaults to sweep_root/gt (the saved test split).')
    p.add_argument('--results_json', type=str, default='pd_results.json',
                   help='JSON file to cache computed metrics (resume-safe)')
    p.add_argument('--plots_dir', type=str, default='pd_plots',
                   help='Output directory for PDF plots')
    return p.parse_args()


def main():
    global DATASET_TAG

    args = parse_args()

    gt_dir   = os.path.join(args.sweep_root, 'gt')
    ref_path = args.ref_path if args.ref_path else gt_dir
    assert os.path.isdir(gt_dir),   f"GT dir not found: {gt_dir}"
    assert os.path.isdir(ref_path), f"ref_path not found: {ref_path}"

    # Register the test-split images as the reference distribution.
    # torch_fidelity expects: dataset_fn(root, download) -> Dataset
    # We store images directly in ref_path (flat folder of PNGs).
    DATASET_TAG = args.dataset
    ref_files = sorted([
        os.path.join(ref_path, f)
        for f in os.listdir(ref_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    assert ref_files, f"No images found in ref_path: {ref_path}"
    register_dataset(
        DATASET_TAG,
        lambda _root, _download, _files=ref_files: ImagesPathDataset(_files),
    )

    all_results = collect_sweep_results(
        args.sweep_root, gt_dir, ref_path, args.results_json
    )

    os.makedirs(args.plots_dir, exist_ok=True)
    torch.save(all_results, os.path.join(args.plots_dir, 'pd_results.pt'))

    make_plots(all_results, args.plots_dir)


if __name__ == '__main__':
    main()
