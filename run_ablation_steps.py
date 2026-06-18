#!/usr/bin/env python
"""
Compute FID and RMSE for each afhq_pnp_flow_{la}_{task}_{N} folder in
generated_samples_ablation_steps, then generate:
  - FID vs steps  (one line per model: la=0.0 and la=1.0)
  - RMSE vs steps (one line per model: la=0.0 and la=1.0)

Usage:
    python ablation_steps_metrics.py
    python ablation_steps_metrics.py --samples_dir ./generated_samples_ablation_steps --plots_dir ablation_steps_plots
"""

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_fidelity.datasets import ImagesPathDataset

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def parse_folder(name):
    """
    Parse 'afhq_pnp_flow_{la}_{task}_{N}_{suffix}' into (la, task, N) or None.
    Examples:
        afhq_pnp_flow_0.0_super_resolution_5_0.01  -> (0.0, 'super_resolution', 5)
        afhq_pnp_flow_1.0_super_resolution_25_0.01 -> (1.0, 'super_resolution', 25)
    """
    m = re.match(r'afhq_pnp_flow_([\d.]+)_(.+?)_(\d+)(?:_[\d.]+)?$', name)
    # m = re.match(r'afhq_pnp_flow_([\d.]+)_(.+?)_(\d+)(?:_[\d.]+)?$', name)
    if m:
        return float(m.group(1)), m.group(2), int(m.group(3))
    return None


def discover(samples_dir):
    """
    Returns {task: {la: {N: abs_folder_path}}}.
    Incomplete runs (zero restored PNGs) are silently skipped.
    """
    data = {}
    for name in os.listdir(samples_dir):
        parsed = parse_folder(name)
        if parsed is None:
            continue
        la, task, N = parsed
        folder = os.path.join(samples_dir, name)
        n = len([f for f in os.listdir(folder)
                 if 'restored' in f and f.endswith('.png')])
        if n == 0:
            continue
        data.setdefault(task, {}).setdefault(la, {})[N] = folder

    return data


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def sorted_restored_pngs(folder):
    """Return abs paths of restored PNGs sorted by (batch_idx, img_idx)."""
    files = [f for f in os.listdir(folder)
             if 'restored' in f and f.endswith('.png')]
    files.sort(key=lambda f: (int(f.split('_')[1]), int(f.split('_')[3].split('.')[0])))
    return [os.path.join(folder, f) for f in files]


def sorted_gt_pngs(gt_dir):
    """Return sorted abs paths of GT PNGs from the ground-truth folder."""
    files = sorted(
        f for f in os.listdir(gt_dir)
        if f.lower().endswith(('.png', '.jpg'))
    )
    return [os.path.join(gt_dir, f) for f in files]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(rec_files, gt_files):
    """RMSE in [0, 255] pixel space."""
    assert len(rec_files) == len(gt_files), \
        f"rec={len(rec_files)} vs gt={len(gt_files)}"

    rec_ds = ImagesPathDataset(rec_files)
    gt_ds  = ImagesPathDataset(gt_files)
    rec_dl = DataLoader(rec_ds, batch_size=128, shuffle=False, num_workers=4)
    gt_dl  = DataLoader(gt_ds,  batch_size=128, shuffle=False, num_workers=4)

    mse_sum = 0.0
    for rec_b, gt_b in zip(rec_dl, gt_dl):
        rec_b = rec_b.float()
        gt_b  = gt_b.float()
        mse_sum += ((rec_b - gt_b) ** 2).mean() * rec_b.shape[0]

    return math.sqrt((mse_sum / len(rec_ds)).item())


def compute_fid(rec_files, gt_dir):
    """FID via pytorch_fid between rec_files (written to a temp folder) and gt_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        for i, src in enumerate(rec_files):
            dst = os.path.join(tmp, f"{i:06d}.png")
            os.symlink(os.path.abspath(src), dst)
        cmd = [
            "python", "-m", "pytorch_fid",
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            tmp, gt_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"pytorch_fid failed:\n{result.stderr}")
        out = result.stdout
    for line in out.splitlines():
        if "FID" in line:
            return float(line.split()[-1])
    raise RuntimeError(f"Could not parse pytorch_fid output:\n{out}")


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run(samples_dir, gt_dir, fid_ref_dir, cache_json):
    data   = discover(samples_dir)
    all_gt = sorted_gt_pngs(gt_dir)

    if os.path.exists(cache_json):
        with open(cache_json) as f:
            cache = json.load(f)
    else:
        cache = {}

    results = {}

    for task in sorted(data):
        for la in sorted(data[task]):
            for N in sorted(data[task][la]):
                folder   = data[task][la][N]
                rec_pngs = sorted_restored_pngs(folder)
                gt_pngs  = all_gt[:len(rec_pngs)]

                key = f"{task}|{la}|{N}"
                if key not in cache or 'rmse' not in cache[key]:
                    print(f"  computing  task={task}  la={la}  N={N}  ({len(rec_pngs)} images)")
                    rmse = compute_rmse(rec_pngs, gt_pngs)
                    fid  = compute_fid(rec_pngs, fid_ref_dir)
                    cache[key] = {'rmse': rmse, 'fid': fid}
                    with open(cache_json, 'w') as f:
                        json.dump(cache, f, indent=2)
                else:
                    print(f"  [cached]   task={task}  la={la}  N={N}")

                results[(task, la, N)] = cache[key]

    return results, data


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

TASK_LABELS = {
    'super_resolution':    'Super-resolution x4',
    'deblurring_gaussian': 'Gaussian deblurring',
    'deblurring_motion':   'Motion deblurring',
    'inpainting_random':   'Inpainting (random)',
}

MODEL_STYLES = {
    0.0: dict(color='#185FA5', linestyle='-',  marker='o', label='$t = s$'),
    1.0: dict(color='#D85A30', linestyle='--', marker='s', label='$t = 1.0$'),
}


def make_ablation_plot(results, data, plots_dir):
    """One figure per task: FID vs steps and RMSE vs steps side by side."""
    for task in sorted(data):
        las = sorted(data[task].keys())
        fig, (ax_fid, ax_rmse) = plt.subplots(1, 2, figsize=(10, 4))

        for la in las:
            steps_available = sorted(data[task][la].keys())
            # Only include steps that have computed results
            steps = [N for N in steps_available if (task, la, N) in results]
            if not steps:
                continue
            fid_vals  = [results[(task, la, N)]['fid']  for N in steps]
            rmse_vals = [results[(task, la, N)]['rmse'] for N in steps]
            style = MODEL_STYLES.get(la, dict(marker='o'))

            ax_fid.plot(steps, fid_vals, linewidth=2, markersize=6, **style)
            ax_rmse.plot(steps, rmse_vals, linewidth=2, markersize=6, **style)

        label = TASK_LABELS.get(task, task)

        ax_fid.set_xlabel('Number of steps', fontsize=20)
        ax_fid.set_ylabel('FID', fontsize=20)
        # ax_fid.set_title(f'{label} — FID', fontsize=20)
        ax_fid.legend(fontsize=18)
        # Change size of values in the axis
        ax_fid.tick_params(axis='both', which='major', labelsize=20)
        ax_fid.grid(True, alpha=0.3)

        ax_rmse.set_xlabel('Number of steps', fontsize=20)
        ax_rmse.set_ylabel('RMSE', fontsize=20)
        # ax_rmse.set_title(f'{label} — RMSE', fontsize=20)
        ax_rmse.legend(fontsize=18)
        ax_rmse.grid(True, alpha=0.3)
        ax_rmse.tick_params(axis='both', which='major', labelsize=20)

        fig.tight_layout()
        safe_task = task.replace('/', '_')
        path = os.path.join(plots_dir, f'ablation_steps_{safe_task}.pdf')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def make_combined_plot(results, data, plots_dir):
    """Single figure with all tasks: FID vs steps (left) and RMSE vs steps (right)."""
    tasks = sorted(data.keys())
    if not tasks:
        return

    fig, (ax_fid, ax_rmse) = plt.subplots(1, 2, figsize=(12, 5))

    task_colors = plt.cm.tab10.colors
    handles_seen = set()
    legend_handles = []

    for t_idx, task in enumerate(tasks):
        las = sorted(data[task].keys())
        label_base = TASK_LABELS.get(task, task)

        for la in las:
            steps_available = sorted(data[task][la].keys())
            steps = [N for N in steps_available if (task, la, N) in results]
            if not steps:
                continue
            fid_vals  = [results[(task, la, N)]['fid']  for N in steps]
            rmse_vals = [results[(task, la, N)]['rmse'] for N in steps]

            color = task_colors[t_idx % len(task_colors)]
            linestyle = MODEL_STYLES.get(la, {}).get('linestyle', '-')
            marker    = MODEL_STYLES.get(la, {}).get('marker', 'o')
            full_label = f"{label_base}  ($\\lambda={la}$)"

            line_fid,  = ax_fid.plot(steps, fid_vals,   color=color, linestyle=linestyle,
                                      marker=marker, linewidth=2, markersize=6, label=full_label)
            line_rmse, = ax_rmse.plot(steps, rmse_vals,  color=color, linestyle=linestyle,
                                       marker=marker, linewidth=2, markersize=6, label=full_label)

            if full_label not in handles_seen:
                handles_seen.add(full_label)
                legend_handles.append(line_fid)

    for ax, ylabel, title in [
        (ax_fid,  'FID',  'FID vs Steps'),
        (ax_rmse, 'RMSE', 'RMSE vs Steps'),
    ]:
        ax.set_xlabel('Number of steps', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        # ax.set_title(title, fontsize=20)
        ax.grid(True, alpha=0.3)

    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               fontsize=20, loc='lower center',
               bbox_to_anchor=(0.5, 0), ncol=min(len(legend_handles), 4),
               frameon=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    path = os.path.join(plots_dir, 'ablation_steps_combined.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_dir', default='./generated_samples_ablation_steps')
    p.add_argument('--gt_dir',      default='./generated_samples_256/fid_folder_ground_truth',
                   help='Paired GT PNGs for RMSE (sorted, 1-to-1 with restored)')
    p.add_argument('--fid_ref_dir', default='./generated_samples_256/fid_folder_ground_truth',
                   help='Reference folder for FID computation')
    p.add_argument('--cache_json',  default='ablation_steps_cache.json')
    p.add_argument('--plots_dir',   default='./ablation_steps_plots')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"Samples dir:   {args.samples_dir}")
    print(f"FID reference: {args.fid_ref_dir}")
    assert os.path.isdir(args.fid_ref_dir) and any(
        f.lower().endswith(('.png', '.jpg')) for f in os.listdir(args.fid_ref_dir)
    ), f"No images found in {args.fid_ref_dir}"

    results, data = run(
        args.samples_dir, args.gt_dir, args.fid_ref_dir,
        args.cache_json,
    )

    make_ablation_plot(results, data, args.plots_dir)
    make_combined_plot(results, data, args.plots_dir)

    print("\n" + "=" * 65)
    print(f"{'task':<25} {'la':>5} {'N':>6} | {'RMSE':>8} {'FID':>8}")
    print("-" * 65)
    for (task, la, N), m in sorted(results.items()):
        print(f"{task:<25} {la:>5} {N:>6} | {m['rmse']:>8.4f} {m['fid']:>8.2f}")
    print("=" * 65)


if __name__ == '__main__':
    main()
