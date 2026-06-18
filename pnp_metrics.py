#!/usr/bin/env python
"""
Compute FID and RMSE for each celeba_pnp_flow_{la}_{task}_{N} folder,
then generate:
  - Figure 1 : FID vs RMSE curve  (one line per task, colored dots per lookahead)
  - Figure 2+ : image strip per task  (measurement | restored_la0 | restored_la1 | ...)

Usage:
    python pnp_metrics.py
    python pnp_metrics.py --samples_dir ./generated_samples --plots_dir pnp_plots
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch_fidelity.datasets import ImagesPathDataset

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Full CelebA test set — read directly from zip for FID reference
# ---------------------------------------------------------------------------

class CelebAZipDataset(Dataset):
    """Serves CelebA test-split images from the zip (no extraction needed)."""

    def __init__(self, zip_path, partition_file, split='test'):
        split_id = {'train': 0, 'val': 1, 'test': 2}[split]
        with open(partition_file) as f:
            entries = [line.strip().split() for line in f]
        valid = {e[0].replace('.jpg', '.png') for e in entries if int(e[1]) == split_id}
        with zipfile.ZipFile(zip_path) as z:
            self.names = sorted(n for n in z.namelist() if n in valid)
        self.zip_path = zip_path
        self._local = threading.local()

    def _get_zip(self):
        if not hasattr(self._local, 'zip'):
            self._local.zip = zipfile.ZipFile(self.zip_path)
        return self._local.zip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with self._get_zip().open(self.names[idx]) as f:
            # torch_fidelity expects uint8 HWC numpy or CHW uint8 tensor
            img = Image.open(f).convert('RGB')
        return torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (C,H,W) uint8


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def parse_folder(name):
    """
    Parse '{dataset}_pnp_flow_{la}_{task}_{N}' into (dataset, la, task, N) or None.
    Examples:
        celeba_pnp_flow_0.0_deblurring_100         -> ('celeba', 0.0, 'deblurring', 100)
        afhq_pnp_flow_1.0_deblurring_motion_100    -> ('afhq',   1.0, 'deblurring_motion', 100)
        celeba_pnp_flow_0.0_inpainting_random_300  -> ('celeba', 0.0, 'inpainting_random', 300)
    """
    m = re.match(r'(celeba|afhq)_pnp_flow_([\d.]+)_(.+?)_(\d+)(?:_[\d.]+)?$', name)
    if m:
        return m.group(1), float(m.group(2)), m.group(3), int(m.group(4))
    return None


def discover(samples_dir):
    """
    Returns {(dataset, task, N): {la: abs_folder_path}}.
    Within each group, only la values whose restored-PNG count equals the
    group maximum are included (incomplete/partial runs are silently skipped).
    """
    groups = {}
    for name in os.listdir(samples_dir):
        parsed = parse_folder(name)
        if parsed is None:
            continue
        dataset, la, task, N = parsed
        folder = os.path.join(samples_dir, name)
        n = len([f for f in os.listdir(folder)
                 if 'restored' in f and f.endswith('.png')])
        if n == 0:
            continue
        groups.setdefault((dataset, task, N), {})[la] = (folder, n)

    clean = {}
    for (dataset, task, N), la_map in groups.items():
        max_n = max(v[1] for v in la_map.values())
        complete = {la: path for la, (path, n) in la_map.items() if n == max_n}
        if complete:
            skipped = [la for la, (_, n) in la_map.items() if n < max_n]
            if skipped:
                print(f"  [skip incomplete] dataset={dataset} task={task} N={N} la={skipped} "
                      f"(only {[la_map[la][1] for la in skipped]} / {max_n} images)")
            clean[(dataset, task, N)] = complete
    return clean


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def sorted_restored_pngs(folder):
    """Return abs paths of restored PNGs sorted by (batch_idx, img_idx)."""
    files = [f for f in os.listdir(folder)
             if 'restored' in f and f.endswith('.png')]
    files.sort(key=lambda f: (int(f.split('_')[1]), int(f.split('_')[3].split('.')[0])))
    return [os.path.join(folder, f) for f in files]


def sorted_noisy_pngs(folder):
    """Return abs paths of noisy PNGs sorted by (batch_idx, img_idx) — for image strips."""
    files = [f for f in os.listdir(folder)
             if 'noisy' in f and f.endswith('.png')]
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
# Metrics  (matching PD_experiment/compute_pd_metrics.py conventions)
# ---------------------------------------------------------------------------

def compute_rmse(rec_files, gt_files):
    """
    RMSE in [0, 255] pixel space — identical convention to PD_experiment.
    Images loaded via ImagesPathDataset (uint8-range float tensors).
    """
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
        # Symlink restored PNGs into tmp so pytorch_fid can read them as a folder
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
    # pytorch_fid prints "FID:  <value>"
    for line in out.splitlines():
        if "FID" in line:
            return float(line.split()[-1])
    raise RuntimeError(f"Could not parse pytorch_fid output:\n{out}")


def compute_kid(rec_files, gt_dir, subset_size, cache_root=None, ref_cache_name=None):
    """KID (mean, std) via torch_fidelity between rec_files and gt_dir.

    Unlike FID, KID is an unbiased estimator and is far better behaved at low
    sample counts; torch_fidelity reports a standard deviation across the random
    subsets it draws, giving an honest error bar on the metric.

    The std is only meaningful when `subset_size` is *strictly smaller* than each
    input set: if the subset equals the whole set there is nothing to resample
    and the std collapses to 0. We therefore clamp to min(subset_size, #rec,
    #ref) and rely on the caller to pass a subset below the restored count.

    If `cache_root`/`ref_cache_name` are given, the (large) reference's Inception
    features are cached on disk and reused across calls, so the ~20k-image
    reference is only featurised once.
    """
    from torch_fidelity import calculate_metrics

    n_ref  = len([f for f in os.listdir(gt_dir)
                  if f.lower().endswith(('.png', '.jpg'))])
    subset = max(2, min(subset_size, len(rec_files), n_ref))

    kw = dict(
        cuda=torch.cuda.is_available(),
        fid=False, isc=False, kid=True,
        kid_subset_size=subset,
        verbose=False,
    )
    if cache_root and ref_cache_name:
        kw.update(cache=True, cache_root=cache_root, input2_cache_name=ref_cache_name)

    with tempfile.TemporaryDirectory() as tmp:
        for i, src in enumerate(rec_files):
            dst = os.path.join(tmp, f"{i:06d}.png")
            os.symlink(os.path.abspath(src), dst)
        metrics = calculate_metrics(input1=tmp, input2=gt_dir, **kw)
    return (metrics['kernel_inception_distance_mean'],
            metrics['kernel_inception_distance_std'])


def materialize_celeba_test_ref(zip_path, partition_file, out_dir, split='test'):
    """Extract the full CelebA *test* split from the zip into a flat folder, to
    serve as a large FID/KID distribution reference (vs. the 100 paired GT images
    used for RMSE). Byte-copies the PNGs (no re-encoding); skips if already done."""
    split_id = {'train': 0, 'val': 1, 'test': 2}[split]
    with open(partition_file) as f:
        entries = [line.strip().split() for line in f]
    valid = {e[0].replace('.jpg', '.png') for e in entries if int(e[1]) == split_id}

    os.makedirs(out_dir, exist_ok=True)
    existing = set(os.listdir(out_dir))
    with zipfile.ZipFile(zip_path) as z:
        members = sorted(n for n in z.namelist() if n in valid)
        if len(existing) >= len(members):
            return out_dir
        for m in members:
            base = os.path.basename(m)
            if base in existing:
                continue
            with z.open(m) as src, open(os.path.join(out_dir, base), 'wb') as out:
                out.write(src.read())
    return out_dir


def materialize_afhq_test_ref(zip_path, out_dir, train_ratio=0.9):
    """Extract the AFHQ *test* split into a flat folder for use as the KID
    distribution reference. The split is taken straight from ImageFolderDataset
    (sorted filenames, last 1-train_ratio fraction) so it matches exactly how the
    test set is built for generation. Zip entries live under cat/dog/wild, so the
    subdir is folded into the filename to keep names unique."""
    from training.dataset import ImageFolderDataset

    base   = ImageFolderDataset(path=zip_path, split='test', train_ratio=train_ratio)
    fnames = list(base._image_fnames)   # exact test-split members, in dataset order
    base.close()

    os.makedirs(out_dir, exist_ok=True)
    existing = set(os.listdir(out_dir))
    with zipfile.ZipFile(zip_path) as z:
        if len(existing) >= len(fnames):
            return out_dir
        for fn in fnames:
            flat = fn.replace('/', '_')
            if flat in existing:
                continue
            with z.open(fn) as src, open(os.path.join(out_dir, flat), 'wb') as out:
                out.write(src.read())
    return out_dir


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def load_fid_seed(cache_json):
    """Build {(dataset, task, N, la) -> fid} from a previous cache, tolerating both
    the legacy 'task|N|la' (celeba-only, no dataset prefix) and the current
    'dataset|task|N|la[|reftag]' key formats. Used to reuse already-reported FID
    instead of recomputing it."""
    if not os.path.exists(cache_json):
        return {}
    with open(cache_json) as f:
        cache = json.load(f)
    seed = {}
    for key, entry in cache.items():
        if 'fid' not in entry:
            continue
        parts = key.split('|')
        if len(parts) >= 4 and parts[0] in ('celeba', 'afhq'):
            dataset, task, N, la = parts[0], parts[1], int(parts[2]), float(parts[3])
        elif len(parts) == 3:                 # legacy celeba key without dataset
            dataset, task, N, la = 'celeba', parts[0], int(parts[1]), float(parts[2])
        else:
            continue
        seed.setdefault((dataset, task, N, la), entry['fid'])
    return seed


def run(samples_dir, rmse_gt_dirs, fid_ref_dirs, kid_ref_dirs, cache_json,
        kid_subset_size, fidelity_cache_dir, fid_seed=None):
    """Three decoupled references, on purpose:
      - `rmse_gt_dirs[ds]` : paired ground-truth images, aligned to restored order (RMSE).
      - `fid_ref_dirs[ds]` : FID reference (only used if FID has to be computed).
      - `kid_ref_dirs[ds]` : KID reference. Uses the *full* test set: KID is a kernel-MMD
                             estimator that stays well-defined at low sample count and
                             reports a meaningful std.

    FID is *not* recomputed here when a value is available in `fid_seed`
    (mapping (dataset, task, N, la) -> fid). FID at n~=100 is numerically a
    coin-flip (rank-deficient 2048-dim covariance -> unstable matrix sqrt), so
    we reuse the originally reported FID and only compute KID fresh.
    """
    fid_seed = fid_seed or {}
    groups   = discover(samples_dir)
    present  = {ds for ds, _, _ in groups}
    all_gt_by_dataset = {ds: sorted_gt_pngs(rmse_gt_dirs[ds]) for ds in present}

    # Per-dataset reference bookkeeping: counts (-> cache tag) and KID feature-cache name.
    ref_info = {}
    for ds in present:
        n_fid = len([f for f in os.listdir(fid_ref_dirs[ds])
                     if f.lower().endswith(('.png', '.jpg'))])
        n_kid = len([f for f in os.listdir(kid_ref_dirs[ds])
                     if f.lower().endswith(('.png', '.jpg'))])
        ref_info[ds] = dict(
            fid_dir=fid_ref_dirs[ds], kid_dir=kid_ref_dirs[ds],
            n_fid=n_fid, n_kid=n_kid,
            kid_cache=f"{ds}_kidref{n_kid}",
            tag=f"fid{n_fid}_kid{n_kid}",
        )

    if os.path.exists(cache_json):
        with open(cache_json) as f:
            cache = json.load(f)
    else:
        cache = {}

    results = {}

    for (dataset, task, N), la_map in sorted(groups.items()):
        info   = ref_info[dataset]
        all_gt = all_gt_by_dataset[dataset]

        for la in sorted(la_map):
            folder   = la_map[la]
            rec_pngs = sorted_restored_pngs(folder)
            gt_pngs  = all_gt[:len(rec_pngs)]       # paired GT in order

            # Reference identities are baked into the key so changing either
            # reference recomputes rather than reusing stale cached values.
            key   = f"{dataset}|{task}|{N}|{la}|{info['tag']}"
            entry = cache.get(key, {})
            # Compute only the metrics that are missing.
            need = [m for m in ('rmse', 'fid', 'kid_mean') if m not in entry]
            if need:
                print(f"  computing {need}  dataset={dataset}  task={task}  N={N}  la={la}  "
                      f"({len(rec_pngs)} imgs | fid ref={info['n_fid']} | kid ref={info['n_kid']})")
                if 'rmse' not in entry:
                    entry['rmse'] = compute_rmse(rec_pngs, gt_pngs)
                if 'fid' not in entry:
                    seeded = fid_seed.get((dataset, task, N, la))
                    entry['fid'] = seeded if seeded is not None \
                        else compute_fid(rec_pngs, info['fid_dir'])
                if 'kid_mean' not in entry:
                    kid_mean, kid_std = compute_kid(
                        rec_pngs, info['kid_dir'], kid_subset_size,
                        cache_root=fidelity_cache_dir, ref_cache_name=info['kid_cache'])
                    entry['kid_mean'] = kid_mean
                    entry['kid_std']  = kid_std
                cache[key] = entry
                with open(cache_json, 'w') as f:
                    json.dump(cache, f, indent=2)
            else:
                print(f"  [cached]   dataset={dataset}  task={task}  N={N}  la={la}")

            results[(dataset, task, N, la)] = cache[key]

    return results, groups


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

TASK_LABELS = {
    'super_resolution':     'Super-resolution x 4',
    'deblurring_gaussian': 'Gaussian deblurring',
    'deblurring_motion':   'Motion deblurring',
    'inpainting_random':   'Inpainting (random)',
    'inpainting_box':      'Inpainting (box)',
}

TASK_STYLES = {
    'super_resolution':    dict(linestyle='-',  color="#D400FF"),
    'deblurring_gaussian': dict(linestyle='-',  color='#185FA5'),
    'deblurring_motion':   dict(linestyle='-', color='#D85A30'),
    'inpainting_random':   dict(linestyle='-',  color="#08C431"),
    'inpainting_box':      dict(linestyle='-',  color="#FF6B6B"),

}

# Linestyle by dataset.
DATASET_STYLES = {'celeba': '-', 'afhq': '--'}
DATASET_LABELS = {'celeba': 'CelebA', 'afhq': 'AFHQ'}
_DATASET_STYLE_FALLBACK = ['-', '--', ':', '-.']


def dataset_linestyle(dataset, all_datasets):
    if dataset in DATASET_STYLES:
        return DATASET_STYLES[dataset]
    return _DATASET_STYLE_FALLBACK[sorted(all_datasets).index(dataset) % len(_DATASET_STYLE_FALLBACK)]


def make_metric_vs_rmse_plot(results, groups, plots_dir, filename,
                             metric_key, metric_label, std_key=None):
    """Figure: <metric> vs RMSE, one curve per (dataset, task, N), colored dots per la.
    Color encodes task; all curves use a solid linestyle. If `std_key` is given,
    horizontal error bars (±1 std on the metric) are drawn at each point."""
    fig, ax = plt.subplots(figsize=(6, 5))

    la_values = sorted(set(la for (_, _, _, la) in results))
    cmap = plt.cm.tab10.colors

    seen_tasks = []

    for (dataset, task, N), la_map in sorted(groups.items()):
        las = sorted(la_map)
        x_vals    = [results[(dataset, task, N, la)][metric_key] for la in las]
        rmse_vals = [results[(dataset, task, N, la)]['rmse']     for la in las]

        style = dict(TASK_STYLES.get(task, {}))
        style['linestyle'] = '-'
        ax.plot(x_vals, rmse_vals, linewidth=1.5, zorder=1, **style)

        if std_key:
            xerr = [results[(dataset, task, N, la)][std_key] for la in las]
            ax.errorbar(x_vals, rmse_vals, xerr=xerr, fmt='none',
                        ecolor='gray', elinewidth=1, capsize=3,
                        alpha=0.4, zorder=2)

        if task not in seen_tasks:
            seen_tasks.append(task)

        for i, la in enumerate(las):
            ci = la_values.index(la)
            ax.scatter(x_vals[i], rmse_vals[i],
                        color=cmap[ci], s=80, zorder=3,
                        edgecolor='black', linewidths=1.2,
                        label=f'$t = \min(s + {la}, 1.0)$' if (dataset, task, N) == sorted(groups)[0] else None)

    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('RMSE $\\rightarrow$ better', fontsize=12)
    ax.invert_xaxis()
    ax.invert_yaxis()

    # la (scatter) legend inside axes
    handles, labels = ax.get_legend_handles_labels()
    scatter_handles = [(h, l) for h, l in zip(handles, labels) if l and l.startswith('$t')]
    if scatter_handles:
        sh, sl = zip(*scatter_handles)
        leg_la = ax.legend(sh, sl, fontsize=9, framealpha=0.8)
        ax.add_artist(leg_la)

    # task (color) legend below the figure
    task_proxies = [Line2D([0], [0],
                           color=TASK_STYLES.get(t, {}).get('color', 'black'),
                           linestyle='-', linewidth=1.5,
                           label=TASK_LABELS.get(t, t))
                    for t in seen_tasks]
    if task_proxies:
        fig.legend(handles=task_proxies, fontsize=9, framealpha=0.8,
                   loc='lower center', bbox_to_anchor=(0.5, 0),
                   ncol=min(len(task_proxies), 3), frameon=True)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


STRIP_LAS = {0.0, 0.1, 0.2, 0.3, 1.0}

# Image index (0-based) to display per task — pick one that shows clear improvement.
STRIP_IMG_IDX = {
    'deblurring':         0,
    'deblurring_motion':  1,
    'inpainting_random':  1,
    'inpainting_box':     5,
    'super_resolution':   4,
}

# Zoom region as (y0, y1, x0, x1) in pixel coords — centre-crop of the face.
# Adjust these to the actual image resolution if needed.
def _zoom_box(img_np):
    """Return a centre crop covering ~40% of the image height/width."""
    H, W = img_np.shape[:2]
    y0, y1 = int(0.25 * H), int(0.65 * H)
    x0, x1 = int(0.25 * W), int(0.75 * W)
    return y0, y1, x0, x1


def _add_zoom_inset(ax, img_np, edge_color):
    """Overlay a zoomed crop in the top portion of ax with a coloured border."""
    y0, y1, x0, x1 = _zoom_box(img_np)
    crop = img_np[y0:y1, x0:x1]

    # Inset axes: top-right corner, occupying 45% width × 35% height of the panel
    inset_w, inset_h = 0.45, 0.35
    ax_ins = ax.inset_axes([1 - inset_w - 0.02, 1 - inset_h - 0.02, inset_w, inset_h])
    ax_ins.imshow(crop)
    ax_ins.set_xticks([]); ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(2)

    # Draw a rectangle on the main image showing the zoomed region
    rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               linewidth=1.5, edgecolor=edge_color, facecolor='none')
    ax.add_patch(rect)


def make_image_strip(dataset, task, N, la_map, samples_dir, plots_dir):
    """Figure per (dataset, task): [measurement | restored_la0 | restored_la1 | ...]."""
    las = sorted(la for la in la_map if la in STRIP_LAS)
    n_panels = 1 + len(las)
    cmap = plt.cm.tab10.colors

    img_idx = STRIP_IMG_IDX.get(task, 0)

    first_folder = la_map[las[0]]
    noisy_files  = sorted_noisy_pngs(first_folder)
    img_idx      = min(img_idx, len(noisy_files) - 1)
    noisy_img_np = np.array(Image.open(noisy_files[img_idx]).convert('RGB'))

    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.5))

    # Measurement panel
    axes[0].imshow(noisy_img_np)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title('Measurement', fontsize=20)
    for spine in axes[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    _add_zoom_inset(axes[0], noisy_img_np, 'black')

    # Restored panels
    for i, la in enumerate(las):
        folder    = la_map[la]
        rec_files = sorted_restored_pngs(folder)
        img_np    = np.array(Image.open(rec_files[img_idx]).convert('RGB'))

        axes[i + 1].imshow(img_np)
        axes[i + 1].set_xticks([]); axes[i + 1].set_yticks([])
        axes[i + 1].set_title(f'$t = \min(s + {la}, 1.0)$', fontsize=20)
        color = cmap[i]
        for spine in axes[i + 1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        _add_zoom_inset(axes[i + 1], img_np, color)

    label = TASK_LABELS.get(task, task)
    fig.suptitle(label, fontsize=20, y=1.02)
    fig.tight_layout()
    safe_task = task.replace('/', '_')
    path = os.path.join(plots_dir, f'pnp_strip_{dataset}_{safe_task}_N{N}.pdf')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_dir',    default='./generated_samples_section62',)
    p.add_argument('--celeba_ref_dir', default='./generated_samples_128/fid_folder_ground_truth',
                   help='Reference dir for CelebA (paired RMSE GT and FID reference)')
    p.add_argument('--afhq_ref_dir',   default='./generated_samples_256/fid_folder_ground_truth',
                   help='Reference dir for AFHQ (paired RMSE GT and FID reference)')
    p.add_argument('--celeba_zip',     default='./data/celeba/celeba-128x128.zip',
                   help='CelebA zip for full test-set FID reference (preferred)')
    p.add_argument('--celeba_partition', default='./data/celeba/list_eval_partition.txt')
    p.add_argument('--full_celeba_ref', action='store_true', default=True,
                   help='Use the full CelebA test split (from the zip) as the KID '
                        'reference, instead of the 100 paired GT images.')
    p.add_argument('--no_full_celeba_ref', dest='full_celeba_ref', action='store_false',
                   help='Use the paired GT folder as the KID reference (legacy behaviour).')
    p.add_argument('--afhq_zip', default='/home/nvidia/flow_maps/data/afhq-256x256.zip',
                   help='AFHQ zip used to build the full test-split KID reference '
                        '(same zip/split used for generation).')
    p.add_argument('--afhq_train_ratio', type=float, default=0.9,
                   help='train_ratio for the AFHQ ImageFolderDataset test split.')
    p.add_argument('--full_afhq_ref', action='store_true', default=True,
                   help='Use the full AFHQ test split (from the zip) as the KID reference.')
    p.add_argument('--no_full_afhq_ref', dest='full_afhq_ref', action='store_false',
                   help='Use the paired GT folder as the AFHQ KID reference (legacy behaviour).')
    p.add_argument('--kid_subset_size', type=int, default=50,
                   help='KID subset size. Must be < the restored-image count for the '
                        'reported KID std to be non-zero (resampling needs room to vary).')
    p.add_argument('--fid_cache', default='pnp_metrics_cache.json',
                   help='Cache JSON to reuse already-reported FID values from (FID at '
                        'n~=100 is not reproducible, so we do not recompute it).')
    p.add_argument('--cache_json',     default='pnp_metrics_cache.json')
    p.add_argument('--cache_dir',      default='./fidelity_cache')
    p.add_argument('--plots_dir',      default='./pnp_plots')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Paired GT (for RMSE) — one folder per dataset, images aligned to restored order.
    rmse_gt_dirs = {
        'celeba': args.celeba_ref_dir,
        'afhq':   args.afhq_ref_dir,
    }
    # FID reference stays at the original (paired-GT) reference: FID against a large
    # reference is numerically ill-posed with only ~100 restored images.
    fid_ref_dirs = dict(rmse_gt_dirs)
    # KID reference defaults to the FID reference, but for CelebA we use the full
    # test split (extracted from the zip) — KID stays well-behaved at low n and
    # gives a meaningful std against a large, stable reference.
    kid_ref_dirs = dict(rmse_gt_dirs)

    present = {ds for ds, _, _ in discover(args.samples_dir)}

    if 'celeba' in present and args.full_celeba_ref:
        celeba_kid_ref = os.path.join(args.cache_dir, 'celeba_test_ref')
        print(f"Building full CelebA test reference (KID) -> {celeba_kid_ref}")
        materialize_celeba_test_ref(args.celeba_zip, args.celeba_partition, celeba_kid_ref)
        kid_ref_dirs['celeba'] = celeba_kid_ref

    if 'afhq' in present and args.full_afhq_ref:
        afhq_kid_ref = os.path.join(args.cache_dir, 'afhq_test_ref')
        print(f"Building full AFHQ test reference (KID) -> {afhq_kid_ref}")
        materialize_afhq_test_ref(args.afhq_zip, afhq_kid_ref, train_ratio=args.afhq_train_ratio)
        kid_ref_dirs['afhq'] = afhq_kid_ref

    # Validate all reference kinds for present datasets only.
    for ds in sorted(present):
        for kind, d in (('rmse-gt', rmse_gt_dirs[ds]),
                        ('fid-ref', fid_ref_dirs[ds]),
                        ('kid-ref', kid_ref_dirs[ds])):
            assert os.path.isdir(d) and any(
                f.lower().endswith(('.png', '.jpg')) for f in os.listdir(d)
            ), f"No images found for {ds} {kind}: {d}"
            print(f"{ds} {kind}: {d}")

    # Seed FID from previously reported values rather than recomputing it.
    fid_seed = load_fid_seed(args.fid_cache)
    if fid_seed:
        print(f"Loaded {len(fid_seed)} reported FID values from {args.fid_cache}")

    results, groups = run(
        args.samples_dir, rmse_gt_dirs, fid_ref_dirs, kid_ref_dirs, args.cache_json,
        args.kid_subset_size, args.cache_dir, fid_seed=fid_seed,
    )

    for ds in sorted({d for (d, _, _) in groups}):
        ds_groups  = {k: v for k, v in groups.items()  if k[0] == ds}
        ds_results = {k: v for k, v in results.items() if k[0] == ds}
        make_metric_vs_rmse_plot(ds_results, ds_groups, args.plots_dir,
                                 filename=f'pnp_fid_vs_rmse_{ds}.pdf',
                                 metric_key='fid', metric_label='FID $\\rightarrow$ better')
        make_metric_vs_rmse_plot(ds_results, ds_groups, args.plots_dir,
                                 filename=f'pnp_kid_vs_rmse_{ds}.pdf',
                                 metric_key='kid_mean', metric_label='KID $\\rightarrow$ better',
                                 std_key='kid_std')

    for (dataset, task, N), la_map in sorted(groups.items()):
        make_image_strip(dataset, task, N, la_map, args.samples_dir, args.plots_dir)

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'dataset':<8} {'task':<25} {'N':>5} {'la':>5} | "
          f"{'RMSE':>8} {'FID':>8} {'KID':>9} {'KID_std':>9}")
    print("-" * 75)
    for (dataset, task, N, la), m in sorted(results.items()):
        print(f"{dataset:<8} {task:<25} {N:>5} {la:>5} | "
              f"{m['rmse']:>8.4f} {m['fid']:>8.2f} "
              f"{m.get('kid_mean', float('nan')):>9.5f} {m.get('kid_std', float('nan')):>9.5f}")
    print("=" * 75)


if __name__ == '__main__':
    main()
