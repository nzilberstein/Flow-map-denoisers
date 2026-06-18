"""
Comparison grid plotter.
Each column = one method (or degradation / ground truth).
Each row = one image sample.

Configure the sections below, then run:
    python plot_comparison.py
"""

import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

BASE = "/home/nvidia/easy_meanflow/generated_samples_ablation_steps"
BASE_FID = "/home/nvidia/easy_meanflow/generated_samples_256"

# List of (folder_name_under_BASE, display_label).
# The 'noisy' key from the FIRST entry is used as the degradation column.
degradation = "phase_retrieval"
EXPERIMENTS = [
#     ("afhq_d_flow_deblurring_motion", "D-Flow"),
#     ("afhq_flow_priors_deblurring_motion", "Flow Priors"),
#     ("afhq_ot_ode_deblurring_motion", "OT-ODE"),
    # ("afhq_dps_" + degradation, "DPS"),
    ("afhq_pnp_flow_0.0_" + degradation + "_1000_0.01", "PnP-Flow (t=s)"),
    ("afhq_pnp_flow_1.0_" + degradation + "_1000_0.01", "PnP-Flow (t=1.0)"),
#     # ("afhq_ot_ode_deblurring_gaussian_100",                        "OT-ODE"),
# ]
# BASE = "/home/nvidia/easy_meanflow/generated_samples_128"
# BASE_FID = "/home/nvidia/easy_meanflow/generated_samples_128"
# degradation = "deblurring_motion"
# EXPERIMENTS = [
    # ("afhq_pnp_flow_1.0_phase_retrieval_500_0.001", "PnP-Flow (t=1.0)"),
    # ("afhq_pnp_flow_0.0_phase_retrieval_500_0.001", "PnP-Flow (t=0.0)"),
    # ("celeba_d_flow_" + degradation, "D-Flow"),
    # ("celeba_flow_priors_" + degradation, "Flow Priors"),
    # ("celeba_ot_ode_" + degradation, "OT-ODE"),
    # ("celeba_dps_" + degradation, "DPS"),
    # ("celeba_pnp_flow_0.0_" + degradation + "_50_0.05_test", "PnP-Flow (t=s)"),
    # ("celeba_pnp_flow_1.0_" + degradation + "_50_0.05", "PnP-Flow (t=1.0)"),
    # ("celeba_ot_ode_super_resolution_100",                        "OT-ODE"),
]


# Ground truth: path to a folder of sequentially numbered PNGs (000000.png …).
# Set to None to skip the ground truth column.
GROUND_TRUTH_DIR = os.path.join(BASE_FID, "fid_folder_ground_truth")

# Images to display. START_IDX is the global image index (same meaning across
# methods). START_BATCH is a shortcut: start_idx = (START_BATCH - 1) * START_BATCH_UNIT.
# START_BATCH wins if set. Batch file layout is auto-detected per folder, so
# D-Flow (1 image/file) and others (5/file) are handled transparently.
START_IDX = 0
START_BATCH = 80
START_BATCH_UNIT = 1   # images-per-batch used by START_BATCH (typical run size)
NUM_IMAGES = 5       # how many images to show (stacked vertically per column)

# Output file (set to None to call plt.show() instead).
OUTPUT_FILE = "comparison.pdf"

# Display size per image cell (inches) and save DPI. Raise DPI if the saved
# file looks blurry; raise INCHES_PER_IMAGE to blow up labels along with it.
INCHES_PER_IMAGE = 3.0
SAVE_DPI = 120

# Zoom display mode:
#   "overlay" — small zoom inset in the top-right of each full image
#   "row"     — zoom crop as a full-size cell directly below each example
#   "none"    — no zoom
# ZOOM_BOX is (y0, y1, x0, x1) as fractions of image height/width.
ZOOM_MODE = "none"
# ZOOM_BOX = (0.20, 0.70, 0.25, 0.75)   # centre face crop for AFHQ portraits
ZOOM_BOX = (0.45, 0.95, 0.45, 0.95)   # centre face crop for AFHQ portraits
ZOOM_INSET_SIZE = (0.45, 0.35)        # (width, height) fractions of the cell (overlay only)

# ─── END CONFIGURATION ────────────────────────────────────────────────────────


def _batch_path(folder: str, batch_idx: int) -> str:
    return os.path.join(folder, f"batch_{batch_idx}_results.pt")


def _detect_batch_size(folder: str) -> int:
    data = torch.load(_batch_path(folder, 1), map_location="cpu", weights_only=False)
    first_tensor = next(v for v in data.values() if torch.is_tensor(v))
    return first_tensor.shape[0]


def load_tensors(folder: str, key: str, start: int, n: int, batch_size: int) -> torch.Tensor:
    """Load `n` tensors starting at global index `start` from batch .pt files."""
    chunks = []
    remaining = n
    idx = start
    while remaining > 0:
        batch_num = idx // batch_size + 1
        offset = idx % batch_size
        path = _batch_path(folder, batch_num)
        data = torch.load(path, map_location="cpu", weights_only=False)
        chunk = data[key][offset : offset + remaining]
        if len(chunk) == 0:
            raise RuntimeError(
                f"No data at idx={idx} in {path} (offset={offset}, "
                f"file has {len(data[key])} items, batch_size={batch_size}). "
                "Batch size likely mismatched."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
        idx += len(chunk)
    return torch.cat(chunks, dim=0)


def load_gt_images(gt_dir: str, start: int, n: int) -> torch.Tensor:
    """Load n ground-truth PNGs starting at global index `start` → (N,3,H,W) in [-1,1]."""
    from PIL import Image
    imgs = []
    for i in range(start, start + n):
        path = os.path.join(gt_dir, f"{i:06d}.png")
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [0,255] → [-1,1]
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(imgs)


def to_images(t: torch.Tensor) -> np.ndarray:
    """(N,3,H,W) in [-1,1] → (N,H,W,3) uint8."""
    t = t.clamp(-1.0, 1.0)
    t = (t + 1.0) * 127.5
    return t.permute(0, 2, 3, 1).to(torch.uint8).numpy()


def _zoom_bbox(img: np.ndarray):
    H, W = img.shape[:2]
    fy0, fy1, fx0, fx1 = ZOOM_BOX
    return int(fy0 * H), int(fy1 * H), int(fx0 * W), int(fx1 * W)


def _draw_zoom_rect(ax, img: np.ndarray, edge_color):
    y0, y1, x0, x1 = _zoom_bbox(img)
    rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=1.5, edgecolor=edge_color, facecolor="none")
    ax.add_patch(rect)


def _style_zoom_axes(ax, edge_color):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(2)


def _add_zoom_overlay(ax, img: np.ndarray, edge_color):
    """Overlay a zoomed crop in the top-right of ax with a coloured border."""
    y0, y1, x0, x1 = _zoom_bbox(img)
    inset_w, inset_h = ZOOM_INSET_SIZE
    ax_ins = ax.inset_axes([1 - inset_w - 0.01, 1 - inset_h - 0.01, inset_w, inset_h])
    ax_ins.imshow(img[y0:y1, x0:x1], interpolation="nearest")
    _style_zoom_axes(ax_ins, edge_color)
    _draw_zoom_rect(ax, img, edge_color)


def main():
    # columns: list of (label, (N,H,W,3) uint8 array, edge_color)
    columns = []

    first_folder = os.path.join(BASE, EXPERIMENTS[0][0])

    start_idx = (START_BATCH - 1) * START_BATCH_UNIT if START_BATCH is not None else START_IDX
    print(f"start_idx={start_idx}, num_images={NUM_IMAGES}")

    cmap = plt.cm.tab10.colors

    def _load(folder, key, start, n):
        bs = _detect_batch_size(folder)
        print(f"  {os.path.basename(folder)}: batch_size={bs}")
        return load_tensors(folder, key, start, n, bs)

    # Degradation column from the first experiment's 'noisy' key.
    noisy = _load(first_folder, "noisy", start_idx, NUM_IMAGES)
    columns.append(("Degraded", to_images(noisy), "black"))

    # One restored column per experiment.
    for i, (folder_name, label) in enumerate(EXPERIMENTS):
        folder = os.path.join(BASE, folder_name)
        restored = _load(folder, "restored", start_idx, NUM_IMAGES)
        columns.append((label, to_images(restored), cmap[i]))

    # Ground truth column.
    if GROUND_TRUTH_DIR is not None:
        gt = load_gt_images(GROUND_TRUTH_DIR, start_idx, NUM_IMAGES)
        # if gt.shape[-2:] != noisy.shape[-2:]:
        #     import torch.nn.functional as F
        #     gt = F.interpolate(gt, size=noisy.shape[-2:], mode="bilinear",
        #                        align_corners=False)
        columns.append(("Ground Truth", to_images(gt), "black"))

    n_cols = len(columns)
    rows_per_example = 2 if ZOOM_MODE == "row" else 1
    total_rows = NUM_IMAGES * rows_per_example

    fig, axes = plt.subplots(
        total_rows, n_cols,
        figsize=(n_cols * INCHES_PER_IMAGE, total_rows * INCHES_PER_IMAGE),
        squeeze=False,
    )

    for col, (label, imgs, color) in enumerate(columns):
        axes[0, col].set_title(label, fontsize=20, pad=6)
        for row in range(NUM_IMAGES):
            full_ax = axes[row * rows_per_example, col]
            full_ax.imshow(imgs[row], interpolation="nearest")
            full_ax.set_xticks([]); full_ax.set_yticks([])

            if ZOOM_MODE == "overlay":
                _add_zoom_overlay(full_ax, imgs[row], color)
            elif ZOOM_MODE == "row":
                _draw_zoom_rect(full_ax, imgs[row], color)
                y0, y1, x0, x1 = _zoom_bbox(imgs[row])
                zoom_ax = axes[row * rows_per_example + 1, col]
                zoom_ax.imshow(imgs[row][y0:y1, x0:x1], interpolation="nearest")
                _style_zoom_axes(zoom_ax, color)

    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99,
                        top=0.97, bottom=0.01)

    if OUTPUT_FILE:
        fig.savefig(OUTPUT_FILE, bbox_inches="tight", pad_inches=0.05, dpi=SAVE_DPI)
        print(f"Saved → {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
