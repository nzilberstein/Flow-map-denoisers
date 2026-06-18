"""
compute_metrics.py

Compute PSNR, LPIPS, and DISTS between restored images in a generated_samples
subfolder and the corresponding ground-truth images from the dataset.

Ground-truth images are drawn from the test split of the dataset in index
order, matching the order used by run_baselines.py / prepare_fid_folders.py.

Usage:
    python compute_metrics.py generated_samples/celeba_flow_priors_deblurring
    python compute_metrics.py generated_samples/afhq_pnp_flow_1.0_inpainting_random_300
"""

import argparse
import os
import sys
import zipfile

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

import lpips as lpips_lib
from DISTS_pytorch import DISTS


# ---------------------------------------------------------------------------
# Dataset helpers (mirrors prepare_fid_folders.py / run_baselines.py)
# ---------------------------------------------------------------------------

class CelebAZip(torch.utils.data.Dataset):
    def __init__(self, zip_path, partition_file, split="test", transform=None, max_items=None):
        self.zip_path = zip_path
        self.transform = transform
        split_map = {"train": 0, "val": 1, "test": 2}
        split_id = split_map[split]
        with open(partition_file) as f:
            entries = [line.strip().split() for line in f]
        valid = {e[0].replace(".jpg", ".png") for e in entries if int(e[1]) == split_id}
        with zipfile.ZipFile(zip_path) as z:
            self.names = sorted(n for n in z.namelist() if n in valid)
        if max_items is not None:
            self.names = self.names[:max_items]
        self._zf = None

    def _zip(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with self._zip().open(self.names[idx]) as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def load_gt_dataset(model_name: str, max_items: int | None = None) -> torch.utils.data.Dataset:
    norm = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if model_name == "celeba":
        return CelebAZip(
            zip_path="/home/nvidia/flow_maps/data/celeba/celeba-128x128.zip",
            partition_file="/home/nvidia/flow_maps/data/celeba/list_eval_partition.txt",
            split="test",
            transform=norm,
            max_items=max_items,
        )

    if model_name == "afhq":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

        return _AFHQWrapper()

    if model_name == "cifar10":
        from torchvision.datasets import CIFAR10
        return CIFAR10(
            "data", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        )

    raise ValueError(f"Unknown model/dataset: {model_name!r}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def detect_model(folder_name: str) -> str:
    base = os.path.basename(folder_name.rstrip("/"))
    for model in ("celeba", "afhq", "cifar10"):
        if base.startswith(model):
            return model
    raise ValueError(
        f"Cannot detect model from folder name {base!r}. "
        "Expected name to start with 'celeba', 'afhq', or 'cifar10'."
    )


def sorted_pt_files(folder: str):
    files = [f for f in os.listdir(folder) if f.endswith("_results.pt")]

    def _batch_num(name):
        parts = name.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    return [os.path.join(folder, f) for f in sorted(files, key=_batch_num)]


def to_01(t: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] tensor to [0, 1] and clamp."""
    return torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """x, y in [0, 1], shape [B, C, H, W]. Returns mean PSNR over batch."""
    mse = ((x - y) ** 2).mean(dim=[1, 2, 3])           # [B]
    psnr = -10.0 * torch.log10(mse + 1e-10)
    return psnr.mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_folder", help="Path to a generated_samples subfolder")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    folder = os.path.abspath(args.samples_folder)
    if not os.path.isdir(folder):
        sys.exit(f"Error: {folder} is not a directory")

    device = torch.device(args.device)
    model_name = detect_model(folder)
    print(f"Detected model/dataset : {model_name}")
    print(f"Device                 : {device}")

    pt_files = sorted_pt_files(folder)
    if not pt_files:
        sys.exit(f"No batch_*_results.pt files found in {folder}")

    # Count total restored images so we only load that many GT images.
    total_restored = 0
    for pt_path in pt_files:
        d = torch.load(pt_path, map_location="cpu")
        total_restored += d["restored"].shape[0]
    print(f"Found {len(pt_files)} batch file(s), {total_restored} restored images\n")

    # Load metric models
    # print("Loading LPIPS (AlexNet) …")
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    # print("Loading DISTS …")
    dists_fn = DISTS().to(device)
    dists_fn.eval()

    # Load ground-truth dataset
    # print(f"Loading {model_name} test dataset …\n")
    dataset = load_gt_dataset(model_name, max_items=total_restored)

    psnr_list, lpips_list, dists_list = [], [], []
    gt_idx = 0

    for pt_path in pt_files:
        batch_num = os.path.basename(pt_path).split("_")[1]
        d = torch.load(pt_path, map_location="cpu")
        restored = to_01(d["restored"])          # [B, 3, H, W] in [0, 1]
        B = restored.shape[0]

        if gt_idx + B > len(dataset):
            print(f"Warning: ran out of GT images at batch {batch_num}, stopping early.")
            break

        # Collect matching GT images
        gt_imgs = torch.stack([to_01(dataset[gt_idx + i][0]) for i in range(B)])  # [B, 3, H, W]
        gt_idx += B

        restored_dev = restored.to(device)
        gt_dev = gt_imgs.to(device)

        with torch.no_grad():
            # PSNR
            batch_psnr = compute_psnr(restored_dev, gt_dev)

            # LPIPS expects [-1, 1] input
            restored_m1p1 = restored_dev * 2 - 1
            gt_m1p1 = gt_dev * 2 - 1
            batch_lpips = lpips_fn(restored_m1p1, gt_m1p1).mean().item()

            # DISTS expects [0, 1] input
            batch_dists = dists_fn(restored_dev, gt_dev).mean().item()

        psnr_list.append(batch_psnr)
        lpips_list.append(batch_lpips)
        dists_list.append(batch_dists)

        # print(
        #     f"  batch {batch_num:>4s}  PSNR={batch_psnr:.2f} dB  "
        #     f"LPIPS={batch_lpips:.4f}  DISTS={batch_dists:.4f}"
        # )

    if not psnr_list:
        sys.exit("No batches processed.")

    psnr_t = torch.tensor(psnr_list)
    lpips_t = torch.tensor(lpips_list)
    dists_t = torch.tensor(dists_list)

    print()
    print("=" * 60)
    print(f"SUMMARY  ({len(psnr_list)} batches, {gt_idx} images total)")
    print(f"Folder: {folder}")
    print("=" * 60)
    print(f"  PSNR  : {psnr_t.mean():.3f} ± {psnr_t.std():.3f} dB")
    print(f"  LPIPS : {lpips_t.mean():.4f} ± {lpips_t.std():.4f}")
    print(f"  DISTS : {dists_t.mean():.4f} ± {dists_t.std():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
