"""
prepare_fid_folders.py

Given a folder of generated samples (containing batch_*_results.pt files),
extracts restored images into <output_dir>/fid_folder and the matching
ground-truth clean images into <output_dir>/fid_folder_ground_truth.

Usage:
    python prepare_fid_folders.py <samples_folder> [--output_dir <dir>]

The dataset (celeba / afhq / cifar10) is auto-detected from the folder name.
Images are saved as PNGs numbered 000000.png, 000001.png, …
"""

import argparse
import os
import sys
import zipfile

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Inline dataset helpers (copied from run_baselines.py so this script is
# self-contained and does not depend on the rest of the project being importable)
# ---------------------------------------------------------------------------

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


def load_gt_dataset(model_name: str) -> torch.utils.data.Dataset:
    """Return the test-split dataset for the given model name."""
    norm = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    if model_name == "celeba":
        return CelebAZip(
            zip_path="/home/nvidia/easy_meanflow/data/celeba/celeba-128x128.zip",
            partition_file="/home/nvidia/easy_meanflow/data/celeba/list_eval_partition.txt",
            split="test",
            transform=norm,
        )

    if model_name == "afhq":
        sys.path.insert(0, os.path.dirname(__file__))
        from training.dataset import ImageFolderDataset

        base = ImageFolderDataset(
            path="/home/nvidia/easy_meanflow/data/afhq-256x256.zip",
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
# Helpers
# ---------------------------------------------------------------------------

def tensor_to_01(t: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] tensors to [0, 1] and clamp."""
    return torch.clamp((t + 1.0) / 2.0, 0.0, 1.0)


def detect_model(folder_name: str) -> str:
    """Guess the model name from the experiment folder name."""
    base = os.path.basename(folder_name.rstrip("/"))
    for model in ("celeba", "afhq", "cifar10"):
        if base.startswith(model):
            return model
    raise ValueError(
        f"Cannot detect model from folder name {base!r}. "
        "Expected name to start with 'celeba', 'afhq', or 'cifar10'."
    )


def sorted_pt_files(folder: str):
    """Return batch_*_results.pt files sorted by batch number."""
    files = [f for f in os.listdir(folder) if f.endswith("_results.pt")]

    def _batch_num(name):
        # batch_<N>_results.pt
        parts = name.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    return [os.path.join(folder, f) for f in sorted(files, key=_batch_num)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples_folder", help="Path to the generated-samples experiment folder")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to create fid_folder and fid_folder_ground_truth "
             "(default: same parent as samples_folder)",
    )
    parser.add_argument(
        "--full_gt",
        action="store_true",
        help="Save the entire test dataset to fid_folder_ground_truth instead of "
             "only the images matching the number of restored samples.",
    )
    parser.add_argument(
        "--skip_gt",
        action="store_true",
        help="Skip ground-truth folder creation entirely (reuse an existing one).",
    )
    args = parser.parse_args()

    samples_folder = os.path.abspath(args.samples_folder)
    if not os.path.isdir(samples_folder):
        sys.exit(f"Error: {samples_folder} is not a directory")

    output_dir = args.output_dir or os.path.dirname(samples_folder)
    fid_restored = os.path.join(output_dir, "fid_folder")
    fid_gt       = os.path.join(output_dir, "fid_folder_ground_truth")
    os.makedirs(fid_restored, exist_ok=True)
    os.makedirs(fid_gt,       exist_ok=True)

    model_name = detect_model(samples_folder)
    print(f"Detected model/dataset: {model_name}")

    pt_files = sorted_pt_files(samples_folder)
    if not pt_files:
        sys.exit(f"No batch_*_results.pt files found in {samples_folder}")
    print(f"Found {len(pt_files)} batch file(s)")

    # --- Pass 1: collect all restored tensors and count images ---------------
    restored_batches = []
    total = 0
    for pt_path in pt_files:
        d = torch.load(pt_path, map_location="cpu")
        restored = tensor_to_01(d["restored"])  # [B, 3, H, W]
        restored_batches.append(restored)
        total += restored.shape[0]

    print(f"Total restored images: {total}")

    # --- Save restored images ------------------------------------------------
    img_idx = 0
    for restored in restored_batches:
        for i in range(restored.shape[0]):
            out_path = os.path.join(fid_restored, f"{img_idx:06d}.png")
            save_image(restored[i], out_path)
            img_idx += 1
    print(f"Saved {img_idx} restored images → {fid_restored}")

    # --- Load ground-truth dataset and save images ---------------------------
    if args.skip_gt:
        print(f"Skipping ground-truth folder (reusing existing {fid_gt})")
        print("Done.")
        return


    print(f"Loading {model_name} test dataset …")
    dataset = load_gt_dataset(model_name)
    n_gt = len(dataset) if args.full_gt else min(total, len(dataset))
    if not args.full_gt and total > len(dataset):
        print(
            f"Warning: requested {total} GT images but dataset only has {len(dataset)}; "
            "saving all available."
        )

    for i in range(n_gt):
        img_tensor, _ = dataset[i]            # [-1, 1]
        img_01 = tensor_to_01(img_tensor)
        out_path = os.path.join(fid_gt, f"{i:06d}.png")
        save_image(img_01, out_path)
        if (i + 1) % 100 == 0 or (i + 1) == n_gt:
            print(f"  GT {i + 1}/{n_gt}", end="\r", flush=True)

    print(f"\nSaved {n_gt} ground-truth images → {fid_gt}")
    print("Done.")


if __name__ == "__main__":
    main()
