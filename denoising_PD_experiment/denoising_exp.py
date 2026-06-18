#!/usr/bin/env python
"""
Generate reconstructions for P-D sweep (Gaussian denoising).

Sweeps the lookahead t at multiple noise levels alpha, saves all
reconstructions as PNG, then compute metrics externally.

Usage:
    python generate_pd_sweep.py --model celeba
    python generate_pd_sweep.py --model afhq --batch_size 20
"""

import argparse
import os
import pickle
import threading
import zipfile

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Checkpoint / dataset configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "celeba": {
        "checkpoint": (
            "/home/nvidia/easy_meanflow/logs/aniso_flow_map/"
            "lsd/00054-celeba-128x128-uncond-ddpmpp-mf-gpus8-batch128-fp32/"
            "network-snapshot-087584.pkl"
        ),
        "img_size": 128,
        "channels": 3,
        "data": "celeba",
    },
    "afhq": {
        "checkpoint": (
            "/home/nvidia/easy_meanflow/logs/aniso_flow_map/"
            "lsd/00055-afhq-cat-192x192-uncond-ddpmpp-mf-gpus8-batch64-fp32/"
            "network-snapshot-035033.pkl"
        ),
        "img_size": 192,
        "channels": 3,
        "data": "afhq",
    },
}


# ---------------------------------------------------------------------------
# Dataset classes
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
        # Thread-local zip handles avoid reopening the archive on every __getitem__
        self._local = threading.local()

    def _get_zip(self):
        if not hasattr(self._local, "zip"):
            self._local.zip = zipfile.ZipFile(self.zip_path)
        return self._local.zip

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with self._get_zip().open(self.names[idx]) as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


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
            zip_path="/home/nvidia/easy_meanflow/data/celeba/celeba-128x128.zip",
            partition_file="/home/nvidia/easy_meanflow/data/celeba/list_eval_partition.txt",
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
            path="/home/nvidia/easy_meanflow/data/afhq-cat-192x192.zip",
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
    else:
        raise ValueError(f"Unknown dataset: {data}")

    loader = NextDataLoader(
        dataset, batch_size, num_workers=1, prefetch_factor=2,
        pin_memory=True, shuffle=False,
    )
    return loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate P-D sweep reconstructions")
    p.add_argument("--model", choices=list(MODEL_CONFIGS), default="celeba")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=50,
                   help="Batch size for inference (not dataset size)")
    p.add_argument("--n_points", type=int, default=15,
                   help="Number of lookahead values per noise level")
    p.add_argument("--output_root", type=str, default="./pd_sweep_images")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    seed = args.seed
    batch_size = args.batch_size
    n_points = args.n_points
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
    output_root = os.path.join(args.output_root, args.model)

    # Load model and full test set
    net, cfg = load_model(args.model, device)
    dataset = load_dataset(cfg, args.batch_size)
    chunks = []
    count = 0
    N = 500  # adjust as needed
    for batch, _ in dataset:
        chunks.append(batch)
        count += batch.shape[0]
        print(f"  loaded {count} images...", end='\r')
        if count >= N:
            break
    clean_img = torch.cat(chunks, dim=0)[:N].to(device)
    N = clean_img.shape[0]
    print(f"Loaded {N} test images ({cfg['data']}, {cfg['img_size']}x{cfg['img_size']})")

    # Save ground truth (once)
    gt_dir = os.path.join(output_root, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    clean_01 = torch.clamp((clean_img + 1) / 2, 0, 1)
    for i in range(N):
        save_image(clean_01[i], os.path.join(gt_dir, f'{i:05d}.png'))
    print(f"Saved {N} ground truth images to {gt_dir}")

    # Sweep
    alphas = [0.1, 0.3, 0.5, 0.7]

    for alpha in alphas:
        s_val = 1.0 - alpha

        torch.manual_seed(seed)
        eps = torch.randn_like(clean_img)
        noisy_img = alpha * eps + s_val * clean_img

        lookaheads = np.linspace(0, alpha, n_points)

        print(f"\n=== alpha={alpha}, s={s_val} ===")

        for la in lookaheads:
            t_val = s_val + la
            t_str = f't_{t_val:.3f}'
            save_dir = os.path.join(output_root, f'alpha_{alpha}', t_str)
            os.makedirs(save_dir, exist_ok=True)

            # Generate all reconstructions in batches
            idx = 0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_noisy = noisy_img[start:end]
                s_batch = torch.ones(end - start, device=device) * s_val
                t_batch = torch.clamp(s_batch + la, max=1.0)

                with torch.no_grad():
                    v_pred = net(batch_noisy,
                                s_batch.view(-1, 1, 1, 1),
                                t_batch.view(-1, 1, 1, 1))
                    recon = batch_noisy + (1.0 - s_batch.view(-1, 1, 1, 1)) * v_pred

                recon_01 = torch.clamp((recon + 1) / 2, 0, 1)

                for j in range(recon_01.shape[0]):
                    save_image(recon_01[j],
                               os.path.join(save_dir, f'{idx:05d}.png'))
                    idx += 1

            print(f"  t={t_val:.3f} | saved {idx} images to {save_dir}")

    # Print metric commands
    print("\n\nTo compute metrics, run:")
    print("=" * 60)
    for alpha in alphas:
        s_val = 1.0 - alpha
        lookaheads = np.linspace(0, alpha, n_points)
        for la in lookaheads:
            t_val = s_val + la
            t_str = f't_{t_val:.3f}'
            rec_path = os.path.join(output_root, f'alpha_{alpha}', t_str)
            gt_path = os.path.join(output_root, 'gt')
            print(f"python compute_metrics.py "
                  f"--rec_path {rec_path} "
                  f"--gt_path {gt_path} "
                  f"--parent_ffhq_256_path /path/to/parent")


if __name__ == "__main__":
    main()