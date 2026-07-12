# Flow Map Denoisers: Traversing the Distortion–Perception Plane for Inverse Problems

Summary: This GitHub repository contains the code for our paper using flow map denoisers as prior in plug-and-play methods. We also provide a training, and we hope it serves as pytorch implementation for using flow map for solving inverse problems.

Link to paper: https://arxiv.org/abs/2606.19802

Link to checkpoits: [https://drive.google.com/drive/folders/1WRbvpDFQ4gWAJv1jP5YPaJZbhu3jWaUh?usp=sharing](https://drive.google.com/drive/u/1/folders/1iCyTTQHKakKQsZNJ9_9di68YGs2aNd6C)



This repo contains:
- the **flow-map trainer** (`train_mf.py`)
- the **PnP-Flow solver** with a tunable lookahead (`baselines/` + `run_baselines.py`)
- four reference baselines for direct comparison: OT-ODE, Flow-Priors, D-Flow, and DPS
- the **toy DP-frontier experiments** that validate the Gaussian theorem
- the **ablation pipeline** that sweeps lookahead × steps and produces FID / RMSE plots

---

## Contents
- [Setup](#setup)
- [Data](#data)
- [Training a flow map](#training-a-flow-map)
- [One-step sampling](#one-step-sampling)
- [Inverse problems with PnP-Flow](#inverse-problems-with-pnp-flow)
- [Tracing the DP frontier (the lookahead knob)](#tracing-the-dp-frontier-the-lookahead-knob)
- [Ablations](#ablations)
- [Toy experiments](#toy-experiments)
- [Metrics](#metrics)
- [Acknowledgements](#acknowledgements)

---

## Setup

```bash
git clone <this-repo>
cd flow_maps
conda env create -f environment.yml
conda activate easy_meanflow
```

GPU + CUDA toolkit are required for training and most inverse-problem experiments.

## Data

Datasets live under `data/` (gitignored) in the [StyleGAN3](https://github.com/NVlabs/stylegan3) zip layout:

```
data/
  afhq-256x256.zip
  afhq-cat-192x192.zip
  celeba/
    celeba-128x128.zip
    list_eval_partition.txt
  cifar10-32x32.zip
```

For the inverse-problem demos and qualitative figures, a small curated set lives under [`data_examples/`](data_examples/) (5 AFHQ + 5 CelebA images), so the PnP pipeline can be exercised without the full datasets.

For FID, the EDM reference statistics are at [https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/).

## Training a flow map

Flow-map training learns the *average velocity* `u(x, s, t) = (1/(t-s)) ∫_s^t v(x_τ, τ) dτ` between any pair of times `s ≤ t`. Once trained, the lookahead `t` becomes a free parameter at inference time.

The trainer is [`train_mf.py`](train_mf.py), launched via `torchrun`. The reference 8-GPU AFHQ-256 run is wrapped by [`runners/train_script_anisotropic.sh`](runners/train_script_anisotropic.sh):

```bash
bash runners/train_script_anisotropic.sh
```

which expands to

```bash
torchrun --standalone --nproc_per_node=8 train_mf.py \
    --detach_tgt=1 \
    --outdir=logs/aniso_flow_map/lsd/ \
    --data=./data/afhq-256x256.zip \
    --cond=0 --arch=ddpmpp --lr 2e-4 --batch 32 \
    --noise_dist=uniform --loss_type=lsd \
    --log_weights=1 --duration=100 --data_proportion=0.75 \
    --metrics=none
```

Key flags:

| flag | description |
|------|-------------|
| `--data` | path to dataset zip |
| `--outdir` | run directory under `logs/` |
| `--arch` | backbone (`ddpmpp` / `ncsnpp` / `adm`) |
| `--batch` | total batch size across GPUs |
| `--lr` | learning rate |
| `--loss_type` | flow-map loss variant (e.g. `lsd`) |
| `--detach_tgt` | detach the bootstrapped target velocity |
| `--noise_dist` | distribution of `(s, t)` pairs |
| `--data_proportion` | fraction of dataset used per epoch |
| `--resume` | resume from a `.pkl` snapshot |

CelebA-128 and CIFAR-10 variants are commented out at the bottom of the same script — uncomment the block you want.

## One-step sampling

Single-step generation from a trained checkpoint uses [`generate_onestep_mf.py`](generate_onestep_mf.py). Wrapper at [`generate.sh`](generate.sh):

```bash
python generate_onestep_mf.py \
    --outdir=image_experiment/out \
    --seeds=0-127 --batch=128 \
    --network=/path/to/network-snapshot.pkl
```

## Inverse problems with PnP-Flow

[`run_baselines.py`](run_baselines.py) is the single entry point for all inverse-problem experiments. It supports five solvers and eight degradations:

| solver | source |
|--------|--------|
| `pnp_flow` (ours) | inline in [run_baselines.py](run_baselines.py) |
| `ot_ode` | [baselines/ot_ode.py](baselines/ot_ode.py) |
| `flow_priors` | [baselines/flow_priors.py](baselines/flow_priors.py) |
| `d_flow` | [baselines/d_flow.py](baselines/d_flow.py) |
| `dps` | [baselines/dps.py](baselines/dps.py) |

| degradation | parameters |
|-------------|------------|
| `deblurring_gaussian` | `--blur_kernel_size`, `--blur_std` |
| `deblurring_motion` | `--blur_kernel_size`, `--blur_std` |
| `super_resolution` | `--sr_factor` |
| `inpainting_box` | `--box_size` |
| `inpainting_random` | `--mask_ratio` |
| `colorization` | — |
| `phase_retrieval` | — (nonlinear) |
| `jpeg` | `--jpeg_qf` |

Minimal invocation:

```bash
python run_baselines.py \
    --model afhq --method pnp_flow \
    --degradation super_resolution --sr_factor 4 \
    --blur_kernel_size 61 --blur_std 0.5 \
    --lookahead 1.0 --alpha 0.05 --num_steps 100 --sigma_noise 5e-2 \
    --num_batches 20
```

Pre-canned scripts under [`runners/`](runners/) — each enumerates the exact CLIs that produced the paper numbers (uncomment the line you want):

| script | task |
|--------|------|
| [run_deblurring_gaussian.sh](runners/run_deblurring_gaussian.sh) | Gaussian deblurring |
| [run_deblurrion_motion.sh](runners/run_deblurrion_motion.sh) | motion deblurring |
| [run_sr.sh](runners/run_sr.sh) | super-resolution |
| [run_inpainting.sh](runners/run_inpainting.sh) | random / box inpainting |
| [run_jpeg.sh](runners/run_jpeg.sh) | JPEG / phase retrieval |

### Poisson noise

For Poisson observation noise (only `pnp_flow` and `dps` honour it in their data-fidelity terms — the rest assume Gaussian internally), use the wrapper [`run_baselines_poisson.py`](run_baselines_poisson.py):

```bash
python run_baselines_poisson.py --peak 20 \
    --model afhq --method pnp_flow \
    --degradation deblurring_gaussian --num_steps 100 \
    --blur_kernel_size 61 --blur_std 3.0 --alpha 0.9
```

`--peak` is the photon count (lower = noisier). All other flags forward to `run_baselines.parse_args`.

## Tracing the DP frontier (the lookahead knob)

The central claim of the paper is that **a single trained flow map traces the entire DP frontier by sweeping `--lookahead`**, with no retraining or auxiliary models. Concretely:

- `--lookahead 0.0` → MMSE end of the frontier (low distortion, high LPIPS / FID)
- `--lookahead 1.0` → perceptual end (low LPIPS / FID, higher distortion)
- intermediate values → smoothly interpolated operating points

A typical sweep:

```bash
for LA in 0.0 0.2 0.5 0.8 1.0; do
  python run_baselines.py --model afhq --method pnp_flow \
      --degradation super_resolution --sr_factor 4 \
      --blur_kernel_size 61 --blur_std 0.5 \
      --lookahead $LA --alpha 0.05 --num_steps 100 --sigma_noise 5e-2 \
      --num_batches 20
done
```

Results are written under `generated_samples_ablation_steps/afhq_pnp_flow_{LA}_{task}_{N}_{sigma}/`, which the metrics scripts (below) ingest to draw the DP curve.

## Ablations

Two ablation entry points:

1. **Sweep over steps × lookahead** — [`runners/run_ablation.sh`](runners/run_ablation.sh) iterates `num_steps × lookahead` and runs `pnp_flow` on the chosen task:
   ```bash
   bash runners/run_ablation.sh
   ```
   Edit the script's two `for` loops to change the grid.

2. **Plot FID & RMSE vs steps** — [`run_ablation_steps.py`](run_ablation_steps.py) discovers `generated_samples_ablation_steps/afhq_pnp_flow_{LA}_{task}_{N}_*` folders, computes FID and RMSE per folder, caches results in `ablation_steps_cache.json`, and writes plots to `ablation_steps_plots/`:
   ```bash
   python run_ablation_steps.py \
       --samples_dir ./generated_samples_ablation_steps \
       --plots_dir ./ablation_steps_plots
   ```
   Two curves per plot (`la=0.0` and `la=1.0`) — the gap between them is the DP-frontier width at a given step budget.

A companion sweep for the §6.2 PnP-Flow experiments lives in [`pnp_metrics.py`](pnp_metrics.py).

## Toy experiments

The Gaussian case (§3 of the paper) and a 2-D Mixture-of-Gaussians validation live in [`toy_experiments/`](toy_experiments/):

- **[`toy_example.ipynb`](toy_experiments/toy_example.ipynb)** — interactive walkthrough of the flow-map / lookahead-as-DP-knob construction on 2-D toys.
- **[`mog_dp_comparison.py`](toy_experiments/mog_dp_comparison.py)** — empirical DP comparison on a 2-D MoG: the average denoiser `D_{s,t}` swept over `t ∈ [s, 1]` against the Freirich et al. optimal DP curve (`W2`-displacement interpolant between MMSE output and `p_1`).
  ```bash
  cd toy_experiments
  python mog_dp_comparison.py --seed 0
  # → mog_dp_comparison.pdf, mog_dp_results.csv
  ```
- **[`mog_dp_comparison_v2.py`](toy_experiments/mog_dp_comparison_v2.py)** — same experiment with a vectorised RK4 integrator, a coupling-validity check, and `N=5000` samples by default:
  ```bash
  python mog_dp_comparison_v2.py --n 5000 --steps 100
  ```

Both scripts require `pip install pot` for the Wasserstein computations.

## Metrics

- **PSNR / LPIPS / DISTS** for a single sample folder — [`compute_metrics.py`](compute_metrics.py):
  ```bash
  python compute_metrics.py generated_samples_256/afhq_pnp_flow_super_resolution
  ```
  Wrapped by [`runners/compute_metrics.sh`](runners/compute_metrics.sh).

- **FID** against a ground-truth folder — [`runners/run_fid.sh`](runners/run_fid.sh) extracts PNGs via [`utils/prepare_fid_folders.py`](utils/prepare_fid_folders.py) and then calls `python -m pytorch_fid`:
  ```bash
  bash runners/run_fid.sh
  ```

## Acknowledgements

This codebase builds directly on three excellent open-source projects:

- **[easy_meanflow](https://github.com/pkulwj1994/easy_meanflow)** by Weijian Luo (Humane Intelligence Lab, Xiaohongshu Inc. & Peking University — [pkulwj1994@icloud.com](mailto:pkulwj1994@icloud.com)) and Yifei Wang (Rice University — [yw251@rice.edu](mailto:yw251@rice.edu)) — the flow-map trainer and CIFAR-10 reference implementation.
- **[NVlabs/edm](https://github.com/NVlabs/edm)** — the trainer scaffolding, network architectures, and FID evaluation.
- **[nmboffi/flow-maps](https://github.com/nmboffi/flow-maps/tree/main)** — flow-map formulation and reference baselines.
- **[pnp-flow](https://github.com/annegnx/PnP-Flow/tree/main)** pnp-flow matching method, which we used as template for implementing the baselines
