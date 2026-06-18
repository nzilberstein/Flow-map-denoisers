#!/usr/bin/env python3
"""
mog_dp_comparison.py

Empirical DP-curve comparison on 2D MoG (paper §3.3):
  - Average denoiser D_{s,t} swept over t ∈ [s, 1.0]
  - Freirich et al. optimal DP curve (W2-displacement interpolant
    between MMSE output distribution and p_1)

Outputs: mog_dp_comparison.pdf, mog_dp_results.csv

Usage:
    python mog_dp_comparison.py [--seed SEED]

Requirements: numpy, matplotlib, pot  (pip install pot)
"""

import argparse
import csv
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import ot
except ImportError:
    sys.exit("POT not installed.  Run:  pip install pot")

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="global random seed")
args = parser.parse_args()
np.random.seed(args.seed)

# ─── MoG parameters ───────────────────────────────────────────────────────────
mu_a    = np.array([4.5, 3.5])
mu_b    = np.array([3.5, 5.5])
sigma_c = 0.35          # within-component std

N           = 2000      # samples for estimation / perception metric
N_STEPS_ODE = 200       # ODE steps inside flow_map_denoiser
S_VALUES    = [0.3, 0.5, 0.7]

# ─── Core MoG functions (copied from toy_example) ─────────────────────────────

def mog_denoiser(x, t):
    """MMSE denoiser E[x1 | x_t=x] for 2D equal-weight MoG under linear interpolant."""
    sig_t_sq = (1 - t) ** 2 + t ** 2 * sigma_c ** 2
    diff_a   = x - t * mu_a
    diff_b   = x - t * mu_b
    log_w_a  = -0.5 * np.sum(diff_a ** 2) / sig_t_sq
    log_w_b  = -0.5 * np.sum(diff_b ** 2) / sig_t_sq
    max_log  = max(log_w_a, log_w_b)
    w_a      = np.exp(log_w_a - max_log)
    w_b      = np.exp(log_w_b - max_log)
    w_sum    = w_a + w_b
    w_a     /= w_sum
    w_b     /= w_sum
    A_t      = t * sigma_c ** 2 / sig_t_sq
    post_a   = mu_a + A_t * (x - t * mu_a)
    post_b   = mu_b + A_t * (x - t * mu_b)
    return w_a * post_a + w_b * post_b


def mog_velocity(x, t):
    t = min(t, 0.9999)
    D = mog_denoiser(x, t)
    return (D - x) / (1 - t)


def simulate_flow(x_start, s, t, n_steps=N_STEPS_ODE):
    x  = x_start.copy()
    dt = (t - s) / n_steps
    for i in range(n_steps):
        tau = s + i * dt
        x   = x + dt * mog_velocity(x, tau)
    return x


def flow_map_denoiser(x, s, t, n_steps=N_STEPS_ODE):
    """
    Flow-map denoiser D_{s,t}(x).
    At t == s returns the exact MMSE denoiser (identity shortcut).
    """
    if np.abs(t - s) < 1e-10:
        return mog_denoiser(x, s)
    x_t   = simulate_flow(x, s, t, n_steps=n_steps)
    v_avg = (x_t - x) / (t - s)
    return x + (1 - s) * v_avg


def sample_p1(n, rng):
    labels = rng.binomial(1, 0.5, n)
    return np.where(
        labels[:, None],
        mu_b + sigma_c * rng.standard_normal((n, 2)),
        mu_a + sigma_c * rng.standard_normal((n, 2)),
    )


# ─── W2² and OT coupling via POT ──────────────────────────────────────────────

def w2_squared(A, B):
    """Exact W2²(empirical A, empirical B) using POT emd2."""
    n  = len(A)
    ua = np.ones(n) / n
    ub = np.ones(n) / n
    M  = ot.dist(A, B, metric="sqeuclidean")
    return float(ot.emd2(ua, ub, M))


def ot_permutation(A, B):
    """
    Returns index array pi s.t. A[i] ↔ B[pi[i]] under the W2-optimal
    coupling between uniform empirical distributions of equal size.
    For equal-weight empirical measures the EMD plan is a permutation
    matrix scaled by 1/N; argmax per row recovers the permutation.
    """
    n  = len(A)
    ua = np.ones(n) / n
    ub = np.ones(n) / n
    M  = ot.dist(A, B, metric="sqeuclidean")
    T  = ot.emd(ua, ub, M)
    return np.argmax(T, axis=1)


# ─── Sanity check 1 ───────────────────────────────────────────────────────────

def sanity_check_identity(s):
    """D_{s,s}(x) must equal mog_denoiser(x, s) to machine precision."""
    rng   = np.random.RandomState(9999)
    batch = rng.randn(50, 2) * 0.5 + 4.0
    errs  = [
        np.linalg.norm(flow_map_denoiser(x, s, s) - mog_denoiser(x, s))
        for x in batch
    ]
    max_e  = max(errs)
    status = "PASS" if max_e < 1e-4 else "WARN (>1e-4)"
    print(f"  [Check 1] s={s:.1f}: max ||D_{{s,s}} - MMSE|| = {max_e:.2e}  [{status}]")


# ─── Per-s experiment ─────────────────────────────────────────────────────────

def run_experiment(s, seed_base):
    print(f"\n{'─' * 60}")
    print(f"  s = {s}")
    print(f"{'─' * 60}")

    rng_gt  = np.random.RandomState(seed_base * 1000 + 10)
    rng_eps = np.random.RandomState(seed_base * 1000 + 20)
    rng_ref = np.random.RandomState(seed_base * 1000 + 30)
    rng_ot  = np.random.RandomState(seed_base * 1000 + 40)

    x1_gt  = sample_p1(N, rng_gt)                           # (N,2) ground truth
    eps    = rng_eps.standard_normal((N, 2))
    x_s    = s * x1_gt + (1 - s) * eps                      # noised inputs x_s^(i)
    x1_ref = sample_p1(N, rng_ref)                          # fresh p1 for W2² metric

    # ── Average denoiser curve: sweep t ∈ [s, 1.0] ──────────────────────────
    t_grid   = np.round(np.clip(np.arange(s, 1.001, 0.05), s, 1.0), 10)
    avg_rows = []

    print(f"  Average denoiser: {len(t_grid)} t-values, {N} pts each")
    for t in t_grid:
        t0    = time.time()
        hat_x = np.array([flow_map_denoiser(x_s[i], s, t) for i in range(N)])
        mse   = float(np.mean(np.sum((hat_x - x1_gt) ** 2, axis=1)))
        w2sq  = w2_squared(hat_x, x1_ref)
        avg_rows.append({"t": float(t), "mse": mse, "w2_squared": w2sq})
        print(f"    t={t:.2f}: MSE={mse:.4f}  W2²={w2sq:.4f}  ({time.time()-t0:.1f}s)")

    # ── Freirich optimal curve: OT interpolation ─────────────────────────────
    # Step 1: MMSE outputs  D_s(x_s^(i)) = mog_denoiser(x_s^(i), s)
    hat_x0 = np.array([mog_denoiser(x_s[i], s) for i in range(N)])

    # Step 2: fresh p1 targets and W2-optimal coupling
    x1_ot  = sample_p1(N, rng_ot)
    print(f"  Freirich: computing {N}×{N} OT coupling ...", end="", flush=True)
    t0     = time.time()
    pi     = ot_permutation(hat_x0, x1_ot)
    x1m    = x1_ot[pi]          # matched p1 samples  x_1^(π(i))
    print(f" done ({time.time()-t0:.1f}s)")

    # Step 3: interpolate over alpha ∈ [0, 1]
    alpha_grid = np.round(np.clip(np.arange(0.0, 1.001, 0.05), 0.0, 1.0), 10)
    frei_rows  = []

    print(f"  Freirich: {len(alpha_grid)} alpha-values")
    for alpha in alpha_grid:
        hat_xa = alpha * hat_x0 + (1 - alpha) * x1m
        mse    = float(np.mean(np.sum((hat_xa - x1_gt) ** 2, axis=1)))
        w2sq   = w2_squared(hat_xa, x1_ref)
        frei_rows.append({"alpha": float(alpha), "mse": mse, "w2_squared": w2sq})
        print(f"    alpha={alpha:.2f}: MSE={mse:.4f}  W2²={w2sq:.4f}")

    # ── Sanity checks 2 & 3 ─────────────────────────────────────────────────
    a_mmse = avg_rows[0]    # t = s     → MMSE endpoint
    f_mmse = frei_rows[-1]  # alpha = 1 → MMSE endpoint
    a_perf = avg_rows[-1]   # t = 1     → (near) perfect perception
    f_perf = frei_rows[0]   # alpha = 0 → perfect perception

    print(f"\n  [Check 2] MMSE endpoint (t=s vs α=1) — should match to 2–3 decimals:")
    print(f"    avg  (t=s) : MSE={a_mmse['mse']:.4f}  W2²={a_mmse['w2_squared']:.4f}")
    print(f"    Frei (α=1) : MSE={f_mmse['mse']:.4f}  W2²={f_mmse['w2_squared']:.4f}")
    print(
        f"    gaps       : ΔMSE={abs(a_mmse['mse']-f_mmse['mse']):.2e}"
        f"  ΔW2²={abs(a_mmse['w2_squared']-f_mmse['w2_squared']):.2e}"
    )

    print(f"\n  [Check 3] Perfect-perception endpoint (t=1 vs α=0) — small ODE gap expected:")
    print(f"    avg  (t=1) : MSE={a_perf['mse']:.4f}  W2²={a_perf['w2_squared']:.4f}")
    print(f"    Frei (α=0) : MSE={f_perf['mse']:.4f}  W2²={f_perf['w2_squared']:.4f}")
    print(
        f"    gaps       : ΔMSE={abs(a_perf['mse']-f_perf['mse']):.2e}"
        f"  ΔW2²={abs(a_perf['w2_squared']-f_perf['w2_squared']):.2e}"
        f"  (ODE numerical error)"
    )

    return avg_rows, frei_rows


# ─── Main ─────────────────────────────────────────────────────────────────────

t_wall = time.time()

print("=" * 60)
print(f"MoG DP-curve comparison  (N={N}, n_steps={N_STEPS_ODE}, seed={args.seed})")
print("=" * 60)

# Sanity check 1
print("\nSanity check 1: D_{s,s} ≈ mog_denoiser")
for s in S_VALUES:
    sanity_check_identity(s)

# Run experiments
all_avg  = {}
all_frei = {}
for s in S_VALUES:
    all_avg[s], all_frei[s] = run_experiment(s, seed_base=args.seed)

# ─── CSV ──────────────────────────────────────────────────────────────────────
csv_path = "mog_dp_results.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["s", "method", "param", "mse", "w2_squared"])
    w.writeheader()
    for s in S_VALUES:
        for r in all_avg[s]:
            w.writerow({"s": s, "method": "avg_denoiser",
                        "param": r["t"], "mse": r["mse"], "w2_squared": r["w2_squared"]})
        for r in all_frei[s]:
            w.writerow({"s": s, "method": "freirich_optimal",
                        "param": r["alpha"], "mse": r["mse"], "w2_squared": r["w2_squared"]})
print(f"\nSaved: {csv_path}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
C_AVG   = "#1A73E8"   # blue  — average denoiser
C_FREI  = "#E05C3A"   # orange-red — Freirich optimal

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

for ax, s in zip(axes, S_VALUES):
    avg_r  = all_avg[s]
    frei_r = all_frei[s]

    aw = [r["w2_squared"] for r in avg_r]
    am = [r["mse"]        for r in avg_r]
    fw = [r["w2_squared"] for r in frei_r]
    fm = [r["mse"]        for r in frei_r]

    # Curves
    ax.plot(aw, am, "-o", color=C_AVG,  lw=2,    ms=5, zorder=3,
            label=r"Avg denoiser $D_{s,t}$")
    ax.plot(fw, fm, "-s", color=C_FREI, lw=2,    ms=5, zorder=3,
            label="Optimal DP (Freirich)")

    # Shared MMSE endpoint  (t=s / α=1) — both curves start here
    ax.scatter([aw[0]], [am[0]], s=160, marker="*", color="#111111",
               zorder=6, label="MMSE (t=s, α=1)")

    # Perfect-perception endpoints  (t=1 / α=0) — may differ slightly
    ax.scatter([aw[-1]], [am[-1]], s=90, marker="^", color=C_AVG,
               zorder=6, edgecolors="k", linewidths=0.6)
    ax.scatter([fw[0]],  [fm[0]],  s=90, marker="^", color=C_FREI,
               zorder=6, edgecolors="k", linewidths=0.6)

    # Annotations for endpoints
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.annotate(
        "MMSE\n(t=s, α=1)",
        xy=(aw[0], am[0]),
        xytext=(0.62, 0.08), textcoords="axes fraction",
        fontsize=7.5, ha="center", color="#333333",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )
    ax.annotate(
        "Perf. perc.\n(t=1, α=0)",
        xy=(min(aw[-1], fw[0]), max(am[-1], fm[0])),
        xytext=(0.28, 0.92), textcoords="axes fraction",
        fontsize=7.5, ha="center", color="#333333",
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )

    ax.set_xlabel(r"Perception  $W_2^2(\hat{p},\,p_1)$  [lower = better]", fontsize=9)
    ax.set_ylabel(r"Distortion  MSE  [lower = better]",                    fontsize=9)
    ax.set_title(f"$s = {s}$", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, frameon=True, edgecolor="#CCCCCC",
              loc="upper right", handlelength=1.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
        sp.set_color("#AAAAAA")

plt.suptitle(
    r"DP curve: average denoiser $D_{s,t}$ vs.\ Freirich et al.\ optimal  (2D MoG)",
    fontsize=12, fontweight="bold", y=1.03,
)
plt.tight_layout()
fig.savefig("mog_dp_comparison.pdf", bbox_inches="tight", dpi=300)
print("Saved: mog_dp_comparison.pdf")

print(f"\nTotal runtime: {time.time() - t_wall:.1f}s")
