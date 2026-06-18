#!/usr/bin/env python3
"""
mog_dp_comparison_v2.py  —  RK4 integrator, N=5000, coupling check, vectorized ODE

Three improvements over v1:
  1. Vectorized batch ODE (all N pts in one numpy pass, ~100x faster than scalar loop)
     → makes RK4 + N=5000 feasible in minutes instead of hours
  2. RK4 instead of Euler (4th-order, n_steps=100 >> Euler n_steps=200 in accuracy)
     → fixes the t=1 endpoint artifact seen at s=0.7 in v1
  3. Coupling sanity check (Check 2b): W2²({x_1^(π(i))}, fresh p_1) ≈ 0
     → confirms the OT plan is a valid permutation of a p_1 sample
  4. N=5000 default: finite-sample W2² bias scales as O(1/sqrt(N));
     going from 2000→5000 shrinks it by ~0.63×, exposing whether the
     interior DP gap at s=0.5 is real or sampling noise

Usage:
    python mog_dp_comparison_v2.py [--seed S] [--n N] [--steps K]
    python mog_dp_comparison_v2.py --n 2000 --steps 100   # quick cross-check

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
parser.add_argument("--seed",  type=int, default=0,    help="global random seed")
parser.add_argument("--n",     type=int, default=5000, help="sample size N")
parser.add_argument("--steps", type=int, default=100,  help="RK4 ODE steps (100 >> Euler 200)")
args = parser.parse_args()
np.random.seed(args.seed)

# ─── MoG parameters (identical to toy_example / v1) ──────────────────────────
mu_a    = np.array([4.5, 3.5])
mu_b    = np.array([3.5, 5.5])
sigma_c = 0.35

N           = args.n
N_STEPS_ODE = args.steps
S_VALUES    = [0.3, 0.5, 0.7]

# ─── Scalar MoG (reference, used only in sanity check 1) ─────────────────────

def mog_denoiser_scalar(x, t):
    sig_t_sq = (1 - t)**2 + t**2 * sigma_c**2
    da = x - t * mu_a;  db = x - t * mu_b
    lwa = -0.5 * np.sum(da**2) / sig_t_sq
    lwb = -0.5 * np.sum(db**2) / sig_t_sq
    m   = max(lwa, lwb)
    wa  = np.exp(lwa - m);  wb = np.exp(lwb - m)
    wa /= (wa + wb);        wb  = 1.0 - wa
    A   = t * sigma_c**2 / sig_t_sq
    return wa * (mu_a + A*(x - t*mu_a)) + wb * (mu_b + A*(x - t*mu_b))


# ─── Vectorized batch functions ───────────────────────────────────────────────

def _denoiser_batch(X, t):
    """MMSE denoiser for batch X (N,2) → (N,2), all numpy, no Python loop."""
    sig_t_sq = (1 - t)**2 + t**2 * sigma_c**2
    da  = X - t * mu_a                                       # (N,2)
    db  = X - t * mu_b
    lwa = -0.5 * np.einsum("ni,ni->n", da, da) / sig_t_sq   # (N,)
    lwb = -0.5 * np.einsum("ni,ni->n", db, db) / sig_t_sq
    m   = np.maximum(lwa, lwb)
    wa  = np.exp(lwa - m);  wb = np.exp(lwb - m)
    wa /= (wa + wb);        wb  = 1.0 - wa
    A   = t * sigma_c**2 / sig_t_sq
    pa  = mu_a + A * (X - t * mu_a)                         # (N,2)
    pb  = mu_b + A * (X - t * mu_b)
    return wa[:, None] * pa + wb[:, None] * pb


def _vel_batch(X, t):
    t = min(t, 0.9999)
    return (_denoiser_batch(X, t) - X) / (1.0 - t)


def _rk4_batch(X_start, s, t, n_steps):
    """RK4 batch integration from time s to t.  X_start: (N,2)."""
    X  = X_start.copy()
    dt = (t - s) / n_steps
    for i in range(n_steps):
        tau = s + i * dt
        th  = min(tau + 0.5*dt, 0.9999)   # midpoint time (capped)
        te  = min(tau +     dt, 0.9999)   # endpoint time (capped)
        k1  = _vel_batch(X,              tau)
        k2  = _vel_batch(X + 0.5*dt*k1, th)
        k3  = _vel_batch(X + 0.5*dt*k2, th)
        k4  = _vel_batch(X +     dt*k3, te)
        X  += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return X


def flow_map_batch(X, s, t, n_steps=None):
    """D_{s,t} applied to batch X: (N,2) → (N,2)."""
    if n_steps is None:
        n_steps = N_STEPS_ODE
    if np.abs(t - s) < 1e-10:
        return _denoiser_batch(X, s)
    X_t   = _rk4_batch(X, s, t, n_steps)
    v_avg = (X_t - X) / (t - s)
    return X + (1.0 - s) * v_avg


def sample_p1(n, rng):
    labels = rng.binomial(1, 0.5, n)
    return np.where(labels[:, None],
                    mu_b + sigma_c * rng.standard_normal((n, 2)),
                    mu_a + sigma_c * rng.standard_normal((n, 2)))


# ─── W2² and OT coupling ──────────────────────────────────────────────────────

_OT_ITERS = 10_000_000   # default 100k is too small for N=5000; 10M is safe


def w2_squared(A, B):
    n = len(A)
    M = ot.dist(A, B, metric="sqeuclidean")
    return float(ot.emd2(np.ones(n)/n, np.ones(n)/n, M, numItermax=_OT_ITERS))


def ot_permutation(A, B):
    """W2-optimal coupling → argmax permutation index."""
    n = len(A)
    M = ot.dist(A, B, metric="sqeuclidean")
    T = ot.emd(np.ones(n)/n, np.ones(n)/n, M, numItermax=_OT_ITERS)
    return np.argmax(T, axis=1)


# ─── Sanity checks ────────────────────────────────────────────────────────────

def check1_denoiser_identity(s):
    """Batch D_{s,s} must match scalar mog_denoiser to machine precision."""
    rng    = np.random.RandomState(9999)
    X      = rng.randn(50, 2) * 0.5 + 4.0
    batch  = flow_map_batch(X, s, s)
    scalar = np.array([mog_denoiser_scalar(x, s) for x in X])
    max_e  = float(np.max(np.linalg.norm(batch - scalar, axis=1)))
    status = "PASS" if max_e < 1e-4 else "WARN"
    print(f"  [Check 1] s={s:.1f}: max ||D_{{s,s}}(batch) - MMSE(scalar)|| = {max_e:.2e}  [{status}]")


def check2b_coupling(hat_x0, x1_ot, pi, x1_chk, s):
    """
    x1_ot[pi] is a permutation of x1_ot (a p1 sample), so
    W2²(x1_ot[pi], fresh_p1) should equal W2²(x1_ot, fresh_p1)
    to within sampling noise.  Both should be near 0 for large N.
    """
    w2_perm = w2_squared(x1_ot[pi], x1_chk)
    w2_raw  = w2_squared(x1_ot,     x1_chk)
    delta   = abs(w2_perm - w2_raw)
    status  = "PASS" if delta < 0.01 else "WARN"
    print(f"  [Check 2b] s={s:.1f}: W2²(x1_ot[π], p1_fresh)={w2_perm:.5f}  "
          f"W2²(x1_ot, p1_fresh)={w2_raw:.5f}  Δ={delta:.2e}  [{status}]")


# ─── Per-s experiment ─────────────────────────────────────────────────────────

def run_experiment(s, seed_base):
    print(f"\n{'─'*60}")
    print(f"  s = {s}  (N={N}, RK4 n_steps={N_STEPS_ODE})")
    print(f"{'─'*60}")

    rng_gt  = np.random.RandomState(seed_base * 1000 + 10)
    rng_eps = np.random.RandomState(seed_base * 1000 + 20)
    rng_ref = np.random.RandomState(seed_base * 1000 + 30)
    rng_ot  = np.random.RandomState(seed_base * 1000 + 40)
    rng_chk = np.random.RandomState(seed_base * 1000 + 50)

    x1_gt  = sample_p1(N, rng_gt)
    x_s    = s * x1_gt + (1 - s) * rng_eps.standard_normal((N, 2))
    x1_ref = sample_p1(N, rng_ref)   # fixed reference for all W2² calls this s
    x1_chk = sample_p1(N, rng_chk)   # separate sample used only in check 2b

    # ── Average denoiser curve: sweep t ∈ [s, 1.0] ──────────────────────────
    t_grid   = np.round(np.clip(np.arange(s, 1.001, 0.05), s, 1.0), 10)
    avg_rows = []

    print(f"  Average denoiser: {len(t_grid)} t-values")
    for t in t_grid:
        t0    = time.time()
        hat_x = flow_map_batch(x_s, s, t)
        mse   = float(np.mean(np.sum((hat_x - x1_gt)**2, axis=1)))
        w2sq  = w2_squared(hat_x, x1_ref)
        avg_rows.append({"t": float(t), "mse": mse, "w2_squared": w2sq})
        print(f"    t={t:.2f}: MSE={mse:.5f}  W2²={w2sq:.5f}  ({time.time()-t0:.1f}s)")

    # ── Freirich optimal curve: W2-geodesic from MMSE dist → p1 ─────────────
    hat_x0 = _denoiser_batch(x_s, s)     # MMSE outputs, no ODE needed
    x1_ot  = sample_p1(N, rng_ot)

    print(f"  Freirich OT coupling ({N}×{N}) ...", end="", flush=True)
    t0  = time.time()
    pi  = ot_permutation(hat_x0, x1_ot)
    x1m = x1_ot[pi]
    print(f" done ({time.time()-t0:.1f}s)")

    # Check 2b immediately after coupling
    check2b_coupling(hat_x0, x1_ot, pi, x1_chk, s)

    alpha_grid = np.round(np.clip(np.arange(0.0, 1.001, 0.05), 0.0, 1.0), 10)
    frei_rows  = []

    print(f"  Freirich: {len(alpha_grid)} alpha-values")
    for alpha in alpha_grid:
        hat_xa = alpha * hat_x0 + (1.0 - alpha) * x1m
        mse    = float(np.mean(np.sum((hat_xa - x1_gt)**2, axis=1)))
        w2sq   = w2_squared(hat_xa, x1_ref)
        frei_rows.append({"alpha": float(alpha), "mse": mse, "w2_squared": w2sq})
        print(f"    alpha={alpha:.2f}: MSE={mse:.5f}  W2²={w2sq:.5f}")

    # ── Endpoint checks 2 & 3 ───────────────────────────────────────────────
    a_mmse = avg_rows[0];   f_mmse = frei_rows[-1]   # t=s  / α=1
    a_perf = avg_rows[-1];  f_perf = frei_rows[0]    # t=1  / α=0

    print(f"\n  [Check 2] MMSE endpoint (t=s vs α=1):")
    print(f"    avg  (t=s) : MSE={a_mmse['mse']:.5f}  W2²={a_mmse['w2_squared']:.5f}")
    print(f"    Frei (α=1) : MSE={f_mmse['mse']:.5f}  W2²={f_mmse['w2_squared']:.5f}")
    print(f"    gaps: ΔMSE={abs(a_mmse['mse']-f_mmse['mse']):.2e}  "
          f"ΔW2²={abs(a_mmse['w2_squared']-f_mmse['w2_squared']):.2e}")

    print(f"\n  [Check 3] Perfect-perception endpoint (t=1 vs α=0):")
    print(f"    avg  (t=1) : MSE={a_perf['mse']:.5f}  W2²={a_perf['w2_squared']:.5f}")
    print(f"    Frei (α=0) : MSE={f_perf['mse']:.5f}  W2²={f_perf['w2_squared']:.5f}")
    print(f"    gaps: ΔMSE={abs(a_perf['mse']-f_perf['mse']):.2e}  "
          f"ΔW2²={abs(a_perf['w2_squared']-f_perf['w2_squared']):.2e}  (residual ODE error)")

    return avg_rows, frei_rows


# ─── Main ─────────────────────────────────────────────────────────────────────

t_wall = time.time()
print("=" * 60)
print(f"MoG DP-curve comparison v2  (N={N}, RK4 n_steps={N_STEPS_ODE}, seed={args.seed})")
print("=" * 60)

print("\nSanity check 1: D_{s,s}(batch) ≈ mog_denoiser(scalar)")
for s in S_VALUES:
    check1_denoiser_identity(s)

all_avg  = {}
all_frei = {}
for s in S_VALUES:
    all_avg[s], all_frei[s] = run_experiment(s, seed_base=args.seed)

# ─── CSV ──────────────────────────────────────────────────────────────────────
csv_path = "mog_dp_results_v2.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["s", "method", "param", "mse", "w2_squared"])
    writer.writeheader()
    for s in S_VALUES:
        for r in all_avg[s]:
            writer.writerow({"s": s, "method": "avg_denoiser",
                             "param": r["t"],     "mse": r["mse"], "w2_squared": r["w2_squared"]})
        for r in all_frei[s]:
            writer.writerow({"s": s, "method": "freirich_optimal",
                             "param": r["alpha"], "mse": r["mse"], "w2_squared": r["w2_squared"]})
print(f"\nSaved: {csv_path}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
C_AVG  = "#1A73E8"
C_FREI = "#E05C3A"

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

for ax, s in zip(axes, S_VALUES):
    avg_r  = all_avg[s]
    frei_r = all_frei[s]
    aw = [r["w2_squared"] for r in avg_r]
    am = [r["mse"]        for r in avg_r]
    fw = [r["w2_squared"] for r in frei_r]
    fm = [r["mse"]        for r in frei_r]

    ax.plot(aw, am, "-o", color=C_AVG,  lw=2, ms=5, zorder=3,
            label=r"Avg denoiser $D_{s,t}$")
    ax.plot(fw, fm, "-s", color=C_FREI, lw=2, ms=5, zorder=3,
            label="Optimal DP (Freirich)")

    # Shared MMSE endpoint
    ax.scatter([aw[0]], [am[0]], s=160, marker="*", color="#111111",
               zorder=6, label="MMSE (t=s, α=1)")
    # Perfect-perception endpoints (triangles, one per curve)
    ax.scatter([aw[-1]], [am[-1]], s=90, marker="^", color=C_AVG,
               zorder=6, edgecolors="k", linewidths=0.6)
    ax.scatter([fw[0]],  [fm[0]],  s=90, marker="^", color=C_FREI,
               zorder=6, edgecolors="k", linewidths=0.6)

    ax.annotate("MMSE\n(t=s, α=1)", xy=(aw[0], am[0]),
                xytext=(0.62, 0.08), textcoords="axes fraction",
                fontsize=7.5, ha="center", color="#333333",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))
    ax.annotate("Perf. perc.\n(t=1, α=0)",
                xy=(min(aw[-1], fw[0]), max(am[-1], fm[0])),
                xytext=(0.28, 0.92), textcoords="axes fraction",
                fontsize=7.5, ha="center", color="#333333",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    ax.set_xlabel(r"Perception  $W_2^2(\hat{p},\,p_1)$  [lower = better]", fontsize=9)
    ax.set_ylabel(r"Distortion  MSE  [lower = better]", fontsize=9)
    ax.set_title(f"$s = {s}$", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, frameon=True, edgecolor="#CCCCCC", loc="upper right", handlelength=1.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
        sp.set_color("#AAAAAA")

# plt.suptitle(
#     r"DP curve: $D_{s,t}$ vs.\ Freirich optimal  (2D MoG, RK4, $N=" + str(N) + r"$)",
#     fontsize=12, fontweight="bold", y=1.03,
# )
plt.tight_layout()
fig.savefig("mog_dp_comparison_v2.pdf", bbox_inches="tight", dpi=300)
print("Saved: mog_dp_comparison_v2.pdf")

print(f"\nTotal runtime: {time.time()-t_wall:.1f}s")
