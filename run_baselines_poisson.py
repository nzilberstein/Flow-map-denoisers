#!/usr/bin/env python
"""Thin wrapper around run_baselines.py that defaults to a Poisson noise model.

Equivalent to invoking `run_baselines.py --noise_model poisson ...`. Use
`--peak <rate>` to control the Poisson rate (lower = noisier; typical 1, 5,
20, 50). All other flags forward to `run_baselines.parse_args`.

Note: only the data-fidelity term in PnP-Flow and DPS is switched to the
Poisson NLL. OT_ODE / Flow-Priors / D-FLOW solvers still assume Gaussian
noise internally and are included for completeness.
"""

import sys

import run_baselines


def _ensure_poisson_in_argv():
    if "--noise_model" not in sys.argv:
        sys.argv.insert(1, "poisson")
        sys.argv.insert(1, "--noise_model")


if __name__ == "__main__":
    _ensure_poisson_in_argv()
    run_baselines.main()
