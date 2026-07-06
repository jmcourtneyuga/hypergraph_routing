#!/usr/bin/env python3
"""Paper III, Table 5.3 cross-seed bands.

The original lp_random_voltage_sweep.py uses a single master seed (42)
to draw 500 voltage assignments per L value and reports the fraction
Ramanujan as a point estimate. This wrapper re-runs that procedure
across multiple master seeds and reports the cross-seed mean ± std
of (frac_ram, mean beta_lift) for each L. This closes the
``PAPER3_FINAL_AUDIT_VERDICT.md`` finding about hard-coded RNG seeds
in Table 5.3.

Reuses ``lift_spectrum_check`` and ``biregular_3x5_mask`` from
lp_random_voltage_sweep.py without modifying that script (golden
behavior preserved for the original 33-line output).

USAGE:
    python experiments/paper3/lp_voltage_bands.py                  # default 5 seeds
    python experiments/paper3/lp_voltage_bands.py --master-seeds 42 43 44 45 46
    python experiments/paper3/lp_voltage_bands.py --n-samples 1000  # tighter bands
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.paper3.lp_random_voltage_sweep import (
    biregular_3x5_mask,
    lift_spectrum_check,
)


def sweep_with_seed(M, L_values, n_samples, master_seed):
    """Single-seed sweep. Returns dict {L: (frac_ram, mean_beta_lift)}."""
    rng = np.random.default_rng(master_seed)
    out = {}
    for L in L_values:
        beta_lifts = []
        is_rams = []
        for _ in range(n_samples):
            res = lift_spectrum_check(M, L, rng)
            beta_lifts.append(res['beta_lift'])
            is_rams.append(res['is_ramanujan'])
        out[L] = (sum(is_rams) / len(is_rams), float(np.mean(beta_lifts)))
    return out


def run_bands(master_seeds, L_values, n_samples):
    M = biregular_3x5_mask()
    print("=" * 90)
    print(f"LP voltage-sweep cross-seed bands: {len(master_seeds)} master "
          f"seeds * {n_samples} samples per L per seed")
    print(f"master seeds: {master_seeds}")
    print(f"L values:     {L_values}")
    print("=" * 90)

    all_runs = []
    for s in master_seeds:
        t0 = time.time()
        out = sweep_with_seed(M, L_values, n_samples, s)
        elapsed = time.time() - t0
        all_runs.append(out)
        fr_summary = ", ".join(f"L={L}:{out[L][0]:.2%}" for L in L_values)
        print(f"  seed={s}: {fr_summary}  ({elapsed:.1f}s)")

    print()
    print("=" * 90)
    print("CROSS-SEED BANDS (mean +/- std over {} seeds)".format(len(master_seeds)))
    print("=" * 90)
    print(f"  {'L':>3}  {'N_LP':>6}  {'frac Ram (mean +/- std)':>25}  "
          f"{'mean beta_lift (mean +/- std)':>32}  {'frac min':>9}  {'frac max':>9}")
    print(f"  {'-' * 90}")
    for L in L_values:
        N_LP = (5 * L) ** 2 + (3 * L) ** 2
        fracs = np.array([r[L][0] for r in all_runs])
        betas = np.array([r[L][1] for r in all_runs])
        f_mean, f_std = fracs.mean(), fracs.std(ddof=1) if len(fracs) > 1 else 0.0
        b_mean, b_std = betas.mean(), betas.std(ddof=1) if len(betas) > 1 else 0.0
        print(f"  {L:>3}  {N_LP:>6}  "
              f"{f_mean:>9.2%} +/- {f_std:>9.2%}  "
              f"{b_mean:>12.4f}  +/- {b_std:>12.4f}  "
              f"{fracs.min():>8.2%}  {fracs.max():>8.2%}")

    print()
    print("Interpretation: the cross-seed std on frac Ramanujan is the")
    print("statistical uncertainty in the table 5.3 numbers. If <2pp the")
    print("single-seed point estimate (Table 5.3 as drafted) is robust.")
    print("If >5pp Table 5.3 should be reported as mean +/- std.")

    return all_runs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master-seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    p.add_argument("--n-samples", type=int, default=500,
                    help="voltage samples per L per seed (default: 500, "
                    "matching lp_random_voltage_sweep.py)")
    p.add_argument("--L", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 8, 12, 16])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bands(args.master_seeds, args.L, args.n_samples)
