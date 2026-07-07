import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.published_codes import (
    BRAVYI_BB_CODES, XU_LP_CODES,
    get_bravyi_bb_tanner, get_xu_lp_tanner,
)


T_AOD_TRANSFER_MS = 0.7
T_ATOM_MOTION_MS = 2.3
T_PER_REARRANGEMENT_MS = T_AOD_TRANSFER_MS + T_ATOM_MOTION_MS
T_AOL_SWITCH_US = 30


def measure_chi_prime(A):
    degs = A.sum(axis=1).astype(int)
    return int(degs.max()), A.shape[0]


def per_cycle_xu_ms(chi_prime):
    return chi_prime * T_PER_REARRANGEMENT_MS


def per_cycle_ours_ms(chi_prime, L_layers, scenario="prestored"):
    n_batches = math.ceil(chi_prime / L_layers)
    if scenario == "prestored":
        return n_batches * T_AOL_SWITCH_US * 1e-3
    elif scenario == "motion":
        return n_batches * T_PER_REARRANGEMENT_MS
    else:
        raise ValueError(scenario)


def setup_cost_ms(N, k_emp=0.5):
    return k_emp * math.log2(N) * T_PER_REARRANGEMENT_MS


def run(skip_bb=False, skip_xu=False, L_values=(1, 2, 4, 8, 16),
         R_values=(10, 100, 1000, 10000)):
    print("=" * 100)
    print("PER-CYCLE COMPARISON ON PUBLISHED CODES")
    print(f"L_layers sweep: {L_values}")
    print("=" * 100)
    print()

    results = []

    if not skip_bb:
        print("Bravyi et al. 2024 BB codes (used in Bluvstein 2024 hardware):")

        for spec in BRAVYI_BB_CODES:
            t0 = time.time()
            A, H_X, H_Z = get_bravyi_bb_tanner(spec)
            chi, N = measure_chi_prime(A)
            elapsed = time.time() - t0
            print(f"  {spec['label']:>20} (declared N={spec['N']}, K={spec['K']}, "
                  f"d={spec['d']}): Tanner N={N}, chi'={chi}, build {elapsed:.2f}s")
            xu_ms = per_cycle_xu_ms(chi)
            for L in L_values:
                ours_ms = per_cycle_ours_ms(chi, L, "prestored")
                speedup = xu_ms / ours_ms
                results.append({
                    "code": spec["label"], "family": "BB", "N_decl": spec["N"],
                    "K": spec["K"], "d": spec["d"],
                    "N_tanner": N, "chi": chi, "L": L,
                    "xu_ms": xu_ms, "ours_ms": ours_ms, "speedup": speedup,
                })
        print()

    if not skip_xu:
        print("Xu et al. 2024 LP codes (random voltages on 3x5 base mask):")

        for spec in XU_LP_CODES:
            t0 = time.time()
            A, H = get_xu_lp_tanner(spec)
            chi, N = measure_chi_prime(A)
            elapsed = time.time() - t0
            print(f"  {spec['label']:>22}: Tanner N={N}, "
                  f"chi'={chi}, build {elapsed:.2f}s")
            xu_ms = per_cycle_xu_ms(chi)
            for L in L_values:
                ours_ms = per_cycle_ours_ms(chi, L, "prestored")
                speedup = xu_ms / ours_ms
                results.append({
                    "code": spec["label"], "family": "LP", "L_lift": spec["L_lift"],
                    "N_tanner": N, "chi": chi, "L": L,
                    "xu_ms": xu_ms, "ours_ms": ours_ms, "speedup": speedup,
                })
        print()

    print("=" * 100)
    print("PER-CYCLE WALL-CLOCK SUMMARY (pre-stored AOL scenario)")
    print("=" * 100)
    print(f"{'code':>22} | {'N':>5} | {'chi':>4} | {'Xu (ms)':>8} | "
          + ' | '.join(f'L={L:>2} (ms)' for L in L_values))
    print("-" * 100)
    seen_codes = set()
    for r in results:
        if r["code"] in seen_codes:
            continue
        seen_codes.add(r["code"])
        same_code = [x for x in results if x["code"] == r["code"]]
        cells = []
        for L in L_values:
            entry = next(x for x in same_code if x["L"] == L)
            cells.append(f"{entry['ours_ms']:>7.3f}")
        print(f"{r['code']:>22} | {r['N_tanner']:>5} | {r['chi']:>4} | "
              f"{r['xu_ms']:>7.2f} | " + ' | '.join(cells))
    print()

    print(f"TOTAL-TIME SPEEDUP at L_layers=8, varying R rounds")
    print(f"{'code':>22} | {'chi':>4} | "
          + ' | '.join(f'R={R:>5}' for R in R_values))
    for code in sorted(seen_codes):
        same_code = [x for x in results if x["code"] == code]
        L8 = next((x for x in same_code if x["L"] == 8), None)
        if L8 is None:
            continue
        cells = []
        T_set = setup_cost_ms(L8["N_tanner"])
        for R in R_values:
            T_xu = R * L8["xu_ms"]
            T_us = T_set + R * L8["ours_ms"]
            cells.append(f'{T_xu/T_us:>5.1f}x')
        print(f"{code:>22} | {L8['chi']:>4} | " + ' | '.join(cells))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-bb", action="store_true",
                    help="skip Bravyi BB codes")
    p.add_argument("--skip-xu", action="store_true",
                    help="skip Xu LP codes")
    p.add_argument("--L", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="L_layers sweep (default: 1 2 4 8 16)")
    p.add_argument("--R", type=int, nargs="+", default=[10, 100, 1000, 10000],
                    help="R rounds for total-time speedup (default: 10 100 1000 10000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(skip_bb=args.skip_bb, skip_xu=args.skip_xu,
         L_values=args.L, R_values=args.R)
