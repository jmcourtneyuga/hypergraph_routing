import math
import numpy as np
from collections import OrderedDict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix,
    lp_check_matrix_circulant,
    hgp_tanner_graph,
    hgp_code_params,
)


def hgp_distance_lower_bound(H):
    m, n = H.shape
    col_weights = H.sum(axis=0)
    return int(col_weights.max())  # rough lower bound on row distance


def make_code_sample():
    codes = []
    for n_base, seed in [(8, 1), (12, 1), (16, 2), (20, 3), (24, 4),
                         (28, 5), (32, 6), (40, 7), (60, 8), (80, 9)]:
        H = random_3_4_bipartite_check_matrix(n_base, seed=seed)
        N, K = hgp_code_params(H, H)
        d_lb = hgp_distance_lower_bound(H)
        codes.append({
            "label": f"HGP-rand[{n_base}]",
            "family": "random HGP",
            "qc": False,
            "H": H,
            "n_base": n_base,
            "N": N, "K": K, "d_lb": d_lb,
            "delta_c": 4,  # max(row_deg, col_deg) for (3,4)-biregular
        })

    rng = np.random.default_rng(42)
    B_34 = rng.integers(0, 7, size=(3, 4))
    for L_lift in [4, 8, 12, 16, 20]:
        H = lp_check_matrix_circulant(B_34, L=L_lift)
        N, K = hgp_code_params(H, H)
        d_lb = hgp_distance_lower_bound(H)
        codes.append({
            "label": f"LP-3x4[L={L_lift}]",
            "family": "LP (3,4)-biregular",
            "qc": True,
            "H": H,
            "L_lift": L_lift,
            "N": N, "K": K, "d_lb": d_lb,
            "delta_c": 4,
        })

    B_35 = rng.integers(0, 7, size=(3, 5))
    for L_lift in [4, 8, 12, 20, 32]:
        H = lp_check_matrix_circulant(B_35, L=L_lift)
        N, K = hgp_code_params(H, H)
        d_lb = hgp_distance_lower_bound(H)
        codes.append({
            "label": f"LP-3x5[L={L_lift}]",
            "family": "LP (3,5)-biregular",
            "qc": True,
            "H": H,
            "L_lift": L_lift,
            "N": N, "K": K, "d_lb": d_lb,
            "delta_c": 5,
        })

    return codes

T_MOVE_PERCYCLE_MS = 0.200  # Bluvstein 2022/2024: short-range per-gate move (PER CYCLE)
T_AOD_TRANSFER_MS  = 0.7     # long-range setup: Xu Methods after Eq. 7: 14 trap transfers x 50 us
T_ATOM_MOTION_MS   = 2.3     # long-range setup: Xu Methods cubic spline trajectory, L=100, d=5um
T_MOVE_SETUP_MS    = T_AOD_TRANSFER_MS + T_ATOM_MOTION_MS  # = 3.0 ms (SETUP / long-range only)
T_AOL_SWITCH_US    = 30      # Bluvstein Methods, conservative midrange

T_PER_REARRANGEMENT_MS = T_MOVE_PERCYCLE_MS


def xu_per_cycle_ms(delta_c):
    return 2 * delta_c * T_MOVE_PERCYCLE_MS


def ours_per_cycle_ms(chi_prime, L_layers, scenario="prestored"):
    n_batches = math.ceil(chi_prime / L_layers)
    if scenario == "prestored":
        return n_batches * T_AOL_SWITCH_US * 1e-3  # us -> ms
    elif scenario == "motion":
        return n_batches * T_MOVE_PERCYCLE_MS
    else:
        raise ValueError(scenario)


def setup_cost_ms(N, k_emp=0.5):
    T_setup_motions = k_emp * math.log2(N)
    return T_setup_motions * T_MOVE_SETUP_MS


def crossover_round(N, chi_prime, L_layers, scenario="prestored", k_emp=0.5):
    T_xu = xu_per_cycle_ms(chi_prime / 2)  # delta_c = chi'/2
    T_us = ours_per_cycle_ms(chi_prime, L_layers, scenario)
    T_set = setup_cost_ms(N, k_emp)
    if T_us >= T_xu:
        return float('inf')  # never crosses
    return max(1, math.ceil(T_set / (T_xu - T_us)))


def total_wallclock_ms(N, chi_prime, L_layers, R, scenario="prestored", k_emp=0.5):
    T_xu_total = R * xu_per_cycle_ms(chi_prime / 2)
    T_us_total = setup_cost_ms(N, k_emp) + R * ours_per_cycle_ms(chi_prime, L_layers, scenario)
    return T_xu_total, T_us_total


def main():
    codes = make_code_sample()
    L_values = [1, 2, 4, 8, 16, 32]
    R_breakdown = [10, 100, 1000, 10000]
    scenarios = ["prestored", "motion"]


    print("CODE SAMPLE")

    print(f"{'label':>20} | {'family':>22} | {'QC?':>4} | {'N':>6} | "
          f"{'K':>5} | {'d_lb':>4} | {'chi':>4}")

    for c in codes:
        chi = 2 * c['delta_c']  # by Konig
        print(f"{c['label']:>20} | {c['family']:>22} | {str(c['qc']):>4} | "
              f"{c['N']:>6} | {c['K']:>5} | {c['d_lb']:>4} | {chi:>4}")
    print()


    print("PER-CYCLE WALL-CLOCK (ms), pre-stored AOL scenario")

    print(f"{'label':>20} | {'N':>6} | {'chi':>4} | {'Xu':>7} | "
          + ' | '.join(f'L={L:>3}' for L in L_values))

    for c in codes:
        chi = 2 * c['delta_c']
        xu_ms = xu_per_cycle_ms(c['delta_c'])
        cells = [f'{ours_per_cycle_ms(chi, L, "prestored"):>5.3f}'
                  for L in L_values]
        print(f"{c['label']:>20} | {c['N']:>6} | {chi:>4} | "
              f"{xu_ms:>6.2f} | " + ' | '.join(cells))
    print()

    print("CROSSOVER ROUND R* (rounds needed for ours to beat Xu in TOTAL time)")
    print("Pre-stored AOL scenario, k_emp = 0.5")

    print(f"{'label':>20} | {'N':>6} | {'T_setup(ms)':>11} | "
          + ' | '.join(f'L={L:>3}' for L in L_values))

    for c in codes:
        chi = 2 * c['delta_c']
        T_set = setup_cost_ms(c['N'])
        cells = []
        for L in L_values:
            R_star = crossover_round(c['N'], chi, L, "prestored")
            if R_star == float('inf'):
                cells.append(' inf')
            else:
                cells.append(f'{R_star:>4d}')
        print(f"{c['label']:>20} | {c['N']:>6} | {T_set:>10.2f} | "
              + ' | '.join(cells))
    print()

    print("TOTAL-TIME SPEEDUP (Xu/ours) at fixed R rounds, L_layers = 8")
    print("Pre-stored AOL scenario")

    print(f"{'label':>20} | {'N':>6} | "
          + ' | '.join(f'R={R:>5}' for R in R_breakdown))

    for c in codes:
        chi = 2 * c['delta_c']
        cells = []
        for R in R_breakdown:
            T_xu, T_us = total_wallclock_ms(c['N'], chi, 8, R, "prestored")
            speedup = T_xu / T_us
            cells.append(f'{speedup:>6.1f}x')
        print(f"{c['label']:>20} | {c['N']:>6} | " + ' | '.join(cells))
    print()

    print("CROSSOVER ROUND R* — ATOM-MOTION SCENARIO (no pre-storing)")
    print("Each chromatic batch incurs a ~200 us short-range atom rearrangement.")

    print(f"{'label':>20} | {'N':>6} | {'T_setup(ms)':>11} | "
          + ' | '.join(f'L={L:>3}' for L in L_values))

    for c in codes:
        chi = 2 * c['delta_c']
        T_set = setup_cost_ms(c['N'])
        cells = []
        for L in L_values:
            R_star = crossover_round(c['N'], chi, L, "motion")
            if R_star == float('inf'):
                cells.append(' inf')
            else:
                cells.append(f'{R_star:>4d}')
        print(f"{c['label']:>20} | {c['N']:>6} | {T_set:>10.2f} | "
              + ' | '.join(cells))
    print()
    print("Note: at L=1, ours requires chi' atom motions per cycle = same as Xu;")
    print("ours never wins under atom-motion scenario at L=1 (R* = inf).")
    print("Multi-layer advantage starts at L=2 and saturates at L=chi'.")
    print()


    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "crossover_data.csv")
    with open(csv_path, "w") as f:
        f.write("label,family,QC,N,K,d_lb,delta_c,chi_prime,L_layers,scenario,"
                "T_setup_ms,T_xu_per_cycle_ms,T_ours_per_cycle_ms,"
                "R_crossover,speedup_R10,speedup_R100,speedup_R1000,speedup_R10000\n")
        for c in codes:
            chi = 2 * c['delta_c']
            T_set = setup_cost_ms(c['N'])
            T_xu = xu_per_cycle_ms(c['delta_c'])
            for scenario in scenarios:
                for L in L_values:
                    T_us = ours_per_cycle_ms(chi, L, scenario)
                    R_star = crossover_round(c['N'], chi, L, scenario)
                    R_star_str = "inf" if R_star == float('inf') else str(R_star)
                    speedups = []
                    for R in R_breakdown:
                        T_xu_tot = R * T_xu
                        T_us_tot = T_set + R * T_us
                        speedups.append(T_xu_tot / T_us_tot)
                    f.write(f"{c['label']},{c['family']},{c['qc']},{c['N']},"
                            f"{c['K']},{c['d_lb']},{c['delta_c']},{chi},{L},"
                            f"{scenario},{T_set:.4f},{T_xu:.4f},{T_us:.4f},"
                            f"{R_star_str},{speedups[0]:.3f},{speedups[1]:.3f},"
                            f"{speedups[2]:.3f},{speedups[3]:.3f}\n")
    print(f"CSV exported to: {csv_path}")
    print()

if __name__ == "__main__":
    main()
