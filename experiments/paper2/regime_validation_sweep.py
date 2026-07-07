#!/usr/bin/env python3
import numpy as np

from routing_lib.routing import simulate_block_routing


def run_block_simulation(d, r, d_C, N_L, n_trials=3, n_phys_factor=8, seed=42):
    np.random.seed(seed)
    block_size = d_C * d_C
    N_phys = max(N_L * block_size * n_phys_factor, 256)
    if (N_phys * d * (r - 1)) % 2 != 0:
        N_phys += 1
    stats = simulate_block_routing(
        N_phys=N_phys, d=d, r=r, d_C=d_C,
        guard_dist=1, N_L=N_L, n_trials=n_trials,
    )
    d_prime = d * (r - 1)
    return {
        'd': d, 'r': r, 'd_prime': d_prime, 'd_C': d_C, 'N_L': N_L,
        'N_phys': N_phys,
        'T_sim': stats['T_mean'],
        'C_max': stats['C_max_mean'],
        'D_Q': stats['D_Q_mean'],
        'beta_host': stats['beta_mean'],
        'beta_Q': stats['beta_Q_mean'],
        'successful_trials': stats['successful_trials'],
    }


def regime_threshold(d_C, beta_host, r=3):
    """The minimum d' satisfying d' > d_C^2 (r-1) / (1 - beta_host)."""
    return d_C * d_C * (r - 1) / max(1e-9, 1.0 - beta_host)


def part_a_rerun_dC7():

    print("(a) Rerun d_C = 7 rows at d' = 200 (was d' = 100 in V1)")

    print()
    print(f"{'d_C':>4} {'N_L':>4} {'T_sim':>7} {'d_C log2 N_L':>13} "
          f"{'alpha':>7} {'beta_Q':>7} {'beta_h':>7} "
          f"{'threshold':>10} {'margin':>9}")
    print("-" * 78)

    rows = []
    d_C = 7
    d = 100   # d' = d(r-1) = 200 for r=3
    r = 3
    for N_L in [8, 16, 32, 64]:
        result = run_block_simulation(d=d, r=r, d_C=d_C, N_L=N_L,
                                      n_trials=3, seed=42 + N_L)
        T_sim = result['T_sim']
        log2_NL = np.log2(N_L)
        target = d_C * log2_NL
        alpha = T_sim / target if target > 0 else float('nan')
        beta_h = result['beta_host']
        beta_Q = result['beta_Q']
        thr = regime_threshold(d_C, beta_h, r=r)
        margin = result['d_prime'] - thr
        print(f"{d_C:>4} {N_L:>4} {T_sim:>7.1f} {target:>13.1f} "
              f"{alpha:>7.2f} {beta_Q:>7.3f} {beta_h:>7.3f} "
              f"{thr:>10.1f} {margin:>+9.1f}")
        rows.append(result)
    print()
    return rows

def part_b_dprime_sweep():

    print("(b) d' sweep at fixed d_C = 7, N_L = 32 (regime threshold demo)")

    print()
    print(f"{'d_prime':>8} {'d':>4} {'T_sim':>7} {'d_C log2 N_L':>13} "
          f"{'alpha':>7} {'beta_Q':>7} {'beta_h':>7} "
          f"{'threshold':>10} {'in_regime':>10}")
    print("-" * 78)

    d_C = 7
    N_L = 32
    r = 3
    rows = []
    for d_prime in [50, 100, 200, 400]:
        d = d_prime // (r - 1)
        if d * (r - 1) != d_prime:
            print(f"  skipping d_prime={d_prime}: not a multiple of (r-1)={r-1}")
            continue
        result = run_block_simulation(d=d, r=r, d_C=d_C, N_L=N_L,
                                      n_trials=3, seed=42 + d_prime)
        T_sim = result['T_sim']
        log2_NL = np.log2(N_L)
        target = d_C * log2_NL
        alpha = T_sim / target if target > 0 else float('nan')
        beta_h = result['beta_host']
        beta_Q = result['beta_Q']
        thr = regime_threshold(d_C, beta_h, r=r)
        in_regime = "YES" if d_prime > thr else "NO"
        print(f"{d_prime:>8} {d:>4} {T_sim:>7.1f} {target:>13.1f} "
              f"{alpha:>7.2f} {beta_Q:>7.3f} {beta_h:>7.3f} "
              f"{thr:>10.1f} {in_regime:>10}")
        rows.append(result)
    print()
    return rows


def latex_table2_dc7_rows(rows):
    out = []
    for r in rows:
        d_C = r['d_C']
        N_L = r['N_L']
        T_sim = r['T_sim']
        target = d_C * np.log2(N_L)
        alpha = T_sim / target
        beta_Q = r['beta_Q']
        out.append(f"{d_C}  & {N_L:>2}  & {T_sim:>4.1f}  "
                   f"& {target:>4.1f}  & {alpha:>.2f} & {beta_Q:>.3f} \\\\")
    return out


def latex_sweep_table(rows):
    out = []
    for r in rows:
        d_prime = r['d_prime']
        T_sim = r['T_sim']
        d_C = r['d_C']
        N_L = r['N_L']
        target = d_C * np.log2(N_L)
        alpha = T_sim / target
        beta_Q = r['beta_Q']
        beta_h = r['beta_host']
        thr = regime_threshold(d_C, beta_h)
        in_regime = "in" if d_prime > thr else "out"
        out.append(f"{d_prime}  & {beta_h:>.3f} & {thr:>5.1f}  "
                   f"& {in_regime} & {T_sim:>4.1f} & {alpha:>.2f} "
                   f"& {beta_Q:>.3f} \\\\")
    return out


def main():
    rows_a = part_a_rerun_dC7()
    rows_b = part_b_dprime_sweep()

    print("LaTeX drop-ins")

    print()
    print("(a) New d_C=7 rows for tab:simulation (replace old four rows):")
    for line in latex_table2_dc7_rows(rows_a):
        print("  " + line)
    print()
    print("(b) New sweep table (d_C=7, N_L=32, varying d'):")
    for line in latex_sweep_table(rows_b):
        print("  " + line)


if __name__ == "__main__":
    main()
