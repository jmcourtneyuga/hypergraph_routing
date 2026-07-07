#!/usr/bin/env python3
import numpy as np
from numpy.linalg import eigvalsh
from itertools import combinations, product as cart_product
from math import ceil, log2
from collections import deque
import time

np.random.seed(42)


from routing_lib.graphs import (
    fano_plane,
    pg23,
    clique_expansion,
    voltage_covering,
    random_voltage_covering,
    trivial_covering,
)
from routing_lib.spectral import (
    spectral_params_covering_tower as spectral_params,
    ramanujan_bound_hyper,
)
from routing_lib.routing import bfs_path, valiant_trial

def fiber_cross_decompose(pi, N0, k):
    N = N0 * k
    cross_count = 0
    fiber_count = 0
    for v in range(N):
        b_src = v // k
        b_tgt = pi[v] // k
        if b_src != b_tgt:
            cross_count += 1
        else:
            fiber_count += 1

    return fiber_count, cross_count


def main():
    t_start = time.time()

    print("  DIRECTION 5: COVERING TOWERS AND RECURSIVE LIFT")

    print("\n" + "=" * 90)
    print("  PART 1: BASE HYPERGRAPH SPECTRA")

    N0_f, r_f, he_f = fano_plane()
    A_f = clique_expansion(N0_f, r_f, he_f)
    sp_f = spectral_params(A_f)
    d_f = 3  # each vertex in 3 hyperedges
    rb_f = ramanujan_bound_hyper(d_f, r_f)

    print(f"\n  Fano plane: N0={N0_f}, (d,r)=({d_f},{r_f})")
    print(f"  Clique expansion degree: {sp_f['d_prime']:.0f}")
    print(f"  Eigenvalues: {np.round(sp_f['all_eigs'], 3)}")
    print(f"  lambda_2 = {sp_f['lambda_2']:.3f}, lambda_N = {sp_f['lambda_N']:.3f}")
    print(f"  SFM Ramanujan bound: |lam - (r-2)| <= {rb_f:.3f}")
    print(f"  Centered: |lam_2 - {r_f-2}| = {abs(sp_f['lambda_2'] - (r_f-2)):.3f} "
      f"<= {rb_f:.3f}? {'YES' if abs(sp_f['lambda_2'] - (r_f-2)) <= rb_f + 1e-10 else 'no'}")
    print(f"  beta = {sp_f['beta']:.4f}")

    # PG(2,3)
    N0_p, r_p, he_p = pg23()
    A_p = clique_expansion(N0_p, r_p, he_p)
    sp_p = spectral_params(A_p)
    d_p = 4
    rb_p = ramanujan_bound_hyper(d_p, r_p)

    print(f"\n  PG(2,3): N0={N0_p}, (d,r)=({d_p},{r_p})")
    print(f"  Clique expansion degree: {sp_p['d_prime']:.0f}")
    print(f"  lambda_2 = {sp_p['lambda_2']:.3f}, lambda_N = {sp_p['lambda_N']:.3f}")
    print(f"  SFM Ramanujan bound: |lam - (r-2)| <= {rb_p:.3f}")
    print(f"  Centered: |lam_2 - {r_p-2}| = {abs(sp_p['lambda_2'] - (r_p-2)):.3f} "
      f"<= {rb_p:.3f}? {'YES' if abs(sp_p['lambda_2'] - (r_p-2)) <= rb_p + 1e-10 else 'no'}")
    print(f"  beta = {sp_p['beta']:.4f}")

    print("\n\n" + "=" * 90)
    print("  PART 2: VOLTAGE COVERINGS OF THE FANO PLANE")
    print("  k-fold coverings via Z_k voltage assignments")


    for k in [2, 3, 4, 5, 7]:
        N_lift = N0_f * k
        print(f"\n  --- k = {k} (N = {N_lift}) ---")

        # Trivial covering (k disjoint copies)
        A_triv, _ = trivial_covering(N0_f, r_f, he_f, k)
        sp_triv = spectral_params(A_triv)

        # Random voltage coverings (sample several)
        best_beta = 1.0
        best_voltages = None
        worst_beta = 0.0
        betas_random = []
        ramanujan_count = 0

        n_voltage_trials = min(200, k ** len(he_f))

        for trial in range(n_voltage_trials):
            A_rand, voltages = random_voltage_covering(N0_f, r_f, he_f, k)
            sp_rand = spectral_params(A_rand)

            # Check Ramanujan: all non-base eigenvalues satisfy SFM bound
            base_eigs = set(np.round(sp_f['all_eigs'], 6))
            new_eigs = []
            used_base = list(sp_f['all_eigs'].copy())
            for e in sorted(sp_rand['all_eigs'], reverse=True):
                matched = False
                for i, be in enumerate(used_base):
                    if abs(e - be) < 1e-4:
                        used_base.pop(i)
                        matched = True
                        break
                if not matched:
                    new_eigs.append(e)

            is_ram = True
            for e in new_eigs:
                if abs(e - (r_f - 2)) > rb_f + 1e-6:
                    is_ram = False
                    break

            if is_ram:
                ramanujan_count += 1

            betas_random.append(sp_rand['beta'])
            if sp_rand['beta'] < best_beta:
                best_beta = sp_rand['beta']
                best_voltages = voltages
            worst_beta = max(worst_beta, sp_rand['beta'])

        print(f"  Trivial (disconnected):  beta = {sp_triv['beta']:.4f}")
        print(f"  Random voltages ({n_voltage_trials} trials):")
        print(f"    Best beta  = {best_beta:.4f}")
        print(f"    Worst beta = {worst_beta:.4f}")
        print(f"    Mean beta  = {np.mean(betas_random):.4f}")
        print(f"    Ramanujan: {ramanujan_count}/{n_voltage_trials} "
          f"({100 * ramanujan_count / n_voltage_trials:.0f}%)")

        if best_voltages is not None:
            print(f"    Best voltages: {best_voltages}")


    print("\n\n" + "=" * 90)
    print("  PART 3: EXHAUSTIVE VOLTAGE SEARCH (Fano plane, k=2)")
    print("  All 2^7 = 128 voltage assignments on Z_2")


    k = 2
    N_lift = N0_f * k
    n_he = len(he_f)

    print(f"\n  Fano plane ({n_he} hyperedges), k={k}, N_lift={N_lift}")

    all_betas = []
    all_ramanujan = []
    best_beta = 1.0
    best_v = None

    for bits in range(2 ** n_he):
        voltages = [(bits >> i) & 1 for i in range(n_he)]
        A_lift = voltage_covering(N0_f, r_f, he_f, k, voltages)
        sp = spectral_params(A_lift)

        # Check new eigenvalue Ramanujan
        base_eigs_sorted = sorted(sp_f['all_eigs'], reverse=True)
        lift_eigs_sorted = sorted(sp['all_eigs'], reverse=True)
        new_eigs = []
        remaining = list(base_eigs_sorted)
        for e in lift_eigs_sorted:
            matched = False
            for i, be in enumerate(remaining):
                if abs(e - be) < 1e-4:
                    remaining.pop(i)
                    matched = True
                    break
            if not matched:
                new_eigs.append(e)

        is_ram = all(abs(e - (r_f - 2)) <= rb_f + 1e-6 for e in new_eigs) if new_eigs else True
        all_betas.append(sp['beta'])
        all_ramanujan.append(is_ram)

        if sp['beta'] < best_beta:
            best_beta = sp['beta']
            best_v = voltages
            best_new_eigs = new_eigs
            best_sp = sp

    print(f"  Total assignments: {2**n_he}")
    print(f"  Ramanujan: {sum(all_ramanujan)}/{2**n_he} "
      f"({100 * sum(all_ramanujan) / (2**n_he):.1f}%)")
    print(f"  Best beta: {best_beta:.4f} (voltages: {best_v})")
    print(f"  Best lift eigenvalues: {np.round(best_sp['all_eigs'], 3)}")
    print(f"  New eigenvalues: {np.round(best_new_eigs, 3)}")
    print(f"  SFM Ramanujan bound: {rb_f:.3f}")
    print(f"  Max |new_eig - (r-2)|: {max(abs(e - (r_f-2)) for e in best_new_eigs):.3f}")

    print(f"\n  Beta distribution:")
    beta_arr = np.array(all_betas)
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        frac = np.mean(beta_arr < threshold)
        print(f"    beta < {threshold}: {frac:.1%}")


    print("\n\n" + "=" * 90)
    print("  PART 4: ROUTING ON COVERING LIFTS")
    print("  Valiant routing: black-box vs recursive decomposition")

    k = 2
    A_lift = voltage_covering(N0_f, r_f, he_f, k, best_v)
    sp_lift = spectral_params(A_lift)
    N_lift = N0_f * k

    print(f"\n  Best 2-fold covering: N={N_lift}, beta={sp_lift['beta']:.4f}")

    n_trials = 30
    bb_Ts = []
    for _ in range(n_trials):
        T, C, D = valiant_trial(A_lift)
        bb_Ts.append(T)

    bb_T_med = int(np.median(bb_Ts))
    print(f"  Black-box Valiant: median T = {bb_T_med}, "
      f"T/log2(N) = {bb_T_med / log2(N_lift):.2f}")

    fano_Ts = []
    for _ in range(n_trials):
        T, C, D = valiant_trial(A_f)
        fano_Ts.append(T)
    fano_T_med = int(np.median(fano_Ts))
    print(f"  Base Fano plane: median T = {fano_T_med}, "
      f"T/log2(N0) = {fano_T_med / log2(N0_f):.2f}")

    print(f"\n  Fiber/cross-fiber decomposition:")
    frac_cross_list = []
    for _ in range(100):
        pi = np.random.permutation(N_lift)
        n_fiber, n_cross = fiber_cross_decompose(pi, N0_f, k)
        frac_cross_list.append(n_cross / N_lift)

    print(f"    Mean fraction cross-fiber: {np.mean(frac_cross_list):.3f}")
    print(f"    Expected (1 - 1/k): {1 - 1/k:.3f}")

    for k in [3, 4, 5]:
        N_lift = N0_f * k
        A_rand, _ = random_voltage_covering(N0_f, r_f, he_f, k)
        sp_rand = spectral_params(A_rand)

        Ts = []
        for _ in range(n_trials):
            T, C, D = valiant_trial(A_rand)
            Ts.append(T)
        T_med = int(np.median(Ts))

        frac_cross = []
        for _ in range(50):
            pi = np.random.permutation(N_lift)
            _, n_cross = fiber_cross_decompose(pi, N0_f, k)
            frac_cross.append(n_cross / N_lift)

        print(f"\n  k={k}-fold covering: N={N_lift}, beta={sp_rand['beta']:.4f}")
        print(f"    Black-box: median T = {T_med}, T/log2(N) = {T_med / log2(N_lift):.2f}")
        print(f"    Cross-fiber fraction: {np.mean(frac_cross):.3f} "
          f"(expected {1 - 1/k:.3f})")


    print("\n\n" + "=" * 90)
    print("  PART 5: TWO-LEVEL COVERING TOWER")
    print("  H_0 <- H_1 <- H_2, testing Theorem 5.1")

    k = 2
    A_H1 = voltage_covering(N0_f, r_f, he_f, k, best_v)
    N1 = N0_f * k  # 14
    sp_H1 = spectral_params(A_H1)

    print(f"\n  H_0 = Fano plane: N0 = {N0_f}, beta = {sp_f['beta']:.4f}")
    print(f"  H_1 = 2-fold covering: N1 = {N1}, beta = {sp_H1['beta']:.4f}")

    k2 = 4  # k^2 = 4, so H_2 has N = 7*4 = 28 vertices
    N2 = N0_f * k2

    # Build via direct k^2 covering with random voltages
    A_H2, v2 = random_voltage_covering(N0_f, r_f, he_f, k2)
    sp_H2 = spectral_params(A_H2)

    print(f"  H_2 = 4-fold covering: N2 = {N2}, beta = {sp_H2['beta']:.4f}")

    print(f"\n  Routing depth comparison:")
    for name, A, N in [("H_0 (Fano)", A_f, N0_f),
                        ("H_1 (2-fold)", A_H1, N1),
                        ("H_2 (4-fold)", A_H2, N2)]:
        Ts = []
        for _ in range(30):
            T, C, D = valiant_trial(A)
            Ts.append(T)
        T_med = int(np.median(Ts))
        logN = log2(N)
        print(f"  {name:<20}: N={N:>3}, T_med={T_med:>3}, "
          f"log2(N)={logN:.1f}, T/logN={T_med / logN:.2f}")

    L = 2
    logk = log2(k)
    logN0 = log2(N0_f)
    print(f"\n  Theorem 5.1 prediction (L={L}, k={k}):")
    print(f"    rt(H_2) = O(L*log(k) + log(N0)) = O({L}*{logk:.1f} + {logN0:.1f}) "
      f"= O({L*logk + logN0:.1f})")
    print(f"    vs log2(N2) = {log2(N2):.1f}")
    print(f"    Ratio: {(L*logk + logN0) / log2(N2):.2f}")

    print("\n\n" + "=" * 90)
    print("  PART 6: CROSS-FIBER EXPANDER (Lemma 5.2)")
    print("  Spectral gap of inter-fiber subgraph")


    for k in [2, 3, 4]:
        N_lift = N0_f * k
        A_lift = voltage_covering(N0_f, r_f, he_f, k, best_v if k == 2
                                   else [np.random.randint(0, k) for _ in he_f])
        sp_lift = spectral_params(A_lift)

        A_cross = np.zeros((N_lift, N_lift))
        A_fiber = np.zeros((N_lift, N_lift))
        for u in range(N_lift):
            for v in range(u + 1, N_lift):
                if A_lift[u, v] > 0:
                    base_u = u // k
                    base_v = v // k
                    if base_u != base_v:
                        A_cross[u, v] = A_lift[u, v]
                        A_cross[v, u] = A_lift[v, u]
                    else:
                        A_fiber[u, v] = A_lift[u, v]
                        A_fiber[v, u] = A_lift[v, u]

        sp_cross = spectral_params(A_cross)
        sp_fiber = spectral_params(A_fiber)

        n_cross_edges = int(A_cross.sum() / 2)
        n_fiber_edges = int(A_fiber.sum() / 2)
        n_total_edges = int(A_lift.sum() / 2)

        print(f"\n  k={k}-fold covering (N={N_lift}):")
        print(f"    Total edges: {n_total_edges}, "
          f"cross-fiber: {n_cross_edges} ({100*n_cross_edges/n_total_edges:.0f}%), "
          f"intra-fiber: {n_fiber_edges}")
        print(f"    Full graph:   beta = {sp_lift['beta']:.4f}")
        print(f"    Cross-fiber:  beta = {sp_cross['beta']:.4f}, "
          f"d' = {sp_cross['d_prime']:.1f}")
        print(f"    Intra-fiber:  d' = {sp_fiber['d_prime']:.1f}")

    print("\n\n" + "=" * 90)
    print("  PART 7: COVERINGS OF PG(2,3)")


    for k in [2, 3]:
        N_lift = N0_p * k
        print(f"\n  PG(2,3), k={k}-fold covering (N={N_lift}):")

        best_beta_pg = 1.0
        n_trials_v = min(200, k ** len(he_p))
        ramanujan_ct = 0

        for _ in range(n_trials_v):
            A_rand, _ = random_voltage_covering(N0_p, r_p, he_p, k)
            sp_rand = spectral_params(A_rand)
            if sp_rand['beta'] < best_beta_pg:
                best_beta_pg = sp_rand['beta']
                best_A_pg = A_rand
                
            new_eigs = []
            remaining = list(sorted(sp_p['all_eigs'], reverse=True))
            for e in sorted(sp_rand['all_eigs'], reverse=True):
                matched = False
                for i, be in enumerate(remaining):
                    if abs(e - be) < 1e-4:
                        remaining.pop(i)
                        matched = True
                        break
                if not matched:
                    new_eigs.append(e)
            is_ram = all(abs(e - (r_p - 2)) <= rb_p + 1e-6 for e in new_eigs) if new_eigs else True
            if is_ram:
                ramanujan_ct += 1

        print(f"    Best beta: {best_beta_pg:.4f}")
        print(f"    Ramanujan: {ramanujan_ct}/{n_trials_v} "
          f"({100*ramanujan_ct/n_trials_v:.0f}%)")

        Ts = []
        for _ in range(30):
            T, C, D = valiant_trial(best_A_pg)
            Ts.append(T)
        T_med = int(np.median(Ts))
        print(f"    Routing: median T = {T_med}, T/log2(N) = {T_med / log2(N_lift):.2f}")


    print("\n\n" + "=" * 90)
    print("  SUMMARY OF RESULTS")


    print(.format(sp_f['beta'], sp_p['beta'],
               2 * log2(2) + log2(7), log2(28)))

    t_end = time.time()
    print(f"  Total computation time: {t_end - t_start:.1f}s")


if __name__ == "__main__":
    main()
