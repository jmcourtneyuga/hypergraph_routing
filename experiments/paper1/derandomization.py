import numpy as np
from itertools import combinations
from collections import defaultdict, deque
import time

from routing_lib.graphs import build_circulant_clique
from routing_lib.routing import (
    bfs_shortest_paths,
    canonical_path,
    path_edges,
    compute_congestion_paths as compute_congestion,
    derandomize_valiant,
    random_valiant,
)

def run_analysis():
    test_cases = [
        ("Circ(21,1,5) r=3", 21, [1, 5], 3),
        ("Circ(31,1,5) r=3", 31, [1, 5], 3),
        ("Circ(21,1,4,9) r=4", 21, [1, 4, 9], 4),
        ("Circ(31,1,4,9) r=4", 31, [1, 4, 9], 4),
        ("Circ(51,1,4,9) r=4", 51, [1, 4, 9], 4),
    ]

    num_perms = 20  # test permutations per graph
    num_rand_trials = 500  # random sigma trials per permutation

    print("DERANDOMIZATION OF VALIANT ROUTING")
    print("Method of Conditional Expectations vs Random Intermediate")
    print("=" * 90)

    all_results = []

    for name, N, offsets, r in test_cases:
        print(f"\n--- {name} (N={N}) ---")
        hyperedges, adj = build_circulant_clique(N, offsets, r)

        d_prime = min(len(adj[v]) for v in range(N))
        D_vals = []
        for v in range(N):
            d, _ = bfs_shortest_paths(adj, N, v)
            D_vals.append(max(d))
        D = max(D_vals)

        print(f"  d'={d_prime}, D={D}, |E|={sum(len(adj[v]) for v in range(N))//2}")

        derand_results = []
        rand_results_all = []

        np.random.seed(42)
        test_perms = [np.random.permutation(N).tolist() for _ in range(num_perms)]
        # Also test some structured permutations
        test_perms.append([(v + 1) % N for v in range(N)])  # cyclic shift
        test_perms.append([(N - 1 - v) for v in range(N)])  # reversal

        for pi_idx, pi in enumerate(test_perms):
            t0 = time.time()
            sigma, C_s, C_g = derandomize_valiant(dict(adj), N, pi)
            t_derand = time.time() - t0
            C_derand = max(C_s, C_g)

            # Random baseline
            t0 = time.time()
            rand_res = random_valiant(dict(adj), N, pi, num_trials=num_rand_trials)
            t_rand = time.time() - t0

            rand_totals = [r[2] for r in rand_res]
            C_rand_mean = np.mean(rand_totals)
            C_rand_min = min(rand_totals)
            C_rand_median = np.median(rand_totals)

            derand_results.append({
                'C_derand': C_derand, 'C_s': C_s, 'C_g': C_g,
                'C_rand_mean': C_rand_mean, 'C_rand_min': C_rand_min,
                'C_rand_median': C_rand_median,
                't_derand': t_derand
            })

        # Summary for this graph
        avg_C_derand = np.mean([r['C_derand'] for r in derand_results])
        avg_C_rand_mean = np.mean([r['C_rand_mean'] for r in derand_results])
        avg_C_rand_min = np.mean([r['C_rand_min'] for r in derand_results])
        avg_t = np.mean([r['t_derand'] for r in derand_results])
        max_C_derand = max(r['C_derand'] for r in derand_results)
        max_C_rand_min = max(r['C_rand_min'] for r in derand_results)

        print(f"\n  Results over {len(test_perms)} test permutations:")
        hdr = f"  {'Perm':>5} {'C_derand':>8} {'C_s':>4} {'C_g':>4} {'C_rand_mean':>11} {'C_rand_min':>10} {'Ratio':>6}"
        print(hdr)
        print(f"  {'-'*60}")
        for i, r in enumerate(derand_results):
            ratio = r['C_derand'] / max(r['C_rand_min'], 1)
            label = f"pi_{i}" if i < num_perms else ("cyclic" if i == num_perms else "reversal")
            print(f"  {label:>5} {r['C_derand']:>8} {r['C_s']:>4} {r['C_g']:>4} "
                  f"{r['C_rand_mean']:>11.1f} {r['C_rand_min']:>10} {ratio:>6.2f}")

        print(f"\n  Summary:")
        print(f"    Avg derandomized congestion: {avg_C_derand:.1f}")
        print(f"    Avg random mean congestion:  {avg_C_rand_mean:.1f}")
        print(f"    Avg random best-of-{num_rand_trials} congestion: {avg_C_rand_min:.1f}")
        print(f"    Worst-case derand / worst-case rand-best: {max_C_derand}/{max_C_rand_min} = {max_C_derand/max(max_C_rand_min,1):.2f}")
        print(f"    Avg construction time: {avg_t:.3f}s")

        all_results.append({
            'name': name, 'N': N, 'r': r, 'd_prime': d_prime, 'D': D,
            'avg_C_derand': avg_C_derand,
            'max_C_derand': max_C_derand,
            'avg_C_rand_mean': avg_C_rand_mean,
            'avg_C_rand_min': avg_C_rand_min,
            'max_C_rand_min': max_C_rand_min,
            'avg_time': avg_t
        })

    # Final summary
    print(f"\n\n{'='*90}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*90}")
    dp_hdr = "d'"
    print(f"  {'Name':<25} {'N':>3} {dp_hdr:>3} {'D':>2} {'C_derand':>8} {'C_rand_mean':>11} {'C_rand_best':>11} {'Ratio':>6} {'Time':>6}")
    print(f"  {'-'*82}")
    for r in all_results:
        ratio = r['max_C_derand'] / max(r['max_C_rand_min'], 1)
        print(f"  {r['name']:<25} {r['N']:>3} {r['d_prime']:>3} {r['D']:>2} "
              f"{r['max_C_derand']:>8} {r['avg_C_rand_mean']:>11.1f} {r['max_C_rand_min']:>11} "
              f"{ratio:>6.2f} {r['avg_time']:>5.2f}s")

    print(f"\n  Ratio = worst-case derandomized / worst-case random-best-of-{num_rand_trials}")
    print(f"  Ratio near 1.0 means derandomization matches the best random outcome.")
    print(f"  Ratio < 1.0 means derandomization beats random search.")

if __name__ == "__main__":
    run_analysis()
