import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from routing_lib.graphs import random_regular_matching_union as rrg
from routing_lib.routing import valiant_trial


def estimate_dense_memory_gb(N):
    """Dense N x N float64 adjacency matrix size in GB."""
    return (N * N * 8) / (2**30)


DENSE_MEMORY_THRESHOLD_GB = 4.0


def _trial_worker(N, L_layers, d_0, seed, queue):
    try:
        np.random.seed(seed)
        use_sparse = estimate_dense_memory_gb(N) > DENSE_MEMORY_THRESHOLD_GB
        if use_sparse:
            from scipy.sparse import csr_matrix
            A_union = csr_matrix((N, N), dtype=np.float64)
            for _ in range(L_layers):
                A_union = A_union + rrg(N, d_0, sparse=True)
            # Ensure CSR format for valiant_trial
            A_union = A_union.tocsr() if hasattr(A_union, 'tocsr') else A_union
        else:
            A_union = np.zeros((N, N))
            for _ in range(L_layers):
                A_union += rrg(N, d_0)
        T, C, D = valiant_trial(A_union)
        queue.put(("ok", T, C, D))
    except Exception as e:
        import traceback
        queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _trial_with_timeout(N, L_layers, d_0, seed, timeout_s):
    queue = mp.Queue()
    p = mp.Process(target=_trial_worker, args=(N, L_layers, d_0, seed, queue))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(2)
        if p.is_alive():
            p.kill()
            p.join()
        return {"status": "timeout"}
    if not queue.empty():
        result = queue.get_nowait()
        if result[0] == "ok":
            return {"status": "ok", "T": result[1], "C": result[2], "D": result[3]}
        return {"status": "err", "message": result[1]}
    return {"status": "err", "message": "subprocess exited without result"}


def empirical_T_Valiant(N, L_layers, d_0=8, n_trials=3, seed=42,
                          max_runtime_s=None, per_trial_timeout_s=None,
                          verbose=False):
    if per_trial_timeout_s is None:
        if max_runtime_s:
            per_trial_timeout_s = max_runtime_s / n_trials
        else:
            # Heuristic: scale ~ N^2 / 50k.  At N=10k expect ~2 min, scales up.
            per_trial_timeout_s = max(120.0, (N * N) / 50000.0)

    Ts = []
    timeouts = 0
    errors = 0
    elapsed_total = 0.0
    for trial in range(n_trials):
        if max_runtime_s is not None and elapsed_total > max_runtime_s:
            break
        t0 = time.time()
        result = _trial_with_timeout(N, L_layers, d_0, seed + trial,
                                       per_trial_timeout_s)
        dt = time.time() - t0
        elapsed_total += dt
        if result["status"] == "ok":
            Ts.append(result["T"])
            if verbose:
                print(f"  trial {trial+1}/{n_trials}: T={result['T']} "
                      f"C={result['C']} D={result['D']} ({dt:.1f}s)", flush=True)
        elif result["status"] == "timeout":
            timeouts += 1
            if verbose:
                print(f"  trial {trial+1}/{n_trials}: TIMEOUT after "
                      f"{per_trial_timeout_s}s", flush=True)
        else:
            errors += 1
            if verbose:
                print(f"  trial {trial+1}/{n_trials}: ERROR: "
                      f"{result.get('message','?')}", flush=True)

    return {
        "median": int(np.median(Ts)) if Ts else None,
        "all_T": Ts,
        "elapsed_s": elapsed_total,
        "n_trials_completed": len(Ts),
        "n_timeouts": timeouts,
        "n_errors": errors,
        "per_trial_timeout_s": per_trial_timeout_s,
    }


def run_sweep(n_values, L_values, n_trials=3, d_0=8, max_runtime_s=None,
               per_trial_timeout_s=None, seed=42, verbose=True):
    print("=" * 90)
    print("k_emp MEASUREMENT ON MULTI-LAYER OVERLAY UNION GRAPHS")
    print(f"d_0 = {d_0} (per-layer regularity), trials per (N,L) = {n_trials}")
    if max_runtime_s:
        print(f"max runtime per (N, L) = {max_runtime_s} s")
    if per_trial_timeout_s:
        print(f"per-trial hard timeout = {per_trial_timeout_s} s")
    print("=" * 90)
    print(f"{'N':>7} | {'L':>3} | {'log_2 N':>9} | {'T_median':>9} | "
          f"{'k_emp':>7} | {'time(s)':>8} | {'ok':>3} | {'timeout':>7} | notes")
    print("-" * 95)

    results = []
    for N in n_values:
        log_n = np.log2(N)
        mem_gb = estimate_dense_memory_gb(N)
        sparse_mode = mem_gb > DENSE_MEMORY_THRESHOLD_GB
        for L in L_values:
            if verbose:
                mode_str = (f"SPARSE (dense would need {mem_gb:.1f} GB)"
                              if sparse_mode else f"dense ({mem_gb:.2f} GB)")
                print(f"  -> Measuring N={N}, L={L} ({n_trials} trials, "
                      f"{mode_str}) ...", flush=True)
            res = empirical_T_Valiant(N, L, d_0=d_0, n_trials=n_trials,
                                        seed=seed, max_runtime_s=max_runtime_s,
                                        per_trial_timeout_s=per_trial_timeout_s,
                                        verbose=verbose)  # propagate to per-trial output
            if res["median"] is None:
                print(f"  {N:>7} | {L:>3} | {log_n:>9.2f} | {'?':>9} | "
                      f"{'?':>7} | {res['elapsed_s']:>8.1f} | {0:>3} | "
                      f"{res['n_timeouts']:>7} | all trials failed/timeout")
                continue
            k = res["median"] / log_n
            note = ""
            if res["n_timeouts"] > 0:
                note = f"{res['n_timeouts']} timeouts at {res['per_trial_timeout_s']}s cap"
            print(f"  {N:>7} | {L:>3} | {log_n:>9.2f} | {res['median']:>9} | "
                  f"{k:>7.3f} | {res['elapsed_s']:>8.1f} | "
                  f"{res['n_trials_completed']:>3} | {res['n_timeouts']:>7} | {note}")
            results.append({
                "N": N, "L": L, "log_n": log_n, "T_median": res["median"],
                "k_emp": k, "elapsed_s": res["elapsed_s"],
                "n_trials": res["n_trials_completed"],
                "n_timeouts": res["n_timeouts"],
            })

    print()
    print("=" * 90)
    print("SUMMARY: k_emp by L_layers")
    print("=" * 90)
    by_L = {}
    for r in results:
        by_L.setdefault(r["L"], []).append(r["k_emp"])
    for L in sorted(by_L):
        ks = by_L[L]
        print(f"  L = {L:>3}: k_emp = {np.mean(ks):.3f} +/- {np.std(ks):.3f} "
              f"(n = {len(ks)}; range [{min(ks):.3f}, {max(ks):.3f}])")
    print()
    print("Paper draft Theorem 4.1 uses k_emp ~= 0.5 for L >= 8.")
    print("Validate that the value stays ~0.5 across the tested N range.")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=None,
                    help="N values (default: 225, 625, 1225, 2500, 5625, 11025)")
    p.add_argument("--L", type=int, nargs="+", default=[4, 8, 16],
                    help="L_layers values (default: 4 8 16)")
    p.add_argument("--trials", type=int, default=3,
                    help="trials per (N, L) (default: 3)")
    p.add_argument("--d0", type=int, default=8,
                    help="per-layer regularity (default: 8)")
    p.add_argument("--max-runtime", type=int, default=None,
                    help="max seconds per (N, L) before truncating trials")
    p.add_argument("--per-trial-timeout", type=int, default=None,
                    help="hard per-trial timeout in seconds (default: max-runtime/trials, or 1800)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = parse_args()
    n_default = [225, 625, 1225, 2500, 5625, 11025]
    n_values = args.n if args.n is not None else n_default
    run_sweep(n_values, args.L, n_trials=args.trials, d_0=args.d0,
               max_runtime_s=args.max_runtime,
               per_trial_timeout_s=args.per_trial_timeout, seed=args.seed)
