#!/usr/bin/env python3
import argparse
import numpy as np

def geometric_shift_generators(n):
    gens = []
    s = 1
    while s < n:
        gens.append((s % n, 0))
        gens.append((0, s % n))
        s *= 2
    return gens


def random_shift_generators(n, n_layers, seed=0):
    rng = np.random.default_rng(seed)
    gens = set()
    while len(gens) < n_layers:
        g = (int(rng.integers(1, n)), int(rng.integers(0, n)))
        gens.add(g)
    return list(gens)


def cayley_beta(n, gens):
    a = np.arange(n)
    A, B = np.meshgrid(a, a, indexing="ij")
    lam = np.zeros((n, n), dtype=np.float64)
    for g1, g2 in gens:
        lam += 2.0 * np.cos(2.0 * np.pi / n * (A * g1 + B * g2))
    d = 2 * len(gens)
    flat = np.sort(lam.ravel())[::-1]
    lam_star = max(abs(flat[1]), abs(flat[-1]))  # exclude Perron eigenvalue d
    # torus distance of each shift, in grid units
    def tdist(g):
        dx = min(g[0], n - g[0]); dy = min(g[1], n - g[1])
        return np.hypot(dx, dy)
    max_shift = max(tdist(g) for g in gens)
    return d, lam_star / d, max_shift

def ramanujan_bound(d):
    return 2.0 * np.sqrt(d - 1) / d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+",
                    default=[16, 32, 64, 128, 256, 512],
                    help="grid side lengths n (N = n^2 atoms)")
    ap.add_argument("--family", choices=["geometric", "random"],
                    default="geometric", help="shift-generator family")
    ap.add_argument("--layers", type=int, default=8,
                    help="number of layers for --family random")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (random family)")
    args = ap.parse_args()

    print(f"# Shifted-Cayley motion barrier  (family={args.family})")
    print(f"# {'n':>5} {'N':>9} {'deg':>4} {'beta':>7} {'ram_bd':>7} "
          f"{'Ramanujan?':>11} {'max_shift':>10} {'~N^1/4':>8}")
    prev = None
    monotone = True
    for n in args.n:
        N = n * n
        if args.family == "geometric":
            gens = geometric_shift_generators(n)
        else:
            gens = random_shift_generators(n, args.layers, args.seed)
        d, beta, max_shift = cayley_beta(n, gens)
        rb = ramanujan_bound(d)
        is_ram = "yes" if beta <= rb + 1e-9 else "no"
        print(f"  {n:>5} {N:>9} {d:>4} {beta:>7.3f} {rb:>7.3f} "
              f"{is_ram:>11} {max_shift:>10.1f} {N**0.25:>8.1f}")
        if prev is not None and beta < prev - 1e-9:
            monotone = False
        prev = beta

    print()
    print(f"#   beta monotone non-decreasing in n: {monotone}")


if __name__ == "__main__":
    main()
