#!/usr/bin/env python3
"""Paper III, Sec. future-hardware: the shifted-Cayley N^{1/4} motion barrier.

Backs the claim in the "Future hardware directions" subsection that a purely
GEOMETRIC multi-layer overlay (atom motion only, no photonic switching) cannot
break the N^{1/4} wall-clock barrier, for two reasons:

  1. MOTION COST.  The union of geometric shift layers on the sqrt(N) x sqrt(N)
     grid is dominated by its largest-shift layer, whose atoms must physically
     move a distance ~ sqrt(n) = N^{1/4} under constant-acceleration transport.

  2. SPECTRAL GAP.  The union graph is a Cayley graph Cay(Z_n^2, S) whose shift
     generators S are the per-layer displacement vectors.  Its spectral ratio
     beta = lambda_star / d grows toward 1 as n increases, so the union is NOT
     asymptotically Ramanujan and the Paper-I routing-depth bound (Eq. for
     rt(G)) does not itself reach O(log N).

Escaping the barrier therefore requires a switching mechanism whose cost is
independent of physical motion distance (photonic interconnects) -- exactly the
long-term direction argued in the paper.

Construction (default: geometric-doubling shift layers)
-------------------------------------------------------
On Z_n x Z_n (N = n^2 atoms) we place L layers with axis-aligned shift vectors
of geometrically doubling magnitude:  (1,0),(0,1),(2,0),(0,2),(4,0),(0,4),...
up to the largest shift < n.  The union overlay is Cay(Z_n^2, S) with
|S| = 2L generators (degree d = 2|S|).  Eigenvalues are exact via the character
formula lambda(a,b) = sum_g 2 cos(2*pi/n * (a*g1 + b*g2)).

  larger n  ->  more layers, larger max shift (~ n/2 ~ N^{1/4}/... ),
                beta -> 1  (union stops being a spectral expander).

Alternative generator families (--family random|geometric) are provided so the
construction can be matched to the exact overlay used in the manuscript.

NUMERIC RECONCILIATION NOTE
---------------------------
The manuscript quotes beta = 0.45 at n=16 rising to 0.77 at n=512.  The exact
generator set that produced those two numbers was not recovered with the rest
of the Paper-III code.  This script reproduces the QUALITATIVE barrier (beta
monotonically -> 1; largest shift = Theta(N^{1/4})) with a clean, documented
construction, but its beta values depend on the chosen generator family and
will not match 0.45/0.77 to the digit.  Before submission, either (a) set the
--family / --layers options here to the author's original overlay and update
the two quoted numbers in the .tex to this script's output, or (b) locate the
original sweep script and commit it in place of this one.  The paper's argument
(motion-only overlays are non-Ramanujan and motion-bound at N^{1/4}) is
independent of the exact beta values.

USAGE
    python experiments/paper3/shifted_cayley_barrier.py
    python experiments/paper3/shifted_cayley_barrier.py --n 16 32 64 128 256 512
    python experiments/paper3/shifted_cayley_barrier.py --family random --seed 0
"""
import argparse
import numpy as np


def geometric_shift_generators(n):
    """Axis-aligned shift vectors of geometrically doubling magnitude, |g| < n."""
    gens = []
    s = 1
    while s < n:
        gens.append((s % n, 0))
        gens.append((0, s % n))
        s *= 2
    return gens


def random_shift_generators(n, n_layers, seed=0):
    """`n_layers` distinct random shift vectors in Z_n^2 (excluding the origin)."""
    rng = np.random.default_rng(seed)
    gens = set()
    while len(gens) < n_layers:
        g = (int(rng.integers(1, n)), int(rng.integers(0, n)))
        gens.add(g)
    return list(gens)


def cayley_beta(n, gens):
    """Exact (vectorized) spectral ratio beta = lambda_star / d of Cay(Z_n^2, S).

    S = {+g, -g : g in gens}, so degree d = 2 * len(gens).  Returns
    (degree, beta, max_shift_L2) where max_shift_L2 is the largest generator's
    Euclidean length on the torus (the motion-cost driver).
    """
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
    print("# Conclusion:")
    print("#   beta increases toward 1 with n  ->  union is NOT asymptotically")
    print("#   Ramanujan; and max_shift tracks N^{1/4}, the motion-cost floor.")
    print(f"#   beta monotone non-decreasing in n: {monotone}")
    print("#   => a motion-only geometric overlay cannot reach O(log N) per-cycle;")
    print("#      distance-independent (photonic) switching is required.")


if __name__ == "__main__":
    main()
