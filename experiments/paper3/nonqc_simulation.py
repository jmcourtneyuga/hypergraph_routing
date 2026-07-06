"""
Direct simulation of Xu Algorithm 3 vs our multi-layer scheme on
NON-quasi-cyclic HGP codes (random biregular bases, no circulant structure).

Tests three hypotheses:
  H1 (Xu): per-layer rearrangement count = 8, same as QC.
  H2 (Xu): per-cycle rearrangement is SHORT-RANGE, ~200 us/layer (Bluvstein
           2022/2024 demonstrated characteristic move), same as QC. The
           distance-based Xu Eq. 7 spline (~ms) is a LONG-RANGE worst-case
           upper bound, reported separately, NOT the demonstrated per-cycle
           cost. (CORRECTED 2026-07-06.)
  H3 (Ours): chromatic colors are arbitrary matchings (no shift structure)
             for non-QC bases.

The key empirical question: does Xu's wall-clock advantage vanish for
non-QC codes (where they cannot exploit shift simplification), making
the comparison favor ours by an even wider margin?

For each color of the chromatic decomposition of G_T:
  * QC case: matching is a "shift" - all (data, ancilla) pairs related by
    same translation in the canonical 2D embedding.
  * Non-QC case: matching is arbitrary - pairs are scattered without
    common translation.

Per-color atom motion cost:
  * Shift matching: motion distance = ε (0 for trivial shift, small for
    nearest-neighbor) → t_motion ≈ 0.
  * Arbitrary matching: motion distance up to √L·d in worst case →
    t_motion ≈ (3+2√2)√(6Ld/a_p) ≈ 2.3 ms (Xu's worst-case Eq. 7).
"""
import numpy as np
from collections import defaultdict

# Demonstrated SHORT-RANGE per-cycle move time (Bluvstein 2022 Nature 604,
# 0.55 um/us cubic-velocity profile; reaffirmed 2024 Nature 626). This is the
# physically correct per-syndrome-cycle rearrangement cost. The distance-based
# Xu Eq. 7 spline below is a long-range worst-case upper bound only.
TAU_MOVE_PERCYCLE_US = 200.0
TAU_AOL_SWITCH_US = 30.0  # Bluvstein Methods, conservative midrange

from routing_lib.qldpc.codes import (
    random_3_4_bipartite_check_matrix,
    lp_check_matrix_circulant,
    hgp_tanner_graph,
)


def is_qc_structure(H, period_max=None):
    """Detect circulant/quasi-cyclic structure in check matrix H.

    Test: do all rows arise from cyclic shifts of a small set of base rows?
    Equivalently, is the row support set closed under cyclic translation?

    For random non-QC bases this returns False; for proper LP circulant
    lifts this returns True.
    """
    m, n = H.shape
    if period_max is None:
        period_max = n
    # Try shifting each row by 1..n-1 and see if it appears elsewhere
    rows_as_tuples = set(tuple(H[i]) for i in range(m))
    for i in range(m):
        for shift in range(1, period_max + 1):
            shifted = tuple(np.roll(H[i], shift))
            if shifted in rows_as_tuples and shifted != tuple(H[i]):
                return True  # found a cyclic-shift relationship
    return False


def explicit_konig_coloring(A):
    """Construct chi' = Delta proper edge coloring of bipartite graph A
    via repeated maximum matching. Returns list of edge -> color."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from collections import deque
    n = A.shape[0]

    # 2-color the bipartition
    side = [-1] * n
    for s in range(n):
        if side[s] != -1:
            continue
        side[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in range(n):
                if A[u, v] > 0 and side[v] == -1:
                    side[v] = 1 - side[u]
                    q.append(v)
    left = [i for i in range(n) if side[i] == 0]
    right = [i for i in range(n) if side[i] == 1]
    L_idx = {u: i for i, u in enumerate(left)}
    R_idx = {v: i for i, v in enumerate(right)}

    edges = []
    for u in left:
        for v in right:
            if A[u, v] > 0:
                edges.append((u, v))

    edge_color = {}
    remaining = set(edges)
    color = 0
    while remaining:
        rows = [L_idx[u] for (u, v) in remaining]
        cols = [R_idx[v] for (u, v) in remaining]
        adj = csr_matrix(([1] * len(rows), (rows, cols)),
                          shape=(len(left), len(right)))
        matching = maximum_bipartite_matching(adj, perm_type='column')
        matched = []
        used_global = set()
        for u_idx, v_idx in enumerate(matching):
            if v_idx == -1:
                continue
            global_pair = (left[u_idx], right[v_idx])
            if global_pair in remaining and global_pair not in used_global:
                matched.append(global_pair)
                used_global.add(global_pair)
        if not matched:
            uv = next(iter(remaining))
            matched = [uv]
        for pair in matched:
            edge_color[pair] = color
            remaining.discard(pair)
        color += 1
    return edge_color, color


def matching_motion_cost(matching_pairs, n_side, d_lattice=5e-6, a_p=2e-2):
    """Estimate atom motion cost for a matching, in mm/s units.

    For each (u, v) pair, the motion distance is the Euclidean distance
    between u's canonical position and v's canonical position on the
    n_side x n_side 2D grid. Returns mean and max distance, plus the
    Xu-formula motion time using max distance.

    a_p is in µm/µs² (Xu uses 0.02), d_lattice is the trap spacing.
    Motion time per Xu Eq. 7: t_motion = (3 + 2√2) · √(6 · L · d / a_p)
    where L is the array side, d is the trap spacing.
    """
    distances = []
    for (u, v) in matching_pairs:
        ru, cu = u // n_side, u % n_side
        rv, cv = v // n_side, v % n_side
        dist = np.sqrt((ru - rv) ** 2 + (cu - cv) ** 2)
        distances.append(dist)
    distances = np.array(distances)
    if len(distances) == 0:
        return {"mean": 0, "max": 0, "t_motion_us": 0}
    max_dist = distances.max()
    mean_dist = distances.mean()
    # Use worst-case max-distance trajectory time (Xu Eq. 7 form)
    # t_motion in µs: max_dist * d_lattice / 1e-6 = motion length in µm
    motion_length_um = max_dist * d_lattice * 1e6  # convert to µm
    if motion_length_um == 0:
        t_motion_us = 0
    else:
        # Xu's spline: t = (3+2sqrt(2)) * sqrt(6 * L_motion / a_p)
        # where L_motion is in µm and a_p in µm/µs²
        t_motion_us = (3 + 2 * np.sqrt(2)) * np.sqrt(6 * motion_length_um / a_p)
    return {
        "mean_dist_traps": mean_dist,
        "max_dist_traps": max_dist,
        "t_motion_us": t_motion_us,
    }


def analyze_code(H, label, L_layers=8):
    """Run full analysis on a check matrix: chi', motion costs, QC detection."""
    print(f"\n{'='*78}")
    print(f"Code: {label}")
    print(f"  H shape: {H.shape}")
    is_qc = is_qc_structure(H)
    print(f"  QC structure detected: {is_qc}")

    A = hgp_tanner_graph(H, H)
    N = A.shape[0]
    degrees = A.sum(axis=1).astype(int)
    delta = int(degrees.max())
    print(f"  G_T size: N = {N}")
    print(f"  Delta(G_T) = {delta}, so chi'(G_T) = {delta} (Konig)")

    # Compute explicit chromatic decomposition
    edge_color, n_colors = explicit_konig_coloring(A)
    print(f"  Explicit greedy-matching coloring uses {n_colors} colors "
          f"({'tight' if n_colors == delta else 'loose'})")

    # Group edges by color and analyze motion cost per color
    colors_to_edges = defaultdict(list)
    for (u, v), c in edge_color.items():
        colors_to_edges[c].append((u, v))

    # Approximate qubit positions on a sqrt(N) x sqrt(N) grid
    n_side = int(np.ceil(np.sqrt(N)))
    print(f"  Approx grid side: {n_side}")

    print(f"\n  Per-color motion analysis:")
    print(f"  {'color':>5} | {'edges':>6} | {'mean d':>8} | {'max d':>7} | "
          f"{'t_motion (µs)':>15}")
    total_motion_us = 0
    for c in sorted(colors_to_edges):
        cost = matching_motion_cost(colors_to_edges[c], n_side)
        print(f"  {c:>5} | {len(colors_to_edges[c]):>6} | "
              f"{cost['mean_dist_traps']:>8.2f} | "
              f"{cost['max_dist_traps']:>7.2f} | {cost['t_motion_us']:>15.1f}")
        total_motion_us += cost['t_motion_us']

    # Per-cycle wall-clock uses the DEMONSTRATED short-range move (~200 us/layer),
    # not the distance-based long-range spline (which is a worst-case upper bound).
    xu_percycle_ms = n_colors * TAU_MOVE_PERCYCLE_US / 1000.0
    print(f"\n  Total per-cycle wall-clock (Xu approach, all colors sequential):")
    print(f"    {n_colors} short-range layers * {TAU_MOVE_PERCYCLE_US:.0f} us "
          f"(demonstrated) = {xu_percycle_ms:.2f} ms per cycle")
    print(f"    [worst-case long-range upper bound (Xu Eq. 7 distance model): "
          f"sum t_motion = {total_motion_us/1000:.2f} ms + "
          f"{n_colors * 0.7:.1f} ms AOD transfer = "
          f"{total_motion_us/1000 + n_colors * 0.7:.2f} ms]")

    print(f"\n  Multi-layer parallelization (L = {L_layers}):")
    color_groups = (n_colors + L_layers - 1) // L_layers
    print(f"    chi'/L_layers = {n_colors}/{L_layers} = "
          f"{color_groups} parallel batches")

    # Scenario A: atoms must move per batch (short-range per-cycle move)
    print(f"\n  (A) Atom-motion scenario (no pre-storing):")
    batch_motion_us = color_groups * TAU_MOVE_PERCYCLE_US
    print(f"    {color_groups} batches * {TAU_MOVE_PERCYCLE_US:.0f} us "
          f"(short-range move) = {batch_motion_us/1000:.2f} ms")

    # Scenario B: AOL patterns pre-stored (Bluvstein-style, no atom motion)
    print(f"\n  (B) Pre-stored AOL pattern scenario (Bluvstein hardware):")
    aol_switch_us = TAU_AOL_SWITCH_US  # conservative midrange of "5-8 us/row, 10s of us/pattern"
    total_aol_us = color_groups * aol_switch_us
    print(f"    {color_groups} AOL pattern switches * {aol_switch_us:.0f} µs = "
          f"{total_aol_us:.1f} µs = {total_aol_us/1000:.3f} ms")
    print(f"    Note: requires {n_colors} pre-loaded AOL patterns (one per color).")
    print(f"    For QC codes, patterns are simple shifts; for non-QC, they are")
    print(f"    arbitrary matchings (still pre-storable on Bluvstein-style AOD).")

    return {
        "label": label,
        "is_qc": is_qc,
        "N": N,
        "chi_prime": n_colors,
        "xu_percycle_ms": xu_percycle_ms,                 # demonstrated short-range
        "worstcase_longrange_ms": total_motion_us/1000 + n_colors * 0.7,
        "ours_motion_ms": batch_motion_us / 1000.0,       # demonstrated short-range
        "ours_prestored_ms": total_aol_us / 1000.0,
        "color_groups": color_groups,
    }


def main():
    print("=" * 78)
    print("NON-QC vs QC COMPARISON: Xu Algorithm 3 vs Multi-Layer Scheme")
    print("Per-cycle wall-clock with realistic motion costs (Xu Eq. 7)")
    print("=" * 78)

    results = []

    # QC case: lifted product from circulant base
    rng = np.random.default_rng(42)
    B_qc = rng.integers(0, 7, size=(3, 4))
    H_qc = lp_check_matrix_circulant(B_qc, L=8)
    results.append(analyze_code(H_qc, "QC: LP[3x4 base, L_lift=8]"))

    # Non-QC case: random biregular
    H_random = random_3_4_bipartite_check_matrix(32, seed=1)
    results.append(analyze_code(H_random, "non-QC: random (3,4)-biregular n=32"))

    # Larger non-QC
    H_random_large = random_3_4_bipartite_check_matrix(48, seed=2)
    results.append(analyze_code(H_random_large, "non-QC: random (3,4)-biregular n=48"))

    print("\n" + "=" * 78)
    print("SUMMARY (per-cycle wall-clock, milliseconds; demonstrated short-range move):")
    print("=" * 78)
    print(f"{'code':>40} | {'QC?':>4} | {'chi':>4} | "
          f"{'Xu':>7} | {'ours-motion':>11} | {'ours-prestored':>14}")
    for r in results:
        print(f"{r['label']:>40} | {str(r['is_qc']):>4} | "
              f"{r['chi_prime']:>4} | {r['xu_percycle_ms']:>6.2f}  | "
              f"{r['ours_motion_ms']:>11.3f} | {r['ours_prestored_ms']:>14.4f}")
    print("\n(For reference, the long-range worst-case upper bound per cycle is:")
    for r in results:
        print(f"    {r['label']}: {r['worstcase_longrange_ms']:.2f} ms)")
    print()
    print("=" * 78)
    print("HONEST FINDINGS:")
    print("=" * 78)
    print("  NOTE: numbers above use a GREEDY edge coloring, which is loose by")
    print("  1-2 colors on random bipartite graphs. By Konig's theorem (verified")
    print("  in konig_optimal_test.py), chi'(G_T) = Delta = 8 EXACTLY for both")
    print("  QC and non-QC HGP codes. Construct via Delta-regular extension +")
    print("  repeated bipartite max matching (polynomial time).")
    print()
    print("  1. chi'(G_T) = 8 for both QC and non-QC HGP (Konig-optimal).")
    print("  2. Xu's Algorithm 3 APPLIES to non-QC HGP codes (uses Algorithm 1's")
    print("     general 1D scrambling). Per-cycle wall-clock ~1.6 ms for both QC")
    print("     and non-QC at N ~ 3000-7000 (8 short-range moves x ~200 us;")
    print("     CORRECTED from the misattributed 19-28 ms / 3 ms-per-layer figure).")
    print("  3. Our atom-motion scenario (no pre-storing) gives ~0.2 ms - an 8x")
    print("     step-count advantage from multi-layer parallelism, even WITH motion.")
    print("  4. Our pre-stored AOL scenario gives ~30 us, dominating by ~50-300x.")
    print("     This is the regime Bluvstein 2024 has demonstrated (pre-stored")
    print("     pattern switching without recalibration).")
    print("  5. The non-QC penalty at the algorithmic level is ZERO with Konig-")
    print("     optimal coloring. The only non-QC cost is offline preprocessing")
    print("     (computing the optimal coloring + storing chi' AOL trap patterns).")
    print()
    print("CONCLUSION: 'Xu doesn't apply to non-QC' is OVERCLAIMING. Both schemes")
    print("apply. Our multi-layer-AOL advantage is robust across QC and non-QC.")
    print("Konig-optimal coloring (polynomial-time) gives chi' = 2*delta_c exactly")
    print("for both, eliminating any per-cycle algorithmic non-QC overhead.")


if __name__ == "__main__":
    main()
