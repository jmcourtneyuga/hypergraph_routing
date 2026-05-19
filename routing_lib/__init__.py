"""routing_lib — shared utilities for the Hypergraph routing papers.

This package consolidates primitives that were previously duplicated
across 18 standalone scripts. Submodules:

    graphs     — graph and hypergraph constructors
    spectral   — eigenvalue / beta / Ramanujan-bound helpers
    routing    — BFS, Valiant routing, Koenig edge-coloring
    blocks     — Paper II block placement and quotient-graph machinery
    greedy     — displacement-energy / greedy-matching helpers (Paper I §9)

Several primitives appear in multiple variants under distinct names. The
docstring on each function names which scripts use it. This is deliberate:
the original scripts diverged on beta normalization, torus-vs-open grid
distance, eigvalsh source library, etc., and the refactor preserves
bit-identical output by keeping all variants.
"""
