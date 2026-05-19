"""Smoke tests: every routing_lib module imports and exposes its
documented public surface."""
import pytest

import routing_lib


def test_package_has_docstring():
    assert routing_lib.__doc__ and "routing_lib" in routing_lib.__doc__


def test_all_submodules_importable():
    import routing_lib.blocks  # noqa: F401
    import routing_lib.graphs  # noqa: F401
    import routing_lib.greedy  # noqa: F401
    import routing_lib.routing  # noqa: F401
    import routing_lib.spectral  # noqa: F401
    import routing_lib._sparse_compat  # noqa: F401


@pytest.mark.parametrize("module,attrs", [
    ("routing_lib.blocks",
     ["Block", "BlockConfig", "random_block_config", "place_blocks_bfs",
      "quotient_graph", "build_quotient_graph"]),
    ("routing_lib.spectral",
     ["spectral_params_lam2_over_d", "spectral_params_minimal",
      "spectral_params_abs_eigs", "spectral_params_covering_tower",
      "compute_spectral_params", "compute_graph_spectral_params",
      "ramanujan_bound", "ramanujan_bound_hyper",
      "theoretical_routing_bound", "diameter_from_matrix"]),
    ("routing_lib.graphs",
     ["random_regular_config_model", "random_regular_matching_union",
      "random_regular_networkx", "build_2d_grid_clique",
      "build_3d_aol_clique", "build_host_graph_d_dprime",
      "fano_plane", "pg23", "clique_expansion", "voltage_covering"]),
    ("routing_lib.routing",
     ["bfs_path", "compute_edge_congestion", "valiant_routing_simulation",
      "simulate_block_routing", "derandomize_valiant",
      "koenig_edge_color_simple", "chromatic_index_block_translation",
      "chromatic_index_corridor"]),
    ("routing_lib.greedy",
     ["grid_distance_open", "grid_distance_torus", "displacement_energy",
      "n_displaced", "greedy_matching", "random_matching", "apply_matching"]),
])
def test_module_exposes_public_api(module, attrs):
    import importlib
    m = importlib.import_module(module)
    missing = [a for a in attrs if not hasattr(m, a)]
    assert not missing, f"{module} is missing: {missing}"
