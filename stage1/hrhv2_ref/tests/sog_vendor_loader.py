"""Load the SOG distance-aware overlap functions from the pinned SOG checkout
(github junwon0/what-converges-prh, commit be654a37, vendored read-only at
~/hrh_v2_stage1_20261008/vendor/sog_what_converges_prh) WITHOUT importing the
package (its __init__ chain pulls model/data dependencies).  Each wanted
top-level function is extracted with `ast` and exec'd into one namespace; the
vendored files are not modified.
"""
from __future__ import annotations

import __future__
import ast
import os
import subprocess
import typing

import numpy as np

SOG_ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008/vendor/sog_what_converges_prh")
SOG_COMMIT = "be654a37"

WANTED = {
    "aristotelian/metrics/knn.py": ["_compute_knn_indices"],
    "aristotelian/metrics/topology.py": ["_pairwise_distances", "_quantile_normalize"],
    "aristotelian/experiments/layerwise_engine.py": [
        "_normalize_distance_matrix", "_knn_distance_matrix_from_layer",
        "_knn_log_distance_values_from_layer", "_knn_distance_overlap_from_log_values"],
    "aristotelian/prh/topology_alignment.py": [
        "_normalize_distance_array", "_mst_edge_codes_numpy", "_permute_edge_codes",
        "_mst_distance_overlap_similarity_from_cached"],
}


def sog_head_commit(root=SOG_ROOT):
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return None


def load_sog(root=SOG_ROOT):
    import torch
    ns = {"torch": torch, "np": np, "EPS": 1e-8}  # metrics/utils.py EPS (used by _quantile_normalize)
    ns.update({k: getattr(typing, k) for k in ("Literal", "Sequence", "Dict", "Tuple")})
    flags = __future__.annotations.compiler_flag
    for rel, names in WANTED.items():
        path = os.path.join(root, rel)
        tree = ast.parse(open(path).read(), filename=path)
        found = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                code = compile(ast.Module([node], []), path, "exec", flags=flags, dont_inherit=True)
                exec(code, ns)
                found.add(node.name)
        missing = set(names) - found
        if missing:
            raise RuntimeError(f"{rel}: functions not found in vendored SOG code: {sorted(missing)}")
    # check EPS really is 1e-8 in the vendored metrics/utils.py
    utils_src = open(os.path.join(root, "aristotelian/metrics/utils.py")).read()
    if "EPS = 1e-8" not in utils_src:
        raise RuntimeError("vendored SOG metrics/utils.py EPS changed")
    return ns
