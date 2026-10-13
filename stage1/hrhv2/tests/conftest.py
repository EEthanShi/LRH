"""Shared fixtures for the hrhv2 test suite.

Vendored reference code is read from this project's ``vendor/`` directory
(PRH platonic-rep @ dcd76ba3, SOG/ARH what-converges-prh @ be654a37). Only
the needed top-level functions are compiled out of each file (AST
extraction), so the heavy optional imports of those packages are never run.
"""
import ast
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                  # .../hrhv2
ROOT = os.path.dirname(PKG)                  # project dir containing the package
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VENDOR = os.path.join(ROOT, "vendor")
PRH_METRICS = os.path.join(VENDOR, "platonic-rep", "metrics.py")
ARH_DIR = os.path.join(VENDOR, "sog_what_converges_prh", "aristotelian")
ARH_UTILS = os.path.join(ARH_DIR, "metrics", "utils.py")
ARH_PREPROCESS = os.path.join(ARH_DIR, "prh", "preprocess.py")
SOG_TOPO_ALIGN = os.path.join(ARH_DIR, "prh", "topology_alignment.py")
SOG_LAYERWISE = os.path.join(ARH_DIR, "experiments", "layerwise_engine.py")
SOG_TOPOLOGY = os.path.join(ARH_DIR, "metrics", "topology.py")


def rand_features(m, d, key):
    from hrhv2.util import rng_from_key
    return rng_from_key(key).standard_normal((m, d))


def rand_dist(m, key, d=5):
    """Squared Euclidean distances of random Gaussian points (tie-free a.s.)."""
    from hrhv2.prep import Geom
    return Geom.from_features(rand_features(m, d, key), "none").D


def int_dist(m, key):
    """Symmetric hollow matrix whose upper-triangle entries are a random
    permutation of 1..m(m-1)/2 (all distinct integers, exactly representable)."""
    from hrhv2.util import rng_from_key
    n = m * (m - 1) // 2
    vals = rng_from_key(key).permutation(n).astype(np.float64) + 1.0
    D = np.zeros((m, m))
    iu = np.triu_indices(m, 1)
    D[iu] = vals
    return D + D.T


def tied_dist(m, key, levels=3):
    """Symmetric hollow matrix with entries in {1..levels} (massive ties)."""
    from hrhv2.util import rng_from_key
    r = rng_from_key(key).integers(1, levels + 1, size=(m, m)).astype(np.float64)
    D = np.triu(r, 1)
    return D + D.T


def load_functions_from_file(path, names, extra_globals=None):
    """Compile only the named top-level functions / class methods of a file."""
    src = open(path).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            keep.append(node)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in names:
                    sub.decorator_list = []
                    keep.append(sub)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = dict(extra_globals or {})
    exec(compile(mod, path, "exec"), ns)
    missing = set(names) - set(ns)
    if missing:
        raise RuntimeError(f"functions not found in {path}: {missing}")
    return ns


@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="session")
def prh(torch):
    if not os.path.exists(PRH_METRICS):
        pytest.skip("vendored PRH metrics.py not available")
    return load_functions_from_file(
        PRH_METRICS, {"mutual_knn", "compute_nearest_neighbors", "hsic_unbiased", "hsic_biased",
                      "cka", "remove_outliers"},
        {"torch": torch, "np": np})


@pytest.fixture(scope="session")
def arh(torch):
    if not os.path.exists(ARH_UTILS):
        pytest.skip("vendored ARH code not available")
    ns = load_functions_from_file(ARH_UTILS, {"hsic_unbiased", "hsic_biased", "center_gram"},
                                  {"torch": torch, "np": np})
    ns.update(load_functions_from_file(ARH_PREPROCESS, {"remove_outliers"}, {"torch": torch}))
    return ns


@pytest.fixture(scope="session")
def sog(torch):
    if not os.path.exists(SOG_TOPO_ALIGN):
        pytest.skip("vendored SOG code not available")
    ns = load_functions_from_file(
        SOG_TOPO_ALIGN, {"_mst_distance_overlap_similarity_from_cached", "_normalize_distance_array"},
        {"torch": torch, "np": np})
    ns.update(load_functions_from_file(SOG_LAYERWISE, {"_knn_distance_overlap_from_log_values"},
                                       {"torch": torch, "np": np}))
    ns.update(load_functions_from_file(SOG_TOPOLOGY, {"_quantile_normalize"},
                                       {"torch": torch, "np": np, "EPS": 1e-8}))
    return ns
