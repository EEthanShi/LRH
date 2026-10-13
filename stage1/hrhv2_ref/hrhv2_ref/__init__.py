"""hrhv2_ref — independent reference implementation of SPEC v1.0 §0–§5 (+§4 projector).

Public API:
    ladder_ref(X_A, X_B, pre, kgrid=None, ...) -> dict
        features (default), gram=True (Gram inputs) or dist=True (squared distances);
        parts= subset of ("knn","comp","rank","edge","mst","geom","cka","rsa","sog","quot","proj").
    calibrate_ref(X_A, X_B, score_fn, K=500, alpha=0.05, key=..., perms=None) -> dict
    ladder_score(name, pre) -> score_fn for calibrate_ref
    compare.compare_ladders(main_dict, ref_dict) -> SPEC §8 tolerance check

Usage (server):
    cd ~/hrh_v2_stage1_20261008/hrhv2_ref && source ../venv/bin/activate
    python -c "from hrhv2_ref import ladder_ref; ..."
    python -m pytest -q tests

All documented choices for SPEC ambiguities are listed in ladder.py (A1-A13),
quotient.py, calibrate.py and core.py docstrings.
"""
from .calibrate import calibrate_ref, critical_rank, perm_from_key
from .core import hash_halves, preprocess, prh_clip_threshold, sqdist_from_gram
from .ladder import ladder_ref, ladder_score

__all__ = ["ladder_ref", "calibrate_ref", "ladder_score", "critical_rank", "perm_from_key",
           "hash_halves", "preprocess", "prh_clip_threshold", "sqdist_from_gram"]
__version__ = "0.2.0"
