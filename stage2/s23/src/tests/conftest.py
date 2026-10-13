"""Excluded unit tests of S2.3 (tiny instances, keys 'hrhv2-s23-TEST|...', outputs in a temp dir).
Run with:  CUDA_VISIBLE_DEVICES= python -m pytest -q src/tests   (no GPU, never the real STATUS.md)."""
import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="s23test_")
os.environ["S23_ROOT"] = _TMP
os.environ["S23_KEY"] = "hrhv2-s23-TEST"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tiny_cfg():
    import cfg as CF
    c = CF.tiny()
    c["g0"].update({"lam_min": 0.0, "omega": 0.0, "twoview_ptp": 0.0, "ex8_dgap": 0.0})
    c["grid"].update({"j_lo": -2, "j_hi": 2, "br_offset": 8, "br_min_log2": 8, "dep_offsets": [-6, 0, 6],
                      "pilot_offsets": [-6, 6]})
    c["runs"].update({"g1b_offsets": [-6, 0, 6], "rm_points": ["br", -6]})
    return c


@pytest.fixture(scope="session")
def tiny_inst(tiny_cfg):
    import instance as I
    import s23common as C
    return I.build(C.GEOM_KEY, tiny_cfg["t"], tiny_cfg["geometry"])
