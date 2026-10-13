"""SPEC §2.10 check against the pinned SOG code (be654a37), run in float64.

SPEC: "d~ = d / Q_0.9(positive off-diagonal d). Use plain distance d unless the
pinned SOG code uses another quantity; record the check."  Recorded result:
  * SOG uses the plain (chord) distance d on L2-normalised rows in both variants
    (kNN: sqrt(clamp(2 - 2 x.x')); MST: torch.cdist) -> SPEC default is right.
  * Q_0.9 is taken over the off-diagonal matrix ENTRIES (ordered pairs):
    MST variant np.quantile(D[D > 0]); kNN variant torch.quantile(D[offdiag])
    (zeros not removed — identical unless two items coincide).
  * log is taken of max(d~, 1e-12).
hrhv2_ref v0.2 follows all three; these tests assert agreement to 1e-9 relative
when the SOG functions are fed float64 inputs (SOG itself runs float32: see
scripts/sog_crosscheck.py for the float32 gap).
"""
import os

import numpy as np
import pytest

from hrhv2_ref import ladder_ref
from hrhv2_ref.core import gram_sym, neighbor_order, preprocess, sqdist_from_gram
from hrhv2_ref import structural as st

torch = pytest.importorskip("torch")
from sog_vendor_loader import SOG_COMMIT, SOG_ROOT, load_sog, sog_head_commit  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(SOG_ROOT), reason="vendored SOG code not present")

TAUS = (1.0, 0.1, 0.01, 0.001, 0.0001)


def pair(seed, m=300):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(m, 5))
    A = Z @ rng.normal(size=(5, 32)) + 0.05 * rng.normal(size=(m, 32))
    B = np.tanh(A @ rng.normal(size=(32, 24)) / 3) + 0.15 * rng.normal(size=(m, 24))
    return A, B


def test_sog_commit_pinned():
    head = sog_head_commit()
    assert head is None or head.startswith(SOG_COMMIT)


@pytest.mark.parametrize("seed", [70, 71])
def test_sog_knn_and_mst_match_vendored_float64(seed):
    ns = load_sog()
    A, B = pair(seed)
    Ap, _ = preprocess(A, "l2")
    Bp, _ = preprocess(B, "l2")
    m = Ap.shape[0]
    out = ladder_ref(A, B, "l2", parts={"sog"})
    DA = sqdist_from_gram(gram_sym(Ap))
    DB = sqdist_from_gram(gram_sym(Bp))
    oA = neighbor_order(DA, 10)
    oB = neighbor_order(DB, 10)
    tA, tB = torch.from_numpy(Ap), torch.from_numpy(Bp)
    # SOG kNN support (torch.topk on cosine similarity) == SPEC tie-rule sets (tie-free data)
    kA = ns["_compute_knn_indices"](tA, 10).numpy()
    kB = ns["_compute_knn_indices"](tB, 10).numpy()
    assert np.array_equal(np.sort(kA, 1), np.sort(oA, 1)) and np.array_equal(np.sort(kB, 1), np.sort(oB, 1))
    nA, nB = torch.from_numpy(oA), torch.from_numpy(oB)
    lA = ns["_knn_log_distance_values_from_layer"](tA, nA, q=0.9)
    lB = ns["_knn_log_distance_values_from_layer"](tB, nB, q=0.9)
    # SOG MST variant (value path of prepare_mst_overlap_cache, fed our float64 d)
    dA, dB = np.sqrt(DA), np.sqrt(DB)
    vA = ns["_normalize_distance_array"](dA, q=0.9)
    vB = ns["_normalize_distance_array"](dB, q=0.9)
    eA = np.sort(ns["_mst_edge_codes_numpy"](vA))
    eB = np.sort(ns["_mst_edge_codes_numpy"](vB))
    assert np.array_equal(eA, st.mst_kruskal(DA)) and np.array_equal(eB, st.mst_kruskal(DB))
    assert out["sog_Q_A"] == pytest.approx(float(np.quantile(dA[dA > 0], 0.9)), rel=1e-14)
    for t in TAUS:
        sk = float(ns["_knn_distance_overlap_from_log_values"](nA, nB, lA, lB, topk=10, tau=t))
        sm = ns["_mst_distance_overlap_similarity_from_cached"](vA, eA, vB, eB, n=m, distance_tau=t)
        assert out[f"sog_knn@{t:g}"] == pytest.approx(sk, rel=1e-9, abs=1e-14), t
        assert out[f"sog_mst@{t:g}"] == pytest.approx(sm, rel=1e-9, abs=1e-14), t


def test_sog_one_zero_distance_convention():
    # duplicated item in A only -> an off-diagonal zero distance in A.
    # SOG kNN path: Q over ALL off-diagonal entries (zeros kept), log(max(d~, 1e-12)).
    # Ref: Q over positive entries (SPEC wording); weights use the same eps clamp.
    from hrhv2_ref import geometric as geo
    ns = load_sog()
    A, B = pair(72, m=120)
    A[5] = A[9]
    B[5] = B[9] + 1e-3 * np.ones(B.shape[1])  # nearly (not exactly) duplicated in B
    Ap, _ = preprocess(A, "l2")
    Bp, _ = preprocess(B, "l2")
    m = Ap.shape[0]
    DA, DB = sqdist_from_gram(gram_sym(Ap)), sqdist_from_gram(gram_sym(Bp))
    oA, oB = neighbor_order(DA, 10), neighbor_order(DB, 10)
    assert oA[5, 0] == 9 and oB[5, 0] == 9
    nA, nB = torch.from_numpy(oA), torch.from_numpy(oB)
    tA, tB = torch.from_numpy(Ap), torch.from_numpy(Bp)
    lA = ns["_knn_log_distance_values_from_layer"](tA, nA, q=0.9)
    lB = ns["_knn_log_distance_values_from_layer"](tB, nB, q=0.9)
    dA, dB = np.sqrt(DA), np.sqrt(DB)
    off = ~np.eye(m, dtype=bool)
    QA_sogknn = float(np.quantile(dA[off], 0.9))  # zeros kept, as SOG kNN path
    QB_sogknn = float(np.quantile(dB[off], 0.9))
    assert QA_sogknn != geo.sog_scale(DA)  # differs from the ref (positive-only) Q
    ii = np.repeat(np.arange(m), 10)
    sh = np.array([j in set(oB[i]) for i, j in zip(ii, oA.ravel())])
    I, J = ii[sh], oA.ravel()[sh]
    for t in (1.0, 0.01, 0.0001):
        sk = float(ns["_knn_distance_overlap_from_log_values"](nA, nB, lA, lB, topk=10, tau=t))
        w = geo.sog_weights(dA[I, J], dB[I, J], QA_sogknn, QB_sogknn, t)
        assert w.sum() / (m * 10) == pytest.approx(sk, rel=1e-9, abs=1e-15), t
    # the zero-distance pair (5, 9) gets the SOG eps weight, not 0
    w59 = geo.sog_weights(np.array([0.0]), np.array([dB[5, 9]]), 1.0, 1.0, 1.0)[0]
    assert 0.0 < w59 < 1e-6
