"""Cross-checks against vendored PRH (dcd76ba) and ARH (785054b) code.

PRH metrics.py imports torchaudio/sklearn (absent from the venv), so the needed
functions are extracted with `ast` from the vendored file without modifying it.
ARH metric files are verbatim copies in tests/vendor_arh_785054b/.
"""
import ast
import os
import textwrap

import numpy as np
import pytest

from hrhv2_ref import ladder_ref
from hrhv2_ref.core import gram_sym, preprocess, prh_clip_threshold
from hrhv2_ref import geometric as geo

torch = pytest.importorskip("torch")

# prefer this project's vendored PRH (dcd76ba); fall back to the identical copy of the earlier project
PRH = os.path.expanduser("~/hrh_v2_stage1_20261008/vendor/platonic-rep/metrics.py")
if not os.path.exists(PRH):
    PRH = os.path.expanduser("~/hrh_finite_20260922/vendor/platonic-rep/metrics.py")


def load_prh():
    src = open(PRH).read()
    tree = ast.parse(src)
    ns = {"torch": torch, "np": np}
    wanted_funcs = {"hsic_unbiased", "hsic_biased", "compute_nearest_neighbors", "remove_outliers"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            exec(compile(ast.Module([node], []), PRH, "exec"), ns)
        if isinstance(node, ast.ClassDef) and node.name == "AlignmentMetrics":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in ("mutual_knn", "cka"):
                    fn = ast.FunctionDef(name="prh_" + sub.name, args=sub.args, body=sub.body,
                                         decorator_list=[], returns=None, type_comment=None,
                                         type_params=[])
                    ast.fix_missing_locations(fn)
                    exec(compile(ast.Module([fn], []), PRH, "exec"), ns)
    return ns


def feats(seed, m=300, d=48, noise=0.6):
    rng = np.random.default_rng(seed)
    X = rng.standard_t(4, size=(m, d))
    Y = X @ rng.normal(size=(d, d)) / np.sqrt(d) + noise * rng.standard_t(4, size=(m, d))
    return X, Y


@pytest.mark.skipif(not os.path.exists(PRH), reason="vendored PRH metrics.py not present")
def test_prh_mutual_knn_and_cka():
    ns = load_prh()
    X, Y = feats(40)
    Xp, _ = preprocess(X, "l2")
    Yp, _ = preprocess(Y, "l2")
    tX, tY = torch.from_numpy(Xp), torch.from_numpy(Yp)
    out = ladder_ref(X, Y, "l2", kgrid=[1, 3, 10, 30], parts={"knn", "cka"})
    for k in (1, 3, 10, 30):
        # PRH builds float32 masks, so its score is the float32-rounded mean: compare
        # the implied integer overlap count exactly, and the value to float32 precision
        v = ns["prh_mutual_knn"](tX, tY, k)
        assert round(v * 300 * k) == round(out[f"s_N@{k}"] * 300 * k)
        assert v == pytest.approx(out[f"s_N@{k}"], abs=1e-7)
    prh_b = ns["prh_cka"](tX, tY, kernel_metric="ip", unbiased=False)
    prh_u = ns["prh_cka"](tX, tY, kernel_metric="ip", unbiased=True)
    KA, KB = gram_sym(Xp), gram_sym(Yp)
    ours_b_eps6 = geo.cka_from_hsic(geo.hsic_biased(KA, KB), geo.hsic_biased(KA, KA),
                                    geo.hsic_biased(KB, KB), eps=1e-6)
    assert prh_b == pytest.approx(ours_b_eps6, abs=1e-12)
    assert prh_b == pytest.approx(out["cka_biased"], abs=1e-8)
    # PRH adds +1e-6 to the denominator
    assert prh_u == pytest.approx(out["cka_debiased_eps6"], abs=1e-12)
    hab = ns["hsic_unbiased"](tX @ tX.T, tY @ tY.T).item()
    assert hab == pytest.approx(out["hsic_u_AB"], rel=1e-10)


@pytest.mark.skipif(not os.path.exists(PRH), reason="vendored PRH metrics.py not present")
def test_prh_remove_outliers():
    ns = load_prh()
    X, _ = feats(41, m=50, d=33)
    t = torch.from_numpy(X)
    ref = ns["remove_outliers"](t, q=0.95).numpy()
    c = prh_clip_threshold(X)
    assert np.allclose(ref, np.clip(X, -c, c), atol=1e-13, rtol=0)
    # stacked (n, L, d) -> one threshold per model
    S = np.stack([X, 2 * X], axis=1)
    ref_s = ns["remove_outliers"](torch.from_numpy(S), q=0.95).numpy()
    cs = prh_clip_threshold(S)
    assert np.allclose(ref_s, np.clip(S, -cs, cs), atol=1e-13, rtol=0)


def test_arh_hsic_cka_and_preprocess():
    from vendor_arh_785054b import estimators as arh_est
    from vendor_arh_785054b import preprocess as arh_pre
    from vendor_arh_785054b import utils as arh_utils
    X, Y = feats(42)
    Xp, _ = preprocess(X, "prh_clip+l2")
    Yp, _ = preprocess(Y, "prh_clip+l2")
    KA, KB = gram_sym(Xp), gram_sym(Yp)
    tX, tY = torch.from_numpy(Xp), torch.from_numpy(Yp)
    # ARH utils.hsic_unbiased (Song) on uncentred Grams, as in CKAUnbiased
    for (Ka, Kb, tK, tL) in ((KA, KB, tX @ tX.T, tY @ tY.T), (KA, KA, tX @ tX.T, tX @ tX.T)):
        assert arh_utils.hsic_unbiased(tK, tL).item() == pytest.approx(geo.hsic_song(Ka, Kb), rel=1e-11)
    out = ladder_ref(X, Y, "prh_clip+l2", parts={"cka"})
    hab = arh_utils.hsic_unbiased(tX @ tX.T, tY @ tY.T)
    haa = arh_utils.hsic_unbiased(tX @ tX.T, tX @ tX.T)
    hbb = arh_utils.hsic_unbiased(tY @ tY.T, tY @ tY.T)
    arh_cka_unb = (hab / (torch.sqrt(haa * hbb) + arh_utils.EPS)).item()  # CKAUnbiased._compute_raw
    assert arh_cka_unb == pytest.approx(out["cka_debiased_eps8"], abs=1e-12)
    # ARH estimators.cka_debiased (U-centering, sums without 1/(m(m-3))) == exact Song CKA
    assert arh_est.cka_debiased(tX, tY) == pytest.approx(out["cka_debiased"], abs=1e-9)
    assert arh_est.cka_biased(tX, tY) == pytest.approx(out["cka_biased"], abs=1e-9)
    # ARH preprocess.remove_outliers
    c_ref = arh_pre.remove_outliers(torch.from_numpy(X), q=0.95).numpy()
    c = prh_clip_threshold(X)
    assert np.allclose(c_ref, np.clip(X, -c, c), atol=1e-13, rtol=0)
