import os

import numpy as np
import pytest

from hrhv2_ref import calibrate_ref, critical_rank, ladder_score, perm_from_key

ARH_CAL = os.path.expanduser("~/hrh_finite_20260922/vendor/aristotelian/calibrated_similarity/calibration.py")


def test_critical_rank():
    assert critical_rank(500, 0.05) == 476
    assert critical_rank(19, 0.05) == 19
    assert critical_rank(99, 0.05) == 95
    assert critical_rank(9, 0.05) == 10
    assert critical_rank(4, 0.4) == 3


def _first_row_score(A, B):
    return B[0, 0] / 10.0


def test_calibrate_hand():
    XA = np.zeros((5, 1))
    XB = np.array([[5.0], [1.0], [2.0], [3.0], [4.0]])
    perms = [np.roll(np.arange(5), -s) for s in (1, 2, 3, 4)]  # pi(0) = 1, 2, 3, 4
    r = calibrate_ref(XA, XB, _first_row_score, perms=perms, alpha=0.05)
    assert r["null"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert r["c_alpha"] == pytest.approx(0.5) and r["s_cal"] == 0.0
    assert r["p_value"] == pytest.approx(0.2)
    r = calibrate_ref(XA, XB, _first_row_score, perms=perms, alpha=0.4)
    assert r["rank_r"] == 3 and r["c_alpha"] == pytest.approx(0.3)
    assert r["s_cal"] == pytest.approx(0.2 / 0.7)
    assert r["uncensored"] == pytest.approx(0.25)


def test_layer_max_uses_same_perm_for_all_layers():
    XA = [np.zeros((5, 1))]
    B1 = np.array([[5.0], [1.0], [2.0], [3.0], [4.0]])
    B2 = B1[::-1].copy()  # row 0 = 4, row1 = 3, ...
    perms = [np.roll(np.arange(5), -1)]
    r = calibrate_ref(XA, [B1, B2], _first_row_score, perms=perms)
    # observed: max(0.5, 0.4) ; perm pi(0)=1 -> max(0.1, 0.3)
    assert r["s_obs"] == pytest.approx(0.5)
    assert r["null"].tolist() == pytest.approx([0.3])


def test_perm_from_key_deterministic():
    a = perm_from_key("x", 3, 50)
    assert np.array_equal(a, perm_from_key("x", 3, 50))
    assert not np.array_equal(a, perm_from_key("x", 4, 50))
    assert sorted(a.tolist()) == list(range(50))


def test_knn_calibration_runs_and_is_significant():
    rng = np.random.default_rng(30)
    X = rng.normal(size=(60, 8))
    Y = X + 0.3 * rng.normal(size=(60, 8))
    r = calibrate_ref(X, Y, ladder_score("s_N@5", pre="l2"), K=49, key="t")
    assert r["p_value"] == pytest.approx(1 / 50)
    assert r["s_cal"] > 0


@pytest.mark.skipif(not os.path.exists(ARH_CAL), reason="vendored ARH calibration.py not present")
def test_matches_vendored_arh_calibrate():
    torch = pytest.importorskip("torch")
    import importlib.util
    spec = importlib.util.spec_from_file_location("arh_calibration", ARH_CAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(31)
    X = rng.normal(size=(40, 6))
    Y = X + 0.8 * rng.normal(size=(40, 6))
    K = 99
    perms = [perm_from_key("arh-match", k, 40) for k in range(K)]
    score = ladder_score("cka_biased", pre="none")
    ours = calibrate_ref(X, Y, score, perms=perms)
    it = iter(perms)

    def perm_fn(n, device):
        return torch.as_tensor(next(it), dtype=torch.long)

    def sim(A, B):
        return score(A.numpy(), B.numpy())

    scal, p, tau = mod.calibrate(torch.from_numpy(X), torch.from_numpy(Y), sim, K=K, alpha=0.05,
                                 perm_fn=perm_fn)
    assert float(tau) == pytest.approx(ours["c_alpha"], abs=1e-6)  # ARH is float32
    assert float(p) == pytest.approx(ours["p_value"], abs=1e-6)
    assert float(scal) == pytest.approx(ours["s_cal"], abs=1e-6)


def test_ladder_score_part_mapping():
    from hrhv2_ref.ladder import part_of
    assert part_of("s_N@10") == "knn" and part_of("cka_debiased_eps8") == "cka"
    assert part_of("sog_mst@0.01") == "sog" and part_of("Gs_rms") == "geom" and part_of("G_max") == "geom"
    assert part_of("quot_iso_expl_xfit") == "quot" and part_of("s_T_bridge") == "mst"
    rng = np.random.default_rng(32)
    X = rng.normal(size=(30, 4))
    f = ladder_score("cka_debiased", pre="l2")
    assert f(X, X) == pytest.approx(1.0)
