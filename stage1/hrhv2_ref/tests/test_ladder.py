import numpy as np
import pytest

from hrhv2_ref import ladder_ref
from hrhv2_ref.core import gram_sym, preprocess

REQUIRED = (["s_N@1", "s_N@3", "s_N@10", "comp_rand@1", "comp_n_A@3", "comp_n_B@10", "delta_O",
             "kendall_global", "spearman_global", "s_T", "s_T_forest", "s_T_bridge", "forest_share_A",
             "Gs_rms", "Gs_max", "G_rms", "G_max", "cka_biased", "cka_debiased", "rsa_pearson",
             "proj_dist"]
            + [f"sog_{v}@{t}" for v in ("knn", "mst") for t in ("1", "0.1", "0.01", "0.001", "0.0001")]
            + [f"quot_{q}_{s}" for q in ("scale", "affine", "iso", "rowiso") for s in ("expl", "rms", "max")])


def cloud(seed, m, p=3, d=16):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(m, p))
    Q, _ = np.linalg.qr(rng.normal(size=(d, p)))
    return Z @ Q.T


def test_keys_and_identity():
    X = cloud(50, 120)
    out = ladder_ref(X, X, "l2")
    for k in REQUIRED:
        assert k in out, k
    for k in (1, 3, 10, 30, 100):
        assert out[f"s_N@{k}"] == 1.0
    assert out["comp_rand@1"] == 0.0 and out["delta_O"] == 0.0
    assert out["kendall_global"] == pytest.approx(1.0) and out["s_T"] == 1.0
    assert out["cka_biased"] == pytest.approx(1.0) and out["cka_debiased"] == pytest.approx(1.0)
    assert out["F_A_subset_T_A"] and out["prim_eq_kruskal_A"]
    assert out["sog_knn@0.0001"] == pytest.approx(1.0)
    assert out["k_q"] == 1 and out["proj_dist"] == pytest.approx(0.0, abs=1e-10)


def spec_names(kq):
    """Every SPEC v1.0 §1-§4 metric name ladder_ref must return (default grids)."""
    ks = sorted({1, 3, 10, 30, 100, kq})
    names = [f"s_N@{k}" for k in ks] + [f"delta_N@{k}" for k in ks] + [f"chance@{k}" for k in ks]
    names += [f"near_tie_{s}@{k}" for s in "AB" for k in ks]
    names += [f"comp_{x}@{k}" for x in ("rand", "n_A", "n_B", "degenerate") for k in (1, 3, 10)]
    names += ["delta_O", "kendall_global", "spearman_global", "delta_E", "s_T", "s_T_forest",
              "s_T_bridge", "forest_share_A", "F_A_subset_T_A", "Gs_rms", "Gs_max", "G_rms", "G_max",
              "cka_biased", "cka_debiased", "rsa_pearson", "proj_dist", "cca_mean_sq", "k_q"]
    names += [f"sog_{v}@{t}" for v in ("knn", "mst") for t in ("1", "0.1", "0.01", "0.001", "0.0001")]
    names += [f"quot_{q}_{s}{x}{d}" for q in ("scale", "affine", "iso", "rowiso")
              for s in ("expl", "rms", "max") for x in ("", "_xfit") for d in ("", "_BA", "_mean")]
    return names


def test_all_spec_names_present_and_finite():
    rng = np.random.default_rng(61)
    m = 330  # k_q = round(3.29) = 3; all grid k < m
    X = cloud(61, m, p=4, d=24)
    Y = np.tanh(X @ rng.normal(size=(24, 20))) + 0.2 * rng.normal(size=(m, 20))
    out = ladder_ref(X, Y, "l2")
    assert out["k_q"] == 3
    missing = [n for n in spec_names(out["k_q"]) if n not in out]
    assert not missing, missing
    for n in spec_names(out["k_q"]):
        v = float(out[n])
        assert np.isfinite(v), n
        if n.startswith(("s_N@", "delta_N@", "comp_rand", "s_T", "forest_share", "sog_", "proj_dist",
                         "delta_O", "delta_E", "near_tie")):
            assert 0.0 <= v <= 1.0, (n, v)
    for k in (1, 3, 10, 30, 100):
        assert out[f"delta_N@{k}"] == 1.0 - out[f"s_N@{k}"]
    # s_N@1 is reported by the MST block even when 1 is not in an explicit kgrid
    o = ladder_ref(X, Y, "l2", kgrid=[10], parts={"knn", "mst"})
    assert o["s_N@1"] == out["s_N@1"] and "s_N@3" not in o
    # same values whether computed jointly or part by part
    for part, key in (("sog", "sog_knn@0.01"), ("quot", "quot_rowiso_expl_xfit"), ("proj", "proj_dist")):
        assert ladder_ref(X, Y, "l2", parts={part})[key] == out[key]


def test_pbern_plant_ordinal_rungs_identical():
    # SPEC §7 P-Bern on a small cloud: K_B = exp(beta (K_A - 1))
    X = cloud(51, 150, p=4, d=64)
    Xl, _ = preprocess(X, "l2")
    KA = gram_sym(Xl)
    for beta in (1.0, 3.0, 10.0):
        KB = np.exp(beta * (KA - 1.0))
        out = ladder_ref(KA, KB, gram=True, kgrid=[1, 3, 10, 30, 100])
        for k in (1, 3, 10, 30, 100):
            assert out[f"s_N@{k}"] == 1.0
        for k in (1, 3, 10):
            assert out[f"comp_rand@{k}"] == 0.0
        assert out["delta_O"] == 0.0 and out["delta_E"] == 0.0 and out["s_T"] == 1.0
        assert out["quot_iso_expl"] >= 0.999
        assert out["quot_rowiso_expl"] >= 0.999


def test_rotation_invariance_and_noise_monotone():
    rng = np.random.default_rng(52)
    X = cloud(52, 200, p=5, d=20)
    Q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    o = ladder_ref(X, X @ Q, "l2", parts={"knn", "cka", "geom"})
    assert o["s_N@10"] == 1.0 and o["cka_debiased"] == pytest.approx(1.0, abs=1e-12)
    assert o["G_max"] < 1e-12
    prev = 1.1
    for s in (0.05, 0.2, 0.8):
        Y = X + s * rng.normal(size=X.shape)
        o = ladder_ref(X, Y, "l2", parts={"knn"})
        assert o["s_N@10"] < prev
        prev = o["s_N@10"]


def test_delta_O_brute_equals_merge():
    X = cloud(53, 70)
    Y = cloud(54, 70)
    a = ladder_ref(X, Y, "l2", parts={"rank"}, kendall_brute_max_m=1000)
    b = ladder_ref(X, Y, "l2", parts={"rank"}, kendall_brute_max_m=10)
    assert a["delta_O_method"] == "brute" and b["delta_O_method"] == "mergesort"
    assert a["delta_O"] == pytest.approx(b["delta_O"], abs=1e-15)


def test_k_q_and_large_k_nan():
    X = cloud(55, 90)
    Y = cloud(56, 90)
    out = ladder_ref(X, Y, "l2", parts={"knn"})
    assert out["k_q"] == 1  # round(89/100) = 1
    assert np.isnan(out["s_N@100"])
    X = cloud(57, 256)
    out = ladder_ref(X, X + 0.1, "l2", parts={"knn"})
    assert out["k_q"] == 3  # round(2.55) = 3


def test_rank_detail_monotone_with_saturation_ties():
    # B = monotone transform of A that merges some values -> only B-only ties, no strict discordance
    rng = np.random.default_rng(58)
    X = cloud(58, 60)
    Xl, _ = preprocess(X, "l2")
    KA = gram_sym(Xl)
    KB = np.round(KA, 1)  # coarse monotone (non-decreasing) map of K -> creates ties in B only
    np.fill_diagonal(KB, 1.0)
    out = ladder_ref(KA, KB, gram=True, parts={"rank", "edge"})
    assert out["E_n_disc_strict"] == 0 and out["E_ties_B_only"] > 0
    assert out["delta_O_strict_disc"] == 0.0 and out["delta_O"] > 0
    assert out["delta_E"] > 0


def test_dist_mode_matches_feature_mode():
    X = cloud(59, 80)
    Y = X + 0.3 * np.random.default_rng(59).normal(size=X.shape)
    Xl, _ = preprocess(X, "l2")
    Yl, _ = preprocess(Y, "l2")
    from hrhv2_ref.core import sqdist_from_gram
    DA, DB = sqdist_from_gram(gram_sym(Xl)), sqdist_from_gram(gram_sym(Yl))
    f = ladder_ref(X, Y, "l2", parts={"knn", "cka", "mst", "rank"})
    d = ladder_ref(DA, DB, dist=True, parts={"knn", "cka", "mst", "rank"})
    for k in ("s_N@1", "s_N@10", "s_T", "delta_O"):
        assert f[k] == d[k]
    for k in ("cka_biased", "cka_debiased"):
        assert f[k] == pytest.approx(d[k], abs=1e-12)


def test_pbern_distance_level_has_no_strict_discordance():
    # D-level plant: D_A = 2 - 2K, D_B = -2 expm1(beta (K - 1)) -> monotone; only B-only ties possible
    X = cloud(60, 300, p=2, d=64)
    Xl, _ = preprocess(X, "l2")
    KA = gram_sym(Xl)
    DA = 2.0 - 2.0 * KA
    DB = -2.0 * np.expm1(10.0 * (KA - 1.0))
    out = ladder_ref(DA, DB, dist=True, parts={"rank", "edge", "knn", "mst"})
    assert out["E_n_disc_strict"] == 0 and out["delta_O_strict_disc"] == 0.0
    assert out["s_N@10"] == 1.0 and out["s_T"] == 1.0


def test_projector_equals_one_minus_mean_sq_cca_and_pe2_keys():
    # ||P_A - P_B||_F^2 = r_A + r_B - 2 sum rho^2  ->  with r_A = r_B = r: proj_dist = 1 - mean rho^2
    rng = np.random.default_rng(62)
    X = rng.normal(size=(150, 6))
    Y = X @ rng.normal(size=(6, 6)) + 0.7 * rng.normal(size=(150, 6))
    o = ladder_ref(X, Y, "none", parts={"proj", "mst"})
    assert o["proj_rank_A"] == 6 and o["proj_rank_B"] == 6
    assert o["proj_dist"] == pytest.approx(1.0 - o["cca_mean_sq"], abs=1e-12)
    for k in ("forest_share_shared_A", "forest_share_shared_B", "forest_share_shared_union"):
        assert 0.0 <= o[k] <= 1.0
    assert o["forest_share_shared_union"] >= max(o["forest_share_shared_A"], o["forest_share_shared_B"])
