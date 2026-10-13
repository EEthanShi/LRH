import numpy as np
import pytest

from hrhv2.prep import Geom, preprocess, prh_clip_threshold, gram_to_dist, center_gram
from conftest import rand_features


def test_l2_rows_unit():
    X = rand_features(50, 7, "prep|l2")
    Xp = preprocess(X, "l2")
    assert np.allclose(np.linalg.norm(Xp, axis=1), 1.0, atol=1e-15)
    assert Xp.dtype == np.float64


def test_zero_row_is_error():
    X = np.ones((4, 3))
    X[2] = 0
    with pytest.raises(ValueError):
        preprocess(X, "l2")


def test_unknown_pre_is_error():
    with pytest.raises(ValueError):
        preprocess(np.ones((3, 2)), "zscore")


def test_prh_clip_hand_example():
    # 2 samples, 5 coords; per-sample 0.95-quantile (linear) of |x|
    X = np.array([[0., 1., 2., 3., 10.], [-4., 0., 0., 0., 0.]])
    # sample 0: sorted |x| = 0,1,2,3,10 -> pos 0.95*4 = 3.8 -> 3 + 0.8*7 = 8.6
    # sample 1: sorted = 0,0,0,0,4 -> pos 3.8 -> 0 + 0.8*4 = 3.2
    c = prh_clip_threshold(X, 0.95)
    assert c == pytest.approx((8.6 + 3.2) / 2, rel=1e-14)
    Xp = preprocess(X, "prh_clip+l2")
    Xc = np.clip(X, -5.9, 5.9)
    assert np.allclose(Xp, Xc / np.linalg.norm(Xc, axis=1, keepdims=True), atol=1e-15)


def test_prh_clip_matches_vendored_arh_and_prh(arh, prh, torch):
    X = rand_features(64, 40, "prep|arh") ** 3  # heavy tails
    for ns in (arh, prh):
        ref = ns["remove_outliers"](torch.tensor(X, dtype=torch.float64), q=0.95).numpy()
        ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        ours = preprocess(X, "prh_clip+l2")
        assert np.max(np.abs(ours - ref)) < 1e-12
    # stacked (n, L, d): ARH/PRH pool all layers of a sample for the threshold
    S = rand_features(64, 3 * 40, "prep|arh3").reshape(64, 3, 40) ** 3
    c_ref = float(torch.quantile(torch.tensor(S).abs().flatten(start_dim=1), 0.95, dim=1).mean())
    assert prh_clip_threshold(S, 0.95) == pytest.approx(c_ref, rel=1e-12)
    # preprocess a single layer with the all-layer threshold (clip_ref)
    ref3 = arh["remove_outliers"](torch.tensor(S), q=0.95).numpy()[:, 1, :]
    ref3 = ref3 / np.linalg.norm(ref3, axis=1, keepdims=True)
    assert np.max(np.abs(preprocess(S[:, 1, :], "prh_clip+l2", clip_ref=S) - ref3)) < 1e-12


def test_prep_float32_input_is_upcast():
    X = rand_features(10, 4, "prep|f32").astype(np.float32)
    assert preprocess(X, "l2").dtype == np.float64


def test_gram_to_dist_hand():
    K = np.array([[2., 1., 0.], [1., 1., -1.], [0., -1., 5.]])
    D = gram_to_dist(K)
    expect = np.array([[0, 2 + 1 - 2, 2 + 5 - 0], [1, 0, 1 + 5 + 2], [7, 8, 0]], dtype=float)
    assert np.array_equal(D, expect)


def test_gram_to_dist_clips_and_zero_diag():
    K = np.array([[1., 1.0000000001], [1.0000000001, 1.]])
    D = gram_to_dist(K)
    assert D[0, 1] == 0.0 and D[1, 0] == 0.0 and D[0, 0] == 0.0


def test_center_gram_double_centering():
    K = rand_features(9, 9, "prep|cg")
    K = K @ K.T
    H = np.eye(9) - 1.0 / 9
    assert np.allclose(center_gram(K), H @ K @ H, atol=1e-12)


def test_geom_from_features_dist_matches_direct():
    X = rand_features(30, 5, "prep|geom")
    G = Geom.from_features(X, "none")
    direct = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    assert np.allclose(G.D, direct, atol=1e-12)
    assert np.all(np.diag(G.D) == 0)


def test_geom_from_dist_requires_symmetry():
    D = np.array([[0, 1.], [2., 0]])
    with pytest.raises(ValueError):
        Geom.from_dist(D)


def test_dist_centered_gram_equals_gram_centered():
    X = rand_features(20, 4, "prep|mds")
    A = Geom.from_features(X, "none")
    B = Geom.from_dist(A.D)
    assert np.allclose(A.centered_gram(), B.centered_gram(), atol=1e-12)


def test_subset_slices_bitwise():
    X = rand_features(40, 6, "prep|sub")
    G = Geom.from_features(X, "l2")
    idx = np.arange(25)
    S = G.subset(idx)
    assert np.array_equal(S.D, G.D[:25, :25])
