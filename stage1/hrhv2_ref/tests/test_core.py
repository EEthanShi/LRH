import numpy as np
import pytest

from hrhv2_ref.core import (gram_sym, hash_halves, l2_rows, neighbor_order, preprocess,
                            prh_clip_threshold, sqdist_from_gram)


def test_l2_zero_row_is_error():
    with pytest.raises(ValueError):
        l2_rows(np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_prh_clip_threshold_hand():
    # row 1: |x| sorted [1,2,3,4,100], pos = .95*4 = 3.8 -> 4 + .8*96 = 80.8
    # row 2: [0,0,0,0,10] -> 0 + .8*10 = 8 ; mean = 44.4
    X = np.array([[1.0, -2.0, 3.0, 4.0, -100.0], [0.0, 0.0, 0.0, 0.0, 10.0]])
    assert prh_clip_threshold(X) == pytest.approx(44.4, abs=1e-12)
    Xp, info = preprocess(X, "prh_clip+l2")
    assert info["clip_c"] == pytest.approx(44.4, abs=1e-12)
    raw1 = np.array([1.0, -2.0, 3.0, 4.0, -44.4])
    assert np.allclose(Xp[0], raw1 / np.linalg.norm(raw1), atol=1e-15)
    assert np.allclose(Xp[1], [0, 0, 0, 0, 1.0], atol=1e-15)


def test_prh_clip_matches_torch_quantile():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X = rng.standard_t(3, size=(50, 37))
    c_t = torch.quantile(torch.from_numpy(np.abs(X)), 0.95, dim=1).mean().item()
    assert prh_clip_threshold(X) == pytest.approx(c_t, rel=1e-14)


def test_sqdist_line_and_symmetry():
    X = np.array([[0.0], [1.0], [3.0]])
    D = sqdist_from_gram(gram_sym(X))
    assert np.array_equal(D, np.array([[0, 1, 9], [1, 0, 4], [9, 4, 0]], dtype=float))
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(40, 7))
    DY = sqdist_from_gram(gram_sym(Y))
    assert np.array_equal(DY, DY.T)
    assert (DY >= 0).all() and (np.diag(DY) == 0).all()


def test_neighbor_order_tie_rule():
    # anchor 0 at x=0; candidates 1 (x=1), 2 (x=-1), 3 (x=2): tie between 1 and 2 -> 1 first
    X = np.array([[0.0], [1.0], [-1.0], [2.0]])
    D = sqdist_from_gram(gram_sym(X))
    o = neighbor_order(D)
    assert o[0].tolist() == [1, 2, 3]
    # anchor 1 (x=1): 0 (d=1), 3 (d=1), 2 (d=4) -> [0, 3, 2]
    assert o[1].tolist() == [0, 3, 2]
    # a row of exact ties is ordered by index
    Dt = np.ones((5, 5)) - np.eye(5)
    assert neighbor_order(Dt)[3].tolist() == [0, 1, 2, 4]


def test_hash_halves_deterministic_and_balanced():
    a = hash_halves(101)
    b = hash_halves(101)
    assert np.array_equal(a, b) and a.sum() == 50
    c = hash_halves(101, salt="other")
    assert not np.array_equal(a, c)
