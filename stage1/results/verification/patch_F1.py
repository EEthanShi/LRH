"""Verifier patch F1: debiased CKA evaluated on H K H (see hrhv2/FIXES.md)."""
import os

ROOT = os.path.expanduser("~/hrh_v2_stage1_20261008")

p = os.path.join(ROOT, "hrhv2/geometric.py")
s = open(p).read()
start = s.index("def cka_pair(A: Geom, B: Geom) -> dict:")
end = s.index("    deb = hab / np.sqrt(haa * hbb)")
old = s[start:end]
assert old.count("KA = A.hsic_gram()") == 1, "unexpected cka_pair body"
new = '''def cka_pair(A: Geom, B: Geom) -> dict:
    """Biased and debiased linear CKA (plus the ARH-EPS variant of debiased).

    Both HSICs are evaluated on the double-centred Grams H K H. The biased
    HSIC is <HKH, HLH>_F by definition. The Song et al. (2012) unbiased HSIC
    is invariant to K_ij -> K_ij + a_i + a_j, and H K H = K + a 1' + 1 a' for
    symmetric K, so evaluating it on H K H is the same estimator in exact
    arithmetic; it avoids the large row/column-mean terms of a raw Gram that
    otherwise cancel (verifier fix F1, hrhv2/FIXES.md: on raw unit-norm Grams
    the cancelling terms exceeded the result by up to 3e5 on WIT-1024
    features, costing up to 1.2e-9 relative accuracy; on H K H the error vs
    exact rational arithmetic is <= 3e-15)."""
    KcA = A.centered_gram()
    KcB = B.centered_gram()
    biased = cka_biased_from_centered(KcA, KcB)
    hab = hsic_unbiased(KcA, KcB)
    haa = hsic_unbiased(KcA, KcA)
    hbb = hsic_unbiased(KcB, KcB)
    del KcA, KcB
'''
s = s[:start] + new + s[end:]
open(p, "w").write(s)

p = os.path.join(ROOT, "hrhv2/tests/test_geometric.py")
s = open(p).read()
assert "test_cka_debiased_accurate_on_ill_conditioned_gram" not in s
s += '''

def _song_exact(K, L):
    """Song et al. unbiased HSIC in exact rational arithmetic (float64 inputs are dyadic)."""
    from fractions import Fraction
    m = K.shape[0]
    Kf = [[Fraction(float(K[i, j])) if i != j else Fraction(0) for j in range(m)] for i in range(m)]
    Lf = [[Fraction(float(L[i, j])) if i != j else Fraction(0) for j in range(m)] for i in range(m)]
    t1 = sum(Kf[i][j] * Lf[i][j] for i in range(m) for j in range(m))
    rK = [sum(r) for r in Kf]
    rL = [sum(r) for r in Lf]
    t3 = sum(a * b for a, b in zip(rK, rL))
    return (t1 + sum(rK) * sum(rL) / ((m - 1) * (m - 2)) - Fraction(2, m - 2) * t3) / (m * (m - 3))


def test_cka_debiased_accurate_on_ill_conditioned_gram():
    """Verifier fix F1: rows sharing a large common mean make the raw-Gram Song
    terms cancel heavily; evaluating on H K H keeps the unbiased HSIC and the
    debiased CKA accurate to <= 1e-12 relative vs exact rational arithmetic
    (the raw-Gram evaluation misses this by orders of magnitude)."""
    mu = 30.0 * rand_features(1, 10, "illc0")
    XA = rand_features(40, 10, "illc1") + mu
    XB = 0.4 * XA + rand_features(40, 10, "illc2") + 2 * mu
    A, B = Geom.from_features(XA, "none"), Geom.from_features(XB, "none")
    r = geo.cka_pair(A, B)
    hab = _song_exact(A.K, B.K)
    haa = _song_exact(A.K, A.K)
    hbb = _song_exact(B.K, B.K)
    assert abs(r["hsicu_AB"] - float(hab)) <= 1e-12 * abs(float(hab))
    deb = float(hab) / np.sqrt(float(haa) * float(hbb))
    assert abs(r["cka_debiased"] - deb) <= 1e-12 * abs(deb)
    # the old raw-Gram evaluation is far less accurate on this input
    raw = geo.hsic_unbiased(A.K, B.K)
    assert abs(raw - float(hab)) > 100 * abs(r["hsicu_AB"] - float(hab))
'''
open(p, "w").write(s)
print("patched")
