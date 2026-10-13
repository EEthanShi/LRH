# Exploratory design scenarios for the Pythia axis (NOT citable; no Pythia data used).
import itertools, numpy as np
from scipy.stats import rankdata
prv = np.array([70.4,122.6,196.5,253.1,18.9,32.9,39.5,126.1,192.4,239.1,194.2,128.4,142.8,160.9,193.9,211.2,242.5])
N = np.array([70.4e6,162.3e6,405.3e6,1011.8e6,1414.6e6,2775.2e6,6857.3e6]); x = np.log2(N/N[0])
P = np.array(list(itertools.permutations(range(7))))  # 5040
def rho_rows(Y, ranks):  # Y (V,7) scores; ranks (7,) size ranks -> per-ViT Spearman
    R = np.apply_along_axis(rankdata, 1, Y)
    a = ranks - ranks.mean(); out=[]
    for r in R:
        b = r - r.mean(); d = np.sqrt((a*a).sum()*(b*b).sum()); out.append((a*b).sum()/d if d>0 else np.nan)
    return np.array(out)
def tr_all(Y):  # Tr under every relabelling: ranks of sizes permuted
    R = np.apply_along_axis(rankdata, 1, Y); Rc = R - R.mean(1, keepdims=True); nr = np.sqrt((Rc**2).sum(1))
    S = P + 1.0; Sc = S - 4.0; ns = np.sqrt((Sc**2).sum(1))
    return ((Sc @ Rc.T) / (ns[:,None]*nr[None,:])).mean(1)  # (5040,)
rng = np.random.default_rng(20261010)
print("PR70 PR69 a  | Tr_raw  Tr_cal  dTr  p_one(5040)  flips_c2d")
for pr0, pr1 in [(80,220),(120,260),(100,300)]:
    PR = pr0*(pr1/pr0)**(x/x[-1])
    B = np.sqrt(np.outer(prv, PR))/1023            # (17,7)
    for a in [0.0, 0.004, 0.008, 0.012, 0.016, 0.024]:
        res=[]
        for rep in range(20):
            c = B + 0.0019 + rng.normal(0, 0.0002, B.shape)
            T = c[:, :1] + 0.18 + a*x[None,:] + rng.normal(0, 0.004, B.shape)
            s = np.maximum((T-c)/(1-c), 0)
            tT, tS = tr_all(T), tr_all(s); d = tS - tT
            p = np.mean(d <= d[0] + 1e-9)  # row 0 = identity
            c2d = sum(((T[:,j2]>T[:,j1]) & (s[:,j2]<s[:,j1])).sum() for j1 in range(7) for j2 in range(j1+1,7))
            res.append((tT[0], tS[0], d[0], p, c2d))
        r = np.array(res); m = np.median(r, 0)
        print(f"{pr0:4d} {pr1:4d} {a:.3f} | {m[0]:+.2f}  {m[1]:+.2f}  {m[2]:+.2f}  {m[3]:.5f} [{r[:,3].min():.5f},{r[:,3].max():.5f}]  {m[4]:.0f}/357")
