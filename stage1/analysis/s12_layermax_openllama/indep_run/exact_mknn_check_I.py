#!/usr/bin/env python
"""OpenLLaMA axis, role oll-run-indep: integer-only re-derivation of every mKNN order, trend and order-change count
of implementation I (plan F2: mKNN quantities are exact; ties only on exact equality). NEW helper, written by
this role after the main run and before the I aggregate output was opened; it is not in the S0b manifest and
changes no output of the frozen code. It reads only I's own pair files (indep/<pre>/pairs/) and, at the end,
I's own verdicts_ol.json, and writes results/S1_2_layermax_openllama/indep/exact_mknn_check_I.json.

Arithmetic: no float and no Fraction touches an order decision. With MK = m k = 10,240 and K = 500:
  raw T              -> integer count T
  c_alpha            -> integer count c (476th ascending order statistic of {T} u {T_k})
  s_cal              -> rational (max(T - c, 0), MK - c), compared by integer cross-multiplication
                        (s_cal = 0 if T <= c or MK - c <= 0, the I/E convention)
  uncensored         -> integer K T - sum_k T_k (common denominator K MK)
  ctm M              -> max over entries of the entry s_cal (rational, row-major first maximum),
      M_unc          -> max over entries of K S_uv - sum_k S_uv^(k),  TM -> S at the M-argmax
Ranks of the 3 sizes per ViT are average ranks from exact comparisons; rho_v is the Pearson formula on ranks
(float only after the ranks are fixed). A ViT with 3 exactly equal scores is excluded. Order-change counts use
d = sign(score_b - score_a) from the exact comparisons.
Compared with I's verdicts_ol.json: tr (|d| <= 1e-12), dtr (|d| <= 1e-12), tr_nconst, chg|...|{new,pilot}|
{down,up,c2d,d2c} (exact). Also reported: the number of exact ties per (pre, Q, X) among the 51 comparisons,
which is where float rounding could break a tie.

usage: analysis/s12_layermax_openllama/launch_ol.sh ol_I_exact_mknn 2 -- venv/bin/python -I \
         analysis/s12_layermax_openllama/indep_run/exact_mknn_check_I.py
"""
import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
sys.path.insert(0, AXIS)
import guard  # noqa: E402

OUT = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "indep")
guard.require_partb_frozen(OUT, script="indep_run/exact_mknn_check_I.py")   # F1(a): first action

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

M_ITEMS, KNN, K = 1024, 10, 500
MK = M_ITEMS * KNN
ORDER = 475          # 0-based index of the 476th ascending order statistic of 501 values
PRES = ("prh_clip+l2", "l2")
OL = ["openlm-research__open_llama_3b", "openlm-research__open_llama_7b", "openlm-research__open_llama_13b"]
VITS = ["vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k",
        "vit_base_patch16_224.augreg_in21k", "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae",
        "vit_large_patch16_224.mae", "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m",
        "vit_base_patch16_clip_224.laion2b", "vit_large_patch14_clip_224.laion2b", "vit_huge_patch14_clip_224.laion2b",
        "vit_base_patch16_clip_224.laion2b_ft_in12k", "vit_large_patch14_clip_224.laion2b_ft_in12k",
        "vit_huge_patch14_clip_224.laion2b_ft_in12k"]
PAIRS3 = ((0, 1), (0, 2), (1, 2))
NEW = ((0, 2), (1, 2))
PILOT = ((0, 1),)


def fl(n, r):
    return int(math.floor(r * (n - 1) + 0.5))


def grid(n, k):
    return sorted({fl(n, j / (k - 1)) for j in range(k)})


def subsets(LA, LB):
    A, B = list(range(LA)), list(range(LB))
    s = {"full": (A, B)}
    for n in (2, 3, 5, 9, 17):
        s[f"grid{n}"] = (grid(LA, n), grid(LB, n))
    s["fixed_r0.5"] = ([fl(LA, 0.5)], [fl(LB, 0.5)])
    s["fixed_r1.0"] = ([LA - 1], [LB - 1])
    s["LMall_ViTlast"] = (A, [LB - 1])
    s["LMlast_ViTall"] = ([LA - 1], B)
    s["noLMemb"] = (list(range(1, LA)), B)
    s["LMdeep"] = ([a for a in range(LA) if a / (LA - 1) >= 0.5], B)
    s["LMemb_only"] = ([0], B)
    return s


# exact score objects: ("int", n) compares by n; ("rat", num, den) by cross-multiplication (den > 0)
def cmp(x, y):
    if x[0] == "int":
        return (x[1] > y[1]) - (x[1] < y[1])
    l, r = x[1] * y[2], y[1] * x[2]
    return (l > r) - (l < r)


def scal(T, c):
    if T > c and MK - c > 0:
        return ("rat", T - c, MK - c)
    return ("rat", 0, 1)


def summ(S, N):
    """S (la, lb) int64 observed counts; N (K, la, lb) int64 null counts -> exact raw, s_cal, unc."""
    T = int(S.max())
    Tk = N.reshape(N.shape[0], -1).max(axis=1)
    c = int(np.sort(np.append(Tk, T))[ORDER])
    return {"raw": ("int", T), "s_cal": scal(T, c), "unc": ("int", K * T - int(Tk.sum())), "c": c}


def ctm(S, N):
    Ce = np.sort(np.concatenate([S[None], N], axis=0), axis=0)[ORDER]
    best, ba, bb = ("rat", 0, 1), 0, 0
    LA, LB = S.shape
    for u in range(LA):
        for w in range(LB):
            q = scal(int(S[u, w]), int(Ce[u, w]))
            if cmp(q, best) > 0:
                best, ba, bb = q, u, w
    U = K * S - N.sum(axis=0)
    return {"M": best, "M_unc": ("int", int(U.max())), "TM": ("int", int(S[ba, bb]))}


def avg_ranks(vals):
    out = []
    for x in vals:
        lt = sum(1 for y in vals if cmp(y, x) < 0)
        eq = sum(1 for y in vals if cmp(y, x) == 0)
        out.append(1 + lt + 0.5 * (eq - 1))
    return out


PERM6 = [tuple(float(v) for v in p) for p in itertools.permutations((1, 2, 3))]
ID6 = PERM6.index((1.0, 2.0, 3.0))


def trend(cols):
    valid = [j for j, c in enumerate(cols) if not (cmp(c[0], c[1]) == 0 and cmp(c[1], c[2]) == 0)]
    if not valid:
        return None, None, len(cols)
    R = np.array([avg_ranks(cols[j]) for j in valid], dtype=np.float64)
    Rc = R - R.mean(axis=1, keepdims=True)
    X = np.array(PERM6)
    Xc = X - X.mean(axis=1, keepdims=True)
    rho = (Xc @ Rc.T) / (np.sqrt((Xc ** 2).sum(axis=1))[:, None] * np.sqrt((Rc ** 2).sum(axis=1))[None, :])
    T = rho.mean(axis=1)
    return T, float(T[ID6]), len(cols) - len(valid)


def main():
    t0 = time.time()
    recs = {}
    for pre in PRES:
        recs[pre] = {}
        for lm in OL:
            for v in VITS:
                z = np.load(os.path.join(OUT, pre, "pairs", f"{lm}__{v}.npz"))
                assert int(z["K"]) == K and str(z["key"]) == f"S1.2-layermax|{lm}|{v}"
                S = z["obs_cnt"].astype(np.int64)
                N = z["null_cnt"].astype(np.int64)
                LA, LB = S.shape
                r = {}
                for q, (ra, rb) in subsets(LA, LB).items():
                    r[q] = summ(S[np.ix_(ra, rb)], N[:, ra][:, :, rb])
                r["ctm"] = ctm(S, N)
                recs[pre][(lm, v)] = r
        print(f"[{time.time() - t0:.0f}s] {pre}: 51 pair records (integer)", flush=True)
    mine, ties = {}, {}
    qs = list(subsets(41, 12).keys())
    for pre in PRES:
        R = recs[pre]
        TR = {}
        for q in qs + ["ctm"]:
            xs = ("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")
            for x in xs:
                cols = [[R[(lm, v)][q][x] for lm in OL] for v in VITS]
                T, obs, nconst = trend(cols)
                TR[(q, x)] = T
                k = f"{pre}|mknn|{q}|{x}"
                mine[f"tr|{k}"] = obs
                mine[f"tr_nconst|{k}"] = nconst
                ties[k] = sum(1 for c in cols for a, b in PAIRS3 if cmp(c[a], c[b]) == 0)
            base = TR[("full", "raw")] if q == "ctm" else TR[(q, "raw")]
            for x in (("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")):
                T = TR[(q, x)]
                mine[f"dtr|{pre}|mknn|{q}|{x}"] = None if T is None or base is None else float((T - base)[ID6])
            for x in (("M", "M_unc") if q == "ctm" else ("s_cal", "unc")):
                rq = "full" if q == "ctm" else q
                for grp, prs in (("new", NEW), ("pilot", PILOT)):
                    c = {"down": 0, "up": 0, "c2d": 0, "d2c": 0}
                    for v in VITS:
                        for a, b in prs:
                            dr = cmp(R[(OL[b], v)][rq]["raw"], R[(OL[a], v)][rq]["raw"])
                            dx = cmp(R[(OL[b], v)][q][x], R[(OL[a], v)][q][x])
                            c["down"] += dx < dr
                            c["up"] += dx > dr
                            c["c2d"] += (dr == 1 and dx == -1)
                            c["d2c"] += (dr == -1 and dx == 1)
                    for kk, vv in c.items():
                        mine[f"chg|{pre}|mknn|{q}|{x}|{grp}|{kk}"] = vv
    # ---- compare with I's verdicts_ol.json
    VI = json.load(open(os.path.join(OUT, "verdicts_ol.json")))["v"]
    n = bad = 0
    fails = []
    for k, a in mine.items():
        n += 1
        b = VI.get(k, "MISSING")
        if isinstance(a, float) or isinstance(b, float):
            ok = (a is None and b is None) or (a is not None and b not in (None, "MISSING") and abs(float(a) - float(b)) <= 1e-12)
        else:
            ok = a == b
        if not ok:
            bad += 1
            fails.append([k, a, b])
    res = {"what": "integer-only re-derivation of I's mKNN trends and order changes (plan F2)",
           "n_compared": n, "n_mismatch": bad, "mismatches": fails[:50], "pass": bad == 0,
           "exact_ties_per_pre_Q_X_of_51": ties,
           "n_keys_with_ties": sum(1 for v in ties.values() if v > 0),
           "values": mine, "secs": time.time() - t0,
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    fn = os.path.join(OUT, "exact_mknn_check_I.json")
    with open(fn + ".tmp", "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(fn + ".tmp", fn)
    print(json.dumps({k: res[k] for k in ("n_compared", "n_mismatch", "pass", "n_keys_with_ties")}), flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
