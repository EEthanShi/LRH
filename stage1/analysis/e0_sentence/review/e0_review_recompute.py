#!/usr/bin/env python3
"""E0 review (role e0-review): independent recomputation of the confirmatory decisions and of the
freeze order.

Own code. Imports numpy, pandas and scipy.stats.rankdata only; imports no e0_* module, no s11copy /
s11v11* module and no hrhv2 module. Every definition below is written from the TEXT of
ANALYSIS_PLAN_E0.md v1.0 (sha256 e859c6ea52f941b18ca27b83150f489dded0085a4d99218d909b36c19603f0ee),
sections 6, 7 and 13, not from the E0 or S1.1 code.

Read only:
  results/E0_sentence/obs/obs_<arm>.parquet          (E0 obs, new pairs and old pairs under ctr)
  results/S1_1/v11/obs/obs_clip.parquet, obs_l2.parquet (frozen S1.1 obs, old pairs)
  results/E0_sentence/cache/clip/<enc>.<inp>/d<d>_knn.npz and results/S1_1/v11/cache/clip/<vit>/d<d>_knn.npz
                                                    (obs-step check from the stored neighbour lists)
  extract/e0_sentence/coco_captions/captions_val2014.json, features/coco_pool/items.parquet (caption fact)
  results/E0_sentence/VERDICTS_E0.json               (only to compare against, after recomputation)
  STATUS.md and file birth times                     (freeze-order audit)
Writes one new JSON file (refuses to overwrite).

    OMP_NUM_THREADS=1 taskset -c 0-5 venv/bin/python -I analysis/e0_sentence/review/e0_review_recompute.py \
        --out analysis/e0_sentence/review/review_recompute_<utc>.json
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
E0R = os.path.join(PROJ, "results/E0_sentence")
OBSD = os.path.join(E0R, "obs")
S11 = os.path.join(PROJ, "results/S1_1/v11")
M_GRID = [256, 512, 1024, 2048, 4096, 8192, 16384]
VITS = ["augreg_b", "clip_b", "dinov2_b", "mae_b", "siglip_b"]
ENCS = ["contriever", "mpnet", "qwen3emb06"]
LMS = ["bloomz_1b7", "openllama_3b", "pythia410m_s1"]
TEXT_ORDER = LMS + ENCS  # plan 7.2 fixed text order

# ---- frozen constants, typed from the plan text (sections 7.3, 7.4) ----
LINE_A, LINE_C = 0.1357059076472474, 0.22805384255107658
S_RES = 0.05647474462525915
COMP = {"augreg_b": -0.5752592991292489, "clip_b": -0.5499745403938117, "dinov2_b": -0.5890678384804802,
        "mae_b": -0.6444729401596141, "siglip_b": -0.511091295171744}
BETA_M_CONST = -0.570555396995647


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ols(x, y):
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return beta  # intercept, slope


# ---------------------------------------------------------------- quantities (plan section 6)
NCOL = {"obs": "n_anchors", "obs_allH2": "n_allH2", "obs_clean": "n_clean"}


def draw_slopes(df, col="obs"):
    """b(k) = mean over draws of the OLS slope of ln max(s, eps_d) on ln m over all 7 m; eps_d = 0.5/(n k)."""
    n = df[NCOL[col]].astype(float)
    d = df.assign(y=np.log(np.maximum(df[col].astype(float), 0.5 / (n * df["k"]))), x=np.log(df["m"].astype(float)))
    rows = []
    for (p, k, dr), g in d.groupby(["pair", "k", "draw"], sort=True):
        assert len(g) == 7 and sorted(g.m) == M_GRID, (p, k, dr)
        rows.append((p, k, dr, ols(g.x.values, g.y.values)[1]))
    ds = pd.DataFrame(rows, columns=["pair", "k", "draw", "slope"])
    b = ds.groupby(["pair", "k"]).slope.agg(["mean", "std", "count"]).reset_index()
    assert (b["count"] == 5).all()
    return b.rename(columns={"mean": "b", "std": "b_draw_sd"}), ds


def drawmean_cells(df, col="obs"):
    """s-bar = mean over draws; floor eps-bar = 0.5/(5 n-bar k)."""
    c = df.groupby(["pair", "m", "k"]).agg(s=(col, "mean"), n=(NCOL[col], "mean"), nd=("draw", "nunique")).reset_index()
    assert (c.nd == 5).all()
    c["eps"] = 0.5 / (5.0 * c.n * c.k)
    c["lns"] = np.log(np.maximum(c.s, c.eps))
    return c


def elasticities(cells, K):
    out = {}
    for p, g in cells[cells.k.isin(K)].groupby("pair"):
        assert len(g) == 7 * len(K)
        X = np.column_stack([np.ones(len(g)), np.log(g.k.astype(float)), np.log(g.m.astype(float))])
        beta, *_ = np.linalg.lstsq(X, g.lns.values, rcond=None)
        out[p] = {"e_k": beta[1], "e_m": beta[2], "sum": beta[1] + beta[2]}
    return out


def level(cells, k=10, mref=1024):
    c = cells[(cells.k == k) & (cells.m == mref)].set_index("pair")
    return c.s.to_dict(), c.lns.to_dict()


def per_draw_elasticities(df, K, col="obs"):
    n = df[NCOL[col]].astype(float)
    d = df.assign(lns=np.log(np.maximum(df[col].astype(float), 0.5 / (n * df["k"]))))
    out = {}
    for dr, gd in d.groupby("draw"):
        cc = gd.rename(columns={})
        out[int(dr)] = elasticities(cc[["pair", "m", "k", "lns"]], K)
    return out


# ---------------------------------------------------------------- decision rules (plan section 7)
def rule_P1(el):
    sums = np.array([v["sum"] for v in el.values()])
    em = np.array([v["e_m"] for v in el.values()])
    inf = np.abs(em) >= 0.10
    n_inf = int(inf.sum())
    if n_inf < 8:
        return {"label": "uninformative", "n_informative": n_inf}
    s = sums[inf]
    med = float(np.median(s))
    frac = float(np.mean(np.abs(s) <= 0.05))
    if abs(med) <= 0.03 and frac >= 0.80:
        lab = "holds"
    elif abs(med) > 0.06 or frac <= 0.50:
        lab = "fails"
    else:
        lab = "partly holds"
    return {"label": lab, "n_informative": n_inf, "median_sum": med, "frac_abs_sum_le_0.05": frac,
            "max_abs_sum": float(np.max(np.abs(s))), "e_m_range": [float(em.min()), float(em.max())]}


def bhat_star(lnL, a=LINE_A, c=LINE_C):
    return np.minimum(a + c * lnL, 0.0)


def rule_P2b(pairs, lnL, b10, a=LINE_A, c=LINE_C, s_res=S_RES, comp=COMP):
    g = np.array([bhat_star(lnL[p], a, c) - comp[p.split("__")[0]] for p in pairs])
    y = np.array([b10[p] - comp[p.split("__")[0]] for p in pairs])
    enc = np.array([p.split("__")[1] for p in pairs])
    guard = 2 * s_res
    med_abs_g = float(np.median(np.abs(g)))
    enc_med_abs_g = {e: float(np.median(np.abs(g[enc == e]))) for e in sorted(set(enc))}
    inf_enc = [e for e, v in enc_med_abs_g.items() if v >= guard]
    lam = float(np.sum(g * y) / np.sum(g * g))
    lam_e_all = {e: float(np.sum(g[enc == e] * y[enc == e]) / np.sum(g[enc == e] ** 2)) for e in sorted(set(enc))}
    res = {"median_abs_g": med_abs_g, "guard": guard, "enc_median_abs_g": enc_med_abs_g, "informative_encoders": inf_enc,
           "lambda_descriptive": lam, "lambda_e_descriptive": lam_e_all,
           "g": dict(zip(pairs, g.tolist())), "y": dict(zip(pairs, y.tolist()))}
    if med_abs_g < guard:
        res["label"] = "uninformative"
        return res
    if len(inf_enc) < 2:
        res["label"] = "inconclusive"
        return res
    le = {e: lam_e_all[e] for e in inf_enc}
    if 0.75 <= lam <= 1.25 and all(0.5 <= v <= 1.5 for v in le.values()):
        res["label"] = "holds"
    elif lam <= 0.25 and all(v <= 0.5 for v in le.values()):
        res["label"] = "fails"
    else:
        res["label"] = "inconclusive"
    return res


def rule_P2c(pairs, ladders):
    """ladders[p] = list of (lnL_n, b_n) for n = 1..5."""
    gam, dln = {}, {}
    for p in pairs:
        lnL = np.array([t[0] for t in ladders[p]])
        b = np.array([t[1] for t in ladders[p]])
        gam[p] = float(ols(lnL, b)[1])
        dln[p] = float(lnL[4] - lnL[0])
    enc = {p: p.split("__")[1] for p in pairs}
    med_g = float(np.median(list(gam.values())))
    med_d = float(np.median(list(dln.values())))
    enc_d = {e: float(np.median([dln[p] for p in pairs if enc[p] == e])) for e in ENCS}
    enc_g = {e: float(np.median([gam[p] for p in pairs if enc[p] == e])) for e in ENCS}
    inf_enc = [e for e in ENCS if enc_d[e] >= 0.3]
    res = {"median_gamma": med_g, "median_dln": med_d, "enc_median_dln": enc_d, "enc_median_gamma": enc_g,
           "informative_encoders": inf_enc, "gamma": gam, "dln": dln}
    if med_d < 0.3:
        res["label"] = "uninformative"
    elif len(inf_enc) < 2:
        res["label"] = "inconclusive"
    elif 0.17 <= med_g <= 0.29 and all(0.154 <= enc_g[e] <= 0.342 for e in inf_enc):
        res["label"] = "holds"
    elif med_g <= 0.12 and all(enc_g[e] <= 0.154 for e in inf_enc):
        res["label"] = "fails"
    else:
        res["label"] = "inconclusive"
    return res


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------- obs step from the stored neighbour lists
def obs_from_caches(vit, encdir, draw, m, k):
    A = np.load(os.path.join(S11, "cache/clip", vit, f"d{draw}_knn.npz"))
    B = np.load(os.path.join(E0R, "cache/clip", encdir, f"d{draw}_knn.npz"))
    pa, pb = A[f"pos_m{m}"], B[f"pos_m{m}"]
    assert np.array_equal(pa, pb)
    ia, ib = A[f"idx_m{m}"], B[f"idx_m{m}"]
    self_first = bool(np.all(ia[:, 0] == pa) and np.all(ib[:, 0] == pb))
    off = 1 if self_first else 0
    na, nb = ia[:, off:off + k], ib[:, off:off + k]
    ov = np.array([len(np.intersect1d(na[i], nb[i], assume_unique=True)) for i in range(len(pa))]) / k
    anc = pa < 1024
    return float(ov[anc].mean()), int(anc.sum()), float(ov.mean()), int(len(pa)), self_first


# ---------------------------------------------------------------- freeze-order audit (own code)
def status_lines(path):
    with open(path, "rb") as f:
        raw = f.read()
    lines = raw.split(b"\n")
    return raw, lines


def utc_epoch(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def births(paths):
    out = {}
    for i in range(0, len(paths), 300):
        r = subprocess.run(["stat", "--printf", "%W|%Y|%n\n"] + paths[i:i + 300], capture_output=True, text=True, check=True)
        for ln in r.stdout.splitlines():
            w, y, n = ln.split("|", 2)
            out[n] = (int(w), int(y))
    return out


def walk(root):
    out = []
    for d, _, fs in os.walk(root):
        out += [os.path.join(d, f) for f in fs]
    return sorted(out)


def audit():
    res = {"violations": []}
    raw, lines = status_lines(os.path.join(PROJ, "STATUS.md"))
    txt = [l.decode("utf-8") for l in lines]
    # (a) STATUS prefix hashes quoted by the E0 lines
    pref = []
    for i, l in enumerate(txt):
        mm = re.search(r"STATUS(?:\.md)? before this line(?: sha256 ([0-9a-f]{64}) \((\d+) lines\)| (\d+) lines sha256 ([0-9a-f]{64}))", l)
        if mm and "[e0-" in l:
            h = mm.group(1) or mm.group(4)
            n = int(mm.group(2) or mm.group(3))
            prefix = b"\n".join(lines[:n]) + b"\n"
            ok = hashlib.sha256(prefix).hexdigest() == h
            pref.append({"line": i + 1, "n_lines": n, "quoted": h, "ok": ok})
            if not ok:
                res["violations"].append({"status_prefix_hash": i + 1})
    res["status_prefix_checks"] = pref
    # (b) milestones in order, with times
    pats = {"plan": r"\[e0-freeze\] PLAN FROZEN", "code": r"\[e0-freeze\] E0 CODE FROZEN", "add1": r"\[e0-run\] ADDENDUM 1 LOGGED",
            "extract": r"\[e0-extract\] EXTRACTION DONE", "ctrcal": r"\[e0-run\] CTR CALIBRATION FROZEN",
            "guard": r"\[e0-run\] GUARD CLEAN", "verdicts": r"\[e0-run\] VERDICTS"}
    ms = {}
    for key, pat in pats.items():
        hits = [(i + 1, l) for i, l in enumerate(txt) if re.search(pat, l)]
        if hits:
            i, l = hits[0]
            ms[key] = {"line": i, "utc": l.split(" ")[0], "epoch": utc_epoch(l.split(" ")[0])}
    res["milestones"] = ms
    order = ["plan", "code", "add1", "extract", "ctrcal", "guard", "verdicts"]
    ln = [ms[k]["line"] for k in order if k in ms]
    res["milestone_order_ok"] = (len(ln) == len(order)) and ln == sorted(ln)
    if not res["milestone_order_ok"]:
        res["violations"].append({"milestone_order": ln})
    # (c) hashes named in milestone lines
    named = {
        "analysis/e0_sentence/ANALYSIS_PLAN_E0.md": "e859c6ea52f941b18ca27b83150f489dded0085a4d99218d909b36c19603f0ee",
        "analysis/e0_sentence/CODE_MANIFEST_E0.json": "6b0111c85aa5416b1d7c97b7f75fe2fe413ce8e2741bdb4e852e43e707664b13",
        "analysis/e0_sentence/ANALYSIS_PLAN_E0_ADDENDUM_1.md": "dcf0ecfc9b3e79063782a9e0d51af7db3e91c4719dee5ab8cec9490cb2509aad",
        "analysis/e0_sentence/addendum1/e0_evaluate_add1.py": "d0c1ce4af6cff66b099ed9ab1c5dd074bca9f7f2d99324b4d3d0c2805e521da8",
        "results/E0_sentence/extraction/EXTRACTION_E0.json": "27bdcb149f1b386be886b9e5539b2b72b352c54be1867a4a42f9a87c6416f8de",
        "results/E0_sentence/eval/ctr_calibration.json": "33d1e796fa1a80185b053e237fd93b58e1ccf3f72e84506ecedb484948d63da2",
        "results/E0_sentence/guard/GUARD_CHECK_20261010T140640Z.json": "f5099bd2b48b3b4d6d31a235710c99b81913cceee4d0e201feb5a28e7c20af7d",
        "results/E0_sentence/VERDICTS_E0.json": "5b04e46098e39056f6e9b2d534d53fa54777b1670ccfa68442cb7ef3fd1568a7",
        "results/E0_sentence/eval/E0_tables.md": "22b6b04b7b400b75a6c936306525d32bf90256dceb275a9a606d3a455e268a66",
        "results/E0_sentence/summary.json": "b1d6fcddeb706ba607246207d8d70b6ebc215d8f412f76877955306f1ea822b2",
        "results/E0_sentence/REPORT.md": "a7ec345754e034a30bbf89f87678678169ec2bb74974e0994d4e75c6f8f7cdb4",
        "results/E0_sentence/diag/diag_singlemodel.json": "e60bf07c222312f6073f818211423270d3edf29656ada576e85d7863d6a35c7f",
        "results/E0_sentence/obs/obs_ctrold.parquet": "f08e6699d1b23b27ed2ea262573fe3cf2c497eecdee2a798e357ac0d6f882a03",
        "results/E0_sentence/obs/obs_P.parquet": "1e13c7df10571a250de0faf44f9767ca4a835f7d9aff960a96f5b51ea68c6cbc",
        "features/coco_e0/captions_index.parquet": "ee979cc81d7d69be888438b72c2cf921489f1475db1b07c23c747caec60ed76b",
    }
    hashes = {}
    for rel, h in named.items():
        got = sha256(os.path.join(PROJ, rel))
        hashes[rel] = got == h
        if got != h:
            res["violations"].append({"hash": rel})
    res["named_hashes_ok"] = hashes
    man = json.load(open(os.path.join(PROJ, "analysis/e0_sentence/CODE_MANIFEST_E0.json")))
    bad = [rel for rel, h in man["files"].items() if sha256(os.path.join(PROJ, rel)) != h]
    res["manifest_files"] = len(man["files"])
    res["manifest_mismatch"] = bad
    if bad:
        res["violations"].append({"manifest": bad})
    # (d) birth and modification times against the milestones
    def check(paths, lo=None, hi=None, tag="", use_mtime_hi=False):
        b = births(paths)
        early, late, unknown = [], [], []
        for p in paths:
            w, y = b[p]
            if w <= 0:
                unknown.append(p)
                continue
            if lo is not None and w < lo:
                early.append(p)
            t = y if use_mtime_hi else w
            if hi is not None and t >= hi:
                late.append(p)
        rec = {"n": len(paths), "born_before_lower": early, "after_upper": late, "unknown_birth": unknown,
               "first_birth": min((b[p][0] for p in paths), default=None), "last_birth": max((b[p][0] for p in paths), default=None)}
        res.setdefault("births", {})[tag] = rec
        if early or late or unknown:
            res["violations"].append({"births": tag})
    ep = {k: v["epoch"] for k, v in ms.items()}
    # plan: content unchanged since birth, born before the PLAN FROZEN line
    check([os.path.join(PROJ, "analysis/e0_sentence/ANALYSIS_PLAN_E0.md")], hi=ep["plan"], tag="plan_before_freeze_line", use_mtime_hi=True)
    # manifest code: last modified before CODE FROZEN line
    check([os.path.join(PROJ, r) for r in man["files"]], hi=ep["code"], tag="code_mtime_before_code_freeze", use_mtime_hi=True)
    # addendum files born before the ADDENDUM line, which precedes the guard
    check([os.path.join(PROJ, "analysis/e0_sentence/ANALYSIS_PLAN_E0_ADDENDUM_1.md"),
           os.path.join(PROJ, "analysis/e0_sentence/addendum1/e0_evaluate_add1.py")], hi=ep["add1"], tag="addendum_before_line",
          use_mtime_hi=True)
    # sentence-encoder features: after plan freeze and code freeze, before EXTRACTION DONE
    feats = [p for p in walk(os.path.join(PROJ, "features/coco_e0"))]
    check(feats, lo=ep["code"], hi=ep["extract"], tag="features_between_code_freeze_and_extraction_done")
    # caches: after code freeze, before guard
    caches = walk(os.path.join(E0R, "cache"))
    check(caches, lo=ep["code"], hi=ep["guard"], tag="caches_between_code_freeze_and_guard")
    # old-pair ctr obs + calibration: before CTR line
    check([os.path.join(OBSD, "obs_ctrold.parquet"), os.path.join(E0R, "eval/ctr_calibration.json")], lo=ep["extract"],
          hi=ep["ctrcal"], tag="ctrold_and_calibration_before_ctr_line")
    # new-pair obs/eval/verdicts: after GUARD CLEAN
    old_ok = {"obs_ctrold.parquet", "obs_ctrold.json", "ctr_calibration.json"} | {f"E0_{t}_ctrold.parquet" for t in ("cells", "drawslopes", "slopes", "elasticity")}
    newp = [p for p in walk(OBSD) + walk(os.path.join(E0R, "eval")) if os.path.basename(p) not in old_ok]
    newp.append(os.path.join(E0R, "VERDICTS_E0.json"))
    check(newp, lo=ep["guard"], tag="new_pair_files_after_guard")
    # dry run of ADDENDUM 1 happened before any sentence-encoder feature existed
    dry = walk(os.path.join(PROJ, "analysis/e0_sentence/addendum1"))
    dry = [p for p in dry if "__pycache__" not in p]
    first_feat = min(births([p for p in feats if p.endswith(".npy")]).values())[0]
    check(dry, hi=first_feat, tag="addendum_dryrun_before_first_sentence_feature")
    # no other files under results/E0_sentence besides the known subdirectories
    res["results_top_level"] = sorted(os.listdir(E0R))
    res["result"] = "PASS" if not res["violations"] else "VIOLATION"
    return res


# ---------------------------------------------------------------- caption fact (plan 3.3 (c), 13.5)
def caption_fact():
    cj = json.load(open(os.path.join(PROJ, "extract/e0_sentence/coco_captions/captions_val2014.json")))
    by_img = {}
    for a in cj["annotations"]:
        by_img.setdefault(a["image_id"], []).append((a["id"], a["caption"].strip()))
    items = pd.read_parquet(os.path.join(PROJ, "features/coco_pool/items.parquet"), columns=["image_id", "text"])
    ranks, nmatch, ncap, json_pos = [], 0, [], []
    for iid, t in zip(items.image_id, items.text):
        caps = by_img.get(int(iid), [])
        ncap.append(len(caps))
        srt = [c for _, c in sorted(caps)]
        t0 = t.strip()
        if t0 in srt:
            nmatch += 1
            ranks.append(srt.index(t0))
            json_pos.append([c for _, c in caps].index(t0))  # position in the order of the json file
        else:
            ranks.append(-1)
    r = np.array(ranks)
    jp = np.array(json_pos)
    return {"n_items": len(items), "frac_c1_equals_official": nmatch / len(items),
            "rank_by_annotation_id_hist": {int(k): int(v) for k, v in zip(*np.unique(r, return_counts=True))},
            "frac_rank0": float(np.mean(r == 0)),
            "position_in_json_order_hist": {int(k): int(v) for k, v in zip(*np.unique(jp, return_counts=True))},
            "captions_per_image_hist": {int(k): int(v) for k, v in zip(*np.unique(ncap, return_counts=True))}}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"refusing to overwrite {a.out}")
    t0 = time.time()
    R = {"role": "e0-review", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "script_sha256": sha256(os.path.abspath(__file__)),
         "imports": "numpy, pandas, scipy.stats.rankdata only; no e0_*, s11copy, s11v11, hrhv2"}

    # ---- S1.1 old pairs, recomputed from the frozen S1.1 obs (clip)
    s11 = pd.read_parquet(os.path.join(S11, "obs/obs_clip.parquet"))
    s11 = s11[s11.primary == True]  # noqa: E712
    R["s11_inputs_sha256"] = {"obs_clip.parquet": sha256(os.path.join(S11, "obs/obs_clip.parquet"))}
    b_old, _ = draw_slopes(s11)
    c_old = drawmean_cells(s11)
    L_old, lnL_old = level(c_old)
    b10_old = b_old[b_old.k == 10].set_index("pair").b.to_dict()
    cls_old = s11.groupby("pair").cls.first().to_dict()
    pairs_old = sorted(b10_old)
    assert len(pairs_old) == 29
    x = np.array([lnL_old[p] for p in pairs_old]); y = np.array([b10_old[p] for p in pairs_old])
    a_, c_ = ols(x, y)
    resid = y - (a_ + c_ * x)
    vl_old = [p for p in pairs_old if cls_old[p] == "cross-modal"]
    comp_re = {v: float(np.mean([b10_old[p] for p in vl_old if p.split("__")[0] == v])) for v in VITS}
    R["frozen_constants_recomputed"] = {
        "line_a": a_, "line_c": c_, "diff_a": a_ - LINE_A, "diff_c": c_ - LINE_C,
        "s_res_ddof2": float(np.std(resid, ddof=2)), "s_res_ddof1": float(np.std(resid, ddof=1)),
        "s_res_frozen": S_RES,
        "competitor": comp_re, "competitor_max_abs_diff": max(abs(comp_re[v] - COMP[v]) for v in VITS),
        "vl_median_slope": float(np.median([b10_old[p] for p in vl_old])),
        "vl_mean_slope": float(np.mean([b10_old[p] for p in vl_old])), "beta_M_const_frozen": BETA_M_CONST}

    # ---- arm P (confirmatory P1, P2b)
    P = pd.read_parquet(os.path.join(OBSD, "obs_P.parquet"))
    R["inputs_sha256"] = {f"obs_{arm}.parquet": sha256(os.path.join(OBSD, f"obs_{arm}.parquet"))
                          for arm in ["P", "D1", "D2", "D3", "D4", "A", "Aall", "C", "CA", "CAall", "L", "Pdd", "ctrold"]}
    pairs_new = sorted(P.pair.unique())
    assert len(pairs_new) == 15
    bP, dsP = draw_slopes(P)
    cP = drawmean_cells(P)
    elP = elasticities(cP, [10, 100])
    LP, lnLP = level(cP)
    b10P = bP[bP.k == 10].set_index("pair").b.to_dict()
    p1 = rule_P1(elP)
    p1["per_pair"] = elP
    p1["per_encoder_median_sum"] = {e: float(np.median([elP[p]["sum"] for p in pairs_new if p.endswith("__" + e)])) for e in ENCS}
    # per-draw P1 (robustness, plan 9)
    pd_el = per_draw_elasticities(P, [10, 100])
    p1["per_draw_labels"] = {d: rule_P1(v)["label"] for d, v in pd_el.items()}
    p1["per_draw_median_sum"] = {d: rule_P1(v).get("median_sum") for d, v in pd_el.items()}
    p1["all_k_median_sum"] = float(np.median([v["sum"] for v in elasticities(cP, [1, 10, 100]).values()]))
    p1["k1_10_median_sum"] = float(np.median([v["sum"] for v in elasticities(cP, [1, 10]).values()]))
    R["P1"] = p1

    p2b = rule_P2b(pairs_new, lnLP, b10P)
    # leave-one-out lambda (descriptive)
    loo = {}
    for e in ENCS:
        pp = [p for p in pairs_new if not p.endswith("__" + e)]
        loo["without_" + e] = rule_P2b(pp, lnLP, b10P)["lambda_descriptive"]
    for v in VITS:
        pp = [p for p in pairs_new if not p.startswith(v + "__")]
        loo["without_" + v] = rule_P2b(pp, lnLP, b10P)["lambda_descriptive"]
    p2b["leave_one_out_lambda"] = loo
    # what level would make the median |g| reach the guard, per ViT (orientation)
    p2b["level_needed_per_vit"] = {v: float(np.exp((COMP[v] + 2 * S_RES - LINE_A) / LINE_C)) for v in VITS}
    p2b["levels"] = LP
    p2b["b10"] = b10P
    # same rule with the recomputed constants
    p2b["label_with_recomputed_constants"] = rule_P2b(pairs_new, lnLP, b10P, a_, c_, float(np.std(resid, ddof=2)), comp_re)["label"]
    R["P2b"] = p2b

    # ---- P2c: arms D1..D4, A (= D5)
    ladders = {p: [] for p in pairs_new}
    lv_arm = {}
    for arm in ["D1", "D2", "D3", "D4", "A"]:
        D = pd.read_parquet(os.path.join(OBSD, f"obs_{arm}.parquet"))
        bD, _ = draw_slopes(D)
        cD = drawmean_cells(D)
        LD, lnLD = level(cD)
        b10D = bD[bD.k == 10].set_index("pair").b.to_dict()
        lv_arm[arm] = {"median_L": float(np.median(list(LD.values()))), "median_b10": float(np.median(list(b10D.values())))}
        for p in pairs_new:
            ladders[p].append((lnLD[p], b10D[p]))
    p2c = rule_P2c(pairs_new, ladders)
    p2c["ladder_medians"] = lv_arm
    p2c["ladder"] = {p: [[float(np.exp(t[0])), t[1]] for t in ladders[p]] for p in pairs_new}
    R["P2c"] = p2c

    # ---- secondary: P2b on arm A with the arm-P line (plan 7.5), P1 in every secondary arm
    A = pd.read_parquet(os.path.join(OBSD, "obs_A.parquet"))
    bA, _ = draw_slopes(A); cA = drawmean_cells(A); LA, lnLA = level(cA)
    b10A = bA[bA.k == 10].set_index("pair").b.to_dict()
    sA = rule_P2b(pairs_new, lnLA, b10A)
    R["secondary_P2b_armA"] = {k: sA[k] for k in ("label", "median_abs_g", "lambda_descriptive", "lambda_e_descriptive", "informative_encoders", "enc_median_abs_g")}
    sec_p1 = {}
    for arm in ["D1", "D2", "D3", "D4", "A", "Aall", "C", "CA", "CAall", "L", "Pdd", "ctrold"]:
        X = pd.read_parquet(os.path.join(OBSD, f"obs_{arm}.parquet"))
        if arm == "ctrold":
            X = X[X.cls == "cross-modal"]
        r1 = rule_P1(elasticities(drawmean_cells(X), [10, 100]))
        sec_p1[arm + ("(VL)" if arm == "ctrold" else "")] = {"label": r1["label"], "median_sum": r1.get("median_sum")}
    R["secondary_P1"] = sec_p1

    # ---- descriptive P2 (plan 7.2), with the frozen permutation rule
    vl_all = vl_old + pairs_new
    Lall = {**L_old, **LP}; lnall = {**lnL_old, **lnLP}; ball = {**b10_old, **b10P}
    def text_of(p):
        return p.split("__")[1]
    order = [f"{v}__{t}" for v in VITS for t in TEXT_ORDER]
    assert sorted(order) == sorted(vl_all)
    lv = np.array([Lall[p] for p in order]); sv = np.array([ball[p] for p in order])
    rho = spearman(lv, sv)
    seed = int(hashlib.sha256(b"e0-p2-perm").hexdigest()[:16], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    rl = rankdata(lv); rl = (rl - rl.mean()) / np.sqrt(((rl - rl.mean()) ** 2).sum())
    rs = rankdata(sv); rs = (rs - rs.mean()) / np.sqrt(((rs - rs.mean()) ** 2).sum())
    Rn = 100000
    cnt = 0
    for _ in range(Rn):
        idx = np.concatenate([6 * j + rng.permutation(6) for j in range(5)])
        if float(np.dot(rl, rs[idx])) >= rho - 1e-12:
            cnt += 1
    p_perm = (1 + cnt) / (1 + Rn)
    Lt = np.array([np.median([Lall[f"{v}__{t}"] for v in VITS]) for t in TEXT_ORDER])
    bt = np.array([np.median([ball[f"{v}__{t}"] for v in VITS]) for t in TEXT_ORDER])
    rho_t = spearman(Lt, bt)
    import itertools
    perms = list(itertools.permutations(range(6)))
    p_exact = sum(spearman(Lt, bt[list(pi)]) >= rho_t - 1e-12 for pi in perms) / len(perms)
    R["P2_descriptive"] = {"rho_30": rho, "p_within_vit_perm": p_perm, "seed": seed, "rho_6": rho_t, "p_exact_6": p_exact,
                           "level_by_text": dict(zip(TEXT_ORDER, Lt.tolist())), "slope_by_text": dict(zip(TEXT_ORDER, bt.tolist()))}

    # ---- level wording (plan 13.3), arms P and C
    C = pd.read_parquet(os.path.join(OBSD, "obs_C.parquet")); cC = drawmean_cells(C); LC, _ = level(cC)
    CO = pd.read_parquet(os.path.join(OBSD, "obs_ctrold.parquet")); cCO = drawmean_cells(CO); LCO, _ = level(cCO)
    cls_co = CO.groupby("pair").cls.first().to_dict()
    def med(d, keys):
        return float(np.median([d[k] for k in keys]))
    vis_old = [p for p in pairs_old if cls_old[p] == "cross-vision"]
    lev = {"P": {"sentence": med(LP, pairs_new), "lm_token": med(L_old, vl_old), "vision": med(L_old, vis_old)},
           "C": {"sentence": med(LC, pairs_new), "lm_token": med(LCO, [p for p in LCO if cls_co[p] == "cross-modal"]),
                 "vision": med(LCO, [p for p in LCO if cls_co[p] == "cross-vision"])}}
    def word(d):
        if d["sentence"] > d["vision"]:
            return "above"
        if d["sentence"] > d["lm_token"]:
            return "between"
        return "not above"
    wP, wC = word(lev["P"]), word(lev["C"])
    lev["wording_13_3"] = wP if wP == wC else "mixed"
    lev["within_vit_min_enc_over_max_lm"] = {v: min(LP[f"{v}__{e}"] for e in ENCS) > max(L_old[f"{v}__{t}"] for t in LMS) for v in VITS}
    lev["within_vit_ranges"] = {v: {"lm": [min(L_old[f"{v}__{t}"] for t in LMS), max(L_old[f"{v}__{t}"] for t in LMS)],
                                    "enc": [min(LP[f"{v}__{e}"] for e in ENCS), max(LP[f"{v}__{e}"] for e in ENCS)]} for v in VITS}
    R["levels_13_3"] = lev

    # ---- Schnaus-matched cells (P3h), arms CA and P
    def L10000(cells, pair, k):
        c = cells[(cells.pair == pair) & (cells.k == k)].set_index("m")
        l8, l16 = c.loc[8192, "lns"], c.loc[16384, "lns"]
        return float(np.exp(l8 + (np.log(10000) - np.log(8192)) / np.log(2) * (l16 - l8)))
    CA = pd.read_parquet(os.path.join(OBSD, "obs_CA.parquet")); cCA = drawmean_cells(CA)
    R["schnaus_matched"] = {
        "CA_dinov2_mpnet_k100": L10000(cCA, "dinov2_b__mpnet", 100), "CA_dinov2_mpnet_k10": L10000(cCA, "dinov2_b__mpnet", 10),
        "P_dinov2_mpnet_k100": L10000(cP, "dinov2_b__mpnet", 100), "P_dinov2_mpnet_k10": L10000(cP, "dinov2_b__mpnet", 10),
        "CA_median_k100": float(np.median([L10000(cCA, p, 100) for p in pairs_new])),
        "CA_max_k100": float(np.max([L10000(cCA, p, 100) for p in pairs_new])),
        "ctrold_vision_median_k100": float(np.median([L10000(cCO, p, 100) for p in LCO if cls_co[p] == "cross-vision"])),
        "ctrold_vision_max_k100": float(np.max([L10000(cCO, p, 100) for p in LCO if cls_co[p] == "cross-vision"]))}

    # ---- slope classes and class ordering (P3d, P3f)
    so = {}
    for k in (1, 10, 100):
        bo = b_old[b_old.k == k].set_index("pair").b.to_dict(); bn = bP[bP.k == k].set_index("pair").b.to_dict()
        Lo, _ = level(c_old, k); Ln, _ = level(cP, k)
        classes = {"seed": [p for p in pairs_old if cls_old[p] == "seed-seed"], "vision": vis_old, "vl_lm": vl_old}
        mb = {c: float(np.median([bo[p] for p in ps])) for c, ps in classes.items()}
        mL = {c: float(np.median([Lo[p] for p in ps])) for c, ps in classes.items()}
        mb["vl_sentence"] = float(np.median(list(bn.values()))); mL["vl_sentence"] = float(np.median(list(Ln.values())))
        so[k] = {"median_b": mb, "median_L_m1024": mL,
                 "slope_order": sorted(mb, key=lambda c: -mb[c]), "level_order": sorted(mL, key=lambda c: -mL[c]),
                 "new_pairs_falling": int(sum(v <= -0.1 for v in bn.values())), "new_b_range": [min(bn.values()), max(bn.values())]}
        so[k]["orders_equal"] = so[k]["slope_order"] == so[k]["level_order"]
    R["slope_classes_and_order"] = so

    # ---- obs step from the stored neighbour lists (arms P and D1..D5; all draws, m, k)
    arm_inp = {"P": "c1", "D1": "n1", "D2": "n2", "D3": "n3", "D4": "n4", "A": "n5"}
    maxdiff, maxdiff_all, n_cells, self_first_all, n_mismatch_anchors = 0.0, 0.0, 0, True, 0
    for arm, inp in arm_inp.items():
        X = pd.read_parquet(os.path.join(OBSD, f"obs_{arm}.parquet")).set_index(["pair", "draw", "m", "k"])
        for p in pairs_new:
            v, e = p.split("__")
            for dr in range(5):
                for m in M_GRID:
                    for k in (1, 10, 100):
                        o, na, oall, nall, sf = obs_from_caches(v, f"{e}.{inp}", dr, m, k)
                        row = X.loc[(p, dr, m, k)]
                        maxdiff = max(maxdiff, abs(o - row["obs"]))
                        maxdiff_all = max(maxdiff_all, abs(oall - row["obs_allH2"]))
                        n_mismatch_anchors += int(na != row["n_anchors"] or nall != row["n_allH2"])
                        self_first_all &= sf
                        n_cells += 1
    R["obs_step_from_caches"] = {"n_cells": n_cells, "max_abs_diff_obs": maxdiff, "max_abs_diff_obs_allH2": maxdiff_all,
                                 "anchor_count_mismatches": n_mismatch_anchors, "self_listed_first_in_any_cell": self_first_all,
                                 "note": "checks the overlap step from the stored neighbour lists; the caches themselves are V-f's job"}

    # ---- caption fact and freeze audit
    R["caption_fact"] = caption_fact()
    R["audit"] = audit()

    # ---- compare with VERDICTS_E0.json
    V = json.load(open(os.path.join(E0R, "VERDICTS_E0.json")))
    cmp = {"labels_equal": {"P1": V["labels"]["P1"] == R["P1"]["label"], "P2b": V["labels"]["P2b"] == R["P2b"]["label"],
                            "P2c": V["labels"]["P2c"] == R["P2c"]["label"]},
           "P1_median_sum_diff": abs(V["P1"]["median_sum"] - R["P1"]["median_sum"]),
           "P2b_median_abs_gap_diff": abs(V["P2b"]["median_abs_gap"] - R["P2b"]["median_abs_g"]),
           "P2c_median_gamma_diff": abs(V["P2c"]["median_gamma"] - R["P2c"]["median_gamma"]),
           "P2c_median_dln_diff": abs(V["P2c"]["median_dln"] - R["P2c"]["median_dln"]),
           "P2_rho_diff": abs(V["P2_descriptive"]["within_vit_permutation"]["rho"] - R["P2_descriptive"]["rho_30"]),
           "P2_p_diff": abs(V["P2_descriptive"]["within_vit_permutation"]["p_one_sided"] - R["P2_descriptive"]["p_within_vit_perm"]),
           "leave_mpnet_out_lambda_diff": abs(V["P2b"]["leave_mpnet_out_lambda"] - R["P2b"]["leave_one_out_lambda"]["without_mpnet"]),
           "secondary_P2b_A_label": V["secondary"]["P2b"]["A"].get("label")}
    R["comparison_with_VERDICTS"] = cmp
    R["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    tmp = a.out + ".tmp"
    with open(tmp, "x") as f:
        json.dump(conv(R), f, indent=1)
    os.rename(tmp, a.out)
    print(a.out, sha256(a.out))


if __name__ == "__main__":
    main()
