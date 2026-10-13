#!/usr/bin/env python
"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/lmx_ol_aggregate.py (sha256 90796c04...), sub-role pyt-code-I
(S0b, 2026-10-10): aggregation of implementation I on the Pythia axis, writing the verdict contract
analysis/s12_layermax_pythia/VERDICTS_SPEC_PYT.md (S0b v1) sections A, B, C, H, R, D, G.

Reads (only after the order gate, plan section 4 F1(a)):
  results/S1_2_layermax_pythia/indep/<pre>/pairs/<lm>__<vit>.npz       the 7 Pythia x 17 ViT pairs of this run
  results/S1_2_layermax_pythia/indep/prh_clip+l2/prep_meta.json         V-P1 flag of section G
  results/S1_2_layermax_pythia/partB/partB_features_pyt.json            frozen Part B (spec 11 field names; s, s_late,
                                                                        membership, H2 branch, B-bar, B_late, bands,
                                                                        flags, num_params, width, depth; never recomputed)
  results/S1_2_layermax/indep/, results/S1_2_layermax_llama/indep/,
  results/S1_2_layermax_openllama/indep/  <pre>/pairs/                  the 11 known LMs (section D only, full layer-max)
  analysis/s12_layermax_pythia/verdict_keys.py                          the contract's required key list (names only)
Writes:
  results/S1_2_layermax_pythia/indep/verdicts_pyt.json   {"schema": "PYT-VERDICTS v1", "impl": "I", "inputs", "v"}:
                                                          every key of verdict_keys.required_keys("EI") (exit 1 if one is
                                                          missing; every l2 key null if l2 was not run), plus x| extras
                                                          (I's own category, modifiers, readings and P-P2b as x|I|...)
  results/S1_2_layermax_pythia/indep/summary.json        full detail (per-pair records, rho vectors, provenance)
An earlier output is kept aside (<name>.prev-<UTC>), never overwritten. The verdict arithmetic lives in pyt_verdicts.py.
Changes against the OpenLLaMA copy (PYT-AXIS): model list and paths; pair records unchanged except the added entry null
mean at the M_unc-argmax (muM) and n_lm_layers; the OpenLLaMA sections A-D, G are replaced by the contract's sections;
"--pres" allows the primary pre alone (plan 11: l2 may be "not run (budget)"); "--legacy" writes the V-P8 file
(n = 3, OpenLLaMA key names, schema OL-VERDICTS v1; S0b test mode only); "--fixture"/"--partb" (S0b test mode only)
run the V-P9 7-size stand-in on the shared stand-in Part B (spec 13).

usage: launch_pyt.sh pyt_I_aggregate 4 -- venv/bin/python analysis/s12_layermax_pythia/indep/lmx_pyt_aggregate.py
       [--pres prh_clip+l2[,l2]]
CPU only, <= 4 threads, about a minute.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True   # PYT-AXIS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402  PYT-AXIS
OUT_ROOT = pyt_gate.gate(__file__)   # PYT-AXIS F1(a): exits 3 unless the plan's gates hold (S0b test mode: test root)

_T = os.environ.get("PYT_THREADS", "1")   # PYT-AXIS
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402

import lmx_common as L  # noqa: E402
from lmx_common import C, cal  # noqa: E402
import pyt_verdicts as PV  # noqa: E402  PYT-AXIS

sys.path.insert(0, os.path.join(L.PROJ, "analysis", "s12"))
import aggregate as S12  # noqa: E402  (S1.2 helper boot_vits, read-only import)

HERE = os.path.dirname(os.path.abspath(__file__))
PARTB_DEFAULT = (os.path.join(os.path.dirname(C.CLIP_PARTB), "partB_features_pyt.json") if pyt_gate.test_mode()
                 else pyt_gate.PARTB_JSON)   # PYT-AXIS (test: the shared stand-in s0b_scratch/PB/partB_vp9/, spec 13)
sys.path.insert(0, pyt_gate.AXIS)
import verdict_keys as VK  # noqa: E402  PYT-AXIS: the shared contract's key lists (names only, read-only import)
PLAN = os.path.join(pyt_gate.AXIS, "ANALYSIS_PLAN_PYTHIA.md")   # PYT-AXIS
PLAN_SHA256 = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"   # PYT-AXIS
METRICS = PV.METRICS
NPZ = {"mknn": ("obs_cnt", "null_cnt"), "cka_biased": ("obs_cka_b", "null_cka_b"),
       "cka_debiased": ("obs_cka_d", "null_cka_d")}
NV = len(L.VIT_STEMS)
PYT = list(L.LM_STEMS)   # PYT-AXIS
TAG = [C.SIZE_OF[s] for s in PYT]   # PYT-AXIS
POOLED = list(L.POOLED_LM_STEMS)
K = L.K_PERM
MK = L.M * L.KNN
ORDER_IDX = cal.order_index(L.K_PERM, L.ALPHA) - 1          # 475: the 476th ascending order statistic
PRIMARY = "prh_clip+l2"
FIXTURE = False   # PYT-AXIS: set by --fixture (S0b test mode)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- layer subsets
def fixed_layer(n: int, r: float) -> int:
    return int(math.floor(r * (n - 1) + 0.5))


def grid_layers(n: int, k: int) -> list:
    return sorted({fixed_layer(n, j / (k - 1)) for j in range(k)})


def subsets(LA, LB):
    """the 13 layer subsets of plan section 3 (iii) (= OpenLLaMA spec section 1)."""
    allA, allB = list(range(LA)), list(range(LB))
    out = {"full": (allA, allB)}
    for n in (2, 3, 5, 9, 17):
        out[f"grid{n}"] = (grid_layers(LA, n), grid_layers(LB, n))
    out["fixed_r0.5"] = ([fixed_layer(LA, 0.5)], [fixed_layer(LB, 0.5)])
    out["fixed_r1.0"] = ([LA - 1], [LB - 1])
    out["LMall_ViTlast"] = (allA, [LB - 1])
    out["LMlast_ViTall"] = ([LA - 1], allB)
    out["noLMemb"] = (list(range(1, LA)), allB)
    out["LMdeep"] = ([a for a in range(LA) if a / (LA - 1) >= 0.5], allB)
    out["LMemb_only"] = ([0], allB)
    return out


# ----------------------------------------------------------------------------- per pair records
def _summ(met, Ss, Ns):
    """layer-max summary of one subset. mKNN: exact Fractions from the integer counts."""
    if met == "mknn":
        T = int(Ss.max())
        Tk = Ns.reshape(Ns.shape[0], -1).max(axis=1).astype(np.int64)
        c = int(np.sort(np.append(Tk, T))[ORDER_IDX])
        s_cal = Fraction(T - c, MK - c) if (MK - c > 0 and T > c) else Fraction(0)
        mu = Fraction(int(Tk.sum()), K * MK)
        return {"raw": Fraction(T, MK), "c": Fraction(c, MK), "s_cal": s_cal, "mu": mu, "unc": Fraction(T, MK) - mu,
                "p": (1.0 + float(np.sum(Tk >= T))) / (K + 1), "null_sd": float(np.std(Tk / MK, ddof=1))}
    T = float(Ss.max())
    Tk = Ns.reshape(Ns.shape[0], -1).max(axis=1)
    sm = L.summarize(T, Tk)
    return {"raw": sm["s_obs"], "c": sm["c_alpha"], "s_cal": sm["s_cal"], "mu": sm["null_mean"],
            "unc": sm["uncensored"], "p": sm["p_value"], "null_sd": sm["null_sd"]}


def _entry_c(met, S, N, a, b):
    if met == "mknn":
        ce = int(np.sort(np.append(N[:, a, b], S[a, b]))[ORDER_IDX])
        return Fraction(ce, MK), Fraction(int(N[:, a, b].sum()), K * MK)
    sm = L.summarize(float(S[a, b]), N[:, a, b])
    return sm["c_alpha"], sm["null_mean"]


def _ctm(met, S, N):
    """(iv) calibrate-then-max: every entry against its own c_alpha,uv from the same 500 pi."""
    Ce = np.sort(np.concatenate([S[None], N], axis=0), axis=0)[ORDER_IDX]
    LA, LB = S.shape
    if met == "mknn":
        best, ba, bb = Fraction(0), 0, 0
        vals = {}
        for (u, w), sv in np.ndenumerate(S):
            cu = int(Ce[u, w])
            q = Fraction(int(sv) - cu, MK - cu) if (MK - cu > 0 and int(sv) > cu) else Fraction(0)
            vals[(u, w)] = q
            if q > best:
                best, ba, bb = q, u, w
        U = K * S.astype(np.int64) - N.sum(axis=0)
        ua, ub = np.unravel_index(int(np.argmax(U)), U.shape)
        return {"M": best, "M_unc": Fraction(int(U.max()), K * MK), "TM": Fraction(int(S[ba, bb]), MK),
                "cM": Fraction(int(Ce[ba, bb]), MK), "a": int(ba), "b": int(bb), "unc_a": int(ua), "unc_b": int(ub),
                "muM": Fraction(int(N[:, ua, ub].sum()), K * MK),   # PYT-AXIS: entry null mean at the M_unc-argmax
                "rel_a": ba / (LA - 1), "n_entries_scal_pos": int(sum(1 for q in vals.values() if q > 0)),
                "n_tied_max": int(sum(1 for q in vals.values() if q == best))}
    den = 1.0 - Ce
    scal = np.where(den > 0, np.maximum((S - Ce) / np.where(den > 0, den, 1.0), 0.0), 0.0)
    unc = S - N.mean(axis=0)
    ia, ib = np.unravel_index(int(np.argmax(scal)), scal.shape)
    ua, ub = np.unravel_index(int(np.argmax(unc)), unc.shape)
    return {"M": float(scal.max()), "M_unc": float(unc.max()), "TM": float(S[ia, ib]), "cM": float(Ce[ia, ib]),
            "muM": float(N[:, ua, ub].mean()),   # PYT-AXIS: entry null mean at the M_unc-argmax (spec 5 P-P5a, x = unc)
            "a": int(ia), "b": int(ib), "unc_a": int(ua), "unc_b": int(ub), "rel_a": ia / (LA - 1),
            "n_entries_scal_pos": int(np.sum(scal > 0)), "n_tied_max": int(np.sum(scal == scal.max()))}


def pair_record(z, detail=True):
    rec = {}
    for met in METRICS:
        ko, kn = NPZ[met]
        if met == "mknn":
            S, N = z[ko].astype(np.int64), z[kn].astype(np.int64)
        else:
            S, N = np.asarray(z[ko], dtype=np.float64), np.asarray(z[kn], dtype=np.float64)
        LA, LB = S.shape
        subs = subsets(LA, LB) if detail else {"full": (list(range(LA)), list(range(LB)))}
        r = {}
        for q, (ra, rb) in subs.items():
            ra, rb = np.asarray(ra), np.asarray(rb)
            Ss = S[np.ix_(ra, rb)]
            Ns = N[:, ra][:, :, rb]
            d = _summ(met, Ss, Ns)
            ia, ib = np.unravel_index(int(np.argmax(Ss)), Ss.shape)
            d.update({"a": int(ra[ia]), "b": int(rb[ib]), "n_entries": int(Ss.size),
                      "n_tied_max": int(np.sum(Ss == Ss.max())), "LA": LA, "LB": LB,
                      "n_lm_layers": int(len(ra))})   # PYT-AXIS: grid saturation flag (plan 3)
            r[q] = d
        a, b = r["full"]["a"], r["full"]["b"]
        r["full"]["c_entry"], r["full"]["mu_entry"] = _entry_c(met, S, N, a, b)
        if met == "mknn":
            r["full"]["max_entry_mu"] = Fraction(int(N.sum(axis=0).max()), K * MK)
        else:
            r["full"]["max_entry_mu"] = float(N.mean(axis=0).max())
        if detail:
            r["ctm"] = _ctm(met, S, N)
        rec[met] = r
    return rec


def pair_path(base, pre, lm, vit):   # PYT-AXIS: (path, key stem) of a pair file; test fixture -> frozen stand-in
    if FIXTURE and lm in C.STANDIN:
        src = C.source_stem(lm)
        return os.path.join(L.frozen_indep_root(src), pre, "pairs", f"{src}__{vit}.npz"), src
    return os.path.join(base, pre, "pairs", f"{lm}__{vit}.npz"), lm


def load_z(base, pre, lm, vit):
    path, kstem = pair_path(base, pre, lm, vit)   # PYT-AXIS
    z = np.load(path)
    assert int(z["K"]) == K and str(z["key"]) == L.perm_key(kstem, vit), (path, str(z["key"]))
    return z


def jrec(o):
    """JSON form of a record (Fractions -> float)."""
    if isinstance(o, dict):
        return {str(k): jrec(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jrec(v) for v in o]
    if isinstance(o, Fraction):
        return float(o)
    return L.jsonable(o)


# ----------------------------------------------------------------------------- PYT-AXIS: section D (spec 8)
def section_d(recs_pyt, pres, npar_pyt):
    """cross-family table and the 18-LM panel (descriptive): famtr, famdtr, panel18. Known LMs from their frozen I
    counterparts (plan 3: each implementation reads its own), full layer-max records only."""
    from hrhv2 import util
    V = {}
    fam_rows = {f: list(VK.LM_FAMILIES[f]) for f in ("BLOOMZ", "LLaMA", "OpenLLaMA")}
    fam_rows["Pythia"] = list(PYT)
    rows18 = [s_ for f in ("BLOOMZ", "LLaMA", "OpenLLaMA", "Pythia") for s_ in fam_rows[f]]
    npar = {s_: C.n_params(s_) for f in ("BLOOMZ", "LLaMA", "OpenLLaMA") for s_ in fam_rows[f]}
    npar.update(dict(zip(PYT, npar_pyt)))
    x18 = np.log(np.array([float(npar[s_]) for s_ in rows18], dtype=np.float64))
    n_known = 0
    for pre in pres:
        recs = {}
        for f in ("BLOOMZ", "LLaMA", "OpenLLaMA"):
            for lm in fam_rows[f]:
                recs[lm] = {v: pair_record(load_z(L.frozen_indep_root(lm), pre, lm, v), detail=False)
                            for v in L.VIT_STEMS}
                n_known += NV
        recs.update({lm: recs_pyt[pre][lm] for lm in PYT})
        for met in METRICS:
            for x in ("raw", "s_cal", "unc"):
                G = np.array([[float(recs[lm][v][met]["full"][x]) for v in L.VIT_STEMS] for lm in rows18])
                r = PV.pear(G, x18)
                k = f"panel18|{pre}|{met}|{x}"
                V[f"{k}|r"] = None if not math.isfinite(r) else r
                rng = util.rng_from_key(f"S1.2|bootvit|layermax-pythia|panel18|{pre}|{met}|{x}")
                bs = np.empty(PV.NB)
                for b in range(PV.NB):
                    j = rng.integers(0, NV, NV)
                    bs[b] = PV.pear(G[:, j], x18)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                V[f"{k}|ci_lo"], V[f"{k}|ci_hi"] = float(lo), float(hi)
                for f, rows in fam_rows.items():
                    V[f"famtr|{pre}|{met}|{x}|{f}"] = PV.trend(
                        [[recs[lm][v][met]["full"][x] for lm in rows] for v in L.VIT_STEMS])["obs"]
            for x in PV.XS:
                for f, rows in fam_rows.items():
                    dt = PV.dtrend([[recs[lm][v][met]["full"][x] for lm in rows] for v in L.VIT_STEMS],
                                   [[recs[lm][v][met]["full"]["raw"] for lm in rows] for v in L.VIT_STEMS], "common")
                    V[f"famdtr|{pre}|{met}|{x}|{f}"] = None if dt is None else float(np.mean(dt["drho"]))
    return V, n_known


# ----------------------------------------------------------------------------- PYT-AXIS: legacy mode (V-P8)
OL_SLUGS = ["openlm-research__open_llama_3b", "openlm-research__open_llama_7b", "openlm-research__open_llama_13b"]
OL_PARTB = os.path.join(L.PROJ, "results", "S1_2_layermax_openllama", "partB", "partB_features_ol.json")


def ol_root(lm):
    """the OpenLLaMA I aggregator's source rule: OpenLLaMA models from its own run, llama-13b/30b from the LLaMA-axis
    I run, BLOOMZ and llama-7b from the frozen 8-LM I run."""
    if lm in OL_SLUGS:
        return L.OL_INDEP
    if lm in C.LLAMA_AXIS_STEMS:
        return L.LLAMA_INDEP
    return L.OLD_INDEP


def legacy_verdicts(mode="legacy"):
    """V-P8: the generalised n-size code in legacy mode (n = 3) on the frozen OpenLLaMA I pair files; returns the
    OpenLLaMA sections A, B, C (P-O4's pooled 11-LM r from the frozen I files of the 11 LMs) and the records."""
    pb = json.load(open(OL_PARTB))
    rec = {p: {lm: {v: pair_record(load_z(ol_root(lm), p, lm, v), detail=True) for v in L.VIT_STEMS} for lm in OL_SLUGS}
           for p in L.PRES}
    pooled = {"r": {}}
    arh = np.array([L.ARH_RANK[s_] for s_ in L.POOLED_LM_STEMS], dtype=np.float64)
    for p in L.PRES:
        G = {x: [] for x in ("raw", "s_cal")}
        for lm in L.POOLED_LM_STEMS:
            rr = rec[p][lm] if lm in OL_SLUGS else {v: pair_record(load_z(ol_root(lm), p, lm, v), detail=False)
                                                    for v in L.VIT_STEMS}
            for x in G:
                G[x].append([float(rr[v]["cka_debiased"]["full"][x]) for v in L.VIT_STEMS])
        for x in G:
            r = PV.pear(np.array(G[x]), arh)
            pooled["r"][("11", p, "cka_debiased", x, "arh")] = None if not math.isfinite(r) else r
    ex = {}
    V = PV.ol_verdicts(rec, OL_SLUGS, list(L.VIT_STEMS), pb, pooled, ex, mode=mode)
    return {k: (v if isinstance(v, str) else PV.f(v)) for k, v in V.items()}, rec


# ----------------------------------------------------------------------------- main
def main():
    global FIXTURE
    ap = argparse.ArgumentParser(description="implementation I aggregate, Pythia axis (VERDICTS_SPEC_PYT A,B,C,H,R,D,G)")
    ap.add_argument("--pres", default=",".join(L.PRES), help="comma list; the primary pre alone if l2 is not run")
    ap.add_argument("--fixture", action="store_true", help="S0b test mode only: the 7-size stand-in fixture (V-P9)")
    ap.add_argument("--partb", default=None, help="S0b test mode only: stand-in Part B json")
    ap.add_argument("--legacy", action="store_true", help="S0b test mode only: V-P8 legacy file (OL-VERDICTS v1)")
    args = ap.parse_args()
    assert os.path.realpath(L.OUT) == os.path.realpath(OUT_ROOT)
    if (args.fixture or args.partb or args.legacy) and not pyt_gate.test_mode():
        raise SystemExit("--fixture / --partb / --legacy are S0b test options")
    t0 = time.time()
    os.makedirs(L.OUT, exist_ok=True)
    if args.legacy:   # spec 0 "Legacy mode": n = 3 on the frozen OpenLLaMA pair files, OpenLLaMA key names
        V, _ = legacy_verdicts("legacy")
        out = {"schema": "OL-VERDICTS v1", "impl": "I", "mode": "legacy (pyt_verdicts.ol_verdicts, n = 3)",
               "inputs": {os.path.relpath(OL_PARTB, L.PROJ): sha(OL_PARTB)}, "v": V}
        os.makedirs(os.path.join(L.OUT, "legacy"), exist_ok=True)
        L.write_json(os.path.join(L.OUT, "legacy", "verdicts_ol_legacy.json"), out)
        log(f"wrote legacy file: {len(V)} keys ({time.time() - t0:.0f}s)")
        return
    FIXTURE = bool(args.fixture)
    pres = tuple(args.pres.split(","))
    assert pres[0] == PRIMARY and set(pres) <= set(L.PRES), pres
    partb_path = args.partb or PARTB_DEFAULT
    pb = json.load(open(partb_path))
    assert list(pb["pyt_lms"]) == PYT and list(pb["vits"]) == list(L.VIT_STEMS) and list(pb["sizes"]) == TAG, \
        "Part B model, ViT or size lists differ from implementation I"
    recs = {}
    n_read = 0
    for pre in pres:
        recs[pre] = {}
        for lm in PYT:
            recs[pre][lm] = {}
            for vit in L.VIT_STEMS:
                recs[pre][lm][vit] = pair_record(load_z(L.OUT, pre, lm, vit), detail=True)
                n_read += 1
        log(f"{pre}: {len(PYT) * NV} pair records built ({time.time() - t0:.0f}s)")
    aux = {"num_params": [int(pb["num_params"][s_]) for s_ in PYT], "width": [int(pb["width"][s_]) for s_ in PYT],
           "depth": [int(pb["depth_blocks"][s_]) for s_ in PYT], "L_stored": [int(pb["L_stored"][s_]) for s_ in PYT],
           "vit_family": [VK.VIT_FAMILY4[v] for v in L.VIT_STEMS]}
    pbv = {pre: PV.partb_view(pb, pre, PYT, list(L.VIT_STEMS), TAG) for pre in pres}
    extra = {}
    V = PV.verdicts_pyt(recs, PYT, list(L.VIT_STEMS), pbv, aux, extra, pres, tags=TAG)
    log(f"sections A, B, C, H, R done ({time.time() - t0:.0f}s)")
    VD, n_known = section_d(recs, pres, aux["num_params"])
    V.update(VD)
    log(f"section D done ({time.time() - t0:.0f}s)")
    # ---- section G (spec 9)
    clip_eq = None
    meta_fn = os.path.join(L.OUT, PRIMARY, "prep_meta.json")
    if not FIXTURE and os.path.exists(meta_fn):
        meta = json.load(open(meta_fn))
        clip_eq = all(bool(meta[s_]["clip_bitwise_equal"]) for s_ in PYT if s_ in meta) and all(s_ in meta for s_ in PYT)
    V.update({"G|n_pairs_read": int(n_read), "G|n_known_records": int(n_known), "G|l2_run": "l2" in pres,
              "G|partB_sha256": sha(partb_path), "G|clip_equal_partB": clip_eq})
    # ---- every l2 key null if l2 was not run (spec 9); required keys present?
    required = VK.required_keys("EI")
    if "l2" not in pres:
        for k in required:
            if k not in V and "|l2|" in f"{k}|":
                V[k] = None
    missing = [k for k in required if k not in V]
    # ---- descriptive extras (x|): ViT bootstrap CIs of Tr (key S1.2|bootvit|layermax-pythia|<stat>), vision axis
    assert list(S12.C.VITS) == list(C.VITS), "ViT order differs between the S1.2 registry and the copy"
    for p in pres:
        for m in METRICS:
            for q in ("full", "fixed_r0.5", "fixed_r1.0", "ctm"):
                for x in (("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")):
                    rho = extra.get(f"x|rho|{p}|{m}|{q}|{x}")
                    if rho is None or all(r is None for r in rho):
                        continue
                    vec = np.array([np.nan if r is None else r for r in rho], dtype=np.float64)
                    _, lo, hi = S12.boot_vits(vec, f"layermax-pythia|{p}|{m}|{q}|{x}")
                    extra[f"x|tr_ci_lo|{p}|{m}|{q}|{x}"], extra[f"x|tr_ci_hi|{p}|{m}|{q}|{x}"] = lo, hi
            for x in ("raw", "s_cal", "unc"):
                G = np.array([[float(recs[p][lm][v][m]["full"][x]) for v in L.VIT_STEMS] for lm in PYT])
                vt = S12.vision_trend(G)
                extra[f"x|I|vis|{p}|{m}|{x}|tr"], extra[f"x|I|vis|{p}|{m}|{x}|p_one"] = vt["Tr"], vt["p_one"]
    rho = {k: v for k, v in list(extra.items()) if k.startswith("x|rho|")}
    for k in rho:
        extra.pop(k)
    vout = {k: (v if isinstance(v, str) else PV.f(v)) for k, v in {**V, **extra}.items()}
    for k, v in vout.items():
        assert v is None or isinstance(v, (int, float, str, bool)), (k, type(v))

    def digest(lms_, pres_=pres):
        h = hashlib.sha256()
        for pre in pres_:
            for lm in lms_:
                for vit in L.VIT_STEMS:
                    fn, _ = pair_path(L.OUT, pre, lm, vit)
                    h.update(f"{pre}/pairs/{lm}__{vit}.npz {sha(fn)}\n".encode())
        return h.hexdigest()
    inputs = {os.path.relpath(partb_path, L.PROJ): sha(partb_path),
              os.path.relpath(L.OUT, L.PROJ) + "/<pre>/pairs/{7 Pythia x 17 ViTs} (digest)": digest(PYT)}
    for fpath in (PLAN, os.path.join(pyt_gate.AXIS, "VERDICTS_SPEC_PYT.md"), os.path.join(pyt_gate.AXIS, "verdict_keys.py")):
        if os.path.exists(fpath):
            inputs[os.path.relpath(fpath, L.PROJ)] = sha(fpath)
    if os.path.exists(pyt_gate.PARTB_MD) and not pyt_gate.test_mode():
        inputs[os.path.relpath(pyt_gate.PARTB_MD, L.PROJ)] = sha(pyt_gate.PARTB_MD)
    for fn in ("pyt_verdicts.py", "lmx_pyt_aggregate.py", "lmx_common.py", "common_pyt.py", "pyt_gate.py"):
        inputs[os.path.relpath(os.path.join(HERE, fn), L.PROJ)] = sha(os.path.join(HERE, fn))
    verd = {"schema": "PYT-VERDICTS v1", "impl": "I", "inputs": inputs, "v": vout}
    summary = {
        "role": "pyt-axis I (implementation I aggregate)", "implementation": "I",
        "test_mode": bool(pyt_gate.test_mode()), "fixture": FIXTURE,
        "plan": "analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md", "plan_sha256_expected": PLAN_SHA256,
        "spec": "analysis/s12_layermax_pythia/VERDICTS_SPEC_PYT.md (S0b v1)",
        "K": K, "alpha": L.ALPHA, "pyt_order": PYT, "size_tags": TAG, "vit_order": list(L.VIT_STEMS),
        "standin": dict(C.STANDIN), "aux": aux, "pres": list(pres),
        "missing_required_keys": missing[:200], "n_missing_required_keys": len(missing),
        "n_required_keys": len(required),
        "verdicts": vout,
        "rho_per_vit": {k[len("x|rho|"):]: v for k, v in rho.items()},
        "pairs": {pre: {f"{lm}|{v}": jrec(recs[pre][lm][v]) for lm in PYT for v in L.VIT_STEMS} for pre in pres},
        "inputs": inputs,
    }
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["secs"] = time.time() - t0
    for name, obj in (("summary.json", summary), ("verdicts_pyt.json", verd)):
        L.write_json(os.path.join(L.OUT, name), obj,   # PYT-AXIS: an earlier output is kept aside (no-delete rule)
                     default=lambda o: float(o) if isinstance(o, Fraction) else L.jsonable(o))
    log(f"wrote {L.OUT}/summary.json and verdicts_pyt.json: {len(vout)} keys, {len(required)} required, "
        f"{len(missing)} missing ({time.time() - t0:.0f}s)")
    if missing:
        log(f"MISSING required keys (first 20): {missing[:20]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
