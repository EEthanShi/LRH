"""Pythia axis, sub-role R: V-P8 for R's generalised aggregator (TEST ONLY; NEW FILE, PYT-AXIS).

Plan ANALYSIS_PLAN_PYTHIA.md (6de89603...) section 11 V-P8 (S0b, CPU), R's share:
 (a) legacy mode, n = 3, on the frozen OpenLLaMA pair files: R's generalised sections reproduce every Section A-C
     verdict key of results/S1_2_layermax_openllama/layermax_openllama_summary.json (2777dadf..., the embedded E and I
     verdicts), 0 mismatches; as an extra, sections A-D equal the frozen original R outputs review/{E,I}/.
 (b) common-ViT-set mode, n = 3: the keys that change against legacy are listed, each with its reason (a ViT
     excluded under one score only);
 (c) legacy mode, n = 5, on the frozen BLOOMZ pair files: every BLOOMZ layer-max trend Tr, exact one-sided p, two-sided
     p and excluded-ViT count of results/S1_2_layermax/layermax_summary.json (0dddd17a...) reproduced to 1e-12.
Reads only frozen files (read-only). The legacy key lists are R's own copy, checked against the contract's
verdict_keys.legacy_required_keys (which imports the frozen OpenLLaMA verdict_keys.py after a hash check); as the
contract (section 0) prescribes, (a) also writes the legacy verdict files (schema OL-VERDICTS v1) and compares them with
the shared compare_verdicts.py --legacy --b-sub embedded.verdicts.<impl> --sections A,B,C.
Runs only in S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN; else exit 3). Outputs under s0b_scratch/R/ only.
usage: PYT_S0B_TEST=1 launch_pyt.sh NAME 4 -- env PYTHONDONTWRITEBYTECODE=1 venv/bin/python \\
           analysis/s12_layermax_pythia/review/tests/test_vp8_r.py --out F [--parts a,b,c]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.dirname(HERE)
sys.path.insert(0, REVIEW)
PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
OL_AX = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
FROZEN8 = os.path.join(PROJ, "results", "S1_2_layermax")
LLAMA_AX = os.path.join(PROJ, "results", "S1_2_layermax_llama")
OL_SUMMARY = os.path.join(OL_AX, "layermax_openllama_summary.json")
OL_SUMMARY_SHA = "2777dadf287dfc02433eb07e84cefad0ec67381258a199b9d306b6c7d32aebaa"
BZ_SUMMARY = os.path.join(FROZEN8, "layermax_summary.json")
BZ_SUMMARY_SHA = "0dddd17a588dda2be47b19ec30b182f61f01dcfb5f791f237b39bf421d2c5613"
OL_PARTB = os.path.join(OL_AX, "partB", "partB_features_ol.json")
OL_PARTB_SHA = "e1104bd71e5eff9ced65ed402a61c187590bce5dee306c25ee2f64b18fdd7e51"
PRES = ("prh_clip+l2", "l2")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def same(a, b):
    """compare_verdicts.same of the OpenLLaMA axis (VERDICTS_SPEC.md section 0), re-typed here."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        fa, fb = float(a), float(b)
        if math.isnan(fa) or math.isnan(fb):
            return math.isnan(fa) and math.isnan(fb)
        return abs(fa - fb) <= 1e-12 * max(abs(fa), abs(fb)) + 1e-15
    return type(a) is type(b) and a == b


def ol_keys_ABC(VITS, METRICS, SUBSETS, PROTOCOLS, scores, dscores, cscores):
    """R's own copy of the OpenLLaMA verdict_keys.required_keys('A'/'B'/'C') lists (VERDICTS_SPEC.md sections 3-5)."""
    fmt = lambda x: repr(float(x))  # noqa: E731
    A, B, C = [], [], []
    for p in PRES:
        for m in METRICS:
            for q in PROTOCOLS:
                for x in scores(q):
                    A += [f"{f}|{p}|{m}|{q}|{x}" for f in ("tr", "tr_nconst", "tr_pge", "tr_ple", "tr_ptwo")]
                    if m != "mknn":
                        A.append(f"neartie|{p}|{m}|{q}|{x}")
                for x in dscores(q):
                    A += [f"{f}|{p}|{m}|{q}|{x}" for f in ("dtr", "dtr_pge", "dtr_ple", "dtr_ptwo")]
            A += [f"d3cal|{p}|{m}", f"d3sel|{p}|{m}"]
            for q in PROTOCOLS:
                for x in cscores(q):
                    for g in ("new", "pilot"):
                        B += [f"chg|{p}|{m}|{q}|{x}|{g}|{c}" for c in ("down", "up", "c2d", "d2c")]
                    B += [f"fnew|{p}|{m}|{q}|{x}|{c}" for c in ("F", "fminus", "fplus", "cp_lo", "cp_hi")]
        C.append(f"PO1|{p}|holds")
        for x in ("s_cal", "unc"):
            C += [f"PO1|{p}|zeta|{x}|{c}" for c in ("nq", "nrev", "ntie")]
            C += [f"PO2|{p}|{x}|{c}" for c in ("cat", "dist", "kedge")]
        C += [f"PO2ai|{p}|{c}" for c in ("median", "in_band", "n", "x", "cp_lo", "cp_hi", "lp_median_abs", "lp_le_0.01")]
        C += [f"PO2ai|{p}|offset|{v}" for v in VITS]
        C += [f"PO2aii|{p}|{c}" for c in ("n", "x", "need", "verdict", "cp_lo", "cp_hi")]
        for r in ("0.5", "1.0"):
            for x in ("s_cal", "unc"):
                C.append(f"PO3|{p}|{r}|{x}|desc")
        for m in METRICS:
            for q in SUBSETS:
                for x in ("s_cal", "unc"):
                    C += [f"keepid|{p}|{m}|{q}|{x}|nviol", f"signrule|{p}|{m}|{q}|{x}|nviol"]
        C += [f"PO4|{p}|s_cal|holds", f"PO4|{p}|unc|holds", f"PO4|{p}|arh|holds"]
        for x in ("M", "M_unc"):
            C += [f"PO5|{p}|{x}|pred_holds", f"PO5|{p}|{x}|cat"]
        C += [f"PO5a|{p}|{c}" for c in ("n", "x", "need", "verdict")]
        C.append(f"D3|{p}|nviol")
        C += [f"plugin|{p}|tr", f"plugin|{p}|diff", f"plugin|{p}|within", f"readO|{p}|cka_biased", f"readO|{p}|cka_debiased"]
        for t in (0.03, 0.06, 0.10, 0.15, 0.20):
            for x in ("s_cal", "unc"):
                C.append(f"curve|{p}|PO2|{x}|tau={fmt(t)}")
            C.append(f"curve|{p}|PO1|tau={fmt(t)}")
            for x in ("M", "M_unc"):
                C.append(f"curve|{p}|PO5|{x}|tau={fmt(t)}")
        for d in (0.0025, 0.005, 0.0075, 0.01):
            C.append(f"curve|{p}|material|db={fmt(d)}")
            C += [f"curve|{p}|PO2aii|db={fmt(d)}|{c}" for c in ("n", "x", "verdict")]
        for m in METRICS:
            for s in ("3b", "7b", "13b"):
                C += [f"D1|{p}|{m}|{s}", f"D2|{p}|{m}|{s}|argmax_rel", f"D2|{p}|{m}|{s}|c_minus_centry"]
            C.append(f"frac7gt3|{p}|{m}")
    return A, B, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="a,b,c")
    args = ap.parse_args()
    import pyt_review_gate as G
    if not G.test_mode():
        sys.exit(3)
    out = os.path.realpath(args.out)
    if not out.startswith(os.path.realpath(G.test_root()) + os.sep):
        print(f"--out must lie under {G.test_root()}", file=sys.stderr)
        sys.exit(3)
    import subprocess
    os.makedirs(os.path.dirname(out), exist_ok=True)
    AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")
    sys.path.insert(0, AXIS)
    import verdict_keys as VK
    import numpy as np
    import pyt_review_conf as CF
    import pyt_review_lib as RL
    import pyt_review_sections as RS
    parts = set(args.parts.split(","))
    rep = {"what": "V-P8 R share (S0b test mode)", "inputs": {}, "parts": {}}
    t0 = time.time()
    vits = list(CF.VITS)
    for p, want in ((OL_SUMMARY, OL_SUMMARY_SHA), (BZ_SUMMARY, BZ_SUMMARY_SHA), (OL_PARTB, OL_PARTB_SHA)):
        got = sha(p)
        rep["inputs"][os.path.relpath(p, PROJ)] = {"sha256": got, "equals_frozen": got == want}

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    if parts & {"a", "b"}:
        A, B, C = ol_keys_ABC(vits, RL.METRICS, RL.SUBSETS, RL.SUBSETS + ("ctm",), RS.scores, RS.dscores, RS.cscores)
        rep["legacy_keys_equal_contract"] = {sec: sorted(mine) == sorted(VK.legacy_required_keys(sec))
                                             for sec, mine in (("A", A), ("B", B), ("C", C))}
        S = json.load(open(OL_SUMMARY))["embedded"]["verdicts"]
        pb = json.load(open(OL_PARTB))
        ax = CF.OPENLLAMA
        slug = dict(ax.slug)
        llama = ("huggyllama__llama-13b", "huggyllama__llama-30b")
        for impl in ("E", "I"):
            def rd(lm, P, vit, own):
                if impl == "E":
                    if own:
                        b = os.path.join(OL_AX, "early", "pairs", P, f"{lm}__{vit}")
                    elif lm in llama:
                        b = os.path.join(LLAMA_AX, "early", "pairs", P, f"{lm}__{vit}")
                    else:
                        b = os.path.join(FROZEN8, "pairs", P, f"{lm}__{vit}")
                    if own:
                        return RL.read_pair_E(b + ".json", b + ".npz")
                    return RL.read_pair_E(b + ".json", b + ".npz", protocols=("full",), want_ctm=False, allow_missing=True)
                if own:
                    return RL.read_pair_I(os.path.join(OL_AX, "indep", P, "pairs", f"{lm}__{vit}.npz"))
                base = os.path.join(LLAMA_AX, "indep") if lm in llama else os.path.join(FROZEN8, "indep")
                return RL.read_pair_I(os.path.join(base, P, "pairs", f"{lm}__{vit}.npz"), protocols=("full",), want_ctm=False)
            ol = {P: {s: {v: rd(slug[s], P, v, True) for v in vits} for s in ax.sizes} for P in PRES}
            pool = {P: {} for P in PRES}
            for P in PRES:
                for lm in CF.POOLED11:
                    own = [s for s in ax.sizes if slug[s] == lm]
                    pool[P][lm] = ol[P][own[0]] if own else {v: rd(lm, P, v, False) for v in vits}
            log(f"{impl}: OpenLLaMA and pooled pair files read")
            PB = RS.PartB(pb, slug)
            if "a" in parts:
                T = RS.Trends(ol, vits, ax.sizes, mode="legacy")
                v = {}
                RS.section_A(v, T, PRES, RL.METRICS, RL.SUBSETS + ("ctm",))
                RS.section_B(v, ol, PB, PRES, RL.METRICS, RL.SUBSETS + ("ctm",), vits)
                panels = RS.panel_spec(list(CF.POOLED11), slug["13b"], pb["num_params"])
                D = RS.section_D(v, pool, panels, PRES, vits)
                RS.section_C(v, ol, T, PB, D, PRES, vits)
                log(f"{impl}: legacy sections A-D computed")
                ref = S[impl]
                res = {}
                for sec, keys in (("A", A), ("B", B), ("C", C)):
                    miss_R = [k for k in keys if k not in v]
                    miss_ref = [k for k in keys if k not in ref]
                    mm = [{"key": k, "R": v[k], "summary": ref[k]} for k in keys if k in v and k in ref and not same(v[k], ref[k])]
                    res[sec] = {"n_keys": len(keys), "n_compared": len([k for k in keys if k in v and k in ref]),
                                "missing_in_R": miss_R[:50], "missing_in_summary": miss_ref[:50], "n_mismatch": len(mm),
                                "mismatches": mm[:200], "pass": not miss_R and not miss_ref and not mm}
                # extra: sections A-D against the frozen original R output of this implementation
                orig = json.load(open(os.path.join(OL_AX, "review", impl, "verdicts_review.json")))["v"]
                kk = [k for k in orig if not k.startswith("x|") and k.split("|")[0] in
                      ("tr", "tr_nconst", "tr_pge", "tr_ple", "tr_ptwo", "neartie", "dtr", "dtr_pge", "dtr_ple", "dtr_ptwo",
                       "d3cal", "d3sel", "chg", "fnew", "arh", "dd", "rule7a", "size_not_reproduced", "fp6")
                      or k in C]
                mm2 = [{"key": k, "R": v.get(k, "<missing>"), "orig_R": orig[k]} for k in kk if k not in v or not same(v[k], orig[k])]
                res["vs_original_R_A_to_D"] = {"n_compared": len(kk), "n_mismatch": len(mm2), "mismatches": mm2[:100],
                                               "pass": not mm2}
                # contract section 0 procedure: legacy file + shared comparator (names only)
                lf = os.path.join(os.path.dirname(out), f"vp8a_legacy_{impl}.json")
                with open(lf, "w") as fh:
                    json.dump({"schema": "OL-VERDICTS v1", "impl": f"R-{impl} (Pythia R, legacy n=3)", "v": v}, fh,
                              indent=0, default=str)
                cr = subprocess.run([sys.executable, os.path.join(AXIS, "compare_verdicts.py"), lf, OL_SUMMARY,
                                     "--b-sub", f"embedded.verdicts.{impl}", "--legacy", "--sections", "A,B,C",
                                     "--out", lf.replace(".json", "_cmp.json")], capture_output=True, text=True)
                res["compare_verdicts_legacy"] = {"exit": cr.returncode, "stdout": cr.stdout.strip()[-500:]}
                res["pass"] = all(res[s]["pass"] for s in ("A", "B", "C")) and cr.returncode == 0
                rep["parts"][f"a|{impl}"] = res
                log(f"{impl}: V-P8(a) A {res['A']['n_mismatch']}, B {res['B']['n_mismatch']}, C {res['C']['n_mismatch']} "
                    f"mismatches; vs original R {res['vs_original_R_A_to_D']['n_mismatch']}/{len(kk)}")
            if "b" in parts:
                Tl = RS.Trends(ol, vits, ax.sizes, mode="legacy")
                Tc = RS.Trends(ol, vits, ax.sizes, mode="common")
                vl, vc = {}, {}
                RS.section_A(vl, Tl, PRES, RL.METRICS, RL.SUBSETS + ("ctm",))
                RS.section_A(vc, Tc, PRES, RL.METRICS, RL.SUBSETS + ("ctm",))
                changed = []
                for k in sorted(set(vl) & set(vc)):
                    if not same(vl[k], vc[k]):
                        if len(k.split("|")) != 5:
                            changed.append({"key": k, "legacy": vl[k], "common": vc[k], "explained": False})
                            continue
                        f, P, M, Q, X = k.split("|")
                        b = Tl.base(Q)
                        ux, ub = Tl.parts(P, M, Q, X).usable, Tl.parts(P, M, b[0], b[1]).usable
                        only = [vits[i] for i in range(len(vits)) if ux[i] != ub[i]]
                        changed.append({"key": k, "legacy": vl[k], "common": vc[k],
                                        "vits_valid_under_one_score_only": only, "explained": bool(only)})
                rep["parts"][f"b|{impl}"] = {"n_changed": len(changed), "changed": changed,
                                             "all_explained": all(c["explained"] for c in changed),
                                             "new_keys_in_common_mode": sorted(set(vc) - set(vl))[:5] + ["..."],
                                             "n_new_keys": len(set(vc) - set(vl))}
                log(f"{impl}: V-P8(b) {len(changed)} keys change, all explained {rep['parts'][f'b|{impl}']['all_explained']}")
    if "c" in parts:
        S = json.load(open(BZ_SUMMARY))["per_pre"]
        ax = CF.BLOOMZ
        subs10 = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17", "LMall_ViTlast",
                  "LMlast_ViTall")
        col = {"raw": "s_obs", "s_cal": "s_cal", "unc": "uncensored"}
        for impl in ("E", "I"):
            data = {}
            for P in PRES:
                data[P] = {}
                for s in ax.sizes:
                    lm = ax.slug[s]
                    if impl == "E":
                        b = os.path.join(FROZEN8, "pairs", P, f"{lm}__{{}}")
                        data[P][s] = {v: RL.read_pair_E(b.format(v) + ".json", b.format(v) + ".npz", protocols=subs10,
                                                        want_ctm=False, allow_missing=True) for v in vits}
                    else:
                        data[P][s] = {v: RL.read_pair_I(os.path.join(FROZEN8, "indep", P, "pairs", f"{lm}__{v}.npz"),
                                                        protocols=subs10, want_ctm=False) for v in vits}
            T = RS.Trends(data, vits, ax.sizes, mode="legacy")
            n = bad = 0
            mm = []
            for P in PRES:
                for M in RL.METRICS:
                    for Q in subs10:
                        for X in ("raw", "s_cal", "unc"):
                            ref = S[P]["trends"][M][f"{Q}|{col[X]}"]["tr_lm"]
                            tab = T.table(P, M, Q, X)
                            pc = RL.p_counts(tab["T"])
                            for name, got, want in (("Tr", pc["obs"], ref["Tr"]), ("p_one", pc["pge"], ref["p_one"]),
                                                    ("p_two", pc["ptwo"], ref["p_two"]),
                                                    ("n_vit_constant", tab["nconst"], ref["n_vit_constant"])):
                                n += 1
                                ok = got is not None and abs(float(got) - float(want)) <= 1e-12
                                if not ok:
                                    bad += 1
                                    mm.append({"key": f"{P}|{M}|{Q}|{X}|{name}", "R": got, "summary": want})
            rep["parts"][f"c|{impl}"] = {"n_values": n, "n_mismatch": bad, "mismatches": mm[:200], "pass": bad == 0,
                                         "tolerance": "absolute 1e-12"}
            log(f"{impl}: V-P8(c) {bad}/{n} BLOOMZ values differ")
    rep["pass"] = all(p.get("pass", p.get("all_explained", True)) for p in rep["parts"].values()) and \
        all(x["equals_frozen"] for x in rep["inputs"].values()) and all(rep.get("legacy_keys_equal_contract", {}).values())
    rep["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: (v.get("pass", v.get("all_explained"))) for k, v in rep["parts"].items()}), "pass", rep["pass"])
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
