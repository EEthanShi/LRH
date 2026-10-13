"""OpenLLaMA axis, sub-role R: S0b reproduction check of review/ol_axis_review.py (TEST ONLY; NEW FILE, OL-AXIS).

Runs only in S0b test mode (guard.test_mode()). Reads R's verdict files of a fixture (make_test_fixture.py) and the
frozen LLaMA-axis results (read-only), and checks:
  variant llama3 (3b/7b/13b <- llama-7b/13b/30b): every OpenLLaMA-axis trend Tr / p / dTr of R-E and R-I equals the
      LLaMA-axis value: CKA vs the E summary (trends_L, delta_L) and the I summary (blocks), mKNN vs the exact-rational
      rows of the merged summary (cross_validation.part4_mknn_exact_rational) and vs E/I floats; J dTr|B, dTr|L vs
      the plan section 0 values; G flags true.
  variant ol2 (3b/7b <- frozen OpenLLaMA 3b/7b, 13b <- llama-13b): panel-10 r (arh, dense, logp), dd and panel-8 r
      equal the LLaMA-axis pooled values of E (pooled, readings) and I (readings).
Tolerance 1e-12 relative + 1e-15 (compare_verdicts.same). Writes <fixture>/out/reproduce_<variant>.json.
usage: OL_S0B_TEST=1 launch_ol.sh R_repro 1 -- venv/bin/python .../review/test_reproduce_llama.py --variant V --root DIR
"""
import argparse
import json
import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
COLS = {"raw": "s_obs", "s_cal": "s_cal", "unc": "uncensored"}
I_SUB = {"LMall_ViTlast": "lmall_x_vitlast", "LMlast_ViTall": "lmlast_x_vitall"}
I_MET = {"mknn": "mknn", "cka_biased": "cka_b", "cka_debiased": "cka_d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("llama3", "ol2"), required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    sys.path.insert(0, AXIS)
    import guard
    import verdict_keys as VK
    from compare_verdicts import same
    if not guard.test_mode():
        sys.exit(3)
    root = os.path.realpath(os.path.expanduser(args.root))
    S = json.load(open(os.path.join(PROJ, "results", "S1_2_layermax_llama", "layermax_llama_summary.json")))
    E, I = S["embedded"]["early_summary"], S["embedded"]["indep_summary"]
    exact = {(r["pre"], r["sub"], r["field"]): r for r in S["cross_validation"]["part4_mknn_exact_rational"]["rows"]}
    R = {impl: json.load(open(os.path.join(root, "out", "review", impl, "verdicts_review.json")))["v"] for impl in ("E", "I")}
    rows, n_ok, n_bad = [], 0, 0

    def chk(name, mine, ref, kind):
        nonlocal n_ok, n_bad
        ok = same(mine, ref)
        n_ok += ok
        n_bad += not ok
        if not ok or len(rows) < 40:
            rows.append({"what": name, "R": mine, "ref": ref, "ref_kind": kind, "ok": ok})

    if args.variant == "llama3":
        for impl in ("E", "I"):
            v = R[impl]
            for P in ("prh_clip+l2", "l2"):
                pe = E["per_pre"][P]
                pi = I["per_pre"][P]["blocks"]
                for M in VK.METRICS:
                    for Q in VK.SUBSETS:
                        for X, col in COLS.items():
                            k = f"{M}|{Q}|{col}"
                            t = pe["trends_L"][k]
                            if M == "mknn":
                                ex = exact.get((P, Q, col))
                                if ex is not None:
                                    chk(f"R-{impl} tr|{P}|{M}|{Q}|{X} vs exact", v[f"tr|{P}|{M}|{Q}|{X}"], ex["Tr_exact"], "exact-rational")
                            else:
                                chk(f"R-{impl} tr|{P}|{M}|{Q}|{X} vs E", v[f"tr|{P}|{M}|{Q}|{X}"], t["Tr"], "E float")
                            chk(f"R-{impl} tr_pge|{P}|{M}|{Q}|{X} vs E p_one", v[f"tr_pge|{P}|{M}|{Q}|{X}"], t["p_one"], "E")
                            chk(f"R-{impl} tr_ptwo|{P}|{M}|{Q}|{X} vs E p_two", v[f"tr_ptwo|{P}|{M}|{Q}|{X}"], t["p_two"], "E")
                            chk(f"R-{impl} tr_nconst|{P}|{M}|{Q}|{X} vs E", v[f"tr_nconst|{P}|{M}|{Q}|{X}"], t["n_vit_constant"], "E")
                            ib = pi[I_MET[M]].get(I_SUB.get(Q, Q), {}).get("llama_axis", {}).get(col)
                            if ib is not None and M != "mknn":
                                chk(f"R-{impl} tr|{P}|{M}|{Q}|{X} vs I", v[f"tr|{P}|{M}|{Q}|{X}"], ib["Tr"], "I float")
                            if X != "raw":
                                d = pe["delta_L"][k]
                                if M != "mknn":
                                    chk(f"R-{impl} dtr|{P}|{M}|{Q}|{X} vs E", v[f"dtr|{P}|{M}|{Q}|{X}"], d["diff"], "E")
                                chk(f"R-{impl} dtr_pge vs E p_upper {P}|{M}|{Q}|{X}", v[f"dtr_pge|{P}|{M}|{Q}|{X}"], d["p_upper"], "E")
                                chk(f"R-{impl} dtr_ple vs E p_lower {P}|{M}|{Q}|{X}", v[f"dtr_ple|{P}|{M}|{Q}|{X}"], d["p_lower"], "E")
                                chk(f"R-{impl} dtr_ptwo vs E p_two {P}|{M}|{Q}|{X}", v[f"dtr_ptwo|{P}|{M}|{Q}|{X}"], d["p_two"], "E")
                    for X in ("M", "M_unc"):
                        t = pe["trends_L"][f"{M}|ctm|{X}"]
                        chk(f"R-{impl} tr|{P}|{M}|ctm|{X} vs E", v[f"tr|{P}|{M}|ctm|{X}"], t["Tr"], "E float")
                        d = pe["delta_L"][f"{M}|ctm|{X}"]
                        chk(f"R-{impl} dtr|{P}|{M}|ctm|{X} vs E", v[f"dtr|{P}|{M}|ctm|{X}"], d["diff"], "E")
                    # mKNN derived dTr against the exact-rational Tr differences
                    for Q in VK.SUBSETS:
                        for X, col in (("s_cal", "s_cal"), ("unc", "uncensored")):
                            a, b = exact.get((P, Q, col)), exact.get((P, Q, "s_obs"))
                            if a and b and M == "mknn":
                                chk(f"R-{impl} dtr|{P}|mknn|{Q}|{X} vs exact", v[f"dtr|{P}|mknn|{Q}|{X}"],
                                    a["Tr_exact"] - b["Tr_exact"], "exact-rational")
            # J family values known from the plan (section 0 table; rounded to 3 decimals there)
            for X, (b_, l_) in (("s_cal", (-0.912, -1.529)), ("unc", (-1.182, -1.676))):
                chk(f"R-{impl} J|{X}|dTr|B ~ plan s0 (3 dp)", round(v[f"J|{X}|dTr|B"], 3), b_, "plan s0")
                chk(f"R-{impl} J|{X}|dTr|L ~ plan s0 (3 dp)", round(v[f"J|{X}|dTr|L"], 3), l_, "plan s0")
                chk(f"R-{impl} J|{X}|dTr|O == dTr|L (stand-in)", v[f"J|{X}|dTr|O"], v[f"J|{X}|dTr|L"], "self")
            for g in ("G|llama_pairs_equal_merged_summary", "G|rerun_equals_frozen_3b7b"):
                chk(f"R-{impl} {g}", v[g], True, "expected")
            chk(f"R-{impl} G|n_pairs_read", v["G|n_pairs_read"], 374, "spec")
    else:
        for impl in ("E", "I"):
            v = R[impl]
            for P in ("prh_clip+l2", "l2"):
                pe = E["per_pre"][P]
                for M in VK.METRICS:
                    for X, col in COLS.items():
                        blk = pe["pooled"][f"{M}|full|{col}"]
                        chk(f"R-{impl} arh|10|{P}|{M}|{X}|arh|r vs E", v[f"arh|10|{P}|{M}|{X}|arh|r"], blk["arh"]["r"], "E")
                        chk(f"R-{impl} arh|10|{P}|{M}|{X}|dense|r vs E", v[f"arh|10|{P}|{M}|{X}|dense|r"], blk["dense"], "E")
                        chk(f"R-{impl} arh|10|{P}|{M}|{X}|logp|r vs E", v[f"arh|10|{P}|{M}|{X}|logp|r"], blk["logp"], "E")
                        chk(f"R-{impl} arh|8|{P}|{M}|{X}|arh|r vs E r_8LM", v[f"arh|8|{P}|{M}|{X}|arh|r"], blk["r_8LM"], "E")
                for M, Mi in (("cka_biased", "cka_b"), ("cka_debiased", "cka_d")):
                    rd = pe["readings"][M]
                    chk(f"R-{impl} dd|10|{P}|{M} vs E", v[f"dd|10|{P}|{M}"], rd["delta_diff"], "E")
                    chk(f"R-{impl} dd|8|{P}|{M} vs E", v[f"dd|8|{P}|{M}"], rd["delta_diff_8LM"], "E")
                    ri = I["per_pre"][P]["readings"][Mi]
                    chk(f"R-{impl} dd|10|{P}|{M} vs I", v[f"dd|10|{P}|{M}"], ri["arh_pooled10"]["dDelta"], "I")
                    chk(f"R-{impl} r_after cka 10 vs I", v[f"arh|10|{P}|{M}|s_cal|arh|r"], ri["inputs"]["r_after_cka"], "I")
                    chk(f"R-{impl} r_before mknn 10 vs I", v[f"arh|10|{P}|mknn|raw|arh|r"], ri["inputs"]["r_before_mknn"], "I")
    out = {"variant": args.variant, "n_ok": n_ok, "n_bad": n_bad, "rows_bad_or_first40": rows}
    od = os.path.join(root, "out")
    with open(os.path.join(od, f"reproduce_{args.variant}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"variant": args.variant, "n_ok": n_ok, "n_bad": n_bad}))
    for r in rows:
        if not r["ok"]:
            print("MISMATCH", r)
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
