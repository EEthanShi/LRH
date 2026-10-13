"""Required verdict keys of the OpenLLaMA axis (VERDICTS_SPEC.md, S0b v1). Names only, no computation.

Written at S0b by role oll-axis-extract (2026-10-09); hashed in the CODE FROZEN line.
Usage: ``required_keys("A")`` ... ``required_keys("J")``; ``required_keys("EI")`` = A+B+C+D+G (what each
aggregate writes); ``python verdict_keys.py`` prints the counts per section.
"""
from __future__ import annotations

import sys

PRES = ("prh_clip+l2", "l2")
METRICS = ("mknn", "cka_biased", "cka_debiased")
SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
PROTOCOLS = SUBSETS + ("ctm",)
SIZES = ("3b", "7b", "13b")
OL_SLUGS = {"3b": "openlm-research__open_llama_3b", "7b": "openlm-research__open_llama_7b",
            "13b": "openlm-research__open_llama_13b"}
VITS = (
    "vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k", "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae", "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b", "vit_huge_patch14_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b_ft_in12k", "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
)
POOLED11 = (
    "bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
    "bigscience__bloomz-7b1", "openlm-research__open_llama_3b", "openlm-research__open_llama_7b",
    "openlm-research__open_llama_13b", "huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b",
)
PANELS = ("11", "10", "8")
KINDS = ("arh", "dense", "logp")
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
HS = (0.0003, 0.0005, 0.001, 0.0015, 0.002)


def scores(q: str) -> tuple:
    return ("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")


def dscores(q: str) -> tuple:
    return ("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")


def cscores(q: str) -> tuple:
    return ("M", "M_unc") if q == "ctm" else ("s_cal", "unc")


def _A():
    k = []
    for p in PRES:
        for m in METRICS:
            for q in PROTOCOLS:
                for x in scores(q):
                    for f in ("tr", "tr_nconst", "tr_pge", "tr_ple", "tr_ptwo"):
                        k.append(f"{f}|{p}|{m}|{q}|{x}")
                    if m != "mknn":
                        k.append(f"neartie|{p}|{m}|{q}|{x}")
                for x in dscores(q):
                    for f in ("dtr", "dtr_pge", "dtr_ple", "dtr_ptwo"):
                        k.append(f"{f}|{p}|{m}|{q}|{x}")
            k += [f"d3cal|{p}|{m}", f"d3sel|{p}|{m}"]
    return k


def _B():
    k = []
    for p in PRES:
        for m in METRICS:
            for q in PROTOCOLS:
                for x in cscores(q):
                    for grp in ("new", "pilot"):
                        for c in ("down", "up", "c2d", "d2c"):
                            k.append(f"chg|{p}|{m}|{q}|{x}|{grp}|{c}")
                    for c in ("F", "fminus", "fplus", "cp_lo", "cp_hi"):
                        k.append(f"fnew|{p}|{m}|{q}|{x}|{c}")
    return k


def _fmt(v: float) -> str:
    return repr(float(v))


def _C():
    k = []
    for p in PRES:
        k.append(f"PO1|{p}|holds")
        for x in ("s_cal", "unc"):
            k += [f"PO1|{p}|zeta|{x}|{c}" for c in ("nq", "nrev", "ntie")]
            k += [f"PO2|{p}|{x}|{c}" for c in ("cat", "dist", "kedge")]
        k += [f"PO2ai|{p}|{c}" for c in ("median", "in_band", "n", "x", "cp_lo", "cp_hi", "lp_median_abs",
                                         "lp_le_0.01")]
        k += [f"PO2ai|{p}|offset|{v}" for v in VITS]
        k += [f"PO2aii|{p}|{c}" for c in ("n", "x", "need", "verdict", "cp_lo", "cp_hi")]
        for r in ("0.5", "1.0"):
            for x in ("s_cal", "unc"):
                k.append(f"PO3|{p}|{r}|{x}|desc")
        for m in METRICS:
            for q in SUBSETS:
                for x in ("s_cal", "unc"):
                    k.append(f"keepid|{p}|{m}|{q}|{x}|nviol")
                    k.append(f"signrule|{p}|{m}|{q}|{x}|nviol")
        k += [f"PO4|{p}|s_cal|holds", f"PO4|{p}|unc|holds", f"PO4|{p}|arh|holds"]
        for x in ("M", "M_unc"):
            k += [f"PO5|{p}|{x}|pred_holds", f"PO5|{p}|{x}|cat"]
        k += [f"PO5a|{p}|{c}" for c in ("n", "x", "need", "verdict")]
        k.append(f"D3|{p}|nviol")
        k += [f"plugin|{p}|tr", f"plugin|{p}|diff", f"plugin|{p}|within"]
        k += [f"readO|{p}|cka_biased", f"readO|{p}|cka_debiased"]
        for t in TAUS:
            for x in ("s_cal", "unc"):
                k.append(f"curve|{p}|PO2|{x}|tau={_fmt(t)}")
            k.append(f"curve|{p}|PO1|tau={_fmt(t)}")
            for x in ("M", "M_unc"):
                k.append(f"curve|{p}|PO5|{x}|tau={_fmt(t)}")
        for d in DBS:
            k.append(f"curve|{p}|material|db={_fmt(d)}")
            k += [f"curve|{p}|PO2aii|db={_fmt(d)}|{c}" for c in ("n", "x", "verdict")]
        for m in METRICS:
            for s in SIZES:
                k += [f"D1|{p}|{m}|{s}", f"D2|{p}|{m}|{s}|argmax_rel", f"D2|{p}|{m}|{s}|c_minus_centry"]
            k.append(f"frac7gt3|{p}|{m}")
    return k


def _D():
    k = []
    for p in PRES:
        for pan in PANELS:
            for m in METRICS:
                for x in ("raw", "s_cal", "unc"):
                    for kd in KINDS:
                        k.append(f"arh|{pan}|{p}|{m}|{x}|{kd}|r")
                        if pan == "11":
                            k += [f"arh|11|{p}|{m}|{x}|{kd}|ci_lo", f"arh|11|{p}|{m}|{x}|{kd}|ci_hi"]
                    if pan == "11":
                        k.append(f"arh|11|{p}|{m}|{x}|arh|pge")
            for m in ("cka_biased", "cka_debiased"):
                k.append(f"dd|{pan}|{p}|{m}")
        for m in ("cka_biased", "cka_debiased"):
            k += [f"dd|11|{p}|{m}|ci_lo", f"dd|11|{p}|{m}|ci_hi", f"dd|11|{p}|{m}|pge"]
        k += [f"rule7a|{p}", f"size_not_reproduced|{p}"]
        k += [f"fp6|{p}|{c}" for c in ("raw_cat", "gap_ci_lo", "gap_ci_hi", "kedge", "cat")]
    return k


def _F():
    k = []
    for p in PRES:
        for x in ("s_cal", "unc"):
            for a in ("3b", "7b"):
                for v in VITS:
                    k += [f"PO2b|{p}|{x}|dec|{a}|{v}", f"PO2b|{p}|{x}|obs|{a}|{v}", f"PO2b|{p}|{x}|ok|{a}|{v}"]
            k += [f"PO2b|{p}|{x}|{c}" for c in ("n_decided", "x_correct", "n_undecided", "cp_lo", "cp_hi",
                                               "verdict")]
        for h in HS:
            k += [f"PO2b|{p}|s_cal|h={_fmt(h)}|n_decided", f"PO2b|{p}|s_cal|h={_fmt(h)}|x_correct"]
    return k


def _J():
    k = ["J|label", "J|provenance"]
    for x in ("s_cal", "unc"):
        k += [f"J|{x}|{c}" for c in ("sO", "Jmin", "pmin", "pmin_product", "Jmean", "pmean", "nO", "cond_size",
                                     "dTr|B", "dTr|L", "dTr|O")]
    return k


def _G():
    # section G: provenance checks each aggregate writes (plan section 3 sources, V-O4b source rule)
    k = ["G|llama_pairs_equal_merged_summary", "G|rerun_equals_frozen_3b7b", "G|n_pairs_read"]
    return k


_SECTIONS = {"A": _A, "B": _B, "C": _C, "D": _D, "F": _F, "J": _J, "G": _G}


def required_keys(section: str) -> list[str]:
    if section == "EI":
        return [k for s in "ABCDG" for k in _SECTIONS[s]()]
    if section == "ALL":
        return [k for s in "ABCDGFJ" for k in _SECTIONS[s]()]
    return _SECTIONS[section]()


if __name__ == "__main__":
    tot = 0
    for s in "ABCDGFJ":
        n = len(required_keys(s))
        tot += n
        print(s, n)
    print("total", tot)
    ks = required_keys("ALL")
    assert len(ks) == len(set(ks)), "duplicate keys"
    sys.exit(0)
