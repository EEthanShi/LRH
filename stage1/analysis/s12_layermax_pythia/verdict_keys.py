"""Required verdict keys of the Pythia axis (VERDICTS_SPEC_PYT.md, S0b v1). Names only, no computation.  # PYT-AXIS

PYT-AXIS copy of analysis/s12_layermax_openllama/verdict_keys.py (sha256 95a152c8...), written at S0b by role  # PYT-AXIS
pyt-code-infra (2026-10-10) for analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md v1.0 (sha256 6de89603...);  # PYT-AXIS
hashed in the CODE FROZEN line. Every changed or added line carries the marker PYT-AXIS; diff: verdict_keys.py.diff.  # PYT-AXIS
Usage: ``required_keys("A")`` ... ``required_keys("K")``; ``required_keys("EI")`` = A+B+C+H+R+D+G (what each  # PYT-AXIS
aggregate writes); ``required_keys("FK")`` = F+K (what forecast_pyt.py writes per implementation);  # PYT-AXIS
``legacy_required_keys(section)`` = the frozen OpenLLaMA key list (V-P8 legacy mode, n = 3);  # PYT-AXIS
``python verdict_keys.py`` prints the counts per section.  # PYT-AXIS
"""
from __future__ import annotations

import hashlib  # PYT-AXIS
import importlib.util  # PYT-AXIS
import itertools  # PYT-AXIS
import os  # PYT-AXIS
import sys

PRES = ("prh_clip+l2", "l2")
METRICS = ("mknn", "cka_biased", "cka_debiased")
SUBSETS = ("full", "fixed_r0.5", "fixed_r1.0", "grid2", "grid3", "grid5", "grid9", "grid17",
           "LMall_ViTlast", "LMlast_ViTall", "noLMemb", "LMdeep", "LMemb_only")
PROTOCOLS = SUBSETS + ("ctm",)
SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")  # PYT-AXIS: size ranks 1..7 (num_params order)
PYT_SLUGS = {s: f"EleutherAI__pythia-{s}" for s in SIZES}  # PYT-AXIS
SIZE_PAIRS = tuple(f"{a}|{b}" for a, b in itertools.combinations(SIZES, 2))  # PYT-AXIS: 21, smaller|larger
ADJ_PAIRS = tuple(f"{SIZES[i]}|{SIZES[i + 1]}" for i in range(len(SIZES) - 1))  # PYT-AXIS: 6 adjacent steps
D4_STEPS = ("410m|1b", "1b|1.4b", "2.8b|6.9b")  # PYT-AXIS: plan section 6 D4 contrasts
VITS = (
    "vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k", "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae", "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b", "vit_huge_patch14_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b_ft_in12k", "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
)
FAMILIES4 = ("in21k", "MAE", "DINOv2", "CLIP")  # PYT-AXIS: the 4 ViT pretraining families of plan section 1
VIT_FAMILY4 = {v: ("in21k" if "augreg_in21k" in v else "MAE" if v.endswith(".mae") else  # PYT-AXIS
                   "DINOv2" if "dinov2" in v else "CLIP") for v in VITS}  # PYT-AXIS
LM_FAMILIES = {  # PYT-AXIS: cross-family descriptive table and the 18-LM panel (plan sections 1, 5.5(g))
    "BLOOMZ": ("bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7",  # PYT-AXIS
               "bigscience__bloomz-3b", "bigscience__bloomz-7b1"),  # PYT-AXIS
    "LLaMA": ("huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"),  # PYT-AXIS
    "OpenLLaMA": ("openlm-research__open_llama_3b", "openlm-research__open_llama_7b",  # PYT-AXIS
                  "openlm-research__open_llama_13b"),  # PYT-AXIS
    "Pythia": tuple(PYT_SLUGS[s] for s in SIZES),  # PYT-AXIS
}  # PYT-AXIS
PANEL18 = tuple(s for f in ("BLOOMZ", "LLaMA", "OpenLLaMA", "Pythia") for s in LM_FAMILIES[f])  # PYT-AXIS
XS = ("s_cal", "unc")  # PYT-AXIS: primary (s_cal) and uncensored (secondary) versions of every statistic
HYPS = ("H2", "H3", "H4")  # PYT-AXIS: secondary family behind the H1 gate (plan section 5.4)
TOST_COMBOS = tuple((m, q) for m in METRICS for q in ("full", "ctm"))  # PYT-AXIS: plan 5.4 TOST key family
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
HS = (0.0004, 0.0006, 0.0008, 0.001, 0.0015)  # PYT-AXIS: plan 5.6 h curve around f(PR*_i)
MARGINS = (0.05, 0.10, 0.15)  # PYT-AXIS: plan 5.6 TOST and magnitude margins
CATEGORIES = ("A", "A-att", "A-prime", "B", "C", "D", "E", "F")  # PYT-AXIS: plan section 7 (A-prime = A')
OL_VERDICT_KEYS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # PYT-AXIS
                               "s12_layermax_openllama", "verdict_keys.py")  # PYT-AXIS: V-P8 legacy key list
OL_VERDICT_KEYS_SHA = "95a152c8fcecc51ac03b79a0905d7aaa2baf754e2a12a40295d5d36749b22f8f"  # PYT-AXIS (plan section 3)


def scores(q: str) -> tuple:
    return ("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")


def dscores(q: str) -> tuple:
    return ("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")


def cscores(q: str) -> tuple:
    return ("M", "M_unc") if q == "ctm" else ("s_cal", "unc")


def _fmt(v: float) -> str:  # PYT-AXIS
    return repr(float(v))  # PYT-AXIS


def comparisons() -> list[tuple[str, str, str]]:  # PYT-AXIS: the 357 size comparisons (j < j', v)
    return [(a, b, v) for a, b in itertools.combinations(SIZES, 2) for v in VITS]  # PYT-AXIS


def _A():
    k = []
    for p in PRES:
        for m in METRICS:
            for q in PROTOCOLS:
                for x in scores(q):
                    for f in ("tr", "tr_nconst", "tr_pge", "tr_ple", "tr_ptwo", "tr0"):  # PYT-AXIS: + tr0 (rho_v := 0)
                        k.append(f"{f}|{p}|{m}|{q}|{x}")
                    if m != "mknn":
                        k.append(f"neartie|{p}|{m}|{q}|{x}")
                for x in dscores(q):
                    for f in ("dtr", "dtr_nv", "dtr_pge", "dtr_ple", "dtr_ptwo", "dtr0"):  # PYT-AXIS: common ViT set
                        k.append(f"{f}|{p}|{m}|{q}|{x}")
            k += [f"d3cal|{p}|{m}", f"d3sel|{p}|{m}"]
    return k


def _B():
    k = []
    for p in PRES:
        for m in METRICS:
            for q in PROTOCOLS:
                for x in cscores(q):
                    for c in ("down", "up", "c2d", "d2c", "c2t", "t2c", "d2t", "t2d"):  # PYT-AXIS: all 357
                        k.append(f"chg|{p}|{m}|{q}|{x}|all|{c}")  # PYT-AXIS
                    for c in ("n", "fminus", "fplus", "F", "fm_cp_lo", "fm_cp_hi", "fp_cp_lo", "fp_cp_hi"):  # PYT-AXIS
                        k.append(f"fcnt|{p}|{m}|{q}|{x}|{c}")  # PYT-AXIS: plan 5.3
    return k


def _C():  # PYT-AXIS: plan sections 6, 8, 9 (premise rules, implementation checks, attribution diagnostics)
    k = []
    for p in PRES:
        k.append(f"PP1|{p}|holds")  # PYT-AXIS
        for x in XS:  # PYT-AXIS
            k += [f"PP1|{p}|{x}|{c}" for c in ("nq", "nrev", "ntie", "holds")]  # PYT-AXIS
            for s in SIZES:  # PYT-AXIS: P-P2a(i) per model
                k += [f"PP2ai|{p}|{x}|{c}|{s}" for c in ("median", "lo", "hi", "in_band", "x", "cp_lo", "cp_hi")]  # PYT-AXIS
                k += [f"PP2ai|{p}|{x}|offset|{s}|{v}" for v in VITS]  # PYT-AXIS
            k += [f"PP2ai|{p}|{x}|{c}" for c in ("n_models_in_band", "holds", "failing")]  # PYT-AXIS
            k += [f"PP2aii|{p}|{x}|{c}" for c in ("n", "x", "need", "need67", "verdict", "verdict67",  # PYT-AXIS
                                                  "cp_lo", "cp_hi")]  # PYT-AXIS
            k.append(f"PP2a|{p}|{x}|verdict")  # PYT-AXIS
            k += [f"PP5a|{p}|{x}|{c}" for c in ("n", "x", "need", "verdict", "cp_lo", "cp_hi")]  # PYT-AXIS
            for r in ("0.5", "1.0"):  # PYT-AXIS
                k.append(f"PP3|{p}|{r}|{x}|desc")  # PYT-AXIS
        k += [f"PP2ai|{p}|s_cal|flat|in_band|{s}" for s in SIZES]  # PYT-AXIS: flat-band sensitivity
        k += [f"PP2ai|{p}|s_cal|flat|{c}" for c in ("n_models_in_band", "holds")]  # PYT-AXIS
        for m in METRICS:
            for q in SUBSETS:
                for x in XS:  # PYT-AXIS
                    k.append(f"keepid|{p}|{m}|{q}|{x}|nviol")
                    k.append(f"signrule|{p}|{m}|{q}|{x}|nviol")
                    k += [f"signvit|{p}|{m}|{q}|{x}|{c}" for c in ("nmono_up", "nviol_up", "nmono_down",  # PYT-AXIS
                                                                   "nviol_down", "nexcl_tie")]  # PYT-AXIS
        k += [f"D3|{p}|nviol", f"D3|{p}|nexcl_tie"]  # PYT-AXIS: + censoring-tie exclusions
        k += [f"plugin|{p}|tr", f"plugin|{p}|diff", f"plugin|{p}|within", f"plugin|{p}|nv"]  # PYT-AXIS: + nv
        k += [f"readO|{p}|cka_biased", f"readO|{p}|cka_debiased"]
        for m in METRICS:
            for s in SIZES:
                k += [f"D1|{p}|{m}|{s}", f"D2|{p}|{m}|{s}|argmax_rel", f"D2|{p}|{m}|{s}|c_minus_centry"]
        for qty in ("PRstar", "Bbar", "calpha", "T"):  # PYT-AXIS: D4
            k += [f"D4|{p}|{qty}|{a}" for a in ("logparams", "width", "depth")]  # PYT-AXIS
            k += [f"D4|{p}|{qty}|step|{st}" for st in D4_STEPS]  # PYT-AXIS
        for st in ADJ_PAIRS:  # PYT-AXIS: D5
            k += [f"D5|{p}|{st}|raw_gain", f"D5|{p}|{st}|thr_gain"]  # PYT-AXIS
        k += [f"D5|{p}|g", f"D5|{p}|h"]  # PYT-AXIS
        for m in ("mknn", "cka_debiased"):  # PYT-AXIS: D6
            k += [f"D6|{p}|{m}|{s}|c_minus_base" for s in SIZES]  # PYT-AXIS
            k += [f"D6|{p}|{m}|spearman_L", f"D6|{p}|{m}|step|410m|1b"]  # PYT-AXIS
    return k


def _H():  # PYT-AXIS: plan section 5.4 (confirmatory family; secondary versions descriptive)
    k = []  # PYT-AXIS
    for p in PRES:  # PYT-AXIS
        for x in XS:  # PYT-AXIS
            h = f"H|{p}|{x}"  # PYT-AXIS
            k += [f"{h}|H1|{c}" for c in ("member", "s", "D", "nV", "R", "C", "p_perm", "p_mag", "p",  # PYT-AXIS
                                          "confirmed", "ci_lo", "ci_hi", "kedge")]  # PYT-AXIS
            k += [f"{h}|H1|p_mag|m={_fmt(mg)}" for mg in MARGINS]  # PYT-AXIS
            k += [f"{h}|H2|{c}" for c in ("branch", "s_late", "D", "nV", "p_perm", "p_mag", "p_lo", "p_hi", "p",  # PYT-AXIS
                                          "degenerate", "ranks_identical")]  # PYT-AXIS
            k += [f"{h}|H2|p_mag|m={_fmt(mg)}" for mg in MARGINS]  # PYT-AXIS
            k += [f"{h}|H2|p_tost|m={_fmt(mg)}" for mg in MARGINS]  # PYT-AXIS
            for hy in ("H3", "H4"):  # PYT-AXIS
                k += [f"{h}|{hy}|{c}" for c in ("D", "nV", "p_lo", "p_hi", "p", "degenerate", "ranks_identical")]  # PYT-AXIS
                k += [f"{h}|{hy}|p_tost|m={_fmt(mg)}" for mg in MARGINS]  # PYT-AXIS
            k += [f"{h}|gate", f"{h}|holm_order"]  # PYT-AXIS
            for hy in HYPS:  # PYT-AXIS
                k += [f"{h}|{hy}|{c}" for c in ("tested", "p_holm", "holm_reject", "confirmed")]  # PYT-AXIS
            for m, q in TOST_COMBOS:  # PYT-AXIS: descriptive TOST for every (metric, protocol)
                k += [f"TOST|{p}|{x}|{m}|{q}|{c}" for c in ("D", "nV", "p_lo", "p_hi", "p", "degenerate",  # PYT-AXIS
                                                            "ranks_identical")]  # PYT-AXIS
    return k  # PYT-AXIS


def _R():  # PYT-AXIS: plan section 5.5 robustness (descriptive; wording modifiers only)
    k = []  # PYT-AXIS
    for p in PRES:  # PYT-AXIS
        k.append(f"R|{p}|n_scal_zero")  # PYT-AXIS: M4 input (119 primary pairs, cka_biased full)
        k += [f"R|{p}|median_scal|{s}" for s in SIZES]  # PYT-AXIS
        for x in XS:  # PYT-AXIS
            r = f"R|{p}|{x}"  # PYT-AXIS
            for s in SIZES:  # PYT-AXIS: (a) leave-one-size-out
                k += [f"{r}|loo|{s}|{c}" for c in ("D1", "p_perm", "p_mag", "p", "dtr_mknn", "dtr_dcka")]  # PYT-AXIS
            k += [f"{r}|fragile", f"{r}|fragile_sizes"]  # PYT-AXIS
            k += [f"{r}|five|{c}" for c in ("D1", "p_perm", "carried", "carried_D")]  # PYT-AXIS: (b)
            k += [f"{r}|grid5|{c}" for c in ("D1", "dtr_mknn", "dtr_dcka")]  # PYT-AXIS: (c)
            for f in FAMILIES4:  # PYT-AXIS: (d)
                k += [f"{r}|lofo|{f}|{c}" for c in ("D1", "dtr_mknn", "dtr_dcka")]  # PYT-AXIS
            k.append(f"{r}|rho0|D1")  # PYT-AXIS: (e)
    return k  # PYT-AXIS


def _D():  # PYT-AXIS: cross-family table and the 18-LM panel (descriptive; plan sections 1, 5.5(g))
    k = []
    for p in PRES:
        for m in METRICS:  # PYT-AXIS
            for x in ("raw", "s_cal", "unc"):  # PYT-AXIS
                k += [f"panel18|{p}|{m}|{x}|{c}" for c in ("r", "ci_lo", "ci_hi")]  # PYT-AXIS
                k += [f"famtr|{p}|{m}|{x}|{f}" for f in LM_FAMILIES]  # PYT-AXIS
            for x in XS:  # PYT-AXIS
                k += [f"famdtr|{p}|{m}|{x}|{f}" for f in LM_FAMILIES]  # PYT-AXIS
    return k


def _F():  # PYT-AXIS: P-P2b (forecast_pyt.py), plan section 9 B6
    k = []
    for p in PRES:
        for x in XS:  # PYT-AXIS
            for a, b, v in comparisons():  # PYT-AXIS
                k += [f"PP2b|{p}|{x}|{c}|{a}|{b}|{v}" for c in ("dec", "obs", "ok")]  # PYT-AXIS
            k += [f"PP2b|{p}|{x}|{c}" for c in ("n_decided", "x_correct", "n_wrong", "n_undecided", "n_tie_fc",  # PYT-AXIS
                                                "n_flip_fc_rawconc", "cp_lo", "cp_hi", "verdict",  # PYT-AXIS
                                                "wrong_all_extrap", "wrong_sizes")]  # PYT-AXIS
        for h in HS:
            k += [f"PP2b|{p}|s_cal|h={_fmt(h)}|n_decided", f"PP2b|{p}|s_cal|h={_fmt(h)}|x_correct"]  # PYT-AXIS
        k += [f"PP2b|{p}|s_cal|flat|{c}" for c in ("n_decided", "x_correct", "verdict")]  # PYT-AXIS
    return k


def _K():  # PYT-AXIS: plan section 7 category, modifiers and readings (forecast_pyt.py, after F)
    k = []  # PYT-AXIS
    for p in PRES:  # PYT-AXIS
        for x in XS:  # PYT-AXIS
            c = f"K|{p}|{x}"  # PYT-AXIS
            k += [f"{c}|{f}" for f in ("category", "premise", "F_reason", "R", "C", "nV", "M1", "M1_sizes", "M2",  # PYT-AXIS
                                       "M2_x", "M2_H3", "M2_H4", "M3", "M4", "M4_sizes", "M4_n_zero",  # PYT-AXIS
                                       "M5_extrap_sizes", "M5_B", "M5_A", "M5_A_sizes", "H2_reading",  # PYT-AXIS
                                       "H2_same_as_H1", "H3_reading", "H4_reading", "H2_counted", "H3_counted",  # PYT-AXIS
                                       "H4_counted", "kedge", "kedge_which", "h_consistent")]  # PYT-AXIS
            k += [f"{c}|curve|tau={_fmt(t)}" for t in TAUS]  # PYT-AXIS
            k += [f"{c}|curve|db={_fmt(d)}" for d in DBS]  # PYT-AXIS
    return k  # PYT-AXIS


def _G():  # PYT-AXIS
    # section G: provenance checks each aggregate writes  # PYT-AXIS
    k = ["G|n_pairs_read", "G|n_known_records", "G|l2_run", "G|partB_sha256", "G|clip_equal_partB"]  # PYT-AXIS
    return k


_SECTIONS = {"A": _A, "B": _B, "C": _C, "H": _H, "R": _R, "D": _D, "G": _G, "F": _F, "K": _K}  # PYT-AXIS
EI_SECTIONS = "ABCHRDG"  # PYT-AXIS: aggregate files (E early/aggregate_pyt.py, I indep/lmx_pyt_aggregate.py)
FK_SECTIONS = "FK"  # PYT-AXIS: forecast_pyt.py --impl E|I


def required_keys(section: str) -> list[str]:
    if section == "EI":
        return [k for s in EI_SECTIONS for k in _SECTIONS[s]()]  # PYT-AXIS
    if section == "FK":  # PYT-AXIS
        return [k for s in FK_SECTIONS for k in _SECTIONS[s]()]  # PYT-AXIS
    if section == "ALL":
        return [k for s in EI_SECTIONS + FK_SECTIONS for k in _SECTIONS[s]()]  # PYT-AXIS
    return _SECTIONS[section]()


def _sha(path: str) -> str:  # PYT-AXIS
    h = hashlib.sha256()  # PYT-AXIS
    with open(path, "rb") as f:  # PYT-AXIS
        h.update(f.read())  # PYT-AXIS
    return h.hexdigest()  # PYT-AXIS


def legacy_module():  # PYT-AXIS: the frozen OpenLLaMA verdict_keys.py (read-only import, hash-checked), V-P8 only
    got = _sha(OL_VERDICT_KEYS)  # PYT-AXIS
    if got != OL_VERDICT_KEYS_SHA:  # PYT-AXIS
        raise SystemExit(f"frozen OpenLLaMA verdict_keys.py changed: {got}")  # PYT-AXIS
    spec = importlib.util.spec_from_file_location("ol_verdict_keys_frozen", OL_VERDICT_KEYS)  # PYT-AXIS
    mod = importlib.util.module_from_spec(spec)  # PYT-AXIS
    spec.loader.exec_module(mod)  # PYT-AXIS
    return mod  # PYT-AXIS


def legacy_required_keys(section: str) -> list[str]:  # PYT-AXIS: V-P8(a) compares sections A, B, C (n = 3, legacy)
    return legacy_module().required_keys(section)  # PYT-AXIS


if __name__ == "__main__":
    tot = 0
    for s in EI_SECTIONS + FK_SECTIONS:  # PYT-AXIS
        n = len(required_keys(s))
        tot += n
        print(s, n)
    print("total", tot)
    ks = required_keys("ALL")
    assert len(ks) == len(set(ks)), "duplicate keys"
    assert len(comparisons()) == 357 and len(SIZE_PAIRS) == 21  # PYT-AXIS
    assert sorted(FAMILIES4) == sorted(set(VIT_FAMILY4.values()))  # PYT-AXIS
    assert [sum(1 for v in VITS if VIT_FAMILY4[v] == f) for f in FAMILIES4] == [4, 3, 4, 6]  # PYT-AXIS: plan section 1
    sys.exit(0)
