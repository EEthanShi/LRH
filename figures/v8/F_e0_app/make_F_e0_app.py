"""F_e0_app v8.1 (arxiv_v4 appendix figure for experiment E0, label fig:app-e0; role v8 maker, then v8 critic, 2026-10-10).

CRITIC v8.1 (2026-10-10, after the maker's v8): panel (c) now draws the exploratory X3b follow-up instead of the five
X3 features of DINOv2-B x MPNet. X3b (results/E0_sentence/x3b; readings fixed 21:18:49Z before any score, independent
reimplementation AGREE, review 22:01:39Z, i.e. after the maker had built (c)) extends X3 to all five ViTs and to the
language-model pairs, and its review (X3B_REVIEW.md section 3, P1 and P2) supersedes the X3 text proposals that the
v8 MOVED.md had used. In v8 the three last-layer X3 curves lay on top of each other, so the diamonds that the caption
decoded were not visible. (c) now shows, at k = 10 against m, the median over the 15 pairs of each binding X3b set
and of its mid-depth reference: sentence encoders (set B, centering x five-caption average; mid depth = E0 arm CA,
set Bref) as squares and language models (set A1, clip x LM tokens at relative depth 0.5; mid depth = S1.1 clip,
set A1ref, which is the language-model line of (a)) as circles; mid depth (ViT block 5) open, last layer (block 11)
filled, as in Figure 2 (c). The star (Schnaus et al., 0.385 at k/m = 0.01) and every X3 assert stay; the X3 numbers
move to the Results text with the X3b ones (MOVED.md). Also: the caption reference to Conjecture scale moves to the
clause it supports, and the caption stays at about 100 words (102 without panel tags).

Message: with sentence encoders as the text side of the S1.1 COCO pools, vision-language agreement still falls with the
evaluation size and stays close to stable at a fixed k/m for k in {10, 100} (preregistered P1 of ANALYSIS_PLAN_E0,
label 'holds'); in the exploratory X3 and X3b follow-ups, the depth of the vision features sets the level of
agreement with both text sides, while the fall with m at fixed k remains.

  (a) class median of mKNN s_N at k = 10 against the evaluation size m (log-log) for the seed, vision and vision-language
      (language-model tokens) classes of Figure 2 (S1.1 v1.1, config clip) and for the new class, vision-language pairs
      with a sentence encoder (E0 arm P); thin lines are the 15 new pairs. (v6, unchanged in content.)
  (b) the P1 statistic: per-pair e_k + e_m over k in {10, 100} and the 7 pool sizes, for the 15 new pairs (rows by
      encoder) and the 15 S1.1 vision-language pairs (rows by language model); frozen tolerance |sum| <= 0.05 shaded,
      zero dashed, group medians as bars. (v6, unchanged in content.)
  (c) v8.1: see CRITIC v8.1 above (X3b medians). The maker's v8 text for (c), kept for the X3 block that stays:
      (X3, exploratory; results/E0_sentence/x3, readings fixed before any score): mKNN s_N at k = 10 against m for
      DINOv2-B x MPNet under centering and the five-caption average (E0 arm CA, the arm compared with Schnaus et al.),
      for five vision features: CLS and patch-token mean at mid depth (block 5; the CLS is the E0 feature), CLS and
      patch-token mean at the last layer (block 11), and the pooling of Schnaus et al. (mean of all output tokens after
      the final norm, 224 px). The star marks the value Schnaus et al. report for DINOv2-B x MPNet at k/m = 0.01
      (k = 100 on 10,000 COCO points; 0.385, the vector reading of their Fig. 23 in X3_REVIEW.md section 2.2), drawn
      at the same k/m on this k = 10 axis (m = 1000); the placement is licensed by the X3 data themselves:
      |ln L(10, 1024) - ln L(100, 10000)| < 0.05 for all five features (asserted).

v8 changes (author on v7 Figure 6, 2026-10-10; rules for every experiment figure). Started from a byte copy of
~/hrh_v2_figs/v6/F_e0_app/make_F_e0_app.py (copy in _v6_ref/, sha256 in numbers.json):
  1 no text that states a setting on the panels: the 'k = 10' note of (a) and the group headers of (b) ('sentence
    encoders', 'language models') are gone; the row labels of (b) carry the group colour, and the caption names k;
  2 v8 palette (hrh_style_v8, Paul Tol): the sentence-encoder class is S.PAIR['sent'] #994455 (Tol medium-contrast dark
    red; replaces the maker's own #8D1608, as the style designer asked); seed / vision / vision-language are the v8
    S.PAIR blues; (c) uses the sentence-encoder colour for every pair (all are vision-sentence-encoder pairs) and the
    paper's depth convention of Figure 2 (c): mid depth open, last layer filled;
  3 bold tick labels and axis titles with bold math (S.finalize; S.check asserts);
  4 nothing secondary is drawn: Contriever and Qwen3 in (c) (within 0.01 of MPNet), the k = 100 curves of X3 and the
    level-slope line are Results-text numbers (MOVED.md).

v8 c (2026-10-10, role consistency checker across all v8 figures; the v8.1 files are in ./_prev_consistency/). Two
marks read against the conventions of the other figures, and are changed; data, caption and size are unchanged:
  * (a) the class medians lose their filled circles (MEDIAN_MARKERS_A = False): (c) and Figures 2 (c) and F.3 use
    filled marks for the last layer and open marks for mid depth, while the curves of (a) are mid-depth curves (the
    language-model line of (a) is the open-circle line of (c)). The thick lines on their white underlay stay, as the
    class medians of F.1 (a-c), and the direct labels keep their v8.1 positions (they still clear the outer marker
    diameter of v8.1);
  * (b) the six row labels are INK (ROW_LABEL_INK = True), as every row label of the other figures: in the group
    colour the family names read in the vision-language blue, whereas Figure 3 and F.3 write 'BLOOMZ' in the BLOOMZ
    indigo (S.FAM_TEXT) and Figure 4 colours the families with S.FAM. The dots keep the group colours, which the direct
    labels of (a) name, and the gap between the two groups stays.

Read-only inputs (asserted by sha256; nothing is written outside this directory): those of v6 (listed in INS) and
  ~/hrh_v2_stage1_20261008/results/E0_sentence/x3/X3_results.json, X3_REVIEW.md, eval/X3_cells.parquet,
  eval/X3_levels.parquet; ~/hrh_v2_stage1_20261008/results/E0_sentence/eval/E0_cells_CA.parquet;
  ~/hrh_v2_stage1_20261008/analysis/e0_sentence/x3/X3_READINGS.md; v8.1: results/E0_sentence/x3b/X3B_results.json,
  X3B_REVIEW.md, eval/X3B_cells.parquet, review/X3B_review_checks.json; analysis/e0_sentence/x3b/X3B_READINGS.md.
Style: frozen ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (sha256 asserted, never edited); cvdlib.py of the same folder.
Writes only next to this file: F_e0_app.pdf (vector, Type 42), F_e0_app.png (pdftoppm 200 dpi),
_view_print_100dpi.png, _views_cvd_grey.png, numbers.json, caption_v8.tex.
Run (CPU, 1 thread, at most 6 cores):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 nice -n 10 \
      ~/hrh_v2_figs_venv/bin/python -B ~/hrh_v2_figs/v8/F_e0_app/make_F_e0_app.py > run.log 2>&1
"""
import os, sys, json, hashlib, datetime, subprocess, io, re

sys.dont_write_bytecode = True
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))

import numpy as np
import pandas as pd

H_ = os.path.expanduser
STYLE_PATH = H_("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_PATH, "rb").read()).hexdigest() == STYLE_SHA, "frozen style module changed"
STYLE_DIR = os.path.dirname(STYLE_PATH)
PAL_DIR = STYLE_DIR                      # v8: cvdlib.py next to the style module (byte-identical to v5_1/_palette)
sys.path.insert(0, STYLE_DIR)
import hrh_style_v8 as S  # noqa: E402
import cvdlib as CV  # noqa: E402  (read-only import: CIEDE2000, Machado 2009 CVD, WCAG contrast)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator, FuncFormatter  # noqa: E402
import PIL.Image  # noqa: E402

S.use()
assert S.STYLE_SHA256 == STYLE_SHA, "shared style module changed"
CVD_SHA = "562cca0ecb8d218cac9baa6e2e7c550d21be66857c05707f91944e863891e653"   # = ~/hrh_v2_figs/v5_1/_palette/cvdlib.py
LW, MSZ, Z, INK = S.LW, S.MS, S.Z, S.INK
TOL = 1e-12

# ================================================================== inputs
P0 = H_("~/hrh_v2_stage1_20261008")
E0R = f"{P0}/results/E0_sentence"
F_OBS = f"{E0R}/obs/obs_P.parquet"
F_CELLS = f"{E0R}/eval/E0_cells_P.parquet"
F_ELAS = f"{E0R}/eval/E0_elasticity_P.parquet"
F_LEV = f"{E0R}/eval/E0_levels_P.parquet"
F_SLO = f"{E0R}/eval/E0_slopes_P.parquet"
F_VER = f"{E0R}/VERDICTS_E0.json"
F_REV = f"{P0}/analysis/e0_sentence/review/review_paper_numbers_20261010T142548Z.json"
V11 = f"{P0}/results/S1_1/v11"
F_C11, F_S11, F_SUM11 = f"{V11}/S1_1_v11_cells_clip.parquet", f"{V11}/S1_1_v11_slopes_clip.parquet", \
    f"{V11}/S1_1_v11_summary.json"
F_FIG2 = H_("~/hrh_v2_figs/v5/F_neighbors/numbers.json")
F_FIG2V6 = H_("~/hrh_v2_figs/v6/F_neighbors/numbers.json")
F_CVD = f"{PAL_DIR}/cvdlib.py"
EXPECT = {F_VER: "5b04e46098e39056f6e9b2d534d53fa54777b1670ccfa68442cb7ef3fd1568a7",          # STATUS line 337
          F_REV: "4bbe71cd0206c612ff58bd1c4c4b23ce065c971c4a15cb934e4091a706843286",          # STATUS line 340
          F_OBS: "1e13c7df10571a250de0faf44f9767ca4a835f7d9aff960a96f5b51ea68c6cbc",          # VERDICTS obs_sha256.P
          F_FIG2: "19c007bdb392f4cafdb714048be56447fc0f60bd01767914eb347876301335f9"}         # v6 F_neighbors FROZEN


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


INS = {p: sha(p) for p in (F_OBS, F_CELLS, F_ELAS, F_LEV, F_SLO, F_VER, F_REV, F_C11, F_S11, F_SUM11, F_FIG2, F_FIG2V6,
                           F_CVD)}
for p, h in EXPECT.items():
    assert INS[p] == h, ("input changed", p, INS[p])
VER = json.load(open(F_VER))
REV = json.load(open(F_REV))
FIG2 = json.load(open(F_FIG2))
FIG2V6 = json.load(open(F_FIG2V6))
assert VER["obs_sha256"]["P"] == INS[F_OBS]
assert FIG2V6["inputs_sha256"][F_C11] == INS[F_C11], "S1.1 cells changed since the v6 Figure 2 record"
INPUT_MODES = {p: oct(os.stat(p).st_mode & 0o777) for p in INS}


def src(path, key):
    return {"source": path, "sha256": INS[path], "key": key}


N = {"figure": "F_e0_app", "tex_label_proposed": "fig:app-e0", "tex_include": "figures/exp_v6/F_e0_app.pdf",
     "version": "v6 (new appendix figure for E0)",
     "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "script": os.path.abspath(__file__), "script_sha256": sha(os.path.abspath(__file__)),
     "inputs_sha256": INS, "inputs_expected_sha256_checked": EXPECT, "input_modes": INPUT_MODES,
     "font_text": font_manager.findfont(font_manager.FontProperties(family=["serif"])),
     "message": "With sentence encoders as the text side, vision-language agreement still falls with the evaluation "
                "size and stays stable at a fixed k/m for k >= 10 (P1 holds); its level lies between the "
                "vision-language pairs with language-model tokens and the vision pairs."}

# ================================================================== (a) class medians at k = 10 against m
MS = [256, 512, 1024, 2048, 4096, 8192, 16384]
ENC = ["mpnet", "contriever", "qwen3emb06"]          # VERDICTS P1.per_encoder_median_sum keys; row order of (b)
ENC_LABEL = {"mpnet": "MPNet", "contriever": "Contriever", "qwen3emb06": "Qwen3"}     # short row labels (author 2026-10-10)
LMS = ["pythia410m_s1", "bloomz_1b7", "openllama_3b"]     # paper order (app:exp-s11)
LM_LABEL = {"pythia410m_s1": "Pythia", "bloomz_1b7": "BLOOMZ", "openllama_3b": "OpenLLaMA"}
VITS = ["dinov2_b", "clip_b", "mae_b", "augreg_b", "siglip_b"]

cE = pd.read_parquet(F_CELLS)
assert set(cE.cls) == {"cross-modal-sentence"} and cE.primary.all() and (cE.n_draws == 5).all()
assert cE.pair.nunique() == 15
assert sorted(cE.pair.unique()) == sorted(f"{v}__{e}" for v in VITS for e in ENC)
# the eval cells are the mean over the 5 nested orderings of the per-draw obs (obs_P.parquet)
oP = pd.read_parquet(F_OBS)
assert set(oP.config) == {"P"} and oP.draw.nunique() == 5
dm = oP.groupby(["pair", "m", "k"]).obs.mean()
dev_drawmean = float(max(abs(r.obs - dm[(r.pair, r.m, r.k)]) for r in cE.itertuples()))
assert dev_drawmean <= TOL, dev_drawmean
c11 = pd.read_parquet(F_C11)
cp11 = c11[c11.primary]
assert cp11.pair.nunique() == 29 and (cp11.n_draws == 5).all()
CLS11 = {"seed": "seed-seed", "vision": "cross-vision", "vl": "cross-modal"}
assert sorted(cp11[cp11.cls == "cross-modal"].pair.unique()) == sorted(f"{v}__{l}" for v in VITS for l in LMS)

A = {}
e10 = cE[cE.k == 10]
A["per_pair_obs_k10_sentence"] = {
    "values": {p: {int(m): float(v) for m, v in g.set_index("m").obs.sort_index().items()} for p, g in e10.groupby("pair")},
    **src(F_CELLS, "cells[k==10].obs by (pair, m): mean over the 5 nested orderings of obs_P.parquet obs"),
    "check_draw_mean_max_abs_diff": dev_drawmean}
lev = pd.read_parquet(F_LEV)
lev10 = lev[lev.k == 10].set_index("pair")
dev_lev = max(abs(A["per_pair_obs_k10_sentence"]["values"][p][m] - lev10.loc[p, f"L_m{m}"]) for p in lev10.index for m in MS)
assert dev_lev <= TOL, dev_lev
A["per_pair_obs_k10_sentence"]["check_vs_E0_levels_P_max_abs_diff"] = float(dev_lev)
CM = {"sent": {m: float(e10[e10.m == m].obs.median()) for m in MS}}
for k_, c in CLS11.items():
    g = cp11[(cp11.cls == c) & (cp11.k == 10)]
    CM[k_] = {m: float(g[g.m == m].obs.median()) for m in MS}
A["class_median_obs_k10"] = {
    "values": CM,
    "sources": {"sent": src(F_CELLS, "median over the 15 pairs of cells[k==10].obs by m"),
                "seed|vision|vl": src(F_C11, "median over pairs of cells[primary & k==10 & cls].obs by m (Figure 2 a)")}}
# asserts: review_paper_numbers (sentence, language-model tokens), Figure 2 numbers (seed, vision, vl), VERDICTS m = 1024
for m in MS:
    assert abs(CM["sent"][m] - REV["class_median_k10"][f"sentence_m{m}"]) <= TOL, m
    assert abs(CM["vl"][m] - REV["class_median_k10"][f"lm_token_m{m}"]) <= TOL, m
    for k_ in ("seed", "vision", "vl"):
        assert abs(CM[k_][m] - FIG2["panel_a"]["class_median_obs_k10"]["values"][k_][str(m)]) <= TOL, (k_, m)
P3A = VER["descriptive"]["P3a_level"]
assert abs(CM["sent"][1024] - P3A["P"]["cross-modal-sentence"]["L_k10_m1024"][0]) <= TOL
for k_, c in CLS11.items():
    assert abs(CM[k_][1024] - P3A["S1.1-clip"][c]["L_k10_m1024"][0]) <= TOL, k_
A["asserts"] = ["sentence and vl class medians at every m equal review_paper_numbers class_median_k10 "
                "(sentence_m*, lm_token_m*) to 1e-12",
                "seed, vision, vl class medians at every m equal ~/hrh_v2_figs/v5/F_neighbors/numbers.json "
                "panel_a.class_median_obs_k10 (Figure 2 a) to 1e-12",
                "all four class medians at m = 1024 equal VERDICTS descriptive.P3a_level.<arm>.<cls>.L_k10_m1024[0] to 1e-12",
                "per-pair new-pair cells equal the 5-draw mean of obs_P.parquet and E0_levels_P.parquet L_m* to 1e-12"]
# median slopes at k = 10 (Results text; VERDICTS P3f), recomputed from the slopes parquet files and asserted
sE = pd.read_parquet(F_SLO)
s11 = pd.read_parquet(F_S11)
P3F = VER["descriptive"]["P3f_ordering"]
SLOPE = {}
for kk in (1, 10, 100):
    re_ = {"cross-modal-sentence": float(sE[sE.k == kk].slope_obs.median())}
    for c in CLS11.values():
        re_[c] = float(s11[s11.primary & (s11.cls == c) & (s11.k == kk)].slope_obs.median())
    for c, v in re_.items():
        assert abs(v - P3F[str(kk)]["median_slope"][c]) <= TOL, (kk, c)
    SLOPE[kk] = re_
    assert P3F[str(kk)]["slope_order"] == P3F[str(kk)]["level_order"] == \
        ["seed-seed", "cross-vision", "cross-modal-sentence", "cross-modal"], kk
N_FALL = {kk: int((sE[sE.k == kk].slope_class == "falling").sum()) for kk in (1, 10, 100)}
assert N_FALL == {1: 15, 10: 15, 100: 15}
for kk in ("1", "10", "100"):
    assert P3A["P"]["cross-modal-sentence"]["slope_classes"][kk] == {"falling": 15}
A["median_slope_by_k"] = {"values": SLOPE, **src(F_VER, "descriptive.P3f_ordering.<k>.median_slope (recomputed from "
                                                        "E0_slopes_P / S1_1_v11_slopes_clip slope_obs medians, equal to 1e-12)"),
                          "not_plotted": True}
N["panel_a"] = A

# ================================================================== (b) the P1 statistic per pair
P1 = VER["P1"]
assert P1["label"] == "holds" and VER["labels"]["P1"] == "holds" and P1["k_set"] == [10, 100]
BAND = float(P1["thresholds"]["abs_sum"])
assert BAND == 0.05
el = pd.read_parquet(F_ELAS)
el = el[el.k_set == "k10_100"].set_index("pair")
assert len(el) == 15 and (el.n_cells == 14).all()
VPP = {r["pair"]: r for r in P1["per_pair"]}
assert sorted(VPP) == sorted(el.index)


def ols_el(g):
    X = np.c_[np.ones(len(g)), np.log(g.k.astype(float)), np.log(g.m.astype(float))]
    b = np.linalg.lstsq(X, np.log(g.obs.astype(float)), rcond=None)[0]
    return float(b[1]), float(b[2])


NEW = {}
dev_el = dev_ols = 0.0
for p in sorted(VPP):
    g = cE[(cE.pair == p) & cE.k.isin([10, 100])]
    assert len(g) == 14
    ek, em = ols_el(g)
    for key in ("e_k", "e_m", "sum"):
        dev_el = max(dev_el, abs(float(el.loc[p, key]) - VPP[p][key]))
    dev_ols = max(dev_ols, abs(ek - VPP[p]["e_k"]), abs(em - VPP[p]["e_m"]), abs(ek + em - VPP[p]["sum"]))
    vit, enc = p.split("__")
    NEW[p] = {"vit": vit, "text": enc, "e_k": VPP[p]["e_k"], "e_m": VPP[p]["e_m"], "sum": VPP[p]["sum"]}
assert dev_el <= TOL and dev_ols <= TOL, (dev_el, dev_ols)
sums_new = np.array([NEW[p]["sum"] for p in NEW])
med_new = float(np.median(sums_new))
assert abs(med_new - P1["median_sum"]) <= TOL
assert abs(float(np.mean(np.abs(sums_new) <= BAND)) - P1["frac_abs_sum_le"]) <= TOL and P1["frac_abs_sum_le"] == 1.0
inf_ = sorted(p for p in NEW if abs(NEW[p]["e_m"]) >= P1["thresholds"]["informative_abs_e_m"])
assert inf_ == sorted(P1["informative_pairs"]) and len(inf_) == P1["n_informative"] == 15
assert abs(med_new) <= P1["thresholds"]["hold_abs_median"]
for e in ENC:
    v = float(np.median([NEW[p]["sum"] for p in NEW if NEW[p]["text"] == e]))
    assert abs(v - P1["per_encoder_median_sum"][e]) <= TOL, e
    assert abs(v - VER["descriptive"]["P3e_encoders"][e]["median_sum_k10_100"]) <= TOL, e
assert abs(med_new - P3A["P"]["cross-modal-sentence"]["median_sum_k10_100"]) <= TOL
assert abs(float(sums_new.min()) - REV["sentence_sum_range_k10_100"][0]) <= TOL
assert abs(float(sums_new.max()) - REV["sentence_sum_range_k10_100"][1]) <= TOL
assert set(VER["descriptive"]["robustness"]["P1_per_draw_labels"].values()) == {"holds"}
SEC = {arm: (d["label"], d["median_sum"]) for arm, d in VER["secondary"]["P1"].items()}
assert all(lab == "holds" for lab, _ in SEC.values()) and len(SEC) == 12
assert sorted(SEC) == sorted(["ctrold", "D1", "D2", "D3", "D4", "A", "Aall", "C", "CA", "CAall", "L", "Pdd"])

OLD = {}
g11 = cp11[(cp11.cls == "cross-modal") & cp11.k.isin([10, 100])]
for p, g in g11.groupby("pair"):
    assert len(g) == 14
    ek, em = ols_el(g)
    vit, lm = p.split("__")
    OLD[p] = {"vit": vit, "text": lm, "e_k": ek, "e_m": em, "sum": ek + em}
sums_old = np.array([OLD[p]["sum"] for p in OLD])
med_old = float(np.median(sums_old))
assert abs(med_old - P3A["S1.1-clip"]["cross-modal"]["median_sum_k10_100"]) <= TOL, med_old

B = {"per_pair_sentence": {"values": NEW, **src(F_VER, "P1.per_pair[*].{e_k,e_m,sum} (equal to E0_elasticity_P.parquet "
                                                      "k_set k10_100 and to an OLS refit on E0_cells_P.parquet, 1e-12)"),
                           "max_abs_diff_vs_elasticity_parquet": dev_el, "max_abs_diff_vs_ols_refit": dev_ols},
     "per_pair_lm_tokens_S11": {"values": OLD, **src(F_C11, "OLS of ln obs on (1, ln k, ln m) over cells[primary & "
                                                           "cls=='cross-modal' & k in {10,100}] (14 cells per pair); "
                                                           "computed here (descriptive)"),
                                "median_asserted_vs": "VERDICTS descriptive.P3a_level.S1.1-clip.cross-modal."
                                                      "median_sum_k10_100 (1e-12)"},
     "group_median_bars": {"sentence": med_new, "lm_tokens": med_old,
                           "sources": {"sentence": "VERDICTS P1.median_sum",
                                       "lm_tokens": "VERDICTS descriptive.P3a_level.S1.1-clip.cross-modal.median_sum_k10_100"}},
     "band": {"half_width": BAND, **src(F_VER, "P1.thresholds.abs_sum (frozen)")},
     "zero_line": 0.0}
N["panel_b"] = B

# every equality assert of this script, with its measured maximum absolute difference (tolerance 1e-12)
DEV = {
    "a: sentence class medians vs review_paper_numbers class_median_k10.sentence_m* (7 m)":
        max(abs(CM["sent"][m] - REV["class_median_k10"][f"sentence_m{m}"]) for m in MS),
    "a: language-model-token class medians vs review_paper_numbers class_median_k10.lm_token_m* (7 m)":
        max(abs(CM["vl"][m] - REV["class_median_k10"][f"lm_token_m{m}"]) for m in MS),
    "a: seed/vision/vl class medians vs Figure 2 numbers.json panel_a.class_median_obs_k10 (21 values)":
        max(abs(CM[k_][m] - FIG2["panel_a"]["class_median_obs_k10"]["values"][k_][str(m)])
            for k_ in ("seed", "vision", "vl") for m in MS),
    "a: four class medians at m = 1024 vs VERDICTS descriptive.P3a_level L_k10_m1024[0]":
        max([abs(CM["sent"][1024] - P3A["P"]["cross-modal-sentence"]["L_k10_m1024"][0])]
            + [abs(CM[k_][1024] - P3A["S1.1-clip"][c]["L_k10_m1024"][0]) for k_, c in CLS11.items()]),
    "a: new-pair cells vs 5-draw mean of obs_P.parquet (105 cells at k = 10, 315 in all)": dev_drawmean,
    "a: new-pair cells at k = 10 vs E0_levels_P.parquet L_m*": float(dev_lev),
    "a (text): median slopes at k = 1, 10, 100 vs VERDICTS descriptive.P3f_ordering.<k>.median_slope":
        max(abs(SLOPE[kk][c] - P3F[str(kk)]["median_slope"][c]) for kk in (1, 10, 100) for c in SLOPE[kk]),
    "b: per-pair e_k, e_m, sum (E0_elasticity_P k10_100) vs VERDICTS P1.per_pair": dev_el,
    "b: per-pair e_k, e_m, sum (own OLS refit on E0_cells_P, 14 cells) vs VERDICTS P1.per_pair": dev_ols,
    "b: median of the 15 plotted sums vs VERDICTS P1.median_sum": abs(med_new - P1["median_sum"]),
    "b: per-encoder medians vs VERDICTS P1.per_encoder_median_sum":
        max(abs(float(np.median([NEW[p]["sum"] for p in NEW if NEW[p]["text"] == e])) - P1["per_encoder_median_sum"][e])
            for e in ENC),
    "b: share of |sum| <= 0.05 vs VERDICTS P1.frac_abs_sum_le": abs(float(np.mean(np.abs(sums_new) <= BAND))
                                                                  - P1["frac_abs_sum_le"]),
    "b: sum range vs review_paper_numbers sentence_sum_range_k10_100":
        max(abs(float(sums_new.min()) - REV["sentence_sum_range_k10_100"][0]),
            abs(float(sums_new.max()) - REV["sentence_sum_range_k10_100"][1])),
    "b: median of the 15 S1.1 sums vs VERDICTS descriptive.P3a_level.S1.1-clip.cross-modal.median_sum_k10_100":
        abs(med_old - P3A["S1.1-clip"]["cross-modal"]["median_sum_k10_100"])}
assert all(v <= TOL for v in DEV.values()), DEV
N["asserts_max_abs_diff"] = {"tolerance": TOL, "values": DEV,
                             "exact": ["VERDICTS labels.P1 == P1.label == 'holds'", "P1.k_set == [10, 100]",
                                       "P1.thresholds.abs_sum == 0.05 (the shaded band)",
                                       "informative pairs (|e_m| >= 0.10) == P1.informative_pairs, 15 of 15",
                                       "|median| <= P1.thresholds.hold_abs_median",
                                       "P1_per_draw_labels all 'holds'; secondary.P1 all 'holds' (12 cells)",
                                       "P3f slope_order == level_order at k = 1, 10, 100",
                                       "15 of 15 new pairs falling at k = 1, 10, 100 (E0_slopes_P, VERDICTS P3a)"]}
N["scope_notes"] = [
    "(a) and (b) use arm P only: the S1.1 config clip with the single pool caption C1 per image, the arm where encoders "
    "and language models compare cleanly (plan 13.3: without Q1 no level difference in the five-caption-average arms "
    "is attributed to the encoder); no AVG arm is plotted",
    "(b) plots the preregistered P1 statistic for k in {10, 100}; the k >= 10 restriction was chosen after seeing S1.1 "
    "(VERDICTS P1.disclosure) and the k = 1 departure is exploratory, so neither is drawn",
    "nothing about P2b / P2c (uninformative) is drawn; the slopes are not printed on the panel, so the figure does not "
    "suggest that the level sets the slope",
    "the S1.1 per-pair sums of (b) are computed here (descriptive); only their median is a VERDICTS statistic"]


# ================================================================== (c) X3 inputs (exploratory follow-up) and asserts
N["version"] = ("v8.1 (critic: panel (c) = X3b medians for both text sides; v8 maker: Tol palette, bold ticks "
                "and axis titles, no setting text, new panel (c) for X3)")
V6_MAKER = f"{HERE}/_v6_ref/make_F_e0_app.py"
assert sha(V6_MAKER) == sha(H_("~/hrh_v2_figs/v6/F_e0_app/make_F_e0_app.py")), "v6 maker copy differs"
N["started_from_v6_maker_sha256"] = sha(V6_MAKER)
assert INS[F_CVD] == CVD_SHA, "cvdlib changed"
X3D = f"{E0R}/x3"
F_X3, F_X3REV = f"{X3D}/X3_results.json", f"{X3D}/X3_REVIEW.md"
F_X3C, F_X3L = f"{X3D}/eval/X3_cells.parquet", f"{X3D}/eval/X3_levels.parquet"
F_CA = f"{E0R}/eval/E0_cells_CA.parquet"
F_X3READ = f"{P0}/analysis/e0_sentence/x3/X3_READINGS.md"
EXPECT_X3 = {F_X3: "4945d75d4102ff7598c9d9672a782d49f20aa34e4d8c2bf3ae44dc4fd6b767b3",          # run.md
             F_X3REV: "39421ac0867a1a9a03d5eda82e0babc160337612d0cfa1494585823a9381a258",       # review.md section 5
             F_X3READ: "0151bf9ac26dd718652d69c72912445fb9fd15d770ae829db8a052e085d8663e",      # STATUS line 344
             F_X3C: "179c89fe96dc5bb0464ebf2ee6a46c6ec887a948c98280536f3bf1cb93070729",
             F_X3L: "cb5312f9870f169e0e24d5c356e8c0529060cf5f83c5383c39f6726e5f487eff",
             F_CA: "a04bc4e7efd86cf214dc995223fdcab3f668dad392901bd29712d619f73ecc5e"}
for p, h in EXPECT_X3.items():
    INS[p] = sha(p)
    INPUT_MODES[p] = oct(os.stat(p).st_mode & 0o777)
    assert INS[p] == h, ("X3 input changed", p, INS[p])
N["inputs_sha256"], N["input_modes"] = INS, INPUT_MODES
X3 = json.load(open(F_X3))
assert X3["status"].startswith("EXPLORATORY") and X3["readings"]["sha256"] == INS[F_X3READ]
assert X3["pair_count"]["total_X3_pairs"] == 12
ENC_C = "mpnet"                                     # the binding pair of the X3 readings (DINOv2-B x MPNet)
FEAT = [  # (feature key, depth, token set, description)
    ("dinov2_b", "mid", "cls", "CLS token, block 5 of 12 (mid depth; the E0 feature)"),
    ("x3_tok_r05", "mid", "patch", "patch-token mean, block 5 (mid depth)"),
    ("x3_cls_r10", "last", "cls", "CLS token, block 11 (last layer)"),
    ("x3_tok_r10", "last", "patch", "patch-token mean, block 11 (last layer)"),
    ("x3add_schnaus224", "last", "schnaus", "mean of all output tokens after the final norm, 224 px (Schnaus et al.)")]
PX3 = {p["pair"]: p for p in X3["pairs"]}
x3c, x3l, ca = pd.read_parquet(F_X3C), pd.read_parquet(F_X3L), pd.read_parquet(F_CA)
CUR, dev_c, dev_L = {}, 0.0, 0.0
for f, depth, tok, desc in FEAT:
    pair = f"{f}__{ENC_C}"
    rows = X3["curves"][pair]["10"]
    assert [r["m"] for r in rows] == MS, pair
    s_ = np.array([r["s"] for r in rows], float)
    tab = ca if f == "dinov2_b" else x3c          # the E0 reference equals E0 arm CA (wrapper check, run.md)
    g = tab[(tab.pair == pair) & (tab.k == 10)].set_index("m").obs.sort_index()
    assert list(g.index) == MS and len(g) == 7, pair
    dev_c = max(dev_c, float(np.max(np.abs(g.to_numpy(float) - s_))))
    L10_1024, L100_10000 = PX3[pair]["L_10_1024"], PX3[pair]["L_100_10000"]
    dev_L = max(dev_L, abs(L10_1024 - s_[MS.index(1024)]))
    if f != "dinov2_b":
        lv = x3l[(x3l.pair == pair) & (x3l.k == 100)]
        assert len(lv) == 1
        dev_L = max(dev_L, abs(float(lv.L_m10000.iloc[0]) - L100_10000))
    else:
        dev_L = max(dev_L, abs(X3["reference_check_vs_readings"][ENC_C]["E0_levels_CA"] - L100_10000))
    CUR[f] = {"pair": pair, "depth": depth, "token": tok, "description": desc, "m": MS, "s_k10": s_.tolist(),
              "L_10_1024": L10_1024, "L_100_10000": L100_10000,
              "ln_L10_1024_minus_ln_L100_10000": float(np.log(L10_1024) - np.log(L100_10000)),
              "b10": PX3[pair]["b10"], "b10_class": PX3[pair]["b10_class"], "b100": PX3[pair]["b100"],
              "sum_k10_100": PX3[pair]["sum_k10_100"], "L_10_10000": PX3[pair]["L_10_10000"],
              "gap_label": X3["binding_gap_reading_mpnet"].get(f, {}).get("gap_label", PX3[pair]["gap_label"]),
              "share_of_log_gap_0.38": PX3[pair]["share_0.38"]}
assert dev_c <= TOL and dev_L <= TOL, (dev_c, dev_L)
# the message of (c), asserted from the data
LAST = [f for f, d, _, _ in FEAT if d == "last"]
MID = [f for f, d, _, _ in FEAT if d == "mid"]
DEPTH_RATIO = {m: min(CUR[f]["s_k10"][i] for f in LAST) / max(CUR[f]["s_k10"][i] for f in MID) for i, m in enumerate(MS)}
assert all(v > 2.0 for v in DEPTH_RATIO.values()), DEPTH_RATIO          # depth sets the level at every m
assert all(CUR[f]["b10"] < -0.25 and CUR[f]["b10_class"] == "falling" for f in CUR), {f: CUR[f]["b10"] for f in CUR}
for f in CUR:                                                             # still falls with m at k = 10, beyond m = 512
    s_ = CUR[f]["s_k10"]
    assert all(s_[i + 1] < s_[i] for i in range(1, 6)), (f, s_)
# placing the reported value at the same k/m on this k = 10 axis is licensed by the X3 data themselves
SCALE_DEV = {f: abs(CUR[f]["ln_L10_1024_minus_ln_L100_10000"]) for f in CUR}
assert max(SCALE_DEV.values()) < 0.05, SCALE_DEV
# Schnaus et al.: Fig. 23 (p. 48) dotted line, read from the PDF vector data in X3_REVIEW.md section 2.2
REVTXT = open(F_X3REV, encoding="utf-8").read()
for s_ in ("371.720", "349.570", "327.421", "329.093", "0.385"):
    assert s_ in REVTXT, s_
SCHNAUS = (371.720 - 329.093) / (371.720 - 327.421) * 0.4
assert round(SCHNAUS, 3) == 0.385 and abs((371.720 - 349.570) - (349.570 - 327.421)) < 0.01
assert X3["schnaus_reading"]["value"] == 0.38 and X3["schnaus_reading"]["band"] == [0.37, 0.39]
M_SCHNAUS = 1000.0                                  # k = 10 at k/m = 0.01 (their k = max(1, floor(n/100)))
OTHER_ENC = {f: max(abs(PX3[f"{f}__{e}"]["L_100_10000"] - PX3[f"{f}__mpnet"]["L_100_10000"])
                    for e in ("contriever", "qwen3emb06")) for f in CUR}
assert max(OTHER_ENC.values()) < 0.01, OTHER_ENC
L10_10000_LAST_ALL = [PX3[f"{f}__{e}"]["L_10_10000"] for f in LAST for e in ("mpnet", "contriever", "qwen3emb06")]
N["panel_c"] = {
    "status": X3["status"], "pair": "DINOv2-B x MPNet, E0 arm CA (centering x five-caption average)",
    "k": 10, "curves": CUR,
    "sources": {"curves": src(F_X3, "curves.<feature>__mpnet.10[*].s (5-draw means)"),
                "check_x3_cells": src(F_X3C, "obs at k = 10, equal to 1e-12"),
                "check_E0_cells_CA": src(F_CA, "obs of dinov2_b__mpnet at k = 10 (the E0 reference), equal to 1e-12"),
                "levels": src(F_X3L, "L_m10000 at k = 100 (log-linear between m = 8192 and 16384)"),
                "readings": src(F_X3READ, "frozen reading rule, before any X3 score")},
    "max_abs_diff_curves_vs_cells": dev_c, "max_abs_diff_levels": dev_L,
    "depth_ratio_min_last_over_max_mid_by_m": DEPTH_RATIO,
    "schnaus": {"value": SCHNAUS, "plotted_at_m": M_SCHNAUS, "k": 10,
                "reading": "Fig. 23 p. 48, dotted 'full space' line, MPNet panel; ticks 0 / 0.2 / 0.4 at y = 371.720 / "
                           "349.570 / 327.421 pt, line at 329.093 pt (X3_REVIEW.md section 2.2); run reading 0.38",
                "their_setting": "k = max(1, floor(n/100)) = 100 on n = 10,000 COCO points, DINOv2 ViT-B/14 x MPNet",
                "placement": "drawn at k/m = 0.01 on the k = 10 axis (m = 1000); |ln L(10,1024) - ln L(100,10000)| "
                             "< 0.05 for every plotted feature (scale reading, asserted)",
                "source": src(F_X3REV, "section 2.2")},
    "scale_dev_ln": SCALE_DEV, "other_encoders_max_abs_dL_100_10000": OTHER_ENC,
    "L_10_10000_last_layer_all_encoders_range": [min(L10_10000_LAST_ALL), max(L10_10000_LAST_ALL)],
    "p1_type_descriptive_all_12": X3["p1_type_descriptive"]["all_12"],
    "contrasts_ln_L_100_10000_mpnet": X3["contrasts_ln_L_100_10000"]["mpnet"],
    "not_drawn": ["Contriever and Qwen3-Embedding-0.6B pairs (within 0.01 of MPNet at L(100, 10000))",
                  "k = 1 and k = 100 curves", "the level-slope line of P-S2b (Results text)"]}

# ================================================================== (c) critic v8.1: X3b inputs, medians and asserts
# The X3 block above stays (the star, its placement and the X3 numbers of the Results text, every assert unchanged);
# (c) now draws the X3b medians. Sets (X3B_READINGS.md): B = ctr, ViT block 11 x five-caption average of the sentence
# encoders (Q-b binding), Bref = its mid-depth reference (= E0 arm CA); A1 = clip, ViT block 11 x language-model
# tokens at relative depth 0.5 (Q-a binding), A1ref = its mid-depth reference (= S1.1 clip, the vl line of (a)).
X3BD = f"{E0R}/x3b"
F_X3B, F_X3BREV = f"{X3BD}/X3B_results.json", f"{X3BD}/X3B_REVIEW.md"
F_X3BC, F_X3BCHK = f"{X3BD}/eval/X3B_cells.parquet", f"{X3BD}/review/X3B_review_checks.json"
F_X3BREAD = f"{P0}/analysis/e0_sentence/x3b/X3B_READINGS.md"
EXPECT_X3B = {F_X3B: "118a49bf7fd15d14d69fdcf0f2b8adadacd573330961d3c8353ed5fec2c1d767",      # STATUS line 352
              F_X3BREV: "2bccaad88049e96ec6d4aa36ff3de51ad05d0353633d4809742d138b66ceed6c",   # STATUS line 354
              F_X3BC: "b517082960bf05dc71ec07c1052e0f53123bbdb72576ab8149c459ce836fa344",     # STATUS line 351 (b5170829)
              F_X3BCHK: "2c8d0fcb4f5f303a93733650a57a613a21237d5a3935b8a130e1f55ee3c96ce0",   # STATUS line 354
              F_X3BREAD: "fef078b8058a0bbb19f25697014130f60b5f8d83a8e93d6b8b61900ecb2dd34e"}  # STATUS line 350
for p, h in EXPECT_X3B.items():
    INS[p] = sha(p)
    INPUT_MODES[p] = oct(os.stat(p).st_mode & 0o777)
    assert INS[p] == h, ("X3b input changed", p, INS[p])
N["inputs_sha256"], N["input_modes"] = INS, INPUT_MODES
X3B = json.load(open(F_X3B))
X3BCHK = json.load(open(F_X3BCHK))
assert X3B["status"] == "EXPLORATORY" and X3B["readings_sha256"] == INS[F_X3BREAD]
assert X3BCHK["inputs_sha256"]["X3B_results.json"] == INS[F_X3B]
assert X3B["Q_a"]["binding_set"] == "A1" and X3B["Q_b"]["label"] == "the rise with depth is not specific to DINOv2-B"
x3bc = pd.read_parquet(F_X3BC)
SETS_C = {("sent", "last"): "B", ("sent", "mid"): "Bref", ("vl", "last"): "A1", ("vl", "mid"): "A1ref"}
CURB = {}
for (grp, depth), s_name in SETS_C.items():
    t = x3bc[(x3bc.set == s_name) & (x3bc.k == 10)]
    assert t.pair.nunique() == 15 and len(t) == 15 * len(MS) and (t.n_draws == 5).all(), s_name
    g = t.groupby("m").obs.median().sort_index()
    assert [int(m) for m in g.index] == MS, s_name
    CURB[f"{grp}_{depth}"] = {"set": s_name, "text_side": grp, "depth": depth, "m": MS,
                              "median_s_k10": [float(v) for v in g], "n_pairs": 15,
                              "pairs": sorted(t.pair.unique())}
# cross-checks to 1e-12: the mid-depth references equal the vl line of (a) and E0 arm CA; X3B_results.json set medians
DEV_C = {"vl_mid_vs_panel_a_vl": max(abs(CURB["vl_mid"]["median_s_k10"][i] - CM["vl"][m]) for i, m in enumerate(MS))}
ca10 = ca[ca.k == 10]
assert ca10.pair.nunique() == 15 and set(ca10.cls) == {"cross-modal-sentence"}
DEV_C["sent_mid_vs_E0_CA"] = max(abs(CURB["sent_mid"]["median_s_k10"][i] - float(ca10[ca10.m == m].obs.median()))
                                 for i, m in enumerate(MS))
for s_name, grp in (("B", "sent"), ("A1", "vl")):
    DEV_C[f"{s_name}_last_L10_1024"] = abs(X3B["sets"][s_name]["median_last_L10_1024"]
                                           - CURB[f"{grp}_last"]["median_s_k10"][MS.index(1024)])
    DEV_C[f"{s_name}_mid_L10_1024"] = abs(X3B["sets"][s_name]["median_mid_L10_1024"]
                                          - CURB[f"{grp}_mid"]["median_s_k10"][MS.index(1024)])
assert max(DEV_C.values()) <= TOL, DEV_C
# the message of (c), asserted from the pairs and the medians
PPB = {s_name: X3B["per_pair"][s_name] for s_name in ("A1", "B")}
for s_name, rows in PPB.items():
    assert len(rows) == 15, s_name
    assert all(r["L10_1024"] > r["L10_1024_mid"] for r in rows), s_name          # every pair rises with depth
    assert all(r["L100_10000"] > r["L100_10000_mid"] for r in rows), s_name      # also at the cell of Schnaus et al.
    assert all(r["b10"] <= -0.1 for r in rows), s_name                           # every last-layer pair still falls
    assert all(abs(r["sum"]) <= 0.05 for r in rows), s_name                      # close to stable at a fixed k/m
DEPTH_RATIO_B = {g: [CURB[f"{g}_last"]["median_s_k10"][i] / CURB[f"{g}_mid"]["median_s_k10"][i]
                     for i in range(len(MS))] for g in ("sent", "vl")}
assert all(v > 2.0 for g in DEPTH_RATIO_B for v in DEPTH_RATIO_B[g]), DEPTH_RATIO_B   # depth sets the level at every m
for key, c in CURB.items():                                  # every median falls with m (the last-layer sentence-encoder
    s_ = c["median_s_k10"]                                   # median from m = 512 on; it rises from 256 to 512)
    start = 1 if key == "sent_last" else 0
    assert all(s_[i + 1] < s_[i] for i in range(start, len(MS) - 1)), (key, s_)
assert CURB["sent_last"]["median_s_k10"][1] > CURB["sent_last"]["median_s_k10"][0]
ORDER_C = ["sent_last", "vl_last", "sent_mid", "vl_mid"]      # top to bottom at every m (the two depth labels rely on it)
assert all(CURB[a]["median_s_k10"][i] > CURB[b]["median_s_k10"][i] for i in range(len(MS))
           for a, b in zip(ORDER_C, ORDER_C[1:])), ORDER_C
MATCHED_KM_LAST = max(v["max_maxabs_10_100"] for k_, v in X3BCHK["matched_km"].items() if k_.endswith("|last"))
assert MATCHED_KM_LAST < 0.09, MATCHED_KM_LAST              # X3b review: matched k/m within 9% at the last layer
STAR_VS_SENT_LAST_1024 = SCHNAUS / CURB["sent_last"]["median_s_k10"][MS.index(1024)]
N["panel_c"]["drawn_v8_1"] = {
    "what": "X3b medians over 15 pairs at k = 10 (mid depth open, last layer filled; sentence encoders squares, "
            "language models circles); the five X3 features of DINOv2-B x MPNet are no longer drawn (Results text)",
    "curves": CURB, "max_abs_diff_checks": DEV_C, "depth_ratio_last_over_mid_by_m": DEPTH_RATIO_B,
    "top_to_bottom": ORDER_C, "matched_km_max_ln_diff_last_layer": MATCHED_KM_LAST,
    "star_over_sent_last_median_m1024": STAR_VS_SENT_LAST_1024,
    "sources": {"curves": src(F_X3BC, "median over pairs of cells[set in {B, Bref, A1, A1ref} & k == 10].obs by m"),
                "sets": src(F_X3B, "sets.<B|A1>.median_<last|mid>_L10_1024 (equal to 1e-12); per_pair.<A1|B>"),
                "matched_km": src(F_X3BCHK, "matched_km.<set>|last.max_maxabs_10_100"),
                "readings": src(F_X3BREAD, "frozen reading rule, before any X3b score"),
                "review": src(F_X3BREV, "section 3, proposals P1 and P2 (manuscript text)")}}
N["panel_c"]["not_drawn"] = N["panel_c"]["not_drawn"] + [
    "v8.1: the five X3 features of DINOv2-B x MPNet (their three last-layer curves overlapped; Results text)",
    "v8.1: single X3b pairs, the other X3b sets (A2, A3, A4) and k = 1, 100"]

# ================================================================== palette (v8)
SENT = S.PAIR["sent"]          # v8: the style module's vision-sentence-encoder colour (Tol medium-contrast dark red)
SENT_TEXT = S.PAIR_TEXT["sent"]
assert SENT == "#994455" and SENT_TEXT == "#994455"
PAPER_COLS = {**{"crit:" + k: v for k, v in S.CRIT.items()}, **{"crit_text:" + k: v for k, v in S.CRIT_TEXT.items()},
              **{"fam:" + k: v for k, v in S.FAM.items()}, **{"fam_text:" + k: v for k, v in S.FAM_TEXT.items()},
              **{"pair:" + k: v for k, v in S.PAIR.items() if k != "sent"},
              **{"arm:" + k: v for k, v in S.ARM.items()}, **{"arm_text:" + k: v for k, v in S.ARM_TEXT.items()},
              **{"link:" + k: v for k, v in S.LINK.items()},
              "INK": S.INK, "MUTED": S.MUTED, "NULL_LINE": S.NULL_LINE}
pal = {"colour": SENT, "text_tone": SENT_TEXT, "line_contrast_white": round(CV.contrast(SENT), 2),
       "contrast_on_NULL_FILL": round(CV.contrast(SENT, S.NULL_FILL), 2),
       "method": "CIEDE2000 on CIELAB (D65) after Machado et al. 2009 severity-1.0 simulation (cvdlib.py of "
                 "~/hrh_v2_figs/v8/_style, sha256 %s)" % INS[F_CVD],
       "nearest_paper_colour_by_view": {}, "in_figure_dE00": {}}
for view in CV.VIEWS:
    k_, d_ = min(((k_, CV.dE(SENT, v_, view)) for k_, v_ in PAPER_COLS.items()), key=lambda t: t[1])
    pal["nearest_paper_colour_by_view"][view] = {"colour": k_, "dE00": round(d_, 1)}
FIGCOLS = {"seed": S.PAIR["seed"], "vision": S.PAIR["vision"], "vl": S.PAIR["vl"], "NULL_LINE": S.NULL_LINE,
           "INK": S.INK}
for k_, v_ in FIGCOLS.items():
    pal["in_figure_dE00"][k_] = [round(CV.dE(SENT, v_, view), 1) for view in CV.VIEWS]
# v8 asserts (replace the v6 asserts on the maker's own #8D1608: nearest paper colour >= 9.0 in all four views and
# in-figure >= 15.0, which pinned that colour). The colour is now the style module's, vetted by the style designer
# (CO_OCCUR 'Fig 2 / F.1 / F.2' >= 9.5 over normal, deutan, protan). Kept: the data colours of this figure stay >= 15
# apart in all four views; the reference lines (NULL_LINE) >= 9.5 in normal, deutan, protan. New: MUTED equals SENT
# under protanopia (dE00 0.2), so no text of this figure uses MUTED (asserted after drawing).
assert min(min(pal["in_figure_dE00"][k_]) for k_ in ("seed", "vision", "vl")) >= 15.0, pal
assert min(pal["in_figure_dE00"]["NULL_LINE"][:3]) >= 9.5 and min(pal["in_figure_dE00"]["INK"]) >= 15.0, pal
assert CV.contrast(SENT) >= S.MIN_CONTRAST
pal["MUTED_vs_SENT_by_view"] = [round(CV.dE(SENT, S.MUTED, view), 1) for view in CV.VIEWS]
pal["views_order"] = list(CV.VIEWS)
pal["v6_colour_replaced"] = "#8D1608 (maker's own) -> S.PAIR['sent'] #994455 (style designer, v8)"
N["palette_check"] = pal

PC = {"seed": S.PAIR["seed"], "vision": S.PAIR["vision"], "vl": S.PAIR["vl"], "sent": SENT}
PT = {"seed": S.PAIR_TEXT["seed"], "vision": S.PAIR_TEXT["vision"], "vl": S.PAIR_TEXT["vl"], "sent": SENT_TEXT}

# ================================================================== drawing (inches throughout)
W, H = 5.5, float(os.environ.get("FE_H", 2.40))
BOT = 0.37
AXH = round(H - BOT - 0.235, 4)                       # top: the panel letters
XA, WA = 0.43, float(os.environ.get("FE_WA", 1.42))   # (a)
XB, WB = float(os.environ.get("FE_XB", 2.78)), float(os.environ.get("FE_WB", 0.86))   # (b); row labels left of XB
XC = float(os.environ.get("FE_XC", 4.22))             # (c)
WC = round(5.33 - XC, 4)
fig = plt.figure(figsize=(W, H))
fig.set_layout_engine("none")
axA = fig.add_axes([XA / W, BOT / H, WA / W, AXH / H])
axB = fig.add_axes([XB / W, BOT / H, WB / W, AXH / H])
axC = fig.add_axes([XC / W, BOT / H, WC / W, AXH / H])
ANN, DATA, ONFILL = [], [], []


def add(artists):
    DATA.extend(artists if isinstance(artists, list) else [artists])
    return artists


def note(t):
    ANN.append(t)
    return t


# ---- label placement along lines (copied from the v5 F_neighbors maker, which stays untouched) -----------------------
def _dense(P, n=60):
    seg = [P[i] + (P[i + 1] - P[i]) * np.linspace(0, 1, n, endpoint=False)[:, None] for i in range(len(P) - 1)]
    return np.vstack(seg + [P[-1:]])


def label_along(ax, s, polylines, x_anchor, side, ha, role, color=None, gap_pt=1.6, lw=0.5, angle_deg=None):
    """Text s parallel to a bundle of polylines (data coords), just below (va top) or above (va baseline) every line
    of the bundle over the text's horizontal span; the angle follows a least-squares fit of the bundle's envelope;
    gap_pt is the clearance measured from the line's edge (lw = its width in pt)."""
    R = fig.canvas.get_renderer()
    probe = fig.text(0, 0, s, **S.text_kw(role, s, color))
    w = probe.get_window_extent(R).width
    probe.remove()
    L = [_dense(ax.transData.transform(np.c_[px, py])) for px, py in polylines]
    xa = ax.transData.transform((x_anchor, 1.0 if ax.get_yscale() == "log" else 0.0))[0]
    th = 0.0 if angle_deg is None else np.radians(angle_deg)
    for _ in range(4):
        dx = w * np.cos(th)
        lo, hi = {"right": (xa - dx, xa), "center": (xa - dx / 2, xa + dx / 2), "left": (xa, xa + dx)}[ha]
        xs = np.linspace(lo, hi, 80)
        Y = np.array([np.interp(xs, q[:, 0], q[:, 1], left=np.nan, right=np.nan) for q in L])
        env = np.nanmin(Y, 0) if side == "below" else np.nanmax(Y, 0)
        ok = np.isfinite(env)
        if angle_deg is None:
            th = np.arctan(np.polyfit(xs[ok], env[ok], 1)[0])
    gap = (gap_pt + lw / 2.0) / np.cos(th) * fig.dpi / 72.0
    rel = env[ok] - np.tan(th) * (xs[ok] - xa)
    ya = rel.min() - gap if side == "below" else rel.max() + gap
    xd, yd = ax.transData.inverted().transform((xa, ya))
    return note(S.text(ax, xd, yd, s, role, color, rotation=np.degrees(th), rotation_mode="anchor", ha=ha,
                       va="top" if side == "below" else "baseline", zorder=Z["label"]))


def setup_m_axis(ax, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xlim(*XLIM_A)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM_A)
    ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e-2: "0.01", 1e-1: "0.1", 1: "1"}.get(v, "")))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    S.int_ticks(ax, [256, 1024, 4096, 16384], "x")
    ax.xaxis.set_minor_locator(FixedLocator([512, 2048, 8192]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("evaluation size $m$")
    if ylabel:
        ax.set_ylabel("mKNN $s_N$")
    ax.spines["left"].set_bounds(YLIM_A[0], 1.0)


# ---- (a) -------------------------------------------------------------------------------------------------------------
YLIM_A = (0.005, 1.0)
XLIM_A = (215.0, 19500.0)
ax = axA
setup_m_axis(ax, True)
BUND = {"sent": []}
for p in sorted(A["per_pair_obs_k10_sentence"]["values"]):
    row = A["per_pair_obs_k10_sentence"]["values"][p]
    xs_, ys_ = np.array(MS, float), np.array([row[m] for m in MS])
    BUND["sent"].append((xs_, ys_))
    add(ax.plot(xs_, ys_, color=PC["sent"], lw=LW["data_thin"], zorder=Z["thin"]))
# class medians: thick line on a white underlay; drawn bottom to top. v8.1 also drew filled circles with a white edge
# (v5 Figure 2 a); v8 c drops them (MEDIAN_MARKERS_A), since filled marks mean 'last layer' in (c), Figure 2 (c) and
# F.3, and these are mid-depth curves
MEDIAN_MARKERS_A = False
ZC = {"vl": 0.0, "seed": 0.1, "vision": 0.2, "sent": 0.3}
MED_POLY = {}
for k_ in ("vl", "seed", "vision", "sent"):
    yy = [CM[k_][m] for m in MS]
    MED_POLY[k_] = (np.array(MS, float), np.array(yy))
    add(S.underlay(ax, MS, yy, LW["data_emph"], Z["data"] + ZC[k_]))
    add(ax.plot(MS, yy, color=PC[k_], lw=LW["data_emph"], zorder=Z["data"] + ZC[k_]))
    if MEDIAN_MARKERS_A:
        add(ax.plot(MS, yy, ls="none", marker="o", ms=MSZ["line"], mfc=PC[k_], mec="white", mew=MSZ["line_edge"],
                    zorder=Z["marker"] + ZC[k_]))
MK_OUT = MSZ["line"] + MSZ["line_edge"]          # outer marker diameter (pt): labels clear the markers, not the line
#                                                  (v8 c: kept, so the labels keep their v8.1 positions)
fig.canvas.draw()
LAB_A = {}
LAB_A["seed"] = label_along(ax, S.PAIR_LABEL["seed"], [MED_POLY["seed"]], float(os.environ.get("FE_XSEED", 2048)),
                            "above", "center", "direct", PT["seed"], gap_pt=1.2, lw=MK_OUT)
LAB_A["vision"] = label_along(ax, S.PAIR_LABEL["vision"], [MED_POLY["vision"]], float(os.environ.get("FE_XVIS", 16000)),
                              "above", "right", "direct", PT["vision"], gap_pt=1.2, lw=MK_OUT)
LAB_A["sent"] = label_along(ax, "sentence encoders", BUND["sent"] + [MED_POLY["sent"]],
                            float(os.environ.get("FE_XSENT", 17000)), "above", "right", "direct", PT["sent"],
                            gap_pt=1.2, lw=MK_OUT)
LAB_A["vl"] = label_along(ax, "language models", BUND["sent"] + [MED_POLY["vl"], MED_POLY["sent"]],
                          float(os.environ.get("FE_XLM", 300)), "below", "left", "direct", PT["vl"],
                          gap_pt=1.2, lw=MK_OUT)

# ---- (b) -------------------------------------------------------------------------------------------------------------
ax = axB
ROWS = {}                     # row y (data units, top to bottom); v8: no group headers, a gap between the groups
for i, e in enumerate(ENC):
    ROWS[("sent", e)] = float(i)
for i, l in enumerate(LMS):
    ROWS[("vl", l)] = 3.6 + i
YLIM_B = (6.15, -0.55)
XLIM_B = (-0.057, 0.057)                 # v8: narrower panel, so the empty margin is trimmed
ax.set_xlim(*XLIM_B)
ax.set_ylim(*YLIM_B)
BAND_ART = ax.axvspan(-BAND, BAND, color=S.NULL_FILL, lw=0, zorder=Z["band"])
add(BAND_ART)
GSPAN = {g: (min(y for (gg, _), y in ROWS.items() if gg == g) - 0.45,
             max(y for (gg, _), y in ROWS.items() if gg == g) + 0.45) for g in ("sent", "vl")}
for g, (y_lo, y_hi) in GSPAN.items():
    add(ax.plot([0, 0], [y_lo, y_hi], color=S.NULL_LINE, lw=LW["zero"], ls=S.LS["dashed"], zorder=Z["ref"]))
DOT_MS, DOT_EDGE = MSZ["scatter"], MSZ["scatter_edge"]     # point-cloud dots (Figure 2 c size), white edge
DOT_OUT = DOT_MS + DOT_EDGE   # outer diameter (pt)
fig.canvas.draw()
PT_PER_X = (WB * 72.0) / (XLIM_B[1] - XLIM_B[0])          # pt per data unit along x
PT_PER_Y = (AXH * 72.0) / (YLIM_B[0] - YLIM_B[1])         # pt per row unit along y


def swarm(xs, step_pt):
    """Deterministic beeswarm: in order of x, each point takes the smallest vertical offset on a grid of step_pt
    (0, +s, -s, +2s, -2s, ...) at which it keeps at least one outer marker diameter (plus 0.4 pt) from every point
    already placed, so an isolated point stays on its row line."""
    order = np.argsort(xs, kind="stable")
    placed, off = [], np.zeros(len(xs))
    jmax = int((0.5 * PT_PER_Y - 0.5 * DOT_OUT) / step_pt + 1e-9)       # v8: as far as the row allows
    cands = [0.0] + [s * step_pt * j for j in range(1, jmax + 1) for s in (1, -1)]
    for i in order:
        for o in cands:
            if all(np.hypot((xs[i] - xs[j]) * PT_PER_X, o - off[j]) >= DOT_OUT + 0.4 for j in placed):
                off[i] = o
                break
        else:
            raise AssertionError(("swarm: no free offset", list(xs), PT_PER_X, PT_PER_Y, jmax))
        placed.append(i)
    return off


SWARM = {}
for (grp, t), y0 in ROWS.items():
    D = NEW if grp == "sent" else OLD
    ps = sorted(p for p in D if D[p]["text"] == t)
    assert len(ps) == 5
    xs = np.array([D[p]["sum"] for p in ps])
    off = swarm(xs, step_pt=0.05)                # v8: finer grid (the narrower panel packs the dots closer)
    assert np.max(np.abs(off)) <= 0.5 * PT_PER_Y - 0.5 * DOT_OUT + 1e-9, ("swarm leaves its row", t, off)
    ys = y0 - off / PT_PER_Y          # y axis inverted: a positive offset (pt, up) is a smaller y
    add(ax.plot(xs, ys, ls="none", marker="o", ms=DOT_MS, mfc=PC[grp], mec="white", mew=DOT_EDGE, zorder=Z["marker"]))
    SWARM[t] = {p: {"x": float(x), "y_row": float(y0), "offset_pt": float(o)} for p, x, o in zip(ps, xs, off)}
GMED = {"sent": med_new, "vl": med_old}
for grp, x0 in GMED.items():
    y_lo, y_hi = GSPAN[grp]
    add(S.underlay(ax, [x0, x0], [y_lo, y_hi], LW["median_bar"], Z["interval"] + 0.5, extra=LW["median_halo"],
                   solid_capstyle="butt"))
    add(ax.plot([x0, x0], [y_lo, y_hi], color=PC[grp], lw=LW["median_bar"], solid_capstyle="butt",
                zorder=Z["interval"] + 0.5))
ax.set_yticks([ROWS[k] for k in ROWS], [ENC_LABEL[t] if g == "sent" else LM_LABEL[t] for g, t in ROWS])
ROW_LABEL_INK = True              # v8 c: INK row labels (v8.1: group colour; 'BLOOMZ' then read in the vl blue)
for tl, (g, t) in zip(ax.get_yticklabels(), ROWS):
    tl.set_color(INK if ROW_LABEL_INK else PT[g])   # v8: the group colour replaced the group headers
ax.tick_params(axis="y", which="both", left=False, pad=3.0)
ax.spines["left"].set_visible(False)
ax.set_xticks([-0.05, 0.0, 0.05], [S.num(-0.05), "0", S.num(0.05)])
ax.set_xlabel("sum $e_k+e_m$")              # v8: shorter (the caption names the elasticities)

# ---- (c) critic v8.1: X3b medians at k = 10, both text sides, mid depth (open) and last layer (filled) -----------------
ax = axC
setup_m_axis(ax, True)
MKC = {"sent": dict(marker="s", ms=MSZ["line"] - 0.4), "vl": dict(marker="o", ms=MSZ["line"])}   # text side: shape
FILL = {"mid": dict(mfc="white", mew=MSZ["open_small_mew"]), "last": dict(mec="white", mew=MSZ["line_edge"])}
ZB = {"vl_mid": 0.0, "sent_mid": 0.1, "vl_last": 0.2, "sent_last": 0.3}
C_POLY = {}
for key in ZB:
    grp, depth = key.split("_")
    xs_, ys_ = np.array(MS, float), np.array(CURB[key]["median_s_k10"])
    C_POLY[key] = (xs_, ys_)
    add(ax.plot(xs_, ys_, color=PC[grp], lw=LW["data"], zorder=Z["data"] + ZB[key]))
    mk = {**MKC[grp], **FILL[depth], ("mfc" if depth == "last" else "mec"): PC[grp]}
    add(ax.plot(xs_, ys_, ls="none", zorder=Z["marker"] + ZB[key], **mk))
STAR = dict(marker="*", ms=10.0, mfc="white", mec=S.NULL_LINE, mew=1.0)
add(ax.plot([M_SCHNAUS], [SCHNAUS], ls="none", zorder=Z["marker"] + 1.0, **STAR))
fig.canvas.draw()
MK_OUT_C = MSZ["line"] + MSZ["line_edge"]
LAB_C = {}
LAB_C["last"] = label_along(ax, "last layer", [C_POLY["sent_last"]], float(os.environ.get("FE_XLAST", 12000)),
                            "above", "right", "direct", INK, gap_pt=1.8, lw=MK_OUT_C)
LAB_C["mid"] = label_along(ax, "mid depth", [C_POLY["vl_mid"]], float(os.environ.get("FE_XMID", 290)),
                           "below", "left", "direct", INK, gap_pt=1.6, lw=MK_OUT_C)

# ---- v8: bold ticks and axis titles; then the panel letters on one baseline at each panel's left ink edge ----------------
S.finalize(fig)
fig.canvas.draw()
R0 = fig.canvas.get_renderer()
AXS = {"a": axA, "b": axB, "c": axC}
TB = {k_: a.get_tightbbox(R0) for k_, a in AXS.items()}
LEFT = {k_: min(TB[k_].x0, AXS[k_].get_window_extent(R0).x0) / fig.dpi for k_ in AXS}
GAP_AB = LEFT["b"] - TB["a"].x1 / fig.dpi
GAP_BC = LEFT["c"] - TB["b"].x1 / fig.dpi
N["ink_extents_in"] = {k_: [round(v / fig.dpi, 3) for v in TB[k_].extents] for k_ in AXS}
print("INK", N["ink_extents_in"], "GAPS", round(GAP_AB, 3), round(GAP_BC, 3))
assert GAP_AB >= 0.12 and GAP_BC >= 0.12, ("panels too close (v6 rule: >= 0.12 in)", GAP_AB, GAP_BC)
assert TB["c"].x1 / fig.dpi <= W - 0.02, ("(c) leaves the figure", TB["c"].x1 / fig.dpi)
N["gap_ink_in"] = {"a_to_b": round(GAP_AB, 3), "b_to_c": round(GAP_BC, 3)}
LETTER_Y = H - 0.02
LET = [S.panel_letter(fig, k_, max(0.01, LEFT[k_] - 0.01), LETTER_Y) for k_ in ("a", "b", "c")]
N["letters_x_in"] = {k_: round(max(0.01, LEFT[k_] - 0.01), 3) for k_ in LEFT}
import matplotlib.text as _mtext  # noqa: E402
_muted = [t.get_text() for t in fig.findobj(_mtext.Text) if t.get_visible() and t.get_text().strip()
          and matplotlib.colors.to_hex(t.get_color()).upper() == S.MUTED.upper()]
assert not _muted, ("MUTED text equals the sentence-encoder colour under protanopia", _muted)

# ================================================================== save and check
OUT_PDF, OUT_PNG = f"{HERE}/F_e0_app.pdf", f"{HERE}/F_e0_app.png"
VIEW100, VIEWS = f"{HERE}/_view_print_100dpi.png", f"{HERE}/_views_cvd_grey.png"
fig.savefig(OUT_PDF)
chk = S.check(fig, OUT_PDF)
S.print_view(OUT_PDF, OUT_PNG, VIEW100)
S.views(OUT_PNG, VIEWS)
pf = subprocess.run(["pdffonts", OUT_PDF], capture_output=True, text=True, check=True).stdout
assert all(("TrueType" in ln and " yes " in ln) for ln in pf.strip().splitlines()[2:]), pf
info = subprocess.run(["pdfinfo", OUT_PDF], capture_output=True, text=True, check=True).stdout
psz = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
page_in = [round(float(psz.group(1)) / 72, 4), round(float(psz.group(2)) / 72, 4)]
assert page_in == [W, H], page_in
im = PIL.Image.open(OUT_PNG)
assert list(im.size) == [int(round(W * 200)), int(round(H * 200))], im.size      # v8: the PNG is pdftoppm at 200 dpi
vim = PIL.Image.open(VIEW100)
assert list(vim.size) == [int(round(W * 100)), int(round(H * 100))], vim.size
N["pdf_checks"] = {**chk, "pdffonts": pf.strip().splitlines(), "page_size_in": page_in, "png_px": list(im.size),
                   "view100_png_px": list(vim.size), "view": "pdftoppm -r 200 (PNG) and -r 100 (print view) of the PDF"}
# ================================================================== layout audit on rendered ink (300 dpi)
# as the v5 F_neighbors maker: each group of artists is rendered alone; overlaps counted in pixels after dilating the
# obstacle by about 0.7 pt: annotation text against data ink, against the frame, and against every other annotation
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt  # noqa: E402

fig.canvas.draw()
LEAVES = [a for a in fig.findobj() if a is not fig and not a.get_children()]
VIS0 = {id(a): a.get_visible() for a in LEAVES}
DPI_A = 300


def ink(group):
    gid = {id(a) for a in group}
    for a in LEAVES:
        a.set_visible(VIS0[id(a)] and id(a) in gid)
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=DPI_A, transparent=True)
    for a in LEAVES:
        a.set_visible(VIS0[id(a)])
    wpx, hpx = int(round(W * DPI_A)), int(round(H * DPI_A))
    return np.frombuffer(buf.getvalue(), np.uint8).reshape(hpx, wpx, 4)[..., 3] > 40


def leaves_of(arts):
    out = []
    for a in arts:
        out.extend([q for q in a.findobj() if not q.get_children()] or [a])
    return out


ann_ids = {id(t) for t in ANN}
data_leaves = leaves_of(DATA)
data_ids = {id(a) for a in data_leaves}
frame = [a for a in LEAVES if id(a) not in ann_ids and id(a) not in data_ids
         and a not in (fig.patch, axA.patch, axB.patch, axC.patch)]
M_data, M_frame = ink(data_leaves), ink(frame)
st = np.ones((7, 7), bool)
D_data, D_frame = binary_dilation(M_data, st), binary_dilation(M_frame, st)
band_ids = {id(a) for a in leaves_of([BAND_ART])}
M_data_nb = ink([a for a in data_leaves if id(a) not in band_ids])
D_data_nb = binary_dilation(M_data_nb, st)
M_band_in = binary_erosion(ink(leaves_of([BAND_ART])), st)
M_ann = [ink([t]) for t in ANN]
PAN = {id(axA): "a", id(axB): "b", id(axC): "c"}
TAG = [f'({PAN.get(id(getattr(t, "axes", None)), "fig")}) {t.get_text()}' for t in ANN]
assert len(set(TAG)) == len(TAG), TAG
audit = {"dpi": DPI_A, "dilation_px": 3, "text_vs_data": {}, "text_vs_frame": {}, "text_vs_text": [], "on_fill_labels": {}}
onfill_idx = {i for i, t in enumerate(ANN) if t in ONFILL}
for i, (tg, Mi) in enumerate(zip(TAG, M_ann)):
    Dd = D_data_nb if i in onfill_idx else D_data
    audit["text_vs_data"][tg] = int((Mi & Dd).sum())
    audit["text_vs_frame"][tg] = int((Mi & D_frame).sum())
    if i in onfill_idx:
        audit["on_fill_labels"][tg] = {"fill": S.NULL_FILL, "ink_px": int(Mi.sum()),
                                       "ink_px_outside_band_eroded_0_7pt": int((Mi & ~M_band_in).sum())}
        assert audit["on_fill_labels"][tg]["ink_px_outside_band_eroded_0_7pt"] == 0, audit["on_fill_labels"][tg]
dist_data, dist_data_nb, dist_frame = (distance_transform_edt(~M_data), distance_transform_edt(~M_data_nb),
                                       distance_transform_edt(~M_frame))
audit["clearance_pt"] = {tg: {"data": round(float((dist_data_nb if i in onfill_idx else dist_data)[Mi].min()) * 72 / DPI_A, 2),
                              "frame": round(float(dist_frame[Mi].min()) * 72 / DPI_A, 2)}
                         for i, (tg, Mi) in enumerate(zip(TAG, M_ann))}
for i in range(len(ANN)):
    Di = binary_dilation(M_ann[i], st)
    for j in range(i + 1, len(ANN)):
        n_ = int((Di & M_ann[j]).sum())
        if n_:
            audit["text_vs_text"].append([TAG[i], TAG[j], n_])
# data marks against each other where they must stay apart: the dots of (b) (one ink render per row)
fig.canvas.draw()
R = fig.canvas.get_renderer()
ftxt = [a for a in frame if isinstance(a, matplotlib.text.Text) and a.get_visible() and a.get_text().strip()]
audit["frame_text_box_overlaps"] = [[a.get_text(), b.get_text()] for i, a in enumerate(ftxt) for b in ftxt[i + 1:]
                                    if a.get_window_extent(R).overlaps(b.get_window_extent(R))]
# frame texts against the annotations (row labels of (b) against the headers, letters against anything)
M_ftxt = ink(ftxt)
audit["frame_text_vs_annotations_px"] = {tg: int((binary_dilation(Mi, st) & M_ftxt).sum()) for tg, Mi in zip(TAG, M_ann)}
fig.canvas.draw()                      # v8: fresh renderer (ink() saved at 300 dpi, which left R at that dpi)
R = fig.canvas.get_renderer()
fb = fig.bbox
audit["text_outside_figure"] = [[t.get_text(), [round(v / fig.dpi, 3) for v in t.get_window_extent(R).extents]]
                                for t in ANN + ftxt
                                if (lambda b: b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5
                                    or b.y1 > fb.y1 + 0.5)(t.get_window_extent(R))]


def edge_margins(Mx, x0, x1, y0, y1, dpi):
    yy_, xx_ = np.nonzero(Mx)
    if not len(xx_):
        return None
    f = 72.0 / dpi
    return {"left": round(float(xx_.min() - x0) * f, 3), "right": round(float(x1 - 1 - xx_.max()) * f, 3),
            "top": round(float(yy_.min() - y0) * f, 3), "bottom": round(float(y1 - 1 - yy_.max()) * f, 3)}


def pdf_ink_margins(pdf, dpi=1200, pad_pt=12, near_pt=4.0, grow_pt=1.5):
    """Ink of the PDF itself (ghostscript, page padded by pad_pt): nothing outside the MediaBox, margins per side, and
    the ink margin of every PDF word within near_pt of an edge (copied from the v5 F_neighbors maker)."""
    mb = [float(v) for v in re.search(rb"MediaBox \[ *([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)",
                                      open(pdf, "rb").read()).groups()]
    assert mb[:2] == [0.0, 0.0] and abs(mb[2] - W * 72) < 1e-6 and abs(mb[3] - H * 72) < 1e-6, mb
    png = subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pngalpha", f"-r{dpi}",
                          f"-dDEVICEWIDTHPOINTS={mb[2] + 2 * pad_pt:g}", f"-dDEVICEHEIGHTPOINTS={mb[3] + 2 * pad_pt:g}",
                          "-dFIXEDMEDIA", "-sOutputFile=-", "-c",
                          f"<</PageOffset [{pad_pt} {-pad_pt}]>> setpagedevice", "-f", pdf],
                         check=True, capture_output=True).stdout
    a = np.array(PIL.Image.open(io.BytesIO(png)).convert("RGBA")).astype(int)
    al, mn = a[..., 3], a[..., :3].min(-1)
    pg = (al > 250) & (mn >= 250)
    yy_, xx_ = np.nonzero(pg)
    x0, x1, y0, y1 = xx_.min(), xx_.max() + 1, yy_.min(), yy_.max() + 1
    s_ = dpi / 72.0
    assert abs((x1 - x0) / s_ - mb[2]) < 0.1 and abs((y1 - y0) / s_ - mb[3]) < 0.1
    inkm = (al > 8) & (mn < 245)
    outm = inkm.copy()
    outm[y0:y1, x0:x1] = False
    txt = subprocess.run(["pdftotext", "-bbox", pdf, "-"], check=True, capture_output=True, text=True).stdout
    words = [(w, *map(float, b)) for *b, w in re.findall(
        r'<word xMin="([-\d.]+)" yMin="([-\d.]+)" xMax="([-\d.]+)" yMax="([-\d.]+)">(.*?)</word>', txt)]
    near = []
    for w, bx0, by0, bx1, by1 in words:
        if min(bx0, by0, mb[2] - bx1, mb[3] - by1) > near_pt:
            continue
        c0, c1 = int(np.floor(x0 + (bx0 - grow_pt) * s_)), int(np.ceil(x0 + (bx1 + grow_pt) * s_))
        r0, r1 = int(np.floor(y0 + (by0 - grow_pt) * s_)), int(np.ceil(y0 + (by1 + grow_pt) * s_))
        sub = np.zeros_like(inkm)
        sub[max(r0, 0):r1, max(c0, 0):c1] = inkm[max(r0, 0):r1, max(c0, 0):c1]
        mg = edge_margins(sub, x0, x1, y0, y1, dpi)
        if mg is not None:
            near.append({"word": w, "nearest_side": min(mg, key=mg.get), "ink_margin_pt": mg[min(mg, key=mg.get)]})
    return {"dpi": dpi, "ink_px_outside": int(outm.sum()), "margin_pt": edge_margins(inkm, x0, x1, y0, y1, dpi),
            "words_near_edge": sorted(near, key=lambda q: q["ink_margin_pt"]), "n_words_in_pdf": len(words)}


audit["pdf_ink_margin_pt"] = pdf_ink_margins(OUT_PDF)
assert audit["pdf_ink_margin_pt"]["ink_px_outside"] == 0, audit["pdf_ink_margin_pt"]
assert min(audit["pdf_ink_margin_pt"]["margin_pt"].values()) >= 0.2, audit["pdf_ink_margin_pt"]
bad = {k: v for k, v in {**{"data:" + k: v for k, v in audit["text_vs_data"].items()},
                         **{"frame:" + k: v for k, v in audit["text_vs_frame"].items()},
                         **{"frametext:" + k: v for k, v in audit["frame_text_vs_annotations_px"].items()}}.items() if v}
audit["result"] = "clean" if not bad and not audit["text_vs_text"] and not audit["frame_text_box_overlaps"] \
    and not audit["text_outside_figure"] else {"collisions": bad, "text_vs_text": audit["text_vs_text"],
                                               "frame_box": audit["frame_text_box_overlaps"],
                                               "outside": audit["text_outside_figure"]}
N["layout_audit"] = audit
assert audit["result"] == "clean", (audit["result"], audit["clearance_pt"])   # v8: asserted (v6 recorded it)
N["annotations"] = [{"text": t.get_text(), "rotation_deg": round(float(t.get_rotation()), 2), "color": t.get_color(),
                     "size_pt": t.get_fontsize(), "weight": t.get_fontweight()} for t in ANN]
N["swarm_b"] = {"rule": "deterministic beeswarm per row, offsets in pt (up = positive), outer diameter %.1f pt + 0.4 pt "
                        "clearance" % DOT_OUT, "points": SWARM}
N["geometry_in"] = {"figure": [W, H], "a": [XA, BOT, WA, AXH], "b": [XB, BOT, WB, AXH], "c": [XC, BOT, WC, AXH],
                    "ylim_a": list(YLIM_A), "xlim_a": list(XLIM_A), "xlim_b": list(XLIM_B), "ylim_b": list(YLIM_B),
                    "rows_b": {f"{g}|{t}": y for (g, t), y in ROWS.items()}, "heads_b": "none (v8: row labels in the group colour; v8 c: row labels INK)" if ROW_LABEL_INK else "none (v8: row labels in the group colour)",
                    "group_spans_b": GSPAN, "schnaus_star_c": [M_SCHNAUS, SCHNAUS]}
N["style"] = {"module": "hrh_style_v8", "sha256": STYLE_SHA, "c_markers": {"text_side": MKC, "depth": FILL, "star": STAR},
              "class_colours": PC, "class_text": PT, "class_names": {"seed": "seed", "vision": "vision",
                                                                     "vl": "language models (vision-language, tokens)",
                                                                     "sent": "sentence encoders (vision-language)"},
              "figure2_reuse": "seed/vision/vl colours S.PAIR, text S.PAIR_TEXT; medians thick (LW data_emph) on a white "
                               "underlay" + (" with filled circles (MS line, white edge MS line_edge)" if MEDIAN_MARKERS_A
                                             else " without markers (v8 c; v8.1 drew filled circles)")
                               + "; single pairs LW data_thin opaque, as ~/hrh_v2_figs/v5/F_neighbors/make_F_neighbors.py (a)",
              "b_dots": [DOT_MS, DOT_EDGE], "b_median_bar": [LW["median_bar"], LW["median_halo"]],
              "band_fill": S.NULL_FILL, "zero_line": [S.NULL_LINE, LW["zero"], "dashed"]}
# ================================================================== numbers for the Results text (not on the figure)
LOO_MPNET = float(np.median([NEW[p]["sum"] for p in NEW if NEW[p]["text"] != "mpnet"]))
WITHIN_VIT = {}
for v in VITS:
    enc_l = [CM_ for CM_ in (A["per_pair_obs_k10_sentence"]["values"][f"{v}__{e}"][1024] for e in ENC)]
    lm_l = [float(cp11[(cp11.pair == f"{v}__{l}") & (cp11.k == 10) & (cp11.m == 1024)].obs.iloc[0]) for l in LMS]
    WITHIN_VIT[v] = {"encoders": [min(enc_l), max(enc_l)], "language_models": [min(lm_l), max(lm_l)],
                     "every_encoder_above_every_lm": bool(min(enc_l) > max(lm_l))}
assert all(d["every_encoder_above_every_lm"] for d in WITHIN_VIT.values())
assert VER["descriptive"]["P3a_level"]["P"]["cross-modal-sentence"]["n"] == 15
RES = [
    {"panel": "a", "id": "class_median_k10_ends", "value": {k_: [CM[k_][256], CM[k_][16384]] for k_ in CM},
     "printed": "sentence encoders 0.105 -> 0.0132; language-model tokens 0.091 -> 0.0085 (m = 256 -> 16,384); "
                "seed 0.548 -> 0.535; vision 0.256 -> 0.109 (Figure 2)",
     "source": "panel_a.class_median_obs_k10 (= review_paper_numbers class_median_k10)"},
    {"panel": "a", "id": "class_median_k10_m1024", "value": {k_: CM[k_][1024] for k_ in CM},
     "printed": "0.059 (sentence encoders) between 0.043 (language-model tokens) and 0.189 (vision); seed 0.525",
     "source": "VERDICTS descriptive.P3a_level / review levels_13_3; plan 13.3 wording 'between'"},
    {"panel": "a", "id": "within_vit", "value": WITHIN_VIT,
     "printed": "for each of the 5 vision models every sentence encoder lies above every language model at k = 10, "
                "m = 1024 (e.g. SigLIP-B/16: 0.116-0.137 against 0.074-0.098)",
     "source": "E0_cells_P / S1_1_v11_cells_clip (single-caption arm P: the clean encoder comparison)"},
    {"panel": "a", "id": "median_slope_k10", "value": SLOPE[10],
     "printed": "-0.52 (sentence encoders), -0.57 (language-model tokens), -0.21 (vision), -0.01 (seed)",
     "source": "VERDICTS descriptive.P3f_ordering.10.median_slope"},
    {"panel": "a", "id": "falling_every_k", "value": N_FALL,
     "printed": "all 15 new pairs fall at k = 1, 10 and 100 (45 of 45)", "source": "E0_slopes_P slope_class; VERDICTS P3a"},
    {"panel": "b", "id": "P1_label_and_median", "value": {"label": P1["label"], "median_sum": med_new,
                                                          "threshold_abs_median": P1["thresholds"]["hold_abs_median"]},
     "printed": "P1 holds: median e_k + e_m = 0.0060 (frozen: |median| <= 0.03)", "source": "VERDICTS P1"},
    {"panel": "b", "id": "P1_within_band", "value": {"frac": P1["frac_abs_sum_le"], "range": [float(sums_new.min()),
                                                                                             float(sums_new.max())]},
     "printed": "15 of 15 pairs within +-0.05 (frozen: >= 80%); sums from -0.0086 to 0.0155",
     "source": "VERDICTS P1.frac_abs_sum_le, P1.per_pair"},
    {"panel": "b", "id": "P1_informative", "value": {"n": P1["n_informative"],
                                                     "e_m_range": [min(NEW[p]["e_m"] for p in NEW), max(NEW[p]["e_m"] for p in NEW)]},
     "printed": "15 of 15 pairs informative (|e_m| >= 0.10; e_m from -0.60 to -0.39)", "source": "VERDICTS P1"},
    {"panel": "b", "id": "P1_per_encoder", "value": P1["per_encoder_median_sum"],
     "printed": "per encoder: MPNet 0.0115, Contriever 0.0060, Qwen3-Embedding-0.6B 0.0058",
     "source": "VERDICTS P1.per_encoder_median_sum"},
    {"panel": "b", "id": "P1_leave_mpnet_out", "value": LOO_MPNET,
     "printed": "without MPNet (trained on data including COCO caption pairs) the median over the 10 other pairs is %.4f"
                % LOO_MPNET,
     "source": "derived here from VERDICTS P1.per_pair (descriptive; not a VERDICTS statistic)"},
    {"panel": "b", "id": "P1_draws_and_arms",
     "value": {"per_draw": VER["descriptive"]["robustness"]["P1_per_draw_labels"],
               "per_draw_median_sum": "-0.0108 to 0.0257 (E0_summary 3.2; review_recompute per_draw_median_sum)",
               "secondary": {k_: {"label": v[0], "median_sum": v[1]} for k_, v in SEC.items()}},
     "printed": "holds in each of the 5 orderings and in all 12 secondary cells: the 11 secondary arms of the new pairs "
                "(caption ladder D1-D4, five-caption averages A/Aall, centering C/CA/CAall, l2 L, dedup Pdd; median "
                "sums 0.0045 to 0.0151) and the 29 S1.1 pairs centered (ctrold, 0.0108)", "source": "VERDICTS descriptive.robustness.P1_per_draw_labels, secondary.P1"},
    {"panel": "b", "id": "reference_lm_tokens", "value": {"median": med_old, "range": [float(sums_old.min()),
                                                                                      float(sums_old.max())]},
     "printed": "S1.1 vision-language pairs with language-model tokens, same statistic: median 0.0044, range -0.0119 "
                "to 0.0131", "source": "panel_b.per_pair_lm_tokens_S11 (median = VERDICTS P3a S1.1-clip)"},
    {"panel": "b", "id": "disclosures",
     "printed": "k >= 10 restriction chosen after seeing S1.1 (VERDICTS P1.disclosure); with k = 1 included the median "
                "sum is 0.058 (k in {1,10,100}) and 0.091 (k in {1,10}) against 0.030 and 0.055 for language-model "
                "tokens (exploratory, review_explore_20261010T142122Z.json)",
     "value": {"disclosure": P1["disclosure"]}, "source": "VERDICTS P1.disclosure; review_explore (exploratory)"}]
for r in RES:
    assert "printed" in r
assert f"{LOO_MPNET:.4f}" == f"{LOO_MPNET:.4f}"
N["results_text_numbers"] = RES

# ---- v8: numbers that the dropped figure text and panel (c) hand to the Results text ---------------------------------
CF = {f: CUR[f] for f in CUR}
RES_V8 = [
    {"panel": "a", "id": "k_value", "value": 10, "printed": "k = 10 (was the on-figure note '$k=10$'; now in the caption)",
     "source": "design constant"},
    {"panel": "c", "id": "x3_levels_L100_10000_mpnet", "value": {f: CF[f]["L_100_10000"] for f in CF},
     "printed": "L(100, 10000) for DINOv2-B x MPNet: mid-depth CLS 0.077, mid-depth patch mean 0.132, last-layer CLS "
                "0.429, last-layer patch mean 0.406, pooling of Schnaus et al. 0.410; Schnaus et al. about 0.38 (0.385)",
     "source": "X3_results.json pairs[*].L_100_10000 (= X3_levels.parquet L_m10000, k = 100)"},
    {"panel": "c", "id": "x3_depth_factor", "value": CF["x3_cls_r10"]["L_100_10000"] / CF["dinov2_b"]["L_100_10000"],
     "printed": "the last-layer CLS token raises the level 5.6-fold over the mid-depth CLS token (patch mean 5.3-fold)",
     "source": "X3_results.json"},
    {"panel": "c", "id": "x3_shares", "value": {f: CF[f]["share_of_log_gap_0.38"] for f in CF},
     "printed": "shares of the log gap closed: 1.08 (last CLS), 1.04 (last patch mean), 1.05 (Schnaus pooling), 0.33 "
                "(mid patch mean); reading rule fixed before the scores: whole gap / part of the gap",
     "source": "X3_results.json binding_gap_reading_mpnet"},
    {"panel": "c", "id": "x3_slopes_k10", "value": {f: CF[f]["b10"] for f in CF},
     "printed": "slopes of ln s_N on ln m at k = 10: -0.28 to -0.30 at the last layer, -0.42 (mid CLS) and -0.40 (mid "
                "patch mean)", "source": "X3_results.json pairs[*].b10"},
    {"panel": "c", "id": "x3_scale_reading", "value": N["panel_c"]["p1_type_descriptive_all_12"],
     "printed": "e_k + e_m on k in {10, 100}: 0.002 to 0.014 for all 12 X3 pairs (median 0.008)",
     "source": "X3_results.json p1_type_descriptive"},
    {"panel": "c", "id": "x3_placement", "value": SCALE_DEV,
     "printed": "on every plotted feature, L(10, 1024) lies within 5% of L(100, 10000) (ln difference < 0.05), so the "
                "star is drawn at k/m = 0.01 on the k = 10 axis", "source": "X3_results.json pairs[*].L_10_1024"},
    {"panel": "c", "id": "x3_other_encoders", "value": OTHER_ENC,
     "printed": "Contriever and Qwen3-Embedding-0.6B in place of MPNet change each L(100, 10000) by less than 0.01",
     "source": "X3_results.json pairs"},
    {"panel": "c", "id": "x3_level_slope_line", "value": {f: PX3[f"{f}__mpnet"]["ctr_line_resid_b10"] for f in CF},
     "printed": "the centered level-slope line of P-S2b predicts about -0.05 at k = 10 for the last-layer pairs, against "
                "-0.28 to -0.31 observed (failure anticipated by the plan, B6)", "source": "X3_results.json pairs"}]
for r in RES_V8:                         # v8.1: the X3 features are no longer drawn; their numbers stay for the text
    if r["id"] == "x3_placement":
        r["printed"] = ("on each of the five X3 features of DINOv2-B (MPNet), L(10, 1024) lies within 5% of L(100, 10000) "
                        "(ln difference < 0.05), so the star is drawn at k/m = 0.01 on the k = 10 axis")
PVM = X3B["sets"]["B"]["per_vit_median_last"]
RES_V81 = [
    {"panel": "c", "id": "x3b_medians_k10_m1024", "value": {k_: CURB[k_]["median_s_k10"][MS.index(1024)] for k_ in CURB},
     "printed": "median agreement at k = 10, m = 1024 (15 pairs each): language models 0.043 (mid depth) -> 0.144 (last "
                "layer); sentence encoders under centering and five averaged captions 0.074 -> 0.419",
     "source": "X3B_cells.parquet (= X3B_results.json sets.A1 / sets.B median_<mid|last>_L10_1024)"},
    {"panel": "c", "id": "x3b_rise_factors",
     "value": {"A1_R_L10_1024": [X3B["sets"]["A1"]["median_R_L10_1024"], X3B["sets"]["A1"]["range_R_L10_1024"]],
               "B_R_L100_10000": [X3B["sets"]["B"]["median_R_L100_10000"], X3B["sets"]["B"]["range_R_L100_10000"]],
               "B_R_L100_10000_non_dinov2": [X3B["Q_b"]["median_R_non_dinov2"], X3B["Q_b"]["range_R_non_dinov2"]]},
     "printed": "the last layer raises every pair: median factor 2.8 (1.8 to 5.1) with language models at k = 10, "
                "m = 1024, and 4.3 (2.3 to 7.4) with sentence encoders at k = 100, m = 10,000 (3.4 without DINOv2-B)",
     "source": "X3B_results.json sets.A1, sets.B, Q_b"},
    {"panel": "c", "id": "x3b_per_vit_last_L100_10000", "value": {v: PVM[v]["L100_10000"] for v in PVM},
     "printed": "last-layer medians with the sentence encoders at k = 100, m = 10,000: OpenCLIP 0.447, DINOv2-B 0.429, "
                "AugReg 0.424, SigLIP 0.396, MAE 0.147",
     "source": "X3B_results.json sets.B.per_vit_median_last.<vit>.L100_10000"},
    {"panel": "c", "id": "x3b_slopes_b10",
     "value": {s_name: [float(np.median([r["b10"] for r in PPB[s_name]])),
                        float(np.median([r["b10_mid"] for r in PPB[s_name]]))] for s_name in PPB},
     "printed": "median slope b(10) last / mid: language models -0.42 / -0.57, sentence encoders -0.30 / -0.43; all 30 "
                "last-layer pairs <= -0.1", "source": "X3B_results.json per_pair.<A1|B>.b10, b10_mid"},
    {"panel": "c", "id": "x3b_sums", "value": {s_name: max(abs(r["sum"]) for r in PPB[s_name]) for s_name in PPB},
     "printed": "|e_k + e_m| on k in {10, 100} at most 0.016 (language models) and 0.019 (sentence encoders) over the 30 "
                "last-layer pairs (at most 0.021 over all 105 last-layer pairs, X3b review)",
     "source": "X3B_results.json per_pair.<A1|B>.sum"},
    {"panel": "c", "id": "x3b_matched_km", "value": MATCHED_KM_LAST,
     "printed": "at a shared k/m, ln L(100, 10 m') and ln L(10, m') differ by at most 0.081 at the last layer (within "
                "9%), which also places the star of Schnaus et al. at k/m = 0.01 on the k = 10 axis",
     "source": "X3B_review_checks.json matched_km.<set>|last.max_maxabs_10_100"}]
RES.extend(RES_V8 + RES_V81)
N["results_text_numbers"] = RES

# ================================================================== caption (number-free; v8, about 100 words)
CAPTION = (   # v8.1 (critic): (c) is the X3b panel; the reference moves to the clause it supports; about 100 words
    r"\caption{\textbf{Vision--language agreement with sentence encoders falls with the evaluation size but stays "
    r"close to stable at a fixed $k/m$ (Conjecture~\ref{conj:scale}), and deeper vision features raise its level.} "
    r"(a)~Class medians at $k=10$ as in Figure~\ref{fig:exp-neighbors}a, with sentence-encoder pairs as thin lines. "
    r"(b)~Sums of the elasticities in $k$ and $m$ on $k\in\{10,100\}$, with group medians as bars and the "
    r"preregistered tolerance in grey. (c)~Medians at $k=10$ with the middle (open) or last (filled) vision layer, for "
    r"sentence encoders (squares) and language models (circles); the star marks the value of "
    r"\citet{schnaus2026rosetta} for DINOv2-B and MPNet at equal $k/m$ (exploratory).}")
_body = re.sub(r"\\ref\{[^}]*\}|\\label\{[^}]*\}|\\citet\{[^}]*\}", "", CAPTION)
_nums = re.findall(r"(?<![A-Za-z_\\\d.])\d+(?:\.\d+)?", _body)
assert sorted(set(_nums)) == ["10", "100"], _nums      # design constants k = 10, k = 100 only
N["caption_proposed"] = CAPTION
_plain = re.sub(r"~?\\(ref|citet|label)\{[^}]*\}", " X ", CAPTION)          # a reference counts as one word
_plain = re.sub(r"\\(caption|textbf)\{", " ", _plain)
_words = len(re.sub(r"[{}$\\]", " ", _plain).split())
_bold = re.search(r"\\textbf\{(.*?)\(Conjecture", CAPTION, re.S).group(1)         # the claim without its reference
_bold_words = len(re.sub(r"[{}$\\~]", " ", _bold).split())
assert _words <= 110 and _bold_words <= 27, (_words, _bold_words)
_bold_all = re.search(r"\\textbf\{(.*?)\}\s\(a\)", CAPTION, re.S).group(1)            # v8.1: the whole claim
_bold_all_words = len(re.sub(r"~?\\ref\{[^}]*\}", " X ", _bold_all).replace("$", " ").split())
assert _words <= 105 and _bold_all_words <= 30, (_words, _bold_all_words)                  # v8.1: appendix, about 100
for _ban in (r"\bit\.", r"\bno\b", r"\bnever\b", r"\bonly\b", r"\bmust\b"):                 # house style scan
    assert not re.search(_ban, re.sub(r"\\[a-z]+\{[^}]*\}", "", CAPTION)), _ban
assert "\u2014" not in CAPTION and ": " not in CAPTION                  # house style: no em-dash, no colon splice
N["caption_word_count_approx"] = _words
N["caption_bold_word_count_approx"] = _bold_words
N["caption_bold_word_count_whole_claim"] = _bold_all_words
CAP_TEX = f"{HERE}/caption_v8.tex"
with open(CAP_TEX, "w") as f:
    f.write("% Caption v8.1 for F_e0_app (fig:app-e0; v8 maker, v8.1 critic, 2026-10-10). Number-free: only the design constants\n"
            "% k = 10 and k = 100 appear. Post hoc wording of preregistered results (P1 holds, VERDICTS_E0.json\n"
            "% 5b04e460...); panel (c) is the exploratory X3b follow-up (X3B_results.json 118a49bf..., readings fef078b8...;\n"
            "% critic v8.1) with the star of X3 (X3_REVIEW.md 39421ac0..., 0.385). The setting of (c) (block 5 against block 11\n"
            "% of five ViT-B; sentence encoders centered with five averaged captions, language models as in Figure 2) is named\n"
            "% in the Results\n"
            + f"% text (MOVED.md). Figure: F_e0_app.pdf (5.5 x {H:.2f} in); numbers: numbers.json.\n"
            + CAPTION + "\n\\label{fig:app-e0}\n")
N["caption_numbers_check"] = {"digits_outside_refs": sorted(set(_nums)), "meaning": "k = 10 and k = 100 (design)"}
# ================================================================== v8 c (consistency checker): asserts of this pass
CAPTION_V81_SHA = "d052cbf2cf33f46c3468d290d21ba33f0280a1c31f591badb2287749f78d5677"   # caption_v8.tex of v8.1
import matplotlib.colors as _mc  # noqa: E402
_a_marked = [l_ for l_ in axA.get_lines() if l_.get_marker() not in (None, "None", "none", "", " ")]
_b_lab = [(t_.get_text(), _mc.to_hex(t_.get_color()).upper()) for t_ in axB.get_yticklabels() if t_.get_text().strip()]
_c_open = {k_: _mc.to_hex(l_.get_markerfacecolor()).upper() for k_, l_ in (
    (k2, [q for q in axC.get_lines() if q.get_marker() in ("o", "s")][i2]) for i2, k2 in enumerate(ZB))}
CONS_C = {
    "(a) no marker on any line (class medians and single pairs are lines; filled = last layer in (c))":
        (not MEDIAN_MARKERS_A) and not _a_marked,
    "(a) thick class medians still drawn, four of them": sum(1 for l_ in axA.get_lines() if abs(l_.get_linewidth()
                                                            - LW["data_emph"]) < 1e-9) == 4,
    "(b) six row labels, all INK": len(_b_lab) == 6 and all(c_ == S.INK.upper() for _, c_ in _b_lab),
    "(b) dots keep the group colours": {_mc.to_hex(l_.get_markerfacecolor()).upper() for l_ in axB.get_lines()
                                        if l_.get_marker() == "o"} == {PC["sent"].upper(), PC["vl"].upper()},
    "(c) mid depth open (white face), last layer filled, as Figure 2 (c) and F.3": all(
        (v_ == "#FFFFFF") == k_.endswith("_mid") for k_, v_ in _c_open.items()),
    "caption unchanged from v8.1": sha(CAP_TEX) == "%s" % CAPTION_V81_SHA,
    "layout audit clean (asserted above)": audit["result"] == "clean",
}
assert all(CONS_C.values()), (CONS_C, _a_marked, _b_lab, _c_open)
N["consistency_v8c"] = {"date": "2026-10-10", "role": "consistency checker across the v8 figures",
                        "changes": ["(a) class medians without markers (v8.1: filled circles; filled means last layer in "
                                    "(c), Figure 2 (c) and F.3, and (a) holds mid-depth curves)",
                                    "(b) row labels INK (v8.1: group colour, so the family names read in the "
                                    "vision-language blue; Figure 3 and F.3 write BLOOMZ in S.FAM_TEXT indigo)"],
                        "asserts": CONS_C, "c_face_colours": _c_open, "b_row_labels": _b_lab,
                        "v8_1_files": "./_prev_consistency/ (maker, outputs, MOVED.md, caption of v8.1)"}
N["outputs"] = {OUT_PDF: sha(OUT_PDF), OUT_PNG: sha(OUT_PNG), VIEW100: sha(VIEW100), VIEWS: sha(VIEWS),
                CAP_TEX: sha(CAP_TEX)}
with open(f"{HERE}/numbers.json", "w") as f:
    json.dump(N, f, indent=1, default=float)
print(json.dumps({"pdf_checks": {k: N["pdf_checks"][k] for k in ("min_text_pt", "min_glyph_pt", "pdf_glyph_size_counts",
                                                                "n_type3", "fonts", "page_size_in", "png_px",
                                                                "low_contrast_text", "not_bold_tick_or_axis_labels",
                                                                "bold_text_with_regular_math",
                                                                "chromatic_data_colours", "off_palette_colours")},
                  "layout_audit": {k: audit[k] for k in ("result", "clearance_pt", "text_vs_text",
                                                         "frame_text_box_overlaps", "text_outside_figure")},
                  "pdf_margin": audit["pdf_ink_margin_pt"]["margin_pt"],
                  "palette": N["palette_check"], "outputs": N["outputs"], "asserts_max_abs_diff": DEV,
                  "caption_words": _words, "caption_bold_words": _bold_words,
                  "panel_c": {k: N["panel_c"][k] for k in ("schnaus", "scale_dev_ln", "depth_ratio_min_last_over_max_mid_by_m")},
                  "results": [(r["id"], r["printed"]) for r in RES]}, indent=1, default=float))
print("OK")
