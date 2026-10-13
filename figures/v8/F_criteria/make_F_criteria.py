"""F_criteria (v8, 2026-10-10; main-text Figure 3, label fig:exp-criteria, Section 5.3 sec:exp-criteria).

v8 = a copy of the v7 maker (~/hrh_v2_figs/v7/F_criteria/make_F_criteria.py, sha256 2451dceb..., kept byte-identical
next to this file as make_F_criteria_v7_copy.py) with the author's v8 rules (on v7 Figure 6, 2026-10-10, for EVERY
figure): (1) no title or header that states a setting (the headers '(b) CKA' and '(c) mKNN' name the plotted criterion,
not a setting, and stay); (2) the frozen v8 style ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (Paul Tol palette, sha256
asserted): CKA becomes Tol vibrant red #CC3311 (was Okabe-Ito vermilion), mKNN Tol vibrant blue #0077BB, the BLOOMZ word
Tol muted indigo, theory INK, references NULL_LINE; (3) tick labels and axis titles bold with bold math (S.finalize);
(4) a secondary comparison that the Results text states is not drawn: (a) shows mid depth only (the depth of (b, c));
the last-layer median and its 136 thin lines, the depth dodge and the key 'mid depth' / 'last layer' leave the figure
(FC_DEPTHS=0.5,1.0 restores them). Their values stay in numbers.json (panel_a.median_parts, median_cumulative) and in
the text (Section 5.3 Results: 0.68 at both depths, 0.84 after the affine fit, 0.0012, 0.034, 0.12; Appendix
app:exp-s12, Warps and Margins). staircase.py is the unchanged v6 module (sha asserted; shared with F_link).
Run: OMP_NUM_THREADS=1 taskset -c 0-5 nice -n 10 ~/hrh_v2_figs_venv/bin/python -B ~/hrh_v2_figs/v8/F_criteria/make_F_criteria.py

---- v7 header (unchanged) ----
F_criteria (v7, 2026-10-10; main-text Figure 3, label fig:exp-criteria, Section 5.3 sec:exp-criteria).

v7 redesign (role maker for Figure 3, author feedback 2026-10-10 on Figures 4 and 6: panels fill the row, little text
on the figure, short captions). Started from a copy of the v6 maker (~/hrh_v2_figs/v6/F_criteria/make_F_criteria.py,
sha256 recorded in numbers.json; copy kept here as make_F_criteria_v6_copy.py), of staircase.py (unchanged, so the
staircase convention shared with F_link stays intact) and of the frozen style hrh_style_v5_1.py (unchanged, sha asserted).
Changes against v6:
  1. (b, c) share ONE log y axis: each line (mean over the 17 vision models of the score minus its permutation-null
     mean, and its +-1 s.e.) is divided by its own value at 0.56B. On a log axis this division only moves a line
     vertically, so every line keeps its v6 shape (asserted: the plotted log-ratios equal the v6 log-means minus a
     constant). The broken axis of (b), the second y axis, the second y label and the axis-break marks are gone, and
     (c) carries no y tick labels. A dotted NULL_LINE at 1 marks 'no change from 0.56B'.
  2. The Tr column (header and four numbers) is dropped; the Results paragraph of Section 5.3 quotes the four trends.
  3. Criterion names move into the panel headers '(b) CKA', '(c) mKNN' (criterion text tones); the y labels
     'CKA - null mean' and 'mKNN - null mean' become the one label 'relative to 0.56B'; one shared x label.
  4. (a) shorter texts: theory label 'monotone link' (one line, was 'a monotone link / would reach 1 here'), key
     'mid depth' / 'last layer' (was 'median, mid depth' / 'median, last layer'); 'median' and 'pair' go to the caption.
  5. Height lowered from 1.75 in to H = 1.44 in, the smallest that passes every check below (at 1.43 in the y label of
     (a), kept in the shared staircase wording, ends 0.0009 in past the top edge); panel widths solved
     so that the three panels fill the 5.5 in row with MIN_PANEL_GAP between the panels.
  6. Panel letters: '(a)' sits right of the (long, shared-convention) y label of (a) instead of above it, because at
     the lower height the y label of (a) reaches the letter row; '(b)' follows the same rule (letter at the left edge
     of the y tick-label column) and '(c)', which has no tick labels, sits 0.02 in left of its axes; one common top.
Message: between pretrained vision and language models, the difference left after an affine change of scale is a
residual, not a monotone link; and the preprocessing can reverse the sign of a size trend.

Read-only inputs:
  ~/hrh_v2_stage1_20261008/results/S1_2/panel_main.parquet   m = 1024, 544 rows
  ~/hrh_v2_stage1_20261008/results/S1_2/s12_summary.json     warp medians (asserted), PE3 trends (asserted)
  ./hrh_style_v5_1.py (copy of ~/hrh_v2_figs/v5/_style/hrh_style_v5_1.py, sha asserted), ./staircase.py (copy of v6)
Writes only into this directory (or _trials/<FC_TRIAL>): F_criteria.pdf, F_criteria.png (300 dpi), numbers.json.
Run: nice taskset -c 0-3 ~/hrh_v2_figs_venv/bin/python -B ~/hrh_v2_figs/v7/F_criteria/make_F_criteria.py > run.log 2>&1
"""
import os, sys, json, hashlib, itertools, subprocess, re

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, rankdata  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox  # noqa: E402

STYLE_DIR = HERE                       # v7: the copy of the frozen style file in this folder
sys.path.insert(0, HERE)
import hrh_style_v5_1 as S51  # noqa: E402  (v8: the v7 style copy, kept and asserted; no longer used for drawing)
import staircase as SC  # noqa: E402
import importlib.util  # noqa: E402

# v8: the frozen v8 style, imported from its own folder after its sha256 is asserted
STYLE_V8_PATH = os.path.expanduser("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")
STYLE_V8_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_V8_PATH, "rb").read()).hexdigest() == STYLE_V8_SHA, "v8 style changed"
_spec8 = importlib.util.spec_from_file_location("hrh_style_v8", STYLE_V8_PATH)
S = importlib.util.module_from_spec(_spec8)
_spec8.loader.exec_module(S)
assert S.STYLE_SHA256 == STYLE_V8_SHA and S.STYLE_VERSION == "v8"
S.use()
STYLE_PATH = os.path.join(STYLE_DIR, "hrh_style_v5_1.py")
STYLE_SHA = "97d0ed5388b2b7a06829ce42645bc545070ecbd3e65549b5020568fecc9f41a6"
STAIRCASE_SHA_V6 = "56214a368e83179068a3a92a247d4140e9d137d2262d6bcf5c4d12c138291e4e"
V6_MAKER = os.path.expanduser("~/hrh_v2_figs/v6/F_criteria/make_F_criteria.py")
V7 = os.path.expanduser("~/hrh_v2_figs/v7/F_criteria")
FROZEN_V7 = {f"{V7}/make_F_criteria.py": "2451dcebbba506e47f4e8e7a552ad79e92a1424c2d3f15d86e3b304ded11f01a",
             f"{V7}/F_criteria.pdf": "714c247a2b1588f516adcc0b42a2df5f3db2c5edabeb27f40655360f2bd087fa",
             f"{V7}/F_criteria.png": "3bed980566d0485070a4d572bc45d33f707124a9551702c444ff2367c6830a31",
             f"{V7}/caption_v7.tex": "3844236cfd37ce81da67dc5af884c6ffae45423893cda62d7f023f24784531af",
             f"{V7}/numbers.json": "8193203f92f205353490594d0efcebcee8f95a5e4187dd85321cdf6f218dcb17",
             os.path.join(HERE, "make_F_criteria_v7_copy.py"): "2451dcebbba506e47f4e8e7a552ad79e92a1424c2d3f15d86e3b304ded11f01a"}
LW, MS, LS, Z = S.LW, S.MS, S.LS, S.Z


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


assert sha(STYLE_PATH) == STYLE_SHA == S51.STYLE_SHA256 and S51.STYLE_VERSION == "v5_1", "frozen style module changed"
assert sha(os.path.join(HERE, "staircase.py")) == STAIRCASE_SHA_V6, "staircase.py differs from the v6 module"
for _p, _h in FROZEN_V7.items():                    # v8: the v7 maker, render, caption and numbers (read only)
    assert sha(_p) == _h, ("v7 file changed", _p)
O7 = json.load(open(f"{V7}/numbers.json"))

TRIAL = os.environ.get("FC_TRIAL", "")
OUT = os.path.join(HERE, "_trials", TRIAL) if TRIAL else HERE
os.makedirs(OUT, exist_ok=True)
THIN_W = float(os.environ.get("FC_THIN_W", S.SPAN_TINT))  # share of white in the thin-line tint of (a) (spec: SPAN_TINT)

S1 = os.path.expanduser("~/hrh_v2_stage1_20261008/results/S1_2")
P_MAIN, P_SUM = f"{S1}/panel_main.parquet", f"{S1}/s12_summary.json"
PRE_CLIP, PRE_L2 = "prh_clip+l2", "l2"
DEPTH = {0.5: "mid depth", 1.0: "last layer"}
SUM = json.load(open(P_SUM))
d = pd.read_parquet(P_MAIN)
assert len(d) == 544 and (d["L.m"] == 1024).all() and (d["K"] == 500).all()

# =============================================================================================
# (a) explained shares, prh_clip+l2, both depths (in-sample, the values the 5.3 text quotes)   [v6, unchanged]
# =============================================================================================
A = d[d.pre == PRE_CLIP]
CUM, PARTS, PAIRS, MAXEX = {}, {}, {}, {}
for r in (0.5, 1.0):
    g = A[A.r == r].sort_values(["lm", "vit"]).reset_index(drop=True)
    assert len(g) == 136 and len(g[["lm", "vit"]].drop_duplicates()) == 136
    assert g.lm.nunique() == 8 and g.vit.nunique() == 17
    CUM[r] = SC.cumulative(g)
    PARTS[r] = {k: float(g["L.warpshare_expl_" + k].median()) for k in SC.STEPS + ("residual",)}
    W_ = SUM["warp"][f"r={r}|{PRE_CLIP}"]["expl"]
    for k in SC.STEPS + ("residual",):
        assert abs(PARTS[r][k] - W_[k][0]) < 1e-12, (r, k, PARTS[r][k], W_[k][0])
    MAXEX[r] = {k: float(g["L.warpshare_expl_" + k].max()) for k in ("iso_extra", "row_extra")}
    MAXEX[r]["min_residual"] = float(g["L.warpshare_expl_residual"].min())
    MAXEX[r]["max_residual"] = float(g["L.warpshare_expl_residual"].max())
    MAXEX[r]["frac_iso_extra_ge_0.10"] = float((g["L.warpshare_expl_iso_extra"] >= 0.10).mean())
    assert MAXEX[r]["frac_iso_extra_ge_0.10"] == W_["frac_pairs_iso_extra_ge_0.10"]
    PAIRS[r] = g[["lm", "vit", "lm_family", "vit_family"]]
MEDCUM = {r: np.median(CUM[r], axis=0) for r in CUM}
# spec asserts (PLAN.md 3.2, verified 2026-10-10)
assert round(PARTS[0.5]["iso_extra"], 4) == 0.0004 and round(PARTS[1.0]["iso_extra"], 4) == 0.0012
assert round(PARTS[0.5]["residual"], 4) == 0.1024 and round(PARTS[1.0]["residual"], 4) == 0.1234
assert [round(float(v), 3) for v in MEDCUM[0.5]] == [0.681, 0.872, 0.873, 0.898], MEDCUM[0.5]
for r in CUM:
    assert abs(MEDCUM[r][3] - (1 - PARTS[r]["residual"])) < 1e-12

# =============================================================================================
# (b, c) BLOOMZ size trends at mid depth under both preprocessings (uncensored calibrated scores)   [v6, unchanged]
# =============================================================================================
B = d[(d.r == 0.5) & (d.lm_family == "BLOOMZ")]
lms = B.drop_duplicates("lm").sort_values("lm_params")[["lm", "lm_params"]]
LM_ORDER, LM_PAR = list(lms.lm), [float(v) for v in lms.lm_params]
XB_SIZES = np.array(LM_PAR) / 1e9
VITS = sorted(B.vit.unique())
assert len(LM_ORDER) == 5 and len(VITS) == 17
PERMS = np.array(list(itertools.permutations(range(5))))


def grid(g, col):
    return g.pivot(index="lm", columns="vit", values=col).loc[LM_ORDER, VITS].to_numpy(float)


def trend(Y):
    """Tr = mean over vision models of Spearman(size rank, score); exact p over the 120 relabellings
    (copied from F_mst_app, which copied F2/make_F2.py)."""
    R = np.apply_along_axis(rankdata, 0, Y)

    def tr_of(x):
        return float(np.mean([spearmanr(x, R[:, j]).statistic for j in range(R.shape[1])]))
    T = np.array([tr_of(p) for p in PERMS])
    t = tr_of(np.arange(5))
    return t, float(np.mean(T >= t - 1e-12)), float(np.mean(np.abs(T) >= abs(t) - 1e-12))


CRITS = {"cka": ("cal.cka_biased", "Tr_raw", "raw"), "mknn": ("cal.s_N_k10", "Tr_mKNN", "mKNN")}
LINES = {}
for pre in (PRE_CLIP, PRE_L2):
    g = B[B.pre == pre]
    assert len(g) == 85
    pe3 = SUM["pe3"][f"r=0.5|{pre}|biased"]
    for c, (col, key, mkey) in CRITS.items():
        Y = grid(g, col + ".uncensored")
        assert np.allclose(Y, grid(g, col + ".s_obs") - grid(g, col + ".null_mean"), atol=1e-12)
        t, p1, p2 = trend(Y)
        ref = pe3[key]
        assert abs(t - ref["Tr"]) < 1e-12 and abs(p1 - ref["p_one"]) < 1e-12 and abs(p2 - ref["p_two"]) < 1e-12, (pre, c)
        t_scal = trend(grid(g, col + ".s_cal"))[0]
        allp = d[(d.r == 0.5) & (d.pre == pre)]
        assert abs(float(allp[col + ".uncensored"].mean()) - pe3["mean_uncensored"][mkey][0]) < 1e-12
        mean = Y.mean(1)
        se = Y.std(1, ddof=1) / np.sqrt(Y.shape[1])
        LINES[(pre, c)] = {"mean": mean, "se": se, "Tr": t, "p_one": p1, "p_two": p2, "Y": Y,
                           "EXPLORATORY_Tr_of_censored_s_cal": t_scal,
                           "n_vit_rising_monotone": int(sum(np.all(np.diff(Y[:, j]) > 0) for j in range(17))),
                           "n_vit_falling_monotone": int(sum(np.all(np.diff(Y[:, j]) < 0) for j in range(17))),
                           "ratio_largest_over_smallest_mean": float(mean[-1] / mean[0])}
assert [round(LINES[(PRE_CLIP, c)]["Tr"], 2) for c in ("cka", "mknn")] == [0.70, 0.84]
assert [round(LINES[(PRE_L2, c)]["Tr"], 2) for c in ("cka", "mknn")] == [-1.00, -0.92]

# v7: the plotted (b, c) values, each line divided by its own value at 0.56B (a vertical shift on the log axis)
for key_, L in LINES.items():
    assert (L["Y"] > 0).all(), key_                 # every score positive, so the log axis and the ratio are defined
    L["ratio"] = L["mean"] / L["mean"][0]
    L["se_ratio"] = L["se"] / L["mean"][0]
    assert L["ratio"][0] == 1.0
    shift = np.log10(L["ratio"]) - np.log10(L["mean"])
    assert np.allclose(shift, shift[0], atol=1e-12, rtol=0), key_   # same shape as v6, shifted on the log axis
    lo_, hi_ = np.log10(L["ratio"] - L["se_ratio"]), np.log10(L["ratio"] + L["se_ratio"])
    assert np.allclose(lo_ - np.log10(L["mean"] - L["se"]), shift[0]) and np.allclose(hi_ - np.log10(L["mean"] + L["se"]), shift[0])
    assert abs(L["ratio"][-1] - L["ratio_largest_over_smallest_mean"]) < 1e-12
# the sign of each plotted change from 0.56B to 7.1B agrees with the sign of its trend Tr
for key_, L in LINES.items():
    assert np.sign(np.log(L["ratio"][-1])) == np.sign(L["Tr"]), key_

# =============================================================================================
# figure: 5.5 in wide, exact print size, axes placed in inches
# =============================================================================================
CK, CKT = S.CRIT["cka"], S.CRIT_TEXT["cka"]
MK, MKT = S.CRIT["mknn"], S.CRIT_TEXT["mknn"]
INK = S.INK
THIN = S.tint(CK, THIN_W)
W = S.TEXTWIDTH_IN
H = float(os.environ.get("FC_H", "1.48"))       # v8: smallest passing height in 0.01 in steps with the bold y label of (a)
#   (1.46: the bold label ends 0.006 in past the top; 1.47: 0.0004 in past, fails the 0.002 in inside margin; v7: 1.44)
LET_DX = float(os.environ.get("FC_LET_DX", "0.0"))         # letter left edge relative to the y tick-label column (in)
H_V6 = 1.75
MIN_PANEL_GAP = 0.08                 # v6 assert, kept
PANEL_GAP = float(os.environ.get("FC_PANEL_GAP", "0.12"))   # target gap between neighbouring panels' outer extents (in)
Y0 = float(os.environ.get("FC_Y0", "0.34"))                 # common bottom of the data areas (in)
TOP_PAD = float(os.environ.get("FC_TOP", "0.16"))           # from the data top to the figure top (letters, headers)
AH = H - Y0 - TOP_PAD
XA = float(os.environ.get("FC_XA", "0.40"))                 # left edge of (a) axes
WBC = float(os.environ.get("FC_WBC", "1.00"))               # data width of (b) and of (c)
GAP_BC = float(os.environ.get("FC_GAP_BC", "0.15"))         # (b) right spine to (c) left spine (shared y)
RIGHT_PAD = 0.03
XC = W - RIGHT_PAD - WBC
XB = XC - GAP_BC - WBC
DASH_ON, DASH_OFF = [v * LW["data"] for v in matplotlib.rcParams["lines.dashed_pattern"]]   # plotted dashes (pt)
KEY_HANDLE_PT = 3 * DASH_ON + 2 * DASH_OFF   # key handle length in (a) and (c): three full dashes when dashed
# v8: (a) draws the depths in PLOT_DEPTHS (default mid depth only, the depth of (b, c)); with one depth there is no dodge
PLOT_DEPTHS = tuple(float(v) for v in os.environ.get("FC_DEPTHS", "0.5").split(","))
assert set(PLOT_DEPTHS) <= {0.5, 1.0} and 0.5 in PLOT_DEPTHS, PLOT_DEPTHS
DODGE = 0.10 if len(PLOT_DEPTHS) == 2 else 0.0   # (a) x offset of the two depths (steps); mid depth right, last layer left
XLIM_A = (0.68, 4.32)
YTOP_A = float(os.environ.get("FC_YTOP_A", "1.16"))
X_BEND = 3 + DODGE                   # the theory path bends at the mid-depth marker position of the global step
X_BRACKET = 4 + DODGE + 0.14
RES_GAP = 0.06                       # (a) bracket to its label (steps)
STEP_SET = os.environ.get("FC_STEP_LABELS", "words")   # 'words' (default, staged) or 'math' (fallback trial only)
MED_MK = {0.5: dict(marker="o", ms=MS["open_line"], mfc="white", mec=CK, mew=MS["open_line_mew"]),
          1.0: dict(marker="o", ms=MS["open_line"] + MS["open_line_mew"] - MS["line_edge"], mfc=CK, mec="white",
                    mew=MS["line_edge"])}
DODGES = {0.5: +DODGE, 1.0: -DODGE}
DODGES = {r: DODGES[r] for r in PLOT_DEPTHS}
XL = (0.42, 9.6)
XT = [float(v) for v in XB_SIZES]
XT_LAB = ["%.2g" % XT[0], "", "%.2g" % XT[2], "", "%.2g" % XT[4]]
LINE_LS = {PRE_CLIP: "-", PRE_L2: LS["dashed"]}
YT_BC = [0.5, 0.7, 1.0]
Y_LABEL_BC = "relative to 0.56B"
assert "%.2g" % XT[0] == "0.56", XT[0]
KEY_INSET_PT = 5.0                   # key of (b, c): inset from the y spine of (c) (v6 assert >= 3 pt kept)
KEY_LIFT_PT = 3.0                    # key of (b, c): lifted above the x spine of (c)


def build(WA):
    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")

    def axes_in(x0, y0, w, h):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    R = {"fig": fig}
    # ------------------------------------------------------------------------------------------ (a)
    axA = axes_in(XA, Y0, WA, AH)
    SC.setup(axA, S, labels=SC.STEP_LABELS if STEP_SET == "words" else SC.STEP_LABELS_MATH, xlim=XLIM_A,
             ylim=(0.0, YTOP_A))
    R["in_per_step_a"] = WA / (XLIM_A[1] - XLIM_A[0])
    drawn = {}
    for r in (1.0, 0.5):
        if r not in PLOT_DEPTHS:
            continue
        drawn[DEPTH[r]] = SC.draw_series(axA, S, CUM[r], CK, thin_color=THIN, dodge=DODGES[r], marker=MED_MK[r])
    R["drawn_a"] = drawn
    R["full_a"] = SC.full_share_line(axA, S, x1=X_BRACKET)   # dotted 'everything explained' line, closed by the bracket
    R["ref_a"], R["t_ref"] = SC.link_reference(axA, S, 2 + DODGES[0.5], float(MEDCUM[0.5][1]), x_mono=X_BEND, x_end=None,
                                               label="monotone link", label_xy=(X_BEND, 1.04),
                                               label_kw={"multialignment": "center"})
    R["br_a"], R["t_res"] = SC.residual_bracket(axA, S, X_BRACKET, float(MEDCUM[0.5][3]), 1.0, tick=0.06,
                                                label="residual", gap=RES_GAP)
    assert axA.get_ylabel() == SC.Y_LABEL
    keyA = None
    if len(PLOT_DEPTHS) == 2:                     # v8: the depth key exists only when both depths are drawn
        keyA = axA.legend(handles=[S.key_handle(color=CK, lw=LW["data_emph"], **MED_MK[0.5]),
                                   S.key_handle(color=CK, lw=LW["data_emph"], **MED_MK[1.0])],
                          labels=["mid depth", "last layer"],
                          loc="lower right", bbox_to_anchor=(1.0, 0.0), bbox_transform=axA.transAxes, frameon=False,
                          handlelength=KEY_HANDLE_PT / S.FS["key"], labelspacing=0.22, borderaxespad=0.15, borderpad=0.0,
                          handletextpad=0.5)
        for t_ in keyA.get_texts():
            t_.set_fontsize(S.FS["key"])
            t_.set_color(S.FC["key"])
    R["axA"], R["keyA"] = axA, keyA

    # ------------------------------------------------------------------------------------------ (b, c)
    axB = axes_in(XB, Y0, WBC, AH)
    axC = axes_in(XC, Y0, WBC, AH)
    lo = min(float((L["ratio"] - L["se_ratio"]).min()) for L in LINES.values())
    hi = max(float((L["ratio"] + L["se_ratio"]).max()) for L in LINES.values())
    YL = (lo / 10 ** 0.03, hi * 10 ** 0.03)
    R["ylim_bc"] = [float(v) for v in YL]

    def draw_trend(ax, pre, c, hue):
        L = LINES[(pre, c)]
        ax.errorbar(XB_SIZES, L["ratio"], yerr=L["se_ratio"], fmt="none", ecolor=hue, elinewidth=LW["errorbar"],
                    zorder=Z["interval"])
        ax.plot(XB_SIZES, L["ratio"], color=hue, lw=LW["data"], ls=LINE_LS[pre], zorder=Z["data"])
        ax.plot(XB_SIZES, L["ratio"], ls="none", zorder=Z["marker"],
                marker="o", ms=MS["line"], mew=MS["open_small_mew"], mfc="white", mec=hue)

    for ax, c, hue in ((axB, "cka", CK), (axC, "mknn", MK)):
        ax.axhline(1.0, color=S.NULL_LINE, lw=LW["ref"], ls=LS["dotted"], zorder=Z["ref"],
                   dash_capstyle="butt", solid_capstyle="butt")
        draw_trend(ax, PRE_CLIP, c, hue)
        draw_trend(ax, PRE_L2, c, hue)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*XL)
        ax.set_ylim(*YL)
        ax.minorticks_off()
        ax.set_xticks(XT, XT_LAB)
        ax.set_yticks(YT_BC, ["%g" % v for v in YT_BC])
        ax.minorticks_off()
    axC.tick_params(axis="y", labelleft=False)
    # key of the preprocessing (neutral INK, applies to (b) and (c)): lower left of (c), the empty region below the
    # dashed mKNN line; line-only handles (every line carries the same open circle, so only the line style is keyed)
    keyC = axC.legend(handles=[S.key_handle(color=INK, lw=LW["data"], ls=LINE_LS[PRE_CLIP]),
                               S.key_handle(color=INK, lw=LW["data"], ls=LINE_LS[PRE_L2])],
                      labels=["clip + ℓ2", "plain ℓ2"], loc="lower left",
                      bbox_to_anchor=(KEY_INSET_PT / 72 / WBC, KEY_LIFT_PT / 72 / AH), bbox_transform=axC.transAxes,
                      frameon=False, handlelength=KEY_HANDLE_PT / S.FS["key"], labelspacing=0.22, borderaxespad=0.0,
                      borderpad=0.0, handletextpad=0.45, numpoints=1)
    for t_ in keyC.get_texts():
        t_.set_fontsize(S.FS["key"])
        t_.set_color(S.FC["key"])
    R["axB"], R["axC"], R["keyC"] = axB, axC, keyC
    # v8: bold tick labels and axis titles (bold math) before any text is measured for the layout
    R["n_finalized"] = S.finalize(fig)

    fig.canvas.draw()
    REN = fig.canvas.get_renderer()
    INV = fig.dpi_scale_trans.inverted()

    # y label of (b), centred on the data height, right edge 0.035 in left of the tick labels
    x0_tl_b = min(t.get_window_extent(REN).transformed(INV).x0 for t in axB.get_yticklabels() if t.get_visible() and t.get_text())
    R["ylab_b"] = S.text(fig, (x0_tl_b - 0.035) / W, (Y0 + AH / 2) / H, Y_LABEL_BC, "label", rotation=90,
                         ha="right", va="center")
    # shared x label of (b, c): 'BLOOMZ parameters (B)', family name in the BLOOMZ colour (as v6 and Figure F.2 c)
    axB.set_xlabel("BLOOMZ parameters (B)")
    fig.canvas.draw()
    bbl = axB.xaxis.label.get_window_extent(REN).transformed(fig.transFigure.inverted())
    axB.set_xlabel("")
    pack = HPacker(children=[TextArea("BLOOMZ", textprops=S.text_kw("label", "BLOOMZ", color=S.FAM["bloomz"])),
                             TextArea("parameters (B)", textprops=S.text_kw("label", "parameters (B)"))],
                   align="baseline", pad=0, sep=2.0)
    R["xlab_pack"] = AnchoredOffsetbox(loc="upper center", child=pack, pad=0, borderpad=0, frameon=False,
                                       bbox_to_anchor=((XB + XC + WBC) / 2 / W, bbl.y1), bbox_transform=fig.transFigure)
    fig.add_artist(R["xlab_pack"])

    # panel letters: 10 pt bold, common top LET_Y; x = left edge of the panel's y tick-label column + LET_DX
    # ((c): left edge of its axes - 0.02 in); headers of (b, c) on the letters' baseline, criterion text tones
    fig.canvas.draw()
    LET_Y = H - 0.02
    x0_tl_a = min(t.get_window_extent(REN).transformed(INV).x0 for t in axA.get_yticklabels() if t.get_visible() and t.get_text())
    let_x = {"a": x0_tl_a + LET_DX, "b": x0_tl_b + LET_DX, "c": XC - 0.02}
    R["letters"] = {k_: S.panel_letter(fig, k_, let_x[k_], LET_Y) for k_ in ("a", "b", "c")}
    tmp = fig.text(0.0, 0.5, "(b)", fontsize=S.FS["letter"], fontweight=S.FW["letter"], va="baseline")
    fig.canvas.draw()
    asc = tmp.get_window_extent(REN).transformed(INV).y1 - 0.5 * H
    tmp.remove()
    base_y = LET_Y - asc
    R["headers"] = {}
    for k_, s_, tone in (("b", "CKA", CKT), ("c", "mKNN", MKT)):
        fig.canvas.draw()
        lb = R["letters"][k_].get_window_extent(REN).transformed(INV)
        R["headers"][k_] = S.text(fig, (lb.x1 + 0.04) / W, base_y / H, s_, "header", color=tone, ha="left",
                                  va="baseline")
    R["let_x"], R["let_y"], R["header_baseline_y"] = let_x, LET_Y, base_y
    R["YL"] = YL
    return R


# ---------------------------------------------------------------------------------------------- solve the row widths
# (a) is widened until its right extent (the 'residual' label) sits PANEL_GAP left of the left extent of (b) (y label)
WA = float(os.environ.get("FC_WA0", "2.0"))
for it in range(4):
    R = build(WA)
    fig = R["fig"]
    fig.canvas.draw()
    REN = fig.canvas.get_renderer()
    INV = fig.dpi_scale_trans.inverted()
    a_right = max(R["axA"].get_tightbbox(REN).transformed(INV).x1, R["t_res"].get_window_extent(REN).transformed(INV).x1,
                  R["t_ref"].get_window_extent(REN).transformed(INV).x1)
    b_left = min(R["axB"].get_tightbbox(REN).transformed(INV).x0, R["ylab_b"].get_window_extent(REN).transformed(INV).x0)
    slack = b_left - a_right - PANEL_GAP
    if abs(slack) < 1e-3:
        break
    # the residual label moves with the data x of the bracket: d(a_right)/d(WA) = (X_BRACKET + RES_GAP - x0) / range
    frac = (X_BRACKET + RES_GAP - XLIM_A[0]) / (XLIM_A[1] - XLIM_A[0])
    plt.close(fig)
    WA = WA + slack / frac
axA, axB, axC, keyA, keyC = R["axA"], R["axB"], R["axC"], R["keyA"], R["keyC"]
t_ref, t_res, YLAB_B, XLAB_PACK = R["t_ref"], R["t_res"], R["ylab_b"], R["xlab_pack"]
IN_PER_STEP_A = R["in_per_step_a"]
assert STEP_SET != "words" or IN_PER_STEP_A >= SC.MIN_IN_PER_STEP_WORDS - 1e-9, IN_PER_STEP_A

PDF, PNG = os.path.join(OUT, "F_criteria.pdf"), os.path.join(OUT, "F_criteria.png")
N_FINALIZED_2 = S.finalize(fig)                     # v8: last call before savefig (idempotent)
fig.savefig(PDF)
CHK = S.check(fig, PDF)
assert CHK["n_axis_titles_checked"] == 1 and CHK["n_tick_labels_checked"] > 0, CHK["n_axis_titles_checked"]
assert not CHK["not_bold_tick_or_axis_labels"] and not CHK["bold_text_with_regular_math"] and CHK["chromatic_data_colours"]
assert CHK["off_palette_colours"] == [], CHK["off_palette_colours"]
# v8: the staged PNG is the PDF rendered by poppler at 200 dpi (v7: matplotlib at 300 dpi); print view and CVD views
from PIL import Image  # noqa: E402
VIEW_PP = os.path.join(OUT, "_view_print_100dpi.png")
S.print_view(PDF, PNG, VIEW_PP)
_im = Image.open(PNG)
_im.load()
_im.save(PNG, dpi=(200, 200))
PNG_PX = list(Image.open(PNG).size)
assert PNG_PX == [int(round(W * 200)), int(round(H * 200))], PNG_PX
VIEWS = os.path.join(OUT, "_views_cvd_grey.png")
S.views(PNG, VIEWS)
assert abs(CHK["size_in"][0] - 5.5) < 1e-9 and abs(CHK["size_in"][1] - H) < 1e-9
assert H < H_V6
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
font_rows = [ln.split() for ln in pf.splitlines()[2:] if ln.strip()]
assert font_rows and all(row[-5] == "yes" for row in font_rows), pf
assert not any("Type 3" in ln for ln in pf.splitlines()), pf

# =============================================================================================
# layout checks (v6 checks, adapted where a v6 element was removed; see numbers.json asserts_adapted)
# =============================================================================================
fig.canvas.draw()
REN = fig.canvas.get_renderer()
INV = fig.dpi_scale_trans.inverted()
TB = []
for t in fig.findobj(mtext.Text):
    if t.get_visible() and t.get_text().strip():
        TB.append((t.get_text(), mtext.Text.get_window_extent(t, REN).transformed(INV), t))
# (1) text against text
clash = [(a, b) for i, (a, ba, _) in enumerate(TB) for (b, bb, _) in TB[i + 1:] if ba.overlaps(bb)
         and ba.intersection(ba, bb).width > 0.005 and ba.intersection(ba, bb).height > 0.005]
assert not clash, clash
# (2) every text inside the figure
outside = [(a, [round(v, 3) for v in bb.extents]) for a, bb, _ in TB
           if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]
assert not outside, outside
# v7: strictly inside, with at least 0.002 in to every figure edge (the v6 check above allows 0.001 in past an edge)
INSIDE_MARGIN_IN = float(min(min(bb.x0, bb.y0, W - bb.x1, H - bb.y1) for _, bb, _ in TB))
assert INSIDE_MARGIN_IN >= 0.002, INSIDE_MARGIN_IN
AXES = (axA, axB, axC)


def rot_frame(t, bb):
    ang = float(t.get_rotation()) % 360
    if not ang:
        return None
    tmp = fig.text(0, 0, t.get_text(), fontsize=t.get_fontsize(), fontproperties=t.get_fontproperties(),
                   linespacing=t.get_linespacing())
    ub = tmp.get_window_extent(REN).transformed(INV)
    tmp.remove()
    cxy = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
    th = np.radians(ang)
    return ub, cxy, np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])


ROT = {id(t): rot_frame(t, bb) for _, bb, t in TB}
# (3) text against data markers (outer disc)
mark_hits = []
for ax in AXES:
    pts_in = []
    for ln in ax.get_lines():
        if ln.get_marker() not in ("None", "none", "", None) and ln.get_visible():
            xy = INV.transform(ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float)))
            pts_in.append((xy, (ln.get_markersize() + ln.get_markeredgewidth()) / 2 / 72))
    for s_, bb, t in TB:
        fr = ROT[id(t)]
        for xy, rad in pts_in:
            if fr:
                ub, cxy, rot = fr
                q = (xy - cxy) @ rot.T
                hit = (np.abs(q[:, 0]) < ub.width / 2 + rad) & (np.abs(q[:, 1]) < ub.height / 2 + rad)
            else:
                hit = (xy[:, 0] > bb.x0 - rad) & (xy[:, 0] < bb.x1 + rad) & (xy[:, 1] > bb.y0 - rad) & (xy[:, 1] < bb.y1 + rad)
            if hit.any():
                mark_hits.append((s_, int(hit.sum())))
assert not mark_hits, mark_hits


# (4) text against drawn lines (data, thin pairs, references, bracket, error bars); dashed tested as solid
def drawn_polylines(ax):
    out = []
    for ln in ax.get_lines():
        if not ln.get_visible() or str(ln.get_linestyle()).lower() in ("none", ""):
            continue
        xy = np.asarray(ln.get_xydata(), float)
        if len(xy) >= 2:
            out.append(("line", ln.get_transform().transform(xy)))
    for c in ax.collections:
        if isinstance(c, LineCollection) and c.get_visible():
            for seg in c.get_segments():
                out.append(("collection", c.get_transform().transform(np.asarray(seg, float))))
    return out


def dense(P, step_px=0.25):
    pts = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(np.ceil(np.hypot(*(b - a)) / step_px)) + 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(pts)


line_hits = []
for ax in AXES:
    PL = [(kind, INV.transform(dense(P))) for kind, P in drawn_polylines(ax)]
    for s_, bb, t in TB:
        fr = ROT[id(t)]
        for kind, Q in PL:
            if fr:
                ub, cxy, rot = fr
                q = (Q - cxy) @ rot.T
                hit = (np.abs(q[:, 0]) < ub.width / 2) & (np.abs(q[:, 1]) < ub.height / 2)
            else:
                hit = (Q[:, 0] > bb.x0) & (Q[:, 0] < bb.x1) & (Q[:, 1] > bb.y0) & (Q[:, 1] < bb.y1)
            if hit.any():
                line_hits.append((s_, kind, ax is axA))
assert not line_hits, line_hits
# (5) panels do not overlap (data area + labels); v7: the shared x label belongs to (b, c) together
PANBOX = {}
for k_, axs, extra in (("a", (axA,), [t_res, t_ref]), ("b", (axB,), [YLAB_B, R["headers"]["b"]]),
                       ("c", (axC,), [R["headers"]["c"]])):
    bbs = [ax.get_tightbbox(REN).transformed(INV) for ax in axs] + [e.get_window_extent(REN).transformed(INV) for e in extra]
    PANBOX[k_] = [min(b.x0 for b in bbs), max(b.x1 for b in bbs)]
GAPS = {"a|b": PANBOX["b"][0] - PANBOX["a"][1], "b|c": PANBOX["c"][0] - PANBOX["b"][1]}
assert min(GAPS.values()) >= MIN_PANEL_GAP, GAPS
# v7: the row is filled (outer extents within 0.06 in of both figure edges)
ROW_FILL = {"left_margin_in": float(min(PANBOX["a"][0], R["letters"]["a"].get_window_extent(REN).transformed(INV).x0)),
            "right_margin_in": float(W - max(PANBOX["c"][1], XLAB_PACK.get_window_extent(REN).transformed(INV).x1))}
assert ROW_FILL["left_margin_in"] < 0.06 and ROW_FILL["right_margin_in"] < 0.06, ROW_FILL
DATA_SHARE = float((axA.get_position().width + axB.get_position().width + axC.get_position().width))
# (6) (a) x tick labels keep a clear gap; (b, c) x tick labels too
GAP_XT = {}
for k_, ax in (("a", axA), ("b", axB), ("c", axC)):
    tl = [t.get_window_extent(REN).transformed(INV) for t in ax.get_xticklabels() if t.get_text()]
    GAP_XT[k_] = float(min(b.x0 - a.x1 for a, b in zip(tl[:-1], tl[1:])))
assert min(GAP_XT.values()) > 0.04, GAP_XT
# (7) the theory label clears the dashed reference by at least 2 pt
tb_ = t_ref.get_window_extent(REN).transformed(INV)
y_ref_at = INV.transform(axA.transData.transform([[X_BEND, 1.0]]))[0][1]
REF_CLEAR_PT = float((tb_.y0 - y_ref_at) * 72)
assert REF_CLEAR_PT >= 2.0, REF_CLEAR_PT
# the label is centred over the bend (it names the bend)
LABEL_CENTRE_OFFSET_IN = float((tb_.x0 + tb_.x1) / 2 - INV.transform(axA.transData.transform([[X_BEND, 1.0]]))[0][0])
assert abs(LABEL_CENTRE_OFFSET_IN) < 0.01, LABEL_CENTRE_OFFSET_IN
# (8) key of (b, c): inset >= 3 pt from the y spine of (c); line-only dashed handle long enough for three full dashes;
# v7 (the 'Tr' header is gone): the key keeps >= 0.05 in from every drawn point and error bar of (c) above it
spine_x = INV.transform(axC.transAxes.transform([[0.0, 0.0]]))[0][0]
KEY_BB = keyC.get_window_extent(REN).transformed(INV)
KEY_INSET_MEASURED_PT = float((KEY_BB.x0 - spine_x) * 72)
assert KEY_INSET_MEASURED_PT >= 3.0, KEY_INSET_MEASURED_PT
HANDLE_PT = keyC.handlelength * S.FS["key"]
assert HANDLE_PT >= 3 * DASH_ON + 2 * DASH_OFF - 1e-9, HANDLE_PT
assert all(h.get_marker() in (None, "None", "none", "") for h in keyC.legend_handles), "key handles must be line-only"
low_c = min(float(INV.transform(axC.transData.transform([[x, L["ratio"][i] - L["se_ratio"][i]]]))[0][1])
            for (pre, c), L in LINES.items() if c == "mknn" for i, x in enumerate(XB_SIZES))
KEY_TO_DATA_IN = float(low_c - KEY_BB.y1)
assert KEY_TO_DATA_IN >= 0.05, KEY_TO_DATA_IN
assert KEY_BB.y0 >= INV.transform(axC.transAxes.transform([[0, 0]]))[0][1] - 1e-9   # inside (c)
# (9) v7: (b) and (c) share one y axis; (c) has no y tick labels; letters share one top; headers on the letters' baseline
assert axB.get_ylim() == axC.get_ylim() and axB.get_yscale() == axC.get_yscale() == "log"
assert not [t for t in axC.get_yticklabels() if t.get_visible() and t.get_text()], "(c) shows y tick labels"
LET_TOPS = {k_: float(t.get_window_extent(REN).transformed(INV).y1) for k_, t in R["letters"].items()}
assert max(LET_TOPS.values()) - min(LET_TOPS.values()) < 1e-6, LET_TOPS
# (10) v7: the plotted points of (b, c) are the ratios (read back from the artists)
for ax, c in ((axB, "cka"), (axC, "mknn")):
    marker_lines = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    assert len(marker_lines) == 2
    got = sorted([tuple(np.round(ln.get_ydata(), 12)) for ln in marker_lines])
    want = sorted([tuple(np.round(LINES[(p, c)]["ratio"], 12)) for p in (PRE_CLIP, PRE_L2)])
    assert got == want, (c, got, want)
LAYOUT = {"size_in": [W, H], "height_v6_in": H_V6,
          "axes_in": {"a": [XA, Y0, float(WA), AH], "b": [XB, Y0, WBC, AH], "c": [XC, Y0, WBC, AH]},
          "data_width_share_of_row": DATA_SHARE, "row_fill": ROW_FILL,
          "ylim_b_c_shared_log": R["ylim_bc"], "inches_per_decade_b_and_c": float(AH / np.log10(R["YL"][1] / R["YL"][0])),
          "panel_letters_in": {k_: [R["let_x"][k_], R["let_y"]] for k_ in R["let_x"]},
          "header_baseline_in": R["header_baseline_y"], "panel_extents_in": PANBOX,
          "panel_gaps_in": GAPS, "min_text_to_figure_edge_in": INSIDE_MARGIN_IN, "x_ticklabel_min_gap_in": GAP_XT, "theory_label_clearance_pt": REF_CLEAR_PT,
          "theory_label_centre_minus_bend_in": LABEL_CENTRE_OFFSET_IN, "in_per_step_a": IN_PER_STEP_A,
          "key_c": {"inset_from_y_spine_pt": KEY_INSET_MEASURED_PT, "handle_pt": HANDLE_PT, "handles": "line only",
                    "dash_on_off_pt": [DASH_ON, DASH_OFF], "full_dashes_in_dashed_handle": 3,
                    "gap_to_lowest_mknn_point_in": KEY_TO_DATA_IN, "extent_in": [float(v) for v in KEY_BB.extents]},
          "key_a_handle_pt": None if keyA is None else float(keyA.handlelength * S.FS["key"]),
          "key_a": "none (v8: one depth drawn in (a))" if keyA is None else "mid depth / last layer",
          "text_text_overlaps": clash, "text_marker_overlaps": mark_hits, "text_line_overlaps": line_hits}

# =============================================================================================
# numbers.json
# =============================================================================================
SRC_A = {"path": P_MAIN, "filter": "pre == 'prh_clip+l2', r in {0.5, 1.0} (136 rows each; L.m == 1024)",
         "columns": ["L.warpshare_expl_" + k for k in SC.STEPS + ("residual",)], "fit": "in-sample"}
SRC_BC = {"path": P_MAIN, "filter": "r == 0.5, lm_family == 'BLOOMZ', pre in {'prh_clip+l2', 'l2'} (85 rows each)",
          "columns": {"cka": "cal.cka_biased.uncensored", "mknn": "cal.s_N_k10.uncensored", "x": "lm_params"}}
TEXTS = sorted({s_ for s_, _, _ in TB})
N = {
    "figure": {"id": "F_criteria", "label": "fig:exp-criteria", "section": "5.3 sec:exp-criteria", "version": "v8",
               "status": "v8 proposal (staged in build/figs_v8_staging/F_criteria); arxiv_v4.tex untouched", "trial": TRIAL,
               "png": "poppler (pdftoppm) render of the PDF at 200 dpi", "png_px": PNG_PX,
               "views": {"print_100dpi": VIEW_PP, "cvd_grey": VIEWS},
               "size_in": CHK["size_in"], "height_v6_in": H_V6, "min_text_pt": CHK["min_text_pt"],
               "min_glyph_pt": CHK["min_glyph_pt"], "n_type3": CHK["n_type3"], "check": CHK, "pdffonts": pf.splitlines(),
               "layout": LAYOUT, "visible_texts": TEXTS},
    "inputs": {p: sha(p) for p in (P_MAIN, P_SUM)},
    "style": {"path": STYLE_V8_PATH, "sha256": S.STYLE_SHA256, "version": S.STYLE_VERSION,
              "semantic_map_used": {"criterion.CKA": S.CRIT["cka"], "criterion.mKNN": S.CRIT["mknn"],
                                    "family.BLOOMZ": S.FAM["bloomz"], "theory": S.INK, "reference.line": S.NULL_LINE,
                                    "thin_lines_(a)": THIN, "text.CKA": CKT, "text.mKNN": MKT},
              "fw": {r: S.FW[r] for r in ("label", "tick", "header", "direct", "theory_label", "key")},
              "v7_style_copy_kept": {"path": STYLE_PATH, "sha256": STYLE_SHA, "version": S51.STYLE_VERSION,
                                     "copied_from": "~/hrh_v2_figs/v5/_style/hrh_style_v5_1.py (identical sha256)"}},
    "script_sha256": sha(os.path.abspath(__file__)),
    "v6_maker_sha256": sha(V6_MAKER) if os.path.exists(V6_MAKER) else None,
    "staircase_module_sha256": sha(os.path.join(HERE, "staircase.py")),
    "deviation_from_spec": {
        "b_c_column": "spec named cal.<crit>.s_cal; plotted cal.<crit>.uncensored (s_obs - permutation-null mean), the "
                      "score whose BLOOMZ trend is the PE3 Tr quoted in the text (s12_summary.json pe3[...]['Tr_raw'/'Tr_mKNN'], "
                      "computed on .uncensored in analysis/s12/aggregate.py l.304-306; arxiv_v4 l.4232). Tr of the censored "
                      "s_cal under l2 would be %s (CKA) and %s (mKNN), not the quoted -1.00 / -0.92." % (
                          S.num(LINES[(PRE_L2, "cka")]["EXPLORATORY_Tr_of_censored_s_cal"], 2),
                          S.num(LINES[(PRE_L2, "mknn")]["EXPLORATORY_Tr_of_censored_s_cal"], 2)),
        "b_c_axis_v7": "one shared log y axis for (b) and (c); each line (mean and +-1 s.e.) divided by its own mean at "
                       "0.56B, a vertical shift on the log axis (asserted), replacing the v6 two-segment axis of (b)"},
    "v7_changes": {
        "removed_texts_v6": ["a monotone link / would reach 1 here", "median, mid depth", "median, last layer", "Tr (x2)",
                             "0.70", "−1.00", "0.84", "−0.92", "CKA − null mean", "mKNN − null mean",
                             "BLOOMZ parameters (B) (second copy)", "y tick labels 0.11 0.14 0.008 0.012 0.02 0.03 0.04"],
        "new_texts_v7": ["monotone link", "mid depth", "last layer", "CKA (header of b)", "mKNN (header of c)",
                         Y_LABEL_BC, "y tick labels 0.5 0.7 1"],
        "removed_marks_v6": ["axis-break marks of (b)", "second y axis of (c) and of the lower segment of (b)"],
        "new_marks_v7": ["dotted NULL_LINE at 1 in (b) and (c) (no change from 0.56B)"],
        "panels": "three panels kept; (b) and (c) keep their letters and messages, so the text references "
                  "'Figure fig:exp-criteria b,c' (arxiv_v4 l.1352, l.4302) stay valid"},
    "asserts_adapted": {
        "size": "v6 asserted 5.5 x 1.75 in; v7 asserts 5.5 x H with H < 1.75",
        "key_c_vs_Tr_header": "v6 asserted the key top >= 0.05 in below the 'Tr' header, which v7 removes; v7 asserts the key "
                              "top >= 0.05 in below the lowest drawn point minus s.e. of (c), and the key inside (c)",
        "panel_boxes": "v6 counted the Tr column and both x labels in the panel boxes; v7 counts the y label and header of "
                       "(b), the header of (c); the shared x label spans (b, c) and is checked against the right margin",
        "y_label_b_c": "v6 placed one y label per panel; v7 one y label on (b), and (c) has no y tick labels (asserted)"},
    "panel_a": {
        "source": SRC_A, "summary_check": f"{P_SUM} warp['r=<r>|prh_clip+l2']['expl'][<step>][0] == median (1e-12, asserted)",
        "steps": list(SC.STEP_LABELS), "dodge_steps": {DEPTH[r]: DODGES[r] for r in DODGES},
        "plotted_depths": [DEPTH[r] for r in PLOT_DEPTHS],
        "not_plotted_v8": {DEPTH[r]: "values kept in median_parts, median_cumulative, per_pair_*; stated in the Results "
                                     "of Section 5.3 and in Appendix app:exp-s12 (Warps and Margins)"
                           for r in (0.5, 1.0) if r not in PLOT_DEPTHS},
        "median_parts": {DEPTH[r]: PARTS[r] for r in PARTS},
        "median_cumulative": {DEPTH[r]: [float(v) for v in MEDCUM[r]] for r in MEDCUM},
        "per_pair_extremes": {DEPTH[r]: MAXEX[r] for r in MAXEX},
        "quantiles_cumulative_10_50_90": {DEPTH[r]: np.percentile(CUM[r], [10, 50, 90], axis=0).round(6).tolist() for r in CUM},
        "drawn": {k: {kk: vv for kk, vv in v.items()} for k, v in R["drawn_a"].items()},
        "theory_reference": R["ref_a"], "full_share_line": R["full_a"], "residual_bracket": R["br_a"],
        "thin_tint_share_of_white": THIN_W, "key_labels": [] if keyA is None else ["mid depth", "last layer"],
        "per_pair_cumulative": {DEPTH[r]: [{"lm": a, "vit": b, "cum": [float(x) for x in row]}
                                           for (a, b), row in zip(PAIRS[r][["lm", "vit"]].itertuples(index=False), CUM[r])]
                                for r in CUM}},
    "panels_b_c": {
        "source": SRC_BC, "depth": "mid depth (r = 0.5)", "lm_order": LM_ORDER, "lm_params": LM_PAR, "vits": VITS,
        "summary_check": f"{P_SUM} pe3['r=0.5|<pre>|biased'][Tr_raw (CKA) | Tr_mKNN] Tr, p_one, p_two equal (1e-12, asserted)",
        "encoding": {"marker": "open circle (mid depth) for every line", "line_style": {PRE_CLIP: "solid", PRE_L2: "dashed"},
                     "colour": {"cka": "CRIT cka (b)", "mknn": "CRIT mknn (c)"},
                     "y": "mean over the 17 vision models divided by its value at 0.56B (log axis shared by b and c); "
                          "error bars +-1 s.e. over vision models divided by the same value",
                     "reference": "dotted NULL_LINE at 1"},
        "y_ticks": YT_BC, "ylim": R["ylim_bc"],
        "lines": {f"{c}|{pre}": {"plotted_ratio_to_0.56B": L["ratio"].tolist(), "plotted_se_ratio": L["se_ratio"].tolist(),
                                 "mean_over_17_vits": L["mean"].tolist(), "se_over_17_vits": L["se"].tolist(),
                                 "divisor_mean_at_0.56B": float(L["mean"][0]),
                                 "Tr": L["Tr"], "p_one_exact": L["p_one"], "p_two_exact": L["p_two"], "n_perm": 120,
                                 "EXPLORATORY_Tr_of_censored_s_cal": L["EXPLORATORY_Tr_of_censored_s_cal"],
                                 "n_vits_strictly_rising": L["n_vit_rising_monotone"],
                                 "n_vits_strictly_falling": L["n_vit_falling_monotone"],
                                 "ratio_mean_7.1B_over_0.56B": L["ratio_largest_over_smallest_mean"],
                                 "range_of_mean": [float(L["mean"].min()), float(L["mean"].max())],
                                 "per_vit_grid_rows_lm_cols_vit": L["Y"].tolist()}
                  for (pre, c), L in LINES.items()}},
    "on_panel_numbers": {},
    "on_panel_numbers_note": "v8 (as v7) prints no result number on the figure; the four Tr values that v6 printed are "
                             "quoted in the Results paragraph of Section 5.3 (arxiv_v4 l.1420-1423 as read 2026-10-10 22:40).",
}
RESULTS = {
    "a: median share explained by scale (mid / last)": ([PARTS[0.5]["scale"], PARTS[1.0]["scale"]], "panel_a.median_parts.<depth>.scale"),
    "a: median cumulative after + affine (mid / last)": ([float(MEDCUM[0.5][1]), float(MEDCUM[1.0][1])], "panel_a.median_cumulative.<depth>[1]"),
    "a: median iso extra (mid / last)": ([PARTS[0.5]["iso_extra"], PARTS[1.0]["iso_extra"]], "panel_a.median_parts.<depth>.iso_extra; S1_2_report.md 6.1"),
    "a: largest iso extra in any pair (mid / last)": ([MAXEX[0.5]["iso_extra"], MAXEX[1.0]["iso_extra"]], "panel_a.per_pair_extremes.<depth>.iso_extra"),
    "a: median row extra (mid / last)": ([PARTS[0.5]["row_extra"], PARTS[1.0]["row_extra"]], "panel_a.median_parts.<depth>.row_extra"),
    "a: median residual (mid / last)": ([PARTS[0.5]["residual"], PARTS[1.0]["residual"]], "panel_a.median_parts.<depth>.residual; S1_2_report.md 6.1"),
    "a: residual range over pairs (mid / last)": ([[MAXEX[0.5]["min_residual"], MAXEX[0.5]["max_residual"]], [MAXEX[1.0]["min_residual"], MAXEX[1.0]["max_residual"]]], "panel_a.per_pair_extremes"),
    "b, c: Tr clip CKA / mKNN": ([LINES[(PRE_CLIP, "cka")]["Tr"], LINES[(PRE_CLIP, "mknn")]["Tr"]], "panels_b_c.lines.<crit>|prh_clip+l2.Tr"),
    "b, c: Tr l2 CKA / mKNN": ([LINES[(PRE_L2, "cka")]["Tr"], LINES[(PRE_L2, "mknn")]["Tr"]], "panels_b_c.lines.<crit>|l2.Tr"),
    "b, c: exact one-sided p, clip CKA / mKNN": ([LINES[(PRE_CLIP, "cka")]["p_one"], LINES[(PRE_CLIP, "mknn")]["p_one"]], "panels_b_c.lines.<crit>|prh_clip+l2.p_one_exact"),
    "b, c: exact two-sided p, l2 CKA / mKNN": ([LINES[(PRE_L2, "cka")]["p_two"], LINES[(PRE_L2, "mknn")]["p_two"]], "panels_b_c.lines.<crit>|l2.p_two_exact"),
    "b, c: vision models with a strictly falling line under l2, CKA / mKNN (of 17)": ([LINES[(PRE_L2, "cka")]["n_vit_falling_monotone"], LINES[(PRE_L2, "mknn")]["n_vit_falling_monotone"]], "panels_b_c.lines.<crit>|l2.n_vits_strictly_falling"),
    "b, c: mean at 7.1B / mean at 0.56B, clip CKA / l2 CKA / clip mKNN / l2 mKNN": ([LINES[(PRE_CLIP, "cka")]["ratio_largest_over_smallest_mean"], LINES[(PRE_L2, "cka")]["ratio_largest_over_smallest_mean"], LINES[(PRE_CLIP, "mknn")]["ratio_largest_over_smallest_mean"], LINES[(PRE_L2, "mknn")]["ratio_largest_over_smallest_mean"]], "panels_b_c.lines.<crit>|<pre>.ratio_mean_7.1B_over_0.56B"),
    "b, c (v7, levels no longer on the axis): range over sizes of the mean, clip CKA / l2 CKA / clip mKNN / l2 mKNN": ([[float(LINES[(p, c)]["mean"].min()), float(LINES[(p, c)]["mean"].max())] for p, c in ((PRE_CLIP, "cka"), (PRE_L2, "cka"), (PRE_CLIP, "mknn"), (PRE_L2, "mknn"))], "panels_b_c.lines.<crit>|<pre>.range_of_mean"),
}
N["results_paragraph_numbers"] = {k: {"value": v, "key": key} for k, (v, key) in RESULTS.items()}

# =============================================================================================
# v8: the data equal v7; the figure text is v7 minus the depth key; caption v8
# =============================================================================================
def _rt(x):
    return json.loads(json.dumps(x, default=float))


assert _rt(N["results_paragraph_numbers"]) == O7["results_paragraph_numbers"], "Results numbers differ from v7"
for k_ in ("median_parts", "median_cumulative", "per_pair_extremes", "quantiles_cumulative_10_50_90", "per_pair_cumulative"):
    assert _rt(N["panel_a"][k_]) == O7["panel_a"][k_], k_
assert _rt(N["panels_b_c"]["lines"]) == O7["panels_b_c"]["lines"] and N["panels_b_c"]["y_ticks"] == O7["panels_b_c"]["y_ticks"]
assert _rt(N["inputs"]) == O7["inputs"], "input files changed since v7"
for r in PLOT_DEPTHS:                                   # the drawn medians are the v7 medians (x moves by the dodge only)
    assert np.allclose(N["panel_a"]["drawn"][DEPTH[r]]["median_cumulative"], O7["panel_a"]["drawn"][DEPTH[r]]["median_cumulative"],
                       rtol=0, atol=1e-15)
    assert np.allclose(np.array(N["panel_a"]["drawn"][DEPTH[r]]["x"]) - DODGES[r],
                       np.array(O7["panel_a"]["drawn"][DEPTH[r]]["x"]) - O7["panel_a"]["dodge_steps"][DEPTH[r]])
V7_TEXTS = set(O7["figure"]["visible_texts"])
DROPPED_TEXTS = sorted(V7_TEXTS - set(TEXTS))
ADDED_TEXTS = sorted(set(TEXTS) - V7_TEXTS)
if PLOT_DEPTHS == (0.5,):
    assert DROPPED_TEXTS == ["last layer", "mid depth"] and ADDED_TEXTS == [], (DROPPED_TEXTS, ADDED_TEXTS)
    assert len(R["drawn_a"]) == 1 and R["drawn_a"]["mid depth"]["n"] == 136

CAPTION_V7 = (
    r"\caption{\textbf{Beyond an affine change of scale, pretrained vision and language models differ by a residual, not a "
    r"monotone link, and preprocessing can reverse size trends (Proposition~\ref{prop:cka}, Theorem~\ref{thm:link}).} "
    r"(a)~Share of the squared-distance discrepancy explained by the nested fits, per pair (thin), median (thick), and "
    r"under a monotone link (dashed). (b,~c)~CKA and mKNN minus their permutation-null means, relative to the smallest "
    r"size, as mean $\pm$ s.e.\ over vision models on log axes. Appendix~\ref{app:exp-s12} gives the shares and the "
    r"exact $p$-values.}")
assert hashlib.sha256((CAPTION_V7 + "\n").encode()).hexdigest() == FROZEN_V7[f"{V7}/caption_v7.tex"], "v7 caption record"
CAPTION_V8 = (
    r"\caption{\textbf{Beyond an affine change of scale, pretrained vision and language models differ by a residual, not a "
    r"monotone link, and preprocessing can reverse size trends (Proposition~\ref{prop:cka}, Theorem~\ref{thm:link}).} "
    r"(a)~Share of the squared-distance discrepancy at mid depth explained by the nested fits, per pair (thin), median "
    r"(thick), and under a monotone link (dashed). (b,~c)~CKA and mKNN minus their permutation-null means, as mean "
    r"$\pm$ s.e.\ over vision models on log axes. Appendix~\ref{app:exp-s12} gives the shares and the exact "
    r"$p$-values.}")
assert CAPTION_V8.replace(" at mid depth explained", " explained").replace(
    "means, as mean", "means, relative to the smallest size, as mean") == CAPTION_V7, "v8 = v7 caption with two edits"


def caption_words(cap):
    """(words as printed, panel-tag tokens, bold-claim words without refs); each \\ref and each inline formula is one
    token, as in the F_neighbors makers since v7."""
    w = re.sub(r"\\ref\{[^}]*\}", " 1 ", cap)
    w = re.sub(r"\$[^$]*\$", " x ", w)
    w = re.sub(r"\\caption\{|\\textbf\{|[{}~]", " ", w)
    toks = [x for x in re.split(r"\s+", w) if re.search(r"[A-Za-z0-9]", x)]
    ntag = sum(1 for x in toks if re.fullmatch(r"\(?[a-c][),]", x) or x in ("(a)", "(b,", "c)"))
    b = re.search(r"\\textbf\{(.*?)\}\s*\(a\)", cap, re.S).group(1)
    nb = len([x for x in re.split(r"\s+", re.sub(r"\([^)]*\)", " ", b)) if re.search(r"[A-Za-z]", x)])
    return len(toks), ntag, nb


NW8, NT8, NB8 = caption_words(CAPTION_V8)
NW7, NT7, NB7 = caption_words(CAPTION_V7)
assert NT8 == 3 and NB8 <= 25 and 60 <= NW8 - NT8 <= 80, (NW8, NT8, NB8)
assert re.findall(r"(?<![A-Za-z_\\])\d+(?:\.\d+)?", re.sub(r"\\ref\{[^}]*\}", "", CAPTION_V8)) == [], "number in caption"
assert "---" not in CAPTION_V8 and "—" not in CAPTION_V8 and ": " not in CAPTION_V8, "em-dash or colon in the caption"
assert not re.search(r"\bit[.,;]", CAPTION_V8), "a clause ends on 'it'"
for w_ in ("must", "never", "always", "nothing", "none", "cannot", "only"):
    assert not re.search(r"\b%s\b" % w_, CAPTION_V8), w_
assert len(re.findall(r"Appendi(?:x|ces)~\\ref", CAPTION_V8)) == 1, "one appendix pointer"
assert "mid depth" in CAPTION_V8, "the caption names the depth of (a)"
with open(os.path.join(OUT, "caption_v8.tex"), "w") as f:
    f.write(CAPTION_V8 + "\n\\label{fig:exp-criteria}\n")
N["caption_v7"] = CAPTION_V7
N["caption_v8"] = CAPTION_V8
N["caption_v8_sha256"] = hashlib.sha256(CAPTION_V8.encode()).hexdigest()
N["caption_words"] = {"v8": {"total_incl_bold_tags_and_refs": NW8, "without_panel_tags": NW8 - NT8, "bold_claim_without_refs": NB8},
                      "v7": {"total_incl_bold_tags_and_refs": NW7, "without_panel_tags": NW7 - NT7, "bold_claim_without_refs": NB7},
                      "how": "each \\ref and each inline formula one token; panel tags (a), (b, c) = 3 tokens"}
N["height_trials_v8"] = json.loads(os.environ.get("FC_HEIGHT_TRIALS_V8", "{}"))
N["v8_changes"] = {
    "style": "hrh_style_v5_1 -> hrh_style_v8 (sha256 %s...): CKA %s -> %s, mKNN %s -> %s, thin lines of (a) %s -> %s, "
             "CKA text %s -> %s, mKNN text %s -> %s, BLOOMZ word %s (unchanged)"
             % (S.STYLE_SHA256[:12], S51.CRIT["cka"], S.CRIT["cka"], S51.CRIT["mknn"], S.CRIT["mknn"],
                S51.tint(S51.CRIT["cka"], THIN_W), THIN, S51.CRIT_TEXT["cka"], CKT, S51.CRIT_TEXT["mknn"], MKT,
                S.FAM["bloomz"]),
    "bold": "tick labels, the axis title of (a), the y label of (b) and the shared x label bold (S.finalize and the v8 "
            "'label' role)",
    "panel_a_depths": "plotted %s (v7: mid depth and last layer, dodged by 0.10 step, with a key)" % [DEPTH[r] for r in PLOT_DEPTHS],
    "removed_texts_v7": DROPPED_TEXTS, "added_texts_v8": ADDED_TEXTS,
    "removed_marks_v7": [] if len(PLOT_DEPTHS) == 2 else
    ["last-layer median line with its filled circles", "136 last-layer thin lines", "the depth key handles"],
    "png": "poppler 200 dpi render of the PDF (v7: matplotlib 300 dpi)",
    "caption": "'at mid depth' added to (a); 'relative to the smallest size,' dropped from (b, c) (the y label reads "
               "'relative to 0.56B')",
    "height": "%.2f in (v7 1.44 in)" % H}
N["asserts_adapted_v8"] = {
    "style_sha": "v7 asserted the imported style = hrh_style_v5_1; v8 asserts the imported style = hrh_style_v8 "
                 "(fe994067...) and keeps the v7 assert on the v5.1 copy (S51), which stays next to this file",
    "png": "v7 wrote the PNG with matplotlib at 300 dpi; v8 renders the PDF with poppler at 200 dpi and asserts its size",
    "key_a": "v7 recorded the handle length of the depth key of (a); v8 records None when one depth is drawn",
    "new": "v8 style sha; v7 maker, render, caption and numbers frozen; Results numbers, panel_a medians, per-pair shares "
           "and the (b, c) lines equal v7; inputs unchanged since v7; figure words = v7 minus the depth key; S.check v8 "
           "rules (bold ticks and axis titles, bold math, chromatic data, no off-palette data colour); v8 caption rules"}
json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
print("ok", OUT)
print(json.dumps({"check": {k: CHK[k] for k in ("min_text_pt", "min_glyph_pt", "n_type3", "fonts", "off_palette_colours",
                                                "n_tick_labels_checked", "n_axis_titles_checked")},
                  "caption_words": N["caption_words"], "removed_texts_v7": DROPPED_TEXTS,
                  "layout": {k: LAYOUT[k] for k in ("size_in", "axes_in", "data_width_share_of_row", "row_fill", "panel_gaps_in",
                                                    "x_ticklabel_min_gap_in", "theory_label_clearance_pt", "in_per_step_a",
                                                    "key_c", "inches_per_decade_b_and_c")}}, indent=1, default=float))
