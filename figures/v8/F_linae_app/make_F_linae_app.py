#!/usr/bin/env python3
"""F_linae_app, v8 render (arxiv_v4 appendix figure \\label{fig:exp-linae}; find it by the label, line numbers move).
Message (v8 caption, bold sentence): gradient descent without a regularizer stays unreliable at every sample size on
the criteria that see the gauge, whereas weight decay, tied weights and small initialization become reliable
(Theorem thm:gauge).

v8 (2026-10-10, role maker for the appendix gauge figures): copy of ~/hrh_v2_figs/v6/F_linae_app/make_F_linae_app.py
(sha256 c724662d...9614, untouched; its outputs stay in ~/hrh_v2_figs/v6/F_linae_app/), switched to the frozen v8
style ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (sha256 asserted). Author feedback of 2026-10-10 on v7 Figure 6, applied
to this appendix figure:
  1 no header that states a setting or a reading: removed the key group heads 'covered by the theorem:' / 'outside the
    theorem:', the group heads of (e), the row-label parentheticals of (e) ('(part 1)', '(part 1, C_0 ~ 0)',
    '(part 3)', '(C not conserved)'), the tolerance lines under the a-d heads, the decision-rule text in (d), the pair
    counts in the a-d x label, the title of (e), the 'measured' / 'predicted' key of (e), and 'batch 16' after SGD and
    Adam. The caption and the Results text carry them (MOVED.md says where each one lives).
  2 colour: v8 palette (Tol). GD dark blue #222255 (v6 black), every other rule keeps its v5.1 hue; the predictions stay
    INK (theory). Series stay apart in greyscale and under CVD by marker and line style (filled / open, solid / dashed).
  3 bold tick labels and axis titles with bold math (S.finalize before savefig; S.check asserts it).
  4 little text: the direct labels 'SGD', 'Adam', 'GD' of (a, c) are gone (the key names every rule); panel heads are the
    criterion symbols T_{N,10}, G_s, 1 - CKA, Delta_Pi, on the baseline of the panel letter.
  Layout: 5.5 x 3.22 in (v6 3.75): the key in two rows (rules covered by the theorem, then the rules outside it, which
  the line style also tells), the a-d row, and (e) with one row per rule (no group-head rows).
  Data, statistics and every number are those of v6 (asserted equal to the v6, v5.1, v4 and v3 numbers.json; only
  geometry fields and colour fields differ, listed in numbers.json v8.changed_fields).
v8 r2 (2026-10-10, role critic; the maker's r1 files are in ./_prev_r1_maker/):
  * x title of (e): '$EE^{\\top}$' -> '$EE^{\\mathrm{T}}$'. Liberation Serif has no \\top, so the bold math fell back
    to a thin Computer Modern glyph set apart from 'EE' (the only thin glyph in an axis title of the figure); the
    upright bold T is the same transpose in the bold math font.
  * caption: (e) says which rows carry a prediction ('the predictions for the covered rules in black'), so the thin
    line of (e) and the rows without a triangle decode from the caption; 'against the sample size $N$' -> 'against
    $N$' (the x title names the sample size) keeps the caption at 100 words.
v8 c (2026-10-10, role consistency checker across all v8 figures; the r2 files are in ./_prev_consistency/): panel (e)
is the full version of Figure 5 (c), and both now draw the same marks the same way:
  * the grey line at cond = 1 is gone, as Figure 5 (c) dropped it in its critic pass: the prediction triangles of the
    rows at 1 sat on this line and read as arrows pointing at the markers; the tick '1' marks the isotropic value
    (SHOW_COND1_LINE_E = False brings it back);
  * x title of (e) '$\\mathrm{cond}(EE^{\\mathrm{T}})$', the x title of Figure 5 (c) (r2: 'condition number of
    $EE^{\\mathrm{T}}$'); the caption already names the quantity ('Anisotropy of the learned gauge').
  Data, every assert of r2 and the caption unchanged.
Earlier history: the docstring of the v6 maker (r2, v6, v5.1, v5 r1-r3) in ~/hrh_v2_figs/v6/F_linae_app/.

Read-only inputs:
  SUM  ~/hrh_v2_stage1_20261008/results/S1_3a_linae_summary.json   decisions, Clopper-Pearson bounds, cond quantiles
  RUNS ~/hrh_v2_stage1_20261008/linae/criteria/<arm>_Ninf_runs.csv  per run: diverged, eig_fC0_0..3, kappa
  NPZ  ~/hrh_v2_stage1_20261008/linae/runs/WD/Ninf/chunk*.npz       S_eigs, spec_EEt (part-3 check for weight decay)
  OLD  ~/hrh_v2_figs/F5/numbers.json                                 v3 numbers; every value is asserted equal
  V4   ~/hrh_v2_figs/v4/F_linae_app/numbers.json                     v4 numbers; every value is asserted equal
  V51  ~/hrh_v2_figs/v5_1/F_linae_app/numbers.json                   v5.1 numbers; every data block asserted equal
  V6   ~/hrh_v2_figs/v6/F_linae_app/numbers.json                     v6 numbers; every data block asserted equal (v8)
  STYLE ~/hrh_v2_figs/v8/_style/hrh_style_v8.py                      frozen shared style (imported, never edited)
Writes only into this directory: F_linae_app.pdf, F_linae_app.png (200 dpi, pdftoppm), _view_print_100dpi.png,
_views_cvd_grey.png, numbers.json, caption_v8.tex.
Run (v8): cd ~/hrh_v2_figs/v8/F_linae_app && OMP_NUM_THREADS=1 taskset -c 0-5 nice -n 10 \
     ~/hrh_v2_figs_venv/bin/python -B make_F_linae_app.py > run.log 2>&1
"""
import os, sys, json, hashlib, glob, datetime, subprocess, re

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


STYLE_PATH = os.path.expanduser("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert sha(STYLE_PATH) == STYLE_SHA, "frozen v8 style module changed"
sys.path.insert(0, os.path.dirname(STYLE_PATH))
import hrh_style_v8 as S  # noqa: E402
assert S.STYLE_SHA256 == STYLE_SHA
S.use()
INK, MUTED, LW, MS, Z = S.INK, S.MUTED, S.LW, S.MS, S.Z

HOME = os.path.expanduser("~")
S1 = f"{HOME}/hrh_v2_stage1_20261008"
SUM = f"{S1}/results/S1_3a_linae_summary.json"
CRIT_DIR = f"{S1}/linae/criteria"
NPZ_WD = sorted(glob.glob(f"{S1}/linae/runs/WD/Ninf/chunk*.npz"))
LINAE_CORE = f"{S1}/linae/linae_core.py"
OLD = f"{HOME}/hrh_v2_figs/F5/numbers.json"
V4NUM = f"{HOME}/hrh_v2_figs/v4/F_linae_app/numbers.json"
V51NUM = f"{HOME}/hrh_v2_figs/v5_1/F_linae_app/numbers.json"
V51NUM_SHA = "7c6fda32cfcc6d713a372afd0c7df67c140bbc131c1985c274bd00034c63dacf"   # v5.1 render, staged 2026-10-10
V6NUM = f"{HOME}/hrh_v2_figs/v6/F_linae_app/numbers.json"
V6NUM_SHA = "c38c0307d5982be6e82bac353c1e2b2288e2a4611bde6e69089c0bb4c2c0b724"     # v6 r2 render, staged 2026-10-10
V6MAKER = f"{HOME}/hrh_v2_figs/v6/F_linae_app/make_F_linae_app.py"
V6MAKER_SHA = "c724662d1e25f598beeb23da4bef0a76fd9f4fd71ebd291583c4808c06a99614"
assert NPZ_WD, "no WD chunks"
assert sha(V51NUM) == V51NUM_SHA, "v5.1 numbers.json changed"
assert sha(V6NUM) == V6NUM_SHA, "v6 numbers.json changed"
assert sha(V6MAKER) == V6MAKER_SHA, "v6 maker changed"
STEM = "F_linae_app"

# row labels of e in earlier renders (history; asserted against the v5.1 and v6 numbers.json below)
ROW_LABELS_V51 = {"P-GD-flow": "GD, step 0.002 (secondary, part 1)", "small-init": "small init. (part 1)",
                  "P-GD": "GD, step 0.05 (preregistered)"}
ROW_LABELS_V6 = {"P-GD-flow": "GD, step 0.002 (part 1)", "small-init": "small init. (part 1, $C_0\\approx 0$)",
                 "WD": "weight decay (part 3)", "TIED": "tied weights (part 3)",
                 "P-GD": "GD, step 0.05 ($C$ not conserved)", "P-SGD": "SGD, batch 16", "P-Adam": "Adam, batch 16"}

summ = json.load(open(SUM))
meta, cells = summ["meta"], summ["cells"]
old = json.load(open(OLD))
v4num = json.load(open(V4NUM))
v51num = json.load(open(V51NUM))
v6num = json.load(open(V6NUM))

# ---------------------------------------------------------------------------------------------------
# training rules: (summary key, key label, row label in e, covered by the theorem, colour, marker, face, ls, arm key)
# order = dodge order of a-d = row order of e. v8: row label = key label = the rule's name alone (the setting words
# moved to the caption and the Results text); colours S.ARM of hrh_style_v8 (GD #222255; the others as v5.1).
# ---------------------------------------------------------------------------------------------------
DASH = S.LS["dashed"]
ARMS = [
    ("P-GD-flow",  "GD, step 0.002", "GD, step 0.002", True,  S.ARM["gd"],        "o", "fill", "-",  "gd"),
    ("small-init", "small init.",    "small init.",    True,  S.ARM["smallinit"], "^", "fill", "-",  "smallinit"),
    ("WD",         "weight decay",   "weight decay",   True,  S.ARM["wd"],        "s", "fill", "-",  "wd"),
    ("TIED",       "tied weights",   "tied weights",   True,  S.ARM["tied"],      "D", "fill", "-",  "tied"),
    ("P-GD",       "GD, step 0.05",  "GD, step 0.05",  False, S.ARM["nowd"],      "o", "open", DASH, "nowd"),
    ("P-SGD",      "SGD",            "SGD",            False, S.ARM["sgd"],       "X", "fill", DASH, "sgd"),
    ("P-Adam",     "Adam",           "Adam",           False, S.ARM["adam"],      "P", "fill", DASH, "adam"),
]
AK = [a[0] for a in ARMS]
ARM_OF = {a[0]: a for a in ARMS}
NS = ["256", "4096", "65536", "inf"]
XPOS = {"256": 8.0, "4096": 12.0, "65536": 16.0, "inf": 20.5}     # log2 N; infinity drawn at 20.5 after a break
XBRK = 18.25
XT = [S.pow_label(2, 8), S.pow_label(2, 12), S.pow_label(2, 16), "$\\infty$"]
DODGE = np.linspace(-1.38, 1.38, len(ARMS))                         # per-rule offset in log2 N, same in a-d (as v4)
COL_HW = 1.66                                                       # half-width of the light column behind each N
XLIM_AD = (XPOS["256"] - COL_HW - 0.17, XPOS["inf"] + COL_HW + 0.17)
EPS = meta["eps"]
assert EPS == {"kNN": 0.05, "Gs": 0.02, "CKA": 0.01, "proj": 0.01}, EPS
CRITS = [  # summary key, panel head (v8: the criterion symbol, as F_gauge_app (a) and Figure 6 (d)), v6 head + tolerance
    ("kNN",  "$T_{N,10}$",           "neighbor sets",         "$1-s_N(10)\\leq %g$" % EPS["kNN"]),
    ("Gs",   "$G_s$",                "distances up to scale", "$G_s\\leq %g$" % EPS["Gs"]),
    ("CKA",  "$1-\\mathrm{CKA}$",    "linear CKA",            "$1-\\mathrm{CKA}\\leq %g$" % EPS["CKA"]),
    ("proj", "$\\Delta_\\Pi$",       "projector",             "$\\Delta_\\Pi\\leq %g$" % EPS["proj"]),
]
REL_LO, UNREL_HI = 0.9, 0.8   # preregistered decision levels (PREREG S1.3a; ANALYSIS_PLAN)

# marker area matching across shapes (geometry of the glyphs, not style)
SHAPE = {"P": 1.12, "X": 1.12, "D": 0.85, "^": 1.15}
EDGE = MS["line_edge"]     # 0.6 pt, drawn in the rule colour (v5 r3)


def mstyle(arm, where="ad"):
    """Marker of a rule, drawn S.MS['line'] = 4.0 pt across in a-d and the key and S.MS['summary'] = 5.0 pt across in
    e, edge included (circle and square; the other shapes scaled by SHAPE); edge in the rule colour; the open GD ring
    has the outer diameter of the filled marks (v5 r3, unchanged)."""
    ak, kl, rl, cov, col, mk, face, ls, rk = arm
    size = MS["line"] if where == "ad" else MS["summary"]
    if face == "open":
        mew = MS["open_small_mew"] if where == "ad" else MS["open_line_mew"]
        return dict(marker=mk, ms=size - mew, mfc="white", mec=col, mew=mew)
    return dict(marker=mk, ms=(size - EDGE) * SHAPE.get(mk, 1.0), mfc=col, mec=col, mew=EDGE)


def underlay(m):
    """White underlay of a median mark of e: the same marker in white, its edge LW['median_halo'] wider."""
    return dict(marker=m["marker"], ms=m["ms"], mfc="white", mec="white", mew=m["mew"] + LW["median_halo"])


def arm_width_pt(arm, where):
    """Visible arm width (pt) of an 'X' or '+' mark, edge included."""
    m = mstyle(arm, where)
    return {"P": 1 / 3, "X": 0.25 * np.sqrt(2)}[m["marker"]] * m["ms"] + m["mew"]


DATA = []      # data artists (marks, curves, bars), for the text-against-mark clearance check
FILLS = []     # (patch, fill hex): fills that texts may sit on (column bands of a-d)
KEY_ART = []

# ===================================================================================================
# geometry (inches from the bottom-left corner); exact print size 5.5 x H
# ===================================================================================================
W = S.TEXTWIDTH_IN
fig = plt.figure(figsize=(W, 4.0))     # height set below, once the blocks are stacked
renderer = fig.canvas.get_renderer()


def text_w(s, role):
    t = fig.text(0, 0, s, **S.text_kw(role, s))
    w = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return w


# rows of e: one per rule; the four rules covered by the theorem carry a prediction triangle above their median mark,
# so their rows are COV_PITCH apart (the triangle sits in the extra space; v5 r1); a GROUP_GAP and a thin line separate
# them from the three rules outside the theorem (v8: no group-head rows)
COV_PITCH, GROUP_GAP = 1.3, 0.25
rows, y = [], 0.0
for i, a in enumerate(ARMS[:4]):
    rows.append((y, a)); y += COV_PITCH if i < 3 else 1 + GROUP_GAP
for a in ARMS[4:]:
    rows.append((y, a)); y += 1
ymax = y - 1
Y_SEP = (rows[3][0] + rows[4][0]) / 2
Y_TOP, Y_BOT = -0.95, ymax + 0.5   # y limits of e (inverted axis: rows run downward); v8: the first row (GD, step
# 0.002) carries a prediction triangle, so the top leaves room for it (v6 had a group-head row there)
PT_ROW_TARGET = 8.6                # pt per row unit (v6 8.43): bold 7.5 pt row labels one unit apart
E_B = 0.36
E_T = E_B + (Y_BOT - Y_TOP) * PT_ROW_TARGET / 72
AX_B = E_T + 0.49                  # a-d axes bottom: tick labels, the x label and a gap above e
AX_H = 0.85
AX_T = AX_B + AX_H
HEAD_BASE = AX_T + 0.085           # baseline of the panel letters (a)-(d) and of the criterion heads
KEY_Y = [AX_T + 0.400, AX_T + 0.270]   # centre baselines of the two key rows
H = AX_T + 0.49
fig.set_size_inches(W, H)


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


E_PAD = 3.0 / 72                                  # gap between the row labels of e and its axis, inches
E_LABW = max(text_w(a[2], "tick") for a in ARMS)  # bold 7.5 (v8 tick role)
AX_L, AX_R, GAP = 0.50, 5.44, 0.13
AW = (AX_R - AX_L - 3 * GAP) / 4
E_L, E_R = 0.03 + E_LABW + E_PAD, 5.44
axs = [add_ax(AX_L + i * (AW + GAP), AX_B, AW, AX_H) for i in range(4)]
axe = add_ax(E_L, E_B, E_R - E_L, E_T - E_B)

numbers = {"panels_a_to_d": {}, "panel_e": {}}

# ===================================================================================================
# panels a-d: fraction of disjoint run pairs within tolerance against N, one-sided 95% CP bounds
# ===================================================================================================
HEADS = []
for ci, (ck, chead, cname_v6, tol_v6) in enumerate(CRITS):
    ax = axs[ci]
    numbers["panels_a_to_d"][ck] = {}
    for N in NS:   # light column behind each N, so that the seven offset rules read as one group
        FILLS.append((ax.axvspan(XPOS[N] - COL_HW, XPOS[N] + COL_HW, color=S.REF_FILL, lw=0, zorder=Z["band"]),
                      S.REF_FILL))
    for yv in (REL_LO, UNREL_HI):
        ax.axhline(yv, color=S.NULL_LINE, lw=LW["level"], zorder=Z["guide"])
    for ai, arm in enumerate(ARMS):
        ak, kl, rl, cov, col, mk, face, ls, rk = arm
        xs, ps, los, his = [], [], [], []
        numbers["panels_a_to_d"][ck][ak] = {}
        for N in NS:
            c = cells[f"{ak}|{N}"]["primary"][ck]
            xs.append(XPOS[N] + DODGE[ai]); ps.append(c["p_hat"]); los.append(c["cp_lower"]); his.append(c["cp_upper"])
            numbers["panels_a_to_d"][ck][ak][N] = dict(
                x_log2N_plotted=XPOS[N] + float(DODGE[ai]),
                passes=c["passes"], n_pairs=c["n_pairs"], p_hat=c["p_hat"], cp_lower=c["cp_lower"],
                cp_upper=c["cp_upper"], decision=c["decision"], n_fail_by_divergence=c["n_fail_by_divergence"],
                source=f"{SUM} :: cells['{ak}|{N}']['primary']['{ck}']")
        xs, ps, los, his = map(np.asarray, (xs, ps, los, his))
        DATA.extend(ax.plot(xs[:3], ps[:3], color=col, lw=LW["data"], ls=ls, zorder=Z["data"]))
        DATA.extend(ax.plot(xs[2:], ps[2:], color=col, lw=LW["break"], ls=S.LS["dotted"], zorder=Z["data"]))
        DATA.append(ax.vlines(xs, los, his, color=col, lw=LW["errorbar"], zorder=Z["interval"]))
        DATA.extend(ax.plot(xs, ps, ls="none", zorder=Z["marker"], **mstyle(arm)))
    ax.set_xlim(*XLIM_AD)
    ax.set_xticks([XPOS[N] for N in NS])
    ax.set_xticklabels(XT)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(-0.045, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ci == 0:
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
        ax.set_ylabel("fraction of pairs\nwithin tolerance", linespacing=1.1)
    else:
        ax.set_yticklabels([])
    # axis break between 2^16 and infinity: a white gap in the spine, then two // strokes
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([XBRK - 0.28, XBRK + 0.28], [0, 0], transform=tr, color="white", lw=LW["spine"] + 1.4, clip_on=False,
            zorder=Z["label"] - 0.5, solid_capstyle="butt")
    for dx in (-0.2, 0.2):
        ax.plot([XBRK + dx - 0.22, XBRK + dx + 0.22], [-0.035, 0.035], transform=tr, color=INK,
                lw=LW["axis_break"], clip_on=False, zorder=Z["label"], solid_capstyle="butt")

_NP = {N: sorted({cells[f"{ak}|{N}"]["primary"][ck]["n_pairs"] for ak in AK for ck in EPS}) for N in NS}
assert _NP == {"256": [48], "4096": [48], "65536": [48], "inf": [500]}, _NP
N_PAIRS_FIN, N_PAIRS_INF = _NP["256"][0], _NP["inf"][0]
XLAB_TEXT_V6 = ("training sample size $N$ (%d disjoint pairs per finite $N$; %d at $N=\\infty$)"
                % (N_PAIRS_FIN, N_PAIRS_INF))
XLAB_TEXT = "sample size $N$"           # v8: the pair counts live in Appendix app:exp-s13a (MOVED.md)
XLAB_AD = S.text(fig, (AX_L + AX_R) / 2 / W, (AX_B - 0.33) / H, XLAB_TEXT, "label", ha="center", va="baseline")
P = numbers["panels_a_to_d"]

# panel letters (a)-(d) and the criterion heads on one baseline: the letter at the left edge of the panel block, the
# head right after the letter (v7 Figure 5 layout)
LETTER_X = [0.02] + [AX_L + i * (AW + GAP) - 0.075 for i in (1, 2, 3)]
LET = {}
for i, l in enumerate("abcd"):
    LET[l] = S.panel_letter(fig, l, LETTER_X[i], HEAD_BASE)
    LET[l].set_va("baseline")
    LET[l].set_y(HEAD_BASE / H)
fig.canvas.draw()
for i, (ck, chead, _, _) in enumerate(CRITS):
    x_head = LET["abcd"[i]].get_window_extent(renderer).x1 / fig.dpi + 0.05
    HEADS.append(S.text(fig, x_head / W, HEAD_BASE / H, chead, "header", ha="left", va="baseline"))

# ===================================================================================================
# key for a-d: two rows, the rules covered by the theorem (solid lines) and the rules outside it (dashed lines);
# columns aligned, the whole key centred on the a-d block
# ===================================================================================================
KEY_ROWS = [ARMS[:4], ARMS[4:]]
HL, HG, CS = 0.25, 0.045, 0.16     # handle length, handle-text gap, column gap (in)
colw = []
for j in range(4):
    labs = [r[j][1] for r in KEY_ROWS if j < len(r)]
    colw.append(HL + HG + max(text_w(s, "key") for s in labs))
KEY_TOTAL = sum(colw) + 3 * CS
x0 = (AX_L + AX_R) / 2 - KEY_TOTAL / 2
KEY_TXT = []
for r, arms in enumerate(KEY_ROWS):
    xc = x0
    for j, arm in enumerate(arms):
        ak, kl, rl, cov, col, mk, face, ls, rk = arm
        yy = KEY_Y[r] + 0.026          # x-height centre of the 7.5 pt labels above their baseline
        KEY_ART.append(fig.add_artist(Line2D([xc / W, (xc + HL) / W], [yy / H, yy / H], transform=fig.transFigure,
                                             color=col, lw=LW["data"], ls=ls)))
        KEY_ART.append(fig.add_artist(Line2D([(xc + HL / 2) / W], [yy / H], transform=fig.transFigure, ls="none",
                                             **mstyle(arm))))
        KEY_TXT.append(S.text(fig, (xc + HL + HG) / W, KEY_Y[r] / H, kl, "key", ha="left", va="baseline"))
        xc += colw[j] + CS

# ===================================================================================================
# panel e: condition number of the learned gauge at N = infinity against the predictions of the theorem
# ===================================================================================================
runs_inf = {ak: pd.read_csv(f"{CRIT_DIR}/{ak}_Ninf_runs.csv") for ak in AK}


def kappa_pred(df):
    """part 1: cond of f(spec C0), f applied per run (columns eig_fC0_0..3 written by run_s13a.py)."""
    ok = df[~df.diverged.astype(bool)]
    f = ok[[f"eig_fC0_{i}" for i in range(4)]].to_numpy()
    return f.max(1) / f.min(1)


kp_flow = kappa_pred(runs_inf["P-GD-flow"])
band = np.quantile(kp_flow, [0.1, 0.5, 0.9])
kp_check = {a: float(np.median(kappa_pred(runs_inf[a]))) for a in ("P-GD", "P-SGD", "P-Adam", "WD")}

# part 3 for weight decay, per run from its own sample eigenvalues (N = inf: exact Sigma)
lam = meta["wd_lambda"]
wd_pred, wd_meas, wd_relerr = [], [], []
for f in NPZ_WD:
    z = np.load(f, allow_pickle=False)
    for r in range(len(z["run_ids"])):
        if bool(z["diverged"][r]):
            continue
        sp = np.sort(1.0 - lam / z["S_eigs"][r][:4])     # spec(E^T E) on U_hat = 1 - lam / lam_hat_i
        me = np.sort(z["spec_EEt"][r])
        wd_pred.append(sp[-1] / sp[0]); wd_meas.append(me[-1] / me[0])
        wd_relerr.append(float(np.max(np.abs(me - sp)) / sp.max()))
wd_pred, wd_meas = np.asarray(wd_pred), np.asarray(wd_meas)
kp_small = kappa_pred(runs_inf["small-init"])
pred_marker = {"P-GD-flow": float(band[1]), "small-init": float(np.median(kp_small)),
               "WD": float(np.median(wd_pred)), "TIED": 1.0}
pred_q = {"P-GD-flow": [float(v) for v in band],
          "small-init": [float(v) for v in np.quantile(kp_small, [0.1, 0.5, 0.9])],
          "WD": [float(v) for v in np.quantile(wd_pred, [0.1, 0.5, 0.9])], "TIED": [1.0, 1.0, 1.0]}
for _k in pred_q:
    assert abs(pred_q[_k][1] - pred_marker[_k]) <= 1e-12, _k
PRED_PART = {"P-GD-flow": "part 1", "small-init": "part 1", "WD": "part 3", "TIED": "part 3"}   # v8: was in row labels
pred_source = {
    "P-GD-flow": f"median over runs of cond(f(spec C0)), {CRIT_DIR}/P-GD-flow_Ninf_runs.csv :: eig_fC0_0..3",
    "small-init": f"median over runs of cond(f(spec C0)), {CRIT_DIR}/small-init_Ninf_runs.csv :: eig_fC0_0..3",
    "WD": "median over non-diverged runs of cond(1 - lambda_wd/lambda_hat_i), "
          f"{S1}/linae/runs/WD/Ninf/chunk*.npz :: S_eigs[:, :4]; lambda_wd = {SUM} :: meta['wd_lambda']",
    "TIED": "part 3: tied weights W = E^T give E E^T = I on U_hat, cond = 1 exactly"}

PRED_GAP_OWN_PT = 0.6             # prediction triangle: its tip this far (pt) above the top of its own median mark
PT_PER_ROW = (E_T - E_B) * 72 / (Y_BOT - Y_TOP)
assert abs(PT_PER_ROW - PT_ROW_TARGET) < 1e-9


def stroke_box(marker, ms, mew):
    """Extents (pt, relative to the marker centre: x0, y0, x1, y1) of a drawn marker, edge included, with the miter
    joins matplotlib uses on polygon markers (v5 r3, unchanged)."""
    from matplotlib.markers import MarkerStyle
    from matplotlib.path import Path as MPath
    st = MarkerStyle(marker)
    path = st.get_path().transformed(st.get_transform())
    e = path.get_extents()
    box = np.array([e.x0, e.y0, e.x1, e.y1]) * ms + np.array([-1, -1, 1, 1]) * mew / 2
    codes = path.codes
    poly = codes is None or set(int(c) for c in codes) <= {MPath.MOVETO, MPath.LINETO, MPath.CLOSEPOLY}
    if mew == 0 or str(st.get_joinstyle()) != "miter" or not poly:
        return box
    v = path.vertices * ms
    if np.allclose(v[0], v[-1]):
        v = v[:-1]
    sgn = 1.0 if np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]) > 0 else -1.0
    pts = []
    for i in range(len(v)):
        a, b, c = v[i - 1], v[i], v[(i + 1) % len(v)]
        d1, d2 = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        n1, n2 = sgn * np.array([d1[1], -d1[0]]), sgn * np.array([d2[1], -d2[0]])
        pts.append(b + (mew / 2) * (n1 + n2) / (1 + float(n1 @ n2)))
    pts = np.asarray(pts)
    return np.array([min(box[0], pts[:, 0].min()), min(box[1], pts[:, 1].min()),
                     max(box[2], pts[:, 0].max()), max(box[3], pts[:, 1].max())])


def marker_half_h(marker, ms, mew):
    b = stroke_box(marker, ms, mew)
    return max(abs(b[1]), abs(b[3]))


PRED_HALF = marker_half_h("v", MS["pred"], 0.0)
PRED_DY = {}
SHOW_COND1_LINE_E = False      # v8 c: as Figure 5 (c) (critic r1 there), no line at cond = 1
if SHOW_COND1_LINE_E:
    axe.axvline(1.0, color=S.NULL_LINE, lw=LW["zero"], zorder=Z["ref"])
PRED_ART, PRED_LINE = {}, {}

yt, yl = [], []
for yy, item in rows:
    ak, kl, rl, cov, col, mk, face, ls, rk = item
    yt.append(yy); yl.append(rl)
    rr = cells[f"{ak}|inf"]["runs"]
    q10, q50, q90 = rr["kappa_q10_50_90"]
    DATA.extend(axe.plot([q10, q90], [yy, yy], color=col, lw=LW["range_bar"], solid_capstyle="butt",
                         zorder=Z["interval"]))
    axe.plot([q50], [yy], ls="none", zorder=Z["marker"] - 0.1, **underlay(mstyle(item, "e")))
    DATA.extend(axe.plot([q50], [yy], ls="none", zorder=Z["marker"], **mstyle(item, "e")))
    if ak in pred_marker:   # above the median mark, pointing down at it
        _m = mstyle(item, "e")
        PRED_DY[ak] = (marker_half_h(_m["marker"], _m["ms"], _m["mew"]) + PRED_GAP_OWN_PT + PRED_HALF) / PT_PER_ROW
        PRED_ART[ak] = axe.plot([pred_marker[ak]], [yy - PRED_DY[ak]], ls="none", marker="v", ms=MS["pred"], mfc=INK,
                                mec=INK, mew=0, zorder=Z["marker"] + 0.5)[0]
        DATA.append(PRED_ART[ak])
        PRED_LINE[ak] = axe.plot([pred_q[ak][0], pred_q[ak][2]], [yy - PRED_DY[ak]] * 2, color=INK, lw=LW["errorbar"],
                                 solid_capstyle="butt", zorder=Z["marker"] + 0.4)[0]
        DATA.append(PRED_LINE[ak])
    numbers["panel_e"][ak] = dict(
        y_row=yy, row_label=rl, covered_by_theorem=cov,
        cond_q10_50_90=[q10, q50, q90],
        cond_source=f"{SUM} :: cells['{ak}|inf']['runs']['kappa_q10_50_90'] (quantiles over non-diverged runs)",
        n_runs=rr["n_runs"], n_diverged=rr["n_diverged"], n_runs_in_quantiles=rr["n_runs"] - rr["n_diverged"],
        C_drift_rel_median=rr["C_drift_rel_median"], flow_pred_relerr_median=rr["flow_pred_relerr_median"],
        runs_source=f"{SUM} :: cells['{ak}|inf']['runs'][n_runs, n_diverged, C_drift_rel_median, "
                    f"flow_pred_relerr_median]",
        predicted_marker=pred_marker.get(ak), predicted_source=pred_source.get(ak),
        predicted_q10_50_90=pred_q.get(ak))

axe.set_yticks(yt)
axe.set_yticklabels(yl)
axe.tick_params(axis="y", length=0, pad=3)
axe.spines["left"].set_visible(False)
axe.axhline(Y_SEP, color=S.NULL_LINE, lw=LW["guide"], zorder=Z["guide"])
axe.set_ylim(Y_BOT, Y_TOP)
axe.set_xscale("log")
axe.set_xlim(0.93, 8.6)
E_XT = [1, 1.5, 2, 3, 5]               # at most 5 labelled ticks; minor ticks only on this log axis (v5 r2)
E_XT_MINOR = [4, 6, 7, 8]
axe.set_xticks(E_XT)
axe.set_xticklabels([("%g" % v) for v in E_XT])
axe.xaxis.set_minor_locator(mticker.FixedLocator(E_XT_MINOR))
axe.xaxis.set_minor_formatter(mticker.NullFormatter())
E_XLABEL_R1 = "condition number of $EE^{\\top}$"     # v8 r1: \top fell back to a thin CM glyph in the bold title
E_XLABEL_R2 = "condition number of $EE^{\\mathrm{T}}$"  # v8 r2 (critic): bold upright T, same font as the title
E_XLABEL = "$\\mathrm{cond}(EE^{\\mathrm{T}})$"           # v8 c: the x title of Figure 5 (c), character for character
axe.set_xlabel(E_XLABEL)
_le = S.panel_letter(fig, "e", 0.02, E_T + 0.135)
LET["e"] = _le

# ===================================================================================================
# v8: bold ticks, axis titles and math; texts on a fill declared (from the drawn extents)
# ===================================================================================================
N_FINAL = S.finalize(fig)
fig.canvas.draw()


def _lum(h):
    return S.contrast(h, "#000000")


ON_FILL = {}
for t in fig.findobj(mtext.Text):
    if not (t.get_visible() and t.get_text().strip()):
        continue
    bb = t.get_window_extent(renderer)
    under = [fc for p, fc in FILLS if bb.overlaps(p.get_window_extent(renderer))]
    if under:
        fc = min(under, key=_lum)
        S.on_fill(t, fc)
        ON_FILL[t.get_text()] = fc

# ===================================================================================================
# save, check, render, views
# ===================================================================================================
PDF, PNG = os.path.join(HERE, STEM + ".pdf"), os.path.join(HERE, STEM + ".png")
PNG100, VIEWS = os.path.join(HERE, "_view_print_100dpi.png"), os.path.join(HERE, "_views_cvd_grey.png")
fig.savefig(PDF)
CHK = S.check(fig, PDF)
assert abs(CHK["size_in"][0] - 5.5) < 1e-9 and abs(CHK["size_in"][1] - H) < 1e-9, CHK
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
assert not any("Type 3" in ln for ln in pf.splitlines()), pf
S.print_view(PDF, PNG, PNG100)
S.views(PNG, VIEWS)
assert set(CHK["chromatic_data_colours"]) >= {S.ARM[k].upper() for k in ("gd", "smallinit", "wd", "tied", "sgd", "adam")}


def text_boxes():
    out = []
    for t in fig.findobj(mtext.Text):
        if t.get_visible() and t.get_text().strip():
            bb = t.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            out.append((t.get_text(), bb))
    return out


TB = text_boxes()
clash = [(a, b) for i, (a, ba) in enumerate(TB) for (b, bb) in TB[i + 1:] if ba.overlaps(bb)
         and ba.intersection(ba, bb).width > 0.005 and ba.intersection(ba, bb).height > 0.005]
outside = [(a, [round(v, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)]) for a, bb in TB
           if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]
assert not outside, outside
assert not clash, clash

# v6 r2 rule, kept in v8 form: the a-d x label belongs to its own axes. Its gap to the top of the e block (the e axes
# and the letter (e)) is at least 4.5 pt and larger than its gap to the a-d tick labels (layout boxes, pt).
_in = fig.dpi_scale_trans.inverted()
_xl = XLAB_AD.get_window_extent(renderer).transformed(_in)
_tk = min(t.get_window_extent(renderer).transformed(_in).y0 for ax in axs for t in ax.get_xticklabels()
          if t.get_visible() and t.get_text())
_e_top = max(E_T, max(t.get_window_extent(renderer).transformed(_in).y1 for t in axe.get_yticklabels()))
XLAB_GAP = dict(to_top_of_e_axes_pt=round((_xl.y0 - _e_top) * 72, 2),
                to_a_to_d_tick_labels_pt=round((_tk - _xl.y1) * 72, 2),
                top_text_headroom_pt=round((H - max(bb.y1 for _, bb in TB)) * 72, 2))
assert XLAB_GAP["to_top_of_e_axes_pt"] >= 4.5 and XLAB_GAP["to_top_of_e_axes_pt"] > XLAB_GAP["to_a_to_d_tick_labels_pt"], \
    XLAB_GAP
assert XLAB_GAP["to_a_to_d_tick_labels_pt"] >= 2.0, XLAB_GAP
assert XLAB_GAP["top_text_headroom_pt"] >= 0.5, XLAB_GAP
# v8: the heads sit above the axes, clear of the data at 1 (marker tops) by >= 2 pt; the key row above the heads
_head_y0 = min(t.get_window_extent(renderer).transformed(_in).y0 for t in HEADS + [LET[l] for l in "abcd"])
_mark_top = AX_B + (1.0 + 0.045) / 1.095 * AX_H + MS["line"] / 2 / 72
HEAD_GAP_PT = round((_head_y0 - _mark_top) * 72, 2)
assert HEAD_GAP_PT >= 2.0, HEAD_GAP_PT
_key_y0 = min(t.get_window_extent(renderer).transformed(_in).y0 for t in KEY_TXT)
_head_y1 = max(t.get_window_extent(renderer).transformed(_in).y1 for t in HEADS + [LET[l] for l in "abcd"])
KEY_HEAD_GAP_PT = round((_key_y0 - _head_y1) * 72, 2)
assert KEY_HEAD_GAP_PT >= 1.5, KEY_HEAD_GAP_PT


# --- text against data marks: marker outlines, curves and bars (inches; text boxes are layout boxes) ---
def _marker_half(line):
    m = line.get_marker()
    if m in (None, "None", "", " "):
        return None
    b = stroke_box(m, line.get_markersize(), line.get_markeredgewidth())
    return max(abs(b[0]), abs(b[2])) / 72, max(abs(b[1]), abs(b[3])) / 72


def _marker_box(line):
    return [v / 72 for v in stroke_box(line.get_marker(), line.get_markersize(), line.get_markeredgewidth())]


def data_shapes(arts):
    import matplotlib.collections as mcoll
    boxes, segs = [], []
    for a in arts:
        if isinstance(a, mcoll.LineCollection):
            lw = float(np.max(a.get_linewidth())) / 72
            for sg in a.get_segments():
                p = a.get_transform().transform(np.asarray(sg)) / fig.dpi
                segs.append((p[0], p[1], lw / 2))
            continue
        pts = a.get_transform().transform(np.column_stack(a.get_data())) / fig.dpi
        hm = _marker_half(a)
        if hm is not None:
            boxes += [(x - hm[0], y - hm[1], x + hm[0], y + hm[1]) for x, y in pts]
        if a.get_linestyle() not in ("None", "", " ") and a.get_linewidth() > 0:
            segs += [(pts[i], pts[i + 1], a.get_linewidth() / 2 / 72) for i in range(len(pts) - 1)]
    return boxes, segs


def rect_gap(r, q):
    dx = max(q[0] - r[2], r[0] - q[2], 0.0)
    dy = max(q[1] - r[3], r[1] - q[3], 0.0)
    inside = q[0] < r[2] and r[0] < q[2] and q[1] < r[3] and r[1] < q[3]
    return -1.0 if inside else float(np.hypot(dx, dy))


def seg_gap(r, sg):
    p0, p1, hw = sg
    t = np.linspace(0, 1, 400)[:, None]
    pts = p0 + t * (p1 - p0)
    dx = np.maximum.reduce([r[0] - pts[:, 0], pts[:, 0] - r[2], np.zeros(len(pts))])
    dy = np.maximum.reduce([r[1] - pts[:, 1], pts[:, 1] - r[3], np.zeros(len(pts))])
    d = np.hypot(dx, dy) - hw
    return float(d.min())


def text_box_in(t):
    bb = t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    return (bb.x0, bb.y0, bb.x1, bb.y1)


fig.canvas.draw()
BOXES, SEGS = data_shapes(DATA + KEY_ART)
CLEAR = []
for t in fig.findobj(mtext.Text):
    if not (t.get_visible() and t.get_text().strip()):
        continue
    r = text_box_in(t)
    g = min([rect_gap(r, q) for q in BOXES] + [seg_gap(r, sg) for sg in SEGS])
    CLEAR.append((t.get_text(), round(g, 4)))
HIT = [c for c in CLEAR if c[1] < 0]
if HIT:
    for t in fig.findobj(mtext.Text):
        if t.get_visible() and t.get_text() in dict(HIT):
            r = text_box_in(t)
            print("HIT", repr(t.get_text()), [round(v, 3) for v in r],
                  [tuple(round(v, 3) for v in q) for q in BOXES if rect_gap(r, q) < 0],
                  [(tuple(np.round(sg[0], 3)), tuple(np.round(sg[1], 3))) for sg in SEGS if seg_gap(r, sg) < 0])
assert not HIT, HIT

# --- e: each prediction triangle against the median marks of the OTHER rows (its own row's mark is its target) ---
PRED_GAP = {}
for ak, art in PRED_ART.items():
    (px, py), = art.get_transform().transform(np.column_stack(art.get_data())) / fig.dpi
    tb = _marker_box(art)
    tri = (px + tb[0], py + tb[1], px + tb[2], py + tb[3])
    gaps = {}
    for ak2 in AK:
        line = [a for a in DATA if isinstance(a, Line2D) and a.get_marker() == ARM_OF[ak2][5]
                and a.axes is axe and a.get_linestyle() in ("None", "", " ")
                and abs(a.get_ydata()[0] - numbers["panel_e"][ak2]["y_row"]) < 1e-9]
        assert len(line) == 1, (ak2, line)
        (mx, my), = line[0].get_transform().transform(np.column_stack(line[0].get_data())) / fig.dpi
        mb = _marker_box(line[0])
        gaps[ak2] = round(rect_gap(tri, (mx + mb[0], my + mb[1], mx + mb[2], my + mb[3])) * 72, 2)   # pt
    PRED_GAP[ak] = gaps
    others = {k: v for k, v in gaps.items() if k != ak}
    assert min(others.values()) >= 0.3, ("prediction triangle on another row's mark", ak, gaps)
    assert gaps[ak] >= 0.3, ("prediction triangle touches its own mark", ak, gaps)

# v8: every prediction triangle lies inside the e axes (the first row is a covered rule now), >= 1 pt below the top
TRI_TOP_GAP_PT = {}
for ak, art in PRED_ART.items():
    (px, py), = art.get_transform().transform(np.column_stack(art.get_data())) / fig.dpi
    TRI_TOP_GAP_PT[ak] = round((E_T - (py + _marker_box(art)[3])) * 72, 2)
assert min(TRI_TOP_GAP_PT.values()) >= 1.0, TRI_TOP_GAP_PT
ARM_W = {w: {ak: round(arm_width_pt(ARM_OF[ak], w), 3) for ak in ("P-SGD", "P-Adam")} for w in ("ad", "e")}
assert min(ARM_W["ad"].values()) >= LW["data"] and min(ARM_W["e"].values()) >= LW["range_bar"], ARM_W
for _w, _size in (("ad", MS["line"]), ("e", MS["summary"])):
    for _a in ARMS:
        _m = mstyle(_a, _w)
        if _m["marker"] in ("o", "s"):
            assert abs(_m["ms"] + _m["mew"] - _size) < 1e-9, (_a[0], _w, _m)

# v8: one row of e per rule, labels = key labels = the rule names; no text states a setting (asserted on the drawn texts)
DRAWN = sorted({t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()})
BANNED = ["theorem", "part", "batch", "preregistered", "secondary", "reliable", "conserved", "48", "500",
          "anisotropy", "measured", "predicted", "\\leq"]
assert not [s for s in DRAWN if any(b in s for b in BANNED)], [s for s in DRAWN if any(b in s for b in BANNED)]
assert [t.get_text() for t in axe.get_yticklabels()] == [a[2] for a in ARMS]

# ===================================================================================================
# cross-checks against the v3, v4, v5.1 and v6 numbers (all previous asserts kept; v8 differences are geometry and
# colour fields only, named in numbers.json v8.changed_fields)
# ===================================================================================================
for ck in P:
    for ak in P[ck]:
        for N in NS:
            a, b, c4 = P[ck][ak][N], old["panels_a_to_d"][ck][ak][N], v4num["panels_a_to_d"][ck][ak][N]
            for k in ("passes", "n_pairs", "p_hat", "cp_lower", "cp_upper", "decision", "n_fail_by_divergence"):
                assert a[k] == b[k], (ck, ak, N, k, a[k], b[k])
            assert a == c4, (ck, ak, N, a, c4)
_SKIP_E = {"y_row", "row_label", "predicted_q10_50_90"}   # y_row is geometry; row labels changed (v6 r2, v8)
for ak in AK:
    assert numbers["panel_e"][ak]["cond_q10_50_90"] == old["panel_e"][ak]["kappa_q10_50_90"], ak
    assert numbers["panel_e"][ak]["predicted_marker"] == old["panel_e"][ak]["thm4_prediction_marker"], ak
    assert {k: v for k, v in numbers["panel_e"][ak].items() if k not in _SKIP_E} == \
        {k: v for k, v in v4num["panel_e"][ak].items() if k not in _SKIP_E}, ak
    # v6 asserted y_row equal to v5.1 too; v8 removed the two group-head rows of e, so y_row is now excluded as
    # geometry (as against v4)
    assert {k: v for k, v in numbers["panel_e"][ak].items() if k not in _SKIP_E} == \
        {k: v for k, v in v51num["panel_e"][ak].items() if k not in ("row_label", "y_row")}, ak
    assert v51num["panel_e"][ak]["row_label"] == ROW_LABELS_V51.get(ak, v51num["panel_e"][ak]["row_label"]), ak
    assert v6num["panel_e"][ak]["row_label"] == ROW_LABELS_V6[ak], (ak, v6num["panel_e"][ak]["row_label"])
    assert {k: v for k, v in json.loads(json.dumps(numbers["panel_e"][ak], default=float)).items()
            if k not in ("y_row", "row_label")} == \
        {k: v for k, v in v6num["panel_e"][ak].items() if k not in ("y_row", "row_label")}, ak
for ck in P:   # a-d entries equal to v5.1 and v6 (values, plotted positions, sources)
    assert P[ck] == v51num["panels_a_to_d"][ck], ck
    assert json.loads(json.dumps(P[ck], default=float)) == v6num["panels_a_to_d"][ck], ck
FG_NUM = f"{HOME}/hrh_v2_figs/v6/F_gauge/numbers.json"
FG_CHECK = {"path": FG_NUM}
try:
    _fg = json.load(open(FG_NUM))
    _fgq = _fg.get("panel_c_predicted_q10_50_90") or _fg.get("c_predicted_q10_50_90")
    FG_CHECK.update(sha256=sha(FG_NUM), equal_to_1e12={k: bool(np.allclose(pred_q[k], _fgq[k], rtol=0, atol=1e-12))
                                                          for k in pred_q})
    assert all(FG_CHECK["equal_to_1e12"].values()), FG_CHECK
except (OSError, ValueError, KeyError, TypeError) as _e:
    FG_CHECK.update(status="not compared: %r" % (_e,))
ob = old["panel_e"]["flow_prediction_band_default_init"]
assert np.allclose(band, ob["kappa_pred_q10_50_90"], rtol=0, atol=1e-12)
assert kp_check == ob["crn_check_median_kappa_pred_same_init"]
ow = old["panel_e"]["thm43_weight_decay_check"]
assert float(np.median(wd_pred)) == ow["kappa_pred_median"] and float(np.median(wd_meas)) == ow["kappa_meas_median"]
assert round(float(np.median(wd_pred)), 3) == 1.033 and pred_marker["TIED"] == 1.0

# ===================================================================================================
# numbers.json
# ===================================================================================================
inputs = {p: sha(p) for p in [SUM, LINAE_CORE, OLD, V4NUM, V51NUM, V6NUM, V6MAKER]
          + [f"{CRIT_DIR}/{ak}_Ninf_runs.csv" for ak in AK] + NPZ_WD}
c_inf = lambda ak, ck: cells[f"{ak}|inf"]["primary"][ck]  # noqa: E731
NOW_UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
out = {
    "figure": {"id": "F_linae_app", "label": "fig:exp-linae", "version": "v8",
               "v4": "arxiv_v4.tex, figure environment with \\label{fig:exp-linae} (appendix F); locate by label",
               "message": "Gradient descent without a regularizer stays unreliable at every sample size on the "
                          "criteria that see the gauge, whereas weight decay, tied weights, and small initialization "
                          "become reliable (Theorem thm:gauge).",
               "size_in": CHK["size_in"], "min_text_pt": CHK["min_text_pt"],
               "min_glyph_pt_from_pdf_Tf": CHK["min_glyph_pt"], "pdf_glyph_size_counts": CHK["pdf_glyph_size_counts"],
               "smallest_layout_glyphs": CHK["smallest_layout_glyphs"],
               "mathtext_script_scale": CHK["mathtext_shrink_factor"], "n_type3": CHK["n_type3"],
               "fonts": CHK["fonts"], "pdffonts": pf.splitlines(),
               "low_contrast_text": CHK["low_contrast_text"], "text_on_fill_declared": ON_FILL,
               "text_overlaps_gt_0.005in": clash, "text_vs_data_overlaps": HIT,
               "closest_text_to_data_in": sorted(CLEAR, key=lambda c: c[1])[:12],
               "e_prediction_triangle_gap_to_median_marks_pt": PRED_GAP,
               "e_triangle_to_axes_top_pt": TRI_TOP_GAP_PT,
               "a_to_d_xlabel_gaps_pt_layout_boxes": XLAB_GAP,
               "heads_to_data_top_gap_pt": HEAD_GAP_PT, "key_to_heads_gap_pt": KEY_HEAD_GAP_PT,
               "drawn_texts": DRAWN, "n_texts_set_bold_by_finalize": N_FINAL,
               "generated_utc": NOW_UTC},
    "inputs": inputs,
    "style": {"path": STYLE_PATH, "sha256": STYLE_SHA, "version": S.STYLE_VERSION,
              "elements_pt": {"rule_lines_a_to_d": LW["data"], "break_segments_dotted": LW["break"],
                              "cp_bars": LW["errorbar"], "decision_levels": LW["level"], "axis_break": LW["axis_break"],
                              "markers_a_to_d_across_with_edge": MS["line"], "marker_edge_in_rule_colour": EDGE,
                              "open_marker_edge_a_to_d": MS["open_small_mew"], "range_bars_e": LW["range_bar"],
                              "median_marks_e_across_with_edge": MS["summary"],
                              "open_marker_edge_e": MS["open_line_mew"],
                              "marker_kw": {w: {ak: {k: (round(v, 4) if isinstance(v, float) else v)
                                                     for k, v in mstyle(ARM_OF[ak], w).items()} for ak in AK}
                                            for w in ("ad", "e")},
                              "x_and_plus_arm_width_with_edge": ARM_W,
                              "white_underlay_e_extra_edge": LW["median_halo"],
                              "predicted_triangle_e": MS["pred"],
                              "cond_1_line_e": LW["zero"] if SHOW_COND1_LINE_E else "not drawn (v8 c, as Figure 5 c)",
                              "group_separator_e": LW["guide"], "marker_shape_scale": SHAPE},
              "check": CHK},
    "script_sha256": sha(os.path.abspath(__file__)),
    "crosscheck_v3": {"path": OLD, "sha256": sha(OLD),
                      "result": "panels_a_to_d (passes, n_pairs, p_hat, cp bounds, decision, divergences), "
                                "panel_e cond quantiles, prediction markers, flow band, CRN check and the "
                                "weight-decay check equal F5/numbers.json exactly (asserted)"},
    "crosscheck_v4": {"path": V4NUM, "sha256": sha(V4NUM),
                      "result": "panels_a_to_d entries and panel_e entries (y_row, row_label and the prediction range "
                                "excepted) equal the v4 numbers.json exactly (asserted)"},
    "summary_meta": dict(prereg_sha256=meta["prereg_sha256"], spec_sha256=meta["spec_sha256"],
                         analysis_plan_sha256=meta["analysis_plan_sha256"], eps=EPS, wd_lambda=lam,
                         small_sigma0=meta["small_sigma0"], arms={ak: meta["arms"][ak] for ak in AK},
                         source=f"{SUM} :: meta"),
    "encoding": {"x_a_to_d": "log2 N (tick labels 2^8, 2^12, 2^16 at 8 pt); N = inf drawn at 20.5 after an axis break "
                             "at 18.25; light column (S.REF_FILL) of half-width %g behind each N; per-rule offset "
                             "in log2 N = " % COL_HW
                             + json.dumps({ak: round(float(d), 4) for ak, d in zip(AK, DODGE)}),
                 "lines_a_to_d": "solid = covered by the theorem, dashed = outside it; dotted = the segment across "
                                 "the axis break (2^16 to infinity), in the rule colour",
                 "bars_a_to_d": "one-sided 95% Clopper-Pearson lower and upper bounds (preregistered decision "
                                "bounds); reliable if lower > 0.9, unreliable if upper < 0.8",
                 "grey_lines_a_to_d": [REL_LO, UNREL_HI],
                 "e": "v8: one row per rule, labelled by the rule's name; measured: segment = 10th to 90th percentile "
                      "over non-diverged runs at N = inf, mark = median (rule colour); predicted (covered rules): INK "
                      "down-triangle = median over runs of the per-run predicted value (%s), its tip %g pt above the "
                      "median mark, on an INK line (%g pt) over the 10th to 90th percentile of the per-run "
                      "predictions; rows of the covered rules %g apart, a gap of %g and a thin line before the rules "
                      "outside the theorem (%.3f pt per row); x axis log, 0.93 to 8.6, labelled ticks %s, minor ticks "
                      "%s; no title, no key (the caption decodes the marks)"
                      % (json.dumps(PRED_PART), PRED_GAP_OWN_PT, LW["errorbar"], COV_PITCH, GROUP_GAP, PT_PER_ROW,
                         E_XT, E_XT_MINOR),
                 "colours": {ak: a[4] for ak, a in zip(AK, ARMS)},
                 "text_tones": {ak: S.ARM_TEXT[a[8]] for ak, a in zip(AK, ARMS)},
                 "markers": {ak: a[5] + ("(open)" if a[6] == "open" else "") for ak, a in zip(AK, ARMS)}},
    "panels_a_to_d": P,
    "panel_e": numbers["panel_e"],
    "panel_e_band": dict(cond_f_spec_C0_q10_50_90=[float(v) for v in band], n_runs=int(len(kp_flow)),
                         source=f"{CRIT_DIR}/P-GD-flow_Ninf_runs.csv :: eig_fC0_0..3 (non-diverged rows), "
                                "cond = max/min per run, then quantiles 0.1/0.5/0.9",
                         crn_check_median_cond_f_spec_C0_same_init=kp_check),
    "panel_e_weight_decay_check": dict(
        status="computed for the figure, not part of the preregistered analysis",
        cond_pred_median=float(np.median(wd_pred)), cond_meas_median=float(np.median(wd_meas)),
        spec_relerr_median=float(np.median(wd_relerr)), spec_relerr_max=float(np.max(wd_relerr)),
        n_runs=int(len(wd_pred)), source=f"{S1}/linae/runs/WD/Ninf/chunk0..7.npz :: S_eigs, spec_EEt, diverged"),
    "quoted_in_v4_text_and_caption": {
        "n_pairs_finite_N": cells["P-GD|4096"]["primary"]["kNN"]["n_pairs"],
        "n_pairs_inf": cells["P-GD|inf"]["primary"]["kNN"]["n_pairs"],
        "n_runs_per_rule_inf": cells["P-GD|inf"]["runs"]["n_runs"],
        "P-GD_C_drift_rel_median(53%)": cells["P-GD|inf"]["runs"]["C_drift_rel_median"],
        "P-GD-flow_C_drift_rel_median(0.18%)": cells["P-GD-flow|inf"]["runs"]["C_drift_rel_median"],
        "P-GD-flow_flow_pred_relerr_median(0.1%)": cells["P-GD-flow|inf"]["runs"]["flow_pred_relerr_median"],
        "P-GD-flow_cond_q10_50_90(2.98,4.66,7.43)": cells["P-GD-flow|inf"]["runs"]["kappa_q10_50_90"],
        "WD_cond_predicted(1.033)": float(np.median(wd_pred)),
        "P-SGD_cond_median(1.19)": cells["P-SGD|inf"]["runs"]["kappa_q10_50_90"][1],
        "P-Adam_cond_median(1.54)": cells["P-Adam|inf"]["runs"]["kappa_q10_50_90"][1],
        "P-SGD_CKA_passes_inf(322/500)": c_inf("P-SGD", "CKA")["passes"],
        "P-Adam_CKA_passes_inf(172/500)": c_inf("P-Adam", "CKA")["passes"],
        "source": f"{SUM} :: cells[...]['runs'|'primary']; WD from the chunk*.npz check above"},
    "prereg_verdicts": {k: dict(statement=v.get("statement"), verdict=v.get("verdict"),
                                source=f"{SUM} :: verdicts['{k}']")
                        for k, v in summ["verdicts"].items() if isinstance(v, dict) and "verdict" in v},
}
for k in ("quoted_in_v4_text_and_caption", "panel_e_band", "panel_e_weight_decay_check", "prereg_verdicts",
          "summary_meta"):
    assert json.loads(json.dumps(out[k], default=float)) == v4num[k], ("numbers.json content differs from v4", k)
for k in ("panels_a_to_d", "panel_e_band", "panel_e_weight_decay_check", "quoted_in_v4_text_and_caption",
          "prereg_verdicts", "summary_meta"):
    assert json.loads(json.dumps(out[k], default=float)) == v51num[k], ("numbers.json content differs from v5.1", k)
    assert json.loads(json.dumps(out[k], default=float)) == v6num[k], ("numbers.json content differs from v6", k)
# encoding: v6 asserted equal to v5.1 except 'e'; v8 changes 'e' (rows), 'colours' and 'text_tones' (Tol palette)
for k in v51num["encoding"]:
    assert k in ("e", "colours", "text_tones") or out["encoding"][k] == v51num["encoding"][k], k
    assert k in ("e", "colours", "text_tones") or out["encoding"][k] == v6num["encoding"][k], k
_COL_CHANGE = {ak: [v6num["encoding"]["colours"][ak], out["encoding"]["colours"][ak]] for ak in AK
               if v6num["encoding"]["colours"][ak] != out["encoding"]["colours"][ak]}
assert sorted(_COL_CHANGE) == ["P-GD", "P-GD-flow"] and all(v == ["#000000", "#222255"] for v in _COL_CHANGE.values()), \
    _COL_CHANGE
out["crosscheck_v5_1"] = {"path": V51NUM, "sha256": V51NUM_SHA,
                          "result": "panels_a_to_d, panel_e_band, panel_e_weight_decay_check, "
                                    "quoted_in_v4_text_and_caption, prereg_verdicts and summary_meta equal the v5.1 "
                                    "numbers.json exactly; panel_e equal except row_label, y_row (geometry) and the "
                                    "prediction range added in v6 r2 (asserted)",
                          "row_labels_v5_1": ROW_LABELS_V51}
out["crosscheck_v6"] = {"path": V6NUM, "sha256": V6NUM_SHA,
                        "result": "panels_a_to_d, panel_e (except y_row and row_label), panel_e_band, "
                                  "panel_e_weight_decay_check, quoted_in_v4_text_and_caption, prereg_verdicts, "
                                  "summary_meta and results_text_numbers equal the v6 numbers.json exactly; encoding "
                                  "equal except e, colours and text_tones (asserted)",
                        "row_labels_v6": ROW_LABELS_V6, "colour_changes_v6_to_v8": _COL_CHANGE}
out["crosscheck_F_gauge_c_prediction_ranges"] = FG_CHECK

SUM_SHA = sha(SUM)
G3 = ("kNN", "Gs", "CKA")                    # the three criteria that see the gauge
NB = ("4096", "65536", "inf")                # N >= 2^12
REG = ("small-init", "WD", "TIED")


def cl(ak, N, ck):
    return cells[f"{ak}|{N}"]["primary"][ck]


def frac(ak, N, ck):
    c = cl(ak, N, ck)
    return "%d/%d" % (c["passes"], c["n_pairs"])


def cellkey(aks, Ns, cks, fields):
    return "cells['<%s>|<%s>']['primary']['<%s>'][%s]" % ("|".join(aks), "|".join(Ns), "|".join(cks), fields)


def runkey(aks, fields):
    return "cells['<%s>|inf']['runs'][%s]" % ("|".join(aks), fields)


RES = []


def res(id_, printed, value, in_v4, use, key, source=SUM, sha256=SUM_SHA):
    RES.append(dict(id=id_, printed=printed, value=value, in_v4=in_v4, use_in_results=use, source=source,
                    sha256=sha256, key=key))


# results_text_numbers: the v6 blocks, rebuilt from the same cells and asserted equal to the v6 numbers.json below
_gd = {ak: {ck: {N: frac(ak, N, ck) for N in NS} for ck in G3} for ak in ("P-GD", "P-GD-flow")}
assert all(cl(ak, N, ck)["passes"] == 0 and cl(ak, N, ck)["decision"] == "unreliable"
           for ak in ("P-GD", "P-GD-flow") for ck in G3 for N in NS)
res("gd_no_regularizer_zero_passes",
    "0 of %d pairs at each finite N and 0 of %d at N = inf on neighbor sets, G_s and CKA, at step 0.05 "
    "(preregistered) and step 0.002; unreliable in all 24 cells" % (N_PAIRS_FIN, N_PAIRS_INF),
    _gd, "caption (a-d)", "Results (a-c): the first half of the message", cellkey(("P-GD", "P-GD-flow"), NS, G3,
                                                                                   "passes, n_pairs, decision"))
_reg = {ak: {ck: {N: dict(passes=cl(ak, N, ck)["passes"], n_pairs=cl(ak, N, ck)["n_pairs"],
                          cp_lower=cl(ak, N, ck)["cp_lower"], decision=cl(ak, N, ck)["decision"]) for N in NB}
             for ck in EPS} for ak in REG}
assert all(v["decision"] == "reliable" for a in _reg.values() for c in a.values() for v in c.values())
_same = {ak: {N: {frac(ak, N, ck) for ck in EPS} for N in NB} for ak in REG}
assert all(len(s) == 1 for a in _same.values() for s in a.values()), _same
_cnt = {ak: [frac(ak, N, "kNN") for N in NB] for ak in REG}
_minlo = min(v["cp_lower"] for a in _reg.values() for c in a.values() for v in c.values())
res("regularized_reliable_from_2^12",
    "on all four criteria: small init. %s, tied weights %s, weight decay %s pairs (N = 2^12, 2^16, inf); "
    "smallest one-sided lower bound %.3f (> 0.9)" % (", ".join(_cnt["small-init"]), ", ".join(_cnt["TIED"]),
                                                      ", ".join(_cnt["WD"]), _minlo),
    _reg, "caption (a-d), without counts", "Results (a-d): the second half of the message",
    cellkey(REG, NB, tuple(EPS), "passes, n_pairs, cp_lower, decision"))
_r8 = {ak: {ck: dict(frac=frac(ak, "256", ck), decision=cl(ak, "256", ck)["decision"],
                     n_fail_by_divergence=cl(ak, "256", ck)["n_fail_by_divergence"]) for ck in EPS} for ak in REG}
assert all(_r8[ak][ck]["frac"] == "0/48" and _r8[ak][ck]["decision"] == "unreliable" for ak in REG for ck in ("kNN", "Gs"))
assert all(_r8[ak][ck]["frac"] == "48/48" for ak in ("small-init", "TIED") for ck in ("CKA", "proj"))
assert all(_r8["WD"][ck]["frac"] == "46/48" and _r8["WD"][ck]["n_fail_by_divergence"] == 2 for ck in ("CKA", "proj"))
res("regularized_at_2^8",
    "at N = 2^8 the three rules are unreliable on neighbor sets and G_s (0/48 each); on CKA and the projector small "
    "init. and tied weights pass 48/48 (reliable) and weight decay 46/48 (inconclusive; both failing pairs contain a "
    "diverged run)", _r8, "not in v4 (visible in a-d)", "Results (a, b): the onset at 2^12 is on neighbor sets and G_s",
    cellkey(REG, ("256",), tuple(EPS), "passes, n_pairs, decision, n_fail_by_divergence"))
_pj = {ak: {N: dict(frac=frac(ak, N, "proj"), decision=cl(ak, N, "proj")["decision"]) for N in NS} for ak in AK}
assert all(_pj[ak][N]["decision"] == "reliable" for ak in AK for N in NB)
_pj8 = sorted(ak for ak in AK if _pj[ak]["256"]["decision"] != "reliable")
assert _pj8 == ["P-GD", "WD"] and all(_pj[ak]["256"]["decision"] == "inconclusive" for ak in _pj8)
res("projector_reliable_every_rule",
    "reliable for all seven rules at N >= 2^12 (fewest passes %s at 2^12 and 2^16, %s at inf, GD step 0.05); at 2^8 "
    "GD step 0.05 and weight decay are inconclusive (%s, %s), the other five reliable"
    % (frac("P-GD", "4096", "proj"), frac("P-GD", "inf", "proj"), frac("P-GD", "256", "proj"),
       frac("WD", "256", "proj")),
    _pj, "caption (a-d)", "Results (d): part 4 of the theorem; P3", cellkey(AK, NS, ("proj",), "passes, n_pairs, decision"))
_fail = {}
for ak in AK:
    for N in NB:
        for ck in (tuple(EPS) if ak == "WD" else ("proj",)):
            c = cl(ak, N, ck)
            if c["passes"] < c["n_pairs"]:
                assert c["n_pairs"] - c["passes"] == c["n_fail_by_divergence"], (ak, N, ck, c)
                _fail.setdefault(ak, {})[N] = c["n_fail_by_divergence"]
assert sorted(_fail) == ["P-GD", "P-SGD", "WD"] and list(_fail["P-SGD"]) == ["inf"], _fail
_div_phase = {ak: dict(n_diverged=cells[f"{ak}|inf"]["runs"]["n_diverged"],
                       stochastic_phase=cells[f"{ak}|inf"]["runs"]["n_diverged_stochastic_phase"],
                       gd_or_polish=cells[f"{ak}|inf"]["runs"]["n_diverged_gd_or_polish"]) for ak in _fail}
assert _div_phase["P-SGD"]["stochastic_phase"] == _fail["P-SGD"]["inf"] == 2
res("failures_are_divergences",
    "every failing pair at N >= 2^12 (projector, and every criterion for weight decay) contains a diverged run: "
    "GD step 0.05 %s, weight decay %s pairs (2^12, 2^16, inf; both at step 0.05), SGD %d pairs at inf (minibatch "
    "phase)" % (", ".join(str(_fail["P-GD"][N]) for N in NB), ", ".join(str(_fail["WD"][N]) for N in NB),
                _fail["P-SGD"]["inf"]),
    dict(failing_pairs=_fail, diverged_runs_at_inf=_div_phase), "caption (a-d)", "Results (d) or a footnote",
    cellkey(AK, NB, tuple(EPS), "passes, n_pairs, n_fail_by_divergence") + "; "
    + runkey(tuple(_fail), "n_diverged, n_diverged_stochastic_phase, n_diverged_gd_or_polish"))
assert all(cl(ak, N, ck)["decision"] == "unreliable" for ak in ("P-SGD", "P-Adam") for ck in G3 for N in NS)
_sa = {ak: {ck: dict(frac=frac(ak, "inf", ck), p_hat=cl(ak, "inf", ck)["p_hat"], cp_upper=cl(ak, "inf", ck)["cp_upper"])
            for ck in G3} for ak in ("P-SGD", "P-Adam")}
res("sgd_adam_unreliable",
    "unreliable on neighbor sets, G_s and CKA at every N; at N = inf SGD passes %s (neighbor sets), %s (G_s), %s "
    "(CKA) and Adam %s, %s, %s" % tuple(_sa[ak][ck]["frac"] for ak in ("P-SGD", "P-Adam") for ck in G3),
    _sa, "caption (322 and 172 of 500 on CKA in the v4 text)", "Results (a-c): rules outside the theorem; P4 "
    "(no prediction)", cellkey(("P-SGD", "P-Adam"), NS, G3, "passes, n_pairs, decision, p_hat, cp_upper"))
_nr = {ak: dict(n_runs=cells[f"{ak}|inf"]["runs"]["n_runs"], n_in_quantiles=numbers["panel_e"][ak]["n_runs_in_quantiles"])
       for ak in AK}
assert {v["n_runs"] for v in _nr.values()} == {2 * N_PAIRS_INF}
res("runs_per_rule_e", "%d runs per rule at N = inf (%s non-diverged for GD step 0.05, %s weight decay, %s SGD)"
    % (2 * N_PAIRS_INF, _nr["P-GD"]["n_in_quantiles"], _nr["WD"]["n_in_quantiles"], _nr["P-SGD"]["n_in_quantiles"]),
    _nr, "caption (e): '1000 runs'", "Results (e) or the setup paragraph", runkey(AK, "n_runs, n_diverged"))
_rf, _rg = cells["P-GD-flow|inf"]["runs"], cells["P-GD|inf"]["runs"]
res("C_drift_and_flow_fit",
    "relative drift of C: median %.2f%% at step 0.002 against %.0f%% at step 0.05; learned spectrum = f(spec C0) to a "
    "median relative error of %.1f%% at step 0.002" % (100 * _rf["C_drift_rel_median"], 100 * _rg["C_drift_rel_median"],
                                                       100 * _rf["flow_pred_relerr_median"]),
    dict(C_drift_rel_median={"P-GD-flow": _rf["C_drift_rel_median"], "P-GD": _rg["C_drift_rel_median"]},
         flow_pred_relerr_median={"P-GD-flow": _rf["flow_pred_relerr_median"], "P-GD": _rg["flow_pred_relerr_median"]}),
    "Appendix F text (Gradient Flow and Other Optimizers)", "Results (e): why step 0.002 stands in for gradient flow",
    runkey(("P-GD-flow", "P-GD"), "C_drift_rel_median, flow_pred_relerr_median"))
_q = {ak: numbers["panel_e"][ak]["cond_q10_50_90"] for ak in AK}
res("cond_flow_vs_band",
    "step 0.002: median %.2f (10-90%%: %.2f-%.2f); predicted cond f(spec C0) over the same runs: %.2f (%.2f-%.2f)"
    % (_q["P-GD-flow"][1], _q["P-GD-flow"][0], _q["P-GD-flow"][2], band[1], band[0], band[2]),
    dict(measured_q10_50_90=_q["P-GD-flow"], predicted_q10_50_90=[float(v) for v in band]),
    "Appendix F text (2.98 / 4.66 / 7.43)", "Results (e): part 1 holds; the initialization sets the gauge",
    runkey(("P-GD-flow",), "kappa_q10_50_90") + "; band: numbers.json panel_e_band (from "
    f"{CRIT_DIR}/P-GD-flow_Ninf_runs.csv :: eig_fC0_0..3)")
res("cond_step005", "step 0.05 (outside the theorem, C drifts): median %.2f (10-90%%: %.2f-%.2f), in the lower part "
    "of the range predicted for gradient flow from the same initializations" % (_q["P-GD"][1], _q["P-GD"][0], _q["P-GD"][2]), _q["P-GD"], "not in v4 (visible in e)",
    "Results (e), optional", runkey(("P-GD",), "kappa_q10_50_90"))
res("cond_weight_decay",
    "predicted (part 3) %.4f, measured median %.4f (10-90%%: %.4f-%.4f); spectra agree to a median relative error of "
    "%.1e" % (float(np.median(wd_pred)), _q["WD"][1], _q["WD"][0], _q["WD"][2], float(np.median(wd_relerr))),
    dict(predicted_median=float(np.median(wd_pred)), measured_q10_50_90=_q["WD"], spec_relerr_median=float(
        np.median(wd_relerr)), status="computed for the figure, not preregistered"),
    "caption and Appendix F text (1.033)", "Results (e): part 3 holds",
    "numbers.json panel_e_weight_decay_check (from %s/linae/runs/WD/Ninf/chunk*.npz :: S_eigs, spec_EEt, diverged; "
    "lambda_wd = meta['wd_lambda']); measured: %s" % (S1, runkey(("WD",), "kappa_q10_50_90")))
_rs = cells["small-init|inf"]["runs"]
res("small_init_C_near_zero",
    "small init.: |C| median %.4f at the start and %.4f at the end (relative drift is large only because C0 is tiny); "
    "learned spectrum = f(spec C0) to a median relative error of %.3f%%"
    % (_rs["C0_norm_median"], _rs["Cend_norm_median"], 100 * _rs["flow_pred_relerr_median"]),
    dict(C0_norm_median=_rs["C0_norm_median"], Cend_norm_median=_rs["Cend_norm_median"],
         C_drift_rel_median=_rs["C_drift_rel_median"], flow_pred_relerr_median=_rs["flow_pred_relerr_median"]),
    "not in v4", "Results (e) or the caption clause 'from small initialization C stays near zero'",
    runkey(("small-init",), "C0_norm_median, Cend_norm_median, C_drift_rel_median, flow_pred_relerr_median"))
_steps = {ak: dict(opt=meta["arms"][ak]["opt"], lr=meta["arms"][ak]["lr"]) for ak in AK}
assert all(_steps[ak] == dict(opt="gd", lr=0.05) for ak in ("P-GD", "small-init", "WD", "TIED"))
res("covered_rules_same_step",
    "weight decay, tied weights and small init. train with full-batch GD at step %g, the preregistered GD step "
    "(linae ANALYSIS_PLAN deviation 2)" % _steps["WD"]["lr"], _steps, "not in v4",
    "caption clause and Appendix F setup: why GD at step 0.05 is outside part 1 while these rules are covered",
    "meta['arms'][<rule>][opt, lr]")
res("pred_ranges_e",
    "predicted 10-90%% over runs: small init. %.5f-%.5f, weight decay %.5f-%.5f, tied weights 1"
    % (pred_q["small-init"][0], pred_q["small-init"][2], pred_q["WD"][0], pred_q["WD"][2]),
    {k: pred_q[k] for k in pred_q}, "not in v4 (drawn in e)", "Results (e), optional",
    "numbers.json panel_e['<rule>']['predicted_q10_50_90'] (small init.: %s :: eig_fC0_0..3; weight decay: "
    "%s/linae/runs/WD/Ninf/chunk*.npz :: S_eigs)" % (f"{CRIT_DIR}/small-init_Ninf_runs.csv", S1))
res("cond_tied_and_small_init",
    "tied weights: predicted 1, measured median 1 + %.0e; small init.: predicted median %.4f, measured median %.4f "
    "(10-90%%: %.4f-%.4f)" % (_q["TIED"][1] - 1, pred_marker["small-init"], _q["small-init"][1], _q["small-init"][0],
                              _q["small-init"][2]),
    dict(TIED=dict(predicted=1.0, measured_q10_50_90=_q["TIED"]),
         small_init=dict(predicted_median=pred_marker["small-init"], measured_q10_50_90=_q["small-init"])),
    "caption (tied weights reach 1); small init. not in v4", "Results (e)",
    runkey(("TIED", "small-init"), "kappa_q10_50_90") + "; small-init prediction: numbers.json panel_e['small-init']"
    "['predicted_marker'] (median cond f(spec C0) over runs)")
res("cond_sgd_adam", "SGD median %.2f (10-90%%: %.2f-%.2f), Adam %.2f (%.2f-%.2f): between the balanced value 1 and "
    "the gradient-flow prediction for step 0.002" % (_q["P-SGD"][1], _q["P-SGD"][0], _q["P-SGD"][2], _q["P-Adam"][1], _q["P-Adam"][0],
                       _q["P-Adam"][2]),
    {ak: _q[ak] for ak in ("P-SGD", "P-Adam")}, "Appendix F text (1.19, 1.54)", "Results (e): outside the theorem",
    runkey(("P-SGD", "P-Adam"), "kappa_q10_50_90"))
_ver = {}
for k in ("P1", "P1_secondary_flow", "P2", "P3", "P4"):
    v = summ["verdicts"][k]
    _ver[k] = dict(verdict=v["verdict"], failed_parts=[(d.get("part"), d.get("Q"), d.get("N")) for d in v.get(
        "detail", []) if d.get("verdict") == "FALSIFIED"])
assert _ver["P1"]["verdict"] == "FALSIFIED" and {p for p, _, _ in _ver["P1"]["failed_parts"]} == {"b"}
assert _ver["P1_secondary_flow"]["verdict"] == "FALSIFIED" and \
    {p for p, _, _ in _ver["P1_secondary_flow"]["failed_parts"]} == {"b"}, _ver["P1_secondary_flow"]
assert _ver["P2"]["verdict"] == "CONFIRMED" and _ver["P3"]["verdict"] == "CONFIRMED"
res("prereg_verdicts",
    "P2 and P3 confirmed; P1 falsified on its plateau test only (part b fails on %s; part a, unreliable at every N, "
    "holds in 12 of 12 cells); the step-0.002 replicate (P1_secondary_flow) is falsified the same way, with part b "
    "failing on %s; P4 (SGD, Adam) had no prediction"
    % (" and ".join(q for _, q, _ in _ver["P1"]["failed_parts"]),
       ", ".join(q for _, q, _ in _ver["P1_secondary_flow"]["failed_parts"])),
    _ver, "caption (last sentence)", "Results or Observation box: status of the predictions",
    "verdicts['<P1|P1_secondary_flow|P2|P3|P4>'][verdict, detail]")
out["results_text_numbers"] = RES
assert json.loads(json.dumps(RES, default=float)) == v6num["results_text_numbers"], "results_text_numbers differ from v6"

out["on_panel_numbers"] = {
    "decision_levels_a_to_d": dict(values=[REL_LO, UNREL_HI], status="preregistered design constants, drawn as grey "
                                   "lines without a label (v8; v6 named them in (d))",
                                   source="PREREG S1.3a / linae ANALYSIS_PLAN (decision rule)"),
    "note": "v8: no number is printed on the panels besides tick labels (v6 printed the tolerances under the heads and "
            "the pair counts in the a-d x label; both now live in Appendix app:exp-s13a, see MOVED.md)"}
out["on_panel_numbers_v6"] = v6num["on_panel_numbers"]

# history: the v6 caption, its digit check and the v6 moved-text notes, kept verbatim
for k in ("caption_v6", "caption_lines_at_print", "caption_v6_moved_non_result_text", "caption_v6_numbers_check",
          "setup_paragraph_sentence_needed", "caption_v6_variant_rowv3_note"):
    out[k] = v6num[k]

# v8 caption: bold claim (no numbers), then only what decodes marks, lines and colours, one pointer
CAPTION_V8 = (
    r"\caption{\textbf{Gradient descent without a regularizer stays unreliable at every sample size on the criteria "
    r"that see the gauge, whereas weight decay, tied weights, and small initialization become reliable "
    r"(Theorem~\ref{thm:gauge}).} (a--d)~Fraction of disjoint pairs of linear-autoencoder runs within tolerance "
    r"against $N$, with one-sided $95\%$ Clopper--Pearson bounds and grey decision levels. Dashed "
    r"lines mark rules outside the theorem, and the projector criterion~(d) is a control that ignores the gauge. "
    r"(e)~Anisotropy of the learned gauge at $N=\infty$ as the median and the 10th to 90th percentile of runs, with "
    r"the predictions for the covered rules in black. Table~\ref{tab:app-s13a} gives every decision.}")
CAPTION_V8_R1 = (
    r"\caption{\textbf{Gradient descent without a regularizer stays unreliable at every sample size on the criteria "
    r"that see the gauge, whereas weight decay, tied weights, and small initialization become reliable "
    r"(Theorem~\ref{thm:gauge}).} (a--d)~Fraction of disjoint pairs of linear-autoencoder runs within tolerance "
    r"against the sample size $N$, with one-sided $95\%$ Clopper--Pearson bounds and grey decision levels. Dashed "
    r"lines mark rules outside the theorem, and the projector criterion~(d) is a control that ignores the gauge. "
    r"(e)~Anisotropy of the learned gauge at $N=\infty$ as the median and the 10th to 90th percentile of runs, with "
    r"the predictions in black. Table~\ref{tab:app-s13a} gives every decision.}")
_stripped = re.sub(r"\\(ref|label)\{[^}]*\}", "", CAPTION_V8)
_digits = sorted(set(re.findall(r"\d+(?:\.\d+)?", _stripped)), key=float)
assert _digits == ["10", "90", "95"], _digits      # method constants only: percentiles and the CP level
def caption_words(cap):
    """Words of a caption: refs count as one word, panel tags such as (a--d) do not."""
    t = re.sub(r"\\caption\{|\\textbf\{", "", cap).replace("~", " ")
    t = re.sub(r"\\ref\{[^}]*\}", "REF", t)
    return [x for x in t.split() if re.search(r"[A-Za-z0-9]", x) and not re.fullmatch(r"\([a-e](--[a-e])?\)", x)]


_bold = re.search(r"\\textbf\{(.*?)\(Theorem", CAPTION_V8).group(1)
N_WORDS = len(caption_words(CAPTION_V8))
N_WORDS_BOLD = len(_bold.split())
assert N_WORDS <= 100 and N_WORDS_BOLD <= 28, (N_WORDS, N_WORDS_BOLD)
assert "—" not in CAPTION_V8 and "---" not in CAPTION_V8, "em-dash"
with open(os.path.join(HERE, "caption_v8.tex"), "w") as f:
    f.write(CAPTION_V8 + "\n")
out["caption_v8"] = CAPTION_V8
out["caption_v8_check"] = {"digits_outside_refs": _digits, "meaning": "10, 90: percentiles of the bars of e; 95: "
                           "one-sided Clopper-Pearson level (method constants); no result number",
                           "words_total": N_WORDS, "words_bold_sentence_before_theorem_ref": N_WORDS_BOLD}
out["v8"] = {
    "revision": "v8 (2026-10-10): v8 style (Tol palette, bold tick labels and axis titles with bold math); the "
                "author's rules of 2026-10-10 applied (no setting text on the figure, little text, colour); key in "
                "two rows without group heads; panel heads = criterion symbols on the letters' baseline; (e) one row "
                "per rule with the rule's name only, no title, no key; caption_v8 (number-free)",
    "removed_elements": {
        "key_group_heads": ["covered by the theorem:", "outside the theorem:"],
        "a_to_d_tolerance_lines": [c[3] for c in CRITS],
        "a_to_d_heads_v6": [c[2] for c in CRITS],
        "a_to_d_xlabel_v6": XLAB_TEXT_V6,
        "d_annotation": "reliable if / lower bound > 0.9 / unreliable if / upper bound < 0.8",
        "direct_labels_a_c": ["SGD (a)", "SGD (c)", "Adam (c)", "GD (c)"],
        "e_title": "anisotropy of the learned gauge",
        "e_group_heads": ["covered by the theorem", "outside the theorem"],
        "e_key": ["measured", "predicted"],
        "e_xlabel_v6": "condition number $\\mathrm{cond}(EE^{\\!\\top}\\!)$ at $N=\\infty$",
        "row_labels_v6": ROW_LABELS_V6},
    "kept_elements": "every data series of v6 (7 rules x 4 criteria x 4 sizes, CP bounds, decision levels, columns, "
                     "axis break; e: 7 measured rows, 4 predictions with their 10-90% ranges; v8 c: no cond = 1 line, "
                     "as Figure 5 c)",
    "prediction_part_per_row": PRED_PART,
    "changed_fields": ["panel_e.<rule>.y_row (rows of e without the two group-head rows)",
                       "panel_e.<rule>.row_label", "encoding.e", "encoding.colours (GD #000000 -> #222255)",
                       "encoding.text_tones", "figure.*", "style.*"],
}
# v8 r2 (critic): the record of what changed against the maker's r1 render, with its asserts
_drawn_txt = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
assert not [s for s in _drawn_txt if "\\top" in s], "a thin \\top glyph is drawn again"
assert axe.xaxis.label.get_text() == E_XLABEL and str(axe.xaxis.label.get_fontweight()) in ("bold", "700")
assert "the predictions for the covered rules in black" in CAPTION_V8 and "against $N$" in CAPTION_V8
assert len(caption_words(CAPTION_V8_R1)) == 99
out["v8"]["critic_r2"] = {
    "date": "2026-10-10",
    "changes": ["e x title: '$EE^{\\top}$' -> '$EE^{\\mathrm{T}}$' (\\top has no Liberation Serif glyph and fell back "
                "to a thin Computer Modern glyph in the bold title)",
                "caption: (e) 'with the predictions in black' -> 'with the predictions for the covered rules in "
                "black'; (a--d) 'against the sample size $N$' -> 'against $N$'"],
    "e_xlabel_r1": E_XLABEL_R1, "caption_r1": CAPTION_V8_R1, "caption_r1_words": len(caption_words(CAPTION_V8_R1)),
    "data_unchanged": "every data block asserted equal to v6 / v5.1 / v4 / v3 as in r1",
    "r1_files": "./_prev_r1_maker/ (maker, outputs, MOVED.md of the maker's r1)"}
out["v8"]["changed_fields"].append("figure.* and output.* (r2 render)")
# v8 c (consistency checker): (e) draws Figure 5 (c)'s marks the same way; asserted against the Figure 5 v8 maker,
# read only (its x title text and its decision to drop the cond = 1 line)
FIG5_MAKER = os.path.expanduser("~/hrh_v2_figs/v8/F_gauge/make_F_gauge.py")
_f5 = open(FIG5_MAKER).read()
_vlines_at_1 = [l_ for l_ in axe.get_lines() if len(l_.get_xdata()) == 2 and np.allclose(l_.get_xdata(), 1.0)
                and l_.get_transform() is not axe.transData]
CONS_C = {
    "(e) no line at cond = 1 (Figure 5 c dropped it; its maker docstring says so)": (not SHOW_COND1_LINE_E)
    and not _vlines_at_1 and "loses the grey reference line at cond = 1" in _f5,
    "(e) x title equals the Figure 5 (c) x title": E_XLABEL == "$\\mathrm{cond}(EE^{\\mathrm{T}})$"
    and '"$\\\\mathrm{cond}(EE^{\\\\mathrm{T}})$"' in _f5,
    "(e) x title bold": str(axe.xaxis.label.get_fontweight()) in ("bold", "700"),
    "S.check passed (bold ticks and titles, Type 42, >= 7 pt, chromatic data)": CHK["min_glyph_pt"] >= 7.0
    and CHK["n_type3"] == 0 and not CHK["not_bold_tick_or_axis_labels"],
}
assert all(CONS_C.values()), CONS_C
out["v8"]["consistency_c"] = {
    "date": "2026-10-10", "role": "consistency checker across the v8 figures",
    "changes": ["(e) the cond = 1 line is not drawn (Figure 5 c)", "(e) x title '%s' -> '%s' (Figure 5 c)"
                % (E_XLABEL_R2, E_XLABEL)],
    "figure5_maker": {"path": FIG5_MAKER.replace(os.path.expanduser("~"), "~"), "sha256": sha(FIG5_MAKER)},
    "asserts": CONS_C, "r2_files": "./_prev_consistency/ (maker, outputs, MOVED.md of r2)"}
out["v8"]["changed_fields"].append("figure.* and output.* (v8 c render)")
out["output"] = {"pdf": PDF, "pdf_sha256": sha(PDF), "png": PNG, "png_dpi": 200, "png_sha256": sha(PNG),
                 "png_renderer": "pdftoppm (S.print_view)", "print_view_100dpi": PNG100, "views_cvd_grey": VIEWS}
json.dump(out, open(os.path.join(HERE, "numbers.json"), "w"), indent=1, default=float)
print(json.dumps({"check": {k: CHK[k] for k in ("size_in", "min_text_pt", "min_glyph_pt", "n_type3", "fonts",
                                                "not_bold_tick_or_axis_labels", "bold_text_with_regular_math",
                                                "chromatic_data_colours", "off_palette_colours")},
                  "clash": clash, "hit": HIT, "on_fill": ON_FILL, "pred_gap_pt": PRED_GAP, "xlabel_gap_pt": XLAB_GAP,
                  "head_gap_pt": HEAD_GAP_PT, "key_head_gap_pt": KEY_HEAD_GAP_PT, "E_L": E_L, "H": H,
                  "key_total_in": KEY_TOTAL, "caption_words": [N_WORDS, N_WORDS_BOLD]}, indent=1))
print("closest text-to-data clearances (in):", sorted(CLEAR, key=lambda c: c[1])[:12])
print("OK")
