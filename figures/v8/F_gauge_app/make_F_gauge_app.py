"""F_gauge_app, v8 render (arxiv_v4 appendix figure \\label{fig:exp-gauge}; find it by the label): without weight decay,
the gauge of a linear head lowers agreement on scale-normalized geometry and leaves neighbor sets unchanged, and weight
decay restores agreement (Theorem thm:gauge). S2.2 gauge network on CIFAR-5m.

v8 (2026-10-10, role maker for the appendix gauge figures): copy of ~/hrh_v2_figs/v6/F_gauge_app/make_F_gauge_app.py
(sha256 7dd257b3...3219, untouched; its outputs stay in ~/hrh_v2_figs/v6/F_gauge_app/), switched to the frozen v8 style
~/hrh_v2_figs/v8/_style/hrh_style_v8.py (sha256 asserted). The data loader and every statistic are the v6 code, kept
verbatim (and so v3 make_F4.py); every number is asserted equal to the v3, v4 and v6 numbers.json. The drawing is new.
Author feedback of 2026-10-10 (binding for every experiment figure) and what it changed here:
  * 'never compress into count tables' and 'little text on the figure': panel (a) was a table of 54 printed counts
    (pairs within eps of 96, for y and u in three arms, nine criteria) with decision fills, preregistration frames,
    failure crosses and a two-row key. v8 draws the same decisions as a dot plot: one small multiple per arm, the nine
    criteria as rows in lattice order (bracketed 'geometry' and 'order'), one mark per criterion at the fraction of
    the 96 pairs of u within eps, and the two decision boundaries (69.5 and 91.5 of 96, the boundaries v6 drew in (b))
    as grey lines, so a mark right of the right line is reliable and one left of the left line unreliable. The outputs
    y (96 of 96 in every cell), the preregistration frames and the two failure crosses leave the figure: Table
    tab:app-s22-full gives every count and decision, Table tab:exp-verdicts and the Results text the failed predictions.
  * no setting text: removed 'N = 250k' and 'pairs within eps (of 96)' in (a), 'G_s at N = 250k' in (b), '(order)' and
    '(geometry)' in the (c) heads, the '(pilot)' tick and the arrows 'u pairs disagree more / agree more' in (c), the
    per-arm relative imbalance '(0.31)' etc. in the (c) key, the band names 'reliable / inconclusive / unreliable', the
    words 'secondary' and 'frozen eps', and the four direct curve labels of (b).
  * secondary comparisons not drawn: the secondary tolerance line of (b) (its counts are in Table tab:app-s22-eps2) and
    the pilot pairs at N = 100,000 in (c) (their median excess is in the Paired Excess paragraph). Both are still computed
    and asserted (numbers.json panel_b.curves[*].x_at_eps_q90, panel_c.cells[*|100000]).
  * colour: v8 palette (Tol): no weight decay #222255 (v6 black), weight decay #44AA99, small init. #AA4499; the arm
    marker shapes (circle, square, triangle) tie (a) to the dots at eps in (b); bold tick labels and axis titles with bold
    math (S.finalize; S.check asserts it).
  * the key: the arm names head the three columns of (a) in the arm text tones; (b) and (c) use the same colours, in
    the same left-to-right order in (c).
Layout 5.5 x H in (v6 3.75): (a) and (b) side by side, (c) the four criteria T_N,10, T_E, G_s, 1 - CKA at the two
training sizes.
v8 r2 (2026-10-10, role critic; the maker's r1 files are in ./_prev_r1_maker/): r1 dropped the preregistration frames
and the two failure crosses with the table, so (a) showed outcomes without the theory they test. Author rule (A) asks
for theory against data, so r2 draws each preregistered u cell's predicted decision as an open INK box over the zone of
that decision (reliable: right of the right grey line; unreliable: left of the left one). A mark outside its box is a
failed prediction: T_N,10 without weight decay (predicted unreliable, 96 of 96) and G_s under small init. (predicted
reliable, 82 of 96). No text is added; the caption decodes the boxes in seven words.
Family-names pass (2026-10-10; the files before this pass are in ./_prev_family_names/). The author restored the family
names of Figure 4: names that tell the reader what a column, row, colour or series IS are identifiers, not setting
headers. In (b) the solid and the dashed curve of each arm are two different series, the head coordinates u and the
outputs y. The colour (the arm) is decoded by the arm heads of (a); the line style had no decoder on the figure since
r1 dropped the four direct curve labels. (b) gets a two-entry line-style key in its empty lower right, "$u$" (solid)
and "$y$" (dashed), drawn in INK as the line-style key of Figure 3 (c). The arm half of the old direct labels stays
off (the colour carries it). Data, caption, size and every other mark unchanged; every assert unchanged; new asserts
check that the key keeps clear of the data.
Earlier history: the docstring of the v6 maker (r2, v6, v5.1, v5 r1-r3, v4) in ~/hrh_v2_figs/v6/F_gauge_app/.

Read-only inputs:
  ~/hrh_v2_stage2_20261008/results/S2_2/s22_summary.json                          preregistered decisions
  ~/hrh_v2_stage2_20261008/results/review_final/s22-recompute/recompute_summary.json  independent recompute
  ~/hrh_v2_stage2_20261008/results/review_final/s22-recompute/recompute_pairs.json    per-pair discrepancies
  ~/hrh_v2_stage2_20261008/prereg/eps_S2_2.json                                    frozen tolerances, pilot rows
  ~/hrh_v2_stage2_20261008/results/review_final/s22-recompute/eps_sensitivity.json secondary tolerance counts
  ~/hrh_v2_figs/F4/numbers.json, ~/hrh_v2_figs/v4/F_gauge_app/numbers.json, ~/hrh_v2_figs/v6/F_gauge_app/numbers.json
  ~/hrh_v2_figs/v8/_style/hrh_style_v8.py  frozen shared style (imported, never edited; sha256 asserted)
Writes only into this directory: F_gauge_app.pdf, F_gauge_app.png (200 dpi, pdftoppm), _view_print_100dpi.png,
_views_cvd_grey.png, numbers.json, caption_v8.tex.
Run: cd ~/hrh_v2_figs/v8/F_gauge_app && OMP_NUM_THREADS=1 taskset -c 0-5 nice -n 10 ~/hrh_v2_figs_venv/bin/python -B \
       make_F_gauge_app.py > run.log 2>&1
"""
import hashlib
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
OUT = HERE
KEY_MODE = "words"      # v6 switch, kept for the data code below (the v8 figure prints no key values)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

STYLE_PATH = os.path.expanduser("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_PATH, "rb").read()).hexdigest() == STYLE_SHA, "frozen v8 style module changed"
sys.path.insert(0, os.path.dirname(STYLE_PATH))
import hrh_style_v8 as S  # noqa: E402
assert S.STYLE_SHA256 == STYLE_SHA
S.use()

# ======== data and statistics: the v6 code, verbatim (v6 lines 144-357) ========
HOME = os.path.expanduser("~")
S2 = f"{HOME}/hrh_v2_stage2_20261008"
SRC = {
    "summary": f"{S2}/results/S2_2/s22_summary.json",
    "recompute_summary": f"{S2}/results/review_final/s22-recompute/recompute_summary.json",
    "recompute_pairs": f"{S2}/results/review_final/s22-recompute/recompute_pairs.json",
    "eps": f"{S2}/prereg/eps_S2_2.json",
    "eps_sensitivity": f"{S2}/results/review_final/s22-recompute/eps_sensitivity.json",
}
P_OLD = f"{HOME}/hrh_v2_figs/F4/numbers.json"
P_V4 = f"{HOME}/hrh_v2_figs/v4/F_gauge_app/numbers.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p) as f:
        return json.load(f)


assert sha(STYLE_PATH) == STYLE_SHA == S.STYLE_SHA256, "frozen style module changed"
SUM = load(SRC["summary"])
REC = load(SRC["recompute_summary"])
PAIRS = load(SRC["recompute_pairs"])
EPS = load(SRC["eps"])
EPSS = load(SRC["eps_sensitivity"])
OLD = load(P_OLD)
for k, v in SRC.items():   # the v3 figure was built from the same files
    assert OLD["sources"][k]["sha256"] == sha(v), ("input changed since F4", k)

assert SUM["interim"] is False and SUM["R"] == 192
assert REC["mismatches"] == [], "independent recompute reports mismatches"

# =============================================================================================
# data and statistics: copied from ~/hrh_v2_figs/F4/make_F4.py
# =============================================================================================
ARMS = ["noWD", "WD", "smallinit"]
ARM_COL = {"noWD": S.ARM["nowd"], "WD": S.ARM["wd"], "smallinit": S.ARM["smallinit"]}
ARM_NAME = {"noWD": "no weight decay", "WD": "weight decay", "smallinit": "small init."}
NS = [25000, 250000]
N_PILOT = 100000
DEC_SHORT = {"reliable": "rel", "inconclusive": "inc", "unreliable": "unr"}

# ---- (a) decision table at 250k; rows bottom -> top in lattice order -------------------------------
ROWS = [
    # v6 r2: '(mKNN)' names the main text's measure (s_N is the mKNN score, arxiv_v4 l.372-374; lattice figure node
    # 'mKNN T_N,k'), a name like '(rankings)' and '(projector)'; '(1 - mKNN)' is 0.96 in wide and crosses the left
    # figure edge; v6 r1 and v5 '(1 - s_N)'
    ("order", "dN_k10", r"$T_{N,10}$  (mKNN)"),
    ("order", "dO", r"$T_O$  (rankings)"),
    ("order", "dMST", r"$T_{\mathrm{MST}}$  ($1-s_{\mathcal{T}}\,$)"),   # r3: the T overhang ran into ')'
    ("order", "dE", r"$T_E$  (global order)"),
    ("geom", "dSOG_t0p1", r"SOG, $\tau=0.1$"),
    ("geom", "dAff", r"$T_{G/\mathrm{Aff}}$  ($1-\mathrm{RSA}$)"),
    ("geom", "Gs", r"$G_s$"),                         # tab:app-s22-full header names
    ("geom", "dCKA", r"$1-\mathrm{CKA}$"),
    ("quot", "dPi", r"$\Delta_\Pi$  (projector)"),
]
REPS = ["y", "u"]
# preregistered predictions on 250k cells (PREREG_stage2 sec. 3.4): (arm, rep, rung) -> predicted
PRED = {
    ("noWD", "u", "dN_k10"): ("unreliable", "P1"), ("noWD", "u", "Gs"): ("unreliable", "P1"),
    ("noWD", "y", "dN_k10"): ("reliable", "P2"), ("noWD", "y", "Gs"): ("reliable", "P2"),
    ("noWD", "y", "dPi"): ("reliable", "P2"),
    ("WD", "u", "dN_k10"): ("reliable", "P3"), ("WD", "u", "Gs"): ("reliable", "P3"),
    ("smallinit", "u", "dN_k10"): ("reliable", "P3"), ("smallinit", "u", "Gs"): ("reliable", "P3"),
}

cells_a = []
for arm in ARMS:
    for rep in REPS:
        key = f"{arm}_250000|{rep}"
        for grp, rung, lab in ROWS:
            t = SUM["tables"][key][rung]
            r = REC["tables"][key][rung]
            assert t["n"] == 96 and r["x"] == t["x"] and DEC_SHORT[t["decision"]] == r["dec"], (key, rung)
            pred = PRED.get((arm, rep, rung))
            cells_a.append({"arm": arm, "rep": rep, "rung": rung, "group": grp, "x": t["x"], "n": t["n"],
                            "decision": t["decision"], "cp_lo": t["cp_lo"], "cp_hi": t["cp_hi"],
                            "eps": t["eps"], "mean_delta": t["mean"],
                            "source_key": f"summary:tables[{key}][{rung}].{{x,n,decision}}",
                            "prereg": None if pred is None else {"prediction": pred[1], "predicted": pred[0],
                                                                 "held": pred[0] == t["decision"]}})
cells_25k = {}
for arm in ARMS:
    for rep in REPS:
        key = f"{arm}_25000|{rep}"
        cells_25k[key] = {rung: SUM["tables"][key][rung]["x"] for _, rung, _ in ROWS}
RULE = REC["cp_thresholds_n96"]
assert RULE == {"reliable_min_x": 92, "unreliable_max_x": 69}, RULE

# ---- per-pair values (independent recompute from saved features) ----------------------------------
PP = PAIRS["pairs"]


def pair_vals(cell, rep, rung):
    return np.array([PP[f"{a}__{b}"]["indep"][rep][rung] for a, b in PAIRS["sets"][cell]], float)


def pilot_vals(arm, rep, rung):
    return np.array([PP[f"{a}__{b}"]["indep"][rep][rung] for (ar, a, b) in PAIRS["pilot"]["pairs"] if ar == arm],
                    float)


# ---- (b) fraction of pairs below threshold, G_s, 250k ---------------------------------------------
eps_gs = EPS["eps"]["u"]["Gs"]
assert eps_gs == EPS["eps"]["y"]["Gs"] == SUM["tables"]["noWD_250000|u"]["Gs"]["eps"]
eps_gs_q90 = EPS["eps_q90"]["u"]["Gs"]
panel_b = {"source": "recompute_pairs:pairs[A__B].indep[rep].Gs over sets[arm_250000]; eps = eps:eps.u.Gs; "
                     "secondary = eps:eps_q90.u.Gs (pooled pilot 0.9-quantile); counts at the secondary tolerance "
                     "from eps_sensitivity:q90[arm_250000|u|Gs].x",
           "eps": eps_gs, "eps_secondary": eps_gs_q90, "eps_secondary_ratio": eps_gs_q90 / eps_gs, "curves": {}}
RATIOS = {}
for arm in ARMS:
    for rep in REPS:
        v = pair_vals(f"{arm}_250000", rep, "Gs")
        x_at_eps = int((v <= eps_gs).sum())
        x_at_q90 = int((v <= eps_gs_q90).sum())
        assert x_at_eps == SUM["tables"][f"{arm}_250000|{rep}"]["Gs"]["x"], (arm, rep)
        if rep == "u":
            assert x_at_q90 == EPSS["q90"][f"{arm}_250000|u|Gs"]["x"]
        c = REC["curves"][f"{arm}_250000|{rep}|Gs"]["p_at_mult"]
        mult = {m: float(np.mean(v <= float(m) * eps_gs)) for m in c}
        for m in c:
            assert abs(mult[m] - c[m]) < 1e-9 or abs(round(mult[m], 2) - round(c[m], 2)) < 1e-9, (arm, rep, m)
        RATIOS[(arm, rep)] = np.sort(v) / eps_gs
        panel_b["curves"][f"{arm}|{rep}"] = {
            "n": int(v.size), "x_at_eps": x_at_eps, "x_at_eps_q90": x_at_q90,
            "p_hat_at_mult": mult, "min_ratio": float(v.min() / eps_gs), "max_ratio": float(v.max() / eps_gs),
            "median_ratio": float(np.median(v) / eps_gs),
            "closest_pair_to_eps": float(np.min(np.abs(v - eps_gs))),
            "plotted_sorted_ratios": [float(z) for z in RATIOS[(arm, rep)]]}

# ---- (c) paired excess Delta(u) - Delta(y) ---------------------------------------------------------
RUNGS_C = [("dN_k10", r"$T_{N,10}$ (order)"), ("dE", r"$T_E$ (order)"),
           ("Gs", r"$G_s$ (geometry)"), ("dCKA", r"$1-\mathrm{CKA}$ (geometry)")]
INDEP_RUNGS = ("dN_k10", "Gs", "dCKA", "dPi")
PILOT_ROWS = {(r["A"], r["B"], r["rep"]): r for r in EPS["pilot_rows"]}


def pair_ladder(cell, rep, rung):
    v = np.array([PP[f"{a}__{b}"]["ladder"][rep][rung] for a, b in PAIRS["sets"][cell]], float)
    if rung in INDEP_RUNGS:
        assert np.max(np.abs(v - pair_vals(cell, rep, rung))) < 1e-10
    return v


def pilot_ladder(arm, rep, rung):
    prs = [(a, b) for (ar, a, b) in PAIRS["pilot"]["pairs"] if ar == arm]
    v = np.array([PILOT_ROWS[(a, b, rep)][rung] for a, b in prs], float)
    assert all(PILOT_ROWS[(a, b, rep)]["arm"] == arm for a, b in prs)
    if rung in INDEP_RUNGS:
        assert np.max(np.abs(v - pilot_vals(arm, rep, rung))) < 1e-10
    return v


panel_c = {"source": "main pairs: recompute_pairs:pairs[A__B].ladder[u|y][rung] over sets[arm_N] (cross-checked "
                     "against the independent feature recompute for T_N10, Gs, CKA, < 1e-10, and against "
                     "recompute_summary:exploratory medians); pilot pairs: eps:pilot_rows[A,B,rep][rung]; u and y "
                     "from the same run pair", "cells": {}}
EXC = {}
for rung, _ in RUNGS_C:
    for arm in ARMS:
        for N in [25000, N_PILOT, 250000]:
            if N == N_PILOT:
                d = pilot_ladder(arm, "u", rung) - pilot_ladder(arm, "y", rung)
            else:
                d = pair_ladder(f"{arm}_{N}", "u", rung) - pair_ladder(f"{arm}_{N}", "y", rung)
                ex = REC["exploratory"].get(f"{arm}_{N}|{rung}")
                if ex is not None:
                    assert abs(np.median(d) - ex["median_u_minus_y"]) < 1e-10, (rung, arm, N)
                    assert abs(np.mean(d > 0) - ex["frac_u_gt_y"]) < 1e-12
            EXC[(rung, arm, N)] = d
            panel_c["cells"][f"{rung}|{arm}|{N}"] = {
                "n": int(d.size), "median": float(np.median(d)), "mean": float(np.mean(d)),
                "q05": float(np.quantile(d, 0.05)), "q95": float(np.quantile(d, 0.95)),
                "min": float(d.min()), "max": float(d.max()),
                "n_u_gt_y": int((d > 0).sum()), "n_u_lt_y": int((d < 0).sum()), "pilot": N == N_PILOT,
                "plotted_values": [float(z) for z in d]}

# ---- cross-check against the v3 figure's numbers.json ----------------------------------------------
old_a = {(c["arm"], c["rep"], c["rung"]): c for c in OLD["panel_a"]["cells"]}
for c in cells_a:
    o = old_a[(c["arm"], c["rep"], c["rung"])]
    for k in ("x", "n", "decision", "cp_lo", "cp_hi", "eps", "mean_delta", "prereg"):
        assert o[k] == c[k], (c["arm"], c["rep"], c["rung"], k)
assert OLD["panel_a"]["cells_25k_x"] == cells_25k
assert OLD["panel_b"]["eps"] == eps_gs and OLD["panel_b"]["eps_q90"] == eps_gs_q90
for k, c in panel_b["curves"].items():
    o = OLD["panel_b"]["curves"][k]
    for f in o:
        assert o[f] == c[f], (k, f)
for k, c in panel_c["cells"].items():
    o = OLD["panel_c"]["cells"][k]
    for f in o:
        assert abs(o[f] - c[f]) < 1e-12 if isinstance(o[f], float) else o[f] == c[f], (k, f)
# spot values quoted by the audit
A = {(c["arm"], c["rep"], c["rung"]): c["x"] for c in cells_a}
assert A[("noWD", "u", "Gs")] == 65 and A[("noWD", "u", "dCKA")] == 73 and A[("smallinit", "u", "Gs")] == 82

# ---- v6: per-arm relative imbalance of the head at the end of training, printed in the (c) key -------------------
# relative imbalance = ||M M^T - W^T W||_F / (||M M^T||_F + ||W^T W||_F) at the end of training, median over the 192
# runs of a cell (v3: rho_end; arxiv_v4.tex Appendix F 'Exploratory Readings' and the S2.2 v1.1 'Checks and Gates')
RHO = {f"{arm}_{N}": SUM["covariates"][f"{arm}_{N}"]["imb_rel_median"] for arm in ARMS for N in NS}
assert OLD["panel_c"]["rho_end_median_imb_rel"]["source"] == "summary.covariates[cell].imb_rel_median"
assert OLD["panel_c"]["rho_end_median_imb_rel"]["values"] == RHO, "relative imbalance differs from v3 F4"
RHO_TXT = {arm: f"{RHO[f'{arm}_250000']:.2f}" for arm in ARMS}
assert RHO_TXT == {"noWD": "0.31", "WD": "0.04", "smallinit": "0.28"}, RHO_TXT       # the spec's printed values

# =============================================================================================
# v8 figure: 5.5 x H in, every position in figure inches
# =============================================================================================
P_V6 = f"{HOME}/hrh_v2_figs/v6/F_gauge_app/numbers.json"
P_V6_SHA = "90452288e26bfc0757bf8686ce23e848230967f0ba336ff2405eb86fcb6c38aa"
P_V6_MAKER = f"{HOME}/hrh_v2_figs/v6/F_gauge_app/make_F_gauge_app.py"
P_V6_MAKER_SHA = "7dd257b385c890c644eb939ef3a8a6748e2f85db05343e8bf4fc5cc327553219"
assert sha(P_V6) == P_V6_SHA and sha(P_V6_MAKER) == P_V6_MAKER_SHA, "v6 files changed"
V6 = load(P_V6)
INK, MUTED = S.INK, S.MUTED
LW, MS, Z = S.LW, S.MS, S.Z
ARM_TXT = {"noWD": S.ARM_TEXT["nowd"], "WD": S.ARM_TEXT["wd"], "smallinit": S.ARM_TEXT["smallinit"]}
ARM_MK = {"noWD": "o", "WD": "s", "smallinit": "^"}            # as Figure F.4 / F_linae_app (GD, weight decay, small init.)
MK_SCALE = {"o": 1.0, "s": 1.0, "^": 1.15}                      # a triangle fills half its box (F_linae_app SHAPE)
assert ARM_COL == {"noWD": "#222255", "WD": "#44AA99", "smallinit": "#AA4499"}, ARM_COL
W = S.TEXTWIDTH_IN
fig = plt.figure(figsize=(W, 4.0))
fig.set_layout_engine("none")
renderer = fig.canvas.get_renderer()


def ext(t):
    return t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())


def text_w(s, role):
    t = fig.text(0, 0, s, **S.text_kw(role, s))
    w = ext(t).width
    t.remove()
    return w


# ---- vertical stack ---------------------------------------------------------------------------------------------
# (a) rows: Delta_Pi, then the geometry criteria, then the order criteria (top to bottom, lattice order), groups apart
ROWS_A = list(reversed(ROWS))                       # ROWS runs bottom -> top (v6 table); (a) draws top -> bottom
ROW_SYM = {"dPi": r"$\Delta_\Pi$", "dCKA": r"$1-\mathrm{CKA}$", "Gs": r"$G_s$", "dAff": r"$T_{G/\mathrm{Aff}}$",
           "dSOG_t0p1": r"$\mathrm{SOG}_{0.1}$", "dE": r"$T_E$", "dMST": r"$T_{\mathrm{MST}}$", "dO": r"$T_O$",
           "dN_k10": r"$T_{N,10}$"}                 # the symbols of Figure 6 (d)
GROUP_GAP_A = 0.55
ROW_Y, yy, prev = {}, 0.0, None
for grp, rung, _ in ROWS_A:
    if prev is not None and grp != prev:
        yy += GROUP_GAP_A
    ROW_Y[rung] = yy
    yy += 1.0
    prev = grp
A_YLIM = (yy - 1.0 + 0.6, -0.6)                     # inverted: first row on top
PT_ROW_A = 9.5                                      # pt per row unit: bold 8 pt row labels with subscripts
A_H = (A_YLIM[0] - A_YLIM[1]) * PT_ROW_A / 72
C_B, C_H = 0.33, 0.78                               # (c) axes
C_T = C_B + C_H
C_HEAD = C_T + 0.075                                # baseline of the (c) heads and the letter (c)
A_B = C_HEAD + 0.13 + 0.40                          # (a, b) axes bottom: (c) heads, a gap, the (a) x label and ticks
A_T = A_B + A_H
HEAD_A = A_T + 0.075                                # baseline of the (a) arm heads and the letters (a), (b)
H = HEAD_A + 0.14
fig.set_size_inches(W, H)


def axes_in(x0, y0, w, h, **kw):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


# ---- (a) one small multiple per arm --------------------------------------------------------------------------------
BRK_X = 0.03                                         # left edge of the rotated bracket labels
BRK_LAB_W = 0.115                                    # room for a rotated bold 8 pt label
ROWLAB_W = max(text_w(ROW_SYM[r], "tick") for _, r, _ in ROWS_A)
A_L = BRK_X + BRK_LAB_W + 0.11 + ROWLAB_W + 0.05     # left edge of the first small multiple
SM_W, SM_G = 0.70, 0.06
ax_a = [axes_in(A_L + i * (SM_W + SM_G), A_B, SM_W, A_H) for i in range(3)]
A_R = A_L + 3 * SM_W + 2 * SM_G
A_XLIM = (0.62, 1.03)
LO_DEC, HI_DEC = (RULE["unreliable_max_x"] + 0.5) / 96, (RULE["reliable_min_x"] - 0.5) / 96   # 69.5 / 96, 91.5 / 96
DATA = []
PANEL_A_PLOT = {}
for i, (arm, ax) in enumerate(zip(ARMS, ax_a)):
    for rung in ROW_Y:
        ax.axhline(ROW_Y[rung], color=S.GUIDE, lw=LW["guide"], zorder=Z["guide"])
    for xv in (LO_DEC, HI_DEC):
        ax.axvline(xv, color=S.NULL_LINE, lw=LW["level"], zorder=Z["ref"])
    mk = ARM_MK[arm]
    for c in cells_a:
        if c["arm"] != arm or c["rep"] != "u":
            continue
        p = c["x"] / c["n"]
        DATA.extend(ax.plot([p], [ROW_Y[c["rung"]]], ls="none", marker=mk, ms=MS["dot"] * MK_SCALE[mk],
                            mfc=ARM_COL[arm], mec="white", mew=MS["dot_edge"], zorder=Z["marker"], clip_on=False))
        PANEL_A_PLOT[f"{arm}|u|{c['rung']}"] = {"x_plotted": p, "y_row": ROW_Y[c["rung"]], "x": c["x"], "n": c["n"],
                                               "decision": c["decision"]}
        # the plotted position decides as the preregistered rule does (>= 92 reliable, <= 69 unreliable)
        assert (p > HI_DEC) == (c["decision"] == "reliable") and (p < LO_DEC) == (c["decision"] == "unreliable"), c
    ax.set_xlim(*A_XLIM)
    ax.set_ylim(*A_YLIM)
    ax.set_xticks([0.7, 1.0])
    ax.set_xticklabels(["0.7", "1"])
    ax.xaxis.set_minor_locator(mticker.FixedLocator([0.8, 0.9]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if i == 0:
        ax.set_yticks([ROW_Y[r] for _, r, _ in ROWS_A])
        ax.set_yticklabels([ROW_SYM[r] for _, r, _ in ROWS_A])
        ax.tick_params(axis="y", pad=3)
    else:
        ax.set_yticks([])
assert all(c["x"] == 96 for c in cells_a if c["rep"] == "y"), "outputs y: 96 of 96 in every cell (stated in the text)"
# ---- v8 r2 (critic): the preregistered predictions on u, drawn as INK boxes over the predicted decision zone ---------
# Theory against data (author rule A): each preregistered u cell at 250,000 (PREREG_stage2 sec. 3.4, the PRED map of
# make_F4.py) gets an open INK box spanning the zone of its predicted decision (reliable: right of the right decision
# line; unreliable: left of the left one), so a mark outside its box is a failed prediction. This restores, without
# text, the v6 frames ('preregistered') and crosses ('prediction failed').
from matplotlib.patches import Rectangle  # noqa: E402
BOX_HALF_H = 0.40                                   # row units (rows 1.0 apart, 9.5 pt per unit)
BOX_X_IN = {"reliable": (HI_DEC, A_XLIM[1] - 0.003), "unreliable": (A_XLIM[0] + 0.005, LO_DEC)}
PRED_BOX = {}
for c in cells_a:
    if c["rep"] != "u" or c["prereg"] is None:
        continue
    ax = ax_a[ARMS.index(c["arm"])]
    x0, x1 = BOX_X_IN[c["prereg"]["predicted"]]
    yy0 = ROW_Y[c["rung"]] - BOX_HALF_H
    ax.add_patch(Rectangle((x0, yy0), x1 - x0, 2 * BOX_HALF_H, fill=False, edgecolor=INK, lw=LW["frame"],
                           joinstyle="miter", zorder=Z["marker"] - 0.5, clip_on=False))
    p = c["x"] / c["n"]
    inside = x0 < p < x1
    assert inside == c["prereg"]["held"], ("a mark lies outside its box iff its prediction failed", c)
    PRED_BOX[f"{c['arm']}|u|{c['rung']}"] = {"prediction": c["prereg"]["prediction"],
                                            "predicted": c["prereg"]["predicted"], "decision": c["decision"],
                                            "held": c["prereg"]["held"], "x_plotted": p, "box_x": [x0, x1],
                                            "box_y": [yy0, yy0 + 2 * BOX_HALF_H], "mark_inside_box": inside}
assert sorted(PRED_BOX) == sorted(["noWD|u|dN_k10", "noWD|u|Gs", "WD|u|dN_k10", "WD|u|Gs", "smallinit|u|dN_k10",
                                   "smallinit|u|Gs"]), sorted(PRED_BOX)
assert sorted(k for k, v in PRED_BOX.items() if not v["held"]) == ["noWD|u|dN_k10", "smallinit|u|Gs"], PRED_BOX
# arm heads (the key of the figure): bold, arm text tone, centred on each small multiple
HEAD_ARM = {}
for arm, ax in zip(ARMS, ax_a):
    pos = ax.get_position()
    HEAD_ARM[arm] = S.text(fig, (pos.x0 + pos.x1) / 2, HEAD_A / H, ARM_NAME[arm], "direct", color=ARM_TXT[arm],
                           ha="center", va="baseline")
XLAB_A = S.text(fig, (A_L + A_R) / 2 / W, (A_B - 0.29) / H, r"fraction of pairs within $\varepsilon$", "label",
                ha="center", va="baseline")
# brackets and rotated group labels, as Figure 6 (d)
tr_a = blended_transform_factory(fig.transFigure, ax_a[0].transData)
BR_X = BRK_X + BRK_LAB_W + 0.03
BRACKET_TXT = {}
for grp, word in (("geom", "geometry"), ("order", "order")):
    ys = [ROW_Y[r] for g, r, _ in ROWS_A if g == grp]
    lo, hi = min(ys) - 0.32, max(ys) + 0.32
    fig.add_artist(Line2D([(BR_X + 0.035) / W, BR_X / W, BR_X / W, (BR_X + 0.035) / W], [lo, lo, hi, hi],
                          transform=tr_a, color=INK, lw=LW["bracket"], solid_joinstyle="miter",
                          solid_capstyle="butt"))
    BRACKET_TXT[grp] = S.text(fig, (BR_X - 0.025) / W, (lo + hi) / 2, word, "header", rotation=90, ha="right",
                              va="center", transform=tr_a)

# ---- (b) fraction of pairs below threshold, G_s, N = 250,000 -------------------------------------------------------
BX0 = A_R + 0.66
BX1 = 5.44
ax_b = axes_in(BX0, A_B, BX1 - BX0, A_H)
XLO, XHI = 0.835, 1.085
ax_b.axvline(1.0, color=S.NULL_LINE, lw=LW["ref"], zorder=Z["ref"])
for yb_ in (LO_DEC, HI_DEC):
    ax_b.axhline(yb_, color=S.NULL_LINE, lw=LW["level"], zorder=Z["guide"])
for arm in ARMS:
    for rep in REPS:
        v = RATIOS[(arm, rep)]
        assert XLO < v.min() and v.max() < XHI, (arm, rep, v.min(), v.max())
        n = v.size
        xs = np.concatenate([[XLO], [v[0]], v])
        ys = np.concatenate([[0.0], [0.0], np.arange(1, n + 1) / n])
        xs = np.concatenate([xs, [XHI]])
        ys = np.concatenate([ys, [1.0]])
        if rep == "u":
            DATA.extend(ax_b.step(xs, ys, where="post", color=ARM_COL[arm],
                                  lw=LW["step_emph" if arm == "noWD" else "step"],
                                  zorder=Z["data"] - (0 if arm == "noWD" else 0.1)))
        else:
            DATA.extend(ax_b.step(xs, ys, where="post", color=ARM_COL[arm], lw=LW["aux"], ls=S.LS["dashed"],
                                  zorder=Z["connector"]))
    pu = panel_b["curves"][f"{arm}|u"]["x_at_eps"] / 96
    mk = ARM_MK[arm]
    DATA.extend(ax_b.plot([1.0], [pu], ls="none", marker=mk, ms=MS["dot"] * MK_SCALE[mk], mfc=ARM_COL[arm],
                          mec="white", mew=MS["dot_edge"], zorder=Z["marker"], clip_on=False))
    assert abs(pu - PANEL_A_PLOT[f"{arm}|u|Gs"]["x_plotted"]) < 1e-12   # the dot at eps = the G_s mark of (a)
ax_b.set_xlim(XLO, XHI)
ax_b.set_ylim(-0.03, 1.03)
ax_b.set_xticks([0.9, 1.0])
ax_b.set_xticklabels(["0.9", "1"])
ax_b.xaxis.set_minor_locator(mticker.FixedLocator([0.85, 0.95, 1.05]))
ax_b.xaxis.set_minor_formatter(mticker.NullFormatter())
ax_b.set_yticks([0, 0.5, 1.0])
ax_b.set_yticklabels(["0", "0.5", "1"])
# x label as a figure text on the baseline of the (a) x label (an axis label sat 2.3 pt higher)
XLAB_B = S.text(fig, (BX0 + BX1) / 2 / W, (A_B - 0.29) / H, r"$G_s$ threshold / $\varepsilon$", "label",
                ha="center", va="baseline")
ax_b.set_ylabel("fraction of pairs\nbelow threshold", linespacing=1.1)
# family-names pass: line-style key of (b), u solid and y dashed (identifiers of the two series of each arm)
B_KEY_X0, B_KEY_LEN, B_KEY_TXT = 0.74, 0.16, 0.04      # start (axes fraction); sample length and text gap (inches)
B_KEY_Y = {"u": 0.30, "y": 0.16}                       # data y: the empty lower right of (b), right of eps
_bw = BX1 - BX0
_trb = blended_transform_factory(ax_b.transAxes, ax_b.transData)
B_KEY_LINES, B_KEY_TEXT = {}, {}
for rep_, ls_, lw_ in (("u", "-", LW["step"]), ("y", S.LS["dashed"], LW["aux"])):
    yk_ = B_KEY_Y[rep_]
    B_KEY_LINES[rep_] = Line2D([B_KEY_X0, B_KEY_X0 + B_KEY_LEN / _bw], [yk_, yk_], transform=_trb, color=INK, lw=lw_,
                               ls=ls_, solid_capstyle="butt", dash_capstyle="butt", zorder=Z["data"], clip_on=False)
    ax_b.add_line(B_KEY_LINES[rep_])
    B_KEY_TEXT[rep_] = S.text(ax_b, B_KEY_X0 + (B_KEY_LEN + B_KEY_TXT) / _bw, yk_, "$%s$" % rep_, "key", ha="left",
                              va="center", transform=_trb)

# ---- (c) paired excess Delta(u) - Delta(y), two training sizes ------------------------------------------------------
C_L, C_R, C_G = 0.50, 5.44, 0.08
CW = (C_R - C_L - 3 * C_G) / 4
ax_c = [axes_in(C_L + i * (CW + C_G), C_B, CW, C_H) for i in range(4)]
rng = np.random.default_rng(20261009)   # display jitter only (main pairs); same seed and draw order as v6
XPOS_C = {25000: 0.0, 250000: 1.0}
OFF = {"noWD": -0.27, "WD": 0.0, "smallinit": 0.27}
JIT, MED_HALF = 0.07, 0.10
C_XLIM = (-0.52, 1.52)
YL = (-0.049, 0.035)
IN_PER_X = CW / (C_XLIM[1] - C_XLIM[0])
DOT_R = np.sqrt(MS["jitter_s"]) / 2 / 72
GAPS_C = {}
HEAD_C = []
for ax, (rung, title_v6) in zip(ax_c, RUNGS_C):
    ax.axhline(0.0, color=S.NULL_LINE, lw=LW["zero"], ls=S.LS["dashed"], zorder=Z["ref"])
    clouds = {}
    for arm in ARMS:
        for N in [25000, N_PILOT, 250000]:
            if N == N_PILOT:
                continue                               # v8: the pilot pairs are not drawn (Results text, MOVED.md)
            d = EXC[(rung, arm, N)]
            assert YL[0] < d.min() and d.max() < YL[1], (rung, arm, N)
            med = float(np.median(d))
            lx = XPOS_C[N] + OFF[arm]
            u = rng.uniform(-1.0, 1.0, d.size)        # same draws as v6 (uniform(-JIT, JIT) = JIT * uniform(-1, 1))
            xl = lx + JIT * u
            DATA.append(ax.scatter(xl, d, s=MS["jitter_s"], color=ARM_COL[arm], alpha=MS["jitter_alpha"],
                                   linewidths=0, zorder=Z["thin"]))
            bx = [lx - MED_HALF, lx + MED_HALF]
            S.underlay(ax, bx, [med, med], LW["median_bar"], Z["data"], extra=LW["median_halo"], solid_capstyle="butt")
            DATA.extend(ax.plot(bx, [med, med], color=ARM_COL[arm], lw=LW["median_bar"], zorder=Z["data"],
                                solid_capstyle="butt"))
            clouds[(arm, N)] = ((xl.min() * IN_PER_X) - DOT_R, (xl.max() * IN_PER_X) + DOT_R,
                                (lx - MED_HALF) * IN_PER_X, (lx + MED_HALF) * IN_PER_X)
    for N in NS:
        for a1, a2 in (("noWD", "WD"), ("WD", "smallinit")):
            g = min(clouds[(a2, N)][0] - clouds[(a1, N)][1], clouds[(a2, N)][2] - clouds[(a1, N)][3])
            GAPS_C[f"{rung}|{N}|{a1}-{a2}"] = round(g, 4)
            assert g >= 0.02, (rung, N, a1, a2, g)
    ax.set_xlim(*C_XLIM)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([S.fmt_int(25000), S.fmt_int(250000)])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(*YL)
    ax.set_yticks([-0.04, -0.02, 0, 0.02])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:g}".replace("-", S.MINUS)))
    pos = ax.get_position()
    HEAD_C.append(S.text(fig, (pos.x0 + pos.x1) / 2, C_HEAD / H, ROW_SYM[rung], "header", ha="center",
                         va="baseline"))
for ax in ax_c[1:]:
    ax.tick_params(axis="y", labelleft=False)
ax_c[0].set_ylabel(r"$\Delta(u)-\Delta(y)$")
XLAB_C = S.text(fig, (C_L + C_R) / 2 / W, 0.035 / H, r"sample size $N$", "label", ha="center", va="baseline")

# ---- panel letters ---------------------------------------------------------------------------------------------
fig.canvas.draw()
LET = {}
LET["a"] = S.panel_letter(fig, "a", 0.02, HEAD_A)
LET["b"] = S.panel_letter(fig, "b", ext(ax_b.yaxis.label).x0, HEAD_A)
LET["c"] = S.panel_letter(fig, "c", 0.02, C_HEAD)
for l, yb in (("a", HEAD_A), ("b", HEAD_A), ("c", C_HEAD)):
    LET[l].set_va("baseline")
    LET[l].set_y(yb / H)

# =============================================================================================
# v8: bold ticks, axis titles and math; save; checks; renders
# =============================================================================================
N_FINAL = S.finalize(fig)
fig.canvas.draw()
PDF, PNG = os.path.join(OUT, "F_gauge_app.pdf"), os.path.join(OUT, "F_gauge_app.png")
PNG100, VIEWS = os.path.join(OUT, "_view_print_100dpi.png"), os.path.join(OUT, "_views_cvd_grey.png")
fig.savefig(PDF)
CHK = S.check(fig, PDF)
assert abs(CHK["size_in"][0] - 5.5) < 1e-9 and abs(CHK["size_in"][1] - H) < 1e-9
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
font_rows = [ln.split() for ln in pf.splitlines()[2:] if ln.strip()]
assert all(row[-5] == "yes" for row in font_rows), pf
assert not any("Type 3" in ln for ln in pf.splitlines()), pf
S.print_view(PDF, PNG, PNG100)
S.views(PNG, VIEWS)
assert {c.upper() for c in ARM_COL.values()} <= set(CHK["chromatic_data_colours"]), CHK["chromatic_data_colours"]


def text_poly(t):
    """Rotated text: its rotated layout box as a polygon (inches); otherwise the axis-aligned box."""
    inv = fig.dpi_scale_trans.inverted()
    bb = t.get_window_extent(renderer).transformed(inv)
    return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]]), bb


TB = [(t.get_text(), text_poly(t)[1]) for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
clash = [(a, b) for i, (a, ba) in enumerate(TB) for (b, bb) in TB[i + 1:] if ba.overlaps(bb)
         and ba.intersection(ba, bb).width > 0.005 and ba.intersection(ba, bb).height > 0.005]
outside = [(a, [round(v, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)]) for a, bb in TB
           if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]
assert not clash, clash
assert not outside, outside


# text against data marks (layout boxes against marker boxes, scatter dots and line segments), inches
def data_shapes(arts):
    import matplotlib.collections as mcoll
    boxes, segs = [], []
    for a in arts:
        if isinstance(a, mcoll.PathCollection):
            r = np.sqrt(np.max(a.get_sizes())) / 2 / 72
            pts = a.get_offset_transform().transform(a.get_offsets()) / fig.dpi
            boxes += [(x - r, y_ - r, x + r, y_ + r) for x, y_ in pts]
            continue
        pts = a.get_transform().transform(np.column_stack(a.get_data())) / fig.dpi
        m = a.get_marker()
        if m not in (None, "None", "", " "):
            hm = (a.get_markersize() + a.get_markeredgewidth()) / 2 / 72
            boxes += [(x - hm, y_ - hm, x + hm, y_ + hm) for x, y_ in pts]
        if a.get_linestyle() not in ("None", "", " ") and a.get_linewidth() > 0:
            if a.get_drawstyle() and a.get_drawstyle().startswith("steps"):
                p = a._get_transformed_path().get_fully_transformed_path().vertices / fig.dpi
                pts = p
            segs += [(pts[i], pts[i + 1], a.get_linewidth() / 2 / 72) for i in range(len(pts) - 1)]
    return np.array(boxes), segs


BOX_ARR, SEGS = data_shapes(DATA)


def clearance(r):
    dx = np.maximum.reduce([BOX_ARR[:, 0] - r[2], r[0] - BOX_ARR[:, 2], np.zeros(len(BOX_ARR))])
    dy = np.maximum.reduce([BOX_ARR[:, 1] - r[3], r[1] - BOX_ARR[:, 3], np.zeros(len(BOX_ARR))])
    inside = (BOX_ARR[:, 0] < r[2]) & (r[0] < BOX_ARR[:, 2]) & (BOX_ARR[:, 1] < r[3]) & (r[1] < BOX_ARR[:, 3])
    gb = float(np.where(inside, -1.0, np.hypot(dx, dy)).min())
    gs = np.inf
    for p0, p1, hw in SEGS:
        if min(p0[0], p1[0]) - 0.3 > r[2] or max(p0[0], p1[0]) + 0.3 < r[0] or min(p0[1], p1[1]) - 0.3 > r[3] \
                or max(p0[1], p1[1]) + 0.3 < r[1]:
            continue
        tt = np.linspace(0, 1, 200)[:, None]
        pts = p0 + tt * (p1 - p0)
        ddx = np.maximum.reduce([r[0] - pts[:, 0], pts[:, 0] - r[2], np.zeros(len(pts))])
        ddy = np.maximum.reduce([r[1] - pts[:, 1], pts[:, 1] - r[3], np.zeros(len(pts))])
        gs = min(gs, float((np.hypot(ddx, ddy) - hw).min()))
    return min(gb, gs)


CLEAR = sorted(((a, round(clearance((bb.x0, bb.y0, bb.x1, bb.y1)), 4)) for a, bb in TB), key=lambda c: c[1])
HIT = [c for c in CLEAR if c[1] < 0]
assert not HIT, HIT
# family-names pass: the (b) key samples and texts keep at least 2 pt from every data mark and line, inside the axes
B_KEY_CLEAR = {}
for rep_ in ("u", "y"):
    _lb = B_KEY_LINES[rep_].get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    _tb = ext(B_KEY_TEXT[rep_])
    B_KEY_CLEAR[rep_] = {"line_in": round(clearance((_lb.x0, _lb.y0, _lb.x1, _lb.y1)), 4),
                         "text_in": round(clearance((_tb.x0, _tb.y0, _tb.x1, _tb.y1)), 4)}
    _ab = ax_b.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    assert _ab.x0 < _lb.x0 and _tb.x1 < _ab.x1 - 0.02 and _ab.y0 < _tb.y0 and _tb.y1 < _ab.y1, (rep_, _lb, _tb, _ab)
assert min(v for d in B_KEY_CLEAR.values() for v in d.values()) >= 2.0 / 72, B_KEY_CLEAR
# layout gaps (pt): (a) x label to the (c) heads below it and to its own tick labels; heads to the top of the axes
INV = fig.dpi_scale_trans.inverted()
_tk_a = min(ext(t).y0 for ax in ax_a for t in ax.get_xticklabels() if t.get_visible() and t.get_text())
_tk_b = min(ext(t).y0 for t in ax_b.get_xticklabels() if t.get_visible() and t.get_text())
GAPS = {"a_xlabel_to_a_ticks_pt": round((_tk_a - ext(XLAB_A).y1) * 72, 2),
        "a_xlabel_to_c_heads_pt": round((ext(XLAB_A).y0 - max(ext(t).y1 for t in HEAD_C + [LET["c"]])) * 72, 2),
        "b_xlabel_to_c_heads_pt": round((ext(XLAB_B).y0 - max(ext(t).y1 for t in HEAD_C)) * 72, 2),
        "b_xlabel_to_b_ticks_pt": round((_tk_b - ext(XLAB_B).y1) * 72, 2),
        "a_heads_to_axes_top_pt": round((min(ext(t).y0 for t in HEAD_ARM.values()) - A_T) * 72, 2),
        "c_heads_to_axes_top_pt": round((min(ext(t).y0 for t in HEAD_C) - C_T) * 72, 2),
        "a_bracket_label_to_row_labels_pt": round((min(ext(t).x0 for t in ax_a[0].get_yticklabels())
                                                   - max(ext(t).x1 for t in BRACKET_TXT.values())) * 72, 2),
        "a_bracket_ticks_to_row_labels_pt": round((min(ext(t).x0 for t in ax_a[0].get_yticklabels())
                                                   - (BR_X + 0.035)) * 72, 2),
        "a_right_to_b_ylabel_in": round(ext(ax_b.yaxis.label).x0 - A_R, 3),
        "top_headroom_pt": round((H - max(bb.y1 for _, bb in TB)) * 72, 2)}
assert GAPS["a_xlabel_to_c_heads_pt"] > GAPS["a_xlabel_to_a_ticks_pt"] >= 2.0, GAPS
assert GAPS["b_xlabel_to_c_heads_pt"] > GAPS["b_xlabel_to_b_ticks_pt"], GAPS
assert GAPS["a_heads_to_axes_top_pt"] >= 2.0 and GAPS["c_heads_to_axes_top_pt"] >= 2.0, GAPS
assert GAPS["a_bracket_label_to_row_labels_pt"] >= 3.0 and GAPS["a_right_to_b_ylabel_in"] >= 0.12, GAPS
assert GAPS["top_headroom_pt"] >= 0.5 and GAPS["a_bracket_ticks_to_row_labels_pt"] >= 2.0, GAPS
# v8 r2: the prediction boxes keep clear of the row labels (first small multiple) and stay inside their own rows
_box_l = min(ax_a[0].transData.transform((v["box_x"][0], 0.0))[0] / fig.dpi for k, v in PRED_BOX.items()
             if k.startswith("noWD|"))
GAPS["a_row_labels_to_prediction_boxes_pt"] = round((_box_l - max(ext(t).x1 for t in ax_a[0].get_yticklabels()))
                                                    * 72 - LW["frame"] / 2, 2)
assert GAPS["a_row_labels_to_prediction_boxes_pt"] >= 2.0, GAPS
assert all(abs(v["box_y"][1] - v["box_y"][0]) < 1.0 for v in PRED_BOX.values())     # one row per box, rows 1.0 apart
HEAD_GAP = {f"{l}|{r}": round(ext(HEAD_ARM[r]).x0 - ext(HEAD_ARM[l]).x1, 3) for l, r in (("noWD", "WD"), ("WD", "smallinit"))}
assert min(HEAD_GAP.values()) >= 0.06, HEAD_GAP
# no text on the figure states a setting, a count or a reading (v8 rule)
DRAWN = sorted({t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()})
BANNED = ["250k", "250{,}000$", "pilot", "secondary", "frozen", "reliable", "inconclusive", "prediction", "of 96",
          "(order)", "(geometry)", "disagree", "agree more", "(0.31)", "(0.28)", "(0.04)", "preregistered"]
assert not [s for s in DRAWN if any(b in s for b in BANNED)], [s for s in DRAWN if any(b in s for b in BANNED)]

# =============================================================================================
# numbers.json
# =============================================================================================
cap = {
    "phat_noWD_u_Gs_at_0.9_1.0_1.1_eps": [panel_b["curves"]["noWD|u"]["p_hat_at_mult"][m] for m in ("0.9", "1.0", "1.1")],
    "count_noWD_u_Gs_at_secondary": panel_b["curves"]["noWD|u"]["x_at_eps_q90"],
    "secondary_over_eps": eps_gs_q90 / eps_gs,
    "Gs_median_excess_noWD_25k_100k_250k": [panel_c["cells"][f"Gs|noWD|{N}"]["median"] for N in (25000, N_PILOT, 250000)],
    "Gs_n_u_gt_y_noWD_main": {N: panel_c["cells"][f"Gs|noWD|{N}"]["n_u_gt_y"] for N in NS},
    "Gs_median_excess_WD_250k": panel_c["cells"]["Gs|WD|250000"]["median"],
    "Gs_median_excess_smallinit_250k": panel_c["cells"]["Gs|smallinit|250000"]["median"],
    "TN10_median_range_all_cells_incl_pilot": [min(panel_c["cells"][f"dN_k10|{a}|{N}"]["median"] for a in ARMS
                                                   for N in (25000, N_PILOT, 250000)),
                                               max(panel_c["cells"][f"dN_k10|{a}|{N}"]["median"] for a in ARMS
                                                   for N in (25000, N_PILOT, 250000))],
    "TN10_median_range_main_cells": [min(panel_c["cells"][f"dN_k10|{a}|{N}"]["median"] for a in ARMS for N in NS),
                                     max(panel_c["cells"][f"dN_k10|{a}|{N}"]["median"] for a in ARMS for N in NS)],
    "TE_median_excess_250k": {a: panel_c["cells"][f"dE|{a}|250000"]["median"] for a in ARMS},
    "TE_n_u_lt_y_250k": {a: panel_c["cells"][f"dE|{a}|250000"]["n_u_lt_y"] for a in ARMS},
    "all_25k_cells_zero": all(v == 0 for c in cells_25k.values() for v in c.values()),
    "failed_predictions_250k": [{"arm": c["arm"], "rep": c["rep"], "rung": c["rung"], "x": c["x"],
                                 "predicted": c["prereg"]["predicted"], "decision": c["decision"]}
                                for c in cells_a if c["prereg"] is not None and not c["prereg"]["held"]],
}
NUM = {
    "figure": {"id": "F_gauge_app", "label": "fig:exp-gauge", "version": "v8",
               "message": "Without weight decay, the gauge of a linear head lowers agreement on scale-normalized "
                          "geometry and leaves neighbor sets unchanged, and weight decay restores agreement "
                          "(Theorem thm:gauge).",
               "size_in": CHK["size_in"], "min_text_pt": CHK["min_text_pt"], "min_glyph_pt": CHK["min_glyph_pt"],
               "n_type3": CHK["n_type3"], "style_check": CHK, "pdffonts": pf.splitlines(),
               "text_overlaps_gt_0.005in": clash, "text_vs_data_overlaps": HIT,
               "closest_text_to_data_in": CLEAR[:12], "layout_gaps": GAPS, "arm_head_gaps_in": HEAD_GAP,
               "panel_c_arm_gaps_in": GAPS_C, "drawn_texts": DRAWN, "n_texts_set_bold_by_finalize": N_FINAL},
    "sources": {k: {"path": v, "sha256": sha(v)} for k, v in SRC.items()},
    "style": {"path": STYLE_PATH, "sha256": STYLE_SHA, "version": S.STYLE_VERSION},
    "script": {"path": os.path.abspath(__file__), "sha256": sha(os.path.abspath(__file__))},
    "crosscheck_v3": {"path": P_OLD, "sha256": sha(P_OLD),
                      "result": "panel_a.cells, cells_25k_x, panel_b (eps, eps_q90, curves) and panel_c.cells "
                                "equal to F4/numbers.json (asserted; floats within 1e-12)"},
    "crosscheck_v4": None,
    "crosscheck_v6": None,
    "panel_a": {"source": "summary:tables[arm_250000|rep][rung] (decisions); recompute_summary:tables "
                          "(cross-check, 0 mismatches); predictions PREREG_stage2 sec. 3.4 as in make_F4.py",
                "N": 250000, "decision_rule": RULE, "cells": cells_a, "cells_25k_x": cells_25k},
    "panel_b": panel_b,
    "panel_c": panel_c,
    "plotted_v8": {
        "a": {"what": "fraction of the 96 disjoint pairs of u within eps at N = 250,000, per arm (small multiple) and "
                      "criterion (row, lattice order top to bottom: Delta_Pi; geometry 1-CKA, G_s, T_G/Aff, SOG_0.1; "
                      "order T_E, T_MST, T_O, T_N,10); grey lines at the decision boundaries 69.5/96 and 91.5/96 "
                      "(right of the right line = reliable, left of the left line = unreliable; asserted per mark); "
                      "y (96 of 96 everywhere) not drawn",
              "marks": PANEL_A_PLOT, "decision_lines": [LO_DEC, HI_DEC], "xlim": list(A_XLIM),
              "row_y": ROW_Y, "row_symbols": ROW_SYM},
        "b": {"what": "ECDF of G_s / eps over the 96 pairs at N = 250,000, u solid (no weight decay heavier), y "
                      "dashed, one colour per arm; marks at eps = the G_s marks of (a) (asserted); grey horizontal "
                      "lines at the decision boundaries; vertical line at eps", "xlim": [XLO, XHI]},
        "c": {"what": "paired excess Delta(u) - Delta(y) of each main pair at N = 25,000 and 250,000 (pilot pairs at "
                      "100,000 not drawn), dots jittered (seed 20261009, v6 draw order), bars = medians on a white "
                      "underlay; arms left to right as the columns of (a)",
              "arm_offsets": OFF, "jitter_half": JIT, "median_half": MED_HALF, "xlim": list(C_XLIM), "ylim": list(YL)},
        "colours": ARM_COL, "text_tones": ARM_TXT, "markers": ARM_MK},
    "panel_c_key": {"quantity": "median over the runs of a cell of ||M M^T - W^T W||_F / (||M M^T||_F + "
                                "||W^T W||_F) of the head at the end of training (relative imbalance; v3 rho_end)",
                    "printed_N": 250000, "printed_v6": RHO_TXT, "printed_v8": None,
                    "values": RHO, "source_path": SRC["summary"],
                    "source_key": "covariates[<arm>_<N>].imb_rel_median",
                    "note": "v8 prints no key value; the Results text states 0.31, 0.04 and 0.28 (MOVED.md)"},
    "family_names_pass": {"date": "2026-10-10", "previous_files": "_prev_family_names/",
                          "change": "(b) line-style key: '$u$' solid and '$y$' dashed in INK, lower right of (b); "
                                    "the two series of each arm (head coordinates u, outputs y) had no in-figure "
                                    "decoder since r1 dropped the direct curve labels; the arm is the colour",
                          "key_geometry": {"x0_axes_fraction": B_KEY_X0, "sample_in": B_KEY_LEN,
                                           "text_gap_in": B_KEY_TXT, "y_data": B_KEY_Y,
                                           "lw_pt": {"u": LW["step"], "y": LW["aux"]}},
                          "clearance_to_data_in": B_KEY_CLEAR,
                          "unchanged": "data, caption, size, every other mark, every earlier assert"},
    "results_numbers": V6["results_numbers"],
    "results_numbers_note": V6["results_numbers_note"],
    "proposed_caption_v6": V6["proposed_caption"],
}
V4 = load(P_V4)
for k in ("panel_a", "panel_b", "panel_c"):
    assert json.loads(json.dumps(NUM[k])) == V4[k], ("numbers changed against v4", k)
    assert json.loads(json.dumps(NUM[k])) == V6[k], ("numbers changed against v6", k)
assert json.loads(json.dumps(cap)) == V4["caption_numbers"], "former caption numbers changed against v4"
assert V4["sources"] == json.loads(json.dumps(NUM["sources"])) == V6["sources"], "inputs changed against v4 / v6"
assert V6["panel_c_key"]["values"] == RHO and V6["panel_c_key"]["printed"] == RHO_TXT
# the v6 results numbers are re-derived here from the same blocks (spot values the MOVED.md sentences quote)
_RN = {r["statement"]: r["value"] for r in V6["results_numbers"]}
assert _RN["(a) u without weight decay, G_s: pairs within eps at N=250k (unreliable)"] == 65
assert _RN["(a) u under small init., G_s: inconclusive, preregistered P3 (reliable) failed"] == 82
assert _RN["(b) pairs within the secondary tolerance: u without weight decay; u under small init."] == \
    [panel_b["curves"]["noWD|u"]["x_at_eps_q90"], panel_b["curves"]["smallinit|u"]["x_at_eps_q90"]] == [39, 58]
assert abs(panel_c["cells"]["Gs|noWD|100000"]["median"] - 0.0188) < 5e-5 and panel_c["cells"]["Gs|noWD|100000"]["n"] == 8
NUM["crosscheck_v4"] = {"path": P_V4, "sha256": sha(P_V4),
                        "result": "panel_a, panel_b, panel_c, the former caption numbers and sources identical to the "
                                  "v4 numbers.json (asserted, exact after JSON round trip)"}
NUM["crosscheck_v6"] = {"path": P_V6, "sha256": P_V6_SHA,
                        "result": "panel_a, panel_b, panel_c, sources and the panel_c_key values identical to the v6 "
                                  "numbers.json; results_numbers copied from v6 and spot-checked (asserted)"}

# ---- v8 caption -----------------------------------------------------------------------------------------------
CAPTION_V8 = (
    r"\caption{\textbf{Without weight decay, the gauge of a linear head lowers agreement on scale-normalized "
    r"geometry and leaves neighbor sets unchanged, and weight decay restores agreement (Theorem~\ref{thm:gauge}).} "
    r"(a)~Fraction of disjoint pairs of CIFAR-5m runs at $N=250{,}000$ whose head coordinates $u$ lie within the "
    r"frozen tolerance $\varepsilon$, with the preregistered predictions boxed in black. (b)~The same fraction on "
    r"$G_s$ against the threshold, for $u$ (solid) and the outputs $y$ (dashed). Grey lines are the decision levels. "
    r"(c)~Excess $\Delta(u)-\Delta(y)$ per pair of runs, with median bars and the arms ordered as in (a). Panels (b) "
    r"and (c) are exploratory, and Table~\ref{tab:app-s22-full} gives every decision.}")
CAPTION_V8_R1 = (    # the maker's r1 caption (no prediction boxes in (a)), kept for the record
    r"\caption{\textbf{Without weight decay, the gauge of a linear head lowers agreement on scale-normalized "
    r"geometry and leaves neighbor sets unchanged, and weight decay restores agreement (Theorem~\ref{thm:gauge}).} "
    r"(a)~Fraction of disjoint pairs of CIFAR-5m runs at $N=250{,}000$ whose head coordinates $u$ lie within the "
    r"frozen tolerance $\varepsilon$. (b)~The same fraction on $G_s$ against the threshold, for $u$ (solid) and the "
    r"outputs $y$ (dashed). Grey lines are the decision levels. (c)~Excess $\Delta(u)-\Delta(y)$ of each pair of "
    r"runs, with bars at the medians and the arms ordered as in (a). Panels (b) and (c) are exploratory, and "
    r"Table~\ref{tab:app-s22-full} gives every decision.}")


def caption_words(cap_):
    """Words of a caption: refs count as one word, panel tags such as (a--d) do not."""
    t = re.sub(r"\\caption\{|\\textbf\{", "", cap_).replace("~", " ")
    t = re.sub(r"\\ref\{[^}]*\}", "REF", t)
    return [x for x in t.split() if re.search(r"[A-Za-z0-9]", x) and not re.fullmatch(r"\([a-e](--[a-e])?\)", x)]


_bold = re.search(r"\\textbf\{(.*?)\(Theorem", CAPTION_V8).group(1)
_digits = sorted(set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\\(ref|label)\{[^}]*\}", "", CAPTION_V8))), key=float)
assert _digits == ["000", "5", "250"], _digits          # N = 250,000 and CIFAR-5m: design constants
N_WORDS, N_WORDS_BOLD = len(caption_words(CAPTION_V8)), len(_bold.split())
assert N_WORDS <= 100 and N_WORDS_BOLD <= 25, (N_WORDS, N_WORDS_BOLD)
assert "—" not in CAPTION_V8 and "---" not in CAPTION_V8
with open(os.path.join(OUT, "caption_v8.tex"), "w") as f:
    f.write(CAPTION_V8 + "\n")
NUM["caption_v8"] = CAPTION_V8
NUM["caption_v8_check"] = {"digits_outside_refs": _digits, "meaning": "250,000: the training size of (a, b); 5 in "
                           "CIFAR-5m: the data set's name; no result number", "words_total": N_WORDS,
                           "words_bold_sentence_before_theorem_ref": N_WORDS_BOLD}
NUM["v8"] = {
    "removed_elements": {
        "a_table": "54 printed counts (y and u, three arms, nine criteria), decision fills, preregistration frames, "
                   "two failure crosses, the two-row decision key, 'N = 250k', 'pairs within eps (of 96)', the y/u "
                   "column heads, bracket words 'gauge quotient', '(above T_E)', '(<= T_E)', the long row names "
                   "'(projector)', '(1 - RSA)', '(global order)', '(1 - s_T)', '(rankings)', '(mKNN)'",
        "b": ["secondary (dotted line at eps_0.9)", "frozen eps (word)", "reliable / inconclusive / unreliable",
              "u, weight decay", "u, small init.", "y of each arm", "u, no weight decay", "G_s at N = 250k"],
        "c": ["(order) and (geometry) in the heads", "100k (pilot) column and the pilot rings and bars",
              "up arrow u pairs disagree more", "down arrow u pairs agree more", "key: (0.31), (0.04), (0.28), "
              "pilot pairs, median", "(log scale) in the x label"]},
    "kept_messages": {"a": "the preregistered decision of every u cell at 250,000 (now a mark against the decision "
                           "lines): weight decay reliable everywhere; no weight decay unreliable on G_s, inconclusive "
                           "on 1 - CKA, reliable on Delta_Pi, T_G/Aff, SOG, order criteria and T_N,10; small init. "
                           "inconclusive on G_s and 1 - CKA",
                      "b": "the knife edge of G_s without weight decay (its u curve crosses the decision band at eps)",
                      "c": "the gauge raises the G_s and 1 - CKA discrepancy of u over y without weight decay and "
                           "under small init., leaves T_N,10 at zero, and lowers T_E; weight decay removes it"},
    "changed_fields": ["figure.*", "style.*", "display (replaced by plotted_v8)", "panel_c_key.printed (none)"],
}
assert len(caption_words(CAPTION_V8_R1)) == 96
NUM["plotted_v8"]["a"]["prediction_boxes"] = PRED_BOX
NUM["plotted_v8"]["a"]["prediction_boxes_encoding"] = (
    "v8 r2: open INK box (LW frame) over the zone of the preregistered decision of each preregistered u cell "
    "(PREREG_stage2 sec. 3.4: P1 no weight decay T_N,10 and G_s unreliable; P3 weight decay and small init. T_N,10 "
    "and G_s reliable); reliable zone = right of the right decision line, unreliable zone = left of the left one; a "
    "mark outside its box = failed prediction (asserted per box against summary decisions)")
NUM["v8"]["critic_r2"] = {
    "date": "2026-10-10",
    "changes": ["(a): open INK boxes over the predicted decision zone of the six preregistered u cells, so theory "
                "stands against data in the figure and the two failed predictions (T_N,10 without weight decay; G_s "
                "under small init.) show as marks outside their boxes; this restores, without text, the v6 frames "
                "and crosses that r1 had moved to the tables",
                "caption: '(a) ... with the preregistered predictions boxed in black'; (c) 'of each pair of runs, "
                "with bars at the medians' -> 'per pair of runs, with median bars' (100 words)"],
    "caption_r1": CAPTION_V8_R1, "caption_r1_words": len(caption_words(CAPTION_V8_R1)),
    "data_unchanged": "panel_a, panel_b, panel_c, sources asserted equal to v3 / v4 / v6 as in r1",
    "r1_files": "./_prev_r1_maker/ (maker, outputs, MOVED.md of the maker's r1)"}
NUM["v8"]["kept_messages"]["a"] += ("; v8 r2: the preregistered predictions as boxes, with the two failed predictions "
                                    "visible as marks outside their boxes")
NUM["outputs"] = {p: sha(os.path.join(OUT, p)) for p in ("F_gauge_app.pdf", "F_gauge_app.png")}
NUM["outputs"]["png"] = "200 dpi, pdftoppm (S.print_view)"
with open(os.path.join(OUT, "numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1)
print(json.dumps({"check": {k: CHK[k] for k in ("size_in", "min_text_pt", "min_glyph_pt", "n_type3", "fonts",
                                                "not_bold_tick_or_axis_labels", "bold_text_with_regular_math",
                                                "chromatic_data_colours", "off_palette_colours")},
                  "clash": clash, "hit": HIT, "gaps": GAPS, "head_gap": HEAD_GAP, "H": H,
                  "caption_words": [N_WORDS, N_WORDS_BOLD]}, indent=1))
print("closest", CLEAR[:12])
print("OK")
