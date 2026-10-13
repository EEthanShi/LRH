#!/usr/bin/env python3
"""F_gauge_v11_app, v8 render (arxiv_v4 appendix figure \\label{fig:app-s22v11}, second network experiment,
app:exp-s22-v11; find it by the label).
Message (v8 caption, bold sentence): the imbalance kept by head flow from the initialization adds to the disagreement of
the head coordinates, by less than the imbalance created by momentum, and weight decay removes the excess on G_s
(Theorem thm:gauge, Remark rem:optimizers).

v8 (2026-10-10, role maker for the appendix gauge figures): copy of
~/hrh_v2_figs/v6/F_gauge_v11_app/make_F_gauge_v11_app.py (sha256 54d54430...2078, untouched; its outputs stay in
~/hrh_v2_figs/v6/F_gauge_v11_app/), switched to the frozen v8 style ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (sha256
asserted). The data code and every assert of v6 are kept verbatim; the drawing is rebuilt. What changed, and why
(author feedback of 2026-10-10, binding for every experiment figure):
  * no setting text, little text: the panel titles 'distortion by the imbalance (exploratory)' and 'drift of the
    imbalance C', the head '(distances up to scale)', the group label '(C_0 = 0)', the cloud labels '(C = C_0)' and
    '(C = C_end)', the x-label definition '(s: smallest singular value of the head)', the long x label of (a, b), the
    'limit of check C1' label and the N = 25,000 / 250,000 / mean legend are gone; the caption decodes them.
  * no wide gutter for row labels: the bold two-line group labels of (a, b) ('default init.', 'balanced init.',
    'small init.', 'weight decay') leave the gutter for one key row at the top (a dot in each initialization colour and
    its name); the rows keep only the rule of the head ('momentum', 'head flow'), so (a, b) gain 0.55 in of width.
  * secondary comparison not drawn: the Euler-step arm (SGD-K, 32 pairs) leaves (a, b). The decomposition of its G_s
    excess is in Table tab:app-s22v11-decomp; its drift of C, the reason the arm exists, stays in (d) with the
    substep cells KS-4 and KS-16.
  * colour: v8 palette (Tol): default init. #222255 (v6 black), balanced init. #6699CC (v6 grey #555555, a data series
    in grey), small init. #AA4499, weight decay #44AA99; bold tick labels and axis titles with bold math (S.finalize;
    S.check asserts both, and that the figure has chromatic data colours).
Earlier history: the docstring of the v6 maker (r2, r1) in ~/hrh_v2_figs/v6/F_gauge_v11_app/.

Panels (5.5 x H in)
  (a) G_s, (b) 1 - CKA: per pair of runs, the excess X = Delta(u) - Delta(y), its balanced part B and its imbalance
      part I (X = B + I per pair), in SD of Delta(y) over the cell's main pairs; seven arms (SGD-K computed and
      asserted, not drawn); per arm an upper strip N = 25,000 and a lower strip N = 250,000 in the same colour; dots =
      pairs, bars = means; one scale (inches per SD) in all six sub-axes.
  (c) exploratory (I6): per run, distortion D (log) against ||C0||_op / s under head flow (FLOW) and against
      ||C_end||_op / s under momentum (MOM), both sizes; the momentum clouds are labelled by size.
  (d) relative drift of C per run (log), rows = rule of the head, colours = initialization, both sizes pooled; dashed
      line = the limit of check C1.
Read-only inputs (sha256 recorded in numbers.json; three frozen hashes asserted):
  LAD   ~/hrh_v2_stage2_20261008/results/S2_2_v11/ladders/S2_2v11_main/{y,u,ubal}/S2.2v11_<ARM>_N<N>_r*__*.json
  SV11  ~/hrh_v2_stage2_20261008/results/S2_2_v11/s22v11_summary.json
  A4    ~/hrh_v2_stage2_20261008/results/S2_2_v11/render/a4_parts.csv
  COV   ~/hrh_v2_stage2_20261008/results/S2_2_v11/render/covariates_by_cell.csv
  V6    ~/hrh_v2_figs/v6/F_gauge_v11_app/numbers.json (every data field asserted equal)
  STYLE ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (frozen; imported, sha256 asserted)
Writes only into this directory: F_gauge_v11_app.pdf, F_gauge_v11_app.png (200 dpi, pdftoppm),
_view_print_100dpi.png, _views_cvd_grey.png, numbers.json, caption_v8.tex.
Run: cd ~/hrh_v2_figs/v8/F_gauge_v11_app && OMP_NUM_THREADS=1 nice -n 10 taskset -c 0-5 ~/hrh_v2_figs_venv/bin/python \
     -B make_F_gauge_v11_app.py > run.log 2>&1
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import matplotlib.collections as mcoll  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


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

# ======== data: the v6 code (v6 lines 83-277), verbatim except GROUP (v8 colours) ========
FIG_ID = "F_gauge_v11_app"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


assert sha(STYLE_PATH) == STYLE_SHA == S.STYLE_SHA256, "frozen style module changed"

HOME = os.path.expanduser("~")
S2 = f"{HOME}/hrh_v2_stage2_20261008"
RES = f"{S2}/results/S2_2_v11"
LAD = f"{RES}/ladders/S2_2v11_main"
SV11 = f"{RES}/s22v11_summary.json"
A4 = f"{RES}/render/a4_parts.csv"
COV = f"{RES}/render/covariates_by_cell.csv"
FROZEN = {SV11: "c05262cba20219458eb132be822c7114a881fbf2de2b8ea1adc24d525fd2437c",
          A4: "d348a70782fec5fa3547c33d76137529d404de854b03a59d979401b84803f149",
          COV: "73f90a78a02a5839f91ba43235440a6932eea72f62a18d3a19d228d03ac4d8f6"}
for _p, _h in FROZEN.items():
    assert sha(_p) == _h, ("frozen S2.2 v1.1 output changed", _p)

sv11 = json.load(open(SV11))
a4 = {}
with open(A4) as fh:
    for row in csv.DictReader(fh):
        a4[(row["cell"], row["rung"], row["part"])] = row
cov = {}
with open(COV) as fh:
    for row in csv.DictReader(fh):
        cov[row["cell"]] = row

# =====================================================================================================================
# arms, groups, colours, labels (as F_gauge v6 r2 (d, e): balanced init. = neutral MUTED grey, a variant of the default
# init.; one colour per initialization, teal = weight decay; N is coded by position, not by lightness)
# =====================================================================================================================
ARMS = [("MOM", "default", "momentum"), ("FLOW", "default", "head flow"), ("SGD-K", "default", "Euler step"),
        ("BAL-MOM", "balanced", "momentum"), ("BAL-FLOW", "balanced", "head flow"),
        ("SMALL-MOM", "small", "momentum"), ("SMALL-FLOW", "small", "head flow"),
        ("WD", "wd", "momentum")]
GROUP = {  # v8: colour, text tone, key name (the gutter labels of v6 became the key row)
    "default":  (S.ARM["gd"], S.ARM_TEXT["gd"], ["default init."]),
    "balanced": (S.ARM["balanced"], S.ARM_TEXT["balanced"], ["balanced init."]),
    "small":    (S.ARM["smallinit"], S.ARM_TEXT["smallinit"], ["small init."]),
    "wd":       (S.ARM["wd"], S.ARM_TEXT["wd"], ["weight decay"]),
}
NPAIRS = {"MOM": 96, "FLOW": 96, "BAL-FLOW": 96, "SMALL-FLOW": 96, "WD": 96, "BAL-MOM": 32, "SMALL-MOM": 32,
          "SGD-K": 32}
NS = (25000, 250000)
NLAB = {25000: "$N=25{,}000$", 250000: "$N=250{,}000$"}
QS = [("Gs", "$G_s$ (distances up to scale)"), ("dCKA", "$1-$CKA")]
PARTS = ["X", "B", "I"]


def ncol(grp, N=None):
    """One colour per group at both sizes (r2)."""
    return GROUP[grp][0]


# =====================================================================================================================
# data (a, b): per pair X, B, I on G_s and 1 - CKA, every ladder arm and N; asserted against a4_parts.csv and the
# frozen summary (values, sigma_y, A4, I3, I4)
# =====================================================================================================================
NET = {}
lad_files = []
for arm, grp, _ in ARMS:
    for N in NS:
        cell = f"{arm}_{N}"
        names = sorted(os.path.basename(p) for p in glob.glob(f"{LAD}/y/S2.2v11_{arm}_N{N}_r*__*.json"))
        assert len(names) == NPAIRS[arm], (cell, len(names))
        for tag in ("u", "ubal"):
            assert sorted(os.path.basename(p) for p in glob.glob(f"{LAD}/{tag}/S2.2v11_{arm}_N{N}_r*__*.json")) == names
        raw = {Q: {} for Q, _ in QS}
        for tag in ("y", "u", "ubal"):
            vals = {Q: [] for Q, _ in QS}
            for n in names:
                fp = f"{LAD}/{tag}/{n}"
                lad_files.append(fp)
                rec = json.load(open(fp))
                assert rec["tag"] == tag and not rec["failed_run"] and rec["cohort"] == "S2_2v11_main", (tag, n)
                for Q, _ in QS:
                    v = rec["deltas"][Q]
                    assert v == sv11["values"][f"{tag}|{rec['A']}|{rec['B']}"][Q], (tag, n, Q)
                    vals[Q].append(v)
            for Q, _ in QS:
                raw[Q][tag] = np.asarray(vals[Q], dtype=float)
        NET[cell] = {"arm": arm, "group": grp, "N": N, "names": names}
        for Q, _ in QS:
            D = raw[Q]
            assert np.all(np.isfinite(D["y"])) and np.all(np.isfinite(D["u"])) and np.all(np.isfinite(D["ubal"]))
            sig = float(np.std(D["y"], ddof=1))
            rawp = {"X": D["u"] - D["y"], "B": D["ubal"] - D["y"], "I": D["u"] - D["ubal"]}
            sp = {p: rawp[p] / sig for p in PARTS}
            assert np.allclose(sp["X"], sp["B"] + sp["I"], rtol=0, atol=1e-12)
            assert abs(sig - sv11["sigma_y"][cell][Q]) <= 1e-9 * sig, (cell, Q)
            for p in PARTS:
                r = a4[(cell, Q, p)]
                assert int(r["n"]) == len(names), (cell, Q, p)
                assert abs(sig - float(r["sigma_y"])) <= 1e-9 * sig, (cell, Q, p)
                assert abs(sp[p].mean() - float(r["mean_sigma_y"])) <= 1e-9, (cell, Q, p)
                assert abs(rawp[p].mean() - float(r["mean"])) <= 1e-12, (cell, Q, p)
                assert abs(float(np.median(rawp[p])) - float(r["median"])) <= 1e-12, (cell, Q, p)
                assert abs((rawp[p] > 0).mean() - float(r["share_gt0"])) <= 1e-12, (cell, Q, p)
                b = sv11["A"]["A4"]["cells"][cell][Q][p]
                assert abs(b["sigma_units"]["mean"] - sp[p].mean()) <= 1e-9 and b["n"] == len(names), (cell, Q, p)
            NET[cell][Q] = {"sigma_y": sig, "raw": rawp, "sd": sp}

# I-block: I4 (FLOW, the imbalance part under head flow) and I3 (momentum minus head flow from BAL and SMALL)
I4 = sv11["I"]["I4"]["intervals"]
for N in NS:
    for Q, _ in QS:
        iv = I4[f"{N}|{Q}|I"]
        d = NET[f"FLOW_{N}"][Q]
        assert iv["n"] == 96 and abs(iv["mean_sigma"] - d["sd"]["I"].mean()) <= 1e-9, (N, Q)
        assert abs(iv["sigma_y"] - d["sigma_y"]) <= 1e-12, (N, Q)
        for g in ("BAL", "SMALL"):
            t = sv11["I"]["I3"]["tests"][f"{g}|{Q}|{N}"]
            diff = NET[f"{g}-MOM_{N}"][Q]["raw"]["I"].mean() - NET[f"{g}-FLOW_{N}"][Q]["raw"]["I"].mean()
            assert abs(t["diff"] - diff) <= 1e-12 and t["holm"]["reject"], (g, Q, N)
# values checked by the plan and the analysis report (rounded), 250k G_s
for arm, x_, i_ in (("MOM", 5.61, 3.21), ("FLOW", 3.45, 0.59), ("BAL-FLOW", 3.14, 0.00), ("WD", -0.04, 0.17),
                    ("BAL-MOM", 6.56, 3.86)):
    d = NET[f"{arm}_250000"]["Gs"]["sd"]
    assert round(d["X"].mean(), 2) == x_ and abs(round(d["I"].mean(), 2)) == abs(i_), arm

# =====================================================================================================================
# data (c), (d): per run from the frozen summary; medians asserted against covariates_by_cell.csv and the C blocks
# =====================================================================================================================
PR = sv11["per_run"]


def runs_of(arm, N):
    rs = [(k, r) for k, r in PR.items() if r["arm"] == arm and r["N"] == N]
    assert rs and all(r["ok"] and r["state"] == "done" and not r["diverged"] for _, r in rs), (arm, N)
    c = cov[f"{arm}_{N}"]
    assert int(c["n"]) == len(rs) == int(c["n_ok"]), (arm, N)
    return sorted(rs)


def med_ok(vals, key, cell):
    m, ref = float(np.median(vals)), float(cov[cell][key])
    assert abs(m - ref) <= 1e-12 * max(1.0, abs(ref)), (cell, key, m, ref)
    return m


PANEL_C = {}
for arm, xkey in (("FLOW", "C0_op"), ("MOM", "C_end_op")):
    for N in NS:
        rs = runs_of(arm, N)
        cell = f"{arm}_{N}"
        Dv = np.array([r["D"] for _, r in rs], float)
        sv = np.array([r["s"] for _, r in rs], float)
        cx = np.array([r[xkey] for _, r in rs], float)
        x = cx / sv
        rho = float(spearmanr(np.log(Dv), x).statistic)
        i6 = sv11["I"]["I6"][cell]
        assert i6["n"] == len(rs) and i6["x"] == f"{xkey}/s" and abs(i6["spearman"] - rho) <= 1e-12, (cell, rho)
        med_ok(Dv, "median_D", cell)
        med_ok(sv, "median_s", cell)
        med_ok(cx, f"median_{xkey}", cell)
        PANEL_C[cell] = {"runs": [k for k, _ in rs], "x": x, "D": Dv, "rho": rho, "p_one_sided": i6["p_one_sided"],
                         "xkey": xkey}

D_ROWS = [  # (row label, [(arm, group)], Ns)
    ("momentum", [("MOM", "default"), ("BAL-MOM", "balanced"), ("SMALL-MOM", "small")], NS),
    ("weight decay", [("WD", "wd")], NS),
    ("Euler step", [("SGD-K", "default")], NS),
    ("4 substeps", [("KS-4", "default")], (250000,)),
    ("16 substeps", [("KS-16", "default")], (250000,)),
    ("head flow", [("FLOW", "default"), ("BAL-FLOW", "balanced"), ("SMALL-FLOW", "small")], NS),
]
PANEL_D = {}
for lab, arms, ns in D_ROWS:
    for arm, grp in arms:
        for N in ns:
            rs = runs_of(arm, N)
            cell = f"{arm}_{N}"
            dc = np.array([r["delta_C"] for _, r in rs], float)
            assert np.all(np.isfinite(dc)) and np.all(dc > 0), cell
            m = med_ok(dc, "median_delta_C", cell)
            c5 = sv11["C"]["C5"]["medians"].get(cell)
            if c5 is not None:
                assert abs(c5["median"] - m) <= 1e-12 and c5["n"] == len(rs), cell
            c1 = sv11["C"]["C1"]["cells"].get(cell)
            if c1 is not None:
                assert abs(c1["delta_C_max"] - dc.max()) <= 1e-15 and c1["n_viol"] == 0, cell
            PANEL_D[cell] = {"row": lab, "group": grp, "runs": [k for k, _ in rs], "delta_C": dc,
                             "rho_end": [float(r["rho_end"]) for _, r in rs]}
FLOW_MAX = max(PANEL_D[f"{a}_{N}"]["delta_C"].max() for a in ("FLOW", "BAL-FLOW", "SMALL-FLOW") for N in NS)
assert sv11["C"]["C1"]["pass"] and round(FLOW_MAX, 7) == 2.75e-5, FLOW_MAX
C1_LIMIT = 2e-4                    # preregistered limit of check C1 (prereg section 10; v4 app:exp-s22-v11 "Checks")

# =====================================================================================================================
# v8 figure geometry (inches from the bottom-left corner), 5.5 x H
# =====================================================================================================================
V6NUM = f"{HOME}/hrh_v2_figs/v6/F_gauge_v11_app/numbers.json"
V6NUM_SHA = "56e091c3b128f2c53aca9ec870c822345214aa3674a2387bd9002a7e767183ea"
V6MAKER = f"{HOME}/hrh_v2_figs/v6/F_gauge_v11_app/make_F_gauge_v11_app.py"
V6MAKER_SHA = "54d544302185234c14b70c3215e444ec482b7b81746df708e0c37d8bb0162078"
assert sha(V6NUM) == V6NUM_SHA and sha(V6MAKER) == V6MAKER_SHA, "v6 files changed"
v6num = json.load(open(V6NUM))
ARMS_AB = [a for a in ARMS if a[0] != "SGD-K"]   # v8: the Euler-step arm leaves (a, b) (secondary; kept in (d))
assert [a[0] for a in ARMS_AB] == ["MOM", "FLOW", "BAL-MOM", "BAL-FLOW", "SMALL-MOM", "SMALL-FLOW", "WD"]
W = S.TEXTWIDTH_IN
fig = plt.figure(figsize=(W, 4.0))
fig.set_layout_engine("none")
renderer = fig.canvas.get_renderer()


def text_w(s, role):
    t = fig.text(0, 0, s, **S.text_kw(role, s))
    w = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return w


def ext(t):
    return t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())


DATA, KEY_ART, ANNOT = [], [], []
# rows of (a, b): each arm 1 unit with two strips, N = 25,000 above and N = 250,000 below (same colour)
SUB, JIT_Y, BAR_HALF, GAP = 0.21, 0.085, 0.155, 0.40
ROW_Y, y = {}, 0.0
prev = None
for arm, grp, _ in ARMS_AB:
    if prev is not None and grp != prev:
        y += GAP
    ROW_Y[arm] = y
    y += 1.0
    prev = grp
YLIM_AB = (y - 1.0 + 0.5 + 0.08, -0.5 - 0.08)
UNIT_IN = 0.19                                   # inches per row unit (v6 0.194)
Q_B, Q_T = 0.37, 1.30                            # (c), (d) axes
P_B = Q_T + 0.56                                 # (a, b) axes bottom: (c, d) letters, the (a, b) x label and ticks
P_T = P_B + (YLIM_AB[0] - YLIM_AB[1]) * UNIT_IN
SUBT_Y = P_T + 0.045                             # bottom of the sub-axis titles 'excess = balanced + imbalance'
HEAD_Y = P_T + 0.215                             # baseline of the heads G_s, 1 - CKA and the letters (a), (b)
KEY_Y = P_T + 0.355                              # centre baseline of the key row
H = P_T + 0.44
fig.set_size_inches(W, H)
assert abs((P_T - P_B) / (YLIM_AB[0] - YLIM_AB[1]) - UNIT_IN) < 1e-12


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


GUT_L = 0.02
RULE_W = max(text_w(rl, "tick") for _, _, rl in ARMS_AB)
RULE_R = GUT_L + RULE_W                                                  # right edge of the rule labels
XR = {"X": (-2.8, 10.2), "B": (-3.2, 9.1), "I": (-0.9, 7.0)}           # SD of Delta(y); one scale everywhere (v6)
AB_L, AB_R = RULE_R + 0.06, 5.47
G_SUB, G_PAN = 0.12, 0.22
span = sum(XR[p][1] - XR[p][0] for p in PARTS)
K_IN = (AB_R - AB_L - 4 * G_SUB - G_PAN) / (2 * span)                 # inches per SD
SUBAX = {}
x0 = AB_L
for qi, (Q, _) in enumerate(QS):
    for pi, p in enumerate(PARTS):
        w = (XR[p][1] - XR[p][0]) * K_IN
        SUBAX[(Q, p)] = add_ax(x0, P_B, w, P_T - P_B)
        x0 += w + (G_SUB if pi < 2 else G_PAN)
assert abs(x0 - G_PAN - AB_R) < 1e-9
SEPS, ARM_SEPS = [], []
for (a1, g1, _), (a2, g2, _) in zip(ARMS_AB[:-1], ARMS_AB[1:]):
    (SEPS if g1 != g2 else ARM_SEPS).append((ROW_Y[a1] + ROW_Y[a2]) / 2)
SUBY = {25000: -SUB, 250000: SUB}
rng = np.random.default_rng(20261010)
JIT = {cell: rng.uniform(-JIT_Y, JIT_Y, NPAIRS[NET[cell]["arm"]]) for cell in NET}   # v6 draws (all cells, v6 order)
for (Q, p), ax in SUBAX.items():
    ax.axvline(0.0, color=S.NULL_LINE, lw=LW["zero"], ls=S.LS["dashed"], zorder=Z["ref"])
    for ys in SEPS + ARM_SEPS:
        ax.axhline(ys, color=S.GUIDE, lw=LW["guide"], zorder=Z["guide"])
    for arm, grp, _ in ARMS_AB:
        for N in NS:
            cell = f"{arm}_{N}"
            v = NET[cell][Q]["sd"][p]
            yc = ROW_Y[arm] + SUBY[N]
            col = ncol(grp)
            DATA.append(ax.scatter(v, yc + JIT[cell], s=MS["jitter_s"], color=col, alpha=MS["jitter_alpha"],
                                   linewidths=0, zorder=Z["thin"]))
            mu = float(v.mean())
            by = [yc - BAR_HALF, yc + BAR_HALF]
            S.underlay(ax, [mu, mu], by, LW["median_bar"], Z["data"], extra=LW["median_halo"], solid_capstyle="butt")
            DATA.extend(ax.plot([mu, mu], by, color=col, lw=LW["median_bar"], zorder=Z["data"],
                                solid_capstyle="butt"))
    ax.set_xlim(*XR[p])
    ax.set_ylim(*YLIM_AB)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    maj = [v for v in (0, 4, 8) if XR[p][0] <= v <= XR[p][1]]
    ax.set_xticks(maj)
    ax.set_xticklabels([S.num(v, 0) for v in maj])
    ax.xaxis.set_minor_locator(mticker.FixedLocator([v for v in (-2, 2, 6, 10) if XR[p][0] <= v <= XR[p][1]]))
for cell in NET:
    for Q, _ in QS:
        for p in PARTS:
            v = NET[cell][Q]["sd"][p]
            assert XR[p][0] + 0.15 < v.min() and v.max() < XR[p][1] - 0.15, (cell, Q, p, v.min(), v.max())

# sub-axis titles "excess = balanced + imbalance" and the criterion heads
PART_T = {"X": "excess", "B": "balanced", "I": "imbalance"}
SUBT, HEADS = [], []
for Q, qlab in QS:
    for pi, p in enumerate(PARTS):
        ax = SUBAX[(Q, p)]
        SUBT.append(S.text(ax, 0.5, 1.0 + 0.045 / (P_T - P_B), PART_T[p], "subheader", color=INK,
                           transform=ax.transAxes, ha="center", va="bottom"))
    for p0, p1, sym in (("X", "B", "="), ("B", "I", "+")):
        xa = SUBAX[(Q, p0)].get_position().x1 * W
        xb = SUBAX[(Q, p1)].get_position().x0 * W
        SUBT.append(S.text(fig, (xa + xb) / 2 / W, SUBT_Y / H, sym, "subheader", color=INK, ha="center", va="bottom"))
    xl = SUBAX[(Q, "X")].get_position().x0 * W
    xr = SUBAX[(Q, "I")].get_position().x1 * W
    HEADS.append(S.text(fig, (xl + xr) / 2 / W, HEAD_Y / H, {"Gs": "$G_s$", "dCKA": "$1-\\mathrm{CKA}$"}[Q], "header",
                        ha="center", va="baseline"))
XLAB_AB_TEXT = "parts of the excess, in SD of $\\Delta(y)$ over pairs"
XLAB_AB = S.text(fig, (AB_L + AB_R) / 2 / W, (P_B - 0.29) / H, XLAB_AB_TEXT, "label", ha="center", va="baseline")

# row labels: the rule of the head, right-aligned at the axes (INK, bold tick role); guides between arms and groups
tr_ab = blended_transform_factory(fig.transFigure, SUBAX[("Gs", "X")].transData)
ROW_LABELS = []
for arm, grp, rl in ARMS_AB:
    ROW_LABELS.append(S.text(fig, RULE_R / W, ROW_Y[arm], rl, "tick", ha="right", va="center", transform=tr_ab))
for ys in SEPS:
    fig.add_artist(Line2D([GUT_L / W, AB_L / W], [ys, ys], transform=tr_ab, color=S.GUIDE, lw=LW["guide"],
                          zorder=Z["guide"]))
for ys in ARM_SEPS:
    fig.add_artist(Line2D([(RULE_R - RULE_W) / W, AB_L / W], [ys, ys], transform=tr_ab, color=S.GUIDE,
                          lw=LW["guide"], zorder=Z["guide"]))

# key row: one dot per initialization (the colour of its rows and runs) and its name in the group text tone
KEY_NAMES = {"default": "default init.", "balanced": "balanced init.", "small": "small init.", "wd": "weight decay"}
DOT, DG, KG = 0.075, 0.04, 0.22               # dot slot, dot-text gap, gap between entries (in)
widths = {g: text_w(KEY_NAMES[g], "direct") for g in GROUP}
KEY_TOTAL = sum(DOT + DG + widths[g] for g in GROUP) + KG * (len(GROUP) - 1)
kx = (AB_L + AB_R) / 2 - KEY_TOTAL / 2
KEY_TXT = {}
for g in GROUP:
    KEY_ART.append(fig.add_artist(Line2D([(kx + DOT / 2) / W], [(KEY_Y + 0.027) / H], transform=fig.transFigure,
                                         ls="none", marker="o", ms=MS["line"], mfc=GROUP[g][0], mec=GROUP[g][0],
                                         mew=0)))
    KEY_TXT[g] = S.text(fig, (kx + DOT + DG) / W, KEY_Y / H, KEY_NAMES[g], "direct", color=GROUP[g][1], ha="left",
                        va="baseline")
    kx += DOT + DG + widths[g] + KG

# ---- bottom block (c), (d) -----------------------------------------------------------------------------------------
C_L, C_R = 0.47, 2.78
ax_c = add_ax(C_L, Q_B, C_R - C_L, Q_T - Q_B)
D_GUT = 3.02                                     # left edge of the row labels of (d)
D_LABW = max(text_w(lab, "tick") for lab, _, _ in D_ROWS)
D_L, D_R = D_GUT + D_LABW + 0.06, 5.45
ax_d = add_ax(D_L, Q_B, D_R - D_L, Q_T - Q_B)

C_XLIM, C_YLIM = (0.05, 2.6), (1.0, 1.95)
for arm in ("FLOW", "MOM"):
    for N in NS:
        d = PANEL_C[f"{arm}_{N}"]
        assert C_XLIM[0] < d["x"].min() and d["x"].max() < C_XLIM[1], (arm, N)
        assert C_YLIM[0] < d["D"].min() and d["D"].max() < C_YLIM[1], (arm, N)
        DATA.append(ax_c.scatter(d["x"], d["D"], s=MS["jitter_s"], color=ncol("default"), alpha=MS["jitter_alpha"],
                                 linewidths=0, zorder=Z["thin"]))
ax_c.set_xscale("log")
ax_c.set_yscale("log")
ax_c.set_xlim(*C_XLIM)
ax_c.set_ylim(*C_YLIM)
ax_c.set_xticks([0.1, 0.3, 1, 2])
ax_c.set_xticklabels(["0.1", "0.3", "1", "2"])
ax_c.xaxis.set_minor_locator(mticker.FixedLocator([0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
ax_c.xaxis.set_minor_formatter(mticker.NullFormatter())
ax_c.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
ax_c.set_yticklabels(["1.0", "1.2", "1.4", "1.6", "1.8"])
ax_c.yaxis.set_minor_locator(mticker.NullLocator())
ax_c.yaxis.set_minor_formatter(mticker.NullFormatter())
ax_c.set_ylabel("distortion $D$")
ax_c.set_xlabel("$\\|C\\|_{\\mathrm{op}}/s$")
# direct labels of the clouds (they identify data subsets: rule of the head, and the size of each momentum cloud)
ANNOT.append(S.text(ax_c, 0.058, 1.265, "head flow", "direct", color=S.ARM_TEXT["gd"], ha="left", va="center"))
ANNOT.append(S.text(ax_c, 2.5, 1.20, "momentum", "direct", color=S.ARM_TEXT["gd"], ha="right", va="center"))
for N, (xr_, yr_) in ((25000, (0.85, 1.38)), (250000, (1.23, 1.645))):
    ANNOT.append(S.text(ax_c, xr_, yr_, NLAB[N], "annot", ha="right", va="center"))

D_H = {lab: (1.0 if len(arms) > 1 else 0.56) for lab, arms, _ in D_ROWS}
D_ROW_Y, yy0 = {}, 0.0
for lab, _, _ in D_ROWS:
    D_ROW_Y[lab] = yy0 + D_H[lab] / 2
    yy0 += D_H[lab]
D_YMAX = yy0
INIT_OFF = {"default": -0.29, "balanced": 0.0, "small": 0.29, "wd": 0.0}
D_JIT = 0.075
rng_d = np.random.default_rng(20261011)
JIT_D = {cell: rng_d.uniform(-D_JIT, D_JIT, len(PANEL_D[cell]["delta_C"])) for cell in PANEL_D}
D_XLIM = (1e-5, 1.0)
ax_d.axvline(C1_LIMIT, color=S.NULL_LINE, lw=LW["ref"], ls=S.LS["dashed"], zorder=Z["ref"])
D_YPOS = {}
for lab, arms, ns in D_ROWS:
    for arm, grp in arms:
        for N in ns:
            cell = f"{arm}_{N}"
            d = PANEL_D[cell]
            D_YPOS[cell] = D_ROW_Y[lab] + (INIT_OFF[grp] if len(arms) > 1 else 0.0)
            DATA.append(ax_d.scatter(d["delta_C"], D_YPOS[cell] + JIT_D[cell], s=MS["jitter_s"], color=ncol(grp),
                                     alpha=MS["jitter_alpha"], linewidths=0, zorder=Z["thin"]))
            assert D_XLIM[0] < d["delta_C"].min() and d["delta_C"].max() < D_XLIM[1], cell
ax_d.set_xscale("log")
ax_d.set_xlim(*D_XLIM)
ax_d.set_ylim(D_YMAX + 0.04, -0.04)
ax_d.set_yticks([])
ax_d.spines["left"].set_visible(False)
ax_d.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
ax_d.set_xticklabels([S.pow_label(10, -5), S.pow_label(10, -4), S.pow_label(10, -3), S.pow_label(10, -2),
                      S.pow_label(10, -1), "1"])
ax_d.xaxis.set_minor_locator(mticker.NullLocator())
ax_d.set_xlabel("relative drift of $C$")
tr_d = blended_transform_factory(fig.transFigure, ax_d.transData)
D_SEPS = list(np.cumsum([D_H[lab] for lab, _, _ in D_ROWS])[:-1])
for ys in D_SEPS:
    ax_d.axhline(ys, color=S.GUIDE, lw=LW["guide"], zorder=Z["guide"])
    fig.add_artist(Line2D([D_GUT / W, D_L / W], [ys, ys], transform=tr_d, color=S.GUIDE, lw=LW["guide"],
                          zorder=Z["guide"]))
D_LABELS = []
for lab, _, _ in D_ROWS:
    D_LABELS.append(S.text(fig, (D_L - 0.05) / W, D_ROW_Y[lab], lab, "tick", transform=tr_d, ha="right", va="center"))

# panel letters: (a), (b) on the baseline of the criterion heads; (c), (d) above their axes
LET = {}
LET["a"] = S.panel_letter(fig, "a", 0.02, HEAD_Y)
LET["b"] = S.panel_letter(fig, "b", SUBAX[("dCKA", "X")].get_position().x0 * W - 0.10, HEAD_Y)
for l in "ab":
    LET[l].set_va("baseline")
    LET[l].set_y(HEAD_Y / H)
LET["c"] = S.panel_letter(fig, "c", 0.02, Q_T + 0.165)
LET["d"] = S.panel_letter(fig, "d", D_GUT, Q_T + 0.165)

# =====================================================================================================================
# v8: bold ticks, axis titles and math; save; check; render; views
# =====================================================================================================================
N_FINAL = S.finalize(fig)
fig.canvas.draw()
PDF, PNG = os.path.join(HERE, f"{FIG_ID}.pdf"), os.path.join(HERE, f"{FIG_ID}.png")
PNG100, VIEWS = os.path.join(HERE, "_view_print_100dpi.png"), os.path.join(HERE, "_views_cvd_grey.png")
fig.savefig(PDF)
CHK = S.check(fig, PDF)
assert abs(CHK["size_in"][0] - W) < 1e-9 and abs(CHK["size_in"][1] - H) < 1e-9, CHK
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
assert not any("Type 3" in ln for ln in pf.splitlines()), pf
S.print_view(PDF, PNG, PNG100)
S.views(PNG, VIEWS)
assert {GROUP[g][0].upper() for g in GROUP} <= set(CHK["chromatic_data_colours"]), CHK["chromatic_data_colours"]


def stroke_box(marker, ms, mew):
    from matplotlib.markers import MarkerStyle
    st = MarkerStyle(marker)
    path = st.get_path().transformed(st.get_transform())
    e = path.get_extents()
    return np.array([e.x0, e.y0, e.x1, e.y1]) * ms + np.array([-1, -1, 1, 1]) * mew / 2


def text_boxes():
    out = []
    for t in fig.findobj(mtext.Text):
        if t.get_visible() and t.get_text().strip():
            out.append((t.get_text(), ext(t)))
    return out


TB = text_boxes()
clash = [(a, b) for i, (a, ba) in enumerate(TB) for (b, bb) in TB[i + 1:] if ba.overlaps(bb)
         and ba.intersection(ba, bb).width > 0.005 and ba.intersection(ba, bb).height > 0.005]
outside = [(a, [round(v, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)]) for a, bb in TB
           if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]


def data_shapes(arts):
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
            b = stroke_box(m, a.get_markersize(), a.get_markeredgewidth()) / 72
            boxes += [(x + b[0], y_ + b[1], x + b[2], y_ + b[3]) for x, y_ in pts]
        if a.get_linestyle() not in ("None", "", " ") and a.get_linewidth() > 0:
            segs += [(pts[i], pts[i + 1], a.get_linewidth() / 2 / 72) for i in range(len(pts) - 1)]
    return boxes, segs


def seg_gap(r, sg):
    p0, p1, hw = sg
    tt = np.linspace(0, 1, 200)[:, None]
    pts = p0 + tt * (p1 - p0)
    dx = np.maximum.reduce([r[0] - pts[:, 0], pts[:, 0] - r[2], np.zeros(len(pts))])
    dy = np.maximum.reduce([r[1] - pts[:, 1], pts[:, 1] - r[3], np.zeros(len(pts))])
    return float((np.hypot(dx, dy) - hw).min())


BOXES, SEGS = data_shapes(DATA + KEY_ART)
BOX_ARR = np.array(BOXES)
CLEAR = []
for t in fig.findobj(mtext.Text):
    if not (t.get_visible() and t.get_text().strip()):
        continue
    bb = ext(t)
    r = (bb.x0, bb.y0, bb.x1, bb.y1)
    dx = np.maximum.reduce([BOX_ARR[:, 0] - r[2], r[0] - BOX_ARR[:, 2], np.zeros(len(BOX_ARR))])
    dy = np.maximum.reduce([BOX_ARR[:, 1] - r[3], r[1] - BOX_ARR[:, 3], np.zeros(len(BOX_ARR))])
    inside = (BOX_ARR[:, 0] < r[2]) & (r[0] < BOX_ARR[:, 2]) & (BOX_ARR[:, 1] < r[3]) & (r[1] < BOX_ARR[:, 3])
    gb = np.where(inside, -1.0, np.hypot(dx, dy)).min()
    near = [sg for sg in SEGS if min(sg[0][0], sg[1][0]) - 0.3 < r[2] and max(sg[0][0], sg[1][0]) + 0.3 > r[0]
            and min(sg[0][1], sg[1][1]) - 0.3 < r[3] and max(sg[0][1], sg[1][1]) + 0.3 > r[1]]
    gs = min([seg_gap(r, sg) for sg in near], default=np.inf)
    CLEAR.append((t.get_text(), round(float(min(gb, gs)), 4)))
HIT = [c for c in CLEAR if c[1] < 0]
ANNOT_CLEAR = {t.get_text(): dict(CLEAR).get(t.get_text()) for t in ANNOT}
print("outside", outside)
print("clash", clash)
print("hit", HIT)
print("closest", sorted(CLEAR, key=lambda c: c[1])[:14])
assert not outside, outside
assert not clash, clash
assert not HIT, HIT
assert min(ANNOT_CLEAR.values()) >= 0.02, ANNOT_CLEAR
# layout gaps (pt)
_tk_ab = min(ext(t).y0 for ax in SUBAX.values() for t in ax.get_xticklabels() if t.get_visible() and t.get_text())
_let_cd_top = max(ext(LET[l]).y1 for l in "cd")
GAPS = {"ab_xlabel_to_ab_ticks_pt": round((_tk_ab - ext(XLAB_AB).y1) * 72, 2),
        "ab_xlabel_to_cd_letters_pt": round((ext(XLAB_AB).y0 - _let_cd_top) * 72, 2),
        "cd_letters_to_cd_axes_pt": round((min(ext(LET[l]).y0 for l in "cd") - Q_T) * 72, 2),
        "subtitles_to_heads_pt": round((min(ext(t).y0 for t in HEADS) - max(ext(t).y1 for t in SUBT)) * 72, 2),
        "heads_to_key_pt": round((min(ext(t).y0 for t in KEY_TXT.values()) - max(ext(t).y1 for t in HEADS)) * 72, 2),
        "top_headroom_pt": round((H - max(bb.y1 for _, bb in TB)) * 72, 2),
        "row_labels_to_axes_in": round(AB_L - max(ext(t).x1 for t in ROW_LABELS), 3)}
assert GAPS["ab_xlabel_to_cd_letters_pt"] > GAPS["ab_xlabel_to_ab_ticks_pt"] >= 2.0, GAPS
assert GAPS["cd_letters_to_cd_axes_pt"] >= 2.0 and GAPS["subtitles_to_heads_pt"] >= 1.0, GAPS
assert GAPS["heads_to_key_pt"] >= 1.0 and GAPS["top_headroom_pt"] >= 0.5, GAPS
# v8: no text on the figure states a setting or a reading
DRAWN = sorted({t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()})
BANNED = ["exploratory", "C_0", "C_{\\mathrm{end}}", "limit", "check", "smallest singular", "distances up to scale",
          "drift of the imbalance", "distortion by", "mean", "Euler step,", "over the pairs of runs"]
assert not [s for s in DRAWN if any(b in s for b in BANNED)], [s for s in DRAWN if any(b in s for b in BANNED)]
assert "Euler step" not in [t.get_text() for t in ROW_LABELS] and "Euler step" in [t.get_text() for t in D_LABELS]

# =====================================================================================================================
# numbers.json (v6 content, every data field asserted equal to the v6 numbers.json; v8 geometry and colour fields)
# =====================================================================================================================
f = lambda a: [float(v) for v in a]  # noqa: E731
panels_ab = {}
for Q, qlab in QS:
    panels_ab[Q] = {}
    for cell, d in NET.items():
        q = d[Q]
        drawn = d["arm"] != "SGD-K"
        panels_ab[Q][cell] = dict(
            arm=d["arm"], group=d["group"], N=d["N"], n_pairs=len(d["names"]), sigma_y=q["sigma_y"],
            y_row=(ROW_Y[d["arm"]] + SUBY[d["N"]]) if drawn else None, drawn_v8=drawn, colour=ncol(d["group"]),
            **{f"{p}_mean_sd": float(q["sd"][p].mean()) for p in PARTS},
            **{f"{p}_median_sd": float(np.median(q["sd"][p])) for p in PARTS},
            **{f"{p}_share_gt0": float((q["raw"][p] > 0).mean()) for p in PARTS},
            **{f"{p}_min_max_sd": [float(q["sd"][p].min()), float(q["sd"][p].max())] for p in PARTS},
            **{f"{p}_per_pair_sd": f(q["sd"][p]) for p in PARTS},
            y_jitter_rows=f(JIT[cell]), pairs=d["names"],
            source=f"{LAD}/{{y,u,ubal}}/<pair> :: deltas.{Q}; X = u - y, B = ubal - y, I = u - ubal, divided by "
                   f"sigma_y = SD (ddof 1) of y over the {len(d['names'])} main pairs",
            asserted_against=[f"{A4} :: cell {cell}, rung {Q}, parts X, B, I: n, mean, median, share_gt0, sigma_y, "
                              "mean_sigma_y", f"{SV11} :: sigma_y['{cell}']['{Q}'], A.A4.cells['{cell}']['{Q}']"
                              "[part].sigma_units.mean, values['<tag>|<A>|<B>']['{Q}'] per pair"])
panel_c = {cell: dict(arm=cell.split("_")[0], N=int(cell.split("_")[1]), x_key=f"{d['xkey']}/s",
                      spearman_logD_x=d["rho"], p_one_sided=d["p_one_sided"], n_runs=len(d["runs"]),
                      x_per_run=f(d["x"]), D_per_run=f(d["D"]), runs=d["runs"],
                      median_D=float(np.median(d["D"])), x_min_max=[float(d["x"].min()), float(d["x"].max())],
                      source=f"{SV11} :: per_run.<run>.{{D, {d['xkey']}, s}}; Spearman asserted equal to "
                             f"I.I6['{cell}'].spearman (1e-12); medians of D, s, {d['xkey']} equal {COV} (1e-12)")
           for cell, d in PANEL_C.items()}
panel_d = {cell: dict(row=d["row"], group=d["group"], y_centre=D_YPOS[cell], colour=ncol(d["group"]),
                      n_runs=len(d["runs"]), median=float(np.median(d["delta_C"])),
                      min_max=[float(d["delta_C"].min()), float(d["delta_C"].max())],
                      delta_C_per_run=f(d["delta_C"]), rho_end_per_run=d["rho_end"], runs=d["runs"],
                      y_jitter=f(JIT_D[cell]),
                      source=f"{SV11} :: per_run.<run>.delta_C; median equals {COV} median_delta_C (1e-12) and "
                             "C.C5.medians where listed; max equals C.C1.cells[cell].delta_C_max for flow arms")
           for cell, d in PANEL_D.items()}


def sd(cell, Q, p):
    return float(NET[cell][Q]["sd"][p].mean())


RESULTS_NUMBERS = {
    "I_head_flow_FLOW_sd": {f"{Q}|{N}": {"mean": I4[f"{N}|{Q}|I"]["mean_sigma"],
                                         "ci_simultaneous_98.75": I4[f"{N}|{Q}|I"]["ci_sigma"],
                                         "run_to_run_I_M": I4[f"{N}|{Q}|I_M"]["mean_sigma"],
                                         "run_to_run_ci": I4[f"{N}|{Q}|I_M"]["ci_sigma"]}
                            for Q, _ in QS for N in NS},
    "I_FLOW_share_of_pairs_gt0": {f"{Q}|{N}": float((NET[f"FLOW_{N}"][Q]["raw"]["I"] > 0).mean())
                                  for Q, _ in QS for N in NS},
    "I_MOM_sd": {f"{Q}|{N}": sd(f"MOM_{N}", Q, "I") for Q, _ in QS for N in NS},
    "ratio_I_FLOW_over_I_MOM": {f"{Q}|{N}": sd(f"FLOW_{N}", Q, "I") / sd(f"MOM_{N}", Q, "I") for Q, _ in QS for N in NS},
    "I_BAL_SMALL_FLOW_sd": {f"{a}|{Q}|{N}": sd(f"{a}_{N}", Q, "I") for a in ("BAL-FLOW", "SMALL-FLOW")
                            for Q, _ in QS for N in NS},
    "I3_raw_diff_momentum_minus_flow": {f"{g}|{Q}|{N}": sv11["I"]["I3"]["tests"][f"{g}|{Q}|{N}"]["diff"]
                                        for g in ("BAL", "SMALL") for Q, _ in QS for N in NS},
    "B_flow_arms_sd": {f"{a}|{Q}|{N}": sd(f"{a}_{N}", Q, "B") for a in ("FLOW", "BAL-FLOW", "SMALL-FLOW")
                       for Q, _ in QS for N in NS},
    "X_WD_sd": {f"{Q}|{N}": sd(f"WD_{N}", Q, "X") for Q, _ in QS for N in NS},
    "X_WD_share_gt0": {f"{Q}|{N}": float((NET[f"WD_{N}"][Q]["raw"]["X"] > 0).mean()) for Q, _ in QS for N in NS},
    "I_WD_sd": {f"{Q}|{N}": sd(f"WD_{N}", Q, "I") for Q, _ in QS for N in NS},
    "X_I_SGDK_sd": {f"{Q}|{N}": [sd(f"SGD-K_{N}", Q, "X"), sd(f"SGD-K_{N}", Q, "I")] for Q, _ in QS for N in NS},
    "I_all_arms_sd": {f"{a}|{Q}|{N}": sd(f"{a}_{N}", Q, "I") for a, _, _ in ARMS for Q, _ in QS for N in NS},
    "X_all_arms_sd": {f"{a}|{Q}|{N}": sd(f"{a}_{N}", Q, "X") for a, _, _ in ARMS for Q, _ in QS for N in NS},
    "B_all_arms_sd": {f"{a}|{Q}|{N}": sd(f"{a}_{N}", Q, "B") for a, _, _ in ARMS for Q, _ in QS for N in NS},
    "I6_p_one_sided": {cell: PANEL_C[cell]["p_one_sided"] for cell in PANEL_C},
    "I6_median_D": {cell: float(np.median(PANEL_C[cell]["D"])) for cell in PANEL_C},
    "I6_spearman": {cell: PANEL_C[cell]["rho"] for cell in PANEL_C},
    "C1_max_delta_C_flow": float(FLOW_MAX),
    "C5_median_delta_C": {cell: float(np.median(PANEL_D[cell]["delta_C"])) for cell in PANEL_D},
    "C5_slope_and_MOM_over_plain": {k: sv11["C"]["C5"]["extrapolation_250k"][k]
                                    for k in ("slope", "slope_ci95", "MOM_over_plain", "MOM_over_plain_ci95")},
    "source": f"{SV11} (I.I4, I.I3, I.I6, C.C1, C.C5, per_run) and the ladders under {LAD} (asserted against {A4})",
}
# ---- equality with the v6 numbers.json: every data field; geometry and colour fields differ by design -------------
_rt = lambda o: json.loads(json.dumps(o))  # noqa: E731
assert _rt(RESULTS_NUMBERS) == v6num["results_numbers"], "results_numbers differ from v6"
_GEOM_AB = {"y_row", "drawn_v8", "colour"}
for Q in panels_ab:
    assert set(panels_ab[Q]) == set(v6num["panels_a_b"][Q]), Q
    for cell in panels_ab[Q]:
        a_, b_ = _rt(panels_ab[Q][cell]), v6num["panels_a_b"][Q][cell]
        assert {k: v for k, v in a_.items() if k not in _GEOM_AB} == {k: v for k, v in b_.items() if k not in _GEOM_AB}, \
            (Q, cell)
assert _rt(panel_c) == v6num["panel_c"], "panel_c differs from v6"
for cell in panel_d:
    a_, b_ = _rt(panel_d[cell]), v6num["panel_d"][cell]
    assert {k: v for k, v in a_.items() if k != "colour"} == {k: v for k, v in b_.items() if k != "colour"}, cell
COLOUR_CHANGES = {g: [v6num["encoding"]["group_colours"][g], GROUP[g][0]] for g in GROUP
                  if v6num["encoding"]["group_colours"][g] != GROUP[g][0]}
assert COLOUR_CHANGES == {"default": ["#000000", "#222255"], "balanced": ["#555555", "#6699CC"]}, COLOUR_CHANGES

ON_PANEL = {
    "none_besides_design_sizes": "v8 prints no result number on any panel; the only numbers drawn are axis ticks and "
                                 "the design sizes N = 25,000 and 250,000 that label the two momentum clouds of (c)",
    "d_check_limit": {"value": C1_LIMIT, "drawn_as": "dashed line without a label (v8; v6 'limit of check C1')",
                      "source": "preregistered limit of C1 (PREREG_S2_2_v1.1 section 10; "
                                f"{SV11} :: C.C1 pass, n_viol 0 in every flow cell)"},
}
CAPTION_V8 = (
    r"\caption{\textbf{The initial imbalance kept by head flow adds less disagreement than the imbalance created by "
    r"momentum, and weight decay removes the excess on $G_s$ (Theorem~\ref{thm:gauge}, "
    r"Remark~\ref{rem:optimizers}).} (a, b)~Excess $\Delta(u)-\Delta(y)$ of each pair of runs in the second network "
    r"experiment and its balanced and imbalance parts. Dots are pairs, bars means, and each row shows "
    r"$N=25{,}000$ above $N=250{,}000$. (c)~Distortion against the imbalance relative to the smallest singular value "
    r"$s$ of the head, initial under head flow and final under momentum (exploratory). (d)~Relative drift of $C$ per "
    r"run against the limit of check C1 (dashed). Appendix~\ref{app:exp-s22-v11} gives the tests.}")


def caption_words(cap_):
    t = re.sub(r"\\caption\{|\\textbf\{", "", cap_).replace("~", " ")
    t = re.sub(r"\\ref\{[^}]*\}", "REF", t)
    return [x for x in t.split() if re.search(r"[A-Za-z0-9]", x) and not re.fullmatch(r"\([a-e](--[a-e])?\)", x)
            and not re.fullmatch(r"\([a-e],?", x) and not re.fullmatch(r"[a-e]\)", x)]


_bold = re.search(r"\\textbf\{(.*?)\(Theorem", CAPTION_V8).group(1)
_digits = sorted(set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\\(ref|label)\{[^}]*\}", "", CAPTION_V8))), key=float)
N_WORDS, N_WORDS_BOLD = len(caption_words(CAPTION_V8)), len(_bold.split())
assert _digits == ["000", "1", "25", "250"], _digits    # N = 25,000 and 250,000; the name of check C1
assert N_WORDS <= 100 and N_WORDS_BOLD <= 25, (N_WORDS, N_WORDS_BOLD)
assert "—" not in CAPTION_V8 and "---" not in CAPTION_V8
with open(os.path.join(HERE, "caption_v8.tex"), "w") as fh:
    fh.write(CAPTION_V8 + "\n")

inputs = {p: sha(p) for p in (SV11, A4, COV, STYLE_PATH, V6NUM, V6MAKER)}
inputs[f"{LAD}/{{y,u,ubal}}/S2.2v11_<ARM>_N<N>_*.json ({len(lad_files)} files)"] = hashlib.sha256(
    "".join(sha(p) for p in lad_files).encode()).hexdigest() + " (sha256 of the concatenated per-file sha256s)"
out = {
    "figure": {"id": FIG_ID, "version": "v8", "label": "fig:app-s22v11",
               "section": "Appendix F, app:exp-s22-v11 (second network experiment)",
               "message": "The initial imbalance kept by head flow adds less disagreement than the imbalance created "
                          "by momentum, and weight decay removes the excess on G_s (on "
                          "1 - CKA at 250,000 weight decay leaves a small positive excess, stated in the text).",
               "size_in": CHK["size_in"], "min_text_pt": CHK["min_text_pt"],
               "min_glyph_pt_from_pdf_Tf": CHK["min_glyph_pt"], "pdf_glyph_size_counts": CHK["pdf_glyph_size_counts"],
               "n_type3": CHK["n_type3"], "fonts": CHK["fonts"], "pdffonts": pf.splitlines(),
               "low_contrast_text": CHK["low_contrast_text"], "text_overlaps_gt_0.005in": clash,
               "text_vs_data_overlaps": HIT, "annotation_clearance_to_nearest_data_mark_in": ANNOT_CLEAR,
               "closest_text_to_data_in": sorted(CLEAR, key=lambda c: c[1])[:14], "layout_gaps": GAPS,
               "drawn_texts": DRAWN, "n_texts_set_bold_by_finalize": N_FINAL,
               "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
    "inputs": inputs,
    "style": {"path": STYLE_PATH, "sha256": STYLE_SHA, "version": S.STYLE_VERSION, "check": CHK},
    "script_sha256": sha(os.path.abspath(__file__)),
    "asserts": dict(v6num["asserts"], v6_equality=(
        "panels_a_b (every field except y_row, drawn_v8, colour), panel_c (all), panel_d (every field except colour) "
        "and results_numbers equal the v6 numbers.json exactly after a JSON round trip; group colours differ only for "
        "default (#000000 -> #222255) and balanced (#555555 -> #6699CC)")),
    "encoding": {
        "a_b": "rows = the seven arms other than SGD-K, labelled by the rule of the head (momentum, head flow); colour = "
               "initialization of the head (key row: default init., balanced init., small init., weight decay); in "
               "each row N = 25,000 is the upper strip and N = 250,000 the lower strip, same colour (alpha %.2f), strip "
               "centres +-%.2f row units; thin guides between arms and groups; dots = pairs (%.1f pt^2, vertical "
               "jitter +-%.3f row units, seed 20261010, v6 draws); bars = means on a white underlay; dashed line = 0; "
               "one scale %.4f in per SD in all sub-axes, ranges %s" % (MS["jitter_alpha"], SUB, MS["jitter_s"],
                                                                         JIT_Y, K_IN, json.dumps(XR)),
        "c": v6num["encoding"]["c"] + "; v8: cloud labels 'head flow', 'momentum', N = 25,000, N = 250,000 only",
        "d": v6num["encoding"]["d"].replace(", labelled in the head-flow row", ", without a label (v8)"),
        "group_colours": {g: v[0] for g, v in GROUP.items()},
        "group_text_tones": {g: v[1] for g, v in GROUP.items()},
        "group_labels": KEY_NAMES,
        "colour_changes_v6_to_v8": COLOUR_CHANGES,
    },
    "on_panel_numbers": ON_PANEL,
    "panels_a_b": panels_ab,
    "panel_c": panel_c,
    "panel_d": panel_d,
    "results_numbers": RESULTS_NUMBERS,
    "caption_v6": open(os.path.expanduser("~/hrh_v2_figs/v6/F_gauge_v11_app/caption_v6.tex")).read(),
    "caption_v8": CAPTION_V8,
    "caption_v8_check": {"digits_outside_refs": _digits, "meaning": "25,000 and 250,000: the two training sizes "
                         "(design); 1 in C1: the name of a check; no result number", "words_total": N_WORDS,
                         "words_bold_sentence_before_theorem_ref": N_WORDS_BOLD},
    "v8": {"removed_elements": ["(a, b) legend: N = 25,000 / N = 250,000 / mean", "(a, b) gutter group labels "
                                "'default init.', 'balanced init. (C_0 = 0)', 'small init.', 'weight decay' (now the "
                                "key row)", "(a, b) rows of the Euler step (SGD-K)", "head '(distances up to scale)'",
                                "x label 'standard deviations of Delta(y) over the pairs of runs of each arm and size'",
                                "(c) title 'distortion by the imbalance (exploratory)', '(C = C_0)', '(C = C_end)', "
                                "'(s: smallest singular value of the head)'", "(d) title 'drift of the imbalance C', "
                                "'limit of check C1', 'during training' in the x label"],
           "kept_messages": {"a_b": "head flow from the default init. keeps an imbalance part above 0, smaller than "
                                    "under momentum; from a balanced or small start head flow has none and momentum "
                                    "creates one; weight decay leaves no G_s excess",
                             "c": "the distortion rises with the imbalance relative to s",
                             "d": "head flow conserves C (below the C1 limit), momentum, weight decay and the Euler step "
                                  "change it, and substeps shrink the Euler drift"},
           "changed_fields": ["panels_a_b.<cell>.y_row (None for SGD-K)", "panels_a_b.<cell>.colour",
                              "panels_a_b.<cell>.drawn_v8 (new)", "panel_d.<cell>.colour", "encoding.*", "figure.*",
                              "style.*"]},
}
with open(os.path.join(HERE, "numbers.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({"check": {k: CHK[k] for k in ("size_in", "min_text_pt", "min_glyph_pt", "n_type3", "fonts",
                                                "not_bold_tick_or_axis_labels", "bold_text_with_regular_math",
                                                "chromatic_data_colours", "off_palette_colours")},
                  "K_IN": K_IN, "H": H, "gaps": GAPS, "annot_clear": ANNOT_CLEAR,
                  "caption_words": [N_WORDS, N_WORDS_BOLD]}, indent=1))
print("OK")
