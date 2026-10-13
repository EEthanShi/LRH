#!/usr/bin/env python3
"""F_gauge, v8 render (main text, Section 5.5 sec:exp-gauge, Figure 5, label fig:exp-gauge-case). Nothing here enters
the paper; the author decides.

Message: the training rule decides whether independent runs agree on the geometry of a linear head by setting the
imbalance that fixes its gauge (Theorem thm:gauge, Remark rem:optimizers).

Why v8 (author 2026-10-10, on v7 Figure 6, binding for every experiment figure): no black-and-white figures, one
attractive, widely recognized scientific palette (Paul Tol, via hrh_style_v8); tick labels and axis titles bold, the
math in them too; no text on the figure that states a setting (the caption or the Results text names it when it calls
a panel); secondary comparisons that the Results text can state are not drawn. Changes against v7:
  * style hrh_style_v8 (frozen, sha256 asserted): training-rule colours from the semantic map, so GD (default init.)
    is Tol dark blue S.ARM['gd'] (v7 black) and balanced init. Tol light blue S.ARM['balanced'] (v7 grey MUTED);
    small init., weight decay, tied weights keep their Tol hexes; S.finalize(fig) sets every tick label and axis
    title bold with bold math;
  * GD at the preregistered step 0.05 (P-GD) is no longer drawn: in (a) it lies on the same zero line as GD at step
    0.002 (0 of 48 and 0 of 500 pairs on G_s at every N for both), in (b) it differs only at N = 2^8 (46 of 48,
    inconclusive), and in (c) its row carried no prediction ("outside" the theorem). The Results text states the
    decisions of both steps, and Figure F.4 (fig:exp-linae) draws every rule. Its data are still read, asserted
    (against F.4 v5.1 and v6) and written to numbers.json, marked not drawn;
  * the four drawn rules are re-centred on each N column (offsets -0.9 ... 0.9, v7 -1.2 ... 0.6 with the P-GD slot at
    +1.2) and the light columns narrowed by the same 0.3 on each side;
  * key strip: four rules, GD named "GD" (no step size on the figure; the caption says GD takes small steps);
  * (c): the "outside" row, its separator and the in-panel key (measured / predicted) are gone; the caption decodes
    the marks (black = theory, as in every figure); x label "cond(EE^T)" without "at N = infinity" (a setting, now in
    the caption);
  * PNG = the PDF rendered by pdftoppm at 200 dpi (v7: savefig at 300 dpi), plus a 100 dpi print view and the
    normal / deuteranopia / protanopia / greyscale views.
Every plotted value of v7 is kept (asserted against the v6 numbers.json as in v7) except the P-GD series, which is
computed and asserted but not drawn.

Critic r1 (2026-10-10), on top of the v8 maker (r0 kept as _r0_maker/make_F_gauge.py):
  * (c) loses its row labels "part 1 / part 1 / part 3 / part 3": they name the theorem part that predicts each row,
    the same kind of theory label as "inside the premises of Theorem 3" that the author rejected, and the r0 caption
    no longer decoded them. Rows are told apart by the rule colour and marker of the key (same order as the key);
    the part of each row is in the theorem statement and in the Results text (MOVED.md). F.4 v8 drops the same labels.
  * (c) loses the grey reference line at cond = 1: the prediction triangles of the three rows at 1 sat on this line
    and read as arrows pointing at the markers; the tick "1" marks the isotropic value.
  * (c) widens into the freed gutter (left edge = the r0 left edge of the row labels), so (a), (b), (d), (e) keep
    their r0 geometry and the letter of (c) keeps its r0 position.

Read-only inputs (sha256 recorded in numbers.json):
  SUM    ~/hrh_v2_stage1_20261008/results/S1_3a_linae_summary.json         (a, b, c)
  RUNS   ~/hrh_v2_stage1_20261008/linae/criteria/<rule>_Ninf_runs.csv       (c: band, small-init prediction)
  NPZ    ~/hrh_v2_stage1_20261008/linae/runs/WD/Ninf/chunk*.npz             (c: weight-decay prediction)
  F4NUM  ~/hrh_v2_figs/v5_1/F_linae_app/numbers.json                        (a-c asserted equal to 1e-12)
  LAD    ~/hrh_v2_stage2_20261008/results/S2_2_v11/ladders/S2_2v11_main/{y,u,ubal}/S2.2v11_<ARM>_N250000_r*__*.json
  A4     ~/hrh_v2_stage2_20261008/results/S2_2_v11/render/a4_parts.csv     (d, e means and sigma_y asserted, 1e-9)
  SV11   ~/hrh_v2_stage2_20261008/results/S2_2_v11/s22v11_summary.json     (sigma_y, per-pair values, I3/I4 asserted)
  V6NUM  ~/hrh_v2_figs/v6/F_gauge/numbers.json                              (every plotted value asserted equal)
  V7NUM  ~/hrh_v2_figs/v7/F_gauge/numbers.json                              (v8 = v7 for every drawn value)
  STYLE  ./hrh_style_v8.py, a byte copy of ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (frozen; both sha256 asserted)
Code start: ~/hrh_v2_figs/v7/F_gauge/make_F_gauge.py (copied here as _v7_copy_make_F_gauge.py; not edited).
Writes only into this directory: F_gauge.pdf, F_gauge.png (200 dpi from the PDF), _view_print_100dpi.png,
_views_cvd_grey.png, numbers.json.
Run: OMP_NUM_THREADS=1 nice -n 10 taskset -c 0-5 ~/hrh_v2_figs_venv/bin/python -B make_F_gauge.py > run.log 2>&1
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import matplotlib.collections as mcoll  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

STYLE_DIR = HERE                               # v8: byte copy of ~/hrh_v2_figs/v8/_style/hrh_style_v8.py
STYLE_PATH = os.path.join(STYLE_DIR, "hrh_style_v8.py")
STYLE_ORIG = os.path.expanduser("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_PATH, "rb").read()).hexdigest() == STYLE_SHA, "style copy changed"
assert hashlib.sha256(open(STYLE_ORIG, "rb").read()).hexdigest() == STYLE_SHA, "frozen v8 style module changed"
sys.path.insert(0, STYLE_DIR)
import hrh_style_v8 as S  # noqa: E402

S.use()
INK, MUTED, LW, MS, Z = S.INK, S.MUTED, S.LW, S.MS, S.Z


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


assert sha(STYLE_PATH) == STYLE_SHA == S.STYLE_SHA256 == sha(STYLE_ORIG), "frozen style module changed"
assert S.STYLE_VERSION == "v8"

HOME = os.path.expanduser("~")
S1 = f"{HOME}/hrh_v2_stage1_20261008"
S2 = f"{HOME}/hrh_v2_stage2_20261008"
SUM = f"{S1}/results/S1_3a_linae_summary.json"
CRIT_DIR = f"{S1}/linae/criteria"
NPZ_WD = sorted(glob.glob(f"{S1}/linae/runs/WD/Ninf/chunk*.npz"))
F4NUM = f"{HOME}/hrh_v2_figs/v5_1/F_linae_app/numbers.json"
LAD = f"{S2}/results/S2_2_v11/ladders/S2_2v11_main"
A4 = f"{S2}/results/S2_2_v11/render/a4_parts.csv"
SV11 = f"{S2}/results/S2_2_v11/s22v11_summary.json"
SV11_SHA = "c05262cba20219458eb132be822c7114a881fbf2de2b8ea1adc24d525fd2437c"     # frozen (analyze.md)
A4_SHA = "d348a70782fec5fa3547c33d76137529d404de854b03a59d979401b84803f149"       # frozen (analyze.md)
assert NPZ_WD, "no WD chunks"
assert sha(SV11) == SV11_SHA and sha(A4) == A4_SHA, "frozen S2.2 v1.1 outputs changed"

summ = json.load(open(SUM))
meta, cells = summ["meta"], summ["cells"]
f4 = json.load(open(F4NUM))

# =====================================================================================================================
# rules of the linear autoencoder (a-c). Marks as F.4 (v5.1), colours from the v8 semantic map: (key, key label, row
# label in c, covered by the theorem, colour, marker, face, line style, arm key of S.ARM_TEXT). v8: the key label of
# GD at step 0.002 is "GD" (the step is a setting; the caption says GD takes small steps); P-GD is computed, asserted
# and written to numbers.json, but not drawn (DRAWN below).
# =====================================================================================================================
DASH = S.LS["dashed"]
AE = {
    "P-GD-flow":  ("P-GD-flow",  "GD",             "GD, step 0.002 (part 1)", True,  S.ARM["gd"],        "o", "fill", "-",  "gd"),
    "small-init": ("small-init", "small init.",    "small init. (part 1)",    True,  S.ARM["smallinit"], "^", "fill", "-",  "smallinit"),
    "WD":         ("WD",         "weight decay",   "weight decay (part 3)",   True,  S.ARM["wd"],        "s", "fill", "-",  "wd"),
    "TIED":       ("TIED",       "tied weights",   "tied weights (part 3)",   True,  S.ARM["tied"],      "D", "fill", "-",  "tied"),
    "P-GD":       ("P-GD",       "GD, step 0.05",  "GD, step 0.05 (finite step)", False, S.ARM["nowd"], "o", "open", DASH, "nowd"),
}
DODGE_ORDER = ["P-GD-flow", "small-init", "WD", "TIED", "P-GD"]      # as F.4: covered rules first, then outside
DRAWN = ["P-GD-flow", "small-init", "WD", "TIED"]                      # v8: P-GD not drawn (docstring)
KEY_ORDER = ["P-GD-flow", "small-init", "WD", "TIED"]                  # v8: four rules (v7 also "P-GD", second)
NS = ["256", "4096", "65536", "inf"]
XPOS = {"256": 8.0, "4096": 12.0, "65536": 16.0, "inf": 20.5}
XBRK = 18.25
XT = ["$2^{8}$", "$2^{12}$", "$2^{16}$", "$\\infty$"]
DODGE = dict(zip(DODGE_ORDER, np.linspace(-1.2, 1.2, len(DODGE_ORDER))))         # v7 offsets (kept for the v6 assert)
DODGE_V8 = dict(zip(DRAWN, np.linspace(-0.9, 0.9, len(DRAWN))))                   # v8: the four drawn rules, centred
for _ak in DRAWN:                                                                 # same spacing, shifted by +0.3
    assert abs(DODGE_V8[_ak] - (DODGE[_ak] + 0.3)) <= 1e-12, _ak
COL_HW_V7 = 1.66
COL_HW = COL_HW_V7 - 0.3                     # v8: the same 0.46 margin outside the outermost drawn rule as v7
assert abs((COL_HW - max(DODGE_V8.values())) - (COL_HW_V7 - max(DODGE.values()))) <= 1e-12
XLIM_AB = (XPOS["256"] - COL_HW - 0.17, XPOS["inf"] + COL_HW + 0.17)
EPS = meta["eps"]
assert EPS["Gs"] == 0.02 and EPS["proj"] == 0.01, EPS
CRITS = [("Gs", "distances up to scale", "$G_s\\leq %g$" % EPS["Gs"]),
         ("proj", "projector", "$\\Delta_\\Pi\\leq %g$" % EPS["proj"])]
REL_LO, UNREL_HI = 0.9, 0.8
SHAPE = {"P": 1.12, "X": 1.12, "D": 0.85, "^": 1.15}
EDGE = MS["line_edge"]
# v8 semantic map (hrh_style_v8.SEM): one colour per training rule in every figure
SEM_RULE = {"P-GD-flow": "rule.GD", "small-init": "rule.small_init", "WD": "rule.weight_decay", "TIED": "rule.tied_weights",
            "P-GD": "rule.GD"}
for _ak, _sk in SEM_RULE.items():
    assert AE[_ak][4].upper() == S.SEM[_sk].upper(), (_ak, AE[_ak][4], S.SEM[_sk])
assert S.ARM["gd"].upper() == "#222255" and S.ARM["balanced"].upper() == "#6699CC"


def mstyle(arm, where="ab"):
    """F.4 r3 marks: S.MS['line'] = 4.0 pt across in a, b and the key; S.MS['summary'] = 5.0 pt in c (edge included,
    edge in the rule colour; the open GD ring has the same outer diameter)."""
    ak, kl, rl, cov, col, mk, face, ls, rk = arm
    size = MS["line"] if where == "ab" else MS["summary"]
    if face == "open":
        mew = MS["open_small_mew"] if where == "ab" else MS["open_line_mew"]
        return dict(marker=mk, ms=size - mew, mfc="white", mec=col, mew=mew)
    return dict(marker=mk, ms=(size - EDGE) * SHAPE.get(mk, 1.0), mfc=col, mec=col, mew=EDGE)


def mark_underlay(m):
    return dict(marker=m["marker"], ms=m["ms"], mfc="white", mec="white", mew=m["mew"] + LW["median_halo"])


def stroke_box(marker, ms, mew):
    """Extents (pt, relative to the centre) of a drawn marker with matplotlib's joins (copied from F.4 r3)."""
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


# =====================================================================================================================
# data (a, b): decisions of the five rules on G_s and the projector (v8: four drawn; P-GD read and asserted only)
# =====================================================================================================================
numbers = {"panels_a_b": {}, "panel_c": {}, "panels_d_e": {}}
for ck, _, _ in CRITS:
    numbers["panels_a_b"][ck] = {}
    for ak in DODGE_ORDER:
        numbers["panels_a_b"][ck][ak] = {}
        for N in NS:
            c = cells[f"{ak}|{N}"]["primary"][ck]
            numbers["panels_a_b"][ck][ak][N] = dict(
                drawn_v8=ak in DRAWN,
                x_log2N_plotted=(XPOS[N] + float(DODGE_V8[ak])) if ak in DRAWN else None,
                x_log2N_v7=XPOS[N] + float(DODGE[ak]), passes=c["passes"], n_pairs=c["n_pairs"],
                p_hat=c["p_hat"], cp_lower=c["cp_lower"], cp_upper=c["cp_upper"], decision=c["decision"],
                source=f"{SUM} :: cells['{ak}|{N}']['primary']['{ck}']")
            o = f4["panels_a_to_d"][ck][ak][N]          # v5.1 F.4 numbers.json
            for k in ("passes", "n_pairs", "decision"):
                assert o[k] == c[k], (ck, ak, N, k)
            for k in ("p_hat", "cp_lower", "cp_upper"):
                assert abs(o[k] - c[k]) <= 1e-12, (ck, ak, N, k, o[k], c[k])

# =====================================================================================================================
# data (c): cond(EE^T) at N = inf, predictions (copied from F.4: part 1 per run from f(spec C0); part 3 for weight
# decay from each run's own sample eigenvalues; tied = 1 exactly)
# =====================================================================================================================
runs_inf = {ak: pd.read_csv(f"{CRIT_DIR}/{ak}_Ninf_runs.csv") for ak in ("P-GD-flow", "small-init")}


def kappa_pred(df):
    ok = df[~df.diverged.astype(bool)]
    f = ok[[f"eig_fC0_{i}" for i in range(4)]].to_numpy()
    return f.max(1) / f.min(1)


kp_flow = kappa_pred(runs_inf["P-GD-flow"])
band = np.quantile(kp_flow, [0.1, 0.5, 0.9])
lam = meta["wd_lambda"]
wd_pred = []
for fpath in NPZ_WD:
    z = np.load(fpath, allow_pickle=False)
    for r in range(len(z["run_ids"])):
        if bool(z["diverged"][r]):
            continue
        sp = np.sort(1.0 - lam / z["S_eigs"][r][:4])
        wd_pred.append(sp[-1] / sp[0])
pred_marker = {"P-GD-flow": float(band[1]), "small-init": float(np.median(kappa_pred(runs_inf["small-init"]))),
               "WD": float(np.median(wd_pred)), "TIED": 1.0}
pred_q = {"P-GD-flow": [float(v) for v in band],
          "small-init": [float(v) for v in np.quantile(kappa_pred(runs_inf["small-init"]), [0.1, 0.5, 0.9])],
          "WD": [float(v) for v in np.quantile(wd_pred, [0.1, 0.5, 0.9])], "TIED": [1.0, 1.0, 1.0]}
for _k in pred_q:
    assert abs(pred_q[_k][1] - pred_marker[_k]) <= 1e-12, _k
pred_source = {
    "P-GD-flow": f"median over runs of cond(f(spec C0)), {CRIT_DIR}/P-GD-flow_Ninf_runs.csv :: eig_fC0_0..3",
    "small-init": f"median over runs of cond(f(spec C0)), {CRIT_DIR}/small-init_Ninf_runs.csv :: eig_fC0_0..3",
    "WD": "median over non-diverged runs of cond(1 - lambda_wd/lambda_hat_i), "
          f"{S1}/linae/runs/WD/Ninf/chunk*.npz :: S_eigs[:, :4]; lambda_wd = {SUM} :: meta['wd_lambda']",
    "TIED": "part 3: tied weights give E E^T = I on U_hat, cond = 1 exactly"}
assert np.allclose(band, f4["panel_e_band"]["cond_f_spec_C0_q10_50_90"], rtol=0, atol=1e-12)
for ak in AE:
    q = cells[f"{ak}|inf"]["runs"]["kappa_q10_50_90"]
    assert np.allclose(q, f4["panel_e"][ak]["cond_q10_50_90"], rtol=0, atol=1e-12), ak
    pm, opm = pred_marker.get(ak), f4["panel_e"][ak]["predicted_marker"]
    assert (pm is None and opm is None) or abs(pm - opm) <= 1e-12, (ak, pm, opm)

# =====================================================================================================================
# data (d, e): S2.2 v1.1 main cohort at N = 250,000, G_s, per pair matched by filename
# =====================================================================================================================
NET = [  # (arm, group, row label)
    ("MOM", "default", "momentum"), ("FLOW", "default", "head flow"),
    ("BAL-MOM", "balanced", "momentum"), ("BAL-FLOW", "balanced", "head flow"),
    ("SMALL-MOM", "small", "momentum"), ("SMALL-FLOW", "small", "head flow"),
    ("WD", "wd", "momentum"),
]
GROUP = {  # label lines, line colour, text tone. v8: balanced init. = Tol light blue S.ARM['balanced'] (semantic map:
           # the same optimizer as the default init. from a balanced start; v7 grey MUTED, now reserved for references)
    "default":  (["default init."], S.ARM["gd"], S.ARM_TEXT["gd"]),
    "balanced": (["balanced init.", "($C_0=0$)"], S.ARM["balanced"], S.ARM_TEXT["balanced"]),
    "small":    (["small init."], S.ARM["smallinit"], S.ARM_TEXT["smallinit"]),
    "wd":       (["weight decay"], S.ARM["wd"], S.ARM_TEXT["wd"]),
}
SEM_GROUP = {"default": "rule.GD", "balanced": "rule.balanced_init", "small": "rule.small_init", "wd": "rule.weight_decay"}
for _g, _sk in SEM_GROUP.items():
    assert GROUP[_g][1].upper() == S.SEM[_sk].upper(), (_g, GROUP[_g][1], S.SEM[_sk])
    assert S.contrast(GROUP[_g][2]) >= S.MIN_CONTRAST, _g
NPAIRS = {"MOM": 96, "FLOW": 96, "BAL-FLOW": 96, "SMALL-FLOW": 96, "WD": 96, "BAL-MOM": 32, "SMALL-MOM": 32}
sv11 = json.load(open(SV11))
a4 = {}
with open(A4) as fh:
    for row in csv.DictReader(fh):
        a4[(row["cell"], row["rung"], row["part"])] = row
NET_DATA = {}
for arm, grp, rl in NET:
    names = sorted(os.path.basename(p) for p in glob.glob(f"{LAD}/y/S2.2v11_{arm}_N250000_r*__*.json"))
    assert len(names) == NPAIRS[arm], (arm, len(names))
    for tag in ("u", "ubal"):
        assert sorted(os.path.basename(p) for p in glob.glob(f"{LAD}/{tag}/S2.2v11_{arm}_N250000_r*__*.json")) == names
    D = {}
    for tag in ("y", "u", "ubal"):
        vals = []
        for n in names:
            rec = json.load(open(f"{LAD}/{tag}/{n}"))
            assert rec["tag"] == tag and not rec["failed_run"] and rec["cohort"] == "S2_2v11_main", (tag, n)
            v = rec["deltas"]["Gs"]
            sv = sv11["values"][f"{tag}|{rec['A']}|{rec['B']}"]["Gs"]      # per-pair value of the frozen summary
            assert v == sv, (tag, n, v, sv)
            vals.append(v)
        D[tag] = np.asarray(vals, dtype=float)
    sig = float(np.std(D["y"], ddof=1))
    X = (D["u"] - D["y"]) / sig
    I = (D["u"] - D["ubal"]) / sig
    cell = f"{arm}_250000"
    rX, rI = a4[(cell, "Gs", "X")], a4[(cell, "Gs", "I")]
    assert abs(sig - float(rX["sigma_y"])) <= 1e-9 * sig and abs(sig - sv11["sigma_y"][cell]["Gs"]) <= 1e-9 * sig
    assert abs(X.mean() - float(rX["mean_sigma_y"])) <= 1e-9 and abs(I.mean() - float(rI["mean_sigma_y"])) <= 1e-9
    assert abs((D["u"] - D["y"]).mean() - float(rX["mean"])) <= 1e-12
    assert abs((D["u"] - D["ubal"]).mean() - float(rI["mean"])) <= 1e-12
    assert int(rX["n"]) == len(names) == int(rI["n"])
    assert abs(float(rI["share_gt0"]) - float((D["u"] - D["ubal"] > 0).mean())) <= 1e-12
    NET_DATA[arm] = dict(names=names, X=X, I=I, sigma_y=sig, raw_X=D["u"] - D["y"], raw_I=D["u"] - D["ubal"])
# I-block: I4 (FLOW, head flow from the default init.) and I3 (momentum minus head flow, raw I)
i4 = sv11["I"]["I4"]["intervals"]["250000|Gs|I"]
assert abs(i4["mean_sigma"] - NET_DATA["FLOW"]["I"].mean()) <= 1e-9 and i4["n"] == 96
assert abs(i4["sigma_y"] - NET_DATA["FLOW"]["sigma_y"]) <= 1e-12
for g, mom, flow in (("BAL", "BAL-MOM", "BAL-FLOW"), ("SMALL", "SMALL-MOM", "SMALL-FLOW")):
    t = sv11["I"]["I3"]["tests"][f"{g}|Gs|250000"]
    assert abs(t["diff"] - (NET_DATA[mom]["raw_I"].mean() - NET_DATA[flow]["raw_I"].mean())) <= 1e-12, g
# values verified 2026-10-10 by the plan (rounded to 2 decimals)
for arm, x_, i_ in (("MOM", 5.61, 3.21), ("FLOW", 3.45, 0.59), ("BAL-FLOW", 3.14, 0.00), ("WD", -0.04, 0.17),
                    ("BAL-MOM", 6.56, 3.86)):
    assert round(NET_DATA[arm]["X"].mean(), 2) == x_ and abs(round(NET_DATA[arm]["I"].mean(), 2)) == abs(i_), arm


# =====================================================================================================================
# figure geometry (inches from the bottom-left corner). v7: every vertical position hangs from the top edge, and H is
# the smallest height whose bottom label keeps a margin of BOT_MARGIN (asserted after drawing).
# =====================================================================================================================
W = S.TEXTWIDTH_IN
H = float(os.environ.get("F_GAUGE_H", "2.87"))      # v6: 3.95; r2 2.99 (looser)
fig = plt.figure(figsize=(W, H))
fig.set_layout_engine("none")
renderer = fig.canvas.get_renderer()


def add_ax(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def text_w(s, role):
    t = fig.text(0, 0, s, **S.text_kw(role, s))
    w = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return w


def letter_w(letter):
    t = fig.text(0, 0, "(%s)" % letter, fontsize=S.FS["letter"], fontweight=S.FW["letter"])
    w = t.get_window_extent(renderer).width / fig.dpi
    t.remove()
    return w


DATA, KEY_ART, FILLS, ANNOT = [], [], [], []
TITLES = {}
LT_GAP = 0.05                                   # letter-to-title gap (in)


def letter_title(letter, x_in, base_in, title):
    """Panel letter (10 pt bold) and its one-line title (header role) on one baseline."""
    fig.text(x_in / W, base_in / H, "(%s)" % letter, fontsize=S.FS["letter"], fontweight=S.FW["letter"],
             color=S.FC["letter"], ha="left", va="baseline")
    if title:
        TITLES[letter] = S.text(fig, (x_in + letter_w(letter) + LT_GAP) / W, base_in / H, title, "header",
                                ha="left", va="baseline")


# ---- vertical layout, from the top -----------------------------------------------------------------------------------
R1_H = float(os.environ.get("F_GAUGE_R1H", "0.78"))   # axes height of a-c
R2_H = float(os.environ.get("F_GAUGE_R2H", "0.82"))   # axes height of d, e (7 rows)
KEY_Y = H - 0.085                               # centre line of the key strip (rules of a-c)
LET1 = H - 0.28                                 # baseline of the letters and titles of row 1
R1_T = LET1 - 0.075                             # axes top of a-c
R1_B = R1_T - R1_H                              # axes bottom of a-c
XLAB1_BOT = R1_B - 0.325                        # bottom of the x labels of row 1
LET2 = XLAB1_BOT - 0.19                         # baseline of the letters and titles of row 2
R2_T = LET2 - 0.075
R2_B = R2_T - R2_H
XLAB2_BOT = R2_B - 0.31
BOT_MARGIN = 0.015
# ---- horizontal layout -----------------------------------------------------------------------------------------------
X_R = 5.46                                      # right edge of the rightmost axes in both rows
A_L, AB_GAP = 0.48, 0.06                        # (a) left edge (room for 2-line y label and ticks), gap a-b
C_ROWLAB = ["part 1", "part 3"]                 # r0 row labels of (c); critic r1: not drawn, width kept for layout
C_LAB_PAD = 3.0 / 72
C_LABW = max(text_w(s, "tick") for s in C_ROWLAB)
C_W_R0 = 1.80                                   # r0 width of the axes of (c)
C_GUT_L = X_R - C_W_R0 - C_LAB_PAD - C_LABW     # r0 left edge of the row labels of (c) = left edge of panel (c)
C_L = C_GUT_L                                   # critic r1: (c) starts where its r0 row labels started
C_W = X_R - C_L                                 # critic r1: and fills the freed gutter
assert C_W > C_W_R0 and abs(C_L - C_GUT_L) < 1e-12
BC_GAP = 0.15                                   # (b) right edge to the left edge of panel (c)
AB_W = (C_GUT_L - BC_GAP - A_L - AB_GAP) / 2
B_L = A_L + AB_W + AB_GAP
ax_a = add_ax(A_L, R1_B, AB_W, R1_T - R1_B)
ax_b = add_ax(B_L, R1_B, AB_W, R1_T - R1_B)
ax_c = add_ax(C_L, R1_B, C_W, R1_T - R1_B)
NET_TICK = ["momentum", "head flow"]
GUT_L = 0.02
D_TICK_PAD = 3.0 / 72
D_L = GUT_L + max(text_w(s, "tick") for s in NET_TICK) + D_TICK_PAD
DE_GAP = 0.10
DE_W = (X_R - D_L - DE_GAP) / 2
E_L = D_L + DE_W + DE_GAP
ax_d = add_ax(D_L, R2_B, DE_W, R2_T - R2_B)
ax_e = add_ax(E_L, R2_B, DE_W, R2_T - R2_B)

# =====================================================================================================================
# (a), (b)
# =====================================================================================================================
for ax, (ck, cname, tol) in zip((ax_a, ax_b), CRITS):
    for N in NS:
        FILLS.append((ax.axvspan(XPOS[N] - COL_HW, XPOS[N] + COL_HW, color=S.REF_FILL, lw=0, zorder=Z["band"]),
                      S.REF_FILL))
    for yv in (REL_LO, UNREL_HI):
        ax.axhline(yv, color=S.NULL_LINE, lw=LW["level"], zorder=Z["guide"])
    for ak in DRAWN:
        arm = AE[ak]
        col, ls = arm[4], arm[7]
        P = numbers["panels_a_b"][ck][ak]
        xs = np.array([P[N]["x_log2N_plotted"] for N in NS])
        ps = np.array([P[N]["p_hat"] for N in NS])
        los = np.array([P[N]["cp_lower"] for N in NS])
        his = np.array([P[N]["cp_upper"] for N in NS])
        DATA.extend(ax.plot(xs[:3], ps[:3], color=col, lw=LW["data"], ls=ls, zorder=Z["data"]))
        DATA.extend(ax.plot(xs[2:], ps[2:], color=col, lw=LW["break"], ls=S.LS["dotted"], zorder=Z["data"]))
        DATA.append(ax.vlines(xs, los, his, color=col, lw=LW["errorbar"], zorder=Z["interval"]))
        DATA.extend(ax.plot(xs, ps, ls="none", zorder=Z["marker"], **mstyle(arm)))
    ax.set_xlim(*XLIM_AB)
    ax.set_xticks([XPOS[N] for N in NS])
    ax.set_xticklabels(XT, **S.text_kw("tick", XT[0]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(-0.045, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    if ax is ax_a:
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_ylabel("fraction of pairs\nwithin tolerance", linespacing=1.1)
    else:
        ax.set_yticklabels([])
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([XBRK - 0.28, XBRK + 0.28], [0, 0], transform=tr, color="white", lw=LW["spine"] + 1.4, clip_on=False,
            zorder=Z["label"] - 0.5, solid_capstyle="butt")
    for dx in (-0.2, 0.2):
        ax.plot([XBRK + dx - 0.22, XBRK + dx + 0.22], [-0.035, 0.035], transform=tr, color=INK,
                lw=LW["axis_break"], clip_on=False, zorder=Z["label"], solid_capstyle="butt")
XLAB_AB = S.text(fig, (A_L + B_L + AB_W) / 2 / W, XLAB1_BOT / H, "sample size $N$", "label", ha="center",
                 va="bottom")

# =====================================================================================================================
# (c): anisotropy of the learned gauge at N = inf against the theorem
# =====================================================================================================================
COV_PITCH = 1.3
C_ROWS = []      # (y, arm key). v8: the four rules covered by the theorem; no separator, no "outside" row
y = 0.0
for i, ak in enumerate(["P-GD-flow", "small-init", "WD", "TIED"]):
    C_ROWS.append((y, ak)); y += COV_PITCH
assert [ak for _, ak in C_ROWS] == DRAWN
Y_TOP, Y_BOT = -0.95, C_ROWS[-1][0] + 0.55
PT_PER_ROW = (R1_T - R1_B) * 72 / (Y_BOT - Y_TOP)
PRED_GAP_OWN_PT = 0.6
PRED_HALF = marker_half_h("v", MS["pred"], 0.0)
# critic r1: no reference line at cond = 1 (with the prediction triangles on it, it read as arrows)
PRED_ART, PRED_LINE = {}, {}
ROW_Y_C = {}
for item in AE:                                   # numbers for all five rules (P-GD: computed, not drawn)
    arm = AE[item]
    rr = cells[f"{item}|inf"]["runs"]
    q10, q50, q90 = rr["kappa_q10_50_90"]
    numbers["panel_c"][item] = dict(
        drawn_v8=item in DRAWN, y_row=None, row_label=arm[2], covered_by_theorem=arm[3], cond_q10_50_90=[q10, q50, q90],
        cond_source=f"{SUM} :: cells['{item}|inf']['runs']['kappa_q10_50_90'] (quantiles over non-diverged runs)",
        n_runs_in_quantiles=rr["n_runs"] - rr["n_diverged"],
        predicted_marker=pred_marker.get(item), predicted_q10_50_90=pred_q.get(item),
        predicted_source=pred_source.get(item),
        f4_key=f"{F4NUM} :: panel_e['{item}'].{{cond_q10_50_90, predicted_marker}}")
for yy, item in C_ROWS:
    ROW_Y_C[item] = yy
    numbers["panel_c"][item]["y_row"] = yy
    arm = AE[item]
    q10, q50, q90 = numbers["panel_c"][item]["cond_q10_50_90"]
    DATA.extend(ax_c.plot([q10, q90], [yy, yy], color=arm[4], lw=LW["range_bar"], solid_capstyle="butt",
                          zorder=Z["interval"]))
    m = mstyle(arm, "c")
    ax_c.plot([q50], [yy], ls="none", zorder=Z["marker"] - 0.1, **mark_underlay(m))
    DATA.extend(ax_c.plot([q50], [yy], ls="none", zorder=Z["marker"], **m))
    assert item in pred_marker, item
    dy = (marker_half_h(m["marker"], m["ms"], m["mew"]) + PRED_GAP_OWN_PT + PRED_HALF) / PT_PER_ROW
    PRED_ART[item] = ax_c.plot([pred_marker[item]], [yy - dy], ls="none", marker="v", ms=MS["pred"], mfc=INK,
                               mec=INK, mew=0, zorder=Z["marker"] + 0.5)[0]
    DATA.append(PRED_ART[item])
    # predicted 10-90% over runs (INK = theory, errorbar width): visible only for GD (default init.)
    PRED_LINE[item] = ax_c.plot([pred_q[item][0], pred_q[item][2]], [yy - dy, yy - dy], color=INK,
                                lw=LW["errorbar"], solid_capstyle="butt", zorder=Z["marker"] + 0.4)[0]
    DATA.append(PRED_LINE[item])
ax_c.set_yticks([])
ax_c.spines["left"].set_visible(False)
ax_c.set_ylim(Y_BOT, Y_TOP)
ax_c.set_xscale("log")
ax_c.set_xlim(0.93, 8.6)
C_XT, C_XT_MINOR = [1, 1.5, 2, 3, 5], [4, 6, 7, 8]
ax_c.set_xticks(C_XT)
ax_c.set_xticklabels([("%g" % v) for v in C_XT])
ax_c.xaxis.set_minor_locator(mticker.FixedLocator(C_XT_MINOR))
ax_c.xaxis.set_minor_formatter(mticker.NullFormatter())
# v8: "at N = infinity" (a setting) moved to the caption; the transpose is an upright bold T, because the bold math
# font has no \top glyph (it falls back to a thin, oversized Computer Modern symbol)
XLAB_C = S.text(fig, (C_L + C_W / 2) / W, XLAB1_BOT / H, "$\\mathrm{cond}(EE^{\\mathrm{T}})$", "label",
                ha="center", va="bottom")
# row labels of (c), one per row: the part of Theorem thm:gauge that predicts the row (r2: r1 centred one label on each
# pair of rows, which read as labelling the row next to it only). v8: no "outside" row.
C_ROW_LAB = {"P-GD-flow": "part 1", "small-init": "part 1", "WD": "part 3", "TIED": "part 3"}
assert set(C_ROW_LAB.values()) == set(C_ROWLAB)
# critic r1: the theorem part of each row is recorded in numbers.json (panel_c.<rule>.theorem_part), not drawn; the rows
# of (c) follow the order of the key strip, and colour and marker name the rule
for ak in C_ROW_LAB:
    numbers["panel_c"][ak]["theorem_part"] = C_ROW_LAB[ak]
assert [ak for _, ak in C_ROWS] == KEY_ORDER
# v8: no key inside (c); the caption decodes the marks (coloured bar and mark = runs, black triangle and thin line =
# the prediction of the theorem; black is the theory colour in every figure). Critic r1: no row labels.
ax_c.set_ylim(Y_BOT, Y_TOP)
ax_c.set_xlim(0.93, 8.6)

# =====================================================================================================================
# (d), (e): per-pair excess X and imbalance part I on G_s, N = 250,000
# =====================================================================================================================
NROW = len(NET)
ROW_Y = {arm: float(i) for i, (arm, _, _) in enumerate(NET)}
DE_YLIM = (NROW - 0.5, -0.5)
DE_XLIM = (-1.6, 10.3)
JIT_Y = 0.27                      # vertical jitter half-width (rows)
BAR_HALF = 0.36                   # half-height of the mean bar (rows)
rng = np.random.default_rng(20261010)
JIT = {arm: rng.uniform(-JIT_Y, JIT_Y, NPAIRS[arm]) for arm, _, _ in NET}     # same jitter in (d) and (e)
for ax, part in ((ax_d, "X"), (ax_e, "I")):
    ax.axvline(0.0, color=S.NULL_LINE, lw=LW["zero"], ls=S.LS["dashed"], zorder=Z["ref"])
    for gsep in (1.5, 3.5, 5.5):
        ax.axhline(gsep, color=S.GUIDE, lw=LW["guide"], zorder=Z["guide"])
    for arm, grp, rl in NET:
        col = GROUP[grp][1]
        v = NET_DATA[arm][part]
        yy = ROW_Y[arm] + JIT[arm]
        DATA.append(ax.scatter(v, yy, s=MS["jitter_s"], color=col, alpha=MS["jitter_alpha"], linewidths=0,
                               zorder=Z["thin"]))
        mu = float(v.mean())
        by = [ROW_Y[arm] - BAR_HALF, ROW_Y[arm] + BAR_HALF]
        S.underlay(ax, [mu, mu], by, LW["median_bar"], Z["data"], extra=LW["median_halo"], solid_capstyle="butt")
        DATA.extend(ax.plot([mu, mu], by, color=col, lw=LW["median_bar"], zorder=Z["data"], solid_capstyle="butt"))
    ax.set_ylim(*DE_YLIM)
    ax.set_xlim(*DE_XLIM)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xticklabels([S.num(v, 0) for v in (0, 2, 4, 6, 8)])
    ax.xaxis.set_minor_locator(mticker.FixedLocator([-1, 1, 3, 5, 7, 9, 10]))
assert all(DE_XLIM[0] < NET_DATA[a][p].min() and NET_DATA[a][p].max() < DE_XLIM[1] for a in NET_DATA for p in "XI")
# rule of the head as tick labels of (d); (e) shares the rows
ax_d.set_yticks([ROW_Y[a] for a, _, _ in NET])
ax_d.set_yticklabels([rl for _, _, rl in NET])
ax_d.tick_params(axis="y", length=0, pad=D_TICK_PAD * 72)
ax_e.set_yticks([])
assert set(rl for _, _, rl in NET) == set(NET_TICK)
tr_d = blended_transform_factory(fig.transFigure, ax_d.transData)
for gsep in (1.5, 3.5, 5.5):          # group separators continue through the tick-label gutter
    fig.add_artist(Line2D([GUT_L / W, D_L / W], [gsep, gsep], transform=tr_d, color=S.GUIDE, lw=LW["guide"],
                          zorder=Z["guide"]))
# initialization named directly inside (e), right of every cloud, in the colour of its rows (one line per group)
GROUP_LABEL_V7 = {"default": "default init.", "balanced": "balanced init.", "small": "small init.",
                  "wd": "weight decay"}
GL_X = DE_XLIM[1] - 0.15
GROUP_ART = {}
for grp, lab in GROUP_LABEL_V7.items():
    rows = [ROW_Y[a] for a, g, _ in NET if g == grp]
    GROUP_ART[grp] = S.text(ax_e, GL_X, float(np.mean(rows)), lab, "header", color=GROUP[grp][2], ha="right",
                            va="center", zorder=Z["label"])
    ANNOT.append(GROUP_ART[grp])
XLAB_DE = S.text(fig, (D_L + X_R) / 2 / W, XLAB2_BOT / H, "$G_s$ excess, in SD of $\\Delta(y)$ over pairs", "label",
                 ha="center", va="bottom")     # critic r1: names the quantity and its unit (was "SD of ... , on G_s")

# =====================================================================================================================
# key strip (rules of a-c) across the top, letters and titles
# =====================================================================================================================
HL, HG, CS = 0.25, 0.045, 0.16
items = [AE[k] for k in KEY_ORDER]
total = sum(HL + HG + text_w(a[1], "key") for a in items) + CS * (len(items) - 1)
x0 = (W - total) / 2
xc = x0
for arm in items:
    yy = KEY_Y
    KEY_ART.append(fig.add_artist(Line2D([xc / W, (xc + HL) / W], [yy / H, yy / H], transform=fig.transFigure,
                                         color=arm[4], lw=LW["data"], ls=arm[7])))
    KEY_ART.append(fig.add_artist(Line2D([(xc + HL / 2) / W], [yy / H], transform=fig.transFigure, ls="none",
                                         **mstyle(arm))))
    ANNOT.append(S.text(fig, (xc + HL + HG) / W, KEY_Y / H, arm[1], "key", ha="left", va="center"))
    xc += HL + HG + text_w(arm[1], "key") + CS
letter_title("a", 0.02, LET1, "$G_s$")
letter_title("b", B_L - 0.02, LET1, "projector")
letter_title("c", C_GUT_L, LET1, "anisotropy of the learned gauge")
letter_title("d", 0.02, LET2, "excess of $u$ over $y$")
letter_title("e", E_L - 0.02, LET2, "imbalance part")

# =====================================================================================================================
# v8: bold tick labels and axis titles with bold math (the last styling call before saving), then texts on a fill:
# declare the darkest fill under each text, so S.check measures contrast against it
# =====================================================================================================================
N_FINALIZED = S.finalize(fig)
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

# =====================================================================================================================
# save and check
# =====================================================================================================================
PDF, PNG = os.path.join(HERE, "F_gauge.pdf"), os.path.join(HERE, "F_gauge.png")
PNG100, VIEWS = os.path.join(HERE, "_view_print_100dpi.png"), os.path.join(HERE, "_views_cvd_grey.png")
fig.savefig(PDF)
CHK = S.check(fig, PDF)
assert abs(CHK["size_in"][0] - W) < 1e-9 and abs(CHK["size_in"][1] - H) < 1e-9, CHK
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
assert not any("Type 3" in ln for ln in pf.splitlines()), pf
# v8 rules (author 2026-10-10), asserted on top of S.check (bold ticks / axis titles, bold math, a chromatic colour):
#   every font embedded TrueType (Type 42); the axis titles drawn as figure texts are bold; data colours are the rule
#   colours of the semantic map plus neutrals; no text on the figure states a setting or a number other than ticks
assert all("TrueType" in f_ for f_ in CHK["fonts"]), CHK["fonts"]
for _t in (XLAB_AB, XLAB_C, XLAB_DE, ax_a.yaxis.label):
    assert S._is_bold(_t.get_fontweight()), _t.get_text()
    assert not S.has_math(_t.get_text()) or _t.get_fontproperties().get_math_fontfamily() == S.BOLD_MATH, _t.get_text()
ALLOWED_DATA = {S.ARM["gd"], S.ARM["balanced"], S.ARM["smallinit"], S.ARM["wd"], S.ARM["tied"], INK, S.NULL_LINE,
                S.REF_FILL, S.GUIDE, S.WHITE}
assert set(CHK["data_colours"]) <= {c.upper() for c in ALLOWED_DATA}, set(CHK["data_colours"]) - ALLOWED_DATA
assert {S.ARM["gd"], S.ARM["balanced"], S.ARM["smallinit"], S.ARM["wd"], S.ARM["tied"]} <= set(CHK["chromatic_data_colours"])
assert not CHK["off_palette_colours"], CHK["off_palette_colours"]
_TXT = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
FORBIDDEN = ["step", "0.002", "0.05", "N=", "N =", "250", "outside", "inside", "premise", "Theorem", "autoencoder",
             "CIFAR", "network", "measured", "predicted", "preregistered", "C_0", "tolerance =",
             "part 1", "part 2", "part 3", "part 4", "Part"]                 # critic r1: theorem parts off the figure
assert not [(f_, t) for t in _TXT for f_ in FORBIDDEN if f_ in t], [(f_, t) for t in _TXT for f_ in FORBIDDEN if f_ in t]
# critic r1: (c) carries no text inside its axes area or its left gutter other than the panel letter and title, and
# no vertical reference line (the cond = 1 line read as arrows with the prediction triangles)
assert not [ln for ln in ax_c.get_lines() if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
            and ln.get_ydata()[0] != ln.get_ydata()[1]], "vertical line in (c)"
assert not ax_c.get_yticklabels() or not any(t.get_text().strip() for t in ax_c.get_yticklabels())
_c_box = ax_c.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
_in_c = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()
         and t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()).y0 > _c_box.y0 + 0.01
         and t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()).y1 < _c_box.y1 - 0.01
         and t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()).x1 > B_L + AB_W + 0.01]
assert not _in_c, _in_c
S.print_view(PDF, PNG, PNG100)
S.views(PNG, VIEWS)


def text_boxes():
    out = []
    for t in fig.findobj(mtext.Text):
        if t.get_visible() and t.get_text().strip():
            bb = t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            out.append((t.get_text(), bb))
    return out


TB = text_boxes()
clash = [(a, b) for i, (a, ba) in enumerate(TB) for (b, bb) in TB[i + 1:] if ba.overlaps(bb)
         and ba.intersection(ba, bb).width > 0.005 and ba.intersection(ba, bb).height > 0.005]
outside = [(a, [round(v, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)]) for a, bb in TB
           if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]
MARGINS = {"bottom_in": round(min(bb.y0 for _, bb in TB), 4), "top_in": round(H - max(bb.y1 for _, bb in TB), 4),
           "left_in": round(min(bb.x0 for _, bb in TB), 4), "right_in": round(W - max(bb.x1 for _, bb in TB), 4)}


def _marker_half(line):
    m = line.get_marker()
    if m in (None, "None", "", " "):
        return None
    b = stroke_box(m, line.get_markersize(), line.get_markeredgewidth())
    return max(abs(b[0]), abs(b[2])) / 72, max(abs(b[1]), abs(b[3])) / 72


def data_shapes(arts):
    boxes, segs = [], []
    for a in arts:
        if isinstance(a, mcoll.PathCollection):
            r = np.sqrt(np.max(a.get_sizes())) / 2 / 72
            pts = a.get_offset_transform().transform(a.get_offsets()) / fig.dpi
            boxes += [(x - r, y_ - r, x + r, y_ + r) for x, y_ in pts]
            continue
        if isinstance(a, mcoll.LineCollection):
            lw = float(np.max(a.get_linewidth())) / 72
            for sg in a.get_segments():
                p = a.get_transform().transform(np.asarray(sg)) / fig.dpi
                segs.append((p[0], p[1], lw / 2))
            continue
        pts = a.get_transform().transform(np.column_stack(a.get_data())) / fig.dpi
        hm = _marker_half(a)
        if hm is not None:
            boxes += [(x - hm[0], y_ - hm[1], x + hm[0], y_ + hm[1]) for x, y_ in pts]
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
    tt = np.linspace(0, 1, 400)[:, None]
    pts = p0 + tt * (p1 - p0)
    dx = np.maximum.reduce([r[0] - pts[:, 0], pts[:, 0] - r[2], np.zeros(len(pts))])
    dy = np.maximum.reduce([r[1] - pts[:, 1], pts[:, 1] - r[3], np.zeros(len(pts))])
    return float((np.hypot(dx, dy) - hw).min())


fig.canvas.draw()
BOXES, SEGS = data_shapes(DATA + KEY_ART)
CLEAR = []
for t in fig.findobj(mtext.Text):
    if not (t.get_visible() and t.get_text().strip()):
        continue
    bb = t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    r = (bb.x0, bb.y0, bb.x1, bb.y1)
    g = min([rect_gap(r, q) for q in BOXES] + [seg_gap(r, sg) for sg in SEGS])
    CLEAR.append((t.get_text(), round(g, 4)))
HIT = [c for c in CLEAR if c[1] < 0]
ANNOT_CLEAR = {t.get_text(): dict(CLEAR).get(t.get_text()) for t in ANNOT if t is not None}
print("outside", outside)
print("clash", clash)
print("hit", HIT)
print("closest", sorted(CLEAR, key=lambda c: c[1])[:12])
print("margins", MARGINS)
assert not outside, outside
assert not clash, clash
assert not HIT, HIT
assert MARGINS["bottom_in"] >= BOT_MARGIN - 0.004 and MARGINS["top_in"] >= 0.0, MARGINS

# prediction triangles of (c) against the marks of the other rows
PRED_GAP = {}
for ak, art in PRED_ART.items():
    (px, py), = art.get_transform().transform(np.column_stack(art.get_data())) / fig.dpi
    tb = [v / 72 for v in stroke_box("v", MS["pred"], 0.0)]
    tri = (px + tb[0], py + tb[1], px + tb[2], py + tb[3])
    gaps = {}
    for ak2 in DRAWN:                     # v8: the four drawn rows (v7: also the P-GD row)
        line = [a for a in DATA if isinstance(a, Line2D) and a.axes is ax_c and a.get_marker() == AE[ak2][5]
                and a.get_linestyle() in ("None", "", " ")
                and abs(a.get_ydata()[0] - numbers["panel_c"][ak2]["y_row"]) < 1e-9]
        assert len(line) == 1, (ak2, line)
        (mx, my), = line[0].get_transform().transform(np.column_stack(line[0].get_data())) / fig.dpi
        mb = [v / 72 for v in stroke_box(line[0].get_marker(), line[0].get_markersize(),
                                         line[0].get_markeredgewidth())]
        gaps[ak2] = round(rect_gap(tri, (mx + mb[0], my + mb[1], mx + mb[2], my + mb[3])) * 72, 2)
    PRED_GAP[ak] = gaps
    assert min(v for k, v in gaps.items() if k != ak) >= 0.3, (ak, gaps)
    assert gaps[ak] >= 0.3, (ak, gaps)

# v7: every panel of v6 is kept, and every plotted value equals v6 (asserted against the v6 numbers.json, read-only)
V6NUM = f"{HOME}/hrh_v2_figs/v6/F_gauge/numbers.json"
v6 = json.load(open(V6NUM))
for ck in numbers["panels_a_b"]:
    for ak in numbers["panels_a_b"][ck]:
        for N, c in numbers["panels_a_b"][ck][ak].items():
            o = v6["panels_a_b"][ck][ak][N]
            for k in ("passes", "n_pairs", "decision"):
                assert o[k] == c[k], ("v6 a/b", ck, ak, N, k)
            assert o["x_log2N_plotted"] == c["x_log2N_v7"], ("v6 a/b x (v7 offsets)", ck, ak, N)
            for k in ("p_hat", "cp_lower", "cp_upper"):
                assert abs(o[k] - c[k]) <= 1e-12, ("v6 a/b", ck, ak, N, k)
for ak, c in numbers["panel_c"].items():
    o = v6["panel_c"][ak]
    assert np.allclose(o["cond_q10_50_90"], c["cond_q10_50_90"], rtol=0, atol=1e-12), ak
    assert (o["predicted_marker"] is None and c["predicted_marker"] is None) or \
        abs(o["predicted_marker"] - c["predicted_marker"]) <= 1e-12, ak
    assert (o["predicted_q10_50_90"] is None and c["predicted_q10_50_90"] is None) or \
        np.allclose(o["predicted_q10_50_90"], c["predicted_q10_50_90"], rtol=0, atol=1e-12), ak
# v8 against v7 (read-only): every value equal; the drawn x of (a, b) moved by exactly +0.3 (re-centring); P-GD not drawn
V7NUM = f"{HOME}/hrh_v2_figs/v7/F_gauge/numbers.json"
v7 = json.load(open(V7NUM))
for ck in numbers["panels_a_b"]:
    for ak in numbers["panels_a_b"][ck]:
        for N, c in numbers["panels_a_b"][ck][ak].items():
            o = v7["panels_a_b"][ck][ak][N]
            for k in ("passes", "n_pairs", "decision"):
                assert o[k] == c[k], ("v7 a/b", ck, ak, N, k)
            for k in ("p_hat", "cp_lower", "cp_upper"):
                assert abs(o[k] - c[k]) <= 1e-12, ("v7 a/b", ck, ak, N, k)
            assert o["x_log2N_plotted"] == c["x_log2N_v7"], ("v7 a/b x", ck, ak, N)
            if ak in DRAWN:
                assert abs(c["x_log2N_plotted"] - (o["x_log2N_plotted"] + 0.3)) <= 1e-12, ("v8 x", ck, ak, N)
            else:
                assert c["x_log2N_plotted"] is None and not c["drawn_v8"], (ck, ak, N)
for ak, c in numbers["panel_c"].items():
    o = v7["panel_c"][ak]
    assert np.allclose(o["cond_q10_50_90"], c["cond_q10_50_90"], rtol=0, atol=1e-12), ak
    assert (o["predicted_marker"] is None and c["predicted_marker"] is None) or \
        abs(o["predicted_marker"] - c["predicted_marker"]) <= 1e-12, ak
    assert (ak in DRAWN) == c["drawn_v8"] and ((c["y_row"] is None) == (ak not in DRAWN)), ak
    if ak in DRAWN:
        assert c["y_row"] == o["y_row"], ("v8 rows keep the v7 positions", ak)

# =====================================================================================================================
# numbers.json
# =====================================================================================================================
for arm, grp, rl in NET:
    d = NET_DATA[arm]
    cell = f"{arm}_250000"
    numbers["panels_d_e"][arm] = dict(
        row=int(ROW_Y[arm]), group=grp, row_label=rl, n_pairs=len(d["names"]),
        sigma_y=d["sigma_y"], X_mean_sigma=float(d["X"].mean()), I_mean_sigma=float(d["I"].mean()),
        B_mean_sigma=float((d["X"] - d["I"]).mean()),
        X_share_gt0=float((d["raw_X"] > 0).mean()), I_share_gt0=float((d["raw_I"] > 0).mean()),
        X_min_max=[float(d["X"].min()), float(d["X"].max())], I_min_max=[float(d["I"].min()), float(d["I"].max())],
        X_per_pair_sigma=[float(v) for v in d["X"]], I_per_pair_sigma=[float(v) for v in d["I"]],
        y_jitter_rows=[float(v) for v in JIT[arm]],
        pairs=d["names"],
        source=f"{LAD}/{{y,u,ubal}}/<pair> :: deltas.Gs; sigma_y = SD (ddof 1) of y over the {len(d['names'])} pairs",
        asserted_against=[f"{A4} :: cell {cell}, rung Gs, parts X and I: mean, mean_sigma_y, sigma_y, n, share_gt0",
                          f"{SV11} :: sigma_y['{cell}']['Gs']; values['<tag>|<A>|<B>']['Gs'] per pair"])
    o = v6["panels_d_e"][arm]
    for k in ("n_pairs", "pairs", "row", "group", "row_label"):
        assert o[k] == numbers["panels_d_e"][arm][k], ("v6 d/e", arm, k)
    for k in ("X_per_pair_sigma", "I_per_pair_sigma", "y_jitter_rows"):
        assert np.allclose(o[k], numbers["panels_d_e"][arm][k], rtol=0, atol=1e-12), ("v6 d/e", arm, k)
    o7 = v7["panels_d_e"][arm]
    for k in ("n_pairs", "pairs", "row", "group", "row_label"):
        assert o7[k] == numbers["panels_d_e"][arm][k], ("v7 d/e", arm, k)
    for k in ("X_per_pair_sigma", "I_per_pair_sigma", "y_jitter_rows"):
        assert np.allclose(o7[k], numbers["panels_d_e"][arm][k], rtol=0, atol=1e-12), ("v7 d/e", arm, k)
numbers["panels_d_e_I_block"] = {
    "I4_FLOW_250000_Gs_I": {"mean_sigma": i4["mean_sigma"], "ci_sigma_simultaneous_98.75": i4["ci_sigma"],
                            "source": f"{SV11} :: I.I4.intervals['250000|Gs|I']"},
    "I4_FLOW_250000_Gs_I_M": {"mean_sigma": sv11["I"]["I4"]["intervals"]["250000|Gs|I_M"]["mean_sigma"],
                              "ci_sigma_simultaneous_98.75": sv11["I"]["I4"]["intervals"]["250000|Gs|I_M"]["ci_sigma"],
                              "source": f"{SV11} :: I.I4.intervals['250000|Gs|I_M']"},
    "I3_tests_Gs_250000": {g: dict(diff_raw=sv11["I"]["I3"]["tests"][f"{g}|Gs|250000"]["diff"],
                                   holm_reject=sv11["I"]["I3"]["tests"][f"{g}|Gs|250000"]["holm"]["reject"],
                                   source=f"{SV11} :: I.I3.tests['{g}|Gs|250000']") for g in ("BAL", "SMALL")},
}

RESULTS_NUMBERS = {   # numbers a Results paragraph may quote about this figure (none of them is printed on it)
    "a_GD_both_steps_Gs_passes": {f"{ak}|{N}": f"{cells[f'{ak}|{N}']['primary']['Gs']['passes']}/"
                                  f"{cells[f'{ak}|{N}']['primary']['Gs']['n_pairs']}"
                                  for ak in ("P-GD-flow", "P-GD") for N in NS},
    "a_gauge_fixing_rules_Gs_passes": {f"{ak}|{N}": f"{cells[f'{ak}|{N}']['primary']['Gs']['passes']}/"
                                       f"{cells[f'{ak}|{N}']['primary']['Gs']['n_pairs']}"
                                       for ak in ("small-init", "WD", "TIED") for N in NS},
    "b_projector_decisions": {f"{ak}|{N}": cells[f"{ak}|{N}"]["primary"]["proj"]["decision"]
                              for ak in DODGE_ORDER for N in NS},
    "c_cond_q10_50_90": {ak: cells[f"{ak}|inf"]["runs"]["kappa_q10_50_90"] for ak in AE},
    "c_predicted": pred_marker,
    "c_band_cond_f_spec_C0_q10_50_90": [float(v) for v in band],
    "c_predicted_q10_50_90": pred_q,
    "d_X_mean_sigma": {a: float(NET_DATA[a]["X"].mean()) for a in NET_DATA},
    "e_I_mean_sigma": {a: float(NET_DATA[a]["I"].mean()) for a in NET_DATA},
    "e_I_share_gt0": {a: float((NET_DATA[a]["raw_I"] > 0).mean()) for a in NET_DATA},
    "e_ratio_I_FLOW_over_I_MOM": float(NET_DATA["FLOW"]["I"].mean() / NET_DATA["MOM"]["I"].mean()),
    "balanced_part_B_mean_sigma_X_minus_I": {a: float((NET_DATA[a]["X"] - NET_DATA[a]["I"]).mean()) for a in NET_DATA},
    "source": f"{SUM} :: cells[...]; {LAD} ladders (asserted against {A4} and {SV11})",
}
for k in ("a_GD_both_steps_Gs_passes", "a_gauge_fixing_rules_Gs_passes", "b_projector_decisions"):
    assert RESULTS_NUMBERS[k] == v6["numbers_for_results_text"][k], ("v6 results numbers", k)
for k in ("d_X_mean_sigma", "e_I_mean_sigma", "e_I_share_gt0"):
    for a in NET_DATA:
        assert abs(RESULTS_NUMBERS[k][a] - v6["numbers_for_results_text"][k][a]) <= 1e-12, ("v6 results numbers", k, a)
for k in ("a_GD_both_steps_Gs_passes", "a_gauge_fixing_rules_Gs_passes", "b_projector_decisions"):
    assert RESULTS_NUMBERS[k] == v7["numbers_for_results_text"][k], ("v7 results numbers", k)
for k in ("d_X_mean_sigma", "e_I_mean_sigma", "e_I_share_gt0"):
    for a in NET_DATA:
        assert abs(RESULTS_NUMBERS[k][a] - v7["numbers_for_results_text"][k][a]) <= 1e-12, ("v7 results numbers", k, a)
# v8: the sentence that carries the undrawn P-GD arm (MOVED.md) quotes these: G_s 0 of 48 / 0 of 500 at every N for
# both GD steps; projector decisions of both GD steps equal for N >= 2^12 (reliable), and differ only at 2^8
for N in NS:
    assert RESULTS_NUMBERS["a_GD_both_steps_Gs_passes"][f"P-GD|{N}"] == RESULTS_NUMBERS["a_GD_both_steps_Gs_passes"][
        f"P-GD-flow|{N}"] == ("0/500" if N == "inf" else "0/48"), N
for N in ("4096", "65536", "inf"):
    assert RESULTS_NUMBERS["b_projector_decisions"][f"P-GD|{N}"] == "reliable" == \
        RESULTS_NUMBERS["b_projector_decisions"][f"P-GD-flow|{N}"], N
assert RESULTS_NUMBERS["b_projector_decisions"]["P-GD|256"] == "inconclusive"
assert RESULTS_NUMBERS["b_projector_decisions"]["P-GD-flow|256"] == "reliable"
# MOVED.md edit 4 ("These counts hold both at the preregistered step of 0.05 and at a small step of 0.002"): the counts
# of l.1657-1660 (0 of 48 at every finite N, 0 of 500 at N = inf, on 10-NN sets, G_s and CKA) hold for both GD arms
for _ak in ("P-GD", "P-GD-flow"):
    for N in NS:
        for _ck in ("kNN", "Gs", "CKA"):
            _c = cells[f"{_ak}|{N}"]["primary"][_ck]
            assert _c["passes"] == 0 and _c["n_pairs"] == (500 if N == "inf" else 48), (_ak, N, _ck, _c["passes"])
RESULTS_NUMBERS["edit4_GD_both_steps_kNN_Gs_CKA_passes"] = {
    f"{_ak}|{N}|{_ck}": "%d/%d" % (cells[f"{_ak}|{N}"]["primary"][_ck]["passes"], cells[f"{_ak}|{N}"]["primary"][_ck]["n_pairs"])
    for _ak in ("P-GD", "P-GD-flow") for N in NS for _ck in ("kNN", "Gs", "CKA")}
RESULTS_NUMBERS["v8_not_drawn_P_GD"] = {
    "Gs_passes": {N: RESULTS_NUMBERS["a_GD_both_steps_Gs_passes"][f"P-GD|{N}"] for N in NS},
    "proj_passes_decision": {N: "%d/%d %s" % (cells[f"P-GD|{N}"]["primary"]["proj"]["passes"],
                                              cells[f"P-GD|{N}"]["primary"]["proj"]["n_pairs"],
                                              cells[f"P-GD|{N}"]["primary"]["proj"]["decision"]) for N in NS},
    "c_cond_q10_50_90": cells["P-GD|inf"]["runs"]["kappa_q10_50_90"],
    "note": "GD at the preregistered step 0.05; drawn in Figure F.4 (fig:exp-linae), not in Figure 5 (v8)"}

inputs = {p: sha(p) for p in [SUM, F4NUM, A4, SV11, STYLE_PATH, STYLE_ORIG, V6NUM, V7NUM] + NPZ_WD
          + [f"{CRIT_DIR}/{ak}_Ninf_runs.csv" for ak in ("P-GD-flow", "small-init")]}
inputs[f"{LAD}/{{y,u,ubal}}/S2.2v11_<ARM>_N250000_*.json"] = hashlib.sha256(
    "".join(sha(f"{LAD}/{t}/{n}") for a in NET_DATA for t in ("y", "u", "ubal")
            for n in NET_DATA[a]["names"]).encode()).hexdigest() + " (sha256 of the concatenated per-file sha256s)"
ALL_TEXT = sorted(set(t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()))
out = {
    "figure": {"id": "F_gauge", "version": "v8 (author feedback 2026-10-10 on v7 Figure 6: Tol palette from the "
                                           "semantic map, bold ticks and axis titles with bold math, no setting text "
                                           "on the figure, secondary comparisons not drawn; values as v7, GD at step "
                                           "0.05 not drawn; critic r1: no row labels and no cond = 1 line in (c), "
                                           "(c) widened into the freed gutter)",
               "label": "fig:exp-gauge-case", "section": "5.5 sec:exp-gauge",
               "message": "The training rule decides whether independent runs agree on the geometry of a linear head "
                          "by setting the imbalance that fixes its gauge (Theorem thm:gauge, Remark rem:optimizers).",
               "size_in": CHK["size_in"], "height_v6_in": 3.95, "height_v7_in": 2.87, "min_text_pt": CHK["min_text_pt"],
               "min_glyph_pt_from_pdf_Tf": CHK["min_glyph_pt"], "pdf_glyph_size_counts": CHK["pdf_glyph_size_counts"],
               "n_type3": CHK["n_type3"], "fonts": CHK["fonts"], "pdffonts": pf.splitlines(),
               "low_contrast_text": CHK["low_contrast_text"], "text_on_fill_declared": ON_FILL,
               "text_overlaps_gt_0.005in": clash, "text_vs_data_overlaps": HIT, "text_margins_in": MARGINS,
               "annotation_clearance_to_nearest_data_mark_in": ANNOT_CLEAR,
               "c_prediction_triangle_gap_to_marks_pt": PRED_GAP,
               "every_text_on_the_figure": ALL_TEXT,
               "layout_in": {"A_L": A_L, "AB_W": AB_W, "B_L": B_L, "C_GUT_L": C_GUT_L, "C_L": C_L, "C_W": C_W,
                             "D_L": D_L, "DE_W": DE_W, "E_L": E_L, "R1": [R1_B, R1_T], "R2": [R2_B, R2_T],
                             "gutter_row2_v6": 1.40, "gutter_row2_v8": D_L,
                             "row_labels_c_v6": 1.6, "row_labels_c_v8_r0": C_LAB_PAD + C_LABW,
                             "row_labels_c_v8_critic_r1": 0.0, "C_W_r0": C_W_R0, "col_halfwidth_ab": COL_HW,
                             "c_rows_y": ROW_Y_C, "c_ylim": [Y_BOT, Y_TOP]},
               "finalize_texts_changed": N_FINALIZED,
               "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
    "inputs": inputs,
    "style": {"path": STYLE_PATH, "original": STYLE_ORIG, "sha256": STYLE_SHA, "version": S.STYLE_VERSION,
              "check": CHK, "semantic_map_used": {k: S.SEM[k] for k in sorted(set(SEM_RULE.values()) | set(SEM_GROUP.values())
                                                                         | {"theory"})}},
    "script_sha256": sha(os.path.abspath(__file__)),
    "asserts": {
        "a_b_c": f"every plotted value of (a)-(c) equals {F4NUM} (panels_a_to_d.Gs, panels_a_to_d.proj, "
                 "panel_e.<rule>.cond_q10_50_90 and .predicted_marker, panel_e_band.cond_f_spec_C0_q10_50_90) "
                 "to 1e-12; passes, n_pairs and decisions exactly",
        "d_e": f"sigma_y equals {A4} (sigma_y) and {SV11} (sigma_y[cell].Gs) to 1e-9 relative; means of X and I in "
               "sigma_y units equal a4_parts mean_sigma_y to 1e-9 and raw means equal a4_parts mean to 1e-12; "
               "share_gt0 of I equal; every per-pair Delta_Gs of y, u and ubal equals the frozen summary's "
               "values['<tag>|<A>|<B>']['Gs'] exactly; FLOW I equals I4 intervals['250000|Gs|I'].mean_sigma; "
               "I3 raw differences BAL and SMALL at 250k equal mean I(momentum) - mean I(head flow)",
        "v7_equals_v6": f"every plotted value (a-c cells and positions, c quantiles and predictions, d-e per-pair values "
                        f"and jitter) and the results numbers equal {V6NUM} to 1e-12 (counts exactly)",
        "frozen_hashes": {"s22v11_summary.json": SV11_SHA, "a4_parts.csv": A4_SHA},
        "v8_equals_v7": f"every value equals {V7NUM} (cells, quantiles, predictions, per-pair values, jitter, results "
                        "numbers); drawn x of (a, b) = v7 x + 0.3 exactly (re-centred four rules); c rows keep the v7 y; "
                        "the P-GD series is computed and asserted, not drawn (x_log2N_plotted None, drawn_v8 False); "
                        "the v6 assert on x_log2N_plotted now reads the field x_log2N_v7",
        "v8_style": "S.check (v8: bold ticks and axis titles, bold math, a chromatic colour, Type 42, >= 7 pt, contrast); "
                    "rule colours equal S.SEM; data colours only rule colours and neutrals; no off-palette colour; "
                    "no figure text holds a setting word or number (FORBIDDEN list); the four axis titles bold with "
                    "bold math",
    },
    "encoding": {
        "key_strip": {AE[k][1]: {"colour": AE[k][4], "marker": AE[k][5] + (" open" if AE[k][6] == "open" else ""),
                                 "line": "dashed" if AE[k][7] == DASH else "solid"} for k in KEY_ORDER},
        "a_b": "fraction of disjoint run pairs within the preregistered tolerance (G_s in a, projector in b; the "
               "tolerances are not printed); x = log2 N, infinity at 20.5 after an axis break at 18.25; per-rule "
               "offsets " + json.dumps({k: round(float(v), 3) for k, v in DODGE_V8.items()}) + " (P-GD not drawn)"
               + "; one-sided 95% Clopper-Pearson bars; grey lines at 0.8 and 0.9 (values not printed); light "
                 "columns behind each N; y ticks 0, 0.5, 1",
        "c": "range bar = 10-90% over non-diverged runs, mark = median (N = inf), rule colour; INK triangle-down = "
             "median and INK line = 10-90% over runs of the per-run prediction (part 1: cond f(spec C0); part 3: weight "
             "decay from each run's sample eigenvalues, tied = 1), drawn just above the measured bar; rows in the order "
             "of the key strip, named by colour and marker only (critic r1: the r0 row labels part 1 / part 3 are "
             "recorded as panel_c.<rule>.theorem_part: part 1 for GD step 0.002 and small init., part 3 for weight "
             "decay and tied weights); no reference line at cond = 1 (critic r1); no outside row and no in-panel key "
             "in v8 (the caption decodes the marks); x log 0.93 to 8.6",
        "d_e": "one dot per pair of runs (alpha %.1f, %.1f pt^2), vertical jitter +-%.2f row (seed 20261010, the same "
               "in d and e); vertical bar = mean on a white underlay; dashed line = 0; shared x %s in SD of "
               "Delta_Gs(y) over the cell's pairs; tick labels = rule of the head; initialization named inside (e) in "
               "the group colour" % (MS["jitter_alpha"], MS["jitter_s"], JIT_Y, list(DE_XLIM)),
        "group_colours": {g: v[1] for g, v in GROUP.items()},
        "group_text_tones": {g: v[2] for g, v in GROUP.items()},
        "group_labels_v7": GROUP_LABEL_V7,
        "rule_colours_semantic_map": {k: (AE[k][4], SEM_RULE[k]) for k in DRAWN},
    },
    "panels_a_b": numbers["panels_a_b"],
    "panel_c": numbers["panel_c"],
    "panel_c_predicted_q10_50_90": pred_q,
    "panel_c_band": {"cond_f_spec_C0_q10_50_90": [float(v) for v in band], "n_runs": int(len(kp_flow)),
                     "drawn_as": "INK line on the GD step-0.002 row (r2; r1 drew a full-height grey band)",
                     "source": f"{CRIT_DIR}/P-GD-flow_Ninf_runs.csv :: eig_fC0_0..3 (non-diverged rows)",
                     "f4_key": f"{F4NUM} :: panel_e_band.cond_f_spec_C0_q10_50_90"},
    "panels_d_e": numbers["panels_d_e"],
    "panels_d_e_I_block": numbers["panels_d_e_I_block"],
    "numbers_for_results_text": RESULTS_NUMBERS,
    "on_panel_numbers": {"printed": "tick labels only (v8: the step sizes left the key with the P-GD arm)",
                         "removed_in_v8": {"key_step_sizes": [0.002, 0.05], "c_x_label_N": "infinity"},
                         "removed_in_v7": {"a_b_tolerances": {"Gs": EPS["Gs"], "proj": EPS["proj"],
                                                              "source": f"{SUM} :: meta.eps"},
                                           "b_decision_levels_annotation": [REL_LO, UNREL_HI],
                                           "d_e_N": 250000}},
}
out["output"] = {"pdf": PDF, "pdf_sha256": sha(PDF), "png": PNG, "png_dpi": 200, "png_from": "pdftoppm of the PDF",
                 "png_sha256": sha(PNG), "print_view_100dpi": PNG100, "views_normal_deutan_protan_grey": VIEWS}
json.dump(out, open(os.path.join(HERE, "numbers.json"), "w"), indent=1, default=float)
print(json.dumps({"check": {k: CHK[k] for k in ("min_text_pt", "min_glyph_pt", "n_type3", "fonts",
                                                  "low_contrast_text")},
                  "on_fill": ON_FILL, "pred_gap_pt": PRED_GAP, "annot_clear_in": ANNOT_CLEAR,
                  "layout": out["figure"]["layout_in"], "H": H, "key_total_in": total}, indent=1))
print({a: (round(NET_DATA[a]["X"].mean(), 3), round(NET_DATA[a]["I"].mean(), 3)) for a in NET_DATA})
print("OK")
