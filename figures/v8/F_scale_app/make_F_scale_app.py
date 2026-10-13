"""F_scale_app v8 (arxiv_v4 Figure F.1, fig:exp-scale, appendix F): the crossing-law slope predictions.

v8 (2026-10-10, role maker for the appendix figures F_scale_app / F_mst_app / F_aggregation_app). A copy of
~/hrh_v2_figs/v6/F_scale_app/make_F_scale_app.py (sha256 dd33ff24...d721, untouched; byte copy next to this file as
make_F_scale_app_v6_copy.py). The author on v7 Figure 6 (2026-10-10): no black-and-white figures (one recognized
scientific palette), tick labels and axis titles bold (math included), no row headers that state a setting, and no
secondary comparison drawn that the Results text states. Data, statistics and every v6 assert unchanged. Changes:
  * style hrh_style_v8 (sha256 fe994067...a8, Paul Tol schemes; asserted): pair classes seed #77AADD, vision #0077BB,
    vision-language #004488 (the light / mid / dark steps of the mKNN hue); the crossing law and N1 stay INK (theory,
    as the v8 semantic map lists for F.1); S.finalize(fig) bolds every tick label and axis title (bold math);
  * the no-change null N0 is no longer drawn: its dashed lines in (a-c), its dashed zero line and 'N0' label in (d)
    and its key entry are gone (a secondary comparison; the Results text gives N0's 4 of 29 classes and its level
    error 2.26). Every N0 number stays in numbers.json and its asserts stay;
  * outputs: PDF, then pdftoppm renders at 200 dpi (staged PNG) and 100 dpi (print view), and the normal /
    deuteranopia / protanopia / greyscale views (S.print_view, S.views); caption_v8.tex (bold claim, decoding only).
Critic-v8 pass (2026-10-10; the maker-v8 outputs are kept in _prev_maker_v8/): figure, data and asserts unchanged;
caption_v8.tex only: the sentence joining the class rule of (d) and the long N1 bars with 'and' is split into two
sentences (house style: one idea per sentence). 30 plotted values recomputed from the per-ordering cells in
_critic/recompute.py (max |diff| 5e-5, the 4-decimal rounding of numbers.json).

v6 fix round (fig-fixer-v6, 2026-10-10). The v6 maker kept the v5.1 figure, whose panels (a, b) repeat Figure 2 (a, b)
(F_neighbors) value for value (fig-critic-v6: 203 thin-line values and every class median agree to 5e-5), so F.1 showed
nothing new except its slope panel, and that panel had no stated message. Rebuilt around what Appendix F adds:

  message: Read descriptively, the unvalidated crossing law predicts the slope class of every pair at k = 10, but too
           shallow a slope for vision--language pairs, whereas the affine null N1 predicts every slope too steep.

  (a-c) mechanism, one panel per pair class: the class median of mKNN s_N at k = 10 against the evaluation size m,
        observed (thick line, class colour, as Figure 2 a) and as predicted by the crossing law (filled INK dots), the
        affine null N1 (open INK rings) and the no-change null N0 (dashed), all three computed from the pool of size
        m = 1024 (thin vertical line) and extrapolated to the other sizes.
  (d)   outcome: per-pair slope of ln s_N on ln m predicted against observed (the v5.1 panel c, enlarged), +-1 s.d. over
        the 5 nested orderings, flat bands of the slope-class rule |slope| < 0.1 on both axes, dotted y = x.

Critic findings fixed here: (1) redundancy with Figure 2 (a, b) removed; (2) flat bands darkened from REF_FILL #F2F2F2
(1.12:1) to NULL_FILL #E5E5E5 and their crossing to S.tint(NULL_LINE, 0.65) = #CDCDCD (was #E5E5E5, 1.13:1 against
the band); (3) caption names the data, defines filled / open / dashed, says the predictors use m = 1024, says the dotted
diagonal marks exact predictions (caption_v6.tex); (4) one key for the whole figure, in a row above the panels.

Read-only inputs (S1.1 under PREREG v1.1, descriptive mode D4.7; config clip = prh_clip+l2, relative depth 0.5):
  ~/hrh_v2_stage1_20261008/results/S1_1/v11/S1_1_v11_{cells,slopes,drawslopes}_clip.parquet, S1_1_v11_summary.json
  ~/hrh_v2_figs/v5/F_scale_app/numbers.json (v5.1 render: panel (c) values asserted equal to panel (d) here)
  ~/hrh_v2_figs/v5/F_neighbors/numbers.json (Figure 2 a class medians, asserted equal to the observed lines here)
Frozen style ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (imported, never edited; sha256 asserted).
Writes only next to this file: F_scale_app.pdf (vector, Type 42 fonts), F_scale_app.png (200 dpi, pdftoppm),
_view_print_100dpi.png, _views_cvd_grey.png, numbers.json.
Run (CPU, 1 thread):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 nice -n 10 \
      ~/hrh_v2_figs_venv/bin/python -B ~/hrh_v2_figs/v8/F_scale_app/make_F_scale_app.py > run.log 2>&1
"""
import os, sys, json, hashlib, datetime, subprocess, io, re

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd

STYLE_DIR = os.path.expanduser("~/hrh_v2_figs/v8/_style")
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"   # hrh_style_v8.py (frozen)
assert hashlib.sha256(open(os.path.join(STYLE_DIR, "hrh_style_v8.py"), "rb").read()).hexdigest() == STYLE_SHA
sys.path.insert(0, STYLE_DIR)
import hrh_style_v8 as S  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator, FuncFormatter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

S.use()
assert S.STYLE_SHA256 == STYLE_SHA and S.STYLE_VERSION == "v8", "shared style module changed"
LW, MSZ, Z = S.LW, S.MS, S.Z

V11 = os.path.expanduser("~/hrh_v2_stage1_20261008/results/S1_1/v11")
F_CELLS = f"{V11}/S1_1_v11_cells_clip.parquet"
F_SLOPES = f"{V11}/S1_1_v11_slopes_clip.parquet"
F_DSLOPES = f"{V11}/S1_1_v11_drawslopes_clip.parquet"
F_SUMMARY = f"{V11}/S1_1_v11_summary.json"
STYLE_FILE = os.path.join(STYLE_DIR, "hrh_style_v8.py")
V5_NUM = os.path.expanduser("~/hrh_v2_figs/v5/F_scale_app/numbers.json")
FN_NUM = os.path.expanduser("~/hrh_v2_figs/v5/F_neighbors/numbers.json")
PLAN_S11 = os.path.expanduser("~/hrh_v2_stage1_20261008/analysis/s11/ANALYSIS_PLAN_S11_v1.1.md")

CLS = ["seed-seed", "cross-vision", "cross-modal"]
CKEY = {"seed-seed": "seed", "cross-vision": "vision", "cross-modal": "vl"}
MS = [256, 512, 1024, 2048, 4096, 8192, 16384]
KS = [1, 10, 100]
M_FIT = 1024                    # plan: 'the fit size m_f = 1024'
PRED = "iii-LSlog-EB"           # crossing-law predictor, unvalidated (PREREG v1.1 D4.7)
FLAT = 0.1                      # plan V5 slope-class threshold |slope| < 0.1
MINUS = S.MINUS


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def jr(o):
    return json.loads(json.dumps(o, default=float))


def neg(s):
    return s.replace("-", MINUS)


assert sha(STYLE_FILE) == STYLE_SHA, "shared style module changed"
INS = {p: sha(p) for p in (F_CELLS, F_SLOPES, F_DSLOPES, F_SUMMARY, STYLE_FILE, V5_NUM, FN_NUM, PLAN_S11)}


def src(path, key):
    return {"source": path, "key": key, "sha256": INS[path]}


CAPTION = open(os.path.join(HERE, "caption_v8.tex")).read().strip()
MESSAGE = ("Read descriptively, the unvalidated crossing law predicts the slope class of every pair at k = 10, but too "
           "shallow a slope for vision-language pairs, whereas the affine null N1 predicts every slope too steep.")
N = {"figure": "F_scale_app", "version": "v8 (hrh_style_v8; N0 not drawn; bold ticks)", "tex_label": "fig:exp-scale",
     "paper_figure": "Figure F.1 (appendix F, arxiv_v4)", "tex_include_proposed": "figures/exp_v8/F_scale_app.pdf",
     "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "script": os.path.abspath(__file__), "script_sha256": sha(os.path.abspath(__file__)), "inputs_sha256": INS,
     "font_text": font_manager.findfont(font_manager.FontProperties(family=["serif"])), "font_math": "mathtext cm",
     "message": MESSAGE, "style_version": S.STYLE_VERSION,
     "orderings_note": "'draw' in the result files = one of 5 hashed nested orderings of the pool ('orderings' in the paper)",
     "class_names": {c: {"paper": S.PAIR_LABEL[CKEY[c]], "colour": S.PAIR[CKEY[c]], "text_colour": S.PAIR_TEXT[CKEY[c]]}
                     for c in CLS}}

# ================================================================== data
summ = json.load(open(F_SUMMARY))
C = summ["configs"]["clip"]
N["s11_setting"] = {"summary_mode": summ["mode"], "summary_G1": summ["G1"], "config_spec": C["spec"],
                    "layer": "relative depth 0.5 (PREREG_stage1.md, S1.1 models)", "orderings_per_cell": 5,
                    "pool": "COCO nested pools m = 256..16384", "fit_size": M_FIT}
cells = pd.read_parquet(F_CELLS)
slopes = pd.read_parquet(F_SLOPES)
dsl = pd.read_parquet(F_DSLOPES)
cp = cells[cells.primary].copy()
sp = slopes[slopes.primary].copy()
assert cp.pair.nunique() == 29 and (cp.n_draws == 5).all()
N["n_primary_pairs"] = {CKEY[c]: int(cp[cp.cls == c].pair.nunique()) for c in CLS}
assert N["n_primary_pairs"] == {"seed": 4, "vision": 10, "vl": 15}
o5 = json.load(open(V5_NUM))
fn = json.load(open(FN_NUM))
chk = {}

# ---- (a-c) class medians at k = 10 of observed and predicted s_N, by m
ca = cp[cp.k == 10]
assert (ca.groupby("pair").m.apply(sorted).apply(list) == pd.Series({p: MS for p in ca.pair.unique()})).all()
SER = {"obs": "obs", "crossing": PRED, "N1": "N1", "N0": "N0"}
MED = {c: {name: {int(mm): r(v) for mm, v in ca[ca.cls == c].groupby("m")[col].median().items()}
           for name, col in SER.items()} for c in CLS}
# N0 is each pair's own observed value at the fit size, held for every m (plan V3): checked per pair
for p, g in ca.groupby("pair"):
    g = g.set_index("m")
    assert np.allclose(g.N0.values, g.obs[M_FIT], rtol=0, atol=1e-12), p
assert all(bool(v) for v in ca[ca.m == M_FIT].is_fit_m) and not any(bool(v) for v in ca[ca.m != M_FIT].is_fit_m)
A = {"class_median_k10": {"values": MED, **src(F_CELLS, "median over the pairs of a class of cells[primary & k==10]."
                                                        "{obs, iii-LSlog-EB, N1, N0} by m (each cell = mean over the 5 "
                                                        "orderings); obs = thick line, crossing = filled INK dots, "
                                                        "N1 = open INK rings, N0 = dashed")},
     "fit_size_line": M_FIT,
     "definitions": {"crossing": "crossing-law predictor iii-LSlog-EB (unvalidated), from each anchor's distance rows "
                                 "at m = 1024, extrapolated to every m",
                     "N1": "affine null with Gaussian noise fitted at m = 1024 (replaced by N0 where its slope is not "
                           "positive or it is undefined; plan D4.8)",
                     "N0": "each pair's observed value at m = 1024, held for every m (checked per pair)"}}
chk["obs class medians == v5.1 F.1 panel_a.class_median_obs_k10"] = \
    jr({c: MED[c]["obs"] for c in CLS}) == o5["panel_a"]["class_median_obs_k10"]["values"]
_fnv = fn["panel_a"]["class_median_obs_k10"]["values"]          # unrounded there, keyed seed / vision / vl
chk["obs class medians == Figure 2 (F_neighbors v5) panel_a.class_median_obs_k10 (to 5e-5)"] = all(
    abs(MED[c]["obs"][mm] - _fnv[CKEY[c]][str(mm)]) <= 5e-5 for c in CLS for mm in MS)
N["panel_abc"] = A

# ---- (d) per-pair slopes at k = 10 (the v5.1 panel c, values unchanged)
s10 = sp[sp.k == 10].set_index("pair")
d10 = dsl[(dsl.primary) & (dsl.k == 10)]
assert (d10.groupby("pair").size() == 5).all()
dsd = d10.groupby("pair")[[f"slope_{PRED}", "slope_N1"]].std(ddof=1)
D = {}
D["per_pair_k10"] = {
    "values": {p: {"cls": row.cls, "slope_obs": r(row.slope_obs), "slope_obs_draw_sd": r(row.slope_obs_draw_sd),
                   "slope_crossing": r(row[f"slope_{PRED}"]), "slope_crossing_draw_sd": r(dsd.loc[p, f"slope_{PRED}"]),
                   "slope_N1": r(row.slope_N1), "slope_N1_draw_sd": r(dsd.loc[p, "slope_N1"]), "slope_N0": r(row.slope_N0)}
               for p, row in s10.iterrows()},
    "source": {"points": src(F_SLOPES, f"slopes[primary & k==10]: x = slope_obs, filled y = slope_{PRED}, open y = "
                                       "slope_N1, N0 = slope_N0 (identically 0, dashed line)"),
               "x_error": src(F_SLOPES, "slopes[primary & k==10].slope_obs_draw_sd (+-1 s.d. over the 5 orderings)"),
               "y_error": src(F_DSLOPES, f"s.d. (ddof 1) over the 5 orderings of drawslopes[primary & k==10]."
                                         f"slope_{PRED} and .slope_N1")}}
assert all(abs(q["slope_N0"]) < 1e-9 for q in D["per_pair_k10"]["values"].values())
chk["(d) per-pair values == v5.1 F.1 panel_c.per_pair_k10"] = jr(D["per_pair_k10"]["values"]) == o5["panel_c"]["per_pair_k10"]["values"]
PCC = D["per_pair_k10"]["values"]


def klass(x):
    return np.where(x >= FLAT, "rising", np.where(x <= -FLAT, "falling", "flat"))


agree = {}
for kk in KS:
    g = sp[sp.k == kk]
    for name, col in (("crossing", f"slope_{PRED}"), ("N1", "slope_N1"), ("N0", "slope_N0")):
        ok = klass(g[col].values) == g.slope_class.values
        agree[f"{name}|k={kk}"] = {CKEY[c]: f"{int(ok[g.cls.values == c].sum())}/{int((g.cls.values == c).sum())}"
                                   for c in CLS}
D["slope_class_agreement"] = {"values": agree, **src(F_SLOPES, "slopes[primary]: class of the predicted slope (+-0.1 "
                                                               "rule) == slope_class, by k and pair class")}
TOT = {name: "%d/%d" % (sum(int(v.split("/")[0]) for v in agree[f"{name}|k=10"].values()),
                        sum(int(v.split("/")[1]) for v in agree[f"{name}|k=10"].values())) for name in ("crossing", "N1", "N0")}
D["slope_class_agreement_k10_total"] = TOT
chk["slope classes at k=10: crossing 29/29, N1 25/29, N0 4/29"] = TOT == {"crossing": "29/29", "N1": "25/29", "N0": "4/29"}
chk["== v5.1 panel_c.slope_class_agreement_k10_total"] = TOT == o5["panel_c"]["slope_class_agreement_k10_total"]
nf = sp[sp.slope_class != "flat"]
D["sign_agreement_nonflat_units"] = {"values": {name: f"{int((np.sign(nf[col]) == np.sign(nf.slope_obs)).sum())}/{len(nf)}"
                                                for name, col in (("crossing", f"slope_{PRED}"), ("N1", "slope_N1"))},
                                     **src(F_SLOPES, "slopes[primary & slope_class != flat], all k")}
# the message, measured: predicted minus observed slope per pair, k = 10 (and k = 1, 100 for the text)
GAP = {}
for kk in KS:
    g = sp[sp.k == kk]
    for c in CLS:
        h = g[g.cls == c]
        dc = h[f"slope_{PRED}"] - h.slope_obs
        dn = h.slope_N1 - h.slope_obs
        GAP[f"{CKEY[c]}|k={kk}"] = {
            "crossing_minus_obs": {"min": r(dc.min(), 3), "median": r(dc.median(), 3), "max": r(dc.max(), 3),
                                   "n_shallower": int((dc > 0).sum()), "n": int(len(h))},
            "N1_minus_obs": {"min": r(dn.min(), 3), "median": r(dn.median(), 3), "max": r(dn.max(), 3),
                             "n_steeper": int((dn < 0).sum()), "n": int(len(h))},
            "median_slope": {"obs": r(h.slope_obs.median(), 3), "crossing": r(h[f"slope_{PRED}"].median(), 3),
                             "N1": r(h.slope_N1.median(), 3)}}
D["slope_gaps"] = {"values": GAP, **src(F_SLOPES, "slopes[primary & k & cls]: slope_iii-LSlog-EB - slope_obs and "
                                                  "slope_N1 - slope_obs (positive = shallower than observed)")}
assert GAP["vl|k=10"]["crossing_minus_obs"]["n_shallower"] == 15 and GAP["vl|k=10"]["crossing_minus_obs"]["min"] > 0.25
assert sum(GAP[f"{CKEY[c]}|k=10"]["N1_minus_obs"]["n_steeper"] for c in CLS) == 29
assert GAP["vl|k=10"]["median_slope"] == {"obs": -0.571, "crossing": -0.238, "N1": GAP["vl|k=10"]["median_slope"]["N1"]}
D["n1_undefined_units_set_to_N0"] = {"values": C["n1_undefined_units"],
                                     **src(F_SUMMARY, "configs.clip.n1_undefined_units (N1 replaced by N0 there)")}
D["n_pair_orderings_k10"] = int(len(d10))
lb = {p: q for p, q in PCC.items() if q["slope_N1_draw_sd"] > 0.2}
D["long_N1_bars"] = {"pairs": sorted(lb), "slope_N1_draw_sd": {p: q["slope_N1_draw_sd"] for p, q in lb.items()},
                     "max_other_slope_N1_draw_sd": r(max(q["slope_N1_draw_sd"] for p, q in PCC.items() if p not in lb))}
assert sorted({u.split("@")[0] for u in C["n1_undefined_units"]}) == sorted(lb), (lb, C["n1_undefined_units"])
acc = C["accuracy"]["primary"]
D["level_error_factor"] = {"values": {name: {CKEY[c]: r(np.exp(acc[key][c]["median_abs_log_ratio"]), 2) for c in CLS}
                                      for name, key in (("crossing", "iii-LSlog-EB (unvalidated)"), ("N0", "N0"), ("N1", "N1"))},
                           **src(F_SUMMARY, "exp of configs.clip.accuracy.primary.<pred>.<cls>.median_abs_log_ratio "
                                            "(cells with m != 1024 and obs above twice chance)")}
assert D["level_error_factor"]["values"]["crossing"]["vl"] == 2.86 and D["level_error_factor"]["values"]["N0"]["vl"] == 2.26
N["panel_d"] = D
N["asserts"] = chk
bad = [k_ for k_, v in chk.items() if not v]
assert not bad, bad

# ================================================================== drawing (inches throughout)
W, H = S.TEXTWIDTH_IN, 2.48
BOT, AXH = 0.37, 1.76
XA, WP, GP = 0.47, 0.85, 0.22  # (a-c): left edge, width, gap (the gap takes '16,384' of one panel and '256' of the next)
XS = [XA + i * (WP + GP) for i in range(3)]
XD, WD = 3.98, 1.49            # (d)
KEY_Y = H - 0.085              # centre line of the key row
LET_Y = H - 0.19               # top of the panel letters
fig = plt.figure(figsize=(W, H))
fig.set_layout_engine("none")
AX = [fig.add_axes([x / W, BOT / H, WP / W, AXH / H]) for x in XS]
for a in AX[1:]:
    a.sharey(AX[0])
    a.sharex(AX[0])
axD = fig.add_axes([XD / W, BOT / H, WD / W, AXH / H])
axK = fig.add_axes([0, (KEY_Y - 0.07) / H, 1, 0.14 / H])
axK.set_axis_off()
axK.set_xlim(0, W)
axK.set_ylim(-0.07, 0.07)
axK.set_xticks([])          # v8: S.finalize() would otherwise fill the (undrawn) tick labels of this key axes
axK.set_yticks([])

PC = {c: S.PAIR[CKEY[c]] for c in CLS}
TC = {c: S.PAIR_TEXT[CKEY[c]] for c in CLS}
NAME = {c: S.PAIR_LABEL[CKEY[c]] for c in CLS}
DASH = S.LS["dashed"]
ANN, ANN_NAME, DATA, BG = [], [], [], []
YLIM = (8e-4, 2.6)             # s_N <= 1; the space above 1 holds the class names
XLIM = (256 / 2 ** 0.35, 16384 * 2 ** 0.35)
MS_CROSS, MEW_CROSS = MSZ["dot"], MSZ["dot_edge"]                         # filled crossing law, as v5.1 (c)
MS_N1 = round(MSZ["dot"] - MSZ["dot_edge"] - MSZ["open_small_mew"], 3)   # N1 ring outer = the filled disc, as v5.1 (c)
MEW_N1 = MSZ["open_small_mew"]
PRED_LW = LW["aux"]            # thin connectors of the predicted curves (secondary data curves)
FIT_LW = LW["guide"]
ST = {"crossing": dict(color=S.INK, lw=PRED_LW, marker="o", ms=MS_CROSS, mfc=S.INK, mec="white", mew=MEW_CROSS),
      "N1": dict(color=S.INK, lw=PRED_LW, marker="o", ms=MS_N1, mfc="white", mec=S.INK, mew=MEW_N1),
      "N0": dict(color=S.NULL_LINE, lw=LW["ref"], ls=DASH),
      "fit": dict(color=S.NULL_LINE, lw=FIT_LW, ls="-")}


def add(artists):
    DATA.extend(artists if isinstance(artists, (list, tuple)) else [artists])


def note(a, name=None):
    ANN.append(a)
    ANN_NAME.append(name)
    return a


# ---- (a-c) one panel per pair class
for i, (ax, c) in enumerate(zip(AX, CLS)):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    m_ = np.array(MS, float)
    add(ax.plot([M_FIT, M_FIT], [YLIM[0], 1.0], zorder=Z["ref"], **ST["fit"]))
    # layers: the thin predictor connectors under the observed line, so the observed line stays visible where the
    # curves coincide (seed); the predictor markers on top of everything. v8: N0 (dashed) no longer drawn
    for nm_ in ("N1", "crossing"):
        yy_ = [MED[c][nm_][mm] for mm in MS]
        add(ax.plot(m_, yy_, color=S.INK, lw=PRED_LW, zorder=Z["connector"] + 0.1))
        add(ax.plot(m_, yy_, ls="none", zorder=Z["marker"] + (0.1 if nm_ == "crossing" else 0.0),
                    **{k_: v for k_, v in ST[nm_].items() if k_ not in ("color", "lw")}))
    add(ax.plot(m_, [MED[c]["obs"][mm] for mm in MS], color=PC[c], lw=LW["data_emph"], zorder=Z["data"],
                solid_capstyle="round"))
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1: "1"}.get(v, "")))
    ax.yaxis.set_minor_locator(FixedLocator([b * 10.0 ** e for e in (-3, -2, -1) for b in range(2, 10)]))   # none above 1
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator([256, 2048, 16384]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: S.fmt_int(v)))
    ax.xaxis.set_minor_locator(FixedLocator([512, 1024, 4096, 8192]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.spines["left"].set_bounds(YLIM[0], 1.0)
    ax.spines["bottom"].set_bounds(256, 16384)
    if i:
        ax.tick_params(axis="y", which="both", labelleft=False)
    # class name at the top of the panel, above s_N = 1 (seed in INK, spec 3.3)
    note(S.text(ax, np.sqrt(256 * 16384), 1.62, NAME[c], "direct", color=TC[c], ha="center", va="center",
                zorder=Z["label"]), f"({'abc'[i]}) name")
AX[0].set_ylabel("mKNN $s_N$ ($k=10$)")
AX[1].set_xlabel("evaluation size $m$")

# ---- (d) predicted against observed slopes at k = 10
ax = axD
XLD, YLD = (-0.76, 0.12), (-1.29, 0.12)
BAND_FILL = S.NULL_FILL                                 # #E5E5E5 (v5.1: REF_FILL #F2F2F2)
CROSS_FILL = S.tint(S.NULL_LINE, 0.65)                  # #CDCDCD (v5.1: NULL_FILL #E5E5E5)
BG.append(ax.axvspan(-FLAT, FLAT, facecolor=BAND_FILL, edgecolor="none", lw=0, zorder=Z["band"]))
BG.append(ax.axhspan(-FLAT, FLAT, facecolor=BAND_FILL, edgecolor="none", lw=0, zorder=Z["band"]))
BG.append(ax.add_patch(Rectangle((-FLAT, -FLAT), 2 * FLAT, 2 * FLAT, facecolor=CROSS_FILL, edgecolor="none", lw=0,
                                 zorder=Z["band"] + 0.1)))
YX = (np.array(XLD), np.array(XLD))
add(ax.plot(*YX, color=S.NULL_LINE, lw=LW["ref"], ls=S.LS["dotted"], zorder=Z["ref"]))
# v8: the dashed N0 line at slope 0 is no longer drawn (the flat band marks slope 0)
LONG = set(D["long_N1_bars"]["pairs"])
MK = {}
for c in CLS:
    col = PC[c]
    for p in sorted(pp for pp, q in PCC.items() if q["cls"] == c):
        q = PCC[p]
        e1 = ax.errorbar(q["slope_obs"], q["slope_N1"], xerr=q["slope_obs_draw_sd"], yerr=q["slope_N1_draw_sd"], fmt="o",
                         ms=MS_N1, mfc="white", mec=col, mew=MEW_N1, ecolor=col, elinewidth=LW["errorbar"],
                         zorder=Z["connector"])
        if p in LONG:   # N1 undefined in one ordering (replaced by N0): thin bar, as v5.1 revision 3
            vert = [lc for lc in e1[2] if all(abs(s[0, 0] - s[-1, 0]) < 1e-12 for s in lc.get_segments())]
            assert len(vert) == 1, p
            vert[0].set_linewidth(LW["data_thin"])
        e2 = ax.errorbar(q["slope_obs"], q["slope_crossing"], xerr=q["slope_obs_draw_sd"], yerr=q["slope_crossing_draw_sd"],
                         fmt="o", ms=MS_CROSS, mfc=col, mec="white", mew=MEW_CROSS, ecolor=col,
                         elinewidth=LW["errorbar"], zorder=Z["data"])
        for e in (e1, e2):
            add([e[0], *e[1], *e[2]])
        MK[p] = (e2[0], e1[0])
ax.set_xlim(*XLD)
ax.set_ylim(*YLD)
ax.set_xticks([-0.6, -0.4, -0.2, 0])
ax.set_yticks([-1.2, -0.8, -0.4, 0])
ax.yaxis.set_minor_locator(FixedLocator([]))
fmt = FuncFormatter(lambda v, _: neg(f"{v:.1f}") if abs(v) > 1e-9 else "0")
ax.xaxis.set_major_formatter(fmt)
ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel("observed slope ($k=10$)")
ax.set_ylabel("predicted slope")
YPT = AXH / (YLD[1] - YLD[0])          # inches per slope unit (y)
# band labels, MUTED on the band fill (declared): predicted flat at the band's left end, observed flat at its foot
note(S.on_fill(S.text(ax, XLD[0] + 0.02, -FLAT + 1.65 / 72 / YPT, "flat", "annot", ha="left", va="baseline",
                      zorder=Z["label"]), BAND_FILL), "(d) flat predicted")
note(S.on_fill(S.text(ax, 0.0, YLD[0] + 0.03, "flat", "annot", ha="center", va="bottom", zorder=Z["label"]), BAND_FILL),
     "(d) flat observed")
# v8: no 'N0' label (N0 is not drawn)


def text_width_in(s, role):
    kw = S.text_kw(role, s)
    R = fig.canvas.get_renderer()
    t = fig.text(0, 0, s, fontsize=kw["fontsize"], fontweight=kw["fontweight"])
    w = t.get_window_extent(R).width / fig.dpi
    t.remove()
    return w


# y = x label, along the diagonal in the empty space above the vision cluster
_ang = np.degrees(np.arctan((AXH / (YLD[1] - YLD[0])) / (WD / (XLD[1] - XLD[0]))))
note(S.text(ax, -0.43, -0.43 + 0.035, "$y=x$", "annot", ha="center", va="bottom", rotation=_ang,
            rotation_mode="anchor", zorder=Z["label"]), "(d) y=x")

# ---- one key for the whole figure, in a row above the panels (handles at exactly the plotted size and style)
HL = 0.20                       # handle length (in)
KEYS = [("observed", "obs"), ("crossing law", "crossing"), ("N1 (affine)", "N1"),
        ("fit size $m=1024$", "fit")]                  # v8: 'N0 (no change)' gone with N0
GAP_HT, GAP_ENT = 0.05, 0.20
wid = [HL + GAP_HT + text_width_in(lab, "key") for lab, _ in KEYS]
x0 = (W - (sum(wid) + GAP_ENT * (len(KEYS) - 1))) / 2
KEYPOS = {}
for (lab, kind), w_ in zip(KEYS, wid):
    hx = np.array([x0, x0 + HL])
    if kind == "obs":           # thick line in the three class colours, one third each (the class sets the colour)
        for j, c in enumerate(CLS):
            note(axK.plot(x0 + HL * np.array([j, j + 1]) / 3, [0, 0], color=PC[c], lw=LW["data_emph"],
                          solid_capstyle="butt")[0], f"key handle obs {CKEY[c]}")
    elif kind in ("crossing", "N1"):
        note(axK.plot(hx, [0, 0], color=S.INK, lw=PRED_LW)[0], f"key handle {kind} line")
        st = {k_: v for k_, v in ST[kind].items() if k_ not in ("color", "lw")}
        note(axK.plot([hx.mean()], [0], ls="none", **st)[0], f"key handle {kind} marker")
    elif kind == "N0":
        note(axK.plot(hx, [0, 0], **ST["N0"])[0], "key handle N0")
    else:                       # fit size: a short vertical line
        note(axK.plot([hx.mean()] * 2, [-0.045, 0.045], **ST["fit"])[0], "key handle fit")
    note(S.text(axK, x0 + HL + GAP_HT, 0, lab, "key", ha="left", va="center_baseline"), f"key {kind}")
    KEYPOS[kind] = [round(x0, 3), round(x0 + w_, 3)]
    x0 += w_ + GAP_ENT
N["key_row_in"] = {"y_centre": KEY_Y, "entries_x_in": KEYPOS}


# ---- v8: bold tick labels and axis titles (bold math) before anything is measured from the frame
N_FINALIZED = S.finalize(fig)


# ---- panel letters
def _rgba(dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=dpi)
    return np.frombuffer(buf.getvalue(), np.uint8).reshape(int(round(H * dpi)), int(round(W * dpi)), 4)


def ink_cols(artist, dpi=300):
    full = _rgba(dpi)
    sub = [q for q in artist.findobj()]
    vis = {id(q): q.get_visible() for q in sub}
    for q in sub:
        q.set_visible(False)
    without = _rgba(dpi)
    for q in sub:
        q.set_visible(vis[id(q)])
    cols = np.nonzero((np.abs(full.astype(int) - without.astype(int)).max(-1) > 40).any(0))[0]
    return int(cols.min()), int(cols.max())


XLET_D = ink_cols(axD.yaxis.label)[0] / 300
LET = [S.panel_letter(fig, "a", 0.02, LET_Y), S.panel_letter(fig, "b", XS[1] - 0.04, LET_Y),
       S.panel_letter(fig, "c", XS[2] - 0.04, LET_Y), S.panel_letter(fig, "d", XLET_D, LET_Y)]
LET_INK = {"a": [ink_cols(LET[0])[0], ink_cols(AX[0].yaxis.label)[0]],
           "d": [ink_cols(LET[3])[0], ink_cols(axD.yaxis.label)[0]]}
assert all(abs(u - v) <= 3 for u, v in LET_INK.values()), LET_INK

out_pdf, out_png = os.path.join(HERE, "F_scale_app.pdf"), os.path.join(HERE, "F_scale_app.png")
out_v100, out_views = os.path.join(HERE, "_view_print_100dpi.png"), os.path.join(HERE, "_views_cvd_grey.png")
fig.savefig(out_pdf)
pc = S.check(fig, out_pdf)                       # v8: also bold ticks / axis titles, bold math, a chromatic colour
pf = subprocess.run(["pdffonts", out_pdf], capture_output=True, text=True).stdout
N["pdf_checks"] = {**pc, "pdffonts": pf.strip().splitlines()}
assert all(("TrueType" in ln and " yes " in ln) for ln in pf.strip().splitlines()[2:]), pf
S.print_view(out_pdf, out_png, out_v100)         # v8: staged PNG = pdftoppm render of the PDF at 200 dpi
S.views(out_png, out_views)                      # normal / deuteranopia / protanopia / greyscale
import PIL.Image  # noqa: E402
N["png_px"] = list(PIL.Image.open(out_png).size)
assert abs(N["png_px"][0] - W * 200) <= 1 and abs(N["png_px"][1] - H * 200) <= 1, N["png_px"]
gl = S.layout_glyph_sizes(fig)
N["glyph_checks"] = {"min_glyph_pt_by_text": {k_: v for k_, v in sorted(gl.items(), key=lambda kv: kv[1])[:8]},
                     "min_pdf_Tf_pt": pc["min_glyph_pt"], "floor_pt": S.MIN_PT, "asserted_by": "S.check (hrh_style_v8)"}

# ================================================================== layout audit on rendered ink (300 dpi)
# As the v5.1 maker (copied from make_F_neighbors.py): annotations (texts AND key handles) against data ink, the frame
# (spines, ticks, tick labels, axis labels, panel letters) and each other, after dilating by about 0.7 pt. The flat
# bands of (d) are background.
from scipy.ndimage import binary_dilation, distance_transform_edt  # noqa: E402

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
    return np.frombuffer(buf.getvalue(), np.uint8).reshape(int(round(H * DPI_A)), int(round(W * DPI_A)), 4)[..., 3] > 40


def leaves_of(arts):
    out = []
    for a in arts:
        out.extend([q for q in a.findobj() if not q.get_children()] or [a])
    return out


ann_ids = {id(t) for t in ANN}
data_leaves = leaves_of(DATA)
data_ids = {id(a) for a in data_leaves}
bg_ids = {id(a) for a in BG}
frame = [a for a in LEAVES if id(a) not in ann_ids and id(a) not in data_ids and id(a) not in bg_ids
         and a not in (fig.patch, axD.patch, axK.patch, *[a_.patch for a_ in AX])]
M_data, M_frame = ink(data_leaves), ink(frame)
st = np.ones((7, 7), bool)
D_data, D_frame = binary_dilation(M_data, st), binary_dilation(M_frame, st)
M_ann = [ink([t]) for t in ANN]
TAG = [nm or t.get_text() for t, nm in zip(ANN, ANN_NAME)]
assert len(set(TAG)) == len(TAG), TAG
audit = {"dpi": DPI_A, "dilation_px": 3, "text_vs_data": {}, "text_vs_frame": {}, "text_vs_text": [], "where_in": {}}
for tg, Mi in zip(TAG, M_ann):
    audit["text_vs_data"][tg] = int((Mi & D_data).sum())
    audit["text_vs_frame"][tg] = int((Mi & D_frame).sum())
dist_data = distance_transform_edt(~M_data)
dist_frame = distance_transform_edt(~M_frame)
audit["clearance_pt"] = {tg: {"data": round(float(dist_data[Mi].min()) * 72 / DPI_A, 2),
                              "frame": round(float(dist_frame[Mi].min()) * 72 / DPI_A, 2)} for tg, Mi in zip(TAG, M_ann)}
# key handles of one entry touch each other by design (marker on its line; the three colour thirds of "observed")
for i in range(len(ANN)):
    Di = binary_dilation(M_ann[i], st)
    for j in range(i + 1, len(ANN)):
        n_ = int((Di & M_ann[j]).sum())
        if n_:
            audit["text_vs_text"].append([TAG[i], TAG[j], n_])
audit["text_vs_text_same_key_entry"] = [x for x in audit["text_vs_text"]
                                        if x[0].startswith("key handle") and x[1].startswith("key handle")
                                        and x[0].split()[2] == x[1].split()[2]]
audit["text_vs_text"] = [x for x in audit["text_vs_text"] if x not in audit["text_vs_text_same_key_entry"]]
fig.canvas.draw()
R = fig.canvas.get_renderer()
ftxt = [a for a in frame if isinstance(a, matplotlib.text.Text) and a.get_visible() and a.get_text().strip()]
audit["frame_text_box_overlaps"] = [[a.get_text(), b.get_text()] for i, a in enumerate(ftxt) for b in ftxt[i + 1:]
                                    if a.get_window_extent(R).overlaps(b.get_window_extent(R))]
fb = fig.bbox
atxt = [t for t in ANN if isinstance(t, matplotlib.text.Text)]
audit["text_outside_figure"] = [[t.get_text(), [round(v / fig.dpi, 3) for v in t.get_window_extent(R).extents]]
                                for t in atxt + ftxt
                                if (lambda b: b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5
                                    or b.y1 > fb.y1 + 0.5)(t.get_window_extent(R))]
# tick labels of neighbouring (a-c) panels: least horizontal gap between '16,384' of one and '256' of the next
xt = {i: [t for t in AX[i].get_xticklabels() if t.get_visible() and t.get_text()] for i in range(3)}
audit["tick_label_gap_between_panels_in"] = {
    f"{'abc'[i]}-{'abc'[i + 1]}": round((min(t.get_window_extent(R).x0 for t in xt[i + 1])
                                         - max(t.get_window_extent(R).x1 for t in xt[i])) / fig.dpi, 3) for i in range(2)}
# (c) x tick labels against every frame text of (d) (y label, tick labels): least 2-D gap between their boxes
_dtx = [a for a in ftxt if getattr(a, "axes", None) is axD or a is axD.yaxis.label]


def _box_gap(b1, b2):
    dx = max(b2.x0 - b1.x1, b1.x0 - b2.x1, 0.0)
    dy = max(b2.y0 - b1.y1, b1.y0 - b2.y1, 0.0)
    return float(np.hypot(dx, dy)) / fig.dpi


audit["gap_c_ticklabels_to_d_frame_text_in"] = round(min(_box_gap(t.get_window_extent(R), u.get_window_extent(R))
                                                         for t in xt[2] for u in _dtx), 3)
audit["n_annotations"] = len(ANN)
M_bg = ink(BG)
audit["undeclared_on_band_px"] = {tg: int((Mi & M_bg).sum()) for t, tg, Mi in zip(ANN, TAG, M_ann)
                                  if getattr(t, "_hrh_bg", None) is None and getattr(t, "axes", None) is axD}


def colour_px(artist):
    keep = {id(q) for q in artist.findobj()}
    for a in LEAVES:
        a.set_visible(VIS0[id(a)] and id(a) in keep)
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=DPI_A, transparent=True)
    for a in LEAVES:
        a.set_visible(VIS0[id(a)])
    P = np.frombuffer(buf.getvalue(), np.uint8).reshape(int(round(H * DPI_A)), int(round(W * DPI_A)), 4)
    return int(((P[..., 3] > 128) & (P[..., :3].min(-1) < 200)).sum())


audit["marker_colour_ink_px"] = {}
for c in CLS:
    p0 = sorted(pp for pp, q in PCC.items() if q["cls"] == c)[0]
    audit["marker_colour_ink_px"][CKEY[c]] = {"pair": p0, "crossing_filled": colour_px(MK[p0][0]), "N1_ring": colour_px(MK[p0][1])}
audit["panel_letter_ink_px"] = {k_: {"letter": u, "leftmost_element": v} for k_, (u, v) in LET_INK.items()}
# contrast of the (d) shading (was 1.12:1 band on white and 1.13:1 square on band)
audit["shading_contrast"] = {"band_on_white": round(S.contrast(BAND_FILL), 2),
                             "square_on_band": round(S.contrast(CROSS_FILL, BAND_FILL), 2),
                             "square_on_white": round(S.contrast(CROSS_FILL), 2),
                             "MUTED_on_band": round(S.contrast(S.MUTED, BAND_FILL), 2),
                             "seed_marker_on_square": round(S.contrast(S.PAIR["seed"], CROSS_FILL), 2),
                             "colours": {"band": BAND_FILL, "square": CROSS_FILL}}
bad = {k_: v for k_, v in {**{"data:" + k_: v for k_, v in audit["text_vs_data"].items()},
                           **{"frame:" + k_: v for k_, v in audit["text_vs_frame"].items()},
                           **{"band:" + k_: v for k_, v in audit["undeclared_on_band_px"].items()}}.items() if v}
bad.update({"ring_heavier:" + k_: v for k_, v in audit["marker_colour_ink_px"].items() if v["N1_ring"] >= v["crossing_filled"]})
bad.update({"tick_gap:" + k_: v for k_, v in audit["tick_label_gap_between_panels_in"].items() if v < 0.04})
if audit["gap_c_ticklabels_to_d_frame_text_in"] < 0.04:
    bad["tick_gap:c-d"] = audit["gap_c_ticklabels_to_d_frame_text_in"]
for tg, Mi in zip(TAG, M_ann):
    if any(k_.endswith(tg) for k_ in bad):
        audit["where_in"][tg] = (lambda yy, xx: None if not len(xx) else
                                 [round(float(xx.min()) / DPI_A, 3), round(float(H - yy.max() / DPI_A), 3),
                                  round(float(xx.max()) / DPI_A, 3), round(float(H - yy.min() / DPI_A), 3)])(
            *np.nonzero(Mi & (D_data | D_frame)))
audit["result"] = "clean" if not bad and not audit["text_vs_text"] and not audit["frame_text_box_overlaps"] \
    and not audit["text_outside_figure"] else {"collisions": bad, "text_vs_text": audit["text_vs_text"]}
N["layout_audit"] = audit
N["axes_in"] = {"a": [XS[0], BOT, WP, AXH], "b": [XS[1], BOT, WP, AXH], "c": [XS[2], BOT, WP, AXH],
                "d": [XD, BOT, WD, AXH], "figure": [W, H]}
N["style"] = {"style": "hrh_style_v8", "style_sha256": STYLE_SHA, "pairs": {CKEY[c]: PC[c] for c in CLS},
              "text_pairs": {CKEY[c]: TC[c] for c in CLS}, "observed_lw_pt": LW["data_emph"],
              "predicted_connector_lw_pt": PRED_LW, "crossing_ms_pt": MS_CROSS, "crossing_white_edge_pt": MEW_CROSS,
              "N1_ring_ms_pt": MS_N1, "N1_ring_edge_pt": MEW_N1, "N0": "not drawn in v8 (v6: NULL_LINE dashed LW ref)",
              "fit_line": "NULL_LINE solid LW guide %.1f" % FIT_LW, "band_fill": BAND_FILL, "square_fill": CROSS_FILL,
              "y_equals_x": "dotted NULL_LINE LW ref", "long_N1_bar_lw_pt": LW["data_thin"],
              "panel_letter_x_in": {"a": 0.02, "b": round(XS[1] - 0.04, 4), "c": round(XS[2] - 0.04, 4),
                                    "d": round(XLET_D, 4)}}

# ================================================================== numbers for the Results text (none in the caption)
plan_txt = open(PLAN_S11).read().splitlines()
_mf = [i + 1 for i, ln in enumerate(plan_txt) if "m_f = 1024" in ln]
_fl = [i + 1 for i, ln in enumerate(plan_txt) if "| flat |" in ln and "0.1" in ln]
assert _mf and _fl, (_mf, _fl)
G10 = {CKEY[c]: GAP[f"{CKEY[c]}|k=10"] for c in CLS}
RES = [
    {"id": "T1", "quantity": "fit size of the three predictors", "kind": "design", "value": M_FIT, "printed": "m = 1024",
     "source": PLAN_S11, "key": "line %d" % _mf[0], "sha256": INS[PLAN_S11], "where": "caption (design constant) and text"},
    {"id": "T2", "quantity": "flat slope class", "kind": "design", "value": FLAT, "printed": "|slope| below 0.1",
     "source": PLAN_S11, "key": "V5 table, line %d" % _fl[0], "sha256": INS[PLAN_S11], "where": "text (already stated)"},
    {"id": "T3", "quantity": "slope classes recovered at k = 10", "kind": "result", "value": TOT,
     "printed": "crossing law 29 of 29, N1 25, N0 4", **src(F_SLOPES, "panel_d.slope_class_agreement_k10_total"),
     "where": "text (already stated)"},
    {"id": "T4", "quantity": "slope classes recovered by k and class", "kind": "result", "value": agree,
     **src(F_SLOPES, "panel_d.slope_class_agreement"), "where": "optional"},
    {"id": "T5", "quantity": "slope signs right on non-flat pair-k combinations", "kind": "result",
     "value": D["sign_agreement_nonflat_units"]["values"], "printed": "69 of 69 for both",
     **src(F_SLOPES, "panel_d.sign_agreement_nonflat_units"), "where": "text (already stated)"},
    {"id": "T6", "quantity": "crossing-law minus observed slope at k = 10, per class (min, max)", "kind": "result",
     "value": {k_: v["crossing_minus_obs"] for k_, v in G10.items()},
     "printed": "seed %s to %s; vision %s to %s; vision-language %s to %s (shallower for all 15)" % tuple(
         neg("%.2f" % G10[k_]["crossing_minus_obs"][e]) for k_ in ("seed", "vision", "vl") for e in ("min", "max")),
     **src(F_SLOPES, "panel_d.slope_gaps.<cls>|k=10.crossing_minus_obs"), "where": "add"},
    {"id": "T7", "quantity": "class median slopes at k = 10: observed, crossing law, N1", "kind": "result",
     "value": {k_: v["median_slope"] for k_, v in G10.items()},
     "printed": "observed %s / %s / %s; crossing law %s / %s / %s; N1 %s / %s / %s" % tuple(
         neg("%.2f" % G10[k_]["median_slope"][s_]) for s_ in ("obs", "crossing", "N1") for k_ in ("seed", "vision", "vl")),
     **src(F_SLOPES, "panel_d.slope_gaps.<cls>|k=10.median_slope"), "where": "vision-language pair already stated; add N1"},
    {"id": "T8", "quantity": "N1 steeper than observed at k = 10", "kind": "result",
     "value": {k_: f'{v["N1_minus_obs"]["n_steeper"]}/{v["N1_minus_obs"]["n"]}' for k_, v in G10.items()},
     "printed": "all 29 pairs", **src(F_SLOPES, "panel_d.slope_gaps.<cls>|k=10.N1_minus_obs.n_steeper"), "where": "add"},
    {"id": "T9", "quantity": "same comparisons at k = 1 and 100", "kind": "result",
     "value": {k_: v for k_, v in GAP.items() if not k_.endswith("k=10")},
     "note": "at k = 1, N1 is steeper for only 5 of 15 vision-language pairs, so the 'every slope too steep' reading "
             "is stated at k = 10", **src(F_SLOPES, "panel_d.slope_gaps"), "where": "caveat for the text"},
    {"id": "T10", "quantity": "vision-language class medians at m = 1024 and 16,384: observed, crossing law, N1",
     "kind": "result", "value": {s_: {mm: MED["cross-modal"][s_][mm] for mm in (1024, 16384)} for s_ in ("obs", "crossing", "N1")},
     **src(F_CELLS, "panel_abc.class_median_k10.values.cross-modal"),
     "note": "the crossing-law curve already sits above the observed one at the fit size (factor %.1f), and the gap "
             "widens to a factor %.1f at m = 16,384" % (MED["cross-modal"]["crossing"][1024] / MED["cross-modal"]["obs"][1024],
                                                       MED["cross-modal"]["crossing"][16384] / MED["cross-modal"]["obs"][16384]),
     "where": "optional"},
    {"id": "T11", "quantity": "median level error factor (cells m != 1024, obs above twice chance)", "kind": "result",
     "value": D["level_error_factor"]["values"], "printed": "vision-language: crossing law 2.86, N0 2.26",
     **src(F_SUMMARY, "panel_d.level_error_factor"), "where": "text (already stated, tab:app-s11-accuracy)"},
    {"id": "T12", "quantity": "pair-orderings with N1 undefined, set to N0 (the two long thin N1 bars in d)",
     "kind": "result", "value": {"n": len(C["n1_undefined_units"]), "of": D["n_pair_orderings_k10"],
                                 "units": C["n1_undefined_units"]}, "printed": "2 of 145",
     **src(F_SUMMARY, "configs.clip.n1_undefined_units"), "where": "add"},
    {"id": "T13", "quantity": "status", "kind": "status", "value": summ["G1"], "printed": "G1 not run; unvalidated; descriptive",
     **src(F_SUMMARY, "G1"), "where": "caption (status words) and text"},
]
N["results_numbers"] = RES
N["caption_proposed_tex"] = CAPTION
_cw = re.sub(r"\\(?:ref|label)\{[^}]*\}", "1", re.sub(r"\$[^$]*\$", "x", CAPTION))
N["caption_word_count"] = len(re.sub(r"[{}~]", " ", re.sub(r"\\[a-zA-Z]+", " ", _cw)).split())
N["caption_digits_outside_math_and_refs"] = re.findall(r"(?<![\\a-zA-Z{])\d+(?:\.\d+)?",
                                                       re.sub(r"\\(?:ref|label)\{[^}]*\}|\$[^$]*\$", "", CAPTION))
N["caption_math"] = re.findall(r"\$[^$]*\$", CAPTION)
N["on_panel_numbers"] = [{"panel": "key", "text": "fit size m = 1024", "kind": "design constant"},
                         {"panel": "a, d", "text": "k = 10 (axis labels)", "kind": "design constant"}]


# ================================================================== v8: caption rules and v8 asserts
def _brace_arg(tex, i):
    """argument of the brace group opening at tex[i] == '{' (nested braces allowed)."""
    assert tex[i] == "{"
    depth = 0
    for j in range(i, len(tex)):
        depth += {"{": 1, "}": -1}.get(tex[j], 0)
        if depth == 0:
            return tex[i + 1:j], j
    raise AssertionError("unbalanced braces")


def caption_v8_check(tex, max_words, max_bold_words=26, allowed_digits=()):
    """v8 caption rules: bold first sentence (claim, <= about 25 words, no digits but names such as N1), whole caption
    <= max_words prose tokens, no digits outside math and \\ref, house style section 1 scans (no em-dash, no colon
    outside math and refs, no sentence ending on 'it', no sentence opening on a negation)."""
    assert tex.startswith("\\caption{\\textbf{"), tex[:30]
    body_all, _ = _brace_arg(tex, len("\\caption"))
    bold, j = _brace_arg(body_all, len("\\textbf"))
    rest = body_all[j + 1:]

    def plain(t):
        t = re.sub(r"[A-Za-z]+s?~\\(?:ref|eqref)\{[^}]*\}", "REF", t)     # 'Theorem~\ref{x}' counts as one word
        t = re.sub(r"\\(?:ref|eqref|label|cite[pt]?)\{[^}]*\}", "REF", t)
        t = re.sub(r"\$[^$]*\$", "x", t)
        t = re.sub(r"\\[a-zA-Z]+", " ", t)
        return re.sub(r"[{}~]", " ", t)

    def words(t):
        return len([w for w in plain(t).split() if re.search(r"[A-Za-z0-9]", w)])
    no_refs = re.sub(r"\\(?:ref|eqref|label|cite[pt]?)\{[^}]*\}", "", body_all)
    no_math = re.sub(r"\$[^$]*\$", "", no_refs)
    bold_claim = re.sub(r"\s*\([^()]*\\ref\{[^}]*\}[^()]*\)", "", bold)   # the claim without its citation
    out = {"bold_sentence": bold, "bold_words": words(bold_claim), "caption_words": words(body_all),
           "max_words": max_words, "max_bold_words": max_bold_words,
           "digits_in_bold_outside_names_and_refs": re.findall(r"(?<![\\A-Za-z{])\d+(?:\.\d+)?",
                                                               re.sub(r"\$[^$]*\$", "", re.sub(
                                                                   r"\\(?:ref|eqref)\{[^}]*\}", "", bold))),
           "digits_outside_math_and_refs": re.findall(r"(?<![\\A-Za-z{])\d+(?:\.\d+)?", no_math),
           "em_dash": ("---" in body_all) or ("\u2014" in body_all),
           "colon_outside_math_and_refs": ":" in no_math,
           "sentence_ending_on_it": re.findall(r"\bit[.?!]", body_all),
           "sentence_opening_on_negation": re.findall(r"(?:^|[.?!]\s+)(No|Not|None|Nothing|Never|Neither|Zero)\b",
                                                      plain(rest))}
    assert out["bold_words"] <= max_bold_words, out
    assert out["caption_words"] <= max_words, out
    assert not out["digits_in_bold_outside_names_and_refs"], out
    out["allowed_digits"] = list(allowed_digits)
    assert all(d_ in allowed_digits for d_ in out["digits_outside_math_and_refs"]), out
    assert not out["em_dash"] and not out["colon_outside_math_and_refs"], out
    assert not out["sentence_ending_on_it"] and not out["sentence_opening_on_negation"], out
    return out


CAP_FILE = os.path.join(HERE, "caption_v8.tex")
N["caption_v8"] = {"tex": CAPTION, "sha256": sha(CAP_FILE), "checks": caption_v8_check(CAPTION, max_words=100)}
fig.canvas.draw()
_lines = [ln for a_ in (*AX, axD) for ln in a_.get_lines()]
_texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_visible()]
CHROM_OK = {S.PAIR[k_].upper() for k_ in ("seed", "vision", "vl")}
V8 = {
    "style hrh_style_v8 sha256 asserted": S.STYLE_SHA256 == STYLE_SHA,
    "N0 not drawn: no dashed line in (a-d)": not any(ln.get_linestyle() in ("--", "dashed") for ln in _lines),
    "N0 not drawn: no 'N0' text and no N0 key entry": not any("N0" in t for t in _texts),
    "N0 numbers kept (class medians, slope classes 4/29, level error 2.26)": (
        all(len(MED[c]["N0"]) == 7 for c in CLS) and TOT["N0"] == "4/29"
        and D["level_error_factor"]["values"]["N0"]["vl"] == 2.26),
    "tick labels and axis titles bold (S.check)": not pc["not_bold_tick_or_axis_labels"] and pc["n_tick_labels_checked"] > 0,
    "bold math in bold texts (S.check)": not pc["bold_text_with_regular_math"],
    "chromatic data colours are the three pair classes": set(pc["chromatic_data_colours"]) == CHROM_OK,
    "crossing law and N1 drawn in INK (theory)": ST["crossing"]["mfc"] == S.INK and ST["N1"]["mec"] == S.INK,
    "min glyph >= 7 pt": pc["min_glyph_pt"] >= 7.0,
    "Type 42 only": pc["n_type3"] == 0,
    "class names are the only panel titles (no setting header)": sorted(
        t.get_text() for a_ in AX for t in a_.texts) == sorted(NAME.values()),
}
N["asserts_v8"] = V8
assert all(V8.values()), {k_: v for k_, v in V8.items() if not v}
N["finalize_texts_changed"] = N_FINALIZED
N["views"] = {"print_100dpi": out_v100, "cvd_grey": out_views}
N["outputs"] = {out_pdf: sha(out_pdf), out_png: sha(out_png), out_v100: sha(out_v100), out_views: sha(out_views)}
json.dump(N, open(os.path.join(HERE, "numbers.json"), "w"), indent=1, default=float)
print(json.dumps({"pdf_checks": {k_: N["pdf_checks"][k_] for k_ in ("min_text_pt", "min_glyph_pt", "n_type3", "fonts",
                                                                     "low_contrast_text", "size_in")},
                  "glyph_checks": N["glyph_checks"], "asserts": chk,
                  "layout_audit": {k_: v for k_, v in audit.items() if k_ not in ("text_vs_data", "text_vs_frame")},
                  "k10_totals": TOT, "gaps_k10": G10, "caption_words": N["caption_word_count"],
                  "caption_digits": N["caption_digits_outside_math_and_refs"], "T10": RES[9]["note"],
                  "v8": {"asserts_v8": V8, "caption_v8": N["caption_v8"]["checks"],
                         "chromatic": pc["chromatic_data_colours"], "off_palette": pc["off_palette_colours"],
                         "fonts": pc["fonts"]}},
                 indent=1, default=float))
assert audit["result"] == "clean", audit["result"]
