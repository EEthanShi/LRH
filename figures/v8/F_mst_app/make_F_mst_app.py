"""F_mst_app (arxiv_v4 Figure F.2, fig:exp-mst, appendix F): spanning-tree overlap mostly reads the
one-nearest-neighbor scale (Proposition prop:mst).

v8 (2026-10-10, role maker for the appendix figures F_scale_app / F_mst_app / F_aggregation_app). A copy of
~/hrh_v2_figs/v6/F_mst_app/make_F_mst_app.py (sha256 36ab581e...c8b2, untouched; byte copy next to this file as
make_F_mst_app_v6_copy.py). The author on v7 Figure 6 (2026-10-10): one recognized scientific palette, tick labels
and axis titles bold (math included), little text on the figure, no result numbers on it (they live in the Results
text). Data, statistics and every v6 assert unchanged. Changes:
  * style hrh_style_v8 (sha256 fe994067...a8; asserted): MST hue Tol vibrant teal #009988, its text tone #008575;
    bound INK, null and chance NULL_LINE / NULL_FILL; S.finalize(fig) bolds every tick label and axis title;
  * (b) the x label 'depth (136 pairs each)' is gone (the category names say the depth; the pair count is in the
    Spanning Trees paragraph and Table tab:app-pe2);
  * (c) 'Tr = 0.57', 'Tr = -0.16' and 'power-limited: p >= 1/120' are gone (the Spanning Trees paragraph states all
    three); the depth names stay, placed by the same search as before;
  * outputs: PDF, pdftoppm renders at 200 dpi (staged PNG) and 100 dpi (print view), the normal / deuteranopia /
    protanopia / greyscale views; caption_v8.tex.
  Panel (a) repeats Figure 2 (c) (the same 272 points); it stays, so that F.2 reads alone (outcome in a, mechanism in
  b, the global part in c) and the Results sentence that cites the chance cross of (a) keeps its target.
v8 c (2026-10-10, role consistency checker across all v8 figures; the critic-v8 files are in ./_prev_consistency/):
  * (b) the median bars (and their key handle) are a dark step of the MST hue, MST_MED = the Tol vibrant teal with
    CIELAB a*, b* kept and L* lowered to 8:1 on white (#005B4D, the method of the module's text tones; cvdlib
    text_tone), instead of INK. Black means theory in every v8 figure (here the bound of (a)), and every other figure
    draws its medians and means in the series colour (F.2 b, Figure 5 d-e, F.6 c, F.7 a-b). The bars stay under the
    swarm, so no dot is hidden; dE2000 of MST_MED to the MST dots 22.0 / 20.6 / 23.1 (normal / deutan / protan), to
    INK 25.7 / 13.9 / 17.1, to the null line 23.9 / 12.8 / 11.2. The key text 'median' stays bold INK. Data, caption
    and size unchanged.

v6 fixer round (2026-10-10, role fig-fixer-v6; the maker's v6 script and outputs are kept in _prev_r2/, the maker's
rejected first layout in _prev_r1/). Critic findings fixed on the panel; data, statistics and (a), (b) unchanged:
  * (c) the note under 'Tr = 0.57' read 'smallest p = 1/120' and could be taken for the observed p (0.058). It now
    states a floor, 'power-limited: / p >= 1/120', and moves with the last-layer name and Tr into the empty upper left
    of (c), left-aligned with 'null 95%' and set as low as 2.5 pt clearances allow (v3 F2 (c) put its statistics
    there). The y range of (c) is back to v5's 0.06, so the left spine ends at the top tick as in (a) and (b);
  * (c) the mid-depth block sat nearer the last-layer interval at 7.1B than its own end point; it is now set just
    above its own curve, at the rightmost anchor where its clearance to the last-layer data is at least twice its
    clearance to its own data;
  * numbers.json: tree_forest_fraction_by_config (n_forest_A/(m-1) by pool, preprocessing and depth, for the appendix
    wording '84 to 85%'), caption_proposed (no result numbers), on-panel note recorded as a floor.
The caption fixes (F_A and bridges glossed, 'log scale', 'median per-pair 95% permutation quantile', pointer to the
Spanning Trees paragraph) are in caption_proposed. The v6 maker's notes follow.

v6 render (2026-10-10, role fig-maker-v6; plan build/figs_v6_staging/_design/PLAN.md section 4, row F.2). A copy of
~/hrh_v2_figs/v5/F_mst_app/make_F_mst_app.py (sha256 430deacd...b3ad, left untouched). The author's rule of
2026-10-10: captions carry no result numbers, so the two things v3 F2 showed ON the panel and v5 had moved into the
caption go back onto the panel. Data, statistics and every v5 drawing choice not listed here are unchanged.
  * style module hrh_style_v5_1 (differs from v5 only in the training-rule colours ARM / ARM_TEXT, unused here);
  * (b) x label 'depth (136 pairs each)' (v3 F2 b), set on the row of the x labels of (a) and (c); the median / null
    key, which v5 had put on that row, moves into the empty band under the preregistered level (no data below 0.76);
  * (c) 'Tr = 0.57' under the 'last layer' label and 'Tr = -0.16' under the 'mid depth' label (bold, MST text tone),
    and under 'Tr = 0.57' a MUTED 7.5 pt note 'power-limited: / smallest p = 1/120' (v3 F2 c), all right-aligned at the
    last points and placed from measured text extents with 2 pt clearances asserted; the y range is raised (ticks
    unchanged) only as far as the labels need it, recorded in numbers.json;
  * every on-panel number is asserted equal to the v5 numbers.json (panel_c) and to S1_2_report.md section 3 / 5;
  * numbers.json: 'on_panel_numbers' (each with source path and key) and 'results_text_numbers' (the numbers the
    caption no longer carries, for the Results paragraph); 'caption_numbers' is empty by rule.
Everything below this paragraph is the v5 history.

v5 render (2026-10-10): a copy of ~/hrh_v2_figs/v4/F_mst_app/make_F_mst_app.py (left untouched) switched to the frozen
v5 style (~/hrh_v2_figs/v5/_style/hrh_style_v5.py) and the per-figure notes of STYLE_V5_SPEC.md section 5 (F.2). The
data loader and every statistic are unchanged (still asserted equal to ~/hrh_v2_figs/F2/numbers.json); only the drawing
changed. What changed against v4:
  * every stroke, marker size, text size and weight comes from the style module (LW, MS, FS/FW via S.text);
  * (a) points 3.6 pt (open edge 0.9, filled white edge 0.4); bound 1.6 pt INK, 'bound' bold; y = x dotted 1.0 pt;
    'last layer' / 'mid depth' bold in the MST text tone #098763; leaders 0.7 pt;
  * (b) beeswarm dots 2.6 pt across (open edge 0.7, filled white edge 0.3) instead of 2.2 pt, still with no overlap
    (asserted); the spec's 3.0 pt cannot be drawn without overlap at this width (see SWARM_NOTE). To make room the
    'median' / 'null' labels moved from the right of the panel into a key under the category names (the strip that the
    x labels of (a) and (c) use), the data area widened from 1.11 to 1.41 in, and the y range starts at 0.65
    instead of 0.6; median bar 2.2 pt INK on a white underlay (+1.0), null dashed 1.0 pt, level 1.0 pt;
  * (c) lines 1.5 pt, markers 4.2 pt (filled white edge 0.6, open edge 0.9), error bars 1.0 pt, zero line 0.8 pt;
    labels bold #098763;
  * panel letters 10 pt bold; S.check() asserts glyph sizes from the PDF, fonts and text contrast.
v5 round 2 (critic of round 1; the round-1 outputs are kept in _prev_v5r1/):
  * (b) the median and null bars are drawn UNDER the swarm (z below Z['interval']) with no white underlay, so no dot
    is hidden or cut (round 1: the 3.2 pt white underlay of the median bar hid the 7 dots tied at the median in each
    depth and cut 13 / 19 more). Each median bar runs 3 pt past the outer edge of the outermost dot that touches it,
    on both sides, so it reads beside the swarm; the null bars keep their round-1 length (they already reach far
    past the narrow swarm at the null height). Clearances are asserted (axes edges, the other swarm, the other bars);
  * (a) x label $s_{N{,}1}$: mathtext puts a space after a bare comma ('N, 1'); the braced comma prints 'N,1' as
    LaTeX does in script style;
  * (a) the filled-marker edge and the mid-depth leader colour are switches (FM_FILL_EDGE = white | self | halo,
    FM_LEADER = ink | tone; defaults halo / ink) so that F.2 (a) can be set to whatever Figure 2 (c) (F_neighbors,
    the same 272 points) ends up using; FM_TRIAL=<name> writes to _trials/<name>. The default 'halo' draws the filled
    cloud in two passes (all white 4.0 pt discs, then all green 3.2 pt discs): a lone point looks exactly like the
    spec's filled 3.6 pt marker with a white 0.4 edge, but stacked points no longer show white arcs inside the cloud
    (with per-marker white edges the dense last-layer clusters near s_N,1 = 0.035-0.05 read as open rings, i.e. as
    mid-depth points; with edges in the fill colour, as F_neighbors round 1 draws, the cloud fuses into one mass of
    4.0 pt discs). Leader INK as spec section 4.2. Trials: _trials/fig2c_r1_match (self / tone), _trials/edge_white.
v5 consistency pass (fig-consistency-v5, 2026-10-10; round-2 maker and outputs kept in _prev_v5r2/):
  * (a) FM_FILL_EDGE default halo -> self, the edge Figure 2 (c) draws for the same 272 points (its critic accepted
    it; the halo pass left the filled points 3.2 pt green inside 4.0 pt white, lighter than the 4.5 pt open rings and
    than Figure 2 (c)). The leader stays INK; Figure 2 (c) now uses INK too. Trial: _trials/cons_self.
Read-only inputs:
  ~/hrh_v2_stage1_20261008/results/S1_2/panel_pool2047.parquet  designated pool (m = 2047), prh_clip+l2; all panels
  ~/hrh_v2_stage1_20261008/results/S1_2/panel_main.parquet      m = 1024; only for the 816-configuration bound check
  ~/hrh_v2_stage1_20261008/results/S1_2/s12_summary.json        preregistered PE2 numbers and Holm-adjusted p
  ~/hrh_v2_stage1_20261008/analysis/s12/ANALYSIS_PLAN_S12.md    preregistered level 0.70 (line 98)
  ~/hrh_v2_stage1_20261008/results/S1_2/S1_2_report.md          section 3 (PE2, m = 2047) and 5 (Holm): asserted
  ~/hrh_v2_figs/F2/numbers.json                                  v3 numbers; every recomputed value is asserted equal
  ~/hrh_v2_figs/v5/F_mst_app/numbers.json                        v5 numbers; panel_c / panel_b values asserted equal
  ~/hrh_v2_figs/v5/_style/hrh_style_v5_1.py                      frozen shared style (imported, never edited)
Writes only into this directory: F_mst_app.pdf, F_mst_app.png (300 dpi, exact print size), numbers.json.
Run: OMP_NUM_THREADS=1 nice -n 10 taskset -c 0-5 ~/hrh_v2_figs_venv/bin/python -B \
         ~/hrh_v2_figs/v8/F_mst_app/make_F_mst_app.py > run.log 2>&1
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
import matplotlib.transforms  # noqa: E402
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox  # noqa: E402

STYLE_DIR = os.path.expanduser("~/hrh_v2_figs/v8/_style")
STYLE_PATH = os.path.join(STYLE_DIR, "hrh_style_v8.py")
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"   # hrh_style_v8.py (frozen)
assert hashlib.sha256(open(STYLE_PATH, "rb").read()).hexdigest() == STYLE_SHA, "frozen style module changed"
sys.path.insert(0, STYLE_DIR)
import hrh_style_v8 as S  # noqa: E402

S.use()                  # rcParams and mathtext scripts at 7/8 of their base (the module owns both)
assert S.STYLE_SHA256 == STYLE_SHA and S.STYLE_VERSION == "v8"
LW, MS, LS, Z = S.LW, S.MS, S.LS, S.Z

S1 = os.path.expanduser("~/hrh_v2_stage1_20261008/results/S1_2")
P_POOL, P_MAIN, P_SUM = f"{S1}/panel_pool2047.parquet", f"{S1}/panel_main.parquet", f"{S1}/s12_summary.json"
P_REPORT = f"{S1}/S1_2_report.md"
P_PLAN = os.path.expanduser("~/hrh_v2_stage1_20261008/analysis/s12/ANALYSIS_PLAN_S12.md")
P_OLD = os.path.expanduser("~/hrh_v2_figs/F2/numbers.json")
P_V5 = os.path.expanduser("~/hrh_v2_figs/v5/F_mst_app/numbers.json")
P_V5_CODE = os.path.expanduser("~/hrh_v2_figs/v5/F_mst_app/make_F_mst_app.py")
V5_SHA = {"numbers": "ccbff9dade4125b8b8a6443006d746dc087462bdf7450114eb5b6a2a2221f9aa",
          "code": "430deacd284479ac180828d2951682db1d9c356fd46a34a8c9773d45cb9db3ad"}
PRE = "prh_clip+l2"
READING = "A (SPEC-literal, F_A = LM forest)"
# (a) switches shared with Figure 2 (c) (same 272 points; the two figures must draw them alike)
FILL_EDGE = os.environ.get("FM_FILL_EDGE", "self")       # halo: two passes, white 4.0 pt discs under green 3.2 pt
#                                                          (default); white: per-marker white 0.4 edge; self: edge in fill colour
LEADER = os.environ.get("FM_LEADER", "ink")              # ink: spec section 4.2 (leaders INK); tone: MST text tone
assert FILL_EDGE in ("white", "self", "halo") and LEADER in ("ink", "tone"), (FILL_EDGE, LEADER)
TRIAL = os.environ.get("FM_TRIAL", "")
OUT = os.path.join(HERE, "_trials", TRIAL) if TRIAL else HERE
os.makedirs(OUT, exist_ok=True)
NB = 10000
SEED = 20261009
DEPTH = {0.5: "mid depth", 1.0: "last layer"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


assert sha(STYLE_PATH) == STYLE_SHA, "frozen style module changed"
SUM = json.load(open(P_SUM))
OLD = json.load(open(P_OLD))

# =============================================================================================
# data and statistics: copied from ~/hrh_v2_figs/F2/make_F2.py (same order of RNG calls)
# =============================================================================================
d = pd.read_parquet(P_POOL)
d = d[d.pre == PRE].copy()
assert len(d) == 272 and (d["L.m"] == 2047).all()
m = 2047.0
d["sN1"] = d["L.s_N_k1_for_mst"].astype(float)
d["sN10"] = d["L.s_N_k10"].astype(float)
d["sT"] = d["L.s_T"].astype(float)
d["nmut"] = d["L.m"] - d["M.n_forest_A"]
d["n_sh"] = d["M.n_shared_edges"].astype(float)
d["n_sh_br"] = d["M.n_shared_bridges_A"].astype(float)
d["n_sh_f"] = d.n_sh - d.n_sh_br
d["bound"] = np.maximum(m * d.sN1 - d.nmut, m * d.sN1 / 2) / (m - 1)
assert np.allclose(d.n_sh / (m - 1), d.sT, atol=1e-12)
assert np.allclose(d.n_sh_f / d.n_sh, d["M.forest_share_shared_A"], atol=1e-12)
assert (d["M.n_bridges_A"] == d.nmut - 1).all()
assert np.allclose(d["L.s_N_k1"], d.sN1)

allc = []
for p in (P_POOL, P_MAIN):
    x = pd.read_parquet(p)
    mm = x["L.m"].astype(float)
    s1 = x["L.s_N_k1_for_mst"].astype(float)
    nm = mm - x["M.n_forest_A"]
    b = np.maximum(mm * s1 - nm, mm * s1 / 2) / (mm - 1)
    allc.append(pd.DataFrame({"pool": os.path.basename(p), "pre": x.pre, "r": x.r, "ok": x["L.s_T"] >= b - 1e-12,
                              "ratio": x["L.s_T"] / b, "half_binds": mm * s1 / 2 >= mm * s1 - nm,
                              "sN1": s1, "two_nmut_over_m": 2 * nm / mm}))
allc = pd.concat(allc)
BOUND = {
    "n_configs": int(len(allc)), "n_bound_holds": int(allc.ok.sum()),
    "n_half_term_binds": int(allc.half_binds.sum()),
    "max_sN1": float(allc.sN1.max()), "min_2nmut_over_m": float(allc.two_nmut_over_m.min()),
    "note": "the m*sN1 - n_mut branch would bind only if sN1 > 2 n_mut/m; never reached on this panel",
    "sT_over_bound_median_by_cell": {f"{a}|{b}|r={c}": float(g.ratio.median()) for (a, b, c), g in allc.groupby(["pool", "pre", "r"])},
    "sT_over_bound_min": float(allc.ratio.min()), "sT_over_bound_max": float(allc.ratio.max())}

NA = {}
for r, g in d.groupby("r"):
    NA[f"r={r}"] = {
        "n_pairs": int(len(g)),
        "median_sT": float(g.sT.median()), "median_sN1": float(g.sN1.median()),
        "median_sT_over_sN1": float((g.sT / g.sN1).median()),
        "q05_q95_sT_over_sN1": [float((g.sT / g.sN1).quantile(.05)), float((g.sT / g.sN1).quantile(.95))],
        "median_sT_over_bound": float((g.sT / g.bound).median()), "min_sT_over_bound": float((g.sT / g.bound).min()),
        "frac_sT_above_sN1": float((g.sT > g.sN1).mean()),
        "max_abs_sT_minus_sN1": float((g.sT - g.sN1).abs().max()),
        "null_sN1_exact_1_over_m_minus_1": 1 / (m - 1),
        "null_sT_perm_mean_median": float(g["cal.s_T.null_mean"].median()),
        "frac_sT_above_null_scal_gt0": float((g["cal.s_T.s_cal"] > 0).mean()),
        "EXPLORATORY_spearman_sT_sN1": float(spearmanr(g.sT, g.sN1).statistic),
        "EXPLORATORY_spearman_sT_sN10": float(spearmanr(g.sT, g.sN10).statistic)}
NA["pooled"] = {"EXPLORATORY_spearman_sT_sN1": float(spearmanr(d.sT, d.sN1).statistic),
                "EXPLORATORY_spearman_sT_sN10": float(spearmanr(d.sT, d.sN10).statistic), "n": int(len(d))}

NBd = {}
for r, g in d.groupby("r"):
    exp_br = g["cal.s_T_bridge.null_mean"] * g["M.n_bridges_A"]
    exp_all = g["cal.s_T.null_mean"] * (m - 1)
    NBd[f"r={r}"] = {
        "median_forest_share_of_shared": float(g["M.forest_share_shared_A"].median()),
        "min_forest_share_of_shared": float(g["M.forest_share_shared_A"].min()),
        "frac_pairs_forest_share_ge_0.70": float((g["M.forest_share_shared_A"] >= 0.70).mean()),
        "mean_shared_edges": float(g.n_sh.mean()), "mean_shared_forest_edges": float(g.n_sh_f.mean()),
        "mean_shared_bridges": float(g.n_sh_br.mean()), "n_pairs_zero_shared_bridges": int((g.n_sh_br == 0).sum()),
        "mean_null_expected_shared_bridges": float(exp_br.mean()),
        "mean_null_expected_shared_edges": float(exp_all.mean()),
        "median_bridge_overlap": float(g["L.s_T_bridge"].median()),
        "median_bridge_null_mean": float(g["cal.s_T_bridge.null_mean"].median()),
        "median_bridge_null_c_alpha": float(g["cal.s_T_bridge.c_alpha"].median()),
        "frac_pairs_bridge_scal_gt0": float((g["cal.s_T_bridge.s_cal"] > 0).mean()),
        "frac_pairs_bridge_p_le_0.05": float((g["cal.s_T_bridge.p_value"] <= 0.05).mean()),
        "mean_bridges_in_LM_tree": float(g["M.n_bridges_A"].mean()),
        "median_tree_forest_fraction_nforest_over_m_minus_1": float((g["M.n_forest_A"] / (m - 1)).median()),
        "EXPLORATORY_median_forest_edge_sharing_rate_sT_forest": float(g["L.s_T_forest"].median()),
        "EXPLORATORY_median_bridge_sharing_rate_sT_bridge": float(g["L.s_T_bridge"].median()),
        "EXPLORATORY_frac_pairs_forest_share_above_tree_forest_fraction": float((g["M.forest_share_shared_A"] > g["M.n_forest_A"] / (m - 1)).mean()),
        "EXPLORATORY_median_ratio_forest_rate_over_bridge_rate": float((g["L.s_T_forest"] / g["L.s_T_bridge"].replace(0, np.nan)).median()),
        "null_note": "under random relabelling every LM-tree edge is in the ViT tree with probability 2/m, so the ratio of expected shared forest edges to expected shared edges is n_forest_A/(m-1)"}

bl = d[d.lm_family == "BLOOMZ"]
lms = bl.drop_duplicates("lm").sort_values("lm_params")[["lm", "lm_params"]]
LM_ORDER, LM_PAR = list(lms.lm), list(lms.lm_params)
VITS = sorted(d.vit.unique())
assert len(LM_ORDER) == 5 and len(VITS) == 17
PERMS = np.array(list(itertools.permutations(range(5))))


def gridv(g, col):
    t = g.pivot(index="lm", columns="vit", values=col).loc[LM_ORDER, VITS]
    return t.to_numpy(float)


def trend(Y):
    """Tr = mean over ViTs of Spearman(size rank, score); exact p over the 120 relabellings."""
    R = np.apply_along_axis(rankdata, 0, Y)

    def tr_of(x):
        return float(np.mean([spearmanr(x, R[:, j]).statistic for j in range(R.shape[1])]))
    T = np.array([tr_of(p) for p in PERMS])
    t = tr_of(np.arange(5))
    return t, float(np.mean(T >= t - 1e-12)), float(np.mean(np.abs(T) >= abs(t) - 1e-12))


rng = np.random.default_rng(SEED)
NC, CUR, BANDS = {}, {}, {}
for r in (0.5, 1.0):
    g = bl[bl.r == r]
    U = gridv(g, "cal.s_T_bridge.uncensored")
    UT = gridv(g, "cal.s_T.uncensored")
    t, p1, p2 = trend(U)
    tT, pT1, pT2 = trend(UT)
    ref = SUM["pe2"][f"m2047|r={r}|{PRE}|{READING}"]["Tr_bridge"]
    assert abs(t - ref["Tr"]) < 1e-12 and abs(p1 - ref["p_one"]) < 1e-12 and abs(p2 - ref["p_two"]) < 1e-12, (t, ref)
    mean = U.mean(1)
    idx = rng.integers(0, U.shape[1], size=(NB, U.shape[1]))
    bs = U[:, idx].mean(2)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=1)
    band = float((g["cal.s_T_bridge.c_alpha"] - g["cal.s_T_bridge.null_mean"]).median())
    hk = f"PE2|r={r}|m2047"
    NC[f"r={r}"] = {
        "lm_order": LM_ORDER, "lm_params": [float(v) for v in LM_PAR],
        "mean_over_17_vits_uncensored_bridge_overlap": mean.tolist(),
        "vit_bootstrap_95ci_lo": lo.tolist(), "vit_bootstrap_95ci_hi": hi.tolist(), "bootstrap_B": NB, "seed": SEED,
        "Tr_bridge": t, "p_one_exact": p1, "p_two_exact": p2, "n_perm": 120, "smallest_attainable_p": 1 / 120,
        "p_one_holm": SUM["holm"]["p_one"]["holm"][hk], "p_two_holm": SUM["holm"]["p_two"]["holm"][hk],
        "Tr_bridge_ci_vits_from_summary": SUM["pe2"][f"m2047|r={r}|{PRE}|{READING}"]["Tr_bridge_ci_vits"],
        "openllama_7b_gt_3b_frac_vits": ref["openllama_frac_larger_higher"],
        "verdict": SUM["verdicts"]["PE2"][f"m2047|r={r}|{PRE}|{READING}"]["verdict"],
        "median_null_c_alpha_minus_null_mean": band,
        "EXPLORATORY_Tr_sT_all_edges_uncensored": tT, "EXPLORATORY_p_one_sT": pT1, "EXPLORATORY_p_two_sT": pT2,
        "EXPLORATORY_mean_over_vits_uncensored_sT": UT.mean(1).tolist()}
    CUR[r] = (mean, lo, hi)
    BANDS[r] = band


# =============================================================================================
# cross-check against the v3 numbers (F2/numbers.json) and the values named in the remake spec
# =============================================================================================
def same(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b))), (path, a, b)
    else:
        assert a == b, (path, a, b)


NEW = json.loads(json.dumps({"panel_a": NA, "panel_b": NBd, "panel_c": NC, "bound_check_all_configs": BOUND}))
for k in NEW:
    same(NEW[k], OLD[k], k)
assert round(NBd["r=0.5"]["median_forest_share_of_shared"], 3) == 0.875
assert all(NBd[f"r={r}"]["frac_pairs_forest_share_ge_0.70"] == 1.0 for r in (0.5, 1.0))     # every pair above 0.70
assert (round(NBd["r=0.5"]["min_forest_share_of_shared"], 2), round(NBd["r=1.0"]["min_forest_share_of_shared"], 2)) == (0.76, 0.78)
assert round(NBd["r=1.0"]["median_forest_share_of_shared"], 3) == 0.889
assert round(NBd["r=0.5"]["median_tree_forest_fraction_nforest_over_m_minus_1"], 3) == 0.850
assert round(NBd["r=1.0"]["median_tree_forest_fraction_nforest_over_m_minus_1"], 3) == 0.852
c1, c5 = NC["r=1.0"], NC["r=0.5"]
assert (round(c1["Tr_bridge"], 2), round(c1["p_one_exact"], 3), round(c1["p_one_holm"], 2)) == (0.57, 0.058, 0.23)
assert (round(c5["Tr_bridge"], 2), round(c5["p_one_exact"], 2)) == (-0.16, 0.74)
assert BOUND["n_configs"] == BOUND["n_bound_holds"] == BOUND["n_half_term_binds"] == 816
PLAN_LINE = open(P_PLAN).read().splitlines()[97].strip()
assert "0.70" in PLAN_LINE and "Forest share" in PLAN_LINE, PLAN_LINE
LEVEL = 0.70
PLAN_TXT = open(P_PLAN).read().splitlines()
PLAN_C = next(ln.strip() for ln in PLAN_TXT if ln.strip().startswith("- (c) The within-family trend Tr"))
PLAN_D = next(ln.strip() for ln in PLAN_TXT if ln.strip().startswith("- (d) Tr"))
assert "two-sided" in PLAN_C and "one-sided" in PLAN_D, (PLAN_C, PLAN_D)

# Conditions (c) and (d) of the PE2 rule. The verdict "SOG-global ~ 1-NN" rests on (c), the two-sided test.
# With 5 sizes the reversed size order is one of the 120 relabellings and flips the sign of Tr, so every exact
# two-sided p is >= 2/120; Holm over the 4-test family multiplies the smallest p by 4, so every adjusted
# two-sided p is >= 4 * 2/120 = 0.067 > 0.05 and (c) cannot fail on any data. (d) uses the one-sided p.
HOLM_FAMILY = sorted(SUM["holm"]["p_two"]["raw"])
assert len(HOLM_FAMILY) == 4 and sorted(SUM["holm"]["p_one"]["raw"]) == HOLM_FAMILY


def holm(raw):
    ks = sorted(raw, key=lambda k: raw[k])
    out, run = {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - i) * raw[k]))
        out[k] = run
    return out


for side in ("p_one", "p_two"):
    hh = holm(SUM["holm"][side]["raw"])
    for k in HOLM_FAMILY:
        assert abs(hh[k] - SUM["holm"][side]["holm"][k]) < 1e-12, (side, k)
REV = PERMS.tolist().index([4, 3, 2, 1, 0])
for r in (0.5, 1.0):
    U = gridv(bl[bl.r == r], "cal.s_T_bridge.uncensored")
    R = np.apply_along_axis(rankdata, 0, U)
    t_rev = float(np.mean([spearmanr(PERMS[REV], R[:, j]).statistic for j in range(R.shape[1])]))
    assert abs(t_rev + NC[f"r={r}"]["Tr_bridge"]) < 1e-12          # the reversal gives -Tr
MIN_P_TWO = 2 / 120
MIN_P_ONE = 1 / 120
MIN_HOLM_TWO = min(1.0, len(HOLM_FAMILY) * MIN_P_TWO)
MIN_HOLM_ONE = min(1.0, len(HOLM_FAMILY) * MIN_P_ONE)
assert MIN_HOLM_TWO > 0.05 and MIN_HOLM_ONE < 0.05
for r in (0.5, 1.0):
    v = SUM["verdicts"]["PE2"][f"m2047|r={r}|{PRE}|{READING}"]
    assert v["c_trend_null"] and not v["d_trend_genuine"] and v["verdict"] == "SOG-global ≈ 1-NN", v
    assert abs(v["p_two_holm"] - NC[f"r={r}"]["p_two_holm"]) < 1e-12 and abs(v["p_one_holm"] - NC[f"r={r}"]["p_one_holm"]) < 1e-12
assert (round(c1["p_two_exact"], 3), round(c1["p_two_holm"], 3)) == (0.117, 0.467)
TREND_RULE = {
    "condition_c": {"text": PLAN_C, "source": f"{P_PLAN}"},
    "condition_d": {"text": PLAN_D, "source": f"{P_PLAN}"},
    "holm_family": HOLM_FAMILY,
    "smallest_attainable_exact_p_two": MIN_P_TWO,
    "smallest_attainable_holm_p_two": MIN_HOLM_TWO,
    "smallest_attainable_exact_p_one": MIN_P_ONE,
    "smallest_attainable_holm_p_one": MIN_HOLM_ONE,
    "why": "the reversed BLOOMZ size order is one of the 120 relabellings and gives -Tr (asserted on the data), so "
           "|T| >= |Tr| for at least 2 of 120 relabellings; Holm over the 4-test family multiplies the smallest raw p "
           "by 4 and never lowers the others below it, so condition (c) holds on any data",
    "per_depth": {DEPTH[r]: {"Tr": NC[f"r={r}"]["Tr_bridge"],
                             "p_two_exact": NC[f"r={r}"]["p_two_exact"], "p_two_holm": NC[f"r={r}"]["p_two_holm"],
                             "p_one_exact": NC[f"r={r}"]["p_one_exact"], "p_one_holm": NC[f"r={r}"]["p_one_holm"],
                             "c_trend_null": True, "d_trend_genuine": False,
                             "source": f"{P_SUM} verdicts['PE2']['m2047|r={r}|{PRE}|{READING}'] and holm[p_one|p_two]['holm']['PE2|r={r}|m2047']"}
                  for r in (0.5, 1.0)}}
N_BLOOMZ_PAIRS = int(len(bl) // 2)
assert N_BLOOMZ_PAIRS == 85 and (bl.groupby("r").size() == 85).all()
POOL_COUNTS = {f"{a}|{b}": int(n) for (a, b), n in allc.groupby(["pool", "pre"]).size().items()}
assert POOL_COUNTS == {"panel_main.parquet|l2": 272, "panel_main.parquet|prh_clip+l2": 272,
                       "panel_pool2047.parquet|prh_clip+l2": 272}, POOL_COUNTS

# =============================================================================================
# v6: the on-panel numbers are unchanged against the v5 numbers.json and the frozen S1.2 report
# =============================================================================================
assert sha(P_V5) == V5_SHA["numbers"] and sha(P_V5_CODE) == V5_SHA["code"], "v5 F_mst_app changed"
V5 = json.load(open(P_V5))
V5_CHECK = {}
for r in (0.5, 1.0):
    dn, nc = DEPTH[r], NC[f"r={r}"]
    s5 = V5["panel_c"]["statistics_for_caption"][dn]
    for k_ in ("Tr_bridge", "p_one_exact", "p_one_holm", "p_two_exact", "p_two_holm"):
        assert abs(nc[k_] - s5[k_]) <= 1e-12, (dn, k_, nc[k_], s5[k_])
    cv5 = V5["panel_c"]["curves"][dn]
    for k_, arr in (("mean", CUR[r][0]), ("ci95_lo", CUR[r][1]), ("ci95_hi", CUR[r][2])):
        same(arr.tolist(), cv5[k_], f"v5.panel_c.curves.{dn}.{k_}")
    same((np.array(LM_PAR, float) / 1e9).tolist(), cv5["params_B"], f"v5.panel_c.curves.{dn}.params_B")
    V5_CHECK[dn] = {"Tr_bridge": s5["Tr_bridge"], "p_one_exact": s5["p_one_exact"], "p_one_holm": s5["p_one_holm"],
                    "curves": "mean, ci95_lo, ci95_hi, params_B equal within 1e-12"}
assert abs(V5["panel_c"]["null_band"]["drawn_upper"] - max(BANDS.values())) <= 1e-12
assert abs(V5["panel_c"]["trend_rule_conditions_c_d"]["smallest_attainable_exact_p_one"] - 1 / 120) <= 1e-15
assert all(abs(NC[f"r={r}"]["smallest_attainable_p"] - 1 / 120) <= 1e-15 for r in (0.5, 1.0))
assert abs(MIN_P_ONE - 1 / 120) <= 1e-15

# S1_2_report.md section 3 (m = 2047 column, primary reading) and section 5 (Holm), parsed as text
REP = open(P_REPORT).read()


def rep_slice(txt, head, nxt=None):
    i = txt.index(head)
    return txt[i:txt.index(nxt, i + len(head))] if nxt else txt[i:]


SEC3 = rep_slice(REP, "## 3. PE2: MST = forest + bridges", "## 4. PE3")
SEC5 = rep_slice(REP, "## 5. Holm correction", "## 6. Diagnostics")
SUB3 = {0.5: rep_slice(SEC3, "### r = 0.5", "### r = 1.0"), 1.0: rep_slice(SEC3, "### r = 1.0")}


def rep_cell(r, starts):
    """m = 2047 cell of the first row of the r subsection whose first cell starts with `starts`."""
    for ln in SUB3[r].splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].startswith(starts):
            return cells[2], ln.strip()
    raise AssertionError((r, starts))


REPORT_CHECK = {}
for r in (0.5, 1.0):
    dn, nc = DEPTH[r], NC[f"r={r}"]
    c_tr, l_tr = rep_cell(r, "(c/d) bridge-overlap trend Tr")
    c_p, l_p = rep_cell(r, "exact p one-sided / two-sided")
    c_fs, l_fs = rep_cell(r, "median forest share of shared edges")
    c_br, l_br = rep_cell(r, "median bridge overlap / median null mean")
    c_sh, l_sh = rep_cell(r, "mean shared edges / A-bridges / shared A-bridges")
    tr_rep = float(c_tr.split()[0])
    p1_rep, p2_rep = (float(v) for v in c_p.split("/"))
    fs_rep = float(c_fs.split()[0])
    br_rep, brn_rep = (float(v) for v in c_br.split("/"))
    sh_rep = [float(v) for v in c_sh.split("/")]
    assert round(nc["Tr_bridge"], 3) == tr_rep, (dn, nc["Tr_bridge"], c_tr)
    assert (round(nc["p_one_exact"], 3), round(nc["p_two_exact"], 3)) == (p1_rep, p2_rep), (dn, c_p)
    assert round(NBd[f"r={r}"]["median_forest_share_of_shared"], 3) == fs_rep, (dn, c_fs)
    assert round(NBd[f"r={r}"]["median_bridge_overlap"], 4) == br_rep, (dn, c_br)
    assert round(NBd[f"r={r}"]["mean_shared_bridges"], 2) == sh_rep[2], (dn, c_sh)
    import re as _re
    mh = _re.search(r"PE2\|r=%s\|m2047 \| ([0-9.]+) → ([0-9.]+) \| ([0-9.]+) → ([0-9.]+)" % _re.escape(str(r)), SEC5)
    assert mh, (dn, "Holm row")
    h_rep = [float(v) for v in mh.groups()]
    assert [round(nc["p_one_exact"], 3), round(nc["p_one_holm"], 3), round(nc["p_two_exact"], 3),
            round(nc["p_two_holm"], 3)] == h_rep, (dn, h_rep)
    REPORT_CHECK[dn] = {"Tr_row": l_tr, "p_row": l_p, "forest_share_row": l_fs, "bridge_row": l_br,
                        "shared_edges_row": l_sh, "holm_row": mh.group(0),
                        "result": "Tr, exact p (one/two-sided), Holm p, forest-share median, bridge-overlap median and "
                                  "mean shared A-bridges equal the report to its printed precision"}
assert "120 perms" in SEC3


# =============================================================================================
# figure: 5.5 x 2.1 in, exact print size, axes placed in inches (no layout engine, no tight bbox)
# =============================================================================================
assert (d["K"] == 500).all()                       # permutation count n_pi of every calibrated score
K_NULL = int(d["K"].iloc[0])
G, GT = S.CRIT["mst"], S.CRIT_TEXT["mst"]          # MST hue for marks and lines; its text tone (>= 4.5:1) for labels
sys.path.insert(0, STYLE_DIR)
import cvdlib as CV  # noqa: E402  (read-only, next to the frozen style module)
MST_MED = CV.text_tone(G, target=8.0)              # v8 c: median bars of (b), a dark step of the MST hue (not INK)
assert MST_MED == "#005B4D" and CV.contrast(MST_MED) >= 8.0, MST_MED
INK, MUTED, NL, NF = S.INK, S.MUTED, S.NULL_LINE, S.NULL_FILL
W, H = S.TEXTWIDTH_IN, 2.1
fig = plt.figure(figsize=(W, H))
fig.set_layout_engine("none")


def axes_in(x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


Y0, AH = 0.36, 1.45                  # common bottom and height of the three data areas (in), as in v4
XA = 0.405                           # (a) square data area, 1.45 x 1.45 in; v4 0.39 (0.5 pt longer ticks)
XB = 2.40                            # (b) left edge; v4 2.38 (ticks are 0.5 pt longer in v5, (a) moved right)
WC, RM_C = 1.15, 0.04                # (c) width and right margin as in v4 (its centred x label needs them)
XC = W - RM_C - WC
MIN_PANEL_GAP = 0.10                 # in, between the label extents of neighbouring panels
SW_D = 2.6                           # (b) outer dot diameter (ms + mew) = beeswarm step: dots touch, never overlap
GAP_SW, PAD_SW = 5.0, 3.0            # (b) pt between the two swarms / between a swarm and the axes edges
YB = (0.65, 1.0)                     # (b) y range (v4 0.6-1.0): taller spread, so the larger dots fit the width
SWARM_NOTE = ("spec F.2 (b) asks for swarm dots of MS['swarm'] = 3.0 pt (open edge 0.7, filled white edge 0.3). A "
              "beeswarm with no overlapping dots (the v4 rule, asserted) of 136 dots per depth needs a data area of "
              "about 1.9 in at a 3.0 pt step, and 1.55-1.73 in at 2.6-2.8 pt, on the v4 y range from 0.6; about "
              "1.41 in is available between (a) and (c). v5 draws 2.6 pt dots (outer diameter; v4 2.2 pt) with the "
              "spec's edge widths on a y range from 0.65, which needs 1.41 in.")
axA = axes_in(XA, Y0, AH, AH)
axC = axes_in(XC, Y0, WC, AH)


def mk_a(r):
    """(a) point cloud, spec F.2 (a) = spec Figure 2 (c): mid depth open (edge 0.9), last layer filled with an edge
    0.4 in white (FILL_EDGE = 'white', the spec) or in the fill colour ('self', what F_neighbors v5 round 1 draws).
    'halo': this is the green pass (3.2 pt, no edge); mk_a_halo() is the white pass under it."""
    if r == 0.5:
        return dict(ls="none", marker="o", ms=MS["scatter"], mfc="white", mec=G, mew=MS["scatter_open_mew"])
    if FILL_EDGE == "halo":
        return dict(ls="none", marker="o", ms=MS["scatter"] - MS["scatter_edge"], mfc=G, mec=G, mew=0.0)
    return dict(ls="none", marker="o", ms=MS["scatter"], mfc=G, mec="white" if FILL_EDGE == "white" else G,
                mew=MS["scatter_edge"])


def mk_a_halo():
    """white pass under the filled cloud for FILL_EDGE = 'halo': discs of the outer diameter 3.6 + 0.4 = 4.0 pt."""
    return dict(ls="none", marker="o", ms=MS["scatter"] + MS["scatter_edge"], mfc="white", mec="white", mew=0.0)


def mk_c(r):
    """(c) five sizes per line: filled with a white edge 0.6 / open with edge 0.9 (spec F.2 c)."""
    if r == 0.5:
        return dict(ls="none", marker="o", ms=MS["line_sparse"], mfc="white", mec=G, mew=MS["open_small_mew"])
    return dict(ls="none", marker="o", ms=MS["line_sparse"], mfc=G, mec="white", mew=MS["line_edge"])


def mk_swarm(r):
    """(b) beeswarm dots, outer diameter (ms + mew) = SW_D: open edge 0.7, filled white edge 0.3 (spec F.2 b)."""
    if r == 0.5:
        return dict(ls="none", marker="o", ms=SW_D - MS["swarm_open_mew"], mfc="white", mec=G,
                    mew=MS["swarm_open_mew"])
    return dict(ls="none", marker="o", ms=SW_D - MS["swarm_edge"], mfc=G, mec="white", mew=MS["swarm_edge"])


def text_wh(ax, s, role, color=None, **kw):
    """width and height of a text of a role in data units of ax (linear axes), measured with the renderer."""
    t = S.text(ax, 0, 0, s, role, color=color, **kw)
    fig.canvas.draw()
    bb = t.get_window_extent()
    t.remove()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x1 - x0, y1 - y0


# (a) s_T against s_{N,1} ----------------------------------------------------------------------
LIM = 0.09
assert d.sN1.max() < LIM and d.sT.max() < LIM and d.sN1.min() > 0 and d.sT.min() > 0
min_2nmut_pool = float((2 * d.nmut / m).min())
assert min_2nmut_pool > LIM       # on the drawn range the bound equals its half term for every plotted pair
axA.set_xlim(0, LIM)
axA.set_ylim(0, LIM)
xs = np.linspace(0, LIM, 200)
slope_b = m / (2 * (m - 1))
axA.fill_between(xs, 0, slope_b * xs, color=NF, lw=0, zorder=Z["fill"])
axA.plot(xs, xs, color=NL, lw=LW["ref"], ls=LS["dotted"], zorder=Z["ref"])
axA.plot(xs, slope_b * xs, color=INK, lw=LW["theory"], zorder=Z["ref"], solid_capstyle="butt")
for r in (0.5, 1.0):
    g = d[d.r == r]
    if r == 1.0 and FILL_EDGE == "halo":
        axA.plot(g.sN1, g.sT, zorder=Z["data"] - 0.02, **mk_a_halo())
    axA.plot(g.sN1, g.sT, zorder=Z["marker"] if r == 0.5 else Z["data"], **mk_a(r))
cx, cy = 1 / (m - 1), float(d["cal.s_T.null_mean"].median())
# chance: NULL_LINE mark; MUTED label and leader (a reference annotation, so it stays grey rather than INK). The
# label sits low in the excluded wedge (declared on NULL_FILL for the contrast check), well below the bound line.
CH_MS, CH_MEW = MS["line"], LW["ref"]
axA.plot([cx], [cy], marker="x", ms=CH_MS, mew=CH_MEW, color=NL, zorder=Z["marker"] + 0.5, clip_on=False)
CH_XY = (0.020, 0.0030)
ann_ch = axA.annotate("chance", (cx, cy), xytext=CH_XY, va="center", ha="left",
                      arrowprops=dict(arrowstyle="-", lw=LW["leader"], color=MUTED, shrinkA=1.0,
                                      shrinkB=(CH_MS + CH_MEW) / 2 + 0.5),
                      **S.text_kw("annot", "chance"))
S.on_fill(ann_ch, NF)
t_ex = S.text(axA, 0.0875, 0.0035, "excluded", "annot", ha="right", va="bottom")
S.on_fill(t_ex, NF)
ang_b = float(np.degrees(np.arctan(slope_b)))
OFF_BOUND = 0.0015                   # v4 0.0012; +0.35 pt for the heavier bound line
S.text(axA, 0.062, slope_b * 0.062 + OFF_BOUND, "bound", "theory_label", rotation=ang_b,
       rotation_mode="anchor", ha="left", va="bottom", transform_rotates_text=True)
# diagonal label '$y=x$', as on the same diagonal in F_neighbors (c) and F_scale_app (c); the caption names the line
# s_T = s_N,1
S.text(axA, 0.0690, 0.0690 - 0.0017, r"$y=x$", "annot", rotation=45, rotation_mode="anchor", ha="left", va="top",
       transform_rotates_text=True)

# direct labels, placed from the data: outer marker radius and gap in data units of (a)
RAD_A = ((MS["scatter"] + MS["scatter_open_mew"]) / 2 / 72) * LIM / AH
GAP_A = (2.0 / 72) * LIM / AH
PX, PY = d.sN1.to_numpy(float), d.sT.to_numpy(float)
# 'last layer': right-aligned, just left of the top of the filled cloud (all points above s_T = 0.0372 are last layer)
wL, hL = text_wh(axA, "last layer", "direct", color=GT)
YL = 0.079
band = np.abs(PY - YL) <= hL / 2 + RAD_A + GAP_A
assert band.any() and (d.r.to_numpy()[band] == 1.0).all()
XL_R = float(PX[band].min()) - RAD_A - GAP_A
assert XL_R - wL > 0.002
S.text(axA, XL_R, YL, "last layer", "direct", color=GT, ha="right", va="center")
# 'mid depth': two lines in the empty upper left, with an INK leader to the upper-left edge of the open cloud.
# The leader stays left of every filled point (the smallest last-layer s_N1 is larger than its x range).
wM, hM = text_wh(axA, "mid\ndepth", "direct", color=GT, linespacing=1.0)
XM = 0.0025
GAP_M = (3.5 / 72) * LIM / AH      # wider clearance, so the label reads as apart from the nearby filled points
YM = None
for yb in np.arange(0.030, 0.06, 0.0005):
    inbox = ((PX + RAD_A + GAP_M >= XM) & (PX - RAD_A - GAP_M <= XM + wM)
             & (PY + RAD_A + GAP_M >= yb) & (PY - RAD_A - GAP_M <= yb + hM))
    if not inbox.any():
        YM = float(yb)
        break
assert YM is not None
openp = d[(d.r == 0.5) & (d.sN1 <= 0.0115)]
tgt = openp.sort_values("sT").iloc[-1]
TX, TY = float(tgt.sN1), float(tgt.sT)
S.text(axA, XM, YM, "mid\ndepth", "direct", color=GT, ha="left", va="bottom", linespacing=1.0)
LX0, LY0 = XM + 0.30 * wM, YM - 0.0009
LEADER_C = INK if LEADER == "ink" else GT
axA.annotate("", xy=(TX, TY), xytext=(LX0, LY0),
             arrowprops=dict(arrowstyle="-", lw=LW["leader"], color=LEADER_C, shrinkA=0,
                             shrinkB=(MS["scatter"] + MS["scatter_open_mew"]) / 2 + 0.8))
fil = d[d.r == 1.0]
assert fil.sN1.min() > max(LX0, TX) + RAD_A           # leader x range lies left of every filled point
axA.set_xticks([0, 0.03, 0.06, 0.09], ["0", "0.03", "0.06", "0.09"])
axA.set_yticks([0, 0.03, 0.06, 0.09], ["0", "0.03", "0.06", "0.09"])
# braced comma: mathtext adds a space after a bare comma ('N, 1'); '{,}' prints 'N,1' as LaTeX does in script style
axA.set_xlabel(r"1-NN overlap $s_{N{,}1}$")
axA.set_ylabel(r"MST overlap $s_{\mathcal{T}}$", labelpad=1.5)   # 7 pt subscript: 1 pt closer, stays inside


# (b) forest share of the shared MST edges ------------------------------------------------------
def beeswarm(y_pt, step, order):
    """Greedy beeswarm in points: y stays exact, each dot takes the admissible x offset closest to the centre line
    (ties go to the emptier side), so no two dots are closer than `step` (centre to centre)."""
    xs = np.zeros(len(y_pt))
    placed = []
    for i in order:
        yi = y_pt[i]
        near = [j for j in placed if abs(y_pt[j] - yi) < step]
        cands = [0.0]
        for j in near:
            dx = np.sqrt(step ** 2 - (y_pt[j] - yi) ** 2)
            cands += [xs[j] + dx, xs[j] - dx]
        n_l = sum(xs[j] < 0 for j in placed)
        n_r = sum(xs[j] > 0 for j in placed)
        cands = sorted(cands, key=lambda c: (round(abs(c), 6), (c > 0) == (n_r > n_l)))
        for c in cands:
            if all((c - xs[j]) ** 2 + (y_pt[j] - yi) ** 2 >= step ** 2 - 1e-7 for j in near):
                xs[i] = c
                break
        else:
            raise AssertionError("no admissible swarm position")
        placed.append(i)
    return xs


assert d["M.forest_share_shared_A"].min() > YB[0] + 0.1 and d["M.forest_share_shared_A"].max() <= YB[1]
V_B = AH * 72 / (YB[1] - YB[0])            # points per y data unit in (b)
PB, SWARM, OFF = {}, {}, {}
for i, r in enumerate((0.5, 1.0)):
    g = d[d.r == r].sort_values(["lm", "vit"])
    fs = g["M.forest_share_shared_A"].to_numpy(float)
    ypt = (fs - YB[0]) * V_B
    # three deterministic placement orders; keep the most compact (x offsets are layout only)
    orders = {"ascending y": np.argsort(ypt, kind="stable"), "descending y": np.argsort(-ypt, kind="stable"),
              "centre out": np.argsort(np.abs(ypt - np.median(ypt)), kind="stable")}
    best = min(orders, key=lambda k: np.abs(beeswarm(ypt, SW_D, orders[k])).max())
    off = beeswarm(ypt, SW_D, orders[best])
    dd = np.sqrt((off[:, None] - off[None, :]) ** 2 + (ypt[:, None] - ypt[None, :]) ** 2)
    np.fill_diagonal(dd, np.inf)
    assert dd.min() >= SW_D - 1e-4, dd.min()          # no two dots overlap
    OFF[r] = off
    PB[r] = dict(g=g, fs=fs, med=float(np.median(fs)), nul=float((g["M.n_forest_A"] / (m - 1)).median()))
    SWARM[DEPTH[r]] = {"order": best, "step_pt": SW_D, "min_centre_distance_pt": float(dd.min()),
                       "x_extent_pt": [float(off.min()), float(off.max())]}
# x scale of (b) from the swarm extents (pt): left swarm | GAP_SW | right swarm, PAD_SW to the axes edges
R_SW = SW_D / 2
A0, B0 = -OFF[0.5].min() + R_SW, OFF[0.5].max() + R_SW
A1, B1 = -OFF[1.0].min() + R_SW, OFF[1.0].max() + R_SW
U_B = B0 + A1 + GAP_SW                     # points per x data unit (category centres at 0 and 1)
XLB = (-(A0 + PAD_SW) / U_B, 1 + (B1 + PAD_SW) / U_B)
WB = (XLB[1] - XLB[0]) * U_B / 72
axB = axes_in(XB, Y0, WB, AH)
for i, r in enumerate((0.5, 1.0)):
    PB[r]["x"] = i + OFF[r] / U_B
    SWARM[DEPTH[r]]["x_extent_data"] = [float(OFF[r].min() / U_B), float(OFF[r].max() / U_B)]
    axB.plot(PB[r]["x"], PB[r]["fs"], zorder=Z["interval"], clip_on=False, **mk_swarm(r))
# null (NULL_LINE dashed 1.0) and median (INK 2.2) bars, both drawn UNDER the swarm with no white underlay, so every
# dot stays whole on top of them (round 1 drew the median bar above the swarm on a 3.2 pt white underlay, which hid
# the dots tied at the median). The null bars keep the round-1 half width HB_PT (the narrowest swarm half-extent; at
# the null height the swarm is much narrower, so the bar reaches well past it on both sides). Each median bar runs
# MED_EXT pt past the outer edge of the outermost dot that touches it, on each side, and is never shorter than the
# null bar; clearances to the axes edges, the other swarm and the other bars are asserted below.
HB_PT = float(np.floor(2 * min(A0, B0, A1, B1) - 2 * R_SW) / 2)
HB = HB_PT / U_B
assert PB[0.5]["x"].max() + R_SW / U_B < 1 - HB - 2.0 / U_B and PB[1.0]["x"].min() - R_SW / U_B > HB + 2.0 / U_B
MED_EXT = 3.0                                          # pt past the outer edge of the outermost dot touching the bar
Z_NULL_B, Z_MED_B = Z["ref"] + 0.3, Z["ref"] + 0.6     # both below the swarm (drawn at Z['interval'])
assert max(Z_NULL_B, Z_MED_B) < Z["interval"]
BARS, BAR_SEG = {}, []                                 # BAR_SEG: (depth index, kind, y, x0_pt, x1_pt, lw) in pt from (b)'s x = 0
for i, r in enumerate((0.5, 1.0)):
    med, nul = PB[r]["med"], PB[r]["nul"]
    ypt_r = (PB[r]["fs"] - YB[0]) * V_B
    touch = np.abs(ypt_r - (med - YB[0]) * V_B) < R_SW + LW["median_bar"] / 2      # dots whose disc meets the bar
    lo_pt = min(-HB_PT, float(OFF[r][touch].min()) - R_SW - MED_EXT)
    hi_pt = max(HB_PT, float(OFF[r][touch].max()) + R_SW + MED_EXT)
    BARS[DEPTH[r]] = {
        "median": {"y": med, "x_pt_from_category_centre": [lo_pt, hi_pt], "lw": LW["median_bar"],
                   "n_dots_touching": int(touch.sum()),
                   "outermost_touching_dot_edges_pt": [float(OFF[r][touch].min()) - R_SW,
                                                       float(OFF[r][touch].max()) + R_SW],
                   "n_dots_exactly_at_median": int(np.sum(np.abs(PB[r]["fs"] - med) < 1e-12))},
        "null": {"y": nul, "x_pt_from_category_centre": [-HB_PT, HB_PT], "lw": LW["ref"]}}
    axB.plot([i - HB, i + HB], [nul, nul], color=NL, lw=LW["ref"], ls=LS["dashed"], zorder=Z_NULL_B)
    axB.plot([i + lo_pt / U_B, i + hi_pt / U_B], [med, med], color=MST_MED, lw=LW["median_bar"], zorder=Z_MED_B,
             solid_capstyle="butt")
    BAR_SEG += [(i, "null", nul, i * U_B - HB_PT, i * U_B + HB_PT, LW["ref"]),
                (i, "median", med, i * U_B + lo_pt, i * U_B + hi_pt, LW["median_bar"])]
# clearances (pt): bar ends inside the axes; bar ends clear of the other depth's dots at the bar's height; bars of the
# two depths at nearly the same height (the two nulls are 0.8 pt apart) clearly separated
BAR_CLEAR = {"to_axes_edge_min_pt": None, "to_other_swarm_min_pt": None, "between_depths_min_pt": None}
XL_PT = (XLB[0] * U_B, XLB[1] * U_B)
BAR_CLEAR["to_axes_edge_min_pt"] = float(min(min(s[3] - XL_PT[0], XL_PT[1] - s[4]) for s in BAR_SEG))
assert BAR_CLEAR["to_axes_edge_min_pt"] >= 1.0, BAR_SEG
oth = []
for i_, kind, y_, x0, x1, lw_ in BAR_SEG:
    ro = (1.0, 0.5)[i_]
    io = 1 - i_
    yo = (PB[ro]["fs"] - y_) * V_B
    near = np.abs(yo) < R_SW + lw_ / 2
    if near.any():
        xo = io * U_B + OFF[ro][near]
        oth.append(float(np.min(np.where(xo > x1, xo - R_SW - x1, x0 - (xo + R_SW)))))
BAR_CLEAR["to_other_swarm_min_pt"] = float(min(oth))
assert BAR_CLEAR["to_other_swarm_min_pt"] >= 3.0, oth
btw = [max(s[3], t[3]) - min(s[4], t[4]) for s in BAR_SEG for t in BAR_SEG     # > 0: gap between the x ranges
       if s[0] < t[0] and abs(s[2] - t[2]) * V_B < (s[5] + t[5]) / 2 + 2.0]
BAR_CLEAR["between_depths_min_pt"] = float(min(btw)) if btw else None
assert not btw or min(btw) >= 3.0, btw
axB.axhline(LEVEL, color=NL, lw=LW["ref"], zorder=Z["ref"])
S.text(axB, XLB[0] + PAD_SW / U_B, LEVEL + (LW["ref"] / 2 + 1.0) / V_B, "preregistered level", "annot",
       va="bottom", ha="left")
assert PB[0.5]["x"].min() - R_SW / U_B > XLB[0] and PB[1.0]["x"].max() + R_SW / U_B < XLB[1]
axB.set_xlim(*XLB)
axB.set_ylim(*YB)
axB.set_xticks([0, 1], [DEPTH[0.5], DEPTH[1.0]])
axB.tick_params(axis="x", length=0, pad=3.5)
axB.set_yticks([0.7, 0.8, 0.9, 1.0], ["0.7", "0.8", "0.9", "1.0"])
axB.set_ylabel("forest share of shared MST edges")
# v6: x label of v3 F2 (b), with the pair count per depth (asserted); its row is aligned with the x label of (c) below
N_PAIRS_B = {DEPTH[r]: int(len(PB[r]["fs"])) for r in (0.5, 1.0)}
assert N_PAIRS_B == {"mid depth": 136, "last layer": 136} and NA["r=0.5"]["n_pairs"] == NA["r=1.0"]["n_pairs"] == 136
XLAB_B = None                     # v8: no x label (v6: 'depth (136 pairs each)'); the categories name the depth
# v5 values of (b) unchanged (medians and nulls of the bars)
for r in (0.5, 1.0):
    assert abs(V5["panel_b"]["median_bar"][DEPTH[r]] - PB[r]["med"]) <= 1e-12
    assert abs(V5["panel_b"]["null_bar"][DEPTH[r]] - PB[r]["nul"]) <= 1e-12
# key for the two bars. v5 set it on the row of the x labels; that row now holds the x label, so the key moves into
# the empty band between the bottom of the axes and the preregistered level (no pair lies below 0.76), centred
# under (b); handles at the drawn widths. 'median' bold INK, 'null' MUTED (spec F.2 b).
assert min(PB[r]["fs"].min() for r in (0.5, 1.0)) > LEVEL + 0.05
KEY_Y = (YB[0] + LEVEL) / 2
KEY_B = axB.legend([S.key_handle(color=MST_MED, lw=LW["median_bar"], solid_capstyle="butt"),
                    S.key_handle(color=NL, lw=LW["ref"], ls=LS["dashed"])], ["median", "null"],
                   ncol=2, loc="center", frameon=False, borderpad=0, borderaxespad=0,
                   bbox_to_anchor=((XLB[0] + XLB[1]) / 2, KEY_Y), bbox_transform=axB.transData)
for t_, kw_ in zip(KEY_B.get_texts(), (S.text_kw("direct", "median", color=INK), S.text_kw("annot", "null"))):
    t_.set(**kw_)

# (c) bridge overlap minus its permutation-null mean along BLOOMZ -------------------------------
xb = np.array(LM_PAR, float) / 1e9
band = max(BANDS.values())
DODGE = {0.5: 0.955, 1.0: 1.047}
axC.axhspan(0, band, color=NF, lw=0, zorder=Z["band"])
axC.axhline(0, color=NL, lw=LW["zero"], zorder=Z["ref"])
S.text(axC, 0.48, band + 0.0006, "null 95%", "annot", ha="left", va="bottom")
for r in (0.5, 1.0):
    mean, lo, hi = CUR[r]
    xx = xb * DODGE[r]
    axC.errorbar(xx, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor=G, elinewidth=LW["errorbar"],
                 zorder=Z["interval"])
    axC.plot(xx, mean, color=G, lw=LW["data"], zorder=Z["data"])
    axC.plot(xx, mean, zorder=Z["marker"], **mk_c(r))
mL, loL, hiL = CUR[1.0]
mM, loM, hiM = CUR[0.5]
axC.set_xscale("log")
axC.set_xlim(0.45, 8.6)
axC.set_xticks(xb, [f"{v:.2g}" for v in xb])
axC.minorticks_off()
axC.set_yticks([0, 0.02, 0.04, 0.06], ["0", "0.02", "0.04", "0.06"])
axC.set_ylabel("bridge overlap − null mean")

# v6 (c) labels, fixer round (critic of the v6 render, 2026-10-10; the maker's render is kept in _prev_r2/). The y range
# is back to v5's 0.06 (ticks unchanged), so the left spine ends at the top tick as in (a) and (b).
#   * last layer: its name over 'Tr = 0.57' (both bold, MST text tone) over the MUTED 7.5 pt note
#     'power-limited: / p >= 1/120' (two lines), left-aligned with the 'null 95%' label in the empty upper left, above
#     the rising part of its own curve (where v3 F2 (c) set its statistics). The note states a floor, the smallest exact
#     p that any trend over five sizes can reach, so it cannot be read as the observed p (the maker's 'smallest p =
#     1/120' directly under 'Tr = 0.57' could be).
#   * mid depth: its name over 'Tr = -0.16', right-aligned and set just above its own curve. The anchor moves left from
#     the last point (0.01 in steps) until the block's nearest data are its own curve's: the rightmost anchor at which
#     its clearance to the last-layer data is at least NEAR_RATIO times its clearance to the mid-depth data (in the
#     maker's render the block sat closer to the last-layer interval at 7.1B than to its own end point).
#   * each block sits at the lowest height, in 0.25 pt steps, at which every line keeps TARGET_PT from every marker,
#     line and interval (asserted >= CLEAR_PT after drawing) and CLEAR_PT from the other labels.
import matplotlib.text as mtext  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
INV_IN = fig.dpi_scale_trans.inverted()
CLEAR_PT = 2.0                                     # pt, asserted minimum: every (c) label against data and other labels
TARGET_PT = 2.5                                    # pt, placement target against data
STEP_PT = 0.25                                     # pt, vertical search step
NEAR_RATIO = 2.0                                   # mid-depth block: clearance to last-layer data / to own data
YTOP_C = 0.060                                     # v5 y range of (c)
X_LEFT_C = 0.48                                    # data x of the left edge of 'null 95%' (the last-layer block aligns)
TR_TXT = {r: "Tr = " + S.num(NC[f"r={r}"]["Tr_bridge"], 2) for r in (0.5, 1.0)}
assert TR_TXT == {0.5: "Tr = " + S.MINUS + "0.16", 1.0: "Tr = 0.57"}, TR_TXT
P_MIN_TXT = "1/%d" % round(1 / MIN_P_ONE)
assert P_MIN_TXT == "1/120"
NOTE_TXT = ("power-limited:", "$p$ ≥ " + P_MIN_TXT)


def bb_in(t):
    fig.canvas.draw()
    return mtext.Text.get_window_extent(t, fig.canvas.get_renderer()).transformed(INV_IN)


def c_obstacles(rs=(0.5, 1.0)):
    """markers (centre, outer radius) and polylines (densified, half width) of (c) for the depths rs, in inches."""
    pts_, segs_ = [], []
    for r in rs:
        mean, lo, hi = CUR[r]
        xx = xb * DODGE[r]
        P = INV_IN.transform(axC.transData.transform(np.column_stack([xx, mean])))
        mk = mk_c(r)
        pts_ += [(p[0], p[1], (mk["ms"] + mk["mew"]) / 2 / 72) for p in P]
        segs_.append((P, LW["data"] / 2 / 72))
        for x_, l_, h_ in zip(xx, lo, hi):
            segs_.append((INV_IN.transform(axC.transData.transform([[x_, l_], [x_, h_]])), LW["errorbar"] / 2 / 72))
    dense_pts = []
    for P, hw in segs_:
        for a_, b_ in zip(P[:-1], P[1:]):
            n_ = max(2, int(np.ceil(np.hypot(*(b_ - a_)) / 0.002)) + 1)
            Q = a_ + (b_ - a_) * np.linspace(0, 1, n_)[:, None]
            dense_pts += [(q[0], q[1], hw) for q in Q]
    return np.array(pts_ + dense_pts)


def clear_pt(bb, obs):
    """smallest distance in pt from bbox bb (inches) to the obstacles (x, y, radius); <= 0 means overlap."""
    dx = np.maximum.reduce([bb.x0 - obs[:, 0], np.zeros(len(obs)), obs[:, 0] - bb.x1])
    dy = np.maximum.reduce([bb.y0 - obs[:, 1], np.zeros(len(obs)), obs[:, 1] - bb.y1])
    return float(np.min(np.hypot(dx, dy) - obs[:, 2]) * 72)


def box_gap_pt(a, b):
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return float(np.hypot(dx, dy) * 72) if (dx > 0 or dy > 0) else -1.0


def y_data(y_in):
    return float(axC.transData.inverted().transform(fig.dpi_scale_trans.transform([[0, y_in]]))[0][1])


def y_in_of(y_d):
    return float(INV_IN.transform(axC.transData.transform([[1.0, y_d]]))[0][1])


def x_data(x_in):
    return float(axC.transData.inverted().transform(fig.dpi_scale_trans.transform([[x_in, 0]]))[0][0])


def x_in_of(x_d):
    return float(INV_IN.transform(axC.transData.transform([[x_d, 0.0]]))[0][0])


T_NULL = [t_ for t_ in axC.texts if t_.get_text() == "null 95%"]
assert len(T_NULL) == 1
axC.set_ylim(-0.002, YTOP_C)
fig.canvas.draw()
OBS_R = {r: c_obstacles((r,)) for r in (0.5, 1.0)}
OBS_C = np.vstack([OBS_R[0.5], OBS_R[1.0]])
ax_bb = axC.get_window_extent().transformed(INV_IN)
bb_null = bb_in(T_NULL[0])
assert abs(bb_null.x0 - x_in_of(X_LEFT_C)) * 72 < 0.05
C_SPEC = {  # key: (text, role, colour). v8: the depth names only; Tr values and the p floor are in the Results text
    "name|last layer": (DEPTH[1.0], "direct", GT),
    "name|mid depth": (DEPTH[0.5], "direct", GT),
}
BLOCK_LL = ["name|last layer"]
BLOCK_MD = ["name|mid depth"]
SIZE_IN = {}                                       # each line's box, measured once; x and y are then set analytically
for k_, (s_, role_, col_) in C_SPEC.items():
    t_ = S.text(axC, 1.0, 0.0, s_, role_, color=col_, ha="left", va="bottom")
    b_ = bb_in(t_)
    SIZE_IN[k_] = (float(b_.width), float(b_.height))
    t_.remove()


def stack(keys, x_in, ha, y_bottom_in):
    out, y_ = {}, y_bottom_in
    for k_ in keys:
        w_, h_ = SIZE_IN[k_]
        x0_ = x_in if ha == "left" else x_in - w_
        out[k_] = Bbox([[x0_, y_], [x0_ + w_, y_ + h_]])
        y_ = y_ + h_
    return out


def lowest(keys, x_in, ha, y_from_in, others=()):
    """lowest bottom (STEP_PT steps up from y_from_in) at which every line keeps TARGET_PT from all data of (c) and
    CLEAR_PT from the boxes in others, inside the axes; None if the block reaches the top first."""
    y_ = y_from_in
    while True:
        B = stack(keys, x_in, ha, y_)
        if max(b.y1 for b in B.values()) > ax_bb.y1 + 1e-9:
            return None
        if (min(b.x0 for b in B.values()) >= ax_bb.x0 and max(b.x1 for b in B.values()) <= ax_bb.x1
                and all(clear_pt(b, OBS_C) >= TARGET_PT for b in B.values())
                and all(box_gap_pt(b, o) >= CLEAR_PT for b in B.values() for o in others)):
            return y_, B
        y_ += STEP_PT / 72


# last-layer block: left-aligned with 'null 95%', searched upwards from the lowest last-layer mean
RES_LL = lowest(BLOCK_LL, x_in_of(X_LEFT_C), "left", y_in_of(float(mL.min())), others=[bb_null])
assert RES_LL is not None, "last-layer block does not fit under y = %.3f" % YTOP_C
Y_LL, BL = RES_LL
# mid-depth block: right-aligned; anchors from the last point leftwards in 0.01 in steps
X_MD_LAST = float(xb[-1] * DODGE[0.5])
MD_TRIALS, MD_CHOICE = [], None
for i_ in range(0, 60):
    xa_in = x_in_of(X_MD_LAST) - 0.01 * i_
    res_ = lowest(BLOCK_MD, xa_in, "right", y_in_of(float(mM.min())), others=[bb_null] + list(BL.values()))
    if res_ is None:
        MD_TRIALS.append({"anchor_x_data": x_data(xa_in), "fits": False})
        continue
    y_, B_ = res_
    d_own = min(clear_pt(b, OBS_R[0.5]) for b in B_.values())
    d_oth = min(clear_pt(b, OBS_R[1.0]) for b in B_.values())
    MD_TRIALS.append({"anchor_x_data": x_data(xa_in), "fits": True, "bottom_data": y_data(y_), "to_own_data_pt": d_own,
                      "to_last_layer_data_pt": d_oth, "ratio": d_oth / d_own})
    if d_oth >= NEAR_RATIO * d_own:
        MD_CHOICE = (xa_in, y_, B_, i_)
        break
assert MD_CHOICE is not None, MD_TRIALS
XA_MD_IN, Y_MD, BM, I_MD = MD_CHOICE
C_LABELS = {}
for k_ in BLOCK_LL:
    s_, role_, col_ = C_SPEC[k_]
    C_LABELS[k_] = S.text(axC, X_LEFT_C, y_data(BL[k_].y0), s_, role_, color=col_, ha="left", va="bottom")
X_MD_ANCHOR = x_data(XA_MD_IN)
for k_ in BLOCK_MD:
    s_, role_, col_ = C_SPEC[k_]
    C_LABELS[k_] = S.text(axC, X_MD_ANCHOR, y_data(BM[k_].y0), s_, role_, color=col_, ha="right", va="bottom")
# the drawn boxes equal the planned ones; consecutive lines touch (no overlap) and share their aligned edge
for k_, b_ in {**BL, **BM}.items():
    d_ = bb_in(C_LABELS[k_])
    assert max(abs(d_.x0 - b_.x0), abs(d_.x1 - b_.x1), abs(d_.y0 - b_.y0), abs(d_.y1 - b_.y1)) * 72 < 0.05, (k_, d_, b_)
for blk, edge in ((BLOCK_LL, "x0"), (BLOCK_MD, "x1")):
    for k0, k1 in zip(blk[:-1], blk[1:]):
        b0, b1_ = bb_in(C_LABELS[k0]), bb_in(C_LABELS[k1])
        assert abs(b1_.y0 - b0.y1) * 72 < 0.05 and abs(getattr(b1_, edge) - getattr(b0, edge)) * 72 < 0.05, (k0, k1)
fig.canvas.draw()
OBS_C = np.vstack([c_obstacles((0.5,)), c_obstacles((1.0,))])
C_CLEAR = {k_: {"to_data_pt": clear_pt(bb_in(t_), OBS_C)} for k_, t_ in C_LABELS.items()}
for k_ in C_CLEAR:
    C_CLEAR[k_]["to_other_labels_pt"] = min(box_gap_pt(bb_in(C_LABELS[k_]), bb_in(C_LABELS[j_]))
                                           for j_ in C_LABELS if j_ != k_ and not (k_ in BLOCK_LL and j_ in BLOCK_LL)
                                           and not (k_ in BLOCK_MD and j_ in BLOCK_MD))
    C_CLEAR[k_]["to_null_label_pt"] = box_gap_pt(bb_in(C_LABELS[k_]), bb_in(T_NULL[0]))
    assert C_CLEAR[k_]["to_data_pt"] >= CLEAR_PT and C_CLEAR[k_]["to_other_labels_pt"] >= CLEAR_PT, (k_, C_CLEAR[k_])
    assert C_CLEAR[k_]["to_null_label_pt"] >= CLEAR_PT, (k_, C_CLEAR[k_])
# the y range is v5's and the whole last-layer block lies inside it
assert tuple(axC.get_ylim()) == (-0.002, YTOP_C)
assert max(bb_in(C_LABELS[k_]).y1 for k_ in C_LABELS) <= ax_bb.y1 + 1e-9
MD_D_OWN = min(clear_pt(bb_in(C_LABELS[k_]), c_obstacles((0.5,))) for k_ in BLOCK_MD)
MD_D_OTH = min(clear_pt(bb_in(C_LABELS[k_]), c_obstacles((1.0,))) for k_ in BLOCK_MD)
assert MD_D_OTH >= NEAR_RATIO * MD_D_OWN, (MD_D_OWN, MD_D_OTH)
C_PLACE = {"ylim": [-0.002, YTOP_C], "ylim_maker_v6": [-0.002, 0.066],
           "last_layer_block": {"align": "left, with 'null 95%'", "x_left_data": X_LEFT_C,
                                "bottoms_data": {k_: y_data(BL[k_].y0) for k_ in BLOCK_LL},
                                "top_data": y_data(BL[BLOCK_LL[-1]].y1)},
           "mid_depth_block": {"align": "right", "anchor_x_data": X_MD_ANCHOR, "last_point_x_data": X_MD_LAST,
                               "anchor_steps_left_of_last_point_in": 0.01 * I_MD,
                               "bottoms_data": {k_: y_data(BM[k_].y0) for k_ in BLOCK_MD},
                               "to_own_data_pt": MD_D_OWN, "to_last_layer_data_pt": MD_D_OTH,
                               "trials": MD_TRIALS},
           "note_text": list(NOTE_TXT), "clearances_pt": C_CLEAR, "clear_pt_required": CLEAR_PT,
           "target_pt": TARGET_PT, "near_ratio": NEAR_RATIO,
           "rule": "y range 0.06 (v5). Last-layer block (name, Tr, note line 1, note line 2; top to bottom) left-aligned "
                   "with 'null 95%%'; mid-depth block (name, Tr) right-aligned at the rightmost anchor (0.01 in steps "
                   "left from its last point) at which its clearance to the last-layer data is at least %.1f times its "
                   "clearance to its own data. Each block at the lowest height (%.2f pt steps) at which every line keeps "
                   "%.1f pt from every marker, line and interval and %.1f pt from the other labels"
                   % (NEAR_RATIO, STEP_PT, TARGET_PT, CLEAR_PT)}
# x label with the family name in the family colour: lay out the plain label once, then replace it by a
# baseline-aligned two-colour pack anchored at the same place. Spec F.2 (c): try 'BLOOMZ parameters (B, log scale)'
# at 7.5 pt; it is used only if, centred under the axes, it stays inside the figure and right of the y label of (c).
XLAB_TRY = ("BLOOMZ", "parameters (B, log scale)", "label_small")
XLAB_V4 = ("BLOOMZ", "parameters (B)", "label")


def xlabel_pack(parts):
    fam, rest, role = parts
    return HPacker(children=[TextArea(fam, textprops=S.text_kw(role, fam, color=S.FAM["bloomz"])),
                             TextArea(rest, textprops=S.text_kw(role, rest))],
                   align="baseline", pad=0, sep=2.0 * S.FS[role] / 8.0)


axC.set_xlabel("BLOOMZ parameters (B)")
fig.canvas.draw()
REN = fig.canvas.get_renderer()
bbl = axC.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
axC.set_xlabel("")
YLAB_C_X0 = float(axC.yaxis.label.get_window_extent(REN).transformed(fig.dpi_scale_trans.inverted()).x0)
XC_MID = XC + WC / 2
_tmp = [fig.text(0, 0, s_, **S.text_kw(XLAB_TRY[2], s_)) for s_ in XLAB_TRY[:2]]
fig.canvas.draw()
_w_try = float(sum(t_.get_window_extent(REN).width for t_ in _tmp) / fig.dpi + 2.0 * S.FS[XLAB_TRY[2]] / 8.0 / 72)
for t_ in _tmp:
    t_.remove()
XLAB_FIT = {"text": "BLOOMZ parameters (B, log scale)", "size_pt": S.FS["label_small"], "width_in": _w_try,
            "centred_x0_in": XC_MID - _w_try / 2, "centred_x1_in": XC_MID + _w_try / 2,
            "limits_in": [YLAB_C_X0, W]}
XLAB_FIT["fits"] = bool(XLAB_FIT["centred_x0_in"] >= YLAB_C_X0 and XLAB_FIT["centred_x1_in"] <= W - 0.01)
XLAB_USED = XLAB_TRY if XLAB_FIT["fits"] else XLAB_V4
XLAB_PACK = AnchoredOffsetbox(loc="upper center", child=xlabel_pack(XLAB_USED), pad=0, borderpad=0, frameon=False,
                              bbox_to_anchor=((bbl.x0 + bbl.x1) / 2, bbl.y1), bbox_transform=fig.transFigure)
fig.add_artist(XLAB_PACK)
fig.canvas.draw()
# v8: the bold label is wider than the regular one; centred under (c) it ran 0.03 in past the right edge of the page,
# so it shifts left until its right end sits XLAB_RIGHT_PAD in inside the figure (it stays under (c))
XLAB_RIGHT_PAD = 0.02
_pb = XLAB_PACK.get_window_extent(REN).transformed(fig.dpi_scale_trans.inverted())
XLAB_SHIFT_IN = float(min(0.0, (W - XLAB_RIGHT_PAD) - _pb.x1))
if XLAB_SHIFT_IN < 0:
    XLAB_PACK.set_bbox_to_anchor(((bbl.x0 + bbl.x1) / 2 + XLAB_SHIFT_IN / W, bbl.y1), transform=fig.transFigure)
    fig.canvas.draw()
_pb = XLAB_PACK.get_window_extent(REN).transformed(fig.dpi_scale_trans.inverted())
assert _pb.x1 <= W - XLAB_RIGHT_PAD + 1e-6 and _pb.x0 >= XC - 0.15, (_pb, XC)
XLAB_FIT["v8_shift_left_in"] = XLAB_SHIFT_IN
XLAB_FIT["v8_drawn_x_in"] = [float(_pb.x0), float(_pb.x1)]
# v8: (b) has no x label, so nothing to align with the x label of (c)
YLAB_ROW_TOP_FIG = float(bbl.y1)
fig.canvas.draw()
bbp = XLAB_PACK.get_window_extent(REN).transformed(fig.transFigure.inverted())
assert abs(bbp.y1 - bbl.y1) * H * 72 < 0.05, (bbp, bbl)

# v8: bold tick labels and axis titles (bold math) before the letters are placed from the y-label extents
N_FINALIZED = S.finalize(fig)
# panel letters: 10 pt bold, top-left of each panel (left edge of the panel's y label, common top)
fig.canvas.draw()
REN = fig.canvas.get_renderer()
LET_Y = H - 0.03
LET_X = {}
for k_, ax in (("a", axA), ("b", axB), ("c", axC)):
    yl = ax.yaxis.label.get_window_extent(REN).transformed(fig.dpi_scale_trans.inverted())
    LET_X[k_] = max(0.02, float(yl.x0) - 0.02)
    S.panel_letter(fig, k_, LET_X[k_], LET_Y)

# every line in (b) (level, nulls, medians) lies under the swarm, so no dot is hidden or cut by a line
B_LINES = [(ln.get_zorder(), ln.get_linewidth()) for ln in axB.get_lines() if ln.get_marker() in ("None", "none", "", None)]
B_SWARM_Z = [ln.get_zorder() for ln in axB.get_lines() if ln.get_marker() == "o"]
assert len(B_SWARM_Z) == 2 and len(B_LINES) == 5 and max(z for z, _ in B_LINES) < min(B_SWARM_Z), (B_LINES, B_SWARM_Z)
assert not any(ln.get_color() == "white" for ln in axB.get_lines()), "no white underlay in (b)"

PDF, PNG = os.path.join(OUT, "F_mst_app.pdf"), os.path.join(OUT, "F_mst_app.png")
V100, VIEWS = os.path.join(OUT, "_view_print_100dpi.png"), os.path.join(OUT, "_views_cvd_grey.png")
fig.savefig(PDF)
CHK = S.check(fig, PDF)                          # v8: also bold ticks / axis titles, bold math, a chromatic colour
S.print_view(PDF, PNG, V100)                     # v8: staged PNG = pdftoppm render of the PDF at 200 dpi
S.views(PNG, VIEWS)
assert abs(CHK["size_in"][0] - 5.5) < 1e-9 and abs(CHK["size_in"][1] - 2.1) < 1e-9
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
font_rows = [ln.split() for ln in pf.splitlines()[2:] if ln.strip()]
assert font_rows and all(row[-5] == "yes" for row in font_rows), pf          # emb column
assert not any("Type 3" in ln for ln in pf.splitlines()), pf
assert all("Liberation" in row[0] or row[0].split("+")[-1] in ("Cmr10", "Cmmi10", "Cmsy10") for row in font_rows), pf


# layout checks ------------------------------------------------------------------------------------
import matplotlib.text as mtext  # noqa: E402
fig.canvas.draw()
REN = fig.canvas.get_renderer()
INV = fig.dpi_scale_trans.inverted()


def text_boxes():
    out = []
    for t in fig.findobj(mtext.Text):
        if t.get_visible() and t.get_text().strip():
            out.append((t.get_text(), mtext.Text.get_window_extent(t, REN).transformed(INV), t))  # text only, no arrow
    return out


TB = text_boxes()
# (1) text against text, anywhere in the figure (panel letters, tick labels, keys and the x-label pack included)
clash = [(a, b) for i, (a, ba, _) in enumerate(TB) for (b, bb, _) in TB[i + 1:] if ba.overlaps(bb)
         and ba.intersection(ba, bb).width > 0.005 and ba.intersection(ba, bb).height > 0.005]
assert not clash, clash
# (2) every text inside the figure
outside = [(a, [round(v, 3) for v in bb.extents]) for a, bb, _ in TB
           if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]
assert not outside, outside
# (3) text against data markers (the outer marker disc, edge included, must stay clear of every text box)
mark_hits = []
for ax in (axA, axB, axC):
    pts_in = []
    for ln in ax.get_lines():
        if ln.get_marker() in ("o", "x") and ln.get_visible():
            xy = ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float))
            xy = INV.transform(xy)
            rad = (ln.get_markersize() + ln.get_markeredgewidth()) / 2 / 72
            pts_in.append((xy, rad))
    for s_, bb, t in TB:
        ang = float(t.get_rotation()) % 360
        if ang:
            # rotated text: test against the rotated glyph rectangle, not its axis-aligned hull
            tmp = fig.text(0, 0, t.get_text(), fontsize=t.get_fontsize(), fontproperties=t.get_fontproperties())
            ub = tmp.get_window_extent(REN).transformed(INV)
            tmp.remove()
            cxy = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
            th = np.radians(ang)
            rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        for xy, rad in pts_in:
            if ang:
                q = (xy - cxy) @ rot.T
                hit = (np.abs(q[:, 0]) < ub.width / 2 + rad) & (np.abs(q[:, 1]) < ub.height / 2 + rad)
            else:
                hit = ((xy[:, 0] > bb.x0 - rad) & (xy[:, 0] < bb.x1 + rad) & (xy[:, 1] > bb.y0 - rad) & (xy[:, 1] < bb.y1 + rad))
            if hit.any():
                mark_hits.append((s_, int(hit.sum())))
assert not mark_hits, mark_hits
# (4) the three panels (data area plus its labels, keys and the x-label pack) do not overlap one another
PAN = {k_: ax.get_tightbbox(REN).transformed(INV) for k_, ax in (("a", axA), ("b", axB), ("c", axC))}
PACK_BB = XLAB_PACK.get_window_extent(REN).transformed(INV)
PAN_C_X0 = min(PAN["c"].x0, PACK_BB.x0)
assert PAN["b"].x0 - PAN["a"].x1 >= MIN_PANEL_GAP, ("a", "b", PAN["a"].x1, PAN["b"].x0)
assert PAN_C_X0 - PAN["b"].x1 >= MIN_PANEL_GAP, ("b", "c", PAN["b"].x1, PAN_C_X0)
# (5) the two category names of (b) keep a clear gap
tl = [t.get_window_extent(REN).transformed(INV) for t in axB.get_xticklabels() if t.get_text()]
GAP_B = float(tl[1].x0 - tl[0].x1)
assert GAP_B > 0.08, GAP_B
# (6) text against drawn lines: data lines, reference lines, underlays, error bars and leaders. A text is not tested
#     against its own leader. Dashed and dotted lines are tested as solid (conservative).
from matplotlib.collections import LineCollection  # noqa: E402


def drawn_polylines(ax):
    out = []
    for ln in ax.get_lines():
        if not ln.get_visible() or str(ln.get_linestyle()).lower() in ("none", ""):
            continue
        xy = np.asarray(ln.get_xydata(), float)
        if len(xy) >= 2:
            out.append(("line", ln.get_transform().transform(xy), None))
    for c in ax.collections:
        if isinstance(c, LineCollection) and c.get_visible():
            for seg in c.get_segments():
                out.append(("errorbar", c.get_transform().transform(np.asarray(seg, float)), None))
    for t in ax.texts:
        if isinstance(t, mtext.Annotation) and t.arrow_patch is not None:
            ap = t.arrow_patch
            out.append(("leader", ap.get_transform().transform_path(ap.get_path()).vertices, t))
    return out


def dense(P, step_px=0.25):
    pts = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(np.ceil(np.hypot(*(b - a)) / step_px)) + 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(pts)


line_hits = []
for ax in (axA, axB, axC):
    PL = [(kind, INV.transform(dense(P)), owner) for kind, P, owner in drawn_polylines(ax)]
    for s_, bb, t in TB:
        ang = float(t.get_rotation()) % 360
        if ang:
            tmp = fig.text(0, 0, t.get_text(), fontsize=t.get_fontsize(), fontproperties=t.get_fontproperties())
            ub = tmp.get_window_extent(REN).transformed(INV)
            tmp.remove()
            cxy = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
            th = np.radians(ang)
            rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        for kind, Q, owner in PL:
            if owner is t:
                continue
            if ang:
                q = (Q - cxy) @ rot.T
                hit = (np.abs(q[:, 0]) < ub.width / 2) & (np.abs(q[:, 1]) < ub.height / 2)
            else:
                hit = (Q[:, 0] > bb.x0) & (Q[:, 0] < bb.x1) & (Q[:, 1] > bb.y0) & (Q[:, 1] < bb.y1)
            if hit.any():
                line_hits.append((s_, kind))
assert not line_hits, line_hits
# (7) clearance of the 'chance' label below the bound line of (a), in points (from the bound's centre line)
bbc = mtext.Text.get_window_extent(ann_ch, REN).transformed(INV)
x_tl = axA.transData.inverted().transform(fig.dpi_scale_trans.transform([[bbc.x0, bbc.y1]]))[0]
y_bound_at = axA.transData.transform([[x_tl[0], slope_b * x_tl[0]]])
CHANCE_CLEAR_PT = float((INV.transform(y_bound_at)[0][1] - bbc.y1) * 72)
assert CHANCE_CLEAR_PT >= 3.0 + LW["theory"] / 2, CHANCE_CLEAR_PT
# (8) v6: the key of (b) sits inside the axes, between the x axis and the preregistered level, with clear gaps
KEY_BB = KEY_B.get_window_extent(REN).transformed(INV)
AXB_BB = axB.get_window_extent(REN).transformed(INV)
Y_LEVEL_IN = float(INV.transform(axB.transData.transform([[0, LEVEL]]))[0][1])
KEY_GAPS_PT = {"to_x_axis": float((KEY_BB.y0 - AXB_BB.y0) * 72 - LW["spine"] / 2),
               "to_level_line": float((Y_LEVEL_IN - KEY_BB.y1) * 72 - LW["ref"] / 2),
               "to_axes_left": float((KEY_BB.x0 - AXB_BB.x0) * 72), "to_axes_right": float((AXB_BB.x1 - KEY_BB.x1) * 72)}
assert min(KEY_GAPS_PT.values()) >= 1.5, KEY_GAPS_PT
KEY_GAP_IN = KEY_GAPS_PT
LAYOUT = {"panel_gaps_in": {"a|b": float(PAN["b"].x0 - PAN["a"].x1), "b|c": float(PAN_C_X0 - PAN["b"].x1)},
          "text_line_overlaps": line_hits, "chance_label_clearance_below_bound_pt": CHANCE_CLEAR_PT,
          "panel_b_category_gap_in": GAP_B, "panel_b_key_gaps_pt": KEY_GAPS_PT,
          "axes_in": {"a": [XA, Y0, AH, AH], "b": [XB, Y0, WB, AH], "c": [XC, Y0, WC, AH]},
          "panel_letters_in": {k_: [LET_X[k_], LET_Y] for k_ in LET_X},
          "text_text_overlaps": clash, "text_marker_overlaps": mark_hits,
          "panel_c_xlabel_log_scale_trial": XLAB_FIT,
          "panel_b_bar_clearances_pt": BAR_CLEAR, "panel_b_lines_under_swarm": {"lines_z_lw": B_LINES, "swarm_z": B_SWARM_Z}}


# =============================================================================================
# numbers.json: every plotted or quoted value, with its source file and key
# =============================================================================================
def pts(g, cols):
    return [{c: (float(v) if isinstance(v, (float, np.floating)) else str(v)) for c, v in zip(cols, row)}
            for row in g[cols].itertuples(index=False)]


SRC_POOL = {"path": P_POOL, "filter": f"pre == '{PRE}' (272 rows = 136 pairs x 2 depths; L.m == 2047)"}
N = {
    "figure": {"id": "F_mst_app", "label": "fig:exp-mst", "render": "v8 (hrh_style_v8: Tol palette, bold ticks and axis titles): the v6 drawing without the on-panel statistics of (c) (Tr values, p floor) and without the x label of (b); depth names of (c) placed by the v6 search", "size_spec_in": [5.5, 2.10], "switches": {"FM_FILL_EDGE": FILL_EDGE, "FM_LEADER": LEADER, "FM_TRIAL": TRIAL},
               "v4": "arxiv_v4.tex, figure with \\label{fig:exp-mst} (Figure F.2; currently includes figures/exp_v3/F2.pdf)",
               "size_in": CHK["size_in"], "min_text_pt": CHK["min_text_pt"], "min_glyph_pt": CHK["min_glyph_pt"],
               "n_type3": CHK["n_type3"], "check": CHK,
               "pdffonts": pf.splitlines(), "layout": LAYOUT},
    "inputs": {p: sha(p) for p in (P_POOL, P_MAIN, P_SUM, P_PLAN, P_REPORT, P_OLD, P_V5, P_V5_CODE)},
    "style": {"path": STYLE_PATH, "sha256": STYLE_SHA, "version": S.STYLE_VERSION},
    "script_sha256": sha(os.path.abspath(__file__)),
    "v5_drawing": {
        "a": {"points": {"mid depth": mk_a(0.5), "last layer": mk_a(1.0)},
              "bound": {"lw": LW["theory"], "color": "INK"}, "diagonal": {"lw": LW["ref"], "ls": "dotted", "color": "NULL_LINE"},
              "chance": {"marker": "x", "ms": CH_MS, "mew": CH_MEW, "color": "NULL_LINE", "leader": [LW["leader"], "MUTED"],
                         "label": "MUTED, declared on NULL_FILL"},
              "labels": {"last layer / mid depth": ["bold", S.FS["direct"], GT], "bound": ["bold", S.FS["theory_label"], "INK"],
                         "chance / excluded / y=x": ["normal", S.FS["annot"], "MUTED"]},
              "mid_depth_leader": [LW["leader"], "INK" if LEADER == "ink" else "MST text tone %s" % GT],
              "filled_edge": FILL_EDGE, "halo_pass": mk_a_halo() if FILL_EDGE == "halo" else None},
        "b": {"swarm": {"mid depth": mk_swarm(0.5), "last layer": mk_swarm(1.0), "outer_diameter_pt": SW_D,
                        "spec_value_not_used": {"MS['swarm']": MS["swarm"], "why": SWARM_NOTE}},
              "median": {"lw": LW["median_bar"], "color": "MST_MED %s (v8 c; critic-v8: INK)" % MST_MED, "zorder": Z_MED_B, "under_swarm": True,
                         "white_underlay": None, "ext_past_outermost_touching_dot_pt": MED_EXT},
              "null": {"lw": LW["ref"], "ls": "dashed", "color": "NULL_LINE", "zorder": Z_NULL_B, "under_swarm": True,
                       "white_underlay": None},
              "swarm_zorder": Z["interval"],
              "level": {"lw": LW["ref"], "color": "NULL_LINE"},
              "key": "v6: inside the axes, centred in the empty band between the x axis and the preregistered level (y = %.3f): 'median' bold INK, 'null' MUTED (v5: on the x-label row, now taken by the x label)" % KEY_Y,
              "x_label": {"text": XLAB_B, "source": "v3 F2 (b); 136 = rows per depth of %s (asserted)" % P_POOL,
                          "row": "same top as the x label of (c)"},
              "ylim_v4": [0.6, 1.0], "ylim_v5": list(YB)},
        "c": {"markers": {"mid depth": mk_c(0.5), "last layer": mk_c(1.0)}, "line_lw": LW["data"],
              "errorbar_lw": LW["errorbar"], "zero_lw": LW["zero"], "labels": ["bold", S.FS["direct"], GT],
              "v6_labels": {"depth names": ["role direct", "bold", S.FS["direct"], GT],
                            "Tr values": ["role number", "bold", S.FS["number"], GT],
                            "note": ["role annot", "normal", S.FS["annot"], "MUTED"],
                            "placement": C_PLACE},
              "x_label_used": " ".join(XLAB_USED[:2]) + " at %g pt" % S.FS[XLAB_USED[2]],
              "x_label_log_scale_trial": XLAB_FIT},
        "letters": {"size_pt": S.FS["letter"], "weight": "bold"}},
    "crosscheck_v5": {"path": P_V5, "sha256": sha(P_V5), "code": P_V5_CODE, "code_sha256": sha(P_V5_CODE),
                      "per_depth": V5_CHECK,
                      "also": "panel_b median_bar / null_bar and panel_c null_band.drawn_upper equal within 1e-12; "
                              "smallest attainable exact one-sided p = 1/120"},
    "crosscheck_report": {"path": P_REPORT, "sha256": sha(P_REPORT), "sections": "3 (PE2, m = 2047 column) and 5 (Holm)",
                          "per_depth": REPORT_CHECK},
    "crosscheck_v3": {"path": P_OLD, "sha256": sha(P_OLD),
                      "keys": ["panel_a", "panel_b", "panel_c", "bound_check_all_configs"],
                      "result": "every recomputed value equal to F2/numbers.json within 1e-12 (asserted)"},
    "setting": {"pool": "panel_pool2047 (m = 2047, designated PE2 pool)", "pre": PRE, "forest_reading": READING,
                "depths": {"0.5": "mid depth", "1.0": "last layer"}, "m": m,
                "K_null": {"value": K_NULL, "source": f"{P_POOL} column K (all 272 rows)"},
                "bound_check_rows_by_pool_and_pre": POOL_COUNTS},
    "panel_a": {
        "x": {"label": "1-NN overlap s_{N,1}", **SRC_POOL, "column": "L.s_N_k1_for_mst"},
        "y": {"label": "MST overlap s_T", **SRC_POOL, "column": "L.s_T"},
        "points": {DEPTH[r]: pts(d[d.r == r].sort_values(["lm", "vit"]).rename(columns={"sN1": "s_N1", "sT": "s_T"}),
                                 ["lm", "vit", "s_N1", "s_T"]) for r in (0.5, 1.0)},
        "bound_line": {"formula": "s_T = m s_N1 / (2 (m - 1)), the half term of max{m s_N1 - n_mut_A, m s_N1 / 2} / (m - 1)",
                       "slope": slope_b, "m": m,
                       "why_half_term_only": "on panel_pool2047 min 2 n_mut_A / m = %.4f > axis limit 0.09, so the other branch is below the half term at every drawn s_N1 for every pair" % min_2nmut_pool,
                       "min_2nmut_over_m_pool2047": min_2nmut_pool},
        "excluded_region": "below the bound line (NULL_FILL)",
        "diagonal": "s_T = s_N1 (dotted NULL_LINE), labelled y = x as in F_neighbors (c)",
        "chance_mark": {"x_1_over_m_minus_1": cx, "colour": "NULL_LINE (mark), MUTED (leader, label)",
                        "label_colour": "MUTED", "label_xy_data": list(CH_XY),
                        "y_median_perm_null_mean_sT": cy,
                        "source_y": {**SRC_POOL, "column": "cal.s_T.null_mean", "stat": "median over 272 rows"}},
        "direct_labels": {"last layer": {"text_right_x": XL_R, "text_center_y": YL,
                                         "rule": "right-aligned left of the leftmost point within the text's height band (all last layer)"},
                          "mid depth": {"text_left_bottom": [XM, YM], "leader_from": [LX0, LY0],
                                        "leader_to_open_point": {"lm": str(tgt.lm), "vit": str(tgt.vit), "s_N1": TX, "s_T": TY},
                                        "rule": "two lines in the empty upper left; leader to the open point with s_N1 <= 0.0115 and largest s_T"}},
        "axes": {"xlim": [0, LIM], "ylim": [0, LIM], "max_plotted_sN1": float(d.sN1.max()), "max_plotted_sT": float(d.sT.max())},
        "summary_from_F2": {k: NA[k] for k in NA},
        "bound_check_all_configs": BOUND,
    },
    "panel_b": {
        "y": {"label": "forest share of shared MST edges", **SRC_POOL, "column": "M.forest_share_shared_A"},
        "x_categories": ["mid depth", "last layer"],
        "points": {DEPTH[r]: pts(PB[r]["g"].rename(columns={"M.forest_share_shared_A": "forest_share",
                                                            "M.n_forest_A": "n_forest_A"}),
                                 ["lm", "vit", "forest_share", "n_forest_A"]) for r in (0.5, 1.0)},
        "median_bar": {DEPTH[r]: PB[r]["med"] for r in (0.5, 1.0)},
        "null_bar": {DEPTH[r]: PB[r]["nul"] for r in (0.5, 1.0)},
        "null_source": {**SRC_POOL, "column": "M.n_forest_A / (m - 1)", "stat": "median over the 136 pairs per depth"},
        "preregistered_level": {"value": LEVEL, "source": f"{P_PLAN} line 98: {PLAN_LINE}",
                                "frac_pairs_ge_level": {DEPTH[r]: NBd[f"r={r}"]["frac_pairs_forest_share_ge_0.70"] for r in (0.5, 1.0)},
                                "frac_from_summary": {DEPTH[r]: SUM["pe2"][f"m2047|r={r}|{PRE}|{READING}"]["frac_forest_ge_0.70"] for r in (0.5, 1.0)}},
        "min_forest_share": {DEPTH[r]: NBd[f"r={r}"]["min_forest_share_of_shared"] for r in (0.5, 1.0)},
        "axes": {"ylim": list(YB), "xlim": list(XLB),
                 "n_points_at_1.0": {DEPTH[r]: int((PB[r]["fs"] >= 1.0 - 1e-12).sum()) for r in (0.5, 1.0)}},
        "swarm": {"rule": "greedy beeswarm in points, y exact; x offsets are layout only; centre-to-centre distance "
                          ">= step = outer marker diameter, so no two dots overlap (asserted); of three deterministic "
                          "placement orders the most compact is kept; the x scale is set from the two swarm extents "
                          "(gap %.1f pt between the swarms, %.1f pt to the axes edges)" % (GAP_SW, PAD_SW),
                  "points_per_x_unit": U_B, "points_per_y_unit": V_B, "per_depth": SWARM,
                  "markers": {"mid depth": mk_swarm(0.5), "last layer": mk_swarm(1.0)}, "note": SWARM_NOTE},
        "bars": {"null_half_width_data": HB, "null_half_width_pt": HB_PT, "per_depth": BARS, "clearances_pt": BAR_CLEAR,
                 "median": "MST_MED (v8 c; critic-v8 INK) %.1f pt, under the swarm, no underlay; runs %.1f pt past the outer edge of the outermost "
                           "dot touching it on each side (never shorter than the null bar)" % (LW["median_bar"], MED_EXT),
                 "null": "NULL_LINE dashed %.1f pt, under the swarm, no underlay; half width %.1f pt" % (LW["ref"], HB_PT),
                 "labels": "key under the category names: 'median' bold INK (handle MST_MED, v8 c), 'null' MUTED"},
        "every_pair_above_level": {DEPTH[r]: bool(NBd[f"r={r}"]["frac_pairs_forest_share_ge_0.70"] == 1.0) for r in (0.5, 1.0)},
        "summary_from_F2": {k: NBd[k] for k in NBd},
    },
    "panel_c": {
        "y": {"label": "bridge overlap - null mean", **SRC_POOL, "column": "cal.s_T_bridge.uncensored",
              "rows": "lm_family == 'BLOOMZ' (5 sizes x 17 vision models per depth)",
              "stat": "mean over the 17 vision models; 95% bootstrap interval over vision models (B = 10000, seed 20261009)"},
        "x": {"label": " ".join(XLAB_USED[:2]), **SRC_POOL, "column": "lm_params / 1e9", "scale": "log",
              "dodge_factor": {DEPTH[r]: DODGE[r] for r in (0.5, 1.0)}},
        "curves": {DEPTH[r]: {"lm": LM_ORDER, "params_B": xb.tolist(),
                              "mean": CUR[r][0].tolist(), "ci95_lo": CUR[r][1].tolist(), "ci95_hi": CUR[r][2].tolist()}
                   for r in (0.5, 1.0)},
        "null_band": {"drawn_upper": band, "drawn_lower": 0.0,
                      "meaning": "median over the 85 BLOOMZ pairs of the per-pair 95% permutation quantile minus the "
                                 "per-pair null mean (c_alpha - null mean, n_pi = 500); the larger depth is drawn. It is "
                                 "a conservative reference, not the null of the plotted mean over 17 vision models",
                      "per_depth_median_c_alpha_minus_null_mean": {DEPTH[r]: BANDS[r] for r in (0.5, 1.0)},
                      "source": {**SRC_POOL, "columns": "cal.s_T_bridge.c_alpha - cal.s_T_bridge.null_mean",
                                 "stat": "median over the 85 BLOOMZ pairs per depth; the larger of the two depths is drawn"}},
        "n_bloomz_pairs_per_depth": N_BLOOMZ_PAIRS,
        "trend_rule_conditions_c_d": TREND_RULE,
        "statistics_for_caption": {
            DEPTH[r]: {"Tr_bridge": NC[f"r={r}"]["Tr_bridge"], "p_one_exact": NC[f"r={r}"]["p_one_exact"],
                       "p_one_holm": NC[f"r={r}"]["p_one_holm"], "n_perm": 120,
                       "p_two_exact": NC[f"r={r}"]["p_two_exact"], "p_two_holm": NC[f"r={r}"]["p_two_holm"],
                       "verdict": NC[f"r={r}"]["verdict"],
                       "source": {"Tr, p": f"{P_SUM} pe2['m2047|r={r}|{PRE}|{READING}']['Tr_bridge'] (recomputed and asserted equal)",
                                  "Holm": f"{P_SUM} holm['p_one']['holm']['PE2|r={r}|m2047']"}}
            for r in (0.5, 1.0)},
    },
}
# v6: numbers printed on the panels (each with its source). v8: none is drawn any more; the v6 record is kept under
# on_panel_numbers_v6_removed, and the Spanning Trees paragraph of app:exp-s12 states each value ------------------
SRC_TR = {r: {"path": P_SUM, "key": f"pe2['m2047|r={r}|{PRE}|{READING}']['Tr_bridge']['Tr']",
              "recomputed": "trend() on cal.s_T_bridge.uncensored of the 85 BLOOMZ rows of %s at r = %s" % (P_POOL, r),
              "also_equal": [f"{P_V5} panel_c.statistics_for_caption['{DEPTH[r]}'].Tr_bridge",
                             f"{P_REPORT} section 3, r = {r}, row '(c/d) bridge-overlap trend Tr', m = 2047 column"]}
          for r in (0.5, 1.0)}
N["on_panel_numbers_v6_removed"] = {
    "(c) " + TR_TXT[1.0]: {"value": NC["r=1.0"]["Tr_bridge"], "printed": TR_TXT[1.0], **SRC_TR[1.0]},
    "(c) " + TR_TXT[0.5]: {"value": NC["r=0.5"]["Tr_bridge"], "printed": TR_TXT[0.5], **SRC_TR[0.5]},
    "(c) power-limited: p ≥ 1/120": {
        "value": MIN_P_ONE, "printed": "p ≥ " + P_MIN_TXT,
        "path": P_V5, "key": "panel_c.trend_rule_conditions_c_d.smallest_attainable_exact_p_one",
        "why": "a floor, not the observed p: the smallest exact one-sided p over the 120 relabellings of the five "
               f"BLOOMZ sizes (5! = 120); {P_REPORT} section 3 '(120 perms; unadjusted)'. The observed last-layer p "
               "(0.058, Holm 0.23) is in results_text_numbers"},
    "(b) 136 pairs each": {"value": N_PAIRS_B, "path": P_POOL, "key": "rows per r with pre == '%s'" % PRE},
}
# v6 fixer: forest fraction of the language-model tree per configuration (the appendix sentence '84 to 85% across
# configurations' holds for prh_clip+l2 only); the designated-pool medians must equal the null bars of (b)
FF = {}
for p_ in (P_POOL, P_MAIN):
    x_ = pd.read_parquet(p_)
    x_ = x_.assign(ff=x_["M.n_forest_A"] / (x_["L.m"] - 1))
    for (pre_, r_), g_ in x_.groupby(["pre", "r"]):
        FF[f"m={int(g_['L.m'].iloc[0])}|{pre_}|{DEPTH[float(r_)]}"] = {
            "median": float(g_.ff.median()), "min": float(g_.ff.min()), "max": float(g_.ff.max()), "n_pairs": int(len(g_))}
assert len(FF) == 6 and sum(v_["n_pairs"] for v_ in FF.values()) == BOUND["n_configs"] == 816
FF_MIN_ALL = min(v_["min"] for v_ in FF.values())
for r in (0.5, 1.0):
    assert abs(FF[f"m=2047|{PRE}|{DEPTH[r]}"]["median"] - PB[r]["nul"]) <= 1e-12
assert all(0.84 <= FF[f"m={mm_}|{PRE}|{DEPTH[r]}"]["median"] < 0.855 for mm_ in (1024, 2047) for r in (0.5, 1.0))
assert FF_MIN_ALL >= 0.82

CAPTION = (
    r"\caption{\textbf{Spanning-tree overlap mostly reads the one-nearest-neighbor scale "
    r"(Proposition~\ref{prop:mst}).} Vision--language pairs on the preregistered pool of $m=2047$ WIT inputs with "
    r"prh\_clip+l2, at mid depth (open markers) and the last layer (filled). The nearest-neighbor forest $F_A$ of the "
    r"language model lies inside its MST (Proposition~\ref{prop:mst}), and the bridges are the tree edges outside "
    r"$F_A$. (a) MST overlap $s_{\Tset}$ against one-nearest-neighbor overlap $s_{N,1}$, one point per pair, with the "
    r"bound of Proposition~\ref{prop:mst} (black line), the region it excludes (shaded), the diagonal (dotted) and "
    r"chance (cross). (b) Share of the shared MST edges that lie in $F_A$, one dot per pair, with the median (black "
    r"bar), the share expected if the shared edges were a random subset of the language-model tree (dashed), and the "
    r"preregistered level (grey line). (c) Bridge overlap minus its permutation null mean against BLOOMZ parameter "
    r"count (log scale), averaged over vision models, with $95\%$ bootstrap intervals over vision models and the "
    r"median per-pair $95\%$ permutation quantile (grey band). $\mathrm{Tr}$ is the preregistered trend statistic; "
    r"with five sizes no exact $p$-value can fall below the printed floor, so the trend test is power-limited. The "
    r"Spanning Trees paragraph of Appendix~\ref{app:exp-s12} and Table~\ref{tab:app-pe2} give the values and both "
    r"pools.}")

# v6: numbers the caption no longer carries, for the Results paragraph of Appendix F (each with its source) ---------
RES = {
    "forest-share medians (mid depth / last layer)": ((PB[0.5]["med"], PB[1.0]["med"]), "panel_b.median_bar",
                                                       f"{P_REPORT} section 3 'median forest share of shared edges' (0.875 / 0.889)"),
    "null medians = forest fraction of the LM tree (mid depth / last layer)": ((PB[0.5]["nul"], PB[1.0]["nul"]), "panel_b.null_bar", None),
    "every pair above the preregistered level 0.70; minimum forest share": (
        ({DEPTH[r]: NBd[f"r={r}"]["frac_pairs_forest_share_ge_0.70"] for r in (0.5, 1.0)},
         (NBd["r=0.5"]["min_forest_share_of_shared"], NBd["r=1.0"]["min_forest_share_of_shared"])),
        "panel_b.preregistered_level.frac_pairs_ge_level, panel_b.min_forest_share", f"{P_REPORT} section 3 row (a)"),
    "preregistered level 0.70": (LEVEL, "panel_b.preregistered_level", f"{P_PLAN} line 98"),
    "last layer: Tr 0.57, exact one-sided p 0.058, Holm 0.23": ((c1["Tr_bridge"], c1["p_one_exact"], c1["p_one_holm"]),
                                                                "panel_c.statistics_for_caption['last layer']",
                                                                f"{P_REPORT} sections 3 and 5"),
    "last layer: exact two-sided p 0.117, Holm 0.47": ((c1["p_two_exact"], c1["p_two_holm"]),
                                                       "panel_c.statistics_for_caption['last layer']", f"{P_REPORT} section 5"),
    "mid depth: Tr -0.16, exact one-sided p 0.74": ((c5["Tr_bridge"], c5["p_one_exact"]),
                                                    "panel_c.statistics_for_caption['mid depth']", f"{P_REPORT} section 3"),
    "smallest attainable exact one-sided p 1/120; Holm-adjusted two-sided p never below 0.067": (
        (MIN_P_ONE, MIN_HOLM_TWO), "panel_c.trend_rule_conditions_c_d", None),
    "bridge overlap median vs its null mean (mid depth / last layer)": (
        ((NBd["r=0.5"]["median_bridge_overlap"], NBd["r=0.5"]["median_bridge_null_mean"]),
         (NBd["r=1.0"]["median_bridge_overlap"], NBd["r=1.0"]["median_bridge_null_mean"])),
        "panel_b.summary_from_F2['r=0.5'|'r=1.0'].{median_bridge_overlap, median_bridge_null_mean}",
        f"{P_REPORT} section 3 'median bridge overlap / median null mean'"),
    "mean shared LM bridges per pair (mid depth / last layer), power target 10": (
        (NBd["r=0.5"]["mean_shared_bridges"], NBd["r=1.0"]["mean_shared_bridges"]),
        "panel_b.summary_from_F2['r=0.5'|'r=1.0'].mean_shared_bridges",
        f"{P_REPORT} section 3 power gate"),
    "bound of Proposition mst holds in all 816 configurations; half term binds in all": (
        (BOUND["n_bound_holds"], BOUND["n_half_term_binds"], BOUND["n_configs"]), "panel_a.bound_check_all_configs", None),
    "s_T over the bound, range": ((BOUND["sT_over_bound_min"], BOUND["sT_over_bound_max"]), "panel_a.bound_check_all_configs", None),
    "chance: 1/(m-1) for s_N1 and median permutation mean for s_T": ((cx, cy), "panel_a.chance_mark", None),
    "pairs: 136 vision-language pairs per depth in (a, b); 85 BLOOMZ pairs per depth in (c); 17 vision models": (
        (NA["r=0.5"]["n_pairs"], N_BLOOMZ_PAIRS, len(VITS)), "panel_a.summary_from_F2, panel_c.n_bloomz_pairs_per_depth", None),
    "m = 2047; n_pi = 500; bootstrap B = 10000": ((m, K_NULL, NB), "setting", None),
    "forest fraction n_forest_A/(m-1) of the LM tree, median by pool, preprocessing and depth; minimum over all 816": (
        ({k_: v_["median"] for k_, v_ in FF.items()}, FF_MIN_ALL), "tree_forest_fraction_by_config",
        "recomputed here from %s and %s; the designated-pool medians equal panel_b.null_bar" % (P_POOL, P_MAIN)),
}
N["results_text_numbers"] = {k: {"value": v, "key": key, "path": "numbers.json (this file)",
                                 "also_in": also} for k, (v, key, also) in RES.items()}
N["caption_numbers"] = {"note": "empty by the author's rule of 2026-10-10: the caption carries no result numbers; "
                                "they are listed under results_text_numbers (design constants only: m = 2047, 95%, "
                                "five sizes)"}
N["tree_forest_fraction_by_config"] = {
    "what": "n_forest_A / (m - 1): share of the language-model MST that is its nearest-neighbor forest F_A (the null "
            "of (b)); per configuration (pool, preprocessing, depth) over its 136 pairs",
    "per_config": FF, "min_over_all_816": FF_MIN_ALL,
    "wording_note": "84 to 85% holds for prh_clip+l2 only (both pools); plain l2 at m = 1024 gives 94% (mid depth) and "
                    "86% (last layer); the minimum 82% holds in every configuration"}
N["caption_proposed_v6"] = CAPTION
N["on_panel_numbers"] = {"note": "v8: no result number is drawn; tick labels only (the v6 numbers are listed under "
                                 "on_panel_numbers_v6_removed with their sources)"}


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
CAPTION_V8 = open(CAP_FILE).read().strip()
N["caption_v8"] = {"tex": CAPTION_V8, "sha256": sha(CAP_FILE),
                   "checks": caption_v8_check(CAPTION_V8, max_words=100, allowed_digits=("95",))}
fig.canvas.draw()
_texts = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
import re as _re8  # noqa: E402
_tick_txt = {t.get_text() for a_ in (axA, axB, axC) for t in a_.get_xticklabels() + a_.get_yticklabels()}
DESIGN_LEVEL_TEXTS = {"null 95%",                # the level of the permutation quantile (design), not a result
                      r"1-NN overlap $s_{N{,}1}$"}   # a name (one-nearest-neighbor), not a number
DIGIT_TEXTS = sorted({t for t in _texts if (t not in _tick_txt)
                      and _re8.search(r"\d", _re8.sub(r"\$[^$]*\$", "", t))})
N["texts_with_digits_besides_ticks"] = DIGIT_TEXTS
V8 = {
    "style hrh_style_v8 sha256 asserted": S.STYLE_SHA256 == STYLE_SHA,
    "no result number drawn: texts with a digit are tick labels, the design level 'null 95%' or the name 1-NN": set(DIGIT_TEXTS)
    <= DESIGN_LEVEL_TEXTS,
    "(c) Tr and p floor not drawn": not any(("Tr" in t) or ("1/120" in t) or ("power" in t) for t in _texts),
    "(b) no x label": axB.xaxis.label.get_text() == "",
    "(c) direct labels are the depth names only": sorted(t.get_text() for t in C_LABELS.values()) == sorted(DEPTH.values()),
    "v6 statistics kept in numbers (Tr 0.57 / -0.16, p 0.058, Holm 0.23, floor 1/120)": (
        (round(c1["Tr_bridge"], 2), round(c1["p_one_exact"], 3), round(c1["p_one_holm"], 2)) == (0.57, 0.058, 0.23)
        and round(c5["Tr_bridge"], 2) == -0.16 and abs(MIN_P_ONE - 1 / 120) < 1e-15),
    "tick labels and axis titles bold (S.check)": not CHK["not_bold_tick_or_axis_labels"] and CHK["n_tick_labels_checked"] > 0,
    "bold math in bold texts (S.check)": not CHK["bold_text_with_regular_math"],
    "chromatic data colours: MST hue and its dark step MST_MED (v8 c; critic-v8: MST hue only; the BLOOMZ word is text)":
        set(CHK["chromatic_data_colours"]) == {S.CRIT["mst"].upper(), MST_MED.upper()},
    "MST hue is the v8 Tol vibrant teal": S.CRIT["mst"].upper() == "#009988" and GT.upper() == "#008575",
    "min glyph >= 7 pt": CHK["min_glyph_pt"] >= 7.0,
    "Type 42 only": CHK["n_type3"] == 0,
}
N["asserts_v8"] = V8
assert all(V8.values()), ({k_: v for k_, v in V8.items() if not v}, DIGIT_TEXTS)
# v8 c (consistency checker): black only for theory; medians in a step of the series hue
import matplotlib.colors as _mc  # noqa: E402
_b_lines = [(l_, _mc.to_hex(l_.get_color()).upper()) for l_ in axB.get_lines()]
_med_lines = [c_ for l_, c_ in _b_lines if abs(l_.get_linewidth() - LW["median_bar"]) < 1e-9]
_ink_data = [(a_n, _mc.to_hex(l_.get_color()).upper()) for a_n, a_ in (("a", axA), ("b", axB), ("c", axC))
             for l_ in a_.get_lines() if _mc.to_hex(l_.get_color()).upper() == INK.upper()
             and l_.get_linewidth() >= LW["theory"] - 1e-9]          # data weight (leaders are 0.7 pt)
CONS_C = {
    "(b) two median bars, both MST_MED": _med_lines == [MST_MED.upper()] * 2,
    "(b) no INK line": not [c_ for _, c_ in _b_lines if c_ == INK.upper()],
    "(b) key handle 'median' MST_MED, key text INK": _mc.to_hex(KEY_B.legend_handles[0].get_color()).upper()
    == MST_MED.upper() and _mc.to_hex(KEY_B.get_texts()[0].get_color()).upper() == INK.upper(),
    "INK data-weight lines only the bound of (a) (theory)": [n_ for n_, _ in _ink_data] == ["a"],
    "median values unchanged (v5 asserted above)": all(abs(V5["panel_b"]["median_bar"][DEPTH[r]] - PB[r]["med"]) <= 1e-12
                                                       for r in (0.5, 1.0)),
}
assert all(CONS_C.values()), (CONS_C, _b_lines, _ink_data)
N["consistency_v8c"] = {"date": "2026-10-10", "role": "consistency checker across the v8 figures",
                        "changes": ["(b) median bars and key handle INK -> MST_MED %s (dark step of the MST hue); black "
                                    "stays for theory only (the bound of a)" % MST_MED],
                        "MST_MED_dE2000_normal_deutan_protan": {
                            "vs_MST": [round(CV.dE(MST_MED, G, v_), 1) for v_ in ("normal", "deutan", "protan")],
                            "vs_INK": [round(CV.dE(MST_MED, INK, v_), 1) for v_ in ("normal", "deutan", "protan")],
                            "vs_NULL_LINE": [round(CV.dE(MST_MED, NL, v_), 1) for v_ in ("normal", "deutan", "protan")]},
                        "asserts": CONS_C, "critic_v8_files": "./_prev_consistency/"}
N["finalize_texts_changed"] = N_FINALIZED
N["outputs"] = {p_: sha(p_) for p_ in (PDF, PNG, V100, VIEWS)}
json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
print(json.dumps({"check": CHK, "layout": LAYOUT, "pdffonts": pf, "asserts_v8": V8,
                  "caption_v8": N["caption_v8"]["checks"]}, indent=1))
print("OK")
