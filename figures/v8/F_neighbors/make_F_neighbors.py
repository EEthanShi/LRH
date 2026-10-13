"""F_neighbors v8 (arxiv_v4 Figure 2, fig:exp-neighbors; role maker v8, 2026-10-10).

v8 = a copy of the v7 maker (~/hrh_v2_figs/v7/F_neighbors/make_F_neighbors.py, sha256 4960c507..., kept byte-identical
next to this file as make_F_neighbors_v7_copy.py) with the author's v8 rules (on v7 Figure 6, 2026-10-10, for EVERY
figure): (1) no title or row header that states a setting (this figure has none; the caption and the Results name the
setting); (2) the frozen v8 style ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (Paul Tol palette, sha256 asserted): pair
classes seed / vision / vision-language take Tol light blue / vibrant blue / high-contrast blue (the light, mid and dark
steps of the mKNN hue), MST takes Tol vibrant teal, theory INK, references NULL_LINE / NULL_FILL; (3) tick labels and
axis titles bold, math included (S.finalize(fig) before the layout is measured and again before savefig); (4) no
secondary comparison drawn (this figure draws none: (b) is the k/m test itself and the two depths of (c) are the two
halves of the plotted configurations, labelled once each). The panels, their data and their on-figure words are those of
v7 (asserted against the v7 numbers.json); only colours, weights and the geometry that bold labels need change.
Old checks that read the v5.1 style (the v5.1 render, the v5 = v5.1 equivalence) use the v5.1 copy next to this file
(S51); the figure itself is drawn with the v8 module (S). numbers.json lists every change (changes_v8) and every adapted
assert (asserts_adapted_v8).
Run (CPU, 1 thread, at most 6 cores):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 nice -n 10 \
      ~/hrh_v2_figs_venv/bin/python -B ~/hrh_v2_figs/v8/F_neighbors/make_F_neighbors.py
Writes only inside ~/hrh_v2_figs/v8/F_neighbors/ (or its _trials/<name>/ with FN_TRIAL=<name>).

---- v7 header (unchanged) ----
F_neighbors v7 (arxiv_v4 Figure 2, fig:exp-neighbors; role fig-maker-v7, 2026-10-10).

Author feedback (2026-10-10, on Figures 4 and 6, applied to every figure): panels fill the whole row; little text on
the figure (no settings, statuses or long annotations; the shortest labels that still decode the encodings); short
captions (a bold claim plus what decodes the marks; settings and numbers live in the Results text, Section 5.1 or the
appendix). The figure keeps its three panels and their messages (the author liked the v5.1 render of this figure).

This script draws the figure again. It is a merge of two copies kept next to it, both untouched:
  make_F_neighbors_v6_copy.py      = ~/hrh_v2_figs/v6/F_neighbors/make_F_neighbors.py (the v6 maker, which drew nothing
                                     and verified the v5.1 render), sha256 d763e4a2...;
  make_F_neighbors_v5_draw_copy.py = ~/hrh_v2_figs/v5/F_neighbors/make_F_neighbors.py (the maker that drew the v5.1
                                     render), sha256 d6c978ee....
Every assert of the v6 maker is kept and runs here (sections V6.1 to V6.4 and V6.CAPTION below; the checks of the frozen
v5.1 render and of the v6 option render are read-only and no longer write a view into the v6 folders). Every assert of
the v5 drawing maker is kept as well (data against F3/F2/v4 numbers, label placement, ink layout audit, PDF ink margins).
The style module is the copy ./hrh_style_v5_1.py of ~/hrh_v2_figs/v5/_style/hrh_style_v5_1.py (sha256 97d0ed53...,
asserted); the path ~/hrh_v2_figs/v5_1/hrh_style_v5_1.py named in the task does not exist on the server.

v7 changes to the figure (numbers.json changes_v7 lists them with their reasons):
  * row filled: the slope column inside (a) and the 0.25 in gap after it are gone; (a) and (b) widen from 1.23 / 1.26 in
    of data to equal widths that take the freed space, (c) stays square at the axes height, and the gaps between panels
    are just wide enough for the tick labels and the y label of (c);
  * less text: the three slope numbers, the 'slope' header, the ', slope -1.00' of the chance label and the 'k = 10'
    corner label of (a) are dropped (the Results paragraph of Section 5.2 states the slopes against chance);
  * (a) class medians are drawn with the open k = 10 square of (b) (the v6 option render): (b)'s key then decodes the
    k of (a) without a label, and a filled circle no longer means k = 10 in (a) and k = 1 in (b);
  * height: the top margin held the 'slope' header and now holds only the panel letters, so the figure is lower:
    1.95 -> 1.70 in, the lowest height in 0.05 in steps whose layout audit is clean (1.65 in puts the 'vision' label
    on the vision-language bundle; trials in _trials/, numbers.json height_trials).
Data and every plotted value are unchanged: the panel blocks are re-derived as in v5 and asserted equal to the v5
numbers.json (which the v6 maker verified) value for value.
Run (CPU, 1 thread, at most 6 cores):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 nice -n 10 \
      ~/hrh_v2_figs_venv/bin/python -B ~/hrh_v2_figs/v7/F_neighbors/make_F_neighbors.py
Writes only inside ~/hrh_v2_figs/v7/F_neighbors/ (or its _trials/<name>/ with FN_TRIAL=<name>).
"""
import os, sys, json, hashlib, datetime, subprocess, re, io
from collections import Counter
import importlib.util

sys.dont_write_bytecode = True
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

H_ = os.path.expanduser
sys.path.insert(0, HERE)
import hrh_style_v5_1 as S51  # noqa: E402  (the copy next to this file; frozen v5.1 style, for the old checks only)
# v8: the frozen v8 style, imported from its own folder after its sha256 is asserted
STYLE_V8_PATH = H_("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")
STYLE_V8_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_V8_PATH, "rb").read()).hexdigest() == STYLE_V8_SHA, "v8 style changed"
_spec8 = importlib.util.spec_from_file_location("hrh_style_v8", STYLE_V8_PATH)
S = importlib.util.module_from_spec(_spec8)
_spec8.loader.exec_module(S)
assert S.STYLE_SHA256 == STYLE_V8_SHA and S.STYLE_VERSION == "v8"
_spec = importlib.util.spec_from_file_location("hrh_style_v5", H_("~/hrh_v2_figs/v5/_style/hrh_style_v5.py"))
S5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S5)                 # the style of the v5.1 render, for the v6 equivalence check (read only)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator, FuncFormatter  # noqa: E402
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt  # noqa: E402

S.use()
LW, MSZ, Z = S.LW, S.MS, S.Z
MINUS = S.MINUS


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r4(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def neg(s):
    return s.replace("-", MINUS)


# ================================================================== paths
STYLE_COPY = os.path.join(HERE, "hrh_style_v5_1.py")
STYLE_DIR = H_("~/hrh_v2_figs/v5/_style")
STYLE_V5, STYLE_V51 = f"{STYLE_DIR}/hrh_style_v5.py", f"{STYLE_DIR}/hrh_style_v5_1.py"
STYLE_V51_SHA = "97d0ed5388b2b7a06829ce42645bc545070ecbd3e65549b5020568fecc9f41a6"
STYLE_V5_SHA = "8c75d5261d4f387fbd9e8f8ebbd22861adbf229a5f43b6c01ec405cbd3bd5ad4"
V5 = H_("~/hrh_v2_figs/v5/F_neighbors")
V6 = H_("~/hrh_v2_figs/v6/F_neighbors")
V7 = H_("~/hrh_v2_figs/v7/F_neighbors")
V7_MAKER_SHA = "4960c507050389e802f5809c662bc6944e6e1fb0844f03e1fe3500fbf0927168"
FROZEN_V7 = {f"{V7}/make_F_neighbors.py": V7_MAKER_SHA,
             f"{V7}/F_neighbors.pdf": "8113d112fef4e9ab635a2c414a6825d9559e2742bdc5188069c1bb8f831b0cd3",
             f"{V7}/F_neighbors.png": "644fd32902d579cd0086e3741557cfb4fa12a8d734dc6597244618267d475b46",
             f"{V7}/caption_v7.tex": "2edd42020dc87783165839ba73148d905e27f01bc1f70f5f45d8c5f9350f483e",
             f"{V7}/numbers.json": "7018783410f70ae3ba8654ba766ec98b279fada5df7a4415b5392bfe7e0ecc7a"}
FROZEN = {f"{V5}/F_neighbors.pdf": "d9336aa58e034928ea4261300e31f40ba5bf2c23e291582a372ad1314c492221",
          f"{V5}/F_neighbors.png": "962fc6484710f6fd97ccef2d3675de26b729329f0c303703ba4bf8363af639b7",
          f"{V5}/numbers.json": "19c007bdb392f4cafdb714048be56447fc0f60bd01767914eb347876301335f9",
          f"{V5}/make_F_neighbors.py": "d6c978eebdcc51bb3fad85b2023bb239404d7d953a7deac58414161e9adfa8bb"}
COPIES = {f"{HERE}/make_F_neighbors_v6_copy.py": ("d763e4a2eb07360d1451ac8dccae24aed9aa758912acca726f99d5448f223d03",
                                                  f"{V6}/make_F_neighbors.py"),
          f"{HERE}/make_F_neighbors_v5_draw_copy.py": (FROZEN[f"{V5}/make_F_neighbors.py"], f"{V5}/make_F_neighbors.py"),
          STYLE_COPY: (STYLE_V51_SHA, STYLE_V51),
          f"{HERE}/make_F_neighbors_v7_copy.py": (V7_MAKER_SHA, f"{V7}/make_F_neighbors.py")}
V11 = H_("~/hrh_v2_stage1_20261008/results/S1_1/v11")
F_CELLS, F_SLOPES, F_SUMMARY = (f"{V11}/S1_1_v11_cells_clip.parquet", f"{V11}/S1_1_v11_slopes_clip.parquet",
                                f"{V11}/S1_1_v11_summary.json")
S12 = H_("~/hrh_v2_stage1_20261008/results/S1_2")
P_POOL, P_MAIN = f"{S12}/panel_pool2047.parquet", f"{S12}/panel_main.parquet"
OLD_F3 = H_("~/hrh_v2_figs/F3/numbers.json")
OLD_F2 = H_("~/hrh_v2_figs/F2/numbers.json")
V4_NUM = H_("~/hrh_v2_figs/v4/F_neighbors/numbers.json")

CLS = ["seed-seed", "cross-vision", "cross-modal"]
CKEY = {"seed-seed": "seed", "cross-vision": "vision", "cross-modal": "vl"}
MS = [256, 512, 1024, 2048, 4096, 8192, 16384]
KS = [1, 10, 100]
KMARK = {1: "o", 10: "s", 100: "^"}
PRE = "prh_clip+l2"
TOL = 1e-12

# copies next to this file are byte-identical to their sources
for p, (h, srcp) in COPIES.items():
    assert sha(p) == h == sha(srcp), ("copy differs from its source", p)
assert sha(STYLE_COPY) == S51.STYLE_SHA256 == STYLE_V51_SHA, "style copy changed"
for p, h in FROZEN_V7.items():                      # v8: the v7 render and maker this one starts from (read only)
    assert sha(p) == h, ("v7 file changed", p)
O7 = json.load(open(f"{V7}/numbers.json"))

INS = {p: sha(p) for p in (F_CELLS, F_SLOPES, F_SUMMARY, P_POOL, P_MAIN, OLD_F3, OLD_F2, V4_NUM, STYLE_COPY)}


def src(path, key):          # as in the v5 maker (key order source, key, sha256)
    return {"source": path, "key": key, "sha256": INS[path]}


# ================================================================== V6.1 the frozen v5.1 render (read only)
for p, h in FROZEN.items():
    assert sha(p) == h, ("v5 render changed", p)
O5 = json.load(open(f"{V5}/numbers.json"))

# ================================================================== V6.2 v5 style == v5.1 style for this figure
assert sha(STYLE_V51) == STYLE_V51_SHA == S51.STYLE_SHA256
assert sha(STYLE_V5) == O5["style"]["sha256"] == S5.STYLE_SHA256 == STYLE_V5_SHA
EXEMPT = {"ARM", "ARM_TEXT", "STYLE_VERSION", "STYLE_SHA256"}
up5 = {k for k in dir(S5) if k.isupper()}
up51 = {k for k in dir(S51) if k.isupper()}
assert up5 == up51, sorted(up5 ^ up51)
style_diff = [k for k in sorted(up5) if k not in EXEMPT and repr(getattr(S5, k)) != repr(getattr(S51, k))]
assert not style_diff, style_diff
mk5 = open(f"{V5}/make_F_neighbors.py").read()
assert "S.ARM" not in mk5 and "ARM_TEXT" not in mk5, "the v5 maker uses a training-rule colour"
mk7 = open(os.path.abspath(__file__)).read()
assert not re.search(r"S\.ARM(_TEXT)?\[", mk7), "this maker uses a training-rule colour"
STYLE_EQ = {"render_style_v5_1_render": "hrh_style_v5 (sha256 %s)" % S5.STYLE_SHA256,
            "frozen_v5_1": "hrh_style_v5_1 (sha256 %s)" % S51.STYLE_SHA256,
            "v7_imports": "./hrh_style_v5_1.py (copy, sha256 %s)" % S51.STYLE_SHA256,
            "v8_draws_with": "%s (sha256 %s); the v5 = v5.1 equivalence above concerns the old renders only"
                             % (STYLE_V8_PATH, S.STYLE_SHA256),
            "n_constants_compared": len(up5 - EXEMPT), "differences_outside_exempt": style_diff,
            "exempt": sorted(EXEMPT), "maker_uses_ARM": False,
            "result": "identical for every constant this figure can use"}

# ================================================================== V6.3 print-size checks of the v5.1 render (read only)
V5_PDF, V5_PNG = f"{V5}/F_neighbors.pdf", f"{V5}/F_neighbors.png"
glyph5 = S51.pdf_glyph_sizes(V5_PDF)
fonts5 = S51.pdf_fonts(V5_PDF)
info5 = subprocess.run(["pdfinfo", V5_PDF], capture_output=True, text=True, check=True).stdout
psz5 = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info5)
page5 = [float(psz5.group(1)) / 72, float(psz5.group(2)) / 72]
im5 = Image.open(V5_PNG)
dpi5 = [round(float(v)) for v in im5.info.get("dpi", (0, 0))]
contrast5 = []
for a in O5["annotations"]:
    back = S51.NULL_FILL if a["text"] == "excluded" else S51.WHITE
    contrast5.append({"text": a["text"], "color": a["color"], "background": back,
                      "ratio": round(S51.contrast(a["color"], back), 2)})
contrast5.append({"text": "axis labels and ticks (INK)", "color": S51.INK, "background": S51.WHITE,
                  "ratio": round(S51.contrast(S51.INK, S51.WHITE), 2)})
V5_CHECKS = {"pdf": V5_PDF, "pdf_sha256": sha(V5_PDF), "page_size_in": [round(v, 4) for v in page5],
             "min_glyph_pt": float(min(glyph5)), "n_type3": open(V5_PDF, "rb").read().count(b"/Type3"),
             "fonts": [f"{r['name']} ({r['type']}, emb {r['emb']})" for r in fonts5],
             "png_px": list(im5.size), "png_dpi": dpi5, "min_text_contrast": min(c["ratio"] for c in contrast5),
             "v5_layout_audit_result": O5["layout_audit"]["result"]}
assert abs(page5[0] - 5.5) < 1e-3 and abs(page5[1] - 1.95) < 1e-3, page5
assert V5_CHECKS["min_glyph_pt"] >= S51.MIN_PT - 1e-9
assert V5_CHECKS["n_type3"] == 0
assert fonts5 and all(r["emb"] == "yes" and "TrueType" in r["type"] for r in fonts5), fonts5
assert tuple(im5.size) == (1650, 585) and dpi5 == [300, 300], (im5.size, dpi5)
assert V5_CHECKS["min_text_contrast"] >= S51.MIN_CONTRAST, contrast5
assert O5["layout_audit"]["result"] == "clean"

# ---- V6.3b the v6 option render (open k = 10 squares in (a)), read only; v7 adopts its marker change
OPT = f"{V6}/_option_a_open_squares"
OPT_PDF, OPT_PNG, OPT_MK = f"{OPT}/F_neighbors.pdf", f"{OPT}/F_neighbors.png", f"{OPT}/make_F_neighbors_option.py"
O_OPT = json.load(open(f"{OPT}/numbers.json"))
mk_opt = open(OPT_MK).read()
assert mk_opt.count('marker="s", ms=MSZ["scatter"], mfc="white", mec=PC[k_], mew=MSZ["open_small_mew"]') == 1
assert 'marker="o", ms=MSZ["line"], mfc=PC[k_], mec="white", mew=MSZ["line_edge"]' not in mk_opt
assert O_OPT["layout_audit"]["result"] == "clean", O_OPT["layout_audit"]["result"]
for blk in ("panel_a", "panel_b", "panel_c"):
    assert json.dumps(O_OPT[blk], sort_keys=True, default=float) == json.dumps(O5[blk], sort_keys=True, default=float), blk
g_opt = S51.pdf_glyph_sizes(OPT_PDF)
f_opt = S51.pdf_fonts(OPT_PDF)
i_opt = subprocess.run(["pdfinfo", OPT_PDF], capture_output=True, text=True, check=True).stdout
p_opt = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", i_opt)
im_opt = Image.open(OPT_PNG)
OPT_CHECKS = {"pdf": OPT_PDF, "pdf_sha256": sha(OPT_PDF), "png_sha256": sha(OPT_PNG), "maker_sha256": sha(OPT_MK),
              "page_size_in": [round(float(p_opt.group(1)) / 72, 4), round(float(p_opt.group(2)) / 72, 4)],
              "min_glyph_pt": float(min(g_opt)), "n_type3": open(OPT_PDF, "rb").read().count(b"/Type3"),
              "png_px": list(im_opt.size), "png_dpi": [round(float(v)) for v in im_opt.info.get("dpi", (0, 0))],
              "layout_audit_result": O_OPT["layout_audit"]["result"]}
assert OPT_CHECKS["page_size_in"] == [5.5, 1.95] and OPT_CHECKS["min_glyph_pt"] >= S51.MIN_PT - 1e-9
assert OPT_CHECKS["n_type3"] == 0 and all(r["emb"] == "yes" and "TrueType" in r["type"] for r in f_opt)
assert OPT_CHECKS["png_px"] == [1650, 585] and OPT_CHECKS["png_dpi"] == [300, 300]

# ================================================================== V6.4 Results numbers recomputed from the results
hash_vs_v5 = {p: ("unchanged" if INS[p] == O5["inputs_sha256"][p] else "CHANGED")
              for p in (F_CELLS, F_SLOPES, F_SUMMARY, P_POOL, P_MAIN)}
assert all(v == "unchanged" for v in hash_vs_v5.values()), hash_vs_v5

cells = pd.read_parquet(F_CELLS)
slopes = pd.read_parquet(F_SLOPES)
summ = json.load(open(F_SUMMARY))
C = summ["configs"]["clip"]
cp = cells[cells.primary].copy()
sp = slopes[slopes.primary].copy()
assert cp.pair.nunique() == 29 and (cp.n_draws == 5).all()
ca = cp[cp.k == 10]
npairs = {CKEY[c]: int(cp[cp.cls == c].pair.nunique()) for c in CLS}
assert npairs == {"seed": 4, "vision": 10, "vl": 15}
cmed = {CKEY[c]: ca[ca.cls == c].groupby("m").obs.median() for c in CLS}
slope10 = {CKEY[c]: float(C["slope_classes"]["primary"][c]["10"]["median_slope"]) for c in CLS}
slope10_re = {CKEY[c]: float(slopes[slopes.primary & (slopes.cls == c) & (slopes.k == 10)].slope_obs.median()) for c in CLS}
order = C["secondary"]["primary"]["slope_ordering"]
chance_slope = float(np.polyfit(np.log(MS), np.log([10 / (m - 1) for m in MS]), 1)[0])
ratio_ch = {CKEY[c]: (ca[ca.cls == c].obs / ca[ca.cls == c].chance).groupby(ca[ca.cls == c].m).min() for c in CLS}
el = C["secondary"]["primary"]["elasticity_by_class"]
order_re = {str(kk): {c: float(slopes[slopes.primary & (slopes.cls == c) & (slopes.k == kk)].slope_obs.median())
                      for c in CLS} for kk in (1, 10, 100)}
for kk in ("1", "10", "100"):
    for c in CLS:
        assert abs(order_re[kk][c] - order[kk]["median_slope_by_class"][c]) < 1e-12, (kk, c)
    assert order_re[kk]["seed-seed"] > order_re[kk]["cross-vision"] > order_re[kk]["cross-modal"], kk
EL_RE = []
for p, g in cp.groupby("pair"):
    assert len(g) == 21
    X = np.c_[np.ones(len(g)), np.log(g.k.astype(float)), np.log(g.m.astype(float))]
    b_ = np.linalg.lstsq(X, np.log(g.obs.astype(float)), rcond=None)[0]
    EL_RE.append({"pair": p, "cls": g.cls.iloc[0], "e_k": float(b_[1]), "e_m": float(b_[2]), "sum": float(b_[1] + b_[2])})
EL_RE = pd.DataFrame(EL_RE)
el_re = {c: {"median_e_k": float(EL_RE[EL_RE.cls == c].e_k.median()), "median_e_m": float(EL_RE[EL_RE.cls == c].e_m.median()),
             "median_sum": float(EL_RE[EL_RE.cls == c]["sum"].median())} for c in CLS}
for c in CLS:
    for kk in ("median_e_k", "median_e_m", "median_sum"):
        assert abs(el_re[c][kk] - el[c][kk]) < 1e-9, (c, kk, el_re[c][kk], el[c][kk])
med = cp.groupby(["cls", "k", "m"]).obs.median()
RAT6 = {}
for c in CLS:
    x10 = np.log([10 / m for m in MS])[::-1]
    y10 = np.log([med[(c, 10, m)] for m in MS])[::-1]
    for kk in (1, 100):
        RAT6[f"{CKEY[c]}|k={kk}"] = {m: float(med[(c, kk, m)] / np.exp(np.interp(np.log(kk / m), x10, y10)))
                                     for m in MS if x10[0] - TOL <= np.log(kk / m) <= x10[-1] + TOL}
allc6 = []
for p in (P_POOL, P_MAIN):
    x = pd.read_parquet(p)
    m_ = x["L.m"].astype(float)
    s1 = x["L.s_N_k1_for_mst"].astype(float)
    b = np.maximum(m_ * s1 - (m_ - x["M.n_forest_A"]), m_ * s1 / 2) / (m_ - 1)
    allc6.append(pd.DataFrame({"ok": x["L.s_T"] >= b - TOL, "ratio": x["L.s_T"] / b, "pool": os.path.basename(p),
                               "pre": x["pre"], "r": x["r"]}))
allc6 = pd.concat(allc6, ignore_index=True)
d6 = pd.read_parquet(P_POOL)
d6 = d6[d6.pre == PRE]
assert len(d6) == 272 and (d6["L.m"] == 2047).all()
d6 = d6.assign(sN1=d6["L.s_N_k1_for_mst"].astype(float), sT=d6["L.s_T"].astype(float))
bydepth = {f"r={r}": {"n": int(len(g)), "median_sT_over_sN1": float((g.sT / g.sN1).median()),
                      "n_sT_above_sN1": int((g.sT > g.sN1).sum())} for r, g in d6.groupby("r")}
A5, B5, C5 = O5["panel_a"], O5["panel_b"], O5["panel_c"]
for k_ in ("seed", "vision", "vl"):
    for m in MS:
        assert abs(cmed[k_][m] - A5["class_median_obs_k10"]["values"][k_][str(m)]) < TOL
        assert abs(ratio_ch[k_][m] - A5["obs_over_chance_k10_min_by_m"]["values"][k_][str(m)]) < TOL
    assert abs(slope10[k_] - A5["median_slope_k10"]["values"][k_]) < TOL and abs(slope10[k_] - slope10_re[k_]) < TOL
    for kk in (1, 100):
        for m, v in RAT6[f"{k_}|k={kk}"].items():
            assert abs(v - B5["ratio_to_k10_at_shared_k_over_m"]["values"][f"{k_}|k={kk}"][str(m)]) < TOL
assert abs(chance_slope - A5["chance_slope_k10"]["value"]) < TOL
assert len(allc6) == 816 == int(allc6.ok.sum()) == C5["bound_check_all_configs"]["n_bound_holds"]
assert abs(allc6.ratio.min() - C5["bound_check_all_configs"]["sT_over_bound_min"]) < TOL
for r in ("0.5", "1.0"):
    assert abs(bydepth[f"r={r}"]["median_sT_over_sN1"] - C5["summary_by_depth"][f"r={r}"]["median_sT_over_sN1"]) < TOL
    assert abs(bydepth[f"r={r}"]["n_sT_above_sN1"] / 136 - C5["summary_by_depth"][f"r={r}"]["frac_sT_above_sN1"]) < TOL
for kk in ("1", "10", "100"):
    assert order[kk]["ordering_holds"] is True


def src6(path, key):         # as in the v6 maker (key order source, sha256, key)
    return {"source": path, "sha256": INS[path], "key": key}


def rng(vals):
    return [float(min(vals)), float(max(vals))]


RES = []


def add_res(rid, printed, value, where_v6, use, **s):
    RES.append({"id": rid, "printed": printed, "value": value, "in_v4_at_v6": where_v6, "use_in_results": use, **s})


add_res("class_median_k10_m256_to_m16384",
        "seed 0.548 -> 0.535; vision 0.256 -> 0.109; vision-language 0.091 -> 0.0085",
        {k_: [float(cmed[k_][256]), float(cmed[k_][16384])] for k_ in cmed},
        "Sec 5.2 text l.1233-1235 and caption", "Results (a): the three class levels at the ends of the pool range",
        **src6(F_CELLS, "median over pairs of cells[primary & k==10 & cls].obs at m = 256 and 16384 "
                        "(obs = mean over 5 nested orderings)"))
add_res("median_slope_k10", "-0.01, -0.21, -0.57 (seed, vision, vision-language)", slope10,
        "caption only (and on panel (a))", "Results (a): the rate per class (v7: no longer on the panel)",
        **src6(F_SUMMARY, "configs.clip.slope_classes.primary.<cls>.10.median_slope; equal to the median of "
                          "S1_1_v11_slopes_clip.parquet slope_obs (primary, k = 10)"))
add_res("chance_slope_k10", "-1.00", chance_slope, "caption only (and on panel (a))",
        "Results (a): chance falls as 1/m; analytic (v7: no longer on the panel)",
        source="analytic", key="OLS slope of ln(10/(m-1)) on ln m over the 7 pool sizes")
add_res("slope_classes_k10", "seed 4/4 flat; vision 10/10 falling; vision-language 15/15 falling; no rising pair",
        {CKEY[c]: {kk: C["slope_classes"]["primary"][c]["10"][kk] for kk in ("rising", "flat", "falling", "n")} for c in CLS},
        "Sec 5.2 text (\"the rising class is absent\")", "Results (a)",
        **src6(F_SUMMARY, "configs.clip.slope_classes.primary.<cls>.10.{rising,flat,falling,n}"))
add_res("slope_ordering_every_k",
        "seed > vision > vision-language at k = 1, 10, 100; median slopes at k = 1: 0.009, -0.076, -0.522; "
        "at k = 100: -0.055, -0.278, -0.598",
        {kk: {CKEY[c]: order[kk]["median_slope_by_class"][c] for c in CLS} for kk in ("1", "10", "100")},
        "Sec 5.2 text, without values", "Results (a)",
        **src6(F_SUMMARY, "configs.clip.secondary.primary.slope_ordering.<k>.{median_slope_by_class,ordering_holds}"))
assert [f'{order[kk]["median_slope_by_class"][c]:.3f}' for kk in ("1", "100") for c in CLS] == \
    ["0.009", "-0.076", "-0.522", "-0.055", "-0.278", "-0.598"]
add_res("vl_over_chance_min", "at m = 16,384 every vision-language pair is at least 8.75 times chance (1.93 at m = 256)",
        {"m=16384": float(ratio_ch["vl"][16384]), "m=256": float(ratio_ch["vl"][256])},
        "Sec 5.2 text l.1235-1236", "Results (a)",
        **src6(F_CELLS, "min over vision-language pairs of cells[primary & k==10].obs / .chance at m"))
NSH = {key: len(v) for key, v in RAT6.items()}
assert set(NSH.values()) == {3}, NSH
assert all(sorted(RAT6[f"{k_}|k=100"]) == [4096, 8192, 16384] and sorted(RAT6[f"{k_}|k=1"]) == [256, 512, 1024]
           for k_ in ("seed", "vision", "vl"))
add_res("ratio_k100_to_k10_shared_k_over_m",
        "k = 100 within 4% of k = 10 in every class (ratios 1.003-1.024, 1.027-1.039, 0.993-1.033), "
        "each range over the 3 shared k/m values of its class",
        {k_: rng(RAT6[f"{k_}|k=100"].values()) for k_ in ("seed", "vision", "vl")},
        "caption only", "Results (b)", n_shared_points_per_class=3, shared_m=[4096, 8192, 16384],
        **src6(F_CELLS, "class median at k = 100 over the k = 10 class median interpolated in (ln k/m, ln s_N)"))
add_res("ratio_k1_to_k10_shared_k_over_m",
        "k = 1 lower by 13-16% (seed, ratios 0.845-0.869) and 17-19% (vision, 0.815-0.829); within 11% for "
        "vision-language (0.891-1.053), each range over the 3 shared k/m values of its class",
        {k_: rng(RAT6[f"{k_}|k=1"].values()) for k_ in ("seed", "vision", "vl")},
        "caption only", "Results (b)", n_shared_points_per_class=3, shared_m=[256, 512, 1024],
        **src6(F_CELLS, "as above for k = 1, m = 256, 512, 1024"))
add_res("elasticity_sum_and_m",
        "e_k + e_m class medians 0.03, 0.06, 0.03; e_m -0.02, -0.19, -0.56 (e_m is not the fixed-k slope -0.57)",
        {CKEY[c]: {"median_sum": el[c]["median_sum"], "median_e_m": el[c]["median_e_m"], "median_e_k": el[c]["median_e_k"]}
         for c in CLS},
        "Sec 5.2 text l.1246-1249", "Results (b)",
        **src6(F_SUMMARY, "configs.clip.secondary.primary.elasticity_by_class.<cls>.{median_sum,median_e_m}"))
for k_, lo, hi in (("seed", "1.003", "1.024"), ("vision", "1.027", "1.039"), ("vl", "0.993", "1.033")):
    assert [f"{v:.3f}" for v in rng(RAT6[f"{k_}|k=100"].values())] == [lo, hi], k_
for k_, lo, hi in (("seed", "0.845", "0.869"), ("vision", "0.815", "0.829"), ("vl", "0.891", "1.053")):
    assert [f"{v:.3f}" for v in rng(RAT6[f"{k_}|k=1"].values())] == [lo, hi], k_
add_res("mst_bound_all_configs", "bound holds in 816 of 816 configurations; s_T at least 1.58 times the bound",
        {"n": int(len(allc6)), "n_holds": int(allc6.ok.sum()), "min_ratio": float(allc6.ratio.min()),
         "max_ratio": float(allc6.ratio.max()),
         "composition": {f"{p}|{pre}|r={r}": int(n) for (p, pre, r), n in allc6.groupby(["pool", "pre", "r"]).size().items()}},
        "Sec 5.3 text and caption", "Results (c)",
        **src6(P_MAIN, "panel_pool2047 (272) + panel_main (544); bound max{m s_N1 - (m - n_forest_A), m s_N1 / 2}/(m-1)"))
add_res("mst_plotted", "272 configurations plotted: 136 vision-language pairs (8 language x 17 vision models) x 2 depths, m = 2047",
        {"n": int(len(d6)), "n_pairs_per_depth": {k: v["n"] for k, v in bydepth.items()}},
        "caption only", "data description", **src6(P_POOL, "rows pre == prh_clip+l2"))
add_res("mst_over_1nn_median_by_depth", "median s_T / s_N,1 = 1.20 at mid depth, 1.32 at the last layer",
        {k: v["median_sT_over_sN1"] for k, v in bydepth.items()}, "caption only", "Results (c)",
        **src6(P_POOL, "median over the 136 pairs of L.s_T / L.s_N_k1_for_mst, rows pre == prh_clip+l2, by r"))
add_res("mst_above_1nn_count_by_depth", "s_T > s_N,1 for 129 of 136 pairs at mid depth and 136 of 136 at the last layer",
        {k: v["n_sT_above_sN1"] for k, v in bydepth.items()}, "not in v4 at v6", "Results (c)",
        **src6(P_POOL, "count of L.s_T > L.s_N_k1_for_mst, rows pre == prh_clip+l2, by r"))
# where each Results number stands in arxiv_v4.tex now (read only, 2026-10-10 21:00; line numbers of that file)
RES_IN_V4_NOW = {"class_median_k10_m256_to_m16384": "l.1218-1220", "median_slope_k10": "l.1220-1222",
                 "chance_slope_k10": "l.1222", "slope_classes_k10": "l.1225 ('The rising class is absent')",
                 "slope_ordering_every_k": "l.1224-1225 (without values)", "vl_over_chance_min": "l.1222-1223",
                 "ratio_k100_to_k10_shared_k_over_m": "l.1232", "ratio_k1_to_k10_shared_k_over_m": "l.1233-1235",
                 "elasticity_sum_and_m": "l.1235-1237", "mst_bound_all_configs": "l.1302-1305",
                 "mst_plotted": "l.1289-1290 (136 pairs of WIT) and l.1297 (pool of m = 2047)",
                 "mst_over_1nn_median_by_depth": "l.1299-1301", "mst_above_1nn_count_by_depth": "l.1301"}
assert set(RES_IN_V4_NOW) == {r["id"] for r in RES}
# v8: the same records in arxiv_v4.tex as read at 2026-10-10 22:40 (sha256 0e4c35b0...; the file was only read)
RES_IN_V4_V8 = {"class_median_k10_m256_to_m16384": "l.1243-1245", "median_slope_k10": "l.1245-1247",
                "chance_slope_k10": "l.1247", "slope_classes_k10": "l.1250 ('The rising class is absent')",
                "slope_ordering_every_k": "l.1249-1250 (without values)", "vl_over_chance_min": "l.1247-1248",
                "ratio_k100_to_k10_shared_k_over_m": "l.1257-1258", "ratio_k1_to_k10_shared_k_over_m": "l.1258-1260",
                "elasticity_sum_and_m": "l.1260-1262", "mst_bound_all_configs": "l.1373-1376",
                "mst_plotted": "l.1360-1361 (136 pairs of WIT) and l.1368 (pool of m = 2047)",
                "mst_over_1nn_median_by_depth": "l.1370-1372", "mst_above_1nn_count_by_depth": "l.1372"}
assert set(RES_IN_V4_V8) == set(RES_IN_V4_NOW)

# ---- V6.CAPTION: the v6 caption (record) keeps its v6 checks
CAPTION_V6 = (
    r"\caption{\textbf{Neighbor agreement changes with the evaluation size at a rate set by the pair of models, and "
    r"spanning-tree overlap tracks one-nearest-neighbor overlap (Theorem~\ref{thm:crossing}, "
    r"Proposition~\ref{prop:mst}).} (a) mKNN $s_N$ at $k=10$ and mid depth against the evaluation size $m$ on nested "
    r"COCO pools, for single pairs (thin), class medians (thick), and chance $k/(m-1)$ (dashed). A pair joins two "
    r"Pythia-410M runs trained with different seeds (seed), two vision models (vision), or a vision and a language "
    r"model (vision--language). Numbers at the right give the median slope of $\ln s_N$ on $\ln m$ in each class. "
    r"(b) The class medians against $k/m$ for three values of $k$, sharing the vertical axis of (a), with the medians "
    r"of (a) as squares. The closer the markers of a class lie to one curve, the more its agreement depends on $k/m$ "
    r"alone. (c) MST overlap against one-nearest-neighbor overlap on WIT for every vision--language pair of "
    r"Section~\ref{sec:exp-protocol}, at mid depth (open) and the last layer (filled). The bound of "
    r"Proposition~\ref{prop:mst} (solid) excludes the grey region, and the dotted line is the diagonal. Panels (a, b) "
    r"are descriptive; Figure~\ref{fig:exp-scale} compares them with the crossing law, and "
    r"Figure~\ref{fig:exp-mst} adds the forest share and the bridges.}")
_b6 = re.sub(r"\\ref\{[^}]*\}|\\label\{[^}]*\}", "", CAPTION_V6).replace("Pythia-410M", "Pythia")
_n6 = re.findall(r"(?<![A-Za-z_\\])\d+(?:\.\d+)?", _b6)
assert sorted(set(_n6)) == ["1", "10"], _n6
for frag in ("Pythia-410M runs trained with different seeds (seed)", "two vision models (vision)",
             "a vision and a language model (vision--language)", "sharing the vertical axis of (a)",
             "with the medians of (a) as squares", r"Proposition~\ref{prop:mst} (solid) excludes the grey region",
             "the dotted line is the diagonal", r"every vision--language pair of Section~\ref{sec:exp-protocol}"):
    assert frag in CAPTION_V6, frag
assert r"s_{\Tset}=s_{N,1}" not in CAPTION_V6

# ================================================================== (a, b): S1.1 loader (v5 maker, unchanged)
N = {"figure": "F_neighbors", "tex_label": "fig:exp-neighbors", "tex_include": "figures/exp_v8/F_neighbors.pdf (proposed)",
     "version": "v8 (2026-10-10): v7 figure in the v8 style (Tol palette, bold tick labels and axis titles with bold "
                "math); same panels, data and on-figure words as v7",
     "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
     "script": os.path.abspath(__file__), "script_sha256": sha(os.path.abspath(__file__)),
     "inputs_sha256": INS,
     "font_text": font_manager.findfont(font_manager.FontProperties(family=["serif"])),
     "claim": O5["claim"]}
N["s11_setting"] = O5["s11_setting"]
assert N["s11_setting"] == {"summary_mode": summ["mode"], "config_spec": C["spec"],
                            "layer": "relative depth 0.5 (PREREG_stage1.md, S1.1 models)", "draws_per_cell": 5,
                            "pool": "COCO nested pools m = 256..16384"}
N["n_primary_pairs"] = npairs

A = {}
A["per_pair_obs_k10"] = {"values": {p: {int(mm): float(v) for mm, v in g.set_index("m").obs.items()}
                                    for p, g in ca.groupby("pair")},
                         **src(F_CELLS, "cells[primary & k==10].obs by (pair, m); mean over 5 nested orderings")}
A["per_pair_cls"] = {p: CKEY[g.cls.iloc[0]] for p, g in ca.groupby("pair")}
A["class_median_obs_k10"] = {"values": {CKEY[c]: {int(mm): float(v) for mm, v in ca[ca.cls == c].groupby("m").obs.median().items()}
                                        for c in CLS},
                             **src(F_CELLS, "median over pairs of cells[primary & k==10 & cls].obs by m")}
A["chance_k10"] = {"values": {int(mm): 10 / (mm - 1) for mm in MS}, "definition": "k/(m-1), k = 10",
                   "check": src(F_CELLS, "cells.chance at k==10")}
assert np.allclose(ca.groupby("m").chance.first().values, [10 / (mm - 1) for mm in MS], rtol=0, atol=1e-12)
A["median_slope_k10"] = {"values": {CKEY[c]: float(C["slope_classes"]["primary"][c]["10"]["median_slope"]) for c in CLS},
                         **src(F_SUMMARY, "configs.clip.slope_classes.primary.<cls>.10.median_slope "
                                          "(median over pairs of the OLS slope of ln s_N on ln m)")}
A["median_slope_k10_recomputed_from_slopes"] = {CKEY[c]: float(sp[(sp.cls == c) & (sp.k == 10)].slope_obs.median()) for c in CLS}
for c in CLS:
    assert abs(A["median_slope_k10"]["values"][CKEY[c]] - A["median_slope_k10_recomputed_from_slopes"][CKEY[c]]) < 1e-12
lc = np.log([A["chance_k10"]["values"][mm] for mm in MS])
A["chance_slope_k10"] = {"value": float(np.polyfit(np.log(MS), lc, 1)[0]),
                         "definition": "OLS slope of ln(10/(m-1)) on ln m over the 7 pool sizes (analytic, about -1)"}
A["slope_classes_k10"] = {CKEY[c]: {kk: C["slope_classes"]["primary"][c]["10"][kk] for kk in ("rising", "flat", "falling", "n")}
                          for c in CLS}
A["obs_over_chance_k10_min_all_m"] = {CKEY[c]: float((ca[ca.cls == c].obs / ca[ca.cls == c].chance).min()) for c in CLS}
A["obs_over_chance_k10_m16384_min"] = {CKEY[c]: float((g.obs / g.chance).min()) for c in CLS
                                       for g in [ca[(ca.cls == c) & (ca.m == 16384)]]}
A["draw_sd_over_obs_k10_max"] = {CKEY[c]: float((ca[ca.cls == c].obs_draw_sd / ca[ca.cls == c].obs).max()) for c in CLS}
N["panel_a"] = A

B = {}
B["class_median_obs"] = {"values": {f"{CKEY[c]}|k={kk}": {int(mm): float(med[(c, kk, mm)]) for mm in MS} for c in CLS for kk in KS},
                         **src(F_CELLS, "median over pairs of cells[primary & cls & k].obs by m")}
B["k_over_m"] = {f"k={kk}": {int(mm): kk / mm for mm in MS} for kk in KS}
B["chance_diagonal"] = "y = x (chance k/(m-1) ~ k/m)"
B["elasticity_by_class_summary"] = {CKEY[c]: C["secondary"]["primary"]["elasticity_by_class"][c]
                                    for c in CLS} if isinstance(C["secondary"]["primary"]["elasticity_by_class"], dict) else \
    C["secondary"]["primary"]["elasticity_by_class"]
B["elasticity_source"] = src(F_SUMMARY, "configs.clip.secondary.primary.elasticity_by_class (quoted in the text, not plotted)")
RAT = {}
for c in CLS:
    k_ = CKEY[c]
    x10 = np.log([10 / mm for mm in MS])[::-1]
    y10 = np.log([B["class_median_obs"]["values"][f"{k_}|k=10"][mm] for mm in MS])[::-1]
    for kk in (1, 100):
        row = {}
        for mm in MS:
            xv = np.log(kk / mm)
            if x10[0] - 1e-12 <= xv <= x10[-1] + 1e-12:
                row[int(mm)] = float(B["class_median_obs"]["values"][f"{k_}|k={kk}"][mm] / np.exp(np.interp(xv, x10, y10)))
        RAT[f"{k_}|k={kk}"] = row
B["ratio_to_k10_at_shared_k_over_m"] = {
    "values": RAT,
    "summary": {f"{k_}|k={kk}": [min(RAT[f"{k_}|k={kk}"].values()), max(RAT[f"{k_}|k={kk}"].values())]
                for k_ in ("seed", "vision", "vl") for kk in (1, 100)},
    "definition": "class median at k divided by the k = 10 class median interpolated in (ln k/m, ln s_N) at the same k/m; "
                  "only k/m inside the k = 10 range [10/16384, 10/256] (k = 1: m = 256, 512, 1024; k = 100: m = 4096, 8192, 16384)",
    **src(F_CELLS, "derived from panel_b.class_median_obs")}
N["panel_b"] = B

o3 = json.load(open(OLD_F3))
OLDC = {"seed": "seed-seed", "vision": "cross-vision", "vl": "cross-modal"}
dev = 0.0
for p, row in A["per_pair_obs_k10"]["values"].items():
    for mm, v in row.items():
        dev = max(dev, abs(r4(v) - o3["panel_a"]["per_pair_obs_k10"][p][str(mm)]))
    assert CKEY[o3["panel_a"]["per_pair_cls"][p]] == A["per_pair_cls"][p]
for k_, c in OLDC.items():
    for mm in MS:
        dev = max(dev, abs(r4(A["class_median_obs_k10"]["values"][k_][mm]) - o3["panel_a"]["class_median_obs_k10"][c][str(mm)]))
        dev = max(dev, abs(r4(A["chance_k10"]["values"][mm], 6) - o3["panel_a"]["chance_k10"][str(mm)]))
        for kk in KS:
            dev = max(dev, abs(r4(B["class_median_obs"]["values"][f"{k_}|k={kk}"][mm]) - o3["panel_b"]["class_median_obs"][f"{c}|k={kk}"][str(mm)]))
    assert abs(A["median_slope_k10"]["values"][k_] - o3["panel_a"]["slope_classes_primary_summary"][f"{c}|k=10"]["median_slope"]) < 1e-12
for kk in KS:
    for mm in MS:
        dev = max(dev, abs(r4(kk / mm, 7) - o3["panel_b"]["k_over_m"][f"k={kk}"][str(mm)]))
assert dev < 1e-9, dev
N["assert_vs_F3_numbers"] = {"file": OLD_F3, "max_abs_diff_after_rounding": dev,
                             "keys": ["panel_a.per_pair_obs_k10", "panel_a.per_pair_cls", "panel_a.class_median_obs_k10",
                                      "panel_a.chance_k10", "panel_a.slope_classes_primary_summary[<cls>|k=10].median_slope",
                                      "panel_b.class_median_obs", "panel_b.k_over_m"], "result": "all equal"}

# ================================================================== (c): S1.2 loader and bound (v5 maker, unchanged)
d = pd.read_parquet(P_POOL)
d = d[d.pre == PRE].copy()
assert len(d) == 272 and (d["L.m"] == 2047).all()
m = 2047.0
d["sN1"] = d["L.s_N_k1_for_mst"].astype(float)
d["sT"] = d["L.s_T"].astype(float)
d["nmut"] = d["L.m"] - d["M.n_forest_A"]
d["bound"] = np.maximum(m * d.sN1 - d.nmut, m * d.sN1 / 2) / (m - 1)
assert np.allclose(d["L.s_N_k1"], d.sN1)
assert np.allclose(d["M.n_shared_edges"] / (m - 1), d.sT, atol=1e-12)
assert (d["M.n_bridges_A"] == d.nmut - 1).all()
allc = []
for p in (P_POOL, P_MAIN):
    x = pd.read_parquet(p)
    mm_ = x["L.m"].astype(float)
    s1 = x["L.s_N_k1_for_mst"].astype(float)
    nm = mm_ - x["M.n_forest_A"]
    b = np.maximum(mm_ * s1 - nm, mm_ * s1 / 2) / (mm_ - 1)
    allc.append(pd.DataFrame({"ok": x["L.s_T"] >= b - 1e-12, "half": mm_ * s1 / 2 >= mm_ * s1 - nm, "ratio": x["L.s_T"] / b}))
allc = pd.concat(allc)
CC = {}
CC["setting"] = {"pool": "WIT, m = 2047 (panel_pool2047, designated PE2 pool)", "pre": PRE,
                 "pairs": "8 language models x 17 vision transformers = 136 vision-language pairs per depth",
                 "depth": {"0.5": "mid depth (open markers)", "1.0": "last layer (filled markers)"}}
CC["points"] = {"values": {f"r={r}": [{"lm": a_, "vit": b_, "sN1": float(u), "sT": float(v)}
                                      for a_, b_, u, v in g[["lm", "vit", "sN1", "sT"]].itertuples(index=False)]
                           for r, g in d.groupby("r")},
                **src(P_POOL, "rows pre == prh_clip+l2; x = L.s_N_k1_for_mst, y = L.s_T, grouped by r")}
CC["bound_line"] = {"definition": "s_T >= max{m s_N1 - n_mut_A, m s_N1 / 2}/(m-1) (Proposition prop:mst); the plotted line "
                                  "is the half branch m s_N1 / (2(m-1)) with m = 2047, which binds in every configuration",
                    "slope": m / (2 * (m - 1)), "m": m}
CC["bound_check_all_configs"] = {"n_configs": int(len(allc)), "n_bound_holds": int(allc.ok.sum()),
                                 "n_half_term_binds": int(allc.half.sum()), "sT_over_bound_min": float(allc.ratio.min()),
                                 "sT_over_bound_max": float(allc.ratio.max()),
                                 **src(P_MAIN, "panel_pool2047 (272) + panel_main (544) rows; L.m, L.s_N_k1_for_mst, M.n_forest_A, L.s_T")}
CC["bound_check_all_configs"]["composition"] = {
    "panel_pool2047 (plotted in (c))": "m = 2047, prh_clip+l2, r in {0.5, 1.0}, 136 pairs each: 272",
    "panel_main (not plotted)": "m = 1024, prh_clip+l2 and l2, r in {0.5, 1.0}, 136 pairs each: 544"}
CC["bound_check_plotted"] = {"n_configs": int(len(d)), "n_bound_holds": int((d.sT >= d.bound - 1e-12).sum()),
                             "sT_over_bound_min": float((d.sT / d.bound).min()),
                             "sT_over_bound_median": float((d.sT / d.bound).median()),
                             **src(P_POOL, "rows pre == prh_clip+l2 (the 272 plotted points); L.s_T / bound")}
assert CC["bound_check_plotted"]["n_bound_holds"] == 272
CC["summary_by_depth"] = {}
for r, g in d.groupby("r"):
    CC["summary_by_depth"][f"r={r}"] = {
        "n_pairs": int(len(g)), "median_sT": float(g.sT.median()), "median_sN1": float(g.sN1.median()),
        "median_sT_over_sN1": float((g.sT / g.sN1).median()),
        "q05_q95_sT_over_sN1": [float((g.sT / g.sN1).quantile(.05)), float((g.sT / g.sN1).quantile(.95))],
        "frac_sT_above_sN1": float((g.sT > g.sN1).mean()),
        "median_sT_over_bound": float((g.sT / g.bound).median()), "min_sT_over_bound": float((g.sT / g.bound).min()),
        "max_sN1": float(g.sN1.max()), "max_sT": float(g.sT.max())}
N["panel_c"] = CC

o2 = json.load(open(OLD_F2))
for r in ("0.5", "1.0"):
    for kk in ("median_sT", "median_sN1", "median_sT_over_sN1", "frac_sT_above_sN1", "median_sT_over_bound", "min_sT_over_bound"):
        assert abs(CC["summary_by_depth"][f"r={r}"][kk] - o2["panel_a"][f"r={r}"][kk]) < 1e-12, (r, kk)
for kk in ("n_configs", "n_bound_holds", "n_half_term_binds", "sT_over_bound_min", "sT_over_bound_max"):
    assert abs(CC["bound_check_all_configs"][kk] - o2["bound_check_all_configs"][kk]) < 1e-12, kk
assert CC["bound_check_all_configs"]["n_bound_holds"] == 816 == CC["bound_check_all_configs"]["n_configs"]
N["assert_vs_F2_numbers"] = {"file": OLD_F2, "keys": ["panel_a['r=0.5'|'r=1.0'] medians and ratios", "bound_check_all_configs"],
                             "result": "all equal to 1e-12"}
N["panel_a"]["obs_over_chance_k10_min_by_m"] = {
    "values": {CKEY[c]: {int(mm): float(v) for mm, v in (ca[ca.cls == c].obs / ca[ca.cls == c].chance).groupby(ca[ca.cls == c].m).min().items()}
               for c in CLS},
    **src(F_CELLS, "min over pairs of cells[primary & k==10 & cls].obs / .chance by m"),
    "note": "8.75 is the vision-language minimum at m = 16384 only; at m = 256 the minimum is 1.93"}


def jdiff(a, b, path=""):
    """Paths where two JSON trees differ (NaN equals NaN; floats compared exactly)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k} (missing on one side)" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += jdiff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} (length {len(a)} vs {len(b)})"]
        return [q for i, (u, v) in enumerate(zip(a, b)) for q in jdiff(u, v, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


# ---- v5 assert kept: every data block equals the v4 render's numbers.json (input file hashes aside)
o4 = json.load(open(V4_NUM))
V4_KEYS = ("claim", "s11_setting", "n_primary_pairs", "panel_a", "panel_b", "panel_c",
           "assert_vs_F3_numbers", "assert_vs_F2_numbers")
DD = {kk: jdiff(json.loads(json.dumps(N[kk], default=float)), o4[kk], kk) for kk in V4_KEYS}
VAL = [q for v in DD.values() for q in v if "/sha256:" not in q]
assert not VAL, ("a value differs from the v4 numbers.json", VAL[:10])
N["assert_vs_v4_numbers"] = {"file": V4_NUM, "sha256": INS[V4_NUM], "keys": list(V4_KEYS),
                             "result": "every value identical after JSON round trip (NaN = NaN); only input file hashes differ",
                             "n_value_differences": len(VAL)}
# ---- v7: every plotted block equals the v5.1 render's numbers.json exactly, input hashes included (the v6 numbers.json
# copied these blocks verbatim), so the redraw plots the same values as the figure in the paper
D5 = {kk: jdiff(json.loads(json.dumps(N[kk], default=float)), O5[kk], kk)
      for kk in ("claim", "s11_setting", "n_primary_pairs", "panel_a", "panel_b", "panel_c")}
assert not any(D5.values()), {k: v[:5] for k, v in D5.items() if v}
O6 = json.load(open(f"{V6}/numbers.json"))
for kk in ("panel_a", "panel_b", "panel_c"):
    assert not jdiff(json.loads(json.dumps(N[kk], default=float)), O6[kk], kk), kk
N["assert_vs_v5_v6_numbers"] = {"files": [f"{V5}/numbers.json", f"{V6}/numbers.json"],
                                "keys": ["claim", "s11_setting", "n_primary_pairs", "panel_a", "panel_b", "panel_c"],
                                "result": "identical, input hashes included"}
# ---- v8: every plotted block and every Results record equals the v7 numbers.json (v7 render of the same data)
D7 = {kk: jdiff(json.loads(json.dumps(N[kk], default=float)), O7[kk], kk)
      for kk in ("claim", "s11_setting", "n_primary_pairs", "panel_a", "panel_b", "panel_c")}
assert not any(D7.values()), {k: v[:5] for k, v in D7.items() if v}
assert json.loads(json.dumps(RES, default=float)) == O7["results_text_numbers"], "Results records differ from v7"
N["assert_vs_v7_numbers"] = {"file": f"{V7}/numbers.json", "sha256": sha(f"{V7}/numbers.json"),
                             "keys": ["claim", "s11_setting", "n_primary_pairs", "panel_a", "panel_b", "panel_c",
                                      "results_text_numbers"],
                             "result": "identical, input hashes included"}

# ================================================================== drawing (inches throughout)
W = 5.5
H = float(os.environ.get("FN_H", 1.70))
BOT = 0.37                                   # x tick labels and x labels
TOPM = float(os.environ.get("FN_TOPM", 0.22))   # panel letters only (v5: 0.28, which held the 'slope' header); the
#                                                 letters must clear the top tick label '0.09' of (c), which they overhang
AXH = round(H - BOT - TOPM, 4)
XA = 0.47                                    # y label and tick labels of (a) (ink-checked on the PDF below)
RM = float(os.environ.get("FN_RM", 0.12))    # right: half of the tick label '0.09' of (c)
G1 = float(os.environ.get("FN_G1", 0.24))    # (a) -> (b): half of the tick label '16,384' and the y ticks of (b)
G2 = float(os.environ.get("FN_G2", 0.50))    # (b) -> (c): y label and tick labels of (c)
WC = AXH                                     # (c) square, equal data ranges
XC = round(W - RM - WC, 4)
WA = WB = round((XC - G2 - G1 - XA) / 2, 4)
XB = round(XA + WA + G1, 4)
fig = plt.figure(figsize=(W, H))
fig.set_layout_engine("none")
axA = fig.add_axes([XA / W, BOT / H, WA / W, AXH / H])
axB = fig.add_axes([XB / W, BOT / H, WB / W, AXH / H], sharey=axA)
axC = fig.add_axes([XC / W, BOT / H, WC / W, AXH / H])
YTOP = float(os.environ.get("FN_YTOP", 1.5))   # headroom for the seed name above the seed bundle (tick 1 kept)
YLIM = (5e-4, YTOP)
XLA = (220, 16384 * 256 / 220)               # the same log margin left of m = 256 and right of m = 16,384
XLB = (4e-5, 0.6)
LIM = 0.09
PC = {CKEY[c]: S.PAIR[CKEY[c]] for c in CLS}
PT = {CKEY[c]: S.PAIR_TEXT[CKEY[c]] for c in CLS}
XCH = float(os.environ.get("FN_XCH", 2200))
ANN, DATA = [], []
LEAD = []


def logy(ax):
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1: "1"}.get(v, "")))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def add(artists):
    DATA.extend(artists if isinstance(artists, list) else [artists])


def note(t):
    ANN.append(t)
    return t


def _dense(P, n=60):
    seg = [P[i] + (P[i + 1] - P[i]) * np.linspace(0, 1, n, endpoint=False)[:, None] for i in range(len(P) - 1)]
    return np.vstack(seg + [P[-1:]])


def label_along(ax, s, polylines, x_anchor, side, ha, role, color=None, gap_pt=1.6, lw=0.5, angle_deg=None):
    """Text s parallel to a bundle of polylines (data coords), just below or above every line of the bundle over the
    text's span (v5 maker, unchanged)."""
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


# ---- (a) mKNN at k = 10 against m -------------------------------------------------------------------------------
ax = axA
ax.set_xscale("log", base=2)
ax.set_xlim(*XLA)
logy(ax)
BUND = {k_: [] for k_ in PC}
for p, g in ca.groupby("pair"):
    g = g.sort_values("m")
    k_ = CKEY[g.cls.iloc[0]]
    BUND[k_].append((g.m.values.astype(float), g.obs.values))
    add(ax.plot(g.m, g.obs, color=PC[k_], lw=LW["data_thin"], zorder=Z["thin"]))
for c in CLS:
    k_ = CKEY[c]
    yy = [A["class_median_obs_k10"]["values"][k_][mm] for mm in MS]
    add(S.underlay(ax, MS, yy, LW["data_emph"], Z["data"]))
    add(ax.plot(MS, yy, color=PC[k_], lw=LW["data_emph"], zorder=Z["data"]))
    # v7 (the v6 option render): the k = 10 medians take the open k = 10 square of (b), so (b)'s key decodes k here
    add(ax.plot(MS, yy, ls="none", marker="s", ms=MSZ["scatter"], mfc="white", mec=PC[k_], mew=MSZ["open_small_mew"],
                zorder=Z["marker"]))
CH = np.array([A["chance_k10"]["values"][mm] for mm in MS])
add(ax.plot(MS, CH, color=S.NULL_LINE, lw=LW["ref"], ls=S.LS["dashed"], zorder=Z["ref"]))
S.int_ticks(ax, [256, 1024, 4096, 16384], "x")
ax.xaxis.set_minor_locator(FixedLocator([512, 2048, 8192]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("evaluation size $m$")
ax.set_ylabel("mKNN $s_N$")
XNAME = float(os.environ.get("FN_XNAME", 15500))
XSEED = float(os.environ.get("FN_XSEED", 2048))
label_along(ax, S.PAIR_LABEL["seed"], BUND["seed"], XSEED, "above", "center", "direct", PT["seed"],
            gap_pt=1.5, lw=LW["data_thin"])
for k_ in ("vision", "vl"):
    label_along(ax, S.PAIR_LABEL[k_], BUND[k_], XNAME, "below", "right", "direct", PT[k_],
                gap_pt=1.5, lw=LW["data_thin"])
# v5 kept the median slopes as a column of numbers right of m = 16,384; v7 drops it (the Results text states them).
# The formatted values are still checked against the summary, as in v5.
SLO = {k_: S.num(A["median_slope_k10"]["values"][k_]) for k_ in PC}
SLO["chance"] = S.num(A["chance_slope_k10"]["value"])
assert SLO == {k_: neg(f'{v:.2f}') for k_, v in {**A["median_slope_k10"]["values"],
                                                  "chance": A["chance_slope_k10"]["value"]}.items()}
assert SLO == {"seed": neg("-0.01"), "vision": neg("-0.21"), "vl": neg("-0.57"), "chance": neg("-1.00")}, SLO
label_along(ax, "chance", [(np.array(MS, float), CH)], XCH, "below", "center", "annot", gap_pt=1.5, lw=LW["ref"])

# ---- (b) class medians against k/m, marker = k (v5 fix-3 layering, unchanged) ----------------------------------
ax = axB
ax.set_xscale("log")
ax.set_xlim(*XLB)
logy(ax)
B_MS = MSZ["line"]
B_EDGE = MSZ["line_edge"]
B_MS_OPEN = MSZ["scatter"]
B_MEW_OPEN = MSZ["open_small_mew"]
KMS = {kk: B_MS + (MSZ["tri_extra"] if KMARK[kk] == "^" else 0.0) for kk in KS}
KORDER = [10, 1, 100]
KRANK = {kk: len(KORDER) - 1 - i for i, kk in enumerate(KORDER)}
CRANK = {"vl": 0, "vision": 1, "seed": 2}


def bmark(kk, col):
    if kk == 10:
        return dict(marker="s", ms=B_MS_OPEN, mfc="white", mec=col, mew=B_MEW_OPEN)
    return dict(marker=KMARK[kk], ms=KMS[kk], mfc=col, mec="white", mew=B_EDGE)


ZB = {}
for c in CLS:
    k_ = CKEY[c]
    for kk in KS:
        xs = [kk / mm for mm in MS]
        ys = [B["class_median_obs"]["values"][f"{k_}|k={kk}"][mm] for mm in MS]
        zm = Z["marker"] + 0.3 * CRANK[k_] + 0.05 * KRANK[kk]
        zc = Z["marker"] + 0.3 * CRANK[k_] + 0.05 * KRANK[KORDER[0]] - 0.025
        ZB[f"{k_}|k={kk}"] = {"marker": round(zm, 3), "connector": round(zc, 3)}
        add(ax.plot(xs, ys, color=PC[k_], lw=LW["connector"], zorder=zc))
        add(ax.plot(xs, ys, ls="none", zorder=zm, **bmark(kk, PC[k_])))
xx = np.logspace(np.log10(XLB[0]), np.log10(XLB[1]), 60)
add(ax.plot(xx, xx, color=S.NULL_LINE, lw=LW["ref"], ls=S.LS["dashed"], zorder=Z["ref"]))
ax.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1"}.get(v, "")))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis="y", which="both", labelleft=False)
ax.set_xlabel("$k/m$")
label_along(ax, "chance", [(xx, xx)], float(os.environ.get("FN_XCHB", 2.2e-3)), "below", "center", "annot",
            gap_pt=1.5, lw=LW["ref"])
# key: rows a fixed 9 pt apart and the lowest row a fixed height above the x axis, whatever the axes height (v5: rows
# 8.9 pt apart from y = 0.0034 down, the lowest 4.6 pt above the axis at 1.30 in)
KEYX = float(os.environ.get("FN_KEYX", 0.020))
KEY_ROW_PT = 9.0
KEY_LOW_PT = float(os.environ.get("FN_KEYLOW", 5.0))
_dec_per_pt = np.log10(YLIM[1] / YLIM[0]) / (AXH * 72)
KEYF = 10 ** (KEY_ROW_PT * _dec_per_pt)
KEYY0 = YLIM[0] * 10 ** ((KEY_LOW_PT + 2 * KEY_ROW_PT) * _dec_per_pt)
for i, kk in enumerate(KS):
    yk = KEYY0 / KEYF ** i
    add(ax.plot([KEYX], [yk], ls="none", zorder=Z["marker"], **bmark(kk, S.INK)))
    note(S.text(ax, KEYX * 1.5, yk, f"$k={kk}$", "key", ha="left", va="center", zorder=Z["label"]))

# ---- (a, b) spines: the shared y spine ends at the top tick 1 (v5)
for a_ in (axA, axB):
    a_.spines["left"].set_bounds(YLIM[0], 1.0)

# ---- (c) MST overlap against 1-NN overlap (v5, unchanged) ------------------------------------------------------
ax = axC
G, GT = S.CRIT["mst"], S.CRIT_TEXT["mst"]
xs = np.linspace(0, LIM, 50)
bl = m * xs / (2 * (m - 1))
FILL_C = ax.fill_between(xs, 0, bl, color=S.NULL_FILL, lw=0, zorder=Z["fill"])
add(FILL_C)
add(ax.plot(xs, xs, color=S.NULL_LINE, lw=LW["ref"], ls=S.LS["dotted"], zorder=Z["ref"]))
add(ax.plot(xs, bl, color=S.INK, lw=LW["theory"], zorder=Z["ref"] + 0.1))
MK = {1.0: dict(ls="none", marker="o", ms=MSZ["scatter"], mfc=G, mec=G, mew=MSZ["scatter_edge"]),
      0.5: dict(ls="none", marker="o", ms=MSZ["scatter"], mfc="white", mec=G, mew=MSZ["scatter_open_mew"])}
for r, g in d.groupby("r"):
    add(ax.plot(g.sN1, g.sT, zorder=Z["marker"] if r == 0.5 else Z["data"], **MK[r]))
ax.set_xlim(0, LIM)
ax.set_ylim(0, LIM)
tk = [0, 0.03, 0.06, 0.09]
ax.set_xticks(tk, ["0", "0.03", "0.06", "0.09"])
ax.set_yticks(tk, ["0", "0.03", "0.06", "0.09"])
ax.set_xlabel(r"1-NN overlap $s_{N{,}1}$")
ax.set_ylabel(r"MST overlap $s_{\mathcal{T}}$")
label_along(ax, "bound", [(xs, bl)], 0.066, "above", "center", "theory_label", gap_pt=1.6, lw=LW["theory"])
label_along(ax, "$y=x$", [(xs, xs)], 0.0745, "below", "center", "annot", gap_pt=1.5, lw=LW["ref"])
T_EXCL = note(S.on_fill(S.text(ax, 0.0875, 0.0035, "excluded", "annot", ha="right", va="bottom",
                               zorder=Z["label"]), S.NULL_FILL))
PX, PY, PR = d.sN1.to_numpy(float), d.sT.to_numpy(float), d.r.to_numpy(float)
RAD = (MSZ["scatter"] / 2 / 72) * LIM / WC
C_LAB = {}


def text_wh(s, role, **kw):
    t = ax.text(0, 0, s, **S.text_kw(role, s, GT), **kw)
    fig.canvas.draw()
    bb = t.get_window_extent()
    t.remove()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x1 - x0, y1 - y0


GAP = (2.0 / 72) * LIM / WC
wL, hL = text_wh("last layer", "direct")
YL = float(os.environ.get("FN_YL", 0.0795))
band = np.abs(PY - YL) <= hL / 2 + RAD + GAP
assert band.any() and (PR[band] == 1.0).all()
XL_R = float(PX[band].min()) - RAD - GAP
assert XL_R - wL > 0.002, (XL_R, wL)
note(S.text(ax, XL_R, YL, "last layer", "direct", GT, ha="right", va="center", zorder=Z["label"]))
C_LAB["last layer"] = {"text_right_x": XL_R, "text_center_y": YL,
                       "rule": "right-aligned left of the leftmost point within the text's height band (all last layer)"}
wM, hM = text_wh("mid\ndepth", "direct", linespacing=1.0)
XM = float(os.environ.get("FN_XM", 0.0025))
GAP_M = (3.5 / 72) * LIM / WC
YM = None
for yb in np.arange(0.030, 0.06, 0.0005):
    inbox = ((PX + RAD + GAP_M >= XM) & (PX - RAD - GAP_M <= XM + wM)
             & (PY + RAD + GAP_M >= yb) & (PY - RAD - GAP_M <= yb + hM))
    if not inbox.any():
        YM = float(yb)
        break
assert YM is not None
openp = d[(d.r == 0.5) & (d.sN1 <= float(os.environ.get("FN_TGTX", 0.0115)))]
tgt = openp.sort_values("sT").iloc[-1]
TX, TY = float(tgt.sN1), float(tgt.sT)
note(S.text(ax, XM, YM, "mid\ndepth", "direct", GT, ha="left", va="bottom", linespacing=1.0, zorder=Z["label"]))
LX0, LY0 = XM + 0.30 * wM, YM - 0.0009
LEAD.append(ax.annotate("", xy=(TX, TY), xytext=(LX0, LY0), zorder=Z["label"],
                        arrowprops=dict(arrowstyle="-", lw=LW["leader"], color=S.INK, shrinkA=0,
                                        shrinkB=MSZ["scatter"] / 2 + MSZ["scatter_open_mew"] / 2 + 0.8)))
fil = d[d.r == 1.0]
assert fil.sN1.min() > max(LX0, TX) + RAD
C_LAB["mid depth"] = {"text_left_bottom": [XM, YM], "leader_from": [LX0, LY0], "leader_to": [TX, TY],
                      "leader_to_pair": [str(tgt.lm), str(tgt.vit)],
                      "rule": "lowest box from s_T = 0.030 clearing every point by 3.5 pt; leader to the highest "
                              "open point with s_N,1 <= 0.0115"}

# ---- v8: bold tick labels and axis titles with bold math, before the layout is measured (letters, row check, audit)
N_FINALIZED = S.finalize(fig)

# ---- panel letters: same baseline, each at the left ink edge of its panel
fig.canvas.draw()
R0 = fig.canvas.get_renderer()
LEFT = {}
for k_, a_ in (("a", axA), ("b", axB), ("c", axC)):
    LEFT[k_] = min(a_.get_tightbbox(R0).x0, a_.get_window_extent(R0).x0) / fig.dpi
LET = [S.panel_letter(fig, k_, max(0.01, LEFT[k_] - 0.01), H - 0.02) for k_ in ("a", "b", "c")]
N["letters_x_in"] = {k_: round(max(0.01, LEFT[k_] - 0.01), 3) for k_ in LEFT}

# ---- v7 row check: gaps between the ink boxes of neighbouring panels (tick labels and axis labels included), the
# right margin, and the clearance of each letter above its axes and from the data (letters sit in the top margin)
fig.canvas.draw()
TB = {k_: a_.get_tightbbox(R0) for k_, a_ in (("a", axA), ("b", axB), ("c", axC))}
ROW = {"gap_a_b_in": round((TB["b"].x0 - TB["a"].x1) / fig.dpi, 3),
       "gap_b_c_in": round((TB["c"].x0 - TB["b"].x1) / fig.dpi, 3),
       "right_margin_in": round((fig.bbox.x1 - TB["c"].x1) / fig.dpi, 3),
       "left_margin_in": round(TB["a"].x0 / fig.dpi, 3),
       "letter_bottom_minus_axes_top_in": {}, "data_width_in": {"a": WA, "b": WB, "c": WC},
       "axes_in": {"a": [XA, BOT, WA, AXH], "b": [XB, BOT, WB, AXH], "c": [XC, BOT, WC, AXH], "figure": [W, H]}}
for t, (k_, a_) in zip(LET, (("a", axA), ("b", axB), ("c", axC))):
    ROW["letter_bottom_minus_axes_top_in"][k_] = round((t.get_window_extent(R0).y0 - a_.get_window_extent(R0).y1) / fig.dpi, 3)
assert ROW["gap_a_b_in"] >= 0.04 and ROW["gap_b_c_in"] >= 0.04, ROW
assert 0.0 <= ROW["right_margin_in"] <= 0.06 and ROW["left_margin_in"] <= 0.06, ROW

TRIAL = os.environ.get("FN_TRIAL")
OUT = os.path.join(HERE, "_trials", TRIAL) if TRIAL else HERE
os.makedirs(OUT, exist_ok=True)
out_pdf, out_png = os.path.join(OUT, "F_neighbors.pdf"), os.path.join(OUT, "F_neighbors.png")
N_FINALIZED_2 = S.finalize(fig)                     # v8: last call before savefig (idempotent)
fig.savefig(out_pdf)
chk = S.check(fig, out_pdf)
assert chk["n_tick_labels_checked"] > 0 and chk["n_axis_titles_checked"] == 5, chk["n_axis_titles_checked"]
assert not chk["not_bold_tick_or_axis_labels"] and not chk["bold_text_with_regular_math"]
assert chk["chromatic_data_colours"], "black-and-white figure"
pf = subprocess.run(["pdffonts", out_pdf], capture_output=True, text=True).stdout
N["pdf_checks"] = {**chk, "pdffonts": pf.strip().splitlines()}
assert all(("TrueType" in ln and " yes " in ln) for ln in pf.strip().splitlines()[2:]), pf
assert chk["n_type3"] == 0 and chk["min_glyph_pt"] >= S.MIN_PT - 1e-9 and not chk["low_contrast_text"]
# the PNG is the PDF rendered at 200 dpi (poppler), and a print-size view at 100 dpi (5.5 in = 550 px)
subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", out_pdf, out_png[:-4]], check=True)
im = Image.open(out_png)
im.load()
im.save(out_png, dpi=(200, 200))
N["png_px"] = list(Image.open(out_png).size)
assert N["png_px"] == [int(round(W * 200)), int(round(H * 200))], N["png_px"]
VIEW = os.path.join(OUT, "_view_print_100dpi.png")
Image.open(out_png).convert("RGB").resize((550, int(round(H * 100))), Image.LANCZOS).save(VIEW, dpi=(100, 100))
# v8: poppler render at 100 dpi (print size) and the normal / deuteranopia / protanopia / greyscale views
VIEW_PP = os.path.join(OUT, "_view_print_100dpi_pdftoppm.png")
_tmp200 = os.path.join(OUT, "_render_200dpi_check.png")
S.print_view(out_pdf, _tmp200, VIEW_PP)
assert list(Image.open(_tmp200).size) == N["png_px"]
os.remove(_tmp200)
VIEWS = os.path.join(OUT, "_views_cvd_grey.png")
S.views(out_png, VIEWS)
# v8: the colours the figure draws, against the semantic map (every data colour from the module's palette)
SEM_USED = {"pair.seed": S.PAIR["seed"], "pair.vision": S.PAIR["vision"], "pair.vision-language": S.PAIR["vl"],
            "criterion.MST": S.CRIT["mst"], "theory": S.INK, "reference.line": S.NULL_LINE,
            "reference.fill": S.NULL_FILL}
for k_, v_ in SEM_USED.items():
    if k_.startswith("reference."):
        continue
    assert S.SEM[k_].upper() == v_.upper(), k_
assert {c.upper() for c in (S.PAIR["seed"], S.PAIR["vision"], S.PAIR["vl"], S.CRIT["mst"])} <= set(chk["data_colours"])
assert chk["off_palette_colours"] == [], chk["off_palette_colours"]
CVD = {}
for v_ in ("normal", "deutan", "protan", "grey"):
    labs = {k_: S.lab(S.simulate(h_, v_)) for k_, h_ in (("seed", S.PAIR["seed"]), ("vision", S.PAIR["vision"]),
                                                         ("vl", S.PAIR["vl"]))}
    CVD[v_] = {"L*": {k_: round(float(l_[0]), 1) for k_, l_ in labs.items()},
               "min_dL_between_classes": round(float(min(abs(labs[a][0] - labs[b][0]) for a, b in
                                                         (("seed", "vision"), ("vision", "vl"), ("seed", "vl")))), 1)}
    assert CVD[v_]["min_dL_between_classes"] >= 15, CVD[v_]   # the three classes stay apart by lightness in every view

# ================================================================== layout audit on rendered ink (v5 maker, unchanged)
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
lead_ids = {id(a) for a in LEAD}
frame = [a for a in LEAVES if id(a) not in ann_ids and id(a) not in data_ids and id(a) not in lead_ids
         and a not in (fig.patch, axA.patch, axB.patch, axC.patch)]
M_data, M_frame = ink(data_leaves), ink(frame)
st = np.ones((7, 7), bool)
D_data, D_frame = binary_dilation(M_data, st), binary_dilation(M_frame, st)
M_ann = [ink([t]) for t in ANN]
audit = {"dpi": DPI_A, "dilation_px": 3, "text_vs_data": {}, "text_vs_frame": {}, "text_vs_text": []}
PAN = {id(axA): "a", id(axB): "b", id(axC): "c"}
TAG = [f'({PAN.get(id(t.axes), "fig")}) {t.get_text()}' for t in ANN]
assert len(set(TAG)) == len(TAG), TAG


def where(Mx):
    yy_, xx_ = np.nonzero(Mx)
    return None if not len(xx_) else [round(float(xx_.min()) / DPI_A, 3), round(float(H - yy_.max() / DPI_A), 3),
                                      round(float(xx_.max()) / DPI_A, 3), round(float(H - yy_.min() / DPI_A), 3)]


fill_ids = {id(FILL_C)}
M_data_nf = ink([a for a in data_leaves if id(a) not in fill_ids])
D_data_nf = binary_dilation(M_data_nf, st)
M_fill_in = binary_erosion(ink([FILL_C]), st)
ON_FILL = {i for i, t in enumerate(ANN) if t is T_EXCL}
audit["on_fill_labels"] = {}
audit["where_in"] = {}
for i, (tg, Mi) in enumerate(zip(TAG, M_ann)):
    audit["text_vs_data"][tg] = int((Mi & (D_data_nf if i in ON_FILL else D_data)).sum())
    audit["text_vs_frame"][tg] = int((Mi & D_frame).sum())
    if i in ON_FILL:
        audit["on_fill_labels"][tg] = {"fill": S.NULL_FILL, "ink_px": int(Mi.sum()),
                                       "ink_px_outside_wedge_eroded_0.7pt": int((Mi & ~M_fill_in).sum())}
        assert audit["on_fill_labels"][tg]["ink_px_outside_wedge_eroded_0.7pt"] == 0, audit["on_fill_labels"][tg]
    if audit["text_vs_data"][tg] or audit["text_vs_frame"][tg]:
        audit["where_in"][tg] = where(Mi & ((D_data_nf if i in ON_FILL else D_data) | D_frame))
dist_data = distance_transform_edt(~M_data)
dist_data_nf = distance_transform_edt(~M_data_nf)
dist_frame = distance_transform_edt(~M_frame)
audit["clearance_pt"] = {tg: {"data": round(float((dist_data_nf if i in ON_FILL else dist_data)[Mi].min()) * 72 / DPI_A, 2),
                              "frame": round(float(dist_frame[Mi].min()) * 72 / DPI_A, 2)}
                         for i, (tg, Mi) in enumerate(zip(TAG, M_ann))}
for i in range(len(ANN)):
    Di = binary_dilation(M_ann[i], st)
    for j in range(i + 1, len(ANN)):
        n_ = int((Di & M_ann[j]).sum())
        if n_:
            audit["text_vs_text"].append([TAG[i], TAG[j], n_])
# v7: the panel letters now sit in a lower top margin, so each letter's ink is also checked against the data ink
M_let = [ink([t]) for t in LET]
audit["letter_vs_data_px"] = {t.get_text(): int((Ml & D_data).sum()) for t, Ml in zip(LET, M_let)}
audit["letter_clearance_data_pt"] = {t.get_text(): round(float(dist_data[Ml].min()) * 72 / DPI_A, 2) for t, Ml in zip(LET, M_let)}
assert not any(audit["letter_vs_data_px"].values()), audit["letter_vs_data_px"]
fig.canvas.draw()
R = fig.canvas.get_renderer()
ftxt = [a for a in frame if isinstance(a, matplotlib.text.Text) and a.get_visible() and a.get_text().strip()]
audit["frame_text_box_overlaps"] = [[a.get_text(), b.get_text()] for i, a in enumerate(ftxt) for b in ftxt[i + 1:]
                                    if a.get_window_extent(R).overlaps(b.get_window_extent(R))]
fb = fig.bbox
audit["text_outside_figure"] = [[t.get_text(), [round(v / fig.dpi, 3) for v in t.get_window_extent(R).extents]] for t in ANN + ftxt
                                if (lambda b: b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5)(t.get_window_extent(R))]
audit["n_annotations"] = len(ANN)
audit["n_frame_texts"] = len(ftxt)


def edge_margins(Mx, x0, x1, y0, y1, dpi):
    yy_, xx_ = np.nonzero(Mx)
    if not len(xx_):
        return None
    f = 72.0 / dpi
    return {"left": round(float(xx_.min() - x0) * f, 3), "right": round(float(x1 - 1 - xx_.max()) * f, 3),
            "top": round(float(yy_.min() - y0) * f, 3), "bottom": round(float(y1 - 1 - yy_.max()) * f, 3)}


def pdf_ink_margins(pdf, dpi=1200, pad_pt=12, near_pt=4.0, grow_pt=1.5):
    mb = [float(v) for v in re.search(rb"MediaBox \[ *([\d.]+) +([\d.]+) +([\d.]+) +([\d.]+)",
                                       open(pdf, "rb").read()).groups()]
    assert mb[:2] == [0.0, 0.0] and abs(mb[2] - W * 72) < 1e-6 and abs(mb[3] - H * 72) < 1e-6, mb
    png = subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pngalpha", f"-r{dpi}",
                          f"-dDEVICEWIDTHPOINTS={mb[2] + 2 * pad_pt:g}", f"-dDEVICEHEIGHTPOINTS={mb[3] + 2 * pad_pt:g}",
                          "-dFIXEDMEDIA", "-sOutputFile=-", "-c",
                          f"<</PageOffset [{pad_pt} {-pad_pt}]>> setpagedevice", "-f", pdf],
                         check=True, capture_output=True).stdout
    a = np.array(Image.open(io.BytesIO(png)).convert("RGBA")).astype(int)
    al, mn = a[..., 3], a[..., :3].min(-1)
    pg = (al > 250) & (mn >= 250)
    yy_, xx_ = np.nonzero(pg)
    x0, x1, y0, y1 = xx_.min(), xx_.max() + 1, yy_.min(), yy_.max() + 1
    s_ = dpi / 72.0
    assert abs((x1 - x0) / s_ - mb[2]) < 0.1 and abs((y1 - y0) / s_ - mb[3]) < 0.1, ((x1 - x0) / s_, (y1 - y0) / s_)
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
            near.append({"word": w, "font_box_pt": [round(v, 2) for v in (bx0, by0, bx1, by1)],
                         "nearest_side": min(mg, key=mg.get), "ink_margin_pt": mg[min(mg, key=mg.get)]})
    return {"dpi": dpi, "renderer": "ghostscript pngalpha, page padded by %d pt" % pad_pt, "mediabox_pt": mb,
            "ink_px_outside": int(outm.sum()), "margin_pt": edge_margins(inkm, x0, x1, y0, y1, dpi),
            "words_within_%g_pt_of_an_edge" % near_pt: sorted(near, key=lambda q: q["ink_margin_pt"]),
            "n_words_in_pdf": len(words)}


audit["pdf_ink_margin_pt"] = pdf_ink_margins(out_pdf)
audit["text_ink_outside_figure_pdf"] = [q for q in audit["pdf_ink_margin_pt"]["words_within_4_pt_of_an_edge"]
                                       if q["ink_margin_pt"] < 0]
assert audit["pdf_ink_margin_pt"]["ink_px_outside"] == 0, audit["pdf_ink_margin_pt"]
assert min(audit["pdf_ink_margin_pt"]["margin_pt"].values()) >= 0.2, audit["pdf_ink_margin_pt"]
assert not audit["text_ink_outside_figure_pdf"], audit["text_ink_outside_figure_pdf"]
bad = {k: v for k, v in {**{"data:" + k: v for k, v in audit["text_vs_data"].items()},
                         **{"frame:" + k: v for k, v in audit["text_vs_frame"].items()}}.items() if v}
audit["result"] = "clean" if not bad and not audit["text_vs_text"] and not audit["frame_text_box_overlaps"] \
    and not audit["text_outside_figure"] and not audit["text_ink_outside_figure_pdf"] \
    and audit["pdf_ink_margin_pt"]["ink_px_outside"] == 0 else {"collisions": bad, "text_vs_text": audit["text_vs_text"],
                                                                 "frame_text_box_overlaps": audit["frame_text_box_overlaps"]}
N["layout_audit"] = audit
N["row_check"] = ROW
N["annotations"] = [{"text": t.get_text(), "rotation_deg": round(float(t.get_rotation()), 2), "color": t.get_color()} for t in ANN]
N["on_panel_numbers"] = [t.get_text() for t in ANN if re.search(r"\d", t.get_text()) and "k=" not in t.get_text()]
assert N["on_panel_numbers"] == [], N["on_panel_numbers"]
N["axes_in"] = ROW["axes_in"]
N["style"] = {"module": "hrh_style_v8 (%s)" % STYLE_V8_PATH, "sha256": S.STYLE_SHA256, "version": S.STYLE_VERSION,
              "semantic_map_used": SEM_USED, "fw": {r: S.FW[r] for r in ("label", "tick", "direct", "annot", "key")},
              "cvd_lightness_of_pair_classes": CVD,
              "palette_pairs": {k_: S.PAIR[k_] for k_ in PC}, "pair_text": {k_: S.PAIR_TEXT[k_] for k_ in PC},
              "mst": S.CRIT["mst"], "mst_text": S.CRIT_TEXT["mst"], "null_line": S.NULL_LINE, "null_fill": S.NULL_FILL,
              "ink": S.INK, "muted": S.MUTED,
              "marker_pt": {"a_median_open_square": [MSZ["scatter"], MSZ["open_small_mew"]], "b_circle_square": KMS[1],
                            "b_triangle": KMS[100], "b_edge": B_EDGE, "b_zorder": ZB,
                            "b_k10_open_ms_mew": [B_MS_OPEN, B_MEW_OPEN], "b_korder_top_down": KORDER,
                            "b_key_row_pt": KEY_ROW_PT, "b_key_lowest_row_pt_above_axis": KEY_LOW_PT, "c_filled": [MSZ["scatter"], MSZ["scatter_edge"]],
                            "c_open": [MSZ["scatter"], MSZ["scatter_open_mew"]]},
              "text_pt": {r: S.FS[r] for r in ("letter", "label", "tick", "direct", "theory_label", "annot", "key", "math")},
              "ylim_ab": list(YLIM), "xlim_a": list(XLA), "xlim_b": list(XLB)}
N["c_direct_labels"] = C_LAB
assert audit["result"] == "clean", audit["result"]

# ================================================================== v7 print-size checks of the new render
info = subprocess.run(["pdfinfo", out_pdf], capture_output=True, text=True, check=True).stdout
psz = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
page_in = [float(psz.group(1)) / 72, float(psz.group(2)) / 72]
assert abs(page_in[0] - 5.5) < 1e-3 and abs(page_in[1] - H) < 1e-3, page_in
N["v7_checks"] = {"page_size_in": [round(v, 4) for v in page_in], "height_in": {"v6": 1.95, "v7": H},
                  "min_glyph_pt": chk["min_glyph_pt"], "n_type3": chk["n_type3"], "fonts": chk["fonts"],
                  "png": "pdftoppm -r 200 of the PDF", "png_px": N["png_px"], "view_print_100dpi_px": list(Image.open(VIEW).size),
                  "n_annotations_v6": len(O5["annotations"]), "n_annotations_v7": len(ANN),
                  "annotations_dropped": sorted((Counter(a["text"] for a in O5["annotations"])
                                                 - Counter(t.get_text() for t in ANN)).elements()),
                  "annotations_added": sorted((Counter(t.get_text() for t in ANN)
                                               - Counter(a["text"] for a in O5["annotations"])).elements()),
                  "annotation_changed": {"(a) chance, slope −1.00": "(a) chance"}}
assert N["v7_checks"]["annotations_dropped"] == sorted(["−0.01", "−0.21", "−0.57", "slope", "chance, slope −1.00",
                                                        "$k=10$"]), N["v7_checks"]["annotations_dropped"]
assert N["v7_checks"]["annotations_added"] == ["chance"], N["v7_checks"]["annotations_added"]   # (a) 'chance'
assert sorted(t.get_text() for t in ANN if t.axes is axA) == ["chance", "seed", "vision", "vision–language"]

# ================================================================== v7 caption (number-free, short)
CAPTION_V7 = (
    r"\caption{\textbf{Neighbor agreement changes with the evaluation size at a rate set by the model pair, and "
    r"spanning-tree overlap tracks one-nearest-neighbor overlap (Theorem~\ref{thm:crossing}, "
    r"Proposition~\ref{prop:mst}).} (a) mKNN against $m$ for single pairs (thin) and class medians (thick). "
    r"(b) Class medians at three values of $k$ against $k/m$, on the scale of (a). Coincident curves show agreement "
    r"set by $k/m$ alone. (c) Vision--language pairs of WIT at two depths. "
    r"Appendices~\ref{app:exp-s11} and~\ref{app:exp-s12} give the details.}")
_b7 = re.sub(r"\\ref\{[^}]*\}|\\label\{[^}]*\}", "", CAPTION_V7)
_n7 = re.findall(r"(?<![A-Za-z_\\])\d+(?:\.\d+)?", _b7)
assert _n7 == [], _n7                                   # no number at all, not even a design constant
# words as printed: each \ref is one token (a number in the PDF), each inline formula one token
_words = re.sub(r"\\ref\{[^}]*\}", " 1 ", CAPTION_V7)
_words = re.sub(r"\$[^$]*\$", " x ", _words)
_words = re.sub(r"\\caption\{|\\textbf\{|[{}~]", " ", _words)
NWORDS = len([w for w in re.split(r"\s+", _words) if re.search(r"[A-Za-z0-9]", w)])
_bold = re.search(r"\\textbf\{(.*?)\}\s*\(a\)", CAPTION_V7, re.S).group(1)
NBOLD = len([w for w in re.split(r"\s+", re.sub(r"\([^)]*\)", " ", _bold)) if re.search(r"[A-Za-z]", w)])
NPANEL = len(re.findall(r"(?:^|[.}] )\(([a-c])\) ", CAPTION_V7.replace(r"\caption{", "")))   # leading (a), (b), (c)
assert NPANEL == 3, NPANEL
assert NWORDS - NPANEL <= 80 and NBOLD <= 25, (NWORDS, NPANEL, NBOLD)
assert "---" not in CAPTION_V7 and "—" not in CAPTION_V7 and ": " not in CAPTION_V7, "em-dash or colon in the caption"
assert not re.search(r"\bit[.,;]", CAPTION_V7), "a clause ends on 'it'"
for w_ in ("must", "never", "always", "nothing", "none", "cannot", "only"):
    assert not re.search(r"\b%s\b" % w_, CAPTION_V7), w_
# v8: the v7 caption is kept as a record only (its checks above still run); v8 writes caption_v8.tex below
assert hashlib.sha256((CAPTION_V7 + "\n\\label{fig:exp-neighbors}\n").encode()).hexdigest() == FROZEN_V7[f"{V7}/caption_v7.tex"]
N["caption_v7"] = CAPTION_V7
N["caption_v7_sha256"] = hashlib.sha256(CAPTION_V7.encode()).hexdigest()
N["caption_v7_words"] = {"total_incl_bold_and_refs": NWORDS, "total_without_the_3_panel_labels": NWORDS - NPANEL,
                         "bold_claim_without_refs": NBOLD,
                         "how": "each \\ref and each inline formula is one word; panel labels (a) (b) (c) at sentence "
                                "starts counted in the first total only"}
N["caption_v6"] = CAPTION_V6
N["caption_v6_sha256"] = hashlib.sha256(CAPTION_V6.encode()).hexdigest()

# ================================================================== v8 caption (number-free, short; names the pools of a)
CAPTION_V8 = (
    r"\caption{\textbf{Neighbor agreement changes with the evaluation size at a rate set by the model pair, and "
    r"spanning-tree overlap tracks one-nearest-neighbor overlap (Theorem~\ref{thm:crossing}, "
    r"Proposition~\ref{prop:mst}).} (a) mKNN on nested COCO pools against $m$, for single pairs (thin) and class "
    r"medians (thick). (b) Class medians at three values of $k$ against $k/m$, on the scale of (a). Coincident curves "
    r"show agreement set by $k/m$ alone. (c) Vision--language pairs of WIT at two depths. "
    r"Appendices~\ref{app:exp-s11} and~\ref{app:exp-s12} give the details.}")


def caption_words(cap):
    """(words as printed, panel labels at sentence starts, bold-claim words without refs): each \\ref is one token,
    each inline formula one token (the v7 rule)."""
    w = re.sub(r"\\ref\{[^}]*\}", " 1 ", cap)
    w = re.sub(r"\$[^$]*\$", " x ", w)
    w = re.sub(r"\\caption\{|\\textbf\{|[{}~]", " ", w)
    nw = len([x for x in re.split(r"\s+", w) if re.search(r"[A-Za-z0-9]", x)])
    b = re.search(r"\\textbf\{(.*?)\}\s*\(a\)", cap, re.S).group(1)
    nb = len([x for x in re.split(r"\s+", re.sub(r"\([^)]*\)", " ", b)) if re.search(r"[A-Za-z]", x)])
    npan = len(re.findall(r"(?:^|[.}] )\(([a-c])\) ", cap.replace(r"\caption{", "")))
    return nw, npan, nb


NW8, NP8, NB8 = caption_words(CAPTION_V8)
assert (NWORDS, NPANEL, NBOLD) == caption_words(CAPTION_V7)            # the helper reproduces the v7 count
assert NP8 == 3 and 60 <= NW8 - NP8 <= 80 and NB8 <= 25, (NW8, NP8, NB8)
_b8 = re.sub(r"\\ref\{[^}]*\}|\\label\{[^}]*\}", "", CAPTION_V8)
assert re.findall(r"(?<![A-Za-z_\\])\d+(?:\.\d+)?", _b8) == [], "a number in the v8 caption"
assert "---" not in CAPTION_V8 and "—" not in CAPTION_V8 and ": " not in CAPTION_V8, "em-dash or colon in the caption"
assert not re.search(r"\bit[.,;]", CAPTION_V8), "a clause ends on 'it'"
for w_ in ("must", "never", "always", "nothing", "none", "cannot", "only"):
    assert not re.search(r"\b%s\b" % w_, CAPTION_V8), w_
assert len(re.findall(r"Appendi(?:x|ces)~\\ref", CAPTION_V8)) == 1, "one appendix pointer"
for setting in ("COCO", "WIT"):                                         # the caption names the setting of each panel
    assert setting in CAPTION_V8, setting
assert CAPTION_V8.replace("(a) mKNN on nested COCO pools against $m$, for", "(a) mKNN against $m$ for") == CAPTION_V7, \
    "v8 caption = v7 caption + the pools of (a)"
with open(os.path.join(OUT, "caption_v8.tex"), "w") as f:
    f.write(CAPTION_V8 + "\n\\label{fig:exp-neighbors}\n")
N["caption_v8"] = CAPTION_V8
N["caption_v8_sha256"] = hashlib.sha256(CAPTION_V8.encode()).hexdigest()
N["caption_v8_words"] = {"total_incl_bold_and_refs": NW8, "total_without_the_3_panel_labels": NW8 - NP8,
                         "bold_claim_without_refs": NB8, "how": N["caption_v7_words"]["how"]}

# ================================================================== v8 checks of the new render against v7
N["v8_checks"] = {"annotation_texts_equal_v7": sorted(t.get_text() for t in ANN) == sorted(a["text"] for a in O7["annotations"]),
                  "n_annotations": len(ANN), "height_in": {"v7": 1.70, "v8": H},
                  "finalize_changed_texts": [N_FINALIZED, N_FINALIZED_2],
                  "n_tick_labels_bold": chk["n_tick_labels_checked"], "n_axis_titles_bold": chk["n_axis_titles_checked"],
                  "fonts": chk["fonts"], "off_palette_colours": chk["off_palette_colours"],
                  "views": {"print_100dpi_pil": VIEW, "print_100dpi_pdftoppm": VIEW_PP, "cvd_grey": VIEWS},
                  "colour_changes_vs_v7": {
                      "seed": [O7["style"]["palette_pairs"]["seed"], S.PAIR["seed"]],
                      "vision": [O7["style"]["palette_pairs"]["vision"], S.PAIR["vision"]],
                      "vision-language": [O7["style"]["palette_pairs"]["vl"], S.PAIR["vl"]],
                      "MST": [O7["style"]["mst"], S.CRIT["mst"]], "MST text": [O7["style"]["mst_text"], S.CRIT_TEXT["mst"]]}}
assert N["v8_checks"]["annotation_texts_equal_v7"], "on-figure words differ from v7"
assert [a["text"] for a in N["annotations"]] == [a["text"] for a in O7["annotations"]]

# ================================================================== records
N["results_text_numbers"] = RES
N["results_numbers_in_arxiv_v4_now"] = RES_IN_V4_V8
N["results_numbers_in_arxiv_v4_at_v7"] = RES_IN_V4_NOW
N["slope_ordering_recomputed"] = {kk: {CKEY[c]: v for c, v in d_.items()} for kk, d_ in order_re.items()}
N["elasticity_recomputed"] = {CKEY[c]: v for c, v in el_re.items()}
N["style_equivalence"] = STYLE_EQ
N["v5_1_render_checks"] = V5_CHECKS
N["v6_option_render_checks"] = OPT_CHECKS
N["frozen_v5_1_render"] = dict(FROZEN)
N["inputs_vs_v5_render"] = hash_vs_v5
N["copies_next_to_maker"] = {p: h for p, (h, _) in COPIES.items()}
# height trials (FN_H=<h> FN_TRIAL=h<h>, outputs and logs in _trials/ and _trials_h<h>.log; same maker and defaults)
HEIGHT_TRIALS = {"1.80": "clean (axes 1.21 in)", "1.75": "clean (axes 1.16 in)", "1.70": "clean (axes 1.11 in); adopted",
                 "1.65": "fails: '(a) vision' touches the vision-language bundle (3 px at 300 dpi)",
                 "1.60": "fails: '(a) vision' touches data (8 px) and '(b) chance' touches the x axis (4 px)"}
N["height_trials"] = HEIGHT_TRIALS
N["changes_v7"] = [
    {"what": "row filled", "how": "slope column of (a) (0.31 in) and the 0.25 in gap after it removed; data widths "
                                  "(a) 1.23 -> %.2f in, (b) 1.26 -> %.2f in, (c) 1.30 -> %.2f in (square); gaps a-b %.2f in, "
                                  "b-c %.2f in between ink boxes; right margin %.2f in"
                                  % (WA, WB, WC, ROW["gap_a_b_in"], ROW["gap_b_c_in"], ROW["right_margin_in"]),
     "why": "author 2026-10-10: panels fill the whole row"},
    {"what": "slope numbers -0.01 / -0.21 / -0.57 and the 'slope' header dropped from (a)",
     "why": "result numbers on the figure; the Results paragraph of Section 5.2 states them (l.1220-1222)"},
    {"what": "'chance, slope -1.00' -> 'chance'", "why": "the number is in the Results text (l.1222)"},
    {"what": "'k = 10' corner label of (a) dropped; (a) medians drawn as the open k = 10 squares of (b)",
     "why": "a setting label; the marker now carries k (the key of (b) decodes it), and the filled circle that meant "
            "k = 10 in (a) but k = 1 in (b) is gone (v6 critic finding, v6 option render)"},
    {"what": "x axis of (a) runs %g to %.0f (same log margin on both sides) with a full spine" % XLA,
     "why": "no slope column to set apart"},
    {"what": "(b) key rows a fixed %.0f pt apart" % KEY_ROW_PT, "why": "the axes are lower than in v5"},
    {"what": "height 1.95 -> %.2f in, top margin 0.28 -> %.2f in" % (H, TOPM),
     "why": "the top margin held the 'slope' header; it now holds only the panel letters; %.2f in is the lowest "
            "height in steps of 0.05 in whose layout audit is clean (HEIGHT_TRIALS)" % H},
    {"what": "caption %d words (v6: about 150)" % NWORDS, "why": "author 2026-10-10: captions short"}]
N["height_trials_v8"] = json.loads(os.environ.get("FN_HEIGHT_TRIALS_V8", "{}"))
N["changes_v8"] = [
    {"what": "style module hrh_style_v5_1 -> hrh_style_v8 (sha256 %s...)" % S.STYLE_SHA256[:12],
     "why": "author 2026-10-10 on v7 Figure 6: an attractive, widely recognized scientific palette (Paul Tol), one "
            "colour per meaning across the paper"},
    {"what": "colours: seed %s -> %s, vision %s -> %s, vision-language %s -> %s, MST %s -> %s (text %s -> %s)"
             % (O7["style"]["palette_pairs"]["seed"], S.PAIR["seed"], O7["style"]["palette_pairs"]["vision"],
                S.PAIR["vision"], O7["style"]["palette_pairs"]["vl"], S.PAIR["vl"], O7["style"]["mst"], S.CRIT["mst"],
                O7["style"]["mst_text"], S.CRIT_TEXT["mst"]),
     "why": "the semantic map of the v8 style (pair classes = light / mid / dark steps of the mKNN hue; MST = Tol "
            "vibrant teal); theory INK, chance and y = x NULL_LINE, excluded region NULL_FILL unchanged"},
    {"what": "tick labels and axis titles bold, math included ($s_N$, $k/m$, $s_{N,1}$, $s_{\\mathcal{T}}$)",
     "why": "author 2026-10-10: tick labels and axis titles bold in every figure; S.finalize(fig) before the layout "
            "is measured and before savefig. The calligraphic T of s_T keeps the regular Computer Modern glyph "
            "(the v8 module has no bold calligraphic face)"},
    {"what": "geometry: XA %s, G1 %s, G2 %s, RM %s in; height %.2f in" % (XA, G1, G2, RM, H),
     "why": "bold tick labels are wider; the v7 row and ink audits are rerun unchanged (row gaps >= 0.04 in, margins, "
            "clean collision audit)"},
    {"what": "caption: '(a) mKNN against $m$' -> '(a) mKNN on nested COCO pools against $m$,'",
     "why": "author 2026-10-10: the caption or Results names the setting when it calls a panel; (c) already names WIT"},
    {"what": "nothing dropped from the figure", "why": "no setting title, no secondary comparison, no number on the "
                                                       "figure in v7; every on-figure word kept (asserted)"}]
N["asserts_adapted_v8"] = {
    "style_sha": "v7 asserted the imported style = hrh_style_v5_1 (97d0ed53...); v8 asserts the imported style = "
                 "hrh_style_v8 (fe994067...) and still asserts the v5.1 copy next to this file (used by the old checks)",
    "old_checks_module": "the v5.1-render checks, the v6-option checks and the v5 = v5.1 equivalence use S51 (the "
                         "v5.1 copy) instead of S, so they test the same files with the same module as in v7",
    "caption_file": "v7 wrote caption_v7.tex; v8 keeps every v7 caption assert, asserts the v7 caption text against "
                    "the frozen v7 caption_v7.tex, and writes caption_v8.tex",
    "new": "imported v8 style sha; v7 maker, render and caption frozen; plotted blocks and Results records equal the "
           "v7 numbers.json; on-figure words equal v7; S.check v8 rules (bold ticks and axis titles, bold math, "
           "chromatic data, no off-palette data colour); pair classes >= 15 apart in L* in normal, deutan, protan and "
           "grey views; v8 caption rules"}
N["outputs"] = {out_pdf: sha(out_pdf), out_png: sha(out_png), VIEW: sha(VIEW), VIEW_PP: sha(VIEW_PP), VIEWS: sha(VIEWS),
                os.path.join(OUT, "caption_v8.tex"): sha(os.path.join(OUT, "caption_v8.tex"))}
json.dump(N, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=float)
print(json.dumps({"H": H, "axes_in": ROW["axes_in"], "row": {k: v for k, v in ROW.items() if k != "axes_in"},
                  "pdf_checks": {k: chk[k] for k in ("min_glyph_pt", "pdf_glyph_size_counts", "n_type3", "fonts")},
                  "png_px": N["png_px"], "layout": audit["result"], "clearance_pt": audit["clearance_pt"],
                  "letter_clearance_data_pt": audit["letter_clearance_data_pt"],
                  "pdf_margin_pt": audit["pdf_ink_margin_pt"]["margin_pt"],
                  "caption_words": N["caption_v8_words"], "v8_checks": {k: N["v8_checks"][k] for k in
                                                                        ("n_tick_labels_bold", "n_axis_titles_bold",
                                                                         "off_palette_colours", "fonts")},
                  "outputs": N["outputs"]}, indent=1, default=float))
