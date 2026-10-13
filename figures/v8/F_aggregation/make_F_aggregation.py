"""F_aggregation v8 (main Figure 4, fig:exp-aggregation). Role maker-v8, 2026-10-10. Starts from a byte copy of the v7
maker (~/hrh_v2_figs/v7/F_aggregation/make_F_aggregation.py, sha256 b4bcb0de..., untouched; the copy is kept next to
this file as make_F_aggregation_v7_copy.py) and imports the frozen v8 style ~/hrh_v2_figs/v8/_style/hrh_style_v8.py by
path (sha256 fe994067..., asserted before and after the import).

Author feedback 2026-10-10 on v7 Figure 6, applied to every figure: (1) no header that states a setting (the caption or
the Results names the setting when it calls a panel); (2) no black-and-white data: Tol palette of the v8 module; (3) tick
labels and axis titles bold, math included; (4) secondary comparisons that the Results text can state are not drawn.
v8 changes (data, layout and every plotted value of v7 unchanged, asserted equal to the v5.1 numbers.json files to 1e-12):
  * style v5.1 -> v8: CKA is Tol vibrant red #CC3311 (was vermilion #D55E00), mKNN Tol vibrant blue #0077BB; tick
    labels, axis titles and the row-1 score names (role 'tick') bold, bold math in bold texts; S.finalize(fig) before
    savefig; S.check() asserts bold ticks and titles, bold math and a chromatic data colour.
  * the family names over (a)-(c) are gone (SHOW_FAMILY_HEADERS = False): a column title names the setting of the
    column, so the caption names the families ("Columns show BLOOMZ, LLaMA-1, and OpenLLaMA") and the Results paragraphs
    call each column by its family. The letters (a)-(c) keep the header line.
  * row 2 lower segment and row 3 had only INK and grey marks. v8 semantics: c_alpha squares and the peak squares of
    row 3 in the CKA hue (c_alpha is the CKA threshold that s_cal subtracts; the square keeps one meaning, the largest
    baseline that c_alpha tracks); the largest baseline predicted from features in INK dashed (theory: the prediction
    of Proposition baselines); the threshold of the selected pair in NULL_LINE dotted (a reference) with a MUTED label.
  * PNG = pdftoppm render of the PDF at 200 dpi (S.print_view), plus a 100 dpi print view and the normal / deutan /
    protan / grey views (S.views).
  * caption_v8.tex (checked here as caption_v7.tex was): bold claim <= 26 words, no result numbers.
Critic-v8 pass (2026-10-10; the maker-v8 outputs are kept in _prev_maker_v8/): author rule 4 (secondary comparisons that
the Results text can state are not drawn) applied to the 'CKA, calibrate then max' row of (a)-(c) and to its mechanism,
the dotted threshold of the selected layer pair in (d)-(f) with its 'selected pair' label. Calibration before the
maximum is a second protocol; v4 states its outcome in every family (l.1539-1542, l.4943-4951, l.5160-5161, Table
tab:exp-verdicts) and the selected-pair thresholds in the threshold paragraphs (l.4820-4829, l.4892-4894,
l.5140-5141, l.5476-5478), and F.3 keeps both. SHOW_CTM_ROW = False (SHOW_SELECTED_PAIR follows it) restores neither;
set it True to draw them again. Row 1 keeps its per-row spacing, so the figure is one row shorter. The loaded data,
the Results numbers (calibrate-then-max included) and every value still drawn are asserted equal to v5.1 and v7.
Family-names pass (2026-10-10, author: "家族名加回来"; the files before this pass are kept in _prev_family_names/): the
family names are identifiers that tell the reader which column is which model family, not setting headers, so
SHOW_FAMILY_HEADERS = True draws "BLOOMZ", "LLaMA-1", "OpenLLaMA" again after the letters (a)-(c), bold, in the family
text tones S.FAM_TEXT (the style's role 'header'), on the existing header line (figure size unchanged). The caption
sentence "Columns show BLOOMZ, LLaMA-1, and OpenLLaMA." is removed from caption_v8.tex as redundant. Data, layout and
every other mark unchanged; every assert unchanged.

v7 header follows.

F_aggregation v7 (main Figure 4, fig:exp-aggregation). Role maker-v7, 2026-10-10. Starts from a copy of the v6.1
maker (~/hrh_v2_figs/v6/F_aggregation/make_F_aggregation.py, sha256 c6eaf184..., untouched) and of the frozen style
~/hrh_v2_figs/v5/_style/hrh_style_v5_1.py (copied next to this file, same sha256 97d0ed53...).

Author feedback 2026-10-10 on the v6 figure: the first row gave a wide left gutter to row labels and a key; the panels
must fill the row, and the figure should carry little text. v7 changes (data, encodings and the three-row design of v6
unchanged; every plotted value asserted equal to the v5.1 numbers.json files to 1e-12 as before):
  * gutter 1.05 in -> GUTTER (only the y tick labels and y labels of rows 2-3); columns 1.42 in -> AX_W.
  * row-1 score names are one-line texts left-aligned at the figure edge; in (a) they run into the empty left part of
    the axes (the row guides start after each name), so they need no gutter of their own.
  * row-1 key: one line across the top of the figure (it was a three-line block in the gutter).
  * removed from the figure: the status lines under the family names ('5 sizes, replication', '3 sizes, predictions
    frozen', '3 sizes, descriptive'), the 'expl.' tag in (a), both 'post hoc' tags (d, g) and the six endpoint numbers
    of row 1 (the Results text states every one of them). 'threshold of the selected pair' (e) -> 'selected pair';
    row-3 y label 'random-pairing baseline' -> 'baseline' (the caption names the random pairing).
  * row-1 headroom for the endpoint numbers dropped; trend x limits +-1.15 -> +-1.10 (they were widened only to keep
    two endpoint numbers apart).
The caption (caption_v7.tex) carries no result numbers; the Results text cites the panels for every number
Critic-v7 pass (2026-10-10): caption tightened (c_alpha named as the threshold; the shading clause cut, since the (h)
label names the shading); CAPTION_MAX_WORDS 100 -> 90. Figure code and data unchanged.
(numbers.json key results_text_numbers, unchanged from v6).

Rows, top to bottom from outcome to mechanism (as v6):
  Row 1 (a-c): size trend Tr over 17 vision models (raw ring, after subtraction dot, s_cal tick, 95% tints), 4 rows.
  Row 2 (d-f): levels by size on a two-segment y axis with the same scale in both segments: observed layer-max CKA
               (upper); threshold c_alpha of the maximum, largest baseline predicted from features and threshold of
               the selected layer pair (lower, dotted, no markers).
  Row 3 (g-i): CKA random-pairing baseline per language-model layer (max over vision-model layers, mean over the 17
               vision models), one line per size (Proposition baselines from features) with permutation means (dots);
               the peak of each profile carries the INK square used for c_alpha in row 2; depths of the observed max
               shaded; mKNN baseline k/(m-1) in blue.

Data: identical to v6.1 / v5.1, through ~/hrh_v2_figs/F1/f1_data.py, ~/hrh_v2_figs/F1/f1_llama_data.py and
~/hrh_v2_figs/v5/F_aggregation_app/ol_data.py, imported read-only by path (no bytecode written).

Run on ds2213 (CPU only):
  cd ~/hrh_v2_figs/v8/F_aggregation && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      nice -n 10 taskset -c 12-17 ~/hrh_v2_figs_venv/bin/python -B make_F_aggregation.py
Writes F_aggregation.{pdf,png}, _view_print_100dpi.png, _views_cvd_grey.png and numbers.json next to this file.
"""
import hashlib, importlib.util, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
HOME = os.path.expanduser("~")
FIGS = f"{HOME}/hrh_v2_figs"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.lines import Line2D


def load_module(name, path):
    """Import a frozen module by path (read-only; sys.dont_write_bytecode is set, so no __pycache__ is written)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


STYLE_PATH = f"{FIGS}/v8/_style/hrh_style_v8.py"              # v8: frozen style, imported by path (read only)
STYLE_V8_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_PATH, "rb").read()).hexdigest() == STYLE_V8_SHA, "hrh_style_v8 changed"
S = load_module("hrh_style_v8", STYLE_PATH)
assert S.STYLE_SHA256 == STYLE_V8_SHA and S.STYLE_VERSION == "v8", S.STYLE_SHA256
D = load_module("f1_data", f"{FIGS}/F1/f1_data.py")
LD = load_module("f1_llama_data", f"{FIGS}/F1/f1_llama_data.py")
OD = load_module("ol_data", f"{FIGS}/v5/F_aggregation_app/ol_data.py")
V51_MAIN = f"{FIGS}/v5/F_aggregation/numbers.json"          # v5.1 Figure 3 (row 1 reference)
V51_APP = f"{FIGS}/v5/F_aggregation_app/numbers.json"       # v5.1 F.3 (rows 2-3 reference)

S.use()
INK, MUTED = S.INK, S.MUTED
LW, MS, Z = S.LW, S.MS, S.Z
C_CKA, C_MKNN = S.CRIT["cka"], S.CRIT["mknn"]
# v8 semantic map of this figure (one colour, one meaning; asserted in main against the module's hexes)
C_THRESH = C_CKA                 # c_alpha squares (row 2) and peak squares (row 3): the CKA threshold / largest baseline
C_PRED = S.THEORY                # largest baseline predicted from features (Proposition baselines): INK, dashed
LS_PRED, LW_PRED = S.LS["dashed"], LW["theory"]
C_SEL = S.NULL_LINE              # threshold of the selected layer pair: a reference, dotted
assert C_CKA == "#CC3311" and C_MKNN == "#0077BB" and C_PRED == "#222222" and C_SEL == "#707070"
SHOW_FAMILY_HEADERS = True       # family-names pass: the names identify the columns (author 2026-10-10); v8 had False
PRE = "prh_clip+l2"
M = 1024
MKNN_BASE = 10 / (M - 1)
FAMS = ["bloomz", "llama1", "openllama"]
LMS = {"bloomz": D.BLOOMZ, "llama1": LD.LLAMA, "openllama": OD.OL}
SIZE_LABEL = {"bigscience__bloomz-560m": "0.56B", "bigscience__bloomz-1b1": "1.1B", "bigscience__bloomz-1b7": "1.7B",
              "bigscience__bloomz-3b": "3B", "bigscience__bloomz-7b1": "7.1B"}
SIZE_LABEL.update(LD.LL_LABEL)
SIZE_LABEL.update(OD.OL_LABEL)
XLAB = {"bloomz": ["0.56", "1.1", "1.7", "3", "7.1"], "llama1": ["7", "13", "30"], "openllama": ["3", "7", "13"]}
STATUS = {"bloomz": "5 sizes, replication", "llama1": "3 sizes, predictions frozen",     # v7: numbers.json only,
          "openllama": "3 sizes, descriptive"}                                            # not drawn
ROWS_ALL = [("cka_biased|full", C_CKA, "CKA, layer max"),                 # loaded, checked, Results numbers
            ("cka_biased|ctm", C_CKA, "CKA, calibrate then max"),
            ("cka_biased|fixed_r0.5", C_CKA, "CKA, mid depth"),
            ("mknn|full", C_MKNN, "mKNN, layer max")]
SHOW_CTM_ROW = False             # critic-v8: author rule 4; the calibrate-then-max row is a second protocol (F.3 keeps it)
SHOW_SELECTED_PAIR = SHOW_CTM_ROW  # its mechanism, the dotted threshold of the selected pair in (d)-(f), goes with it
ROWS = [r for r in ROWS_ALL if SHOW_CTM_ROW or r[0] != "cka_biased|ctm"]   # drawn rows of (a)-(c)

# ------------------------------------------------------------------ geometry (inches; exact print size)
V6_FIG_H = 4.38                                                # v6.1 height, for the report
FIG_W = S.TEXTWIDTH_IN
GUTTER, RIGHT, GAP = 0.48, 5.47, 0.07                          # v6: 1.05 in gutter, right edge 5.43, 0.06 in gaps
                                                               # v8: gutter 0.42 -> 0.48: the bold score names need
                                                               # room before the zero line of (a), and the bold y tick
                                                               # labels of rows 2-3 room before the y labels
AX_W = (RIGHT - GUTTER - 2 * GAP) / 3                          # 1.617 in (v7: 1.637, v6: 1.42)
COL_X = [GUTTER + i * (AX_W + GAP) for i in range(3)]          # 0.42, 2.127, 3.833 -> right edge 5.47
# bottom-up: row 3 x label, row 3, row 2 x label, row 2, row 1 x label, row 1, headers, key line
YB_XLAB3 = 0.055
R3_Y, R3_H = 0.320, 0.80                                       # per-layer baselines (v6: 0.90)
YB_XLAB2 = R3_Y + R3_H + 0.235                                 # letters (g-i) and size keys sit in this strip
R2_Y, R2_H = YB_XLAB2 + 0.268, 0.92                            # levels: two segments (v6: 0.98)
SEG_GAP = 0.06
SEG_H = (R2_H - SEG_GAP) / 2                                   # 0.43 in each (v6: 0.46)
YB_XLAB1 = R2_Y + R2_H + 0.232
R1_H_4ROWS = 0.74                                              # forests (v6: 0.86 with headroom for numbers)
R1_H = round(R1_H_4ROWS * (len(ROWS) + 0.05) / (4 + 0.05), 3)  # critic-v8: same pt per row as the 4-row v7/v8 forest
R1_Y = YB_XLAB1 + 0.268
YB_FAM = R1_Y + R1_H + 0.075                                   # "(a) BLOOMZ": one header line (v6: two)
YB_KEY = YB_FAM + 0.165                                        # row-1 key: one line across the top
FIG_H = round(YB_KEY + 0.085, 3)
FAM_DX = 0.215                                                 # family name starts after "(a) "
LETTER_DY = 0.075                                              # letters of rows 2-3: baseline above the axes top
ROW_LBL_X = 0.02                                               # left edge of the row-1 score names (figure inches)
ROW_LBL_PAD = 3.0                                              # pt between a score name and the start of its guide
KEY_X0, KEY_SEP, KEY_TXT = 0.02, 0.17, 0.045                   # key line: start, entry gap, marker-to-text gap (in)
KEY3_DX = 0.30                                                 # row-3 size keys start this far right of the column
XLIM_TR = (-1.10, 1.10)                                        # trend axes (v6: +-1.15, widened only for numbers)
SHOW_ENDPOINT_NUMBERS = False                                  # v7: the six row-1 numbers are gone (Results text)
YLIM_TR = (len(ROWS) - 0.5, -0.55)                             # v6: (3.5, -0.92), headroom for the numbers
PT_PER_TR = AX_W / (XLIM_TR[1] - XLIM_TR[0]) * 72
SEG_UP, SEG_LO = (0.32, 0.46), (0.095, 0.235)                  # equal spans 0.14. Lower segment 0.005 below the
                                                               # plan's [0.10, 0.24]: there the BLOOMZ 1.7B selected-
                                                               # pair circle (0.1069) crossed the x axis by 0.3 pt
YC_MAX = 0.30
LEADER_GAP = 1.2                                               # pt between a leader's end and a marker edge
LABEL_GAP_MIN_PT = 3.0                                         # least gap between endpoint numbers of two panels
CAPTION_MAX_WORDS = 90                                        # critic-v7: 100 -> 90 regex tokens (panel tags, c+alpha and
                                                               # ref names count; about 80 words of prose; task 60-80)
NAME_TO_MARK_MIN_PT = 3.0                                      # v7: least gap between a score name and the marks
BOLD_MAX_WORDS = 26                                            # v8: bold claim of the caption (v7 allowed 30)
NAME_TO_ZERO_MIN_PT = 8.0                                      # v8: every score name ends left of the zero line of (a)
TOL = 1e-12


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tilde(p):
    return p.replace(HOME, "~")


def arr(a):
    return [float(v) for v in np.asarray(a).ravel()]


# ------------------------------------------------------------------ size ramps darkened to >= 3:1 (critic, palette)
RAMP_MIN_CONTRAST = 3.0
_WP = np.array([0.95047, 1.0, 1.08883])                       # D65 white
_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])


def _lab(hex_color):
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(hex_color))
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (_M @ c) / _WP
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _hex(lab):
    from matplotlib.colors import to_hex
    fy = (lab[0] + 16) / 116
    f = np.array([fy + lab[1] / 500, fy, fy - lab[2] / 200])
    t = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WP
    c = np.linalg.solve(_M, t)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return to_hex(np.clip(c, 0, 1))


def ramp_3to1(fam):
    """The v5.1 ramp S.FAM_RAMP[fam] with its light end moved along the ramp's own CIELAB path until it reaches
    RAMP_MIN_CONTRAST on white, and the inner steps re-spaced to even L* between the new light end and the unchanged
    dark end (same hue path, same number of steps)."""
    base = S.ramp(fam)
    P = np.array([_lab(h) for h in base])                       # L* decreasing along the ramp
    def at(L):                                                  # point on the polyline with lightness L
        for k in range(len(P) - 1):
            if P[k + 1][0] <= L <= P[k][0]:
                w = (P[k][0] - L) / (P[k][0] - P[k + 1][0])
                return P[k] + w * (P[k + 1] - P[k])
        raise ValueError(L)
    hi, lo = P[0][0], P[-1][0]
    if S.contrast(base[0]) >= RAMP_MIN_CONTRAST:
        L0 = hi
    else:
        a, b = lo, hi                                           # bisection on L*: contrast falls as L* rises
        for _ in range(60):
            m = (a + b) / 2
            a, b = (m, b) if S.contrast(_hex(at(m))) >= RAMP_MIN_CONTRAST + 0.02 else (a, m)
        L0 = a
    Ls = np.linspace(L0, lo, len(base))
    out = [_hex(at(L)) for L in Ls[:-1]] + [base[-1]]
    assert all(S.contrast(c) >= RAMP_MIN_CONTRAST for c in out), (fam, out)
    assert all(S.contrast(out[k]) < S.contrast(out[k + 1]) for k in range(len(out) - 1)), (fam, out)
    return out


# ------------------------------------------------------------------ data (as v5.1 F.3 load(), restricted to 4 rows)
def load():
    nums = D.compute()
    vits = sorted(nums["a_trend"]["cka_biased|full"]["s_obs"]["per_vit_rho"].keys())
    L = LD.compute(vits)
    O = OD.compute(vits)
    out = {"vits": vits, "inputs": {}, "trend": {}, "thr": {}, "prof": {}}
    for k, v in nums["inputs"].items():
        out["inputs"]["f1_data:" + k] = v
    for k, v in L["inputs"].items():
        out["inputs"]["f1_llama_data:" + k] = v
    for k, v in O["inputs"].items():
        out["inputs"]["ol_data:" + k] = v
    i_sum = nums["inputs"]["layermax_summary"]
    i_E, i_I = L["inputs"]["E_summary"], L["inputs"]["I_summary"]
    i_O = O["inputs"]["layermax_openllama_summary"]

    def entry(t, path, sha256, key, ci_note=None):
        e = {"Tr": float(t["Tr"]), "ci95": [float(c) for c in t["vit_boot_ci95"]],
             "p_one_sided_exact": float(t["p_one_sided_exact"]), "path": path, "sha256": sha256, "key": key}
        if ci_note:
            e["ci_note"] = ci_note
        if "per_vit_rho" in t:                            # v6.1: per-vision-model Spearman, for the Results text
            e["per_vit_rho"] = {k: float(v) for k, v in t["per_vit_rho"].items()}
        return e

    # trends: BLOOMZ (frozen layermax_summary; calibrate-then-max recomputed in f1_llama_data, as v5.1)
    A = nums["a_trend"]
    TB = {}
    for key in ["cka_biased|full", "cka_biased|fixed_r0.5", "mknn|full"]:
        met, sub = key.split("|")
        TB[key] = {nm: entry(A[key][c], i_sum["path"], i_sum["sha256"], f"per_pre/{PRE}/trends/{met}/{sub}|{c}/tr_lm")
                   for nm, c in [("raw", "s_obs"), ("after", "uncensored"), ("s_cal", "s_cal")]}
    C = L["a_trend_ctm_bloomz"]
    assert abs(C["s_obs"]["Tr"] - A["cka_biased|full"]["s_obs"]["Tr"]) < TOL
    note = "no frozen CI; ViT bootstrap recomputed in f1_llama_data (B = 4000, seed 20261009)"
    TB["cka_biased|ctm"] = {
        "raw": TB["cka_biased|full"]["raw"],
        "after": entry(C["uncensored"], i_I["path"], i_I["sha256"],
                       f"per_pre/{PRE}/blocks/cka_b/ctm/bloomz_axis/uncensored (Tr, p_one); recomputed from "
                       f"S1_2_layermax/pairs/{PRE}/<lm>__<vit>.npz entry arrays", note),
        "s_cal": entry(C["s_cal"], i_I["path"], i_I["sha256"],
                       f"per_pre/{PRE}/blocks/cka_b/ctm/bloomz_axis/s_cal (Tr, p_one); recomputed from entry arrays",
                       note)}
    out["trend"]["bloomz"] = TB
    Dt = L["d_trend"]
    out["trend"]["llama1"] = {key: {nm: entry(Dt[key][c], i_E["path"], i_E["sha256"],
                                              Dt[key][c]["source_Tr_p_ci"].split(" ", 1)[1])
                                    for nm, c in [("raw", "s_obs"), ("after", "uncensored"), ("s_cal", "s_cal")]}
                              for key, _, _ in ROWS_ALL}
    To = O["trend"]
    TO = {}
    for key, _, _ in ROWS_ALL:
        TO[key] = {}
        for nm, c in [("raw", "s_obs"), ("after", "uncensored"), ("s_cal", "s_cal")]:
            t = To[key][c]
            e = entry(t, i_O["path"], i_O["sha256"], t["src"]["key_Tr"])
            e["key_p"], e["key_ci"] = t["src"]["key_p"], t["src"]["key_ci"]
            TO[key][nm] = e
    out["trend"]["openllama"] = TO

    # thresholds by size
    BT = nums["b_thresholds"]
    p_panel = tilde(f"{D.LM_DIR}/panel_layermax.parquet")
    out["thr"]["bloomz"] = {
        "x": [float(BT[lm]["lm_params"]) for lm in D.BLOOMZ],
        "T": [BT[lm]["T"] for lm in D.BLOOMZ], "c_alpha": [BT[lm]["c_alpha"] for lm in D.BLOOMZ],
        "Bbar": [BT[lm]["pred_largest_baseline"] for lm in D.BLOOMZ],
        "sel": [BT[lm]["c_alpha_entry_at_argmax"] for lm in D.BLOOMZ],
        "bbar_style": "INK dashed as in every family (v8); status: post hoc retrodiction from features",
        "src": {"T": f"{p_panel} rows pre={PRE}, metric=cka_biased, subset=full, column s_obs (f1_data b_thresholds)",
                "c_alpha": f"{p_panel} column c_alpha", "sel": f"{p_panel} column entry_c_alpha_at_argmax",
                "Bbar": tilde(f"{D.LM_DIR}/indep/diag_null_profile.json") + f" {PRE}/PR_profile; "
                        "sqrt(max_u PR_lm,u * max_v PR_vit,v)/(m-1) (f1_data b_thresholds pred_largest_baseline)",
                "x": f"{p_panel} column lm_params"}}
    ET = L["e_thresholds"]
    p_pbL = tilde(f"{LD.LL}/partB/partB_features.json")
    out["thr"]["llama1"] = {
        "x": [float(ET[lm]["num_params"]) for lm in LD.LLAMA],
        "T": [ET[lm]["T"] for lm in LD.LLAMA], "c_alpha": [ET[lm]["c_alpha"] for lm in LD.LLAMA],
        "Bbar": [ET[lm]["Bbar_partB"] for lm in LD.LLAMA], "sel": [ET[lm]["c_alpha_entry_at_argmax"] for lm in LD.LLAMA],
        "bbar_style": "INK dashed as in every family (v8); status: frozen from features before any cross-model score, Part B",
        "src": {"T": tilde(f"{LD.LL}/early/pairs/{PRE}/") + "<lm>__<vit>.json summaries/cka_biased/full/s_obs "
                     "(means asserted against early/summary.json means/cka_biased/full|s_obs)",
                "c_alpha": "same files, summaries/cka_biased/full/c_alpha (asserted against early/summary.json)",
                "sel": "same files, summaries/cka_biased/full/entry_c_alpha_at_argmax",
                "Bbar": f"{p_pbL} per_pre/{PRE}/Bbar/<lm>/<vit> (mean asserted against pooled/mean_Bbar_by_lm)",
                "x": f"{p_pbL} meta/inputs/<lm>/num_params"}}
    R = O["thresholds"]
    out["thr"]["openllama"] = {
        "x": [float(R[lm]["num_params"]) for lm in OD.OL],
        "T": [R[lm]["T"] for lm in OD.OL], "c_alpha": [R[lm]["c_alpha"] for lm in OD.OL],
        "Bbar": [R[lm]["Bbar_partB"] for lm in OD.OL], "sel": [R[lm]["c_alpha_entry_at_argmax"] for lm in OD.OL],
        "bbar_style": "INK dashed as in every family (v8); status: frozen from features before any 13B cross-model score, Part B",
        "src": dict(O["thresholds_src"], path_root=tilde(OD.OLD))}
    out["order_counts"] = {"llama1": L.get("e_calpha_minus_Bbar_51_pairs"), "openllama": O["calpha_minus_Bbar"]}

    # baseline per language-model layer
    out["prof"]["bloomz"] = {"P": nums["c_null_profile"], "sel": nums["c_observed_argmax_rel_depth_all_bloomz_pairs"],
                             "meta": nums["c_null_profile_meta"]}
    out["prof"]["llama1"] = {"P": L["f_null_profile"], "sel": L["f_observed_argmax_rel_depth_all_51_pairs"],
                             "meta": L["f_null_profile_meta"]}
    out["prof"]["openllama"] = {"P": O["null_profile"], "sel": O["observed_argmax_rel_depth_all_51_pairs"],
                                "meta": O["null_profile_meta"]}
    for lm in D.BLOOMZ:                                   # harmonise (as v5.1 F.3): relative depth of the argmax
        P = nums["c_null_profile"][lm]
        P["argmax_rel"] = P["argmax_measured"] / (P["L"] - 1)
        P["max_measured"] = float(np.max(P["measured"]))
    out["L"], out["O"], out["nums"] = L, O, nums
    return out


# ------------------------------------------------------------------ drawing helpers
def ax_at(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def connector_visibility(r, a, c):
    """Visible length (pt) of the raw -> after connector: the segment between the centres minus the raw ring, the
    after dot and the s_cal tick stroke drawn above it (as v5.1 Figure 3)."""
    R, A, C = r * PT_PER_TR, a * PT_PER_TR, c * PT_PER_TR
    lo, hi = min(R, A), max(R, A)
    er, ea, et = (MS["open"] + MS["open_mew"]) / 2, (MS["dot"] + MS["dot_edge"]) / 2, LW["cal_tick"] / 2
    pieces, x = [], lo
    for clo, chi in sorted([(R - er, R + er), (A - ea, A + ea), (C - et, C + et)]):
        if chi <= x:
            continue
        if clo >= hi:
            break
        if clo > x:
            pieces.append(clo - x)
        x = max(x, chi)
    if x < hi:
        pieces.append(hi - x)
    return [round(p, 2) for p in pieces]


def trend_panel(ax, T, first):
    """Row 1: forests as v5.1 Figure 3 (a-c): tints, connector, raw ring, after dot, s_cal tick on top. v7: no y tick
    labels (the score names are figure texts, see main), row guides drawn as lines so that in (a) they can start after
    each name, no endpoint numbers unless SHOW_ENDPOINT_NUMBERS."""
    plotted, guides = {}, {}
    for i, (key, col, lab) in enumerate(ROWS):
        guides[i] = ax.plot(XLIM_TR, [i, i], color=S.GUIDE, lw=LW["guide"], zorder=Z["guide"],
                            solid_capstyle="butt")[0]
        raw, aft, cal = T[key]["raw"], T[key]["after"], T[key]["s_cal"]
        for s in (raw, aft):
            ax.plot(s["ci95"], [i, i], color=S.ci_color(col), lw=LW["ci_bar"], solid_capstyle="butt",
                    zorder=Z["interval"])
        ax.plot([raw["Tr"], aft["Tr"]], [i, i], color=col, lw=LW["connector"], solid_capstyle="butt",
                zorder=Z["connector"])
        ax.plot(raw["Tr"], i, "o", ms=MS["open"], mfc="white", mec=col, mew=MS["open_mew"], zorder=Z["data"],
                clip_on=False)
        ax.plot(aft["Tr"], i, "o", ms=MS["dot"], mfc=col, mec="white", mew=MS["dot_edge"], zorder=Z["marker"],
                clip_on=False)
        ax.plot(cal["Tr"], i, "|", ms=MS["cal_tick"], mec=col, mfc=col, mew=LW["cal_tick"], zorder=Z["marker"] + 0.5,
                clip_on=False)
        plotted[key] = {"label": lab, "colour": col, "ci_tint": S.ci_color(col), "raw": raw,
                        "after_subtraction": aft, "s_cal": cal,
                        "connector_visible_pieces_pt": connector_visibility(raw["Tr"], aft["Tr"], cal["Tr"])}
    ax.set_xlim(*XLIM_TR)
    ax.set_ylim(*YLIM_TR)
    arts = {}
    if SHOW_ENDPOINT_NUMBERS:                          # v6 endpoint numbers (off in v7)
        for nm in ("raw", "after"):
            v = T["cka_biased|full"][nm]["Tr"]
            arts[nm] = S.text(ax, v, -0.30, S.num(v), role="number", ha="center", va="bottom", zorder=Z["label"])
        plotted["cka_biased|full"]["on_panel_numbers"] = {nm: t.get_text() for nm, t in arts.items()}
    ax.axvline(0, color=S.NULL_LINE, lw=LW["zero"], zorder=Z["interval"] + 0.25)     # over the tints
    ax.set_xticks([-1, 0, 1], [S.MINUS + "1", "0", "1"])
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_yticks([])
    return plotted, arts, guides


def mean_of(lst):
    return np.array([d["mean"] for d in lst], float)


def level_panels(axu, axl, Th, fam, first):
    """Row 2: observed layer max (upper segment), thresholds (lower segment); same scale in both segments."""
    x = np.asarray(Th["x"], float) / 1e9
    T, ca, Bb, se_ = mean_of(Th["T"]), mean_of(Th["c_alpha"]), mean_of(Th["Bbar"]), mean_of(Th["sel"])
    # upper: observed maximum, open circles in the CKA hue on a 1.5 pt line (raw, as in the row-1 key)
    axu.plot(x, T, color=C_CKA, lw=LW["data"], marker="o", ms=MS["open_line"], mfc="white", mec=C_CKA,
             mew=MS["open_line_mew"], zorder=Z["marker"], clip_on=False)
    # lower (v8 colours): largest baseline predicted from features (INK dashed: the prediction of Proposition
    # baselines, the same style in every family), selected pair (NULL_LINE dotted, no markers: a reference),
    # c_alpha (CKA-red squares, white edge, drawn on top: they carry the claim; v5.3 order)
    axl.plot(x, Bb, color=C_PRED, lw=LW_PRED, ls=LS_PRED, zorder=Z["interval"], dash_capstyle="butt")
    if SHOW_SELECTED_PAIR:                                # critic-v8: off (author rule 4; F.3 keeps it)
        axl.plot(x, se_, color=C_SEL, lw=LW["ref"], ls=S.LS["dotted"], zorder=Z["connector"], clip_on=False)
    axl.plot(x, ca, ls="", marker="s", ms=MS["line"], mfc=C_THRESH, mec="white", mew=MS["line_edge"],
             zorder=Z["data"] + 0.6, clip_on=False)
    for ax, lim, ticks in ((axu, SEG_UP, [0.35, 0.40, 0.45]), (axl, SEG_LO, [0.10, 0.15, 0.20])):
        ax.set_xscale("log")
        ax.set_xlim(x[0] / 1.33, x[-1] * 1.33)
        ax.set_ylim(*lim)
        ax.set_yticks(ticks, ["%.2f" % t for t in ticks] if first else [""] * len(ticks))
        ax.minorticks_off()
    axu.spines["bottom"].set_visible(False)
    axu.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axl.set_xticks(x, XLAB[fam])
    axl.minorticks_off()
    return dict(x=x, T=T, ca=ca, Bb=Bb, sel=se_)


def break_marks(fig, x_in):
    """'//' on the left spine across the gap between the two segments (INK, LW axis_break)."""
    tr = fig.dpi_scale_trans
    dx, dy = 0.035, 0.022
    out = []
    for y in (R2_Y + SEG_H, R2_Y + SEG_H + SEG_GAP):
        fig.add_artist(Line2D([x_in - dx, x_in + dx], [y - dy, y + dy], transform=tr, color=INK,
                              lw=LW["axis_break"], solid_capstyle="butt"))
        out.append([round(x_in, 3), round(y, 3)])
    return out


def profile_lw(k, n):
    return LW["profile_emph"] if k == n - 1 else LW["profile"]


def profile_panel(ax, Pr, fam, first, ramp):
    """Row 3: CKA random-pairing baseline per language-model layer; lines = Proposition baselines from features,
    dots = permutation means, INK square = peak of each size (the largest baseline that sets c_alpha in row 2)."""
    lms = LMS[fam]
    out = {}
    for k, (lm, col) in enumerate(zip(lms, ramp)):
        c = Pr["P"][lm]
        d = np.asarray(c["rel_depth"])
        zl = Z["data"] + 0.1 * k
        ax.plot(d, c["predicted"], color=col, lw=profile_lw(k, len(lms)), zorder=zl)
        ax.plot(d, c["measured"], ls="", marker="o", ms=MS["profile_dot"], mfc=col, mec="white",
                mew=MS["profile_edge"], zorder=zl + 0.05)
        out[lm] = {"colour": col, "line_width_pt": profile_lw(k, len(lms)),
                   "rel_depth": arr(c["rel_depth"]), "measured_permutation_mean": arr(c["measured"]),
                   "predicted_proposition": arr(c["predicted"]), "argmax_rel": float(c["argmax_rel"]),
                   "max_measured": float(c["max_measured"])}
    for k, lm in enumerate(lms):                          # peaks: CKA-red squares (v8; v7 INK) above every line and dot
        ax.plot(out[lm]["argmax_rel"], out[lm]["max_measured"], ls="", marker="s", ms=MS["line"], mfc=C_THRESH,
                mec="white", mew=MS["line_edge"], zorder=Z["marker"] + 0.1 * k, clip_on=False)
    ax.axhline(MKNN_BASE, color=C_MKNN, lw=LW["data"], zorder=Z["data"])
    lo, hi = float(min(Pr["sel"])), float(max(Pr["sel"]))
    ax.axvspan(lo, hi, color=S.span_color(C_CKA), lw=0, zorder=Z["band"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, YC_MAX)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"] if first else [""] * 4)
    ax.minorticks_off()
    return out, [lo, hi]


def size_key(fig, col_x, y_base, lms, ramp):
    """Ordinal size key in the strip above a profile panel, right of the panel letter (as v5.1 F.3)."""
    x0, x1 = col_x + KEY3_DX, col_x + AX_W
    kax = fig.add_axes([x0 / FIG_W, (y_base - 0.06) / FIG_H, (x1 - x0) / FIG_W, 0.12 / FIG_H])
    kax.set_axis_off()
    kax.set_xlim(0, x1 - x0); kax.set_ylim(-1, 1)
    n = len(lms)
    lab_l, lab_r = SIZE_LABEL[lms[0]], SIZE_LABEL[lms[-1]]
    step, half = 0.115, 0.04
    kx = (0.05 * len(lab_l) + 0.10) + step * np.arange(n)
    for k, (xx, col) in enumerate(zip(kx, ramp)):
        kax.plot([xx - half, xx + half], [0, 0], color=col, lw=profile_lw(k, n), clip_on=False, solid_capstyle="butt")
        kax.plot(xx, 0, "o", ms=MS["profile_dot"], mfc=col, mec="white", mew=MS["profile_edge"], clip_on=False)
    S.text(kax, kx[0] - 0.07, 0, lab_l, role="key", ha="right", va="center_baseline")
    S.text(kax, kx[-1] + 0.07, 0, lab_r, role="key", ha="left", va="center_baseline")
    return kax


def aim_leader(fig, ann, ax, xy, half_pt, square=False):
    """End the leader of ann LEADER_GAP pt off the edge of the marker centred on xy (v5.1 F.3)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = mtext.Text.get_window_extent(ann, r)
    cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
    px, py = ax.transData.transform(xy)
    ux, uy = cx - px, cy - py
    n = np.hypot(ux, uy); ux, uy = ux / n, uy / n
    d = half_pt / max(abs(ux), abs(uy)) if square else half_pt
    ann.arrow_patch.shrinkB = d + LEADER_GAP
    fig.canvas.draw()
    ext = ann.arrow_patch.get_window_extent(r)
    return round(float(d + LEADER_GAP), 2), round(float(np.hypot(ext.width, ext.height) * 72 / fig.dpi), 2)


def text_overlaps(fig):
    """Overlapping text boxes, leaders crossing another text, and boxes leaving the figure (v5.1 Figure 3)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    T = [t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
    bb = [mtext.Text.get_window_extent(t, r) for t in T]
    W, H = fig.bbox.width, fig.bbox.height
    out, outside = [], []
    for i in range(len(T)):
        if bb[i].x0 < -0.5 or bb[i].y0 < -0.5 or bb[i].x1 > W + 0.5 or bb[i].y1 > H + 0.5:
            outside.append(T[i].get_text())
        for j in range(i + 1, len(T)):
            if bb[i].overlaps(bb[j]):
                out.append((T[i].get_text(), T[j].get_text()))
    for i, t in enumerate(T):
        ap = getattr(t, "arrow_patch", None)
        if ap is None or not ap.get_visible():
            continue
        P = ap.get_path().transformed(ap.get_transform()).vertices
        pts = np.concatenate([np.linspace(P[k], P[k + 1], max(2, int(np.hypot(*(P[k + 1] - P[k])) * 4)))
                              for k in range(len(P) - 1)])
        for j in range(len(T)):
            if j != i and any(bb[j].contains(px, py) for px, py in pts):
                out.append(("leader of " + t.get_text(), T[j].get_text()))
    return out, outside


def text_vs_marks(fig, texts, axes_marks):
    """Least distance (pt) from each direct label's box to any plotted mark centre in its axes (diagnostic)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    res = {}
    for t, ax, pts in zip(texts, *axes_marks):
        b = mtext.Text.get_window_extent(t, r)
        P = ax.transData.transform(np.asarray(pts))
        dx = np.maximum(np.maximum(b.x0 - P[:, 0], 0), P[:, 0] - b.x1)
        dy = np.maximum(np.maximum(b.y0 - P[:, 1], 0), P[:, 1] - b.y1)
        res[t.get_text()] = round(float(np.min(np.hypot(dx, dy)) * 72 / fig.dpi), 2)
    return res


# ------------------------------------------------------------------ main
def main():
    d = load()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.set_layout_engine("none")
    P = {"trend": {}, "lev": {}, "prof": {}}
    ax1, ax2u, ax2l, ax3, num_arts, guides = {}, {}, {}, {}, {}, {}
    ramps = {f: ramp_3to1(f) for f in FAMS}
    breaks = []
    for j, f in enumerate(FAMS):
        ax1[f] = ax_at(fig, COL_X[j], R1_Y, AX_W, R1_H)
        P["trend"][f], num_arts[f], guides[f] = trend_panel(ax1[f], d["trend"][f], j == 0)
        ax2l[f] = ax_at(fig, COL_X[j], R2_Y, AX_W, SEG_H)
        ax2u[f] = ax_at(fig, COL_X[j], R2_Y + SEG_H + SEG_GAP, AX_W, SEG_H)
        P["lev"][f] = level_panels(ax2u[f], ax2l[f], d["thr"][f], f, j == 0)
        breaks += break_marks(fig, COL_X[j])
        ax3[f] = ax_at(fig, COL_X[j], R3_Y, AX_W, R3_H)
        P["prof"][f] = profile_panel(ax3[f], d["prof"][f], f, j == 0, ramps[f])

    # ---- column headers: "(a) BLOOMZ" on one line (v7: no status line); letters of rows 2-3 -------------------------
    for j, f in enumerate(FAMS):
        lt = S.panel_letter(fig, "abc"[j], COL_X[j], YB_FAM)
        lt.set_verticalalignment("baseline")
        if SHOW_FAMILY_HEADERS:                          # family-names pass: on (v8: off)
            S.text(fig, (COL_X[j] + FAM_DX) / FIG_W, YB_FAM / FIG_H, S.FAM_LABEL[f], role="header",
                   color=S.FAM_TEXT[f], ha="left", va="baseline")
        for row_top, letters in ((R2_Y + R2_H, "def"), (R3_Y + R3_H, "ghi")):
            lt = S.panel_letter(fig, letters[j], COL_X[j], row_top + LETTER_DY)
            lt.set_verticalalignment("baseline")

    # ---- row-1 score names: one line each, left-aligned at the figure edge, running into the empty left part of (a);
    #      the guides of (a) start ROW_LBL_PAD pt after each name (v6: two-line tick labels in a 1.05 in gutter) ----
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    a0 = ax1["bloomz"]
    row_names, name_geom = {}, {}
    for i, (key, col, lab) in enumerate(ROWS):
        y_fig = a0.transData.transform((0, i))[1] / fig.dpi                      # inches from the figure bottom
        t = S.text(fig, ROW_LBL_X / FIG_W, y_fig / FIG_H, lab, role="tick", ha="left", va="center_baseline")
        row_names[key] = t
        bb = t.get_window_extent(rend)
        x_end_data = a0.transData.inverted().transform((bb.x1 + ROW_LBL_PAD / 72 * fig.dpi, 0))[0]
        guides["bloomz"][i].set_xdata([x_end_data, XLIM_TR[1]])
        # least gap (pt) from the name's box to the leftmost drawn extent of this row in (a): tint, ring, dot, tick
        T = d["trend"]["bloomz"][key]
        r_ring, r_dot, h_tick = (MS["open"] + MS["open_mew"]) / 2, (MS["dot"] + MS["dot_edge"]) / 2, LW["cal_tick"] / 2
        left = [(min(T["raw"]["ci95"][0], T["after"]["ci95"][0]), 0.0), (T["raw"]["Tr"], r_ring),
                (T["after"]["Tr"], r_dot), (T["s_cal"]["Tr"], h_tick), (0.0, LW["zero"] / 2)]
        gaps = [(a0.transData.transform((x, i))[0] - bb.x1) * 72 / fig.dpi - rr for x, rr in left]
        gap_zero = (a0.transData.transform((0.0, i))[0] - bb.x1) * 72 / fig.dpi - LW["zero"] / 2
        name_geom[key] = {"text": lab, "x_in": [round(bb.x0 / fig.dpi, 3), round(bb.x1 / fig.dpi, 3)],
                          "guide_starts_at_Tr": round(float(x_end_data), 4),
                          "gap_to_nearest_mark_or_zero_line_pt": round(float(min(gaps)), 2),
                          "gap_to_zero_line_pt": round(float(gap_zero), 2),
                          "gap_to_nearest_coloured_mark_pt": round(float(min(gaps[:4])), 2),
                          "weight": t.get_fontweight(), "colour": t.get_color(), "size_pt": t.get_fontsize()}
        assert min(gaps) >= NAME_TO_MARK_MIN_PT, (key, gaps)
        assert bb.x0 / fig.dpi >= 0.0 and x_end_data > XLIM_TR[0], (key, bb)
        # v8 (task: the names inside (a) must not read as data): set exactly as the tick labels (bold INK, tick size),
        # never in a data hue, and ending left of the zero line, so each name sits in the label zone of its row
        assert t.get_color() == INK and S._is_bold(t.get_fontweight()) and t.get_fontsize() == S.FS["tick"], key
        assert gap_zero >= NAME_TO_ZERO_MIN_PT, (key, gap_zero)

    # ---- row-1 key: ONE line across the top (ring raw, dot after subtraction, tick s_cal, tint 95% interval) --------
    tr = fig.dpi_scale_trans
    dy = 0.026
    x = KEY_X0
    key_items = []
    def key_text(x_txt, s):
        t = S.text(fig, x_txt / FIG_W, YB_KEY / FIG_H, s, role="key", ha="left", va="baseline")
        fig.canvas.draw()
        return t, t.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi
    # ring (raw)
    rr = (MS["open"] + MS["open_mew"]) / 2 / 72
    fig.add_artist(Line2D([x + rr], [YB_KEY + dy], transform=tr, ls="", marker="o", ms=MS["open"], mfc="white",
                          mec=INK, mew=MS["open_mew"]))
    t, x = key_text(x + 2 * rr + KEY_TXT, "raw"); key_items.append(t.get_text())
    # dot (after subtraction)
    x += KEY_SEP
    rd = (MS["dot"] + MS["dot_edge"]) / 2 / 72
    fig.add_artist(Line2D([x + rd], [YB_KEY + dy], transform=tr, ls="", marker="o", ms=MS["dot"], mfc=INK,
                          mec="white", mew=MS["dot_edge"]))
    t, x = key_text(x + 2 * rd + KEY_TXT, "after subtraction"); key_items.append(t.get_text())
    # tick (s_cal)
    x += KEY_SEP
    fig.add_artist(Line2D([x + 0.01], [YB_KEY + dy], transform=tr, ls="", marker="|", ms=MS["cal_tick"], mec=INK,
                          mfc=INK, mew=LW["cal_tick"]))
    t, x = key_text(x + 0.02 + KEY_TXT, r"$s_{\rm cal}$"); key_items.append(t.get_text())
    # tint (95% interval)
    x += KEY_SEP
    fig.add_artist(Line2D([x, x + 0.10], [YB_KEY + dy] * 2, transform=tr, color=S.ci_color(C_CKA), lw=LW["ci_bar"],
                          solid_capstyle="butt"))
    t, x = key_text(x + 0.10 + KEY_TXT, "95% interval"); key_items.append(t.get_text())
    key_right_in = round(float(x), 3)
    assert key_right_in <= FIG_W - 0.02, key_right_in

    # ---- shared x labels; y labels of rows 2-3 ----------------------------------------------------------------------
    xc = (COL_X[0] + COL_X[2] + AX_W) / 2
    S.text(fig, xc / FIG_W, YB_XLAB1 / FIG_H, r"size trend $\mathrm{Tr}$", role="label", ha="center", va="baseline")
    S.text(fig, xc / FIG_W, YB_XLAB2 / FIG_H, "parameters (B, log scale)", role="label", ha="center", va="baseline")
    S.text(fig, xc / FIG_W, YB_XLAB3 / FIG_H, "relative depth (language model)", role="label", ha="center",
           va="baseline")
    YL_X = 0.075
    S.text(fig, YL_X / FIG_W, (R2_Y + R2_H / 2) / FIG_H, "CKA", role="label", ha="center", va="center",
           rotation=90)
    S.text(fig, YL_X / FIG_W, (R3_Y + R3_H / 2) / FIG_H, "baseline", role="label", ha="center",
           va="center", rotation=90)

    lead = dict(arrowstyle="-", color=INK, lw=LW["leader"], shrinkA=1.0, shrinkB=1.8)
    labels, leaders = {}, {}
    # ---- direct labels of row 2 (v3 F1 b style: placed beside the series, bold) -----------------------------------
    q = P["lev"]["bloomz"]
    X_LEFT = q["x"][0] / 1.33 * 1.06                            # left text edge in (d): 1.06x right of the x limit
    labels["observed max"] = S.text(ax2u["bloomz"], X_LEFT, 0.402, "observed max", role="direct",
                                    color=S.CRIT_TEXT["cka"], ha="left", va="baseline", zorder=Z["label"])
    # ONE label names the squares and the grey line together, in (d), where the series separate most; it states the
    # identity the message rests on (c_alpha ~ largest baseline). v7: the 'post hoc' tag under it is gone.
    s = r"$c_\alpha \approx$largest baseline"           # no literal space: mathtext already spaces the relation
    labels[s] = S.text(ax2l["bloomz"], X_LEFT, 0.208, s, role="direct", ha="left", va="baseline",
                       zorder=Z["label"])
    q = P["lev"]["llama1"]
    if SHOW_SELECTED_PAIR:                               # critic-v8: off with the dotted line it names
        s = "selected pair"                              # v6: 'threshold of the\nselected pair'; v8: MUTED (reference)
        labels[s] = S.text(ax2l["llama1"], q["x"][-1] * 1.33 / 1.04, 0.114, s, role="annot", ha="right",
                           va="baseline", zorder=Z["label"])

    # ---- row 3: size keys, mKNN, one 'largest baseline' leader (g), shading label (h) ------------------------------
    for j, f in enumerate(FAMS):
        size_key(fig, COL_X[j], R3_Y + R3_H + LETTER_DY + 0.030, LMS[f], ramps[f])
    ag = ax3["bloomz"]
    lo, hi = P["prof"]["bloomz"][1]
    S.text(ag, lo - 0.03, MKNN_BASE + 0.006, "mKNN", role="direct", color=S.CRIT_TEXT["mknn"], ha="right",
           va="bottom", zorder=Z["label"])
    peak_to = {}
    prof = P["prof"]["bloomz"][0]
    top = LMS["bloomz"][-1]
    xp, yp = prof[top]["argmax_rel"], prof[top]["max_measured"]
    s = "largest baseline"
    an = ag.annotate(s, xy=(xp, yp), xytext=(0.20, 0.222), ha="left", va="bottom", arrowprops=dict(lead),
                     zorder=Z["label"], **S.text_kw("direct", s))
    labels["largest baseline (g)"] = an
    leaders["row3 bloomz"] = aim_leader(fig, an, ag, (xp, yp), (MS["line"] + MS["line_edge"]) / 2, square=True)
    peak_to["bloomz"] = {"lm": top, "rel_depth": xp, "value": yp}
    # shading label in (h), where the band is widest: two lines right-aligned left of the band, leader into the band
    # above the profiles
    ah = ax3["llama1"]
    lo_h, hi_h = P["prof"]["llama1"][1]
    s = "depths of\nobserved max"
    y_in = 0.262
    an = ah.annotate(s, xy=((lo_h + hi_h) / 2, y_in), xytext=(lo_h - 0.06, y_in), ha="right", va="center",
                     linespacing=0.95, arrowprops=dict(lead, shrinkA=1.0, shrinkB=0.0), zorder=Z["label"],
                     **S.text_kw("annot", s))
    labels["depths of observed max (h)"] = an
    fig.canvas.draw()
    ext = an.arrow_patch.get_window_extent(fig.canvas.get_renderer())
    leaders["row3 shading label (h)"] = {"xy": [float((lo_h + hi_h) / 2), y_in], "band": [float(lo_h), float(hi_h)],
                                         "length_pt": round(float(np.hypot(ext.width, ext.height) * 72 / fig.dpi), 2)}
    assert 8.0 <= leaders["row3 shading label (h)"]["length_pt"] <= 20.0, leaders

    # ---- v8: bold tick labels and axis titles (bold math) before every geometry check and the save --------------------
    n_finalized = S.finalize(fig)
    # finalize() asks every axes for its tick labels, which fills the tick texts of the axis-off key axes (size keys of
    # g-i) as well; matplotlib never draws them, so hide them, or the overlap check below would count phantom texts
    n_hidden = 0
    for a in fig.axes:
        if not a.axison:
            for t in S._axis_tick_labels(a):
                if t.get_visible():
                    t.set_visible(False)
                    n_hidden += 1

    # ---- endpoint numbers of row 1: centred on their marks; neighbour gap ------------------------------------------
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    num_geom = {}
    for f in FAMS:
        for nm, t in num_arts[f].items():
            bb = t.get_window_extent(rend)
            x_mark = ax1[f].transData.transform((t.get_position()[0], 0))[0]
            num_geom[f + "/" + nm] = {"text": t.get_text(), "x_in": [round(bb.x0 / fig.dpi, 3), round(bb.x1 / fig.dpi, 3)],
                                      "centre_minus_mark_pt": round(((bb.x0 + bb.x1) / 2 - x_mark) / fig.dpi * 72, 3)}
            assert abs(num_geom[f + "/" + nm]["centre_minus_mark_pt"]) < 0.05
    if SHOW_ENDPOINT_NUMBERS:                         # v6 asserts on the numbers, kept for the case they return
        for fl, fr in (("bloomz", "llama1"), ("llama1", "openllama")):
            g = (num_geom[fr + "/after"]["x_in"][0] - num_geom[fl + "/raw"]["x_in"][1]) * 72
            num_geom[f"gap_{fl}_raw_to_{fr}_after_pt"] = round(g, 2)
            assert g >= LABEL_GAP_MIN_PT, (fl, fr, g)
    else:                                             # v7: row 1 carries no number but its tick labels
        import re
        tick_txt = {t.get_text() for f in FAMS for t in ax1[f].get_xticklabels()}
        stray = [t.get_text() for a in ax1.values() for t in a.texts if re.search(r"\d", t.get_text())]
        assert not num_geom and not stray, (num_geom, stray, tick_txt)

    # ---- label clearance from the plotted marks of row 2 (diagnostic) ----------------------------------------------
    def pts(q, keys):
        return [(xx, yy) for k in keys for xx, yy in zip(q["x"], q[k])]
    lab_c = r"$c_\alpha \approx$largest baseline"
    LOW = ["ca", "Bb", "sel"] if SHOW_SELECTED_PAIR else ["ca", "Bb"]       # critic-v8: the drawn lower series only
    if SHOW_SELECTED_PAIR:
        clear = text_vs_marks(fig, [labels["observed max"], labels[lab_c], labels["selected pair"]],
                              ([ax2u["bloomz"], ax2l["bloomz"], ax2l["llama1"]],
                               [pts(P["lev"]["bloomz"], ["T"]), pts(P["lev"]["bloomz"], LOW),
                                pts(P["lev"]["llama1"], LOW)]))
    else:
        clear = text_vs_marks(fig, [labels["observed max"], labels[lab_c]],
                              ([ax2u["bloomz"], ax2l["bloomz"]],
                               [pts(P["lev"]["bloomz"], ["T"]), pts(P["lev"]["bloomz"], LOW)]))

    # v6.1: ink clearance (pt) between each direct label's text box and every drawn line (densified in display space,
    # minus half its width) and marker (minus its radius) in the same axes; asserted >= 0.5 pt
    sq_r = (MS["line"] + MS["line_edge"]) / 2 * np.sqrt(2)          # square: half-diagonal (conservative)
    ring_r = (MS["open_line"] + MS["open_line_mew"]) / 2
    dot_r = (MS["profile_dot"] + MS["profile_edge"]) / 2

    def lev_geom(f):
        q = P["lev"][f]
        lines = [(q["x"], q["ca"], 0.0), (q["x"], q["Bb"], LW_PRED)]
        if SHOW_SELECTED_PAIR:
            lines.append((q["x"], q["sel"], LW["ref"]))
        return lines, [(q["x"], q["ca"], sq_r)]

    def prof_geom(f):
        prof = P["prof"][f][0]
        lines = [(prof[lm]["rel_depth"], prof[lm]["predicted_proposition"], prof[lm]["line_width_pt"]) for lm in LMS[f]]
        lines.append(([-0.03, 1.03], [MKNN_BASE] * 2, LW["data"]))
        marks = [(prof[lm]["rel_depth"], prof[lm]["measured_permutation_mean"], dot_r) for lm in LMS[f]]
        marks.append(([prof[lm]["argmax_rel"] for lm in LMS[f]], [prof[lm]["max_measured"] for lm in LMS[f]], sq_r))
        return lines, marks

    def ink_clearance(t, ax, lines, marks):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        b = mtext.Text.get_window_extent(t, r)
        k = 72 / fig.dpi
        best = np.inf
        def box_dist(Pp):
            dx = np.maximum(np.maximum(b.x0 - Pp[:, 0], 0), Pp[:, 0] - b.x1)
            dy = np.maximum(np.maximum(b.y0 - Pp[:, 1], 0), Pp[:, 1] - b.y1)
            return np.hypot(dx, dy) * k
        for xs, ys, lw in lines:
            V = ax.transData.transform(np.column_stack([xs, ys]))
            D = np.concatenate([np.linspace(V[i], V[i + 1], 60) for i in range(len(V) - 1)])
            best = min(best, float(np.min(box_dist(D))) - lw / 2)
        for xs, ys, rad in marks:
            V = ax.transData.transform(np.column_stack([xs, ys]))
            best = min(best, float(np.min(box_dist(V))) - rad)
        return round(best, 2)

    def pixel_gap_pt(t, ax, dpi=600):
        """Gap (pt) between the rendered glyphs of t (leader hidden) and the rendered data lines and markers of ax,
        from two renders at dpi with every other artist hidden (non-white pixels; anti-aliasing counts as ink)."""
        from scipy.ndimage import distance_transform_edt
        structural = {fig.patch} | set(fig.axes) | {a.patch for a in fig.axes}
        allart = [a for a in fig.findobj() if a is not fig and a not in structural]
        vis = {a: a.get_visible() for a in allart}
        ap = getattr(t, "arrow_patch", None)
        dpi0 = fig.dpi
        def render(keep):
            for a in allart:
                a.set_visible(vis[a] and a in keep)
            fig.set_dpi(dpi)
            fig.canvas.draw()
            return np.any(np.asarray(fig.canvas.buffer_rgba())[..., :3] < 250, axis=2)
        ap_vis = ap.get_visible() if ap is not None else None
        try:
            if ap is not None:
                ap.set_visible(False)
            A = render({t})
            B = render(set(ax.get_lines()))
        finally:
            for a, v in vis.items():
                a.set_visible(v)
            if ap is not None:
                ap.set_visible(ap_vis)
            fig.set_dpi(dpi0)
            fig.canvas.draw()
        ys, xs = np.nonzero(A)
        pad = int(30 / 72 * dpi)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, A.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, A.shape[1])
        Bc = B[y0:y1, x0:x1]
        if not Bc.any():
            return 30.0
        dist = distance_transform_edt(~Bc)
        return round(float((dist[A[y0:y1, x0:x1]].min() - 1) * 72 / dpi), 2)

    targets = {"(d) " + lab_c: (labels[lab_c], ax2l["bloomz"]),
               "(d) observed max": (labels["observed max"], ax2u["bloomz"]),
               "(g) largest baseline": (labels["largest baseline (g)"], ax3["bloomz"]),
               "(h) depths of observed max": (labels["depths of observed max (h)"], ax3["llama1"])}
    if SHOW_SELECTED_PAIR:
        targets["(e) selected pair"] = (labels["selected pair"], ax2l["llama1"])
    for key, t in row_names.items():                   # v7: the score names that run into (a)
        targets["(a) " + t.get_text()] = (t, ax1["bloomz"])
    ink = {k: pixel_gap_pt(t, ax) for k, (t, ax) in targets.items()}
    lg = lev_geom("bloomz")
    qb = P["lev"]["bloomz"]
    box = {"(d) " + lab_c: ink_clearance(labels[lab_c], ax2l["bloomz"], *lg),
           "(d) observed max": ink_clearance(labels["observed max"], ax2u["bloomz"], [(qb["x"], qb["T"], LW["data"])],
                                             [(qb["x"], qb["T"], ring_r)]),
           "(g) largest baseline": ink_clearance(labels["largest baseline (g)"], ax3["bloomz"], *prof_geom("bloomz")),
           "(h) depths of observed max": ink_clearance(labels["depths of observed max (h)"], ax3["llama1"],
                                                       *prof_geom("llama1"))}
    if SHOW_SELECTED_PAIR:
        box["(e) selected pair"] = ink_clearance(labels["selected pair"], ax2l["llama1"], *lev_geom("llama1"))
    print("glyph-to-data gap of direct labels, rendered at 600 dpi (pt)", ink)
    print("font-box-to-data clearance of direct labels (pt, conservative)", box)
    assert min(ink.values()) >= 1.0, ink

    # ---- checks, save -----------------------------------------------------------------------------------------------
    # ---- v8: the header line holds the letters only; the semantic colours are the drawn ones ------------------------
    fam_words = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible()
                 and t.get_text().strip() in set(S.FAM_LABEL.values())]
    assert SHOW_FAMILY_HEADERS or not fam_words, fam_words
    from matplotlib.colors import to_hex as _hx
    sem = {"c_alpha_squares": sorted({_hx(l.get_markerfacecolor()).upper() for f in FAMS for l in ax2l[f].get_lines()
                                      if l.get_marker() == "s"}),
           "peak_squares_row3": sorted({_hx(l.get_markerfacecolor()).upper() for f in FAMS for l in ax3[f].get_lines()
                                        if l.get_marker() == "s"}),
           "largest_baseline_line": sorted({(_hx(l.get_color()).upper(), l.get_linestyle(), l.get_linewidth())
                                            for f in FAMS for l in ax2l[f].get_lines() if l.get_linestyle() == "--"}),
           "selected_pair_line": sorted({(_hx(l.get_color()).upper(), l.get_linestyle(), l.get_linewidth())
                                         for f in FAMS for l in ax2l[f].get_lines() if l.get_linestyle() == ":"}),
           "observed_max": sorted({_hx(l.get_color()).upper() for f in FAMS for l in ax2u[f].get_lines()}),
           "mknn_baseline_row3": sorted({_hx(l.get_color()).upper() for f in FAMS for l in ax3[f].get_lines()
                                         if list(l.get_ydata()) == [MKNN_BASE, MKNN_BASE]})}
    assert sem["c_alpha_squares"] == sem["peak_squares_row3"] == [C_CKA], sem
    assert sem["largest_baseline_line"] == [(C_PRED, "--", LW_PRED)], sem
    assert sem["selected_pair_line"] == ([(C_SEL, ":", LW["ref"])] if SHOW_SELECTED_PAIR else []), sem
    assert sem["observed_max"] == [C_CKA] and sem["mknn_baseline_row3"] == [C_MKNN], sem
    assert labels["observed max"].get_color() == S.CRIT_TEXT["cka"]
    if SHOW_SELECTED_PAIR:
        assert labels["selected pair"].get_color() == MUTED
    # critic-v8: rule 4 removals are real (no calibrate-then-max name, no 'selected pair' text, 3 forest rows)
    all_txt = [t.get_text() for t in fig.findobj(mtext.Text) if t.get_visible()]
    assert SHOW_CTM_ROW or not any("calibrate" in s for s in all_txt), all_txt
    assert SHOW_SELECTED_PAIR or not any("selected" in s for s in all_txt), all_txt
    assert [k for k, _, _ in ROWS] == list(row_names) and len(ROWS) == (4 if SHOW_CTM_ROW else 3), list(row_names)
    for f in FAMS:
        assert set(P["trend"][f]) == {k for k, _, _ in ROWS}, (f, sorted(P["trend"][f]))

    ov, outside = text_overlaps(fig)
    pdf, png = os.path.join(HERE, "F_aggregation.pdf"), os.path.join(HERE, "F_aggregation.png")
    fig.savefig(pdf)
    chk = S.check(fig, pdf)                          # v8: bold ticks / titles, bold math, chromatic data, >= 7 pt, TT
    assert sha(S.__file__) == S.STYLE_SHA256 == STYLE_V8_SHA, "hrh_style_v8 changed"
    assert chk["n_type3"] == 0 and chk["min_glyph_pt"] >= 7.0 and not chk["not_bold_tick_or_axis_labels"], chk
    view100 = os.path.join(HERE, "_view_print_100dpi.png")
    S.print_view(pdf, png, view100)                  # v8: the staged PNG is the 200 dpi render of the PDF
    views_png = S.views(png, os.path.join(HERE, "_views_cvd_grey.png"))
    print("check", json.dumps(chk, default=float))
    print("text overlaps", ov, "outside", outside)
    print("label clearance from row-2 mark centres (pt)", clear)
    print("leaders", leaders)
    print("semantic colours drawn", sem, "finalized texts", n_finalized)
    assert not ov and not outside, (ov, outside)

    # ---- asserts against the v5.1 numbers ---------------------------------------------------------------------------
    V1, VA = json.load(open(V51_MAIN)), json.load(open(V51_APP))
    n_cmp = 0

    def same(a, b, what):
        nonlocal n_cmp
        a, b = np.asarray(a, float), np.asarray(b, float)
        assert a.shape == b.shape and np.all(np.abs(a - b) <= TOL), (what, a, b)
        n_cmp += a.size

    for f in FAMS:
        for key, _, _ in ROWS:
            ref = V1["top_row_trends"][f]["rows"][key]
            got = P["trend"][f][key]
            for nm, rk in (("raw", "raw"), ("after_subtraction", "after_subtraction"), ("s_cal", "s_cal")):
                same(got[nm]["Tr"], ref[rk]["Tr"], (f, key, nm, "Tr"))
                same(got[nm]["ci95"], ref[rk]["ci95"], (f, key, nm, "ci95"))
        for key, _, _ in ROWS_ALL:                       # critic-v8: loaded rows, drawn or not (Results numbers)
            ref = V1["top_row_trends"][f]["rows"][key]
            for nm, rk in (("raw", "raw"), ("after", "after_subtraction"), ("s_cal", "s_cal")):
                same(d["trend"][f][key][nm]["Tr"], ref[rk]["Tr"], (f, key, nm, "Tr, loaded"))
                same(d["trend"][f][key][nm]["ci95"], ref[rk]["ci95"], (f, key, nm, "ci95, loaded"))
        R2 = VA["row2_thresholds"][f]; q = P["lev"][f]
        same(q["x"], R2["x_params_B"], (f, "x"))
        same(q["T"], R2["observed_layermax_cka_T"]["mean"], (f, "T"))
        same(q["ca"], R2["c_alpha_of_max"]["mean"], (f, "c_alpha"))
        same(q["Bb"], R2["largest_baseline_Bbar_from_features"]["mean"], (f, "Bbar"))
        same(q["sel"], R2["threshold_of_selected_pair"]["mean"], (f, "sel"))
        R3 = VA["row3_profiles"][f]
        for lm in LMS[f]:
            g, r = P["prof"][f][0][lm], R3["profiles"][lm]
            for k in ("rel_depth", "measured_permutation_mean", "predicted_proposition", "argmax_rel", "max_measured"):
                same(g[k], r[k], (f, lm, k))
        same(P["prof"][f][1], R3["shading_observed_argmax_rel_depth"], (f, "shading"))
        same(MKNN_BASE, R3["mknn_baseline_k_over_m_minus_1"], (f, "mknn"))
    # segment spans equal; every plotted level inside its segment (values, and marker extents in pt)
    assert abs((SEG_UP[1] - SEG_UP[0]) - (SEG_LO[1] - SEG_LO[0])) < TOL
    pt_per_unit = SEG_H * 72 / (SEG_UP[1] - SEG_UP[0])
    seg_check = {}
    for f in FAMS:
        q = P["lev"][f]
        up, lo_ = np.asarray(q["T"]), np.concatenate([q[k] for k in LOW])        # critic-v8: drawn series only
        assert np.all((up > SEG_UP[0]) & (up < SEG_UP[1])) and np.all((lo_ > SEG_LO[0]) & (lo_ < SEG_LO[1])), f
        ring = (MS["open_line"] + MS["open_line_mew"]) / 2
        sq = (MS["line"] + MS["line_edge"]) / 2
        seg_check[f] = {"upper_min_max": [float(up.min()), float(up.max())],
                        "lower_min_max": [float(lo_.min()), float(lo_.max())],
                        "clearance_pt": {"observed_ring_to_upper_bottom": round((up.min() - SEG_UP[0]) * pt_per_unit - ring, 2),
                                         "observed_ring_to_upper_top": round((SEG_UP[1] - up.max()) * pt_per_unit - ring, 2),
                                         "c_alpha_square_to_lower_top": round((SEG_LO[1] - np.max(q["ca"])) * pt_per_unit - sq, 2),
                                         "c_alpha_square_to_lower_bottom": round((np.min(q["ca"]) - SEG_LO[0]) * pt_per_unit - sq, 2)}}
        if SHOW_SELECTED_PAIR:
            seg_check[f]["clearance_pt"]["selected_line_to_lower_bottom"] = round(
                (np.min(q["sel"]) - SEG_LO[0]) * pt_per_unit - LW["ref"] / 2, 2)
        assert min(seg_check[f]["clearance_pt"].values()) > 0, seg_check[f]
    # c_alpha order across sizes (text: 51/51 LLaMA-1, 33/33 OpenLLaMA)
    O = d["O"]["calpha_minus_Bbar"]
    assert O["n_calpha_larger_size_higher_recomputed"] == "51/51" and O["PO2aii"]["x"] == 33 == O["PO2aii"]["n"], O
    assert d["L"]["e_calpha_minus_Bbar_51_pairs"]["n_calpha_larger_size_higher"] == "51/51", d["L"]["e_calpha_minus_Bbar_51_pairs"]
    # the caption carries no result numbers (author rule 2026-10-10): beyond refs and 'LLaMA-1', at most the pool size
    # 17 and the interval level 95 (v6 asserted exactly these two; v7 states neither, so the set may be smaller).
    # v7: the caption is short (author 2026-10-10): at most CAPTION_MAX_WORDS words, bold claim at most ~30 words.
    # v8: the same checks on caption_v8.tex; bold claim at most BOLD_MAX_WORDS (task: about 25); the caption names the
    # three families (the column titles are gone) and the setting words of the dropped headers live there.
    # Family-names pass: the column titles are back, so the caption no longer names the families (the assert below
    # holds through SHOW_FAMILY_HEADERS, unchanged).
    cap_path = os.path.join(HERE, "caption_v8.tex")
    caption = open(cap_path).read().strip() if os.path.exists(cap_path) else None
    assert caption is not None, "caption_v8.tex missing"
    import re
    digits = re.findall(r"\d+(?:\.\d+)?", re.sub(r"\\ref\{[^}]*\}|LLaMA-1", "", caption))
    assert set(digits) <= {"17", "95"}, digits
    body = re.sub(r"\\caption\{|\\textbf\{|\\ref\{[^}]*\}|[{}$\\]|~", " ", caption)
    cap_words = len(re.findall(r"[A-Za-z][A-Za-z\-]*|\(\w--\w\)", body))
    bold = re.sub(r"\([^)]*\\ref.*$", "", caption.split(r"\textbf{", 1)[1].split(".}", 1)[0])
    bold_words = len(re.findall(r"[A-Za-z][A-Za-z\-]*", re.sub(r"\\[a-z]+|[{}$~]", " ", bold)))
    assert cap_words <= CAPTION_MAX_WORDS and bold_words <= BOLD_MAX_WORDS, (cap_words, bold_words)
    for bad in ("---", " -- ", ": ", " it.", " it,", "grey line", "black"):   # house style; v8 colour words
        assert bad not in re.sub(r"\(\w--\w\)", "", caption), bad
    assert SHOW_FAMILY_HEADERS or all(S.FAM_LABEL[f] in caption for f in FAMS), "caption must name the families"
    for word in ("squares", "dashed", "lines", "dots") + (("dotted",) if SHOW_SELECTED_PAIR else ()):
        assert word in caption.lower(), word                                # every v8 mark the caption decodes
    # critic-v8: the caption decodes no mark that is not drawn, and names the random pairing of the baseline
    assert SHOW_SELECTED_PAIR or ("dotted" not in caption and "selected" not in caption), caption
    assert SHOW_CTM_ROW or "calibrate then max" not in caption.lower(), caption
    assert "random-pairing" in caption.lower(), caption
    print("compared values", n_cmp, "segments", seg_check)

    # ---- numbers.json -----------------------------------------------------------------------------------------------
    tr_rows = {f: {k: {kk: vv for kk, vv in v.items()} for k, v in P["trend"][f].items()} for f in FAMS}
    N = {"figure": {
            "id": "F_aggregation", "version": "v8", "label": "fig:exp-aggregation",
            "version_note": "v8 (maker-v8, 2026-10-10): author feedback on v7 Figure 6, applied to every figure (no "
                            "setting headers, Tol palette with a colour for every data series, bold ticks and axis "
                            "titles, secondary comparisons left to the Results text); data, layout and every plotted "
                            "value of v7 unchanged",
            "v8_changes": {
                "style": "hrh_style_v5_1 -> hrh_style_v8 (sha256 %s): CKA #D55E00 -> #CC3311, mKNN #0072B2 -> #0077BB; "
                         "bold tick labels, axis titles and row-1 score names (role tick); S.finalize before savefig"
                         % STYLE_V8_SHA,
                "removed_from_figure_v8": ([] if SHOW_FAMILY_HEADERS else
                                           ["family names over (a)-(c) (BLOOMZ, LLaMA-1, OpenLLaMA): column titles "
                                            "that name the setting; the caption names the families"]),
                "family_names_pass": {"date": "2026-10-10", "SHOW_FAMILY_HEADERS": SHOW_FAMILY_HEADERS,
                                      "change": "family names drawn again after the letters (a)-(c), bold, in "
                                                "S.FAM_TEXT; caption sentence 'Columns show BLOOMZ, LLaMA-1, and "
                                                "OpenLLaMA.' removed; size, data and every other mark unchanged",
                                      "reason": "author 2026-10-10: the family names identify the columns; the rule "
                                                "against setting headers never applied to them",
                                      "previous_files": "_prev_family_names/"},
                "recoloured_v8": {"c_alpha squares (d-f)": "INK -> CKA #CC3311",
                                  "peak squares (g-i)": "INK -> CKA #CC3311",
                                  "largest baseline from features (d-f)": "NULL_LINE solid 1.5 -> INK dashed 1.6 "
                                                                          "(theory: prediction of Proposition)",
                                  "threshold of the selected pair (d-f)": "INK dotted -> NULL_LINE dotted (reference)",
                                  "'selected pair' label (e)": "direct (bold INK) -> annot (MUTED)"},
                "kept_not_dropped_maker_v8": "maker-v8 kept the selected-pair line as the mechanism of the 'CKA, "
                                             "calibrate then max' row; superseded by critic_v8",
                "critic_v8": {
                    "removed_from_figure": (["'CKA, calibrate then max' row of (a)-(c) (ring, dot, tick, tints, "
                                             "guide, score name)"] if not SHOW_CTM_ROW else [])
                                           + (["threshold of the selected layer pair in (d)-(f) (NULL_LINE dotted)",
                                               "'selected pair' label in (e)"] if not SHOW_SELECTED_PAIR else []),
                    "reason": "author rule 4 (2026-10-10): a secondary comparison that the Results text can state "
                              "is not drawn. Calibration before the maximum is a second protocol, and the dotted "
                              "line exists only to explain it; v4 states both in every family and F.3 draws both",
                    "flags": {"SHOW_CTM_ROW": SHOW_CTM_ROW, "SHOW_SELECTED_PAIR": SHOW_SELECTED_PAIR},
                    "row1_height_in": {"4_rows": R1_H_4ROWS, "drawn": R1_H, "rows_drawn": len(ROWS)},
                    "caption": "'Dotted lines give the threshold of the selected layer pair.' removed; "
                               "'random-pairing' restored in the (g--i) sentence",
                    "still_loaded_and_asserted": "all four row-1 rows of every family (v5.1 to 1e-12) and "
                                                 "results_text_numbers (cka_ctm_Tr, selected_pair levels; v7 to 1e-12)",
                    "previous_outputs": "~/hrh_v2_figs/v8/F_aggregation/_prev_maker_v8/"},
                "semantic_colours_drawn": sem, "n_texts_finalized": n_finalized,
                "n_hidden_tick_labels_of_axis_off_key_axes": n_hidden,
                "views": {"print_100dpi": os.path.basename(view100), "cvd_grey": os.path.basename(views_png)}},
            "height_in": {"v6.1": V6_FIG_H, "v7": 4.108, "v8_maker": 4.108, "v8": FIG_H},
            "removed_from_figure_v7": ["status lines under the family names (5 sizes, replication / 3 sizes, "
                                       "predictions frozen / 3 sizes, descriptive)", "'expl.' tag on the BLOOMZ "
                                       "calibrate-then-max row (a)", "'post hoc' tags in (d) and (g)",
                                       "six endpoint numbers of row 1 (0.89/-0.29, 0.68/-1.00, 0.88/-0.59)",
                                       "row-1 key block and two-line score names in the 1.05 in gutter"],
            "shortened_v7": {"(e) 'threshold of the\\nselected pair'": "'selected pair'",
                             "row-3 y label 'random-pairing\\nbaseline'": "'baseline'",
                             "'CKA, calibrate\\nthen max'": "'CKA, calibrate then max' (one line)"},
            "row1_score_names": name_geom, "row1_key_line": {"items": key_items, "right_edge_in": key_right_in},
            "message": "A maximum over layers makes the calibration threshold follow the largest random-pairing "
                       "baseline; where this baseline grows with size at least as fast as the observed maximum, "
                       "calibration removes or reverses the CKA size trend, while mKNN, whose baseline is the same at "
                       "every layer, keeps its trend.",
            "size_in": chk["size_in"], "check": chk, "text_overlaps": ov, "text_outside_figure": outside,
            "pdf_sha256": sha(pdf), "png_sha256": sha(png), "script_sha256": sha(__file__),
            "hrh_style_v8_sha256": sha(S.__file__), "hrh_style_v8_path": tilde(S.__file__),
            "f1_data_sha256": sha(D.__file__),
            "f1_llama_data_sha256": sha(LD.__file__), "ol_data_sha256": sha(OD.__file__),
            "ol_data_path": tilde(OD.__file__),
            "v51_reference": {"row1": {"path": tilde(V51_MAIN), "sha256": sha(V51_MAIN), "key": "top_row_trends"},
                              "rows2_3": {"path": tilde(V51_APP), "sha256": sha(V51_APP),
                                          "keys": "row2_thresholds, row3_profiles"},
                              "asserted_equal_to": TOL, "n_values_compared": n_cmp},
            "geometry_in": {"gutter": GUTTER, "column": AX_W, "gap": GAP, "right_edge": RIGHT, "row1_axes": [R1_Y, R1_H],
                            "row2_axes": [R2_Y, R2_H], "row2_segment_height": SEG_H, "row2_segment_gap": SEG_GAP,
                            "row3_axes": [R3_Y, R3_H], "trend_xlim": XLIM_TR, "trend_ylim": YLIM_TR,
                            "header_baseline": YB_FAM, "key_baseline": YB_KEY},
            "encodings": {
                "row1": "open ring = raw Tr (5.5/1.3 pt), filled dot = after subtraction of the permutation mean "
                        "(4.6 pt), tick = s_cal (8.5 pt, edge 1.4), thin connector raw -> after, opaque 95% tints "
                        "(raw and after), zero line over the tints; no endpoint numbers (v7); score names as one-line "
                        "texts from the figure edge into (a), guides start after each name; key on one line at the "
                        "top; no relabelling band (F.3 keeps it); rows drawn: " + ", ".join(r[2] for r in ROWS)
                        + ("" if SHOW_CTM_ROW else " (critic-v8: no calibrate-then-max row; F.3 keeps it)"),
                "row2": "two-segment y axis, equal scale (0.14 each): upper [0.32, 0.46] observed layer-max CKA "
                        "(open CKA circles 4.8/1.1, line 1.5); lower [0.095, 0.235] c_alpha (CKA-red squares "
                        "4.0/0.6, v8), largest baseline from features (INK dashed 1.6, v8; same style in every "
                        "family)" + (", threshold of the selected pair (NULL_LINE dotted 1.0, no markers, v8)"
                                     if SHOW_SELECTED_PAIR else "; critic-v8: no selected-pair line (F.3 keeps it)")
                        + "; one bold label 'c_alpha ~ largest baseline' in (d)"
                        + ("; MUTED 'selected pair' in (e)" if SHOW_SELECTED_PAIR else "")
                        + "; '//' break marks; x log parameters with nominal ticks; no s.e. bars",
                "row3": "per-layer CKA baseline (max over vision-model layers, mean over the 17 vision models), "
                        "lines = Proposition from features (size ramps darkened to >= 3:1, largest size 1.6 pt on "
                        "top), dots = permutation means (2.4/0.3), CKA-red squares (v8; v7 INK) = peak of each size "
                        "(same marker as c_alpha in row 2), CKA span tint = depths of the observed max over all "
                        "pairs, blue 1.5 pt = mKNN k/(m-1); one 'largest baseline' label (g), shading label (h); "
                        "y label 'baseline'",
                "headers": ("family-names pass: '(a) BLOOMZ', '(b) LLaMA-1', '(c) OpenLLaMA' on one line, names bold "
                            "in S.FAM_TEXT (SHOW_FAMILY_HEADERS = True)" if SHOW_FAMILY_HEADERS else
                            "v8: letters (a)-(c) only; no family names (SHOW_FAMILY_HEADERS = False)")},
            "status_not_drawn_v7": {"bloomz": "5 sizes, replication of the ARH layer maximum under a frozen plan; "
                                 "threshold reading (d, g) post hoc; calibrate-then-max row exploratory",
                       "llama1": "Part B predictions frozen from features before any cross-model score",
                       "openllama": "every outcome descriptive (plan R.1.2); B-bar frozen before any 13B score"},
            "endpoint_numbers": num_geom, "leaders": leaders, "label_clearance_row2_pt": clear,
            "label_glyph_to_data_gap_pt_600dpi": ink, "label_fontbox_to_data_clearance_pt": box,
            "break_marks_in": breaks, "segments": seg_check,
            "size_ramps": {f: {"colours": ramps[f], "contrast_on_white": [round(S.contrast(c), 2) for c in ramps[f]],
                               "v51_FAM_RAMP": S.ramp(f),
                               "v51_contrast_on_white": [round(S.contrast(c), 2) for c in S.ramp(f)],
                               "rule": "light end darkened to >= %.1f:1 along the v5.1 CIELAB path, even L* steps "
                                       "(ramp_3to1)" % RAMP_MIN_CONTRAST} for f in FAMS}},
         "inputs": d["inputs"], "vits": d["vits"],
         "caption_v8_tex": caption, "caption_v8_sha256": sha(cap_path), "caption_v8_words": cap_words,
         "caption_v8_bold_words": bold_words,
         "caption_numbers": "no result numbers; digits outside refs limited to {17, 95}; asserted",
         "row1_trends": {}, "row2_levels": {}, "row3_profiles": {}}
    for j, f in enumerate(FAMS):
        N["row1_trends"][f] = {"panel": "abc"[j], "rows": tr_rows[f],
                               "source_note": "values identical to v5.1 Figure 3 numbers.json top_row_trends"}
        q = P["lev"][f]; Th = d["thr"][f]
        N["row2_levels"][f] = {
            "panel": "def"[j], "lms": LMS[f], "x_params_B": arr(q["x"]), "x_tick_labels": XLAB[f],
            "observed_layermax_cka_T_mean": arr(q["T"]), "c_alpha_of_max_mean": arr(q["ca"]),
            "largest_baseline_Bbar_from_features_mean": arr(q["Bb"]), "bbar_style": Th["bbar_style"],
            "threshold_of_selected_pair_mean": arr(q["sel"]),
            "se_not_drawn": {k: arr([r["se"] for r in Th[k]]) for k in ("T", "c_alpha", "sel")},
            "sources": Th["src"],
            "v51_key": f"row2_thresholds.{f}.{{x_params_B, observed_layermax_cka_T.mean, c_alpha_of_max.mean, "
                       f"largest_baseline_Bbar_from_features.mean, threshold_of_selected_pair.mean}}"}
        prof, band = P["prof"][f]
        N["row3_profiles"][f] = {"panel": "ghi"[j], "profiles": prof, "shading_observed_argmax_rel_depth": band,
                                 "n_pairs_shading": len(d["prof"][f]["sel"]), "mknn_baseline_k_over_m_minus_1": MKNN_BASE,
                                 "peak_squares": {lm: [prof[lm]["argmax_rel"], prof[lm]["max_measured"]] for lm in LMS[f]},
                                 "largest_baseline_leader_to": peak_to.get(f), "meta": d["prof"][f]["meta"],
                                 "v51_key": f"row3_profiles.{f}"}
    drawn = {k for k, _, _ in ROWS}                     # critic-v8: loaded rows that are not drawn, for traceability
    N["row1_trends_not_drawn"] = {f: {k: d["trend"][f][k] for k, _, _ in ROWS_ALL if k not in drawn} for f in FAMS}
    N["results_text_numbers"] = results_numbers(d, P)
    # v8: every plotted value and every Results-text number equal to the v7 numbers.json (read only), to 1e-12
    V7 = f"{FIGS}/v7/F_aggregation/numbers.json"
    N7 = json.load(open(V7))
    n7 = [0]

    def same_tree(a, b, path):
        if isinstance(a, dict):
            assert isinstance(b, dict) and set(a) == set(b), (path, sorted(set(a) ^ set(b)))
            for k in a:
                same_tree(a[k], b[k], path + "/" + str(k))
        elif isinstance(a, (list, tuple)) and a and not isinstance(a[0], str):
            assert len(a) == len(b), path
            for k, (u, v) in enumerate(zip(a, b)):
                same_tree(u, v, "%s[%d]" % (path, k))
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            assert abs(float(a) - float(b)) <= TOL, (path, a, b)
            n7[0] += 1
        else:
            assert a == b, (path, a, b)
    for key in ("row1_trends", "row2_levels", "row3_profiles", "results_text_numbers"):
        cur = json.loads(json.dumps(N[key], default=float))
        old = N7[key]
        if key == "row1_trends":                        # v8 colours differ from v7 by design; values must not
            for f in FAMS:                              # critic-v8: rows not drawn leave numbers.json row1_trends
                for k in [k for k in old[f]["rows"] if k not in cur[f]["rows"]]:
                    assert not SHOW_CTM_ROW and k == "cka_biased|ctm", k
                    for nm, rk in (("raw", "raw"), ("after", "after_subtraction"), ("s_cal", "s_cal")):
                        assert abs(d["trend"][f][k][nm]["Tr"] - old[f]["rows"][k][rk]["Tr"]) <= TOL, (f, k, nm)
                        n7[0] += 1
                    old[f]["rows"].pop(k)
                for r in cur[f]["rows"].values():       # (and the connector pieces in pt: geometry, the v8 column
                    r.pop("colour"); r.pop("ci_tint")       # is 0.02 in narrower; recorded, not compared)
                    r.pop("connector_visible_pieces_pt")
                for r in old[f]["rows"].values():
                    r.pop("colour"); r.pop("ci_tint")
                    r.pop("connector_visible_pieces_pt")
        if key == "row2_levels":                        # v8 restates the line style; the status words are unchanged
            for f in FAMS:
                assert old[f]["bbar_style"].split(" (")[1].rstrip(")") == cur[f]["bbar_style"].split("status: ")[1]
                old[f].pop("bbar_style"); cur[f].pop("bbar_style")
        same_tree(cur, old, key)
    N["figure"]["v7_reference"] = {"path": tilde(V7), "sha256": sha(V7), "n_values_asserted_equal": n7[0],
                                   "keys": "row1_trends (colours and connector_visible_pieces_pt excepted), row2_levels, row3_profiles, "
                                           "results_text_numbers"}
    print("values equal to v7 numbers.json", n7[0])
    json.dump(N, open(os.path.join(HERE, "numbers.json"), "w"), indent=1, default=float)
    print(json.dumps(N["results_text_numbers"], indent=1, default=float))


def results_numbers(d, P):
    """Numbers the Results text states about this figure (none appear in the caption), each with its source."""
    out = {}
    for f in FAMS:
        r = d["trend"][f]
        q = P["lev"][f]
        prof, band = P["prof"][f]
        lms = LMS[f]
        mk = max(abs(r["mknn|full"][nm]["Tr"] - r["mknn|full"]["raw"]["Tr"]) for nm in ("after", "s_cal"))
        out[f] = {
            "cka_layermax_Tr": {nm: round(r["cka_biased|full"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
            "cka_layermax_after_ci95": [round(c, 3) for c in r["cka_biased|full"]["after"]["ci95"]],
            "cka_layermax_s_cal_ci95": [round(c, 3) for c in r["cka_biased|full"]["s_cal"]["ci95"]],
            "cka_layermax_p_one_sided_exact": {nm: r["cka_biased|full"][nm]["p_one_sided_exact"]
                                               for nm in ("raw", "after", "s_cal")},
            "cka_ctm_Tr": {nm: round(r["cka_biased|ctm"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
            "cka_mid_depth_Tr": {nm: round(r["cka_biased|fixed_r0.5"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
            "mknn_layermax_Tr": {nm: round(r["mknn|full"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
            "mknn_max_abs_change": round(mk, 4),
            "levels_smallest_to_largest": {"observed_max": [round(float(q["T"][0]), 4), round(float(q["T"][-1]), 4)],
                                           "c_alpha": [round(float(q["ca"][0]), 4), round(float(q["ca"][-1]), 4)],
                                           "Bbar": [round(float(q["Bb"][0]), 4), round(float(q["Bb"][-1]), 4)],
                                           "selected_pair": [round(float(q["sel"][0]), 4), round(float(q["sel"][-1]), 4)]},
            "gain_smallest_to_largest": {"observed_max": round(float(q["T"][-1] - q["T"][0]), 4),
                                         "c_alpha": round(float(q["ca"][-1] - q["ca"][0]), 4),
                                         "Bbar": round(float(q["Bb"][-1] - q["Bb"][0]), 4)},
            "max_abs_c_alpha_minus_Bbar_of_means": round(float(np.max(np.abs(q["ca"] - q["Bb"]))), 4),
            "c_alpha_minus_selected_pair_at_largest": round(float(q["ca"][-1] - q["sel"][-1]), 4),
            "largest_baseline_rel_depth_by_size": {SIZE_LABEL[lm]: round(prof[lm]["argmax_rel"], 4) for lm in lms},
            "largest_baseline_value_by_size": {SIZE_LABEL[lm]: round(prof[lm]["max_measured"], 4) for lm in lms},
            "observed_argmax_rel_depth_range": [round(v, 4) for v in band],
            "sources": {"Tr, ci95, p": f"row1_trends.{f}.rows.<row>.{{raw, after_subtraction, s_cal}} (path, key "
                                       f"per entry)",
                        "levels": f"row2_levels.{f} (sources per series)",
                        "profiles": f"row3_profiles.{f}.profiles.<lm>.{{argmax_rel, max_measured}}"}}
        # v6.1: per-vision-model Spearman of the CKA layer max after subtraction and under s_cal (Tr is their mean)
        for nm in ("raw", "after", "s_cal"):
            pv = r["cka_biased|full"][nm].get("per_vit_rho")
            if pv is not None:
                vals = np.round(np.array(list(pv.values()), float), 6)
                u, c = np.unique(vals, return_counts=True)
                out[f].setdefault("cka_layermax_per_vit_rho_counts", {})[nm] = {
                    "n_vits": int(len(vals)), "n_negative": int(np.sum(vals < 0)), "n_positive": int(np.sum(vals > 0)),
                    "n_zero": int(np.sum(vals == 0)), "value_counts": {"%g" % k: int(n) for k, n in zip(u, c)}}
        out[f]["gain_c_alpha_minus_gain_observed_max"] = round(float((q["ca"][-1] - q["ca"][0]) - (q["T"][-1] - q["T"][0])), 4)
    gb, go = out["bloomz"]["gain_smallest_to_largest"], out["openllama"]["gain_smallest_to_largest"]
    pt_unit = SEG_H * 72 / (SEG_UP[1] - SEG_UP[0])
    out["wording_notes"] = {
        "at_least_as_fast": "BLOOMZ gains from 0.56B to 7.1B: c_alpha %+.3f, observed max %+.3f, largest baseline "
                            "%+.3f, and the trend is removed (s_cal Tr %.2f), so the message says 'at least as fast "
                            "as', not 'faster than'." % (gb["c_alpha"], gb["observed_max"], gb["Bbar"],
                                                          out["bloomz"]["cka_layermax_Tr"]["s_cal"]),
        "openllama_ranks": "In (f) the OpenLLaMA level lines look nearly parallel at print size (gains c_alpha %+.3f, "
                           "observed max %+.3f; the difference is %.1f pt at the axis scale), yet Tr after subtraction "
                           "is %.2f: Tr is the mean over vision models of a per-model Spearman over 3 sizes, so a small "
                           "excess of baseline growth that recurs across vision models gives a large negative rank "
                           "trend (counts in openllama.cka_layermax_per_vit_rho_counts). The Results text should say "
                           "this." % (go["c_alpha"], go["observed_max"], (go["c_alpha"] - go["observed_max"]) * pt_unit,
                                      out["openllama"]["cka_layermax_Tr"]["after"])}
    qb = P["lev"]["bloomz"]
    out["bloomz"]["c_alpha_jump_1.1B_to_1.7B"] = round(float(qb["ca"][2] - qb["ca"][1]), 4)
    out["bloomz"]["observed_max_change_1.1B_to_1.7B"] = round(float(qb["T"][2] - qb["T"][1]), 4)
    out["bloomz"]["largest_baseline_jump_1.1B_to_1.7B"] = round(float(qb["Bb"][2] - qb["Bb"][1]), 4)
    out["pooled"] = {
        "max_se_row2_not_drawn": round(max(max(r["se"] for r in d["thr"][f][k]) for f in FAMS
                                           for k in ("T", "c_alpha", "sel")), 4),
        "exact_one_sided_p_floor_3_sizes": "1/6 (6 relabellings of 3 sizes; LLaMA-1 and OpenLLaMA)",
        "c_alpha_order_across_sizes": {
            "llama1": d["L"]["e_calpha_minus_Bbar_51_pairs"]["n_calpha_larger_size_higher"]
                      + " (f1_llama_data e_calpha_minus_Bbar_51_pairs.n_calpha_larger_size_higher)",
            "openllama": d["O"]["calpha_minus_Bbar"]["n_calpha_larger_size_higher_recomputed"]
                         + " recomputed (ol_data); P-O2a(ii) %d/%d" % (d["O"]["calpha_minus_Bbar"]["PO2aii"]["x"],
                                                                      d["O"]["calpha_minus_Bbar"]["PO2aii"]["n"])},
        "c_alpha_minus_Bbar_per_pair": {
            "llama1": {k: d["L"]["e_calpha_minus_Bbar_51_pairs"][k] for k in ("min", "max", "median_abs")},
            "openllama": {k: d["O"]["calpha_minus_Bbar"]["all_51_pairs"][k] for k in ("min", "max", "median")}},
        "mknn_baseline_k_over_m_minus_1": round(MKNN_BASE, 4)}
    return out


if __name__ == "__main__":
    main()
