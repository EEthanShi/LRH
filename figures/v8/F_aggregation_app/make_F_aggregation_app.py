"""F_aggregation_app v8 (appendix Figure F.3, fig:app-aggregation): full version of Figure 4.
Message: calibration after a maximum over layers subtracts a threshold that follows the largest random-pairing
baseline, in all three families (Proposition baselines, Corollary aggregation of arxiv_v4).

v8 (2026-10-10, role maker for the appendix figures F_scale_app / F_mst_app / F_aggregation_app). A copy of
~/hrh_v2_figs/v6/F_aggregation_app/make_F_aggregation_app.py (sha256 05f9a614...8ecd, untouched; byte copy next to
this file as make_F_aggregation_app_v6_copy.py). The author's rules of 2026-10-10 (on v6 Figure 4 and on v7 Figure 6):
panels fill the row (no gutter for row labels or a key), little text on the figure, no setting or status headers,
no result numbers on the figure, no secondary comparison that the Results text states, one recognized scientific
palette, tick labels and axis titles bold. Data, the three-row design and every plotted value of v6 are unchanged
(asserted against the v5.1 numbers.json to 1e-12 and against the Figure 3 v6 numbers.json as before). Changes:
  * style hrh_style_v8 (sha256 fe994067...a8; asserted): CKA Tol vibrant red #CC3311 (was vermilion #D55E00), mKNN
    Tol vibrant blue #0077BB, debiased CKA Tol vibrant magenta #EE3377, family ramps as before; S.finalize(fig) bolds
    every tick label and axis title; the row-1 score names (tick role) and the shared axis labels are bold;
  * layout of the v7 main Figure 4: gutter 1.00 in -> 0.42 in (only the row 2-3 y ticks and y labels), columns
    1.36 in -> 1.64 in; the six score names of row 1 are one-line texts from the figure edge that run into the empty
    left part of (a), whose row guides start after each name; the row-1 key is one line across the top;
  * removed from the figure (the Results text, Tables tab:app-layermax(-llama, -openllama) and Section 5.1 carry each
    item): the status lines under the family names, the 'expl.' tag and '(expl.)' of the debiased row, both
    'post hoc' tags, the six endpoint numbers of row 1, the note '(b, c) exact one-sided p >= 1/6', and the grey
    relabelling bands of (a) with their key entry (the critical values -0.80 and 0.80 are stated in Appendix
    app:exp-layermax; the band values stay in numbers.json);
  * labels shortened as in Figure 4 v7: 'threshold c_alpha ~ largest baseline from features' (e, with a leader) ->
    'c_alpha ~ largest baseline' (d, beside the squares); 'threshold of the selected pair' -> 'selected pair' (e);
    'random-pairing baseline' -> 'baseline'; one 'largest baseline' leader (g) instead of three; 'depths of observed
    max' moves to (h); row 2 keeps one continuous CKA axis from 0 (the main figure breaks it);
  * outputs: PDF, pdftoppm renders at 200 dpi (staged PNG) and 100 dpi (print view), the normal / deuteranopia /
    protanopia / greyscale views; caption_v8.tex; numbers.json keeps every v6 entry and adds asserts_v8.
Run on ds2213 (CPU only, 1 thread):
  cd ~/hrh_v2_figs/v8/F_aggregation_app && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      nice -n 10 taskset -c 0-5 ~/hrh_v2_figs_venv/bin/python -B make_F_aggregation_app.py > run.log 2>&1

Critic-v8 pass (2026-10-10, role critic for F.1-F.3; the maker-v8 outputs are kept in _prev_maker_v8/). F.3 is the
full version of Figure 4, whose own critic pass (make_F_aggregation.py, critic-v8) changed its encodings after this
maker ran; F.3 now draws the same marks the same way, and every plotted value is unchanged (asserted as before):
  * row 2 still drew its thresholds in INK and grey only: the c_alpha squares (d-f) and the peak squares (g-i) are now
    CKA red (C_THRESH, the CKA threshold that s_cal subtracts), the largest baseline predicted from features INK dashed
    (C_PRED, the prediction of Proposition baselines, LW theory), and the threshold of the selected layer pair a
    NULL_LINE dotted reference with a MUTED 'selected pair' label (C_SEL), all as in Figure 4 v8;
  * the family names over (a)-(c) are gone (SHOW_FAMILY_HEADERS = False, author rule 1: a column title names the
    setting of the column); the caption names the columns, and the size keys and the x ticks of (d)-(f) still
    identify them;
  * caption_v8.tex: columns named; '(d--f) ... dashed lines the largest baseline'; the additions over Figure 4 v8
    listed (calibration before the maximum, CKA at the last layer, debiased CKA, the threshold of the selected pair);
  * asserts_v8 gains the encoding checks (square colours, line colours and styles, no family header, caption names
    the families).

v8 c (2026-10-10, role consistency checker across all v8 figures; the critic-v8 files are in ./_prev_consistency/):
  * (g)-(i) draw the per-layer baselines with the size ramps of Figure 4 v8 (ramp_3to1, copied from
    ~/hrh_v2_figs/v8/F_aggregation/make_F_aggregation.py: the S.FAM_RAMP light end darkened along its own CIELAB path
    to >= 3.0:1 on white, inner steps re-spaced to even L*, darkest step unchanged). Before, this full version drew
    the same lines in the lighter S.FAM_RAMP shades (lightest 2.63-2.66:1), so a line of Figure 4 (g) and the same
    line of F.4 (g) had different colours. The colours are asserted equal to the size_ramps of the Figure 4 v8
    numbers.json (read only). Data, caption and size unchanged.

Family-names pass (2026-10-10, author: "家族名加回来"; the files before this pass are kept in ./_prev_family_names/):
  * the family names are identifiers that tell the reader which column is which model family, not setting headers, so
    SHOW_FAMILY_HEADERS = True draws "BLOOMZ", "LLaMA-1", "OpenLLaMA" again after the letters (a)-(c), bold, in the
    family text tones S.FAM_TEXT (the style's role 'header'), on the existing header line (figure size unchanged);
  * caption_v8.tex loses the now redundant sentence "Columns show BLOOMZ, LLaMA-1, and OpenLLaMA.";
  * data, layout and every other mark unchanged; every assert unchanged.

v6 docstring follows (unchanged).
F_aggregation_app v6 (appendix Figure F.3, fig:app-aggregation): full version of Figure 3.
Message: calibration after a maximum over layers subtracts a threshold that follows the largest random-pairing
baseline, in all three families (Proposition baselines, Corollary aggregation of arxiv_v4).

v6 (2026-10-10, role fig-maker-v6; build/figs_v6_staging/_design/PLAN.md, section 4 row F.3). Author decision of
2026-10-10: captions carry no result numbers; every number goes into the Results text. F.3 keeps its content (PLAN:
"Content unchanged ... caption without numbers"), so this file is the v5.1 maker
~/hrh_v2_figs/v5/F_aggregation_app/make_F_aggregation_app.py (sha256 0cd051c1...; untouched) with these changes only:
  * style module hrh_style_v5_1 (sha256 97d0ed53...) instead of hrh_style_v5. The two differ only in ARM / ARM_TEXT
    (training-rule colours), which this figure does not use, so no drawn colour changes;
  * ol_data.py imported read-only from the v5 folder (no copy; python -B, sys.dont_write_bytecode);
  * numbers.json: 'quoted_in_caption' becomes 'moved_to_text' (same values) plus 'results_text_numbers', a list of the
    numbers the Results paragraph should state, each with its numbers.json key and its source file and key;
    'on_panel_numbers' lists the numbers drawn on the panels; 'caption_v6' is the number-free caption of
    caption_v6.tex, asserted free of digits other than the design levels 95% and 5%;
  * identity check against the v5.1 render: every plotted value equal to the v5.1 numbers.json to 1e-12, the 300-dpi
    PNG equal pixel by pixel, and the PDF equal byte for byte outside its /CreationDate stamp (asserted).
Run on ds2213 (CPU only, 1 thread):
  cd ~/hrh_v2_figs/v6/F_aggregation_app && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      nice -n 10 taskset -c 0-3 ~/hrh_v2_figs_venv/bin/python -B make_F_aggregation_app.py

v6.1 (2026-10-10, role fig-fixer-v6 after the v6 critic pass; the v6 outputs are kept in ./_prev_v6a/). The figure is
no longer pixel-identical to v5.1; every plotted value still is (asserted), and the drawn additions are listed here:
  * key: a fourth line '95% interval' with the tint swatch of Figure 3 v6, so the orange tints in (a) cannot be read
    as the grey relabelling band; the figure grows by 0.12 in at the top (5.00 -> 5.12 in) to hold it, rows 1-3 keep
    their absolute positions;
  * (d-f) one combined direct label 'threshold c_alpha ~ largest baseline from features' in (e), with a leader to
    the 13B square (v3 F1 b, e carried this claim on the panel); it replaces the separate labels 'largest baseline
    from features' (e) and 'threshold c_alpha' (f);
  * (d-f) the threshold of the selected pair in INK (dotted line, open circles), as in Figure 3 v6 (v5.1: NULL_LINE),
    and its label worded as there: 'threshold of the selected pair';
  * (g-i) an INK square on the peak of each size's profile (the largest baseline), the same marker as c_alpha in
    (d-f) and as Figure 3 v6 (g-i); the 'largest baseline' leader now ends off the square of the largest size;
  * identity check: plotted values against the v5.1 numbers.json (unchanged leaves, only the new key peak_squares
    added) and row 2 / row 3 peaks against the Figure 3 v6 numbers.json, both to 1e-12; no pixel identity;
  * caption: encodings of (a-c), (d-f) and (g-i) defined (dot = after subtraction of the permutation mean; squares,
    grey line, dotted line), still without result numbers.

v5.1 docstring follows (unchanged).
F_aggregation_app (appendix Figure F.3, fig:app-aggregation): full version of Figure 3. Calibration after a maximum
over layers subtracts the largest random-pairing baseline, which grows with size in BLOOMZ, LLaMA-1 and OpenLLaMA
(Proposition 4, Corollary 1 of arxiv_v4).

Columns = families (BLOOMZ, LLaMA-1, OpenLLaMA).
  Row 1 (a-c): size trend Tr, raw / after subtraction of the permutation mean / s_cal, six score rows.
  Row 2 (d-f): per size, mean over 17 vision models of layer-max CKA, its threshold c_alpha, the largest baseline
               B-bar from features and the threshold of the layer pair attaining the maximum (+-1 s.e.).
  Row 3 (g-i): CKA random-pairing baseline per language-model layer, permutation means (dots) and the
               proposition sqrt(PR_A PR_B)/(m-1) (lines), max over vision layers, mean over 17 vision models.

Run on ds2213 (CPU only, <= 4 threads):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/hrh_v2_figs_venv/bin/python -B \
      ~/hrh_v2_figs/v5/F_aggregation_app/make_F_aggregation_app.py
Reads only frozen artifacts, through
  ~/hrh_v2_figs/F1/f1_data.py and f1_llama_data.py (imported read-only, no bytecode written) for BLOOMZ and LLaMA-1,
  ./ol_data.py (copy of f1_llama_data.py re-pointed at S1_2_layermax_openllama, six rows + profile) for OpenLLaMA.
Writes F_aggregation_app.{pdf,png} and numbers.json next to this file.

v2 (2026-10-09, fig-fixer after the critic pass; v1 outputs kept in ./prev_v1/):
  * arrows raw -> after subtraction only where the change exceeds 0.2 (v1: 0.06, two arrows of 7 pt were invisible);
  * s_cal '|' markers filled with the criterion colour (v1 wrote matplotlib cycle colours into the PDF);
  * column headers in the layout of the main-text F_aggregation: '(a) BLOOMZ' over '5 sizes, <status>';
    BLOOMZ status 'replication' (tab:exp-verdicts, row 'WIT, layer max'), its (d) and (g) are post hoc (caption);
  * p note on the baseline of the row-1 x label, right-aligned under (c), 7.5 pt;
  * row 2 x label 'parameters (B, log scale)'; ticks are the nominal model names placed at the actual parameter
    counts (e.g. '30' at 32.5B, '3' at 3.43B);
  * selected-pair circles drawn above the c_alpha squares (LLaMA-1 30B was hidden);
  * OpenLLaMA size ramp extended darker than the family hue (tint 0.30 -> FAM -> 30% black), adjacent dE76 >= 12;
  * '(g)' band label 'depths of observed max'.
  The BLOOMZ calibrate-then-max interval has no frozen CI; it is a ViT bootstrap recomputed in f1_llama_data
  (B = 4000, seed 20261009).
v3 (2026-10-09, fig-consistency pass with the main-text F_aggregation; v2 outputs kept in ./prev_cons/):
  * row 1 marks drawn exactly as in Figure 3 (make_F_aggregation.py): raw -> after as a thin connector in every row
    (no arrowheads, so none lands on an s_cal tick), s_cal ticks 8.5 pt tall and drawn above both circles;
  * row 2 observed maximum as open circles (raw, as in the row-1 key and in Figure 3 d-f);
  * mathtext scripts at 7/8 of the base and the two labels with subscripts (s_cal key, threshold c_alpha) at 8 pt,
    so every subscript is >= 7.0 pt as in Figure 3.
v5 (2026-10-10, fig-maker-v5; copied from ~/hrh_v2_figs/v4/F_aggregation_app/, which is untouched): style module
hrh_style_v5 (build/figs_v5_staging/_research/STYLE_V5_SPEC.md, section 5 'Figure F.3'). Data, numbers and panel
structure unchanged. Every width, size, weight and colour now comes from S.LW / S.MS / S.FS / S.FW / S colour dicts:
  * (a-c) 95% bars 3.6 pt opaque tints (S.ci_color; v4 3.0 pt at alpha 0.35), connectors 1.2, s_cal tick edge 1.4,
    raw rings 5.5 / 1.3, after dots 4.6, zero line 0.8, row guides #DDDDDD 0.6; endpoint numbers bold INK; family
    names bold in the family text tone (OpenLLaMA #6B6B24); 'expl.' MUTED, declared on the grey relabelling band;
  * (d-f) observed max 1.5 pt with open circles 4.8 / 1.1, largest baseline 1.5 pt (dashed for BLOOMZ),
    c_alpha squares 4.0 / white 0.6, selected-pair line 1.0 dotted with circles 3.0 / 0.9, error bars 1.0;
    direct labels bold ('observed max' in the CKA text tone #C25604, the others INK), leaders 0.7;
  * (g-i) prediction lines 1.2 pt (largest size 1.6, drawn on top), dots 2.4 / white 0.3, ramps S.FAM_RAMP,
    mKNN baseline 1.5 pt and its label bold; 'depths of observed max' shading opaque #F6DCC7, label MUTED;
    'mKNN' moved left of the shading so that it stays on white (on the shading it had 3.9:1);
  * panel letters 10 pt bold; S.use() sets the 7/8 script factor (the per-process lines are gone); S.check()
    asserts glyph sizes from the PDF, Type 42 fonts (pdffonts) and text contrast.
v5.1 (2026-10-10, fig-fixer-v5 after the v5 critic pass; v5 outputs kept in ./prev_v5a/):
  * (d-f) no +-1 s.e. bars: at the v5 marker sizes they hid behind every observed-max ring (s.e. 1.6-2.7 pt,
    ring radius 2.95 pt) and almost every c_alpha square, and left only specks and grey stubs crossing the squares
    in (e). The s.e. stay in numbers.json; the caption gives the largest (0.018, row2_max_se);
  * (d) 'threshold of selected pair' without a leader (it was 1.8 pt, a stray tick under the 1.7B ring); the label
    sits directly under the dotted line, which the caption names;
  * leaders of 'observed max' and 'threshold c_alpha' now end a fixed gap off the marker edge (they aimed at the
    top of the s.e. bar);
  * numbers.json known_limits: dot spacing per model computed from the layout (41- and 61-layer models overlap).
v5.2 (2026-10-10, fig-consistency-v5; v5.1 outputs kept in ./prev_v5b/):
  * (a-c) no minor ticks at Tr = -0.5 and 0.5: Figure 3 (a-c) draws the same linear Tr axis without them, and the
    v5 spec keeps minor ticks for log axes only.
v5.3 (2026-10-10, fig-fixer-v5 after the v5.2 critic pass; v5.2 outputs kept in ./prev_v5c/):
  * (d-f) c_alpha squares drawn above the selected-pair circles (Z data + 0.6 over + 0.5). c_alpha carries the
    caption's claim and the (f) label; with the circles on top, 8 of 11 squares lost their lower part and the
    LLaMA-1 30B square showed only a ~1 pt cap. The circles now lose their upper part instead (LLaMA-1 30B: hidden
    but for 0.3 pt; the caption gives the 0.004 gap there). This reverses the v2 order, chosen when the v1 circle
    was hidden; c_alpha is the claim, the selected pair the side reading;
  * key swatch 'relabelling null, 5%' outlined in NULL_LINE at LW guide 0.6 pt, as the decision keys of Figure 4 (a)
    and F.5 (a) (#E5E5E5 has 1.26:1 on white); centred under the key markers. The band in (a) stays unoutlined;
  * caption: the selected pair named ('layer pair selected by the maximum', matching the (d) label and Figure 3's
    caption), the LLaMA-1 30B gap, and the flat mKNN baseline k/(m-1) of (g-i); their numbers in quoted_in_caption.
"""
import hashlib, json, logging, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(HERE, ".mplcfg"))
V5_DIR = os.path.expanduser("~/hrh_v2_figs/v5/F_aggregation_app")   # v5.1 maker, outputs, ol_data.py (read-only)
STYLE_DIR = os.path.expanduser("~/hrh_v2_figs/v8/_style")                # v8 frozen style
assert hashlib.sha256(open(os.path.join(STYLE_DIR, "hrh_style_v8.py"), "rb").read()).hexdigest() == \
    "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8", "hrh_style_v8 changed"
sys.path.insert(0, STYLE_DIR)
sys.path.insert(0, os.path.expanduser("~/hrh_v2_figs/F1"))
sys.path.insert(0, V5_DIR)                                     # v6: ol_data.py from the v5 folder, not copied
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.colors
import matplotlib.font_manager
import hrh_style_v8 as S                                       # v8: frozen v8 style (Tol palette, bold ticks)
import f1_data as D
import f1_llama_data as LD
import ol_data as OD

S.use()
INK, MUTED = S.INK, S.MUTED
LW, MS, Z = S.LW, S.MS, S.Z
C_CKA, C_MKNN, C_CKAD = S.CRIT["cka"], S.CRIT["mknn"], S.CRIT["ckad"]
HOME = os.path.expanduser("~")
PRE = "prh_clip+l2"
M = 1024
FAMS = ["bloomz", "llama1", "openllama"]
LMS = {"bloomz": D.BLOOMZ, "llama1": LD.LLAMA, "openllama": OD.OL}
SIZE_LABEL = {"bigscience__bloomz-560m": "0.56B", "bigscience__bloomz-1b1": "1.1B", "bigscience__bloomz-1b7": "1.7B",
              "bigscience__bloomz-3b": "3B", "bigscience__bloomz-7b1": "7.1B"}
SIZE_LABEL.update(LD.LL_LABEL)
SIZE_LABEL.update(OD.OL_LABEL)
XLAB = {"bloomz": ["0.56", "1.1", "1.7", "3", "7.1"], "llama1": ["7", "13", "30"], "openllama": ["3", "7", "13"]}
SIZES_TXT = {"bloomz": "5 sizes", "llama1": "3 sizes", "openllama": "3 sizes"}
STATUS = {"bloomz": "replication", "llama1": "predictions frozen", "openllama": "descriptive"}
ROWS = [("cka_biased|full", C_CKA, "CKA, layer max"),            # v8: one line each, no '(expl.)' tag
        ("cka_biased|ctm", C_CKA, "CKA, calibrate then max"),
        ("cka_biased|fixed_r1.0", C_CKA, "CKA, last layer"),
        ("cka_biased|fixed_r0.5", C_CKA, "CKA, mid depth"),
        ("mknn|full", C_MKNN, "mKNN, layer max"),
        ("cka_debiased|full", C_CKAD, "debiased CKA, layer max")]
# critic-v8: the encodings of Figure 4 v8 (make_F_aggregation.py, critic-v8 pass), so the full version reads the same
C_THRESH = C_CKA                                               # c_alpha squares (d-f) and peak squares (g-i)
C_PRED, LS_PRED, LW_PRED = S.THEORY, S.LS["dashed"], LW["theory"]   # largest baseline predicted from features
C_SEL = S.NULL_LINE                                            # threshold of the selected layer pair (reference)
SHOW_FAMILY_HEADERS = True                                     # family-names pass: names identify the columns (v8: False)
YB_MAX, YC_MAX = 0.50, 0.30
YB_TICKS = [0, 0.25, 0.5]                                      # v8 row-2 ticks (v6: every 0.1)
Y_OBS_LABEL, Y_CA_LABEL, Y_SEL_LABEL = 0.408, 0.235, 0.125     # v8 baselines of the row-2 direct labels (CKA units)
MKNN_BASE = 10 / (M - 1)

# ------------------------------------------------------------------ geometry (inches; exact print size)
# v8: the layout of the v7 main Figure 4 (make_F_aggregation.py v7), with six score rows and a continuous row-2 axis.
V6_FIG_H = 5.12                                                # v6.2 height, for the report
FIG_W = S.TEXTWIDTH_IN
GUTTER, RIGHT, GAP = 0.42, 5.47, 0.07                          # v6: 1.00 in label column, 0.17 in gaps
AX_W = (RIGHT - GUTTER - 2 * GAP) / 3                          # 1.637 in (v6: 1.36)
COL_X = [GUTTER + i * (AX_W + GAP) for i in range(3)]          # 0.42, 2.127, 3.833 -> right edge 5.47
# bottom-up: row 3 x label, row 3, strip (letters g-i, size keys, row 2 x label), row 2, strip, row 1, header, key
YB_XLAB3 = 0.055
R3_Y, R3_H = 0.320, 0.82                                       # per-layer baselines (v6: 1.05)
YB_XLAB2 = R3_Y + R3_H + 0.235
R2_Y, R2_H = YB_XLAB2 + 0.268, 0.95                            # levels on one CKA axis from 0 (v6: 1.07)
YB_XLAB1 = R2_Y + R2_H + 0.232
R1_Y, R1_H = YB_XLAB1 + 0.268, 1.04                            # six forest rows (v6: 1.23 with number headroom)
YB_FAM = R1_Y + R1_H + 0.075                                   # "(a) BLOOMZ": one header line (v6: two)
YB_KEY = YB_FAM + 0.165                                        # row-1 key: one line across the top
FIG_H = round(YB_KEY + 0.085, 3)
FAM_DX = 0.215                                                 # family name starts after "(a) "
LETTER_DY = 0.075                                              # letters of rows 2-3: baseline above the axes top
ROW_LBL_X = 0.02                                               # left edge of the row-1 score names (figure inches)
ROW_LBL_PAD = 3.0                                              # pt between a score name and the start of its guide
NAME_TO_MARK_MIN_PT = 3.0                                      # least gap between a score name and the marks of (a)
KEY_X0, KEY_SEP, KEY_TXT = 0.02, 0.17, 0.045                   # key line: start, entry gap, marker-to-text gap (in)
KEY3_DX = 0.30                                                 # row-3 size keys start this far right of the column
XLIM_TR = (-1.10, 1.10)                                        # trend axes (v6: -1.06, 1.06)
YLIM_TR = (len(ROWS) - 0.5, -0.55)                             # v6: (5.5, -0.95), headroom for the numbers
PT_PER_TR = AX_W / (XLIM_TR[1] - XLIM_TR[0]) * 72
YL_X = 0.075                                                   # centre of the row 2-3 y labels (figure inches)
LABEL_MIN_CLEAR_PT = 1.0                                       # least gap between a direct label and the marks
MKNN_LABEL_DY = 0.011                                          # (g) 'mKNN' bottom above its line (v6: 0.006, 0.4 pt)
STYLE_V8_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"   # frozen hrh_style_v8
STYLE_V5_1_SHA = "97d0ed5388b2b7a06829ce42645bc545070ecbd3e65549b5020568fecc9f41a6"   # (v6 history)
V5_1_FILES = {"script": ("make_F_aggregation_app.py", "0cd051c168fb061a79d3c6e92b9f26a7250fcf95abf4c7d3ce4ab39c06b847d5"),
              "pdf": ("F_aggregation_app.pdf", "f83e6f5a3b690a370395bb74d379cfe9e16f9d702a4aa1c7e0131584d54a5586"),
              "png": ("F_aggregation_app.png", "40901a10bd40bb91b0feec571f7484f605490fd4f78dd2d078025bd58416431a"),
              "numbers": ("numbers.json", "0ac3ea4c24403fd59f816226a3fcfc35433b8071879993d90a490d42d289549c"),
              "ol_data": ("ol_data.py", "e1272e92ad0e4d14de66d24def37b404f08ad2c173adf09f53b14012804323f5")}
STAGE = "build/figs_v8_staging/F_aggregation_app/"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tilde(p):
    return p.replace(HOME, "~")


def lab(hexc):
    """CIE L*a*b* (D65) of an sRGB hex colour."""
    c = np.array(to_rgb(hexc))
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M_ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M_ @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def dE76(a, b):
    return float(np.linalg.norm(lab(a) - lab(b)))


# ------------------------------------------------------------------ v8 c: the size ramps of Figure 4 v8 (ramp_3to1)
# copied from ~/hrh_v2_figs/v8/F_aggregation/make_F_aggregation.py (critic, palette); names prefixed f4 so the
# helpers of this file stay untouched
F4_RAMP_MIN_CONTRAST = 3.0
_F4_WP = np.array([0.95047, 1.0, 1.08883])                       # D65 white
_F4_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
FIG4_V8_NUMBERS = os.path.expanduser("~/hrh_v2_figs/v8/F_aggregation/numbers.json")   # Figure 4 v8 (read-only)


def _f4_lab(hex_color):
    c = np.array(to_rgb(hex_color))
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (_F4_M @ c) / _F4_WP
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _f4_hex(lab_):
    from matplotlib.colors import to_hex
    fy = (lab_[0] + 16) / 116
    f = np.array([fy + lab_[1] / 500, fy, fy - lab_[2] / 200])
    t = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _F4_WP
    c = np.linalg.solve(_F4_M, t)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
    return to_hex(np.clip(c, 0, 1))


def ramp_3to1(fam):
    """Figure 4 v8: the v5.1 ramp S.FAM_RAMP[fam] with its light end moved along the ramp's own CIELAB path until it
    reaches F4_RAMP_MIN_CONTRAST on white, and the inner steps re-spaced to even L* between the new light end and the
    unchanged dark end (same hue path, same number of steps)."""
    base = S.ramp(fam)
    P = np.array([_f4_lab(h) for h in base])
    def at(L):
        for k in range(len(P) - 1):
            if P[k + 1][0] <= L <= P[k][0]:
                w = (P[k][0] - L) / (P[k][0] - P[k + 1][0])
                return P[k] + w * (P[k + 1] - P[k])
        raise ValueError(L)
    hi, lo = P[0][0], P[-1][0]
    if S.contrast(base[0]) >= F4_RAMP_MIN_CONTRAST:
        L0 = hi
    else:
        a, b = lo, hi
        for _ in range(60):
            m = (a + b) / 2
            a, b = (m, b) if S.contrast(_f4_hex(at(m))) >= F4_RAMP_MIN_CONTRAST + 0.02 else (a, m)
        L0 = a
    Ls = np.linspace(L0, lo, len(base))
    out = [_f4_hex(at(L)) for L in Ls[:-1]] + [base[-1]]
    assert all(S.contrast(c) >= F4_RAMP_MIN_CONTRAST for c in out), (fam, out)
    assert all(S.contrast(out[k]) < S.contrast(out[k + 1]) for k in range(len(out) - 1)), (fam, out)
    return out


def size_ramp(fam, n):
    """v8 c: the family's size ramp of Figure 4 v8 (ramp_3to1 above; asserted equal to the Figure 4 v8 numbers.json).
    v5-v8: S.FAM_RAMP itself (even CIELAB L* steps, light = small, lightest >= 2.6:1 on white). Returns the colours,
    the contrast of each on white and the adjacent dE76 (reported)."""
    r = ramp_3to1(fam)
    f4 = json.load(open(FIG4_V8_NUMBERS))["figure"]["size_ramps"][fam]["colours"]
    assert [c.upper() for c in r] == [c.upper() for c in f4], ("Figure 4 v8 ramp differs", fam, r, f4)
    assert len(r) == n, (fam, r)
    steps = [round(dE76(r[i], r[i + 1]), 1) for i in range(n - 1)]
    return r, [round(S.contrast(c), 2) for c in r], steps


# ------------------------------------------------------------------ data
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

    def entry(t, path, sha256, key, ci_note=None, band=False):
        e = {"Tr": float(t["Tr"]), "ci95": [float(c) for c in t["vit_boot_ci95"]],
             "p_one_sided_exact": float(t["p_one_sided_exact"]), "path": path, "sha256": sha256, "key": key}
        if ci_note:
            e["ci_note"] = ci_note
        if band:
            e["band_5pct"] = [float(t["crit_lo_5pct"]), float(t["crit_hi_5pct"])]
            e["band_note"] = ("one-sided 5% critical values of the exact relabelling null (120 relabellings of the "
                              "5 sizes), recomputed from the per-pair scores (f1_data.trend_stats / f1_llama_data.trend)")
        return e

    # ---- trends: BLOOMZ ----------------------------------------------------------------------------------------
    A = nums["a_trend"]
    TB = {}
    for key, _, _ in ROWS:
        if key == "cka_biased|ctm":
            continue
        met, sub = key.split("|")
        TB[key] = {nm: entry(A[key][c], i_sum["path"], i_sum["sha256"],
                             f"per_pre/{PRE}/trends/{met}/{sub}|{c}/tr_lm", band=(nm == "after"))
                   for nm, c in [("raw", "s_obs"), ("after", "uncensored"), ("s_cal", "s_cal")]}
    C = L["a_trend_ctm_bloomz"]
    assert abs(C["s_obs"]["Tr"] - A["cka_biased|full"]["s_obs"]["Tr"]) < 1e-12        # same raw layer-max T
    note = "no frozen CI; ViT bootstrap recomputed in f1_llama_data (B = 4000, seed 20261009)"
    TB["cka_biased|ctm"] = {
        "raw": TB["cka_biased|full"]["raw"],
        "after": entry(C["uncensored"], i_I["path"], i_I["sha256"],
                       f"per_pre/{PRE}/blocks/cka_b/ctm/bloomz_axis/uncensored (Tr, p_one); recomputed from "
                       f"S1_2_layermax/pairs/{PRE}/<lm>__<vit>.npz entry arrays", note, band=True),
        "s_cal": entry(C["s_cal"], i_I["path"], i_I["sha256"],
                       f"per_pre/{PRE}/blocks/cka_b/ctm/bloomz_axis/s_cal (Tr, p_one); recomputed from entry arrays",
                       note)}
    TB["cka_biased|ctm"]["raw_note"] = "open circle = raw layer-max T (same as the 'CKA, layer max' row)"
    out["trend"]["bloomz"] = TB

    # ---- trends: LLaMA-1 ---------------------------------------------------------------------------------------
    Dt = L["d_trend"]
    TL = {}
    for key, _, _ in ROWS:
        TL[key] = {}
        for nm, c in [("raw", "s_obs"), ("after", "uncensored"), ("s_cal", "s_cal")]:
            src = Dt[key][c]["source_Tr_p_ci"].split(" ", 1)[1]
            TL[key][nm] = entry(Dt[key][c], i_E["path"], i_E["sha256"], src)
    out["trend"]["llama1"] = TL

    # ---- trends: OpenLLaMA -------------------------------------------------------------------------------------
    To = O["trend"]
    TO = {}
    for key, _, _ in ROWS:
        TO[key] = {}
        for nm, c in [("raw", "s_obs"), ("after", "uncensored"), ("s_cal", "s_cal")]:
            t = To[key][c]
            e = entry(t, i_O["path"], i_O["sha256"], t["src"]["key_Tr"])
            e["key_p"], e["key_ci"] = t["src"]["key_p"], t["src"]["key_ci"]
            TO[key][nm] = e
    out["trend"]["openllama"] = TO

    # ---- thresholds by size ------------------------------------------------------------------------------------
    BT = nums["b_thresholds"]
    p_panel = tilde(f"{D.LM_DIR}/panel_layermax.parquet")
    out["thr"]["bloomz"] = {
        "x": [float(BT[lm]["lm_params"]) for lm in D.BLOOMZ],
        "T": [BT[lm]["T"] for lm in D.BLOOMZ], "c_alpha": [BT[lm]["c_alpha"] for lm in D.BLOOMZ],
        "Bbar": [BT[lm]["pred_largest_baseline"] for lm in D.BLOOMZ],
        "sel": [BT[lm]["c_alpha_entry_at_argmax"] for lm in D.BLOOMZ],
        "rel_depth_argmax": [BT[lm]["rel_depth_argmax_LM"] for lm in D.BLOOMZ],
        "offset_calpha_minus_Bbar": nums["b_gap_calpha_minus_pred_largest_baseline_bloomz"],
        "bbar_style": "dashed (post hoc retrodiction from features)",
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
        "rel_depth_argmax": [ET[lm]["rel_depth_argmax_LM"] for lm in LD.LLAMA],
        "offset_calpha_minus_Bbar": L["e_calpha_minus_Bbar_51_pairs"],
        "bbar_style": "solid (frozen from features before any cross-model score, Part B)",
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
        "rel_depth_argmax": [R[lm]["rel_depth_argmax_LM"] for lm in OD.OL],
        "offset_calpha_minus_Bbar": O["calpha_minus_Bbar"],
        "bbar_style": "solid (frozen from features before any 13B cross-model score, Part B)",
        "src": dict(O["thresholds_src"], path_root=tilde(OD.OLD))}

    # ---- baseline per language-model layer ---------------------------------------------------------------------
    out["prof"]["bloomz"] = {"P": nums["c_null_profile"], "sel": nums["c_observed_argmax_rel_depth_all_bloomz_pairs"],
                             "meta": nums["c_null_profile_meta"]}
    out["prof"]["llama1"] = {"P": L["f_null_profile"], "sel": L["f_observed_argmax_rel_depth_all_51_pairs"],
                             "meta": L["f_null_profile_meta"]}
    out["prof"]["openllama"] = {"P": O["null_profile"], "sel": O["observed_argmax_rel_depth_all_51_pairs"],
                                "meta": O["null_profile_meta"]}
    for lm in D.BLOOMZ:                                   # harmonise: relative depth of the measured argmax
        P = nums["c_null_profile"][lm]
        P["argmax_rel"] = P["argmax_measured"] / (P["L"] - 1)
        P["max_measured"] = float(np.max(P["measured"]))
    out["L"], out["O"], out["nums"] = L, O, nums
    return out


# ------------------------------------------------------------------ drawing helpers
def ax_at(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def trend_panel(ax, T):
    """Row 1 (v8, as Figure 4 v7): forests with tints, connector raw -> after, raw ring, after dot, s_cal tick on top.
    No y tick labels (the score names are figure texts, see main), row guides drawn as lines so that in (a) they can
    start after each name, no endpoint numbers, no relabelling band (v6 drew it in (a))."""
    plotted, guides = {}, {}
    r_raw = (MS["open"] + MS["open_mew"]) / 2                 # outer radius of the raw ring (pt)
    r_aft = (MS["dot"] + MS["dot_edge"]) / 2                  # outer radius of the after dot (pt)
    for i, (key, col, lab) in enumerate(ROWS):
        guides[i] = ax.plot(XLIM_TR, [i, i], color=S.GUIDE, lw=LW["guide"], zorder=Z["guide"],
                            solid_capstyle="butt")[0]
        raw, aft, cal = T[key]["raw"], T[key]["after"], T[key]["s_cal"]
        for s in (raw, aft):                                  # opaque tint
            ax.plot(s["ci95"], [i, i], color=S.ci_color(col), lw=LW["ci_bar"], solid_capstyle="butt",
                    zorder=Z["interval"])
        ax.plot([raw["Tr"], aft["Tr"]], [i, i], color=col, lw=LW["connector"], solid_capstyle="butt",
                zorder=Z["connector"])
        vis_pt = abs(aft["Tr"] - raw["Tr"]) * PT_PER_TR - r_raw - r_aft
        ax.plot(raw["Tr"], i, "o", ms=MS["open"], mfc="white", mec=col, mew=MS["open_mew"], zorder=Z["marker"],
                clip_on=False)
        ax.plot(aft["Tr"], i, "o", ms=MS["dot"], mfc=col, mec="white", mew=MS["dot_edge"], zorder=Z["marker"] + 0.1,
                clip_on=False)
        ax.plot(cal["Tr"], i, "|", ms=MS["cal_tick"], color=col, mfc=col, mec=col, mew=LW["cal_tick"],
                zorder=Z["marker"] + 0.2, clip_on=False)
        plotted[key] = {"label": lab, "colour": col, "ci_tint": S.ci_color(col), "raw": raw,
                        "after_subtraction": aft, "s_cal": cal, "connector_visible_pt": round(max(vis_pt, 0.0), 2)}
    # v6 printed the layer-max CKA trends before and after subtraction above the top row; v8 prints none (Results
    # text). The record keeps the v6 strings, marked as not drawn.
    r0 = T["cka_biased|full"]
    plotted["cka_biased|full"]["annotated"] = {nm: S.num(r0[nm]["Tr"]) for nm in ("raw", "after")}
    ax.axvline(0, color=S.NULL_LINE, lw=LW["zero"], zorder=Z["interval"] + 0.25)     # over the tints
    ax.set_xlim(*XLIM_TR)
    ax.set_ylim(*YLIM_TR)
    ax.set_xticks([-1, 0, 1], [S.MINUS + "1", "0", "1"])
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_yticks([])
    return plotted, guides


def mean_se(lst):
    return np.array([d["mean"] for d in lst]), np.array([d["se"] for d in lst])


def threshold_panel(ax, Th, fam, first):
    """Row 2: means over vision models by size on one CKA axis from 0: observed layer maximum (CKA rings on a line,
    'raw' as in the key), largest baseline predicted from features (critic-v8: INK dashed, theory; v8 maker: grey
    line), threshold of the selected layer pair (critic-v8: NULL_LINE dotted reference; v8 maker: INK dotted) and the
    threshold c_alpha of the maximum (critic-v8: CKA-red squares; v8 maker: INK), drawn on top, as Figure 4 v8."""
    x = np.asarray(Th["x"], float) / 1e9
    T, Tse = mean_se(Th["T"]); ca, case = mean_se(Th["c_alpha"]); Bb, _ = mean_se(Th["Bbar"])
    se_, sese = mean_se(Th["sel"])
    ax.plot(x, Bb, color=C_PRED, lw=LW_PRED, ls=LS_PRED, zorder=Z["interval"], dash_capstyle="butt")
    ax.plot(x, se_, color=C_SEL, lw=LW["ref"], ls=S.LS["dotted"], zorder=Z["connector"])
    ax.plot(x, ca, ls="", marker="s", ms=MS["line"], mfc=C_THRESH, mec="white", mew=MS["line_edge"],
            zorder=Z["data"] + 0.6)
    ax.plot(x, T, color=C_CKA, lw=LW["data"], marker="o", ms=MS["open_line"], mfc="white", mec=C_CKA,
            mew=MS["open_line_mew"], zorder=Z["marker"])
    ax.set_xscale("log")
    ax.set_xticks(x, XLAB[fam])
    ax.minorticks_off()
    ax.set_xlim(x[0] / 1.33, x[-1] * 1.33)
    ax.set_ylim(0, YB_MAX)
    ax.set_yticks(YB_TICKS, [("%g" % v) for v in YB_TICKS] if first else [""] * len(YB_TICKS))
    return dict(x=x, T=T, Tse=Tse, ca=ca, case=case, Bb=Bb, sel=se_, selse=sese)


def profile_lw(k, n):
    """Prediction line width: the largest size emphasised (S.LW profile_emph), the others S.LW profile."""
    return LW["profile_emph"] if k == n - 1 else LW["profile"]


def profile_panel(ax, Pr, fam, first, ramp):
    lms = LMS[fam]
    out = {}
    for k, (lm, col) in enumerate(zip(lms, ramp)):
        c = Pr["P"][lm]
        d = np.asarray(c["rel_depth"])
        # each size drawn as line then dots, sizes stacked small -> large, so the emphasised largest size is on top
        zl = Z["data"] + 0.1 * k
        ax.plot(d, c["predicted"], color=col, lw=profile_lw(k, len(lms)), zorder=zl)
        ax.plot(d, c["measured"], ls="", marker="o", ms=MS["profile_dot"], mfc=col, mec="white",
                mew=MS["profile_edge"], zorder=zl + 0.05)
        out[lm] = {"colour": col, "line_width_pt": profile_lw(k, len(lms)),
                   "rel_depth": list(map(float, c["rel_depth"])),
                   "measured_permutation_mean": list(map(float, c["measured"])),
                   "predicted_proposition": list(map(float, c["predicted"])),
                   "max_abs_rel_err": float(c["max_abs_rel_err"]), "argmax_rel": float(c["argmax_rel"]),
                   "max_measured": float(c["max_measured"])}
    # square on the peak of each size (the largest baseline, which sets c_alpha in d-f), above every line and dot;
    # critic-v8: CKA red as the c_alpha squares (Figure 4 v8), v8 maker INK
    for k, lm in enumerate(lms):
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
    """Ordinal size key in the strip above a profile panel (outside the data area), right of the panel letter;
    swatches drawn with the plotted line widths and dot size."""
    x0, x1 = col_x + KEY3_DX, col_x + AX_W
    kax = fig.add_axes([x0 / FIG_W, (y_base - 0.06) / FIG_H, (x1 - x0) / FIG_W, 0.12 / FIG_H])
    kax.set_axis_off()
    kax.set_xticks([])          # v8: S.finalize() would otherwise fill the (undrawn) tick labels of this key axes
    kax.set_yticks([])
    kax.set_xlim(0, x1 - x0); kax.set_ylim(-1, 1)
    n = len(lms)
    w = x1 - x0
    lab_l, lab_r = SIZE_LABEL[lms[0]], SIZE_LABEL[lms[-1]]
    step = 0.115
    left_txt_w = 0.05 * len(lab_l) + 0.04
    sx0 = left_txt_w + 0.06
    kx = sx0 + step * np.arange(n)
    half = 0.04
    for k, (xx, col) in enumerate(zip(kx, ramp)):
        kax.plot([xx - half, xx + half], [0, 0], color=col, lw=profile_lw(k, n), clip_on=False,
                 solid_capstyle="butt")
        kax.plot(xx, 0, "o", ms=MS["profile_dot"], mfc=col, mec="white", mew=MS["profile_edge"], clip_on=False)
    S.text(kax, kx[0] - 0.07, 0, lab_l, role="key", ha="right", va="center_baseline")
    S.text(kax, kx[-1] + 0.07, 0, lab_r, role="key", ha="left", va="center_baseline")
    return kax, (kx[-1] + 0.07 + 0.05 * len(lab_r)) <= w + 0.02


def text_overlaps(fig):
    """Overlapping text boxes, leaders crossing another text, and boxes leaving the figure (as Figure 4 v7)."""
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


LEADER_GAP = 1.2                                               # pt between a leader's end and the marker edge


def aim_leader(fig, ann, ax, xy, half_pt, square=False):
    """Point the leader of annotation ann at the marker centred on xy (data coords of ax) and end it LEADER_GAP pt
    off the marker edge: half_pt = outer radius of a circle, or half the outer side of a square (the leader then
    stops where its ray leaves the square). Returns the shrink (pt) and the leader length (pt)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = mtext.Text.get_window_extent(ann, r)                  # text only (Annotation's extent includes the arrow)
    pad = r.points_to_pixels(4) / 2                            # matplotlib clips the arrow start at text bbox + 2 pt
    cx, cy = (tb.x0 + tb.x1) / 2, (tb.y0 + tb.y1) / 2
    px, py = ax.transData.transform(xy)
    ux, uy = cx - px, cy - py
    n = np.hypot(ux, uy); ux, uy = ux / n, uy / n
    d = half_pt / max(abs(ux), abs(uy)) if square else half_pt
    ann.arrow_patch.shrinkB = d + LEADER_GAP
    fig.canvas.draw()
    ext = ann.arrow_patch.get_window_extent(r)
    length = np.hypot(ext.width, ext.height) * 72 / fig.dpi
    return round(float(d + LEADER_GAP), 2), round(float(length), 2)


def text_vs_marks(fig, items):
    """v6.1 diagnostic: clearance (pt) between each direct label (text only, not its leader) and the drawn marks of
    its axes. items: {name: (text, ax, [(x, y, half_pt), ...])} with points in data coordinates; lines are passed
    as dense samples with half_pt = half the line width. Clearance = distance from the mark centre to the text box
    minus half_pt; negative means the label covers a mark."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    out = {}
    for name, (t, ax, pts) in items.items():
        bb = mtext.Text.get_window_extent(t, r)
        best = np.inf
        for x, y, h in pts:
            px, py = ax.transData.transform((x, y))
            dx = max(bb.x0 - px, 0.0, px - bb.x1)
            dy = max(bb.y0 - py, 0.0, py - bb.y1)
            best = min(best, np.hypot(dx, dy) / px_per_pt - h)
        out[name] = round(float(best), 2)
    return out


def line_samples(x, y, half_pt, logx=False, n=60):
    """Dense samples along a polyline (for text_vs_marks)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    out = []
    for i in range(len(x) - 1):
        t = np.linspace(0, 1, n)
        xs = np.exp(np.log(x[i]) + t * (np.log(x[i + 1]) - np.log(x[i]))) if logx else x[i] + t * (x[i + 1] - x[i])
        out += [(float(a), float(b), half_pt) for a, b in zip(xs, y[i] + t * (y[i + 1] - y[i]))]
    return out


# ------------------------------------------------------------------ v6: caption rule, text numbers, identity check
FN = {"bloomz": "BLOOMZ", "llama1": "LLaMA-1", "openllama": "OpenLLaMA"}
ALLOWED_CAPTION_NUMBERS = {"95": "interval level (design constant)", "5": "level of the relabelling null (design)"}


def caption_digits(tex):
    """Digits left in the caption after removing \\ref/\\label/\\cite arguments. Only the design levels 95% and 5%
    may remain (author decision 2026-10-10: no result numbers in captions)."""
    body = re.sub(r"\\(ref|label|cite[pt]?|eqref)\{[^}]*\}", "", tex)
    body = body.replace("LLaMA-1", "LLaMA")                   # critic-v8: a model name, not a number (Figure 4 v8)
    found = re.findall(r"\d+(?:\.\d+)?", body)
    bad = [x for x in found if x not in ALLOWED_CAPTION_NUMBERS]
    assert not bad, ("result numbers in the caption", bad)
    pct = re.findall(r"(\d+)\\%", body)
    assert sorted(pct) == sorted(found), ("a digit that is not a level in percent", found, pct)
    return {"digits_found": found, "allowed": ALLOWED_CAPTION_NUMBERS, "result_numbers": bad,
            "chars_without_tex_markup": len(re.sub(r"\\[a-zA-Z]+|[{}$~]", "", body))}


def src_of(e):
    return f"{e['path']} :: {e['key']}"


def on_panel_numbers(P, d):
    """Every number v6 drew on the panels (besides tick and size labels). v8 draws none of them; the record keeps the
    v6 strings under 'printed_in_v6_not_drawn_in_v8'."""
    out = {}
    for f in FAMS:
        e = d["trend"][f]["cka_biased|full"]
        out[f"endpoints_{f}"] = {
            "panel": "abc"[FAMS.index(f)],
            "printed_in_v6_not_drawn_in_v8": P["trend"][f]["cka_biased|full"]["annotated"],
            "what": "layer-max CKA trend Tr before calibration (raw ring) and after subtraction of the permutation "
                    "mean (dot), printed above the top row",
            "numbers_json_key": f"row1_trends.{f}.rows.cka_biased|full.{{raw,after_subtraction}}.Tr",
            "source": {"raw": src_of(e["raw"]), "after": src_of(e["after"])}}
    ps = [d["trend"][f][k][nm]["p_one_sided_exact"] for f in ("llama1", "openllama") for k, _, _ in ROWS
          for nm in ("raw", "after", "s_cal")]
    assert abs(min(ps) - 1 / 6) < 1e-12, min(ps)
    out["p_note_bc"] = {"panel": "x label of (c)", "printed_in_v6_not_drawn_in_v8": "(b, c) exact one-sided p >= 1/6",
                        "what": "smallest attainable one-sided p with three sizes (3! = 6 orderings); a design "
                                "bound, asserted equal to the smallest p over every LLaMA-1 and OpenLLaMA trend",
                        "value": min(ps),
                        "numbers_json_key": "row1_trends.{llama1,openllama}.rows.*.*.p_one_sided_exact"}
    return out


def results_text_numbers(d, P, Q, N):
    """Numbers a Results paragraph about F.3 should state (they left the caption), each with its numbers.json key
    and its source. 'in_v4_text' notes where arxiv_v4.tex (read 2026-10-10) already states the number outside the
    figure environments; 'caption_only_v5_1' marks numbers that only the v5.1 caption carried."""
    R = []
    T = d["trend"]

    def add(id_, statement, value, key, source, in_v4_text, status=None):
        R.append({"id": id_, "statement": statement, "value": value, "numbers_json_key": key, "source": source,
                  "in_v4_text": in_v4_text, **({"status": status} if status else {})})

    r3 = lambda v: round(float(v), 3)
    r4 = lambda v: round(float(v), 4)
    # (a-c) layer-max CKA
    for f in FAMS:
        e = T[f]["cka_biased|full"]
        add(f"tr_layermax_{f}", f"{FN[f]}: layer-max CKA trend Tr before calibration, after subtraction of the "
            "permutation mean, under s_cal; 95% interval and exact one-sided p after subtraction",
            {"raw": r3(e["raw"]["Tr"]), "after": r3(e["after"]["Tr"]), "s_cal": r3(e["s_cal"]["Tr"]),
             "after_ci95": [r3(c) for c in e["after"]["ci95"]], "after_p": e["after"]["p_one_sided_exact"],
             "raw_p": e["raw"]["p_one_sided_exact"]},
            f"row1_trends.{f}.rows.cka_biased|full.{{raw,after_subtraction,s_cal}}",
            {nm: src_of(e[nm]) for nm in ("raw", "after", "s_cal")},
            "Sec. 5.4 text and Figure 3 caption (0.89/-0.29/-0.02, 0.68/-1.00/-0.85, 0.88/-0.59/0.18)")
    # (a-c) calibrate then max
    stat = {"bloomz": "exploratory (row label expl.)", "llama1": "against frozen prediction P-L5 (failed)",
            "openllama": "descriptive"}
    for f in FAMS:
        e = T[f]["cka_biased|ctm"]
        add(f"tr_ctm_{f}", f"{FN[f]}: CKA calibrated per layer pair before the maximum, Tr after subtraction and "
            "under s_cal, 95% interval after subtraction",
            {"after": r3(e["after"]["Tr"]), "s_cal": r3(e["s_cal"]["Tr"]),
             "after_ci95": [r3(c) for c in e["after"]["ci95"]], "after_p": e["after"]["p_one_sided_exact"]},
            f"row1_trends.{f}.rows.cka_biased|ctm.{{after_subtraction,s_cal}}",
            {nm: src_of(e[nm]) + ("; " + e[nm]["ci_note"] if "ci_note" in e[nm] else "") for nm in ("after", "s_cal")},
            {"bloomz": "caption only (v5.1 F.3 caption: 0.72)", "llama1": "Sec. 5.4 text (-0.88, P-L5)",
             "openllama": "caption only (v5.1 F.3 caption: -0.15)"}[f], stat[f])
    # (a-c) fixed depth keeps the sign
    fx = {}
    for f in FAMS:
        for k in ("cka_biased|fixed_r1.0", "cka_biased|fixed_r0.5"):
            fx[f"{f}|{k}"] = {nm: r3(T[f][k][nm]["Tr"]) for nm in ("raw", "after", "s_cal")}
    mn = min(min(v.values()) for v in fx.values())
    assert mn > 0, fx
    add("tr_fixed_depth", "CKA at a fixed depth (last layer, mid depth): every trend keeps its positive sign after "
        "subtraction and under s_cal (smallest value given)", {"smallest": mn, "by_family_and_row": fx},
        "row1_trends.<fam>.rows.cka_biased|fixed_r{1.0,0.5}.{raw,after_subtraction,s_cal}.Tr",
        {f: src_of(T[f]["cka_biased|fixed_r1.0"]["after"]).split(" :: ")[0] for f in FAMS},
        "BLOOMZ in App. F 'BLOOMZ Trends' (0.88 -> 0.69, 0.76 -> 0.92); LLaMA-1 s_cal 0.97, 1.00 in Sec. 5.4; "
        "OpenLLaMA caption only")
    # (a-c) mKNN and debiased CKA
    add("mknn_debiased_change", "largest change of the mKNN and debiased-CKA layer-max trends under calibration "
        "(after subtraction or s_cal against raw), per family and pooled",
        {"mknn": {f: Q[f]["mknn_max_abs_change"] for f in FAMS},
         "debiased": {f: Q[f]["debiased_max_abs_change"] for f in FAMS},
         "pooled_mknn": Q["pooled"]["max_abs_change_mknn_all_families"],
         "pooled_debiased": Q["pooled"]["max_abs_change_debiased_all_families"]},
        "moved_to_text.<fam>.{mknn,debiased}_max_abs_change; moved_to_text.pooled",
        {f: {"mknn": src_of(T[f]["mknn|full"]["raw"]).split(" :: ")[0]} for f in FAMS},
        "LLaMA-1 mKNN 0.41 -> 0.38 and OpenLLaMA debiased <= 0.06 in Sec. 5.4; pooled 0.03 / 0.06 caption only",
        "debiased CKA rows exploratory")
    # (a) relabelling band and p
    e = T["bloomz"]["cka_biased|full"]["after"]
    add("bloomz_relabelling_band", "BLOOMZ: one-sided 5% critical values of the exact relabelling null (120 "
        "relabellings of the five sizes) for the layer-max trend after subtraction; the observed -0.29 lies inside",
        {"band": [round(v, 3) for v in e["band_5pct"]], "after": r3(e["Tr"]), "p": e["p_one_sided_exact"]},
        "row1_trends.bloomz.rows.cka_biased|full.after_subtraction.band_5pct", src_of(e) + "; " + e["band_note"],
        "p = 0.68 in App. F 'BLOOMZ Trends'; band values nowhere (drawn only)")
    add("p_three_sizes", "LLaMA-1 and OpenLLaMA: with three sizes every trend has exact one-sided p >= 1/6",
        N["on_panel_numbers_v6_removed"]["p_note_bc"]["value"], "on_panel_numbers_v6_removed.p_note_bc",
        "design bound (3! orderings), asserted against every p_one_sided_exact of (b, c)",
        "Sec. 5.4 text and Figure 3 caption")
    # (d-f) offsets of c_alpha over the largest baseline, and order
    th = d["thr"]
    off = {"bloomz": {k: th["bloomz"]["offset_calpha_minus_Bbar"][k] for k in ("n", "min", "max", "mean")},
           "llama1": {k: th["llama1"]["offset_calpha_minus_Bbar"][k] for k in ("min", "max", "median_abs", "mean")},
           "openllama": th["openllama"]["offset_calpha_minus_Bbar"]["all_51_pairs"]}
    off = {f: {k: (r4(v) if isinstance(v, float) else v) for k, v in o.items()} for f, o in off.items()}
    pooled = [min(o["min"] for o in off.values()), max(o["max"] for o in off.values())]
    add("offset_calpha_minus_Bbar", "threshold c_alpha of the maximum minus the largest baseline predicted from "
        "features, per pair: range per family and pooled", {"by_family": off, "pooled_range": pooled},
        "row2_thresholds.<fam>.offset_calpha_minus_Bbar",
        {f: {"c_alpha": th[f]["src"].get("c_alpha"), "Bbar": th[f]["src"].get("Bbar", th[f]["src"].get("Bbar_partB")),
             **({"path_root": th[f]["src"]["path_root"]} if "path_root" in th[f]["src"] else {})} for f in FAMS},
        "BLOOMZ 0.0017-0.0028 in App. F 'BLOOMZ Thresholds'; LLaMA-1 median 0.0018 and OpenLLaMA median 0.0019 in "
        "Sec. 5.4; pooled 0.0015-0.0028 caption only")
    add("calpha_order", "c_alpha follows the order of the predicted largest baselines across sizes",
        {"llama1": th["llama1"]["offset_calpha_minus_Bbar"]["n_calpha_larger_size_higher"],
         "openllama_PO2aii": f"{Q['openllama']['P-O2a(ii)_order']['x']}/{Q['openllama']['P-O2a(ii)_order']['n']}"},
        "row2_thresholds.llama1.offset_calpha_minus_Bbar.n_calpha_larger_size_higher; moved_to_text.openllama."
        "P-O2a(ii)_order", {"llama1": "recomputed from early/pairs and partB_features.json (f1_llama_data)",
                            "openllama": Q["openllama"]["P-O2a(ii)_order"]["src"]},
        "Sec. 5.4 text (51 of 51, 33 of 33)")
    # (d-f) levels
    for f in FAMS:
        q = P["thr"][f]
        lv = {"sizes": [SIZE_LABEL[lm] for lm in LMS[f]], "observed_max": [r4(v) for v in q["T"]],
              "c_alpha": [r4(v) for v in q["ca"]], "Bbar_from_features": [r4(v) for v in q["Bb"]],
              "rise_smallest_to_largest": Q[f]["rise_smallest_to_largest"]}
        if f == "bloomz":
            lv["jump_1.1B_to_1.7B"] = {"c_alpha": [r4(q["ca"][1]), r4(q["ca"][2])],
                                       "observed_max": [r4(q["T"][1]), r4(q["T"][2])]}
        add(f"levels_{f}", f"{FN[f]}: means over vision models of the observed layer maximum, c_alpha and the "
            "largest baseline from features, by size, and their rise from the smallest to the largest size", lv,
            f"row2_thresholds.{f}.{{observed_layermax_cka_T,c_alpha_of_max,largest_baseline_Bbar_from_features}}"
            ".mean; moved_to_text.%s.rise_smallest_to_largest" % f,
            {k: th[f]["src"][k] for k in th[f]["src"] if k in ("T", "c_alpha", "Bbar", "Bbar_partB", "path_root")},
            {"bloomz": "App. F 'BLOOMZ Thresholds' (0.135 -> 0.184, 0.336 -> 0.384, 0.131 -> 0.181)",
             "llama1": "Sec. 5.4 (Bbar 0.1847, 0.2023, 0.2182); rises in the Figure 3 caption only",
             "openllama": "App. 'OpenLLaMA' (Bbar ... 0.1860); rises in the Figure 3 caption only"}[f],
            "BLOOMZ: post hoc reading of the frozen run" if f == "bloomz" else None)
    # (d-f) threshold of the selected layer pair
    add("selected_pair_threshold", "threshold of the layer pair attaining the maximum: range over sizes, and its gap "
        "below c_alpha at the largest size",
        {f: {"range": Q[f]["sel_pair_threshold_range"], "c_alpha_minus_sel_at_largest":
             Q[f]["c_alpha_minus_sel_pair_at_largest"]} for f in FAMS},
        "moved_to_text.<fam>.{sel_pair_threshold_range,c_alpha_minus_sel_pair_at_largest}; "
        "row2_thresholds.<fam>.threshold_of_selected_pair.mean",
        {f: th[f]["src"].get("sel", th[f]["src"].get("c_alpha_entry_at_argmax")) for f in FAMS},
        "BLOOMZ 0.107-0.121 in App. F 'BLOOMZ Thresholds'; LLaMA-1 0.170-0.216 in App. F 'LLaMA-1 Axis'; "
        "OpenLLaMA 0.146-0.172 and the LLaMA-1 30B gap 0.004 caption only")
    # (g-i) where the largest baseline sits, and the depths of the observed maximum
    add("largest_baseline_depth", "relative depth (language model) of the largest CKA baseline by size, and the "
        "range of depths of the observed maximum over all pairs (shading)",
        {f: {"largest_baseline_by_size": {SIZE_LABEL[lm]: v for lm, v in
                                          Q[f]["largest_baseline_rel_depth_by_size"].items()},
             "observed_max_depths": Q[f]["observed_argmax_rel_depth_range"],
             "n_pairs": N["row3_profiles"][f]["n_pairs_shading"]} for f in FAMS},
        "moved_to_text.<fam>.{largest_baseline_rel_depth_by_size,observed_argmax_rel_depth_range}; "
        "row3_profiles.<fam>.{profiles.<lm>.argmax_rel,shading_observed_argmax_rel_depth}",
        {"bloomz": "S1_2_layermax/pairs/prh_clip+l2/<lm>__<vit>.npz cka_biased.entry_null_mean (profile); "
                   "panel_layermax.parquet column rel_A (shading)",
         "llama1": "S1_2_layermax_llama/early/pairs/prh_clip+l2/<lm>__<vit>.npz cka_biased.entry_null_mean; "
                   "<lm>__<vit>.json summaries/cka_biased/full/argmax_A / (L_A - 1)",
         "openllama": "S1_2_layermax_openllama/early/pairs/prh_clip+l2/<lm>__<vit>.npz cka_biased.entry_null_mean "
                      "(argmax asserted against layermax_openllama_summary.json embedded/E_summary/v/"
                      "x|nullprof|prh_clip+l2|cka_biased|<size>|argmax_rel); json argmax_A / (L_A - 1)"},
        "BLOOMZ 'first two layers at four of five sizes' and 112 of 136 maxima >= 0.9 in Sec. 5.4; LLaMA-1 "
        "0.80-0.88 in App. F 'LLaMA-1 Axis'; OpenLLaMA 0.94 in App. 'OpenLLaMA'")
    add("proposition_vs_permutation", "largest relative error between the permutation means (dots) and "
        "Proposition baselines (lines) over every layer of every size; the dots cover the lines on models with this "
        "many layers or more",
        {"by_family": {f: float("%.3g" % Q[f]["max_abs_rel_err_measured_vs_prop"]) for f in FAMS},
         "pooled": float("%.3g" % Q["pooled"]["max_rel_err_all_families"]),
         "min_layers_where_dots_overlap": Q["pooled"]["min_layers_where_dots_overlap"]},
        "moved_to_text.<fam>.max_abs_rel_err_measured_vs_prop; moved_to_text.pooled",
        "row3_profiles.<fam>.profiles.<lm>.{measured_permutation_mean,predicted_proposition} (sources in "
        "row3_profiles.<fam>.meta)",
        "Sec. 5.4 gives the BLOOMZ median 0.078% over 80,960 layer pairs; the maximum 4.4e-4 caption only")
    add("mknn_baseline", "mKNN random-pairing baseline k/(m-1), the same at every layer and size (Proposition "
        "baselines)", Q["pooled"]["mknn_baseline_k_over_m_minus_1"],
        "moved_to_text.pooled.mknn_baseline_k_over_m_minus_1", "k = 10, m = 1024 (design)",
        "Table of chance k/(m-1) in App. (0.0098 column); caption only for this figure")
    se = N["figure"]["row2_se"]
    fam_max = max(FAMS, key=lambda f: max(se["max_by_family_and_series"][f].values()))
    add("row2_se_not_drawn", "largest s.e. over vision models of any mean in (d-f) (not drawn)",
        {"max": se["max"], "where": fam_max + " observed max, largest size",
         "by_family_and_series": {f: {k: r4(v) for k, v in s.items()} for f, s in
                                  se["max_by_family_and_series"].items()}},
        "figure.row2_se; row2_thresholds.<fam>.*.se", "s.e. (ddof 1)/sqrt(17) over the 17 vision models of the "
        "per-pair values in row2_thresholds.<fam>.sources", "caption only")
    add("design_constants", "design of the runs behind the figure (not results; may stay in the Appendix F text)",
        {"dataset": "WIT", "m": M, "preprocessing": PRE, "n_perm": 500, "n_vision_models": len(d["vits"]),
         "relabellings_bloomz": 120, "boot_B_seed_bloomz_ctm": [4000, 20261009],
         "pairs": {"bloomz": 85, "llama1": 51, "openllama": 51}},
        "inputs; vits", "App. F 'Layer Aggregation' first paragraph (m = 1024, n_pi = 500, 80,960 layer pairs)",
        "already in App. F text (m, n_pi)")
    return R


NEW_KEYS_V6_1 = {"peak_squares"}                              # row3_profiles.<fam>.peak_squares (drawn in v6.1)
V8_STYLE_SUFFIXES = (".colour", ".ci_tint", ".shading_colour", ".label", ".style", ".band")   # v8: may change
FIG3_V6_NUMBERS = os.path.expanduser("~/hrh_v2_figs/v6/F_aggregation/numbers.json")   # Figure 3 v6 (read-only)


def identity_vs_v51(N, pdf, png):
    """v6.1: every plotted value of v5.1 unchanged (to 1e-12); the only new numbers.json key in the plotted blocks is
    peak_squares. The render is no longer compared (v6.1 adds marks). Also compares the row-2 means and the row-3
    peaks with the Figure 3 v6 numbers.json, so the full and the main version plot the same values."""
    V = {k: os.path.join(V5_DIR, f) for k, (f, _) in V5_1_FILES.items()}
    for k, (f, h) in V5_1_FILES.items():
        assert sha(V[k]) == h, ("v5.1 file changed", k)
    old = json.load(open(V["numbers"]))
    new = json.loads(json.dumps({k: N[k] for k in ("row1_trends", "row2_thresholds", "row3_profiles")},
                                default=float))
    stats = {"leaves": 0, "max_abs_diff": 0.0, "new_keys": []}

    def cmp(a, b, path):
        if isinstance(a, dict):
            extra = set(a) - set(b)
            assert isinstance(b, dict) and set(b) <= set(a) and extra <= NEW_KEYS_V6_1, \
                ("keys differ", path, set(a) ^ set(b))
            stats["new_keys"] += [path + "." + k for k in sorted(extra)]
            for k in b:
                cmp(a[k], b[k], path + "." + k)
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), ("length differs", path)
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(x, y, f"{path}[{i}]")
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            if path.endswith(".connector_visible_pt"):        # v8: a layout diagnostic in pt at the drawn scale
                if abs(a - b) > 1e-12:
                    stats.setdefault("layout_values_changed_v8", []).append([path, b, a])
                return
            assert isinstance(b, (int, float)) and abs(a - b) <= 1e-12, ("value differs", path, a, b)
            stats["leaves"] += 1
            stats["max_abs_diff"] = max(stats["max_abs_diff"], abs(a - b))
        elif path.endswith(V8_STYLE_SUFFIXES) or ".annotated." in path + ".":
            if a != b:                                         # v8: colours, labels, style notes (no data value)
                stats.setdefault("style_strings_changed_v8", []).append([path, b, a])
        else:
            assert a == b, ("entry differs", path, a, b)
    for k in ("row1_trends", "row2_thresholds", "row3_profiles"):
        cmp(new[k], old[k], k)
    # Figure 3 v6: row-2 means and row-3 peak squares
    F3 = json.load(open(FIG3_V6_NUMBERS))
    f3 = {"leaves": 0, "max_abs_diff": 0.0, "path": tilde(FIG3_V6_NUMBERS), "sha256": sha(FIG3_V6_NUMBERS)}
    pairs = {"observed_layermax_cka_T": "observed_layermax_cka_T_mean", "c_alpha_of_max": "c_alpha_of_max_mean",
             "largest_baseline_Bbar_from_features": "largest_baseline_Bbar_from_features_mean",
             "threshold_of_selected_pair": "threshold_of_selected_pair_mean"}
    for f in FAMS:
        for mine, theirs in pairs.items():
            a, b = new["row2_thresholds"][f][mine]["mean"], F3["row2_levels"][f][theirs]
            assert len(a) == len(b)
            for x, y in zip(a, b):
                assert abs(x - y) <= 1e-12, ("Figure 3 v6 row 2 differs", f, mine, x, y)
                f3["leaves"] += 1; f3["max_abs_diff"] = max(f3["max_abs_diff"], abs(x - y))
        for lm, (xr, yv) in new["row3_profiles"][f]["peak_squares"].items():
            xr2, yv2 = F3["row3_profiles"][f]["peak_squares"][lm]
            assert abs(xr - xr2) <= 1e-12 and abs(yv - yv2) <= 1e-12, ("Figure 3 v6 peak differs", f, lm)
            f3["leaves"] += 2; f3["max_abs_diff"] = max(f3["max_abs_diff"], abs(xr - xr2), abs(yv - yv2))
    return {"numbers_vs_v5_1": {"compared": "row1_trends, row2_thresholds, row3_profiles (every v5.1 leaf)",
                                **stats},
            "numbers_vs_figure3_v6": {"compared": "row2 means (4 series) and row3 peak squares", **f3},
            "render_vs_v5_1": "not compared: v8 changes palette, weights and layout (revision_v8)",
            "v5_1_pdf_sha256": V5_1_FILES["pdf"][1], "v8_pdf_sha256": sha(pdf), "v8_png_sha256": sha(png)}


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
    no_refs = re.sub(r"\\(?:ref|eqref|label|cite[pt]?)\{[^}]*\}", "", body_all).replace("LLaMA-1", "LLaMA")
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


# ------------------------------------------------------------------ v8: caption rules, revision record, v8 asserts
CAPTION_V6_PATH = os.path.expanduser("~/hrh_v2_figs/v6/F_aggregation_app/caption_v6.tex")   # read-only
CAPTION_V6 = open(CAPTION_V6_PATH).read().strip()
REVISION_V8 = [
    "style hrh_style_v8 (Tol): CKA #CC3311, mKNN #0077BB, debiased CKA #EE3377 (v5.1 #D55E00, #0072B2, #CC79A7); "
    "tick labels, axis titles, score names and shared axis labels bold (S.finalize)",
    "layout of Figure 4 v7: gutter 1.00 -> 0.42 in, columns 1.36 -> %.3f in, figure 5.12 -> FIG_H in; score names one "
    "line from the figure edge, running into the empty left part of (a); key one line across the top" % AX_W,
    "not drawn any more: status lines under the family names, 'expl.' and '(expl.)', 'post hoc' tags in (d) and (g), "
    "the six endpoint numbers of row 1, '(b, c) exact one-sided p >= 1/6', the grey relabelling bands of (a) and "
    "their key entry; every value stays in numbers.json",
    "labels: 'c_alpha ~ largest baseline' in (d) beside the squares (v6: two-line label with leader in (e)); "
    "'selected pair' in (e) (v6: 'threshold of the selected pair' in (d)); one 'largest baseline' leader in (g) "
    "(v6: one per panel); 'depths of observed max' in (h) (v6: (g)); y label 'baseline' (v6: 'random-pairing "
    "baseline')",
    "row 2 keeps one continuous CKA axis from 0 with ticks 0, 0.25, 0.5 (v6: 0 to 0.5 every 0.1)",
    "outputs: staged PNG = pdftoppm 200 dpi render of the PDF; 100 dpi print view; CVD and greyscale views",
    "critic-v8: encodings of Figure 4 v8: c_alpha squares (d-f) and peak squares (g-i) INK -> CKA #CC3311; largest "
    "baseline from features grey solid -> INK dashed (theory, LW theory); threshold of the selected pair INK dotted "
    "-> NULL_LINE dotted with a MUTED regular 'selected pair' label",
    "critic-v8: family names over (a)-(c) removed (author rule 1; Figure 4 v8 likewise); the caption names the columns",
    "family-names pass (2026-10-10): family names drawn again after the letters (a)-(c), bold, in S.FAM_TEXT "
    "(SHOW_FAMILY_HEADERS = True; author: the names identify the columns); caption sentence 'Columns show BLOOMZ, "
    "LLaMA-1, and OpenLLaMA.' removed; size, data and every other mark unchanged",
]


V8_TEXTS_WITH_DIGITS = []


def v8_asserts(fig, chk, ax1, ax2, ax3, row_names, row2_labels, row3_labels, shared_labels, d, P):
    """v8 rules, checked on the drawn figure."""
    fig.canvas.draw()
    texts = [t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
    tt = [t.get_text() for t in texts]
    ticks = {t.get_text() for a in (*ax1.values(), *ax2.values(), *ax3.values())
             for t in a.get_xticklabels() + a.get_yticklabels()}
    size_keys = {SIZE_LABEL[LMS[f][0]] for f in FAMS} | {SIZE_LABEL[LMS[f][-1]] for f in FAMS}
    digit_texts = sorted({s for s in tt if s not in ticks and re.search(r"\d", re.sub(r"\$[^$]*\$", "", s))})
    banned = ("post hoc", "expl", "replication", "predictions frozen", "descriptive", "sizes", "relabelling", "p\\geq",
              "≥", "Tr =")
    print("v8 texts with digits besides tick labels", digit_texts)
    V8_TEXTS_WITH_DIGITS[:] = digit_texts
    band_patches = [p for p in ax1["bloomz"].patches if S.NULL_FILL.lower() in str(p.get_facecolor()).lower()
                    or tuple(np.round(p.get_facecolor(), 4)) == tuple(np.round(matplotlib.colors.to_rgba(S.NULL_FILL), 4))]
    return {
        "style hrh_style_v8 sha256 asserted": sha(S.__file__) == S.STYLE_SHA256 == STYLE_V8_SHA,
        "tick labels and axis titles bold (S.check)": not chk["not_bold_tick_or_axis_labels"]
        and chk["n_tick_labels_checked"] > 0,
        "bold math in bold texts (S.check)": not chk["bold_text_with_regular_math"],
        "score names, shared axis labels and direct labels bold": all(
            float(matplotlib.font_manager.weight_dict.get(t.get_fontweight(), t.get_fontweight())
                  if isinstance(t.get_fontweight(), str) else t.get_fontweight()) >= 600
            for t in [*row_names.values(), *shared_labels,
                      *[v for k, v in row2_labels.items() if k != "selected pair"],
                      row3_labels["mKNN"], row3_labels["largest baseline (g)"]]),
        # critic-v8: the encodings of Figure 4 v8
        "critic-v8: c_alpha squares (d-f) and peak squares (g-i) CKA red": all(
            matplotlib.colors.to_hex(l.get_markerfacecolor()).upper() == C_CKA.upper()
            for f in FAMS for a in (ax2[f], ax3[f]) for l in a.get_lines() if l.get_marker() == "s")
        and sum(1 for f in FAMS for a in (ax2[f], ax3[f]) for l in a.get_lines() if l.get_marker() == "s")
        == len(FAMS) + sum(len(LMS[f]) for f in FAMS),
        "critic-v8: largest baseline from features INK dashed (theory), one line per family": all(
            sum(1 for l in ax2[f].get_lines() if l.get_linestyle() == "--"
                and matplotlib.colors.to_hex(l.get_color()).upper() == S.THEORY.upper()) == 1 for f in FAMS),
        "critic-v8: threshold of the selected pair NULL_LINE dotted, label MUTED regular": all(
            sum(1 for l in ax2[f].get_lines() if l.get_linestyle() == ":"
                and matplotlib.colors.to_hex(l.get_color()).upper() == S.NULL_LINE.upper()) == 1 for f in FAMS)
        and matplotlib.colors.to_hex(row2_labels["selected pair"].get_color()).upper() == S.MUTED.upper()
        and row2_labels["selected pair"].get_fontweight() in ("normal", 400),
        "critic-v8: no INK or grey data marker in rows 2-3 (only theory INK dashed and the dotted reference)": all(
            matplotlib.colors.to_hex(l.get_color()).upper() not in (S.INK.upper(), S.NULL_LINE.upper())
            or l.get_linestyle() in ("--", ":")
            for f in FAMS for a in (ax2[f], ax3[f]) for l in a.get_lines() if l.get_marker() not in ("s",)),
        "critic-v8: no family name drawn (SHOW_FAMILY_HEADERS off)": SHOW_FAMILY_HEADERS or not any(
            S.FAM_LABEL[f] == s for f in FAMS for s in tt),
        "criterion hues drawn (CKA, mKNN, debiased CKA)": {S.CRIT[k].upper() for k in ("cka", "mknn", "ckad")}
        <= set(chk["chromatic_data_colours"]),
        "texts with a digit are tick labels, size keys, family names or '95% interval'": set(digit_texts)
        <= size_keys | {"95% interval"} | {S.FAM_LABEL[f] for f in FAMS},
        "no status line, tag, p note or Tr value drawn": not any(b in s for s in tt for b in banned),
        "no relabelling band in (a)": not band_patches,
        "six score names, as ROWS": [t.get_text() for t in row_names.values()] == [r[2] for r in ROWS],
        "panels fill the row (gutter <= 0.45 in, right edge 5.47 in)": COL_X[0] <= 0.45
        and abs(COL_X[2] + AX_W - RIGHT) < 1e-9,
        "figure lower than v6": FIG_H < V6_FIG_H,
        "min glyph >= 7 pt": chk["min_glyph_pt"] >= 7.0,
        "Type 42 only": chk["n_type3"] == 0,
        "data values unchanged (identity vs v5.1 and Figure 3 v6 asserted in identity_vs_v51)": True,
    }


def main():
    d = load()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.set_layout_engine("none")
    P = {"trend": {}, "thr": {}, "prof": {}}
    ax1, ax2, ax3, guides = {}, {}, {}, {}
    ramps = {}
    for j, f in enumerate(FAMS):
        ax1[f] = ax_at(fig, COL_X[j], R1_Y, AX_W, R1_H)
        P["trend"][f], guides[f] = trend_panel(ax1[f], d["trend"][f])
        ax2[f] = ax_at(fig, COL_X[j], R2_Y, AX_W, R2_H)
        P["thr"][f] = threshold_panel(ax2[f], d["thr"][f], f, j == 0)
        ax3[f] = ax_at(fig, COL_X[j], R3_Y, AX_W, R3_H)
        ramps[f] = size_ramp(f, len(LMS[f]))
        P["prof"][f] = profile_panel(ax3[f], d["prof"][f], f, j == 0, ramps[f][0])

    # ---- column headers: "(a) BLOOMZ" on one line (v8: no status line); letters of rows 2-3 -------------------------
    for j, f in enumerate(FAMS):
        lt = S.panel_letter(fig, "abc"[j], COL_X[j], YB_FAM)
        lt.set_verticalalignment("baseline")
        if SHOW_FAMILY_HEADERS:                                # family-names pass: on (critic-v8: off)
            S.text(fig, (COL_X[j] + FAM_DX) / FIG_W, YB_FAM / FIG_H, S.FAM_LABEL[f], role="header",
                   color=S.FAM_TEXT[f], ha="left", va="baseline")
        for row_top, letters in ((R2_Y + R2_H, "def"), (R3_Y + R3_H, "ghi")):
            lt = S.panel_letter(fig, letters[j], COL_X[j], row_top + LETTER_DY)
            lt.set_verticalalignment("baseline")

    # ---- row-1 score names: one line each (tick role, bold), left-aligned at the figure edge, running into the empty
    #      left part of (a); the guides of (a) start ROW_LBL_PAD pt after each name (v6: tick labels in a 1.00 in
    #      gutter, two of them on two lines) -----------------------------------------------------------------------
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
        # least gap (pt) from the name's box to the leftmost drawn extent of this row in (a): tint, ring, dot, tick,
        # and the zero line
        T = d["trend"]["bloomz"][key]
        r_ring, r_dot, h_tick = (MS["open"] + MS["open_mew"]) / 2, (MS["dot"] + MS["dot_edge"]) / 2, LW["cal_tick"] / 2
        left = [(min(T["raw"]["ci95"][0], T["after"]["ci95"][0]), 0.0), (T["raw"]["Tr"], r_ring),
                (T["after"]["Tr"], r_dot), (T["s_cal"]["Tr"], h_tick), (0.0, LW["zero"] / 2)]
        gaps = [(a0.transData.transform((x, i))[0] - bb.x1) * 72 / fig.dpi - rr for x, rr in left]
        name_geom[key] = {"text": lab, "x_in": [round(bb.x0 / fig.dpi, 3), round(bb.x1 / fig.dpi, 3)],
                          "guide_starts_at_Tr": round(float(x_end_data), 4),
                          "gap_to_nearest_mark_or_zero_line_pt": round(float(min(gaps)), 2)}
        assert min(gaps) >= NAME_TO_MARK_MIN_PT, (key, gaps)
        assert bb.x0 / fig.dpi >= 0.0 and x_end_data > XLIM_TR[0], (key, bb)

    # ---- row-1 key: ONE line across the top (ring raw, dot after subtraction, tick s_cal, tint 95% interval) --------
    tr = fig.dpi_scale_trans
    dy = 0.026
    x = KEY_X0
    key_items = []

    def key_text(x_txt, s):
        t = S.text(fig, x_txt / FIG_W, YB_KEY / FIG_H, s, role="key", ha="left", va="baseline")
        fig.canvas.draw()
        return t, t.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi
    rr = (MS["open"] + MS["open_mew"]) / 2 / 72
    fig.add_artist(Line2D([x + rr], [YB_KEY + dy], transform=tr, ls="", marker="o", ms=MS["open"], mfc="white",
                          mec=INK, mew=MS["open_mew"]))
    t, x = key_text(x + 2 * rr + KEY_TXT, "raw"); key_items.append(t.get_text())
    x += KEY_SEP
    rd = (MS["dot"] + MS["dot_edge"]) / 2 / 72
    fig.add_artist(Line2D([x + rd], [YB_KEY + dy], transform=tr, ls="", marker="o", ms=MS["dot"], mfc=INK,
                          mec="white", mew=MS["dot_edge"]))
    t, x = key_text(x + 2 * rd + KEY_TXT, "after subtraction"); key_items.append(t.get_text())
    x += KEY_SEP
    fig.add_artist(Line2D([x + 0.01], [YB_KEY + dy], transform=tr, ls="", marker="|", ms=MS["cal_tick"], mec=INK,
                          mfc=INK, mew=LW["cal_tick"]))
    t, x = key_text(x + 0.02 + KEY_TXT, r"$s_{\rm cal}$"); key_items.append(t.get_text())
    x += KEY_SEP
    fig.add_artist(Line2D([x, x + 0.10], [YB_KEY + dy] * 2, transform=tr, color=S.ci_color(C_CKA), lw=LW["ci_bar"],
                          solid_capstyle="butt"))
    t, x = key_text(x + 0.10 + KEY_TXT, "95% interval"); key_items.append(t.get_text())
    key_right_in = round(float(x), 3)
    assert key_right_in <= FIG_W - 0.02, key_right_in

    # ---- shared x labels; y labels of rows 2-3 (label role: bold in v8) --------------------------------------------
    xc = (COL_X[0] + COL_X[2] + AX_W) / 2
    shared_labels = [
        S.text(fig, xc / FIG_W, YB_XLAB1 / FIG_H, r"size trend $\mathrm{Tr}$", role="label", ha="center",
               va="baseline"),
        S.text(fig, xc / FIG_W, YB_XLAB2 / FIG_H, "parameters (B, log scale)", role="label", ha="center",
               va="baseline"),
        S.text(fig, xc / FIG_W, YB_XLAB3 / FIG_H, "relative depth (language model)", role="label", ha="center",
               va="baseline"),
        S.text(fig, YL_X / FIG_W, (R2_Y + R2_H / 2) / FIG_H, "CKA", role="label", ha="center", va="center",
               rotation=90),
        S.text(fig, YL_X / FIG_W, (R3_Y + R3_H / 2) / FIG_H, "baseline", role="label", ha="center", va="center",
               rotation=90)]

    lead = dict(arrowstyle="-", color=INK, lw=LW["leader"], shrinkA=1.0, shrinkB=1.8)
    leaders, row2_labels, row3_labels = {}, {}, {}
    # ---- direct labels of row 2, placed beside their series (no leaders), each once ---------------------------------
    q = P["thr"]["bloomz"]; ad = ax2["bloomz"]
    X_LEFT = q["x"][0] / 1.33 * 1.06                            # left text edge in (d): 1.06x right of the x limit
    row2_labels["observed max"] = S.text(ad, X_LEFT, Y_OBS_LABEL, "observed max", role="direct",
                                         color=S.CRIT_TEXT["cka"], ha="left", va="baseline", zorder=Z["label"])
    s = r"$c_\alpha\;\approx\;$largest baseline"           # v8: thick spaces on both sides (the bold math font sets
    #                                                         the relation tight; v7 '$c_\alpha \approx$largest')
    row2_labels["c_alpha ~ largest baseline"] = S.text(ad, X_LEFT, Y_CA_LABEL, s, role="direct", ha="left",
                                                       va="baseline", zorder=Z["label"])
    q = P["thr"]["llama1"]
    row2_labels["selected pair"] = S.text(ax2["llama1"], q["x"][-1] * 1.33 / 1.04, Y_SEL_LABEL, "selected pair",
                                          role="annot", ha="right", va="baseline", zorder=Z["label"])
    #                                             critic-v8: MUTED regular, as its NULL_LINE dotted reference line

    # ---- row 3: size keys, mKNN, one 'largest baseline' leader (g), shading label (h) ------------------------------
    key_ok = {}
    for j, f in enumerate(FAMS):
        _, ok = size_key(fig, COL_X[j], R3_Y + R3_H + LETTER_DY + 0.030, LMS[f], ramps[f][0])
        key_ok[f] = ok
    ag = ax3["bloomz"]
    lo, hi = P["prof"]["bloomz"][1]
    row3_labels["mKNN"] = S.text(ag, lo - 0.03, MKNN_BASE + MKNN_LABEL_DY, "mKNN", role="direct", color=S.CRIT_TEXT["mknn"],
                                 ha="right", va="bottom", zorder=Z["label"])
    lbl_pts = {}
    for f in FAMS:                                         # the peak of the largest size (recorded for every family)
        prof = P["prof"][f][0]
        top = LMS[f][-1]
        lbl_pts[f] = (prof[top]["argmax_rel"], prof[top]["max_measured"])
    xp, yp = lbl_pts["bloomz"]
    s = "largest baseline"
    an = ag.annotate(s, xy=(xp, yp), xytext=(0.20, 0.222), ha="left", va="bottom", arrowprops=dict(lead),
                     zorder=Z["label"], **S.text_kw("direct", s))
    row3_labels["largest baseline (g)"] = an
    leaders["row3 bloomz"] = aim_leader(fig, an, ag, (xp, yp), (MS["line"] + MS["line_edge"]) / 2, square=True)
    ah = ax3["llama1"]
    lo_h, hi_h = P["prof"]["llama1"][1]
    s = "depths of\nobserved max"
    y_in = 0.262
    an = ah.annotate(s, xy=((lo_h + hi_h) / 2, y_in), xytext=(lo_h - 0.06, y_in), ha="right", va="center",
                     linespacing=0.95, arrowprops=dict(lead, shrinkA=1.0, shrinkB=0.0), zorder=Z["label"],
                     **S.text_kw("annot", s))
    row3_labels["depths of observed max (h)"] = an
    fig.canvas.draw()
    ext = an.arrow_patch.get_window_extent(fig.canvas.get_renderer())
    leaders["row3 shading label (h)"] = {"xy": [float((lo_h + hi_h) / 2), y_in], "band": [float(lo_h), float(hi_h)],
                                         "length_pt": round(float(np.hypot(ext.width, ext.height) * 72 / fig.dpi), 2)}
    assert 8.0 <= leaders["row3 shading label (h)"]["length_pt"] <= 20.0, leaders

    # ---- label clearance from the drawn marks of rows 2 and 3 --------------------------------------------------------
    def row2_marks(f):
        q = P["thr"][f]
        pts = line_samples(q["x"], q["T"], LW["data"] / 2, logx=True) + \
            line_samples(q["x"], q["Bb"], max(LW["data"], LW_PRED) / 2, logx=True) + \
            line_samples(q["x"], q["sel"], LW["ref"] / 2, logx=True)
        pts += [(xx, yy, (MS["open_line"] + MS["open_line_mew"]) / 2) for xx, yy in zip(q["x"], q["T"])]
        pts += [(xx, yy, (MS["line"] + MS["line_edge"]) / 2) for xx, yy in zip(q["x"], q["ca"])]
        return pts

    def row3_marks(f):
        pts = []
        for k, lm in enumerate(LMS[f]):
            c = P["prof"][f][0][lm]
            pts += line_samples(c["rel_depth"], c["predicted_proposition"], profile_lw(k, len(LMS[f])) / 2)
            pts += [(a, b, (MS["profile_dot"] + MS["profile_edge"]) / 2)
                    for a, b in zip(c["rel_depth"], c["measured_permutation_mean"])]
            pts.append((c["argmax_rel"], c["max_measured"], (MS["line"] + MS["line_edge"]) / 2))
        pts += line_samples([-0.03, 1.03], [MKNN_BASE, MKNN_BASE], LW["data"] / 2)
        return pts
    clear = text_vs_marks(fig, {
        "(d) observed max": (row2_labels["observed max"], ax2["bloomz"], row2_marks("bloomz")),
        "(d) c_alpha ~ largest baseline": (row2_labels["c_alpha ~ largest baseline"], ax2["bloomz"],
                                           row2_marks("bloomz")),
        "(e) selected pair": (row2_labels["selected pair"], ax2["llama1"], row2_marks("llama1")),
        "(g) largest baseline": (row3_labels["largest baseline (g)"], ax3["bloomz"], row3_marks("bloomz")),
        "(g) mKNN": (row3_labels["mKNN"], ax3["bloomz"], row3_marks("bloomz")),
        "(h) depths of observed max": (row3_labels["depths of observed max (h)"], ax3["llama1"], row3_marks("llama1"))})
    print("label clearance from marks (pt)", clear)
    assert all(v >= LABEL_MIN_CLEAR_PT for v in clear.values()), clear

    # ---- v8: bold tick labels and axis titles (bold math), then the checks and the save ----------------------------
    n_finalized = S.finalize(fig)
    ov, outside = text_overlaps(fig)
    pdf, png = os.path.join(HERE, "F_aggregation_app.pdf"), os.path.join(HERE, "F_aggregation_app.png")
    v100, views = os.path.join(HERE, "_view_print_100dpi.png"), os.path.join(HERE, "_views_cvd_grey.png")
    fig.savefig(pdf)
    # the layout diagnostic of S.check parses the two-line (h) label as one mathtext string and logs a dummy glyph
    # for its newline; the render splits the lines first, and glyph sizes are asserted from the PDF Tf operators
    class _NoNewlineGlyph(logging.Filter):
        def filter(self, rec):
            return "glyph for '\\n'" not in rec.getMessage()
    flt = _NoNewlineGlyph()
    logging.getLogger("matplotlib.mathtext").addFilter(flt)
    chk = S.check(fig, pdf)
    logging.getLogger("matplotlib.mathtext").removeFilter(flt)
    S.print_view(pdf, png, v100)
    S.views(png, views)
    assert sha(S.__file__) == S.STYLE_SHA256 == STYLE_V8_SHA, "hrh_style_v8 changed"
    print("check", json.dumps(chk, indent=1, default=float))
    print("text overlaps", ov)
    print("text outside figure", outside)
    print("size keys fit", key_ok)
    assert not ov and not outside and all(key_ok.values()), (ov, outside, key_ok)
    print("leaders (shrinkB pt, length pt)", leaders)
    assert leaders["row3 bloomz"][1] >= 4.0, leaders
    # v5.1: s.e. of row 2 (not drawn) and the spacing of the profile dots of row 3, both read off the layout
    pt_per_cka = R2_H * 72 / YB_MAX
    row2_se = {f: {k: float(np.max(P["thr"][f][k])) for k in ("Tse", "case", "selse")} for f in FAMS}
    row2_max_se = max(max(v.values()) for v in row2_se.values())
    marker_outer_half_pt = {"Tse": (MS["open_line"] + MS["open_line_mew"]) / 2,
                            "case": (MS["line"] + MS["line_edge"]) / 2,
                            "selse": (MS["open_aux"] + MS["open_small_mew"]) / 2}
    pt_per_depth = AX_W * 72 / (1.03 - (-0.03))
    dot_d = MS["profile_dot"] + MS["profile_edge"]
    dot_spacing = {}
    for f in FAMS:
        for lm in LMS[f]:
            rd = np.asarray(P["prof"][f][0][lm]["rel_depth"])
            sp = float(np.min(np.diff(np.sort(rd)))) * pt_per_depth
            dot_spacing[lm] = {"model": S.FAM_LABEL[f] + " " + SIZE_LABEL[lm], "n_layers": int(rd.size),
                               "spacing_pt": round(sp, 2), "overlap": bool(sp < dot_d)}
    print("row-2 max s.e.", round(row2_max_se, 4), {f: {k: round(v, 4) for k, v in d_.items()}
                                                    for f, d_ in row2_se.items()})
    print("row-3 dot spacing (outer diameter %.1f pt)" % dot_d, dot_spacing)
    # v5.3: c_alpha square (above) against the selected-pair circle (below), read off the layout
    sq_half, ci_half = marker_outer_half_pt["case"], marker_outer_half_pt["selse"]
    row2_layering = {}
    for f in FAMS:
        q = P["thr"][f]
        row2_layering[f] = []
        for i, lm in enumerate(LMS[f]):
            gap = float(q["ca"][i] - q["sel"][i]) * pt_per_cka
            hid = min(max(sq_half + ci_half - gap, 0.0), 2 * ci_half)
            row2_layering[f].append({"size": SIZE_LABEL[lm], "c_alpha_minus_sel": round(float(q["ca"][i] - q["sel"][i]), 4),
                                     "centre_gap_pt": round(gap, 2), "circle_height_hidden_pt": round(hid, 2),
                                     "circle_height_visible_pt": round(2 * ci_half - hid, 2)})
    print("row-2 squares over circles", row2_layering)

    # ---- numbers.json -----------------------------------------------------------------------------------------------
    def arr(a):
        return [float(v) for v in np.asarray(a).ravel()]

    # v6.1: distances between the peak squares of row 3 (pt, centre to centre), read off the layout
    pt_per_base = R3_H * 72 / YC_MAX
    peak_gaps = {}
    for f in FAMS:
        pk = [(P["prof"][f][0][lm]["argmax_rel"], P["prof"][f][0][lm]["max_measured"]) for lm in LMS[f]]
        g = {}
        for i in range(len(pk)):
            for j_ in range(i + 1, len(pk)):
                dd = np.hypot((pk[i][0] - pk[j_][0]) * pt_per_depth, (pk[i][1] - pk[j_][1]) * pt_per_base)
                g[SIZE_LABEL[LMS[f][i]] + "-" + SIZE_LABEL[LMS[f][j_]]] = round(float(dd), 2)
        peak_gaps[f] = g
    sq_out = MS["line"] + MS["line_edge"]
    close = [(f, k, v) for f, g in peak_gaps.items() for k, v in g.items() if v < sq_out]
    peak_limit = ("row 3 (v6.1): peak squares of different sizes closer than one square (%.1f pt) overlap: %s; "
                  "the squares stack, small sizes below" % (sq_out, "; ".join("%s %s %.2f pt" % c for c in close)))
    print("row-3 peak square gaps (pt)", peak_gaps)
    revision_v6_1 = ["key: fourth line '95%% interval' (CKA tint swatch, as Figure 3 v6); figure 5.00 -> %.2f in"
                     % V6_FIG_H,
                     "(e): one direct label 'threshold c_alpha ~ largest baseline from features' with a leader to the "
                     "13B square; replaces 'largest baseline from features' (e) and 'threshold c_alpha' (f)",
                     "(d-f): threshold of the selected pair in INK (Figure 3 v6); label 'threshold of the selected "
                     "pair'",
                     "(g-i): INK square on the peak of each size (Figure 3 v6); 'largest baseline' leader ends off the "
                     "square of the largest size",
                     "caption: encodings defined (dot = after subtraction of the permutation mean; squares; grey "
                     "line dashed/solid; dotted line); no result numbers"]

    N = {"figure": {"id": "F_aggregation_app", "label": "fig:app-aggregation",
                    "tex_include": "figures/exp_v8/F_aggregation_app.pdf", "size_in": chk["size_in"],
                    "min_text_pt": chk["min_text_pt"], "min_glyph_pt": chk["min_glyph_pt"],
                    "n_type3": chk["n_type3"], "check": chk, "text_overlaps": ov,
                    "text_outside_figure": outside, "pdf_sha256": sha(pdf), "png_sha256": sha(png),
                    "script_sha256": sha(__file__), "ol_data_sha256": sha(OD.__file__),
                    "f1_data_sha256": sha(D.__file__), "f1_llama_data_sha256": sha(LD.__file__),
                    "style_sha256": sha(S.__file__), "style_version": S.STYLE_VERSION,
                    "version": "v8 (hrh_style_v8, Tol palette, bold ticks; layout of Figure 4 v7 with six score rows; no status lines, tags, endpoint numbers, p note or relabelling bands; v5.1 data unchanged)",
                    "revision_v8": REVISION_V8, "row1_name_geometry": name_geom, "key_line_items": key_items,
                    "key_line_right_in": key_right_in, "finalize_texts_changed": n_finalized,
                    "revision_v6_2": ["BLOOMZ largest baseline from features drawn solid as in the other families and in "
                                      "Figure 3 v6.1; post hoc status as MUTED 'post hoc' tags in (d) and (g)",
                                      "threshold of the selected pair: dotted INK line without open circles, as in "
                                      "Figure 3 v6.1 (an INK open circle means 'raw' in the row-1 key)",
                                      "(g) 'depths of observed max' on two lines, so the 'post hoc' tag fits top left",
                                      "caption: grey line solid everywhere, tag named; diagnostics 'squares_over_circles' "
                                      "and row2_se 'selse' describe the v6.1 circles, kept for the record"],
                    "revision_v6_1": revision_v6_1,
                    "v5_1_source": {k: {"path": tilde(os.path.join(V5_DIR, f)), "sha256": h}
                                    for k, (f, h) in V5_1_FILES.items()},
                    "staged_to": STAGE,
                    "consistency_v5": "v5.2: (a-c) Tr axes without minor ticks, as Figure 3 (a-c)",
                    "revision_v5_3": "c_alpha squares above the selected-pair circles in (d-f); key band swatch "
                                     "outlined as in Figure 4 (a) and F.5 (a); caption names the selected pair, the "
                                     "LLaMA-1 30B gap and the mKNN baseline of (g-i)",
                    "style_spec": "build/figs_v5_staging/_research/STYLE_V5_SPEC.md, section 5 'Figure F.3'",
                    "v4_source": "~/hrh_v2_figs/v4/F_aggregation_app/ (untouched)",
                    "v5_elements": {
                        "trend_rows": {"ci_bar_lw": LW["ci_bar"], "ci_tint": {k: S.ci_color(S.CRIT[k])
                                                                              for k in ("cka", "mknn", "ckad")},
                                       "connector_lw": LW["connector"], "raw_ring": [MS["open"], MS["open_mew"]],
                                       "after_dot": [MS["dot"], MS["dot_edge"]],
                                       "s_cal_tick": [MS["cal_tick"], LW["cal_tick"]], "zero_lw": LW["zero"],
                                       "guide": [S.GUIDE, LW["guide"]],
                                       "band": "v8: not drawn (v6: NULL_FILL boxes over the one-sided 5% critical "
                                               "values in (a), with an outlined key swatch)"},
                        "level_panels": {"observed_max": [LW["data"], MS["open_line"], MS["open_line_mew"]],
                                         "largest_baseline": [LW_PRED, C_PRED, "dashed (critic-v8, theory, as Figure 4 "
                                                              "v8; v8 maker and v6.2: %.1f pt NULL_LINE solid)"
                                                              % LW["data"]],
                                         "c_alpha_square": [MS["line"], MS["line_edge"], C_THRESH,
                                                            "critic-v8 CKA red (v8 maker INK)"],
                                         "selected_pair": [LW["ref"], "dotted", C_SEL,
                                                           "critic-v8: NULL_LINE reference as Figure 4 v8 (v8 maker "
                                                           "and v6.2 INK; v6.2: no circles, v6.1 drew open circles "
                                                           "%.1f pt; v5.1 NULL_LINE)" % MS["open_aux"]],
                                         "layering": "v5.3: c_alpha squares (Z data + 0.6) above the selected-pair "
                                                     "circles (Z data + 0.5); v2-v5.2 had the circles on top",
                                         "squares_over_circles": row2_layering,
                                         "error_bars": "not drawn (v5.1); s.e. kept in row2_thresholds, the largest "
                                                       "in the Results text (v6; v5.1 put it in the caption) (row2_se)",
                                         "leader_lw": LW["leader"], "leader_gap_pt": LEADER_GAP,
                                         "leaders_shrinkB_and_length_pt": leaders,
                                         "threshold_of_selected_pair_label": "v8: 'selected pair' in (e), no "
                                                                             "leader, under the dotted line",
                                         "label_clearance_from_marks_pt": clear},
                        "profile_panels": {"line_lw": LW["profile"], "largest_size_lw": LW["profile_emph"],
                                           "dot": [MS["profile_dot"], MS["profile_edge"]],
                                           "mknn_baseline_lw": LW["data"], "shading": S.span_color(C_CKA),
                                           "peak_squares": [MS["line"], MS["line_edge"], C_THRESH,
                                                            "v6.1; critic-v8 CKA red (v8 maker INK)"],
                                           "peak_square_gaps_pt": peak_gaps},
                        "text": {"letters_pt": S.FS["letter"], "bold_roles": "family headers, endpoint numbers, "
                                 "direct labels (observed max, threshold of the selected pair, threshold c_alpha ~ "
                                 "largest baseline from features, mKNN, largest baseline)",
                                 "regular_roles": "axis and tick labels, row labels, status lines, key entries, "
                                 "'expl.', p note, 'depths of observed max'"}},
                    "size_ramps": {f: {"colours": ramps[f][0], "contrast_on_white": ramps[f][1],
                                       "adjacent_dE76": ramps[f][2]} for f in FAMS},
                    "size_ramp_rule_v8c": "v8 c: the ramps of Figure 4 v8 (ramp_3to1: S.FAM_RAMP light end darkened "
                                          "to >= 3.0:1 along its CIELAB path, even L* steps), asserted equal to %s; "
                                          "critic-v8 and before: the rule below" % tilde(FIG4_V8_NUMBERS),
                    "size_ramp_rule": "S.FAM_RAMP of hrh_style_v5: even CIELAB L* steps, light = small, lightest "
                                      ">= 2.6:1 on white; the largest size's prediction line at S.LW['profile_emph'] "
                                      "and drawn on top",
                    "connector_rule": "connector raw -> after subtraction in every row (no arrowheads, %.1f pt), as "
                                      "in Figure 3; s_cal tick %.1f pt tall with edge %.1f pt, drawn above both circles"
                                      % (LW["connector"], MS["cal_tick"], LW["cal_tick"]),
                    "headers": {f: SIZES_TXT[f] + ", " + STATUS[f] for f in FAMS},
                    "status": {"bloomz": "panel (a): replication with reading rules fixed in a frozen plan (tab:exp-verdicts "
                                         "row 'WIT, layer max'); calibrate-then-max and debiased CKA rows EXPLORATORY; "
                                         "panels (d) and (g) post hoc readings of the frozen run; tagged 'post hoc' in (d) and (g) (v6.2)",
                               "llama1": "Part B predictions frozen from features before any cross-model score",
                               "openllama": "every outcome descriptive (plan R.1.2); B-bar frozen before any 13B score"},
                    "x_ticks_row2": "log scale; tick labels are the nominal model names placed at the actual parameter "
                                    "counts (x_params_B)",
                    "ci_note_row1": "opaque tinted bars (S.ci_color, %.2f white; v4 drew them at alpha 0.35) = 95%% "
                                    "bootstrap intervals over the 17 vision models (frozen CIs; BLOOMZ "
                                    "calibrate-then-max: recomputed, B = 4000, seed 20261009)" % S.CI_TINT,
                    "subscripts": "mathtext SHRINK_FACTOR 7/8 set by S.use(); texts with a subscript at 8 pt, so "
                                  "subscripts are 7.0 pt (checked from the PDF Tf operators by S.check)",
                    "row2_se": {"max": round(row2_max_se, 4), "max_by_family_and_series": row2_se,
                                "pt_per_cka_unit": round(pt_per_cka, 2),
                                "marker_outer_half_pt": marker_outer_half_pt,
                                "note": "Tse = observed max (ring), case = c_alpha (square), selse = threshold of "
                                        "the selected pair (small ring); at 1.0 pt the bars hid behind the "
                                        "rings and almost every square, so v5.1 draws none"},
                    "row3_dot_spacing": {"outer_diameter_pt": dot_d, "pt_per_rel_depth": round(pt_per_depth, 2),
                                         "by_model": dot_spacing},
                    "known_limits": ["row 3: the permutation-mean dots (%.1f pt with a %.1f pt white edge, the "
                                     "module size; v4 drew 1.9 pt for BLOOMZ and 1.7 pt for the others) overlap "
                                     "on the models with 41 or more layers (%s), whose neighbouring dots are "
                                     "%.1f-%.1f pt apart, and there cover the proposition lines (relative error "
                                     "<= 4.4e-4) except on steep stretches; on the 33-layer models (%s) the "
                                     "dots are %.1f pt apart and nearly touch; the BLOOMZ lines (25 and 31 "
                                     "layers) stay visible between the dots"
                                     % (MS["profile_dot"], MS["profile_edge"],
                                        ", ".join(v["model"] for v in dot_spacing.values() if v["n_layers"] >= 41),
                                        min(v["spacing_pt"] for v in dot_spacing.values() if v["n_layers"] >= 41),
                                        max(v["spacing_pt"] for v in dot_spacing.values() if v["n_layers"] >= 41),
                                        ", ".join(v["model"] for v in dot_spacing.values() if v["n_layers"] == 33),
                                        min(v["spacing_pt"] for v in dot_spacing.values() if v["n_layers"] == 33)),
                                    "row 2: at LLaMA-1 30B the threshold of the selected pair lies %.4f (%.2f pt) "
                                    "below c_alpha, so the c_alpha square covers its circle but for %.2f pt; the "
                                    "Results text gives this gap (v6; v5.1: the caption)" % (row2_layering["llama1"][-1]["c_alpha_minus_sel"],
                                                                row2_layering["llama1"][-1]["centre_gap_pt"],
                                                                row2_layering["llama1"][-1]["circle_height_visible_pt"]),
                                    peak_limit]},
         "inputs": d["inputs"], "vits": d["vits"], "row1_trends": {}, "row2_thresholds": {}, "row3_profiles": {},
         "moved_to_text": {}}
    for j, f in enumerate(FAMS):
        N["row1_trends"][f] = {"panel": "abc"[j], "rows": P["trend"][f],
                               "band": ("v8: not drawn; band_5pct values kept under after_subtraction (v6: grey 5% "
                                        "relabelling band drawn)") if f == "bloomz" else "no band (6 relabellings)"}
        q = P["thr"][f]; Th = d["thr"][f]
        N["row2_thresholds"][f] = {
            "panel": "def"[j], "lms": LMS[f], "x_params_B": arr(q["x"]), "x_tick_labels": XLAB[f],
            "observed_layermax_cka_T": {"mean": arr(q["T"]), "se": arr(q["Tse"])},
            "c_alpha_of_max": {"mean": arr(q["ca"]), "se": arr(q["case"])},
            "largest_baseline_Bbar_from_features": {"mean": arr(q["Bb"]), "style": (
                "INK dashed (critic-v8; v8 maker solid NULL_LINE); the BLOOMZ reading is post hoc (Results text; "
                "v8 draws no tag)"
                if f == "bloomz" else Th["bbar_style"])},
            "threshold_of_selected_pair": {"mean": arr(q["sel"]), "se": arr(q["selse"])},
            "rel_depth_of_argmax_LM_mean": arr([r["mean"] for r in Th["rel_depth_argmax"]]),
            "offset_calpha_minus_Bbar": Th["offset_calpha_minus_Bbar"], "sources": Th["src"]}
        prof, band = P["prof"][f]
        N["row3_profiles"][f] = {"panel": "ghi"[j], "profiles": prof, "shading_observed_argmax_rel_depth": band,
                                 "shading_colour": S.span_color(C_CKA),
                                 "n_pairs_shading": len(d["prof"][f]["sel"]), "mknn_baseline_k_over_m_minus_1": MKNN_BASE,
                                 "largest_baseline_leader_to": {"lm": LMS[f][-1], "rel_depth": lbl_pts[f][0],
                                                                "value": lbl_pts[f][1]},
                                 "peak_squares": {lm: [prof[lm]["argmax_rel"], prof[lm]["max_measured"]]
                                                  for lm in LMS[f]},
                                 "meta": d["prof"][f]["meta"]}
    # numbers the v5.1 caption quoted; v6 moves them to the Results text (same computation, renamed)
    Q = N["moved_to_text"]
    for f in FAMS:
        r = d["trend"][f]
        q = P["thr"][f]
        prof = P["prof"][f][0]
        mk = max(abs(r["mknn|full"][nm]["Tr"] - r["mknn|full"]["raw"]["Tr"]) for nm in ("after", "s_cal"))
        db = max(abs(r["cka_debiased|full"][nm]["Tr"] - r["cka_debiased|full"]["raw"]["Tr"]) for nm in ("after", "s_cal"))
        Q[f] = {"cka_layermax_Tr": {nm: round(r["cka_biased|full"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
                "cka_layermax_p": {nm: r["cka_biased|full"][nm]["p_one_sided_exact"] for nm in ("raw", "after", "s_cal")},
                "cka_ctm_Tr": {nm: round(r["cka_biased|ctm"][nm]["Tr"], 3) for nm in ("after", "s_cal")},
                "cka_fixed_r1_Tr": {nm: round(r["cka_biased|fixed_r1.0"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
                "cka_fixed_r05_Tr": {nm: round(r["cka_biased|fixed_r0.5"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
                "mknn_Tr": {nm: round(r["mknn|full"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
                "mknn_max_abs_change": round(mk, 4),
                "debiased_Tr": {nm: round(r["cka_debiased|full"][nm]["Tr"], 3) for nm in ("raw", "after", "s_cal")},
                "debiased_max_abs_change": round(db, 4),
                "rise_smallest_to_largest": {"c_alpha": round(float(q["ca"][-1] - q["ca"][0]), 4),
                                             "observed_T": round(float(q["T"][-1] - q["T"][0]), 4),
                                             "Bbar": round(float(q["Bb"][-1] - q["Bb"][0]), 4)},
                "sel_pair_threshold_range": [round(float(q["sel"].min()), 4), round(float(q["sel"].max()), 4)],
                "c_alpha_minus_sel_pair_at_largest": round(float(q["ca"][-1] - q["sel"][-1]), 4),
                "offset_calpha_minus_Bbar": d["thr"][f]["offset_calpha_minus_Bbar"],
                "largest_baseline_rel_depth_by_size": {lm: round(prof[lm]["argmax_rel"], 4) for lm in LMS[f]},
                "observed_argmax_rel_depth_range": [round(v, 4) for v in P["prof"][f][1]],
                "max_abs_rel_err_measured_vs_prop": max(prof[lm]["max_abs_rel_err"] for lm in LMS[f])}
    Q["openllama"]["P-O2a(ii)_order"] = d["thr"]["openllama"]["offset_calpha_minus_Bbar"]["PO2aii"]
    Q["pooled"] = {"max_abs_change_mknn_all_families": max(Q[f]["mknn_max_abs_change"] for f in FAMS),
                   "max_abs_change_debiased_all_families": max(Q[f]["debiased_max_abs_change"] for f in FAMS),
                   "max_rel_err_all_families": max(Q[f]["max_abs_rel_err_measured_vs_prop"] for f in FAMS),
                   "row2_max_se_not_drawn": round(row2_max_se, 4),
                   "mknn_baseline_k_over_m_minus_1": {"value": round(MKNN_BASE, 4), "k": 10, "m": M,
                                                      "source": "Proposition 4 (E_pi s_N = k/(m-1)); drawn as the "
                                                                "blue line of (g-i), row3_profiles.<fam>."
                                                                "mknn_baseline_k_over_m_minus_1"},
                   "min_layers_where_dots_overlap": min(v["n_layers"] for v in dot_spacing.values() if v["overlap"]),
                   "bloomz_band_5pct_layermax_after": d["trend"]["bloomz"]["cka_biased|full"]["after"]["band_5pct"]}
    # ---- v6: on-panel numbers, numbers for the Results text, number-free caption, identity with v5.1 --------------
    N["on_panel_numbers_v6_removed"] = on_panel_numbers(P, d)
    N["on_panel_numbers"] = {"note": "v8: no result number is drawn (tick labels and the size keys only); the v6 "
                                     "on-panel numbers are listed under on_panel_numbers_v6_removed with sources"}
    N["results_text_numbers"] = results_text_numbers(d, P, Q, N)
    N["caption_v6_tex"] = CAPTION_V6
    cap = os.path.join(HERE, "caption_v8.tex")
    txt = open(cap).read().strip()
    N["caption_v8"] = {"tex": txt, "sha256": sha(cap), "rule": "bold claim, decoding only, no result numbers",
                       "digit_check": caption_digits(txt),
                       "checks": caption_v8_check(txt, max_words=100, allowed_digits=tuple(ALLOWED_CAPTION_NUMBERS))}
    # critic-v8: the caption names the columns (no family header is drawn) and decodes the dashed and dotted lines.
    # Family-names pass: the headers are drawn again and the caption no longer names the columns (assert unchanged)
    assert SHOW_FAMILY_HEADERS or all(S.FAM_LABEL[f] in txt for f in FAMS), "caption must name the families"
    assert "dashed lines" in txt and "dotted lines" in txt and "grey line" not in txt, txt
    N["identity_vs_v5_1"] = identity_vs_v51(N, pdf, png)
    V8 = v8_asserts(fig, chk, ax1, ax2, ax3, row_names, row2_labels, row3_labels, shared_labels, d, P)
    N["asserts_v8"] = V8
    N["texts_with_digits_besides_ticks"] = list(V8_TEXTS_WITH_DIGITS)
    assert all(V8.values()), ({k: v for k, v in V8.items() if not v}, V8_TEXTS_WITH_DIGITS)
    # v8 c (consistency checker): (g)-(i) in the shades of Figure 4 v8
    f4r = json.load(open(FIG4_V8_NUMBERS))["figure"]["size_ramps"]
    CONS_C = {
        "(g)-(i) size ramps equal those of Figure 4 v8 (its numbers.json, read only)": all(
            [c.upper() for c in ramps[f][0]] == [c.upper() for c in f4r[f]["colours"]] for f in FAMS),
        "every ramp step >= 3:1 on white": all(min(ramps[f][1]) >= 3.0 for f in FAMS),
        "profile lines and size keys drawn in the ramp colours": all(
            P["prof"][f][0][lm]["colour"].upper() == ramps[f][0][k].upper() for f in FAMS for k, lm in enumerate(LMS[f])),
        "darkest step unchanged (= the darkest step of S.FAM_RAMP)": all(ramps[f][0][-1].upper() == S.ramp(f)[-1].upper()
                                                                          for f in FAMS),
        "data values unchanged (identity_vs_v51 asserted above)": N["identity_vs_v5_1"]["numbers_vs_v5_1"]["max_abs_diff"]
        <= 1e-12,
    }
    assert all(CONS_C.values()), CONS_C
    N["consistency_v8c"] = {"date": "2026-10-10", "role": "consistency checker across the v8 figures",
                            "changes": ["(g)-(i) size ramps S.FAM_RAMP -> ramp_3to1 of Figure 4 v8"],
                            "figure4_numbers": {"path": tilde(FIG4_V8_NUMBERS), "sha256": sha(FIG4_V8_NUMBERS)},
                            "ramps_before": {f: S.ramp(f) for f in FAMS}, "ramps_now": {f: ramps[f][0] for f in FAMS},
                            "asserts": CONS_C, "critic_v8_files": "./_prev_consistency/"}
    N["outputs"] = {p_: sha(p_) for p_ in (pdf, png, v100, views)}
    json.dump(N, open(os.path.join(HERE, "numbers.json"), "w"), indent=1, default=float)
    print("identity vs v5.1", json.dumps(N["identity_vs_v5_1"], indent=1))
    print("caption digit check", json.dumps(N["caption_v8"]["digit_check"]))
    print("caption v8 checks", json.dumps(N["caption_v8"]["checks"], indent=1))
    print("asserts_v8", json.dumps(V8, indent=1))
    print("results_text_numbers", len(N["results_text_numbers"]), [r["id"] for r in N["results_text_numbers"]])



if __name__ == "__main__":
    main()
