"""Shared figure style v8 for the LRH paper (arxiv_v4.tex figures, main and appendix). Frozen (chmod 444) 2026-10-10; makers assert its sha256.

v8 (2026-10-10, role style designer). Made from a copy of hrh_style_v5_1.py (sha256 97d0ed53...f41a6, frozen, untouched).
Every v5.1 name is still defined here, so a v7 or v6 maker imports this module after changing only the import line and
the asserted sha256. What changed, and why (the author on v7 Figure 6, 2026-10-10):

  1. ONE palette system: Paul Tol's colour schemes (Tol 2021, "Colour schemes", SRON/EPS/TN/09-002; the schemes
     bright, vibrant, muted, high-contrast, medium-contrast, light, dark and pale, and the sequential YlOrBr). Every
     anchor colour below is a Tol hex (the v5 neutrals aside). Derived colours (text tones, size ramps, interval tints) keep a Tol hue and only
     change CIELAB L* or the share of white. v5.1 mixed Okabe-Ito (criteria) with Tol (families, rules); the Okabe-Ito
     criterion hues move to their Tol vibrant twins (mKNN dE2000 2.0, MST 8.0), and CKA moves from vermilion to Tol
     vibrant red (see 3). Comparison (computed, CIEDE2000 under Machado 2009 full-severity CVD): the journal palettes
     of ggsci (NPG, AAAS, NEJM, Lancet) and ColorBrewer Set1/Dark2 and Tableau 10 all have a pair below dE 5 under
     deutan or protan; Okabe-Ito has only 7 hues; Tol's schemes are the only large CVD-tested family. The comparison
     table is in palette_v8_check.json.
  2. NO black-and-white figures. Neutral grey is reserved for within / null / chance references (NULL_LINE,
     NULL_FILL); INK is reserved for theory (predictions, bounds, links of a theorem), frames and text. Every data
     series gets a hue: GD is Tol dark blue, the balanced initialization Tol medium-contrast light blue, the across-
     objective comparison of Figure 6 Tol vibrant orange, the positive control Tol bright blue, the 2-D densities
     Tol YlOrBr, and the decision fills of F.6 (a) Tol pale green / high-contrast yellow / high-contrast red.
  3. CKA = Tol vibrant red #CC3311 (was Okabe-Ito vermilion #D55E00). Reasons: Tol vibrant orange #EE7733 merges with
     OpenLLaMA olive under deutan/protan (dE 4.8 / 4.1; both sit in every column of Figure 4), red does not (10.6);
     red reaches 5.19:1 on white, so CKA text needs no darker tone; and it removes the clash with the orange that
     Figure 1 uses for ARH (dE 6.6 -> 17.4).
  4. BOLD tick labels and axis titles in every figure, with bold math: FW['tick'], FW['label'] and FW['label_small']
     are bold; axes.labelweight is bold; every bold text that holds math is set with math_fontfamily='custom', which
     use() maps to Liberation Serif Bold / Bold Italic (so 2^{20}, 10^{-2}, $N$, $s_N$ are bold and match the text).
     Regular-weight text keeps Computer Modern math. Call finalize(fig) after the last plotting call and before
     savefig: it bolds every tick label and axis title (matplotlib has no rcParam for tick-label weight), applies the
     bold math, and sets tick labels holding a script to FS['math'] = 8 pt (so scripts stay >= 7 pt).
  5. check() asserts, in addition to v5.1: every visible tick label and axis title is bold; every bold text holding
     math uses the bold math family; the figure has at least one chromatic data colour (CIELAB C* >= 15). It reports
     data colours outside the palette (informational).
  6. New helpers: views() writes the normal / deuteranopia / protanopia / greyscale views of a rendered PNG in one
     image; print_view() renders the PDF at 100 dpi (print size) and 200 dpi; ramp_n() makes an even-L* size ramp of
     any anchor; cmap_density() and seq_cmap() give sequential colormaps; simulate() gives the CVD / greyscale
     version of one colour.

Semantic colour map (one hex = one meaning; SEM below is the flat version with the figures that use each entry):
  criteria      mKNN #0077BB  CKA #CC3311  MST #009988  debiased CKA #EE3377  distance-aware #CCBB44 (unused)
  LM families   BLOOMZ #332288  LLaMA-1 #882255  OpenLLaMA #999933  Pythia #662506
  pair classes  seed #77AADD  vision #0077BB  vision-language #004488  vision-sentence-encoder #994455
                (seed/vision/vision-language are the light/mid/dark steps of the mKNN hue: every pair-class curve is
                mKNN, so the hue says mKNN and the lightness says the class; vision shares the mKNN hex on purpose)
  rules         GD (default init., any step) #222255  balanced init. #6699CC  small init. #AA4499
                weight decay #44AA99  tied weights #117733  SGD #CC6677  Adam #663333
  link          across objectives (InfoNCE vs spectral) #EE7733  positive control (exact link) #4477AA
                within one objective -> neutral (NULL_FILL band, NULL_LINE edge); 2-D density -> Tol YlOrBr
  theory        INK #222222 (dashed for predictions and links, solid for bounds)
  references    NULL_LINE #707070 (chance, null, y = x, tolerance), NULL_FILL #E5E5E5 (null bands, excluded regions)
  decisions     reliable #CCDDAA  inconclusive #DDAA33  unreliable #BB5566 (white bold text)

v5.1 / v5 header follows (everything below is v5.1 unless marked v8).

Spec: build/figs_v5_staging/_research/STYLE_V5_SPEC.md (style-research role, 2026-10-09). This file implements it.

Use:
    import hrh_style_v8 as S
    S.use()                      # rcParams, bold-math fonts, and mathtext scripts at 7/8 of their base
    ...                          # plot; colours only from S.SEM / S.CRIT / S.FAM / S.PAIR / S.ARM / S.LINK / neutrals
    S.finalize(fig)              # v8: bold ticks and axis titles, bold math, 8 pt for tick labels with scripts
    fig.savefig(pdf); out = S.check(fig, pdf)   # asserts the print-size and v8 rules; put `out` into numbers.json
    S.print_view(pdf, png200, png100); S.views(png200, views_png)   # look at print size, CVD and greyscale

Rules the module owns (makers import them; a number left in a maker must be geometry, not style):
  * LW  stroke widths in pt per element class        * MS  marker sizes in pt (scatter areas in pt^2)
  * FS  text sizes in pt per text role               * FW  weights per role ("normal" or "bold" only)
  * FC  default text colour per role                 * Z   z-order per layer        * LS  line styles
  * palette: CRIT, FAM, PAIR, ARM, LINK (line colours) and CRIT_TEXT, FAM_TEXT, PAIR_TEXT, ARM_TEXT, LINK_TEXT (text
    tones, >= 4.5:1)
Fonts: Liberation Serif (TrueType, Times metrics; Regular, Bold, Bold Italic) for text and for bold math, mathtext 'cm'
for regular math; pdf.fonttype 42, so every font is embedded as TrueType (CID TrueType) and no Type 3 font appears.
"""
import hashlib
import os
import re
import zlib

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import to_hex, to_rgb, to_rgba

STYLE_VERSION = "v8"                    # v8: Tol system, colour for every data series, bold ticks/axis titles/math
TEXTWIDTH_IN = 5.5                      # neurips_2024: \textwidth = 5.5 in

# ==================================================================================================== palette (v8)
# ---- the source system: Paul Tol's schemes (exact hexes; the only anchors this module uses) -------------------------
TOL = {
    "bright":  {"blue": "#4477AA", "cyan": "#66CCEE", "green": "#228833", "yellow": "#CCBB44", "red": "#EE6677",
                "purple": "#AA3377", "grey": "#BBBBBB"},
    "vibrant": {"blue": "#0077BB", "cyan": "#33BBEE", "teal": "#009988", "orange": "#EE7733", "red": "#CC3311",
                "magenta": "#EE3377", "grey": "#BBBBBB"},
    "muted":   {"rose": "#CC6677", "indigo": "#332288", "sand": "#DDCC77", "green": "#117733", "cyan": "#88CCEE",
                "wine": "#882255", "teal": "#44AA99", "olive": "#999933", "purple": "#AA4499", "pale_grey": "#DDDDDD"},
    "high_contrast": {"blue": "#004488", "yellow": "#DDAA33", "red": "#BB5566"},
    "medium_contrast": {"light_blue": "#6699CC", "dark_blue": "#004488", "light_yellow": "#EECC66",
                        "dark_red": "#994455", "dark_yellow": "#997700", "light_red": "#EE99AA"},
    "light":   {"light_blue": "#77AADD", "orange": "#EE8866", "light_yellow": "#EEDD88", "pink": "#FFAABB",
                "light_cyan": "#99DDFF", "mint": "#44BB99", "pear": "#BBCC33", "olive": "#AAAA00"},
    "dark":    {"blue": "#222255", "cyan": "#225555", "green": "#225522", "yellow": "#666633", "red": "#663333"},
    "pale":    {"blue": "#BBCCEE", "cyan": "#CCEEFF", "green": "#CCDDAA", "yellow": "#EEEECC", "red": "#FFCCCC"},
    # sequential YlOrBr (Tol's recommended scheme for ordered data; monotone in lightness, CVD-safe)
    "YlOrBr":  ["#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"],
}

# ---- criteria (Tol vibrant) ------------------------------------------------------------------------------------
CRIT = {
    "mknn": "#0077BB",   # vibrant blue    (v5.1 Okabe-Ito #0072B2; dE2000 2.0) neighbor sets: mKNN s_N(k), 1-NN s_{N,1}
    "mst":  "#009988",   # vibrant teal    (v5.1 #009E73; dE 8.0) spanning trees: MST overlap s_T, bridges
    "cka":  "#CC3311",   # vibrant red     (v5.1 vermilion #D55E00; header item 3) biased linear CKA
    "ckad": "#EE3377",   # vibrant magenta (v5.1 #CC79A7) debiased CKA (exploratory)
    "dist": "#CCBB44",   # bright yellow   (v5.1 #E69F00) distance-aware criteria when a hue is needed; unused in v7/v6
}
# ---- language-model families (Tol muted dark trio; Pythia = the darkest Tol YlOrBr step, a dark brown) ----------
# Pythia #662506 keeps dE2000 >= 14.2 from every colour of Figure 4 / F.4 in normal, deutan and protan views (Tol dark
# green #225522 would fall to 6.2 against CKA, Tol dark cyan #225555 to 6.0 against LLaMA-1). cmap_density() stops
# before this step, so the hex keeps one meaning.
FAM = {"bloomz": "#332288", "llama1": "#882255", "openllama": "#999933", "pythia": "#662506"}
FAM_LABEL = {"bloomz": "BLOOMZ", "llama1": "LLaMA-1", "openllama": "OpenLLaMA", "pythia": "Pythia"}
FAM_MARKER = {"bloomz": "o", "llama1": "s", "openllama": "D", "pythia": "^"}
# ---- pair classes (every curve is mKNN): light / mid / dark steps of the mKNN hue, all Tol hexes ----------------
# seed = Tol light blue, vision = Tol vibrant blue (= mKNN, on purpose), vision-language = Tol high-contrast blue.
# L* 67.9 / 48.0 / 29.2 (even steps). The sentence-encoder class (F.2) is a mKNN class with its own Tol hue.
PAIR = {"seed": "#77AADD", "vision": "#0077BB", "vl": "#004488", "sent": "#994455"}
PAIR_LABEL = {"seed": "seed", "vision": "vision", "vl": "vision–language", "sent": "sentence encoders"}
# ---- training rules (learning figures only) ---------------------------------------------------------------------
# gd = Tol dark blue (all GD runs from the default initialization; 'nowd', 'default' and 'gd_large' are aliases: the
# large step is told apart by an open marker and a dashed line, not by colour); balanced = Tol medium-contrast light
# blue (a lighter step of the GD blue: the same optimizer from a balanced start); small init., weight decay, tied
# weights, SGD, Adam unchanged from v5.1. Worst pair inside Figure 5 / F.5 / F.7 over normal, deutan and protan:
# Adam vs INK 9.7 (protan; Adam bars against theory triangles, shapes differ), then 12.7.
ARM = {"gd": "#222255", "nowd": "#222255", "default": "#222255", "gd_large": "#222255", "balanced": "#6699CC",
       "smallinit": "#AA4499", "wd": "#44AA99", "tied": "#117733", "sgd": "#CC6677", "adam": "#663333"}
ARM_LABEL = {"gd": "GD", "nowd": "no weight decay", "default": "default init.", "gd_large": "GD, large step",
             "balanced": "balanced init.", "smallinit": "small init.", "wd": "weight decay", "tied": "tied weights",
             "sgd": "SGD", "adam": "Adam"}
# ---- contrastive link (Figure 6, F.8) ---------------------------------------------------------------------------
# across = InfoNCE against spectral (the main data; also 'link undone', with an open marker, and InfoNCE at the other
# temperature against spectral, with another marker); control = the positive control (InfoNCE against the exact link
# of Theorem 3 applied to its own Gram matrix; 'link check, self / peer' in F.8, open / filled). Within one objective
# is a reference: NULL_FILL band with a NULL_LINE edge. The 2-D densities of kernel entries use cmap_density() (YlOrBr).
LINK = {"across": "#EE7733", "control": "#4477AA"}
LINK_LABEL = {"across": "InfoNCE vs spectral", "control": "exact link (positive control)",
              "within": "within one objective"}

# ---- text tones: same hue, CIELAB L* lowered until the colour reaches 4.5:1 on white ----------------------------
CRIT_TEXT = {"mknn": "#0077BB", "mst": "#008575", "cka": "#CC3311", "ckad": "#E1226D", "dist": "#837800"}
FAM_TEXT = {"bloomz": "#332288", "llama1": "#882255", "openllama": "#6B6B24", "pythia": "#662506"}
ARM_TEXT = {"gd": "#222255", "nowd": "#222255", "default": "#222255", "gd_large": "#222255", "balanced": "#447AAB",
            "smallinit": "#AA4499", "wd": "#108576", "tied": "#117733", "sgd": "#BA5668", "adam": "#663333"}
LINK_TEXT = {"across": "#C4550E", "control": "#4477AA"}
INK = "#222222"          # 15.9:1 on white: text, theory lines, frames
# Seed: the darkest seed tone with 4.5:1 (#447AAA) lies within dE2000 ~6 of the vision blue, so the seed name is INK,
# placed where only the seed bundle is (v5 rule, kept).
PAIR_TEXT = {"seed": INK, "vision": "#0077BB", "vl": "#004488", "sent": "#994455"}

# ---- neutrals (unchanged; references only, never the main data colour) -------------------------------------------
MUTED = "#555555"        # 7.5:1; all secondary annotation text (the lightest text allowed)
NULL_LINE = "#707070"    # 4.9:1; chance, nulls, y = x, tolerance and reference lines; never text
NULL_FILL = "#E5E5E5"    # null bands, within-objective bands, excluded region, relabelling band
REF_FILL = "#F2F2F2"     # flat bands (F.1 d), column bands (Figure 5, F.7)
GUIDE = "#DDDDDD"        # forest-row separators
WHITE = "#FFFFFF"
THEORY = INK             # v8: name for the theory colour (predictions, bounds, links of a theorem)
WITHIN = {"line": NULL_LINE, "fill": NULL_FILL}
# ---- decision fills (F.6 a), v8: Tol pale green / high-contrast yellow / high-contrast red, ordered in L* ---------
# (85.7 / 72.5 / 49.5, so the order survives greyscale); text INK, INK, white bold (10.98, 7.48, 4.56:1).
DECISION = {"reliable": "#CCDDAA", "inconclusive": "#DDAA33", "unreliable": "#BB5566"}
DECISION_TEXT = {"reliable": INK, "inconclusive": INK, "unreliable": WHITE}     # white is bold (FW['cell_dark'])
DECISION_V5 = {"reliable": "#EDEDED", "inconclusive": "#A8A8A8", "unreliable": "#3B3B3B"}   # v5.1 greys (history)

# ---- ordinal size ramps (light = small): even steps in CIELAB L* ---------------------------------------------------
FAM_RAMP = {"bloomz": ["#A199C8", "#887AB8", "#6E5CA8", "#523F98", "#332288"],      # 0.56B -> 7.1B (v5)
            "llama1": ["#C390AA", "#A75C7E", "#882255"],                              # 7B -> 30B (v5)
            "openllama": ["#A4A449", "#7E7E33", "#5A5A1E"],                           # 3B -> 13B (v5)
            # v8, provisional (no Pythia figure yet): 70M -> 6.9B = ramp_n('#662506', 7, 2.3); seven steps give
            # adjacent dE2000 5.9-7.5 only, so label the end sizes directly (or show a subset with ramp_n(..., n)).
            "pythia": ["#D1A18C", "#C08B74", "#AE765D", "#9C6147", "#8B4D31", "#78391C", "#662506"]}

# ---- one flat semantic map: meaning -> hex (v8). SEM_USE lists the figures that use each meaning. -----------------
SEM = {
    "criterion.mKNN": CRIT["mknn"], "criterion.CKA": CRIT["cka"], "criterion.MST": CRIT["mst"],
    "criterion.debiased_CKA": CRIT["ckad"], "criterion.distance_aware": CRIT["dist"],
    "family.BLOOMZ": FAM["bloomz"], "family.LLaMA-1": FAM["llama1"], "family.OpenLLaMA": FAM["openllama"],
    "family.Pythia": FAM["pythia"],
    "pair.seed": PAIR["seed"], "pair.vision": PAIR["vision"], "pair.vision-language": PAIR["vl"],
    "pair.vision-sentence_encoder": PAIR["sent"],
    "rule.GD": ARM["gd"], "rule.balanced_init": ARM["balanced"], "rule.small_init": ARM["smallinit"],
    "rule.weight_decay": ARM["wd"], "rule.tied_weights": ARM["tied"], "rule.SGD": ARM["sgd"], "rule.Adam": ARM["adam"],
    "link.across_objectives": LINK["across"], "link.positive_control": LINK["control"],
    "theory": INK, "reference.within_null_chance.line": NULL_LINE, "reference.within_null_chance.fill": NULL_FILL,
    "decision.reliable": DECISION["reliable"], "decision.inconclusive": DECISION["inconclusive"],
    "decision.unreliable": DECISION["unreliable"],
}
SEM_USE = {
    "criterion.mKNN": "Fig 3 (b-c), Fig 4 (a-c forest, g-i baseline), F.4; hue of the pair ramp in Fig 2, F.1, F.2",
    "criterion.CKA": "Fig 3 (a-b), Fig 4, F.4 (intervals: ci_color; depth spans: span_color)",
    "criterion.MST": "Fig 2 (c), F.3",
    "criterion.debiased_CKA": "F.4 (a-c)",
    "criterion.distance_aware": "none in v7/v6 (reserved)",
    "family.BLOOMZ": "Fig 4 (a, d, g), F.4; x-label word in Fig 3 and F.3 (c)",
    "family.LLaMA-1": "Fig 4 (b, e, h), F.4", "family.OpenLLaMA": "Fig 4 (c, f, i), F.4",
    "family.Pythia": "Pythia-axis figure or column (not yet drawn)",
    "pair.seed": "Fig 2 (a-b), F.1, F.2 (a)", "pair.vision": "Fig 2 (a-b), F.1, F.2 (a)",
    "pair.vision-language": "Fig 2 (a-b), F.1, F.2 (language-model text side)",
    "pair.vision-sentence_encoder": "F.2",
    "rule.GD": "Fig 5 (all panels; default init.), F.5, F.6 (no weight decay), F.7",
    "rule.balanced_init": "Fig 5 (d-e), F.5 (a-b)",
    "rule.small_init": "Fig 5, F.5, F.6, F.7", "rule.weight_decay": "Fig 5, F.5, F.6, F.7",
    "rule.tied_weights": "Fig 5 (a-c), F.7", "rule.SGD": "F.7", "rule.Adam": "F.7",
    "link.across_objectives": "Fig 6 (a, c, d, f; densities b, e via cmap_density), F.8",
    "link.positive_control": "Fig 6 (d, f), F.8 (link check, self / peer)",
    "theory": "Fig 2 (c) bound, Fig 3 (a) monotone link, Fig 5 (c) predicted, Fig 6 predictions and link, F.1 crossing "
              "law and N1, F.3 bound, F.7 (e) predicted",
    "reference.within_null_chance.line": "chance, null, y = x, tolerance lines in all figures",
    "reference.within_null_chance.fill": "Fig 2 (c) excluded, Fig 6 within band, F.3 null band, F.4 relabelling null",
    "decision.reliable": "F.6 (a)", "decision.inconclusive": "F.6 (a)", "decision.unreliable": "F.6 (a)",
}
# Groups that share a figure. make_palette_sheet_v8.py asserts dE2000 >= 9.5 for every pair inside a group over normal,
# deutan and protan (worst: Adam vs INK 9.7, CKA vs OpenLLaMA 10.6); the numbers are in palette_v8_check.json.
CO_OCCUR = {
    "Fig 3": ["cka", "mknn", "bloomz"],
    "Fig 4 / F.4 (+Pythia)": ["cka", "mknn", "ckad", "bloomz", "llama1", "openllama", "pythia"],
    "Fig 2 / F.1 / F.2": ["seed", "vision", "vl", "sent", "mst"],
    "Fig 5 / F.5 / F.6 / F.7": ["gd", "balanced", "smallinit", "wd", "tied", "sgd", "adam", "INK"],
    "Fig 6 / F.8": ["across", "control", "INK", "NULL_LINE", "NULL_FILL"],
    "F.6 (a)": ["reliable", "inconclusive", "unreliable"],
}

# ---- opaque tints (share of white), replacing alpha -----------------------------------------------------------
CI_TINT = 0.55           # 95% interval bars in forest rows (CKA #E8A394, mKNN #8CC2E0, debiased CKA #F7A3C2)
SPAN_TINT = 0.78         # shaded x-ranges in a criterion hue (CKA 'depths of observed max' #F4D2CB)

# ==================================================================================================== sizes
# ---- stroke widths by element class (pt) ------------------------------------------------------------------
LW = {
    "spine": 0.8, "tick": 0.8, "tick_minor": 0.6,      # axes spines and ticks
    "data": 1.5,             # main data line of a series
    "data_emph": 2.0,        # headline summary line (class medians); draw with underlay() (2.0 + 1.4 = 3.4 white)
    "data_thin": 0.6,        # single units (one pair, one run), opaque, class colour
    "connector": 1.2,        # lines joining markers of one series; raw -> after connectors
    "ci_bar": 3.6,           # 95% interval bars in forest rows: opaque ci_color(), butt caps
    "errorbar": 1.0,         # +-s.e., +-s.d., Clopper-Pearson bars, no caps
    "theory": 1.6,           # the bound of Proposition (prop:mst) and other theorem lines, INK
    "profile": 1.2, "profile_emph": 1.6,               # per-layer prediction lines; largest size emphasised
    "ref": 1.0,              # chance, nulls, y = x, frozen epsilon, tolerance lines; LS['dashed'] or LS['dotted']
    "zero": 0.8,             # zero lines inside axes (dashed in strip plots)
    "level": 0.8,            # decision levels (F.4 a-d at 0.8 and 0.9) and band lines (F.5 b), NULL_LINE solid
    "aux": 1.0,              # secondary dashed or dotted data curves (y curves and secondary lines, F.5 b)
    "guide": 0.6,            # forest-row separators, GUIDE
    "leader": 0.7,           # annotation leaders: straight, no arrowhead, shrink 1-2 pt, INK
    "frame": 1.4, "bracket": 0.8,                      # preregistered cell frames / group brackets (F.5 a)
    "median_bar": 2.2,       # medians in strip and swarm plots; colour bars under arm heads
    "cal_tick": 1.4,         # edge width of the s_cal tick marker (MS['cal_tick'] tall)
    "range_bar": 2.2,        # 10-90% ranges (F.4 e)
    "step": 1.5, "step_emph": 2.0,                     # ECDF steps (F.5 b); 'u, no weight decay' emphasised
    "break": 1.0,            # segments across an axis break: LS['dotted'], full series colour
    "axis_break": 0.8,       # the // marks, INK
    "halo": 1.4,             # white underlay width added to an emphasised line
    "median_halo": 1.0,      # white underlay width added to a median bar
}
# ---- markers: Line2D sizes are diameters in pt; scatter sizes are areas in pt^2 ----------------------------
MS = {
    "line": 4.0, "line_edge": 0.6,                 # filled markers on data lines, white edge
    "line_sparse": 4.2,                            # F.2 (c): five sizes per line (filled white edge 0.6 / open 0.9)
    "tri_extra": 0.4,                              # triangles drawn this much larger than circles (Fig. 2 b, F.1 b)
    "open": 5.5, "open_mew": 1.3,                  # open markers that carry a result in forest rows and keys (raw)
    "open_line": 4.8, "open_line_mew": 1.1,        # open markers on gain/level lines (observed max)
    "open_small_mew": 0.9,                         # open markers <= 4 pt (N1, mid depth, pilots); also the rc default
    "open_aux": 3.0,                               # threshold of the selected pair (F.3 d-f), edge open_small_mew
    "dot": 4.6, "dot_edge": 0.5,                   # filled 'after subtraction' dots; intersection dots (F.5 b)
    "scatter": 3.6, "scatter_edge": 0.4, "scatter_open_mew": 0.9,   # point clouds (Fig. 2 c, F.2 a)
    "swarm": 3.0, "swarm_edge": 0.3, "swarm_open_mew": 0.7,         # beeswarm (F.2 b)
    "jitter_s": 4.0, "jitter_alpha": 0.6,          # strip-plot dots: scatter AREA in pt^2 (2.0 pt across), no edge
    "pilot": 3.6, "pilot_mew": 0.9,                # pilot open circles (F.5 c)
    "profile_dot": 2.4, "profile_edge": 0.3,       # permutation-mean dots on profiles (F.3 g-i), white edge
    "summary": 5.0,                                # median markers on range bars (F.4 e)
    "pred": 4.0,                                   # predicted triangle-down (F.4 e), INK
    "cal_tick": 8.5,                               # height of the s_cal tick ('|' marker), edge LW['cal_tick']
}
# Key and legend markers are drawn at exactly the plotted size (legend.markerscale = 1); use key_handle().

# ---- text: size (pt), weight and default colour per role ---------------------------------------------------
FS = {"letter": 10.0, "header": 8.0, "subheader": 7.5, "label": 8.0, "label_small": 7.5, "tick": 7.5,
      "direct": 7.5, "number": 7.5, "theory_label": 7.5, "annot": 7.5, "key": 7.5, "key_header": 7.5,
      "cell": 7.5, "cell_dark": 7.5, "math": 8.0}
# v8: label, label_small and tick are bold (the author: tick labels and axis titles bold in every figure)
FW = {"letter": "bold", "header": "bold", "subheader": "normal", "label": "bold", "label_small": "bold",
      "tick": "bold", "direct": "bold", "number": "bold", "theory_label": "bold", "annot": "normal",
      "key": "normal", "key_header": "bold", "cell": "normal", "cell_dark": "bold"}
FC = {"letter": INK, "header": INK, "subheader": MUTED, "label": INK, "label_small": INK, "tick": INK,
      "direct": INK, "number": INK, "theory_label": INK, "annot": MUTED, "key": INK, "key_header": INK,
      "cell": INK, "cell_dark": WHITE}
# Roles, with the spec's examples:
#   letter       (a) ... (i)
#   header       family names (FAM_TEXT); criterion/column/block heads; arm heads; bracket group labels
#   subheader    status lines (MUTED, default) and tolerance lines (pass color=INK); 8 pt when it holds math
#   label        axis labels;  label_small: the one axis label the spec allows at 7.5 pt (F.2 c, x label)
#   direct       direct series labels, in the series text tone (pass color=...)
#   number       numbers printed on a panel that the caption quotes (INK, or the series text tone on its line)
#   theory_label "bound"
#   annot        reference annotations: chance, null, y = x, excluded, flat, preregistered level, ...
#   key          legend and key entries;  key_header: "covered by the theorem:", "slope", ...
#   cell         table numbers on light fills;  cell_dark: white bold on DECISION['unreliable'] (declare on_fill)
# Every text holding a mathtext sub- or superscript is set at FS['math'] = 8 pt, so its scripts are 7.0 pt;
# text_kw() applies this. Sub-headers go to 8 pt whenever they hold math.
# v8: every BOLD text holding math is set with math_fontfamily=BOLD_MATH (Liberation Serif Bold / Bold Italic);
# text_kw() and finalize() apply this.
MIN_PT = 7.0                     # hard floor for any glyph, scripts included
LETTER_PT = FS["letter"]
SCRIPT_FACTOR = MIN_PT / FS["math"]     # 7/8: mathtext script scale set by use()
MIN_CONTRAST = 4.5               # WCAG 2.1 AA, for every visible text against its background
BOLD_MATH = "custom"             # v8: the math font family of bold texts (mapped by RC_BOLD_MATH)
MIN_CHROMA = 15.0                # v8: CIELAB C* above which a data colour counts as chromatic (check() needs >= 1)

# ---- line styles and z-order -------------------------------------------------------------------------------
LS = {"dashed": "--", "dotted": ":", "dashdot": "-.", "dashed_alt": (0, (5.0, 2.0))}
Z = {"fill": 0, "band": 0, "guide": 0.5, "ref": 1, "interval": 2, "thin": 2, "connector": 3, "underlay": 3.95,
     "data": 4, "marker": 5, "label": 6, "label_top": 7}

# ==================================================================================================== rcParams
RC = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],   # Times metrics; Regular + Bold
    "font.weight": "normal",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": True,
    "font.size": 8,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlepad": 3.0, "axes.titlecolor": INK,
    "axes.labelsize": 8, "axes.labelweight": "bold", "axes.labelpad": 2.5, "axes.labelcolor": INK,   # v8: bold
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.title_fontsize": 7.5, "legend.frameon": False,
    "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
    "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2, "legend.markerscale": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 300, "savefig.bbox": "standard", "savefig.pad_inches": 0.0, "figure.dpi": 150,
    "axes.linewidth": 0.8, "axes.edgecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.axisbelow": True, "grid.linewidth": 0.6, "grid.color": "#DDDDDD",
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "lines.linewidth": 1.5, "lines.markersize": 4.0, "lines.markeredgewidth": 0.9,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "lines.dash_capstyle": "butt",
    "lines.dashed_pattern": [3.2, 1.8], "lines.dotted_pattern": [1.0, 1.6],
    "lines.dashdot_pattern": [4.0, 1.6, 1.0, 1.6], "lines.scale_dashes": True,
    "errorbar.capsize": 0,
    "patch.linewidth": 0.8, "hatch.linewidth": 0.6,
    "scatter.edgecolors": "face",
}
# v8: the 'custom' math family, used only by texts set with math_fontfamily=BOLD_MATH (bold texts). Regular math stays
# Computer Modern ('cm'). Symbols missing from Liberation Serif (\top, \mathcal) fall back to Computer Modern.
RC_BOLD_MATH = {
    "mathtext.rm": "Liberation Serif:bold", "mathtext.it": "Liberation Serif:italic:bold",
    "mathtext.bf": "Liberation Serif:bold", "mathtext.bfit": "Liberation Serif:italic:bold",
    "mathtext.sf": "Liberation Serif:bold", "mathtext.tt": "DejaVu Sans Mono", "mathtext.cal": "cmsy10",
    "mathtext.fallback": "cm",
}
RC.update(RC_BOLD_MATH)
# the rc widths are the LW/MS values (one source of truth; asserted here so a later edit cannot split them)
assert RC["axes.linewidth"] == LW["spine"] and RC["xtick.major.width"] == LW["tick"] == RC["ytick.major.width"]
assert RC["xtick.minor.width"] == LW["tick_minor"] == RC["ytick.minor.width"] and RC["grid.linewidth"] == LW["guide"]
assert RC["lines.linewidth"] == LW["data"] and RC["lines.markersize"] == MS["line"]
assert RC["lines.markeredgewidth"] == MS["open_small_mew"] and RC["grid.color"] == GUIDE
assert RC["axes.labelweight"] == FW["label"] == FW["tick"] == "bold"
# one hex = one meaning: every semantic hex is unique, except the deliberate mKNN = vision step (header)
_dup = {}
for _k, _v in SEM.items():
    _dup.setdefault(_v.upper(), []).append(_k)
assert all(len(v) == 1 or sorted(v) == ["criterion.mKNN", "pair.vision"] for v in _dup.values()), _dup

try:
    with open(os.path.abspath(__file__), "rb") as _fh:
        STYLE_SHA256 = hashlib.sha256(_fh.read()).hexdigest()   # record in numbers.json for provenance
except OSError:                                                 # pragma: no cover
    STYLE_SHA256 = None


# ==================================================================================================== setup
def use():
    """Apply RC (v8: bold axis titles, bold-math fonts) and set mathtext sub- and superscripts to 7/8 of their base
    (SCRIPT_FACTOR), so an 8 pt text has 7.0 pt scripts."""
    import logging
    import matplotlib._mathtext as MT
    mpl.rcParams.update(RC)
    MT.SHRINK_FACTOR = SCRIPT_FACTOR
    logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)
    try:                                   # drop layouts parsed before use() at the old factor
        from matplotlib.mathtext import MathTextParser
        MathTextParser._parse_cached.cache_clear()
    except AttributeError:                 # pragma: no cover
        pass


# ==================================================================================================== colour helpers
def tint(hex_color, w):
    """Opaque mix with white; w = share of white (0 = the hue, 1 = white)."""
    c = np.array(to_rgb(hex_color))
    return to_hex(c + (1 - c) * w)


def ci_color(hex_color):
    """Opaque tint for a 95% interval bar (CKA #E8A394, mKNN #8CC2E0, debiased CKA #F7A3C2, MST #8CD1C9)."""
    return tint(hex_color, CI_TINT)


def span_color(hex_color):
    """Opaque tint for a shaded x-range in a criterion hue (CKA 'depths of observed max' #F4D2CB)."""
    return tint(hex_color, SPAN_TINT)


def tints(hex_color, n, lo=0.55, hi=0.0):
    """v4 helper, kept so old makers import cleanly. NOT for family size ramps: use FAM_RAMP or ramp_n()."""
    c = np.array(to_rgb(hex_color))
    return [to_hex(c + (1 - c) * w) for w in np.linspace(lo, hi, n)]


def ramp(family):
    """Ordinal size ramp of a family, light (small) -> dark (large)."""
    return list(FAM_RAMP[family])


# ---- CIELAB (D65), WCAG contrast, Machado 2009 CVD (module-local; no extra dependency) ----------------------------
_M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE_XYZ = _M_XYZ @ np.ones(3)
CVD_MATRIX = {   # Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def _to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _from_lin(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _lin2lab(lin):
    xyz = (_M_XYZ @ np.asarray(lin, float)) / _WHITE_XYZ
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _lab2lin(lab_):
    L, a, b = lab_
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200

    def finv(t):
        return t ** 3 if t > 6 / 29 else 3 * (6 / 29) ** 2 * (t - 4 / 29)
    return np.linalg.solve(_M_XYZ, np.array([finv(fx), finv(fy), finv(fz)]) * _WHITE_XYZ)


def lab(hex_color, view="normal"):
    """CIELAB of a colour, optionally after Machado 2009 CVD simulation (view 'deutan', 'protan' or 'tritan')."""
    lin = _to_lin(to_rgb(hex_color))
    if view != "normal":
        lin = np.clip(CVD_MATRIX[view] @ lin, 0, 1)
    return _lin2lab(lin)


def chroma(hex_color):
    L, a, b = lab(hex_color)
    return float(np.hypot(a, b))


def contrast(hex_color, bg="#FFFFFF"):
    """WCAG 2.1 contrast ratio of two opaque colours."""
    def lum(h):
        c = _to_lin(to_rgb(h))
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted([lum(hex_color), lum(bg)], reverse=True)
    return float((a + 0.05) / (b + 0.05))


def ramp_n(anchor, n, light_contrast=2.6):
    """v8: n-step ordinal ramp of one hue, light (small) -> anchor (large): even CIELAB L* steps, hue angle kept,
    chroma scaled; the light end has `light_contrast`:1 on white. FAM_RAMP['pythia'] = ramp_n('#662506', 7, 2.3)."""
    L0, a0, b0 = lab(anchor)
    C0, h = np.hypot(a0, b0), np.arctan2(b0, a0)

    def hx(L, C):
        return to_hex(_from_lin(np.clip(_lab2lin((L, C * np.cos(h), C * np.sin(h))), 0, 1)))
    L1, C1 = L0, C0
    for L1 in np.arange(L0, 95, 0.05):
        C1 = C0 * max(0.35, 1 - (L1 - L0) / 100)
        if contrast(hx(L1, C1)) <= light_contrast:
            break
    out = [hx(L1 + (L0 - L1) * t, C1 + (C0 - C1) * t).upper() for t in np.linspace(0, 1, n)]
    out[-1] = anchor.upper()
    return out


def seq_cmap(hex_color, name=None, light=0.92, dark=0.45):
    """v8: single-hue sequential colormap: tint (share of white `light`) -> the hue -> a shade (share of black
    `dark` of the way to INK), monotone in lightness."""
    from matplotlib.colors import LinearSegmentedColormap
    c = np.array(to_rgb(hex_color))
    k = np.array(to_rgb(INK))
    return LinearSegmentedColormap.from_list(name or ("seq_" + hex_color.lstrip("#")),
                                             [tuple(c + (1 - c) * light), tuple(c), tuple(c + (k - c) * dark)])


def cmap_density(lo=1, hi=7):
    """v8: Tol YlOrBr for 2-D densities (hexbin of kernel entries in Figure 6 b, e), steps lo..hi-1 of the nine:
    the default drops the near-white first step (so the lowest bins stay visible on white) and the two darkest browns
    (so the densest bins do not swallow the INK theory line). Draw INK lines over it with underlay(); do not draw
    LINK['across'] markers over it (same hue family)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("tol_YlOrBr_%d_%d" % (lo, hi), TOL["YlOrBr"][lo:hi])


def simulate(hex_color, view):
    """v8: the colour seen under a view: 'normal', 'deutan', 'protan', 'tritan' (Machado 2009, severity 1) or 'grey'
    (CIELAB L* kept, chroma removed). Returns an upper-case hex."""
    lin = _to_lin(to_rgb(hex_color))
    if view == "normal":
        return to_hex(to_rgb(hex_color)).upper()
    if view == "grey":
        Y = float(lin @ np.array([0.2126729, 0.7151522, 0.0721750]))
        return to_hex(_from_lin(np.array([Y, Y, Y]))).upper()
    return to_hex(_from_lin(np.clip(CVD_MATRIX[view] @ lin, 0, 1))).upper()


def palette_report():
    """Contrast on white of every text tone and ramp step (for the smoke test and numbers.json)."""
    rep = {}
    for name, d in (("CRIT_TEXT", CRIT_TEXT), ("FAM_TEXT", FAM_TEXT), ("ARM_TEXT", ARM_TEXT),
                    ("PAIR_TEXT", PAIR_TEXT), ("LINK_TEXT", LINK_TEXT)):
        rep[name] = {k: round(contrast(v), 2) for k, v in d.items()}
    rep["neutrals"] = {"INK": round(contrast(INK), 2), "MUTED": round(contrast(MUTED), 2),
                       "NULL_LINE": round(contrast(NULL_LINE), 2)}
    rep["DECISION_TEXT_on_fill"] = {k: round(contrast(DECISION_TEXT[k], DECISION[k]), 2) for k in DECISION}
    rep["FAM_RAMP"] = {k: [round(contrast(v), 2) for v in vs] for k, vs in FAM_RAMP.items()}
    return rep


def palette_hexes():
    """Every colour the module defines (anchors, text tones, ramps, neutrals, decision fills), upper-case hex."""
    out = set()
    for d in (CRIT, FAM, PAIR, ARM, LINK, CRIT_TEXT, FAM_TEXT, PAIR_TEXT, ARM_TEXT, LINK_TEXT, DECISION, DECISION_TEXT):
        out |= {v.upper() for v in d.values()}
    for vs in FAM_RAMP.values():
        out |= {v.upper() for v in vs}
    out |= {v.upper() for v in (INK, MUTED, NULL_LINE, NULL_FILL, REF_FILL, GUIDE, WHITE, "#000000")}
    out |= {v.upper() for v in TOL["YlOrBr"]}
    for d in (CRIT, LINK, ARM, PAIR, FAM):
        for v in d.values():
            out |= {ci_color(v).upper(), span_color(v).upper()}
    return out


# ==================================================================================================== drawing helpers
def underlay(ax, x, y, lw, zorder, extra=None, **kw):
    """White underlay for an emphasised line, drawn as its own line just below it (zorder - 0.05), width lw + extra
    (default LW['halo']; pass extra=LW['median_halo'] for median bars). Use this, not a path-effect halo, whenever the
    line carries markers: a path effect rings every marker in white and cuts the line next to it."""
    kw.setdefault("solid_capstyle", "round")
    w = LW["halo"] if extra is None else extra
    return ax.plot(x, y, color="white", lw=lw + w, zorder=zorder - 0.05, **kw)


def text_halo(width=1.6):
    """Path-effect white stroke for a label that crosses data (labels only, never lines with markers)."""
    return [pe.withStroke(linewidth=width, foreground="white")]


def panel_letter(fig, letter, x_in, y_in):
    """'(a)' in bold 10 pt with its top-left corner at (x_in, y_in) inches from the figure's bottom-left.
    Letters of one row take the same y_in (same baseline); x_in = left edge of the panel's leftmost element."""
    W, H = fig.get_size_inches()
    return fig.text(x_in / W, y_in / H, "(%s)" % letter, fontsize=FS["letter"], fontweight=FW["letter"],
                    ha="left", va="top", color=FC["letter"])


def on_fill(text, fill_hex):
    """Declare the fill a text sits on (table cells, badges, bands, tinted spans), so check() measures contrast
    against it. check() cannot see what lies under a text: any text on a fill or tint must be declared here.
    Texts with a visible bbox patch are measured against the patch without a declaration."""
    text._hrh_bg = fill_hex
    return text


_MATH_RE = re.compile(r"(?<!\\)\$(.*?)(?<!\\)\$", re.S)


def has_math(s):
    return bool(_MATH_RE.search(s or ""))


def has_script(s):
    """True if a mathtext part of s holds a sub- or superscript (then the text must be set at FS['math'])."""
    return any(("_" in m or "^" in m) for m in _MATH_RE.findall(s or ""))


def _is_bold(weight):
    from matplotlib.font_manager import weight_dict
    w = weight_dict.get(weight, weight) if isinstance(weight, str) else weight
    return float(w) >= 600


def text_kw(role, s="", color=None):
    """fontsize / fontweight / color for a text role (FS, FW, FC). PASS THE STRING s: a text with a script (any
    role) or with math (sub-header) goes to FS['math'] = 8 pt. color overrides FC[role] (series text tones).
    v8: a bold role whose string holds math also gets math_fontfamily=BOLD_MATH (bold math)."""
    size = FS[role]
    if has_script(s) or (role == "subheader" and has_math(s)):
        size = max(size, FS["math"])
    kw = {"fontsize": size, "fontweight": FW[role], "color": FC[role] if color is None else color}
    if FW[role] == "bold" and has_math(s):
        kw["math_fontfamily"] = BOLD_MATH
    return kw


def text(target, x, y, s, role="annot", color=None, **kw):
    """target.text(x, y, s) with the size, weight and colour of a role (text_kw(role, s, color)); target is an Axes
    or a Figure. Prefer this to text_kw(): it cannot forget to pass the string, so scripts always get 8 pt."""
    return target.text(x, y, s, **text_kw(role, s, color), **kw)


def key_handle(**kw):
    """Legend/key handle drawn at exactly the plotted size: pass the same marker/ms/mew/lw/ls as the data."""
    from matplotlib.lines import Line2D
    kw.setdefault("ls", "none" if "marker" in kw and "lw" not in kw else "-")
    return Line2D([], [], **kw)


# ---- v8: bold ticks, axis titles and math --------------------------------------------------------------------------
def pow_label(base, exp):
    """v8: power tick label, e.g. pow_label(2, 20) -> '$2^{20}$'; finalize() sets it bold at 8 pt."""
    return r"$%s^{%s}$" % (base, exp)


def _axis_tick_labels(ax):
    out = []
    for axis in (ax.xaxis, ax.yaxis):
        for t in axis.get_majorticklabels() + axis.get_minorticklabels():
            out.append(t)
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            out += [tick.label1, tick.label2]
    seen, uniq = set(), []
    for t in out:
        if id(t) not in seen:
            seen.add(id(t))
            uniq.append(t)
    return uniq


def _bold_text(t, min_size_for_script=True):
    t.set_fontweight("bold")
    s = t.get_text()
    if has_math(s):
        t.set_math_fontfamily(BOLD_MATH)
    if min_size_for_script and has_script(s) and t.get_fontsize() < FS["math"]:
        t.set_fontsize(FS["math"])


def finalize(fig, bold_all_math=True):
    """v8: call after the last plotting call and before savefig.
      1 draws the figure once, so every tick exists, then sets every tick label (major and minor, both sides) and
        every axis title to bold, with bold math (math_fontfamily=BOLD_MATH); tick labels holding a script go to
        FS['math'] = 8 pt (scripts 7.0 pt). New ticks made at save time copy these properties from the first tick.
      2 with bold_all_math, every other bold text that holds math (headers, direct labels) gets bold math too.
    Returns the number of texts changed."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    n = 0
    for ax in fig.axes:
        for t in _axis_tick_labels(ax):
            _bold_text(t)
            n += 1
        for lab_ in (ax.xaxis.label, ax.yaxis.label, ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            if lab_.get_text().strip():
                _bold_text(lab_, min_size_for_script=False)
                n += 1
    for lab_ in (getattr(fig, "_supxlabel", None), getattr(fig, "_supylabel", None)):
        if lab_ is not None and lab_.get_text().strip():
            _bold_text(lab_, min_size_for_script=False)
            n += 1
    if bold_all_math:
        for t in fig.findobj(mtext.Text):
            if t.get_visible() and _is_bold(t.get_fontweight()) and has_math(t.get_text()):
                t.set_math_fontfamily(BOLD_MATH)
                n += 1
    return n


# ---- number format ----------------------------------------------------------------------------------------
MINUS = "−"


def num(x, nd=2, plus=False):
    """Fixed-point number with a true minus sign (U+2212): num(-0.214) -> '−0.21'."""
    s = "%.*f" % (nd, x)
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    s = s.replace("-", MINUS)
    return ("+" + s) if (plus and x > 0) else s


def fmt_int(v):
    """Integer tick label: 5 or more digits take a comma ('16,384', '65,536'); 4 digits stay plain ('4096')."""
    v = int(round(v))
    s = "{:,}".format(abs(v)) if abs(v) >= 10000 else str(abs(v))
    return (MINUS + s) if v < 0 else s


def int_ticks(ax, values, axis="x"):
    """Set at most 5 labelled major ticks with fmt_int labels."""
    values = list(values)
    assert len(values) <= 5, "at most 5 labelled major ticks per axis: %r" % values
    getattr(ax, "set_%sticks" % axis)(values, [fmt_int(v) for v in values])


# ==================================================================================================== review views (v8)
def views(png_path, out_path, scale=1.0, views_=("normal", "deutan", "protan", "grey")):
    """v8: write one PNG with the normal, deuteranopia, protanopia (Machado 2009, severity 1) and greyscale (CIELAB
    L*) views of a rendered figure, in a 2 x 2 grid with a label above each view. Series must stay apart in every
    view, by marker or line style where the colours merge."""
    from PIL import Image, ImageDraw
    im = np.asarray(Image.open(png_path).convert("RGB"), dtype=float) / 255.0
    if scale != 1.0:
        h, w = im.shape[:2]
        im = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((int(w * scale), int(h * scale)),
                                                                           Image.LANCZOS), float) / 255.0
    lin = _to_lin(im)
    tiles = []
    for v in views_:
        if v == "normal":
            t = im
        elif v == "grey":
            Y = lin @ np.array([0.2126729, 0.7151522, 0.0721750])
            t = np.repeat(_from_lin(Y)[..., None], 3, axis=2)
        else:
            t = _from_lin(np.clip(lin @ CVD_MATRIX[v].T, 0, 1))
        tiles.append((np.clip(t, 0, 1) * 255).round().astype(np.uint8))
    h, w, _ = tiles[0].shape
    pad, head = 16, 28
    grid = np.full((2 * (h + head) + pad, 2 * w + pad, 3), 255, np.uint8)
    names = {"normal": "normal vision", "deutan": "deuteranopia", "protan": "protanopia", "grey": "greyscale",
             "tritan": "tritanopia"}
    pos = []
    for k, t in enumerate(tiles):
        r, c = divmod(k, 2)
        y0, x0 = r * (h + head + pad) + head, c * (w + pad)
        grid[y0:y0 + h, x0:x0 + w] = t
        pos.append((x0, y0 - head + 4, names[views_[k]]))
    img = Image.fromarray(grid)
    d = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=20)
    except (TypeError, OSError):         # Pillow < 10.1
        font = None
    for x0, y0, s in pos:
        d.text((x0 + 4, y0), s, fill=(0, 0, 0), font=font)
    img.save(out_path)
    return out_path


def print_view(pdf_path, png200_path, png100_path=None):
    """v8: render the PDF with pdftoppm at 200 dpi (the staged PNG) and, if asked, at 100 dpi (about print size on a
    screen). Returns the paths written."""
    import shutil
    import subprocess
    exe = shutil.which("pdftoppm")
    assert exe, "pdftoppm not found"
    out = []
    for dpi, p in ((200, png200_path), (100, png100_path)):
        if p is None:
            continue
        stem = p[:-4] if p.endswith(".png") else p
        subprocess.run([exe, "-png", "-r", str(dpi), "-singlefile", pdf_path, stem], check=True)
        out.append(stem + ".png")
    return out


# ==================================================================================================== checks
def pdf_glyph_sizes(pdf_path):
    """Every font size set by a Tf operator in the content streams of the saved PDF: one per text run and per
    mathtext glyph, so a script is measured at its drawn size (the PDF is in pt; text matrices carry no scale)."""
    blob = open(pdf_path, "rb").read()
    out = []
    for m in re.finditer(rb"stream\r?\n", blob):
        start = m.end()
        end = blob.find(b"endstream", start)
        raw = blob[start:end]
        try:
            data = zlib.decompressobj().decompress(raw)
        except zlib.error:
            data = raw
        if b"BT" in data and b"ET" in data and b"Tf" in data:
            out += [float(v) for v in re.findall(rb"/[^\s/\[\]()<>{}%]+\s+([0-9]*\.?[0-9]+)\s+Tf", data)]
    return out


def layout_glyph_sizes(fig):
    """Smallest glyph size (pt) of each visible text from the mathtext layout (diagnostic: names the text behind
    a PDF failure). Tick labels outside the view interval are included, so this is reported, not asserted."""
    import matplotlib.cbook as cbook
    from matplotlib.mathtext import MathTextParser
    parser = MathTextParser("path")
    out = {}
    for t in _visible_texts(fig):
        s = t.get_text()
        if cbook.is_math_text(s):
            try:
                vp = parser.parse(s, dpi=72, prop=t.get_fontproperties())
                g = min((gl[1] for gl in vp.glyphs), default=t.get_fontsize())
            except ValueError:
                g = t.get_fontsize()
        else:
            g = t.get_fontsize()
        out[s] = min(out.get(s, np.inf), round(float(g), 3))
    return out


def pdf_fonts(pdf_path):
    """Rows of `pdffonts` (name, type, encoding, emb, sub, uni) or None if pdffonts is not installed."""
    import shutil
    import subprocess
    exe = shutil.which("pdffonts")
    if exe is None:
        return None
    lines = subprocess.run([exe, pdf_path], capture_output=True, text=True, check=True).stdout.splitlines()
    rows = []
    for ln in lines[2:]:
        tok = ln.split()
        if len(tok) < 8:
            continue
        rows.append({"name": tok[0], "type": " ".join(tok[1:-6]), "encoding": tok[-6], "emb": tok[-5],
                     "sub": tok[-4], "uni": tok[-3]})
    return rows


def _visible_texts(fig):
    import matplotlib.text as mtext
    return [t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]


def _over(rgba, bg_rgb):
    a = rgba[3]
    return tuple(np.array(rgba[:3]) * a + np.array(bg_rgb) * (1 - a))


def _text_background(t):
    bg = getattr(t, "_hrh_bg", None)
    if bg is not None:
        return to_hex(bg)
    patch = t.get_bbox_patch()
    if patch is not None and patch.get_visible():
        fc = to_rgba(patch.get_facecolor())
        if fc[3] > 0:
            return to_hex(_over(fc, (1.0, 1.0, 1.0)))
    return WHITE


def _visible_tick_and_axis_labels(fig):
    """(tick labels, axis titles) that are visible, non-empty and inside the drawn view."""
    ticks, axes_titles = [], []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for axis in (ax.xaxis, ax.yaxis):
            vmin, vmax = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not (vmin - 1e-12 * max(1, abs(vmin)) <= loc <= vmax + 1e-12 * max(1, abs(vmax))):
                    continue
                for t in (tick.label1, tick.label2):
                    if t.get_visible() and t.get_text().strip():
                        ticks.append(t)
            if axis.label.get_visible() and axis.label.get_text().strip():
                axes_titles.append(axis.label)
    return ticks, axes_titles


def data_colours(fig, mapped=True):
    """v8: every opaque-enough colour drawn by a data artist (lines, markers, collections, patches other than the
    axes and figure backgrounds), as upper-case hex. mapped=False leaves out colormapped collections (hexbin, images
    of a colormap), whose many shades are not palette entries."""
    import matplotlib.collections as mcoll
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    cols = set()

    def add(c):
        try:
            arr = np.atleast_2d(to_rgba(c)) if isinstance(c, str) or np.ndim(c) == 1 else np.asarray(c)
        except (ValueError, TypeError):
            return
        for r in arr:
            if len(r) == 4 and r[3] > 0.05:
                cols.add(to_hex(r[:3]).upper())
    backgrounds = {id(fig.patch)} | {id(ax.patch) for ax in fig.axes}
    for a in fig.findobj(lambda o: isinstance(o, (mlines.Line2D, mcoll.Collection, mpatches.Patch))):
        if id(a) in backgrounds or not a.get_visible():
            continue
        if not mapped and isinstance(a, mcoll.Collection) and a.get_array() is not None:
            continue
        if isinstance(a, mlines.Line2D):
            if a.get_linestyle() not in ("None", "none", "", " ") and a.get_linewidth() > 0:
                add(a.get_color())
            if a.get_marker() not in (None, "None", "none", "", " "):
                add(a.get_markerfacecolor())
                add(a.get_markeredgecolor())
        elif isinstance(a, mcoll.Collection):
            add(a.get_facecolors())
            add(a.get_edgecolors())
        else:
            if a.get_fill():
                add(a.get_facecolor())
            if a.get_linewidth() > 0:
                add(a.get_edgecolor())
    return sorted(cols)


def check(fig, pdf_path):
    """Assert the print-size and v8 rules after saving the PDF; return a dict for numbers.json.
      1 every visible Text is >= MIN_PT and every glyph written to the PDF (Tf operators, mathtext scripts at
        their drawn size) is >= MIN_PT;
      2 no Type 3 font; with pdffonts installed, every font is embedded TrueType (Type 42 / CID TrueType);
      3 every visible text reaches MIN_CONTRAST against its background (white unless declared with on_fill or
        drawn on a bbox patch); text alpha is blended over that background;
      4 use() was called (mathtext SHRINK_FACTOR == SCRIPT_FACTOR);
      5 v8: every visible tick label inside the view and every axis title is bold (call finalize(fig));
      6 v8: every visible bold text that holds math uses the bold math family (BOLD_MATH);
      7 v8: at least one data colour is chromatic (CIELAB C* >= MIN_CHROMA): no black-and-white figure.
    Data colours outside palette_hexes() (colormapped collections left out) are reported in 'off_palette_colours'
    (not asserted: alpha blends and derived tints land there too)."""
    import matplotlib._mathtext as MT
    fig.canvas.draw()
    texts = _visible_texts(fig)
    glyph = pdf_glyph_sizes(pdf_path)
    layout = layout_glyph_sizes(fig)
    fonts = pdf_fonts(pdf_path)
    blob = open(pdf_path, "rb").read()
    low = []
    for t in texts:
        bg = _text_background(t)
        rgba = to_rgba(t.get_color(), t.get_alpha())
        col = to_hex(_over(rgba, to_rgb(bg)))
        c = contrast(col, bg)
        if c < MIN_CONTRAST:
            low.append((t.get_text(), col, bg, round(c, 2)))
    ticks, titles = _visible_tick_and_axis_labels(fig)
    not_bold = [t.get_text() for t in ticks + titles if not _is_bold(t.get_fontweight())]
    thin_math = [t.get_text() for t in texts
                 if _is_bold(t.get_fontweight()) and has_math(t.get_text())
                 and t.get_fontproperties().get_math_fontfamily() != BOLD_MATH]
    dcols = data_colours(fig)
    chromatic = [c for c in dcols if chroma(c) >= MIN_CHROMA]
    pal = palette_hexes()
    n_type3 = blob.count(b"/Type3")
    out = {
        "style": "hrh_style_%s" % STYLE_VERSION, "style_sha256": STYLE_SHA256,
        "size_in": [float(v) for v in fig.get_size_inches()],
        "min_text_pt": float(min(t.get_fontsize() for t in texts)) if texts else None,
        "min_glyph_pt": float(min(glyph)) if glyph else None,
        "pdf_glyph_size_counts": {("%g" % v): glyph.count(v) for v in sorted(set(glyph))},
        "smallest_layout_glyphs": dict(sorted(layout.items(), key=lambda kv: kv[1])[:6]),
        "mathtext_shrink_factor": float(MT.SHRINK_FACTOR),
        "n_type3": n_type3,
        "fonts": None if fonts is None else ["%s (%s, emb %s)" % (r["name"], r["type"], r["emb"]) for r in fonts],
        "low_contrast_text": low,
        "n_tick_labels_checked": len(ticks), "n_axis_titles_checked": len(titles),
        "not_bold_tick_or_axis_labels": not_bold, "bold_text_with_regular_math": thin_math,
        "data_colours": dcols, "chromatic_data_colours": chromatic,
        "off_palette_colours": [c for c in data_colours(fig, mapped=False) if c not in pal],
    }
    assert abs(MT.SHRINK_FACTOR - SCRIPT_FACTOR) < 1e-12, ("call S.use() before drawing", out)
    assert out["min_text_pt"] is not None and out["min_text_pt"] >= MIN_PT - 1e-9, ("text below MIN_PT", out)
    assert out["min_glyph_pt"] is not None and out["min_glyph_pt"] >= MIN_PT - 1e-9, ("glyph below MIN_PT", out)
    assert n_type3 == 0, ("Type 3 font", out)
    if fonts is not None:
        assert fonts, ("pdffonts lists no font", out)
        assert all(r["emb"] == "yes" and "TrueType" in r["type"] and "Type 3" not in r["type"] for r in fonts), \
            ("font not embedded TrueType", out)
    assert not low, ("text below MIN_CONTRAST", out)
    assert not not_bold, ("tick label or axis title not bold: call S.finalize(fig) before savefig", out)
    assert not thin_math, ("bold text with regular math: call S.finalize(fig) or use text_kw()", out)
    assert chromatic, ("no chromatic data colour: black-and-white figure", out)
    return out
