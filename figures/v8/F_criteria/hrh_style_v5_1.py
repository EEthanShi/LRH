"""Shared figure style v5.1 for the LRH paper (arxiv_v4.tex figures). Frozen (chmod 444) 2026-10-10.

v5.1 (2026-10-10, role fig-colours): a copy of hrh_style_v5.py (sha256 8c75d526...5ad4, frozen, untouched) that changes
only ARM and ARM_TEXT (and STYLE_VERSION, for provenance). Reason: v5 reused family hues for training rules (weight
decay = BLOOMZ #332288, small init. = OpenLLaMA #999933, Adam = LLaMA-1 #882255), and main-text Figures 3 (families)
and 4 (training rules) sit on facing pages. v5.1 gives the rules their own Tol hues: weight decay #44AA99 (muted teal,
text #108576), small init. #AA4499 (muted purple), Adam #663333 (Tol dark red); GD/no weight decay black, tied #117733
and SGD #CC6677 unchanged. CVD check (CIEDE2000 under Machado 2009, the v5 spec method):
~/hrh_v2_figs/v5_1/_palette/palette_v5_1_check.json. Everything below this paragraph is v5 unless marked v5.1.

v5 header follows.

Spec: build/figs_v5_staging/_research/STYLE_V5_SPEC.md (style-research role, 2026-10-09). This file implements it.
Made from a copy of hrh_style_v4.py (sha256 a04b6115...0def9, untouched) and the spec's draft
(_research/hrh_style_v5_draft.py, sha256 1f693529...1cb0); every v4 name is still defined here.

Use:
    import hrh_style_v5 as S
    S.use()                      # rcParams AND mathtext scripts at 7/8 of their base (makers no longer set this)
    ...
    fig.savefig(pdf); out = S.check(fig, pdf)   # asserts the print-size rules; put `out` into numbers.json

Rules the module owns (makers import them; a number left in a maker must be geometry, not style):
  * LW  stroke widths in pt per element class        * MS  marker sizes in pt (scatter areas in pt^2)
  * FS  text sizes in pt per text role               * FW  weights per role ("normal" or "bold" only)
  * FC  default text colour per role                 * Z   z-order per layer        * LS  line styles
  * palette: CRIT, FAM, PAIR, ARM (line colours) and CRIT_TEXT, FAM_TEXT, PAIR_TEXT, ARM_TEXT (text tones, >= 4.5:1)
What changed against v4 (the author, 2026-10-09: make everything that can be bold bolder):
  * strokes heavier (spines and ticks 0.6 -> 0.8 pt, data 1.1 -> 1.5 pt, connectors -> 1.2 pt, references -> 1.0 pt);
  * panel letters 9 -> 10 pt bold; headers, direct series labels, quoted numbers, the bound label in bold;
  * no text below 4.5:1 against its background: coloured text uses the darker text tone of its hue; seed text is INK;
  * opaque tints instead of alpha (only jitter dots keep alpha);
  * mathtext SHRINK_FACTOR = 7/8 set in use(); every text with a script is set at 8 pt, so scripts are 7.0 pt;
  * check() measures glyph sizes from the PDF Tf operators (scripts at their drawn size), lists the fonts with
    pdffonts, and checks text contrast.
Fonts: Liberation Serif (TrueType, Times metrics; weights Regular and Bold) for text, mathtext 'cm' for math;
pdf.fonttype 42, so every font is embedded as TrueType (CID TrueType) and no Type 3 font appears.
Inline math inside bold text stays regular Computer Modern, as \\textbf{... $G_s$ ...} does in the captions.
"""
import hashlib
import os
import re
import zlib

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import to_hex, to_rgb, to_rgba

STYLE_VERSION = "v5_1"                  # v5.1: ARM / ARM_TEXT only (see the header)
TEXTWIDTH_IN = 5.5                      # neurips_2024: \textwidth = 5.5 in

# ==================================================================================================== palette
# Hues are unchanged from v4. Line colours first; text tones (>= 4.5:1 on white) further below.
# ---- criteria: one Okabe-Ito hue per criterion, identical in every figure ---------------------------------
CRIT = {
    "mknn": "#0072B2",   # neighbor sets: mKNN s_N(k), 1-NN s_{N,1}
    "mst":  "#009E73",   # spanning trees: MST overlap s_T, bridges
    "cka":  "#D55E00",   # biased linear CKA
    "ckad": "#CC79A7",   # debiased CKA (exploratory)
    "dist": "#E69F00",   # distance-aware criteria (SOG tau, G_s, RSA) when a hue is needed; edge with INK
}
# ---- language-model families: Tol 'muted' dark trio, distinct from every criterion hue --------------------
FAM = {"bloomz": "#332288", "llama1": "#882255", "openllama": "#999933"}
FAM_LABEL = {"bloomz": "BLOOMZ", "llama1": "LLaMA-1", "openllama": "OpenLLaMA"}
FAM_MARKER = {"bloomz": "o", "llama1": "s", "openllama": "D"}
# ---- vision-model pair classes (ordinal; every curve is mKNN): lightness ramp of the mKNN hue --------------
PAIR = {"seed": "#7FB8DA", "vision": "#0072B2", "vl": "#003A5D"}
PAIR_LABEL = {"seed": "seed", "vision": "vision", "vl": "vision–language"}
# ---- training rules (learning figures only; LM families never appear there) -------------------------------
# v5.1: own Tol hues, none shared with a family or criterion (v5 reused BLOOMZ, OpenLLaMA, LLaMA-1).
# Nearest family/criterion hue, dE2000 in normal vision: wd 9.6 (MST #009E73), smallinit 15.0 (LLaMA-1),
# adam 14.5 (LLaMA-1), sgd 12.0 (debiased CKA), tied 16.4 (MST). Worst pair inside one figure over normal,
# deutan, protan and tritan simulation: Figure 4 / F.5 18.0 (wd-smallinit, deutan); F.4 11.5 (smallinit-sgd,
# tritan; v5 had 11.8, wd-adam protan), 12.7 (tied-adam, deutan): markers and line styles keep those apart too.
ARM = {"gd": "#000000", "nowd": "#000000", "wd": "#44AA99", "smallinit": "#AA4499", "tied": "#117733",
       "sgd": "#CC6677", "adam": "#663333"}

# ---- text tones: same hue, CIELAB L* lowered until the colour reaches 4.5:1 on white ----------------------
CRIT_TEXT = {"mknn": "#0072B2", "mst": "#098763", "cka": "#C25604", "ckad": "#AC5C89", "dist": "#9F6D00"}
FAM_TEXT = {"bloomz": "#332288", "llama1": "#882255", "openllama": "#6B6B24"}
# v5.1: wd #44AA99 is 2.82:1, so its text tone is L*-lowered to #108576 (4.53:1, dE2000 12.8 from the line);
# smallinit #AA4499 (5.26:1) and adam #663333 (10.07:1) are their own text tones.
ARM_TEXT = {"gd": "#000000", "nowd": "#000000", "wd": "#108576", "smallinit": "#AA4499", "tied": "#117733",
            "sgd": "#BA5668", "adam": "#663333"}
# Seed: every seed tone with >= 4.5:1 lies within dE2000 7.5 of the vision blue (#427D9D), and every tone >= 20
# away from it is below 2.7:1. So the seed name and slope are INK, placed where only the seed bundle is.
INK = "#222222"          # 15.9:1 on white: text, theory lines, frames
PAIR_TEXT = {"seed": INK, "vision": "#0072B2", "vl": "#003A5D"}

# ---- neutrals --------------------------------------------------------------------------------------------
MUTED = "#555555"        # 7.5:1; all secondary annotation text (the lightest text allowed)
NULL_LINE = "#707070"    # 4.9:1; chance, nulls, reference lines; never text (v4 #7F7F7F, 4.0:1)
NULL_FILL = "#E5E5E5"    # null bands, excluded region, relabelling band, gradient-flow band
REF_FILL = "#F2F2F2"     # flat bands (F.1 c), column bands (F.4)
GUIDE = "#DDDDDD"        # forest-row separators
WHITE = "#FFFFFF"
# ---- decision fills (tables in Fig. 4 a, F.5 a) and the text set on them -----------------------------------
DECISION = {"reliable": "#EDEDED", "inconclusive": "#A8A8A8", "unreliable": "#3B3B3B"}
DECISION_TEXT = {"reliable": INK, "inconclusive": INK, "unreliable": WHITE}     # white is bold (FW['cell_dark'])

# ---- ordinal size ramps (light = small): even steps in CIELAB L*, lightest >= 2.6:1 -----------------------
FAM_RAMP = {"bloomz": ["#A199C8", "#887AB8", "#6E5CA8", "#523F98", "#332288"],      # 0.56B -> 7.1B
            "llama1": ["#C390AA", "#A75C7E", "#882255"],                              # 7B -> 30B
            "openllama": ["#A4A449", "#7E7E33", "#5A5A1E"]}                           # 3B -> 13B
# pair classes keep their ramp: PAIR (seed -> vision -> vision-language)

# ---- opaque tints (share of white), replacing alpha --------------------------------------------------------
CI_TINT = 0.55           # 95% interval bars in forest rows (v4: alpha 0.35, i.e. 0.65 white)
SPAN_TINT = 0.78         # shaded x-ranges in a criterion hue, e.g. depths of the observed max (v4: alpha 0.16)

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
FW = {"letter": "bold", "header": "bold", "subheader": "normal", "label": "normal", "label_small": "normal",
      "tick": "normal", "direct": "bold", "number": "bold", "theory_label": "bold", "annot": "normal",
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
MIN_PT = 7.0                     # hard floor for any glyph, scripts included
LETTER_PT = FS["letter"]
SCRIPT_FACTOR = MIN_PT / FS["math"]     # 7/8: mathtext script scale set by use()
MIN_CONTRAST = 4.5               # WCAG 2.1 AA, for every visible text against its background

# ---- line styles and z-order -------------------------------------------------------------------------------
# Dash patterns come from rc and scale with the line width (lines.scale_dashes). Use 'dashed_alt' only where two
# dashed styles must differ in one panel (e.g. the post hoc baseline in Fig. 3 d).
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
    "axes.labelsize": 8, "axes.labelweight": "normal", "axes.labelpad": 2.5, "axes.labelcolor": INK,
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
# the rc widths are the LW/MS values (one source of truth; asserted here so a later edit cannot split them)
assert RC["axes.linewidth"] == LW["spine"] and RC["xtick.major.width"] == LW["tick"] == RC["ytick.major.width"]
assert RC["xtick.minor.width"] == LW["tick_minor"] == RC["ytick.minor.width"] and RC["grid.linewidth"] == LW["guide"]
assert RC["lines.linewidth"] == LW["data"] and RC["lines.markersize"] == MS["line"]
assert RC["lines.markeredgewidth"] == MS["open_small_mew"] and RC["grid.color"] == GUIDE

try:
    with open(os.path.abspath(__file__), "rb") as _fh:
        STYLE_SHA256 = hashlib.sha256(_fh.read()).hexdigest()   # record in numbers.json for provenance
except OSError:                                                 # pragma: no cover
    STYLE_SHA256 = None


# ==================================================================================================== setup
def use():
    """Apply RC and set mathtext sub- and superscripts to 7/8 of their base (SCRIPT_FACTOR), so an 8 pt text has
    7.0 pt scripts. Makers no longer set matplotlib._mathtext.SHRINK_FACTOR themselves (and must not assert 0.7)."""
    import logging
    import matplotlib._mathtext as MT
    mpl.rcParams.update(RC)
    MT.SHRINK_FACTOR = SCRIPT_FACTOR
    # fontTools warns on the 1970 'created' stamps of the bundled CM fonts each time it subsets them; harmless
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
    """Opaque tint for a 95% interval bar (CKA #ECB78C, mKNN #8CC0DC, debiased CKA #E8C3D7, MST #8CD3C0)."""
    return tint(hex_color, CI_TINT)


def span_color(hex_color):
    """Opaque tint for a shaded x-range in a criterion hue (CKA 'depths of observed max' #F6DCC7)."""
    return tint(hex_color, SPAN_TINT)


def tints(hex_color, n, lo=0.55, hi=0.0):
    """v4 helper, kept so v4 makers import cleanly. NOT for family size ramps: use FAM_RAMP (even L* steps)."""
    c = np.array(to_rgb(hex_color))
    return [to_hex(c + (1 - c) * w) for w in np.linspace(lo, hi, n)]


def ramp(family):
    """Ordinal size ramp of a family, light (small) -> dark (large)."""
    return list(FAM_RAMP[family])


def contrast(hex_color, bg="#FFFFFF"):
    """WCAG 2.1 contrast ratio of two opaque colours."""
    def lum(h):
        c = np.array(to_rgb(h))
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted([lum(hex_color), lum(bg)], reverse=True)
    return float((a + 0.05) / (b + 0.05))


def palette_report():
    """Contrast on white of every text tone and ramp step (for the smoke test and numbers.json)."""
    rep = {}
    for name, d in (("CRIT_TEXT", CRIT_TEXT), ("FAM_TEXT", FAM_TEXT), ("ARM_TEXT", ARM_TEXT),
                    ("PAIR_TEXT", PAIR_TEXT)):
        rep[name] = {k: round(contrast(v), 2) for k, v in d.items()}
    rep["neutrals"] = {"INK": round(contrast(INK), 2), "MUTED": round(contrast(MUTED), 2),
                       "NULL_LINE": round(contrast(NULL_LINE), 2)}
    rep["DECISION_TEXT_on_fill"] = {k: round(contrast(DECISION_TEXT[k], DECISION[k]), 2) for k in DECISION}
    rep["FAM_RAMP"] = {k: [round(contrast(v), 2) for v in vs] for k, vs in FAM_RAMP.items()}
    return rep


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


def text_kw(role, s="", color=None):
    """fontsize / fontweight / color for a text role (FS, FW, FC). PASS THE STRING s: a text with a script (any
    role) or with math (sub-header) goes to FS['math'] = 8 pt. color overrides FC[role] (series text tones)."""
    size = FS[role]
    if has_script(s) or (role == "subheader" and has_math(s)):
        size = max(size, FS["math"])
    return {"fontsize": size, "fontweight": FW[role], "color": FC[role] if color is None else color}


def text(target, x, y, s, role="annot", color=None, **kw):
    """target.text(x, y, s) with the size, weight and colour of a role (text_kw(role, s, color)); target is an Axes
    or a Figure. Prefer this to text_kw(): it cannot forget to pass the string, so scripts always get 8 pt."""
    return target.text(x, y, s, **text_kw(role, s, color), **kw)


def key_handle(**kw):
    """Legend/key handle drawn at exactly the plotted size: pass the same marker/ms/mew/lw/ls as the data."""
    from matplotlib.lines import Line2D
    kw.setdefault("ls", "none" if "marker" in kw and "lw" not in kw else "-")
    return Line2D([], [], **kw)


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


def check(fig, pdf_path):
    """Assert the print-size rules after saving the PDF; return a dict for numbers.json.
      1 every visible Text is >= MIN_PT (as in v4) and every glyph written to the PDF (Tf operators, mathtext
        scripts at their drawn size) is >= MIN_PT;
      2 no Type 3 font; with pdffonts installed, every font is embedded TrueType (Type 42 / CID TrueType);
      3 every visible text reaches MIN_CONTRAST against its background (white unless declared with on_fill or
        drawn on a bbox patch); text alpha is blended over that background;
      4 use() was called (mathtext SHRINK_FACTOR == SCRIPT_FACTOR)."""
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
    return out
