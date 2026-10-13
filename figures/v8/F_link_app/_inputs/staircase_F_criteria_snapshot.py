"""Explained-share staircase, shared by F_criteria (a) and F_link (c) (v6 figure plan, PLAN.md sections 3.2 and 3.5).

The staircase plots, for each pair of representations, the cumulative share of the squared-distance discrepancy
SS0 = sum (D_B - D_A)^2 that is explained by a sequence of nested fits (hrhv2 warp ladder; S1_2_report.md section 6.1):

    step 1  scale                a common scale factor
    step 2  + affine             an affine map of distances (extra share over step 1)
    step 3  + global monotone    a global isotonic map (extra share over step 2)
    step 4  + per-row monotone   an isotonic map per row / anchor (extra share over step 3)
    and the residual = 1 - cumulative share at step 4.

Theory reference (Theorem link): if the two distance matrices are related by an increasing link, the global monotone fit
explains everything, so the cumulative share reaches 1 at step 3 and stays there.

SHARED CONVENTION (v6 fix 2026-10-10, critic finding "staircases don't match"; both figures must use exactly these):
  x tick labels   STEP_LABELS  (words, not cD / aD+b / g(D) / g_i(D): easier for a general reader); they need
                  MIN_IN_PER_STEP_WORDS = 0.47 in per step. Fallback for a narrow panel, in both figures or neither:
                  STEP_LABELS_MATH.
  y label         Y_LABEL = "share explained (cumulative)", centred on the bounded spine [0, 1]
  reference       full_share_line(): NULL_LINE dotted at y = 1 (LW ref), "everything explained"
  theory path     link_reference(): INK dashed (LW ref) from the affine median to 1 at the global monotone step,
                  ending there (F_criteria a). In F_link (c) the link-check series (INK dashed, open diamonds) is the
                  realised counterpart of this path, so dashed INK means "a monotone link" in both figures.

This module draws only; it holds no data paths. Every function takes the axes and arrays and returns what it drew (for
numbers.json). Style values come from the frozen hrh_style module the caller passes in (no style number lives here
except geometry, as the style module asks).

Functions
  cumulative(df, prefix)                       -> (n, 4) cumulative shares, after integrity asserts
  setup(ax, S, labels=STEP_LABELS, ...)        x ticks at the four steps, y from 0 to 1 (spine bounded at 1), Y_LABEL
  full_share_line(ax, S, ...)                  NULL_LINE dotted line at y = 1
  draw_series(ax, S, cum, color, ...)          thin line per pair (optional), 10-90% band (optional), bold median
  link_reference(ax, S, x0, y0, ...)           dashed INK theory line from (x0, y0) to (x_mono, 1), optional flat part
  residual_bracket(ax, S, x, y_lo, ...)        INK bracket from y_lo to 1 with a label to its right
"""
import numpy as np

STEPS = ("scale", "affine_extra", "iso_extra", "row_extra")
STEP_LABELS = ("scale", "+ affine", "+ global\nmonotone", "+ per-row\nmonotone")
Y_LABEL = "share explained (cumulative)"
# Fallback for a narrow staircase: the word labels need about 0.47 in per step at 7.5 pt (F_criteria a has 0.478 in);
# below that (F_link c has 0.32 in per step) use these compact labels in BOTH figures, set at FS['math'] (8 pt), and
# name the maps in the caption. Choosing between the two sets is the lead's call; never mix them across figures.
STEP_LABELS_MATH = (r"$cD$", r"$aD{+}b$", r"$g(D)$", r"$g_i(D)$")
MIN_IN_PER_STEP_WORDS = 0.47   # F_criteria (a): 0.478 in per step, tick-label gap 0.058 in
X = np.arange(1, 5, dtype=float)


def cumulative(df, prefix="L.warpshare_expl_", tol=1e-9):
    """(n, 4) cumulative explained shares from the per-step columns <prefix>{scale, affine_extra, iso_extra,
    row_extra, residual}. Asserts: every row sums to 1 with its residual; extras are non-negative; shares in [0, 1]."""
    E = df[[prefix + k for k in STEPS + ("residual",)]].to_numpy(float)
    assert np.isfinite(E).all(), "non-finite share"
    assert np.abs(E.sum(1) - 1.0).max() < tol, ("shares do not sum to 1", float(np.abs(E.sum(1) - 1).max()))
    assert (E[:, 1:4] >= -tol).all(), "negative extra share"
    C = np.cumsum(E[:, :4], axis=1)
    assert (C >= -tol).all() and (C <= 1 + tol).all()
    return C


def setup(ax, S, labels=STEP_LABELS, xlim=(0.68, 4.32), ylim=(0.0, 1.2), yticks=(0, 0.5, 1), ylabel=Y_LABEL):
    """Categorical x at the four steps; y from 0 to ylim[1] with the left spine bounded at [0, 1] (shares cannot
    exceed 1; the room above holds the theory label); y label Y_LABEL (pass ylabel=None to set it later)."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(X, list(labels))
    for t in ax.get_xticklabels():
        t.set_linespacing(1.0)
        t.set_verticalalignment("top")
        if S.has_math(t.get_text()):
            t.set_fontsize(S.FS["math"])
    ax.set_yticks(list(yticks), [("%g" % v) for v in yticks])
    ax.spines["left"].set_bounds(0.0, 1.0)
    ax.minorticks_off()
    if ylabel:
        ax.set_ylabel(ylabel)
    # centre the y label on the bounded spine [0, 1], not on the full y range (the room above 1 holds labels)
    ax.yaxis.label.set_y((0.5 - ylim[0]) / (ylim[1] - ylim[0]))


def full_share_line(ax, S, x0=None, x1=None, z_offset=0.5):
    """NULL_LINE dotted line at y = 1 ('everything explained'), LW ref, drawn above the thin pair lines (z thin + 0.5)
    so pairs that end near 1 do not hide it. x0, x1 default to the x limits."""
    lo, hi = ax.get_xlim()
    xs = [lo if x0 is None else x0, hi if x1 is None else x1]
    ax.plot(xs, [1.0, 1.0], color=S.NULL_LINE, lw=S.LW["ref"], ls=S.LS["dotted"], zorder=S.Z["thin"] + z_offset,
            solid_capstyle="butt", dash_capstyle="butt")
    return {"y": 1.0, "x": [float(v) for v in xs], "style": "NULL_LINE dotted, LW ref"}


def draw_series(ax, S, cum, color, thin_color=None, dodge=0.0, marker=None, band=None, band_color=None,
                thin=True, z_offset=0.0):
    """Draw one group of pairs.
      cum         (n, 4) cumulative shares (cumulative())
      color       hue of the bold median line and its markers
      thin_color  colour of the thin per-pair lines (opaque tint); thin=False skips them
      dodge       x offset of this group (units of steps), so two groups do not hide each other's markers
      marker      dict of Line2D marker kwargs for the median (ms, mfc, mec, mew, marker)
      band        (lo, hi) percentiles for a shaded band in band_color (F_link c); None = no band
    Returns a dict with the plotted medians and band edges (for numbers.json)."""
    LW, Z = S.LW, S.Z
    x = X + dodge
    out = {"x": x.tolist(), "n": int(cum.shape[0])}
    if thin:
        tc = thin_color if thin_color is not None else S.tint(color, S.SPAN_TINT)
        segs = [np.column_stack([x, row]) for row in cum]
        from matplotlib.collections import LineCollection
        lc = LineCollection(segs, colors=tc, linewidths=LW["data_thin"], zorder=Z["thin"] + z_offset,
                            capstyle="round", joinstyle="round")
        ax.add_collection(lc)
        out["thin_color"] = tc
    if band is not None:
        lo, hi = np.percentile(cum, band, axis=0)
        bc = band_color if band_color is not None else S.tint(color, S.SPAN_TINT)
        ax.fill_between(x, lo, hi, color=bc, lw=0, zorder=Z["band"] + z_offset)
        out["band_percentiles"] = list(band)
        out["band_lo"], out["band_hi"] = lo.tolist(), hi.tolist()
    med = np.median(cum, axis=0)
    S.underlay(ax, x, med, LW["data_emph"], Z["data"] + z_offset)
    ax.plot(x, med, color=color, lw=LW["data_emph"], zorder=Z["data"] + z_offset, solid_capstyle="round")
    mk = dict(marker="o", ms=S.MS["open_line"], mfc="white", mec=color, mew=S.MS["open_line_mew"])
    if marker:
        mk.update(marker)
    ax.plot(x, med, ls="none", zorder=Z["marker"] + z_offset, **mk)
    out["median_cumulative"] = med.tolist()
    out["marker"] = mk
    return out


def link_reference(ax, S, x0, y0, x_mono=3.0, x_end=None, label=None, label_xy=None, label_kw=None):
    """Dashed INK theory line: from (x0, y0) (the median after the affine step) to (x_mono, 1) (a monotone link
    explains everything at the global monotone step); flat at 1 to x_end only if x_end is given (default: the line
    ends at the bend, on the dotted full-share line). Optional label (INK bold, theory role)."""
    xs, ys = [x0, x_mono], [y0, 1.0]
    if x_end is not None:
        xs, ys = xs + [x_end], ys + [1.0]
    ax.plot(xs, ys, color=S.INK, lw=S.LW["ref"], ls=S.LS["dashed"], zorder=S.Z["ref"] + 2.5,
            solid_capstyle="butt", dash_capstyle="butt")
    t = None
    if label is not None:
        kw = dict(ha="center", va="bottom", linespacing=1.0)
        kw.update(label_kw or {})
        t = S.text(ax, label_xy[0], label_xy[1], label, "theory_label", **kw)
    return {"polyline": [list(map(float, xs)), list(map(float, ys))], "label": label}, t


def residual_bracket(ax, S, x, y_lo, y_hi=1.0, tick=0.06, label="residual", gap=0.05):
    """INK bracket at x from y_lo to y_hi, ticks pointing left (towards the data), label right of it (direct role,
    INK). Drawn outside the axes if x is beyond the x limits (clip_on False)."""
    ax.plot([x - tick, x, x, x - tick], [y_lo, y_lo, y_hi, y_hi], color=S.INK, lw=S.LW["bracket"],
            zorder=S.Z["label"], clip_on=False, solid_capstyle="butt", solid_joinstyle="miter")
    t = S.text(ax, x + gap, (y_lo + y_hi) / 2, label, "direct", ha="left", va="center", clip_on=False)
    return {"x": float(x), "y_lo": float(y_lo), "y_hi": float(y_hi), "label": label}, t
