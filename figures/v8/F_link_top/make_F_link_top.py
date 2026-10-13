"""F_link v8: Figure 6 (label fig:exp-link, Section 5.6 sec:exp-link) and its two alternatives F_link_top and
F_link_net. Role maker, 2026-10-10 (evening).

v8 = a copy of ~/hrh_v2_figs/v7/F_link/make_F_link.py (sha256 4fce26cc...280f, untouched) redone under the author's
feedback on v7 Figure 6 (2026-10-10 evening): no row header states a setting ("inside the premises of Theorem 3",
"Finite Gibbs model", "CIFAR-5m networks"; the caption and the Results text name the setting when they call a panel);
no black-and-white figure; tick labels and axis titles bold, with bold math; secondary comparisons that the Results
text can state are not drawn. Changes:
  * the style is the frozen v8 module ~/hrh_v2_figs/v8/_style/hrh_style_v8.py (sha256 asserted; Tol schemes);
  * colours from its semantic map: InfoNCE vs spectral = LINK['across'] (Tol vibrant orange; filled circles, and
    open squares for 'link undone'), the positive control (exact link of Theorem 3) = LINK['control'] (Tol bright
    blue; open diamonds, dashed in (f)), the best monotone link of (e) = the across hue in its text tone, theory
    (predictions, the link of the theorem, the requirement bar) = INK, within one objective = the neutral NULL_FILL
    band/box with a NULL_LINE edge, the kernel densities = S.cmap_density() (Tol YlOrBr); direct labels in the text
    tone of their series;
  * the two row headers are gone; one centred key line at the top serves the figure;
  * (f) no longer draws InfoNCE at 0.2 vs 0.5 (open circles), its direct label and its key entry; the numbers are
    still computed (numbers.json text_numbers) and MOVED.md gives the Results sentence that carries them;
  * S.finalize(fig) bolds every tick label and axis title (and their math) before each measurement and before saving;
  * the PNG is the PDF rendered at 200 dpi; _view_print_100dpi.png (print size) and _views_cvd_grey.png (normal,
    deuteranopia, protanopia, greyscale) are written for review;
  * one code base, three figures, chosen by the file name: make_F_link.py -> F_link (both rows), make_F_link_top.py
    -> F_link_top (the finite Gibbs model, a-c, rows 1.20 in tall), make_F_link_net.py -> F_link_net (the networks,
    printed a-c); the three makers are byte copies;
  * every v7 assert is kept (same name), replaced by a named v8 assert, or marked not applicable to the variant, and
    this accounting is itself asserted against the archived v7 numbers.json (_v7_ref/).
Data, pick rules, marker shapes and every data assert are unchanged.
Run: OMP_NUM_THREADS=1 nice -n 10 taskset -c 0-5 ~/hrh_v2_figs_venv/bin/python -B make_<id>.py > run.log 2>&1

Critic pass on v8 (2026-10-10 evening; the maker's v8 outputs are kept in _prev_maker_v8/ of each figure directory).
The three figures are unchanged. Findings and fixes:
  * the appendix figure F.8 (fig:app-s21), redrawn in v8 by its own maker (~/hrh_v2_figs/v8/F_link_app), no longer
    draws InfoNCE at 0.2 vs 0.5 in any panel, and its panel (c) shows 500,000 images only. MOVED.md of F_link and
    F_link_net said the series was "still drawn in Figure fig:app-s21 c" and pointed the must-add sentence C5 there, so
    the comparison would have left every figure under a pointer to a panel that no longer shows it. C5 is now one
    sentence for Section 5.6 (the warp share between the two temperatures, a median 11% at both sizes, which carries
    the "thus" that follows it), and the cumulative 39% against 62% is the sentence that F_link_app/MOVED.md item 9
    adds to Appendix app:exp-s21. New checks: the 50,000-image warp share also rounds to 11%; F_link_app/MOVED.md
    carries the 0.39 against 0.62 sentence; MOVED.md makes no "still drawn" claim. The quotes of the v4 F.8 caption
    (l.6703) in MOVED.md items A33, A34, A38 are annotated with the v8 F.8 caption that replaces it.
  * F_link_net caption: "Figure fig:app-s21 gives both sample sizes and every pair set" was no longer true (F.8 v8
    drops one pair set), now "gives both sample sizes".
  * independent recompute (_critic_v8/critic_v8_recompute.py, own numpy/scipy code, raw Grams and embeddings): 198
    plotted values of the three figures, largest difference 7.4e-14; asserted below.

v7 docstring follows.

F_link v7 (main text Section 5.6 sec:exp-link, Figure 6, label fig:exp-link). Role maker, 2026-10-10.

v7 = a copy of ~/hrh_v2_figs/v6/F_link_s23/make_F_link.py (sha256 recorded in numbers.json:v6_maker, untouched) redone
under the author's feedback of 2026-10-10 on Figures 4 and 6 (panels fill the row, little text on the figure, short
captions; Figure 6 with its caption filled page 20). Data, encodings, pick rules and every v6 assert are unchanged; only
text and frame move:
  * no separate key strip: the two top-row encodings (grey within band/box, filled circle) are keyed at the right end
    of the top-row header line, the two bottom-row-only encodings (diamond, open circle) at the right end of the
    bottom-row header line;
  * row headers cut to a few words, without settings ('certified ERMs', 'N = 500,000' and the temperature move to the
    Results text / appendix, see MOVED.md);
  * (d) row labels are the short symbols of Table tab:app-s21-full ('(projector)', '(isotonic)', ', tau = 0.1' dropped),
    so the row-label gutter shrinks and the (d) axes widen; the '<- within  outside ->' annotation and the
    'Delta/eps' symbol of the x label are dropped; the rows are packed tighter (pitch 0.137 in instead of 0.144 in);
  * one-line y labels in (a) and (c), shorter kernel axis labels in (b) and (e) ('kernel' moves to the caption); the
    (e) key lines sit just above the square, below the panel letter; the (b) and (e) squares grow to 1.05 in;
  * (c) y label centred on its bounded spine so that it clears the panel letter;
  * caption_v7.tex is read, hashed and checked here (no result digits, at most 100 words, bold first sentence at most
    30 words); the v6 caption checks still run on the archived v6 caption (_v6_ref/caption_v6.tex).
The style module is a byte copy of the frozen hrh_style_v5_1.py in _style/ (sha256 asserted equal to the frozen one).
Critic pass on v7 (2026-10-10; the maker's v7 outputs are kept in _prev_maker_v7/): the figure is unchanged. The bold
claim ended on "do neither", which read as if the networks did not differ in geometry and ended on a negative; it now
reads "... whereas CIFAR-5m networks disagree even on order" (26 words). MOVED.md: Delta_Pi had no main-text decoding
once '(projector)' left the row label, so one sentence in Section 5.6 now names Delta_Pi and Delta_iso (replacing the
maker's rewrite of l.1863-1866); the (d) box = range and the (b)/(e) lower-median pick rules go to the appendix. New
checks: bold claim at most 28 tokens, no clause ending on a pronoun or a bare negative, MOVED.md carries the symbols.
Every plotted value was recomputed independently from the raw Grams and embeddings (_critic/critic_v7_recompute.py).
Run: OMP_NUM_THREADS=1 nice -n 10 taskset -c 0-5 ~/hrh_v2_figs_venv/bin/python -B make_F_link.py > run.log 2>&1

v6 docstring follows.

F_link with the S2.3 row (main text Section 5.6 sec:exp-link, Figure 6). Role maker, 2026-10-10.

Message (one contrast, two rows): inside the premises of Theorem 3 (thm:link) -- the finite Gibbs model with certified
empirical risk minimizers (S2.3 Stage A, arm 0) -- InfoNCE and the spectral contrastive loss come to agree exactly on
the global order at the predicted sample size, and their scale-normalized geometry stays apart by exactly the link
g_t of the theorem; CIFAR-5m networks trained with the two losses (S2.1, outside the premises) disagree on order
criteria as well, and a monotone link removes little of the difference that an affine map leaves.

Made from the v6 F_link maker (~/hrh_v2_figs/v6/F_link/make_F_link.py, sha256 a19ccc79...3ce4, untouched). The S2.1
panels keep their data, encodings, rules and asserts; they become the bottom row (d, e, f), and only the frame moves
(key strip shared at the top, new header position, letters d-f, key entry 'InfoNCE vs spectral' because the model's
InfoNCE temperature is t = 0.3 and the networks' 0.2, which the caption states). THM holds the theorem number printed
in the headers and keys (v4: Crossing Law 1, Gauge 2, Link 3); change it and rerun if the theorems are renumbered.

Top row (S2.3 Stage A arm 0; 96 pairs per set and grid N = 2^20, 2^22, ..., 2^40; t = 0.3):
 (a) share of pairs whose exact global order T_E on the 8 probes S_ex8 agrees (cache ex8.T_E = frozen P2b curve):
     cross-objective pairs (filled INK circles), within one objective (grey band from the InfoNCE to the spectral
     share, drawn like the grey boxes of (d)); the frozen pre-step Monte Carlo prediction of p_AB (dashed INK) and the
     predicted transition N* = 2^33.75 (dotted NULL_LINE).
 (b) kernel entries (upper triangle, 32,640) of one InfoNCE run against the spectral run of the same index at N_top
     = 2^40, hexbin in the greys and on the extent of (e); the theorem's link K_B = exp((K_A - 1)/t) (INK dashed).
     Pair rule fixed before plotting: the cross pair with the lower-median inverse-link residual of (c) (rank 48 of 96).
 (c) scale-normalized discrepancy Gs_max = ||D_A/psi(D_A) - D_B/psi(D_B)||_max (psi = mean off-diagonal, Thm 3
     part 3), log scale, medians over pairs (10-90% bars hidden under the marks): across objectives (filled circles),
     after the inverse of the theorem's link is applied to the spectral run (open INK squares; frozen
     metrics.inv_link_cf; frozen P3c at 2^34 and 2^40, descriptive at the other six N, computed here), within one
     objective (grey band, InfoNCE to spectral median); the predicted limit omega_psi (INK dashed) and a slope of
     N^(-1/2) (INK dotted, placed below the data).
Bottom row: the v6 panels (a, b, c) as (d, e, f); see the v6 docstring kept below the S2.1 data block.

Critic pass (2026-10-10; maker v1 kept in _prev_pre_critic/): in (c) 'predicted' and 'link undone' stood one under
the other at the right edge and read as one phrase, with 'link undone' far from the squares. 'predicted' now sits
above the dashed limit (the axes run to the height of the '1' of (a); the y spine stops at 1) and 'link undone' sits
just right of the open square at 2^31. Every plotted S2.3 value is asserted equal to an independent numpy recompute
from the raw Grams (~/hrh_v2_figs/v6/F_link_s23_critic/critic_recompute.json). The caption (caption_v6.tex, read
here, hashed and checked for result digits) now names thm:peer for N* and the diamonds of (f). No data value changed.

Read-only inputs (code of S2.1, S2.3 and hrhv2 imported with python -B; nothing written outside this directory):
  ~/hrh_v2_stage2_20261008/results/ladders/S2_1_main/z/*.json, prereg/eps_S2_1.json, results/S2_1/s21_summary.json,
  runs/S2_1_main/S2.1_{A,B}_N500000_r<k>/feats.npz                       (bottom row, as v6)
  ~/hrh_v2_stage2_20261008/s23/results/stageA/stageA_results.json         frozen Stage A analysis (F0, P2b curve, eps)
  ~/hrh_v2_stage2_20261008/s23/results/stageA/cache/arm0_l<N>_{AB,wA,wB}_n96.json   frozen per-pair rows
  ~/hrh_v2_stage2_20261008/s23/results/stageA/review_A2/recompute_A2.json independent recompute (P3c medians, T_E)
  ~/hrh_v2_stage2_20261008/s23/results/prestep/frozen_predictions.json    MC curves, N*
  ~/hrh_v2_stage2_20261008/s23/results/stageA/N_match.json                P7 numbers for the Results paragraph
  ~/hrh_v2_stage2_20261008/s23/runs/0/l<N>/{A,B}/0_{A,B}_l<N>_r<k>.npy    Gram matrices (counterfactual, (b))
  ~/hrh_v2_stage2_20261008/s23/src/{s23common,metrics}.py                 frozen S2.3 code (inv_link_cf, upper)
  _inputs/tab_app_s21_full_excerpt.tex, _inputs/staircase_F_criteria_snapshot.py   (copies of the v6 inputs)
  ~/hrh_v2_figs/v5/_style/hrh_style_v5_1.py                               frozen style (sha256 asserted)
Writes only into this directory: F_link.pdf, F_link.png (300 dpi, print size), numbers.json.
Run: OMP_NUM_THREADS=1 nice -n 10 taskset -c 0-3 ~/hrh_v2_figs_venv/bin/python -B make_F_link.py > run.log 2>&1

v6 docstring (bottom row), kept for provenance:
  Panels (CIFAR-5m, S2.1, embedding z, cosine kernel, 2048 held-out images, N = 500,000):
   (d) Delta / eps per criterion (log x) with a '~0' floor; rows in lattice order; within one objective = NULL_FILL box
       from min to max over the 96 within pairs of the three arms; InfoNCE 0.2 vs spectral = filled INK circle
       (median, 5-95% bar hidden under it); exact link (an InfoNCE run against exp(5(K-1)) of its own Gram matrix, 64
       pairs) = open INK diamond (median), at the floor on Delta_iso and the order rows, none on Delta_Pi.
   (e) cosine-kernel entries of one InfoNCE (0.2) run against the spectral run of the same index, hexbin; best
       monotone link fitted in sample (INK solid) and the theorem's link exp(5(K_A - 1)) (INK dashed); pair = the A-B
       pair at the lower median of the cross-fitted omega (rank 32 of 64), fixed before plotting.
   (f) cumulative cross-fitted explained share: scale, + affine, + global monotone; medians over pairs for the exact
       link (dashed, diamonds), InfoNCE vs spectral (solid, filled) and InfoNCE 0.2 vs 0.5 (solid, open); the
       requirement bar for InfoNCE vs spectral at 0.5 + 0.5 * median(cumulative affine share).
"""

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
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

VARIANTS = {"make_F_link.py": ("F_link", "both"), "make_F_link_top.py": ("F_link_top", "top"),
            "make_F_link_net.py": ("F_link_net", "net")}
FIG_ID, VARIANT = VARIANTS[os.path.basename(os.path.abspath(__file__))]
assert os.path.basename(HERE) == FIG_ID, (HERE, FIG_ID)
STYLE_PATH = os.path.expanduser("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")   # v8: the frozen v8 style (chmod 444)
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_PATH, "rb").read()).hexdigest() == STYLE_SHA, "frozen v8 style module changed"
sys.path.insert(0, os.path.dirname(STYLE_PATH))
import hrh_style_v8 as S  # noqa: E402

assert S.STYLE_SHA256 == STYLE_SHA
S.use()
V6_MAKER = os.path.expanduser("~/hrh_v2_figs/v6/F_link_s23/make_F_link.py")
V7_MAKER = os.path.expanduser("~/hrh_v2_figs/v7/F_link/make_F_link.py")
THM = "3"            # number of thm:link in arxiv_v4 (Crossing Law 1, Gauge 2, Link 3); rerun if renumbered

HOME = os.path.expanduser("~")
S2 = f"{HOME}/hrh_v2_stage2_20261008"
sys.path.insert(0, f"{S2}/src")
import crit  # noqa: E402  (frozen Stage-2 criterion code; imports hrhv2 read-only)
from hrhv2 import quotients as qu, util as hu  # noqa: E402

LD = f"{S2}/results/ladders/S2_1_main/z"
SRC = {
    "eps": f"{S2}/prereg/eps_S2_1.json",
    "summary": f"{S2}/results/S2_1/s21_summary.json",
    "report": f"{S2}/results/S2_1/S2_1_report.md",
    "table_excerpt": f"{HERE}/_inputs/tab_app_s21_full_excerpt.tex",
    "staircase_convention": f"{HERE}/_inputs/staircase_F_criteria_snapshot.py",
    "crit": f"{S2}/src/crit.py",
    "quotients": f"{HOME}/hrh_v2_stage1_20261008/hrhv2/quotients.py",
}
N = 500000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p) as f:
        return json.load(f)


assert sha(STYLE_PATH) == STYLE_SHA == S.STYLE_SHA256, "frozen style module changed"
assert sha(V7_MAKER) == "4fce26cc85613f3afba9c249e49b1962a5be2e823010d90c6a730d1541a2280f", "v7 maker changed"
EPS = load(SRC["eps"])["eps"]["z"]
SUM = load(SRC["summary"])
assert SUM["interim"] is False and SUM["R"] == 64 and SUM["verdict"] == "falsified", (SUM["interim"], SUM["R"])
ASSERTS = []          # (name, result) recorded in numbers.json


def check(name, cond, detail=None):
    assert cond, (name, detail)
    ASSERTS.append({"assert": name, "ok": True, "detail": detail})


# shared staircase convention (F_criteria/staircase.py, copied to _inputs): step words and y label
import ast  # noqa: E402
_stair_consts = {}
for _node in ast.parse(open(SRC["staircase_convention"]).read()).body:
    if isinstance(_node, ast.Assign) and len(_node.targets) == 1 and isinstance(_node.targets[0], ast.Name) \
            and _node.targets[0].id in ("STEP_LABELS", "Y_LABEL"):
        _stair_consts[_node.targets[0].id] = ast.literal_eval(_node.value)
STEP_LABELS_SHARED = ("scale", "+ affine", "+ global\nmonotone", "+ per-row\nmonotone")
Y_LABEL_C = "share explained (cumulative)"
check("staircase step labels and y label equal the shared convention (F_criteria/staircase.py)",
      _stair_consts.get("STEP_LABELS") == STEP_LABELS_SHARED and _stair_consts.get("Y_LABEL") == Y_LABEL_C,
      _stair_consts)

# =============================================================================================
# S2.1 data (bottom row): pair classes by filename (500k), Delta / eps per rung
# =============================================================================================
PAT = re.compile(r"S2\.1_(A|Atau|B)_N(\d+)_r(\d+)__S2\.1_(A|Atau|B)_N(\d+)_r(\d+)(__bern)?\.json$")
FILES = {}
for f in sorted(glob.glob(LD + "/S2.1_*.json")):
    m = PAT.search(os.path.basename(f))
    if not m:
        continue                                   # fixed-epoch controls
    a, na, ra, b, nb, rb, bern = m.groups()
    if int(na) != N or int(nb) != N:
        continue
    if bern:
        assert a == b == "A"
        c = "bern_self" if ra == rb else "bern_peer"
    elif a == b:
        c = "within_" + a
    else:
        c = f"cross_{a}_{b}"
    FILES.setdefault(c, []).append(f)
N_EXPECT = {"within_A": 32, "within_B": 32, "within_Atau": 32, "cross_A_B": 64, "cross_A_Atau": 64,
            "cross_Atau_B": 64, "bern_self": 64, "bern_peer": 32}
check("pair counts per class at 500k", {k: len(v) for k, v in FILES.items()} == N_EXPECT,
      {k: len(v) for k, v in FILES.items()})
LAD = {c: [load(f) for f in fs] for c, fs in FILES.items()}
for c, rows in LAD.items():
    assert all(not r["failed_run"] for r in rows), c

# rows top -> bottom in lattice order; the separator sits between SOG and Delta_iso
ROWS = [   # v7: the short names of Table tab:app-s21-full only (v6 added '(projector)', '(isotonic)', ', tau = 0.1')
    ("quot", "dPi", r"$\Delta_\Pi$"),
    ("geom", "dCKA", r"$1-\mathrm{CKA}$"),
    ("geom", "Gs", r"$G_s$"),
    ("geom", "dAff", r"$T_{G/\mathrm{Aff}}$"),
    ("geom", "dSOG_t0p1", r"SOG$_{0.1}$"),
    ("iso", "dIso", r"$\Delta_{\mathrm{iso}}$"),
    ("order", "dE", r"$T_E$"),
    ("order", "dMST", r"$T_{\mathrm{MST}}$"),
    ("order", "dO", r"$T_O$"),
    ("order", "dN_k10", r"$T_{N,10}$"),
]
RUNGS = [r for _, r, _ in ROWS]
WITHIN = ["within_A", "within_B", "within_Atau"]
CLASS_TABLE = {"within_A": "within_A_500000|z", "within_B": "within_B_500000|z",
               "within_Atau": "within_Atau_500000|z", "cross_A_B": "cross_A_B_500000|z",
               "cross_A_Atau": "cross_A_Atau_500000|z", "cross_Atau_B": "cross_Atau_B_500000|z",
               "bern_self": "bern_self_500000|z", "bern_peer": "bern_peer_500000|z"}


def deltas(c, r):
    return np.array([row["deltas"][r] for row in LAD[c]], float)


panel_a = {"N": N, "eps_source": "eps_S2_1.json:eps.z.<rung>", "rows_top_to_bottom": RUNGS, "classes": {}}
for c in N_EXPECT:
    panel_a["classes"][c] = {}
    for r in RUNGS:
        v = deltas(c, r)
        e = EPS[r]
        t = SUM["tables"][CLASS_TABLE[c]].get(r)
        if c.startswith("bern") and r == "dPi":
            check(f"link check undefined on dPi ({c})", t is None and np.all(np.isnan(v)))
            panel_a["classes"][c][r] = {"n": int(v.size), "undefined": True}
            continue
        assert np.all(np.isfinite(v)), (c, r)
        x = int(np.sum(v <= e))
        check(f"decision count {c} {r} equals s21_summary", t is not None and t["x"] == x and t["n"] == v.size
              and abs(t["eps"] - e) == 0 and abs(t["median"] - float(np.median(v))) < 1e-12,
              {"x": x, "n": int(v.size), "table": None if t is None else [t["x"], t["n"], t["decision"]]})
        q = v / e
        panel_a["classes"][c][r] = {
            "n": int(v.size), "eps": e, "x_within_eps": x, "decision": t["decision"],
            "ratio_min": float(q.min()), "ratio_q05": float(np.quantile(q, 0.05)), "ratio_median": float(np.median(q)),
            "ratio_q95": float(np.quantile(q, 0.95)), "ratio_max": float(q.max()),
            "n_exact_zero": int(np.sum(v == 0)), "values_delta": [float(z) for z in v],
            "source_key": f"{os.path.basename(LD)}/<{c} files>:deltas.{r}"}

# Table app-s21-full (500k block) against the computed counts
TAB_ROW = {"InfoNCE at 0.2, within": "within_A", "InfoNCE at 0.5, within": "within_Atau",
           "spectral, within": "within_B", "InfoNCE at 0.2 vs spectral": "cross_A_B",
           "InfoNCE at 0.2 vs 0.5": "cross_A_Atau", "InfoNCE at 0.5 vs spectral": "cross_Atau_B",
           "link check, self": "bern_self", "link check, peer": "bern_peer"}
TAB_COLS = ["dN_k10", "dO", "dE", "dMST", "dSOG_t0p1", "dAff", "Gs", "dCKA", "dIso", "dPi"]
tex = open(SRC["table_excerpt"]).read().split("\n")
in500, tab_checked = False, 0
for ln in tex:
    if ln.startswith("500k"):
        in500 = True
    if not in500 or "&" not in ln:
        continue
    cells = [s.strip().rstrip("\\").strip() for s in ln.split("&")]
    name = cells[1]
    c = TAB_ROW[name]
    for col, cell in zip(TAB_COLS, cells[2:]):
        if cell == "n/a":
            assert c.startswith("bern") and col == "dPi"
            continue
        mm = re.match(r"(\d+)/(\d+)", cell)
        assert mm, cell
        assert int(mm.group(1)) == panel_a["classes"][c][col]["x_within_eps"], (name, col, cell)
        assert int(mm.group(2)) == panel_a["classes"][c][col]["n"], (name, col, cell)
        tab_checked += 1
check("Table tab:app-s21-full 500k block equals the computed counts", tab_checked == 8 * 10 - 2, tab_checked)

# plotted summaries of (d)
A_WITHIN = {}
for r in RUNGS:
    q = np.concatenate([deltas(c, r) / EPS[r] for c in WITHIN])
    A_WITHIN[r] = (float(q.min()), float(q.max()))
check("every within-objective pair lies within tolerance (boxes end left of the line)",
      max(v[1] for v in A_WITHIN.values()) < 1.0, max(v[1] for v in A_WITHIN.values()))
A_CLASSES = ("cross_A_B", "bern_self")                 # drawn in (d); cross_A_Atau is in (f) and in F.7
A_MARK = {}
for c in A_CLASSES:
    for r in RUNGS:
        d = panel_a["classes"][c][r]
        if d.get("undefined"):
            continue
        A_MARK[(c, r)] = d
FLOOR_ROWS = ["dIso", "dE", "dMST", "dO", "dN_k10"]           # exact link drawn at the '~0' floor
for r in FLOOR_ROWS:
    d = A_MARK[("bern_self", r)]
    assert d["ratio_max"] < 0.002, (r, d["ratio_max"])
for r in ("dMST", "dO", "dN_k10"):
    check(f"link check exactly 0 on {r}", A_MARK[("bern_self", r)]["n_exact_zero"] == 64)
check("link check on T_E: 62 of 64 exactly 0, rest <= 1e-12",
      A_MARK[("bern_self", "dE")]["n_exact_zero"] == 62 and A_MARK[("bern_self", "dE")]["ratio_max"] < 1e-11,
      A_MARK[("bern_self", "dE")]["ratio_max"])
for r in ("dCKA", "Gs", "dAff", "dSOG_t0p1"):
    check(f"link check outside tolerance on every pair on {r} (diamond right of the line)",
          A_MARK[("bern_self", r)]["ratio_min"] > 1.0, A_MARK[("bern_self", r)]["ratio_min"])
for r in RUNGS:
    check(f"InfoNCE vs spectral outside tolerance on every pair on {r}",
          A_MARK[("cross_A_B", r)]["x_within_eps"] == 0 and A_MARK[("cross_A_B", r)]["ratio_min"] > 1.0,
          A_MARK[("cross_A_B", r)]["ratio_min"])
# checked 2026-10-10 by the plan (medians of Delta/eps)
wmed = [panel_a["classes"][c][r]["ratio_median"] for c in WITHIN for r in RUNGS]
wmed_g = [panel_a["classes"][c][r]["ratio_median"] for c in WITHIN for r in RUNGS if r != "dPi"]
check("within medians 0.21-0.97 (Delta_Pi apart; with it the top is %.3f)" % max(wmed),
      (round(min(wmed_g), 2), round(max(wmed_g), 2)) == (0.21, 0.97), [min(wmed_g), max(wmed_g)])
xmed = [panel_a["classes"]["cross_A_B"][r]["ratio_median"] for r in RUNGS if r != "dPi"]
check("across-objective medians 1.21-6.24 (Delta_Pi apart)", (round(min(xmed), 2), round(max(xmed), 2)) == (1.21, 6.24),
      [min(xmed), max(xmed)])
tmed = [panel_a["classes"]["cross_A_Atau"][r]["ratio_median"] for r in RUNGS if r != "dPi"]
check("across-temperature medians 1.05-3.58 (Delta_Pi apart)", (round(min(tmed), 2), round(max(tmed), 2)) == (1.05, 3.58),
      [min(tmed), max(tmed)])
lmed = [panel_a["classes"]["bern_self"][r]["ratio_median"] for r in ("dCKA", "Gs", "dAff", "dSOG_t0p1")]
check("link check medians on geometry 1.20-4.55", (round(min(lmed), 2), round(max(lmed), 2)) == (1.20, 4.55),
      [min(lmed), max(lmed)])

# =============================================================================================
# (f) staircase data: cumulative cross-fitted shares per pair
# =============================================================================================
STEPS = ["scale", "affine_extra", "iso_extra", "row_extra"]
N_STEPS_C = 3                                       # drawn: scale, + affine, + global monotone
C_CLASSES = ["bern_self", "cross_A_B", "cross_A_Atau"]
panel_c = {"source": "ladder row.warpshare_expl_cf_{scale, affine_extra, iso_extra, row_extra} (cross-fitted, mean of "
                     "the A->B and B->A directions, hrhv2/quotients.py); cumulative sum per pair, then median and "
                     "10/90% quantiles over pairs", "steps_all": list(STEP_LABELS_SHARED),
           "steps_drawn": list(STEP_LABELS_SHARED[:N_STEPS_C]), "classes": {}}
STAIR, CUM = {}, {}
for c in C_CLASSES:
    st = np.array([[row["row"]["warpshare_expl_cf_" + s] for s in STEPS] for row in LAD[c]], float)
    res = np.array([row["row"]["warpshare_expl_cf_residual"] for row in LAD[c]], float)
    assert np.max(np.abs(st.sum(1) + res - 1.0)) < 1e-9, c
    cum = np.cumsum(st, axis=1)
    CUM[c] = cum
    STAIR[c] = {"median": np.median(cum, 0), "q10": np.quantile(cum, 0.1, 0), "q90": np.quantile(cum, 0.9, 0)}
    panel_c["classes"][c] = {"n": int(cum.shape[0]),
                             "cum_median": [float(z) for z in STAIR[c]["median"]],
                             "cum_q10": [float(z) for z in STAIR[c]["q10"]],
                             "cum_q90": [float(z) for z in STAIR[c]["q90"]],
                             "step_median": [float(z) for z in np.median(st, 0)],
                             "per_pair_cum": [[float(z) for z in r_] for r_ in cum]}
sm = {c: [round(z, 3) for z in panel_c["classes"][c]["step_median"]] for c in C_CLASSES}
check("step medians as checked 2026-10-10",
      sm["cross_A_B"] == [0.140, 0.393, 0.086, 0.013] and sm["cross_A_Atau"] == [0.011, 0.287, 0.088, 0.009]
      and sm["bern_self"][0] == 0.023 and abs(sm["bern_self"][1] - 0.656) <= 0.0015 and
      abs(sm["bern_self"][2] - 0.319) <= 0.0015 and abs(sm["bern_self"][3]) <= 0.001, sm)
check("per-row monotone step adds at most 0.014 (median) in every class, so (c) may stop at the global step",
      max(abs(panel_c["classes"][c]["step_median"][3]) for c in C_CLASSES) < 0.014,
      {c: panel_c["classes"][c]["step_median"][3] for c in C_CLASSES})

# preregistered requirement, InfoNCE vs spectral: omega >= 0.5, i.e. the link removes half of the affine residual
OM = np.array([row["deltas"]["omega"] for row in LAD["cross_A_B"]], float)
aff_cum = CUM["cross_A_B"][:, 1]
REQ = 0.5 + 0.5 * float(np.median(aff_cum))          # = median over pairs of aff_i + 0.5 (1 - aff_i) (linear map)
req_i = aff_cum + 0.5 * (1.0 - aff_cum)
check("requirement level equals the median of the per-pair levels", abs(REQ - float(np.median(req_i))) < 1e-12)
n_reach = int(np.sum(CUM["cross_A_B"][:, 2] >= req_i))
n_om = int(np.sum(OM >= 0.5))
check("no InfoNCE-vs-spectral pair reaches the requirement, on the drawn shares and on the preregistered omega",
      n_reach == 0 and n_om == 0, [n_reach, n_om])
check("the exact link reaches 1 at the global step (median >= 0.999)", STAIR["bern_self"]["median"][2] >= 0.999,
      float(STAIR["bern_self"]["median"][2]))
panel_c["requirement_cross_A_B"] = {
    "level": REQ, "rule": "0.5 + 0.5 * median over pairs of the cumulative cross-fitted share after the affine step; "
                          "the preregistered endpoint (iv) is median omega >= 0.5 with omega per pair = mean over "
                          "directions of 1 - (1 - iso_cf) / (1 - affine_cf) (crit.py deltas)",
    "median_cum_affine": float(np.median(aff_cum)), "median_cum_global_monotone": float(STAIR["cross_A_B"]["median"][2]),
    "pairs_reaching_level_on_drawn_shares": n_reach, "pairs_with_omega_ge_0.5": n_om, "median_omega": float(np.median(OM))}

# =============================================================================================
# (e) the pair with the median omega among the 64 A-B pairs
# =============================================================================================
check("median omega across objectives = 0.147 (S2_1_report item iv)", round(float(np.median(OM)), 3) == 0.147,
      float(np.median(OM)))
order = np.argsort(OM, kind="stable")
PICK = int(order[31])                         # rule: lower median, rank 32 of 64 in increasing omega
ROWB = LAD["cross_A_B"][PICK]
RUN_A, RUN_B = ROWB["A"], ROWB["B"]
assert RUN_A.split("_r")[1] == RUN_B.split("_r")[1], (RUN_A, RUN_B)     # same index
FA = f"{S2}/runs/S2_1_main/{RUN_A}/feats.npz"
FB = f"{S2}/runs/S2_1_main/{RUN_B}/feats.npz"
with np.load(FA) as z:
    XA = np.array(z["z"])
with np.load(FB) as z:
    XB = np.array(z["z"])
assert XA.shape == XB.shape == (2048, 128)
GA, GB = crit.geom(XA, "l2"), crit.geom(XB, "l2")
h1 = hu.hash_half([str(i) for i in crit.ITEM_IDS], "hrhv2-halves")
xA, xB = hu.upper_values(GA.D), hu.upper_values(GB.D)
Q = qu.global_quotients_dir(xA, xB, hu.upper_values_masked(GA.D, h1), hu.upper_values_masked(GB.D, h1),
                            hu.upper_values_masked(GA.D, ~h1), hu.upper_values_masked(GB.D, ~h1))
R = ROWB["row"]
for k in ("scale_expl", "affine_expl", "iso_expl", "scale_expl_cf", "affine_expl_cf", "iso_expl_cf"):
    check(f"(e) recomputed {k} equals row.quot_AB_{k}", abs(Q[k] - R[f"quot_AB_{k}"]) < 1e-9,
          [Q[k], R[f"quot_AB_{k}"]])
kA, kB = hu.upper_values(GA.K), hu.upper_values(GB.K)
check("(e) D = 2 - 2K on the l2 sphere", max(np.max(np.abs(xA - (2 - 2 * kA))), np.max(np.abs(xB - (2 - 2 * kB))))
      < 1e-12)
KX, KG, _ = qu.iso_fit(xA, xB)                 # knots in D_A, fitted D_B (nondecreasing)
aff_a, aff_b = Q.get("affine_slope"), Q.get("affine_intercept")
if aff_a is None:
    aff_a, aff_b = qu._fit_affine(xA, xB)
check("(e) affine coefficients equal row.quot_AB_affine_{slope,intercept}",
      abs(aff_a - R["quot_AB_affine_slope"]) < 1e-9 and abs(aff_b - R["quot_AB_affine_intercept"]) < 1e-9,
      [aff_a, aff_b])
# kernel coordinates: K = 1 - D/2; an increasing map of D_A is an increasing map of K_A
GRID = np.linspace(kA.min(), kA.max(), 400)
ISO_Y = 1.0 - qu.iso_predict(KX, KG, 2.0 - 2.0 * GRID) / 2.0
AFF_Y = 1.0 - (aff_a * (2.0 - 2.0 * GRID) + aff_b) / 2.0
assert np.all(np.diff(ISO_Y) >= -1e-12)
# the link Theorem 3 predicts: K_B = exp(beta (K_A - 1)), beta = 1/t = 5 (crit.BERN_BETA, the link-check map)
BETA = float(crit.BERN_BETA)
check("theory link uses beta = 1/t = 5 at t = 0.2 (crit.BERN_BETA, the map of the link check)", BETA == 5.0, BETA)
TGRID = np.linspace(kA.min(), 1.0, 400)
THEORY_Y = np.exp(BETA * (TGRID - 1.0))
bins = [(-1.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]
THEORY_GAP = []
for lo_, hi_ in bins:
    msk = (kA >= lo_) & (kA < hi_)
    if not np.any(msk):
        continue
    THEORY_GAP.append({"K_A_bin": [lo_, min(hi_, 1.0)], "n": int(msk.sum()),
                       "median_K_B": float(np.median(kB[msk])),
                       "theory_K_B_at_bin_median_K_A": float(np.exp(BETA * (np.median(kA[msk]) - 1.0)))})
FRAC_NEG = float(np.mean(kB < 0))
check("(e) a sizeable share of spectral kernel entries is negative (no exp link can produce them)", FRAC_NEG > 0.3,
      FRAC_NEG)
# in-sample share of the squared-distance discrepancy left after mapping D_A by the theorem's link g_t (no fit)
gt = 2.0 * (1.0 - np.exp(-BETA * xA / 2.0))          # g_t(D) = 2 (1 - exp(-D / (2t)))
SS0 = float(np.sum((xB - xA) ** 2))
SHARE_THEORY = 1.0 - float(np.sum((xB - gt) ** 2)) / SS0
panel_b = {"rule": "A-B pair at N = 500,000 with the lower-median cross-fitted omega (deltas.omega), rank 32 of 64 in "
                   "increasing omega, fixed before plotting",
           "run_A": RUN_A, "run_B": RUN_B, "feats": {"A": FA, "B": FB, "key": "z", "pre": "l2"},
           "omega_pick": float(OM[PICK]), "omega_median_64": float(np.median(OM)),
           "omega_sorted": [[LAD["cross_A_B"][i]["A"], float(OM[i])] for i in order],
           "shares_in_sample_AB": {"scale": Q["scale_expl"], "affine": Q["affine_expl"], "iso": Q["iso_expl"]},
           "shares_cf_AB": {"scale": Q["scale_expl_cf"], "affine": Q["affine_expl_cf"], "iso": Q["iso_expl_cf"]},
           "affine_on_D": {"slope": aff_a, "intercept": aff_b, "note": "not drawn; its share is step 2 of (f)"},
           "n_entries": int(kA.size), "kernel_A_range": [float(kA.min()), float(kA.max())],
           "kernel_B_range": [float(kB.min()), float(kB.max())],
           "kernel_A_q01_q99": [float(np.quantile(kA, 0.01)), float(np.quantile(kA, 0.99))],
           "kernel_B_q01_q99": [float(np.quantile(kB, 0.01)), float(np.quantile(kB, 0.99))],
           "pearson_kernel": float(np.corrcoef(kA, kB)[0, 1]),
           "iso_line": {"x": [float(z) for z in GRID], "y": [float(z) for z in ISO_Y]},
           "theory_line": {"formula": "K_B = exp(beta (K_A - 1)), beta = 1/t", "beta": BETA,
                           "source": "Theorem thm:link part 1 (K_B -> exp((K_star - 1)/t)); crit.BERN_BETA",
                           "x": [float(z) for z in TGRID], "y": [float(z) for z in THEORY_Y]},
           "theory_vs_data_by_K_A_bin": THEORY_GAP,
           "frac_spectral_entries_negative": FRAC_NEG,
           "in_sample_share_explained_by_theory_link_g_t_no_fit": SHARE_THEORY,
           "sharp_link_secondary": "S2_1_report.md Secondary: fitted beta-hat median 2.039 (A_B, 500k) against the "
                                   "predicted 5; pred_ok False"}


# =============================================================================================
# S2.3 Stage A, arm 0 (finite Gibbs model, certified ERMs; inside the premises of Theorem 3): top row
# =============================================================================================
S23 = f"{S2}/s23"
S23R = f"{S23}/results/stageA"
SRC23 = {
    "stageA_results": f"{S23R}/stageA_results.json",
    "report_stageA": f"{S23R}/REPORT_stageA.md",
    "frozen_predictions": f"{S23}/results/prestep/frozen_predictions.json",
    "review_recompute": f"{S23R}/review_A2/recompute_A2.json",
    "review_items_table": f"{S23R}/review_A2/items_table.tsv",
    "N_match": f"{S23R}/N_match.json",
    "metrics_s23": f"{S23}/src/metrics.py",
    "analyze_A": f"{S23}/src/analyze_A.py",
}
sys.path.insert(0, f"{S23}/src")
import s23common as C23  # noqa: E402  (paths and constants only; nothing is written)
import metrics as M23    # noqa: E402  (frozen S2.3 metrics: inv_link_cf, upper; read-only)
assert os.path.realpath(M23.__file__) == os.path.realpath(SRC23["metrics_s23"]), M23.__file__
assert os.path.realpath(sys.modules["crit"].__file__) == os.path.realpath(SRC["crit"]), "two crit modules"
RES23 = load(SRC23["stageA_results"])
FP23 = load(SRC23["frozen_predictions"])
REV23 = load(SRC23["review_recompute"])
NM23 = load(SRC23["N_match"])
T23 = float(FP23["cfg"]["t"])
check("S2.3 data temperature t = 0.3 (frozen cfg and s23common.T_DATA)", T23 == 0.3 == C23.T_DATA, T23)
GRID23 = [int(v) for v in RES23["grid"]["grid"]]
L0_23, LTOP_23 = int(RES23["grid"]["log2N0"]), int(RES23["grid"]["log2N_top"])
check("S2.3 arm-0 grid log2 N = 20, 22, 25, 28, 31, 34, 37, 40; N0 = 2^34, N_top = 2^40",
      GRID23 == [20, 22, 25, 28, 31, 34, 37, 40] and (L0_23, LTOP_23) == (34, 40), [GRID23, L0_23, LTOP_23])
check("S2.3 frozen verdicts: 9 of 9 F0 tests confirmed after Holm, order verdict PRESENT-order, calibration ok",
      all(v["confirmed"] for v in RES23["F0"].values()) and len(RES23["F0"]) == 9
      and RES23["verdict"]["verdict"] == "PRESENT-order" and RES23["calibration_ok"] is True,
      {k: v["confirmed"] for k, v in RES23["F0"].items()})
OMEGA = float(RES23["eps"]["omega_psi"])            # population limit of the cross-objective Gs_max (Thm 3 part 3)
EPS_GEO = float(RES23["eps"]["eps_geo"])
check("omega_psi = 0.348 and eps_geo = omega_psi / 2", round(OMEGA, 3) == 0.348 and abs(EPS_GEO - OMEGA / 2) < 1e-12,
      [OMEGA, EPS_GEO])
CACHE23 = f"{S23R}/cache"
WH23 = ("AB", "wA", "wB")


def rows23(l2, w):
    p = f"{CACHE23}/arm0_l{l2}_{w}_n96.json"
    rs = load(p)
    assert len(rs) == 96, (p, len(rs))
    return rs, p


def fnum23(v):
    return float(C23.fnum(v))


PER_N = {}
CACHE_FILES23 = {}
for l2 in GRID23:
    e = {}
    for w in WH23:
        rs, p = rows23(l2, w)
        CACHE_FILES23[f"{l2}_{w}"] = p
        te = np.array([bool(r["ex8"]["T_E"]) for r in rs])
        gsm = np.array([fnum23(r["d"]["Gs_max"]) for r in rs])
        assert np.all(np.isfinite(gsm)), (l2, w)
        e[w] = {"n": len(rs), "n_failed": int(sum(bool(r["failed"]) for r in rs)),
                "TE_agree": int(te.sum()), "TE_share": float(te.mean()),
                "Gs_max_median": float(np.median(gsm)), "Gs_max_q10": float(np.quantile(gsm, 0.1)),
                "Gs_max_q90": float(np.quantile(gsm, 0.9)), "Gs_max_mean": float(gsm.mean()),
                "Gs_max_min": float(gsm.min()), "Gs_max_max": float(gsm.max()),
                "Gs_max_within_eps_geo": int(np.sum(gsm <= EPS_GEO)),
                "pairs": [[r["a"], r["b"]] for r in rs]}
        # descriptive means of the frozen analysis (pair means over the same cache)
        dkey = {"AB": "AB", "wA": "AA", "wB": "BB"}[w]
        assert abs(RES23["descriptive"][str(l2)][dkey]["Gs_max"] - e[w]["Gs_max_mean"]) < 1e-12, (l2, w)
        # exact T_E agreement on S_ex8: equal to the frozen P2b curve (phat) and to the review's recount
        pb = RES23["secondary"]["P2b"]["curve"][str(l2)][f"ex8_T_E_{dkey}"]["phat"]
        assert abs(pb - e[w]["TE_share"]) < 1e-12, (l2, w, pb, e[w]["TE_share"])
        assert REV23["descriptive"][str(l2)]["TE8_agree"][dkey] == e[w]["TE_agree"], (l2, w)
    PER_N[l2] = e
check("T_E agreement counts per N equal the frozen P2b curve and the review recount (8 N x 3 pair sets)", True,
      {l2: [PER_N[l2][w]["TE_agree"] for w in WH23] for l2 in GRID23})
check("cross-objective exact T_E: 0/96 up to 2^28, 7/96 at 2^31, 94/96 at 2^34, 96/96 at 2^37 and 2^40",
      [PER_N[l2]["AB"]["TE_agree"] for l2 in GRID23] == [0, 0, 0, 0, 7, 94, 96, 96],
      [PER_N[l2]["AB"]["TE_agree"] for l2 in GRID23])
check("P2 at N_top: 96/96 cross pairs agree exactly on T_E (frozen F0.P2.x)",
      RES23["F0"]["P2"]["x"] == 96 == PER_N[LTOP_23]["AB"]["TE_agree"])
check("P3a: cross Gs_max within eps_geo in 0/96 at N0 and N_top (frozen F0.P3a), and at every grid N",
      RES23["F0"]["P3a"]["N0"]["x"] == 0 == RES23["F0"]["P3a"]["Ntop"]["x"]
      and all(PER_N[l2]["AB"]["Gs_max_within_eps_geo"] == 0 for l2 in GRID23),
      [PER_N[l2]["AB"]["Gs_max_within_eps_geo"] for l2 in GRID23])
check("cross Gs_max median at N_top within 0.001 of omega_psi",
      abs(PER_N[LTOP_23]["AB"]["Gs_max_median"] - OMEGA) < 1e-3, [PER_N[LTOP_23]["AB"]["Gs_max_median"], OMEGA])

# inverse-link counterfactual Gs_max(D_A, g_t^{-1}(D_B)) per cross pair at every grid N (frozen metrics.inv_link_cf).
# The frozen analysis computed it at N0 and N_top only (P3c); the other six N are descriptive, computed here.
RUNS23 = f"{S23}/runs/0"


def kpath(rid_):
    L = rid_.split("_")[1]
    l2 = int(rid_.split("_l")[1].split("_")[0])
    return f"{RUNS23}/l{l2}/{L}/{rid_}.npy"


CF = {}
for l2 in GRID23:
    cv, fr = [], []
    for a, b in PER_N[l2]["AB"]["pairs"]:
        c_, f_ = M23.inv_link_cf(np.load(kpath(a)), np.load(kpath(b)), T23)
        cv.append(c_)
        fr.append(f_)
    cv = np.array(cv, float)
    assert np.all(np.isfinite(cv)) and max(fr) == 0.0, (l2, max(fr))
    CF[l2] = cv
    PER_N[l2]["cf"] = {"median": float(np.median(cv)), "q10": float(np.quantile(cv, 0.1)),
                       "q90": float(np.quantile(cv, 0.9)), "mean": float(cv.mean()), "min": float(cv.min()),
                       "max": float(cv.max()), "frac_KB_nonpos_max": float(max(fr)),
                       "status": ("frozen P3c point (equal to the review recompute)" if l2 in (L0_23, LTOP_23)
                                  else "descriptive, computed by this maker with the frozen metrics.inv_link_cf")}
mc_rev = REV23["F0"]["P3c"]["median_c"]
check("inverse-link counterfactual medians at N0 and N_top equal the review recompute (P3c) to 1e-12",
      abs(PER_N[L0_23]["cf"]["median"] - mc_rev["N0"]) < 1e-12 and abs(PER_N[LTOP_23]["cf"]["median"] - mc_rev["Ntop"])
      < 1e-12, [PER_N[L0_23]["cf"]["median"], mc_rev])
CF_RATIO = PER_N[LTOP_23]["cf"]["median"] / PER_N[L0_23]["cf"]["median"]
check("counterfactual ratio N_top / N0 equals the frozen P3c ratio (root-N rate predicts 1/8)",
      abs(CF_RATIO - RES23["F0"]["P3c"]["ratio"]) < 1e-12, [CF_RATIO, RES23["F0"]["P3c"]["ratio"]])

# Monte Carlo prediction of p_AB(T_E, S_ex8) frozen in the pre-step, and N* (PREREG 4.3)
MCX = [round(20.0 + 0.25 * i, 2) for i in range(int((LTOP_23 - 20) / 0.25) + 1)]
MCY = [float(FP23["mc_curves"]["ex8"]["%.2f" % x]["TE_AB"]) for x in MCX]
NSTAR = float(FP23["nstar"]["log2_Nstar"])
first90 = next(x for x, y in zip(MCX, MCY) if y >= float(FP23["cfg"]["mc"]["level"]))
check("N* = 2^33.75 is the smallest MC grid N with p_AB(T_E, S_ex8) >= 0.9 (frozen nstar)",
      NSTAR == 33.75 == first90 and float(FP23["nstar"]["log2_Nstar_AA"]) == 33.75, [NSTAR, first90])
check("MC p_AB at the grid N equals the mc values stored with the P2b curve",
      all(abs(float(FP23["mc_curves"]["ex8"]["%.2f" % l2]["TE_AB"])
              - RES23["secondary"]["P2b"]["curve"][str(l2)]["ex8_T_E_AB"]["mc"]) < 1e-12 for l2 in GRID23))

# (b) kernel entries of one cross pair at N_top; rule fixed before plotting: lower median of the counterfactual
order23 = np.argsort(CF[LTOP_23], kind="stable")
PICK23 = int(order23[47])                    # rank 48 of 96 in increasing counterfactual residual
RUN_A23, RUN_B23 = PER_N[LTOP_23]["AB"]["pairs"][PICK23]
KA23, KB23 = np.load(kpath(RUN_A23)), np.load(kpath(RUN_B23))
assert KA23.shape == KB23.shape == (256, 256)
kA23, kB23 = M23.upper(KA23), M23.upper(KB23)
TGRID23 = np.linspace(min(kA23.min(), -0.2), 1.0, 400)
THEORY23 = np.exp((TGRID23 - 1.0) / T23)                   # K_B = exp((K_A - 1)/t), Theorem 3 part 1
DEV23 = float(np.max(np.abs(kB23 - np.exp((kA23 - 1.0) / T23))))
check("(b) top pair: every spectral entry lies within 0.01 of exp((K_A - 1)/t)", DEV23 < 0.01, DEV23)


def theory_share(kA, kB, t):
    """In-sample share of the squared-distance discrepancy removed by the theorem's link g_t (no fit)."""
    xA, xB = 2.0 - 2.0 * kA, 2.0 - 2.0 * kB
    g = 2.0 * (1.0 - np.exp(-xA / (2.0 * t)))
    return 1.0 - float(np.sum((xB - g) ** 2)) / float(np.sum((xB - xA) ** 2))


SHARE23_PICK = theory_share(kA23, kB23, T23)
SHARE23_ALL = []
for a, b in PER_N[LTOP_23]["AB"]["pairs"]:
    SHARE23_ALL.append(theory_share(M23.upper(np.load(kpath(a))), M23.upper(np.load(kpath(b))), T23))
SHARE23_ALL = np.array(SHARE23_ALL)
check("(b) top: the theorem's link without any fit removes > 0.999 of the squared-distance discrepancy (pick and "
      "every N_top pair)", SHARE23_PICK > 0.999 and SHARE23_ALL.min() > 0.999, [SHARE23_PICK, SHARE23_ALL.min()])
panel_top = {
    "grid_log2N": GRID23, "t": T23, "omega_psi": OMEGA, "eps_geo": EPS_GEO, "log2_Nstar": NSTAR,
    "per_N": {str(l2): {w: {k: v for k, v in PER_N[l2][w].items() if k != "pairs"} for w in WH23 + ("cf",)}
              for l2 in GRID23},
    "mc_TE_AB_curve": {"x_log2N": MCX, "p": MCY, "source": "frozen_predictions.json:mc_curves.ex8.<log2N>.TE_AB "
                                                             "(Gaussian MC, n_draws 1e6, PREREG 4.3)"},
    "cf_ratio_Ntop_over_N0": CF_RATIO,
    "panel_b_pick": {"rule": "cross pair at N_top = 2^40 with the lower-median inverse-link residual (rank 48 of 96), "
                             "fixed before plotting", "run_A": RUN_A23, "run_B": RUN_B23,
                     "cf": float(CF[LTOP_23][PICK23]), "K_A_range": [float(kA23.min()), float(kA23.max())],
                     "K_B_range": [float(kB23.min()), float(kB23.max())], "n_entries": int(kA23.size),
                     "max_abs_KB_minus_theory": DEV23, "theory_share_no_fit": SHARE23_PICK,
                     "theory_line": {"formula": "K_B = exp((K_A - 1)/t), t = 0.3",
                                     "x": [float(z) for z in TGRID23], "y": [float(z) for z in THEORY23]}},
    "theory_share_no_fit_Ntop_all_pairs": {"median": float(np.median(SHARE23_ALL)), "min": float(SHARE23_ALL.min()),
                                           "max": float(SHARE23_ALL.max())},
}

# critic cross-check (2026-10-10): every plotted S2.3 value against an independent recompute from the raw Grams
# (~/hrh_v2_figs/v6/F_link_s23_critic/critic_recompute.py; numpy only, no hrhv2 / s23 / maker code)
CRIT_P = f"{HOME}/hrh_v2_figs/v6/F_link_s23_critic/critic_recompute.json"
CRIT = load(CRIT_P)
_cmp = {}
for l2 in GRID23:
    c_ = CRIT["per_N"][str(l2)]
    e_ = PER_N[l2]
    _cmp[str(l2)] = {
        "TE_counts_cross_wA_wB": [[e_[w]["TE_agree"] for w in WH23], [c_["cross_TE"], c_["wA_TE"], c_["wB_TE"]]],
        "Gs_max_median_absdiff": max(abs(e_["AB"]["Gs_max_median"] - c_["cross_Gs_max_median"]),
                                     abs(e_["wA"]["Gs_max_median"] - c_["wA_Gs_max_median"]),
                                     abs(e_["wB"]["Gs_max_median"] - c_["wB_Gs_max_median"])),
        "cf_median_q10_q90_absdiff": max(abs(e_["cf"][k] - c_["cf_" + k]) for k in ("median", "q10", "q90")),
        "cross_within_eps_geo": [e_["AB"]["Gs_max_within_eps_geo"], c_["cross_Gs_max_within_eps_geo"]]}
CRIT_OK = (all(v["TE_counts_cross_wA_wB"][0] == v["TE_counts_cross_wA_wB"][1] for v in _cmp.values())
           and max(v["Gs_max_median_absdiff"] for v in _cmp.values()) < 1e-12
           and max(v["cf_median_q10_q90_absdiff"] for v in _cmp.values()) < 1e-12
           and all(v["cross_within_eps_geo"][0] == v["cross_within_eps_geo"][1] for v in _cmp.values())
           and abs(CRIT["omega_psi_recomputed"] - OMEGA) < 1e-12
           and abs(CRIT["per_N"]["40"]["b_pair_r026_max_abs_KB_minus_link"] - DEV23) < 1e-12
           and "0_A_l40_r%03d" % CRIT["per_N"]["40"]["b_pick_lower_median_run_index"] == RUN_A23
           and abs(CRIT["P3c_ratio_40_over_34"] - CF_RATIO) < 1e-12
           and all(abs(v["p_AB"] - v["frozen"]) < 0.0015 for v in CRIT["mc_TE_AB"].values()))
check("critic recompute (independent numpy code from the raw Grams) equals every plotted S2.3 value: T_E counts "
      "(8 N x 3 sets), Gs_max medians (8 N x 3 sets), link-undone medians and 10/90% quantiles (8 N), omega_psi, the "
      "(b) pair and its deviation from the link, the P3c ratio; own Gaussian MC of p_AB within 0.0015 of the frozen "
      "curve at 2^31, 2^33.5, 2^33.75, 2^34", CRIT_OK, _cmp)
panel_top["critic_recompute"] = {"path": CRIT_P, "sha256": sha(CRIT_P),
                                 "script": f"{HOME}/hrh_v2_figs/v6/F_link_s23_critic/critic_recompute.py",
                                 "script_sha256": sha(f"{HOME}/hrh_v2_figs/v6/F_link_s23_critic/critic_recompute.py"),
                                 "comparison": _cmp, "mc_TE_AB": CRIT["mc_TE_AB"],
                                 "P1_ratios": [CRIT["P1_ratio_A_40_over_34"], CRIT["P1_ratio_B_40_over_34"]]}


# =============================================================================================
# v8 figure. One code base draws three figures, chosen by the file name of this maker (VARIANT above):
#   both -> F_link      (a-c) the finite Gibbs model, (d-f) CIFAR-5m networks (Figure 6, fig:exp-link)
#   top  -> F_link_top  the finite Gibbs model only, panels (a-c)
#   net  -> F_link_net  the networks only, panels (a-c) = (d-f) of F_link (appendix figure)
# Every position is in figure inches from the bottom-left corner. No row headers (author, v7 Figure 6): the caption and
# the Results text name the setting of each row. One key line at the top serves the whole figure.
# =============================================================================================
DRAW_TOP = VARIANT in ("both", "top")
DRAW_BOT = VARIANT in ("both", "net")
W = S.TEXTWIDTH_IN
INK = S.INK                          # theory only: predictions, the link of the theorem, the requirement bar; frames
LW, MS = S.LW, S.MS
# ---- v8 colours, all from the semantic map of hrh_style_v8 (one hex = one meaning) -----------------------------------
C_ACROSS = S.LINK["across"]          # InfoNCE vs spectral (filled circles); 'link undone' (open squares, same hue)
C_CONTROL = S.LINK["control"]        # positive control, the exact link of Theorem 3 (open diamonds; dashed in (f))
T_ACROSS, T_CONTROL = S.LINK_TEXT["across"], S.LINK_TEXT["control"]   # text tones of the two hues (>= 4.5:1)
C_FIT = S.LINK_TEXT["across"]        # (e) best monotone link fitted across objectives: the across hue, darker so that
#                                      it reads over the YlOrBr density (white underlay); its share is step 3 of (f)
BOX_FC, BOX_EC, BOX_LW = S.NULL_FILL, S.NULL_LINE, LW["guide"]   # within one objective: neutral reference band/box
BOX_H = 3.6 / 72                                                 # box height (in = y data units of (d))
DENSITY = S.cmap_density()                                       # Tol YlOrBr steps 1-6 for the kernel densities
MEW_OPEN = MS["open_line_mew"]       # 1.1 pt edge for the coloured open markers (v7: 0.9 pt in INK)
MK = {   # marker specs shared by every panel and the key
    "cross_A_B": dict(marker="o", ms=MS["summary"], mfc=C_ACROSS, mec="white", mew=MS["dot_edge"]),
    "bern_self": dict(marker="D", ms=MS["line"], mfc="white", mec=C_CONTROL, mew=MEW_OPEN),
    "cf": dict(marker="s", ms=MS["line"], mfc="white", mec=C_ACROSS, mew=MEW_OPEN),
}
HALF_H_PT = {"cross_A_B": MS["summary"] / 2, "bern_self": MS["line"] * np.sqrt(2) / 2,
             "cf": MS["line"] / 2}   # as v7 ('D' spans sqrt(2) ms tip to tip); InfoNCE 0.2 vs 0.5 is not drawn in v8
SERIES_COL = {"cross_A_B": C_ACROSS, "cf": C_ACROSS, "bern_self": C_CONTROL}
KEY_NAME = {"within": "within one objective", "cross_A_B": "InfoNCE vs spectral",
            "bern_self": f"exact link of Theorem {THM}"}
KEY_ITEMS = {"both": ["within", "cross_A_B", "bern_self"], "top": ["within", "cross_A_B"],
             "net": ["within", "cross_A_B", "bern_self"]}[VARIANT]
C_CLASSES_DRAWN = ["bern_self", "cross_A_B"]      # (f); v8: InfoNCE at 0.2 vs 0.5 (cross_A_Atau) left out (Results text)
check("v8: (f) draws the positive control and InfoNCE vs spectral only; InfoNCE at 0.2 vs 0.5 is computed, not drawn",
      C_CLASSES_DRAWN == [c for c in C_CLASSES if c != "cross_A_Atau"] and "cross_A_Atau" in panel_c["classes"],
      C_CLASSES_DRAWN)

# ---- shared kernel-density scale: the network pair of (e) sets the log colour scale in every variant, so (b) has the
# same colours in F_link and in F_link_top; the counts are computed off-figure and asserted equal to the drawn ones ----
from matplotlib.colors import LogNorm  # noqa: E402
KLO, KHI = -0.72, 1.04
GRIDSIZE = 30


def hex_counts(x, y, norm=None):
    f_ = plt.figure(figsize=(2, 2))
    a_ = f_.add_axes([0, 0, 1, 1])
    if norm is None:
        h_ = a_.hexbin(x, y, gridsize=GRIDSIZE, extent=(KLO, KHI, KLO, KHI), bins="log", mincnt=1)
    else:
        h_ = a_.hexbin(x, y, gridsize=GRIDSIZE, extent=(KLO, KHI, KLO, KHI), mincnt=1, norm=norm)
    out = (np.asarray(h_.get_array(), float), np.asarray(h_.get_offsets(), float), h_.norm)
    plt.close(f_)
    return out


HEX_COUNTS, HEX_XY, _norm_e = hex_counts(kA, kB)
HEX_NORM = [float(_norm_e.vmin), float(_norm_e.vmax)]
HEX_COUNTS_T, HEX_XY_T, _ = hex_counts(kA23, kB23, LogNorm(vmin=HEX_NORM[0], vmax=HEX_NORM[1]))
check("(e) every entry inside the hexbin extent", kA.min() > KLO and kA.max() < KHI and kB.min() > KLO and
      kB.max() < KHI)
check("(e) hexbin holds every entry (counts sum to the number of entries)",
      int(round(float(np.sum(HEX_COUNTS)))) == kA.size, [float(np.sum(HEX_COUNTS)), int(kA.size)])
check("(e) hexbin colours are a log norm (bins='log')", isinstance(_norm_e, LogNorm), type(_norm_e).__name__)
check("(b) every entry inside the hexbin extent (same extent as (e))",
      kA23.min() > KLO and kA23.max() < KHI and kB23.min() > KLO and kB23.max() < KHI)
check("(b) hexbin holds every entry", int(round(float(np.sum(HEX_COUNTS_T)))) == kA23.size,
      [float(np.sum(HEX_COUNTS_T)), int(kA23.size)])
check("v8: (b) shares the log colour scale of (e) (same count, same colour); its counts lie inside that scale",
      HEX_COUNTS_T.max() <= HEX_NORM[1] and HEX_COUNTS_T.min() >= HEX_NORM[0],
      {"top": [float(HEX_COUNTS_T.min()), float(HEX_COUNTS_T.max()), int(len(HEX_COUNTS_T))], "norm": HEX_NORM})

# ---- vertical geometry -------------------------------------------------------------------------------------------
PITCH0, GG0 = 0.137, 0.035          # (d) row pitch and group gap (v7)
AX_BOT_V7 = 0.36
AX_TOP_V7 = AX_BOT_V7 + 10 * PITCH0 + 3 * GG0
LET_DY = 0.13                        # panel letters: top edge this far above the top of their row's axes (v7)
KEY_DY = 0.19                        # key-line baseline above the top of the uppermost axes (v7 header baseline)
TOP_PAD = 0.10                       # figure top above the key-line baseline (v7)
ROW_GAP = 0.135                      # v8 F_link: bottom-row letter tops to the top-row x-label baseline (v7: 0.40
#                                      from the bottom axes top to that baseline, with the bottom header in between)
XLAB_DY = 0.31                       # top-row x-label baseline to the top-row axes bottom (v7)
TH = {"both": 1.05, "top": 1.20, "net": None}[VARIANT]    # top-row axes height = the (b) square
if DRAW_BOT:
    AX_BOT, AX_TOP = AX_BOT_V7, AX_TOP_V7
else:
    AX_BOT = AX_TOP = None
if VARIANT == "both":
    XLB_T = AX_TOP + LET_DY + ROW_GAP
    TY0 = XLB_T + XLAB_DY
elif VARIANT == "top":
    TY0 = AX_BOT_V7                  # same bottom margin as the bottom row of F_link (tick labels and x label)
    XLB_T = TY0 - XLAB_DY
else:
    XLB_T = TY0 = None
TY1 = TY0 + TH if DRAW_TOP else None
UPPER = TY1 if DRAW_TOP else AX_TOP
H = UPPER + KEY_DY + TOP_PAD

fig = plt.figure(figsize=(W, H))
fig.set_layout_engine("none")


def axes_in(x0, y0, w, h, **kw):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


ov = axes_in(0, 0, W, H, zorder=5)
ov.set_xlim(0, W)
ov.set_ylim(0, H)
ov.axis("off")
ov.set_xticks([])                    # v8: no tick objects on the overlay (S.finalize would create and bold them)
ov.set_yticks([])
ov.patch.set_alpha(0)


def ext(t):
    return t.get_window_extent().transformed(fig.dpi_scale_trans.inverted())


def redraw():
    """v8: bold tick labels and axis titles (S.finalize) before any measurement of text extents, then draw."""
    S.finalize(fig)
    fig.canvas.draw()


# ---- x geometry (v7, shared by every variant; the (d) row labels set the left gutter) -----------------------------
X_GRP = 0.075                        # centre of the rotated group labels
X_BR = 0.155                         # group bracket spine
X_LAB0 = 0.215                       # leftmost start of a bracketed row label
BW = 1.05                            # (e) square (and (b) in F_link)
CW_TARGET = 1.25                     # width of the (f) axes
FW_, BRK = 0.13, 0.05                # floor axes ('~0') and the gap to the log axes
B_DEC, C_DEC, R_MARGIN = 0.56, 0.47, 0.09
XLO, XHI = 0.15, 7.5
_probe = [S.text(ov, 0, 0, lab, "label", ha="left", va="baseline") for g, _, lab in ROWS if g in ("geom", "order")]
fig.canvas.draw()
XL = X_LAB0 + max(ext(t).width for t in _probe)
for t in _probe:
    t.remove()
FX0 = XL + 0.06
AX0 = FX0 + FW_ + BRK
AXW = W - R_MARGIN - CW_TARGET - C_DEC - BW - B_DEC - AX0
assert AXW > 1.13, AXW
BX0 = AX0 + AXW + B_DEC
CX0 = BX0 + BW + C_DEC
CW_ = W - R_MARGIN - CX0
assert abs(CW_ - CW_TARGET) < 1e-9 and CW_ > 1.05, CW_
if VARIANT == "top":                 # F_link_top: (b) is a square of the taller row; (a) and (c) share the rest
    TBW = TH
    TC_W_GOAL, TC_OFF = 1.10, 0.60   # (c) width aimed at; (b) right edge to the (c) axes (v7: 0.59)
    TBX0 = W - R_MARGIN - TC_W_GOAL - TC_OFF - TBW
else:
    TBW, TBX0 = BW, BX0
TCX0 = TBX0 + TBW + C_DEC
TCW = W - R_MARGIN - TCX0
if DRAW_TOP:
    assert abs(TBW - TH) < 1e-12

LAB_C, LAB_TA, LAB_TC, LAB_TB, LEG_B = {}, {}, {}, None, []
NUDGE, BOX_CLEAR, ROW_TXT, GROUP_TXT, BRACKETS = {}, {}, {}, {}, {}
C_DX, TC_DODGE = {}, {}

# =============================================================================================
# bottom row: CIFAR-5m networks (F_link d-f; F_link_net a-c)
# =============================================================================================
if DRAW_BOT:
    # ---- (d) Delta / eps ---------------------------------------------------------------------------------------
    GG = GG0
    groups = [g for g, _, _ in ROWS]
    n_gaps = sum(1 for i in range(1, len(groups)) if groups[i] != groups[i - 1])
    PITCH = (AX_TOP - AX_BOT - n_gaps * GG) / len(ROWS)
    assert n_gaps == 3 and abs(PITCH - PITCH0) < 1e-9, (n_gaps, PITCH)
    row_y, y = {}, AX_TOP - PITCH / 2
    for i, (g, r, _) in enumerate(ROWS):
        if i and g != groups[i - 1]:
            y -= GG
        row_y[r] = y
        y -= PITCH
    ax_f = axes_in(FX0, AX_BOT, FW_, AX_TOP - AX_BOT)
    ax_a = axes_in(AX0, AX_BOT, AXW, AX_TOP - AX_BOT)
    for ax in (ax_f, ax_a):
        ax.set_ylim(AX_BOT, AX_TOP)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    ax_a.set_xscale("log")
    ax_a.set_xlim(XLO, XHI)
    ax_a.xaxis.set_major_locator(mticker.FixedLocator([0.2, 0.5, 1, 2, 5]))
    ax_a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{v:g}"))
    ax_a.xaxis.set_minor_locator(mticker.LogLocator(subs=np.arange(2, 10)))
    ax_a.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_f.set_xlim(0, 1)
    ax_f.set_xticks([0.5], ["≈0"])
    for r in RUNGS:                                      # row guides
        for ax in (ax_f, ax_a):
            ax.axhline(row_y[r], color=S.GUIDE, lw=LW["guide"], zorder=S.Z["guide"])
    ax_a.axvline(1.0, color=S.NULL_LINE, lw=LW["ref"], zorder=S.Z["ref"])     # tolerance
    y_sep = (row_y["dSOG_t0p1"] + row_y["dIso"]) / 2     # Theorem 3: disagreement above, agreement below
    for ax, (x0, x1) in ((ax_f, (0, 1)), (ax_a, (XLO, XHI))):
        ax.plot([x0, x1], [y_sep, y_sep], color=S.NULL_LINE, lw=LW["level"], solid_capstyle="butt",
                zorder=S.Z["ref"])
    for xb in (FX0 + FW_, AX0):                          # // marks on the x spine between floor and log axes
        ov.plot([xb - 0.022, xb + 0.022], [AX_BOT - 0.03, AX_BOT + 0.03], color=INK, lw=LW["axis_break"],
                solid_capstyle="butt", zorder=6)
    PT_PER_DEC = AXW * 72 / np.log10(XHI / XLO)
    for r in RUNGS:
        yr = row_y[r]
        lo, hi = A_WITHIN[r]
        ax_a.add_patch(Rectangle((lo, yr - BOX_H / 2), hi - lo, BOX_H, facecolor=BOX_FC, edgecolor=BOX_EC,
                                 lw=BOX_LW, zorder=S.Z["interval"]))
        # marks on the row line; nudge apart vertically only where the filled circle and the diamond would touch
        dy = {"cross_A_B": 0.0, "bern_self": 0.0}
        if ("bern_self", r) in A_MARK and r not in FLOOR_ROWS:
            sep_pt = abs(np.log10(A_MARK[("bern_self", r)]["ratio_median"] / A_MARK[("cross_A_B", r)]["ratio_median"])) \
                * PT_PER_DEC
            need_pt = HALF_H_PT["cross_A_B"] + HALF_H_PT["bern_self"] + 0.4
            if sep_pt < need_pt:
                off = (need_pt + 0.2) / 2 / 72
                dy = {"bern_self": +off, "cross_A_B": -off}
                NUDGE[r] = {"sep_pt": float(sep_pt), "offset_pt": float(off * 72)}
        d = A_MARK[("cross_A_B", r)]
        yy = yr + dy["cross_A_B"]
        ax_a.plot([d["ratio_q05"], d["ratio_q95"]], [yy, yy], color=C_ACROSS, lw=LW["errorbar"],
                  solid_capstyle="butt", zorder=S.Z["connector"])
        ax_a.plot([d["ratio_median"]], [yy], ls="none", zorder=S.Z["marker"] + 0.5, **MK["cross_A_B"])
        if ("bern_self", r) in A_MARK:
            d = A_MARK[("bern_self", r)]
            yy = yr + dy["bern_self"]
            if r in FLOOR_ROWS:
                ax_f.plot([0.5], [yy], ls="none", zorder=S.Z["marker"], **MK["bern_self"])
            else:
                ax_a.plot([d["ratio_median"]], [yy], ls="none", zorder=S.Z["marker"], **MK["bern_self"])
    check("marks are nudged only on the SOG row (the filled circle and the diamond coincide there)",
          list(NUDGE) == ["dSOG_t0p1"], NUDGE)
    for (c, r), d in A_MARK.items():
        if c == "bern_self" and r in FLOOR_ROWS:
            continue
        gap_pt = np.log10(d["ratio_median"] / A_WITHIN[r][1]) * PT_PER_DEC - HALF_H_PT[c]
        BOX_CLEAR[f"{c}|{r}"] = float(gap_pt)
    for (c, r), d in A_MARK.items():
        if c == "bern_self" and r in FLOOR_ROWS:
            continue
        assert XLO < d["ratio_q05"] and d["ratio_q95"] < XHI, (c, r)
    assert XLO < min(v[0] for v in A_WITHIN.values()) and max(v[1] for v in A_WITHIN.values()) < XHI
    ROW_TXT = {r: S.text(ov, XL, row_y[r], lab, "label", ha="right", va="center_baseline") for g, r, lab in ROWS}
    GROUP = {"geom": "geometry", "order": "order"}
    for g, lab in GROUP.items():
        ys = [row_y[r] for gg, r, _ in ROWS if gg == g]
        lo, hi = min(ys) - PITCH * 0.4, max(ys) + PITCH * 0.4
        ov.plot([X_BR + 0.035, X_BR, X_BR, X_BR + 0.035], [lo, lo, hi, hi], color=INK, lw=LW["bracket"],
                solid_joinstyle="miter", solid_capstyle="butt")
        BRACKETS[g] = (lo, hi)
        GROUP_TXT[g] = S.text(ov, X_GRP, (lo + hi) / 2, lab, "header", rotation=90, ha="center", va="center")
    XLAB_A = S.text(ov, (FX0 + AX0 + AXW) / 2, 0.06, "discrepancy / tolerance", "label", ha="center", va="baseline")

    # ---- (e) kernel entries of one pair -----------------------------------------------------------------------
    BY0 = AX_BOT
    ax_b = axes_in(BX0, BY0, BW, BW)
    hb = ax_b.hexbin(kA, kB, gridsize=GRIDSIZE, extent=(KLO, KHI, KLO, KHI), bins="log", mincnt=1, cmap=DENSITY,
                     linewidths=0.25, edgecolors="face", rasterized=True, zorder=S.Z["band"])
    check("v8: (e) drawn hexbin equals the off-figure counts and log scale (YlOrBr density)",
          np.array_equal(np.asarray(hb.get_array(), float), HEX_COUNTS)
          and [float(hb.norm.vmin), float(hb.norm.vmax)] == HEX_NORM and isinstance(hb.norm, LogNorm))
    S.underlay(ax_b, TGRID, THEORY_Y, LW["theory"], S.Z["data"])
    ax_b.plot(TGRID, THEORY_Y, color=INK, lw=LW["theory"], ls=S.LS["dashed"], zorder=S.Z["data"])
    S.underlay(ax_b, GRID, ISO_Y, LW["data"], S.Z["data"] + 0.5)
    ax_b.plot(GRID, ISO_Y, color=C_FIT, lw=LW["data"], zorder=S.Z["data"] + 0.5)
    ax_b.set_xlim(KLO, KHI)
    ax_b.set_ylim(KLO, KHI)
KFMT = mticker.FuncFormatter(lambda v, p: f"{v:g}" if v in (0, 1) else S.num(v, 1))
KX_LAB, KY_LAB = r"InfoNCE $\hat K_A$", r"spectral $\hat K_B$"
if DRAW_BOT:
    for axis in (ax_b.xaxis, ax_b.yaxis):
        axis.set_major_locator(mticker.FixedLocator([0, 0.5, 1]))
        axis.set_major_formatter(KFMT)
    ax_b.set_xlabel(KX_LAB)
    ax_b.set_ylabel(KY_LAB)
    # key of the two lines in the band above the square, flush with the (e) y label (v7)
    redraw()
    XK_B = ext(ax_b.yaxis.label).x0
    Y_KEY_E = (BY0 + BW + 0.205, BY0 + BW + 0.065)      # baselines of the two key lines above the square
    for j, (lab, ls, lw, col) in enumerate(((f"link of Theorem {THM}", S.LS["dashed"], LW["theory"], INK),
                                            ("best monotone link", "-", LW["data"], C_FIT))):
        yk = Y_KEY_E[j]
        ov.plot([XK_B, XK_B + 0.22], [yk + 0.012] * 2, color=col, lw=lw, ls=ls, solid_capstyle="butt",
                dash_capstyle="butt")
        LEG_B.append(S.text(ov, XK_B + 0.28, yk, lab, "key", ha="left", va="baseline"))

    # ---- (f) staircase (shared convention: step words, y label, dotted full-share line) ------------------------
    ax_c = axes_in(CX0, AX_BOT, CW_, AX_TOP - AX_BOT)
    XS_C = np.arange(1, N_STEPS_C + 1, dtype=float)
    C_XLIM = (0.70, N_STEPS_C + 0.36)
    C_YLIM = (-0.06, 1.10)
    C_STYLE = {"bern_self": dict(color=C_CONTROL, lw=LW["connector"], ls=S.LS["dashed"], dash_capstyle="butt"),
               "cross_A_B": dict(color=C_ACROSS, lw=LW["data"], ls="-")}
    C_DODGE_PT = {"cross_A_B": 0.0, "bern_self": +3.0}   # exact link right of vs spectral at the first step (v7)
    ax_c.set_xlim(*C_XLIM)
    ax_c.set_ylim(*C_YLIM)
    fig.canvas.draw()
    X_PER_PT = (C_XLIM[1] - C_XLIM[0]) / (CW_ * 72)
    Y_PER_PT = (C_YLIM[1] - C_YLIM[0]) / ((AX_TOP - AX_BOT) * 72)
    ax_c.plot(list(C_XLIM), [1.0, 1.0], color=S.NULL_LINE, lw=LW["ref"], ls=S.LS["dotted"],
              zorder=S.Z["thin"] + 0.5, dash_capstyle="butt")
    for c in C_CLASSES_DRAWN:
        dx = C_DODGE_PT[c] * X_PER_PT
        xs = XS_C + dx
        med = STAIR[c]["median"][:N_STEPS_C]
        if c == "cross_A_B":
            S.underlay(ax_c, xs, med, LW["data"], S.Z["data"])
        ax_c.plot(xs, med, zorder=S.Z["data"], **C_STYLE[c])
        for j in range(N_STEPS_C):
            ax_c.plot([xs[j], xs[j]], [STAIR[c]["q10"][j], STAIR[c]["q90"][j]], color=SERIES_COL[c],
                      lw=LW["errorbar"], solid_capstyle="butt", zorder=S.Z["connector"])
        ax_c.plot(xs, med, ls="none", zorder=S.Z["marker"] + (0.5 if c == "cross_A_B" else 0), **MK[c])
        C_DX[c] = float(dx)
    # preregistered requirement for InfoNCE vs spectral, at the global monotone step (theory: INK)
    x_req = XS_C[2] + C_DX["cross_A_B"]
    REQ_HALF = 5.0 * X_PER_PT
    y_mark_top = STAIR["cross_A_B"]["median"][2] + (HALF_H_PT["cross_A_B"] + 0.8) * Y_PER_PT
    ax_c.plot([x_req, x_req], [y_mark_top, REQ], color=S.NULL_LINE, lw=LW["leader"], ls=S.LS["dotted"],
              zorder=S.Z["connector"], dash_capstyle="butt")
    ax_c.plot([x_req - REQ_HALF, x_req + REQ_HALF], [REQ, REQ], color=INK, lw=LW["frame"], solid_capstyle="butt",
              zorder=S.Z["marker"])
    ax_c.set_xticks(XS_C, list(STEP_LABELS_SHARED[:N_STEPS_C]))
    for t in ax_c.get_xticklabels():
        t.set_linespacing(1.0)
        t.set_verticalalignment("top")
    ax_c.minorticks_off()
    ax_c.spines["left"].set_bounds(0, 1)
    ax_c.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax_c.set_ylabel(Y_LABEL_C)
    ax_c.yaxis.label.set_y((0.5 - C_YLIM[0]) / (C_YLIM[1] - C_YLIM[0]))     # centred on the bounded spine [0, 1]
    Y_REQKEY = 0.90
    X_REQKEY = C_XLIM[0] + 0.10
    ax_c.plot([X_REQKEY, X_REQKEY + 2 * REQ_HALF], [Y_REQKEY + 1.2 * Y_PER_PT] * 2, color=INK, lw=LW["frame"],
              solid_capstyle="butt", zorder=S.Z["marker"])
    LAB_C = {
        "bern_self": S.text(ax_c, 1.62, 0.62, "exact\nlink", "direct", color=T_CONTROL, ha="right", va="baseline",
                            linespacing=1.0),
        "cross_A_B": S.text(ax_c, 3.15, 0.435, "vs spectral", "direct", color=T_ACROSS, ha="right", va="baseline"),
        "req": S.text(ax_c, X_REQKEY + 2 * REQ_HALF + 3.0 * X_PER_PT, Y_REQKEY, "required", "key", ha="left",
                      va="baseline"),
    }

# =============================================================================================
# top row: S2.3 Stage A, the finite Gibbs model (F_link a-c; F_link_top a-c)
# =============================================================================================
XL_T = (18.3, 41.0)                  # log2 N
GX = np.array(GRID23, float)
MATH_TICK = S.FS["math"]             # tick labels holding a superscript are set at 8 pt, so scripts are 7.0 pt


def within_band(ax, lo, hi, x=GX):
    """'within one objective' in the top row: the range between the InfoNCE and the spectral value, drawn like the
    grey boxes of (d) (NULL_FILL, NULL_LINE edge) and never thinner than a box (BOX_H)."""
    lw_in = BOX_H * 72
    for yv in (lo, hi):
        ax.plot(x, yv, color=BOX_EC, lw=lw_in + 2 * BOX_LW, solid_capstyle="round", solid_joinstyle="round",
                zorder=S.Z["band"])
    ax.fill_between(x, lo, hi, color=BOX_FC, lw=0, zorder=S.Z["band"] + 0.1)
    for yv in (lo, hi):
        ax.plot(x, yv, color=BOX_FC, lw=lw_in, solid_capstyle="round", solid_joinstyle="round",
                zorder=S.Z["band"] + 0.2)


def log2_ticks(ax, majors, minors):
    ax.set_xlim(*XL_T)
    ax.set_xticks(majors, [S.pow_label(2, v) for v in majors])
    ax.xaxis.set_minor_locator(mticker.FixedLocator(minors))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=MATH_TICK)
    ax.set_xlabel(r"sample size $N$")


TE_W_LO = np.array([min(PER_N[l]["wA"]["TE_share"], PER_N[l]["wB"]["TE_share"]) for l in GRID23])
TE_W_HI = np.array([max(PER_N[l]["wA"]["TE_share"], PER_N[l]["wB"]["TE_share"]) for l in GRID23])
TE_AB = np.array([PER_N[l]["AB"]["TE_share"] for l in GRID23])
GS_W_LO = np.array([min(PER_N[l]["wA"]["Gs_max_median"], PER_N[l]["wB"]["Gs_max_median"]) for l in GRID23])
GS_W_HI = np.array([max(PER_N[l]["wA"]["Gs_max_median"], PER_N[l]["wB"]["Gs_max_median"]) for l in GRID23])
GS_AB = {k: np.array([PER_N[l]["AB"]["Gs_max_" + k] for l in GRID23]) for k in ("median", "q10", "q90")}
CF_T = {k: np.array([PER_N[l]["cf"][k] for l in GRID23]) for k in ("median", "q10", "q90")}
TA_YLIM = (-0.06, 1.10)
TC_TOP_FRAC = (1.0 - TA_YLIM[0]) / (TA_YLIM[1] - TA_YLIM[0])
TC_YLIM = (1e-4, 10 ** (np.log10(1.0 / 1e-4) / TC_TOP_FRAC - 4.0))
REF_X = np.array([22.0, 30.0])
REF_Y0 = 0.012
REF_Y = REF_Y0 * 2.0 ** (-(REF_X - REF_X[0]) / 2.0)
TA0 = TAW = TC_X0 = TC_W = None
GX_AB, GX_CF = GX.copy(), GX.copy()
TC_PRED_Y = TC_CF_X = TC_CF_Y = None
if DRAW_TOP:
    # ---- (a) exact agreement on the global order T_E over the eight probes ----------------------------------
    TA0 = 0.40                                   # provisional; moved so that the y label starts at x = 0.01 (v7)
    TA_GAP = 0.60                                # (a) right edge to the (b) square (v7)
    TAW = TBX0 - TA_GAP - TA0
    ax_ta = axes_in(TA0, TY0, TAW, TH)
    ax_ta.set_ylim(*TA_YLIM)
    log2_ticks(ax_ta, [20, 25, 30, 35, 40], [22, 28, 31, 34, 37])   # minor ticks at the other grid N
    ax_ta.spines["left"].set_bounds(0, 1)
    ax_ta.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    TA_YLAB = r"share agreeing on $T_E$"
    ax_ta.set_ylabel(TA_YLAB, **S.text_kw("label", TA_YLAB))
    ax_ta.yaxis.label.set_y((0.5 - TA_YLIM[0]) / (TA_YLIM[1] - TA_YLIM[0]))
    redraw()
    _dx = 0.01 - min(ext(ax_ta.yaxis.label).x0, min(ext(t).x0 for t in ax_ta.get_yticklabels() if t.get_text()))
    TA0 = TA0 + _dx
    TAW = TBX0 - TA_GAP - TA0
    ax_ta.set_position([TA0 / W, TY0 / H, TAW / W, TH / H])
    redraw()
    within_band(ax_ta, TE_W_LO, TE_W_HI)
    ax_ta.plot([NSTAR, NSTAR], [0.0, 1.0], color=S.NULL_LINE, lw=LW["ref"], ls=S.LS["dotted"], dash_capstyle="butt",
               zorder=S.Z["ref"])
    ax_ta.plot(MCX, MCY, color=INK, lw=LW["theory"], ls=S.LS["dashed"], dash_capstyle="butt", zorder=S.Z["data"])
    ax_ta.plot(GX, TE_AB, ls="none", zorder=S.Z["marker"], **MK["cross_A_B"])
    LAB_TA = {
        "nstar": S.text(ax_ta, NSTAR + 0.45, 0.20, r"$N^{*}$", "annot", ha="left", va="baseline"),
        "pred": S.text(ax_ta, 31.9, 0.60, "predicted", "direct", ha="right", va="baseline"),
    }

    # ---- (b) kernel entries of one cross pair at N_top ------------------------------------------------------
    ax_tb = axes_in(TBX0, TY0, TBW, TBW)
    hb_t = ax_tb.hexbin(kA23, kB23, gridsize=GRIDSIZE, extent=(KLO, KHI, KLO, KHI), mincnt=1, cmap=DENSITY,
                        linewidths=0.25, edgecolors="face", rasterized=True, zorder=S.Z["band"],
                        norm=LogNorm(vmin=HEX_NORM[0], vmax=HEX_NORM[1]))
    check("v8: (b) drawn hexbin equals the off-figure counts on the shared log scale (YlOrBr density)",
          np.array_equal(np.asarray(hb_t.get_array(), float), HEX_COUNTS_T)
          and [float(hb_t.norm.vmin), float(hb_t.norm.vmax)] == HEX_NORM)
    ax_tb.plot(TGRID23, THEORY23, color=INK, lw=LW["ref"], ls=S.LS["dashed"], zorder=S.Z["data"])
    ax_tb.set_xlim(KLO, KHI)
    ax_tb.set_ylim(KLO, KHI)
    for axis in (ax_tb.xaxis, ax_tb.yaxis):
        axis.set_major_locator(mticker.FixedLocator([0, 0.5, 1]))
        axis.set_major_formatter(KFMT)
    ax_tb.set_xlabel(KX_LAB)
    ax_tb.set_ylabel(KY_LAB)
    XLINK_T = 0.62                                         # leader end on the curve (K_A)
    LAB_TB = S.text(ax_tb, -0.62, 0.86, f"link of\nTheorem {THM}", "direct", ha="left", va="top", linespacing=1.0)

    # ---- (c) scale-normalized geometry across N: across objectives, after undoing the link, within --------------
    ax_tc = axes_in(TCX0, TY0, TCW, TH)
    ax_tc.set_yscale("log")
    ax_tc.set_ylim(*TC_YLIM)
    ax_tc.spines["left"].set_bounds(1e-4, 1.0)
    log2_ticks(ax_tc, [20, 30, 40], [25, 35])
    ax_tc.yaxis.set_major_locator(mticker.FixedLocator([1e-4, 1e-2, 1.0]))
    ax_tc.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, p: "$1$" if abs(v - 1) < 1e-12 else S.pow_label(10, int(round(np.log10(v))))))
    ax_tc.yaxis.set_minor_locator(mticker.FixedLocator([1e-3, 1e-1]))
    ax_tc.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_tc.tick_params(axis="y", which="major", labelsize=MATH_TICK)
    TC_YLAB = "normalized difference"
    ax_tc.set_ylabel(TC_YLAB)
    ax_tc.yaxis.label.set_y(TC_TOP_FRAC / 2)          # centred on the bounded spine [1e-4, 1], as in (a)
    redraw()
    b_right = max([ext(t).x1 for t in ax_tb.get_xticklabels() if t.get_text()] + [TBX0 + TBW])
    tc_left_ext = TCX0 - min(ext(ax_tc.yaxis.label).x0,
                             min(ext(t).x0 for t in ax_tc.get_yticklabels() if t.get_text()))
    TC_X0 = max(TCX0, b_right + 0.125 + tc_left_ext)
    TC_W = W - R_MARGIN - TC_X0
    ax_tc.set_position([TC_X0 / W, TY0 / H, TC_W / W, TH / H])
    redraw()
    check("(c) keeps at least 0.9 in of width after the gutter", TC_W >= 0.9, [TC_X0, TC_W, tc_left_ext])
    within_band(ax_tc, GS_W_LO, GS_W_HI)
    ax_tc.plot(list(XL_T), [OMEGA, OMEGA], color=INK, lw=LW["theory"], ls=S.LS["dashed"], dash_capstyle="butt",
               zorder=S.Z["ref"] + 0.5)
    PT_PER_DEC_TC = TH * 72 / np.log10(TC_YLIM[1] / TC_YLIM[0])
    X_PER_PT_TC = (XL_T[1] - XL_T[0]) / (TC_W * 72)
    NEED_TC = HALF_H_PT["cross_A_B"] + HALF_H_PT["cf"] + 0.5
    for j, l2 in enumerate(GRID23):
        sep_pt = abs(np.log10(GS_AB["median"][j] / CF_T["median"][j])) * PT_PER_DEC_TC
        if sep_pt < NEED_TC:
            dx_pt = np.sqrt(NEED_TC ** 2 - sep_pt ** 2) / 2 + 0.1
            GX_AB[j] += dx_pt * X_PER_PT_TC
            GX_CF[j] -= dx_pt * X_PER_PT_TC
            TC_DODGE[str(l2)] = {"vertical_sep_pt": float(sep_pt), "dodge_each_pt": float(dx_pt),
                                 "dodge_each_log2N": float(dx_pt * X_PER_PT_TC)}
    check("(c) marks are dodged only at the smallest N, where the two series cross", set(TC_DODGE) <= {"20", "22"},
          TC_DODGE)
    for key_, arr, xs_ in (("cross_A_B", GS_AB, GX_AB), ("cf", CF_T, GX_CF)):
        for j in range(len(GX)):
            ax_tc.plot([xs_[j], xs_[j]], [arr["q10"][j], arr["q90"][j]], color=SERIES_COL[key_], lw=LW["errorbar"],
                       solid_capstyle="butt", zorder=S.Z["connector"])
        ax_tc.plot(xs_, arr["median"], ls="none", zorder=S.Z["marker"] + (0.5 if key_ == "cross_A_B" else 0),
                   **MK[key_])
    ax_tc.plot(REF_X, REF_Y, color=INK, lw=LW["ref"], ls=S.LS["dotted"], dash_capstyle="butt", zorder=S.Z["ref"])
    TC_PRED_Y = OMEGA * 10 ** ((HALF_H_PT["cross_A_B"] + 1.5 + 0.22 * S.FS["direct"]) / PT_PER_DEC_TC)
    TC_CF_J = GRID23.index(31)
    TC_CF_X = GX_CF[TC_CF_J] + (HALF_H_PT["cf"] + 2.0) * X_PER_PT_TC
    TC_CF_Y = 0.05
    LAB_TC = {
        "pred": S.text(ax_tc, 40.9, TC_PRED_Y, "predicted", "direct", ha="right", va="baseline"),
        "cf": S.text(ax_tc, TC_CF_X, TC_CF_Y, "link\nundone", "direct", color=T_ACROSS, ha="left", va="center",
                     linespacing=1.0),
        "ref": S.text(ax_tc, 19.6, 1.9e-4, r"$\propto N^{-1/2}$", "annot", ha="left", va="baseline"),
    }

# ---- one key line at the top, centred (v8: no row headers) ----------------------------------------------------------
Y_KEY = UPPER + KEY_DY
SWW = {"within": 0.17, "bern_self": 0.07, "cross_A_B": 0.07}
SW_GAP, ENT_GAP = 0.05, 0.24
key_txt = {k: S.text(ov, 0, Y_KEY, KEY_NAME[k], "key", ha="left", va="baseline") for k in KEY_ITEMS}
fig.canvas.draw()
KEY_TOTAL = sum(SWW[k] + SW_GAP + ext(key_txt[k]).width for k in KEY_ITEMS) + ENT_GAP * (len(KEY_ITEMS) - 1)
kx = (W - KEY_TOTAL) / 2
KEY_GEOM = {"baseline_in": float(Y_KEY), "x0_in": float(kx), "width_in": float(KEY_TOTAL), "items": list(KEY_ITEMS)}
for k in KEY_ITEMS:
    yk = Y_KEY + 0.026                                            # swatch centre on the x-height of the text
    if k == "within":
        ov.add_patch(Rectangle((kx, yk - BOX_H / 2), SWW[k], BOX_H, facecolor=BOX_FC, edgecolor=BOX_EC, lw=BOX_LW))
    else:
        ov.plot([kx + SWW[k] / 2], [yk], ls="none", **MK[k])
    key_txt[k].set_x(kx + SWW[k] + SW_GAP)
    kx += SWW[k] + SW_GAP + ext(key_txt[k]).width + ENT_GAP

# ---- (b) leader from the direct label to the curve --------------------------------------------------------------
redraw()
inv = fig.dpi_scale_trans.inverted()


def data_to_in(ax, x, y):
    return inv.transform(ax.transData.transform((x, y)))


def in_to_data(ax, x, y):
    return ax.transData.inverted().transform(fig.dpi_scale_trans.transform((x, y)))


if DRAW_TOP:
    bb_tb = ext(LAB_TB)
    p_end = data_to_in(ax_tb, XLINK_T, float(np.exp((XLINK_T - 1.0) / T23)))
    p_beg = np.array([bb_tb.x0 + 0.62 * bb_tb.width, bb_tb.y0 - 1.5 / 72])
    v = p_end - p_beg
    p_end_s = p_end - v / np.linalg.norm(v) * (LW["theory"] / 2 + 1.5) / 72          # stop 1.5 pt short of the line
    qa, qb = in_to_data(ax_tb, *p_beg), in_to_data(ax_tb, *p_end_s)
    ax_tb.plot([qa[0], qb[0]], [qa[1], qb[1]], color=INK, lw=LW["leader"], solid_capstyle="butt",
               zorder=S.Z["label"])

# ---- panel letters (F_link: a-c top, d-f bottom; F_link_top and F_link_net: a-c) -------------------------------------
redraw()
LETTERS = {}
PANEL_LETTER = {}                    # F_link panel name -> letter printed in this variant
if DRAW_TOP:
    for p_, x_ in (("a", 0.0), ("b", ext(ax_tb.yaxis.label).x0), ("c", ext(ax_tc.yaxis.label).x0)):
        LETTERS[p_] = S.panel_letter(fig, p_, x_, TY1 + LET_DY)
        PANEL_LETTER[p_] = p_
if DRAW_BOT:
    for p_, q_, x_ in (("d", "a", 0.0), ("e", "b", ext(ax_b.yaxis.label).x0), ("f", "c", ext(ax_c.yaxis.label).x0)):
        let_ = p_ if VARIANT == "both" else q_
        LETTERS[p_] = S.panel_letter(fig, let_, x_, AX_TOP + LET_DY)
        PANEL_LETTER[p_] = let_
check("v8: panel letters are a-f in F_link and a-c in F_link_top and F_link_net",
      sorted(t.get_text() for t in LETTERS.values()) == ["(%s)" % c for c in ("abcdef" if VARIANT == "both" else "abc")],
      PANEL_LETTER)

PDF, PNG = os.path.join(HERE, f"{FIG_ID}.pdf"), os.path.join(HERE, f"{FIG_ID}.png")
VIEW100, VIEWS = os.path.join(HERE, "_view_print_100dpi.png"), os.path.join(HERE, "_views_cvd_grey.png")
N_FINAL = S.finalize(fig)            # v8: bold tick labels, axis titles and their math (last call before savefig)
fig.savefig(PDF, dpi=600)            # the two densities are rasterized at 600 dpi
CHK = S.check(fig, PDF)              # v8 checks: >= 7 pt, Type 42, contrast, bold ticks/titles, bold math, colour
S.print_view(PDF, PNG, VIEW100)      # PNG = the PDF rendered at 200 dpi; the 100 dpi view is about print size
S.views(PNG, VIEWS)                  # normal / deuteranopia / protanopia / greyscale
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
font_rows = [ln.split() for ln in pf.splitlines()[2:] if ln.strip()]
FONT_NAMES = sorted({row[0].split("+")[-1] for row in font_rows})
assert set(FONT_NAMES) <= {"LiberationSerif", "LiberationSerif-Bold", "LiberationSerif-BoldItalic", "Cmr10", "Cmmi10",
                           "Cmsy10", "Cmex10"}, FONT_NAMES
check("v8: every font is embedded TrueType (Type 42 / CID TrueType); no Type 3 font",
      CHK["n_type3"] == 0 and bool(CHK["fonts"]) and all("TrueType" in f_ and "emb yes" in f_ for f_ in CHK["fonts"]),
      CHK["fonts"])
check("v8: every visible tick label and axis title is bold, with bold math (S.check)",
      CHK["not_bold_tick_or_axis_labels"] == [] and CHK["bold_text_with_regular_math"] == []
      and CHK["n_tick_labels_checked"] > 0 and CHK["n_axis_titles_checked"] > 0,
      [CHK["n_tick_labels_checked"], CHK["n_axis_titles_checked"]])
ALLOWED_DATA = {c.upper() for c in (C_ACROSS, C_CONTROL, C_FIT, INK, S.NULL_LINE, S.NULL_FILL, S.GUIDE, "#FFFFFF")}
check("v8: every data colour outside the densities comes from the semantic map (across, control, the fitted link in the "
      "across text tone, theory INK, the within/null neutrals); the figure has chromatic data colours",
      set(S.data_colours(fig, mapped=False)) <= ALLOWED_DATA and len(CHK["chromatic_data_colours"]) >= 2,
      {"drawn": S.data_colours(fig, mapped=False), "off_palette": CHK["off_palette_colours"]})


# ---- overlap and gutter checks -----------------------------------------------------------------------------
def text_boxes():
    import matplotlib.text as mtext
    out = []
    for t in fig.findobj(mtext.Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        bb = t.get_window_extent().transformed(inv)
        out.append((t.get_text(), bb, t.get_fontsize()))
    return out


TB = text_boxes()
clash = [(a, b) for i, (a, pa, _) in enumerate(TB) for (b, pb, _) in TB[i + 1:]
         if min(pa.x1, pb.x1) - max(pa.x0, pb.x0) > 0.005 and min(pa.y1, pb.y1) - max(pa.y0, pb.y0) > 0.005]
outside = [a for a, bb, _ in TB if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]
small = sorted({(round(fs, 2), a) for a, _, fs in TB if fs < 7.5})
print("CLASH", clash)
print("OUTSIDE", outside)
print("SMALL", small)
assert not clash and not outside, (clash, outside)
GUTTER = {}
if DRAW_BOT:
    x_mark_a = AX0 + AXW * np.log10(max(d["ratio_median"] for d in A_MARK.values()) / XLO) / np.log10(XHI / XLO) \
        + HALF_H_PT["cross_A_B"] / 72
    b_xt = [ext(t) for t in ax_b.get_xticklabels() if t.get_text()]
    c_xt = [ext(t) for t in ax_c.get_xticklabels() if t.get_text()]
    GUTTER.update({"d_mark_to_e_ylabel_in": float(ext(ax_b.yaxis.label).x0 - x_mark_a),
                   "e_xticks_to_f_ylabel_in": float(ext(ax_c.yaxis.label).x0 - max(e.x1 for e in b_xt)),
                   "f_xtick_label_min_gap_in": float(min(c_xt[i + 1].x0 - c_xt[i].x1 for i in range(len(c_xt) - 1))),
                   "f_last_xtick_right_margin_in": float(W - c_xt[-1].x1),
                   "e_key_to_f_ylabel_in": float(ext(ax_c.yaxis.label).x0 - max(ext(t).x1 for t in LEG_B))})
if DRAW_TOP:
    ta_right = max(data_to_in(ax_ta, GX[-1], 0.0)[0] + HALF_H_PT["cross_A_B"] / 72,
                   max(ext(t).x1 for t in ax_ta.get_xticklabels() if t.get_text()))
    tb_xt = [ext(t) for t in ax_tb.get_xticklabels() if t.get_text()]
    tc_xt = [ext(t) for t in ax_tc.get_xticklabels() if t.get_text()]
    GUTTER.update({"a_right_to_b_ylabel_in": float(ext(ax_tb.yaxis.label).x0 - ta_right),
                   "b_xticks_to_c_ylabel_in": float(ext(ax_tc.yaxis.label).x0 - max(e.x1 for e in tb_xt)),
                   "c_xtick_label_min_gap_in": float(min(tc_xt[i + 1].x0 - tc_xt[i].x1 for i in range(len(tc_xt) - 1))),
                   "c_last_xtick_right_margin_in": float(W - tc_xt[-1].x1)})
if VARIANT == "both":       # v8: the bottom header is gone; the rows are kept apart from the top-row x labels to the
    #                         highest element of the bottom row (its panel letters and the (e) key)
    bot_top = max([ext(LETTERS[p_]).y1 for p_ in "def"] + [ext(t).y1 for t in LEG_B])
    GUTTER["top_xlabel_to_bottom_row_top_in"] = float(min(ext(ax_ta.xaxis.label).y0, ext(ax_tc.xaxis.label).y0,
                                                          ext(ax_tb.xaxis.label).y0) - bot_top)
UP_LET = [LETTERS[p_] for p_ in (("a", "b", "c") if DRAW_TOP else ("d", "e", "f"))]
GUTTER["key_line_to_upper_letters_in"] = float(min(ext(t).y0 for t in key_txt.values()) -
                                               max(ext(t).y1 for t in UP_LET))
GUTTER["key_line_to_figure_top_in"] = float(H - max(ext(t).y1 for t in key_txt.values()))
print("GUTTER", GUTTER)
OK_BOT = (not DRAW_BOT) or (GUTTER["d_mark_to_e_ylabel_in"] >= 0.18 and GUTTER["e_xticks_to_f_ylabel_in"] >= 0.12
                            and GUTTER["f_xtick_label_min_gap_in"] >= 0.05
                            and GUTTER["f_last_xtick_right_margin_in"] >= 0.0
                            and GUTTER["e_key_to_f_ylabel_in"] >= 0.15)
OK_TOP = (not DRAW_TOP) or (GUTTER["a_right_to_b_ylabel_in"] >= 0.18 and GUTTER["b_xticks_to_c_ylabel_in"] >= 0.12
                            and GUTTER["c_xtick_label_min_gap_in"] >= 0.05
                            and GUTTER["c_last_xtick_right_margin_in"] >= 0.0)
OK_KEY = GUTTER["key_line_to_upper_letters_in"] >= 0.02 and GUTTER["key_line_to_figure_top_in"] >= 0.0
GUTTER_V7_NAME = ("gutters: (d) marks >= 0.18 in from the (e) y label, (e) ticks >= 0.12 in from the (f) y label, (f) "
                  "tick labels >= 0.05 in apart and inside the figure; top row: (a) >= 0.18 in from the (b) y label, "
                  "(b) ticks >= 0.12 in from the (c) y label, (c) tick labels >= 0.05 apart and inside; >= 0.1 in "
                  "between the rows")
if VARIANT == "both":
    check(GUTTER_V7_NAME, OK_BOT and OK_TOP and OK_KEY and GUTTER["top_xlabel_to_bottom_row_top_in"] >= 0.1,
          {"gutters": GUTTER, "v8_redefined": "'between the rows' now runs from the top-row x labels to the bottom-row "
                                               "letters and (e) key (the bottom header is gone); the v7 header-to-letters "
                                               "gap is now the key line to the top letters (>= 0.02 in)"})
elif VARIANT == "top":
    check("v8 gutters, top row only (F_link_top): (a) >= 0.18 in from the (b) y label, (b) ticks >= 0.12 in from the "
          "(c) y label, (c) tick labels >= 0.05 in apart and inside; key line >= 0.02 in above the letters",
          OK_TOP and OK_KEY, GUTTER)
else:
    check("v8 gutters, bottom row only (F_link_net): (a) marks >= 0.18 in from the (b) y label, (b) ticks >= 0.12 in "
          "from the (c) y label, (c) tick labels >= 0.05 in apart and inside; key line >= 0.02 in above the letters",
          OK_BOT and OK_KEY, GUTTER)
check("v8: one key line, centred, inside the figure, above every panel letter",
      GUTTER["key_line_to_upper_letters_in"] >= 0.02 and KEY_GEOM["x0_in"] > 0
      and KEY_GEOM["x0_in"] + KEY_GEOM["width_in"] < W, KEY_GEOM)
ROW_SEP, GROUP_FIT, GAP_GROUP = {}, {}, {}
if DRAW_BOT:
    MAX_HALF = max(HALF_H_PT[c] for c in ("cross_A_B", "bern_self"))
    ROW_SEP = {"pitch_pt": float(PITCH * 72),
               "max_mark_offset_pt": float(max([v["offset_pt"] for v in NUDGE.values()] + [0.0])),
               "max_mark_half_height_pt": float(MAX_HALF)}
    check("(d) row membership: largest nudge plus half mark < half the pitch",
          ROW_SEP["max_mark_offset_pt"] + MAX_HALF < PITCH * 72 / 2 + GG * 72 / 2, ROW_SEP)
    for g, t in GROUP_TXT.items():
        ys = [row_y[r] for gg, r, _ in ROWS if gg == g]
        e = ext(t)
        GROUP_FIT[g] = {"text_h_in": e.height, "bracket_h_in": max(ys) - min(ys) + PITCH * 0.76}
    GAP_GROUP = {g: ext(ROW_TXT[r]).x0 - (X_BR + 0.035) for g in GROUP for gg, r, _ in ROWS if gg == g}
    assert min(GAP_GROUP.values()) > 0.01, GAP_GROUP


def seg_hits_box(ax, p0, p1, bb, n=80, pad=0.01):
    ts = np.linspace(0, 1, n)
    pts = np.outer(1 - ts, p0) + np.outer(ts, p1)
    pts_in = inv.transform(ax.transData.transform(pts))
    return bool(np.any((pts_in[:, 0] > bb.x0 - pad) & (pts_in[:, 0] < bb.x1 + pad) &
                       (pts_in[:, 1] > bb.y0 - pad) & (pts_in[:, 1] < bb.y1 + pad)))


def poly_hits_box(ax, xs, ys, bb, pad=0.01):
    return any(seg_hits_box(ax, (xs[j], ys[j]), (xs[j + 1], ys[j + 1]), bb, pad=pad) for j in range(len(xs) - 1))


def mark_hits_box(ax, xs, ys, half_pt, bb, pad=0.01):
    for x_, y_ in zip(xs, ys):
        pm = data_to_in(ax, x_, y_)
        r_in = half_pt / 72 + pad
        if bb.x0 - r_in < pm[0] < bb.x1 + r_in and bb.y0 - r_in < pm[1] < bb.y1 + r_in:
            return True
    return False


def band_hits_box(ax, lo, hi, bb, pad=0.01):
    """the within band, inflated by half a box height on either side (its drawn ribbon)"""
    xs = np.linspace(GX[0], GX[-1], 400)
    lo_i, hi_i = np.interp(xs, GX, lo), np.interp(xs, GX, hi)
    if ax.get_yscale() == "log":
        lo_i, hi_i = np.log10(lo_i), np.log10(hi_i)
    for x_, a_, b_ in zip(xs, lo_i, hi_i):
        ya = data_to_in(ax, x_, 10 ** a_ if ax.get_yscale() == "log" else a_)
        yb = data_to_in(ax, x_, 10 ** b_ if ax.get_yscale() == "log" else b_)
        y0_, y1_ = min(ya[1], yb[1]) - BOX_H / 2 - BOX_LW / 72, max(ya[1], yb[1]) + BOX_H / 2 + BOX_LW / 72
        if bb.x0 - pad < ya[0] < bb.x1 + pad and y0_ < bb.y1 + pad and y1_ > bb.y0 - pad:
            return True
    return False


LAB_HITS = {}
if DRAW_BOT:
    for k, t in LAB_C.items():
        bb = ext(t)
        hits = []
        for c in C_CLASSES_DRAWN:
            xs = XS_C + C_DX[c]
            med = STAIR[c]["median"][:N_STEPS_C]
            if poly_hits_box(ax_c, xs, med, bb):
                hits.append(f"{c} line")
            if mark_hits_box(ax_c, xs, med, HALF_H_PT[c], bb):
                hits.append(f"{c} marker")
        if seg_hits_box(ax_c, (x_req - REQ_HALF, REQ), (x_req + REQ_HALF, REQ), bb) and k != "req":
            hits.append("requirement bar")
        if seg_hits_box(ax_c, (x_req, y_mark_top), (x_req, REQ), bb):
            hits.append("requirement leader")
        LAB_HITS["f|" + k] = hits
if DRAW_TOP:
    for k, t in LAB_TA.items():
        bb = ext(t)
        hits = []
        if poly_hits_box(ax_ta, MCX, MCY, bb):
            hits.append("prediction curve")
        if mark_hits_box(ax_ta, GX, TE_AB, HALF_H_PT["cross_A_B"], bb):
            hits.append("cross marker")
        if band_hits_box(ax_ta, TE_W_LO, TE_W_HI, bb):
            hits.append("within band")
        if k != "nstar" and seg_hits_box(ax_ta, (NSTAR, 0.0), (NSTAR, 1.0), bb):
            hits.append("N* line")
        LAB_HITS["a|" + k] = hits
    bb = ext(LAB_TB)
    hx_in = np.array([data_to_in(ax_tb, x_, y_) for x_, y_ in HEX_XY_T])
    hex_r = (KHI - KLO) / GRIDSIZE / np.sqrt(3) * TBW / (KHI - KLO) * 1.2
    LAB_HITS["b|link"] = (["hexagon"] if np.any((hx_in[:, 0] > bb.x0 - hex_r) & (hx_in[:, 0] < bb.x1 + hex_r)
                                                & (hx_in[:, 1] > bb.y0 - hex_r) & (hx_in[:, 1] < bb.y1 + hex_r))
                          else []) + (["curve"] if poly_hits_box(ax_tb, TGRID23, THEORY23, bb) else [])
    for k, t in LAB_TC.items():
        bb = ext(t)
        hits = []
        for key_, arr, xs_ in (("cross_A_B", GS_AB, GX_AB), ("cf", CF_T, GX_CF)):
            if poly_hits_box(ax_tc, xs_, arr["median"], bb):
                hits.append(f"{key_} line")
            if mark_hits_box(ax_tc, xs_, arr["median"], HALF_H_PT[key_], bb):
                hits.append(f"{key_} marker")
        if band_hits_box(ax_tc, GS_W_LO, GS_W_HI, bb):
            hits.append("within band")
        if seg_hits_box(ax_tc, (XL_T[0], OMEGA), (XL_T[1], OMEGA), bb):
            hits.append("predicted limit")
        if poly_hits_box(ax_tc, REF_X, REF_Y, bb):
            hits.append("reference slope")
        LAB_HITS["c|" + k] = hits
print("LAB_HITS", LAB_HITS)
assert not any(LAB_HITS.values()), LAB_HITS


def _pt_seg(p, a, b):
    ab, ap = b - a, p - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


C_CLEAR, C_MIN = {}, None
if DRAW_BOT:
    C_PTS = {c: inv.transform(ax_c.transData.transform(np.column_stack([XS_C + C_DX[c],
                                                                           STAIR[c]["median"][:N_STEPS_C]]))) * 72
             for c in C_CLASSES_DRAWN}
    C_LW = {c: C_STYLE[c]["lw"] for c in C_CLASSES_DRAWN}
    for c in C_CLASSES_DRAWN:
        for j, p in enumerate(C_PTS[c]):
            for c2 in C_CLASSES_DRAWN:
                if c2 == c:
                    continue
                for j2, p2 in enumerate(C_PTS[c2]):
                    C_CLEAR[f"{c}[{j}] vs {c2}[{j2}] marker"] = float(np.linalg.norm(p - p2)) - HALF_H_PT[c] \
                        - HALF_H_PT[c2]
                for j2 in range(N_STEPS_C - 1):
                    C_CLEAR[f"{c}[{j}] vs {c2} line {j2}"] = _pt_seg(p, C_PTS[c2][j2], C_PTS[c2][j2 + 1]) \
                        - HALF_H_PT[c] - C_LW[c2] / 2
    C_MIN = min(C_CLEAR, key=C_CLEAR.get)
    print("C_CLEAR min", C_MIN, C_CLEAR[C_MIN])
    check("(f) every marker clears the markers and lines of the other classes by >= 0.5 pt", C_CLEAR[C_MIN] >= 0.5,
          [C_MIN, C_CLEAR[C_MIN]])
TC_CLEAR = {}
if DRAW_TOP:
    TC_MARKS = [("cross_A_B", GX_AB[j], GS_AB["median"][j], l2) for j, l2 in enumerate(GRID23)] + \
        [("cf", GX_CF[j], CF_T["median"][j], l2) for j, l2 in enumerate(GRID23)]
    _tcc = {}
    for i_, (ka, xa, ya, la) in enumerate(TC_MARKS):
        for kb, xb, yb, lb in TC_MARKS[i_ + 1:]:
            d_ = np.linalg.norm((data_to_in(ax_tc, xa, ya) - data_to_in(ax_tc, xb, yb)) * 72)
            _tcc[f"{ka}@{la} vs {kb}@{lb}"] = float(d_ - HALF_H_PT[ka] - HALF_H_PT[kb])
    TC_CMIN = min(_tcc, key=_tcc.get)
    check("(c) every circle and square clears every other mark by >= 0.3 pt", _tcc[TC_CMIN] >= 0.3,
          [TC_CMIN, _tcc[TC_CMIN]])
    x_spine = data_to_in(ax_tc, XL_T[0], 1.0)[0]
    TC_SPINE = float(min((data_to_in(ax_tc, x_, 1.0)[0] - x_spine) * 72 - HALF_H_PT[k_] for k_, x_, _, _ in TC_MARKS))
    check("(c) every mark clears the y spine by >= 0.5 pt", TC_SPINE >= 0.5, TC_SPINE)
    TC_CLEAR = {"min": [TC_CMIN, _tcc[TC_CMIN]], "spine_pt": TC_SPINE}

# ranges drawn under the marks: their extent at print size against the mark diameter
HIDDEN = {}
if DRAW_BOT:
    for (c, r), d in A_MARK.items():
        if c == "bern_self":
            continue
        w_pt = np.log10(d["ratio_q95"] / d["ratio_q05"]) * PT_PER_DEC
        HIDDEN[f"d|{c}|{r}"] = {"range_pt": float(w_pt), "mark_pt": float(MK[c]["ms"])}
    for c in C_CLASSES_DRAWN:
        for j in range(N_STEPS_C):
            w_pt = (STAIR[c]["q90"][j] - STAIR[c]["q10"][j]) / Y_PER_PT
            HIDDEN[f"f|{c}|{j}"] = {"range_pt": float(w_pt), "mark_pt": float(MK[c]["ms"])}
if DRAW_TOP:
    for key_, arr in (("cross_A_B", GS_AB), ("cf", CF_T)):
        for j, l2 in enumerate(GRID23):
            HIDDEN[f"c|{key_}|{l2}"] = {"range_pt": float(np.log10(arr["q90"][j] / arr["q10"][j]) * PT_PER_DEC_TC),
                                        "mark_pt": float(MK[key_]["ms"])}
HIDDEN_V7_NAME = "every 5-95% range in (d), 10-90% range in (f) and 10-90% range in (c) is shorter than its mark"
HIDDEN_NAME = {"both": HIDDEN_V7_NAME,
               "top": "v8 top row only (F_link_top): every 10-90% range in (c) is shorter than its mark",
               "net": "v8 bottom row only (F_link_net): every 5-95% range in (a) and 10-90% range in (c) is shorter "
                      "than its mark"}[VARIANT]
check(HIDDEN_NAME, all(v["range_pt"] < v["mark_pt"] for v in HIDDEN.values()),
      max(v["range_pt"] / v["mark_pt"] for v in HIDDEN.values()))



def rq(c, r):
    return panel_a["classes"][c][r]["ratio_median"]


ORDER_R = ["dN_k10", "dO", "dE", "dMST"]
GEOM_R = ["dCKA", "Gs", "dAff", "dSOG_t0p1"]
P = "numbers.json:panel_d.classes"
F0 = RES23["F0"]
P7 = RES23["secondary"]["P7"]
SR = "stageA_results.json"
TEXT_NUMBERS_TOP = [
    {"text": "top (a): cross-objective pairs agreeing exactly on T_E over the 8 probes, per grid N (of 96)",
     "value": {str(l2): PER_N[l2]["AB"]["TE_agree"] for l2 in GRID23},
     "source": "cache arm0_l<N>_AB_n96.json ex8.T_E = " + SR + ":secondary.P2b.curve.<N>.ex8_T_E_AB.phat x 96"},
    {"text": "top (a): within-objective pairs agreeing exactly on T_E, InfoNCE / spectral, per grid N (of 96)",
     "value": {str(l2): [PER_N[l2]["wA"]["TE_agree"], PER_N[l2]["wB"]["TE_agree"]] for l2 in GRID23},
     "source": SR + ":secondary.P2b.curve.<N>.ex8_T_E_{AA,BB}.phat x 96"},
    {"text": "top (a): predicted transition N* (smallest N with MC p_AB >= 0.9) and the MC prediction at 2^31, 2^34",
     "value": {"log2_Nstar": NSTAR, "mc_p_AB_2^31": RES23["secondary"]["P2b"]["curve"]["31"]["ex8_T_E_AB"]["mc"],
               "mc_p_AB_2^34": RES23["secondary"]["P2b"]["curve"]["34"]["ex8_T_E_AB"]["mc"]},
     "source": "frozen_predictions.json:nstar.log2_Nstar, mc_curves.ex8"},
    {"text": "P2 (Thm 3 part 2): 96 of 96 cross pairs agree exactly on T_E at N_top = 2^40; Clopper-Pearson lower 0.969",
     "value": {"x": F0["P2"]["x"], "n": F0["P2"]["n"], "cp_lo": F0["P2"]["cp_lo"]}, "source": SR + ":F0.P2"},
    {"text": "top (c): cross-objective Gs_max median per grid N, and its predicted limit omega_psi",
     "value": {"median": {str(l2): PER_N[l2]["AB"]["Gs_max_median"] for l2 in GRID23}, "omega_psi": OMEGA},
     "source": "cache arm0_l<N>_AB_n96.json d.Gs_max; " + SR + ":eps.omega_psi"},
    {"text": "P3a (Thm 3 part 3): cross Gs_max within eps_geo = omega_psi / 2 in 0 of 96 at 2^34 and at 2^40 "
             "(also 0 of 96 at every other grid N, descriptive)",
     "value": {"eps_geo": EPS_GEO, "x_N0": F0["P3a"]["N0"]["x"], "x_Ntop": F0["P3a"]["Ntop"]["x"],
               "x_every_N": {str(l2): PER_N[l2]["AB"]["Gs_max_within_eps_geo"] for l2 in GRID23}},
     "source": SR + ":F0.P3a; cache d.Gs_max"},
    {"text": "P3b: cross 1 - CKA at 2^40 (mean) against its limit; within-objective means",
     "value": {"cross_mean": F0["P3b"]["tost"]["mean"], "limit": RES23["eps"]["one_minus_cka_inf"],
               "within_A": F0["P3b"]["within_A"]["mean"], "within_B": F0["P3b"]["within_B"]["mean"]},
     "source": SR + ":F0.P3b, eps.one_minus_cka_inf"},
    {"text": "top (c): within-objective Gs_max medians per grid N, InfoNCE / spectral",
     "value": {str(l2): [PER_N[l2]["wA"]["Gs_max_median"], PER_N[l2]["wB"]["Gs_max_median"]] for l2 in GRID23},
     "source": "cache arm0_l<N>_w{A,B}_n96.json d.Gs_max"},
    {"text": "P3c (Thm 3 part 3, decomposition): inverse-link counterfactual median at 2^34 and 2^40 and their ratio "
             "(root-N rate predicts 0.125); the other grid N are descriptive (this maker, frozen metrics.inv_link_cf)",
     "value": {"median": {str(l2): PER_N[l2]["cf"]["median"] for l2 in GRID23}, "ratio": F0["P3c"]["ratio"],
               "ci": [F0["P3c"]["ci_lo"], F0["P3c"]["ci_hi"]]},
     "source": SR + ":F0.P3c; review_A2/recompute_A2.json:F0.P3c.median_c"},
    {"text": "P4a (Thm 3 part 4): isotonic discrepancy within eps_geo in 96 of 96 cross pairs at 2^40",
     "value": {"x": F0["P4a"]["x"], "n": F0["P4a"]["n"]}, "source": SR + ":F0.P4a"},
    {"text": "P1 (Thm 3 part 1): median recovery-error ratio 2^40 / 2^34, InfoNCE and spectral (root-N predicts 0.125)",
     "value": {"A": [F0["P1_A"]["ratio"], F0["P1_A"]["ci_lo"], F0["P1_A"]["ci_hi"]],
               "B": [F0["P1_B"]["ratio"], F0["P1_B"]["ci_lo"], F0["P1_B"]["ci_hi"]]},
     "source": SR + ":F0.P1_A, F0.P1_B"},
    {"text": "Holm: 9 of 9 preregistered tests confirmed; largest p and smallest p",
     "value": {"n_confirmed": sum(bool(v["confirmed"]) for v in F0.values()),
               "p_max": max(v["p"] for v in F0.values()), "p_min": min(v["p"] for v in F0.values())},
     "source": SR + ":F0.*.confirmed, .p"},
    {"text": "top (b): the theorem's link exp((K_A - 1)/t) without any fit removes this share of the squared-distance "
             "discrepancy, pair of (b) and median over the 96 pairs at 2^40 (networks, pair of (e): see panel_e)",
     "value": {"pick": SHARE23_PICK, "median_96": float(np.median(SHARE23_ALL)), "networks_pick": SHARE_THEORY},
     "source": "numbers.json:panel_top.panel_b_pick, theory_share_no_fit_Ntop_all_pairs; panel_e"},
    {"text": "P7 at 2^20 (the noisiest model size): within-objective neighbor agreement s_N(10) InfoNCE / spectral, "
             "cross; networks within at 500k (target); order disagreement across does not exceed within (rho)",
     "value": {"within_A_sN10": NM23["table"]["20"]["mean_sN10_withinA"],
               "within_B_sN10": REV23["P7"]["within_B_sN10"], "cross_sN10": REV23["P7"]["cross_sN10"],
               "networks_target": NM23["target_sN10"],
               "rho": {r: P7["Pi"][r]["rho"] for r in P7["Pi"]}},
     "source": "N_match.json; review_A2/recompute_A2.json:P7; " + SR + ":secondary.P7.Pi"},
]
TEXT_NUMBERS = [
    {"text": "pairs of InfoNCE (0.2) and spectral runs within tolerance: 0 of 64 on every criterion at both N",
     "value": {r: [SUM["tables"]["cross_A_B_%d|z" % n][r]["x"] for n in (50000, 500000)] for r in RUNGS},
     "source": "s21_summary.json tables[cross_A_B_<N>|z][rung].x; Table tab:app-s21-full"},
    {"text": "pairs within one objective at N = 500,000: 32 of 32 within tolerance on every criterion, in each arm",
     "value": {c: sorted({panel_a["classes"][c][r]["x_within_eps"] for r in RUNGS}) for c in WITHIN},
     "source": "s21_summary.json tables[within_<arm>_500000|z]; Table tab:app-s21-full"},
    {"text": "median Delta/eps across objectives, order criteria (T_MST lowest, T_E highest)",
     "value": {r: rq("cross_A_B", r) for r in ORDER_R}, "source": P + ".cross_A_B.<rung>.ratio_median"},
    {"text": "median Delta/eps across objectives, geometric criteria (SOG lowest, 1 - CKA highest)",
     "value": {r: rq("cross_A_B", r) for r in GEOM_R}, "source": P + ".cross_A_B.<rung>.ratio_median"},
    {"text": "median Delta/eps across objectives on Delta_iso and Delta_Pi",
     "value": {r: rq("cross_A_B", r) for r in ("dIso", "dPi")}, "source": P + ".cross_A_B.<rung>.ratio_median"},
    {"text": "median Delta/eps within one objective, range over arms and criteria (Delta_Pi apart / with it)",
     "value": {"without_dPi": [min(wmed_g), max(wmed_g)], "with_dPi": [min(wmed), max(wmed)]},
     "source": P + ".within_{A,B,Atau}.<rung>.ratio_median"},
    {"text": "exact link (link check, self, 64 pairs): 64 of 64 within tolerance on T_N,10, T_O, T_E, T_MST and "
             "Delta_iso; 0 of 64 on the four geometric criteria; Delta exactly 0 on T_N,10, T_O, T_MST (62 of 64 on T_E)",
     "value": {r: panel_a["classes"]["bern_self"][r]["x_within_eps"] for r in RUNGS if r != "dPi"},
     "source": "s21_summary.json tables[bern_self_500000|z]; Table tab:app-s21-full"},
    {"text": "exact link median Delta/eps on geometric criteria (SOG lowest, 1 - CKA highest)",
     "value": {r: rq("bern_self", r) for r in GEOM_R}, "source": P + ".bern_self.<rung>.ratio_median"},
    {"text": "InfoNCE at 0.2 vs 0.5 (not drawn in (d); F.7): 64 of 64 within tolerance on Delta_Pi only, 0 of 64 on "
             "every other criterion; median Delta/eps on the others (T_MST lowest, 1 - CKA highest) and on Delta_Pi",
     "value": {"x": {r: panel_a["classes"]["cross_A_Atau"][r]["x_within_eps"] for r in RUNGS},
               "ratio_median": {r: rq("cross_A_Atau", r) for r in RUNGS}},
     "source": "s21_summary.json tables[cross_A_Atau_500000|z]; " + P + ".cross_A_Atau.<rung>.ratio_median"},
    {"text": "cumulative explained share, medians over pairs (cross-fitted): scale, + affine, + global monotone, "
             "+ per-row monotone (the last step is not drawn in (f))",
     "value": {c: panel_c["classes"][c]["cum_median"] for c in C_CLASSES},
     "source": "numbers.json:panel_f.classes.<class>.cum_median (ladder row.warpshare_expl_cf_*)"},
    {"text": "requirement bar in (f): level at which the median InfoNCE-vs-spectral pair's link would remove half of "
             "the affine residual; pairs reaching it",
     "value": panel_c["requirement_cross_A_B"], "source": "numbers.json:panel_f.requirement_cross_A_B"},
    {"text": "median warp share omega across objectives at N = 500,000 (preregistered requirement >= 0.5)",
     "value": float(np.median(OM)), "source": "ladder deltas.omega over the 64 A-B pairs; S2_1_report.md item (iv) "
                                               "0.147 (0.142 at 50k; within 0.012 and 0.002)"},
    {"text": "omega across temperatures, median", "value": float(np.median([row["deltas"]["omega"]
                                                                            for row in LAD["cross_A_Atau"]])),
     "source": "ladder deltas.omega over the 64 A-A' pairs; S2_1_report.md item (iv) omega_cross_A_Atau 0.109"},
    {"text": "pair of panel (e): index, its omega, in-sample A->B shares (affine, monotone link), kernel correlation",
     "value": {"pair": [RUN_A, RUN_B], "omega": float(OM[PICK]), "affine": Q["affine_expl"], "iso": Q["iso_expl"],
               "pearson": panel_b["pearson_kernel"]},
     "source": "numbers.json:panel_e (recomputed; equal to the ladder row quot_AB_* to 1e-9)"},
    {"text": "panel (e), theory against data: median spectral kernel per InfoNCE-kernel bin against exp(5(K_A - 1)); "
             "share of negative spectral entries; share explained by the theorem's g_t without any fit",
     "value": {"bins": THEORY_GAP, "frac_negative": FRAC_NEG, "share_theory_link_no_fit": SHARE_THEORY},
     "source": "numbers.json:panel_e.theory_vs_data_by_K_A_bin (recomputed from feats.npz)"},
] + TEXT_NUMBERS_TOP


# v8: the number the Results text now carries for the comparison removed from (f) (MOVED.md, must-add sentence)
OM_TAU = np.array([row["deltas"]["omega"] for row in LAD["cross_A_Atau"]], float)
TEXT_NUMBERS.append(
    {"text": "v8, not drawn (was the open circles of v7 (f)): InfoNCE at 0.2 vs 0.5 at N = 500,000, median cumulative "
             "explained share after scale, + affine, + global monotone, and the median warp share omega (the share of "
             "the affine residual removed by the monotone link); InfoNCE vs spectral for comparison",
     "value": {"cross_A_Atau_cum_median": panel_c["classes"]["cross_A_Atau"]["cum_median"][:3],
               "cross_A_Atau_omega_median": float(np.median(OM_TAU)),
               "cross_A_B_cum_median": panel_c["classes"]["cross_A_B"]["cum_median"][:3],
               "cross_A_B_omega_median": float(np.median(OM)),
               "rounded_for_text": {"tau_total_pct": int(round(100 * panel_c["classes"]["cross_A_Atau"]["cum_median"][2])),
                                    "tau_omega_pct": int(round(100 * float(np.median(OM_TAU)))),
                                    "AB_total_pct": int(round(100 * panel_c["classes"]["cross_A_B"]["cum_median"][2]))}},
     "source": "numbers.json:panel_f.classes.cross_A_Atau.cum_median; ladder deltas.omega over the 64 A-A' pairs "
               "(S2_1_report.md item (iv) omega_cross_A_Atau 0.109); critic v8: not drawn in Figure fig:app-s21 v8 "
               "either (F_link_app v8 drops the class), so the Results text carries it (MOVED.md C5, and "
               "F_link_app/MOVED.md item 9 for 0.39 against 0.62)"})
check("v8: the numbers of the must-add sentence for InfoNCE at 0.2 vs 0.5 (39%, 11%, against 62%)",
      TEXT_NUMBERS[-1]["value"]["rounded_for_text"] == {"tau_total_pct": 39, "tau_omega_pct": 11, "AB_total_pct": 62},
      TEXT_NUMBERS[-1]["value"])
# critic v8: C5 follows "At each size, ..." in Section 5.6, so its 11% must hold at 50,000 images too (ladder deltas)
_F50 = sorted(glob.glob(LD + "/S2.1_A_N50000_r*__S2.1_Atau_N50000_r*.json"))
OM_TAU_50K = np.array([load(f_)["deltas"]["omega"] for f_ in _F50], float)
TEXT_NUMBERS[-1]["value"]["cross_A_Atau_omega_median_50k"] = float(np.median(OM_TAU_50K))
TEXT_NUMBERS[-1]["value"]["rounded_for_text"]["tau_omega_pct_50k"] = int(round(100 * float(np.median(OM_TAU_50K))))
check("critic v8: the warp share between InfoNCE at 0.2 and 0.5 rounds to 11% at 50,000 images as well (64 pairs), so "
      "sentence C5 holds at each size",
      len(OM_TAU_50K) == 64 and int(round(100 * float(np.median(OM_TAU_50K)))) == 11,
      [len(OM_TAU_50K), float(np.median(OM_TAU_50K))])
# critic v8: independent recompute of 198 plotted values of the three figures (own numpy/scipy, raw Grams and feats)
CRIT8_P = os.path.join(HERE if FIG_ID == "F_link" else os.path.join(os.path.dirname(HERE), "F_link"), "_critic_v8",
                       "critic_v8_recompute.json")
CRIT8 = load(CRIT8_P)
_c8 = CRIT8["bottom"]
_c8_ok = (CRIT8["n_compared"] == 198 and CRIT8["max_absdiff"] < 1e-12
          and all(CRIT8["top"][str(l2)]["AB"]["te"] == PER_N[l2]["AB"]["TE_share"] for l2 in GRID23)
          and max(abs(CRIT8["top"][str(l2)]["AB"]["gs_med"] - PER_N[l2]["AB"]["Gs_max_median"]) for l2 in GRID23) < 1e-12
          and max(abs(CRIT8["top"][str(l2)]["cf_med"] - PER_N[l2]["cf"]["median"]) for l2 in GRID23) < 1e-12
          and all(max(abs(a_ - b_) for a_, b_ in zip(_c8["cum_median_" + c], panel_c["classes"][c]["cum_median"][:3]))
                  < 1e-12 for c in C_CLASSES)
          and abs(_c8["omega_median_cross_A_B"] - float(np.median(OM))) < 1e-12
          and abs(_c8["omega_median_cross_A_Atau"] - float(np.median(OM_TAU))) < 1e-12)
check("critic v8 recompute (independent numpy/scipy code from the raw Grams and embeddings) equals the plotted values: "
      "198 comparisons over the three figures, max abs difference below 1e-12; re-checked here against T_E shares, "
      "Gs_max and link-undone medians (8 N), the (f) cumulative medians of three classes and both median warp shares",
      _c8_ok, {"n_compared": CRIT8["n_compared"], "max_absdiff": CRIT8["max_absdiff"]})
CRITIC_V8 = {"path": CRIT8_P, "sha256": sha(CRIT8_P), "script": CRIT8_P.replace(".json", ".py"),
             "script_sha256": sha(CRIT8_P.replace(".json", ".py")), "n_compared": CRIT8["n_compared"],
             "max_absdiff": CRIT8["max_absdiff"]}

# ---- archived captions and MOVED of earlier versions: their checks run unchanged (v7 did the same for v6) -------------
CAPTION_P = os.path.join(HERE, "_v6_ref", "caption_v6.tex")
check("v7: the archived v6 caption is a byte copy of ~/hrh_v2_figs/v6/F_link_s23/caption_v6.tex",
      sha(CAPTION_P) == sha(os.path.expanduser("~/hrh_v2_figs/v6/F_link_s23/caption_v6.tex")))
with open(CAPTION_P) as f:
    CAPTION = f.read().strip()
_cap = re.sub(r"\\ref\{[^}]*\}", "", CAPTION)
for tok in ("CIFAR-5m", "500{,}000", "$0.2$", "$0.5$", "N^{-1/2}"):
    _cap = _cap.replace(tok, "")
check("caption carries no result number (digits only in \\ref labels, the CIFAR-5m name, N = 500,000, the "
      "temperatures 0.2 and 0.5, and the N^{-1/2} slope)", re.search(r"\d", _cap) is None,
      re.findall(r".{0,12}\d.{0,12}", _cap))
check("caption names thm:link, thm:peer, InfoNCE and the spectral loss",
      all(s in CAPTION for s in ("\\ref{thm:link}", "\\ref{thm:peer}", "InfoNCE", "spectral")))

V7_DIR = os.path.expanduser("~/hrh_v2_figs/v7/F_link")
CAPTION7_P = os.path.join(HERE, "_v7_ref", "caption_v7.tex")
MOVED7_P = os.path.join(HERE, "_v7_ref", "MOVED_v7.md")
V7_NUM_P = os.path.join(HERE, "_v7_ref", "numbers_v7.json")
check("v8: the archived v7 caption, MOVED.md and numbers.json are byte copies of ~/hrh_v2_figs/v7/F_link/",
      sha(CAPTION7_P) == sha(os.path.join(V7_DIR, "caption_v7.tex")) and sha(MOVED7_P) == sha(os.path.join(V7_DIR,
                                                                                                       "MOVED.md"))
      and sha(V7_NUM_P) == sha(os.path.join(V7_DIR, "numbers.json")))
with open(CAPTION7_P) as f:
    CAPTION7 = f.read().strip()
with open(MOVED7_P) as f:
    MOVED7 = f.read()


def cap_words(s):
    s = re.sub(r"\\(ref|label)\{[^}]*\}", "R", s)
    s = re.sub(r"\$[^$]*\$", "M", s)
    s = re.sub(r"\\[A-Za-z]+", " ", s).replace("~", " ")
    s = re.sub(r"[{}]", " ", s)
    return [w for w in s.split() if re.search(r"[A-Za-z0-9]", w)]


assert CAPTION7.startswith("\\caption{\\textbf{") and CAPTION7.endswith("}"), CAPTION7[:40]
_bold7 = re.match(r"\\caption\{\\textbf\{(.*?)\}\s", CAPTION7, re.S).group(1)
CAP7 = {"words_total": len(cap_words(CAPTION7[len("\\caption{"):-1])), "words_bold": len(cap_words(_bold7))}
_cap7 = re.sub(r"\\ref\{[^}]*\}", "", CAPTION7).replace("CIFAR-5m", "")
check("v7 caption: no digit outside \\ref labels and the CIFAR-5m name (no result number, no setting value)",
      re.search(r"\d", _cap7) is None, re.findall(r".{0,12}\d.{0,12}", _cap7))
check("v7 caption: at most 100 words, bold first sentence at most 30 words",
      CAP7["words_total"] <= 100 and CAP7["words_bold"] <= 30, [CAP7["words_total"], CAP7["words_bold"]])
check("v7 critic: bold first sentence at most 28 tokens (the counter splits Theorem~\\ref, so 27 words)",
      CAP7["words_bold"] <= 28, CAP7["words_bold"])
_cap_plain7 = re.sub(r"\\ref\{[^}]*\}", "R", CAPTION7)
check("v7 critic: no clause of the caption ends on it, them, neither, none or nothing",
      re.search(r"\b(it|them|neither|none|nothing)\s*[.;,}]", _cap_plain7) is None,
      re.findall(r".{0,20}\b(?:it|them|neither|none|nothing)\s*[.;,}]", _cap_plain7))
check("v7 caption names thm:link, InfoNCE and the spectral loss, and points to the appendix of both rows",
      all(s in CAPTION7 for s in ("\\ref{thm:link}", "InfoNCE", "spectral", "\\ref{app:exp-s23}",
                                  "\\ref{fig:app-s21}")))
check("v7 caption: no em-dash, no colon, no sentence ending on 'it'",
      ("---" not in CAPTION7) and ("—" not in CAPTION7) and (":" not in re.sub(r"\\ref\{[^}]*\}", "", CAPTION7))
      and re.search(r"\bit[.;]", CAPTION7) is None)
check("v7: thm:peer, dropped from the caption, is carried by MOVED.md as a must-add for Appendix app:exp-s23",
      "\\ref{thm:peer}" not in CAPTION7 and "thm:peer" in MOVED7 and "app:exp-s23" in MOVED7)
check("v7 critic: MOVED.md names Delta_Pi and Delta_iso for Section 5.6 and the (d) box range and (e) pick for app:exp-s21",
      all(x in MOVED7 for x in ("its panel~(d) shows the projector distance $\\Delta_\\Pi$",
                                "the isotonic discrepancy $\\Delta_{\\rm iso}$ of Appendix~\\ref{app:exp-criteria}",
                                "span all pairs within the three arms", "lower median warp share",
                                "lower median difference after the link is undone")))

# ---- v8 caption: bold claim first, number-free, short; settings named when a panel is called ------------------------
CAPTION8_P = os.path.join(HERE, "caption_v8.tex")
MOVED_P = os.path.join(HERE, "MOVED.md")
with open(CAPTION8_P) as f:
    CAPTION8 = f.read().strip()
with open(MOVED_P) as f:
    MOVED = f.read()
PANEL_REF = re.compile(r"\((?:[a-f](?:--[a-f])?)(?:,~?[a-f](?:--[a-f])?)*\)")


def cap_words_v8(s):
    """words of a caption: panel references such as (a), (b,~e), (a--c) not counted, and 'Theorem~\\ref{...}' (or
    Figure, Appendix, Appendices) counted as one word (the v7 counter split it in two)"""
    s = re.sub(r"(Theorem|Figure|Appendix|Appendices|Section|Table)~\\ref\{[^}]*\}", r"\1", s)
    return cap_words(PANEL_REF.sub(" ", s))


assert CAPTION8.startswith("\\caption{\\textbf{") and CAPTION8.endswith("}"), CAPTION8[:40]
_bold8 = re.match(r"\\caption\{\\textbf\{(.*?)\}\s", CAPTION8, re.S).group(1)
CAP8 = {"words_total": len(cap_words_v8(CAPTION8[len("\\caption{"):-1])), "words_bold": len(cap_words_v8(_bold8)),
        "tokens_total_v7_counter": len(cap_words(CAPTION8[len("\\caption{"):-1])),
        "path": CAPTION8_P, "sha256": sha(CAPTION8_P), "text": CAPTION8}
WORD_CAP = {"both": 80, "top": 80, "net": 100}[VARIANT]       # main figure about 60-80 words, appendix at most 100
_cap8 = re.sub(r"\\ref\{[^}]*\}", "", CAPTION8).replace("CIFAR-5m", "")
check("v8 caption: no digit outside \\ref labels and the CIFAR-5m name (no result number, no setting value)",
      re.search(r"\d", _cap8) is None, re.findall(r".{0,12}\d.{0,12}", _cap8))
check(f"v8 caption: at most {WORD_CAP} words (panel references not counted), bold first sentence at most 26 words",
      CAP8["words_total"] <= WORD_CAP and CAP8["words_bold"] <= 26, [CAP8["words_total"], CAP8["words_bold"]])
_cap_plain8 = re.sub(r"\\ref\{[^}]*\}", "R", CAPTION8)
check("v8 caption: no clause ends on it, them, neither, none or nothing; no em-dash; no colon; no 'as well' or 'too' tail",
      re.search(r"\b(it|them|neither|none|nothing)\s*[.;,}]", _cap_plain8) is None and "---" not in CAPTION8
      and "—" not in CAPTION8 and ":" not in _cap_plain8
      and re.search(r"\b(as well|too)\s*[.;,}]", _cap_plain8) is None,
      re.findall(r".{0,20}\b(?:it|them|neither|none|nothing)\s*[.;,}]", _cap_plain8))
NEED8 = {"both": ("\\ref{thm:link}", "finite Gibbs model", "CIFAR-5m", "(a--c)", "(d--f)", "\\ref{app:exp-s23}",
                  "\\ref{app:exp-s21}"),
         "top": ("\\ref{thm:link}", "finite Gibbs model", "\\ref{app:exp-s23}"),
         "net": ("\\ref{thm:link}", "CIFAR-5m", "\\ref{fig:app-s21}")}[VARIANT]
check("v8 caption names the theorem, the setting of every row it shows and its appendix",
      all(x in CAPTION8 for x in NEED8), [x for x in NEED8 if x not in CAPTION8])
check("v8 caption: the comparison of the two InfoNCE temperatures is not mentioned (it is not drawn)",
      not re.search(r"0\.2|0\.5|temperature", CAPTION8))
check("critic v8: the caption does not promise 'every pair set' in fig:app-s21 (F.8 v8 drops InfoNCE at 0.2 vs 0.5)",
      "every pair set" not in CAPTION8)
# critic v8: F.8 v8 (F_link_app) no longer draws InfoNCE at 0.2 vs 0.5, so C5 carries the warp share in Section 5.6 and
# the cumulative 0.39 against 0.62 is the sentence that F_link_app/MOVED.md item 9 adds to Appendix app:exp-s21
APP_MOVED_P = f"{HOME}/hrh_v2_figs/v8/F_link_app/MOVED.md"
with open(APP_MOVED_P) as f:
    APP_MOVED = f.read()
check("critic v8: F_link_app/MOVED.md (F.8 v8, read only) drops InfoNCE at 0.2 vs 0.5 and carries its cumulative shares "
      "(a median 0.39 against 0.62) in a must-add sentence for Appendix app:exp-s21",
      "InfoNCE at 0.2 against 0.5" in APP_MOVED and "a median $0.39$" in APP_MOVED and "against $0.62$" in APP_MOVED
      and TEXT_NUMBERS[-1]["value"]["rounded_for_text"]["tau_total_pct"] == 39
      and TEXT_NUMBERS[-1]["value"]["rounded_for_text"]["AB_total_pct"] == 62)
check("v8 MOVED.md (critic v8): the dropped 0.2 vs 0.5 series has its Section 5.6 sentence C5 (11%, both sizes), cites "
      "the F_link_app sentence for 0.39 against 0.62, and claims no panel of fig:app-s21 still draws it"
      if VARIANT != "top" else "v8 MOVED.md (F_link_top): names the bottom-row content moved to F_link_net",
      ("Between these temperatures, a fitted monotone link removes a median $11\\%$" in MOVED
       and "a median $0.39$" in MOVED and "against $0.62$" in MOVED and "F_link_app/MOVED.md" in MOVED
       and "still drawn" not in MOVED and "\\ref{fig:app-s21}c)" not in MOVED)
      if VARIANT != "top" else ("F_link_net" in MOVED and "fig:exp-link" in MOVED))
_moved_need = {"both": ("Finite Gibbs model, inside the premises of Theorem 3",
                        "CIFAR-5m networks, outside these premises", "thm:peer", "app:exp-s23",
                        "the projector distance $\\Delta_\\Pi$", "span all pairs within the three arms",
                        "lower median warp share", "lower median difference after the link is undone"),
               "top": ("Finite Gibbs model, inside the premises of Theorem 3", "thm:peer", "app:exp-s23",
                       "lower median difference after the link is undone"),
               "net": ("CIFAR-5m networks, outside these premises", "the projector distance $\\Delta_\\Pi$",
                       "span all pairs within the three arms", "lower median warp share", "fig:app-s21")}[VARIANT]
check("v8 MOVED.md carries the dropped row headers, the thm:peer pointer and the pick and range sentences it still needs",
      all(x in MOVED for x in _moved_need), [x for x in _moved_need if x not in MOVED])
import matplotlib.text as _mtext  # noqa: E402
FIG_TEXTS = sorted({t.get_text() for t in fig.findobj(_mtext.Text) if t.get_visible() and t.get_text().strip()})
check("v7: no figure text states a setting or a status (no sample size, 'certified', 'preregistered', 'frozen')",
      not any(re.search(r"(\d{3},\d{3}|\{,\}|certified|preregist|replicat|frozen|empirical risk)", s_)
              for s_ in FIG_TEXTS), FIG_TEXTS)
check("v8: no row header and no setting on the figure (no 'premises', 'Gibbs', 'CIFAR', 'networks', 'model'), and "
      "nothing of InfoNCE at 0.2 vs 0.5",
      not any(re.search(r"(premises|Gibbs|CIFAR|networks|model|0\.2 vs|InfoNCE at)", s_) for s_ in FIG_TEXTS),
      FIG_TEXTS)

# ---- every v7 assert: kept (same name, passing), replaced by a named v8 assert, or not applicable here -------------------
V7_NUM = load(V7_NUM_P)
V7_ASSERTS = [a["assert"] for a in V7_NUM["asserts"]]
REPLACED = {"(b) shares the log colour scale of (e) (same count, same grey); its counts lie inside that scale":
            "v8: (b) shares the log colour scale of (e) (same count, same colour); its counts lie inside that scale",
            "v7: on each header line the key starts at least 0.25 in right of the header text":
            "v8: one key line, centred, inside the figure, above every panel letter"}
NOT_APPLICABLE = {}
if VARIANT == "top":
    REPLACED[GUTTER_V7_NAME] = ("v8 gutters, top row only (F_link_top): (a) >= 0.18 in from the (b) y label, (b) ticks "
                                ">= 0.12 in from the (c) y label, (c) tick labels >= 0.05 in apart and inside; key line "
                                ">= 0.02 in above the letters")
    REPLACED[HIDDEN_V7_NAME] = HIDDEN_NAME
    for n_ in ("marks are nudged only on the SOG row (the filled circle and the diamond coincide there)",
               "(d) row membership: largest nudge plus half mark < half the pitch",
               "(f) every marker clears the markers and lines of the other classes by >= 0.5 pt"):
        NOT_APPLICABLE[n_] = "layout of the network row, which F_link_top does not draw (F_link and F_link_net keep it)"
if VARIANT == "net":
    REPLACED[GUTTER_V7_NAME] = ("v8 gutters, bottom row only (F_link_net): (a) marks >= 0.18 in from the (b) y label, "
                                "(b) ticks >= 0.12 in from the (c) y label, (c) tick labels >= 0.05 in apart and inside; "
                                "key line >= 0.02 in above the letters")
    REPLACED[HIDDEN_V7_NAME] = HIDDEN_NAME
    for n_ in ("(c) keeps at least 0.9 in of width after the gutter",
               "(c) marks are dodged only at the smallest N, where the two series cross",
               "(c) every circle and square clears every other mark by >= 0.3 pt",
               "(c) every mark clears the y spine by >= 0.5 pt"):
        NOT_APPLICABLE[n_] = "layout of the finite-Gibbs row, which F_link_net does not draw (F_link and F_link_top keep it)"
_names8 = {a["assert"] for a in ASSERTS}
ACCOUNT = {"kept": [n_ for n_ in V7_ASSERTS if n_ in _names8],
           "replaced": {n_: REPLACED[n_] for n_ in V7_ASSERTS if n_ not in _names8 and n_ in REPLACED},
           "not_applicable_in_this_variant": {n_: NOT_APPLICABLE[n_] for n_ in V7_ASSERTS if n_ in NOT_APPLICABLE}}
_missing = [n_ for n_ in V7_ASSERTS if n_ not in _names8 and n_ not in REPLACED and n_ not in NOT_APPLICABLE]
_bad_repl = [n_ for n_ in ACCOUNT["replaced"].values() if n_ not in _names8]
check("v8: every v7 assert is kept (same name, passing), replaced by a named v8 assert that passes, or not applicable "
      "to this variant", not _missing and not _bad_repl and len(V7_ASSERTS) == 166,
      {"missing": _missing, "replacement_not_run": _bad_repl, "n_v7": len(V7_ASSERTS),
       "n_kept": len(ACCOUNT["kept"]), "n_replaced": len(ACCOUNT["replaced"]),
       "n_not_applicable": len(ACCOUNT["not_applicable_in_this_variant"])})

# ---- colour report (CIEDE2000 under Machado 2009 CVD; cvdlib of the v8 style folder) --------------------------------
import cvdlib  # noqa: E402
CVD_PAIRS = {}
_cols = {"across": C_ACROSS, "control": C_CONTROL, "fit": C_FIT, "INK": INK, "NULL_LINE": S.NULL_LINE,
         "NULL_FILL": S.NULL_FILL}
_pairs = [("across", "control"), ("across", "INK"), ("control", "INK"), ("across", "NULL_LINE"),
          ("control", "NULL_LINE"), ("across", "NULL_FILL"), ("control", "NULL_FILL"), ("fit", "INK")]
for a_, b_ in _pairs:
    CVD_PAIRS[f"{a_} vs {b_}"] = {v_: round(cvdlib.dE(_cols[a_], _cols[b_], v_), 1) for v_ in ("normal", "deutan",
                                                                                              "protan", "tritan")}
    CVD_PAIRS[f"{a_} vs {b_}"]["grey_dL"] = round(abs(S.lab(S.simulate(_cols[a_], "grey"))[0]
                                                      - S.lab(S.simulate(_cols[b_], "grey"))[0]), 1)
check("v8: InfoNCE vs spectral (orange) and the positive control (blue) stay apart for normal, deuteranopic and "
      "protanopic readers (CIEDE2000 >= 15) and differ in greyscale lightness by >= 10 L*; their marks also differ in "
      "shape and fill", min(CVD_PAIRS["across vs control"][v_] for v_ in ("normal", "deutan", "protan")) >= 15
      and CVD_PAIRS["across vs control"]["grey_dL"] >= 10, CVD_PAIRS["across vs control"])
COLOURS_V8 = {"semantic": {"link.across_objectives (InfoNCE vs spectral; open squares = link undone)": C_ACROSS,
                           "link.positive_control (exact link of Theorem 3)": C_CONTROL,
                           "best monotone link in (e) (LINK_TEXT['across'])": C_FIT,
                           "theory (predictions, link of the theorem, requirement bar)": INK,
                           "within one objective (band, box)": [S.NULL_FILL, S.NULL_LINE],
                           "references (tolerance, separator, N*, full share)": S.NULL_LINE,
                           "row guides": S.GUIDE, "density": "S.cmap_density() = Tol YlOrBr steps 1-6",
                           "direct labels": {"link undone, vs spectral": T_ACROSS, "exact link": T_CONTROL,
                                             "predicted, required, link of Theorem 3": INK}},
              "cvd_dE2000": CVD_PAIRS, "cvdlib_sha256": sha(cvdlib.__file__),
              "text_contrast_on_white": {"across": round(S.contrast(T_ACROSS), 2),
                                         "control": round(S.contrast(T_CONTROL), 2)}}

GEOM = {"H": H, "variant": VARIANT, "x_shared": {"row_label_edge": XL, "d_floor_axes": [FX0, FW_],
                                                 "d_log_axes": [AX0, AXW, XLO, XHI], "e_axes_x": [BX0, BW],
                                                 "f_axes_x": [CX0, CW_]},
        "key_line": KEY_GEOM}
if DRAW_TOP:
    GEOM["top"] = {"axes_y": [TY0, TH], "a_axes_x": [TA0, TAW], "b_axes": [TBX0, TY0, TBW, TBW],
                   "c_axes_x": [TC_X0, TC_W], "letters_top": TY1 + LET_DY}
if DRAW_BOT:
    GEOM["bottom"] = {"axes_y": [AX_BOT, AX_TOP], "row_pitch": PITCH, "group_gap": GG, "e_axes": [BX0, BY0, BW, BW],
                      "f_dodge_pt": C_DODGE_PT, "e_key_baselines": list(Y_KEY_E), "letters_top": AX_TOP + LET_DY}
NUM = {
    "caption": {"path": CAPTION8_P, "sha256": CAP8["sha256"], "text": CAPTION8, "words_total": CAP8["words_total"],
                "words_bold": CAP8["words_bold"], "tokens_total_v7_counter": CAP8["tokens_total_v7_counter"]},
    "caption_v7_archived": {"path": CAPTION7_P, "sha256": sha(CAPTION7_P), "text": CAPTION7},
    "caption_v6_archived": {"path": CAPTION_P, "sha256": sha(CAPTION_P), "text": CAPTION},
    "moved": {"path": MOVED_P, "sha256": sha(MOVED_P)},
    "moved_v7_archived": {"path": MOVED7_P, "sha256": sha(MOVED7_P)},
    "critic_v8": dict(CRITIC_V8, f_link_app_moved={"path": APP_MOVED_P, "sha256": sha(APP_MOVED_P)},
                      prev_maker_outputs=os.path.join(HERE, "_prev_maker_v8")),
    "v6_maker": {"path": V6_MAKER, "sha256": sha(V6_MAKER)},
    "v7_maker": {"path": V7_MAKER, "sha256": sha(V7_MAKER)},
    "v7_numbers": {"path": V7_NUM_P, "sha256": sha(V7_NUM_P), "n_asserts": len(V7_ASSERTS)},
    "v7_assert_accounting": ACCOUNT,
    "figure": {"id": FIG_ID, "variant": VARIANT,
               "version": "v8 (2026-10-10 evening): v7 with no row headers, the v8 Tol semantic colours, bold ticks and "
                          "axis titles, InfoNCE at 0.2 vs 0.5 no longer drawn in (f), one key line at the top; data, "
                          "pick rules and encodings by shape unchanged",
               "height_in": {"v6": 4.23, "v7": 3.885, "v8": float(H)},
               "label": {"both": "fig:exp-link", "top": "fig:exp-link (alternative: top row only)",
                         "net": "appendix alternative (network row of fig:exp-link)"}[VARIANT],
               "section": "5.6 sec:exp-link", "theorem_number": THM,
               "panels_drawn": {p_: {"a": "S2.3 exact T_E agreement vs N", "b": "S2.3 kernel entries at 2^40",
                                     "c": "S2.3 scale-normalized discrepancy vs N", "d": "S2.1 Delta/eps",
                                     "e": "S2.1 kernel entries", "f": "S2.1 staircase"}[p_] for p_ in PANEL_LETTER},
               "panel_letters_printed": PANEL_LETTER,
               "size_in": CHK["size_in"], "min_text_pt": CHK["min_text_pt"], "min_glyph_pt": CHK["min_glyph_pt"],
               "n_type3": CHK["n_type3"], "style_check": CHK, "finalize_texts_changed": N_FINAL,
               "pdffonts": pf.splitlines(), "font_names": FONT_NAMES,
               "text_overlaps_gt_0.005in": clash, "text_outside_figure": outside, "text_below_7.5pt": small,
               "group_label_fit": GROUP_FIT, "label_hits": LAB_HITS, "gutters_in": GUTTER,
               "row_separation": ROW_SEP, "nudged_rows_d": NUDGE, "dodged_N_c": TC_DODGE,
               "panel_f_marker_clearance_pt_min": None if C_MIN is None else [C_MIN, C_CLEAR[C_MIN]],
               "mark_to_box_clearance_pt": BOX_CLEAR, "panel_c_circle_square_clearance_pt": TC_CLEAR,
               "gap_bracket_to_row_label_in": GAP_GROUP, "geometry_in": GEOM, "figure_texts": FIG_TEXTS,
               "colours_v8": COLOURS_V8,
               "views": {"png_200dpi": os.path.basename(PNG), "print_100dpi": os.path.basename(VIEW100),
                         "cvd_grey": os.path.basename(VIEWS)}},
    "sources": {k: {"path": v, "sha256": sha(v)} for k, v in SRC.items()},
    "sources_s23": {k: {"path": v, "sha256": sha(v)} for k, v in SRC23.items()},
    "s23_cache_files": {k: [os.path.basename(v), sha(v)] for k, v in CACHE_FILES23.items()},
    "s23_panel_b_runs": {r_: {"path": kpath(r_), "sha256": sha(kpath(r_))} for r_ in (RUN_A23, RUN_B23)},
    "ladder_files": {c: [[os.path.basename(f), sha(f)] for f in fs] for c, fs in FILES.items()},
    "feats": {"A": {"path": FA, "sha256": sha(FA)}, "B": {"path": FB, "sha256": sha(FB)}},
    "style": {"path": STYLE_PATH, "sha256": STYLE_SHA, "version": S.STYLE_VERSION},
    "script": {"path": os.path.abspath(__file__), "sha256": sha(os.path.abspath(__file__))},
    "panel_top": panel_top,
    "panel_top_plotted": {
        "drawn": DRAW_TOP,
        "a": {"x_log2N": GRID23, "cross_share": [float(v) for v in TE_AB], "within_band_lo": [float(v) for v in TE_W_LO],
              "within_band_hi": [float(v) for v in TE_W_HI], "prediction": "panel_top.mc_TE_AB_curve",
              "vertical_line_log2N": NSTAR},
        "b": {"hexbin": {"gridsize": GRIDSIZE, "extent": [KLO, KHI, KLO, KHI], "norm": "LogNorm %r" % HEX_NORM,
                         "cmap": "S.cmap_density() (Tol YlOrBr steps 1-6)", "mincnt": 1,
                         "n_hex": int(len(HEX_COUNTS_T)),
                         "centres_counts": [[float(a), float(b), float(c)] for (a, b), c in
                                            zip(HEX_XY_T, HEX_COUNTS_T)]},
              "line": "panel_top.panel_b_pick.theory_line"},
        "c": {"x_log2N": GRID23, "x_drawn_cross": [float(v) for v in GX_AB], "x_drawn_cf": [float(v) for v in GX_CF],
              "cross_median_q10_q90": {k: [float(v) for v in GS_AB[k]] for k in GS_AB},
              "cf_median_q10_q90": {k: [float(v) for v in CF_T[k]] for k in CF_T},
              "within_band_lo": [float(v) for v in GS_W_LO], "within_band_hi": [float(v) for v in GS_W_HI],
              "predicted_limit": OMEGA,
              "labels": {"predicted": {"x_log2N_right": 40.9, "y_baseline": None if TC_PRED_Y is None
                                       else float(TC_PRED_Y)},
                         "link_undone": {"x_log2N_left": None if TC_CF_X is None else float(TC_CF_X),
                                         "y_center": TC_CF_Y}, "spine_top": 1.0},
              "reference_slope": {"x_log2N": [float(v) for v in REF_X], "y": [float(v) for v in REF_Y],
                                  "rule": "y = 0.012 * 2^(-(x - 22)/2): slope -1/2 in log N, placed below the data "
                                          "(position not data)"},
              "ylim": list(TC_YLIM)}},
    "panel_d": panel_a,
    "panel_d_plotted": {"drawn": DRAW_BOT, "classes_drawn": list(A_CLASSES), "within_min_max_ratio": A_WITHIN,
                        "marks": {f"{c}|{r}": {k: d[k] for k in ("ratio_median", "ratio_q05", "ratio_q95")}
                                  for (c, r), d in A_MARK.items()},
                        "floor_rows_exact_link": FLOOR_ROWS, "separator_between": ["dSOG_t0p1", "dIso"],
                        "floor_note": "exact link drawn at the '~0' floor: exactly 0 on T_MST, T_O, T_N,10 (64/64); "
                                      "T_E 62/64 exactly 0, max %.2e; Delta_iso max %.2e of eps" % (
                                          A_MARK[("bern_self", "dE")]["ratio_max"],
                                          A_MARK[("bern_self", "dIso")]["ratio_max"])},
    "panel_e": panel_b,
    "panel_e_hexbin": {"drawn": DRAW_BOT, "gridsize": GRIDSIZE, "extent": [KLO, KHI, KLO, KHI], "bins": "log",
                       "mincnt": 1, "cmap": "S.cmap_density() (Tol YlOrBr steps 1-6)", "norm": HEX_NORM,
                       "n_hex": int(len(HEX_COUNTS)),
                       "centres_counts": [[float(a), float(b), float(c)] for (a, b), c in zip(HEX_XY, HEX_COUNTS)]},
    "panel_f": panel_c,
    "panel_f_plotted": {"drawn": DRAW_BOT, "classes_drawn": C_CLASSES_DRAWN,
                        "classes_computed_not_drawn": ["cross_A_Atau"],
                        "requirement_level": REQ, "dodge_data_units": C_DX},
    "ranges_under_marks": HIDDEN,
    "text_numbers": TEXT_NUMBERS,
    "asserts": ASSERTS,
}
with open(os.path.join(HERE, "numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1)
NUM["outputs"] = {p: sha(os.path.join(HERE, p)) for p in (f"{FIG_ID}.pdf", f"{FIG_ID}.png",
                                                          "_view_print_100dpi.png", "_views_cvd_grey.png")}
with open(os.path.join(HERE, "numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1)
print(json.dumps({"id": FIG_ID, "variant": VARIANT,
                  "check": {k: CHK[k] for k in ("size_in", "min_text_pt", "min_glyph_pt", "n_type3", "fonts",
                                               "low_contrast_text", "n_tick_labels_checked", "n_axis_titles_checked",
                                               "chromatic_data_colours", "off_palette_colours")},
                  "group_fit": GROUP_FIT, "gutters": GUTTER, "dodge_c": TC_DODGE, "nudge_d": NUDGE,
                  "top_c": [TC_X0, TC_W], "caption_words": [CAP8["words_total"], CAP8["words_bold"]],
                  "share_theory_top": [SHARE23_PICK, float(np.median(SHARE23_ALL))], "share_theory_net": SHARE_THEORY,
                  "cvd_across_vs_control": CVD_PAIRS["across vs control"],
                  "v7_asserts": {k: len(v) for k, v in ACCOUNT.items()}, "n_asserts": len(ASSERTS)}, indent=1))
print("OK")
