"""F_link_app v8 (Appendix F, Contrastive Networks, app:exp-s21, label fig:app-s21). Role: v8 maker, 2026-10-10.

Message: InfoNCE and the spectral loss disagree on order criteria as well as geometric ones at both sample sizes and for
both InfoNCE temperatures, against the pattern Theorem link proves inside its premises (the link checks, diamonds).

v8 changes (author on v7 Figure 6, 2026-10-10; rules for every experiment figure), started from a byte copy of
~/hrh_v2_figs/v6/F_link_app/make_F_link_app.py (v6.2; copy kept in _v6_ref/, sha256 in numbers.json):
  1 no header that states a setting: the 'N = 50,000' / 'N = 500,000' headers over (a, b), the 'guard fails' / 'guard
    holds' lines, the '<- within  outside ->' cues and the N tags inside (c) are gone; the caption names the setting
    of each panel and the Results text carries the guard status (all still computed and asserted here);
  2 no black-and-white: style hrh_style_v8 (Paul Tol system, sha256 asserted). InfoNCE against spectral (both
    temperatures) = S.LINK['across'] (Tol vibrant orange; circle for 0.2, square for 0.5); the link checks = S.LINK[
    'control'] (Tol bright blue; open diamond self, filled diamond peer); within one objective = NULL_FILL box with a
    NULL_LINE edge (a reference, grey on purpose); theory/tolerance lines NULL_LINE;
  3 tick labels and axis titles bold, bold math (S.finalize before savefig; S.check asserts it); row labels are the
    bare symbols of Figure 6 (no '(projector)', '(isotonic)'), set bold as the tick labels they are;
  4 secondary comparisons that the Results text states are not drawn: InfoNCE at 0.2 against 0.5 (computed, asserted,
    listed in MOVED.md with its sentence) and the 50,000-image staircase, which equals the 500,000-image staircase to
    within the asserted bound except for the peer check (both sizes stay in numbers.json; MOVED.md gives the sentence).
    Panel (c) is therefore one staircase at N = 500,000, and the key holds four classes and the within band.

Panels (CIFAR-5m, S2.1, embedding z, cosine kernel, 2048 held-out images):
 (a), (b) Delta / eps per criterion (log x) at N = 50,000 (a) and 500,000 (b); rows in lattice order as Figure 6 (d).
     Grey box per row: min-max over the 96 within-objective pairs (three arms). Marks (medians over pairs), on four
     levels top -> bottom: link check self (open diamond; at the '~0' floor on order rows and Delta_iso), InfoNCE 0.2 vs
     spectral (filled circle), link check peer (filled diamond), InfoNCE 0.5 vs spectral (filled square). No link check
     on Delta_Pi.
 (c)  Explained-share staircase at N = 500,000: cumulative cross-fitted share of the squared-distance discrepancy
     explained by scale, + affine, + global monotone, + per-row monotone (ladder row.warpshare_expl_cf_*), medians over
     pairs for the four drawn classes; self dashed, peer dotted, the classes across objectives solid.

Read-only inputs (opened read-only; nothing imported from the experiment code):
  ~/hrh_v2_stage2_20261008/results/ladders/S2_1_main/z/*.json        per-pair ladders (deltas, row)
  ~/hrh_v2_stage2_20261008/prereg/eps_S2_1.json                       frozen tolerances, key eps.z
  ~/hrh_v2_stage2_20261008/results/S2_1/s21_summary.json              preregistered decision tables
  ~/hrh_v2_figs/v8/F_link_app/_inputs/tab_app_s21_full_excerpt.tex    copy of the tabular of Table tab:app-s21-full
  ~/hrh_v2_figs/v8/F_link_app/_inputs/staircase_F_criteria_snapshot.py  copy of F_criteria/staircase.py (labels, setup)
  ~/hrh_v2_figs/v8/_style/hrh_style_v8.py                             frozen style (sha256 asserted)
Writes only into this directory: F_link_app.pdf, F_link_app.png (pdftoppm 200 dpi), _view_print_100dpi.png,
_views_cvd_grey.png, numbers.json.
Run: OMP_NUM_THREADS=1 nice -n 10 taskset -c 0-5 ~/hrh_v2_figs_venv/bin/python -B make_F_link_app.py > run.log 2>&1
"""
import glob
import hashlib
import importlib.util
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
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

STYLE_PATH = os.path.expanduser("~/hrh_v2_figs/v8/_style/hrh_style_v8.py")
STYLE_SHA = "fe994067dfefba8c0fe34309a67609e85f82288f0f785e69d2bef98e33fc79a8"
assert hashlib.sha256(open(STYLE_PATH, "rb").read()).hexdigest() == STYLE_SHA, "frozen style module changed"
sys.path.insert(0, os.path.dirname(STYLE_PATH))
import hrh_style_v8 as S  # noqa: E402

assert S.STYLE_SHA256 == STYLE_SHA
S.use()

HOME = os.path.expanduser("~")
S2 = f"{HOME}/hrh_v2_stage2_20261008"
LD = f"{S2}/results/ladders/S2_1_main/z"
SRC = {
    "eps": f"{S2}/prereg/eps_S2_1.json",
    "summary": f"{S2}/results/S2_1/s21_summary.json",
    "report": f"{S2}/results/S2_1/S2_1_report.md",
    "table_excerpt": f"{HERE}/_inputs/tab_app_s21_full_excerpt.tex",
    "staircase_convention": f"{HERE}/_inputs/staircase_F_criteria_snapshot.py",
    "F_link_app_v6_maker_started_from": f"{HERE}/_v6_ref/make_F_link_app.py",
}
SC_SHA = "56214a368e83179068a3a92a247d4140e9d137d2262d6bcf5c4d12c138291e4e"   # F_criteria/staircase.py 10:50
V6_MAKER_SHA = None                                                             # recorded below (provenance)
NS = [50000, 500000]
NTAG = {50000: r"$N=50{,}000$", 500000: r"$N=500{,}000$"}


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
assert sha(SRC["staircase_convention"]) == SC_SHA, "staircase snapshot changed"
_spec = importlib.util.spec_from_file_location("staircase_snapshot", SRC["staircase_convention"])
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)
EPS = load(SRC["eps"])["eps"]["z"]
SUM = load(SRC["summary"])
assert SUM["interim"] is False and SUM["R"] == 64 and SUM["verdict"] == "falsified", (SUM["interim"], SUM["R"])
ASSERTS = []          # (name, result) recorded in numbers.json


def check(name, cond, detail=None):
    assert cond, (name, detail)
    ASSERTS.append({"assert": name, "ok": True, "detail": detail})


# =============================================================================================
# data: pair classes by filename (both N), Delta / eps per rung
# =============================================================================================
PAT = re.compile(r"S2\.1_(A|Atau|B)_N(\d+)_r(\d+)__S2\.1_(A|Atau|B)_N(\d+)_r(\d+)(__bern)?\.json$")
FILES = {}
for f in sorted(glob.glob(LD + "/S2.1_*.json")):
    m = PAT.search(os.path.basename(f))
    if not m:
        continue                                   # fixed-epoch controls
    a, na, ra, b, nb, rb, bern = m.groups()
    assert na == nb, f
    if bern:
        assert a == b == "A"
        c = "bern_self" if ra == rb else "bern_peer"
    elif a == b:
        c = "within_" + a
    else:
        c = f"cross_{a}_{b}"
    FILES.setdefault((int(na), c), []).append(f)
N_EXPECT = {"within_A": 32, "within_B": 32, "within_Atau": 32, "cross_A_B": 64, "cross_A_Atau": 64,
            "cross_Atau_B": 64, "bern_self": 64, "bern_peer": 32}
for N in NS:
    got = {c: len(FILES.get((N, c), [])) for c in N_EXPECT}
    check(f"pair counts per class at {N}", got == N_EXPECT, got)
check("no other N or class in the main ladders", sorted({k[0] for k in FILES}) == NS
      and {k[1] for k in FILES} == set(N_EXPECT))
LAD = {k: [load(f) for f in fs] for k, fs in FILES.items()}
for k, rows in LAD.items():
    assert all(not r["failed_run"] for r in rows), k

# rows top -> bottom in lattice order; labels exactly as F_link (a), single spaces
ROWS = [
    ("quot", "dPi", r"$\Delta_\Pi$ (projector)"),
    ("geom", "dCKA", r"$1-\mathrm{CKA}$"),
    ("geom", "Gs", r"$G_s$"),
    ("geom", "dAff", r"$T_{G/\mathrm{Aff}}$"),
    ("geom", "dSOG_t0p1", r"SOG, $\tau=0.1$"),
    ("iso", "dIso", r"$\Delta_{\mathrm{iso}}$ (isotonic)"),
    ("order", "dE", r"$T_E$"),
    ("order", "dMST", r"$T_{\mathrm{MST}}$"),
    ("order", "dO", r"$T_O$"),
    ("order", "dN_k10", r"$T_{N,10}$"),
]
RUNGS = [r for _, r, _ in ROWS]
ORDER_RUNGS = ["dE", "dMST", "dO", "dN_k10"]
GEOM_RUNGS = ["dCKA", "Gs", "dAff", "dSOG_t0p1"]
WITHIN = ["within_A", "within_B", "within_Atau"]
CROSS_OBJ = ["cross_A_B", "cross_Atau_B"]
MARK_CLASSES = ["bern_self", "bern_peer", "cross_A_B", "cross_Atau_B", "cross_A_Atau"]


def deltas(N, c, r):
    return np.array([row["deltas"][r] for row in LAD[(N, c)]], float)


panel_ab = {}
for N in NS:
    P = {"N": N, "eps_source": "eps_S2_1.json:eps.z.<rung>", "rows_top_to_bottom": RUNGS, "classes": {}}
    for c in N_EXPECT:
        P["classes"][c] = {}
        tab = SUM["tables"][f"{c}_{N}|z"]
        for r in RUNGS:
            v = deltas(N, c, r)
            e = EPS[r]
            t = tab.get(r)
            if c.startswith("bern") and r == "dPi":
                check(f"link check undefined on dPi ({c}, {N})", t is None and np.all(np.isnan(v)))
                P["classes"][c][r] = {"n": int(v.size), "undefined": True}
                continue
            assert np.all(np.isfinite(v)), (N, c, r)
            x = int(np.sum(v <= e))
            check(f"decision count {c} {N} {r} equals s21_summary", t is not None and t["x"] == x and t["n"] == v.size
                  and abs(t["eps"] - e) == 0 and abs(t["median"] - float(np.median(v))) < 1e-12,
                  {"x": x, "n": int(v.size), "table": None if t is None else [t["x"], t["n"], t["decision"]]})
            q = v / e
            P["classes"][c][r] = {
                "n": int(v.size), "eps": e, "x_within_eps": x, "decision": t["decision"],
                "ratio_min": float(q.min()), "ratio_q05": float(np.quantile(q, 0.05)),
                "ratio_median": float(np.median(q)), "ratio_q95": float(np.quantile(q, 0.95)),
                "ratio_max": float(q.max()), "n_exact_zero": int(np.sum(v == 0)),
                "values_delta": [float(z) for z in v],
                "source_key": f"{os.path.basename(LD)}/<{c} N{N} files>:deltas.{r}; s21_summary.json:tables."
                              f"{c}_{N}|z.{r}"}
    panel_ab[N] = P

# Table app-s21-full (both blocks) against the computed counts
TAB_ROW = {"InfoNCE at 0.2, within": "within_A", "InfoNCE at 0.5, within": "within_Atau",
           "spectral, within": "within_B", "InfoNCE at 0.2 vs spectral": "cross_A_B",
           "InfoNCE at 0.2 vs 0.5": "cross_A_Atau", "InfoNCE at 0.5 vs spectral": "cross_Atau_B",
           "link check, self": "bern_self", "link check, peer": "bern_peer"}
TAB_COLS = ["dN_k10", "dO", "dE", "dMST", "dSOG_t0p1", "dAff", "Gs", "dCKA", "dIso", "dPi"]
tex = open(SRC["table_excerpt"]).read().split("\n")
block, tab_checked = None, {50000: 0, 500000: 0}
for ln in tex:
    if ln.startswith("500k"):
        block = 500000
    elif ln.startswith("50k"):
        block = 50000
    if block is None or "&" not in ln:
        continue
    cells = [s.strip().rstrip("\\").strip() for s in ln.split("&")]
    c = TAB_ROW[cells[1]]
    for col, cell in zip(TAB_COLS, cells[2:]):
        if cell == "n/a":
            assert c.startswith("bern") and col == "dPi"
            continue
        mm = re.match(r"(\d+)/(\d+)", cell)
        assert mm, cell
        d = panel_ab[block]["classes"][c][col]
        assert int(mm.group(1)) == d["x_within_eps"], (block, cells[1], col, cell)
        assert int(mm.group(2)) == d["n"], (block, cells[1], col, cell)
        tab_checked[block] += 1
check("Table tab:app-s21-full equals the computed counts (both blocks)",
      tab_checked == {50000: 78, 500000: 78}, tab_checked)

# the message, asserted from the pairs
for N in NS:
    for c in CROSS_OBJ:
        check(f"{c} at {N}: no pair within tolerance on any criterion",
              all(panel_ab[N]["classes"][c][r]["x_within_eps"] == 0 for r in RUNGS))
        check(f"{c} at {N}: median Delta/eps > 1 on every criterion",
              all(panel_ab[N]["classes"][c][r]["ratio_median"] > 1 for r in RUNGS),
              min(panel_ab[N]["classes"][c][r]["ratio_median"] for r in RUNGS))
    check(f"link check self at {N}: 64/64 within on order rungs and Delta_iso, 0/64 on geometry",
          all(panel_ab[N]["classes"]["bern_self"][r]["x_within_eps"] == 64 for r in ORDER_RUNGS + ["dIso"])
          and all(panel_ab[N]["classes"]["bern_self"][r]["x_within_eps"] == 0 for r in GEOM_RUNGS))
check("within one objective at 500k: every pair within tolerance on every criterion",
      all(panel_ab[500000]["classes"][c][r]["x_within_eps"] == 32 for c in WITHIN for r in RUNGS))
GUARD_50K = {c: [r for r in RUNGS if panel_ab[50000]["classes"][c][r]["x_within_eps"] < 32] for c in WITHIN}
check("within-objective guard at 50k fails (spectral on six criteria, InfoNCE 0.2 on four, InfoNCE 0.5 on none)",
      GUARD_50K == {"within_A": ["dCKA", "dAff", "dE", "dO"], "within_B": ["dPi", "Gs", "dSOG_t0p1", "dIso", "dMST",
                                                                          "dN_k10"], "within_Atau": []}, GUARD_50K)
# the per-panel guard status lines ('fails' / 'holds') are read from the table decisions
GUARD_STATUS = {}
for N in NS:
    dec = {r: [panel_ab[N]["classes"][c][r]["decision"] for c in WITHIN] for r in RUNGS}
    fails_on = [r for r in RUNGS if any(d != "reliable" for d in dec[r])]
    GUARD_STATUS[N] = {"fails_on_rungs": fails_on, "decisions": dec,
                       "status": "fails" if fails_on else "holds"}
check("guard status per panel: fails on every criterion at 50k (InfoNCE 0.2 or spectral), holds on all at 500k",
      GUARD_STATUS[50000]["fails_on_rungs"] == RUNGS and GUARD_STATUS[500000]["fails_on_rungs"] == []
      and all(d == "unreliable" for r in RUNGS for d in GUARD_STATUS[50000]["decisions"][r] if d != "reliable"),
      {N: GUARD_STATUS[N]["fails_on_rungs"] for N in NS})
# peer link check equals within InfoNCE 0.2 on order rungs (an exact increasing link leaves order unchanged)
PEER_EQ = {}
for N in NS:
    wk = {os.path.basename(f): i for i, f in enumerate(FILES[(N, "within_A")])}
    pk = {os.path.basename(f).replace("__bern", ""): i for i, f in enumerate(FILES[(N, "bern_peer")])}
    assert set(wk) == set(pk), N
    for r in ORDER_RUNGS:
        d = max(abs(LAD[(N, "bern_peer")][pk[k]]["deltas"][r] - LAD[(N, "within_A")][wk[k]]["deltas"][r]) for k in wk)
        PEER_EQ[f"{N}|{r}"] = d
        check(f"peer link check equals within InfoNCE 0.2 on {r} at {N} (same run pairs)", d < 1e-12, d)

# plotted summaries of (a), (b)
A_WITHIN, A_MARK = {}, {}
for N in NS:
    A_WITHIN[N] = {}
    for r in RUNGS:
        q = np.concatenate([deltas(N, c, r) / EPS[r] for c in WITHIN])
        A_WITHIN[N][r] = (float(q.min()), float(q.max()))
    for c in MARK_CLASSES:
        for r in RUNGS:
            d = panel_ab[N]["classes"][c][r]
            if d.get("undefined"):
                continue
            A_MARK[(N, c, r)] = d
FLOOR_ROWS = ["dIso", "dE", "dMST", "dO", "dN_k10"]           # link check, self drawn at the '~0' floor
for N in NS:
    for r in FLOOR_ROWS:
        assert A_MARK[(N, "bern_self", r)]["ratio_max"] < 0.002, (N, r)
    for r in ("dMST", "dO", "dN_k10"):
        check(f"link check exactly 0 on {r} at {N}", A_MARK[(N, "bern_self", r)]["n_exact_zero"] == 64)
    check(f"link check on T_E at {N}: <= 1e-11 of eps", A_MARK[(N, "bern_self", "dE")]["ratio_max"] < 1e-11,
          A_MARK[(N, "bern_self", "dE")]["ratio_max"])
# reading cues of the caption, asserted
for N in NS:
    for c in CROSS_OBJ:
        check(f"{c} at {N}: median right of the grey band on every criterion",
              all(A_MARK[(N, c, r)]["ratio_median"] > A_WITHIN[N][r][1] for r in RUNGS),
              {r: [A_MARK[(N, c, r)]["ratio_median"], A_WITHIN[N][r][1]] for r in RUNGS})
    check(f"peer link check at {N}: median inside the grey band on every order criterion",
          all(A_WITHIN[N][r][0] <= A_MARK[(N, "bern_peer", r)]["ratio_median"] <= A_WITHIN[N][r][1]
              for r in ORDER_RUNGS))
    check(f"link checks at {N}: medians right of the band and of the line on every geometric criterion",
          all(A_MARK[(N, c, r)]["ratio_median"] > max(1.0, A_WITHIN[N][r][1]) for c in ("bern_self", "bern_peer")
              for r in GEOM_RUNGS))
CROSS_MIN_VS_WITHIN_MAX = {f"{N}|{c}|{r}": [A_MARK[(N, c, r)]["ratio_min"], A_WITHIN[N][r][1]]
                           for N in NS for c in CROSS_OBJ for r in RUNGS}
N_CROSS_PAIRS_ABOVE_ALL_WITHIN = {f"{N}|{c}": sum(A_MARK[(N, c, r)]["ratio_min"] > A_WITHIN[N][r][1] for r in RUNGS)
                                  for N in NS for c in CROSS_OBJ}
# F_link (500k) checks repeated, plus the 50k values recorded
wmed_g = [panel_ab[500000]["classes"][c][r]["ratio_median"] for c in WITHIN for r in RUNGS if r != "dPi"]
check("500k within medians 0.21-0.97 (Delta_Pi apart)", (round(min(wmed_g), 2), round(max(wmed_g), 2)) == (0.21, 0.97),
      [min(wmed_g), max(wmed_g)])
MED_RANGE = {}
for N in NS:
    for c in CROSS_OBJ + ["cross_A_Atau", "bern_self", "bern_peer"] + WITHIN:
        rr = [r for r in RUNGS if r != "dPi" and not (c == "bern_self" and r in FLOOR_ROWS)]
        meds = [panel_ab[N]["classes"][c][r]["ratio_median"] for r in rr]
        MED_RANGE[f"{N}|{c}"] = {"min": min(meds), "argmin": rr[int(np.argmin(meds))], "max": max(meds),
                                 "argmax": rr[int(np.argmax(meds))], "rungs": rr}
check("500k across-objective (0.2) medians 1.21-6.24", (round(MED_RANGE["500000|cross_A_B"]["min"], 2),
                                                        round(MED_RANGE["500000|cross_A_B"]["max"], 2)) == (1.21, 6.24))
check("500k across-temperature medians 1.05-3.58", (round(MED_RANGE["500000|cross_A_Atau"]["min"], 2),
                                                    round(MED_RANGE["500000|cross_A_Atau"]["max"], 2)) == (1.05, 3.58))
check("500k link check medians on geometry 1.20-4.55", (round(MED_RANGE["500000|bern_self"]["min"], 2),
                                                        round(MED_RANGE["500000|bern_self"]["max"], 2)) == (1.20, 4.55))

# =============================================================================================
# (c) staircase data: cumulative cross-fitted shares per pair, both N
# =============================================================================================
STEPS = ["scale", "affine_extra", "iso_extra", "row_extra"]
check("staircase steps and labels are the shared convention (F_criteria staircase.py snapshot)",
      tuple(STEPS) == SC.STEPS and len(SC.STEP_LABELS) == 4 and "link" not in " ".join(SC.STEP_LABELS),
      list(SC.STEP_LABELS))
STEP_LABEL = list(SC.STEP_LABELS)
C_CLASSES = MARK_CLASSES
panel_c = {"source": "ladder row.warpshare_expl_cf_{scale, affine_extra, iso_extra, row_extra} (cross-fitted, mean of "
                     "the A->B and B->A directions, hrhv2/quotients.py); cumulative sum per pair, then median and "
                     "5/10/90/95% quantiles over pairs", "steps": STEP_LABEL, "N": {}}
STAIR = {}
for N in NS:
    STAIR[N] = {}
    panel_c["N"][N] = {}
    for c in C_CLASSES:
        st = np.array([[row["row"]["warpshare_expl_cf_" + s] for s in STEPS] for row in LAD[(N, c)]], float)
        cum = np.cumsum(st, axis=1)
        om = np.array([row["deltas"]["omega"] for row in LAD[(N, c)]], float)
        STAIR[N][c] = {"median": np.median(cum, 0), "q05": np.quantile(cum, 0.05, 0), "q95": np.quantile(cum, 0.95, 0),
                       "q10": np.quantile(cum, 0.1, 0), "q90": np.quantile(cum, 0.9, 0)}
        panel_c["N"][N][c] = {"n": int(cum.shape[0]),
                              "cum_median": [float(z) for z in STAIR[N][c]["median"]],
                              "cum_q05": [float(z) for z in STAIR[N][c]["q05"]],
                              "cum_q10": [float(z) for z in STAIR[N][c]["q10"]],
                              "cum_q90": [float(z) for z in STAIR[N][c]["q90"]],
                              "cum_q95": [float(z) for z in STAIR[N][c]["q95"]],
                              "step_median": [float(z) for z in np.median(st, 0)],
                              "omega_median": float(np.median(om)),
                              "per_pair_cum": [[float(z) for z in r_] for r_ in cum],
                              "source_key": f"{os.path.basename(LD)}/<{c} N{N} files>:row.warpshare_expl_cf_*; "
                                            "deltas.omega"}
sm = {c: [round(z, 3) for z in panel_c["N"][500000][c]["step_median"]] for c in C_CLASSES}
check("500k step medians as F_link (checked 2026-10-10)",
      sm["cross_A_B"] == [0.140, 0.393, 0.086, 0.013] and sm["cross_A_Atau"] == [0.011, 0.287, 0.088, 0.009]
      and sm["bern_self"][0] == 0.023 and abs(sm["bern_self"][1] - 0.656) <= 0.0015 and
      abs(sm["bern_self"][2] - 0.319) <= 0.0015 and abs(sm["bern_self"][3]) <= 0.001, sm)
OMEGA = {f"{N}|{c}": float(np.median([row["deltas"]["omega"] for row in LAD[(N, c)]]))
         for N in NS for c in list(N_EXPECT)}
check("median omega across objectives 0.147 at 500k and 0.142 at 50k (Appendix item iv)",
      round(OMEGA["500000|cross_A_B"], 3) == 0.147 and round(OMEGA["50000|cross_A_B"], 3) == 0.142, OMEGA)
check("median omega within objectives at 500k 0.012 (InfoNCE 0.2) and 0.002 (spectral) (Appendix item iv)",
      round(OMEGA["500000|within_A"], 3) == 0.012 and round(OMEGA["500000|within_B"], 3) == 0.002)
for N in NS:
    for c in CROSS_OBJ:
        check(f"{c} at {N}: cumulative share at '+ global monotone' below 0.65 for 90% of pairs",
              STAIR[N][c]["q90"][2] < 0.65, float(STAIR[N][c]["q90"][2]))
    check(f"link check self at {N}: cumulative share at '+ global monotone' >= 0.999 (median)",
          STAIR[N]["bern_self"]["median"][2] >= 0.999, float(STAIR[N]["bern_self"]["median"][2]))

# =============================================================================================
# v8: what is drawn, and the numbers that move to the text
# =============================================================================================
DRAWN = ["bern_self", "cross_A_B", "bern_peer", "cross_Atau_B"]          # cross_A_Atau: Results text only (MOVED.md)
check("v8: InfoNCE at 0.2 vs 0.5 is computed but not drawn; the four drawn classes are the two cross-objective "
      "classes and the two link checks", set(DRAWN) | {"cross_A_Atau"} == set(MARK_CLASSES) and "cross_A_Atau" not in DRAWN)
STAIR_N = 500000                                                           # (c) is drawn at this N only
STAIR_DIFF = {c: float(np.max(np.abs(STAIR[50000][c]["median"] - STAIR[500000][c]["median"]))) for c in MARK_CLASSES}
check("v8: the 50,000-image staircase medians equal the 500,000-image ones within 0.005 for every class except the "
      "peer check (which differs by 0.03 to 0.05 at the monotone steps)",
      all(STAIR_DIFF[c] < 0.005 for c in MARK_CLASSES if c != "bern_peer") and 0.03 < STAIR_DIFF["bern_peer"] < 0.05,
      STAIR_DIFF)
A_ATAU = {N: {r: panel_ab[N]["classes"]["cross_A_Atau"][r]["ratio_median"] for r in RUNGS} for N in NS}
check("v8 (moved to text): InfoNCE at 0.2 vs 0.5 lies outside tolerance on every criterion but Delta_Pi at both N "
      "(no pair within tolerance), and within it on Delta_Pi (64/64)",
      all(panel_ab[N]["classes"]["cross_A_Atau"][r]["x_within_eps"] == 0 for N in NS for r in RUNGS if r != "dPi")
      and all(panel_ab[N]["classes"]["cross_A_Atau"]["dPi"]["x_within_eps"] == 64 for N in NS), A_ATAU)
TEXT_NUMBERS = {
    "cross_A_Atau_median_ratio_range_without_dPi": {
        str(N): [min(v for r, v in A_ATAU[N].items() if r != "dPi"), max(v for r, v in A_ATAU[N].items() if r != "dPi")]
        for N in NS},
    "cross_A_Atau_median_ratio_dPi": {str(N): A_ATAU[N]["dPi"] for N in NS},
    "cross_A_Atau_staircase_cum_median": {str(N): panel_c["N"][N]["cross_A_Atau"]["cum_median"] for N in NS},
    "staircase_50k_vs_500k_max_abs_diff_of_cum_medians": STAIR_DIFF,
    "peer_cum_median_global_monotone": {str(N): panel_c["N"][N]["bern_peer"]["cum_median"][2] for N in NS},
    "guard_status": {str(N): GUARD_STATUS[N]["status"] for N in NS},
    "guard_50k_failed_rungs": GUARD_50K,
    "cross_A_B_cum_median_500k": panel_c["N"][500000]["cross_A_B"]["cum_median"],
    "cross_Atau_B_cum_median_500k": panel_c["N"][500000]["cross_Atau_B"]["cum_median"],
    "per_row_step_median_500k": {c: panel_c["N"][500000][c]["step_median"][3] for c in DRAWN},
}
print("TEXT_NUMBERS", json.dumps(TEXT_NUMBERS, default=float)[:2000])

# =============================================================================================
# figure
# =============================================================================================
W = S.TEXTWIDTH_IN
INK, MUTED = S.INK, S.MUTED
LW, MS = S.LW, S.MS
ACROSS, CONTROL = S.LINK["across"], S.LINK["control"]
check("v8 colours: InfoNCE against spectral = S.LINK['across'], link checks = S.LINK['control'] (Tol vibrant orange / "
      "bright blue)", (ACROSS, CONTROL) == ("#EE7733", "#4477AA"), [ACROSS, CONTROL])
BAND_FC, BAND_EC, BAND_LW = S.NULL_FILL, S.NULL_LINE, S.LW["guide"]       # within one objective: a reference, grey
MK = {   # one marker spec per drawn class, shared by (a)-(c) and the key (drawn at the plotted size everywhere)
    "cross_A_B": dict(marker="o", ms=4.3, mfc=ACROSS, mec="white", mew=0.5),
    "cross_Atau_B": dict(marker="s", ms=3.9, mfc=ACROSS, mec="white", mew=0.5),
    "bern_self": dict(marker="D", ms=3.3, mfc="white", mec=CONTROL, mew=MS["open_small_mew"]),
    "bern_peer": dict(marker="D", ms=3.3, mfc=CONTROL, mec=CONTROL, mew=MS["open_small_mew"]),
}


def mark_half(c):
    """Half width and half height (pt) of a class marker, edge included (from the marker path)."""
    m = MarkerStyle(MK[c]["marker"])
    p = m.get_path().transformed(m.get_transform())
    ext_ = p.get_extents()
    return ((ext_.x1 - ext_.x0) / 2 * MK[c]["ms"] + MK[c]["mew"] / 2,
            (ext_.y1 - ext_.y0) / 2 * MK[c]["ms"] + MK[c]["mew"] / 2)


def mark_body(c):
    """Full width and height (pt) of the marker body without its edge (what a range bar would have to exceed)."""
    m = MarkerStyle(MK[c]["marker"])
    p = m.get_path().transformed(m.get_transform())
    ext_ = p.get_extents()
    return (ext_.x1 - ext_.x0) * MK[c]["ms"], (ext_.y1 - ext_.y0) * MK[c]["ms"]


HALF = {c: mark_half(c) for c in DRAWN}
COL = {"bern_self": CONTROL, "bern_peer": CONTROL, "cross_A_B": ACROSS, "cross_Atau_B": ACROSS}
C_STYLE = {"bern_self": dict(lw=LW["ref"], ls=S.LS["dashed"]),
           "bern_peer": dict(lw=LW["ref"], ls=S.LS["dotted"]),
           "cross_A_B": dict(lw=LW["data"], ls="-"),
           "cross_Atau_B": dict(lw=LW["data"], ls="-")}
C_DODGE_PT = {"bern_self": -4.5, "bern_peer": -1.5, "cross_A_B": 1.5, "cross_Atau_B": 4.5}
NAME = {"within": "within one objective", "bern_self": "link check, self", "bern_peer": "link check, peer",
        "cross_A_B": "InfoNCE at 0.2 vs spectral", "cross_Atau_B": "InfoNCE at 0.5 vs spectral"}
# v8 row labels: the bare symbols of Figure 6 (d) (Table tab:app-s21-full column heads)
ROW_LABEL = {"dPi": r"$\Delta_\Pi$", "dCKA": r"$1-\mathrm{CKA}$", "Gs": r"$G_s$", "dAff": r"$T_{G/\mathrm{Aff}}$",
             "dSOG_t0p1": r"SOG$_{0.1}$", "dIso": r"$\Delta_{\mathrm{iso}}$", "dE": r"$T_E$",
             "dMST": r"$T_{\mathrm{MST}}$", "dO": r"$T_O$", "dN_k10": r"$T_{N,10}$"}

# ---- row geometry (pt -> in) ------------------------------------------------------------------------------------
D_PT = 3.3                                        # vertical distance between levels
LEVEL_ORDER = ["bern_self", "cross_A_B", "bern_peer", "cross_Atau_B"]   # top -> bottom
LEVEL_PT = {c: (1.5 - i) * D_PT for i, c in enumerate(LEVEL_ORDER)}
LEVEL = {c: v / 72 for c, v in LEVEL_PT.items()}
top_pt = max(LEVEL_PT[c] + HALF[c][1] for c in DRAWN)
bot_pt = min(LEVEL_PT[c] - HALF[c][1] for c in DRAWN)
BAND_HALF_PT = max(top_pt, -bot_pt) + 0.6
ROW_GAP_PT = 1.6
PITCH = (2 * BAND_HALF_PT + ROW_GAP_PT) / 72
M_END = (BAND_HALF_PT + 1.2) / 72                 # first/last row centre to the axes edge
GG = 0.05                                          # extra gap between groups
groups = [g for g, _, _ in ROWS]
n_gaps = sum(1 for i in range(1, len(groups)) if groups[i] != groups[i - 1])
AX_H = (len(ROWS) - 1) * PITCH + n_gaps * GG + 2 * M_END
AX_BOT = 0.36
TOP_PAD = 0.215                                   # panel letters only (no headers in v8)
H = round(AX_BOT + AX_H + TOP_PAD, 3)
AX_TOP = AX_BOT + AX_H

fig = plt.figure(figsize=(W, H))
fig.set_layout_engine("none")


def axes_in(x0, y0, w, h, **kw):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


ov = axes_in(0, 0, W, H, zorder=5)
ov.set_xlim(0, W)
ov.set_ylim(0, H)
ov.set_xticks([])
ov.set_yticks([])
ov.axis("off")
ov.patch.set_alpha(0)


def ext(t):
    return t.get_window_extent().transformed(fig.dpi_scale_trans.inverted())


# ---- (a), (b) geometry --------------------------------------------------------------------------------------
XL = 0.64                            # right edge of the row labels
FW_, BRK, AXW = 0.11, 0.04, 0.98     # floor axes, gap, log axes width
FX0 = {50000: XL + 0.05}
FX0[500000] = FX0[50000] + FW_ + BRK + AXW + 0.12
AX0 = {N: FX0[N] + FW_ + BRK for N in NS}
XLO, XHI = 0.15, 8.0
row_y, y = {}, AX_TOP - M_END
for i, (g, r, _) in enumerate(ROWS):
    if i and g != groups[i - 1]:
        y -= GG
    row_y[r] = y
    y -= PITCH
assert abs((y + PITCH) - (AX_BOT + M_END)) < 1e-9
AXF, AXA = {}, {}
PT_PER_DEC = AXW * 72 / np.log10(XHI / XLO)
for N in NS:
    ax_f = axes_in(FX0[N], AX_BOT, FW_, AX_TOP - AX_BOT)
    ax_a = axes_in(AX0[N], AX_BOT, AXW, AX_TOP - AX_BOT)
    AXF[N], AXA[N] = ax_f, ax_a
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
    for r in RUNGS:
        for ax in (ax_f, ax_a):
            ax.axhline(row_y[r], color=S.GUIDE, lw=LW["guide"], zorder=S.Z["guide"] - 0.4)
    ax_a.axvline(1.0, color=S.NULL_LINE, lw=LW["ref"], zorder=S.Z["ref"])
    # the separator sits between SOG and Delta_iso, where Theorem 3 splits predicted disagreement (above) from
    # predicted agreement (below), solid NULL_LINE as in Figure 6 (d)
    y_sep = (row_y["dSOG_t0p1"] + row_y["dIso"]) / 2
    for ax, (x0, x1) in ((ax_f, (0, 1)), (ax_a, (XLO, XHI))):
        ax.plot([x0, x1], [y_sep, y_sep], color=S.NULL_LINE, lw=LW["level"], solid_capstyle="butt",
                zorder=S.Z["ref"], clip_on=False)
    for xb in (FX0[N] + FW_, AX0[N]):
        ov.plot([xb - 0.022, xb + 0.022], [AX_BOT - 0.03, AX_BOT + 0.03], color=INK, lw=LW["axis_break"],
                solid_capstyle="butt", zorder=6)
    # bands and marks
    for r in RUNGS:
        yr = row_y[r]
        lo, hi = A_WITHIN[N][r]
        ax_a.add_patch(Rectangle((lo, yr - BAND_HALF_PT / 72), hi - lo, 2 * BAND_HALF_PT / 72, facecolor=BAND_FC,
                                 edgecolor=BAND_EC, lw=BAND_LW, zorder=S.Z["band"] + 0.2))
        for c in DRAWN:
            if (N, c, r) not in A_MARK:
                continue
            d = A_MARK[(N, c, r)]
            yy = yr + LEVEL[c]
            zm = S.Z["marker"] + (0.2 if c.startswith("bern") else 0.0)   # diamonds on top: their tips intrude less
            if c == "bern_self" and r in FLOOR_ROWS:
                ax_f.plot([0.5], [yy], ls="none", zorder=zm, clip_on=False, **MK[c])
                continue
            ax_a.plot([d["ratio_median"]], [yy], ls="none", zorder=zm, **MK[c])
    for (NN, c, r), d in A_MARK.items():
        if NN != N or c not in DRAWN or (c == "bern_self" and r in FLOOR_ROWS):
            continue
        assert XLO < d["ratio_q05"] and d["ratio_q95"] < XHI, (N, c, r)
    assert XLO < min(v[0] for v in A_WITHIN[N].values()) and max(v[1] for v in A_WITHIN[N].values()) < XHI

# ranges over pairs narrower than the marks (a, b): 5-95% width in pt < marker body width
RANGE_AB_PT = {}
for (N, c, r), d in A_MARK.items():
    if c not in DRAWN or (c == "bern_self" and r in FLOOR_ROWS):
        continue
    RANGE_AB_PT[f"{N}|{c}|{r}"] = float(np.log10(d["ratio_q95"] / d["ratio_q05"]) * PT_PER_DEC)
check("(a, b): every 5-95% range over pairs is narrower than its marker body",
      all(RANGE_AB_PT[k] < mark_body(k.split("|")[1])[0] for k in RANGE_AB_PT),
      {"max_range_pt": max(RANGE_AB_PT.values()), "argmax": max(RANGE_AB_PT, key=RANGE_AB_PT.get),
       "marker_body_width_pt": {c: mark_body(c)[0] for c in DRAWN}})

# overlaps between marks of one row (pt): reported, and only rows where values coincide may have neighbours at one x
OVERLAP = {}
for N in NS:
    for r in RUNGS:
        pts = []
        for c in DRAWN:
            if (N, c, r) not in A_MARK or (c == "bern_self" and r in FLOOR_ROWS):
                continue
            pts.append((c, np.log10(A_MARK[(N, c, r)]["ratio_median"]) * PT_PER_DEC, LEVEL_PT[c]))
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                ci, xi, yi = pts[i]
                cj, xj, yj = pts[j]
                ox = HALF[ci][0] + HALF[cj][0] - abs(xi - xj)
                oy = HALF[ci][1] + HALF[cj][1] - abs(yi - yj)
                if ox > 0 and oy > 0:
                    OVERLAP[f"{N}|{r}|{ci}|{cj}"] = {"overlap_x_pt": float(ox), "overlap_y_pt": float(oy),
                                                     "dx_pt": float(abs(xi - xj)), "dy_pt": float(abs(yi - yj))}
rows_with_overlap = sorted({k.split("|")[1] for k in OVERLAP})
print("OVERLAP rows", rows_with_overlap)
for k, v in OVERLAP.items():
    print("  ", k, {a: round(b, 2) for a, b in v.items()})
check("mark boxes overlap only on rows where values coincide within a marker width, never by more than 2.2 pt "
      "vertically", all(v["overlap_y_pt"] <= 2.2 for v in OVERLAP.values()),
      {"rows": rows_with_overlap, "max_overlap_y_pt": max([v["overlap_y_pt"] for v in OVERLAP.values()] or [0])})
check("self and peer diamonds never adjacent levels; the two cross-objective classes never adjacent",
      abs(LEVEL_PT["bern_self"] - LEVEL_PT["bern_peer"]) >= 2 * D_PT - 1e-9
      and abs(LEVEL_PT["cross_A_B"] - LEVEL_PT["cross_Atau_B"]) >= 2 * D_PT - 1e-9)

# row labels (bold, as the tick labels they are) and group labels
ROW_TXT = {}
for g, r, _ in ROWS:
    ROW_TXT[r] = S.text(ov, XL, row_y[r], ROW_LABEL[r], "label", ha="right", va="center_baseline")
GROUP = {"geom": "geometry", "order": "order"}
X_BR = 0.125                          # bracket line
GROUP_TXT = {}
for g, lab in GROUP.items():
    ys = [row_y[r] for gg, r, _ in ROWS if gg == g]
    lo, hi = min(ys) - PITCH * 0.36, max(ys) + PITCH * 0.36
    ov.plot([X_BR + 0.035, X_BR, X_BR, X_BR + 0.035], [lo, lo, hi, hi], color=INK, lw=LW["bracket"],
            solid_joinstyle="miter", solid_capstyle="butt")
    GROUP_TXT[g] = S.text(ov, X_BR - 0.068, (lo + hi) / 2, lab, "header", rotation=90, ha="center", va="center")
XLAB_A = S.text(ov, (FX0[50000] + AX0[500000] + AXW) / 2, 0.06, "discrepancy / tolerance", "label", ha="center",
                va="baseline")

# ---- letters (a), (b) --------------------------------------------------------------------------------------------
Y_LET = H - 0.012
LET = {"a": S.panel_letter(fig, "a", 0.0, Y_LET), "b": S.panel_letter(fig, "b", FX0[500000] - 0.03, Y_LET)}

# ---- right column: key (one entry per line), then (c) ------------------------------------------------------------
KEY_ORDER = ["cross_A_B", "cross_Atau_B", "bern_self", "bern_peer", "within"]
KEY_PITCH = 0.135
KEY_Y = {k: H - 0.085 - i * KEY_PITCH for i, k in enumerate(KEY_ORDER)}
SWW, SW_GAP = 0.24, 0.05
KEY_X0 = 3.42
key_txt = {}
for k in KEY_ORDER:
    yk = KEY_Y[k]
    if k == "within":
        ov.add_patch(Rectangle((KEY_X0, yk - BAND_HALF_PT / 72 * 0.55), SWW, 2 * BAND_HALF_PT / 72 * 0.55,
                               facecolor=BAND_FC, edgecolor=BAND_EC, lw=BAND_LW))
    else:
        ov.plot([KEY_X0, KEY_X0 + SWW], [yk, yk], color=COL[k], solid_capstyle="butt", **C_STYLE[k])
        ov.plot([KEY_X0 + SWW / 2], [yk], ls="none", **MK[k])
    key_txt[k] = S.text(ov, KEY_X0 + SWW + SW_GAP, yk, NAME[k], "key", ha="left", va="center_baseline")

CX1 = 5.335                          # right end of the (c) axes: room for the last two-line step label
CW_ = 1.82                           # 0.51 in per step (>= MIN_IN_PER_STEP_WORDS 0.47; bold labels need the extra) with xlim 0.70-4.30
CX0 = CX1 - CW_
C_XLIM = (0.70, 4.30)
check("(c) width per step meets the shared word-label minimum", CW_ / (C_XLIM[1] - C_XLIM[0]) >=
      SC.MIN_IN_PER_STEP_WORDS, CW_ / (C_XLIM[1] - C_XLIM[0]))
C_TOP = KEY_Y["within"] - 0.30
CY0 = AX_BOT
CH = C_TOP - CY0
AXC = axes_in(CX0, CY0, CW_, CH)
C_YLIM = (-0.06, 1.08)


def staircase(ax, series, dodge_pt):
    """Cumulative explained share against the four nested fits (shared convention: SC.setup labels, x at SC.X, y
    spine bounded at [0, 1], dotted full-share line). One line per class in its colour, its marker at each step,
    dodged horizontally by dodge_pt[key] points. No range bars: the 5-95% ranges are narrower than the marks."""
    SC.setup(ax, S, labels=STEP_LABEL, xlim=C_XLIM, ylim=C_YLIM, yticks=(0, 0.5, 1), ylabel=SC.Y_LABEL)
    SC.full_share_line(ax, S)
    fig.canvas.draw()
    x_per_in = (C_XLIM[1] - C_XLIM[0]) / (ax.get_window_extent().width / fig.dpi)
    out = {}
    for k in DRAWN:
        d = series[k]
        dx = dodge_pt[k] / 72 * x_per_in
        xs = SC.X + dx
        if C_STYLE[k]["ls"] == "-":
            S.underlay(ax, xs, d["median"], C_STYLE[k]["lw"], S.Z["data"])
        ax.plot(xs, d["median"], color=COL[k], zorder=S.Z["data"], **C_STYLE[k])
        ax.plot(xs, d["median"], ls="none", zorder=S.Z["marker"], **MK[k])
        out[k] = dx
    return out


C_DX = staircase(AXC, {c: STAIR[STAIR_N][c] for c in DRAWN}, C_DODGE_PT)
AXC.tick_params(axis="x", pad=2.0)
pt_per_share = CH * 72 / (C_YLIM[1] - C_YLIM[0])
RANGE_C_PT = {}
for N in NS:                         # 50k is not drawn; its ranges are checked at the same scale for the record
    for c in DRAWN:
        assert C_YLIM[0] < STAIR[N][c]["q05"].min() and STAIR[N][c]["q95"].max() < C_YLIM[1], (N, c)
        RANGE_C_PT[f"{N}|{c}"] = float(np.max(STAIR[N][c]["q95"] - STAIR[N][c]["q05"]) * pt_per_share)
check("(c): every 5-95% range over pairs is narrower than its marker body (drawn N = 500,000; 50,000 at the same "
      "scale)", all(RANGE_C_PT[k] < mark_body(k.split("|")[1])[1] for k in RANGE_C_PT),
      {"max_range_pt": max(RANGE_C_PT.values()), "argmax": max(RANGE_C_PT, key=RANGE_C_PT.get)})

# ---- v8: bold ticks and axis titles, then the (c) letter at the left ink edge of its y label -------------------------
S.finalize(fig)
fig.canvas.draw()
YLAB_C = AXC.yaxis.label
LET["c"] = S.panel_letter(fig, "c", ext(YLAB_C).x0 - 0.01, C_TOP + 0.20)

PDF = os.path.join(HERE, "F_link_app.pdf")
PNG = os.path.join(HERE, "F_link_app.png")
VIEW100 = os.path.join(HERE, "_view_print_100dpi.png")
VIEWS = os.path.join(HERE, "_views_cvd_grey.png")
fig.savefig(PDF)
CHK = S.check(fig, PDF)
S.print_view(PDF, PNG, VIEW100)
S.views(PNG, VIEWS)
pf = subprocess.run(["pdffonts", PDF], capture_output=True, text=True).stdout
font_rows = [ln.split() for ln in pf.splitlines()[2:] if ln.strip()]
FONT_NAMES = sorted({row[0].split("+")[-1] for row in font_rows})
# v8: bold math adds Liberation Serif Bold Italic (BOLD_MATH); the v6 set is kept otherwise
assert set(FONT_NAMES) <= {"LiberationSerif", "LiberationSerif-Bold", "LiberationSerif-BoldItalic", "Cmr10",
                           "Cmmi10", "Cmsy10", "Cmex10"}, FONT_NAMES
info = subprocess.run(["pdfinfo", PDF], capture_output=True, text=True, check=True).stdout
psz = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
PAGE_IN = [round(float(psz.group(1)) / 72, 4), round(float(psz.group(2)) / 72, 4)]
check("page size 5.5 in wide, height as computed", PAGE_IN == [W, H], PAGE_IN)


# ---- overlap checks ----------------------------------------------------------------------------------------------
def text_boxes():
    import matplotlib.text as mtext
    out = []
    inv = fig.dpi_scale_trans.inverted()
    for t in fig.findobj(mtext.Text):
        if not (t.get_visible() and t.get_text().strip()):
            continue
        bb = t.get_window_extent().transformed(inv)
        out.append((t.get_text(), bb, t.get_fontsize()))
    return out


fig.canvas.draw()
TB = text_boxes()
clash = [(a, b) for i, (a, pa, _) in enumerate(TB) for (b, pb, _) in TB[i + 1:]
         if min(pa.x1, pb.x1) - max(pa.x0, pb.x0) > 0.005 and min(pa.y1, pb.y1) - max(pa.y0, pb.y0) > 0.005]
outside = [a for a, bb, _ in TB if bb.x0 < -1e-3 or bb.y0 < -1e-3 or bb.x1 > W + 1e-3 or bb.y1 > H + 1e-3]
small = sorted({(round(fs, 2), a) for a, _, fs in TB if fs < 7.5})
print("CLASH", clash)
print("OUTSIDE", outside)
print("SMALL", small)
assert not clash and not outside, (clash, outside)
GROUP_FIT = {}
for g, t in GROUP_TXT.items():
    ys = [row_y[r] for gg, r, _ in ROWS if gg == g]
    e = ext(t)
    GROUP_FIT[g] = {"text_h_in": e.height, "bracket_h_in": max(ys) - min(ys) + PITCH * 0.72,
                    "text_x1_in": e.x1, "bracket_x_in": X_BR}
    assert e.height < GROUP_FIT[g]["bracket_h_in"] and e.x1 < X_BR - 0.01, (g, GROUP_FIT[g])
ROW_LABEL_X0 = {r: ext(t).x0 for r, t in ROW_TXT.items()}
for g, r, _ in ROWS:
    if g in GROUP:
        assert ROW_LABEL_X0[r] > X_BR + 0.035 + 0.02, (r, ROW_LABEL_X0[r])
print("ROW_LABEL_X0", {r: round(v, 3) for r, v in ROW_LABEL_X0.items()}, "GROUP_FIT", GROUP_FIT)
# key column clears panel (b) and the (c) y label; the last step label stays inside the figure
KEY_X1 = max(ext(t).x1 for t in key_txt.values())
B_RIGHT = max([ext(t).x1 for t in AXA[500000].get_xticklabels() if t.get_text()] + [AX0[500000] + AXW])
STEP_TICK_X1 = max(ext(t).x1 for t in AXC.get_xticklabels())
STEP_TICK_GAPS = []
_st = sorted((ext(t).x0, ext(t).x1) for t in AXC.get_xticklabels() if t.get_text())
for (a0, a1), (b0, b1) in zip(_st, _st[1:]):
    STEP_TICK_GAPS.append(b0 - a1)
LAYOUT = {"key_x0_in": KEY_X0, "key_x1_in": KEY_X1, "panel_b_right_in": B_RIGHT, "ylabel_c_x0_in": ext(YLAB_C).x0,
          "ylabel_c_x1_in": ext(YLAB_C).x1, "last_step_label_x1_in": STEP_TICK_X1,
          "step_label_gaps_in": STEP_TICK_GAPS, "c_axes_in": [CX0, CY0, CW_, CH],
          "key_bottom_in": KEY_Y["within"], "c_top_in": C_TOP}
print("LAYOUT", LAYOUT)
assert KEY_X0 > B_RIGHT + 0.08 and ext(YLAB_C).x0 > B_RIGHT + 0.03 and STEP_TICK_X1 < W - 0.005, LAYOUT
assert KEY_X1 < W - 0.01 and min(STEP_TICK_GAPS) > 0.03, LAYOUT

V6_MAKER_SHA = sha(SRC["F_link_app_v6_maker_started_from"])
check("v8 maker started from the v6.2 maker (byte copy in _v6_ref/)",
      V6_MAKER_SHA == sha(os.path.expanduser("~/hrh_v2_figs/v6/F_link_app/make_F_link_app.py")), V6_MAKER_SHA)
NUM = {
    "figure": {"id": "F_link_app", "version": "v8 (Tol palette, bold ticks and axis titles, no setting headers, "
                                               "InfoNCE 0.2 vs 0.5 and the 50k staircase moved to the text)",
               "label": "fig:app-s21",
               "section": "Appendix F, Contrastive Networks (app:exp-s21)",
               "message": "InfoNCE and the spectral loss disagree on order criteria as well as geometric ones at both "
                          "sample sizes and for both InfoNCE temperatures, against the pattern Theorem link proves "
                          "inside its premises (diamonds).",
               "size_in": CHK["size_in"], "page_size_in": PAGE_IN, "min_text_pt": CHK["min_text_pt"],
               "min_glyph_pt": CHK["min_glyph_pt"],
               "n_type3": CHK["n_type3"], "style_check": CHK, "pdffonts": pf.splitlines(), "font_names": FONT_NAMES,
               "text_overlaps_gt_0.005in": clash, "text_outside_figure": outside, "text_below_7.5pt": small,
               "group_label_fit": GROUP_FIT, "row_label_x0_in": ROW_LABEL_X0, "layout": LAYOUT,
               "drawn_classes": DRAWN, "staircase_N_drawn": STAIR_N,
               "colours": {"across": ACROSS, "control": CONTROL, "within_fill": BAND_FC, "within_edge": BAND_EC,
                           "tolerance_and_separator": S.NULL_LINE},
               "geometry_in": {"row_pitch": PITCH, "row_end_margin": M_END, "group_gap": GG, "dodge_pt": D_PT,
                               "band_half_pt": BAND_HALF_PT, "levels_pt_top_to_bottom": {c: LEVEL_PT[c]
                                                                                          for c in LEVEL_ORDER},
                               "marker_half_extent_pt": HALF, "marker_specs": MK,
                               "ab_log_axes": {str(N): [AX0[N], AXW, XLO, XHI] for N in NS},
                               "ab_floor_axes": {str(N): [FX0[N], FW_] for N in NS}, "pt_per_decade": PT_PER_DEC}},
    "sources": {k: {"path": v, "sha256": sha(v)} for k, v in SRC.items()},
    "ladder_files": {f"{N}|{c}": [[os.path.basename(f), sha(f)] for f in FILES[(N, c)]] for (N, c) in FILES},
    "style": {"path": STYLE_PATH, "sha256": STYLE_SHA, "version": S.STYLE_VERSION},
    "staircase_convention": {"path": SRC["staircase_convention"], "sha256": SC_SHA,
                             "step_labels": list(SC.STEP_LABELS), "y_label": SC.Y_LABEL},
    "script": {"path": os.path.abspath(__file__), "sha256": sha(os.path.abspath(__file__)),
               "started_from_v6_maker_sha256": V6_MAKER_SHA},
    "panels_a_b": {str(N): panel_ab[N] for N in NS},
    "panels_a_b_plotted": {str(N): {
        "within_band_min_max_ratio": A_WITHIN[N],
        "marks": {f"{c}|{r}": {k: A_MARK[(N, c, r)][k] for k in ("ratio_median", "ratio_q05", "ratio_q95")}
                  for c in DRAWN for r in RUNGS if (N, c, r) in A_MARK},
        "not_drawn_marks_cross_A_Atau": {r: {k: A_MARK[(N, "cross_A_Atau", r)][k]
                                             for k in ("ratio_median", "ratio_q05", "ratio_q95")} for r in RUNGS},
        "floor_rows_link_check_self": FLOOR_ROWS,
        "floor_note": "link check, self drawn at the '~0' floor: exactly 0 on T_MST, T_O, T_N,10 (64/64); T_E max "
                      "%.2e of eps; Delta_iso max %.2e of eps" % (A_MARK[(N, "bern_self", "dE")]["ratio_max"],
                                                                  A_MARK[(N, "bern_self", "dIso")]["ratio_max"]),
        "guard_status_not_drawn": "guard " + GUARD_STATUS[N]["status"],
        "guard_fails_on_rungs": GUARD_STATUS[N]["fails_on_rungs"],
        "guard_decisions_within_A_B_Atau": GUARD_STATUS[N]["decisions"]}
        for N in NS},
    "range_5_95_width_pt_ab": RANGE_AB_PT,
    "range_5_95_height_pt_c": RANGE_C_PT,
    "marker_body_pt_w_h": {c: mark_body(c) for c in DRAWN},
    "mark_overlaps_pt": OVERLAP,
    "cross_objective_min_vs_within_max": CROSS_MIN_VS_WITHIN_MAX,
    "n_rows_every_cross_pair_above_every_within_pair": N_CROSS_PAIRS_ABOVE_ALL_WITHIN,
    "median_ratio_ranges": MED_RANGE,
    "within_guard_50k_failed_rungs": GUARD_50K,
    "peer_minus_within_A_order_max_abs": PEER_EQ,
    "panel_c": panel_c,
    "panel_c_drawn_N": STAIR_N,
    "panel_c_dodge_data_units": C_DX,
    "omega_median_by_class": OMEGA,
    "text_numbers_v8": TEXT_NUMBERS,
    "asserts": ASSERTS,
}
with open(os.path.join(HERE, "numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1, default=float)
NUM["outputs"] = {p: sha(os.path.join(HERE, p)) for p in ("F_link_app.pdf", "F_link_app.png",
                                                          "_view_print_100dpi.png", "_views_cvd_grey.png")}
with open(os.path.join(HERE, "numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1, default=float)
print(json.dumps({"check": {k: CHK[k] for k in ("size_in", "min_text_pt", "min_glyph_pt", "n_type3", "fonts",
                                               "low_contrast_text", "not_bold_tick_or_axis_labels",
                                               "bold_text_with_regular_math", "chromatic_data_colours",
                                               "off_palette_colours", "n_tick_labels_checked")},
                  "n_asserts": len(ASSERTS)}, indent=1, default=float))
print("OK")
