"""Render the tables of the s22-recompute review from recompute_summary.json (markdown to stdout)."""
import json
import os
import sys

OUT = os.path.expanduser("~/hrh_v2_stage2_20261008/results/review_final/s22-recompute")
R = json.load(open(os.path.join(OUT, "recompute_summary.json")))
RUNGS = ["dN_k10", "Gs", "dCKA", "dO", "dE", "dMST", "dAff", "dSOG_t0p1", "dPi"]
SETS = ["noWD_25000", "WD_25000", "smallinit_25000", "noWD-shared_25000", "noWD_250000", "WD_250000", "smallinit_250000"]
f3 = lambda v: f"{v:.3f}"
f4 = lambda v: f"{v:.4f}"
L = []
w = L.append

w("### T1. Decision table, report vs recomputed (all 252 cells)\n")
w("Each cell: recomputed `decision x/96`; `=` means the report (md), s22_summary.json (x and decision) and, for the four "
  "feature-recomputed rungs (dN_k10, Gs, dCKA, dPi), the independent numpy recomputation all give the same x. "
  "Any disagreement is written out as `MISMATCH`.\n")
w("| set | rep | " + " | ".join(RUNGS) + " |")
w("|---|---|" + "---|" * len(RUNGS))
for s in SETS:
    for tag in ("y", "u", "h", "h_l2"):
        t = R["tables"][f"{s}|{tag}"]
        cells = []
        for k in RUNGS:
            c = t[k]
            txt = f"{c['dec']} {c['x']}/{c['n']}"
            cells.append(txt + (" =" if c["match"] else f" MISMATCH(md {c['md']}, json {c['summary_dec']} {c['summary_x']}, indep {c.get('x_indep')})"))
        w(f"| {s} | {tag} | " + " | ".join(cells) + " |")
w("")

w("### T2. Frozen tolerances (prereg/eps_S2_2.json) and their re-derivation from the 24 pilot pairs (N = 100k)\n")
w("| rep | " + " | ".join(RUNGS) + " |")
w("|---|" + "---|" * len(RUNGS))
for tag in ("y", "u", "h", "h_l2"):
    w(f"| {tag} | " + " | ".join(f4(R['eps_check']['rows'][f'{tag}|{k}']['eps_file']) for k in RUNGS) + " |")
w("")
bad = [(k, v) for k, v in R["eps_check"]["rows"].items() if not v["match"]]
mx = max(v.get("absdiff_recomputed", 0) for v in R["eps_check"]["rows"].values())
w(f"- eps = max of the file's pilot rows for every (rep, rung): {'yes' if not bad else 'NO: ' + str(bad)}; "
  f"max |eps − max of my feature recomputation over the 24 pilot pairs| (dN_k10, Gs, dCKA, dPi) = {mx:.2e}")
e = R["eps_check"]
w(f"- pilot cohort {e['file_cohort']}, N = {e['pilot_N']}, {e['pilot_n_runs']} runs, {e['n_pairs']} pairs "
  f"(pairs in file = (2j, 2j+1) per arm: {e['pilot_pairs_match_file']}); anchor {e['file_anchor']}; eps(u) == eps(y) for every rung: {e['u_equals_y']}\n")

w("Pilot (N = 100k) u-pairs measured against eps(y) (how far the gauge already moved u at the pilot N):\n")
w("| arm | rung | u pairs ≤ eps(y) | median Δ(u) | median Δ(y) | max Δ(u) | max Δ(y) |")
w("|---|---|---|---|---|---|---|")
for a in ("noWD", "WD", "smallinit"):
    for k in ("dN_k10", "Gs", "dCKA", "dPi"):
        v = e["pilot_u_vs_eps_y"][f"{a}|{k}"]
        w(f"| {a} | {k} | {v['u_le_eps_y']}/{v['n']} | {f4(v['u_median'])} | {f4(v['y_median'])} | {f4(v['u_max'])} | {f4(v['y_max'])} |")
w("")

w("### T3. Δ distributions and p̂(ε) (feature recomputation; exploratory except the 1.0×ε column)\n")
w("p̂ at multiples of the frozen ε; `ε for rel` = the 92nd smallest Δ (the smallest ε that gives x ≥ 92), shown also as a ratio to the frozen ε; `ε for not-unr` = 70th smallest Δ (x ≥ 70 needs ε ≥ this).\n")
mults = ["0.5", "0.75", "0.9", "1.0", "1.1", "1.25", "1.5", "2.0"]
for tag, k in (("u", "Gs"), ("u", "dN_k10"), ("y", "Gs"), ("y", "dN_k10"), ("u", "dCKA"), ("h", "dN_k10"), ("h", "Gs")):
    w(f"**{tag} / {k}** (frozen ε = {f4(R['curves'][f'noWD_25000|{tag}|{k}']['eps'])})\n")
    w("| cell | min | q10 | median | q90 | max | " + " | ".join(f"p̂({m}ε)" for m in mults) + " | ε for rel (×ε) | ε for not-unr |")
    w("|---|" + "---|" * (5 + len(mults) + 2))
    for s in ["pilot100k_noWD", "pilot100k_WD", "pilot100k_smallinit"] + SETS:
        c = R["curves"].get(f"{s}|{tag}|{k}")
        if c is None:
            continue
        q = c["q"]
        qs = " | ".join(f4(q[x]) for x in ("0", "0.1", "0.5", "0.9", "1"))
        if s.startswith("pilot"):
            w(f"| {s} (8 pairs) | {qs} | " + " | ".join("" for _ in mults) + " | | |")
        else:
            ps = " | ".join(f"{c['p_at_mult'][m]:.2f}" for m in mults)
            w(f"| {s} | {qs} | {ps} | {f4(c['eps_needed_rel_92'])} ({c['eps_needed_rel_ratio']:.2f}×) | {f4(c['eps_unrel_70th'])} |")
    w("")

w("### T4. Exploratory, not preregistered: gauge effect read at the same N (u against the function y of the same pairs)\n")
w("`u ≤ max y (cell)` = #u-pairs with Δ(u) ≤ max Δ(y) over the 96 y-pairs of the same cell; `cross-fit` = same with "
  "the max taken on the other half of the pairs (48/48 split), with the CP decision; `pooled N` = threshold = max Δ(y) "
  "over all 288 independent-corpus pairs at that N; `u>y` = fraction of pairs whose u-discrepancy exceeds their own y-discrepancy.\n")
w("| cell | rung | u ≤ max y (cell) | cross-fit x (dec) | u ≤ pooled-N max y | median Δu−Δy | median Δu/Δy | u>y | mean Δu | mean Δy |")
w("|---|---|---|---|---|---|---|---|---|---|")
for s in SETS:
    for k in ("dN_k10", "Gs", "dCKA"):
        v = R["exploratory"].get(f"{s}|{k}")
        if v is None:
            continue
        w(f"| {s} | {k} | {v['u_le_maxy_samecell']}/96 | {v['u_le_maxy_crossfit']}/96 ({v['dec_crossfit']}) | {v['u_le_pooledmaxy_N']}/96 | "
          f"{v['median_u_minus_y']:+.4f} | {v['median_ratio_u_over_y']:.3f} | {v['frac_u_gt_y']:.2f} | {f4(v['mean_u'])} | {f4(v['mean_y'])} |")
w("")
w("cond(W) per cell (quantiles over the 192 runs):\n")
w("| cell | min | q10 | median | q90 | max |")
w("|---|---|---|---|---|---|")
for s in SETS:
    c = R["condW_by_cell"][s]
    w(f"| {s} | " + " | ".join(f"{c[x]:.2f}" for x in ("0", "0.1", "0.5", "0.9", "1")) + " |")
print("\n".join(L))
