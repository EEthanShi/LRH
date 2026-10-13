#!/usr/bin/env python
"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/lmx_ol_report.py (sha256 b60a5ea4...), sub-role pyt-code-I
(S0b, 2026-10-10): writes results/S1_2_layermax_pythia/indep/REPORT.md from indep/summary.json (whose "verdicts" block
is verdicts_pyt.json) and the validation files next to it only; every number is read from them. The OpenLLaMA report
text (descriptive axis, J, P-O rules) is replaced by the Pythia plan's labels (PYT-AXIS): H1 is the confirmatory
gatekeeper and {H2a|H2b, H3, H4} are Holm-corrected secondaries (plan 5.4); everything else is labelled descriptive;
l2 and uncensored are secondary. Order gate first (plan F1(a)). An earlier REPORT.md is kept aside, never overwritten.
usage: launch_pyt.sh pyt_I_report 1 -- venv/bin/python analysis/s12_layermax_pythia/indep/lmx_pyt_report.py
       (after lmx_pyt_aggregate.py; rerun after cross_validate.py to add the V-P5 lines)
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True   # PYT-AXIS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402  PYT-AXIS
OUT_ROOT = pyt_gate.gate(__file__)   # PYT-AXIS F1(a): exits 3 unless the plan's gates hold (S0b test mode: test root)

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import lmx_common as L  # noqa: E402

OUT = L.OUT
PRI = "prh_clip+l2"
METS = (("mknn", "mKNN"), ("cka_biased", "CKA (biased)"), ("cka_debiased", "CKA (debiased)"))


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, str):
        return x
    return f"{x:+.{n}f}"


def pv(x):
    return "n/a" if x is None else f"{x:.4g}"


def main():
    ap = argparse.ArgumentParser(description="REPORT.md of implementation I (Pythia axis)")
    ap.parse_args()
    S = load("summary.json")
    if S is None:
        raise SystemExit("summary.json missing")
    V = S["verdicts"]
    tags = S["size_tags"]
    pres = S["pres"]
    lines = []
    w = lines.append
    w("# Pythia axis (S1.2, fourth size axis): layer-max calibration, implementation I")
    w("")
    w(f"Panel: 7 Pythia models ({', '.join(tags)}) x the 17 ARH `val` ViTs on WIT-1024 (119 pairs); pre run: "
      f"{', '.join(pres)} (l2 run: {V.get('G|l2_run')}). K = {S['K']}, keys `S1.2-layermax|<lm>|<vit>`. Every number "
      f"is read from `summary.json` (its `verdicts` block = `verdicts_pyt.json`). Test mode: {S.get('test_mode')}; "
      f"stand-in fixture: {S.get('fixture')}.")
    if S.get("test_mode"):
        w("")
        w("**S0b TEST OUTPUT: the 'Pythia' models below are stand-ins (" + ", ".join(
            f"{k} = {v}" for k, v in S.get("standin", {}).items()) + "). Not a result; not citable.**")
    w("")
    w("**Labels (plan 5.4, 7).** H1 is the confirmatory gatekeeper at alpha = 0.05; H2a/H2b, H3, H4 are tested with "
      "Holm at 0.05 only if the gate is open. Everything else is descriptive. l2 and uncensored are secondary.")
    for p in pres:
        if not V.get("G|l2_run") and p == "l2":
            continue
        h = f"H|{p}|s_cal"
        k = f"x|I|K|{p}|s_cal"
        w("")
        w(f"## {p}" + (" (primary)" if p == PRI else " (secondary)"))
        w("")
        w(f"- Part B: H1 member {f(V.get(f'{h}|H1|member'))}, s = {V.get(f'{h}|H1|s')}, H2 branch "
          f"{V.get(f'{h}|H2|branch')}, s_late = {V.get(f'{h}|H2|s_late')}.")
        nviol = sum(int(V.get(kk) or 0) for kk in V if kk.startswith((f"keepid|{p}|", f"signrule|{p}|"))
                    or (kk.startswith(f"signvit|{p}|") and kk.endswith(("nviol_up", "nviol_down")))) \
            + int(V.get(f"D3|{p}|nviol") or 0) + int(V.get(f"PP1|{p}|s_cal|nrev") or 0) + int(V.get(f"PP1|{p}|unc|nrev") or 0)
        w(f"- Deductive checks before any category (plan 7): {nviol} violations (keep identity, sign rules, D3, P-P1).")
        w(f"- **Implementation I's own category (cross-check; the axis category is `forecast_pyt.py` section K): "
          f"{V.get(f'{k}|category')}** (premise {V.get(f'{k}|premise')}; F reason {V.get(f'{k}|F_reason')}); "
          f"R = {f(V.get(f'{h}|H1|R'))}, C = {f(V.get(f'{h}|H1|C'))} on {V.get(f'{h}|H1|nV')} ViTs; knife edge "
          f"{f(V.get(f'{h}|H1|kedge'))}.")
        w(f"- Modifiers (I): M1 {f(V.get(f'{k}|M1'))} ({V.get(f'R|{p}|s_cal|fragile_sizes')}), M2 {f(V.get(f'{k}|M2'))} "
          f"(grid5 D1 {f(V.get(f'R|{p}|s_cal|grid5|D1'))}), M3 {f(V.get(f'{k}|M3'))}, M4 {f(V.get(f'{k}|M4'))} "
          f"({V.get(f'{k}|M4_sizes')}), M5 extrapolated: {V.get(f'{k}|M5_extrap_sizes')}; readings: H2 "
          f"{V.get(f'{k}|H2_reading')}; H3 {V.get(f'{k}|H3_reading')}; H4 {V.get(f'{k}|H4_reading')}.")
        w("")
        w("| test | statistic | p | Holm | status |")
        w("|---|---|---|---|---|")
        w(f"| H1 (gatekeeper) | D1 = {f(V.get(f'{h}|H1|D'))} on {V.get(f'{h}|H1|nV')} ViTs | "
          f"{pv(V.get(f'{h}|H1|p'))} (perm {pv(V.get(f'{h}|H1|p_perm'))}, mag {pv(V.get(f'{h}|H1|p_mag'))}) | n/a | "
          f"confirmed {f(V.get(f'{h}|H1|confirmed'))} |")
        for hy in ("H2", "H3", "H4"):
            stat = (f"D2 = {f(V.get(f'{h}|H2|D'))} ({V.get(f'{h}|H2|branch')})" if hy == "H2" else
                    f"Delta = {f(V.get(f'{h}|{hy}|D'))} (degenerate {f(V.get(f'{h}|{hy}|degenerate'))})")
            w(f"| {hy} | {stat} | {pv(V.get(f'{h}|{hy}|p'))} | {pv(V.get(f'{h}|{hy}|p_holm'))} | "
              f"tested {f(V.get(f'{h}|{hy}|tested'))}, confirmed {f(V.get(f'{h}|{hy}|confirmed'))} |")
        w("")
        w(f"- Gate: {V.get(f'{h}|gate')}; Holm order {V.get(f'{h}|holm_order')}.")
        w(f"- Premise rules: P-P1 holds {f(V.get(f'PP1|{p}|holds'))}; P-P2a(i) holds {f(V.get(f'PP2ai|{p}|s_cal|holds'))} "
          f"({V.get(f'PP2ai|{p}|s_cal|n_models_in_band')} of 7 in band; failing '{V.get(f'PP2ai|{p}|s_cal|failing')}'); "
          f"P-P2a(ii) {V.get(f'PP2aii|{p}|s_cal|verdict')} ({V.get(f'PP2aii|{p}|s_cal|x')}/{V.get(f'PP2aii|{p}|s_cal|n')}, "
          f"need {V.get(f'PP2aii|{p}|s_cal|need')}); P-P2a {V.get(f'PP2a|{p}|s_cal|verdict')}; I's P-P2b "
          f"{V.get(f'x|I|PP2b|{p}|s_cal|verdict')} ({V.get(f'x|I|PP2b|{p}|s_cal|x_correct')}/"
          f"{V.get(f'x|I|PP2b|{p}|s_cal|n_decided')} decided correct); P-P5a {V.get(f'PP5a|{p}|s_cal|verdict')}.")
        w("")
        w("| per model | band | median c - B-bar | in band | pairs in band |")
        w("|---|---|---|---|---|")
        for t in tags:
            w(f"| {t} | [{f(V.get(f'PP2ai|{p}|s_cal|lo|{t}'), 5)}, {f(V.get(f'PP2ai|{p}|s_cal|hi|{t}'), 5)}] | "
              f"{f(V.get(f'PP2ai|{p}|s_cal|median|{t}'), 5)} | {f(V.get(f'PP2ai|{p}|s_cal|in_band|{t}'))} | "
              f"{V.get(f'PP2ai|{p}|s_cal|x|{t}')}/17 |")
        w("")
        w("Trends (layer-max `full`, exact one-sided p over 5040 relabellings; descriptive except through H1-H4):")
        w("")
        w("| metric | Tr raw | Tr s_cal | dTr s_cal (common set) | p (<=) | Tr unc | dTr unc |")
        w("|---|---|---|---|---|---|---|")
        for m, lab in METS:
            w(f"| {lab} | {f(V.get(f'tr|{p}|{m}|full|raw'))} | {f(V.get(f'tr|{p}|{m}|full|s_cal'))} | "
              f"{f(V.get(f'dtr|{p}|{m}|full|s_cal'))} | {pv(V.get(f'dtr_ple|{p}|{m}|full|s_cal'))} | "
              f"{f(V.get(f'tr|{p}|{m}|full|unc'))} | {f(V.get(f'dtr|{p}|{m}|full|unc'))} |")
        w("")
        w(f"Robustness (descriptive): five sizes D1 {f(V.get(f'R|{p}|s_cal|five|D1'))}; grid5 D1 "
          f"{f(V.get(f'R|{p}|s_cal|grid5|D1'))}; rho0 D1 {f(V.get(f'R|{p}|s_cal|rho0|D1'))}; leave-one-size-out D1: "
          + ", ".join(f"{t} {f(V.get(f'R|{p}|s_cal|loo|{t}|D1'))}" for t in tags) + ".")
        w(f"Uncensored (secondary): D1 {f(V.get(f'H|{p}|unc|H1|D'))}, p {pv(V.get(f'H|{p}|unc|H1|p'))}; I's category "
          f"{V.get(f'x|I|K|{p}|unc|category')}.")
    w("")
    w("## Validation files")
    for name in ("validation.json", "validation_vl.json", "validation_vp6_arh.json"):
        d = load(name)
        if d is None:
            w(f"- {name}: absent")
            continue
        flags = {k: v.get("pass") for k, v in d.items() if isinstance(v, dict) and "pass" in v}
        if "pass_1e-6" in d:
            flags["V-P6"] = d["pass_1e-6"]
        w(f"- {name}: " + ", ".join(f"{k} pass {f(v)}" for k, v in flags.items()))
    for fn in sorted(glob.glob(os.path.join(OUT, "validation_cross_*.json"))):
        d = json.load(open(fn))
        w(f"- {os.path.basename(fn)}: pass {f(d.get('pass_V-P5_selected_parts'))} (counts and key names only)")
    w("")
    w(f"Written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by lmx_pyt_report.py from summary.json "
      f"(finished {S.get('finished_utc')}).")
    fn = os.path.join(OUT, "REPORT.md")
    tmp = fn + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    L.keep_aside(fn)
    os.replace(tmp, fn)
    print(f"wrote {fn}")


if __name__ == "__main__":
    main()
