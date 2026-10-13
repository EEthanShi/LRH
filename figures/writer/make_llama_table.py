#!/usr/bin/env python3
"""Appendix table: LLaMA-1 axis trends Tr_L (prh_clip+l2), read from the frozen
implementation-E summary and cross-checked cell by cell against implementation I.
Read-only on S1; writes only under ~/hrh_v2_figs/writer/out_llama/."""
import json, hashlib, os, datetime

S1 = os.path.expanduser("~/hrh_v2_stage1_20261008/results/S1_2_layermax_llama")
E = os.path.join(S1, "early/summary.json")
I = os.path.join(S1, "indep/summary.json")
OUT = os.path.expanduser("~/hrh_v2_figs/writer/out_llama")
os.makedirs(OUT, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


e = json.load(open(E))["per_pre"]["prh_clip+l2"]["trends_L"]


def fmt(d):
    tr, p = d["Tr"], d["p_one"]
    s = "%.2f" % abs(tr)
    if tr < -0.005:
        s = "$-$" + s
    return "%s (%.3f)" % (s, p)


names = [("cka_biased", "CKA"), ("mknn", "mKNN"), ("cka_debiased", "debiased CKA (expl.)")]
prots = [("full", "layer-max"), ("fixed_r0.5", "fixed, $r=0.5$"),
         ("fixed_r1.0", "fixed, $r=1$"), ("ctm", "calibrate, then max")]
rows, cells = [], {}
for mk, mname in names:
    for j, (pk, pname) in enumerate(prots):
        if pk == "ctm":
            keys = [None, mk + "|ctm|M", mk + "|ctm|M_unc"]
        else:
            keys = [mk + "|" + pk + "|s_obs", mk + "|" + pk + "|s_cal", mk + "|" + pk + "|uncensored"]
        c = ["--" if k is None else fmt(e[k]) for k in keys]
        for k in keys:
            if k:
                cells[k] = (e[k]["Tr"], e[k]["p_one"])
        lab = mname if j == 0 else ""
        rows.append("%s & %s & %s & %s & %s \\\\" % (lab, pname, c[0], c[1], c[2]))
    rows.append("\\addlinespace[2pt]")
rows = rows[:-1]
body = ("Metric & layers & raw $s_{\\rm obs}$ & $s_{\\rm cal}$ & $s_{\\rm obs}-\\overline{s}_{\\pi}$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n")

# Cross-check against implementation I (same keys searched anywhere under prh_clip+l2).
ichk = {"checked": 0, "missing": [], "max_abs_diff": 0.0}


def find(d, key):
    if isinstance(d, dict):
        if key in d and isinstance(d[key], dict) and "Tr" in d[key]:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


IMAP = {"cka_biased": "cka_b", "mknn": "mknn", "cka_debiased": "cka_d"}
SMAP = {"s_obs": "s_obs", "s_cal": "s_cal", "uncensored": "uncensored", "M": "s_cal", "M_unc": "uncensored"}
try:
    blocks = json.load(open(I))["per_pre"]["prh_clip+l2"]["blocks"]
    for k, (tr, p) in cells.items():
        mk, pk, sk = k.split("|")
        try:
            r = blocks[IMAP[mk]][pk]["llama_axis"][SMAP[sk]]
        except KeyError:
            r = None
        if r is None:
            ichk["missing"].append(k)
            continue
        ichk["checked"] += 1
        ichk["max_abs_diff"] = max(ichk["max_abs_diff"], abs(r["Tr"] - tr),
                                   abs(r.get("p_one", p) - p))
except Exception as ex:  # report, do not hide
    ichk["error"] = repr(ex)

out_tex = os.path.join(OUT, "tab_layermax_llama_tr.tex")
open(out_tex, "w").write(body)
man = {"generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "script_sha256": sha(os.path.abspath(__file__)),
       "inputs": {E: sha(E), I: sha(I) if os.path.exists(I) else None},
       "output": {"tab_layermax_llama_tr.tex": sha(out_tex)},
       "cells": {k: {"Tr": v[0], "p_one": v[1]} for k, v in cells.items()},
       "check_vs_I": ichk}
json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
print(body)
print(json.dumps(ichk))
