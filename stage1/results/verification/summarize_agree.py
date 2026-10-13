#!/usr/bin/env python
"""Summarize agree.py output: per case family, per tolerance class, worst keys."""
import collections
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agree import key_map, tol_class  # noqa: E402

d = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else None
files = sorted(glob.glob(os.path.join(d, "cases", "*.json")))
summary = {"n_cases": len(files), "cases": {}}
worst = collections.defaultdict(list)
for f in files:
    r = json.load(open(f))
    name = r["case"]["name"]
    mv, rv = r["main_values"], r["ref_values"]
    m = r["m"]
    inv = {}
    for k in mv:
        inv[k] = None
    rows = []
    km = {}
    # rebuild main->ref pairs from the stored values using the same map
    from hrhv2 import structural as st  # noqa
    ks = [k for k in st.default_k_grid(m) if k <= m - 2]
    km = key_map(ks)
    for a, b in km.items():
        if a in mv and b in rv:
            x, y = mv[a], rv[b]
            if x is None or y is None:
                continue
            cls = tol_class(a)
            if (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
                continue
            dd = abs(x - y)
            rel = dd / max(abs(x), abs(y)) if max(abs(x), abs(y)) > 0 else 0.0
            rows.append((a, cls, x, y, dd, rel))
            worst[cls].append((rel, dd, name, a, x, y))
    summary["cases"][name] = {
        "n_compared": r["n_compared"], "n_fail": r["n_fail"],
        "fail_keys": [x["key"] for x in r["fails"]],
        "D_identical": r["D_identical"]["A"] and r["D_identical"]["B"],
        "n_negative_DB": r.get("n_negative_DB"),
        "brute_main_ok": r.get("brute_all_main_ok"), "brute_ref_ok": r.get("brute_all_ref_ok"),
        "max_neartie": max([v for v in r["neartie"].values() if v is not None and v == v] or [0.0]),
        "Delta_O_rows_with_ties": r.get("Delta_O_rows_with_ties"),
        "clip_threshold_absdiff": (max(r["clip_threshold"]["absdiff_A"], r["clip_threshold"]["absdiff_B"])
                                   if "clip_threshold" in r else None),
        "t_main_s": r["t_main_s"], "t_ref_s": r["t_ref_s"],
    }
for cls in worst:
    worst[cls].sort(reverse=True)
summary["worst"] = {cls: [dict(zip(("rel", "abs", "case", "key", "main", "ref"), w)) for w in v[:15]]
                    for cls, v in worst.items()}
fam = collections.defaultdict(lambda: [0, 0])
for n, c in summary["cases"].items():
    f = n.split("|")[0]
    fam[f][0] += 1
    fam[f][1] += int(c["n_fail"] > 0)
summary["families"] = {k: {"n": v[0], "n_with_fail": v[1]} for k, v in fam.items()}
print(json.dumps(summary["families"], indent=1))
for cls, v in summary["worst"].items():
    print(cls)
    for w in v[:8]:
        print("   ", w)
if out_json:
    json.dump(summary, open(out_json, "w"), indent=1)
