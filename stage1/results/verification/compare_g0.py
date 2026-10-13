#!/usr/bin/env python
"""Compare the verifier's G0 re-run (after fix F1) with harness-main's G0 report, leaf by leaf.
Usage: python compare_g0.py ORIG.json RERUN.json OUT.json"""
import json
import math
import sys

IGNORE = ("time", "wall", "created_utc", "sha256", "task_time_s", "host")


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if any(s in str(k) for s in IGNORE):
                continue
            if k not in a or k not in b:
                out["missing"].append(path + "/" + str(k))
                continue
            walk(a[k], b[k], path + "/" + str(k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["missing"].append(path + f" (len {len(a)} vs {len(b)})")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        out["n_num"] += 1
        if a != b:
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                return
            rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
            key = path.split("/")[-1].split("[")[0]
            d = out["diff_by_key"].setdefault(key, {"n": 0, "max_rel": 0.0, "example": path})
            d["n"] += 1
            d["max_rel"] = max(d["max_rel"], rel)
    else:
        out["n_other"] += 1
        if a != b:
            out["other_diff"].append([path, a, b])


def main():
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    out = {"n_num": 0, "n_other": 0, "missing": [], "diff_by_key": {}, "other_diff": []}
    walk(a, b, "", out)
    out["overall_orig"] = a["overall"]
    out["overall_rerun"] = b["overall"]
    out["other_diff"] = out["other_diff"][:50]
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print("numeric leaves", out["n_num"], "other leaves", out["n_other"])
    print("overall orig", a["overall"], "rerun", b["overall"])
    for k, v in sorted(out["diff_by_key"].items(), key=lambda t: -t[1]["max_rel"]):
        print(f"  {k}: n={v['n']} max_rel={v['max_rel']:.3g} e.g. {v['example']}")
    print("other diffs", out["other_diff"][:10])
    print("missing", out["missing"][:10])


if __name__ == "__main__":
    main()
