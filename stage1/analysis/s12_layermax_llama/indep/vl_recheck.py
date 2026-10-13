#!/usr/bin/env python
"""LLaMA axis, implementation I (role llama-axis-run-indep): provenance recheck of
validation_vl.json. Reruns V-L1, V-L2, V-L4b and V-L7 with the CURRENT vl_checks.py
(the V-L1/V-L2/V-L7 entries were written by the version that ran before a later edit
of that file) and compares every entry with the stored one (the 'secs' field
excluded). Nothing stored is overwritten: np.save and every write-mode open() inside
vl_checks are redirected to /dev/null. Output:
results/S1_2_layermax_llama/indep/validation_vl_recheck.json
"""
from __future__ import annotations

import os

_T = os.environ.get("LMX_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import builtins  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vl_checks as V  # noqa: E402

V.np.save = lambda *a, **k: None   # no table cache rewrite


def _open(p, mode="r", *a, **k):
    if any(c in mode for c in "wax+"):
        return builtins.open(os.devnull, mode.replace("x", "w"), *a, **k)
    return builtins.open(p, mode, *a, **k)


V.open = _open   # module-global lookup precedes builtins


def sha(p):
    return hashlib.sha256(builtins.open(p, "rb").read()).hexdigest()


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "secs"}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff_paths(a, b, path="", out=None, cap=50):
    out = [] if out is None else out
    if len(out) >= cap:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k} (missing in {'stored' if k not in a else 'recheck'})")
            else:
                diff_paths(a[k], b[k], f"{path}/{k}", out, cap)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, cap)
    elif a != b:
        out.append(f"{path}: stored {str(a)[:80]} vs recheck {str(b)[:80]}")
    return out


def main():
    fn = os.path.join(V.L.OUT, "validation_vl.json")
    stored = json.load(builtins.open(fn))
    res = {"vl_checks_sha256": sha(os.path.join(HERE, "vl_checks.py")), "stored_file_sha256": sha(fn), "items": {}}
    for it in ("vl1", "vl4b", "vl7", "vl2"):
        t0 = time.time()
        new = strip(V.L.jsonable({"vl1": V.vl1, "vl2": V.vl2, "vl4b": V.vl4b, "vl7": V.vl7}[it]()))
        key = it.upper().replace("VL", "V-L")
        old = strip(stored[key])
        d = diff_paths(old, new)
        res["items"][key] = {"identical_to_stored": not d, "n_diff_paths_shown": len(d), "diff_paths": d,
                             "pass_recheck": new.get("pass"), "secs": time.time() - t0}
        print(f"[{time.strftime('%H:%M:%S')}] {key}: identical={not d} pass={new.get('pass')} ({time.time() - t0:.0f}s)", flush=True)
    res["all_identical"] = all(v["identical_to_stored"] for v in res["items"].values())
    out = os.path.join(V.L.OUT, "validation_vl_recheck.json")
    json.dump(res, builtins.open(out, "w"), indent=1)
    print("wrote", out, "all_identical", res["all_identical"], flush=True)


if __name__ == "__main__":
    main()
