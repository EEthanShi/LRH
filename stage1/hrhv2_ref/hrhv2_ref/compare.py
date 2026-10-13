"""SPEC §8 agreement check between two ladder dicts (e.g. hrhv2 vs hrhv2_ref).

Tolerance classes (SPEC §8):
  exact     set/graph/tree quantities: s_N@k, delta_N@k, comp_*, s_T*, forest_share*, n_bridges*,
            n_shared_bridges, k_q, F_*_subset_*;
  iso       isotonic fits (quot_iso*, quot_rowiso*): |a-b| <= 1e-6;
  float     everything else numeric: |a-b| <= 1e-10 * max(|a|,|b|) + 1e-13.
Exact-class disagreements should be read together with near_tie_* fractions: an
ulp-level difference in D can legitimately flip a near-tied neighbour.

CLI: python -m hrhv2_ref.compare MAIN.json REF.json [KEYMAP.json]
(each JSON either one flat dict or a list of flat dicts in the same order).
"""
from __future__ import annotations

import json
import math
import sys

EXACT_PREFIXES = ("s_N@", "delta_N@", "comp_", "s_T", "forest_share", "n_bridges", "n_shared_bridges", "k_q", "F_")
ISO_PREFIXES = ("quot_iso", "quot_rowiso")


def tol_class(key):
    if key.startswith(EXACT_PREFIXES):
        return "exact"
    if key.startswith(ISO_PREFIXES):
        return "iso"
    return "float"


def _isnan(v):
    return isinstance(v, float) and math.isnan(v)


def compare_ladders(main, ref, key_map=None, rel=1e-10, abs_floor=1e-13, iso_abs=1e-6):
    key_map = key_map or {}
    rows = []
    for k_ref, v_ref in ref.items():
        k_main = key_map.get(k_ref, k_ref)
        if k_main not in main or isinstance(v_ref, str):
            continue
        v_main = main[k_main]
        if isinstance(v_main, str):
            continue
        cls = tol_class(k_ref)
        a, b = float(v_main), float(v_ref)
        if _isnan(a) or _isnan(b):
            ok = _isnan(a) and _isnan(b)
            diff = float("nan")
        else:
            diff = abs(a - b)
            if cls == "exact":
                ok = a == b
            elif cls == "iso":
                ok = diff <= iso_abs
            else:
                ok = diff <= rel * max(abs(a), abs(b)) + abs_floor
        rows.append({"key": k_ref, "class": cls, "main": a, "ref": b, "absdiff": diff, "ok": bool(ok)})
    return rows


def main(argv):
    m = json.load(open(argv[1]))
    r = json.load(open(argv[2]))
    km = json.load(open(argv[3])) if len(argv) > 3 else None
    if isinstance(m, dict):
        m, r = [m], [r]
    nbad = 0
    for i, (dm, dr) in enumerate(zip(m, r)):
        for row in compare_ladders(dm, dr, km):
            if not row["ok"]:
                nbad += 1
                print(f"[{i}] FAIL {row['key']} ({row['class']}): main={row['main']!r} ref={row['ref']!r} "
                      f"diff={row['absdiff']:.3g}")
    print(f"{nbad} failures over {len(m)} records")
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
