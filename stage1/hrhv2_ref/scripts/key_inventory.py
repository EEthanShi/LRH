"""Inventory of the keys ladder_ref returns at a realistic size, checked against the
SPEC v1.0 name list (tests/test_ladder.py::spec_names).  Writes every key + value.
Usage: python key_inventory.py OUT_JSON [m]"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "tests"), HERE):
    sys.path.insert(0, p)
from hrhv2_ref import ladder_ref, __version__  # noqa: E402
from scipy_crosscheck import pair  # noqa: E402
from test_ladder import spec_names  # noqa: E402

out_path = sys.argv[1]
m = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
A, B = pair(m, f"hrhv2_ref/xcheck|{m}")
res = {"version": __version__, "m": m}
for pre in ("l2", "prh_clip+l2"):
    t0 = time.time()
    o = ladder_ref(A, B, pre)
    names = spec_names(o["k_q"])
    res[pre] = {"seconds": round(time.time() - t0, 1), "n_keys": len(o),
                "missing_spec_names": [n for n in names if n not in o],
                "values": {k: (v if isinstance(v, (str, bool, int)) else float(v)) for k, v in sorted(o.items())}}
    print(pre, res[pre]["seconds"], "s,", len(o), "keys, missing:", res[pre]["missing_spec_names"], flush=True)
json.dump(res, open(out_path, "w"), indent=1)
