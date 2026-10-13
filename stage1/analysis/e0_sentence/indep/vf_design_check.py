"""V-f step 0 (role e0-indep): the own design (hrhv2.util + items.parquet)
against the stored S1.1 design output results/S1_1/design/design.npz (read only).

Usage: python vf_design_check.py <outdir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C  # noqa: E402


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    ids, orders, h1 = C.design()
    p = os.path.join(C.ROOT, "results", "S1_1", "design", "design.npz")
    pj = os.path.join(C.ROOT, "results", "S1_1", "design", "design.json")
    z = np.load(p)
    js = json.load(open(pj))
    res = dict(
        design_npz=p, design_npz_sha256=C.sha256_file(p), design_json_sha256=C.sha256_file(pj),
        orders_equal=bool(np.array_equal(orders, z["orders"])),
        h1_equal=bool(np.array_equal(h1, z["h1"])),
        image_id_equal=bool(np.array_equal(ids, z["image_id"])),
    )
    cnt = {}
    ok = True
    for d in range(C.N_DRAWS):
        for m in C.M_GRID:
            pre = orders[d][:m]
            nh1 = int(h1[pre].sum())
            nh2 = m - nh1
            st = js["per_draw"][str(d)][str(m)]
            ok &= (nh1 == st["n_H1"]) and (nh2 == st["n_H2"])
            cnt[f"d{d}_m{m}"] = dict(n_pred_anchors=int(len(C.predicted_anchors(orders[d], h1, m))), n_H2=nh2)
    res["halves_counts_equal_design_json"] = bool(ok)
    res["anchor_counts"] = cnt
    sha = C.write_new_json(os.path.join(outdir, "vf_design_check.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "anchor_counts"}), sha)
    assert res["orders_equal"] and res["h1_equal"] and res["image_id_equal"] and ok


if __name__ == "__main__":
    main(sys.argv[1])
