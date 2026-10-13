"""S1.1 item design from the COCO item table only (ids and texts; no feature array).

Writes results/S1_1/design/design.npz (draw orders, H1 mask) and design.json
(salts, sha256 of the arrays, H1/H2 sizes per draw and m, duplicate-text
groups per draw and m). Run before the plan is frozen.
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s11_common as C  # noqa: E402


def main():
    ids, texts = C.coco_item_ids()
    n = len(ids)
    orders = C.draw_orders(ids)
    h1 = C.h1_mask(ids)
    out = os.path.join(C.RESULTS_DIR, "design")
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, "design.npz")
    np.savez(npz, orders=np.stack(orders), h1=h1, image_id=ids)
    texts = np.asarray(texts, dtype=object)
    cnt = Counter(texts.tolist())
    dup_texts = {t for t, c in cnt.items() if c > 1}
    info = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_pool": n,
            "items_sha256": C.file_sha256(os.path.join(C.COCO_DIR, "items.parquet")),
            "draw_salts": [s if s is not None else "stored order (hrh-v2-coco-pool-v1)" for s in C.DRAW_SALTS],
            "halves_salt": C.HALVES_SALT, "h1_fraction_pool": float(h1.mean()),
            "orders_sha256": hashlib.sha256(np.ascontiguousarray(np.stack(orders)).tobytes()).hexdigest(),
            "h1_sha256": hashlib.sha256(h1.tobytes()).hexdigest(),
            "n_duplicate_text_groups_pool": len(dup_texts),
            "n_items_in_duplicate_groups_pool": int(sum(cnt[t] for t in dup_texts)),
            "per_draw": {}}
    for d, o in enumerate(orders):
        per = {}
        for m in C.M_GRID:
            items = o[:m]
            hm = h1[items]
            tc = Counter(texts[items].tolist())
            groups = [c for c in tc.values() if c > 1]
            fit = items[:C.M_FIT] if m >= C.M_FIT else None
            per[m] = {"n_H1": int(hm.sum()), "n_H2": int((~hm).sum()),
                      "n_dup_groups": len(groups), "n_extra_dup_items": int(sum(c - 1 for c in groups))}
        per["n_fit_H1"] = int(h1[o[:C.M_FIT]].sum())
        info["per_draw"][d] = per
    if len(orders) > 1:
        info["overlap_of_prefix_with_draw0"] = {
            m: [int(np.intersect1d(orders[0][:m], o[:m]).size) for o in orders[1:]] for m in C.M_GRID}
    with open(os.path.join(out, "design.json"), "w") as f:
        json.dump(info, f, indent=1)
    info["design_npz_sha256"] = C.file_sha256(npz)
    with open(os.path.join(out, "design.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "per_draw"}, indent=1))
    for d in info["per_draw"]:
        print(d, {m: (v["n_H2"], v["n_extra_dup_items"]) for m, v in info["per_draw"][d].items() if isinstance(v, dict)},
              "fit H1:", info["per_draw"][d]["n_fit_H1"])


if __name__ == "__main__":
    main()
