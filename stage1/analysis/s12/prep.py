"""S1.2 prep: clip thresholds, PE3 top-128 PC projections (fit on S_sel),
layer table, and a bitwise check of our preprocessing against hrhv2.

Outputs (results/S1_2/cache/):
  clip_thresholds.json        all-layer PRH/ARH threshold per (model, pool)
  layers.json                 layer index per (model, r) under both rules
  pc128/<stem>__r<r>__<pre>.npy  wit1024 rows projected on the top-128 PCs of S_sel
  pc128_meta.json             variance captured by the 128 PCs (S_sel fit, wit1024)
  prep_checks.json            bitwise identity of preprocess_rows vs hrhv2.prep
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/hrh_v2_stage1_20261008"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import common as C  # noqa: E402
from hrhv2.prep import Geom, prh_clip_threshold  # noqa: E402


def main():
    os.makedirs(os.path.join(C.CACHE, "pc128"), exist_ok=True)
    t0 = time.time()
    path = os.path.join(C.CACHE, "clip_thresholds.json")
    tab = json.load(open(path)) if os.path.exists(path) else {}
    models = [s for s, _ in C.LMS] + [s for s, _ in C.VITS]
    layers = {}
    for stem in models:
        s1 = C.load_stack(stem, "wit1024")
        s2 = C.load_stack(stem, "wit4096_sel")
        r2 = C.rows_for("wit4096_sel")
        if f"{stem}|wit1024" not in tab:
            tab[f"{stem}|wit1024"] = prh_clip_threshold(np.asarray(s1, dtype=np.float64))
        if f"{stem}|wit4096_sel" not in tab:
            tab[f"{stem}|wit4096_sel"] = prh_clip_threshold(np.asarray(s2[r2], dtype=np.float64))
        if f"{stem}|S+Ssel" not in tab:
            both = np.concatenate([np.asarray(s1, dtype=np.float64), np.asarray(s2[r2], dtype=np.float64)])
            tab[f"{stem}|S+Ssel"] = prh_clip_threshold(both)
            del both
        json.dump(tab, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        layers[stem] = {"n_stored": C.n_stored(stem), "kind": C.sidecar(stem)["kind"],
                        "num_params": C.n_params(stem),
                        "primary": {str(r): C.layer_index(stem, r) for r in C.DEPTHS},
                        "sidecar_rule": {str(r): C.sidecar_layer_index(stem, r) for r in C.DEPTHS}}
        print(f"[{time.time() - t0:6.0f}s] clip {stem}", flush=True)
    json.dump(layers, open(os.path.join(C.CACHE, "layers.json"), "w"), indent=1)

    # bitwise identity: preprocess_rows(c from cache) == hrhv2 Geom.from_features(clip_ref=stack)
    checks = {}
    for stem in ("huggyllama__llama-7b", "vit_base_patch16_clip_224.laion2b"):
        stack = np.asarray(C.load_stack(stem, "wit1024"), dtype=np.float64)
        li = C.layer_index(stem, 0.5)
        G_ref = Geom.from_features(stack[:, li, :], "prh_clip+l2", clip_ref=stack)
        Xp, _ = C.features(stem, 0.5, "prh_clip+l2")
        G = C.geom(Xp, "prh_clip+l2")
        checks[stem] = {"X_bitwise": bool(np.array_equal(Xp, G_ref.X)),
                        "K_bitwise": bool(np.array_equal(G.K, G_ref.K)),
                        "D_bitwise": bool(np.array_equal(G.D, G_ref.D))}
        Xl, _ = C.features(stem, 0.5, "l2")
        G_l = Geom.from_features(stack[:, li, :], "l2")
        checks[stem]["l2_D_bitwise"] = bool(np.array_equal(C.geom(Xl, "l2").D, G_l.D))
        del stack
    json.dump(checks, open(os.path.join(C.CACHE, "prep_checks.json"), "w"), indent=1)
    print("prep checks", checks, flush=True)

    # PE3 projections: top-128 PCs fit on S_sel (usable rows, same pre/layer), applied to wit1024
    meta = {}
    for stem in models:
        for r in C.DEPTHS:
            for pre in C.PRES:
                out = os.path.join(C.CACHE, "pc128", f"{stem}__r{r}__{pre}.npy")
                Xs, ms = C.features(stem, r, pre, ds="wit4096_sel")
                Xw, mw = C.features(stem, r, pre, ds="wit1024")
                mu = Xs.mean(axis=0)
                _, s, Vt = np.linalg.svd(Xs - mu, full_matrices=False)
                V = Vt[:C.PC_DIM].T
                Z = (Xw - mu) @ V
                np.save(out, Z)
                Xwc = Xw - Xw.mean(axis=0)
                Zc = Z - Z.mean(axis=0)
                meta[f"{stem}|{r}|{pre}"] = {
                    "d": int(Xs.shape[1]), "m_fit": int(Xs.shape[0]), "layer": ms["layer"],
                    "var_frac_sel": float(np.sum(s[:C.PC_DIM] ** 2) / np.sum(s ** 2)),
                    "var_frac_wit1024": float(np.sum(Zc * Zc) / np.sum(Xwc * Xwc)),
                }
        print(f"[{time.time() - t0:6.0f}s] pc128 {stem}", flush=True)
    json.dump(meta, open(os.path.join(C.CACHE, "pc128_meta.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
