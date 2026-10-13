"""S1.1 job: one pair x one preprocessing, all nested draws, both directions.

Real data (COCO pool) -- REFUSED unless the lead has authorized it:

    python s11_run.py --pair dinov2_b__clip_b --pre prh_clip+l2 --clip-mode pool-layer

requires analysis/s11/LEAD_AUTHORIZATION.json (written by the lead, never by
this role) with at least
    {"authorized_by": "lead", "utc": "...",
     "D4": "a-frozen" | "b-v1.1",
     "predictor_variant": "P1-med-B-r8" (or the v1.1 variant),
     "predictor_choice_sha256": sha256 of the PREDICTOR_CHOICE file in force,
     "predict_py_sha256": sha256 of hrhv2/predict.py in force,
     "plan_sha256": sha256 of analysis/s11/ANALYSIS_PLAN_S11.md (frozen),
     "clip_mode": "pool-layer" | "pool-stored",
     "D1_D2_recorded": "<STATUS.md line or file that records the G0 decisions>"}
and checks every hash against the files on disk before touching a feature file.

Synthetic dry run (no real data; validates the pipeline end to end):

    python s11_run.py --synthetic P-Rough --syn-p 4 --syn-rep 0 --out-root results/S1_1/dryrun_synthetic

Outputs: <out-root>/raw/<pre>/<pair>.parquet (one row per draw x m x k),
<pair>.json (fit diagnostics, timings, hashes), <pair>.done (written last).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import s11_common as C  # noqa: E402
from s11_core import run_pair_draw  # noqa: E402
from hrhv2 import predict as pr  # noqa: E402
from hrhv2 import plants, util  # noqa: E402

AUTH_PATH = os.path.join(C.ANALYSIS_DIR, "LEAD_AUTHORIZATION.json")
PLAN_PATH = os.path.join(C.ANALYSIS_DIR, "ANALYSIS_PLAN_S11.md")
PREDICT_PY = os.path.join(C.PROJECT, "hrhv2", "predict.py")
FROZEN_PREDICT_SHA = "a17964b4dc4a04673eda1a0798fa5d27afb3ea5c61ea8a9b251a2a2d5777b74c"
FROZEN_CHOICE_SHA = "75e1ba3dec56b5db64c4d0f83afeac64fd6b655a3f3201b283257e2050badc5b"

SYN_PLANTS = ("P-Gauss", "P-GaussSat", "P-Rough", "P-Conf", "P-Aniso")
SYN_CLASS = {"P-Gauss": "seed-seed", "P-GaussSat": "seed-seed", "P-Aniso": "seed-seed", "P-Conf": "cross-vision",
             "P-Rough": "cross-modal"}   # pseudo-classes, ONLY to exercise the evaluator


def check_authorization(args):
    """Return the authorization record, or raise SystemExit (no COCO access)."""
    if not os.path.exists(AUTH_PATH):
        raise SystemExit(f"REFUSED: {AUTH_PATH} does not exist. S1.1 on COCO needs the lead's D4 "
                         "decision (verifier report section 6) and a frozen ANALYSIS_PLAN_S11.md.")
    with open(AUTH_PATH) as f:
        auth = json.load(f)
    need = ("authorized_by", "utc", "D4", "predictor_variant", "predictor_choice_sha256",
            "predict_py_sha256", "plan_sha256", "clip_mode", "D1_D2_recorded")
    miss = [k for k in need if k not in auth]
    if miss:
        raise SystemExit(f"REFUSED: authorization lacks {miss}")
    if C.file_sha256(PLAN_PATH) != auth["plan_sha256"]:
        raise SystemExit("REFUSED: ANALYSIS_PLAN_S11.md changed after authorization")
    if C.file_sha256(PREDICT_PY) != auth["predict_py_sha256"]:
        raise SystemExit("REFUSED: hrhv2/predict.py differs from the authorized version")
    if args.variant != auth["predictor_variant"]:
        raise SystemExit(f"REFUSED: --variant {args.variant} != authorized {auth['predictor_variant']}")
    if args.pre == "prh_clip+l2" and args.clip_mode != auth["clip_mode"]:
        raise SystemExit(f"REFUSED: --clip-mode {args.clip_mode} != authorized {auth['clip_mode']}")
    return auth


def synthetic_pair(plant: str, p: int, rep: int, m: int):
    """Known-answer plant pair on m nested items (hrhv2.plants; pre='none').
    Returns (DA, DB, item_ids, info)."""
    key = f"S11dry|{plant}|p={p}|rep={rep}"
    if plant in ("P-Gauss", "P-GaussSat"):
        cc = 0.15 if plant == "P-Gauss" else 0.4     # 0.4 = G0's most saturated P-Gauss setting
        ref = plants.feature_dists(plants.base_cloud(1024, p, f"{key}|ref"))
        sigma = cc * plants.knn_d_median(ref, 10)
        DA, DB = plants.gauss_plant(m, p, sigma, key)
        # Dry-run only: clip D_B at 0 (hrhv2_ref convention, SPEC_CHANGELOG D1). The B->A direction
        # uses D_B as the 'A' row, and hrhv2.predict's log covariate is NaN for negative distances
        # (IndexError in CouplingModel.F). Real features give D >= 0 (SPEC section 0), so this
        # never arises on COCO.
        info = {"sigma": sigma, "c": cc, "n_negative_DB_clipped": int((DB < 0).sum())}
        np.maximum(DB, 0.0, out=DB)
    else:
        if plant == "P-Rough":
            base = plants.feature_dists(plants.base_cloud(256, p, f"{key}|ref"))
            s2 = 0.3 * plants.knn_d_median(base, 10)
            XA, XB = plants.rough_features(m, p, key, s2)
            info = {"s2": s2}
        elif plant == "P-Conf":
            XA, XB = plants.conf_features(m, p, key)
            info = {}
        elif plant == "P-Aniso":
            XA, XB = plants.aniso_features(m, p, key)
            info = {}
        else:
            raise ValueError(plant)
        DA = plants.feature_dists(XA)
        DB = plants.feature_dists(XB)
    info.update({"plant": plant, "p": p, "rep": rep, "key": key})
    return DA, DB, [f"syn{rep}-{i}" for i in range(m)], info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default=None, help="pair id from s11_common.PAIRS")
    ap.add_argument("--pre", default="prh_clip+l2", choices=C.PRES + ("none",))
    ap.add_argument("--clip-mode", default=None, choices=C.CLIP_MODES)
    ap.add_argument("--variant", default=pr.FROZEN_P1_VARIANT)
    ap.add_argument("--draws", default="0,1,2,3,4")
    ap.add_argument("--m-grid", default=",".join(map(str, C.M_GRID)))
    ap.add_argument("--directions", default="AB,BA")
    ap.add_argument("--synthetic", default=None, choices=SYN_PLANTS)
    ap.add_argument("--syn-p", type=int, default=4)
    ap.add_argument("--syn-rep", type=int, default=0)
    ap.add_argument("--out-root", default=C.RESULTS_DIR)
    args = ap.parse_args(argv)

    m_grid = [int(x) for x in args.m_grid.split(",")]
    draws = [int(x) for x in args.draws.split(",")]
    dirs = tuple(args.directions.split(","))
    t_start = time.perf_counter()
    meta = {"role": "s11", "argv": sys.argv, "host": platform.node(), "python": sys.version,
            "numpy": np.__version__, "predict_py_sha256": C.file_sha256(PREDICT_PY),
            "variant": args.variant, "m_grid": m_grid, "m_fit": C.M_FIT, "ks": list(C.KS),
            "draws": draws, "directions": list(dirs),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "code_sha256": {f: C.file_sha256(os.path.join(HERE, f)) for f in
                            ("s11_common.py", "s11_core.py", "s11_run.py")},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if args.synthetic:
        if args.pre != "none":
            raise SystemExit("synthetic plants use --pre none")
        pair_id = f"syn_{args.synthetic}_p{args.syn_p}_r{args.syn_rep}"
        cls = SYN_CLASS[args.synthetic]
        m_max = max(m_grid)
        DA, DB, ids, info = synthetic_pair(args.synthetic, args.syn_p, args.syn_rep, m_max)
        meta.update({"dataset": "synthetic", "pair": pair_id, "cls": cls, "primary": True,
                     "a_info": info, "b_info": info})
    else:
        if args.pair not in C.PAIRS:
            raise SystemExit(f"unknown --pair {args.pair}")
        if args.pre == "prh_clip+l2" and args.clip_mode is None:
            raise SystemExit("--clip-mode is required with prh_clip+l2 (verifier D5; ANALYSIS_PLAN_S11.md O2)")
        auth = check_authorization(args)          # raises before any feature file is opened
        P = C.PAIRS[args.pair]
        pair_id, cls = P["pair"], P["cls"]
        ids, _ = C.coco_item_ids()
        DA, a_info = C.model_D(P["a"], args.pre, args.clip_mode)
        DB, b_info = C.model_D(P["b"], args.pre, args.clip_mode)
        meta.update({"dataset": "coco_pool", "pair": pair_id, "cls": cls, "primary": P["primary"],
                     "a_info": a_info, "b_info": b_info, "authorization": auth,
                     "items_sha256": C.sidecar(C.MODELS[P["a"]])["items_sha256"]})
        if args.variant != pr.FROZEN_P1_VARIANT:
            meta["variant_note"] = "non-frozen variant (PREDICTOR_CHOICE v1.1 per authorization)"
    if args.variant == pr.FROZEN_P1_VARIANT and meta["predict_py_sha256"] != FROZEN_PREDICT_SHA:
        meta["warning_predict_hash"] = "predict.py hash differs from the frozen a17964b4..."

    out_dir = os.path.join(args.out_root, "raw", args.pre)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, pair_id)
    for ext in (".done", ".failed"):
        if os.path.exists(stem + ext):
            os.remove(stem + ext)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {pair_id} {args.pre}: {msg}", flush=True)

    try:
        h1 = util.hash_half([str(i) for i in ids], C.HALVES_SALT)
        orders = C.draw_orders(ids)
        all_rows = []
        meta["draw_diag"] = {}
        meta["h1_fraction_pool"] = float(h1.mean())
        import pandas as pd
        ckpt_sig = {k: meta[k] for k in ("predict_py_sha256", "variant", "m_grid", "m_fit", "ks", "directions",
                                        "code_sha256")}
        for d in draws:
            ck = f"{stem}.draw{d}"
            if os.path.exists(ck + ".parquet") and os.path.exists(ck + ".json"):
                with open(ck + ".json") as f:
                    cj = json.load(f)
                if cj.get("sig") == json.loads(json.dumps(ckpt_sig)):
                    all_rows.extend(pd.read_parquet(ck + ".parquet").to_dict("records"))
                    meta["draw_diag"][d] = cj["diag"]
                    log(f"draw {d}: reused checkpoint {ck}.parquet")
                    continue
            log(f"draw {d}")
            t0 = time.perf_counter()
            rows, diag = run_pair_draw(DA, DB, orders[d], h1, m_grid=m_grid, m_fit=C.M_FIT, ks=C.KS,
                                       variant=args.variant, directions=dirs, log=log)
            for r in rows:
                r.update({"pair": pair_id, "cls": cls, "primary": meta["primary"], "pre": args.pre,
                          "draw": d})
            all_rows.extend(rows)
            diag["seconds"] = time.perf_counter() - t0
            meta["draw_diag"][d] = diag
            pd.DataFrame(rows).to_parquet(ck + ".parquet", index=False)
            with open(ck + ".json", "w") as f:
                json.dump({"sig": ckpt_sig, "diag": diag}, f,
                          default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            log(f"draw {d} done in {diag['seconds']:.0f}s (checkpoint {ck}.parquet)")
        df = pd.DataFrame(all_rows)
        df.to_parquet(stem + ".parquet", index=False)
        meta["n_rows"] = len(df)
        meta["seconds_total"] = time.perf_counter() - t_start
        meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["parquet_sha256"] = C.file_sha256(stem + ".parquet")
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        with open(stem + ".done", "w") as f:
            f.write(meta["finished_utc"] + "\n")
        log(f"done in {meta['seconds_total']:.0f}s -> {stem}.parquet")
    except BaseException:
        with open(stem + ".failed", "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
