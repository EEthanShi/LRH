"""s11-select: timing of (a) a nulls-only v1.1 unit (N0, N0H1, N1 in modes d2+mu; no P1 variant)
on COCO-size dev plants, (b) full 16384 x 16384 D construction (hrhv2.prep, prh_clip+l2) and the
all-H2 kNN tables on SYNTHETIC Gaussian features of COCO widths. No real data. Timing only."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sel_common as S
from hrhv2 import predict_v11 as v, prep, structural as st, util

def unit(task):
    plant, p = task
    u = S.plant_unit(plant, p, 0, "S11selCost")
    t0 = time.perf_counter()
    v.run_unit(u["DA_rows"], u["DB_rows"], u["anchor_pos"], u["DA_fit"], u["DB_fit"], u["h1_fit"],
               m_grid=S.PLANT_M_GRID, m_f=S.PLANT_M_FIT, ks=S.PLANT_KS, variants=(), modes=v.MODES)
    return {"task": f"nulls-only {plant} p={p}", "seconds": time.perf_counter() - t0}

def dbuild(d):
    X = util.rng_from_key(f"S11selCost|X|{d}").standard_normal((16384, d)).astype(np.float32)
    t0 = time.perf_counter()
    G = prep.Geom.from_features(X, "prh_clip+l2", keep_X=False)
    t1 = time.perf_counter()
    h1 = util.hash_half([str(i) for i in range(16384)])
    st.knn_table(G.D, 101)
    t2 = time.perf_counter()
    return {"task": f"D build d={d} (1 thread)", "seconds": t1 - t0, "knn_table_all_rows_s": t2 - t1}

if __name__ == "__main__":
    with mp.get_context("fork").Pool(4) as pool:
        out = pool.map(unit, [("P-GaussSat", 2), ("P-Conf", 8)]) + pool.map(dbuild, [768, 3200])
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))
