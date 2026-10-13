"""s11-select: per-unit runtime of the v1.1 protocol with ONE variant (+ N1, N0, N0H1,
modes d2 and mu), i.e. the work of one COCO (pair, pre, draw, direction) job, measured on
dev-seed plants of COCO size (M = 16384, ~512 anchors). Plants: P-GaussSat p=2 (saturated,
worst case: exact k=1 touches ~all columns) and P-Conf p=8 (smooth). Timing only.

    python coco_cost.py OUT.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time, multiprocessing as mp  # noqa: E401,E402
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel_common as S  # noqa: E402
from hrhv2 import predict_v11 as v  # noqa: E402


def work(task):
    plant, p, var = task
    u = S.plant_unit(plant, p, 0, "S11selCost")
    t0 = time.perf_counter()
    res = v.run_unit(u["DA_rows"], u["DB_rows"], u["anchor_pos"], u["DA_fit"], u["DB_fit"], u["h1_fit"],
                     m_grid=S.PLANT_M_GRID, m_f=S.PLANT_M_FIT, ks=S.PLANT_KS, variants=(var,), modes=v.MODES)
    return {"plant": plant, "p": p, "variant": var, "seconds": time.perf_counter() - t0,
            "timing": res["diag"]["timing"], "n_anchors": int(u["anchor_pos"].size)}


if __name__ == "__main__":
    tasks = [(pl, p, var) for pl, p in (("P-GaussSat", 2), ("P-Conf", 8)) for var in ("i-LSlog", "ii-IsoXF", "iii-LSlog-EB")]
    with mp.get_context("fork").Pool(2) as pool:
        out = pool.map(work, tasks)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for o in out:
        print(o["plant"], o["p"], o["variant"], round(o["seconds"]), {k: round(x) for k, x in o["timing"].items()})
