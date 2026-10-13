"""V-f step 3 (role e0-indep): own s_N against E0's obs files under the frozen
agreement rule (plan 10, V-f): overlap counts must be equal. Cell-level total
overlap counts are recovered from E0's obs as obs * n_anchors * k (must be an
integer within 1e-6). Any differing cell is listed for an anchor-level check.

Usage: python vf_compare_obs.py <outdir> <vf_obs.parquet>
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C  # noqa: E402

E0_OBS = os.path.join(C.ROOT, "results", "E0_sentence", "obs")


def main(outdir, vf_path):
    vf = pd.read_parquet(vf_path)
    res = dict(vf_obs=dict(path=vf_path, sha256=C.sha256_file(vf_path)), arms={}, differing_cells=[])
    tot_cells = 0
    for arm, f in C.E0_OBS_FILE.items():
        p = os.path.join(E0_OBS, f + ".parquet")
        e0 = pd.read_parquet(p)
        mine = vf[vf["arm"] == arm]
        mg = mine.merge(e0, on=["pair", "draw", "m", "k"], how="outer", suffixes=("_vf", "_e0"),
                        indicator=True)
        assert (mg["_merge"] == "both").all(), arm
        assert len(mg) == 15 * 5 * 7 * 3
        tot_cells += len(mg)
        cnt_e0 = mg["obs_e0"] * mg["n_anchors_e0"] * mg["k"]
        cnt_e0_int = np.rint(cnt_e0).astype(np.int64)
        int_err = float(np.max(np.abs(cnt_e0 - cnt_e0_int)))
        cnt_all_e0 = mg["obs_allH2_e0"] * mg["n_allH2_e0"] * mg["k"]
        cnt_all_e0_int = np.rint(cnt_all_e0).astype(np.int64)
        same_n = bool((mg["n_anchors_vf"] == mg["n_anchors_e0"]).all())
        same_nall = bool((mg["n_allH2_vf"] == mg["n_allH2_e0"]).all())
        diff_cnt = mg["ov_sum"] != cnt_e0_int
        diff_all = mg["ov_sum_allH2"] != cnt_all_e0_int
        a = dict(
            e0_file=p, e0_sha256=C.sha256_file(p), n_cells=int(len(mg)),
            n_anchors_equal=same_n, n_allH2_equal=same_nall,
            max_dev_e0_count_from_integer=int_err,
            cells_overlap_count_differs=int(diff_cnt.sum()),
            cells_overlap_count_differs_allH2=int(diff_all.sum()),
            max_abs_obs_diff=float(np.max(np.abs(mg["obs_vf"] - mg["obs_e0"]))),
            obs_bitwise_equal_cells=int((mg["obs_vf"].to_numpy() == mg["obs_e0"].to_numpy()).sum()),
            max_abs_obs_allH2_diff=float(np.max(np.abs(mg["obs_allH2_vf"] - mg["obs_allH2_e0"]))),
            # descriptive extras (definitions of E0 columns not given in the plan; informational)
            n_tieB_equal_mine_predicted=int((mg["n_tieB"] == mg["exact_tie_B"]).sum()),
            n_tieB_equal_mine_allH2=int((mg["n_tieB"] == mg["exact_tie_B_allH2"]).sum()),
            n_tieA_equal_mine_predicted=int((mg["n_tieA"] == mg["exact_tie_A"]).sum()),
            n_clean_equal=int((mg["n_clean_vf"] == mg["n_clean_e0"]).sum()),
            obs_clean_maxdiff=float(np.nanmax(np.abs(mg["obs_clean_vf"] - mg["obs_clean_e0"]))),
            neartie_B_maxdiff=float(np.max(np.abs(mg["near_frac_B"] - mg["neartie_B"]))),
            neartie_A_maxdiff=float(np.max(np.abs(mg["near_frac_A"] - mg["neartie_A"]))),
            sum_exact_tie_B_predicted_mine=int(mg["exact_tie_B"].sum()),
            sum_n_tieB_e0=int(mg["n_tieB"].sum()),
        )
        res["arms"][arm] = a
        for r in mg[diff_cnt | diff_all].itertuples(index=False):
            res["differing_cells"].append(dict(arm=arm, pair=r.pair, draw=int(r.draw), m=int(r.m), k=int(r.k),
                                               vf_count=int(r.ov_sum), e0_count=int(np.rint(r.obs_e0 * r.n_anchors_e0 * r.k)),
                                               vf_count_allH2=int(r.ov_sum_allH2),
                                               e0_count_allH2=int(np.rint(r.obs_allH2_e0 * r.n_allH2_e0 * r.k))))
    res["n_cells_total"] = tot_cells
    res["n_cells_differing_predicted"] = int(sum(a["cells_overlap_count_differs"] for a in res["arms"].values()))
    res["n_cells_differing_allH2"] = int(sum(a["cells_overlap_count_differs_allH2"] for a in res["arms"].values()))
    sha = C.write_new_json(os.path.join(outdir, "vf_compare_obs.json"), res)
    for arm, a in res["arms"].items():
        print(arm, {k: v for k, v in a.items() if k not in ("e0_file", "e0_sha256")})
    print("differing cells (predicted anchors):", res["n_cells_differing_predicted"],
          "| allH2:", res["n_cells_differing_allH2"], "| sha", sha)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
