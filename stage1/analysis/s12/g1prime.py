"""Gate G1' re-check through the S1.2 pipeline (3 LM-ViT pairs, r=0.5,
prh_clip+l2, S1.2 layer rule): our mKNN / CKA / calibration vs the pinned
PRH (dcd76ba) and ARH (785054b) code fed the SAME float64 rows, and ARH
calibrate() fed the SAME permutations as the S1.2 main run.

Reuses the verifier's loaders (results/verification/vendor_identity.py,
read-only import). Compares against the values stored by run_pairs.py
(results/S1_2/main/pairs/*.json), so it checks the whole S1.2 driver
(layer choice, clip threshold, fast calibration path), not just hrhv2.

Output: results/S1_2/g1prime.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "results", "verification"))
os.environ.setdefault("VERIF_TORCH_THREADS", "4")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import vendor_identity as vi  # noqa: E402  (verifier helpers; loads pinned PRH/ARH/SOG)
from hrhv2 import calibrate as cal  # noqa: E402
from hrhv2.prep import center_gram  # noqa: E402

PAIRS = [
    ("bigscience__bloomz-560m", "vit_base_patch16_clip_224.laion2b"),
    ("openlm-research__open_llama_7b", "vit_giant_patch14_dinov2.lvd142m"),
    ("huggyllama__llama-7b", "vit_huge_patch14_224.mae"),
]
R = 0.5
PRE = "prh_clip+l2"
KS = (1, 3, 10, 30, 100)
TOL = 1e-6


def main():
    out = {"tolerance": TOL, "pairs": []}
    t0 = time.time()
    worst = 0.0
    for lm, vit in PAIRS:
        rec = json.load(open(os.path.join(C.RES, "main", "pairs", f"{lm}__{vit}__r{R}.json")))
        mine = rec["per_pre"][PRE]
        XA, _ = C.features(lm, R, PRE)
        XB, _ = C.features(vit, R, PRE)
        tA, tB = torch.from_numpy(XA), torch.from_numpy(XB)
        d = {"lm": lm, "vit": vit, "r": R, "pre": PRE, "layer_A": mine["meta_A"]["layer"],
             "layer_B": mine["meta_B"]["layer"], "mknn": {}, "cka": {}, "calib": {}}
        # mKNN
        for k in KS:
            ours = mine["ladder"][f"s_N_k{k}"]
            prh = vi.prh_mknn(tA, tB, k)
            sog = vi.sog_mknn(tA, tB, k)
            d["mknn"][k] = {"ours": ours, "prh": prh, "arh_sog": sog,
                            "absdiff_prh": abs(ours - prh), "absdiff_arh_sog": abs(ours - sog)}
        # CKA (match epsilon conventions)
        KcA, KcB = center_gram(XA @ XA.T), center_gram(XB @ XB.T)
        hab, haa, hbb = float(np.vdot(KcA, KcB)), float(np.vdot(KcA, KcA)), float(np.vdot(KcB, KcB))
        L = mine["ladder"]
        ours = {"biased": L["cka_biased"], "debiased": L["cka_debiased"], "debiased_eps8": L["cka_debiased_arheps"],
                "biased_eps6": hab / (math.sqrt(haa * hbb) + 1e-6), "biased_eps8": hab / (math.sqrt(haa * hbb) + 1e-8),
                "debiased_eps6": L["hsicu_AB"] / (math.sqrt(L["hsicu_AA"] * L["hsicu_BB"]) + 1e-6)}
        ven = vi.vendor_cka(tA, tB)
        for vk, ok in vi.CKA_MAP.items():
            d["cka"][vk] = {"vendored": ven[vk], "ours": ours[ok], "absdiff": abs(ven[vk] - ours[ok])}
        # calibration: ARH calibrate() with the S1.2 permutations
        key = rec["perm_key"]
        K = rec["K"]
        for name, sim in (("s_N_k10", lambda X, Y: vi.prh_mknn(X, Y, 10)),
                          ("cka_biased", lambda X, Y: vi.arh_est.cka_biased(X, Y))):
            feeder = vi.PermFeeder(XA.shape[0], key, K)
            sc, p, tau = vi.arh_calibrate(tA, tB, sim, K=K, alpha=0.05, smax=1.0, perm_fn=feeder)
            m_ = mine["calib"][name]
            d["calib"][name] = {"ours": {k: m_[k] for k in ("s_obs", "s_cal", "c_alpha", "p_value")},
                                "arh_calibrate": {"s_cal": float(sc), "p_value": float(p), "c_alpha": float(tau)},
                                "absdiff_s_cal": abs(float(sc) - m_["s_cal"]),
                                "absdiff_c_alpha": abs(float(tau) - m_["c_alpha"]),
                                "absdiff_p": abs(float(p) - m_["p_value"])}
        diffs = ([v["absdiff_prh"] for v in d["mknn"].values()] + [v["absdiff_arh_sog"] for v in d["mknn"].values()]
                 + [v["absdiff"] for v in d["cka"].values()]
                 + [v[x] for v in d["calib"].values() for x in ("absdiff_s_cal", "absdiff_c_alpha", "absdiff_p")])
        d["max_absdiff"] = max(diffs)
        worst = max(worst, d["max_absdiff"])
        out["pairs"].append(d)
        print(f"[{time.time() - t0:5.0f}s] {lm} {vit}: max absdiff {d['max_absdiff']:.3g}", flush=True)
    out["max_absdiff_all"] = worst
    out["pass"] = bool(worst <= TOL)
    json.dump(C.jsonable(out), open(os.path.join(C.RES, "g1prime.json"), "w"), indent=1)
    print("G1' pass" if out["pass"] else "G1' FAIL", worst, flush=True)


if __name__ == "__main__":
    main()
