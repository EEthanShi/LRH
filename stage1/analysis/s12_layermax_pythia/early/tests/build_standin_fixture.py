"""PYT-AXIS (new, implementation E test helper, S0b only): build the 7-size STAND-IN fixture of plan section 11 V-P9
for E's own tests, from frozen non-Pythia files. Runs only in the S0b test mode (PYT_S0B_TEST=1 and no CODE FROZEN line)
and writes only under the E test root (analysis/s12_layermax_pythia/s0b_scratch/E/...). Computes no score.

Stand-in sizes (VERDICTS_SPEC_PYT section 13): 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b <- bigscience__bloomz-560m, -1b1,
-1b7, -3b, -7b1, huggyllama__llama-13b, huggyllama__llama-30b (num_params increasing). Slugs synth__pyt-<size>.

Writes (root = $PYT_E_TEST_ROOT):
  early/pairs/<pre>/synth__pyt-<size>__<vit>.{json,npz}   E pair files of the known models (results/S1_2_layermax/pairs,
                                                           results/S1_2_layermax_llama/early/pairs); json "lm" relabelled,
                                                           "standin_of" added; npz copied byte for byte
  standins.json                                           size -> feature base of the known model (for sidecars)
  partB/partB_features_pyt.json                           an E-MADE stand-in Part B in the schema of VERDICTS_SPEC_PYT
                                                           section 11, from the PRs of the frozen OpenLLaMA Part B
                                                           (partB_features_ol.json); labelled "E-made stand-in"; the
                                                           shared stand-in is partb_pyt.py --standin vp9 (not E's)
  partB/clip_thresholds_partB_pyt.json                    the known models' frozen clip thresholds under the stand-in slugs
  early/cache/clip.json                                   E clip cache stand-in (stand-ins + 17 ViTs, frozen references)
usage: PYT_S0B_TEST=1 PYT_E_TEST_ROOT=<root> python tests/build_standin_fixture.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

import lm_core as LC  # noqa: E402

MAP = {"70m": "bigscience__bloomz-560m", "160m": "bigscience__bloomz-1b1", "410m": "bigscience__bloomz-1b7",
       "1b": "bigscience__bloomz-3b", "1.4b": "bigscience__bloomz-7b1", "2.8b": "huggyllama__llama-13b",
       "6.9b": "huggyllama__llama-30b"}
BAND = {"prh_clip+l2": (0.001069, 0.012920, 0.0008, 124.5), "l2": (0.001551, 0.006843, 0.0018, 79.6)}
FLAT = {"prh_clip+l2": [0.0010, 0.0040], "l2": [0.0010, 0.0045]}


def src_pairs(pre, lm):
    if lm.startswith("huggyllama__llama-") and lm != "huggyllama__llama-7b":
        return os.path.join(LC.LLAMA_E_RES, "pairs", pre)
    return os.path.join(LC.FROZEN_RES, "pairs", pre)


def feat_base(lm):
    if lm.startswith("huggyllama__llama-") and lm != "huggyllama__llama-7b":
        return os.path.join(LC.PROJ, "features", "wit1024_llama_axis", lm)
    return os.path.join(LC.FEAT_W1024, lm)


def spearman_sizes(vals):
    from scipy.stats import rankdata
    r = rankdata(vals)
    x = np.arange(1, len(vals) + 1, dtype=float)
    r, x = r - r.mean(), x - x.mean()
    return float(np.sum(r * x) / math.sqrt(np.sum(r ** 2) * np.sum(x ** 2)))


def main():
    if not LC.TEST:
        LC._refuse("build_standin_fixture.py runs only with PYT_S0B_TEST=1 (S0b test mode)")
    root = LC.PYT_RES
    assert LC._inside(root, LC.E_TEST_BASE), root
    sizes = list(LC.SIZES)
    slug = {s: LC.SLUG_TEST[s] for s in sizes}
    vits = [v for v, _ in LC.VITS]
    n = 0
    for pre in LC.PRES:
        dst = os.path.join(root, "early", "pairs", pre)
        os.makedirs(dst, exist_ok=True)
        for s in sizes:
            lm = MAP[s]
            for vit in vits:
                b = os.path.join(src_pairs(pre, lm), f"{lm}__{vit}")
                rec = json.load(open(b + ".json"))
                assert rec["lm"] == lm and rec["vit"] == vit
                rec["lm"] = slug[s]
                rec["standin_of"] = lm
                with open(os.path.join(dst, f"{slug[s]}__{vit}.json"), "w") as f:
                    json.dump(rec, f)
                shutil.copyfile(b + ".npz", os.path.join(dst, f"{slug[s]}__{vit}.npz"))
                n += 1
    json.dump({s: feat_base(MAP[s]) for s in sizes}, open(os.path.join(root, "standins.json"), "w"), indent=1)
    # ---- E-made stand-in Part B (schema of VERDICTS_SPEC_PYT section 11) from the frozen OpenLLaMA Part B PRs
    OLPB = json.load(open(LC.OL_PARTB_JSON))
    side = {s: json.load(open(feat_base(MAP[s]) + ".json")) for s in sizes}
    num_params = {slug[s]: int(side[s]["num_params"]) for s in sizes}
    assert [num_params[slug[s]] for s in sizes] == sorted(num_params.values())
    L_stored = {slug[s]: int(side[s]["shape"][1]) for s in sizes}
    for v in vits:
        L_stored[v] = int(json.load(open(os.path.join(LC.FEAT_W1024, v + ".json")))["shape"][1])
    PB = {"schema": "PYT-PARTB v1", "label": "E-made stand-in (tests of implementation E only; not partb_pyt.py)",
          "m": 1024, "delta_b": 0.005, "sizes": sizes, "pyt_lms": [slug[s] for s in sizes], "vits": vits,
          "vit_family4": {v: ("in21k" if "augreg_in21k" in v else "MAE" if v.endswith(".mae") else
                              "DINOv2" if "dinov2" in v else "CLIP") for v in vits},
          "num_params": num_params, "width": {slug[s]: int(side[s]["shape"][2]) for s in sizes},
          "depth_blocks": {slug[s]: int(side[s]["shape"][1]) - 1 for s in sizes}, "L_stored": L_stored,
          "band_constants": {p: {"a": BAND[p][0], "b": BAND[p][1], "h": BAND[p][2], "fit_min_PR": BAND[p][3],
                                 "flat": FLAT[p]} for p in LC.PRES},
          "mu_band": [-0.0003, 0.0008], "standin_of": {slug[s]: MAP[s] for s in sizes}, "per_pre": {}}
    m1 = 1023.0
    for pre in LC.PRES:
        P = OLPB["per_pre"][pre]
        PR = {slug[s]: [float(x) for x in P["PR"][MAP[s]]] for s in sizes}
        PR.update({v: [float(x) for x in P["PR"][v]] for v in vits})
        prs = {k: max(vals) for k, vals in PR.items()}
        ustar = {k: int(np.argmax(vals)) for k, vals in PR.items()}
        Bbar = {slug[s]: {v: math.sqrt(prs[slug[s]] * prs[v]) / m1 for v in vits} for s in sizes}
        Blate = {}
        for s in sizes:
            L = len(PR[slug[s]])
            late = max(PR[slug[s]][u] for u in range(L) if u / (L - 1) >= 0.60)
            Blate[slug[s]] = {v: math.sqrt(late * prs[v]) / m1 for v in vits}
        G = np.array([[Bbar[slug[s]][v] for v in vits] for s in sizes])
        Gl = np.array([[Blate[slug[s]][v] for v in vits] for s in sizes])

        def mat(Gx, db=0.005):
            tr = spearman_sizes(list(Gx[:, 0]))
            sg = 1 if tr >= 0 else -1
            gap = float(np.median(sg * (Gx[-1] - Gx[0])))
            return tr, sg, bool(abs(tr) >= 0.75 - 1e-9 and gap >= 3 * db), gap
        trb, sb, matb, gapb = mat(G)
        trl, sl, matl, gapl = mat(Gl)
        a, b, h, fmin = BAND[pre]
        band = {s: {"f": a + b / math.sqrt(prs[slug[s]])} for s in sizes}
        for s in sizes:
            band[s]["lo"], band[s]["hi"] = band[s]["f"] - h, band[s]["f"] + h
        eq = {s: bool(all(Blate[slug[s]][v] == Bbar[slug[s]][v] for v in vits)) for s in sizes}
        branch = "H2a" if matl else "H2b"
        PB["per_pre"][pre] = {
            "PR": PR, "PR_star": prs, "u_star": ustar,
            "u_star_rel": {k: ustar[k] / (len(PR[k]) - 1) for k in PR},
            "Bbar": Bbar, "Blate": Blate, "Tr_Bbar": trb, "Tr_Blate": trl,
            "materiality": {"Bbar": {"abs_tr": abs(trb), "median_end_gap": gapb, "material": matb},
                            "Blate": {"abs_tr": abs(trl), "median_end_gap": gapl, "material": matl}},
            "material_Bbar": matb, "material_Blate": matl,
            "material_curve_db": {"Bbar": {repr(d): mat(G, d)[2] for d in (0.0025, 0.005, 0.0075, 0.01)},
                                  "Blate": {repr(d): mat(Gl, d)[2] for d in (0.0025, 0.005, 0.0075, 0.01)}},
            "s": sb, "s_flag_zero": trb == 0, "s_late": sl, "s_late_flag_zero": trl == 0,
            "members": (["H1"] if matb else []) + [branch, "H3", "H4"], "H2_branch": branch,
            "Blate_equals_Bbar": eq, "n_Blate_equals_Bbar": int(sum(eq.values())),
            "H2_same_as_H1": bool(sum(eq.values()) >= 6),
            "band": band, "band_flat": FLAT[pre],
            "band_h_curve": {repr(hh): {s: [band[s]["f"] - hh, band[s]["f"] + hh] for s in sizes}
                             for hh in (0.0004, 0.0006, 0.0008, 0.001, 0.0015)},
            "extrapolated": {s: bool(prs[slug[s]] < fmin) for s in sizes}}
    os.makedirs(os.path.join(root, "partB"), exist_ok=True)
    json.dump(PB, open(os.path.join(root, "partB", "partB_features_pyt.json"), "w"), indent=1)
    # ---- clip thresholds (frozen references of the known models) under the stand-in slugs
    s12 = json.load(open(os.path.join(LC.PROJ, "results", "S1_2", "cache", "clip_thresholds.json")))
    llb = json.load(open(os.path.join(LC.PROJ, "results", "S1_2_layermax_llama", "partB", "clip_thresholds_partB.json")))["clip"]

    def clip_of(lm):
        return float(s12[f"{lm}|wit1024"]) if f"{lm}|wit1024" in s12 else float(llb[lm]["c"])
    json.dump({"clip": {slug[s]: {"c": clip_of(MAP[s]), "standin_of": MAP[s]} for s in sizes}},
              open(os.path.join(root, "partB", "clip_thresholds_partB_pyt.json"), "w"), indent=1)
    os.makedirs(os.path.join(root, "early", "cache"), exist_ok=True)
    cc = {slug[s]: {"c": clip_of(MAP[s]), "standin_of": MAP[s]} for s in sizes}
    cc.update({v: {"c": float(s12[f"{v}|wit1024"])} for v in vits})
    json.dump(cc, open(os.path.join(root, "early", "cache", "clip.json"), "w"), indent=1)
    print(f"fixture written under {root}: {n} pair records, stand-in Part B, clip files", flush=True)


if __name__ == "__main__":
    main()
