"""PYT-AXIS COPY of analysis/s12_layermax_openllama/review/make_test_fixture.py (sha256 e565004a..., pristine in
PYT-AXIS: orig_copy/; diff make_test_fixture_pyt.py.diff). Pythia axis, sub-role R: S0b test fixture (TEST ONLY).

PYT-AXIS: Runs only in S0b test mode (PYT_S0B_TEST=1 and no CODE FROZEN line yet; else exit 3) and writes only under
PYT-AXIS: analysis/s12_layermax_pythia/s0b_scratch/R/ (R's sub-root of the shared scratch root, VERDICTS_SPEC_PYT.md
PYT-AXIS: section 13). It builds the plan V-P9 "7-size stand-in fixture assembled from known pair files (the 5 BLOOMZ
PYT-AXIS: sizes + llama-13b + llama-30b, labelled stand-in)" in the production layout, with symlinks to KNOWN files:

  <root>/results/S1_2_layermax_pythia/early/pairs/<pre>/synth__pyt-<size>__<vit>.{json,npz}   -> E-format stand-ins
  <root>/results/S1_2_layermax_pythia/indep/<pre>/pairs/synth__pyt-<size>__<vit>.npz         -> I-format stand-ins
  <root>/features/synth__pyt-<size>.npy                                                     -> the source features
  <root>/results/S1_2_layermax_pythia/partB/partB_features_pyt.json                        stand-in Part B (spec 11)
  <root>/results/S1_2_layermax_pythia/partB/clip_thresholds_partB_pyt.json                 frozen clip values

PYT-AXIS: Sources: BLOOMZ from the frozen 8-LM run, llama-13b/30b from the LLaMA axis; the stand-in Part B takes the
PYT-AXIS: PR lists and B̄ of the frozen partB_features_ol.json and the clip values of the frozen
PYT-AXIS: clip_thresholds_partB_ol.json; its derived fields (trends, materiality and its delta_b curve, directions,
PYT-AXIS: bands with the frozen constants, h-curve bands, extrapolation flags, B_late) are computed here with scipy,
PYT-AXIS: separately from R's code. (The contract's own stand-in Part B comes from partb_pyt.py --standin vp9; this file
PYT-AXIS: is R's, so that R's S0b tests do not depend on another role's code.) No name contains EleutherAI__pythia.
PYT-AXIS: --flip-partb-material writes material_Bbar = not(material_Bbar) for both pres (exercises the non-member path);
PYT-AXIS: --no-l2 links no l2 pair file (exercises the "l2 not run" path of contract section 9).
PYT-AXIS: No-delete rule: an existing link is kept if it points to the same source; an existing Part B file stops the run.

usage: PYT_S0B_TEST=1 launch_pyt.sh NAME 1 -- env PYTHONDONTWRITEBYTECODE=1 venv/bin/python \\
           analysis/s12_layermax_pythia/review/make_test_fixture_pyt.py --root analysis/s12_layermax_pythia/s0b_scratch/R/fx_vp9
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
HERE = os.path.dirname(os.path.abspath(__file__))  # PYT-AXIS
FROZEN8 = os.path.join(PROJ, "results", "S1_2_layermax")
LLAMA_AX = os.path.join(PROJ, "results", "S1_2_layermax_llama")
OL_PB = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "partB")  # PYT-AXIS
FEAT = os.path.join(PROJ, "features")  # PYT-AXIS
PRES = ("prh_clip+l2", "l2")


def link(src, dst):
    assert os.path.exists(src), src
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):  # PYT-AXIS: no-delete rule (never remove; an identical link is kept)
        if os.path.realpath(dst) != os.path.realpath(src):  # PYT-AXIS
            sys.exit(f"{dst} exists and points elsewhere; use a new --root (no-delete rule)")  # PYT-AXIS
        return  # PYT-AXIS
    os.symlink(src, dst)


def spearman_n(y):  # PYT-AXIS: scipy Spearman of the n values against ranks 1..n (0 if constant)
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=float)
    x = np.arange(1, y.size + 1, dtype=float)
    r = rankdata(y)
    x, r = x - x.mean(), r - r.mean()
    return float((x @ r) / math.sqrt((x @ x) * (r @ r))) if np.any(r) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--flip-partb-material", action="store_true")  # PYT-AXIS
    ap.add_argument("--no-l2", action="store_true")  # PYT-AXIS
    args = ap.parse_args()
    sys.path.insert(0, HERE)  # PYT-AXIS
    import pyt_review_gate as G  # PYT-AXIS
    if not G.test_mode():  # PYT-AXIS
        print("make_test_fixture_pyt.py runs only in S0b test mode (PYT_S0B_TEST=1, before CODE FROZEN)", file=sys.stderr)
        sys.exit(3)
    import pyt_review_conf as CF  # PYT-AXIS
    root = os.path.realpath(os.path.expanduser(args.root))
    if not root.startswith(os.path.realpath(G.test_root()) + os.sep):  # PYT-AXIS
        print(f"--root must lie under {G.test_root()}", file=sys.stderr)  # PYT-AXIS
        sys.exit(3)  # PYT-AXIS
    vits = list(CF.VITS)  # PYT-AXIS
    ax = CF.STANDIN  # PYT-AXIS
    res = os.path.join(root, "results", "S1_2_layermax_pythia")  # PYT-AXIS

    def e_files(src, P, vit):  # PYT-AXIS
        if src.startswith("huggyllama"):
            b = os.path.join(LLAMA_AX, "early", "pairs", P, f"{src}__{vit}")
        else:
            b = os.path.join(FROZEN8, "pairs", P, f"{src}__{vit}")
        return b + ".json", b + ".npz"

    def i_file(src, P, vit):  # PYT-AXIS
        base = os.path.join(LLAMA_AX, "indep") if src.startswith("huggyllama") else os.path.join(FROZEN8, "indep")
        return os.path.join(base, P, "pairs", f"{src}__{vit}.npz")

    for P in (PRES[:1] if args.no_l2 else PRES):  # PYT-AXIS
        for s in ax.sizes:  # PYT-AXIS
            src, tgt = CF.STANDIN_OF[s], ax.slug[s]  # PYT-AXIS
            for vit in vits:
                j, z = e_files(src, P, vit)
                b = os.path.join(res, "early", "pairs", P, f"{tgt}__{vit}")
                link(j, b + ".json")
                link(z, b + ".npz")
                link(i_file(src, P, vit), os.path.join(res, "indep", P, "pairs", f"{tgt}__{vit}.npz"))
    for s in ax.sizes:  # PYT-AXIS: feature stacks of the sources (R's clip check)
        src = CF.STANDIN_OF[s]
        d = "wit1024_llama_axis" if src in ("huggyllama__llama-13b", "huggyllama__llama-30b") else "wit1024"
        link(os.path.join(FEAT, d, src + ".npy"), os.path.join(root, "features", ax.slug[s] + ".npy"))
    # ------------------------------------------------------------------ stand-in Part B (contract section 11 fields)
    OL = json.load(open(os.path.join(OL_PB, "partB_features_ol.json")))  # PYT-AXIS
    CL = json.load(open(os.path.join(OL_PB, "clip_thresholds_partB_ol.json")))["clip"]  # PYT-AXIS
    slug = ax.slug
    pb = {"schema": "PYT-PARTB v1", "m": 1024, "delta_b": 0.005, "sizes": list(ax.sizes),
          "pyt_lms": [slug[s] for s in ax.sizes], "vits": vits, "vit_family4": dict(CF.VIT_FAMILY4),
          "num_params": {slug[s]: int(OL["num_params"][CF.STANDIN_OF[s]]) for s in ax.sizes},
          "width": {slug[s]: ax.width[s] for s in ax.sizes}, "depth_blocks": {slug[s]: ax.depth[s] for s in ax.sizes},
          "L_stored": {slug[s]: len(OL["per_pre"]["prh_clip+l2"]["PR"][CF.STANDIN_OF[s]]) for s in ax.sizes},
          "mu_band": [-0.0003, 0.0008], "per_pre": {},
          "_test_fixture": {"stand_ins": {slug[s]: CF.STANDIN_OF[s] for s in ax.sizes}, "flip": args.flip_partb_material,
                            "no_l2": args.no_l2,
                            "note": "S0b test fixture of the R sub-role; not a Part B result; NOT Pythia"}}
    pb["L_stored"].update({vit: len(OL["per_pre"]["prh_clip+l2"]["PR"][vit]) for vit in vits})
    for P in PRES:
        pp = OL["per_pre"][P]
        o = {}
        o["PR_star"] = {slug[s]: float(max(pp["PR"][CF.STANDIN_OF[s]])) for s in ax.sizes}
        o["PR_star"].update({vit: float(max(pp["PR"][vit])) for vit in vits})
        o["Bbar"] = {slug[s]: {vit: float(pp["Bbar"][CF.STANDIN_OF[s]][vit]) for vit in vits} for s in ax.sizes}
        blate = {}
        for s in ax.sizes:
            pr = np.array(pp["PR"][CF.STANDIN_OF[s]], dtype=float)
            relz = np.arange(pr.size) / (pr.size - 1)
            mlm = float(pr[relz >= 0.60].max())
            blate[slug[s]] = {vit: math.sqrt(mlm * float(max(pp["PR"][vit]))) / 1023 for vit in vits}
        o["Blate"] = blate

        def tr_mat(d, db=0.005):
            rho = [spearman_n([d[slug[s]][vit] for s in ax.sizes]) for vit in vits]
            tr = float(np.mean(rho))
            sB = 1 if tr >= 0 else -1
            gap = float(np.median([sB * (d[slug[ax.sizes[-1]]][vit] - d[slug[ax.sizes[0]]][vit]) for vit in vits]))
            return tr, bool(abs(tr) >= 0.75 - 1e-9 and gap >= round(3 * db, 12)), sB

        o["Tr_Bbar"], o["material_Bbar"], o["s"] = tr_mat(o["Bbar"])
        o["Tr_Blate"], o["material_Blate"], o["s_late"] = tr_mat(o["Blate"])
        o["material_curve_db"] = {k: {repr(db): tr_mat(o[k], db)[1] for db in (0.0025, 0.005, 0.0075, 0.01)}
                                  for k in ("Bbar", "Blate")}
        if args.flip_partb_material:
            o["material_Bbar"] = not o["material_Bbar"]
            o["material_curve_db"]["Bbar"] = {k: not x for k, x in o["material_curve_db"]["Bbar"].items()}
        o["H2_branch"] = "H2a" if o["material_Blate"] else "H2b"
        o["members"] = (["H1"] if o["material_Bbar"] else []) + [o["H2_branch"], "H3", "H4"]
        eq = {s: all(blate[slug[s]][vit] == o["Bbar"][slug[s]][vit] for vit in vits) for s in ax.sizes}
        o["Blate_equals_Bbar"] = eq
        o["n_Blate_equals_Bbar"] = int(sum(eq.values()))
        o["H2_same_as_H1"] = o["n_Blate_equals_Bbar"] >= 6
        c = CF.BAND[P]
        f = {s: c["a"] + c["b"] / math.sqrt(o["PR_star"][slug[s]]) for s in ax.sizes}
        o["band"] = {s: {"f": f[s], "lo": f[s] - c["h"], "hi": f[s] + c["h"]} for s in ax.sizes}
        o["band_h_curve"] = {repr(h): {s: [f[s] - h, f[s] + h] for s in ax.sizes}
                             for h in (0.0004, 0.0006, 0.0008, 0.001, 0.0015)}
        o["band_flat"] = list(CF.FLAT_BAND[P])
        o["extrapolated"] = {s: bool(o["PR_star"][slug[s]] < CF.PR_RANGE_LO[P]) for s in ax.sizes}
        pb["per_pre"][P] = o
    os.makedirs(os.path.join(res, "partB"), exist_ok=True)
    out = os.path.join(res, "partB", "partB_features_pyt.json")
    if os.path.exists(out):  # PYT-AXIS: no-delete rule
        sys.exit(f"{out} exists; use a new --root")
    with open(out, "w") as fh:
        json.dump(pb, fh, indent=0)
    with open(os.path.join(res, "partB", "clip_thresholds_partB_pyt.json"), "w") as fh:  # PYT-AXIS
        json.dump({"clip": {slug[s]: {"c": float(CL[CF.STANDIN_OF[s]]["c"])} for s in ax.sizes},
                   "note": "stand-in: frozen clip_thresholds_partB_ol.json values of the sources"}, fh, indent=1)
    print(json.dumps({"root": root, "material_Bbar": {P: pb["per_pre"][P]["material_Bbar"] for P in PRES},
                      "Tr_Bbar": {P: pb["per_pre"][P]["Tr_Bbar"] for P in PRES},
                      "H2_branch": {P: pb["per_pre"][P]["H2_branch"] for P in PRES},
                      "n_Blate_equals_Bbar": {P: pb["per_pre"][P]["n_Blate_equals_Bbar"] for P in PRES},
                      "extrapolated": {P: [k for k, x in pb["per_pre"][P]["extrapolated"].items() if x] for P in PRES}}))


if __name__ == "__main__":
    main()
