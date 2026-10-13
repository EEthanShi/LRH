"""OpenLLaMA axis, sub-role R: S0b test fixture for review/ol_axis_review.py (TEST ONLY).

NEW FILE (OL-AXIS). Runs only in S0b test mode (guard.test_mode(): env OL_S0B_TEST=1 and no CODE FROZEN line yet)
and writes only under ~/oll_s0b_scratch/. It builds a fixture root that mirrors the production layout with symlinks
to KNOWN files (VERDICTS_SPEC.md section 10) plus a synthetic partB_features_ol.json (spec section 8 schema):

  <root>/results/S1_2_layermax_openllama/early/pairs/<pre>/<slug>__<vit>.{json,npz}   -> E-format stand-ins
  <root>/results/S1_2_layermax_openllama/indep/<pre>/pairs/<slug>__<vit>.npz         -> I-format stand-ins
  <root>/results/S1_2_layermax_openllama/partB/partB_features_ol.json               synthetic Part B
  <root>/frozen8/pairs/<pre>/..., <root>/frozen8/indep/<pre>/pairs/...                frozen 8-LM sources

Variants (the stand-in 13b slug is always synth__ol13b; no name contains open_llama_13b):
  llama3  3b <- llama-7b, 7b <- llama-13b, 13b <- llama-30b (LLaMA-axis E and I pair files). The OpenLLaMA-axis
          trends then equal the LLaMA-axis trends (reproduction check). Frozen 3b <- frozen 8-LM llama-7b,
          frozen 7b <- LLaMA-axis llama-13b.
  ol2     3b, 7b <- the frozen 8-LM OpenLLaMA 3b/7b files, 13b <- LLaMA-axis llama-13b. The 10-LM panel of this
          fixture is then exactly the LLaMA-axis pooled panel (pooled-r reproduction check). E-format 3b/7b files
          lack the three newer subsets and calibrate-then-max in JSON (R recomputes mKNN/ctm; CKA subsets -> None).

usage: OL_S0B_TEST=1 launch_ol.sh R_fx 1 -- venv/bin/python analysis/s12_layermax_openllama/review/make_test_fixture.py \
           --variant llama3|ol2 --root ~/oll_s0b_scratch/R/fx_<variant>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
FROZEN8 = os.path.join(PROJ, "results", "S1_2_layermax")
LLAMA_AX = os.path.join(PROJ, "results", "S1_2_layermax_llama")
PRES = ("prh_clip+l2", "l2")
OL3, OL7, SYN = "openlm-research__open_llama_3b", "openlm-research__open_llama_7b", "synth__ol13b"
L7, L13, L30 = "huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"
BLOOMZ = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
          "bigscience__bloomz-7b1"]
HS = (0.0003, 0.0005, 0.001, 0.0015, 0.002)


def link(src, dst):
    assert os.path.exists(src), src
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def spearman3(y):
    from scipy.stats import rankdata
    x = np.array([1.0, 2.0, 3.0]) - 2.0
    r = rankdata(y) - 2.0
    return float((x @ r) / math.sqrt((x @ x) * (r @ r))) if np.any(r) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("llama3", "ol2"), required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    sys.path.insert(0, AXIS)
    import guard
    import verdict_keys as VK
    if not guard.test_mode():
        print("make_test_fixture.py runs only in S0b test mode (OL_S0B_TEST=1, before CODE FROZEN)", file=sys.stderr)
        sys.exit(3)
    root = os.path.realpath(os.path.expanduser(args.root))
    if not root.startswith(guard.SCRATCH + os.sep):
        sys.exit(f"--root must lie under {guard.SCRATCH}")
    vits = list(VK.VITS)
    if args.variant == "llama3":
        src = {"3b": L7, "7b": L13, "13b": L30}
        e_src = {s: ("lla", m) for s, m in src.items()}
        frz_src = {"3b": ("frz", L7), "7b": ("lla", L13)}
    else:
        src = {"3b": OL3, "7b": OL7, "13b": L13}
        e_src = {"3b": ("frz", OL3), "7b": ("frz", OL7), "13b": ("lla", L13)}
        frz_src = {"3b": ("frz", OL3), "7b": ("frz", OL7)}
    tgt = {"3b": OL3, "7b": OL7, "13b": SYN}
    res = os.path.join(root, "results", "S1_2_layermax_openllama")

    def e_files(kind, P, lm, vit):
        b = os.path.join(FROZEN8 if kind == "frz" else os.path.join(LLAMA_AX, "early"), "pairs", P, f"{lm}__{vit}")
        return b + ".json", b + ".npz"

    def i_file(kind, P, lm, vit):
        base = os.path.join(FROZEN8, "indep") if kind == "frz" else os.path.join(LLAMA_AX, "indep")
        return os.path.join(base, P, "pairs", f"{lm}__{vit}.npz")

    for P in PRES:
        for s in ("3b", "7b", "13b"):
            for vit in vits:
                j, z = e_files(e_src[s][0], P, e_src[s][1], vit)
                b = os.path.join(res, "early", "pairs", P, f"{tgt[s]}__{vit}")
                link(j, b + ".json")
                link(z, b + ".npz")
                link(i_file(e_src[s][0], P, e_src[s][1], vit), os.path.join(res, "indep", P, "pairs", f"{tgt[s]}__{vit}.npz"))
        for lm in BLOOMZ + [L7]:
            for vit in vits:
                j, z = e_files("frz", P, lm, vit)
                b = os.path.join(root, "frozen8", "pairs", P, f"{lm}__{vit}")
                link(j, b + ".json")
                link(z, b + ".npz")
                link(i_file("frz", P, lm, vit), os.path.join(root, "frozen8", "indep", P, "pairs", f"{lm}__{vit}.npz"))
        for s in ("3b", "7b"):
            for vit in vits:
                j, z = e_files(frz_src[s][0], P, frz_src[s][1], vit)
                b = os.path.join(root, "frozen8", "pairs", P, f"{tgt[s]}__{vit}")
                link(j, b + ".json")
                link(z, b + ".npz")
                link(i_file(frz_src[s][0], P, frz_src[s][1], vit),
                     os.path.join(root, "frozen8", "indep", P, "pairs", f"{tgt[s]}__{vit}.npz"))
    # ------------------------------------------------------------------ synthetic Part B (spec section 8)
    PB = json.load(open(os.path.join(LLAMA_AX, "partB", "partB_features.json")))
    pooled_real = BLOOMZ + [OL3, OL7, None, L7, L13, L30]
    pooled = [SYN if s is None else s for s in pooled_real]
    src_of = {s: s for s in pooled if s != SYN}
    src_of.update({tgt[s]: src[s] for s in ("3b", "7b", "13b")})   # stand-ins carry their source's features

    def sidecar(slug):
        for d in ("wit1024", "wit1024_llama_axis"):
            f = os.path.join(PROJ, "features", d, slug + ".json")
            if os.path.isfile(f):
                return json.load(open(f))
        raise FileNotFoundError(slug)

    num_params = {s: int(sidecar(src_of[s])["num_params"]) for s in pooled}
    L_stored = {s: int(sidecar(src_of[s])["shape"][1]) for s in pooled}
    pb = {"schema": "OL-PARTB v1", "m": 1024, "delta_b": 0.005,
          "band": {"o": 0.0019, "o_lo": 0.0009, "o_hi": 0.0029, "h": 0.001}, "mu_band": [-0.0002, 0.0002],
          "ol_lms": [OL3, OL7, SYN], "vits": vits, "pooled11": pooled, "num_params": num_params,
          "L_stored": L_stored, "per_pre": {},
          "_test_fixture": {"variant": args.variant, "stand_ins": {tgt[s]: src[s] for s in src},
                            "note": "S0b test fixture of the R sub-role; not a Part B result"}}
    for vit in vits:
        pb["L_stored"][vit] = int(PB["per_pre"]["prh_clip+l2"]["vit"][vit]["L"])
    for P in PRES:
        pp = PB["per_pre"][P]
        o = {}
        o["clip_c"] = {s: (float(PB["meta"]["clip_thresholds"][src_of[s]]) if P == "prh_clip+l2" else None) for s in pooled}
        o["PR"] = {tgt[s]: pp["lm"][src[s]]["PR"] for s in src}
        o["PR"].update({vit: pp["vit"][vit]["PR"] for vit in vits})
        o["PR_star"] = {k: float(max(v)) for k, v in o["PR"].items()}
        o["u_star"] = {k: int(np.argmax(v)) for k, v in o["PR"].items()}
        o["u_star_rel"] = {k: o["u_star"][k] / (len(v) - 1) for k, v in o["PR"].items()}
        o["Bbar"] = {s: {vit: float(pp["Bbar"][src_of[s]][vit]) for vit in vits} for s in pooled}
        o["Bbar_argmax"] = {s: {vit: [int(pp["lm"][src_of[s]]["argmax"]), int(PB["per_pre"][P]["vit_star"][vit]["w_star"])]
                                for vit in vits} for s in pooled}
        o["Bfd"] = {r: {s: {vit: float(pp["Bfd"][r][src[s]][vit]) for vit in vits} for s in src} for r in ("0.5", "1.0")}
        blate = {}
        for s in src:
            pr = np.array(pp["lm"][src[s]]["PR"])
            rel = np.arange(pr.size) / (pr.size - 1)
            mlm = float(pr[rel >= 0.60].max())
            blate[s] = {vit: math.sqrt(mlm * float(max(pp["vit"][vit]["PR"]))) / 1023 for vit in vits}
        o["Blate"] = blate
        B = {"Bbar": {s: o["Bbar"][tgt[s]] for s in src}, "Bfd_0.5": o["Bfd"]["0.5"], "Bfd_1.0": o["Bfd"]["1.0"],
             "Blate": blate}
        o["gaps"] = {}
        for name, d in B.items():
            o["gaps"][name] = {}
            for a, b in (("3b", "7b"), ("7b", "13b"), ("3b", "13b")):
                pv = [d[b][vit] - d[a][vit] for vit in vits]
                o["gaps"][name][f"{a}|{b}"] = {"per_vit": pv, "median": float(np.median(pv))}

        def tr_mat(name):
            d = B[name]
            rho = [spearman3([d[s][vit] for s in ("3b", "7b", "13b")]) for vit in vits]
            tr = float(np.mean(rho))
            mat = abs(abs(tr) - 1) <= 1e-9 and all(o["gaps"][name][k]["median"] >= 0.005 for k in o["gaps"][name])
            return tr, bool(mat)

        o["Tr_pred"], o["material"] = tr_mat("Bbar")
        o["s_O"] = 1 if o["Tr_pred"] >= 0 else -1
        o["s_O_J"] = o["s_O"] if o["material"] else 1
        o["Tr_Bfd"], o["material_Bfd"] = {}, {}
        for r in ("0.5", "1.0"):
            o["Tr_Bfd"][r], o["material_Bfd"][r] = tr_mat(f"Bfd_{r}")
        o["Tr_Blate"], o["material_Blate"] = tr_mat("Blate")
        o["s_late"] = 1 if o["Tr_Blate"] >= 0 else -1
        known = {f: {"3b": {}, "7b": {}} for f in ("T", "c_alpha", "null_mean", "s_cal")}
        for s in ("3b", "7b"):
            for vit in vits:
                j, _ = e_files(frz_src[s][0], P, frz_src[s][1], vit)
                full = json.load(open(j))["summaries"]["cka_biased"]["full"]
                for f, g in (("T", "s_obs"), ("c_alpha", "c_alpha"), ("null_mean", "null_mean"), ("s_cal", "s_cal")):
                    known[f][s][vit] = float(full[g])
        o["known"] = known

        def kap(a, vit, off):
            Ta, ca = known["T"][a][vit], known["c_alpha"][a][vit]
            return Ta + (1 - Ta) * (o["Bbar"][SYN][vit] + off - ca) / (1 - ca)

        bd = {"s_cal": {"lo": {}, "hi": {}}, "unc": {"lo": {}, "hi": {}}, "s_cal_h": {}}
        for a in ("3b", "7b"):
            bd["s_cal"]["lo"][a] = {vit: kap(a, vit, 0.0009) for vit in vits}
            bd["s_cal"]["hi"][a] = {vit: kap(a, vit, 0.0029) for vit in vits}
            bd["unc"]["lo"][a] = {vit: known["T"][a][vit] + o["Bbar"][SYN][vit] - 0.0002 - known["null_mean"][a][vit] for vit in vits}
            bd["unc"]["hi"][a] = {vit: known["T"][a][vit] + o["Bbar"][SYN][vit] + 0.0002 - known["null_mean"][a][vit] for vit in vits}
        for h in HS:
            bd["s_cal_h"][repr(float(h))] = {"lo": {a: {vit: kap(a, vit, 0.0019 - h) for vit in vits} for a in ("3b", "7b")},
                                             "hi": {a: {vit: kap(a, vit, 0.0019 + h) for vit in vits} for a in ("3b", "7b")}}
        o["boundaries"] = bd
        pb["per_pre"][P] = o
    os.makedirs(os.path.join(res, "partB"), exist_ok=True)
    with open(os.path.join(res, "partB", "partB_features_ol.json"), "w") as f:
        json.dump(pb, f, indent=0)
    print(json.dumps({"root": root, "variant": args.variant,
                      "material": {P: pb["per_pre"][P]["material"] for P in PRES},
                      "Tr_pred": {P: pb["per_pre"][P]["Tr_pred"] for P in PRES},
                      "material_Blate": {P: pb["per_pre"][P]["material_Blate"] for P in PRES}}))


if __name__ == "__main__":
    main()
