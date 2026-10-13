#!/usr/bin/env python
"""Pythia axis, step S0b: the cross-implementation fixture X of role pyt-code-freeze (2026-10-10). NEW FILE (PYT-AXIS).

Plan analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md (sha256 6de89603...879e7b) section 11 V-P9, last item: "a 7-size
stand-in fixture assembled from known pair files (the 5 BLOOMZ sizes + llama-13b + llama-30b, labelled stand-in) run end
to end through E, I, R and forecast_pyt.py with 0 mismatches". The OpenLLaMA axis added the same test at its S0b
completion ("Integration fixture X", analysis/s12_layermax_openllama/S0B_NOTES.md).

INTEGRATION ONLY. This file computes no score and no verdict, imports no module of E (early/), I (indep/) or R (review/),
and changes nothing in their logic. It
  inputs    lays out symlinks so that R and forecast_pyt.py --impl I read exactly the pair files that E and I aggregated
            (E: the copies made by E's own builder early/tests/build_standin_fixture.py; I: the frozen I-format files that
            lmx_pyt_aggregate.py --fixture reads), the stand-in feature stacks (R's clip check) and the shared stand-in
            Part B s0b_scratch/PB/partB_vp9/ (partb_pyt.py --standin vp9); it also checks that E's copies equal the frozen
            originals (json equal except the relabelled "lm" and the added "standin_of"; npz byte-identical);
  verdicts  copies the verdict files of E, I and forecast_pyt.py (E and I) into the production layouts that R
            (pyt_axis_review.py), E's V-P5 script (vl5_e_vs_i.py) and I's V-P5 script (cross_validate.py) read;
  compare   runs the shared comparator compare_verdicts.compare() on every pair of outputs and writes counts and key names
            (never a compared value; the branch values of the test input are recorded under "coverage", see the R3
            paragraph below): E vs I (EI), forecast/E vs forecast/I (FK), R-E vs E (EI) and vs forecast/E (FK), R-I vs I
            (EI) and vs forecast/I (FK), R-E vs R-I (ALL); and three semantic checks of R's own fixtures (no required key
            missing; the flipped-materiality fixture gives category E; the no-l2 fixture gives every l2 key null).
            PASS = these 7 comparisons pass under the artefact rule below, E, I, R-E and R-I read the same Part B file
            (the variant's file), and the R-fixture checks hold (base variant only). Recorded but NOT part of PASS (descriptive): the flags
            of R's own comparisons, of E's V-P5 file (pair level and verdict level, with the artefact class of every
            listed failure location) and of I's cross-validation file; their pair-level parts compare the frozen BLOOMZ
            E-format files, which lack three subsets, with the I-format files (see the artefact rule).

Fixture artefact (class 1), fixed before the first run of this file: the frozen BLOOMZ E-format pair files (5 of the 7 stand-in
sizes; results/S1_2_layermax/pairs/) store no summary of the subsets noLMemb, LMdeep and LMemb_only (checked 2026-10-10
02:2xZ on bloomz-560m x ViT-Ti: 10 subsets, against 13 in the LLaMA-axis E files), while the I-format files do. A
mismatch whose key names one of these three subsets is therefore listed separately as "fixture artefact"; every other
mismatch, every missing key and every key present on one side only fails the test. "strict_pass" (0 mismatches of any
kind) is reported too.
Fixture artefact (class 2), ADDED AFTER run 20261010T022655Z (post hoc, disclosed as such in every record): the key
G|clip_equal_partB when implementation I's value is null. I's fixture mode does not evaluate V-P1 by design
(indep/lmx_pyt_aggregate.py: clip_eq stays None "if not FIXTURE and prep_meta.json exists" fails), because the stand-in
has no I single-model prep; E and R evaluate it on the fixture. A null of I on this key in a production run is not
covered by this class (V-P5 and R then flag it). Lead ruling R2 (2026-10-10 ~03:05Z, under the author's delegation)
accepts this class as a pre-registered fixture exclusion of type 2, disclosed as POST HOC, and requires V-P1 to be
evaluated in the real run. The synthetic H1-positive variants below run I without --fixture (on synthetic I-format
files under its own test output root); there too the stand-in has no I single-model prep (prep_meta.json absent), so
I's value is null for the same reason and the same class applies (no new class).

H1-positive synthetic variants h1A, h1Aprime, h1B (lead ruling R3, 2026-10-10 ~03:05Z; ADDED AFTER run 20261010T024645Z,
before any run with them). The stand-in data alone give no confirmed H1 (base, bandfail and bandwide reached category F,
flipH1 category E), so the section 7 rows A, A-att, A' (K value "A-prime"), B and the modifiers M2, M5_A, M5_B and the
premise were never reached. These variants reach them with a SYNTHETIC stand-in whose calibrated ranks drop with size
while H1's permutation and magnitude tests pass. Construction (stage "synth"; numpy only; imports no E, I or R module):
  source    the same 7 stand-in pair files (5 BLOOMZ + llama-13b + llama-30b, both pre) and the same stand-in features and
            shared stand-in Part B s0b_scratch/PB/partB_vp9/ as variant base. Only the biased-CKA arrays change; mKNN
            and debiased CKA are the frozen originals byte for byte (so H3 and H4 keep the stand-in values).
  transform for each pair (size i, ViT v, pre P) one increasing affine map x -> a + b*x (b > 0) is applied to every
            biased-CKA entry of the I-format file, observed matrix obs_cka_b and all 500 null matrices null_cka_b
            (float64). An increasing map commutes with every maximum, order statistic, argmax and tie, so every subset
            and calibrate-then-max quantity of the pair stays a consistent function of the transformed matrices.
            (a, b) solve two targets on the full layer-max: threshold c_alpha' = c*(i, v) = Bbar_P(i, v) + f_P(i) (Part B
            Bbar of the pair plus the centre f(PR*_i) of band_i), and raw T' = T*(i, v) below.
  targets   with c*_1 the 70m value and lambda_v a per-ViT slope: T*_i = c*_1 + beta_v + lambda_v*(c*_i - c*_1), so the
            raw score moves with the threshold at slope lambda_v. Then s_cal_i = (beta_v - (1 - lambda_v)(c*_i -
            c*_1))/(1 - c*_i); while 0 < s_cal < 1 - lambda_v, s_cal moves AGAINST c* at every size step (Corollary 2:
            the baseline grows faster than the score), and for lambda_v > 1 it moves with c*. Because Bbar factorises,
            the size order of c* is the same for every ViT (prh_clip+l2: 70m above 160m, otherwise increasing; l2:
            1b above 1.4b and 2.8b), so the designed rank patterns are exact. beta_v = midpoint of [LB, UB], LB =
            max_i (1 - lambda_v)(c*_i - c*_1) + 0.02 (every s_cal >= 0.02, nothing censored), UB = min_i [(T - c)_i +
            (1 - lambda_v)(c*_i - c*_1)] (b <= 1, so a ~ c(1 - b) >= 0 and the entries stay in [0, 1]); LB if LB > UB.
  designs   "A":      lambda_v = 0.5 for all 17 ViTs: raw follows c*, s_cal reverses it (prh_clip+l2: design R 0.964,
                      C -0.964, D1 -1.929, p_perm 7/5040; l2: R 0.893, C -0.893, D1 -1.786, p_perm 31/5040);
            "Aatt":   lambda_v = 0.5 for 4 ViTs (one per ViT family: ViT-B/16 in21k, MAE-L, DINOv2-B, CLIP-L laion2b),
                      1.5 for the other 13 (s_cal keeps c*'s order): R 0.964 (0.893), C 0.511 (0.473) > 0.40, D1 -0.454
                      (-0.420), p_perm 7/5040 (31/5040), magnitude draws with no dropping ViT about (13/17)^17 = 0.011;
            "Aprime": lambda_v = +0.5 for the 8 odd-indexed ViTs, -0.5 for the 9 even-indexed (raw reversed, s_cal
                      reversed, no change): R -0.057 (-0.053) < 0.60, C -0.964 (-0.893), D1 -0.908 (-0.840), p_perm as A.
  datasets  dA = (prh_clip+l2 "A", l2 "Aatt"); dP = (prh_clip+l2 "Aprime", l2 "A"). Written once per tag under
            FZ/run_<tag>/synth_<dataset>/ (E/<pre>/ and I/<pre>/) and linked into the variant roots; P-P2a holds by
            construction (median offset = f(PR*_i), the band centre; the c* order equals the Bbar order wherever |Bbar
            gap| >= delta_b because |f gap| < 0.0005), and every decided P-P2b forecast is right (the true threshold lies
            inside each band).
  formats   I-format file: the original npz with obs_cka_b and null_cka_b replaced and key = "S1.2-layermax|synth__pyt-
            <size>|<vit>" (I is run WITHOUT --fixture, reading its own test output root <I out>/<pre>/pairs/). E-format
            file: the original json with lm relabelled, "standin_of", "standin_perm_key" and a "synthetic" record added,
            perm_key set to the synth key, and summaries.cka_biased (every subset the original stores) and
            calib_then_max.cka_biased (if stored) recomputed from the transformed I-format matrices with the hrhv2
            calibrate.summarize arithmetic; the npz with every cka_biased.* array recomputed the same way (null_f32 =
            float32 of the transformed nulls); E's other arrays unchanged. So E and I read one underlying data set. The
            stage self-tests this recomputation on the untransformed originals against the frozen E files (identity
            check) and replays R's E-format consistency checks on every written E file. E's clip cache and stand-in map
            (standins.json, early/cache/clip.json, written by E's own builder) are copied from the base variant root of
            the same tag.
  variants  h1A      = dA; Part B: prh_clip+l2 70m and 160m flagged extrapolated (bands unchanged); l2 unchanged.
                       Intended (s_cal): prh_clip+l2 A, premise verified, M5_A true (70m,160m); l2 A-att, verified, M5_A
                       false.
            h1Aprime = dP; Part B: prh_clip+l2 unchanged; l2 every band [f - 0.2, f + 0.2].
                       Intended: prh_clip+l2 A-prime, verified; l2 A, premise unverified (P-P2b not evaluable).
            h1B      = dA; Part B: bands of 70m, 160m, 410m moved to [f + 0.010, f + 0.011]; prh_clip+l2 flags all three
                       extrapolated, l2 only 70m and 160m. Intended: prh_clip+l2 B with M5_B true; l2 B with M5_B false.
            M2 is reached (non-null) in every A-type slot; with one affine map per pair grid5 keeps the full maximum's
            order, so M2 = true is not expected from this construction. The x = unc slots are not designed.
  compare   as for the other variants (7 comparisons, PASS = agreement under the artefact rule), plus "coverage": the
            section-K category and modifier values of each K writer (forecast E, forecast I, R-E, R-I) for every (P, x),
            and "coverage_as_intended" (every K writer reached the intended s_cal branches). These are values of a
            synthetic test input (no Pythia byte), recorded so that the branches reached are on record; coverage is
            reported, not part of PASS.

Part B variants (ADDED AFTER run 20261010T022655Z, before any run with them): the same pair files run with four Part B
files, so that the agreement of E, I, forecast_pyt.py and R is also checked on branches of the plan section 7 table
that the stand-in data alone do not reach (the stand-in gives category A-att with a verified premise):
  base      the shared stand-in Part B s0b_scratch/PB/partB_vp9/ (partb_pyt.py --standin vp9), unchanged;
  flipH1    material_Bbar false and H1 removed from members (both pre; materiality.Bbar and the delta_b curve of Bbar
            set false too): category E, Holm over the three secondary members without the gate;
  bandfail  the bands of 70m, 160m and 410m moved to [f + 0.010, f + 0.011] and these three flagged extrapolated (both
            pre): P-P2a(i) fails at 3 of 7 models, category B, modifier M5_B;
  bandwide  every band set to [f - 0.2, f + 0.2] (both pre): P-P2b decides few comparisons (premise branch).
A variant file is a test input written by stage "partb" under FZ/run_<tag>/partB_<variant>/ (clip file copied); it is
not a Part B result and is never read outside this fixture. PASS of the fixture = PASS of every variant.

S0b test mode only (PYT_S0B_TEST=1 and no CODE FROZEN line yet; otherwise exit 3, nothing written). Writes only under
analysis/s12_layermax_pythia/s0b_scratch/ (its FZ/, E/X_<tag>_<variant>/, I/X_<tag>_<variant>/, R/X_<tag>_<variant>/,
PB/X_<tag>_<variant>/ sub-roots).
usage: PYT_S0B_TEST=1 venv/bin/python analysis/s12_layermax_pythia/tests_freeze/xfixture_pyt.py
           {partb|synth|inputs|verdicts|compare} --tag <TAG>
           --variant {base|flipH1|bandfail|bandwide|h1A|h1Aprime|h1B}
       PYT_S0B_TEST=1 venv/bin/python .../xfixture_pyt.py synth-check --tag <TAG> --dataset {dA|dP}
           (builds the synthetic data set in memory, runs the identity self-test, the construction and design checks,
           and writes only FZ/check_<tag>/synth_check_<dataset>.json; no pair file)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")
sys.path.insert(0, AXIS)
import guard  # noqa: E402  (shared; test_mode() and SCRATCH)

SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")
STANDIN_OF = {"70m": "bigscience__bloomz-560m", "160m": "bigscience__bloomz-1b1", "410m": "bigscience__bloomz-1b7",
              "1b": "bigscience__bloomz-3b", "1.4b": "bigscience__bloomz-7b1", "2.8b": "huggyllama__llama-13b",
              "6.9b": "huggyllama__llama-30b"}  # VERDICTS_SPEC_PYT.md section 13
PRES = ("prh_clip+l2", "l2")
ARTEFACT_Q = ("noLMemb", "LMdeep", "LMemb_only")
# frozen read-only sources (never written by this file)
SRC_RES8 = os.path.join(PROJ, "results", "S1_2_layermax")
SRC_RESL = os.path.join(PROJ, "results", "S1_2_layermax_llama")
SRC_FEAT8 = os.path.join(PROJ, "features", "wit1024")
SRC_FEATL = os.path.join(PROJ, "features", "wit1024_llama_axis")


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


SYNTH_VARIANTS = ("h1A", "h1Aprime", "h1B")   # lead ruling R3 (docstring)
VARIANTS = ("base", "flipH1", "bandfail", "bandwide") + SYNTH_VARIANTS
SHARED_PB = os.path.join(guard.SCRATCH, "PB", "partB_vp9")
K_PERM, ORDER_IDX = 500, 476   # plan section 3: K = 500, c_alpha = 476th ascending order statistic of 501 values
# ---- R3 synthetic designs (docstring "H1-positive synthetic variants")
DROP4 = ("vit_base_patch16_224.augreg_in21k", "vit_large_patch16_224.mae", "vit_base_patch14_dinov2.lvd142m",
         "vit_large_patch14_clip_224.laion2b")   # design Aatt: one dropping ViT per ViT family of plan section 1
DATASETS = {"dA": {"prh_clip+l2": "A", "l2": "Aatt"}, "dP": {"prh_clip+l2": "Aprime", "l2": "A"}}
SYNTH = {
    "h1A": dict(data="dA", pb={"prh_clip+l2": ("extrap", ("70m", "160m")), "l2": None},
                intended={"prh_clip+l2": dict(category="A", premise="verified", M5_A=True, M5_A_sizes="70m,160m"),
                          "l2": dict(category="A-att", premise="verified", M5_A=False)}),
    "h1Aprime": dict(data="dP", pb={"prh_clip+l2": None, "l2": ("bandwide", None)},
                     intended={"prh_clip+l2": dict(category="A-prime", premise="verified", M5_A=False),
                               "l2": dict(category="A", premise="unverified", M5_A=False)}),
    "h1B": dict(data="dA", pb={"prh_clip+l2": ("bandfail", ("70m", "160m", "410m")), "l2": ("bandfail", ("70m", "160m"))},
                intended={"prh_clip+l2": dict(category="B", M5_B=True), "l2": dict(category="B", M5_B=False)}),
}
A_TYPES = ("A", "A-att", "A-prime")
SYNTH_STD_DDOF = 1   # null_sd / entry_null_sd of the frozen E files use ddof 1 (identity self-test of 2026-10-10T03:32Z:
                     # ddof 1 matches to 1e-13, ddof 0 differs by 1e-3; the check is repeated in every build)


def design_lambda(design: str, vi: int, vit: str) -> float:
    if design == "A":
        return 0.5
    if design == "Aatt":
        return 0.5 if vit in DROP4 else 1.5
    if design == "Aprime":
        return 0.5 if vi % 2 == 1 else -0.5
    raise ValueError(design)


def roots(tag: str, variant: str = "base") -> dict:
    sc = guard.SCRATCH
    x = f"X_{tag}_{variant}"
    xr = os.path.join(sc, "R", x)
    fz = os.path.join(sc, "FZ", f"run_{tag}")
    return dict(FZ=fz, XE=os.path.join(sc, "E", x), XI=os.path.join(sc, "I", x), XF=os.path.join(sc, "PB", x), XR=xr,
                RES=os.path.join(xr, "results", "S1_2_layermax_pythia"),
                PB=SHARED_PB if variant == "base" else os.path.join(fz, f"partB_{variant}"), V=variant)


def stage_partb(tag: str, variant: str) -> int:
    R = roots(tag, variant)
    if variant == "base":
        print("base variant: the shared stand-in Part B is used unchanged")
        return 0
    F = json.load(open(os.path.join(SHARED_PB, "partB_features_pyt.json")))
    changed = []
    for pre in PRES:
        P = F["per_pre"][pre]
        if variant in SYNTH:   # R3 variants: one change per pre (docstring), test input only
            ch = SYNTH[variant]["pb"][pre]
            if ch is None:
                changed.append(f"{pre}: unchanged")
            elif ch[0] == "extrap":
                for sz in ch[1]:
                    P["extrapolated"][sz] = True
                changed.append(f"{pre}: {', '.join(ch[1])} flagged extrapolated (bands unchanged)")
            elif ch[0] == "bandwide":
                for sz in SIZES:
                    f0 = P["band"][sz]["f"]
                    P["band"][sz]["lo"], P["band"][sz]["hi"] = f0 - 0.2, f0 + 0.2
                changed.append(f"{pre}: every band [f-0.2, f+0.2]")
            elif ch[0] == "bandfail":
                for sz in ("70m", "160m", "410m"):
                    f0 = P["band"][sz]["f"]
                    P["band"][sz]["lo"], P["band"][sz]["hi"] = f0 + 0.010, f0 + 0.011
                    P["extrapolated"][sz] = sz in ch[1]
                changed.append(f"{pre}: bands of 70m, 160m, 410m moved to [f+0.010, f+0.011]; flagged extrapolated: "
                               f"{', '.join(ch[1])}")
            else:
                raise SystemExit(f"unknown Part B change {ch}")
            continue
        if variant == "flipH1":
            P["material_Bbar"] = False
            P["members"] = [m for m in P["members"] if m != "H1"]
            if isinstance(P.get("materiality", {}).get("Bbar"), dict):
                P["materiality"]["Bbar"]["material"] = False
            P["material_curve_db"]["Bbar"] = {k: False for k in P["material_curve_db"]["Bbar"]}
            changed.append(f"{pre}: material_Bbar false, members {P['members']}")
        elif variant == "bandfail":
            for sz in ("70m", "160m", "410m"):
                f0 = P["band"][sz]["f"]
                P["band"][sz]["lo"], P["band"][sz]["hi"] = f0 + 0.010, f0 + 0.011
                P["extrapolated"][sz] = True
            changed.append(f"{pre}: bands of 70m, 160m, 410m moved to [f+0.010, f+0.011], flagged extrapolated")
        elif variant == "bandwide":
            for sz in SIZES:
                f0 = P["band"][sz]["f"]
                P["band"][sz]["lo"], P["band"][sz]["hi"] = f0 - 0.2, f0 + 0.2
            changed.append(f"{pre}: every band [f-0.2, f+0.2]")
        else:
            raise SystemExit(f"unknown variant {variant}")
    F["_fixture_variant"] = dict(variant=variant, source_sha256=sha(os.path.join(SHARED_PB, "partB_features_pyt.json")),
                                 changes=changed, note="S0b TEST INPUT of fixture X; not a Part B result",
                                 synthetic_data=SYNTH[variant]["data"] if variant in SYNTH else None)
    out = os.path.join(R["PB"], "partB_features_pyt.json")
    if os.path.lexists(out):
        raise SystemExit(f"{out} exists (no-delete rule; use a new tag)")
    os.makedirs(R["PB"], exist_ok=True)
    with open(out, "w") as f:
        json.dump(F, f, indent=1)
    copy_new(os.path.join(SHARED_PB, "clip_thresholds_partB_pyt.json"), os.path.join(R["PB"], "clip_thresholds_partB_pyt.json"))
    print(f"variant {variant}: {out} sha256 {sha(out)}; " + "; ".join(changed))
    return 0


def i_src(pre: str, src: str, vit: str) -> str:
    base = SRC_RESL if src.startswith("huggyllama") else SRC_RES8
    return os.path.join(base, "indep", pre, "pairs", f"{src}__{vit}.npz")


def e_src(pre: str, src: str, vit: str) -> str:
    if src.startswith("huggyllama"):
        return os.path.join(SRC_RESL, "early", "pairs", pre, f"{src}__{vit}")
    return os.path.join(SRC_RES8, "pairs", pre, f"{src}__{vit}")


def feat_src(src: str) -> str:
    base = SRC_FEATL if src.startswith("huggyllama") else SRC_FEAT8
    return os.path.join(base, src + ".npy")


def inside_scratch(p: str) -> bool:
    r = os.path.realpath(guard.SCRATCH)
    q = os.path.realpath(os.path.dirname(p)) if os.path.islink(p) else os.path.realpath(p)
    return q == r or q.startswith(r + os.sep)


def link(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise SystemExit(f"source missing: {src}")
    if not inside_scratch(dst):
        raise SystemExit(f"refused: {dst} is outside {guard.SCRATCH}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):  # no-delete rule: an identical link is kept, anything else stops
        if os.path.realpath(dst) != os.path.realpath(src):
            raise SystemExit(f"{dst} exists and points elsewhere (no-delete rule; use a new tag)")
        return
    os.symlink(src, dst)


def copy_new(src: str, dst: str) -> str:
    if not inside_scratch(dst):
        raise SystemExit(f"refused: {dst} is outside {guard.SCRATCH}")
    if os.path.lexists(dst):
        raise SystemExit(f"{dst} exists (no-delete rule; use a new tag)")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return sha(dst)


def write_report(path: str, doc: dict) -> str:
    if os.path.lexists(path):
        raise SystemExit(f"{path} exists (no-delete rule; use a new tag)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    return sha(path)


# ------------------------------------------------------------------------------------------------------------------
# R3: the H1-positive synthetic stand-in (docstring). Test-input construction only: no verdict of E, I, R or
# forecast_pyt.py is computed here; the "design check" restates the designed rank patterns of the input.
# ------------------------------------------------------------------------------------------------------------------
def _summ(T: float, nmx) -> tuple:
    """hrhv2 calibrate.summarize arithmetic on float64 layer-max nulls (the arithmetic of the frozen E summaries, which
    R's float_from_nullmax restates): (c_alpha, null mean, s_cal, uncensored, p)."""
    null = np.ascontiguousarray(np.asarray(nmx, dtype=np.float64))
    comb = np.sort(np.concatenate(([float(T)], null)))
    c = float(comb[ORDER_IDX - 1])
    den = 1.0 - c
    scal = max((T - c) / den, 0.0) if den > 0 else 0.0
    mu = float(null.mean())
    return c, mu, float(scal), float(T - mu), float((1.0 + np.sum(null >= T)) / (null.size + 1))


def _ctm_entry(obs, ent_c, ent_mu) -> dict:
    """calibrate-then-max from the entry arrays (first maximum in row-major order), the json fields of calib_then_max."""
    LA, LB = obs.shape
    den = 1.0 - ent_c
    scal = np.where(den > 0, np.maximum((obs - ent_c) / np.where(den > 0, den, 1.0), 0.0), 0.0)
    unc = obs - ent_mu
    j = int(np.argmax(scal.reshape(-1)))
    ju = int(np.argmax(unc.reshape(-1)))
    a, b = divmod(j, LB)
    au, bu = divmod(ju, LB)
    M = float(scal.reshape(-1)[j])
    return {"M": M, "argmax_A": int(a), "argmax_B": int(b), "n_tied_max": int(np.sum(scal == M)),
            "n_entries_scal_pos": int(np.sum(scal > 0)), "n_entries": int(LA * LB), "s_obs_at_argmax": float(obs[a, b]),
            "c_alpha_at_argmax": float(ent_c[a, b]), "M_unc": float(unc.reshape(-1)[ju]), "M_unc_argmax_A": int(au),
            "M_unc_argmax_B": int(bu)}


def e_cka_from_matrices(obs, null, jsum: dict, ctm0: dict | None, ddof: int = SYNTH_STD_DDOF):
    """E-format biased-CKA content recomputed from float64 observed (LA, LB) and null (K, LA, LB) matrices: the json
    summaries of exactly the subsets in jsum (layers read from jsum), the npz arrays cka_biased.*, and calib_then_max
    (only if ctm0, the original record, exists). Field order and the fields that do not depend on the data (K, alpha,
    n_layer_pairs, layers_A, layers_B) are kept from jsum."""
    obs = np.asarray(obs, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    K_, LA, LB = null.shape
    ent_c = np.sort(np.concatenate([obs[None], null], axis=0), axis=0)[ORDER_IDX - 1]
    ent_mu = null.mean(axis=0)
    ent_sd = null.std(axis=0, ddof=ddof)
    ent_p = (1.0 + np.sum(null >= obs[None], axis=0)) / (K_ + 1)
    flat = null.reshape(K_, -1)
    fmax = flat.max(axis=1)
    farg = flat.argmax(axis=1)
    arrays = {"entry_obs": obs, "entry_null_mean": ent_mu, "entry_c_alpha": ent_c, "entry_p": ent_p,
              "entry_null_sd": ent_sd, "full_null_max": fmax, "full_null_argmax_A": (farg // LB).astype(np.int16),
              "full_null_argmax_B": (farg % LB).astype(np.int16), "null_f32": null.astype(np.float32)}
    summ = {}
    for q, s0 in jsum.items():
        ia, ib = [int(x) for x in s0["layers_A"]], [int(x) for x in s0["layers_B"]]
        sub = obs[np.ix_(ia, ib)]
        T = float(sub.max())
        j = int(np.argmax(sub.reshape(-1)))
        aA, aB = ia[j // len(ib)], ib[j % len(ib)]
        nmx = fmax if (ia == list(range(LA)) and ib == list(range(LB))) else \
            null[:, ia][:, :, ib].reshape(K_, -1).max(axis=1)
        c, mu, scal, unc, p = _summ(T, nmx)
        s = dict(s0)
        s.update(s_obs=T, c_alpha=c, s_cal=scal, p_value=p, null_mean=mu, null_sd=float(np.std(nmx, ddof=ddof)),
                 uncensored=unc, argmax_A=int(aA), argmax_B=int(aB), n_tied_max=int(np.sum(sub == T)),
                 entry_c_alpha_at_argmax=float(ent_c[aA, aB]), entry_null_mean_at_argmax=float(ent_mu[aA, aB]),
                 entry_p_at_argmax=float(ent_p[aA, aB]))
        summ[q] = s
    ctm = None
    if ctm0 is not None:
        ctm = dict(ctm0)
        ctm.update(_ctm_entry(obs, ent_c, ent_mu))
    return summ, arrays, ctm


def _rel(x, y) -> float:
    x, y = np.atleast_1d(np.asarray(x, dtype=np.float64)), np.atleast_1d(np.asarray(y, dtype=np.float64))
    if x.shape != y.shape:
        return float("inf")
    if x.size == 0:
        return 0.0
    d = np.abs(x - y) / np.maximum(np.maximum(np.abs(x), np.abs(y)), 1e-300)
    d[(x == y)] = 0.0
    return float(d.max())


def identity_diff(summ, arrays, ctm, jsum0, z0, ctm0, acc: dict) -> None:
    """Accumulate the differences between the recomputation from the ORIGINAL I-format matrices and the frozen E file."""
    def up(k, v):
        acc[k] = max(acc.get(k, 0.0), v)

    def cnt(k, v):
        acc[k] = acc.get(k, 0) + int(v)
    for q, s in summ.items():
        s0 = jsum0[q]
        for f in ("s_obs", "c_alpha", "s_cal", "p_value", "null_mean", "uncensored", "entry_c_alpha_at_argmax",
                  "entry_null_mean_at_argmax", "entry_p_at_argmax", "null_sd"):
            up(f"rel|{f}", _rel(s[f], s0[f]))
        for f in ("argmax_A", "argmax_B", "n_tied_max"):
            cnt(f"mismatch|{f}", s[f] != s0[f])
        cnt("n_summaries", 1)
    for f in ("entry_obs", "entry_null_mean", "entry_c_alpha", "entry_p", "entry_null_sd", "full_null_max"):
        up(f"rel|npz|{f}", _rel(arrays[f], z0[f"cka_biased.{f}"]))
    for f in ("full_null_argmax_A", "full_null_argmax_B"):
        cnt(f"mismatch|npz|{f}", np.sum(arrays[f] != z0[f"cka_biased.{f}"]))
    cnt("mismatch|npz|null_f32_elements", np.sum(arrays["null_f32"] != z0["cka_biased.null_f32"]))
    if ctm0 is not None:
        for f in ("M", "M_unc", "s_obs_at_argmax", "c_alpha_at_argmax"):
            up(f"rel|ctm|{f}", _rel(ctm[f], ctm0[f]))
        for f in ("argmax_A", "argmax_B", "n_tied_max", "n_entries_scal_pos", "n_entries", "M_unc_argmax_A",
                  "M_unc_argmax_B"):
            cnt(f"mismatch|ctm|{f}", ctm[f] != ctm0[f])
        cnt("n_ctm", 1)


def r_style_checks(summ, arrays, ctm) -> dict:
    """Replays, on one written E-format biased-CKA record, the consistency checks of R's E-format reader
    (review/pyt_review_lib.py read_pair_E: T and argmax from the entry array, the full layer-max recomputed from
    full_null_max, the subsets from their summary, the float32-null checks, calibrate-then-max from the entry arrays).
    Restated here so that a defect of the synthetic input is caught before any implementation reads it."""
    obs = arrays["entry_obs"]
    f32 = arrays["null_f32"]
    K_ = f32.shape[0]
    n = dict(T=0, argmax=0, float_recomp=0, c_f32=0, p_f32=0, mu_f32_rel_gt_1e6=0, ctm=0)
    for q, s in summ.items():
        ia, ib = [int(x) for x in s["layers_A"]], [int(x) for x in s["layers_B"]]
        sub = obs[np.ix_(ia, ib)]
        T = float(sub.max())
        j = int(np.argmax(sub.reshape(-1)))
        n["T"] += T != s["s_obs"]
        n["argmax"] += (ia[j // len(ib)], ib[j % len(ib)]) != (s["argmax_A"], s["argmax_B"])
        if q == "full":
            c, mu, scal, unc, p = _summ(T, arrays["full_null_max"])
        else:
            c, mu, p = s["c_alpha"], s["null_mean"], s["p_value"]
            den = 1.0 - c
            scal, unc = (max((T - c) / den, 0.0) if den > 0 else 0.0), float(T - mu)
            nmx = f32[:, ia][:, :, ib].reshape(K_, -1).max(axis=1)
            comb = np.sort(np.concatenate(([np.float32(T)], nmx)))
            n["c_f32"] += comb[ORDER_IDX - 1] != np.float32(s["c_alpha"])
            n["mu_f32_rel_gt_1e6"] += abs(float(np.mean(nmx.astype(np.float64))) - mu) / max(abs(mu), 1e-300) > 1e-6
            n["p_f32"] += float((1.0 + np.sum(nmx >= np.float32(T))) / (nmx.size + 1)) != p
        n["float_recomp"] += (c != s["c_alpha"] or scal != s["s_cal"] or unc != s["uncensored"] or p != s["p_value"]
                              or mu != s["null_mean"])
    if ctm is not None:
        e = _ctm_entry(obs, arrays["entry_c_alpha"], arrays["entry_null_mean"])
        n["ctm"] += any(ctm[k] != e[k] for k in ("M", "M_unc", "s_obs_at_argmax", "c_alpha_at_argmax", "argmax_A",
                                                   "argmax_B"))
    return {k: int(v) for k, v in n.items()}


def _avg_ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def design_check(raw: np.ndarray, cal: np.ndarray, s: int) -> dict:
    """Restates the designed rank patterns of the synthetic input (17 x 7 raw and s_cal scores of cka_biased, full):
    R, C, D1 = s(C - R) and the exact relabelling count of D1 over the 5040 size relabellings (plan section 5.4
    formula). A check of the test input, never compared with any implementation."""
    import itertools
    sig = np.array(list(itertools.permutations(range(1, 8))), dtype=np.float64)
    sig -= sig.mean(axis=1, keepdims=True)

    def rho_all(X):   # (17, 7) scores -> (5040, 17) Spearman of every relabelling
        Rk = np.array([_avg_ranks(r) for r in X])
        Rk -= Rk.mean(axis=1, keepdims=True)
        return (sig @ Rk.T) / np.sqrt((sig ** 2).sum(axis=1))[:, None] / np.sqrt((Rk ** 2).sum(axis=1))[None, :]
    rr, rc = rho_all(raw), rho_all(cal)
    D = s * (rc - rr).mean(axis=1)
    return dict(R=float(rr[0].mean()), C=float(rc[0].mean()), D1=float(D[0]),
                p_perm_count=int(np.sum(D <= D[0] + 1e-9)), p_perm=float(np.sum(D <= D[0] + 1e-9) / 5040),
                n_vits_dropping=int(np.sum((rc[0] - rr[0]) * s < -1e-12)))


def synth_root(tag: str, ds: str) -> str:
    return os.path.join(guard.SCRATCH, "FZ", f"run_{tag}", f"synth_{ds}")


def build_dataset(tag: str, ds: str, write: bool) -> dict:
    """Builds data set ds (DATASETS) from the frozen stand-in pair files and the shared stand-in Part B. write=False:
    in memory only (synth-check). Returns the manifest (design, per-pair a and b, checks, hashes)."""
    import verdict_keys as VK
    pbf = os.path.join(SHARED_PB, "partB_features_pyt.json")
    F = json.load(open(pbf))
    out = synth_root(tag, ds)
    if write:
        if os.path.lexists(out):
            raise SystemExit(f"{out} exists (no-delete rule; use a new tag)")
        if not inside_scratch(out):
            raise SystemExit(f"refused: {out} is outside {guard.SCRATCH}")
    man = dict(what="R3 H1-positive synthetic stand-in data set (S0b TEST INPUT of fixture X; not Pythia, not citable)",
               utc=utc(), tag=tag, dataset=ds, designs=DATASETS[ds], partB=os.path.relpath(pbf, PROJ),
               partB_sha256=sha(pbf), lambda_={}, drop4=list(DROP4), pairs={}, identity={}, r_style={},
               construction={}, design_check={}, n_files_written=0)
    for pre in PRES:
        design = DATASETS[ds][pre]
        P = F["per_pre"][pre]
        s_sign = int(P["s"])
        ident, rsty = {}, {}
        cons = dict(n_pairs=0, max_abs_c_minus_target=0.0, max_abs_T_minus_target=0.0, min_b=float("inf"),
                    max_b=0.0, min_entry=float("inf"), max_entry=-float("inf"), min_scal=float("inf"),
                    n_beta_lb_gt_ub=0, n_rank_pattern_raw_bad=0, n_rank_pattern_cal_bad=0)
        RAW, CAL = np.zeros((len(VK.VITS), 7)), np.zeros((len(VK.VITS), 7))
        man["lambda_"][pre] = {}
        for vi, v in enumerate(VK.VITS):
            lam = design_lambda(design, vi, v)
            man["lambda_"][pre][v] = lam
            cst = np.array([P["Bbar"][f"synth__pyt-{sz}"][v] + P["band"][sz]["f"] for sz in SIZES])
            src = {sz: STANDIN_OF[sz] for sz in SIZES}
            I0 = {sz: np.load(i_src(pre, src[sz], v)) for sz in SIZES}
            Tc = {}
            for sz in SIZES:
                o = np.asarray(I0[sz]["obs_cka_b"], dtype=np.float64)
                nl = np.asarray(I0[sz]["null_cka_b"], dtype=np.float64)
                T0 = float(o.max())
                c0 = _summ(T0, nl.reshape(nl.shape[0], -1).max(axis=1))[0]
                Tc[sz] = (T0, c0)
            g = cst - cst[0]
            lb = float(np.max((1.0 - lam) * g)) + 0.02
            ub = float(np.min([Tc[sz][0] - Tc[sz][1] + (1.0 - lam) * g[k] for k, sz in enumerate(SIZES)]))
            beta = 0.5 * (lb + ub) if lb <= ub else lb
            cons["n_beta_lb_gt_ub"] += lb > ub
            Tst = cst[0] + beta + lam * g
            for k, sz in enumerate(SIZES):
                slug = f"synth__pyt-{sz}"
                T0, c0 = Tc[sz]
                b = (Tst[k] - cst[k]) / (T0 - c0)
                a = cst[k] - b * c0
                if not (b > 0):
                    raise SystemExit(f"{pre}|{sz}|{v}: b = {b} not > 0 (original s_cal not > 0)")
                z_i = I0[sz]
                obs2 = a + b * np.asarray(z_i["obs_cka_b"], dtype=np.float64)
                null2 = a + b * np.asarray(z_i["null_cka_b"], dtype=np.float64)
                eb = e_src(pre, src[sz], v)
                rec = json.load(open(eb + ".json"))
                z_e = np.load(eb + ".npz")
                assert rec["lm"] == src[sz] and rec["vit"] == v and int(rec["L_A"]) == obs2.shape[0], (eb, rec["lm"])
                jsum0 = rec["summaries"]["cka_biased"]
                ctm0 = (rec.get("calib_then_max") or {}).get("cka_biased")
                # identity self-test: the recomputation from the ORIGINAL I-format matrices against the frozen E file
                s_id, a_id, c_id = e_cka_from_matrices(z_i["obs_cka_b"], z_i["null_cka_b"], jsum0, ctm0)
                identity_diff(s_id, a_id, c_id, jsum0, z_e, ctm0, ident)
                for dd in (0, 1):
                    nm0 = np.asarray(z_i["null_cka_b"], dtype=np.float64)
                    ident[f"rel|npz|entry_null_sd|ddof{dd}"] = max(ident.get(f"rel|npz|entry_null_sd|ddof{dd}", 0.0),
                                                                   _rel(nm0.std(axis=0, ddof=dd),
                                                                        z_e["cka_biased.entry_null_sd"]))
                # the synthetic record
                summ, arr, ctm = e_cka_from_matrices(obs2, null2, jsum0, ctm0)
                for kk, vv in r_style_checks(summ, arr, ctm).items():
                    rsty[kk] = rsty.get(kk, 0) + vv
                Tn, cn = summ["full"]["s_obs"], summ["full"]["c_alpha"]
                RAW[vi, k], CAL[vi, k] = Tn, summ["full"]["s_cal"]
                cons["n_pairs"] += 1
                cons["max_abs_c_minus_target"] = max(cons["max_abs_c_minus_target"], abs(cn - cst[k]))
                cons["max_abs_T_minus_target"] = max(cons["max_abs_T_minus_target"], abs(Tn - Tst[k]))
                cons["min_b"], cons["max_b"] = min(cons["min_b"], b), max(cons["max_b"], b)
                cons["min_entry"] = min(cons["min_entry"], float(obs2.min()), float(null2.min()))
                cons["max_entry"] = max(cons["max_entry"], float(obs2.max()), float(null2.max()))
                cons["min_scal"] = min(cons["min_scal"], summ["full"]["s_cal"])
                pk = f"{pre}|{slug}|{v}"
                man["pairs"][pk] = dict(standin_of=src[sz], a=a, b=b, lam=lam, beta=beta, c_target=float(cst[k]),
                                        T_target=float(Tst[k]), src_E_json_sha256=sha(eb + ".json"),
                                        src_E_npz_sha256=sha(eb + ".npz"),
                                        src_I_npz_sha256=sha(i_src(pre, src[sz], v)))
                if write:
                    key = f"S1.2-layermax|{slug}|{v}"
                    rec2 = dict(rec)
                    rec2["lm"], rec2["perm_key"] = slug, key
                    rec2["standin_of"], rec2["standin_perm_key"] = src[sz], rec.get("perm_key")
                    rec2["summaries"] = dict(rec["summaries"])
                    rec2["summaries"]["cka_biased"] = summ
                    if ctm0 is not None:
                        rec2["calib_then_max"] = dict(rec["calib_then_max"])
                        rec2["calib_then_max"]["cka_biased"] = ctm
                    rec2["synthetic"] = dict(
                        what="S0b TEST INPUT (fixture X, lead ruling R3): biased CKA of the stand-in transformed by "
                             "x -> a + b x (tests_freeze/xfixture_pyt.py); mKNN and debiased CKA unchanged",
                        dataset=ds, design=design, a=a, b=b, lam=lam, beta=beta, c_target=float(cst[k]),
                        T_target=float(Tst[k]))
                    ej = os.path.join(out, "E", pre, f"{slug}__{v}.json")
                    os.makedirs(os.path.dirname(ej), exist_ok=True)
                    with open(ej, "w") as f:
                        json.dump(rec2, f)
                    earr = {kk: z_e[kk] for kk in z_e.files}
                    for kk, vv in arr.items():
                        assert kk in ("entry_obs", "entry_null_mean", "entry_c_alpha", "entry_p", "entry_null_sd",
                                      "full_null_max", "full_null_argmax_A", "full_null_argmax_B", "null_f32")
                        assert f"cka_biased.{kk}" in earr and earr[f"cka_biased.{kk}"].shape == vv.shape, (eb, kk)
                        earr[f"cka_biased.{kk}"] = vv.astype(earr[f"cka_biased.{kk}"].dtype)
                    np.savez(os.path.join(out, "E", pre, f"{slug}__{v}.npz"), **earr)
                    iarr = {kk: z_i[kk] for kk in z_i.files}
                    iarr["obs_cka_b"], iarr["null_cka_b"] = obs2, null2
                    iarr["key"] = np.array(key)
                    ip = os.path.join(out, "I", pre, f"{slug}__{v}.npz")
                    os.makedirs(os.path.dirname(ip), exist_ok=True)
                    np.savez(ip, **iarr)
                    man["pairs"][pk].update(E_json_sha256=sha(ej), E_npz_sha256=sha(ej[:-5] + ".npz"),
                                            I_npz_sha256=sha(ip))
                    man["n_files_written"] += 3
            # designed rank patterns: raw follows c* (lam > 0) or reverses it; s_cal reverses c* (lam < 1) or follows it
            rc = _avg_ranks(cst)
            want_raw = rc if lam > 0 else 8.0 - rc
            want_cal = 8.0 - rc if lam < 1 else rc
            cons["n_rank_pattern_raw_bad"] += not np.array_equal(_avg_ranks(RAW[vi]), want_raw)
            cons["n_rank_pattern_cal_bad"] += not np.array_equal(_avg_ranks(CAL[vi]), want_cal)
        man["identity"][pre] = ident
        man["r_style"][pre] = rsty
        man["construction"][pre] = cons
        dc = design_check(RAW, CAL, s_sign)
        dc.update(s=s_sign, member_H1=bool(P["material_Bbar"]), design=design,
                  c_star_size_order=[SIZES[i] for i in np.argsort([P["Bbar"][f"synth__pyt-{sz}"][VK.VITS[0]]
                                                                  + P["band"][sz]["f"] for sz in SIZES])])
        man["design_check"][pre] = dc
    # pass flags of the construction (the identity self-test is reported with its numbers; see synth_ok)
    ok = True
    for pre in PRES:
        c, r = man["construction"][pre], man["r_style"][pre]
        ok &= (c["max_abs_c_minus_target"] <= 1e-12 and c["min_b"] > 0 and c["min_entry"] >= 0.0
               and c["max_entry"] <= 1.0 and c["min_scal"] > 0 and c["n_rank_pattern_raw_bad"] == 0
               and c["n_rank_pattern_cal_bad"] == 0 and all(v == 0 for v in r.values()))
        dc, des = man["design_check"][pre], DATASETS[ds][pre]
        ok &= dc["member_H1"] and dc["D1"] <= -0.10 and dc["p_perm"] <= 0.05
        ok &= {"A": dc["R"] >= 0.60 and dc["C"] <= 0.40, "Aatt": dc["R"] >= 0.60 and dc["C"] > 0.40,
               "Aprime": dc["R"] < 0.60}[des]
    man["identity_max_rel"] = {pre: max([v for k, v in man["identity"][pre].items()
                                         if k.startswith("rel|") and "ddof" not in k] or [0.0]) for pre in PRES}
    man["identity_int_mismatches"] = {pre: sum(v for k, v in man["identity"][pre].items() if k.startswith("mismatch|")
                                               and "null_f32" not in k) for pre in PRES}
    # identity self-test: the E-format recomputation reproduces the frozen E files from the frozen I files (E and I of
    # the known models agree to about 1e-12, so 1e-10 relative; integer fields exactly; float32 nulls are only counted)
    man["identity_ok"] = bool(all(man["identity_max_rel"][p] <= 1e-10 and man["identity_int_mismatches"][p] == 0
                                  for p in PRES))
    man["construction_ok"] = bool(ok and man["identity_ok"])
    return man


def stage_synth(tag: str, variant: str) -> int:
    """R3 variants: build the variant's data set once per tag (or verify the one built by an earlier variant of the tag)
    and lay out the variant's E and I roots: synthetic pair files linked, E's stand-in map and clip cache copied from
    the base variant root of the same tag (written there by E's own builder)."""
    import verdict_keys as VK
    if variant not in SYNTH:
        raise SystemExit(f"synth: {variant} is not a synthetic variant")
    ds = SYNTH[variant]["data"]
    R = roots(tag, variant)
    sr = synth_root(tag, ds)
    mp = os.path.join(sr, "synth_manifest.json")
    if os.path.exists(mp):
        man = json.load(open(mp))
        bad = [k for k, d in man["pairs"].items() for kind, path in (
            ("E_json_sha256", os.path.join(sr, "E", k.split("|")[0], f"{k.split('|')[1]}__{k.split('|')[2]}.json")),
            ("E_npz_sha256", os.path.join(sr, "E", k.split("|")[0], f"{k.split('|')[1]}__{k.split('|')[2]}.npz")),
            ("I_npz_sha256", os.path.join(sr, "I", k.split("|")[0], f"{k.split('|')[1]}__{k.split('|')[2]}.npz")))
            if sha(path) != d[kind]]
        if bad:
            raise SystemExit(f"data set {ds} of tag {tag} changed since it was built: {bad[:5]}")
        print(f"data set {ds} reused (built by an earlier variant of tag {tag}; {len(man['pairs'])} pairs re-hashed)")
    else:
        man = build_dataset(tag, ds, write=True)
        write_report(mp, man)
        print(f"data set {ds} built: {man['n_files_written']} files; construction_ok {man['construction_ok']}; "
              f"identity max rel {man['identity_max_rel']}, int mismatches {man['identity_int_mismatches']}; "
              f"design {json.dumps({p: {k: man['design_check'][p][k] for k in ('R', 'C', 'D1', 'p_perm')} for p in PRES})}")
    base = roots(tag, "base")["XE"]
    for fn in ("standins.json", os.path.join("early", "cache", "clip.json")):
        if not os.path.isfile(os.path.join(base, fn)):
            raise SystemExit(f"{os.path.join(base, fn)} missing: run variant base of tag {tag} first (E's builder)")
        copy_new(os.path.join(base, fn), os.path.join(R["XE"], fn))
    n = 0
    for pre in PRES:
        for s in SIZES:
            slug = f"synth__pyt-{s}"
            for v in VK.VITS:
                for ext in (".json", ".npz"):
                    link(os.path.join(sr, "E", pre, f"{slug}__{v}{ext}"),
                         os.path.join(R["XE"], "early", "pairs", pre, f"{slug}__{v}{ext}"))
                    n += 1
                link(os.path.join(sr, "I", pre, f"{slug}__{v}.npz"),
                     os.path.join(R["XI"], "indep", pre, "pairs", f"{slug}__{v}.npz"))
                n += 1
    print(f"variant {variant}: data set {ds}, {n} links; construction_ok {man['construction_ok']}")
    return 0 if man["construction_ok"] else 1


def stage_synth_check(tag: str, ds: str) -> int:
    man = build_dataset(tag, ds, write=False)
    p = os.path.join(guard.SCRATCH, "FZ", f"check_{tag}", f"synth_check_{ds}.json")
    h = write_report(p, man)
    for pre in PRES:
        print(f"[{ds} {pre} {DATASETS[ds][pre]}] construction {json.dumps(man['construction'][pre])}")
        print(f"[{ds} {pre}] R-style E checks {json.dumps(man['r_style'][pre])}")
        print(f"[{ds} {pre}] design check {json.dumps(man['design_check'][pre])}")
        print(f"[{ds} {pre}] identity {json.dumps(man['identity'][pre])}")
    print(f"synth-check {ds}: {p} sha256 {h}; construction_ok {man['construction_ok']}")
    return 0 if man["construction_ok"] else 1


def stage_inputs(tag: str, variant: str) -> int:
    import verdict_keys as VK
    R = roots(tag, variant)
    n_links, bad = 0, []
    if variant in SYNTH:
        return stage_inputs_synth(tag, variant)
    for pre in PRES:
        for s in SIZES:
            slug, src = f"synth__pyt-{s}", STANDIN_OF[s]
            for v in VK.VITS:
                eb = os.path.join(R["XE"], "early", "pairs", pre, f"{slug}__{v}")
                # E's copies against the frozen originals: json equal except lm / standin_of; npz byte-identical
                a = json.load(open(eb + ".json"))
                b = json.load(open(e_src(pre, src, v) + ".json"))
                if not (a.pop("lm", None) == slug and a.pop("standin_of", None) == src and b.pop("lm", None) == src
                        and a == b):
                    bad.append(f"{pre}|{slug}|{v}|json")
                if sha(eb + ".npz") != sha(e_src(pre, src, v) + ".npz"):
                    bad.append(f"{pre}|{slug}|{v}|npz")
                for ext in (".json", ".npz"):
                    link(eb + ext, os.path.join(R["RES"], "early", "pairs", pre, f"{slug}__{v}{ext}"))
                    n_links += 1
                isrc = i_src(pre, src, v)
                link(isrc, os.path.join(R["RES"], "indep", pre, "pairs", f"{slug}__{v}.npz"))
                link(isrc, os.path.join(R["XE"], "indep", pre, "pairs", f"{slug}__{v}.npz"))
                # I's cross_validate.py reads I's pair files under its own output root <I out>/<pre>/pairs/
                # (added after run 20261010T022655Z, where this link was missing: FileNotFoundError, harness bug)
                link(isrc, os.path.join(R["XI"], "indep", pre, "pairs", f"{slug}__{v}.npz"))
                n_links += 3
    for s in SIZES:
        link(feat_src(STANDIN_OF[s]), os.path.join(R["XR"], "features", f"synth__pyt-{s}.npy"))
        n_links += 1
    for fn in ("partB_features_pyt.json", "clip_thresholds_partB_pyt.json"):
        link(os.path.join(R["PB"], fn), os.path.join(R["RES"], "partB", fn))
        n_links += 1
    doc = dict(what="fixture X inputs (links only)", utc=utc(), tag=tag, variant=variant, n_links=n_links,
               partB_sha256=sha(os.path.join(R["PB"], "partB_features_pyt.json")),
               clip_sha256=sha(os.path.join(R["PB"], "clip_thresholds_partB_pyt.json")),
               E_copies_equal_frozen_originals=not bad, n_E_copy_differences=len(bad), E_copy_differences=bad[:50])
    write_report(os.path.join(R["FZ"], f"xfixture_inputs_{variant}.json"), doc)
    print(f"fixture X inputs: {n_links} links; E copies equal frozen originals: {not bad} ({len(bad)} differences)")
    return 0 if not bad else 1


def stage_inputs_synth(tag: str, variant: str) -> int:
    """R3 variants: the R and E-V-P5 layouts link the synthetic files (I's own root already links them, stage synth);
    checks that every synthetic file equals its frozen original outside the biased-CKA content (E json: all fields but
    lm, perm_key, standin_of, standin_perm_key, synthetic, summaries.cka_biased, calib_then_max.cka_biased; E npz:
    every array not named cka_biased.*, byte equal with its dtype; I npz: every array but obs_cka_b, null_cka_b, key)."""
    import verdict_keys as VK
    R = roots(tag, variant)
    ds = SYNTH[variant]["data"]
    sr = synth_root(tag, ds)
    n_links, bad = 0, []
    for pre in PRES:
        for s in SIZES:
            slug, src = f"synth__pyt-{s}", STANDIN_OF[s]
            for v in VK.VITS:
                ej, ez = (os.path.join(sr, "E", pre, f"{slug}__{v}{x}") for x in (".json", ".npz"))
                ip = os.path.join(sr, "I", pre, f"{slug}__{v}.npz")
                a = json.load(open(ej))
                b = json.load(open(e_src(pre, src, v) + ".json"))
                for k in ("lm", "perm_key", "standin_of", "standin_perm_key", "synthetic"):
                    a.pop(k, None)
                for k in ("lm", "perm_key"):
                    b.pop(k, None)
                for d in (a, b):
                    d["summaries"].pop("cka_biased", None)
                    if "calib_then_max" in d:
                        d["calib_then_max"].pop("cka_biased", None)
                if a != b:
                    bad.append(f"{pre}|{slug}|{v}|E-json")
                za, zb = np.load(ez), np.load(e_src(pre, src, v) + ".npz")
                if set(za.files) != set(zb.files) or any(
                        za[k].dtype != zb[k].dtype or not np.array_equal(za[k], zb[k])
                        for k in zb.files if not k.startswith("cka_biased.")):
                    bad.append(f"{pre}|{slug}|{v}|E-npz")
                za, zb = np.load(ip), np.load(i_src(pre, src, v))
                if set(za.files) != set(zb.files) or any(
                        za[k].dtype != zb[k].dtype or not np.array_equal(za[k], zb[k])
                        for k in zb.files if k not in ("obs_cka_b", "null_cka_b", "key")):
                    bad.append(f"{pre}|{slug}|{v}|I-npz")
                for ext, p in ((".json", ej), (".npz", ez)):
                    link(p, os.path.join(R["RES"], "early", "pairs", pre, f"{slug}__{v}{ext}"))
                    n_links += 1
                link(ip, os.path.join(R["RES"], "indep", pre, "pairs", f"{slug}__{v}.npz"))
                link(ip, os.path.join(R["XE"], "indep", pre, "pairs", f"{slug}__{v}.npz"))
                n_links += 2
    for s in SIZES:
        link(feat_src(STANDIN_OF[s]), os.path.join(R["XR"], "features", f"synth__pyt-{s}.npy"))
        n_links += 1
    for fn in ("partB_features_pyt.json", "clip_thresholds_partB_pyt.json"):
        link(os.path.join(R["PB"], fn), os.path.join(R["RES"], "partB", fn))
        n_links += 1
    man = json.load(open(os.path.join(sr, "synth_manifest.json")))
    doc = dict(what="fixture X inputs (links only), synthetic variant", utc=utc(), tag=tag, variant=variant, dataset=ds,
               synth_manifest_sha256=sha(os.path.join(sr, "synth_manifest.json")),
               construction_ok=man["construction_ok"], n_links=n_links,
               partB_sha256=sha(os.path.join(R["PB"], "partB_features_pyt.json")),
               clip_sha256=sha(os.path.join(R["PB"], "clip_thresholds_partB_pyt.json")),
               synthetic_equal_frozen_outside_cka_biased=not bad, n_differences=len(bad), differences=bad[:50])
    write_report(os.path.join(R["FZ"], f"xfixture_inputs_{variant}.json"), doc)
    print(f"fixture X inputs [{variant}]: {n_links} links; synthetic files equal frozen originals outside biased CKA: "
          f"{not bad} ({len(bad)} differences)")
    return 0 if (not bad and man["construction_ok"]) else 1


def stage_verdicts(tag: str, variant: str) -> int:
    R = roots(tag, variant)
    eV = os.path.join(R["XE"], "early", "verdicts_pyt.json")
    iV = os.path.join(R["XI"], "indep", "verdicts_pyt.json")
    fE = os.path.join(R["XF"], "forecast", "E", "verdicts_forecast.json")
    fI = os.path.join(R["XF"], "forecast", "I", "verdicts_forecast.json")
    plan = [(eV, os.path.join(R["RES"], "early", "verdicts_pyt.json")),          # R targets (contract section 0)
            (iV, os.path.join(R["RES"], "indep", "verdicts_pyt.json")),
            (fE, os.path.join(R["RES"], "forecast", "E", "verdicts_forecast.json")),
            (fI, os.path.join(R["RES"], "forecast", "I", "verdicts_forecast.json")),
            (iV, os.path.join(R["XE"], "indep", "verdicts_pyt.json")),          # E's V-P5 layout (<E root>/indep, forecast)
            (fE, os.path.join(R["XE"], "forecast", "E", "verdicts_forecast.json")),
            (fI, os.path.join(R["XE"], "forecast", "I", "verdicts_forecast.json")),
            (fE, os.path.join(R["XI"], "forecast", "E", "verdicts_forecast.json")),  # I's cross_validate layout
            (fI, os.path.join(R["XI"], "forecast", "I", "verdicts_forecast.json"))]
    out = {}
    for src, dst in plan:
        out[os.path.relpath(dst, PROJ)] = dict(source=os.path.relpath(src, PROJ), sha256=copy_new(src, dst))
    write_report(os.path.join(R["FZ"], f"xfixture_verdict_copies_{variant}.json"), dict(utc=utc(), tag=tag, variant=variant, copies=out))
    print(f"fixture X: {len(out)} verdict files copied into the R, E-V-P5 and I-V-P5 layouts")
    return 0


def _classify(rep: dict, va: dict | None = None, vb: dict | None = None, a_is_I: bool = False,
              b_is_I: bool = False) -> dict:
    art = [k for k in rep["mismatching_keys"] if set(k.split("|")) & set(ARTEFACT_Q)]
    art2 = [k for k in rep["mismatching_keys"] if k == "G|clip_equal_partB"
            and ((a_is_I and va is not None and va.get(k) is None) or (b_is_I and vb is not None and vb.get(k) is None))]
    other = [k for k in rep["mismatching_keys"] if k not in set(art) and k not in set(art2)]
    ok = (rep["n_missing_in_a"] == 0 and rep["n_missing_in_b"] == 0 and not rep["only_in_a"] and not rep["only_in_b"]
          and not other and rep["n_mismatch"] == len(rep["mismatching_keys"]))
    return dict(n_required=rep["n_required"], n_compared=rep["n_compared"], n_missing_in_a=rep["n_missing_in_a"],
                n_missing_in_b=rep["n_missing_in_b"], missing_in_a=rep["missing_in_a"][:50],
                missing_in_b=rep["missing_in_b"][:50], n_only_in_a=len(rep["only_in_a"]),
                n_only_in_b=len(rep["only_in_b"]), n_mismatch=rep["n_mismatch"], n_artefact=len(art) + len(art2),
                artefact_subsets={q: sum(1 for k in art if q in k.split("|")) for q in ARTEFACT_Q},
                n_artefact_class2_post_hoc=len(art2), artefact_class2_keys=art2,
                n_other_mismatch=len(other), other_mismatching_keys=other[:200], pass_=ok, strict_pass=rep["pass"])


def _classify_names(names: list) -> dict:
    art = [k for k in names if set(str(k).split("|")) & set(ARTEFACT_Q)]
    return dict(n=len(names), n_artefact=len(art), other=[k for k in names if k not in set(art)][:100])


def _newest(pattern_dir: str, name: str, after: float) -> str | None:
    best = None
    for d, _, fs in os.walk(pattern_dir):
        if name in fs:
            p = os.path.join(d, name)
            if os.path.getmtime(p) >= after and (best is None or os.path.getmtime(p) > os.path.getmtime(best)):
                best = p
    return best


def stage_compare(tag: str, variant: str) -> int:
    import compare_verdicts as CV
    import verdict_keys as VK
    R = roots(tag, variant)
    t_start = os.path.getmtime(os.path.join(R["FZ"], "steps.tsv")) - 5 if os.path.exists(
        os.path.join(R["FZ"], "steps.tsv")) else 0.0
    f = dict(E=os.path.join(R["XE"], "early", "verdicts_pyt.json"), I=os.path.join(R["XI"], "indep", "verdicts_pyt.json"),
             fE=os.path.join(R["XF"], "forecast", "E", "verdicts_forecast.json"),
             fI=os.path.join(R["XF"], "forecast", "I", "verdicts_forecast.json"),
             RE=os.path.join(R["XR"], "out", "review", "E", "verdicts_review.json"),
             RI=os.path.join(R["XR"], "out", "review", "I", "verdicts_review.json"))
    missing = [k for k, p in f.items() if not os.path.isfile(p)]
    res = dict(what="cross-implementation fixture X (plan V-P9 last item): counts and key names of the comparisons; "
                    "section-K branch values of this stand-in/synthetic test input under 'coverage' (no Pythia byte)",
               utc=utc(), tag=tag, variant=variant, partB=os.path.relpath(os.path.join(R["PB"], "partB_features_pyt.json"), PROJ),
               files={k: dict(path=os.path.relpath(p, PROJ), sha256=sha(p) if os.path.isfile(p)
                                                  else None) for k, p in f.items()},
               artefact_rule=("class 1 (fixed before the first run): mismatches on keys naming noLMemb, LMdeep or "
                              "LMemb_only (frozen BLOOMZ E-format pair files store no summary of these subsets); class 2 "
                              "(POST HOC, added after run 20261010T022655Z): G|clip_equal_partB where I's value is null "
                              "(I's fixture mode does not evaluate V-P1 by design; in the synthetic variants I runs "
                              "without --fixture and the stand-in has no I single-model prep either; lead ruling R2); "
                              "every other mismatch, missing key or one-sided key fails"),
               missing_files=missing, comparisons={})
    if not missing:
        V = {k: json.load(open(p))["v"] for k, p in f.items()}
        for name, a, b, secs in (("E_vs_I", "E", "I", "EI"), ("forecastE_vs_forecastI", "fE", "fI", "FK"),
                                 ("RE_vs_E", "RE", "E", "EI"), ("RE_vs_forecastE", "RE", "fE", "FK"),
                                 ("RI_vs_I", "RI", "I", "EI"), ("RI_vs_forecastI", "RI", "fI", "FK"),
                                 ("RE_vs_RI", "RE", "RI", "ALL")):
            rep = CV.compare(V[a], V[b], VK.required_keys(secs))
            res["comparisons"][name] = dict(sections=secs, **_classify(rep, V[a], V[b], a == "I", b == "I"))
        # provenance keys that must agree for "one fixture": the Part B file each implementation read
        res["same_partB_read"] = len({V[k].get("G|partB_sha256") for k in ("E", "I", "RE", "RI")}) == 1
        res["partB_sha256_equals_fixture_partB"] = V["E"].get("G|partB_sha256") == sha(os.path.join(R["PB"],
                                                                                             "partB_features_pyt.json"))
        res["n_keys"] = {k: len(v) for k, v in V.items()}
        # branches reached (every variant): section-K category/modifier values of the four K writers and the H1 and
        # premise inputs of the aggregates. Values of a stand-in or synthetic TEST INPUT (no Pythia byte), recorded so
        # that the branches the fixture exercised are on record (lead ruling R3); not part of PASS.
        cov = {}
        for P in PRES:
            for x in ("s_cal", "unc"):
                kk = {f"K|{P}|{x}|{f}": f for f in ("category", "premise", "F_reason", "M2", "M5_A", "M5_A_sizes",
                                                    "M5_B", "M4")}
                hh = {f"H|{P}|{x}|H1|member": "H1_member", f"H|{P}|{x}|H1|confirmed": "H1_confirmed",
                      f"H|{P}|{x}|gate": "gate", f"PP2a|{P}|{x}|verdict": "PP2a"}
                cov[f"{P}|{x}"] = dict(
                    K={w: {f: V[w].get(k) for k, f in kk.items()} for w in ("fE", "fI", "RE", "RI")},
                    H={w: {f: V[w].get(k) for k, f in hh.items()} for w in ("E", "I", "RE", "RI")},
                    PP2b={w: V[w].get(f"PP2b|{P}|{x}|verdict") for w in ("fE", "fI", "RE", "RI")})
        res["coverage"] = cov
        if variant in SYNTH:
            miss = []
            for P, want in SYNTH[variant]["intended"].items():
                for w in ("fE", "fI", "RE", "RI"):
                    got = cov[f"{P}|s_cal"]["K"][w]
                    for f, val in want.items():
                        if got.get(f) != val:
                            miss.append(f"{w}|{P}|s_cal|{f}")
                    if want["category"] in A_TYPES and got.get("M2") is None:
                        miss.append(f"{w}|{P}|s_cal|M2 (null in an A-type slot)")
            res["intended"] = SYNTH[variant]["intended"]
            res["coverage_as_intended"] = not miss
            res["coverage_misses"] = miss
    # R's own comparisons (review/cmp), E's V-P5 file, I's cross-validation file: pass flags and counts only
    cmpd = os.path.join(R["XR"], "out", "review", "cmp")
    res["R_own_cmp"] = {fn: {k: json.load(open(os.path.join(cmpd, fn))).get(k) for k in ("n_mismatch", "pass")}
                        for fn in sorted(os.listdir(cmpd)) if fn.endswith(".json")} if os.path.isdir(cmpd) else None
    vp5 = os.path.join(R["XE"], "early", "validation", "VP5_E_vs_I.json")
    if os.path.isfile(vp5):  # descriptive (not part of PASS): counts, item names and artefact class of listed locations
        d = json.load(open(vp5))
        pairs = d.get("pairs") or {}
        items = {k: dict(n=x.get("n"), n_fail=x.get("n_fail"),
                         n_listed_artefact=sum(1 for loc in (x.get("fail_locations_first40") or [])
                                               if set(str(loc).split("|")) & set(ARTEFACT_Q)),
                         n_listed=len(x.get("fail_locations_first40") or []))
                 for k, x in pairs.items() if isinstance(x, dict) and "n_fail" in x}
        ver = d.get("verdicts") or {}
        res["E_vl5"] = dict(path=os.path.relpath(vp5, PROJ), sha256=sha(vp5), pass_=d.get("pass"),
                            pairs_pass=pairs.get("pass"), pairs_items_failing={k: v for k, v in items.items()
                                                                               if v["n_fail"]},
                            n_pair_items=len(items),
                            verdicts={s: {k: (ver.get(s) or {}).get(k) for k in ("n_mismatch", "pass")} for s in ver},
                            EI_mismatch_artefact=_classify_names((ver.get("EI") or {}).get("mismatching_keys") or []),
                            FK_mismatch_artefact=_classify_names((ver.get("FK") or {}).get("mismatching_keys") or []))
    else:
        res["E_vl5"] = None
    cx = _newest(os.path.join(R["XI"], "indep"), "cross_EI.json", t_start)
    vx = [os.path.join(R["XI"], "indep", fn) for fn in sorted(os.listdir(os.path.join(R["XI"], "indep")))
          if fn.startswith("validation_cross_") and fn.endswith(".json")] if os.path.isdir(os.path.join(R["XI"], "indep")) else []
    res["I_cross"] = dict(cross_EI=None, validation=[])  # descriptive (not part of PASS)
    if cx:
        d = json.load(open(cx))
        res["I_cross"]["cross_EI"] = dict(path=os.path.relpath(cx, PROJ), sha256=sha(cx), pass_=d.get("pass"),
                                          n_mismatch=d.get("n_mismatch"),
                                          n_missing=(d.get("n_missing_in_a"), d.get("n_missing_in_b")),
                                          mismatch_artefact=_classify_names(d.get("mismatching_keys") or []))
    for p in vx:
        d = json.load(open(p))
        res["I_cross"]["validation"].append(dict(path=os.path.relpath(p, PROJ), sha256=sha(p), pass_=d.get("pass"),
                                                 parts={k: v.get("pass") for k, v in d.items()
                                                        if isinstance(v, dict) and "pass" in v}))
    # R's own fixtures (built by R's make_test_fixture_pyt.py in this battery): semantic checks, booleans only
    rf = {}
    for var in (("", "_flip", "_nol2") if variant == "base" else ()):
        d = os.path.join(guard.SCRATCH, "R", f"fz_{tag}{var}", "out", "review")
        got = {i: json.load(open(os.path.join(d, i, "verdicts_review.json")))["v"]
               for i in ("E", "I") if os.path.isfile(os.path.join(d, i, "verdicts_review.json"))}
        chk = {f"R-{i}_required_missing_0": v.get("x|n_required_missing") == 0 for i, v in got.items()}
        if var == "_flip" and "E" in got:
            chk["flip_category_E"] = got["E"].get("K|prh_clip+l2|s_cal|category") == "E"
        if var == "_nol2" and "I" in got:
            chk["nol2_every_l2_key_null"] = all(x is None for k, x in got["I"].items()
                                                if not k.startswith("x|") and k.split("|")[1:2] == ["l2"])
        rf[f"fz{var or '_main'}"] = dict(impls=sorted(got), checks=chk, pass_=bool(got) and all(chk.values()))
    res["R_own_fixtures"] = rf
    comps = res["comparisons"]
    res["pass"] = (not missing and len(comps) == 7 and all(c["pass_"] for c in comps.values())
                   and res.get("same_partB_read") is True and res.get("partB_sha256_equals_fixture_partB") is True
                   and all(r["pass_"] for r in rf.values()))
    res["strict_pass"] = res["pass"] and all(c["strict_pass"] for c in comps.values())
    h = write_report(os.path.join(R["FZ"], f"xfixture_report_{variant}.json"), res)
    for name, c in comps.items():
        print(f"{name} [{c['sections']}]: compared {c['n_compared']}, missing {c['n_missing_in_a']}/{c['n_missing_in_b']},"
              f" one-sided {c['n_only_in_a']}/{c['n_only_in_b']}, mismatches {c['n_mismatch']} (artefact "
              f"{c['n_artefact']}, other {c['n_other_mismatch']}), pass {c['pass_']}")
    if variant in SYNTH:
        print(f"fixture X [{variant}] coverage_as_intended {res.get('coverage_as_intended')} "
              f"(misses: {res.get('coverage_misses')}) -- reported, not part of PASS")
    print(f"fixture X [{variant}] report {h}: pass {res['pass']}, strict_pass {res['strict_pass']} (counts and key "
          f"names; branch values of the test input under 'coverage')")
    return 0 if res["pass"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("partb", "synth", "inputs", "verdicts", "compare", "synth-check"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    ap.add_argument("--dataset", choices=tuple(DATASETS), default=None, help="synth-check only")
    args = ap.parse_args()
    if not guard.test_mode():  # exits 3 by itself if PYT_S0B_TEST=1 after CODE FROZEN
        print("[xfixture_pyt] REFUSED (exit 3): S0b test mode only (PYT_S0B_TEST=1 before CODE FROZEN)", file=sys.stderr)
        sys.exit(3)
    if not args.tag.replace("T", "").replace("Z", "").isdigit():
        sys.exit("bad --tag")
    if args.stage == "synth-check":
        if not args.dataset:
            sys.exit("synth-check needs --dataset")
        sys.exit(stage_synth_check(args.tag, args.dataset))
    sys.exit({"partb": stage_partb, "synth": stage_synth, "inputs": stage_inputs, "verdicts": stage_verdicts,
              "compare": stage_compare}[args.stage](args.tag, args.variant))


if __name__ == "__main__":
    main()
