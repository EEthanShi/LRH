#!/usr/bin/env python
"""S1.2 OpenLLaMA axis (third size axis), Part B, steps B0-B7 (run by role oll-axis-predict at step S5).  # OL-AXIS

Copy of analysis/s12_layermax_llama/partb_pr.py (sha256 6103a8be8bddcda65b12c7ddca812dc601189550fb5b3c50009899ed8dd2b9ba),  # OL-AXIS
adapted at S0b (sub-role PB of oll-axis-extract) to analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md  # OL-AXIS
section 9 (sha256 4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240) and to the contract  # OL-AXIS
VERDICTS_SPEC.md section 8. Every changed or added line carries the marker OL-AXIS; diff: partb_ol.py.diff.

FEATURES ONLY, plus KNOWN frozen scores. It opens the per-model feature stacks (+ sidecars), the frozen  # OL-AXIS
results/S1_2_layermax_llama/partB/partB_features.json, and the frozen score files of BLOOMZ, LLaMA-1 and  # OL-AXIS
OpenLLaMA 3B/7B (results/S1_2_layermax/pairs/<pre>/ and results/S1_2_layermax_llama/early/pairs/<pre>/).  # OL-AXIS
No cross-model score involving open_llama_13b exists or is computed: every 13B number is a function of the  # OL-AXIS
13B features alone, or a product of two single-model numbers (Proposition 5 baselines).  # OL-AXIS

B0  guard.scan() (plan F1(b)) at start and before writing; refuses (exit 4) unless clean (not in S0b test mode);  # OL-AXIS
    full scan output logged to partB/guard_scan_{start,end}.json.  # OL-AXIS
B1  float64 HKH PRs of every stored layer of the 27 known models and open_llama_13b, both pre; the 27 known  # OL-AXIS
    models are checked against the frozen LLaMA partB_features.json (PR <= 1e-14 relative, clip bitwise);  # OL-AXIS
    13B clip threshold (hrhv2.prep.prh_clip_threshold on the float64 all-layer stack, D5) ->  # OL-AXIS
    partB/clip_thresholds_partB_ol.json.  # OL-AXIS
B2  Bbar (+argmax, u*/(L-1)), B_fd (r = 0.5, 1.0 at floor(r(L-1)+0.5)), B_late (LM states with relative depth  # OL-AXIS
    >= 0.60, all ViT layers), 13B PR at relative depth 0.84-0.90.  # OL-AXIS
B3  Tr_pred = Tr_O(Bbar), Tr_O(B_fd), Tr_O(B_late), per-ViT gaps for the 3 size pairs.  # OL-AXIS
B4  materiality: ||Tr| - 1| <= 1e-9 plus per-ViT strict monotonicity, and median |gap| >= 0.005 for each pair.  # OL-AXIS
B5  Pearson(ARH rank, Bbar) on 187 pairs (and 10, 8 LMs), mean Bbar by LM; retrodictions (OpenLLaMA 3B-7B c_alpha  # OL-AXIS
    order vs Bbar order, LLaMA 51 comparisons, BLOOMZ Tr(Bbar)).  # OL-AXIS
B6  s_O, s_O_J, s_late; band o = 0.0019 [0.0009, 0.0029], h = 0.0010; kappa boundaries for s_cal (+ h curve) and  # OL-AXIS
    the uncensored boundaries, from the frozen 8-LM 3b/7b layer-max values (cka_biased, full); n_B(t), n_L(t)  # OL-AXIS
    and the J tables (known BLOOMZ/LLaMA only).  # OL-AXIS
B7  partB/partB_features_ol.json (every field of VERDICTS_SPEC section 8) and the DRAFT  # OL-AXIS
    analysis/s12_layermax_openllama/PARTB_PREDICTIONS_OL.md (numbers filled in from the JSON).  # OL-AXIS

RUN (step S5: after S4 extraction and its checks; nothing else of this axis running; CPU only):  # OL-AXIS
  cd ~/hrh_v2_stage1_20261008  # OL-AXIS
  analysis/s12_layermax_openllama/launch_ol.sh partb_ol 6 -- venv/bin/python analysis/s12_layermax_openllama/partb_ol.py  # OL-AXIS
  then (sanity/retrodiction on known pairs only):  # OL-AXIS
  analysis/s12_layermax_openllama/launch_ol.sh partb_checks_ol 2 -- venv/bin/python analysis/s12_layermax_openllama/partb_checks_ol.py  # OL-AXIS
  Exit codes: 0 ok; 4 guard scan not clean; 5 a recomputation check failed (27 known models, plan hash,  # OL-AXIS
  frozen input hash, layer rule); nothing is written to partB/ except the scan logs and the check report then.  # OL-AXIS
# OL-AXIS
FREEZE (role oll-axis-predict, plan B7): review the draft PARTB_PREDICTIONS_OL.md (edit wording only; numbers  # OL-AXIS
come from the JSON), run `venv/bin/python analysis/s12_layermax_openllama/guard.py scan` (must exit 0; log it),  # OL-AXIS
chmod 444 both files, then append ONE STATUS line of exactly this form (guard.py parses it):  # OL-AXIS
<UTC>Z [oll-axis-predict] Part B FROZEN (S5, plan section 9 B7): analysis/s12_layermax_openllama/PARTB_PREDICTIONS_OL.md sha256 <64 hex> ; results/S1_2_layermax_openllama/partB/partB_features_ol.json sha256 <64 hex> ; analysis/s12_layermax_openllama/partb_ol.py sha256 <64 hex> ; clip_thresholds_partB_ol.json sha256 <64 hex> ; guard scan clean at start and freeze  # OL-AXIS
  (<UTC> = date -u +%Y-%m-%dT%H:%M:%S at the time of logging.)  # OL-AXIS
# OL-AXIS
S0b TEST MODE (OL_S0B_TEST=1, only before the CODE FROZEN line; guard.test_mode()): outputs go to  # OL-AXIS
~/oll_s0b_scratch/PB/ (partB/ and PARTB_PREDICTIONS_OL.md), a failing scan is logged but not refused, and  # OL-AXIS
--standin selects a stand-in for open_llama_13b (no file named *open_llama_13b* is created):  # OL-AXIS
  llama13b    the llama-13b features stand in for 13B under slug synth__ol13b (3b/7b real)          [default]  # OL-AXIS
  llama_axis  llama-7b/13b/30b stand in for 3b/7b/13b under slugs synth__ol3b/synth__ol7b/synth__ol13b; the  # OL-AXIS
              known 3b/7b values come from the LLaMA-axis E pair files (end-to-end LLaMA reproduction check)  # OL-AXIS
  OL_S0B_TEST=1 OL_LOGDIR=~/oll_s0b_scratch/PB/logs analysis/s12_layermax_openllama/launch_ol.sh pb_test 2 --  # OL-AXIS
      venv/bin/python analysis/s12_layermax_openllama/partb_ol.py --standin llama13b   (one command line)  # OL-AXIS
"""
from __future__ import annotations

import os

_TH = os.environ.get("OL_THREADS", "1")  # OL-AXIS: thread count from launch_ol.sh (plan F5)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _TH  # OL-AXIS

import argparse  # noqa: E402  # OL-AXIS
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402  # OL-AXIS
import sys  # noqa: E402
import time  # noqa: E402

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")  # OL-AXIS
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12"))
sys.path.insert(0, AXIS)  # OL-AXIS

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import common as C  # noqa: E402  (S1.2 registry + preprocess_rows; read-only import)
import guard  # noqa: E402  # OL-AXIS: F1(b) scan, S0b test mode
from hrhv2.prep import prh_clip_threshold, center_gram  # noqa: E402

_spec = importlib.util.spec_from_file_location("s12_aggregate", os.path.join(PROJ, "analysis", "s12", "aggregate.py"))
A12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A12)          # read-only import: _rho_matrix (trend arithmetic of lm_trend)

FEAT = os.path.join(PROJ, "features")
RES_OL = os.path.join(PROJ, "results", "S1_2_layermax_openllama")  # OL-AXIS
OUTDIR_REAL = os.path.join(RES_OL, "partB")  # OL-AXIS
MD_REAL = os.path.join(AXIS, "PARTB_PREDICTIONS_OL.md")  # OL-AXIS
SCRATCH_PB = os.path.join(guard.SCRATCH, "PB")  # OL-AXIS: S0b test outputs only
PLAN = os.path.join(AXIS, "ANALYSIS_PLAN_OPENLLAMA.md")  # OL-AXIS
PLAN_SHA = "4f18d710beda45492b1e191f74580bfc4ef53f6dafe791ad245ff1cf335f3240"  # OL-AXIS
LLAMA_PARTB = os.path.join(PROJ, "results", "S1_2_layermax_llama", "partB", "partB_features.json")  # OL-AXIS
LLAMA_PARTB_SHA = "ede0127589a02037427b343e9c3593ee427d430e9746d7f4942b15f76a3031bf"  # OL-AXIS (STATUS P0 line)
FROZEN8 = os.path.join(PROJ, "results", "S1_2_layermax", "pairs")  # OL-AXIS: frozen 8-LM E-format pair files
LLAMA_E = os.path.join(PROJ, "results", "S1_2_layermax_llama", "early", "pairs")  # OL-AXIS: LLaMA-axis E pairs

M = 1024
KNN_K = 10
DELTA_B = 0.005
DEPTHS = (0.5, 1.0)
PRES = ("prh_clip+l2", "l2")
LATE_REL = 0.60  # OL-AXIS: B_late, LM states with relative depth >= 0.60 (plan B2, spec section 8)
CTM_BAND = (0.84, 0.90)  # OL-AXIS: descriptive 13B PR band (plan B2)
TOL = 1e-9  # OL-AXIS: plan F2 tolerance ("|Tr| = 1", relabelling counts)
MAT = 0.10  # OL-AXIS: Tr materiality (plan 5.1)
BAND_PRIMARY = {"o": 0.0019, "o_lo": 0.0009, "o_hi": 0.0029, "h": 0.0010}  # OL-AXIS: numeric constants, plan B6
MU_BAND = (-0.0002, 0.0002)  # OL-AXIS: uncensored band mu_13 = Bbar_13 + [-0.0002, +0.0002] (plan B6)
HS = (0.0003, 0.0005, 0.001, 0.0015, 0.002)  # OL-AXIS: h curve (plan 5.6)
DBS = (0.0025, 0.005, 0.0075, 0.01)  # OL-AXIS: delta_b curve (plan 5.6)
JT_ROWS = (-0.912, -0.60, -0.50, -0.30, -0.15, -0.10, 0.00)  # OL-AXIS: plan section 12 J_min table rows
NREL = 4320  # OL-AXIS: 5! * 3! * 3!

OL3, OL7, OL13 = ("openlm-research__open_llama_3b", "openlm-research__open_llama_7b",  # OL-AXIS
                  "openlm-research__open_llama_13b")  # OL-AXIS
SIZES = ("3b", "7b", "13b")  # OL-AXIS
PAIRS3 = ((0, 1, "3b|7b"), (1, 2, "7b|13b"), (0, 2, "3b|13b"))  # OL-AXIS: (smaller, larger, spec key)
KNOWN_LMS = [(s, f, "wit1024") for s, f in C.LMS] + [  # OL-AXIS: the 10 LMs of the frozen LLaMA Part B
    ("huggyllama__llama-13b", "LLaMA", "wit1024_llama_axis"),
    ("huggyllama__llama-30b", "LLaMA", "wit1024_llama_axis"),
]
VITS = [(s, f, "wit1024") for s, f in C.VITS]
VIT_STEMS = [s for s, _, _ in VITS]  # OL-AXIS
LLAMA = ["huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"]
BLOOMZ = [s for s, f, _ in KNOWN_LMS if f == "BLOOMZ"]  # OL-AXIS
CFG: dict = {}  # OL-AXIS: filled by configure()
# OL-AXIS
# OL-AXIS
def configure(test: bool, standin: str | None) -> dict:  # OL-AXIS
    """Model sources, slugs, output paths. Real mode: open_llama_13b from features/wit1024_openllama_axis/."""  # OL-AXIS
    if not test:  # OL-AXIS
        ol = {"3b": OL3, "7b": OL7, "13b": OL13}  # OL-AXIS
        new = {OL13: (OL13, "wit1024_openllama_axis")}  # OL-AXIS: slug -> (source stem, feature dir)
        known_src = {"3b": (FROZEN8, OL3), "7b": (FROZEN8, OL7)}  # OL-AXIS
        outdir, md = OUTDIR_REAL, MD_REAL  # OL-AXIS
    elif standin == "llama13b":  # OL-AXIS
        ol = {"3b": OL3, "7b": OL7, "13b": "synth__ol13b"}  # OL-AXIS
        new = {"synth__ol13b": ("huggyllama__llama-13b", "wit1024_llama_axis")}  # OL-AXIS
        known_src = {"3b": (FROZEN8, OL3), "7b": (FROZEN8, OL7)}  # OL-AXIS
        outdir, md = os.path.join(SCRATCH_PB, "partB"), os.path.join(SCRATCH_PB, "PARTB_PREDICTIONS_OL.md")  # OL-AXIS
    elif standin == "llama_axis":  # OL-AXIS
        ol = {"3b": "synth__ol3b", "7b": "synth__ol7b", "13b": "synth__ol13b"}  # OL-AXIS
        new = {"synth__ol3b": ("huggyllama__llama-7b", "wit1024"),  # OL-AXIS
               "synth__ol7b": ("huggyllama__llama-13b", "wit1024_llama_axis"),  # OL-AXIS
               "synth__ol13b": ("huggyllama__llama-30b", "wit1024_llama_axis")}  # OL-AXIS
        known_src = {"3b": (LLAMA_E, "huggyllama__llama-7b"), "7b": (LLAMA_E, "huggyllama__llama-13b")}  # OL-AXIS
        outdir = os.path.join(SCRATCH_PB, "partB_llama_axis")  # OL-AXIS
        md = os.path.join(SCRATCH_PB, "PARTB_PREDICTIONS_OL.llama_axis.md")  # OL-AXIS
    else:  # OL-AXIS
        raise SystemExit(f"unknown --standin {standin}")  # OL-AXIS
    pooled = BLOOMZ + [ol["3b"], ol["7b"], ol["13b"]] + LLAMA  # OL-AXIS: ARH order, ranks 1..11 (plan section 1)
    src = {s: (s, ds) for s, _, ds in KNOWN_LMS + VITS}  # OL-AXIS: key -> (source stem, feature dir)
    src.update(new)  # OL-AXIS
    fam = {s: f for s, f, _ in KNOWN_LMS + VITS}  # OL-AXIS
    fam.update({k: "OpenLLaMA" for k in new})  # OL-AXIS
    CFG.clear()  # OL-AXIS
    CFG.update({"test": test, "standin": standin if test else None, "ol": ol, "new": new, "known_src": known_src,  # OL-AXIS
                "outdir": outdir, "md": md, "pooled": pooled, "src": src, "fam": fam,  # OL-AXIS
                "arh_rank": {s: i + 1 for i, s in enumerate(pooled)}})  # OL-AXIS
    return CFG  # OL-AXIS


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sidecar(stem: str) -> dict:
    s, ds = CFG["src"][stem]  # OL-AXIS
    with open(os.path.join(FEAT, ds, s + ".json")) as f:  # OL-AXIS
        return json.load(f)


def npy_path(stem: str) -> str:  # OL-AXIS
    s, ds = CFG["src"][stem]  # OL-AXIS
    return os.path.join(FEAT, ds, s + ".npy")  # OL-AXIS
# OL-AXIS
# OL-AXIS
def stack(stem: str) -> np.ndarray:
    st = np.load(npy_path(stem), mmap_mode="r")  # OL-AXIS
    assert st.shape[0] == M and st.ndim == 3, (stem, st.shape)
    assert list(st.shape) == list(sidecar(stem)["shape"]), stem
    return st


def fixed_layer(L: int, r: float) -> int:
    return int(math.floor(r * (L - 1) + 0.5))


def layer_pr(X: np.ndarray, pre: str, c: float | None) -> tuple[float, float, float]:
    Xp = C.preprocess_rows(X, pre, c)
    K = Xp @ Xp.T
    Kc = center_gram(K)
    tr = float(np.trace(Kc))
    fr = float(np.vdot(Kc.ravel(), Kc.ravel()))
    return tr * tr / fr, tr, fr


def family_trend(G: np.ndarray, sizes: list[int]) -> dict:
    """lm_trend (analysis/s12/aggregate.py) for an arbitrary family: rows of G
    = family members, columns = ViTs. Tr = mean over ViTs of Spearman(size,
    score); exact p over all relabellings of the sizes (same for every ViT).  # OL-AXIS
    OL-AXIS: tolerance 1e-9 (plan F2), p_le added, Tr(sigma) over all relabellings returned."""
    Y = np.asarray(G, dtype=float)
    ok = np.all(np.isfinite(Y), axis=0)
    Y = Y[:, ok]
    x = rankdata(sizes)
    P = np.array([np.array(p) for p in itertools.permutations(x)])
    rho = A12._rho_matrix(P, Y)
    obs_idx = int(np.flatnonzero(np.all(P == x, axis=1))[0])
    with np.errstate(invalid="ignore"):
        T = np.nanmean(rho, axis=1)
    t_obs = float(T[obs_idx])
    rho_v = rho[obs_idx]
    eps = TOL  # OL-AXIS
    return {"Tr": t_obs, "rho_per_vit": [None if not np.isfinite(v) else float(v) for v in rho_v],
            "n_vit_used": int(np.sum(np.isfinite(rho_v))),
            "n_vit_constant": int(np.sum(~np.isfinite(rho_v)) + np.sum(~ok)),
            "p_one": float(np.mean(T >= t_obs - eps)), "p_two": float(np.mean(np.abs(T) >= abs(t_obs) - eps)),
            "p_le": float(np.mean(T <= t_obs + eps)),  # OL-AXIS
            "n_relabel": int(P.shape[0]), "Tr_all": T, "obs_idx": obs_idx}  # OL-AXIS


def jsonable(v):
    return C.jsonable(v)


def sgn(x: float) -> int:  # OL-AXIS
    return 0 if x == 0 else (1 if x > 0 else -1)  # OL-AXIS
# OL-AXIS
# OL-AXIS
def read_pair_json(base: str, pre: str, lm: str, vit: str) -> tuple[dict, str]:  # OL-AXIS
    p = os.path.join(base, pre, f"{lm}__{vit}.json")  # OL-AXIS
    with open(p) as f:  # OL-AXIS
        js = json.load(f)  # OL-AXIS
    assert js["lm"] == lm and js["vit"] == vit and js["pre"] == pre, p  # OL-AXIS
    return js, p  # OL-AXIS
# OL-AXIS
# OL-AXIS
def score_grid(base: str, pre: str, lms: list[str], met: str, field: str, used: dict) -> np.ndarray:  # OL-AXIS
    """Layer-max (protocol full) summary field of frozen E-format pair files: rows = lms, cols = 17 ViTs."""  # OL-AXIS
    G = np.empty((len(lms), len(VIT_STEMS)))  # OL-AXIS
    for i, lm in enumerate(lms):  # OL-AXIS
        for j, v in enumerate(VIT_STEMS):  # OL-AXIS
            js, p = read_pair_json(base, pre, lm, v)  # OL-AXIS
            used[os.path.relpath(p, PROJ)] = None  # OL-AXIS: hashed at the end
            G[i, j] = float(js["summaries"][met]["full"][field])  # OL-AXIS
    return G  # OL-AXIS
# OL-AXIS
# OL-AXIS
def relabel_dtr(base: str, pre: str, lms: list[str], x: str, used: dict) -> np.ndarray:  # OL-AXIS
    """dTr_f(sigma) = Tr_f(x; sigma) - Tr_f(raw; sigma) over all relabellings (identity first), cka_biased, full."""  # OL-AXIS
    sizes = list(range(1, len(lms) + 1))  # OL-AXIS: lms are in size order
    fx = {"s_cal": "s_cal", "unc": "uncensored"}[x]  # OL-AXIS
    tx = family_trend(score_grid(base, pre, lms, "cka_biased", fx, used), sizes)  # OL-AXIS
    tr = family_trend(score_grid(base, pre, lms, "cka_biased", "s_obs", used), sizes)  # OL-AXIS
    assert tx["obs_idx"] == 0 and tr["obs_idx"] == 0  # OL-AXIS
    return tx["Tr_all"] - tr["Tr_all"]  # OL-AXIS
# OL-AXIS
# OL-AXIS
def run_scan(tag: str) -> dict:  # OL-AXIS
    """Plan F1(b): negative scan, full output logged; refuses unless clean (except S0b test mode)."""  # OL-AXIS
    try:  # OL-AXIS: fail closed if the scan itself cannot run (e.g. guard.py's find -printf %W is
        rep = guard.scan()  # OL-AXIS: unsupported by GNU find 4.9.0 on ds2213: reported at S0b)
    except Exception as e:  # noqa: BLE001  # OL-AXIS
        rep = {"utc": utc(), "clean": False, "find_hits": [], "results_top_level": [],  # OL-AXIS
               "results_top_level_not_allowed": [], "forbidden_processes": [],  # OL-AXIS
               "scan_error": f"{type(e).__name__}: {e}"}  # OL-AXIS
        print(f"[{utc()}] guard.scan() FAILED: {rep['scan_error']} -> treated as NOT clean", flush=True)  # OL-AXIS
    os.makedirs(CFG["outdir"], exist_ok=True)  # OL-AXIS
    with open(os.path.join(CFG["outdir"], f"guard_scan_{tag}.json"), "w") as f:  # OL-AXIS
        json.dump(rep, f, indent=1)  # OL-AXIS
    print(f"[{utc()}] guard scan ({tag}): clean={rep['clean']} find_hits={len(rep['find_hits'])} "  # OL-AXIS
          f"top={rep['results_top_level']} not_allowed={rep['results_top_level_not_allowed']} "  # OL-AXIS
          f"procs={rep['forbidden_processes']}", flush=True)  # OL-AXIS
    if not rep["clean"] and not CFG["test"]:  # OL-AXIS
        raise SystemExit(4)  # OL-AXIS
    return {"utc": rep["utc"], "clean": rep["clean"], "n_find_hits": len(rep["find_hits"]),  # OL-AXIS
            "results_top_level": rep["results_top_level"], "forbidden_processes": rep["forbidden_processes"],  # OL-AXIS
            "scan_error": rep.get("scan_error"),  # OL-AXIS
            "log": os.path.join(CFG["outdir"], f"guard_scan_{tag}.json")}  # OL-AXIS
# OL-AXIS
# OL-AXIS
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # OL-AXIS
    ap.add_argument("--standin", choices=("llama13b", "llama_axis"), default="llama13b",  # OL-AXIS
                    help="S0b test mode only: stand-in for open_llama_13b (ignored otherwise)")  # OL-AXIS
    args = ap.parse_args()  # OL-AXIS
    test = guard.test_mode()  # OL-AXIS
    configure(test, args.standin)  # OL-AXIS
    t0 = time.time()
    started = utc()
    # ---------------- B0 guard ----------------------------------------------  # OL-AXIS
    scan_start = run_scan("start")  # OL-AXIS
    plan_sha = sha256_file(PLAN)
    if plan_sha != PLAN_SHA:
        raise SystemExit(f"ABORT (5): plan hash changed: {plan_sha}")  # OL-AXIS
    llpb_sha = sha256_file(LLAMA_PARTB)  # OL-AXIS
    if llpb_sha != LLAMA_PARTB_SHA:  # OL-AXIS
        raise SystemExit(f"ABORT (5): frozen LLaMA partB_features.json hash changed: {llpb_sha}")  # OL-AXIS
    LLPB = json.load(open(LLAMA_PARTB))  # OL-AXIS
    os.makedirs(CFG["outdir"], exist_ok=True)  # OL-AXIS
    OUTDIR = CFG["outdir"]  # OL-AXIS
    ol = CFG["ol"]  # OL-AXIS
    OLS = [ol[k] for k in SIZES]  # OL-AXIS
    if not CFG["test"]:  # OL-AXIS: the 13B extraction must be complete (S4)
        done = os.path.join(FEAT, "wit1024_openllama_axis", OL13 + ".done")  # OL-AXIS
        if not os.path.isfile(done):  # OL-AXIS
            raise SystemExit(f"ABORT (5): {done} missing (S4 not complete)")  # OL-AXIS

    # ---------------- inputs ------------------------------------------------
    MODELS = [s for s, _, _ in KNOWN_LMS + VITS] + list(CFG["new"])  # OL-AXIS: 27 known + new (13B)
    LM_KEYS = [s for s, _, _ in KNOWN_LMS] + list(CFG["new"])  # OL-AXIS
    inputs = {}
    sha_cache = {}  # OL-AXIS
    for stem in MODELS:  # OL-AXIS
        p = npy_path(stem)  # OL-AXIS
        sc = sidecar(stem)
        if p not in sha_cache:  # OL-AXIS
            sha_cache[p] = sha256_file(p)  # OL-AXIS
        inputs[stem] = {"path": os.path.relpath(p, PROJ), "sha256": sha_cache[p], "family": CFG["fam"][stem],  # OL-AXIS
                        "kind": sc["kind"], "shape": sc["shape"], "num_params": int(sc["num_params"]),  # OL-AXIS
                        "source_stem": CFG["src"][stem][0]}  # OL-AXIS
    if not CFG["test"]:  # OL-AXIS
        assert inputs[OL13]["shape"] == [M, 41, 5120], inputs[OL13]["shape"]  # OL-AXIS: plan section 2
    print(f"[{time.time() - t0:6.0f}s] hashed {len(sha_cache)} feature stacks", flush=True)  # OL-AXIS

    # ---------------- clip thresholds (D5, all-layer float64 stack) --------
    known_clip = LLPB["meta"]["clip_thresholds"]  # OL-AXIS: frozen LLaMA Part B values
    clip, clip_by_src = {}, {}  # OL-AXIS
    for stem in MODELS:  # OL-AXIS
        p = npy_path(stem)  # OL-AXIS
        if p not in clip_by_src:  # OL-AXIS
            st64 = np.asarray(stack(stem), dtype=np.float64)  # OL-AXIS
            clip_by_src[p] = prh_clip_threshold(st64)  # OL-AXIS
            del st64  # OL-AXIS
        c = clip_by_src[p]  # OL-AXIS
        ref = known_clip.get(CFG["src"][stem][0])  # OL-AXIS: known model (or stand-in's source)
        clip[stem] = {"c": c, "c_llama_partB": ref,  # OL-AXIS
                      "bitwise_equal_llama_partB": (None if ref is None else bool(c == float(ref)))}  # OL-AXIS
        print(f"[{time.time() - t0:6.0f}s] clip {stem}: {c!r} (LLaMA Part B equal: {clip[stem]['bitwise_equal_llama_partB']})",  # OL-AXIS
              flush=True)

    # ---------------- B1: per-layer PR --------------------------------------
    PR, TR = {pre: {} for pre in PRES}, {pre: {} for pre in PRES}
    pr_by_src = {}  # OL-AXIS
    for stem in MODELS:  # OL-AXIS
        p = npy_path(stem)  # OL-AXIS
        if p not in pr_by_src:  # OL-AXIS
            st = stack(stem)  # OL-AXIS
            L = st.shape[1]  # OL-AXIS
            res = {}  # OL-AXIS
            for pre in PRES:  # OL-AXIS
                c = clip[stem]["c"] if pre == "prh_clip+l2" else None  # OL-AXIS
                prs, trs = np.empty(L), np.empty(L)  # OL-AXIS
                for l in range(L):  # OL-AXIS
                    prs[l], trs[l], _ = layer_pr(np.asarray(st[:, l, :]), pre, c)  # OL-AXIS
                res[pre] = (prs, trs)  # OL-AXIS
            pr_by_src[p] = res  # OL-AXIS
            del st  # OL-AXIS
        for pre in PRES:  # OL-AXIS
            PR[pre][stem], TR[pre][stem] = pr_by_src[p][pre]  # OL-AXIS
        print(f"[{time.time() - t0:6.0f}s] PR {stem} L={PR['l2'][stem].size} max(prh_clip+l2)={PR['prh_clip+l2'][stem].max():.2f} "  # OL-AXIS
              f"max(l2)={PR['l2'][stem].max():.2f}", flush=True)

    # ---------------- B1 check: 27 known models vs frozen LLaMA Part B (<= 1e-14 rel; clip bitwise) -----  # OL-AXIS
    recheck = {"tolerance_rel": 1e-14, "per_model": {}}  # OL-AXIS
    for stem in MODELS:  # OL-AXIS
        s0 = CFG["src"][stem][0]  # OL-AXIS
        ref = {pre: (LLPB["per_pre"][pre]["lm"].get(s0) or LLPB["per_pre"][pre]["vit"].get(s0)) for pre in PRES}  # OL-AXIS
        if ref["l2"] is None:  # OL-AXIS: open_llama_13b (no frozen reference)
            continue  # OL-AXIS
        e = {}  # OL-AXIS
        for pre in PRES:  # OL-AXIS
            a, b = PR[pre][stem], np.asarray(ref[pre]["PR"], dtype=float)  # OL-AXIS
            e[pre] = float(np.max(np.abs(a - b) / np.abs(b))) if a.shape == b.shape else float("inf")  # OL-AXIS
        recheck["per_model"][stem] = {"max_rel_err_PR": e, "clip_bitwise": clip[stem]["bitwise_equal_llama_partB"]}  # OL-AXIS
    errs = [v for d in recheck["per_model"].values() for v in d["max_rel_err_PR"].values()]  # OL-AXIS
    recheck["n_models"] = len(recheck["per_model"])  # OL-AXIS
    recheck["max_rel_err_PR"] = max(errs)  # OL-AXIS
    recheck["all_clip_bitwise"] = all(d["clip_bitwise"] for d in recheck["per_model"].values())  # OL-AXIS
    recheck["pass"] = bool(recheck["max_rel_err_PR"] <= 1e-14 and recheck["all_clip_bitwise"]  # OL-AXIS
                           and recheck["n_models"] >= 27)  # OL-AXIS
    with open(os.path.join(OUTDIR, "partB_recompute_check_ol.json"), "w") as f:  # OL-AXIS
        json.dump(jsonable(recheck), f, indent=1)  # OL-AXIS
    print(f"[{time.time() - t0:6.0f}s] B1 recheck of {recheck['n_models']} known models: max rel err PR "  # OL-AXIS
          f"{recheck['max_rel_err_PR']:.3e}, clip bitwise {recheck['all_clip_bitwise']} -> pass {recheck['pass']}", flush=True)  # OL-AXIS
    if not recheck["pass"]:  # OL-AXIS
        raise SystemExit("ABORT (5): B1 recomputation of the known models failed (see partB_recompute_check_ol.json)")  # OL-AXIS
# OL-AXIS
    # ---------------- B2-B6 --------------------------------------------------  # OL-AXIS
    lm_stems = CFG["pooled"]  # OL-AXIS: 11 LMs, ARH order
    vit_stems = VIT_STEMS  # OL-AXIS
    nparams = {s: inputs[s]["num_params"] for s in inputs}
    sizes_O = [nparams[s] for s in OLS]  # OL-AXIS
    if not CFG["test"]:  # OL-AXIS
        assert sizes_O == sorted(sizes_O) and len(set(sizes_O)) == 3, sizes_O  # OL-AXIS
    used_scores: dict = {}  # OL-AXIS: frozen score files opened (path -> sha256)
    out = {"schema": "OL-PARTB v1", "m": M, "delta_b": DELTA_B,  # OL-AXIS: VERDICTS_SPEC section 8
           "band": dict(BAND_PRIMARY), "mu_band": list(MU_BAND),  # OL-AXIS
           "ol_lms": OLS, "vits": vit_stems, "pooled11": lm_stems,  # OL-AXIS
           "num_params": {s: nparams[s] for s in lm_stems + vit_stems},  # OL-AXIS
           "L_stored": {s: int(PR["l2"][s].size) for s in lm_stems + vit_stems},  # OL-AXIS
           "meta": {"role": "oll-axis-predict (code: S0b sub-role PB)", "plan": os.path.relpath(PLAN, PROJ),  # OL-AXIS
                    "plan_sha256": plan_sha,  # OL-AXIS
                    "script_sha256": sha256_file(os.path.abspath(__file__)), "m": M, "k": KNN_K, "delta_b": DELTA_B,
                    "guard_sha256": sha256_file(guard.__file__),  # OL-AXIS
                    "test_mode": CFG["test"], "standin": CFG["standin"], "ol_size_labels": ol,  # OL-AXIS
                    "threads": _TH, "python": platform.python_version(), "numpy": np.__version__,  # OL-AXIS
                    "started_utc": started, "guard_scan_start": scan_start, "inputs": inputs,  # OL-AXIS
                    "llama_partB": {"path": os.path.relpath(LLAMA_PARTB, PROJ), "sha256": llpb_sha},  # OL-AXIS
                    "recheck_27": {k: recheck[k] for k in ("n_models", "max_rel_err_PR", "all_clip_bitwise", "pass")},  # OL-AXIS
                    "clip_thresholds": {s: clip[s]["c"] for s in clip},
                    "mknn_baseline_every_pair": KNN_K / (M - 1), "debiased_cka_baseline_every_pair": 0.0,
                    "note": "features only, plus known frozen scores (BLOOMZ, LLaMA, OpenLLaMA 3B/7B); "  # OL-AXIS
                            "no cross-model score involving open_llama_13b computed or opened"},  # OL-AXIS
           "per_pre": {}}
    for pre in PRES:
        P = PR[pre]
        lm_info, vit_info = {}, {}
        for s in lm_stems + vit_stems:  # OL-AXIS
            v = P[s]
            L = v.size
            order = np.argsort(-v, kind="stable")
            rel = np.arange(L) / (L - 1)  # OL-AXIS
            info = {"L": L, "PR": v.tolist(), "trKc": TR[pre][s].tolist(),
                    "PR_star": float(v[order[0]]), "argmax": int(order[0]), "argmax_rel": float(order[0] / (L - 1)),
                    "PR_second": float(v[order[1]]), "argmax_second": int(order[1]),
                    "PR_star_over_second": float(v[order[0]] / v[order[1]]),
                    "PR_first_layer": float(v[0]), "PR_last_layer": float(v[-1]), "PR_min": float(v.min()),
                    "fixed": {str(r): {"layer": fixed_layer(L, r), "PR": float(v[fixed_layer(L, r)])} for r in DEPTHS}}
            if s in lm_stems:  # OL-AXIS: B_late side (LM states with relative depth >= 0.60)
                late = np.flatnonzero(rel >= LATE_REL - 1e-12)  # OL-AXIS
                ul = int(late[np.argmax(v[late])])  # OL-AXIS: first maximum
                band = np.flatnonzero((rel >= CTM_BAND[0] - 1e-12) & (rel <= CTM_BAND[1] + 1e-12))  # OL-AXIS
                info.update({"late_first_layer": int(late[0]), "PR_late_star": float(v[ul]), "u_late": ul,  # OL-AXIS
                             "u_late_rel": float(ul / (L - 1)),  # OL-AXIS
                             "ctm_band_0.84_0.90": [{"u": int(u), "rel": float(rel[u]), "PR": float(v[u])} for u in band],  # OL-AXIS
                             "ctm_band_PR_max": float(v[band].max()) if band.size else None,  # OL-AXIS
                             "ctm_band_PR_median": float(np.median(v[band])) if band.size else None})  # OL-AXIS
            (lm_info if s in lm_stems else vit_info)[s] = info
        # B2: Bbar (11 x 17), argmax, fixed depth, late  # OL-AXIS
        Bbar = np.empty((len(lm_stems), len(vit_stems)))
        Bfd = {r: np.empty_like(Bbar) for r in DEPTHS}
        Blate = np.empty_like(Bbar)  # OL-AXIS
        Bbar_bruteforce_maxdiff = 0.0
        Blate_bruteforce_maxdiff = 0.0  # OL-AXIS
        for i, a in enumerate(lm_stems):
            for j, b in enumerate(vit_stems):
                Buw = np.sqrt(P[a][:, None] * P[b][None, :]) / (M - 1)
                Bbar[i, j] = math.sqrt(lm_info[a]["PR_star"] * vit_info[b]["PR_star"]) / (M - 1)
                Bbar_bruteforce_maxdiff = max(Bbar_bruteforce_maxdiff, abs(float(Buw.max()) - Bbar[i, j]))
                for r in DEPTHS:
                    Bfd[r][i, j] = Buw[lm_info[a]["fixed"][str(r)]["layer"], vit_info[b]["fixed"][str(r)]["layer"]]
                Blate[i, j] = math.sqrt(lm_info[a]["PR_late_star"] * vit_info[b]["PR_star"]) / (M - 1)  # OL-AXIS
                Blate_bruteforce_maxdiff = max(Blate_bruteforce_maxdiff,  # OL-AXIS
                                               abs(float(Buw[lm_info[a]["late_first_layer"]:].max()) - Blate[i, j]))  # OL-AXIS
        oi = [lm_stems.index(s) for s in OLS]  # OL-AXIS
        bi = [lm_stems.index(s) for s in BLOOMZ]  # OL-AXIS
        li = [lm_stems.index(s) for s in LLAMA]  # OL-AXIS
        sizes_B = [nparams[s] for s in BLOOMZ]  # OL-AXIS
        sizes_L = [nparams[s] for s in LLAMA]  # OL-AXIS
        if not CFG["test"]:  # OL-AXIS: plan section 3 layer table
            assert [lm_info[s]["fixed"]["0.5"]["layer"] for s in OLS] == [13, 16, 20]  # OL-AXIS
            assert [lm_info[s]["fixed"]["1.0"]["layer"] for s in OLS] == [26, 32, 40]  # OL-AXIS

        def axis_block(G, rows, stems, sizes):
            tr = family_trend(G[rows], sizes)
            gaps = {}
            for (x, y) in itertools.combinations(range(len(rows)), 2):
                g = np.abs(G[rows[x]] - G[rows[y]])
                d = G[rows[y]] - G[rows[x]]           # larger minus smaller model (rows in size order)
                gaps[f"{stems[x]}|{stems[y]}"] = {"gap_per_vit": g.tolist(), "signed_larger_minus_smaller_per_vit": d.tolist(),
                                                  "median": float(np.median(g)), "min": float(g.min()),
                                                  "max": float(g.max()),
                                                  "median_signed": float(np.median(d)),  # OL-AXIS
                                                  "n_vit_gap_ge_delta_b": int(np.sum(g >= DELTA_B)),
                                                  "n_vit_larger_higher": int(np.sum(d > 0))}
            # |Tr| = 1 tested with the plan F2 tolerance 1e-9 (Spearman via _rho_matrix is not exact in  # OL-AXIS
            # floating point: a perfectly monotone row can give 0.9999999999999998); monotonicity also
            # checked directly on the values.
            Gm = G[rows]
            inc = bool(np.all(np.diff(Gm, axis=0) > 0))
            dec = bool(np.all(np.diff(Gm, axis=0) < 0))
            tr_is_pm1 = bool(abs(abs(tr["Tr"]) - 1.0) <= TOL)  # OL-AXIS
            material = bool(tr_is_pm1 and (inc or dec) and all(v["median"] >= DELTA_B for v in gaps.values()))
            mat_db = {repr(float(db)): bool(tr_is_pm1 and (inc or dec) and all(v["median"] >= db for v in gaps.values()))  # OL-AXIS
                      for db in DBS}  # OL-AXIS
            tr = {k: v for k, v in tr.items() if k not in ("Tr_all", "obs_idx")}  # OL-AXIS
            return {"trend": tr, "gaps": gaps, "material": material, "abs_Tr_is_1_tol1e-9": tr_is_pm1,  # OL-AXIS
                    "material_by_delta_b": mat_db,  # OL-AXIS
                    "strictly_increasing_in_size_every_vit": inc, "strictly_decreasing_in_size_every_vit": dec,
                    "min_median_gap": float(min(v["median"] for v in gaps.values())),
                    "mean_over_vits_by_model": {s: float(G[r].mean()) for s, r in zip(stems, rows)},
                    "per_vit": {s: G[r].tolist() for s, r in zip(stems, rows)}}

        if not CFG["test"]:  # OL-AXIS
            assert sizes_B == sorted(sizes_B) and sizes_L == sorted(sizes_L)  # OL-AXIS
        ol_axis = {"Bbar": axis_block(Bbar, oi, OLS, sizes_O),  # OL-AXIS
                   "Bfd": {str(r): axis_block(Bfd[r], oi, OLS, sizes_O) for r in DEPTHS},  # OL-AXIS
                   "Blate": axis_block(Blate, oi, OLS, sizes_O),  # OL-AXIS
                   "spearman_size_vs_PRstar_LM": family_trend(  # OL-AXIS
                       np.array([[lm_info[s]["PR_star"]] for s in OLS]), sizes_O)["Tr"],  # OL-AXIS
                   "PRstar_LM": {s: lm_info[s]["PR_star"] for s in OLS},  # OL-AXIS
                   "sqrt_PRstar_ratio": {f"{SIZES[x]}|{SIZES[y]}": math.sqrt(lm_info[OLS[y]]["PR_star"] /  # OL-AXIS
                                                                            lm_info[OLS[x]]["PR_star"])  # OL-AXIS
                                         for x, y, _ in PAIRS3},  # OL-AXIS
                   "threshold_setting_LM_layer": {s: {"u_star": lm_info[s]["argmax"], "L_A": lm_info[s]["L"],  # OL-AXIS
                                                      "u_star_rel": lm_info[s]["argmax_rel"],  # OL-AXIS
                                                      "runner_up": lm_info[s]["argmax_second"],  # OL-AXIS
                                                      "PR_star_over_runner_up": lm_info[s]["PR_star_over_second"],  # OL-AXIS
                                                      "u_late": lm_info[s]["u_late"],  # OL-AXIS
                                                      "u_late_rel": lm_info[s]["u_late_rel"],  # OL-AXIS
                                                      "Blate_equals_Bbar": bool(lm_info[s]["argmax_rel"] >= LATE_REL)}  # OL-AXIS
                                                  for s in OLS},  # OL-AXIS
                   "PR_13b_ctm_band": {"band": list(CTM_BAND), "layers": lm_info[ol["13b"]]["ctm_band_0.84_0.90"],  # OL-AXIS
                                       "max": lm_info[ol["13b"]]["ctm_band_PR_max"],  # OL-AXIS
                                       "median": lm_info[ol["13b"]]["ctm_band_PR_median"]}}  # OL-AXIS
        # ViT-side argmax w* (common to all LMs)
        vit_star = {s: {"w_star": vit_info[s]["argmax"], "L_B": vit_info[s]["L"], "w_star_rel": vit_info[s]["argmax_rel"]}
                    for s in vit_stems}
        # B5 pooled ARH-style r of Bbar (187 pairs; 10- and 8-LM panels for comparison)  # OL-AXIS
        def pearson_rank(rows):
            x = np.repeat(np.array([CFG["arh_rank"][lm_stems[i]] for i in rows], float)[:, None], len(vit_stems), axis=1).ravel()  # OL-AXIS
            y = Bbar[rows].ravel()
            return float(np.corrcoef(x, y)[0, 1])
        i10 = [i for i, s in enumerate(lm_stems) if s != ol["13b"]]  # OL-AXIS
        i8 = [i for i, s in enumerate(lm_stems) if s not in (ol["13b"], LLAMA[1], LLAMA[2])]  # OL-AXIS
        xr = np.repeat(np.arange(1, len(lm_stems) + 1, dtype=float)[:, None], len(vit_stems), axis=1).ravel()  # OL-AXIS
        xl = np.repeat(np.log(np.array([nparams[s] for s in lm_stems], float))[:, None], len(vit_stems), axis=1).ravel()
        pooled = {"pearson_arh_rank_Bbar_11LM": pearson_rank(list(range(len(lm_stems)))),  # OL-AXIS
                  "pearson_arh_rank_Bbar_10LM": pearson_rank(i10),  # OL-AXIS
                  "pearson_arh_rank_Bbar_8LM": pearson_rank(i8),  # OL-AXIS
                  "pearson_dense_rank_Bbar_11LM": float(np.corrcoef(xr, Bbar.ravel())[0, 1]),  # OL-AXIS
                  "pearson_log_params_Bbar_11LM": float(np.corrcoef(xl, Bbar.ravel())[0, 1]),  # OL-AXIS
                  "n_pairs_11LM": int(Bbar.size),  # OL-AXIS
                  "mean_Bbar_by_lm": {s: float(Bbar[i].mean()) for i, s in enumerate(lm_stems)}}
        bloomz_retro = {"label": "retrodiction, outcome known, not a test",
                        "Bbar": axis_block(Bbar, bi, BLOOMZ, sizes_B),
                        "Bfd": {str(r): axis_block(Bfd[r], bi, BLOOMZ, sizes_B) for r in DEPTHS},
                        "threshold_setting_LM_layer": {s: {"u_star": lm_info[s]["argmax"], "L_A": lm_info[s]["L"],
                                                           "u_star_rel": lm_info[s]["argmax_rel"]} for s in BLOOMZ}}
        G_cB = score_grid(FROZEN8, pre, BLOOMZ, "cka_biased", "c_alpha", used_scores)  # OL-AXIS
        bloomz_retro["Tr_BLOOMZ_calpha"] = family_trend(G_cB, sizes_B)["Tr"]  # OL-AXIS
        # B5 retrodictions on known pairs: c_alpha order vs Bbar order  # OL-AXIS
        kb, k3 = CFG["known_src"]["3b"]  # OL-AXIS
        _, k7 = CFG["known_src"]["7b"]  # OL-AXIS
        known = {}  # OL-AXIS: spec section 8 "known" (frozen layer-max values, cka_biased, full)
        for fld, key in (("s_obs", "T"), ("c_alpha", "c_alpha"), ("null_mean", "null_mean"), ("s_cal", "s_cal"),  # OL-AXIS
                         ("uncensored", "uncensored")):  # OL-AXIS
            G2 = score_grid(kb, pre, [k3, k7], "cka_biased", fld, used_scores)  # OL-AXIS
            known[key] = {"3b": dict(zip(vit_stems, G2[0].tolist())), "7b": dict(zip(vit_stems, G2[1].tolist()))}  # OL-AXIS
        c37 = np.array([[known["c_alpha"][a][v] for v in vit_stems] for a in ("3b", "7b")])  # OL-AXIS
        b37 = Bbar[oi[:2]]  # OL-AXIS
        agree37 = int(np.sum(np.sign(c37[1] - c37[0]) == np.sign(b37[1] - b37[0])))  # OL-AXIS
        q37 = np.abs(b37[1] - b37[0]) >= DELTA_B  # OL-AXIS
        G_cL = score_grid(LLAMA_E, pre, LLAMA, "cka_biased", "c_alpha", used_scores)  # OL-AXIS
        agreeL = nL = 0  # OL-AXIS
        for x, y, _ in PAIRS3:  # OL-AXIS
            dB_ = Bbar[li[y]] - Bbar[li[x]]  # OL-AXIS
            dc_ = G_cL[y] - G_cL[x]  # OL-AXIS
            q = np.abs(dB_) >= DELTA_B  # OL-AXIS
            nL += int(q.sum())  # OL-AXIS
            agreeL += int(np.sum(np.sign(dB_[q]) == np.sign(dc_[q])))  # OL-AXIS
        pilot_off = (c37 - b37).ravel()  # OL-AXIS: c_alpha - Bbar on the 34 pilot pairs
        mu37 = np.array([[known["null_mean"][a][v] for v in vit_stems] for a in ("3b", "7b")])  # OL-AXIS
        pilot_mu = (mu37 - b37).ravel()  # OL-AXIS
        offL = (G_cL - Bbar[li]).ravel()  # OL-AXIS
        retro = {"label": "retrodiction, outcome known, not a test",  # OL-AXIS
                 "openllama_3b_7b_calpha_vs_Bbar_order": {"agree": agree37, "n": len(vit_stems),  # OL-AXIS
                                                          "agree_gap_ge_delta_b": int(np.sum((np.sign(c37[1] - c37[0]) == np.sign(b37[1] - b37[0]))[q37])),  # OL-AXIS
                                                          "n_gap_ge_delta_b": int(q37.sum())},  # OL-AXIS
                 "llama_calpha_vs_Bbar_order_gap_ge_delta_b": {"agree": agreeL, "n": nL},  # OL-AXIS
                 "bloomz_Tr_Bbar": bloomz_retro["Bbar"]["trend"]["Tr"], "bloomz_Tr_calpha": bloomz_retro["Tr_BLOOMZ_calpha"],  # OL-AXIS
                 "pilot_calpha_minus_Bbar": {"median": float(np.median(pilot_off)), "min": float(pilot_off.min()),  # OL-AXIS
                                             "max": float(pilot_off.max()),  # OL-AXIS
                                             "max_abs_dev_from_median": float(np.max(np.abs(pilot_off - np.median(pilot_off)))),  # OL-AXIS
                                             "per_pair": pilot_off.tolist()},  # OL-AXIS
                 "pilot_null_mean_minus_Bbar": {"median": float(np.median(pilot_mu)), "min": float(pilot_mu.min()),  # OL-AXIS
                                                "max": float(pilot_mu.max())},  # OL-AXIS
                 "llama_calpha_minus_Bbar": {"median": float(np.median(offL)), "min": float(offL.min()),  # OL-AXIS
                                             "max": float(offL.max()), "n": int(offL.size)}}  # OL-AXIS
        # B6: s_O, band, boundaries  # OL-AXIS
        ob = ol_axis["Bbar"]  # OL-AXIS
        Tr_pred = float(ob["trend"]["Tr"])  # OL-AXIS
        s_O = sgn(Tr_pred) if sgn(Tr_pred) != 0 else 1  # OL-AXIS: Tr_pred = 0 -> +1, flagged
        s_O_J = s_O if ob["material"] else 1  # OL-AXIS
        Tr_late = float(ol_axis["Blate"]["trend"]["Tr"])  # OL-AXIS
        s_late = sgn(Tr_late) if sgn(Tr_late) != 0 else 1  # OL-AXIS
        if pre == "prh_clip+l2":  # OL-AXIS: numeric constants of the plan (R.1.4)
            band = dict(BAND_PRIMARY, source="plan section 9 B6 constants (approved, R.1.4)")  # OL-AXIS
        else:  # OL-AXIS: plan section 9 'l2 gets the same rules with its own quantities (its own pilot offsets for the band)'
            o2 = round(float(np.median(pilot_off)), 4)  # OL-AXIS: [op] own pilot median, rounded as 0.00192 -> 0.0019
            band = {"o": o2, "o_lo": round(o2 - BAND_PRIMARY["h"], 4), "o_hi": round(o2 + BAND_PRIMARY["h"], 4),  # OL-AXIS
                    "h": BAND_PRIMARY["h"],  # OL-AXIS
                    "source": "[op] l2: own 34 pilot offsets (median rounded to 4 dp), half-width h = 0.0010 as primary"}  # OL-AXIS
        B13 = Bbar[oi[2]]  # OL-AXIS
        Tk = {a: np.array([known["T"][a][v] for v in vit_stems]) for a in ("3b", "7b")}  # OL-AXIS
        ck = {a: np.array([known["c_alpha"][a][v] for v in vit_stems]) for a in ("3b", "7b")}  # OL-AXIS
        muk = {a: np.array([known["null_mean"][a][v] for v in vit_stems]) for a in ("3b", "7b")}  # OL-AXIS
# OL-AXIS
        def kappa(a, o):  # OL-AXIS: plan B6 kappa_v(j; o)
            return Tk[a] + (1.0 - Tk[a]) * (B13 + o - ck[a]) / (1.0 - ck[a])  # OL-AXIS
# OL-AXIS
        def vd(arr):  # OL-AXIS
            return dict(zip(vit_stems, [float(x) for x in arr]))  # OL-AXIS
        bnd = {"s_cal": {"lo": {a: vd(kappa(a, band["o_lo"])) for a in ("3b", "7b")},  # OL-AXIS
                         "hi": {a: vd(kappa(a, band["o_hi"])) for a in ("3b", "7b")}},  # OL-AXIS
               "unc": {"lo": {a: vd(Tk[a] + B13 + MU_BAND[0] - muk[a]) for a in ("3b", "7b")},  # OL-AXIS
                       "hi": {a: vd(Tk[a] + B13 + MU_BAND[1] - muk[a]) for a in ("3b", "7b")}},  # OL-AXIS
               "s_cal_h": {repr(float(h)): {"lo": {a: vd(kappa(a, band["o"] - h)) for a in ("3b", "7b")},  # OL-AXIS
                                            "hi": {a: vd(kappa(a, band["o"] + h)) for a in ("3b", "7b")}}  # OL-AXIS
                           for h in HS}}  # OL-AXIS
        censored = {a: [v for v in vit_stems if known["s_cal"][a][v] == 0.0] for a in ("3b", "7b")}  # OL-AXIS
        # new comparisons qualifying for P-O2a(ii) / P-O5a (features only; delta_b curve)  # OL-AXIS
        def n_qual(G, db):  # OL-AXIS
            return int(sum(np.sum(np.abs(G[oi[2]] - G[oi[k]]) >= db) for k in (0, 1)))  # OL-AXIS
        quals = {"PO2aii_n_Bbar_gap_ge_delta_b": n_qual(Bbar, DELTA_B),  # OL-AXIS
                 "PO2aii_need": int(math.ceil(2 * n_qual(Bbar, DELTA_B) / 3)),  # OL-AXIS
                 "PO5a_n_Blate_gap_ge_delta_b": n_qual(Blate, DELTA_B),  # OL-AXIS
                 "PO5a_need": int(math.ceil(2 * n_qual(Blate, DELTA_B) / 3)),  # OL-AXIS
                 "curve_PO2aii_n_by_delta_b": {repr(float(db)): n_qual(Bbar, db) for db in DBS}}  # OL-AXIS
        # n_B(t), n_L(t) and the J_min tables (known BLOOMZ / LLaMA; s_B = s_L = +1)  # OL-AXIS
        jt = {}  # OL-AXIS
        for x in ("s_cal", "unc"):  # OL-AXIS
            dB = relabel_dtr(FROZEN8, pre, BLOOMZ, x, used_scores)  # OL-AXIS
            dL = relabel_dtr(LLAMA_E, pre, LLAMA, x, used_scores)  # OL-AXIS
            BL = (dB[:, None] + dL[None, :]) / 2  # OL-AXIS
# OL-AXIS
            def nf(arr, t):  # OL-AXIS
                return int(np.sum(arr <= t + TOL))  # OL-AXIS
            rows = []  # OL-AXIS
            grid = sorted(set([round(v, 12) for v in JT_ROWS] + [round(k / 34, 12) for k in range(-68, 69)]))  # OL-AXIS
            for xo in grid:  # OL-AXIS: xo = s_O_J * dTr_O(id)
                t = max(xo, float(dB[0]), float(dL[0]))  # OL-AXIS
                nb, nl = nf(dB, t), nf(dL, t)  # OL-AXIS
                rows.append({"sO_dTrO": xo, "t": t, "n_B": nb, "n_L": nl, "plan_row": xo in [round(v, 12) for v in JT_ROWS],  # OL-AXIS
                             "p_for_nO": {str(n): nb * nl * n / NREL for n in range(1, 7)}})  # OL-AXIS
            jt[x] = {"dTr_B_id": float(dB[0]), "rank_B": nf(dB, float(dB[0])), "n_B_relabel": int(dB.size),  # OL-AXIS
                     "dTr_L_id": float(dL[0]), "rank_L": nf(dL, float(dL[0])), "n_L_relabel": int(dL.size),  # OL-AXIS
                     "dTr_L_all_sorted": sorted(float(v) for v in dL),  # OL-AXIS
                     "BL_mean_rank_720": int(np.sum(BL <= BL[0, 0] + TOL)),  # OL-AXIS
                     "Jmean_min_p_bound": int(np.sum(BL <= BL[0, 0] + TOL)) / NREL,  # OL-AXIS
                     "s_O_J": s_O_J, "rows": rows,  # OL-AXIS
                     "note": "J_min p = n_B(t) n_L(t) n_O / 4320 with t = max(s_O dTr_O, dTr_B(id), dTr_L(id)); "  # OL-AXIS
                             "n_O <= 3 for t < 0 (relabellings form +- pairs); EXPLORATORY"}  # OL-AXIS
        gapsd = {}  # OL-AXIS: spec section 8 'gaps'
        for name, blk in (("Bbar", ob), ("Bfd_0.5", ol_axis["Bfd"]["0.5"]), ("Bfd_1.0", ol_axis["Bfd"]["1.0"]),  # OL-AXIS
                          ("Blate", ol_axis["Blate"])):  # OL-AXIS
            gapsd[name] = {}  # OL-AXIS
            for x, y, key in PAIRS3:  # OL-AXIS
                g = blk["gaps"][f"{OLS[x]}|{OLS[y]}"]  # OL-AXIS
                gapsd[name][key] = {"per_vit": g["signed_larger_minus_smaller_per_vit"], "median": g["median_signed"],  # OL-AXIS
                                    "median_abs": g["median"], "n_vit_abs_ge_delta_b": g["n_vit_gap_ge_delta_b"]}  # OL-AXIS
        allst = lm_stems + vit_stems  # OL-AXIS
        info_all = dict(lm_info, **vit_info)  # OL-AXIS
        out["per_pre"][pre] = {  # OL-AXIS: spec section 8 fields first, then the full detail
            "clip_c": {s: (clip[s]["c"] if pre == "prh_clip+l2" else None) for s in allst},  # OL-AXIS
            "PR": {s: info_all[s]["PR"] for s in allst},  # OL-AXIS
            "PR_star": {s: info_all[s]["PR_star"] for s in allst},  # OL-AXIS
            "u_star": {s: info_all[s]["argmax"] for s in allst},  # OL-AXIS
            "u_star_rel": {s: info_all[s]["argmax_rel"] for s in allst},  # OL-AXIS
            "Bbar": {a: {b: float(Bbar[i, j]) for j, b in enumerate(vit_stems)} for i, a in enumerate(lm_stems)},  # OL-AXIS
            "Bbar_argmax": {a: {b: [lm_info[a]["argmax"], vit_info[b]["argmax"]] for b in vit_stems} for a in lm_stems},  # OL-AXIS
            "Bfd": {str(r): {a: {b: float(Bfd[r][i, j]) for j, b in enumerate(vit_stems)}  # OL-AXIS
                             for i, a in enumerate(lm_stems)} for r in DEPTHS},  # OL-AXIS
            "Blate": {a: {b: float(Blate[i, j]) for j, b in enumerate(vit_stems)} for i, a in enumerate(lm_stems)},  # OL-AXIS
            "gaps": gapsd,  # OL-AXIS
            "Tr_pred": Tr_pred, "material": ob["material"], "s_O": s_O, "s_O_J": s_O_J,  # OL-AXIS
            "s_O_flag_Tr_pred_zero": bool(sgn(Tr_pred) == 0),  # OL-AXIS
            "Tr_Bfd": {str(r): float(ol_axis["Bfd"][str(r)]["trend"]["Tr"]) for r in DEPTHS},  # OL-AXIS
            "material_Bfd": {str(r): ol_axis["Bfd"][str(r)]["material"] for r in DEPTHS},  # OL-AXIS
            "Tr_Blate": Tr_late, "material_Blate": ol_axis["Blate"]["material"], "s_late": s_late,  # OL-AXIS
            "s_late_flag_Tr_zero": bool(sgn(Tr_late) == 0),  # OL-AXIS
            "known": {k: known[k] for k in ("T", "c_alpha", "null_mean", "s_cal")},  # OL-AXIS
            "known_uncensored": known["uncensored"],  # OL-AXIS
            "known_source": {"dir": os.path.relpath(kb, PROJ), "3b": k3, "7b": k7},  # OL-AXIS
            "boundaries": bnd,  # OL-AXIS
            "band": band, "mu_band": list(MU_BAND),  # OL-AXIS
            "s_cal_censored_known": censored,  # OL-AXIS
            "qualifying": quals,  # OL-AXIS
            "material_curve_db": {k: v for k, v in ob["material_by_delta_b"].items()},  # OL-AXIS
            "J_tables": jt,  # OL-AXIS
            "retro": retro,  # OL-AXIS
            "lm": lm_info, "vit": vit_info, "vit_star": vit_star,  # OL-AXIS
            "Bbar_factorised_vs_bruteforce_maxabs": Bbar_bruteforce_maxdiff,  # OL-AXIS
            "Blate_factorised_vs_bruteforce_maxabs": Blate_bruteforce_maxdiff,  # OL-AXIS
            "ol_axis": ol_axis, "pooled": pooled, "bloomz_retro": bloomz_retro}  # OL-AXIS
        print(f"== {pre}: OL Tr(Bbar) {Tr_pred:+.2f} material {ob['material']} s_O {s_O:+d} "  # OL-AXIS
              f"means {[round(v, 4) for v in ob['mean_over_vits_by_model'].values()]} "  # OL-AXIS
              f"| u* {[(d['u_star'], round(d['u_star_rel'], 2)) for d in ol_axis['threshold_setting_LM_layer'].values()]} "  # OL-AXIS
              f"| Tr(Bfd0.5) {ol_axis['Bfd']['0.5']['trend']['Tr']:+.2f} Tr(Bfd1.0) {ol_axis['Bfd']['1.0']['trend']['Tr']:+.2f} "  # OL-AXIS
              f"| Tr(Blate) {Tr_late:+.2f} mat {ol_axis['Blate']['material']} "  # OL-AXIS
              f"| BLOOMZ Tr(Bbar) {bloomz_retro['Bbar']['trend']['Tr']:+.2f} | pooled r {pooled['pearson_arh_rank_Bbar_11LM']:+.3f} "  # OL-AXIS
              f"| pilot c-Bbar median {retro['pilot_calpha_minus_Bbar']['median']:.5f} | 13B PR* {lm_info[ol['13b']]['PR_star']:.2f}",  # OL-AXIS
              flush=True)
    out["meta"]["score_inputs"] = {p: sha256_file(os.path.join(PROJ, p)) for p in sorted(used_scores)}  # OL-AXIS
    out["meta"]["n_score_inputs"] = len(used_scores)  # OL-AXIS
    # ---------------- B0 end scan, then B7 writes ------------------------------  # OL-AXIS
    scan_end = run_scan("end")  # OL-AXIS
    out["meta"]["guard_scan_end"] = scan_end  # OL-AXIS
    out["meta"]["finished_utc"] = utc()
    with open(os.path.join(OUTDIR, "clip_thresholds_partB_ol.json"), "w") as f:  # OL-AXIS
        json.dump(jsonable({"note": "hrhv2.prep.prh_clip_threshold on the float64 all-layer stack (D5), q=0.95; "  # OL-AXIS
                                    "known models checked bitwise vs the frozen LLaMA Part B (meta.clip_thresholds)",  # OL-AXIS
                            "clip": clip, "written_utc": utc()}), f, indent=1)  # OL-AXIS
    with open(os.path.join(OUTDIR, "partB_features_ol.json.tmp"), "w") as f:  # OL-AXIS
        json.dump(jsonable(out), f, indent=1)
    os.replace(os.path.join(OUTDIR, "partB_features_ol.json.tmp"), os.path.join(OUTDIR, "partB_features_ol.json"))  # OL-AXIS
    print(f"[{time.time() - t0:6.0f}s] wrote {os.path.join(OUTDIR, 'partB_features_ol.json')}", flush=True)  # OL-AXIS
    js_sha = sha256_file(os.path.join(OUTDIR, "partB_features_ol.json"))  # OL-AXIS
    with open(CFG["md"], "w") as f:  # OL-AXIS
        f.write("\n".join(build_md(json.load(open(os.path.join(OUTDIR, "partB_features_ol.json"))), js_sha)) + "\n")  # OL-AXIS
    print(f"[{time.time() - t0:6.0f}s] wrote DRAFT {CFG['md']} (review, chmod 444, STATUS line: see header)", flush=True)  # OL-AXIS
# OL-AXIS
# OL-AXIS
# ---------------------------------------------------------------------------------------------------  # OL-AXIS
# B7 draft of PARTB_PREDICTIONS_OL.md (numbers from the JSON just written)  # OL-AXIS
# ---------------------------------------------------------------------------------------------------  # OL-AXIS
# OL-AXIS
def _f(x, nd=4):  # OL-AXIS
    return "n/a" if x is None else f"{x:.{nd}f}"  # OL-AXIS
# OL-AXIS
# OL-AXIS
def _short(s: str) -> str:  # OL-AXIS
    return s.split("__")[-1]  # OL-AXIS
# OL-AXIS
# OL-AXIS
def build_md(F: dict, js_sha: str) -> list[str]:  # OL-AXIS
    """Plain draft text of PARTB_PREDICTIONS_OL.md; every number is read from partB_features_ol.json."""  # OL-AXIS
    OLS = F["ol_lms"]  # OL-AXIS
    V = F["vits"]  # OL-AXIS
    m = F["meta"]  # OL-AXIS
    L = []  # OL-AXIS
    a = L.append  # OL-AXIS
    a("# S1.2 OpenLLaMA axis, Part B: instantiated rules from features only (DRAFT v1, generated by partb_ol.py)")  # OL-AXIS
    a("")  # OL-AXIS
    a("Role `oll-axis-predict`. B7 output of `analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md` (Part A, sha256 "  # OL-AXIS
      f"`{m['plan_sha256']}`), section 9. Contract: `VERDICTS_SPEC.md` section 8. **DRAFT: the predict role reviews the wording, "  # OL-AXIS
      "chmods this file and the JSON 444 and logs the `[oll-axis-predict] Part B FROZEN` STATUS line.** Every number below is "  # OL-AXIS
      f"read from `results/S1_2_layermax_openllama/partB/partB_features_ol.json` (sha256 `{js_sha}` at generation).")  # OL-AXIS
    if m.get("test_mode"):  # OL-AXIS
        a("")  # OL-AXIS
        a(f"**S0b TEST OUTPUT, NOT A PART B.** Stand-in `{m.get('standin')}`: {json.dumps(m.get('ol_size_labels'))}.")  # OL-AXIS
    a("")  # OL-AXIS
    a("## 0. Freeze conditions and provenance")  # OL-AXIS
    a("")  # OL-AXIS
    for tag in ("guard_scan_start", "guard_scan_end"):  # OL-AXIS
        s = m.get(tag, {})  # OL-AXIS
        a(f"- **{tag.replace('_', ' ')}** ({s.get('utc')}): clean = {s.get('clean')}; `find ... -name '*open_llama_13b__*'` hits = "  # OL-AXIS
          f"{s.get('n_find_hits')}; results top level = {s.get('results_top_level')}; forbidden processes = "  # OL-AXIS
          f"{len(s.get('forbidden_processes', []))}. Full output: `{os.path.relpath(s.get('log', ''), PROJ) if s.get('log') else ''}`."  # OL-AXIS
          + (f" **SCAN ERROR: {s.get('scan_error')}**" if s.get("scan_error") else ""))  # OL-AXIS
    a("- **No cross-model score involving open_llama_13b existed or was computed.** Inputs: the feature stacks "  # OL-AXIS
      f"({len(m['inputs'])} entries, sha256 in `meta.inputs`), the frozen LLaMA Part B `{m['llama_partB']['path']}` "  # OL-AXIS
      f"(`{m['llama_partB']['sha256'][:8]}…`), and {m['n_score_inputs']} frozen score files of BLOOMZ, LLaMA-1 and "  # OL-AXIS
      "OpenLLaMA 3B/7B (`meta.score_inputs`).")  # OL-AXIS
    r27 = m["recheck_27"]  # OL-AXIS
    a(f"- **B1 recomputation of the {r27['n_models']} known models:** max relative PR error {r27['max_rel_err_PR']:.2e} "  # OL-AXIS
      f"(≤ 1e-14 required); clip thresholds bitwise equal: {r27['all_clip_bitwise']}; pass: {r27['pass']}.")  # OL-AXIS
    s13 = OLS[2]  # OL-AXIS
    a(f"- **13B clip threshold** (`hrhv2.prep.prh_clip_threshold`, float64 all-layer stack): {m['clip_thresholds'][s13]!r}, "  # OL-AXIS
      "written to `clip_thresholds_partB_ol.json`; V-O1 must reproduce it bitwise.")  # OL-AXIS
    a(f"- **Code:** `analysis/s12_layermax_openllama/partb_ol.py` sha256 `{m['script_sha256']}`; threads {m['threads']}; "  # OL-AXIS
      f"python {m['python']}, numpy {m['numpy']}; run {m['started_utc']} – {m['finished_utc']}.")  # OL-AXIS
    a("")  # OL-AXIS
    for pre in ("prh_clip+l2", "l2"):  # OL-AXIS
        Q = F["per_pre"][pre]  # OL-AXIS
        prim = pre == "prh_clip+l2"  # OL-AXIS
        oa = Q["ol_axis"]  # OL-AXIS
        a(f"## {'1' if prim else '2'}. {'Primary' if prim else 'Secondary'} instance: {pre}")  # OL-AXIS
        a("")  # OL-AXIS
        a("| model | L_stored | PR* | u* (rel.) | runner-up ratio | PR(emb) | PR(last) | PR at r = 0.5 (layer) | PR_late* (u, rel.) |")  # OL-AXIS
        a("|---|---|---|---|---|---|---|---|---|")  # OL-AXIS
        for s in OLS:  # OL-AXIS
            q = Q["lm"][s]  # OL-AXIS
            a(f"| {_short(s)} | {q['L']} | {q['PR_star']:.2f} | {q['argmax']} ({q['argmax_rel']:.3f}) | {q['PR_star_over_second']:.4f} | "  # OL-AXIS
              f"{q['PR_first_layer']:.2f} | {q['PR_last_layer']:.2f} | {q['fixed']['0.5']['PR']:.2f} ({q['fixed']['0.5']['layer']}) | "  # OL-AXIS
              f"{q['PR_late_star']:.2f} ({q['u_late']}, {q['u_late_rel']:.3f}) |")  # OL-AXIS
        cb = oa["PR_13b_ctm_band"]  # OL-AXIS
        a("")  # OL-AXIS
        a(f"- 13B PR at relative depth 0.84–0.90 (descriptive): " + ", ".join(f"u={d['u']} ({d['rel']:.3f}): {d['PR']:.2f}" for d in cb["layers"])  # OL-AXIS
          + f"; max {_f(cb['max'], 2)}, median {_f(cb['median'], 2)}.")  # OL-AXIS
        ob = oa["Bbar"]  # OL-AXIS
        means = ob["mean_over_vits_by_model"]  # OL-AXIS
        a(f"- **B̄ means over 17 ViTs:** " + " / ".join(f"{means[s]:.4f}" for s in OLS) + "; ρ_v(B̄) common to all ViTs; "  # OL-AXIS
          f"**Tr_pred = Tr_O(B̄) = {Q['Tr_pred']:+.4f}**, so **s_O = {Q['s_O']:+d}**"  # OL-AXIS
          + (" (Tr_pred = 0: s_O := +1, FLAGGED)" if Q["s_O_flag_Tr_pred_zero"] else "") + ".")  # OL-AXIS
        a("")  # OL-AXIS
        a("| quantity | 3B–7B median gap (ViTs ≥ δ_b) | 7B–13B | 3B–13B | Tr_O | material (B4) |")  # OL-AXIS
        a("|---|---|---|---|---|---|")  # OL-AXIS
        for name, lab, trv, matv in (("Bbar", "B̄", Q["Tr_pred"], Q["material"]),  # OL-AXIS
                                     ("Bfd_0.5", "B_fd(0.5)", Q["Tr_Bfd"]["0.5"], Q["material_Bfd"]["0.5"]),  # OL-AXIS
                                     ("Bfd_1.0", "B_fd(1.0)", Q["Tr_Bfd"]["1.0"], Q["material_Bfd"]["1.0"]),  # OL-AXIS
                                     ("Blate", "B_late", Q["Tr_Blate"], Q["material_Blate"])):  # OL-AXIS
            g = Q["gaps"][name]  # OL-AXIS
            cells = [f"{g[k]['median']:+.4f} ({g[k]['n_vit_abs_ge_delta_b']}/17)" for k in ("3b|7b", "7b|13b", "3b|13b")]  # OL-AXIS
            a(f"| {lab} | " + " | ".join(cells) + f" | {trv:+.4f} | **{'MATERIAL' if matv else 'not material'}** |")  # OL-AXIS
        a("")  # OL-AXIS
        a(f"- Gaps are B(larger) − B(smaller), median over ViTs (materiality uses the median of |gap|, LP rule; equal when monotone). "  # OL-AXIS
          f"s_O_J = {Q['s_O_J']:+d}; s_late = {Q['s_late']:+d}; B_late = B̄ for the sizes with u*/(L−1) ≥ 0.60: "  # OL-AXIS
          + ", ".join(f"{_short(s)} {oa['threshold_setting_LM_layer'][s]['Blate_equals_Bbar']}" for s in OLS) + ".")  # OL-AXIS
        a("- δ_b curve (materiality of B̄): " + ", ".join(f"δ_b = {k}: {v}" for k, v in Q["material_curve_db"].items()) + ".")  # OL-AXIS
        if prim:  # OL-AXIS
            a("")  # OL-AXIS
            a("Per-ViT values (triples 3B / 7B / 13B):")  # OL-AXIS
            a("")  # OL-AXIS
            a("| ViT | PR* | w* (of 0..L_B−1) | B̄ | B_fd(0.5) | B_fd(1.0) | B_late |")  # OL-AXIS
            a("|---|---|---|---|---|---|---|")  # OL-AXIS
            for v in V:  # OL-AXIS
                q = Q["vit"][v]  # OL-AXIS
                trip = lambda D: " / ".join(f"{D[s][v]:.4f}" for s in OLS)  # noqa: E731  # OL-AXIS
                a(f"| {v} | {q['PR_star']:.1f} | {q['argmax']} of {q['L'] - 1} | {trip(Q['Bbar'])} | {trip(Q['Bfd']['0.5'])} | "  # OL-AXIS
                  f"{trip(Q['Bfd']['1.0'])} | {trip(Q['Blate'])} |")  # OL-AXIS
        a("")  # OL-AXIS
        rt = Q["retro"]  # OL-AXIS
        po = Q["pooled"]  # OL-AXIS
        a(f"**B5 (retrodictions and extras; outcomes known, not tests).** Pearson(ARH rank, B̄) on {po['n_pairs_11LM']} pairs "  # OL-AXIS
          f"{po['pearson_arh_rank_Bbar_11LM']:+.3f} (10 LMs {po['pearson_arh_rank_Bbar_10LM']:+.3f}, 8 LMs {po['pearson_arh_rank_Bbar_8LM']:+.3f}; "  # OL-AXIS
          f"dense ranks {po['pearson_dense_rank_Bbar_11LM']:+.3f}, log params {po['pearson_log_params_Bbar_11LM']:+.3f}). "  # OL-AXIS
          f"OpenLLaMA 3B–7B c_α order = B̄ order in {rt['openllama_3b_7b_calpha_vs_Bbar_order']['agree']}/{rt['openllama_3b_7b_calpha_vs_Bbar_order']['n']} ViTs; "  # OL-AXIS
          f"LLaMA {rt['llama_calpha_vs_Bbar_order_gap_ge_delta_b']['agree']}/{rt['llama_calpha_vs_Bbar_order_gap_ge_delta_b']['n']} comparisons (gap ≥ δ_b); "  # OL-AXIS
          f"BLOOMZ Tr(B̄) {rt['bloomz_Tr_Bbar']:+.2f} vs Tr(c_α) {rt['bloomz_Tr_calpha']:+.2f}. "  # OL-AXIS
          f"Pilot offsets c_α − B̄ (34 pairs): median {rt['pilot_calpha_minus_Bbar']['median']:.5f}, range "  # OL-AXIS
          f"[{rt['pilot_calpha_minus_Bbar']['min']:.5f}, {rt['pilot_calpha_minus_Bbar']['max']:.5f}]; null mean − B̄ median "  # OL-AXIS
          f"{rt['pilot_null_mean_minus_Bbar']['median']:+.2e} [{rt['pilot_null_mean_minus_Bbar']['min']:+.2e}, {rt['pilot_null_mean_minus_Bbar']['max']:+.2e}]; "  # OL-AXIS
          f"LLaMA offsets median {rt['llama_calpha_minus_Bbar']['median']:.5f} [{rt['llama_calpha_minus_Bbar']['min']:.5f}, {rt['llama_calpha_minus_Bbar']['max']:.5f}].")  # OL-AXIS
        a("- Mean B̄ by LM: " + ", ".join(f"{_short(s)} {v:.4f}" for s, v in po["mean_Bbar_by_lm"].items()) + ".")  # OL-AXIS
        a("")  # OL-AXIS
        bd = Q["band"]  # OL-AXIS
        qu = Q["qualifying"]  # OL-AXIS
        a(f"**B6 (instantiation for 13B).** Offset o = {bd['o']}, band [{bd['o_lo']}, {bd['o_hi']}], h = {bd['h']} ({bd['source']}); "  # OL-AXIS
          f"uncensored μ̂_13 = B̄_13 + [{Q['mu_band'][0]}, {Q['mu_band'][1]}]. Boundaries κ^lo/κ^hi (s_cal) and the uncensored "  # OL-AXIS
          "boundaries for the 34 new comparisons are in `per_pre.<pre>.boundaries`; they use the frozen 8-LM 3B/7B values "  # OL-AXIS
          f"(`per_pre.<pre>.known`, source `{Q['known_source']['dir']}`). Known censored s_cal (forecast 'undecided'): "  # OL-AXIS
          f"3B {len(Q['s_cal_censored_known']['3b'])}, 7B {len(Q['s_cal_censored_known']['7b'])}.")  # OL-AXIS
        for a_ in ("3b", "7b"):  # OL-AXIS
            lo = [Q["boundaries"]["s_cal"]["lo"][a_][v] for v in V]  # OL-AXIS
            hi = [Q["boundaries"]["s_cal"]["hi"][a_][v] for v in V]  # OL-AXIS
            a(f"- κ boundaries vs {a_.upper()}: κ^lo in [{min(lo):.4f}, {max(lo):.4f}], κ^hi − κ^lo in "  # OL-AXIS
              f"[{min(h - l for h, l in zip(hi, lo)):.5f}, {max(h - l for h, l in zip(hi, lo)):.5f}].")  # OL-AXIS
        a(f"- New comparisons with B̄ gap ≥ δ_b (P-O2a(ii)): n = {qu['PO2aii_n_Bbar_gap_ge_delta_b']}, need ⌈2n/3⌉ = {qu['PO2aii_need']}"  # OL-AXIS
          + (" (n < 6: 'not evaluable')" if qu["PO2aii_n_Bbar_gap_ge_delta_b"] < 6 else "") + "; with B_late gap ≥ δ_b (P-O5a): "  # OL-AXIS
          f"n = {qu['PO5a_n_Blate_gap_ge_delta_b']}, need {qu['PO5a_need']}"  # OL-AXIS
          + (" (n < 6: 'not evaluable')" if qu["PO5a_n_Blate_gap_ge_delta_b"] < 6 else "") + ".")  # OL-AXIS
        a("")  # OL-AXIS
    Q = F["per_pre"]["prh_clip+l2"]  # OL-AXIS
    a("## 3. J tables (EXPLORATORY; known BLOOMZ and LLaMA only; primary pre)")  # OL-AXIS
    a("")  # OL-AXIS
    for x in ("s_cal", "unc"):  # OL-AXIS
        j = Q["J_tables"][x]  # OL-AXIS
        a(f"- {x}: ΔTr_B(id) {j['dTr_B_id']:+.3f} (rank {j['rank_B']}/120), ΔTr_L(id) {j['dTr_L_id']:+.3f} (rank {j['rank_L']}/6), "  # OL-AXIS
          f"BLOOMZ+LLaMA mean rank {j['BL_mean_rank_720']}/720; s_O_J = {j['s_O_J']:+d}.")  # OL-AXIS
        a("")  # OL-AXIS
        a("  | s_O·ΔTr_O | t | n_B(t) | n_L(t) | J_min p for n_O = 1 / 2 / 3 |")  # OL-AXIS
        a("  |---|---|---|---|---|")  # OL-AXIS
        for r in j["rows"]:  # OL-AXIS
            if r["plan_row"]:  # OL-AXIS
                a(f"  | {r['sO_dTrO']:+.3f} | {r['t']:+.3f} | {r['n_B']} | {r['n_L']} | "  # OL-AXIS
                  f"{r['p_for_nO']['1']:.4f} / {r['p_for_nO']['2']:.4f} / {r['p_for_nO']['3']:.4f} |")  # OL-AXIS
        a("")  # OL-AXIS
    a("- Every J p is printed only with: the label **EXPLORATORY**; OpenLLaMA's own n_O/6; the conditional size; and "  # OL-AXIS
      "\"BLOOMZ hypothesis-generating; BLOOMZ fails B4; BLOOMZ and LLaMA known before this plan\". No decision is attached.")  # OL-AXIS
    a("")  # OL-AXIS
    s = Q["s_O"]  # OL-AXIS
    mat = Q["material"]  # OL-AXIS
    a("## 4. Instantiated rules (primary prh_clip+l2, biased CKA, layer-max; s_cal primary, uncensored secondary)")  # OL-AXIS
    a("")  # OL-AXIS
    a("No confirmatory test exists on this axis (R.1.2); every OpenLLaMA-only p is ≥ 1/6. Categories use τ = 0.10, δ_b = 0.005, h = 0.0010.")  # OL-AXIS
    a("")  # OL-AXIS
    a("- **P-O1 (mKNN; baseline 10/1023 for every pair).** |ΔTr_O(s_cal)| < 0.10 and |ΔTr_O(unc)| < 0.10; ζ check in exact rationals. "  # OL-AXIS
      "Bet beyond Prop. 5 (multiplicity only), plan §10 item 3.")  # OL-AXIS
    if mat:  # OL-AXIS
        a(f"- **P-O2 (biased CKA, layer-max): B̄ MATERIAL, s = {s:+d}.** Category: 'loss reproduced' if s·ΔTr_O(s_cal) ≤ −0.10; "  # OL-AXIS
          "'trend kept' if |ΔTr_O| < 0.10; 'moved with the baseline' if s·ΔTr_O ≥ +0.10 (plan §10 item 2, premise); "  # OL-AXIS
          f"'not testable' if Tr_O(raw) = {-s:+d}. Same for uncensored (secondary). Reported with F_new (n, x, p̂, CP). "  # OL-AXIS
          "No magnitude is predicted.")  # OL-AXIS
    else:  # OL-AXIS
        a("- **P-O2 (biased CKA, layer-max): B̄ NOT MATERIAL → no category is assigned**; the order changes of the new "  # OL-AXIS
          f"comparisons are reported (s_O := {s:+d} for F_new; J uses s_O_J = +1).")  # OL-AXIS
    a(f"- **P-O2a (premise, new pairs).** (i) median over the 17 new pairs of c_α − B̄ in [0.0009, 0.0029] (count inside the band "  # OL-AXIS
      f"with n, x, p̂, CP; LP's |·| ≤ 0.01 for comparability). (ii) c_α order = B̄ order in ≥ {Q['qualifying']['PO2aii_need']} of the "  # OL-AXIS
      f"{Q['qualifying']['PO2aii_n_Bbar_gap_ge_delta_b']} new comparisons with B̄ gap ≥ 0.005 (n < 6 → 'not evaluable'). Predicted order: "  # OL-AXIS
      + ("c_α(13B) > c_α(smaller)" if Q["Tr_pred"] > 0 else "c_α(13B) < c_α(smaller)" if Q["Tr_pred"] < 0 else "per ViT, as B̄") + ".")  # OL-AXIS
    a("- **P-O2b (primary decision rule).** For each of the 34 new comparisons (j ∈ {3B, 7B}, v): forecast 'below' (13B ranks "  # OL-AXIS
      "below j under s_cal) if T_13,v < κ^lo, 'above' if T_13,v > κ^hi, 'undecided' otherwise (and 'undecided' if the frozen "  # OL-AXIS
      "s_cal of j is 0). **Holds iff x = n and n ≥ 10; fails if x < n; not evaluable if n < 10.** Uncensored the same, secondary. "  # OL-AXIS
      "Applied by `forecast_ol.py` after the main run; it reads only this Part B JSON and the observed T/s_cal/null means.")  # OL-AXIS
    a(f"- **P-O3 (fixed depth r = 0.5, 1.0): descriptive.** B_fd material: r = 0.5 {Q['material_Bfd']['0.5']}, r = 1.0 "  # OL-AXIS
      f"{Q['material_Bfd']['1.0']} (Tr_O(B_fd) {Q['Tr_Bfd']['0.5']:+.3f}, {Q['Tr_Bfd']['1.0']:+.3f}). Keeps sign / lost / reversed is reported, not predicted.")  # OL-AXIS
    a("- **P-O4 (debiased CKA; baseline 0, Prop. 5(3)).** ΔTr_O(s_cal) ≥ −0.10 and ΔTr_O(unc) ≥ −0.10; on 11 LMs r_after ≥ r_before − 0.05.")  # OL-AXIS
    if Q["material_Blate"]:  # OL-AXIS
        a(f"- **P-O5 (calibrate-then-max, secondary): B_late MATERIAL, s_late = {Q['s_late']:+d}.** Prediction s_late·ΔTr_O(M) ≤ 0 with "  # OL-AXIS
          f"P-O2's categories; P-O5a premise: c at the M-argmax ordered as B_late in ≥ {Q['qualifying']['PO5a_need']} of "  # OL-AXIS
          f"{Q['qualifying']['PO5a_n_Blate_gap_ge_delta_b']} new comparisons with B_late gap ≥ 0.005, else 'premise failed'.")  # OL-AXIS
    else:  # OL-AXIS
        a(f"- **P-O5 (calibrate-then-max, secondary): B_late NOT MATERIAL → no category** (s_late = {Q['s_late']:+d} recorded).")  # OL-AXIS
    a("- **§7 readings and J** as in plan §7 and §5.4 (J EXPLORATORY, labels above).")  # OL-AXIS
    a("")  # OL-AXIS
    a("## 5. Summary")  # OL-AXIS
    a("")  # OL-AXIS
    a("| item | primary prh_clip+l2 | secondary l2 |")  # OL-AXIS
    a("|---|---|---|")  # OL-AXIS
    Q2 = F["per_pre"]["l2"]  # OL-AXIS
    for lab, f1 in (("Tr_pred = Tr_O(B̄)", lambda q: f"{q['Tr_pred']:+.4f}"), ("B̄ material", lambda q: str(q["material"])),  # OL-AXIS
                    ("s_O / s_O_J", lambda q: f"{q['s_O']:+d} / {q['s_O_J']:+d}"),  # OL-AXIS
                    ("B_fd material (0.5 / 1.0)", lambda q: f"{q['material_Bfd']['0.5']} / {q['material_Bfd']['1.0']}"),  # OL-AXIS
                    ("Tr_O(B_late), material, s_late", lambda q: f"{q['Tr_Blate']:+.4f}, {q['material_Blate']}, {q['s_late']:+d}"),  # OL-AXIS
                    ("band o [lo, hi]", lambda q: f"{q['band']['o']} [{q['band']['o_lo']}, {q['band']['o_hi']}]"),  # OL-AXIS
                    ("13B PR*", lambda q: f"{q['PR_star'][OLS[2]]:.2f}"),  # OL-AXIS
                    ("P-O2a(ii) n / need", lambda q: f"{q['qualifying']['PO2aii_n_Bbar_gap_ge_delta_b']} / {q['qualifying']['PO2aii_need']}")):  # OL-AXIS
        a(f"| {lab} | {f1(Q)} | {f1(Q2)} |")  # OL-AXIS
    a("")  # OL-AXIS
    a("## 6. Next steps allowed by the plan")  # OL-AXIS
    a("")  # OL-AXIS
    a("- S6 (single-model precompute, V-O1/V-O2, then the gated V-O3/V-O4/V-O7) and S7 start only after the "  # OL-AXIS
      "`[oll-axis-predict] Part B FROZEN` STATUS line exists with the sha256 of this file and of the JSON (guard F1(a)).")  # OL-AXIS
    return L  # OL-AXIS


if __name__ == "__main__":
    main()
