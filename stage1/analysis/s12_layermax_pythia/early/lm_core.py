"""ARH layer-max replication (S1.2 secondary, role arh-layermax-early): core.

PYTHIA-AXIS COPY (implementation E of analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md, sha256
6de89603...879e7b, section 3 "Implementations"; S0b sub-role pyt-code-E). Copied from
analysis/s12_layermax_openllama/early/lm_core.py (sha256 c1572fd9...). Every changed line carries the marker
"PYT-AXIS" and is listed in early/CHANGES.md: model list (Pythia 70M..6.9B, family Pythia), feature paths
(features/wit1024_pythia_axis/), output paths (results/S1_2_layermax_pythia/early, shared single-model cache
results/S1_2_layermax_pythia/cache), Part B reference paths, thread count from PYT_THREADS, the E gate (plan
section 4 F1(a): E's own Part B check plus the shared axis guard, fail closed), the S0b test hooks, the four
known pairs of V-P4, the 4 ViT families of plan section 1, GPU memory fraction 0.30 (plan section 11). The
S-E2 switch of the OpenLLaMA copy is removed (plan R.4: E2 is descriptive on this axis). The kernels
(gpu_layer, ModelGPU, pair_scores, run_pair) are unchanged.

This module holds constants and functions only; importing it reads no data and creates no file (plan section 4
F1(a)); it only reads STATUS.md to decide the S0b test mode.

Switches (environment):
  PYT_S0B_TEST=1  S0b unit tests only (refused, exit 3, once a '<UTC>Z [pyt-axis-<role>] CODE FROZEN' milestone
                  line exists in STATUS.md): every output root moves to PYT_E_TEST_ROOT (default
                  analysis/s12_layermax_pythia/s0b_scratch/E/run; it must lie under analysis/s12_layermax_pythia/
                  s0b_scratch/E/, the E sub-root of the shared guard's scratch root, VERDICTS_SPEC_PYT.md section 13),
                  the 7 Pythia slugs become the stand-ins synth__pyt-<size> whose feature stacks are named in the
                  JSON file PYT_E_TEST_STANDINS ({"<size>": "<path without .npy/.json>"}), and no file name of the
                  axis (EleutherAI__pythia-<size>__*) is ever written. PYT_E_TEST_PARTB=<dir> (test mode only, under
                  s0b_scratch/) reads Part B from <dir> (e.g. the shared stand-in s0b_scratch/PB/partB_vp9/).
  PYT_E_TEST_DEVICE=cpu  (test mode only) run the pair kernel on the CPU (CPU-only S0b tests while the GPU is
                  reserved for other jobs).

Plan: analysis/s12_layermax/ANALYSIS_PLAN_LAYERMAX.md (sha256 in STATUS.md).

GPU implementation of, for one (LM, ViT) pair and one row permutation pi of
the ViT items, the full L_A x L_B matrices of
  * biased linear CKA          (float64; SPEC 2.8, no epsilon)
  * debiased (Song) CKA        (float64; same estimator as hrhv2.geometric.cka_pair, on HKH)
  * mKNN overlap counts, k=10  (exact integers; SPEC s_N(10) = count / (m k))

  HSIC_pi[a,b]  = sum_ij Kc_A^a[i,j] Kc_B^b[pi(i),pi(j)]        (one batched GEMM)
  count_pi[a,b] = sum_ij M_A^a[i,j]  M_B^b[pi(i),pi(j)]         (0/1 kNN adjacency)

pi follows hrhv2 / ARH conventions: new ViT item i is old item pi[i].
Per-layer quantities (Gram, SPEC distance, kNN tables under the (D_ij, j)
tie rule, centred Gram) are computed on the GPU in float64 from rows
preprocessed exactly as hrhv2 (analysis/s12/common.preprocess_rows, checked
bitwise equal to hrhv2.prep in S1.2). kNN tables are validated against
hrhv2.structural.knn_table on every layer (V-P2, precompute.py); the validated
tables are stored in <CACHE>/knn_final/ and the main run uses those.
"""
from __future__ import annotations

import hashlib  # PYT-AXIS
import importlib.util  # PYT-AXIS
import json
import math
import os
import re  # PYT-AXIS
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_pythia")  # PYT-AXIS
EARLY_DIR = os.path.dirname(os.path.abspath(__file__))  # PYT-AXIS
if EARLY_DIR not in sys.path:  # PYT-AXIS
    sys.path.insert(0, EARLY_DIR)  # PYT-AXIS

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C12  # noqa: E402  (S1.2 registry / loaders; read-only import)
from hrhv2 import calibrate as cal  # noqa: E402
from hrhv2.prep import prh_clip_threshold  # noqa: E402

# ----------------------------------------------------------------------------
# PYT-AXIS: plan identity, STATUS milestone readers, S0b test mode, the E gate (plan section 4 F1(a))
# ----------------------------------------------------------------------------
PLAN = os.path.join(AXIS, "ANALYSIS_PLAN_PYTHIA.md")  # PYT-AXIS
PLAN_SHA256 = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"  # PYT-AXIS
STATUS = os.path.join(PROJ, "STATUS.md")  # PYT-AXIS
EXIT_GATE = 3  # PYT-AXIS
_TS = r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ "  # PYT-AXIS: a milestone line starts with its UTC time and the tag
CODE_FROZEN_RE = re.compile(_TS + r"\[pyt-axis-[A-Za-z0-9-]+\] CODE FROZEN")  # PYT-AXIS
PARTB_TAG = "[pyt-axis-predict] Part B FROZEN"  # PYT-AXIS: plan section 4 F1 gate line
PARTB_RE = re.compile(_TS + re.escape(PARTB_TAG))  # PYT-AXIS
SCRATCH = os.path.realpath(os.path.join(AXIS, "s0b_scratch"))  # PYT-AXIS: = guard.SCRATCH (VERDICTS_SPEC_PYT 13)
E_TEST_BASE = os.path.join(SCRATCH, "E")  # PYT-AXIS: every E test output lies here


def _refuse(msg: str):  # PYT-AXIS
    print(f"[early gate] REFUSED (exit {EXIT_GATE}): {msg}", file=sys.stderr, flush=True)
    sys.exit(EXIT_GATE)


def _status_lines() -> list[str]:  # PYT-AXIS
    with open(STATUS, encoding="utf-8") as f:
        return f.read().splitlines()


def code_frozen_line() -> str | None:  # PYT-AXIS
    for line in _status_lines():
        if CODE_FROZEN_RE.match(line):
            return line
    return None


def partb_line() -> str | None:  # PYT-AXIS: the last milestone line '<UTC>Z [pyt-axis-predict] Part B FROZEN'
    hits = [ln for ln in _status_lines() if PARTB_RE.match(ln)]
    return hits[-1] if hits else None


def _inside(path: str, root: str) -> bool:  # PYT-AXIS
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def _test_mode() -> bool:  # PYT-AXIS
    if os.environ.get("PYT_S0B_TEST") != "1":
        return False
    if code_frozen_line() is not None:
        _refuse("PYT_S0B_TEST=1 is refused: the S0b CODE FROZEN line exists in STATUS.md")
    return True


TEST = _test_mode()  # PYT-AXIS

PYT_RES = os.path.join(PROJ, "results", "S1_2_layermax_pythia")  # PYT-AXIS
if TEST:  # PYT-AXIS
    _troot = os.path.realpath(os.path.expanduser(os.environ.get("PYT_E_TEST_ROOT",  # PYT-AXIS
                                                                os.path.join(E_TEST_BASE, "run"))))  # PYT-AXIS
    if not _inside(_troot, E_TEST_BASE):  # PYT-AXIS
        _refuse(f"test root {_troot} is not under {E_TEST_BASE}")  # PYT-AXIS
    PYT_RES = _troot  # PYT-AXIS

# outputs of implementation E (plan section 4 F1 guarded directories); everything else below is read-only
RES = os.path.join(PYT_RES, "early")  # PYT-AXIS
RES_PRIMARY = RES  # PYT-AXIS: kept name (no S-E2 tree on this axis)
CACHE = os.path.join(RES, "cache")
SHARED_CACHE = os.path.join(PYT_RES, "cache")                # shared single-model cache (V-P1 thresholds)
PARTB = os.path.join(PYT_RES, "partB")                       # Part B (read-only for E)  PYT-AXIS
if TEST and os.environ.get("PYT_E_TEST_PARTB"):  # PYT-AXIS: a shared stand-in Part B (test mode only)
    _pb = os.path.realpath(os.environ["PYT_E_TEST_PARTB"])  # PYT-AXIS
    if not _inside(_pb, SCRATCH):  # PYT-AXIS
        _refuse(f"PYT_E_TEST_PARTB {_pb} is not under {SCRATCH}")  # PYT-AXIS
    PARTB = _pb  # PYT-AXIS
PARTB_JSON = os.path.join(PARTB, "partB_features_pyt.json")  # PYT-AXIS
PARTB_CLIP = os.path.join(PARTB, "clip_thresholds_partB_pyt.json")  # PYT-AXIS
PARTB_MD = (os.path.join(PYT_RES, "PARTB_PREDICTIONS_PYT.md") if TEST  # PYT-AXIS
            else os.path.join(AXIS, "PARTB_PREDICTIONS_PYT.md"))  # PYT-AXIS
PYT_INDEP_RES = os.path.join(PYT_RES, "indep")               # implementation I (read by vl5 only)  PYT-AXIS
PYT_FORECAST = os.path.join(PYT_RES, "forecast")             # forecast_pyt.py outputs (read by vl5 only)  PYT-AXIS
PYT_EXTRACTION = os.path.join(PYT_RES, "extraction")         # extraction checks (D7 sink statistics)  PYT-AXIS
FROZEN_RES = os.path.join(PROJ, "results", "S1_2_layermax")  # frozen 8-LM run (read-only)
LLAMA_E_RES = os.path.join(PROJ, "results", "S1_2_layermax_llama", "early")   # LLaMA-axis E (read-only)
LLAMA_MERGED = os.path.join(PROJ, "results", "S1_2_layermax_llama", "layermax_llama_summary.json")
OL_E_RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "early")  # PYT-AXIS: OpenLLaMA-axis E (read-only)
OL_MERGED = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "layermax_openllama_summary.json")  # PYT-AXIS
OL_PARTB_JSON = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "partB", "partB_features_ol.json")  # PYT-AXIS
M_ITEMS = 1024
KNN_K = 10
K_PERM = 500
ALPHA = 0.05
PRES = ("prh_clip+l2", "l2")
PRIMARY_PRE = "prh_clip+l2"
# the Pythia axis (plan section 1); ViTs unchanged  PYT-AXIS
SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")  # PYT-AXIS: size ranks 1..7 (num_params order)
SLUG_REAL = {s: f"EleutherAI__pythia-{s}" for s in SIZES}  # PYT-AXIS
SLUG_TEST = {s: f"synth__pyt-{s}" for s in SIZES}  # PYT-AXIS: S0b stand-ins (never an axis output name)
SLUG = SLUG_TEST if TEST else SLUG_REAL  # PYT-AXIS
FAMILY = "Pythia"  # PYT-AXIS
LMS = [(SLUG[s], FAMILY) for s in SIZES]  # PYT-AXIS
SIZE_LABEL = {SLUG[s]: s for s in SIZES}  # PYT-AXIS
NEW_LMS = tuple(SLUG[s] for s in SIZES)  # PYT-AXIS: every Pythia model is new (plan section 1)
VITS = C12.VITS
# PYT-AXIS: the 4 ViT families of plan section 1 (section 5.5(d)), names of VERDICTS_SPEC_PYT section 1
VIT_FAMILY4 = {s: ("CLIP" if f.startswith("CLIP") else {"AugReg-in21k": "in21k"}.get(f, f)) for s, f in VITS}  # PYT-AXIS
FEAT_PYT = os.path.join(PROJ, "features", "wit1024_pythia_axis")  # PYT-AXIS
FEAT_W1024 = os.path.join(C12.FEAT, "wit1024")
# PYT-AXIS: the four known pairs of V-P4 (plan section 11) and where their frozen inputs live
KNOWN4 = (  # PYT-AXIS: (LM, ViT)
    ("bigscience__bloomz-560m", "vit_tiny_patch16_224.augreg_in21k"),  # PYT-AXIS
    ("openlm-research__open_llama_3b", "vit_giant_patch14_dinov2.lvd142m"),  # PYT-AXIS
    ("huggyllama__llama-13b", "vit_huge_patch14_224.mae"),  # PYT-AXIS
    ("openlm-research__open_llama_13b", "vit_huge_patch14_clip_224.laion2b_ft_in12k"),  # PYT-AXIS
)  # PYT-AXIS
KNOWN_SRC = {  # PYT-AXIS: stem -> (feature base, frozen E pair dir root, frozen knn_final dir, clip reference)
    "bigscience__bloomz-560m": (os.path.join(FEAT_W1024, "bigscience__bloomz-560m"),  # PYT-AXIS
                                os.path.join(FROZEN_RES, "pairs"), os.path.join(FROZEN_RES, "cache", "knn_final"),  # PYT-AXIS
                                ("s12", None)),  # PYT-AXIS
    "openlm-research__open_llama_3b": (os.path.join(FEAT_W1024, "openlm-research__open_llama_3b"),  # PYT-AXIS
                                       os.path.join(FROZEN_RES, "pairs"),  # PYT-AXIS
                                       os.path.join(FROZEN_RES, "cache", "knn_final"), ("s12", None)),  # PYT-AXIS
    "huggyllama__llama-13b": (os.path.join(PROJ, "features", "wit1024_llama_axis", "huggyllama__llama-13b"),  # PYT-AXIS
                              os.path.join(LLAMA_E_RES, "pairs"), os.path.join(LLAMA_E_RES, "cache", "knn_final"),  # PYT-AXIS
                              ("partB", os.path.join(PROJ, "results", "S1_2_layermax_llama", "partB",  # PYT-AXIS
                                                     "clip_thresholds_partB.json"))),  # PYT-AXIS
    "openlm-research__open_llama_13b": (os.path.join(PROJ, "features", "wit1024_openllama_axis",  # PYT-AXIS
                                                     "openlm-research__open_llama_13b"),  # PYT-AXIS
                                        os.path.join(OL_E_RES, "pairs"), os.path.join(OL_E_RES, "cache", "knn_final"),  # PYT-AXIS
                                        ("partB", os.path.join(PROJ, "results", "S1_2_layermax_openllama", "partB",  # PYT-AXIS
                                                               "clip_thresholds_partB_ol.json"))),  # PYT-AXIS
}  # PYT-AXIS
TEST_STANDINS = os.environ.get("PYT_E_TEST_STANDINS")  # PYT-AXIS: JSON {"<size>": "<feature base>"} (test only)
DEV = "cpu" if (TEST and os.environ.get("PYT_E_TEST_DEVICE") == "cpu") else "cuda"  # PYT-AXIS
GPU_MEM_FRAC = 0.30  # PYT-AXIS: plan section 11 ("analysis processes use a GPU memory fraction <= 0.30")

# ARH LLM ordering: prh_models.get_models("val") order (ARH 785054b, SOG be654a37),
# = the x-axis of ARH Fig. 5 ("ranked according to their language performance").
ARH_LLM_ORDER = [
    "bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b", "bigscience/bloomz-7b1", "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b", "openlm-research/open_llama_13b", "huggyllama/llama-7b",
    "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b",
]
# PYT-AXIS: arh_rank / ARH_RANK / DENSE_RANK removed (Pythia is not in ARH's ranked list, plan R.4); the 11 known
# LMs keep ARH_LLM_ORDER for the descriptive cross-family table and the 18-LM panel only.


def perm_key(lm: str, vit: str) -> str:
    return f"S1.2-layermax|{lm}|{vit}"


def n_threads(default: int) -> int:
    """Thread count given by launch_pyt.sh (env PYT_THREADS); default only outside the launcher."""  # PYT-AXIS
    return max(1, int(os.environ.get("PYT_THREADS", default)))  # PYT-AXIS


def setup_torch(threads: int = 2, mem_frac: float = GPU_MEM_FRAC):  # PYT-AXIS: default fraction 0.30
    torch.set_num_threads(n_threads(threads))
    if DEV == "cuda" and torch.cuda.is_available():  # PYT-AXIS: never touch CUDA in the CPU test device
        torch.cuda.set_per_process_memory_fraction(min(mem_frac, GPU_MEM_FRAC), 0)  # PYT-AXIS
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    except Exception:
        pass


def synchronize():  # PYT-AXIS: torch.cuda.synchronize() only on the GPU device
    if DEV == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# PYT-AXIS: the E gate (plan section 4 F1(a)); every gated script calls gate() FIRST
# ----------------------------------------------------------------------------

def sha256_file(path: str) -> str:  # PYT-AXIS
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _shared_guard():  # PYT-AXIS: analysis/s12_layermax_pythia/guard.py, loaded by path (no shadowing); fail closed
    path = os.path.join(AXIS, "guard.py")
    if not os.path.isfile(path):
        _refuse(f"the shared axis guard {path} does not exist")
    try:
        spec = importlib.util.spec_from_file_location("pyt_axis_guard", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        _refuse(f"the shared axis guard {path} failed to import ({e!r})")
    if not hasattr(mod, "require_partb_frozen"):
        _refuse(f"the shared axis guard {path} has no require_partb_frozen()")
    return mod


def gate(script: str) -> dict:  # PYT-AXIS
    """Plan section 4 F1(a) for implementation E. Exits 3 (and has written nothing) unless:
      * S0b test mode (PYT_S0B_TEST=1, no CODE FROZEN line): the E output root lies under the E test base and the
        shared guard's require_partb_frozen() accepts it (its test mode: out_root under its scratch root); or
      * otherwise: (i) STATUS.md has a milestone line '<UTC>Z [pyt-axis-predict] Part B FROZEN' (the last one is
        used) that names the current sha256 of BOTH PARTB_PREDICTIONS_PYT.md and partB_features_pyt.json;
        (ii) both files exist; (iii) the E output root is results/S1_2_layermax_pythia/early; and (iv) the shared
        axis guard (analysis/s12_layermax_pythia/guard.py, plan section 3) exists and its
        require_partb_frozen(<E output root>) passes. A missing or failing shared guard refuses (fail closed)."""
    if TEST:
        if not _inside(RES, E_TEST_BASE):
            _refuse(f"{script}: test mode requires the E output root under {E_TEST_BASE}, got {RES}")
        info = _shared_guard().require_partb_frozen(RES, script=script)
        print(f"[early gate] S0B TEST MODE {script}: no CODE FROZEN line yet; outputs only under {PYT_RES}",
              file=sys.stderr, flush=True)
        return {"mode": "s0b-test", "out_root": os.path.realpath(RES), "shared_guard": info}
    if _inside(RES, E_TEST_BASE):
        _refuse(f"{script}: output root {RES} is the S0b test base but PYT_S0B_TEST is not set")
    line = partb_line()
    if line is None:
        _refuse(f"{script}: no '{PARTB_TAG}' milestone line in STATUS.md")
    sums = {}
    for path in (PARTB_MD, PARTB_JSON):
        if not os.path.isfile(path):
            _refuse(f"{script}: {path} missing")
        h = sha256_file(path)
        if h not in line:
            _refuse(f"{script}: sha256 {h} of {os.path.basename(path)} is not named in the Part B FROZEN line")
        sums[os.path.basename(path)] = h
    want = os.path.join(os.path.realpath(PROJ), "results", "S1_2_layermax_pythia", "early")
    if os.path.realpath(RES) != want:
        _refuse(f"{script}: E output root {RES} is not {want}")
    g = _shared_guard()
    info = g.require_partb_frozen(RES, script=script)
    out = {"mode": "partB-frozen", "out_root": os.path.realpath(RES), "partB_sha256": sums,
           "partB_line_utc": line.split(" ", 1)[0], "shared_guard": info if isinstance(info, dict) else str(info)}
    print(f"[early gate] PASS {script}: Part B frozen ({out['partB_line_utc']}), out_root {out['out_root']}",
          file=sys.stderr, flush=True)
    return out


# ----------------------------------------------------------------------------
# where things are (feature stacks, caches, pair files)
# ----------------------------------------------------------------------------

_STANDIN_MAP = None  # PYT-AXIS


def _standins() -> dict:  # PYT-AXIS: test-mode stand-in feature bases
    global _STANDIN_MAP
    if _STANDIN_MAP is None:
        if not (TEST and TEST_STANDINS and os.path.isfile(TEST_STANDINS)):
            raise RuntimeError("stand-in features need PYT_S0B_TEST=1 and PYT_E_TEST_STANDINS=<json>")
        with open(TEST_STANDINS) as f:
            _STANDIN_MAP = {SLUG_TEST[k]: v for k, v in json.load(f).items()}
    return _STANDIN_MAP


def _feat_base(stem: str) -> str:
    """Path prefix (without .npy/.json) of a model's WIT-1024 feature stack."""
    if TEST and stem in SLUG_TEST.values():  # PYT-AXIS
        return _standins()[stem]  # PYT-AXIS
    if stem in SLUG_REAL.values():  # PYT-AXIS
        return os.path.join(FEAT_PYT, stem)  # PYT-AXIS
    if stem in KNOWN_SRC:  # PYT-AXIS: V-P4 known LMs
        return KNOWN_SRC[stem][0]  # PYT-AXIS
    return os.path.join(FEAT_W1024, stem)


def cache_dir(stem: str) -> str:
    """Single-model cache of a model (E's own cache; plan section 3)."""  # PYT-AXIS
    return CACHE  # PYT-AXIS


def pairs_dir(pre: str, lm: str) -> str:
    """E pair files of a Pythia LM."""  # PYT-AXIS
    return os.path.join(RES, "pairs", pre)  # PYT-AXIS


def clip_value(stem: str):
    """E's own clip threshold of a model (precompute.py output clip.json of the right cache)."""
    with open(os.path.join(cache_dir(stem), "clip.json")) as f:
        return json.load(f)[stem]["c"]


def sidecar(stem: str) -> dict:
    with open(_feat_base(stem) + ".json") as f:
        return json.load(f)


def load_stack(stem: str) -> np.ndarray:
    """Memory-mapped (m, L, d) float32 stack."""
    return np.load(_feat_base(stem) + ".npy", mmap_mode="r")


def n_params(stem: str) -> int:
    return int(sidecar(stem)["num_params"])


def n_stored(stem: str) -> int:
    return int(sidecar(stem)["shape"][1])


def fixed_layer(L: int, r: float) -> int:
    """S1.2 primary rule: floor(r (L-1) + 0.5)."""
    return int(math.floor(r * (L - 1) + 0.5))


def grid_layers(L: int, n: int) -> list[int]:
    if n <= 1:
        return [L - 1]
    return sorted({fixed_layer(L, j / (n - 1)) for j in range(n)})


SUBSET_GRID_N = (2, 3, 5, 9, 17)


def subsets(LA: int, LB: int) -> dict:
    """Pre-specified layer subsets (plan section 7, A2). Values: (A idx, B idx)."""
    allA, allB = list(range(LA)), list(range(LB))
    out = {"full": (allA, allB)}
    for n in SUBSET_GRID_N:
        out[f"grid{n}"] = (grid_layers(LA, n), grid_layers(LB, n))
    out["fixed_r0.5"] = ([fixed_layer(LA, 0.5)], [fixed_layer(LB, 0.5)])
    out["fixed_r1.0"] = ([LA - 1], [LB - 1])
    out["LMall_ViTlast"] = (allA, [LB - 1])
    out["LMlast_ViTall"] = ([LA - 1], allB)
    # pre-specified descriptive subsets of ANALYSIS_PLAN_LLAMA section 3 (iii), kept on this axis
    # (same LM-layer sets as analysis/s12_layermax/posthoc_null_structure.lm_sets), all ViT layers
    out["noLMemb"] = (list(range(1, LA)), allB)
    out["LMdeep"] = ([a for a in range(LA) if a / (LA - 1) >= 0.5], allB)
    out["LMemb_only"] = ([0], allB)
    return out


def grid_saturated(L: int, n: int) -> bool:  # PYT-AXIS: plan section 3 (grid9 at 70m, grid17 at 70m/160m)
    """True if grid<n> uses every stored layer of a model with L stored layers (descriptive flag)."""
    return len(grid_layers(L, n)) == L


# ----------------------------------------------------------------------------
# preprocessing and per-layer GPU quantities
# ----------------------------------------------------------------------------

def clip_c(stem: str) -> float:
    """All-layer PRH/ARH threshold, recomputed with hrhv2 on the float64 stack."""
    st = np.asarray(load_stack(stem), dtype=np.float64)
    c = prh_clip_threshold(st)
    del st
    return float(c)


def s12_clip_c(stem: str) -> float:
    """Reference value for V-P1 (plan section 11): a Pythia model -> the frozen Part B value
    results/S1_2_layermax_pythia/partB/clip_thresholds_partB_pyt.json (format {"clip": {stem: {"c": c}}}, as on
    the OpenLLaMA axis); a V-P4 known LM -> its own frozen reference; every ViT -> the S1.2 cache
    results/S1_2/cache/clip_thresholds.json."""  # PYT-AXIS
    if stem in NEW_LMS:  # PYT-AXIS
        with open(PARTB_CLIP) as f:
            return float(json.load(f)["clip"][stem]["c"])
    if stem in KNOWN_SRC and KNOWN_SRC[stem][3][0] == "partB":  # PYT-AXIS
        with open(KNOWN_SRC[stem][3][1]) as f:  # PYT-AXIS
            return float(json.load(f)["clip"][stem]["c"])  # PYT-AXIS
    with open(os.path.join(PROJ, "results", "S1_2", "cache", "clip_thresholds.json")) as f:
        return float(json.load(f)[f"{stem}|wit1024"])


def layer_rows(stem: str, layer: int, pre: str, c: float | None) -> np.ndarray:
    st = load_stack(stem)   # LLAMA-AXIS: loader
    return C12.preprocess_rows(np.asarray(st[:, layer, :]), pre, c)


def gpu_layer(X: np.ndarray, kmax: int = KNN_K + 1):
    """Gram, SPEC distance, kNN table (key (D_ij, j)), centred Gram, on GPU f64."""
    Xt = torch.from_numpy(X).to(DEV)
    K = Xt @ Xt.T
    dg = torch.diagonal(K).clone()
    D = (dg[:, None] + dg[None, :]) - 2.0 * K
    D.clamp_(min=0.0)
    D.fill_diagonal_(0.0)
    Dk = D.clone()
    Dk.fill_diagonal_(float("inf"))
    vals, idx = torch.sort(Dk, dim=1, stable=True)      # ascending D, ties by index
    nn = idx[:, :kmax].contiguous()
    nv = vals[:, :kmax].contiguous()
    del Dk, vals, idx
    rm = K.mean(dim=1)
    cm = K.mean(dim=0)
    gm = rm.mean()
    Kc = K - rm[:, None]
    Kc -= cm[None, :]
    Kc += gm
    return {"K": K, "D": D, "nn": nn, "nv": nv, "Kc": Kc}


def knn_adjacency(nn: torch.Tensor, k: int = KNN_K) -> torch.Tensor:
    m = nn.shape[0]
    M = torch.zeros((m, m), dtype=torch.float16, device=nn.device)
    M.scatter_(1, nn[:, :k], 1.0)
    return M


class ModelGPU:
    """All layers of one model for one preprocessing, laid out for the pair kernel.

    side 'A' (LM):  Kc as (m, L, m)  [i, a, j];  M as (m, L, m)
    side 'B' (ViT): Kc as (m, m, L)  [i, j, b];  M as (m, m, L)
    """

    def __init__(self, stem: str, pre: str, side: str, c: float | None, knn_tables: np.ndarray | None = None,
                 keep_tables: bool = False):
        self.stem, self.pre, self.side = stem, pre, side
        L = n_stored(stem)
        m = M_ITEMS
        self.L = L
        if side == "A":
            self.Kc = torch.empty((m, L, m), dtype=torch.float64, device=DEV)
            self.M = torch.empty((m, L, m), dtype=torch.float16, device=DEV)
        else:
            self.Kc = torch.empty((m, m, L), dtype=torch.float64, device=DEV)
            self.M = torch.empty((m, m, L), dtype=torch.float16, device=DEV)
        self.diag = torch.empty((L, m), dtype=torch.float64, device=DEV)
        self.csum = torch.empty((L, m), dtype=torch.float64, device=DEV)   # col sums (A) / row sums (B) minus diag
        self.hb_self = torch.empty(L, dtype=torch.float64, device=DEV)    # <Kc, Kc>
        self.hu_self = torch.empty(L, dtype=torch.float64, device=DEV)    # unbiased HSIC(Kc, Kc)
        self.s_tot = torch.empty(L, dtype=torch.float64, device=DEV)      # sum of csum
        self.tables = np.empty((m, L, KNN_K + 1), dtype=np.int64) if keep_tables else None
        self.vals = np.empty((m, L, KNN_K + 1), dtype=np.float64) if keep_tables else None
        self.boundary_tie = np.zeros(L, dtype=bool)        # exact D_(k) == D_(k+1) in some row
        self.n_boundary_tie_rows = np.zeros(L, dtype=np.int64)
        self.D_flag = {}                                   # layer -> D (only B layers with boundary ties)
        st = load_stack(stem)   # LLAMA-AXIS: loader
        for l in range(L):
            X = C12.preprocess_rows(np.asarray(st[:, l, :]), pre, c)
            g = gpu_layer(X)
            nn = g["nn"]
            if knn_tables is not None:
                nn = torch.from_numpy(np.ascontiguousarray(knn_tables[:, l, :])).to(DEV)
            Mi = knn_adjacency(nn)
            Kc = g["Kc"]
            if side == "A":
                self.Kc[:, l, :] = Kc
                self.M[:, l, :] = Mi
                d = torch.diagonal(Kc).clone()
                self.csum[l] = Kc.sum(dim=0) - d
            else:
                self.Kc[:, :, l] = Kc
                self.M[:, :, l] = Mi
                d = torch.diagonal(Kc).clone()
                self.csum[l] = Kc.sum(dim=1) - d
            self.diag[l] = d
            self.hb_self[l] = torch.sum(Kc * Kc)
            self.hu_self[l] = hsic_unbiased_gpu(Kc, Kc)
            self.s_tot[l] = self.csum[l].sum()
            nv = g["nv"]
            tie = (nv[:, KNN_K] == nv[:, KNN_K - 1])
            self.n_boundary_tie_rows[l] = int(tie.sum().item())
            self.boundary_tie[l] = bool(self.n_boundary_tie_rows[l] > 0)
            if side == "B" and self.boundary_tie[l]:
                self.D_flag[l] = g["D"].clone()
            if keep_tables:
                self.tables[:, l, :] = g["nn"].cpu().numpy()
                self.vals[:, l, :] = g["nv"].cpu().numpy()
            del g, Kc, Mi
        del st

    def nbytes(self) -> int:
        return int(self.Kc.numel() * 8 + self.M.numel() * 2)


def hsic_unbiased_gpu(K: torch.Tensor, Lm: torch.Tensor) -> torch.Tensor:
    """Same formula as hrhv2.geometric.hsic_unbiased (float64, GPU)."""
    m = K.shape[0]
    dK = torch.diagonal(K)
    dL = torch.diagonal(Lm)
    t1 = torch.sum(K * Lm) - torch.dot(dK, dL)
    cK = K.sum(dim=0) - dK
    rL = Lm.sum(dim=1) - dL
    sK = cK.sum()
    sL = rL.sum()
    t2 = sK * sL / ((m - 1) * (m - 2))
    t3 = 2.0 * torch.dot(cK, rL) / (m - 2)
    return (t1 + t2 - t3) / (m * (m - 3))


def recompute_M_perm(D: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """SPEC kNN adjacency of the permuted distance matrix (used only for ViT layers
    with exact boundary ties, where relabelling != recomputation)."""
    Dp = D[pi][:, pi].clone()
    Dp.fill_diagonal_(float("inf"))
    _, idx = torch.sort(Dp, dim=1, stable=True)
    return knn_adjacency(idx[:, :KNN_K])


def pair_scores(A: ModelGPU, B: ModelGPU, pi: torch.Tensor) -> dict:
    """L_A x L_B matrices for ViT rows permuted by pi (new item i = old pi[i])."""
    m = M_ITEMS
    Bg = B.Kc[pi[:, None], pi[None, :]]                       # (m, m, LB)
    H = torch.bmm(A.Kc, Bg).sum(dim=0)                        # (LA, LB) = <Kc_A, P Kc_B P^T>
    del Bg
    cka_b = H / torch.sqrt(A.hb_self[:, None] * B.hb_self[None, :])
    # debiased (Song) on HKH, hrhv2 formula with L = P Kc_B P^T
    dBp = B.diag[:, pi]
    rBp = B.csum[:, pi]
    t1 = H - A.diag @ dBp.T
    t2 = A.s_tot[:, None] * B.s_tot[None, :] / ((m - 1) * (m - 2))
    t3 = 2.0 * (A.csum @ rBp.T) / (m - 2)
    hu = (t1 + t2 - t3) / (m * (m - 3))
    den = A.hu_self[:, None] * B.hu_self[None, :]
    cka_u = torch.where(den > 0, hu / torch.sqrt(torch.clamp(den, min=1e-300)), torch.full_like(hu, float("nan")))
    Mg = B.M[pi[:, None], pi[None, :]]                        # (m, m, LB) f16
    for l, D in B.D_flag.items():
        Mg[:, :, l] = recompute_M_perm(D, pi)
    cnt = torch.bmm(A.M, Mg)                                  # (m, LA, LB), per-row counts <= 10: exact in f16
    counts = cnt.sum(dim=0, dtype=torch.float32)              # <= m k = 10240: exact in f32
    del Mg, cnt
    return {"cka_biased": cka_b, "cka_debiased": cka_u, "mknn_count": counts, "hsic_b": H, "hsic_u": hu}


def perms_for(lm: str, vit: str, K: int = K_PERM, key: str | None = None) -> np.ndarray:
    key = key or perm_key(lm, vit)
    return np.stack([cal.permutation(M_ITEMS, key, k) for k in range(K)])


def run_pair(A: ModelGPU, B: ModelGPU, perms: np.ndarray) -> dict:
    """Observed (index 0) + K permutations: tensors (K+1, LA, LB) on GPU."""
    m = M_ITEMS
    Kp = perms.shape[0]
    out = {k: torch.empty((Kp + 1, A.L, B.L), dtype=torch.float64, device=DEV)
           for k in ("cka_biased", "cka_debiased", "mknn")}
    P = torch.from_numpy(perms).to(DEV)
    ident = torch.arange(m, device=DEV)
    for k in range(Kp + 1):
        pi = ident if k == 0 else P[k - 1]
        s = pair_scores(A, B, pi)
        out["cka_biased"][k] = s["cka_biased"]
        out["cka_debiased"][k] = s["cka_debiased"]
        out["mknn"][k] = s["mknn_count"].double() / (m * KNN_K)
    return out


def jsonable(v):
    return C12.jsonable(v)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
