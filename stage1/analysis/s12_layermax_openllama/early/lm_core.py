"""ARH layer-max replication (S1.2 secondary, role arh-layermax-early): core.

OPENLLAMA-AXIS COPY (implementation E of analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md,
sha256 4f18d710..., section 3 "Implementations"; S0b sub-role E of oll-axis-extract). Copied from
analysis/s12_layermax_llama/early/lm_core.py (sha256 26306eac...). Every change is marked "OL-AXIS" and
listed in early/CHANGES.md: model list (OpenLLaMA 3B/7B/13B, family OpenLLaMA), feature paths (13B in
features/wit1024_openllama_axis/), output paths (results/S1_2_layermax_openllama/early, shared single-model
cache results/S1_2_layermax_openllama/cache), Part B reference paths, thread count from OL_THREADS, the
S-E2 sensitivity switch and the S0b test hooks. The kernels (gpu_layer, ModelGPU, pair_scores, run_pair)
are unchanged.

This module holds constants and functions only; importing it reads no data (plan section 4 F1(a)).

Switches (environment):
  OL_SENS_E2=1   plan section 6 S-E2 (used only if E2 is not bitwise): open_llama_3b/7b features are read
                 from features/wit1024_openllama_axis/_e2_check/<slug>.{npy,json}; every output goes under
                 results/S1_2_layermax_openllama/early/sens_e2/; the single-model caches and pair files of
                 the unaffected models (13B, ViTs) are read from the primary early/ tree.
  OL_S0B_TEST=1  S0b unit tests only (guard.test_mode(); refused after the CODE FROZEN line): every output
                 root moves to OL_E_TEST_ROOT (default ~/oll_s0b_scratch/E/run, must lie under
                 ~/oll_s0b_scratch/), the 13B slug becomes the stand-in "synth__ol13b" whose features are
                 OL_E_TEST_13B_FEAT{.npy,.json} (default the LLaMA-axis llama-13b stack, shape [1024, 41, 5120]).

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
hrhv2.structural.knn_table on every layer (V-O2, precompute.py); the validated
tables are stored in <CACHE>/knn_final/ and the main run uses those.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis", "s12"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
sys.path.insert(0, AXIS)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C12  # noqa: E402  (S1.2 registry / loaders; read-only import)
import guard  # noqa: E402  (axis guard; test_mode() only here, the gate is called by each gated script)
from hrhv2 import calibrate as cal  # noqa: E402
from hrhv2.prep import prh_clip_threshold  # noqa: E402

# ----------------------------------------------------------------------------
# switches: S0b test mode and S-E2 sensitivity
# ----------------------------------------------------------------------------
TEST = guard.test_mode()
SENS_E2 = os.environ.get("OL_SENS_E2") == "1"

OL_RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
if TEST:
    _troot = os.path.realpath(os.path.expanduser(os.environ.get("OL_E_TEST_ROOT",
                                                                os.path.join(guard.SCRATCH, "E", "run"))))
    if not (_troot == guard.SCRATCH or _troot.startswith(guard.SCRATCH + os.sep)):
        print(f"[lm_core] test root {_troot} is not under {guard.SCRATCH}", file=sys.stderr)
        sys.exit(3)
    OL_RES = _troot

# outputs of implementation E (plan section 4 F1 table); everything else below is read-only
RES_PRIMARY = os.path.join(OL_RES, "early")
RES = os.path.join(RES_PRIMARY, "sens_e2") if SENS_E2 else RES_PRIMARY
CACHE = os.path.join(RES, "cache")
CACHE_PRIMARY = os.path.join(RES_PRIMARY, "cache")
SHARED_CACHE = os.path.join(OL_RES, "cache")                 # shared single-model cache (V-O1 thresholds)
PARTB = os.path.join(OL_RES, "partB")                        # Part B (read-only for E)
PARTB_JSON = os.path.join(PARTB, "partB_features_ol.json")
PARTB_CLIP = os.path.join(PARTB, "clip_thresholds_partB_ol.json")
OL_INDEP_RES = os.path.join(OL_RES, "indep")                 # implementation I (read by vl5 only)
OL_FORECAST = os.path.join(OL_RES, "forecast")               # forecast_ol.py outputs (read by vl5 only)
OL_JOINT = os.path.join(OL_RES, "joint")                     # joint_summary.py outputs (read by vl5 only)
FROZEN_RES = os.path.join(PROJ, "results", "S1_2_layermax")  # frozen 8-LM run (read-only)
LLAMA_E_RES = os.path.join(PROJ, "results", "S1_2_layermax_llama", "early")   # LLaMA-axis E (read-only)
LLAMA_MERGED = os.path.join(PROJ, "results", "S1_2_layermax_llama", "layermax_llama_summary.json")
M_ITEMS = 1024
KNN_K = 10
K_PERM = 500
ALPHA = 0.05
PRES = ("prh_clip+l2", "l2")
PRIMARY_PRE = "prh_clip+l2"
# the OpenLLaMA axis (plan section 1); ViTs unchanged
SLUG_3B = "openlm-research__open_llama_3b"
SLUG_7B = "openlm-research__open_llama_7b"
SLUG_13B_REAL = "openlm-research__open_llama_13b"
TEST_STANDIN_13B = "synth__ol13b"
SLUG_13B = TEST_STANDIN_13B if TEST else SLUG_13B_REAL
LMS = [
    (SLUG_3B, "OpenLLaMA"),
    (SLUG_7B, "OpenLLaMA"),
    (SLUG_13B, "OpenLLaMA"),
]
SIZE_LABEL = {SLUG_3B: "3b", SLUG_7B: "7b", SLUG_13B: "13b"}
VITS = C12.VITS
NEW_LMS = (SLUG_13B,)
E2_AFFECTED = (SLUG_3B, SLUG_7B)        # models re-extracted by E2 (plan section 2)
FEAT_OL = os.path.join(PROJ, "features", "wit1024_openllama_axis")
FEAT_E2 = os.path.join(FEAT_OL, "_e2_check")
FEAT_W1024 = os.path.join(C12.FEAT, "wit1024")
TEST_13B_FEAT = os.path.expanduser(os.environ.get(
    "OL_E_TEST_13B_FEAT", os.path.join(PROJ, "features", "wit1024_llama_axis", "huggyllama__llama-13b")))
DEV = "cuda"

# ARH LLM ordering: prh_models.get_models("val") order (ARH 785054b, SOG be654a37),
# = the x-axis of ARH Fig. 5 ("ranked according to their language performance").
ARH_LLM_ORDER = [
    "bigscience/bloomz-560m", "bigscience/bloomz-1b1", "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b", "bigscience/bloomz-7b1", "openlm-research/open_llama_3b",
    "openlm-research/open_llama_7b", "openlm-research/open_llama_13b", "huggyllama/llama-7b",
    "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b",
]


def arh_rank(stem: str) -> int:
    """ARH LLM rank (open_llama_13b = 8); the S0b test stand-in takes the 13B rank."""
    s = SLUG_13B_REAL if stem == TEST_STANDIN_13B else stem
    return ARH_LLM_ORDER.index(s.replace("__", "/")) + 1


ARH_RANK = {s: arh_rank(s) for s, _ in LMS}
DENSE_RANK = {s: i + 1 for i, s in enumerate(sorted(ARH_RANK, key=ARH_RANK.get))}


def perm_key(lm: str, vit: str) -> str:
    return f"S1.2-layermax|{lm}|{vit}"


def n_threads(default: int) -> int:
    """Thread count given by launch_ol.sh (env OL_THREADS); default only outside the launcher."""
    return max(1, int(os.environ.get("OL_THREADS", default)))


def setup_torch(threads: int = 2, mem_frac: float = 0.40):
    torch.set_num_threads(n_threads(threads))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(mem_frac, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    except Exception:
        pass


# ----------------------------------------------------------------------------
# where things are (feature stacks, caches, pair files)
# ----------------------------------------------------------------------------

def _feat_base(stem: str) -> str:
    """Path prefix (without .npy/.json) of a model's WIT-1024 feature stack."""
    if stem == TEST_STANDIN_13B:
        return TEST_13B_FEAT
    if stem == SLUG_13B_REAL:
        return os.path.join(FEAT_OL, stem)
    if SENS_E2 and stem in E2_AFFECTED:
        return os.path.join(FEAT_E2, stem)
    return os.path.join(FEAT_W1024, stem)


def cache_dir(stem: str) -> str:
    """Single-model cache of a model: in S-E2 mode only 3B/7B are recomputed, the rest is read from
    the primary cache."""
    if SENS_E2 and stem not in E2_AFFECTED:
        return CACHE_PRIMARY
    return CACHE


def pairs_dir(pre: str, lm: str) -> str:
    """E pair files of an OpenLLaMA LM (S-E2: 13B pairs are read from the primary run)."""
    if SENS_E2 and lm not in E2_AFFECTED:
        return os.path.join(RES_PRIMARY, "pairs", pre)
    return os.path.join(RES, "pairs", pre)


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
    """Reference value for V-O1: open_llama_13b -> the frozen Part B value
    (results/S1_2_layermax_openllama/partB/clip_thresholds_partB_ol.json, format {"clip": {stem: {"c": c}}});
    every other model (3B, 7B, 17 ViTs) -> the S1.2 cache results/S1_2/cache/clip_thresholds.json."""
    if stem in NEW_LMS:
        with open(PARTB_CLIP) as f:
            return float(json.load(f)["clip"][stem]["c"])
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
