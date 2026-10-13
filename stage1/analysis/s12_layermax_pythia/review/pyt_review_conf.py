"""Pythia axis, sub-role R: frozen constants and axis configurations (NEW FILE, PYT-AXIS).

Written at S0b by role pyt-code-R (2026-10-10), before any Pythia byte existed on ds2213; to be hashed in the one
S0b CODE FROZEN line. Every number below is copied from analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md
(sha256 6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b; section cited per constant), from the
shared verdict contract VERDICTS_SPEC_PYT.md / verdict_keys.py (names: families, hypothesis labels, key prefixes,
stand-in slugs), or, for the legacy and stand-in configurations used only by the S0b checks V-P8 and V-P9, from the
frozen OpenLLaMA/LLaMA files. No constant here is chosen by R; R's operational choices are marked [R-op] in
pyt_review_pythia.py and CHANGES.md. The band constants below are used only for R's x| recheck of Part B; every
verdict uses the bands frozen in partB_features_pyt.json (contract section 1).
"""
from __future__ import annotations

import itertools

PLAN_REL = "analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md"
PLAN_SHA256 = "6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b"

PRES = ("prh_clip+l2", "l2")
PRIMARY_PRE = "prh_clip+l2"
METRICS = ("mknn", "cka_biased", "cka_debiased")

# 17 ViTs in ARH val order (analysis/s12/common.py::VITS; the OpenLLaMA verdict_keys.VITS order), plan section 1
VITS = (
    "vit_tiny_patch16_224.augreg_in21k", "vit_small_patch16_224.augreg_in21k", "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k", "vit_base_patch16_224.mae", "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae", "vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m", "vit_giant_patch14_dinov2.lvd142m", "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b", "vit_huge_patch14_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b_ft_in12k", "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
)
# plan section 1 "ViTs" (families used only by section 5.5(d)); labels as verdict_keys.FAMILIES4
FAMILIES4 = ("in21k", "MAE", "DINOv2", "CLIP")
VIT_FAMILIES = {"in21k": VITS[0:4], "MAE": VITS[4:7], "DINOv2": VITS[7:11], "CLIP": VITS[11:17]}
VIT_FAMILY4 = {v: f for f, vs in VIT_FAMILIES.items() for v in vs}
assert sum(len(v) for v in VIT_FAMILIES.values()) == 17 and set(itertools.chain(*VIT_FAMILIES.values())) == set(VITS)
HYPS = ("H2", "H3", "H4")   # the secondary family behind the H1 gate (plan 5.4; contract names)

# ---------------------------------------------------------------------------------------------- plan section 4/5/7/9
MAT = 0.10            # materiality of a trend change, TOST and magnitude margin (5.1, 5.4)
ALPHA = 0.05          # one-sided FWER (5.4)
NB = 2000             # ViT bootstrap draws (5.1, 5.4)
DELTA_B = 0.005       # delta_b (B4)
MAT_TR = 0.75         # |Tr_P(B)| >= 0.75 (B4)
GAP_MULT = 3          # median end-to-end gap >= 3 delta_b (B4)
LOSS_RAW = 0.60       # "loss" needs Tr_P(raw) >= 0.60 ... (section 7, OP section 7 cuts)
LOSS_CAL = 0.40       # ... and Tr_P(s_cal) <= 0.40
NEAR_CHANCE_FRAC = 1.0 / 3.0  # M4: more than 1/3 of the 119 primary pairs with s_cal = 0
KNIFE_SWAP = 0.0021   # one adjacent swap moves Tr_P by 2/56/17 = 0.0021 (F2, 5.6)
PP2B_MIN_DECIDED = 30  # P-P2b (section 9)
PP_MIN_N = 6          # P-P2a(ii) and P-P5a "not evaluable if n < 6"
PP2AI_MIN_MODELS = 6  # P-P2a(i): >= 6 of the 7 models
BASE_MKNN_NUM, BASE_MKNN_DEN = 10, 1023  # Prop. 5 (1): k/(m-1)

# 5.6 curves
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
HS = (0.0004, 0.0006, 0.0008, 0.0010, 0.0015)
MARGINS = (0.05, 0.10, 0.15)

# B6 band constants (frozen): f(PR*) = a + b / sqrt(PR*), band_i = [f - h, f + h]
BAND = {"prh_clip+l2": {"a": 0.001069, "b": 0.012920, "h": 0.0008},
        "l2": {"a": 0.001551, "b": 0.006843, "h": 0.0018}}
FLAT_BAND = {"prh_clip+l2": (0.0010, 0.0040), "l2": (0.0010, 0.0045)}   # sensitivity only (Q4, 9)
UNC_BAND = (-0.0003, 0.0008)                                             # mu_hat - Bbar, secondary (B6)
PR_RANGE_LO = {"prh_clip+l2": 124.5, "l2": 79.6}                        # M5 extrapolation flag (7, B5)

BOOT = "S1.2-layermax-pythia|boot|"   # 5.4 bootstrap key prefix
BOOTVIT = "S1.2|bootvit|layermax-pythia|"  # 5.1 ViT bootstrap CI key prefix


class Axis:
    """One size family as R sees it: sizes in size order (rank 1..n), slugs, and descriptive covariates."""

    def __init__(self, name, sizes, slugs, num_params=None, width=None, depth=None, L_stored=None,
                 five=None, label=None, pair_groups=None):
        self.name = name
        self.sizes = tuple(sizes)
        self.n = len(self.sizes)
        self.slug = dict(slugs)
        self.num_params = dict(num_params or {})
        self.width = dict(width or {})
        self.depth = dict(depth or {})
        self.L_stored = dict(L_stored or {})
        self.five = tuple(five) if five else None
        self.label = label or name
        self.pairs = tuple((a, b) for i, a in enumerate(self.sizes) for b in self.sizes[i + 1:])
        self.pair_groups = pair_groups or {"all": self.pairs}

    def idx(self, s):
        return self.sizes.index(s)


PYT_SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b")
PYTHIA = Axis(
    "pythia", PYT_SIZES, {s: f"EleutherAI__pythia-{s}" for s in PYT_SIZES},
    # plan section 1 registry (expected num_params; width d; blocks L; stored layers L+1)
    num_params={"70m": 70426624, "160m": 162322944, "410m": 405334016, "1b": 1011781632, "1.4b": 1414647808,
                "2.8b": 2775208960, "6.9b": 6857302016},
    width={"70m": 512, "160m": 768, "410m": 1024, "1b": 2048, "1.4b": 2048, "2.8b": 2560, "6.9b": 4096},
    depth={"70m": 6, "160m": 12, "410m": 24, "1b": 16, "1.4b": 24, "2.8b": 32, "6.9b": 32},
    L_stored={"70m": 7, "160m": 13, "410m": 25, "1b": 17, "1.4b": 25, "2.8b": 33, "6.9b": 33},
    five=("410m", "1b", "1.4b", "2.8b", "6.9b"), label="Pythia")
# section 6 D4 contrasts
D4_CONTRASTS = (("410m", "1b"), ("1b", "1.4b"), ("2.8b", "6.9b"))
SMALL_SIZES = ("70m", "160m")   # M3 wording

# fixed depths (plan section 3 table), checked against fixed_layer() in the S0b tests
FIXED_DEPTH = {"0.5": {"70m": 3, "160m": 6, "410m": 12, "1b": 8, "1.4b": 12, "2.8b": 16, "6.9b": 16},
               "1.0": {"70m": 6, "160m": 12, "410m": 24, "1b": 16, "1.4b": 24, "2.8b": 32, "6.9b": 32}}
GRID5_LM = {"70m": [0, 2, 3, 5, 6], "160m": [0, 3, 6, 9, 12], "410m": [0, 6, 12, 18, 24], "1b": [0, 4, 8, 12, 16],
            "1.4b": [0, 6, 12, 18, 24], "2.8b": [0, 8, 16, 24, 32], "6.9b": [0, 8, 16, 24, 32]}

# ---------------------------------------------------------------------------------------------- S0b test configurations
# V-P9 stand-in (contract section 13): the 5 BLOOMZ sizes + llama-13b + llama-30b under the size labels of Pythia,
# slugs synth__pyt-<size> (no name contains EleutherAI__pythia); covariates are those of the SOURCE models
STANDIN_SRC = ("bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
               "bigscience__bloomz-7b1", "huggyllama__llama-13b", "huggyllama__llama-30b")
STANDIN = Axis(
    "standin", PYT_SIZES, {s: f"synth__pyt-{s}" for s in PYT_SIZES},
    num_params={"70m": 559214592, "160m": 1065314304, "410m": 1722408960, "1b": 3002557440, "1.4b": 7069016064,
                "2.8b": 13015864320, "6.9b": 32528943616},
    width={"70m": 1024, "160m": 1536, "410m": 2048, "1b": 2560, "1.4b": 4096, "2.8b": 5120, "6.9b": 6656},
    depth={"70m": 24, "160m": 24, "410m": 24, "1b": 30, "1.4b": 30, "2.8b": 40, "6.9b": 60},
    L_stored={"70m": 25, "160m": 25, "410m": 25, "1b": 31, "1.4b": 31, "2.8b": 41, "6.9b": 61},
    five=PYTHIA.five, label="stand-in (5 BLOOMZ + llama-13b + llama-30b; NOT Pythia)")
STANDIN_OF = dict(zip(PYT_SIZES, STANDIN_SRC))   # stand-in size -> source slug

# V-P8(a)/(b): the OpenLLaMA axis, n = 3, legacy rule
OL_SIZES = ("3b", "7b", "13b")
OPENLLAMA = Axis("openllama", OL_SIZES, {s: f"openlm-research__open_llama_{s}" for s in OL_SIZES},
                 pair_groups={"new": (("3b", "13b"), ("7b", "13b")), "pilot": (("3b", "7b"),)}, label="OpenLLaMA")
# V-P8(c): BLOOMZ, n = 5, legacy rule
BZ_SIZES = ("560m", "1b1", "1b7", "3b", "7b1")
BLOOMZ = Axis("bloomz", BZ_SIZES, {s: f"bigscience__bloomz-{s}" for s in BZ_SIZES}, label="BLOOMZ")

# section 5.5(g) 18-LM panel: the 11 known LMs (ARH order) + the 7 Pythia models; families for per-family trends
POOLED11 = ("bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
            "bigscience__bloomz-7b1", "openlm-research__open_llama_3b", "openlm-research__open_llama_7b",
            "openlm-research__open_llama_13b", "huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b")
# contract section 8 families (verdict_keys.LM_FAMILIES order BLOOMZ, LLaMA, OpenLLaMA, then Pythia)
LM_FAMILIES = {"BLOOMZ": POOLED11[0:5], "LLaMA": POOLED11[8:11], "OpenLLaMA": POOLED11[5:8]}
