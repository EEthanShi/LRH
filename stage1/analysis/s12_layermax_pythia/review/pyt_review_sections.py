"""Pythia axis, sub-role R (review recomputation): the generalised trend sections and the legacy OpenLLaMA sections.

PYT-AXIS COPY of analysis/s12_layermax_openllama/review/ol_review_sections.py (sha256 e2e41713..., pristine in
PYT-AXIS: orig_copy/; diff pyt_review_sections.py.diff). Written at S0b by role pyt-code-R (2026-10-10), before any
PYT-AXIS: Pythia byte existed on ds2213. Changes (marked PYT-AXIS): the trend cache (Trends) and section A are
PYT-AXIS: generalised from 3 sizes to n sizes, with the plan section 5.1 common-ViT-set rule for every difference
PYT-AXIS: (mode "common") and the OpenLLaMA per-score exclusion as mode "legacy" (used only by V-P8); section B
PYT-AXIS: takes the comparison groups from the axis (Pythia: all 21 size pairs = 357 comparisons; OpenLLaMA:
PYT-AXIS: new/pilot); the near-tie count covers every size pair. Sections C, D, F, G, J below are the OpenLLaMA
PYT-AXIS: rules, kept unchanged for the V-P8(a) legacy reproduction; the Pythia rules (sections 5.4, 7, 9 of the
PYT-AXIS: Pythia plan) are in the new file pyt_review_pythia.py.

(Original header of the OpenLLaMA file follows.)
OpenLLaMA axis, sub-role R (review recomputation): the verdict sections A, B, C, D, F, G, J of VERDICTS_SPEC.md.

NEW FILE (OL-AXIS), helper of review/ol_axis_review.py. Written at S0b by the R sub-role of role
oll-axis-extract (2026-10-09), before any OpenLLaMA-13B byte existed on ds2213. Hashed in the CODE FROZEN line.

Every rule is taken from analysis/s12_layermax_openllama/ANALYSIS_PLAN_OPENLLAMA.md (sha256 4f18d710...) and
VERDICTS_SPEC.md; key names come from verdict_keys.py. Choices the plan and the spec leave open are marked
[R-op] here and listed in REVIEW_REPORT.md.

Input context (built by ol_axis_review.py):
  ctx["ol"][P][size][vit]          PairStats of this implementation's reruns (size in 3b/7b/13b)
  ctx["pool"][P][slug][vit]        PairStats of the 11-LM pooled panel (protocol `full`), sources of spec s1
  ctx["llama"][P][slug][vit]       PairStats of the implementation's own LLaMA-axis files (llama-7b/13b/30b)
  ctx["frozen_ol"][P][size][vit]   PairStats of the frozen 8-LM OpenLLaMA 3b/7b files (V-O4b)
  ctx["pb"]                        partB_features_ol.json
  ctx["slug"][size], ctx["pooled11"], ctx["merged_pairs"][P]   (merged LLaMA summary, embedded I pairs)
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np

import pyt_review_lib as RL  # PYT-AXIS

SIZES = ("3b", "7b", "13b")
NEW_PAIRS = (("3b", "13b"), ("7b", "13b"))
PILOT_PAIRS = (("3b", "7b"),)
ALL_PAIRS = (("3b", "7b"), ("7b", "13b"), ("3b", "13b"))
MAT = 0.10
DELTA_B = 0.005
BAND = (0.0009, 0.0029)
O_CENTER = 0.0019
TAUS = (0.03, 0.06, 0.10, 0.15, 0.20)
DBS = (0.0025, 0.005, 0.0075, 0.01)
HS = (0.0003, 0.0005, 0.001, 0.0015, 0.002)
ARH_DDELTA = (0.86 - 0.45) - (0.85 - 0.84)     # 0.40, plan section 7(a)
BASE_MKNN = Fraction(RL.KNN_K, RL.M_ITEMS - 1)  # 10/1023
TOL = 1e-9
NEARTIE_REL = 1e-12
J_LABEL = "EXPLORATORY"
J_PROV = "BLOOMZ hypothesis-generating; BLOOMZ fails B4; BLOOMZ and LLaMA known before this plan"


def fmt(v: float) -> str:
    return repr(float(v))


def scores(q):
    return ("M", "M_unc", "TM") if q == "ctm" else ("raw", "s_cal", "unc")


def dscores(q):
    return ("M", "M_unc", "TM") if q == "ctm" else ("s_cal", "unc")


def cscores(q):
    return ("M", "M_unc") if q == "ctm" else ("s_cal", "unc")


def sign_num(x) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ----------------------------------------------------------------------------------------------------------
# Part B accessors. [R-op] the spec writes Bbar as {lm: ...} and Bfd/Blate as {ol_lm: ...} without saying
# whether the key is the slug or the short size name; both are accepted (slug first for Bbar, short first for
# Blate/Bfd), and the choice actually found is recorded.
# ----------------------------------------------------------------------------------------------------------

class PartB:
    def __init__(self, pb: dict, slug: dict):
        self.pb, self.slug = pb, slug
        self.key_forms = set()

    def pp(self, P):
        return self.pb["per_pre"][P]

    def _get(self, d, size, vit, slug_first):
        ks = (self.slug[size], size) if slug_first else (size, self.slug[size])
        for k in ks:
            if k in d:
                self.key_forms.add("slug" if k == self.slug[size] else "short")
                return float(d[k][vit])
        raise KeyError(f"Part B: no entry for {size} ({self.slug[size]})")

    def bbar(self, P, size, vit):
        return self._get(self.pp(P)["Bbar"], size, vit, True)

    def bbar_slug(self, P, slug, vit):
        return float(self.pp(P)["Bbar"][slug][vit])

    def blate(self, P, size, vit):
        return self._get(self.pp(P)["Blate"], size, vit, False)

    def material(self, P):
        return bool(self.pp(P)["material"])

    def s_O(self, P):
        return int(self.pp(P)["s_O"])

    def s_O_J(self, P):
        return int(self.pp(P)["s_O_J"])

    def s_late(self, P):
        return int(self.pp(P)["s_late"])

    def material_Blate(self, P):
        return bool(self.pp(P)["material_Blate"])

    def Tr_pred(self, P):
        return float(self.pp(P)["Tr_pred"])


# ----------------------------------------------------------------------------------------------------------
# section A helpers
# ----------------------------------------------------------------------------------------------------------

class Trends:
    """Caches trend tables of the OpenLLaMA axis: key (P, M, Q, X) -> trend_table.
    PYT-AXIS: generalised to any size list (n = len(sizes)); mode "common" applies the plan 5.1 common-ViT-set
    PYT-AXIS: rule to every difference, mode "legacy" the OpenLLaMA per-score exclusion."""

    def __init__(self, ol, vits, sizes=SIZES, mode="legacy"):  # PYT-AXIS
        self.ol, self.vits, self.cache = ol, vits, {}
        self.sizes, self.n, self.mode = tuple(sizes), len(sizes), mode  # PYT-AXIS
        assert mode in ("legacy", "common")  # PYT-AXIS

    def rows(self, P, M, Q, X):
        return [[RL.val(self.ol[P][s][v], M, Q, X) for s in self.sizes] for v in self.vits]  # PYT-AXIS

    def table(self, P, M, Q, X):
        k = (P, M, Q, X)
        if k not in self.cache:
            self.cache[k] = RL.trend_table(self.rows(P, M, Q, X), self.n)  # PYT-AXIS
        return self.cache[k]

    def parts(self, P, M, Q, X):  # PYT-AXIS
        return self.table(P, M, Q, X)["parts"]

    def tr(self, P, M, Q, X):
        t = float(self.table(P, M, Q, X)["T"][0])
        return t if math.isfinite(t) else None

    @staticmethod
    def base(Q):  # PYT-AXIS
        return ("full", "raw") if Q == "ctm" else (Q, "raw")

    def ddiff(self, P, M, Q, X, sign=1):  # PYT-AXIS: (vec, per-ViT differences on V, n_V) of sign*[Tr(X)-Tr(base)]
        b = self.base(Q)
        if self.mode == "legacy":
            return sign * (self.table(P, M, Q, X)["T"] - self.table(P, M, b[0], b[1])["T"]), None, None
        return RL.diff_vec(self.parts(P, M, Q, X), self.parts(P, M, b[0], b[1]), "common", sign)

    def dT(self, P, M, Q, X):
        return self.ddiff(P, M, Q, X)[0]  # PYT-AXIS

    def dtr(self, P, M, Q, X):
        d = float(self.dT(P, M, Q, X)[0])
        return d if math.isfinite(d) else None


def neartie_count(rows) -> int:
    n = 0
    for r in rows:
        for i, j in itertools.combinations(range(len(r)), 2):  # PYT-AXIS: every size pair (3 pairs at n = 3)
            a, b = r[i], r[j]
            if a is None or b is None:
                continue
            a, b = float(a), float(b)
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            if abs(a - b) <= NEARTIE_REL * max(abs(a), abs(b)):
                n += 1
    return n


def section_A(v: dict, T: Trends, pres, metrics, protocols):
    for P in pres:
        for M in metrics:
            for Q in protocols:
                for X in scores(Q):
                    tab = T.table(P, M, Q, X)
                    pc = RL.p_counts(tab["T"])
                    v[f"tr|{P}|{M}|{Q}|{X}"] = pc["obs"]
                    v[f"tr_nconst|{P}|{M}|{Q}|{X}"] = int(tab["nconst"])
                    v[f"tr_pge|{P}|{M}|{Q}|{X}"] = pc["pge"]
                    v[f"tr_ple|{P}|{M}|{Q}|{X}"] = pc["ple"]
                    v[f"tr_ptwo|{P}|{M}|{Q}|{X}"] = pc["ptwo"]
                    if M != "mknn":
                        v[f"neartie|{P}|{M}|{Q}|{X}"] = neartie_count(T.rows(P, M, Q, X))
                    if T.mode == "common":  # PYT-AXIS: plan 5.1 sensitivity rho_v := 0 (finite ViTs), descriptive
                        t0 = T.parts(P, M, Q, X).tr0_vec()
                        v[f"tr0|{P}|{M}|{Q}|{X}"] = float(t0[0]) if math.isfinite(t0[0]) else None
                for X in dscores(Q):
                    dvec, _, nV = T.ddiff(P, M, Q, X)  # PYT-AXIS
                    pc = RL.p_counts(dvec)  # PYT-AXIS
                    v[f"dtr|{P}|{M}|{Q}|{X}"] = pc["obs"]
                    v[f"dtr_pge|{P}|{M}|{Q}|{X}"] = pc["pge"]
                    v[f"dtr_ple|{P}|{M}|{Q}|{X}"] = pc["ple"]
                    v[f"dtr_ptwo|{P}|{M}|{Q}|{X}"] = pc["ptwo"]
                    if T.mode == "common":  # PYT-AXIS: n_V of the common ViT set, and the rho := 0 sensitivity
                        v[f"dtr_nv|{P}|{M}|{Q}|{X}"] = nV
                        b = T.base(Q)
                        px, pb_ = T.parts(P, M, Q, X), T.parts(P, M, b[0], b[1])
                        Fm = px.finite & pb_.finite  # spec 3: ViTs with finite scores for both
                        d0 = px.tr0_vec(Fm) - pb_.tr0_vec(Fm)
                        v[f"dtr0|{P}|{M}|{Q}|{X}"] = float(d0[0]) if math.isfinite(d0[0]) else None
            if T.mode == "common":  # PYT-AXIS: D3 decomposition on the common ViT sets (VERDICTS_SPEC_PYT 3)
                d1v = RL.diff_vec(T.parts(P, M, "ctm", "M"), T.parts(P, M, "ctm", "TM"), "common")[0]
                d2v = RL.diff_vec(T.parts(P, M, "ctm", "TM"), T.parts(P, M, "full", "raw"), "common")[0]
                v[f"d3cal|{P}|{M}"] = float(d1v[0]) if math.isfinite(d1v[0]) else None
                v[f"d3sel|{P}|{M}"] = float(d2v[0]) if math.isfinite(d2v[0]) else None
                continue
            a, b, c = T.tr(P, M, "ctm", "M"), T.tr(P, M, "ctm", "TM"), T.tr(P, M, "full", "raw")
            v[f"d3cal|{P}|{M}"] = None if a is None or b is None else a - b
            v[f"d3sel|{P}|{M}"] = None if b is None or c is None else b - c


# ----------------------------------------------------------------------------------------------------------
# section B: order changes
# ----------------------------------------------------------------------------------------------------------

def order_changes(ol, P, M, Q, X, vits, pairs, size_of=None):
    """Counts (down, up, c2d, d2c) over the comparisons (v, a<b). size_of maps a logical size to the model
    whose scores it carries (used by the J conditional size); default identity."""
    size_of = size_of or {s: s for s in ol[P]}  # PYT-AXIS: any size list
    rq = "full" if Q == "ctm" else Q
    down = up = c2d = d2c = 0
    c2t = t2c = d2t = t2d = 0  # PYT-AXIS: tie transitions (VERDICTS_SPEC_PYT 4)
    for vit in vits:
        for a, b in pairs:
            pa, pb_ = ol[P][size_of[a]][vit], ol[P][size_of[b]][vit]
            ra, rb = RL.val(pa, M, rq, "raw"), RL.val(pb_, M, rq, "raw")
            xa, xb = RL.val(pa, M, Q, X), RL.val(pb_, M, Q, X)
            if None in (ra, rb, xa, xb):
                return None
            dr, dx = RL.sgn(ra, rb), RL.sgn(xa, xb)
            down += dx < dr
            up += dx > dr
            c2d += (dr == 1 and dx == -1)
            d2c += (dr == -1 and dx == 1)
            c2t += (dr == 1 and dx == 0)  # PYT-AXIS
            t2c += (dr == 0 and dx == 1)  # PYT-AXIS
            d2t += (dr == -1 and dx == 0)  # PYT-AXIS
            t2d += (dr == 0 and dx == -1)  # PYT-AXIS
    return {"down": int(down), "up": int(up), "c2d": int(c2d), "d2c": int(d2c),
            "c2t": int(c2t), "t2c": int(t2c), "d2t": int(d2t), "t2d": int(t2d)}  # PYT-AXIS


def section_B_all(v: dict, ol, pres, metrics, protocols, vits, pairs):  # PYT-AXIS
    """PYT-AXIS: Pythia section B (VERDICTS_SPEC_PYT 4): 8 transition counts over all comparisons and the plan 5.3
    PYT-AXIS: counts f- = c2d, f+ = d2c, F = (f- - f+)/n with Clopper-Pearson bounds of both."""
    for P in pres:
        for M in metrics:
            for Q in protocols:
                for X in cscores(Q):
                    oc = order_changes(ol, P, M, Q, X, vits, pairs)
                    for c in ("down", "up", "c2d", "d2c", "c2t", "t2c", "d2t", "t2d"):
                        v[f"chg|{P}|{M}|{Q}|{X}|all|{c}"] = None if oc is None else oc[c]
                    n = len(pairs) * len(vits)
                    k = f"fcnt|{P}|{M}|{Q}|{X}"
                    v[f"{k}|n"] = n
                    if oc is None:
                        for c in ("fminus", "fplus", "F", "fm_cp_lo", "fm_cp_hi", "fp_cp_lo", "fp_cp_hi"):
                            v[f"{k}|{c}"] = None
                        continue
                    fm, fp = oc["c2d"], oc["d2c"]
                    v[f"{k}|fminus"], v[f"{k}|fplus"], v[f"{k}|F"] = fm, fp, (fm - fp) / n
                    v[f"{k}|fm_cp_lo"], v[f"{k}|fm_cp_hi"] = RL.cp_bounds(fm, n)
                    v[f"{k}|fp_cp_lo"], v[f"{k}|fp_cp_hi"] = RL.cp_bounds(fp, n)


def section_B(v: dict, ol, PB, pres, metrics, protocols, vits, groups=None, fgroup="new", fkey="fnew", s_of=None):
    """PYT-AXIS: groups = {name: comparison pairs} (default the OpenLLaMA new/pilot groups); F, f-, f+ and the
    PYT-AXIS: CP bounds over the group `fgroup` under key prefix `fkey`; s_of(P) gives the direction s.
    PYT-AXIS: (legacy OpenLLaMA section B, used by V-P8(a); the Pythia section B is section_B_all)"""
    groups = groups or {"new": NEW_PAIRS, "pilot": PILOT_PAIRS}  # PYT-AXIS
    s_of = s_of or PB.s_O  # PYT-AXIS
    for P in pres:
        s = s_of(P)  # PYT-AXIS
        for M in metrics:
            for Q in protocols:
                for X in cscores(Q):
                    for grp, pairs in groups.items():  # PYT-AXIS
                        oc = order_changes(ol, P, M, Q, X, vits, pairs)
                        for c in ("down", "up", "c2d", "d2c"):  # (the legacy key set)
                            v[f"chg|{P}|{M}|{Q}|{X}|{grp}|{c}"] = None if oc is None else oc[c]
                    oc = order_changes(ol, P, M, Q, X, vits, groups[fgroup])  # PYT-AXIS
                    if oc is None:
                        for c in ("F", "fminus", "fplus", "cp_lo", "cp_hi"):
                            v[f"{fkey}|{P}|{M}|{Q}|{X}|{c}"] = None  # PYT-AXIS
                        continue
                    fm, fp = (oc["down"], oc["up"]) if s == 1 else (oc["up"], oc["down"])
                    n_new = len(groups[fgroup]) * len(vits)  # PYT-AXIS
                    lo, hi = RL.cp_bounds(fm, n_new)
                    v[f"{fkey}|{P}|{M}|{Q}|{X}|F"] = (fm - fp) / n_new  # PYT-AXIS
                    v[f"{fkey}|{P}|{M}|{Q}|{X}|fminus"] = fm  # PYT-AXIS
                    v[f"{fkey}|{P}|{M}|{Q}|{X}|fplus"] = fp  # PYT-AXIS
                    v[f"{fkey}|{P}|{M}|{Q}|{X}|cp_lo"] = lo  # PYT-AXIS
                    v[f"{fkey}|{P}|{M}|{Q}|{X}|cp_hi"] = hi  # PYT-AXIS


# ----------------------------------------------------------------------------------------------------------
# section C: Part B decision rules
# ----------------------------------------------------------------------------------------------------------

def cat_po2(material, tr_raw, s, d, tau=MAT):
    """spec section 5 P-O2 order of decisions."""
    if not material:
        return "no category (not material)"
    if tr_raw is not None and abs(tr_raw + s) <= TOL:
        return "not testable"
    if d is None:
        return "undefined"
    if s * d <= -tau:
        return "loss reproduced"
    if s * d >= tau:
        return "moved with the baseline"
    if abs(d) < tau:
        return "trend kept"
    return "undefined"


def cat_po5(material_late, po5a_verdict, tr_raw, s, d, tau=MAT):
    if not material_late:
        return "no category (not material)"
    if po5a_verdict == "fails":
        return "premise failed"
    if tr_raw is not None and abs(tr_raw + s) <= TOL:
        return "not testable"
    return cat_po2(True, None, s, d, tau)


def po2aii(ol, PB: PartB, P, vits, delta):
    n = x = 0
    for vit in vits:
        for a, b in NEW_PAIRS:
            g = PB.bbar(P, b, vit) - PB.bbar(P, a, vit)
            if abs(g) >= delta:
                n += 1
                ca = RL.val(ol[P][a][vit], "cka_biased", "full", "c")
                cb = RL.val(ol[P][b][vit], "cka_biased", "full", "c")
                x += sign_num(cb - ca) == sign_num(g)
    need = int(math.ceil(2 * n / 3))
    verdict = "not evaluable" if n < 6 else ("holds" if x >= need else "fails")
    return n, int(x), need, verdict


def po5a(ol, PB: PartB, P, vits, delta=DELTA_B):
    n = x = 0
    for vit in vits:
        for a, b in NEW_PAIRS:
            g = PB.blate(P, b, vit) - PB.blate(P, a, vit)
            if abs(g) >= delta:
                n += 1
                ca = ol[P][a][vit]["cka_biased"]["ctm"]["cM"]
                cb = ol[P][b][vit]["cka_biased"]["ctm"]["cM"]
                x += sign_num(cb - ca) == sign_num(g)
    need = int(math.ceil(2 * n / 3))
    verdict = "not evaluable" if n < 6 else ("holds" if x >= need else "fails")
    return n, int(x), need, verdict


def keep_and_sign(ol, P, M, Q, X, vits):
    """Keep identity and per-comparison sign rule over the 51 comparisons (spec section 5)."""
    nk = ns = 0
    exact = M == "mknn"
    for vit in vits:
        for a, b in ALL_PAIRS:
            sa, sb = ol[P][a][vit][M].get(Q), ol[P][b][vit][M].get(Q)
            if sa is None or sb is None:
                return None, None
            Ta, Tb = sa["T"], sb["T"]
            Xa, Xb = (sa["s_cal"], sb["s_cal"]) if X == "s_cal" else (sa["unc"], sb["unc"])
            ta, tb = (sa["c"], sb["c"]) if X == "s_cal" else (sa["mu"], sb["mu"])
            # keep identity
            if X == "s_cal":
                use = Xa > 0 and Xb > 0
                if use:
                    lhs, rhs = Tb - Ta, (1 - Ta) * (sb["c"] - sa["c"]) / (1 - sa["c"])
            else:
                use = True
                lhs, rhs = Tb - Ta, sb["mu"] - sa["mu"]
            if use:
                pred, obs = lhs >= rhs, (Xb - Xa) >= 0
                if pred != obs and (exact or abs(float(lhs) - float(rhs)) > 1e-12):
                    nk += 1
            # sign rule
            dr, dx = RL.sgn(Ta, Tb), RL.sgn(Xa, Xb)
            if tb > ta:
                ns += (dr <= 0 and dx == 1)
            elif tb < ta:
                ns += (dr >= 0 and dx == -1)
            else:
                ns += (dx == -dr and dr != 0)
    return int(nk), int(ns)


def section_C(v: dict, ol, T: Trends, PB: PartB, D: dict, pres, vits):
    CK = "cka_biased"
    for P in pres:
        # ---- P-O1 (mKNN, full)
        d1s, d1u = T.dtr(P, "mknn", "full", "s_cal"), T.dtr(P, "mknn", "full", "unc")
        v[f"PO1|{P}|holds"] = bool(d1s is not None and d1u is not None and abs(d1s) < MAT and abs(d1u) < MAT)
        for X in ("s_cal", "unc"):
            nq = nrev = ntie = 0
            for vit in vits:
                for a, b in ALL_PAIRS:
                    sa, sb = ol[P][a][vit]["mknn"]["full"], ol[P][b][vit]["mknn"]["full"]
                    th = "c" if X == "s_cal" else "mu"
                    za, zb = abs(sa[th] - BASE_MKNN), abs(sb[th] - BASE_MKNN)
                    gap = sb["T"] - sa["T"]
                    if abs(gap) > za + zb:
                        nq += 1
                        dx = RL.sgn(sa[X], sb[X])
                        dr = sign_num(gap)
                        nrev += dx == -dr
                        ntie += dx == 0
            v[f"PO1|{P}|zeta|{X}|nq"] = int(nq)
            v[f"PO1|{P}|zeta|{X}|nrev"] = int(nrev)
            v[f"PO1|{P}|zeta|{X}|ntie"] = int(ntie)
        # ---- P-O2 (biased CKA, full), s = s_O(P)
        s, mat = PB.s_O(P), PB.material(P)
        tr_raw = T.tr(P, CK, "full", "raw")
        for X in ("s_cal", "unc"):
            d = T.dtr(P, CK, "full", X)
            v[f"PO2|{P}|{X}|cat"] = cat_po2(mat, tr_raw, s, d)
            # [R-op] dist and kedge are written for every outcome (also when not material)
            dist = None if d is None else min(abs(s * d + MAT), abs(s * d - MAT))
            v[f"PO2|{P}|{X}|dist"] = dist
            v[f"PO2|{P}|{X}|kedge"] = None if dist is None else bool(dist <= 1 / 17 + TOL)
            for tau in TAUS:
                v[f"curve|{P}|PO2|{X}|tau={fmt(tau)}"] = cat_po2(mat, tr_raw, s, d, tau)
        # ---- P-O2a(i)
        offs = {vit: RL.val(ol[P]["13b"][vit], CK, "full", "c") - PB.bbar(P, "13b", vit) for vit in vits}
        med = RL.median(list(offs.values()))
        x_in = sum(BAND[0] <= o <= BAND[1] for o in offs.values())
        lo, hi = RL.cp_bounds(int(x_in), len(vits))
        v[f"PO2ai|{P}|median"] = med
        v[f"PO2ai|{P}|in_band"] = bool(BAND[0] <= med <= BAND[1])
        v[f"PO2ai|{P}|n"] = len(vits)
        v[f"PO2ai|{P}|x"] = int(x_in)
        v[f"PO2ai|{P}|cp_lo"], v[f"PO2ai|{P}|cp_hi"] = lo, hi
        lpm = RL.median([abs(o) for o in offs.values()])
        v[f"PO2ai|{P}|lp_median_abs"] = lpm
        v[f"PO2ai|{P}|lp_le_0.01"] = bool(lpm <= 0.01)
        for vit in vits:
            v[f"PO2ai|{P}|offset|{vit}"] = float(offs[vit])
        # ---- P-O2a(ii) and its delta_b curve
        n, x, need, verdict = po2aii(ol, PB, P, vits, DELTA_B)
        lo, hi = RL.cp_bounds(x, n)
        v.update({f"PO2aii|{P}|n": n, f"PO2aii|{P}|x": x, f"PO2aii|{P}|need": need, f"PO2aii|{P}|verdict": verdict,
                  f"PO2aii|{P}|cp_lo": lo, f"PO2aii|{P}|cp_hi": hi})
        trp = PB.Tr_pred(P)
        for db in DBS:
            n2, x2, _, vd2 = po2aii(ol, PB, P, vits, db)
            v[f"curve|{P}|PO2aii|db={fmt(db)}|n"] = n2
            v[f"curve|{P}|PO2aii|db={fmt(db)}|x"] = x2
            v[f"curve|{P}|PO2aii|db={fmt(db)}|verdict"] = vd2
            ok = abs(abs(trp) - 1) <= TOL
            for a, b in ALL_PAIRS:
                gaps = [PB.bbar(P, b, vit) - PB.bbar(P, a, vit) for vit in vits]
                ok = ok and RL.median(gaps) >= db
            v[f"curve|{P}|material|db={fmt(db)}"] = bool(ok)
        # ---- P-O3 (fixed depth, descriptive)
        for r in ("0.5", "1.0"):
            Q = f"fixed_r{r}"
            trr = T.tr(P, CK, Q, "raw")
            for X in ("s_cal", "unc"):
                trx = T.tr(P, CK, Q, X)
                if trr is None or abs(trr) < MAT:
                    desc = "not applied"
                else:
                    sr = 1 if trr > 0 else -1
                    if trx is None:
                        desc = "undefined"
                    elif sr * trx >= MAT:
                        desc = "keeps sign"
                    elif abs(trx) < MAT:
                        desc = "lost"
                    elif sr * trx <= -MAT:
                        desc = "reversed"
                    else:
                        desc = "undefined"
                v[f"PO3|{P}|{r}|{X}|desc"] = desc
        # ---- keep identity and sign rule (all M, all Q != ctm)
        for M in RL.METRICS:
            for Q in RL.SUBSETS:
                for X in ("s_cal", "unc"):
                    nk, ns = keep_and_sign(ol, P, M, Q, X, vits)
                    v[f"keepid|{P}|{M}|{Q}|{X}|nviol"] = nk
                    v[f"signrule|{P}|{M}|{Q}|{X}|nviol"] = ns
        # ---- P-O4 (debiased CKA)
        for X in ("s_cal", "unc"):
            d = T.dtr(P, "cka_debiased", "full", X)
            v[f"PO4|{P}|{X}|holds"] = bool(d is not None and d >= -MAT)
        rb, ra = D[("11", P, "cka_debiased", "raw", "arh")], D[("11", P, "cka_debiased", "s_cal", "arh")]
        v[f"PO4|{P}|arh|holds"] = bool(math.isfinite(ra) and math.isfinite(rb) and ra >= rb - 0.05)
        # ---- P-O5a, P-O5 (calibrate-then-max), s = s_late(P)
        n5, x5, need5, vd5 = po5a(ol, PB, P, vits)
        v.update({f"PO5a|{P}|n": n5, f"PO5a|{P}|x": x5, f"PO5a|{P}|need": need5, f"PO5a|{P}|verdict": vd5})
        sl, matl = PB.s_late(P), PB.material_Blate(P)
        for X in ("M", "M_unc"):
            d = T.dtr(P, CK, "ctm", X)
            v[f"PO5|{P}|{X}|pred_holds"] = (None if not matl or d is None else bool(sl * d <= 0))
            v[f"PO5|{P}|{X}|cat"] = cat_po5(matl, vd5, tr_raw, sl, d)
            for tau in TAUS:
                v[f"curve|{P}|PO5|{X}|tau={fmt(tau)}"] = cat_po5(matl, vd5, tr_raw, sl, d, tau)
        for tau in TAUS:
            # [R-op] the P-O1 curve value is the boolean "holds" with tau in place of 0.10
            v[f"curve|{P}|PO1|tau={fmt(tau)}"] = bool(d1s is not None and d1u is not None
                                                      and abs(d1s) < tau and abs(d1u) < tau)
        # ---- D3 deductive check
        rM = T.table(P, CK, "ctm", "M")["rho_id"]
        rTM = T.table(P, CK, "ctm", "TM")["rho_id"]
        nv3 = 0
        for k, vit in enumerate(vits):
            c3 = [ol[P][s_][vit][CK]["ctm"]["cM"] for s_ in SIZES]
            if c3[0] < c3[1] < c3[2] and rM[k] is not None and rTM[k] is not None and rM[k] - rTM[k] > 0:
                nv3 += 1
        v[f"D3|{P}|nviol"] = int(nv3)
        # ---- plug-in
        rows = [[RL.val(ol[P][s_][vit], CK, "full", "raw") - PB.bbar(P, s_, vit) for s_ in SIZES] for vit in vits]
        tp = float(RL.trend_table(rows, 3)["T"][0])
        tu = T.tr(P, CK, "full", "unc")
        v[f"plugin|{P}|tr"] = tp if math.isfinite(tp) else None
        diff = None if (not math.isfinite(tp) or tu is None) else tp - tu
        v[f"plugin|{P}|diff"] = diff
        v[f"plugin|{P}|within"] = bool(diff is not None and abs(diff) <= MAT)
        # ---- section 7 OpenLLaMA-axis reading
        tmu = T.tr(P, "mknn", "full", "unc")
        for M in ("cka_biased", "cka_debiased"):
            trr, tru = T.tr(P, M, "full", "raw"), T.tr(P, M, "full", "unc")
            if trr is None or trr < 0.60:
                rd = "not applied"
            elif tru is not None and tmu is not None and tru <= 0.40 and tmu >= 0.60:
                rd = "loses"
            elif tru is not None and tru >= 0.60:
                rd = "no loss"
            else:
                rd = "intermediate"
            v[f"readO|{P}|{M}"] = rd
        # ---- D1, D2, 7B > 3B
        for M in RL.METRICS:
            for s_ in SIZES:
                d1 = [ol[P][s_][vit][M]["full"]["mu"] - ol[P][s_][vit][M]["entry_mu_max"] for vit in vits]
                v[f"D1|{P}|{M}|{s_}"] = RL.median(d1)
                ar = [ol[P][s_][vit][M]["full"]["aA"] / (ol[P][s_][vit]["L_A"] - 1) for vit in vits]
                v[f"D2|{P}|{M}|{s_}|argmax_rel"] = RL.median(ar)
                cc = [ol[P][s_][vit][M]["full"]["c"] - ol[P][s_][vit][M]["c_entry_full_argmax"] for vit in vits]
                v[f"D2|{P}|{M}|{s_}|c_minus_centry"] = RL.median(cc)
            n7 = sum(RL.val(ol[P]["7b"][vit], M, "full", "raw") > RL.val(ol[P]["3b"][vit], M, "full", "raw")
                     for vit in vits)
            v[f"frac7gt3|{P}|{M}"] = n7 / len(vits)


# ----------------------------------------------------------------------------------------------------------
# section D: pooled ARH panel
# ----------------------------------------------------------------------------------------------------------

def panel_spec(pooled11: list, slug13: str, num_params: dict):
    """Panels in ARH order with their rank vectors (spec section 1)."""
    arh11 = {s: i + 1 for i, s in enumerate(pooled11)}
    p11 = list(pooled11)
    p10 = [s for s in p11 if s != slug13]
    llama_new = [s for s in p11 if s.endswith("llama-13b") or s.endswith("llama-30b")]
    p8 = [s for s in p10 if s not in llama_new]
    out = {}
    for name, lms in (("11", p11), ("10", p10), ("8", p8)):
        out[name] = {"lms": lms,
                     "arh": np.array([arh11[s] for s in lms], dtype=float),
                     "dense": np.arange(1, len(lms) + 1, dtype=float),
                     "logp": np.log(np.array([float(num_params[s]) for s in lms], dtype=float))}
    return out


def grid(pool, P, lms, vits, M, X):
    return np.array([[float(RL.val(pool[P][lm][vit], M, "full", X)) for vit in vits] for lm in lms], dtype=float)


def section_D(v: dict, pool, panels, pres, vits, n_threads_note=None):
    """Returns D[(panel, P, M, X, kind)] = r for later use (P-O4)."""
    D = {}
    G = {}
    for P in pres:
        for pan, spec in panels.items():
            for M in RL.METRICS:
                for X in ("raw", "s_cal", "unc"):
                    G[(pan, P, M, X)] = grid(pool, P, spec["lms"], vits, M, X)
                    for kind in ("arh", "dense", "logp"):
                        r = RL.pear(G[(pan, P, M, X)], spec[kind])
                        D[(pan, P, M, X, kind)] = r
                        v[f"arh|{pan}|{P}|{M}|{X}|{kind}|r"] = r if math.isfinite(r) else None
                        if pan == "11":
                            lo, hi = RL.boot_r(G[(pan, P, M, X)], spec[kind],
                                               "S1.2-layermax-openllama|boot|arh_r|" + f"{P}|{M}|{X}|{kind}")
                            v[f"arh|11|{P}|{M}|{X}|{kind}|ci_lo"] = lo
                            v[f"arh|11|{P}|{M}|{X}|{kind}|ci_hi"] = hi
            for M in ("cka_biased", "cka_debiased"):
                dd = ((D[(pan, P, M, "raw", "arh")] - D[(pan, P, M, "s_cal", "arh")])
                      - (D[(pan, P, "mknn", "raw", "arh")] - D[(pan, P, "mknn", "s_cal", "arh")]))
                D[(pan, P, M, "dd")] = dd
                v[f"dd|{pan}|{P}|{M}"] = dd if math.isfinite(dd) else None
        x11 = panels["11"]["arh"]
        for M in ("cka_biased", "cka_debiased"):
            est, lo, hi = RL.delta_boot(G[("11", P, M, "raw")], G[("11", P, M, "s_cal")],
                                        G[("11", P, "mknn", "raw")], G[("11", P, "mknn", "s_cal")], x11,
                                        f"S1.2-layermax-openllama|boot|arh_delta|{P}|{M}")
            v[f"dd|11|{P}|{M}|ci_lo"], v[f"dd|11|{P}|{M}|ci_hi"] = lo, hi
            D[("11", P, M, "dd_ci")] = (lo, hi)
        # section 7(a) rule and FP section 6 category (M = cka_biased)
        lo, hi = D[("11", P, "cka_biased", "dd_ci")]
        v[f"rule7a|{P}"] = ("replicated the direction" if lo > 0 else
                            ("opposite direction" if hi < 0 else "direction not established"))
        v[f"size_not_reproduced|{P}"] = bool(ARH_DDELTA > hi)
        dd = D[("11", P, "cka_biased", "dd")]
        ra_c, ra_m = D[("11", P, "cka_biased", "s_cal", "arh")], D[("11", P, "mknn", "s_cal", "arh")]
        if dd >= 0.20 and ra_c <= ra_m - 0.20:
            raw_cat = "ARH pattern reappears"
        elif dd <= 0.10 or ra_c >= ra_m - 0.10:
            raw_cat = "absent"
        else:
            raw_cat = "partial"
        glo, ghi = RL.gap_boot(G[("11", P, "mknn", "s_cal")], G[("11", P, "cka_biased", "s_cal")], x11,
                               f"S1.2-layermax-openllama|boot|arh_gap|{P}")
        kedge = bool((lo <= 0.10 <= hi) or (glo <= 0.10 <= ghi))
        v[f"fp6|{P}|raw_cat"] = raw_cat
        v[f"fp6|{P}|gap_ci_lo"], v[f"fp6|{P}|gap_ci_hi"] = glo, ghi
        v[f"fp6|{P}|kedge"] = kedge
        v[f"fp6|{P}|cat"] = "not decidable at this panel size (knife edge)" if kedge else raw_cat
    # exact one-sided p over all 11! relabellings, every (P, M, X) r and every dd, in one enumeration
    xc = panels["11"]["arh"] - panels["11"]["arh"].mean()
    cols, keys = [], []
    for P in pres:
        for M in RL.METRICS:
            for X in ("raw", "s_cal", "unc"):
                r = D[("11", P, M, X, "arh")]
                if math.isfinite(r):
                    cols.append((RL.linear_weights(G[("11", P, M, X)], panels["11"]["arh"]), r - TOL))
                    keys.append(f"arh|11|{P}|{M}|{X}|arh|pge")
                else:
                    v[f"arh|11|{P}|{M}|{X}|arh|pge"] = None
        for M in ("cka_biased", "cka_debiased"):
            dd = D[("11", P, M, "dd")]
            if math.isfinite(dd):
                w = lambda m, x: RL.linear_weights(G[("11", P, m, x)], panels["11"]["arh"])  # noqa: E731
                cols.append((w(M, "raw") - w(M, "s_cal") - (w("mknn", "raw") - w("mknn", "s_cal")), dd - TOL))
                keys.append(f"dd|11|{P}|{M}|pge")
            else:
                v[f"dd|11|{P}|{M}|pge"] = None
    if cols:
        W = np.stack([c[0] for c in cols], axis=1)
        thr = np.array([c[1] for c in cols])
        cnt = RL.enumerate_ge_counts(xc, W, thr)
        nfact = math.factorial(xc.size)
        for k, c in zip(keys, cnt):
            v[k] = int(c) / nfact
    return D


# ----------------------------------------------------------------------------------------------------------
# section G: provenance
# ----------------------------------------------------------------------------------------------------------

def _close(a, b, rel=1e-12):
    a, b = float(a), float(b)
    return a == b or abs(a - b) <= rel * max(abs(a), abs(b))


def g_llama_vs_merged(llama, merged_pairs, pres, vits, slugs):
    """llama-13b/30b layer-max values of this implementation vs the merged LLaMA summary (embedded I pairs)."""
    mm = {"mknn": "mknn", "cka_biased": "cka_b", "cka_debiased": "cka_d"}
    fields = (("T", "s_obs"), ("c", "c_alpha"), ("s_cal", "s_cal"), ("unc", "uncensored"), ("mu", "null_mean"),
              ("p", "p_value"))
    n = bad = 0
    worst = []
    for P in pres:
        for lm in slugs:
            for vit in vits:
                ref = merged_pairs[P][f"{lm}|{vit}"]
                for M in RL.METRICS:
                    st = llama[P][lm][vit][M]["full"]
                    for a, b in fields:
                        n += 1
                        if not _close(st[a], ref[mm[M]]["full"][b]):
                            bad += 1
                            if len(worst) < 20:
                                worst.append(f"{P}|{lm}|{vit}|{M}|{a}: {float(st[a])!r} vs {ref[mm[M]]['full'][b]!r}")
    return bad == 0, {"n_values": n, "n_bad": bad, "examples": worst}


def g_rerun_vs_frozen(ol, frozen_ol, pres, vits):
    n = bad = 0
    ex = []
    for P in pres:
        for s in ("3b", "7b"):
            for vit in vits:
                a, b = ol[P][s][vit], frozen_ol[P][s][vit]
                for M in RL.METRICS:
                    for Q in RL.SUBSETS:
                        sa, sb = a[M].get(Q), b[M].get(Q)
                        if sa is None or sb is None:
                            continue
                        for f in ("T", "c", "mu", "s_cal", "unc"):
                            n += 1
                            ok = (sa[f] == sb[f]) if M == "mknn" else _close(sa[f], sb[f])
                            if not ok:
                                bad += 1
                                if len(ex) < 20:
                                    ex.append(f"{P}|{s}|{vit}|{M}|{Q}|{f}")
                        n += 1
                        if sa["p"] != sb["p"]:
                            bad += 1
                            if len(ex) < 20:
                                ex.append(f"{P}|{s}|{vit}|{M}|{Q}|p")
                    if M == "mknn":
                        ca, cb = a[M]["_counts"], b[M]["_counts"]
                        n += 1
                        if ca != cb:
                            bad += 1
                            ex.append(f"{P}|{s}|{vit}|mknn|counts")
    return bad == 0, {"n_values": n, "n_bad": bad, "examples": ex}


# ----------------------------------------------------------------------------------------------------------
# section F: P-O2b
# ----------------------------------------------------------------------------------------------------------

def _hkey(d: dict, h: float):
    for k in (fmt(h), f"{h:.4f}", f"{h:g}", str(h), f"{h:.5f}", f"{h:.6f}"):
        if k in d:
            return k
    for k in d:
        try:
            if abs(float(k) - h) <= 1e-12:
                return k
        except ValueError:
            pass
    return None


def kappa(PB: PartB, P, a, vit, o):
    kn = PB.pp(P)["known"]
    Ta, ca = float(kn["T"][a][vit]), float(kn["c_alpha"][a][vit])
    return Ta + (1 - Ta) * (PB.bbar(P, "13b", vit) + o - ca) / (1 - ca)


def forecast(ol, PB: PartB, P, X, vits, lo_tab, hi_tab):
    res = {}
    n = x = und = 0
    kn = PB.pp(P)["known"]
    for a in ("3b", "7b"):
        for vit in vits:
            T13 = float(RL.val(ol[P]["13b"][vit], "cka_biased", "full", "raw"))
            lo, hi = float(lo_tab[a][vit]), float(hi_tab[a][vit])
            if X == "s_cal" and float(kn["s_cal"][a][vit]) == 0.0:
                dec = "undecided"
            elif T13 < lo:
                dec = "below"
            elif T13 > hi:
                dec = "above"
            else:
                dec = "undecided"
            xa = RL.val(ol[P][a][vit], "cka_biased", "full", X)
            xb = RL.val(ol[P]["13b"][vit], "cka_biased", "full", X)
            obs = "below" if xb < xa else ("above" if xb > xa else "tie")
            ok = None if dec == "undecided" else bool(dec == obs)
            res[(a, vit)] = (dec, obs, ok)
            if dec == "undecided":
                und += 1
            else:
                n += 1
                x += bool(ok)
    return res, n, int(x), und


def section_F(v: dict, ol, PB: PartB, pres, vits, notes: dict):
    for P in pres:
        bd = PB.pp(P)["boundaries"]
        for X in ("s_cal", "unc"):
            res, n, x, und = forecast(ol, PB, P, X, vits, bd[X]["lo"], bd[X]["hi"])
            for (a, vit), (dec, obs, ok) in res.items():
                v[f"PO2b|{P}|{X}|dec|{a}|{vit}"] = dec
                v[f"PO2b|{P}|{X}|obs|{a}|{vit}"] = obs
                v[f"PO2b|{P}|{X}|ok|{a}|{vit}"] = ok
            lo, hi = RL.cp_bounds(x, n)
            v.update({f"PO2b|{P}|{X}|n_decided": n, f"PO2b|{P}|{X}|x_correct": x, f"PO2b|{P}|{X}|n_undecided": und,
                      f"PO2b|{P}|{X}|cp_lo": lo, f"PO2b|{P}|{X}|cp_hi": hi,
                      f"PO2b|{P}|{X}|verdict": "fails" if x < n else ("not evaluable" if n < 10 else "holds")})
        # recomputation of the frozen s_cal boundaries from the known values (verification only)
        mx = 0.0
        for a in ("3b", "7b"):
            for vit in vits:
                mx = max(mx, abs(kappa(PB, P, a, vit, BAND[0]) - float(bd["s_cal"]["lo"][a][vit])),
                         abs(kappa(PB, P, a, vit, BAND[1]) - float(bd["s_cal"]["hi"][a][vit])))
        notes[f"{P}|kappa_recomputed_vs_partB_maxabs"] = mx
        hb = bd.get("s_cal_h", {})
        for h in HS:
            k = _hkey(hb, h)
            if k is not None:
                lo_t, hi_t = hb[k]["lo"], hb[k]["hi"]
                notes[f"{P}|h={fmt(h)}|source"] = f"partB s_cal_h[{k!r}]"
            else:   # [R-op] fallback: boundaries recomputed from the frozen known values (plan B6 formula)
                lo_t = {a: {vit: kappa(PB, P, a, vit, O_CENTER - h) for vit in vits} for a in ("3b", "7b")}
                hi_t = {a: {vit: kappa(PB, P, a, vit, O_CENTER + h) for vit in vits} for a in ("3b", "7b")}
                notes[f"{P}|h={fmt(h)}|source"] = "recomputed (key absent in partB s_cal_h)"
            _, n, x, _ = forecast(ol, PB, P, "s_cal", vits, lo_t, hi_t)
            v[f"PO2b|{P}|s_cal|h={fmt(h)}|n_decided"] = n
            v[f"PO2b|{P}|s_cal|h={fmt(h)}|x_correct"] = x


# ----------------------------------------------------------------------------------------------------------
# section J: joint summary (EXPLORATORY)
# ----------------------------------------------------------------------------------------------------------

def dtr_family(rows_score, rows_raw, n):
    a = RL.trend_table(rows_score, n)
    b = RL.trend_table(rows_raw, n)
    return a["T"] - b["T"], a["perms"]


def section_J(v: dict, ol, pool, llama, PB: PartB, P, vits, bloomz: list, llama_slugs: list):
    sB, sL, sO = 1, 1, PB.s_O_J(P)
    CK = "cka_biased"
    v["J|label"] = J_LABEL
    v["J|provenance"] = J_PROV
    for X in ("s_cal", "unc"):
        rB = lambda x: [[RL.val(pool[P][lm][vit], CK, "full", x) for lm in bloomz] for vit in vits]  # noqa: E731
        rL = lambda x: [[RL.val(llama[P][lm][vit], CK, "full", x) for lm in llama_slugs] for vit in vits]  # noqa
        rO = lambda x: [[RL.val(ol[P][s][vit], CK, "full", x) for s in SIZES] for vit in vits]  # noqa: E731
        dB, _ = dtr_family(rB(X), rB("raw"), 5)
        dL, _ = dtr_family(rL(X), rL("raw"), 3)
        dO, permsO = dtr_family(rO(X), rO("raw"), 3)
        aB, aL, aO = sB * dB, sL * dL, sO * dO
        Jmin = np.maximum(np.maximum(aB[:, None, None], aL[None, :, None]), aO[None, None, :])
        Jmean = (aB[:, None, None] + aL[None, :, None] + aO[None, None, :]) / 3.0
        tot = Jmin.size
        jm0, jx0 = float(Jmin[0, 0, 0]), float(Jmean[0, 0, 0])
        pmin = float(np.sum(Jmin <= jm0 + TOL)) / tot
        nB, nL, nO_t = (int(np.sum(aB <= jm0 + TOL)), int(np.sum(aL <= jm0 + TOL)), int(np.sum(aO <= jm0 + TOL)))
        v[f"J|{X}|sO"] = sO
        v[f"J|{X}|Jmin"] = jm0
        v[f"J|{X}|pmin"] = pmin
        v[f"J|{X}|pmin_product"] = nB * nL * nO_t / tot
        v[f"J|{X}|Jmean"] = jx0
        v[f"J|{X}|pmean"] = float(np.sum(Jmean <= jx0 + TOL)) / tot
        v[f"J|{X}|nO"] = int(np.sum(aO <= float(aO[0]) + TOL))
        v[f"J|{X}|dTr|B"] = float(dB[0])
        v[f"J|{X}|dTr|L"] = float(dL[0])
        v[f"J|{X}|dTr|O"] = float(dO[0])
        # conditional size: each sigma_O taken as the observed labelling
        rej = 0
        for k, perm in enumerate(permsO):            # perm[i] = size rank given to SIZES[i]
            t = max(float(aB[0]), float(aL[0]), float(aO[k]))
            p_k = float(np.sum(Jmin <= t + TOL)) / tot
            by_rank = {perm[i]: SIZES[i] for i in range(3)}      # logical size of rank r <- model
            size_of = {"3b": by_rank[1], "7b": by_rank[2], "13b": by_rank[3]}
            oc = order_changes(ol, P, CK, "full", X, vits, NEW_PAIRS, size_of=size_of)
            fminus = oc["down"] if sO == 1 else oc["up"]
            if p_k <= 0.05 and float(aO[k]) <= -MAT and fminus >= 3:
                rej += 1
        v[f"J|{X}|cond_size"] = rej / len(permsO)
