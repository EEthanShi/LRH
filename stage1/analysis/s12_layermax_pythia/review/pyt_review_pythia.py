"""Pythia axis, sub-role R (review recomputation): the Pythia rules (NEW FILE, PYT-AXIS).

Written at S0b by role pyt-code-R (2026-10-10), before any Pythia byte existed on ds2213; to be hashed in the one S0b
CODE FROZEN line. Rules: analysis/s12_layermax_pythia/ANALYSIS_PLAN_PYTHIA.md (sha256 6de89603...) sections 5-10;
key names and operational details: the shared verdict contract VERDICTS_SPEC_PYT.md + verdict_keys.py (role
pyt-code-infra; read by R as the contract, the counterpart of the OpenLLaMA VERDICTS_SPEC.md). No code of the E or I
implementation writers of the Pythia axis was read (plan section 3).

Sections (VERDICTS_SPEC_PYT numbering): C (5) premise rules, implementation checks, D1-D6; H (6) the confirmatory
family; R (7) robustness; D (8) cross-family table and 18-LM panel; G (9) provenance; F (10) P-P2b; K (10) category,
modifiers, readings, curves. Sections A and B are in pyt_review_sections.py (generalised OpenLLaMA code).
Every Part B value (s, s_late, membership, H2 branch, B̄, B_late, bands, flags, curves, width, depth, num_params) is
read from partB_features_pyt.json and never recomputed for a verdict (spec section 1); R's own recomputation of the
Part B rules is written only as x| extras (a consistency check of Part B, never compared).
Choices left open by the plan and the contract are marked [R-op] and listed in review/CHANGES.md.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

import pyt_review_conf as CF
import pyt_review_lib as RL
import pyt_review_sections as RS  # noqa: F401  (Trends type)

CK = "cka_biased"
TOL = 1e-9
PROTOCOLS = RL.SUBSETS + ("ctm",)
XS = ("s_cal", "unc")
BASE_MKNN = Fraction(CF.BASE_MKNN_NUM, CF.BASE_MKNN_DEN)


def fmt(x) -> str:
    return repr(float(x))


def sgnum(x) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def ceil_div(num: int, den: int) -> int:
    """exact integer ceil(num/den) (spec 12: ceil on exact rationals)."""
    return -(-num // den)


def cscore(x: str) -> str:
    return "M" if x == "s_cal" else "M_unc"


def thr_field(x: str) -> str:
    return "c" if x == "s_cal" else "mu"


def spearman(x, y):
    """Spearman = Pearson on average ranks of both vectors; None if either is constant."""
    rx, ry = RL.avg_ranks(list(x)), RL.avg_ranks(list(y))
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    return RL.pearson_vec(rx, ry)


def cp(x, n):
    return RL.cp_bounds(int(x), int(n)) if n else (None, None)


def fin(x):
    return float(x) if x is not None and math.isfinite(float(x)) else None


def lookup_num_key(d: dict, x: float):
    """value of d under the key that spells the number x ('0.005', '0.0050', ...); None if absent."""
    for k in (fmt(x), f"{x:g}", str(x), f"{x:.4f}", f"{x:.5f}", f"{x:.6f}"):
        if k in d:
            return d[k]
    for k in d:
        try:
            if abs(float(k) - x) <= 1e-12:
                return d[k]
        except ValueError:
            pass
    return None


# ======================================================================================================================
# Part B accessor (spec section 11)
# ======================================================================================================================

class PartBPyt:
    def __init__(self, pb: dict, ax: CF.Axis, vits):
        self.pb, self.ax, self.vits = pb, ax, list(vits)

    def pp(self, P):
        return self.pb["per_pre"][P]

    def _m(self, d, size):
        """a per-model dict keyed by size label or slug (spec: Bbar/Blate by lm, band/extrapolated by size)."""
        for k in (size, self.ax.slug[size]):
            if k in d:
                return d[k]
        raise KeyError(f"Part B: no entry for {size}")

    def bbar(self, P, size, vit):
        return float(self._m(self.pp(P)["Bbar"], size)[vit])

    def blate(self, P, size, vit):
        return float(self._m(self.pp(P)["Blate"], size)[vit])

    def prstar(self, P, size):
        return float(self._m(self.pp(P)["PR_star"], size))

    def s(self, P):
        return int(self.pp(P)["s"])

    def s_late(self, P):
        return int(self.pp(P)["s_late"])

    def member_H1(self, P):
        return bool(self.pp(P)["material_Bbar"])

    def branch(self, P):
        return str(self.pp(P)["H2_branch"])

    def same_as_H1(self, P):
        return bool(self.pp(P)["H2_same_as_H1"])

    def band(self, P, size):
        b = self._m(self.pp(P)["band"], size)
        return (float(b["lo"]), float(b["hi"])) if isinstance(b, dict) else (float(b[0]), float(b[1]))

    def band_h(self, P, h, size):
        d = lookup_num_key(self.pp(P)["band_h_curve"], h)
        if d is None:
            return None
        b = self._m(d, size)
        return (float(b["lo"]), float(b["hi"])) if isinstance(b, dict) else (float(b[0]), float(b[1]))

    def band_flat(self, P):
        b = self.pp(P)["band_flat"]
        return float(b[0]), float(b[1])

    def mu_band(self):
        b = self.pb["mu_band"]
        return float(b[0]), float(b[1])

    def extrapolated(self, P, size):
        return bool(self._m(self.pp(P)["extrapolated"], size))

    def material_db(self, P, db):
        r = lookup_num_key(self.pp(P)["material_curve_db"]["Bbar"], db)
        return None if r is None else bool(r)

    def num_params(self, size):
        return int(self._m(self.pb["num_params"], size))

    def width(self, size):
        return int(self._m(self.pb["width"], size))

    def depth(self, size):
        return int(self._m(self.pb["depth_blocks"], size))

    # ---- R's own recomputation of the Part B rules (x| extras only; plan B3, B4, B6)
    def recheck(self, P) -> dict:
        out, diffs = {}, []
        for name, get in (("Bbar", self.bbar), ("Blate", self.blate)):
            rows = [[get(P, s, vit) for s in self.ax.sizes] for vit in self.vits]
            t = float(RL.trend_table(rows, self.ax.n)["T"][0])
            sB = 1 if t >= 0 else -1
            med = RL.median([sB * (get(P, self.ax.sizes[-1], vit) - get(P, self.ax.sizes[0], vit)) for vit in self.vits])
            mat = bool(abs(t) >= CF.MAT_TR - TOL and med >= round(CF.GAP_MULT * CF.DELTA_B, 12))
            out[name] = (t, sB, med, mat)
        st = self.pp(P)
        for k, want in (("material_Bbar", out["Bbar"][3]), ("s", out["Bbar"][1]), ("material_Blate", out["Blate"][3]),
                        ("s_late", out["Blate"][1]), ("H2_branch", "H2a" if out["Blate"][3] else "H2b")):
            if k in st and st[k] != want:
                diffs.append(f"{P}|{k}: partB {st[k]!r} vs R {want!r}")
        for k, want in (("Tr_Bbar", out["Bbar"][0]), ("Tr_Blate", out["Blate"][0])):
            if k in st and abs(float(st[k]) - want) > 1e-12:
                diffs.append(f"{P}|{k}: partB {st[k]!r} vs R {want!r}")
        c = CF.BAND[P]
        for s in self.ax.sizes:
            f = c["a"] + c["b"] / math.sqrt(self.prstar(P, s))
            lo, hi = self.band(P, s)
            if abs(lo - (f - c["h"])) > 1e-12 or abs(hi - (f + c["h"])) > 1e-12:
                diffs.append(f"{P}|band|{s}: partB [{lo!r}, {hi!r}] vs R [{f - c['h']!r}, {f + c['h']!r}]")
            ex = self.prstar(P, s) < CF.PR_RANGE_LO[P]
            if self.extrapolated(P, s) != ex:
                diffs.append(f"{P}|extrapolated|{s}: partB {self.extrapolated(P, s)} vs R {ex}")
        neq = sum(all(abs(self.blate(P, s, v) - self.bbar(P, s, v)) <= 1e-15 * abs(self.bbar(P, s, v)) for v in self.vits)
                  for s in self.ax.sizes)
        if "n_Blate_equals_Bbar" in st and int(st["n_Blate_equals_Bbar"]) != neq:
            diffs.append(f"{P}|n_Blate_equals_Bbar: partB {st['n_Blate_equals_Bbar']} vs R {neq}")
        if "H2_same_as_H1" in st and bool(st["H2_same_as_H1"]) != (neq >= self.ax.n - 1):
            diffs.append(f"{P}|H2_same_as_H1: partB {st['H2_same_as_H1']} vs R {neq >= self.ax.n - 1}")
        return {"recomputed": {k: list(v) for k, v in out.items()}, "n_Blate_eq_Bbar": neq, "diffs": diffs}


def V(ps, M, Q, X):
    return RL.val(ps, M, Q, X)


# ======================================================================================================================
# section C (spec 5)
# ======================================================================================================================

def pp1(ol, ax, vits, P, x):
    nq = nrev = ntie = 0
    th = thr_field(x)
    for vit in vits:
        for a, b in ax.pairs:
            sa, sb = ol[P][a][vit]["mknn"]["full"], ol[P][b][vit]["mknn"]["full"]
            za, zb = abs(sa[th] - BASE_MKNN), abs(sb[th] - BASE_MKNN)
            gap = sb["T"] - sa["T"]
            if abs(gap) > za + zb:
                nq += 1
                dx, dr = RL.sgn(sa[x], sb[x]), sgnum(gap)
                nrev += (dx == -dr and dr != 0)
                ntie += (dx == 0 and dr != 0)
    return nq, nrev, ntie


def pp2ai(ol, PB, ax, vits, P, x, band_of):
    th = thr_field(x)
    per = {}
    for s in ax.sizes:
        offs = [float(V(ol[P][s][vit], CK, "full", th)) - PB.bbar(P, s, vit) for vit in vits]
        lo, hi = band_of(s)
        med = RL.median(offs)
        per[s] = {"median": med, "in_band": bool(lo <= med <= hi), "x": int(sum(lo <= o <= hi for o in offs)),
                  "lo": lo, "hi": hi, "offs": offs}
    n_in = sum(per[s]["in_band"] for s in ax.sizes)
    return per, int(n_in), bool(n_in >= CF.PP2AI_MIN_MODELS)


def pp2aii(ol, PB, ax, vits, P, x, delta=CF.DELTA_B):
    th = thr_field(x)
    n = xx = 0
    for vit in vits:
        for a, b in ax.pairs:
            g = PB.bbar(P, b, vit) - PB.bbar(P, a, vit)
            if abs(g) >= delta:
                n += 1
                xx += sgnum(V(ol[P][b][vit], CK, "full", th) - V(ol[P][a][vit], CK, "full", th)) == sgnum(g)
    need, need67 = ceil_div(9 * n, 10), ceil_div(2 * n, 3)
    vd = "not evaluable" if n < CF.PP_MIN_N else ("holds" if xx >= need else "fails")
    vd67 = "not evaluable" if n < CF.PP_MIN_N else ("holds" if xx >= need67 else "fails")
    return {"n": n, "x": int(xx), "need": need, "need67": need67, "verdict": vd, "verdict67": vd67}


def pp2a_verdict(holds_i: bool, vii: str) -> str:
    if not holds_i or vii == "fails":
        return "fails"
    if holds_i and vii == "holds":
        return "holds"
    return "not evaluable"


def pp5a(ol, PB, ax, vits, P, x, delta=CF.DELTA_B):
    key = "cM" if x == "s_cal" else "muMu"
    n = xx = 0
    for vit in vits:
        for a, b in ax.pairs:
            g = PB.blate(P, b, vit) - PB.blate(P, a, vit)
            if abs(g) >= delta:
                n += 1
                xx += sgnum(ol[P][b][vit][CK]["ctm"][key] - ol[P][a][vit][CK]["ctm"][key]) == sgnum(g)
    need = ceil_div(2 * n, 3)
    vd = "not evaluable" if n < CF.PP_MIN_N else ("holds" if xx >= need else "fails")
    return {"n": n, "x": int(xx), "need": need, "verdict": vd}


def keep_and_sign(ol, ax, vits, P, M, Q, x):
    """OpenLLaMA keep_and_sign over every size pair of the axis (spec 5, plan section 8, per comparison)."""
    nk = ns = 0
    exact = M == "mknn"
    for vit in vits:
        for a, b in ax.pairs:
            sa, sb = ol[P][a][vit][M].get(Q), ol[P][b][vit][M].get(Q)
            if sa is None or sb is None:
                return None, None
            Ta, Tb = sa["T"], sb["T"]
            Xa, Xb = sa[x], sb[x]
            ta, tb = sa[thr_field(x)], sb[thr_field(x)]
            if x == "s_cal":
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
            dr, dx = RL.sgn(Ta, Tb), RL.sgn(Xa, Xb)
            if tb > ta:
                ns += (dr <= 0 and dx == 1)
            elif tb < ta:
                ns += (dr >= 0 and dx == -1)
            else:
                ns += (dx == -dr and dr != 0)
    return int(nk), int(ns)


def _tie_not_raw(xrow, rrow) -> bool:
    """two sizes tied under the score but not under raw (a censoring tie, spec 5 [plan-gap])."""
    n = len(xrow)
    return any(xrow[i] == xrow[j] and rrow[i] != rrow[j] for i in range(n) for j in range(i + 1, n))


def sign_vit(ol, T, ax, vits, P, M, Q, x):
    th = thr_field(x)
    rows_x, rows_r = T.rows(P, M, Q, x), T.rows(P, M, Q, "raw")
    px, pr = T.parts(P, M, Q, x), T.parts(P, M, Q, "raw")
    rx, rr = px.rho_id(), pr.rho_id()
    nu = vu = nd = vd = ne = 0
    for k, vit in enumerate(vits):
        if not (px.usable[k] and pr.usable[k]):
            continue
        t = [ol[P][s][vit][M][Q][th] for s in ax.sizes]
        up = all(t[i] <= t[i + 1] for i in range(len(t) - 1))
        down = all(t[i] >= t[i + 1] for i in range(len(t) - 1))
        if not (up or down):
            continue
        if _tie_not_raw(rows_x[k], rows_r[k]):
            ne += 1
            continue
        if up:
            nu += 1
            vu += rx[k] > rr[k] + 1e-12
        if down:
            nd += 1
            vd += rx[k] < rr[k] - 1e-12
    return {"nmono_up": nu, "nviol_up": int(vu), "nmono_down": nd, "nviol_down": int(vd), "nexcl_tie": ne}


def section_C(v, ol, T, PB, ax, vits, pres):
    """Returns {(P, x): {"pp2ai": per-model dict}} for section K."""
    res = {}
    for P in pres:
        hold_all = True
        for x in XS:
            nq, nrev, ntie = pp1(ol, ax, vits, P, x)
            v.update({f"PP1|{P}|{x}|nq": nq, f"PP1|{P}|{x}|nrev": nrev, f"PP1|{P}|{x}|ntie": ntie,
                      f"PP1|{P}|{x}|holds": bool(nrev == 0)})
            hold_all = hold_all and nrev == 0
            band_of = (lambda s: PB.band(P, s)) if x == "s_cal" else (lambda s: PB.mu_band())
            per, n_in, holds_i = pp2ai(ol, PB, ax, vits, P, x, band_of)
            for s in ax.sizes:
                lo, hi = cp(per[s]["x"], len(vits))
                v.update({f"PP2ai|{P}|{x}|median|{s}": per[s]["median"], f"PP2ai|{P}|{x}|lo|{s}": per[s]["lo"],
                          f"PP2ai|{P}|{x}|hi|{s}": per[s]["hi"], f"PP2ai|{P}|{x}|in_band|{s}": per[s]["in_band"],
                          f"PP2ai|{P}|{x}|x|{s}": per[s]["x"], f"PP2ai|{P}|{x}|cp_lo|{s}": lo,
                          f"PP2ai|{P}|{x}|cp_hi|{s}": hi})
                for k, vit in enumerate(vits):
                    v[f"PP2ai|{P}|{x}|offset|{s}|{vit}"] = per[s]["offs"][k]
            v[f"PP2ai|{P}|{x}|n_models_in_band"] = n_in
            v[f"PP2ai|{P}|{x}|holds"] = holds_i
            v[f"PP2ai|{P}|{x}|failing"] = ",".join(s for s in ax.sizes if not per[s]["in_band"])
            r2 = pp2aii(ol, PB, ax, vits, P, x)
            lo, hi = cp(r2["x"], r2["n"])
            for k in ("n", "x", "need", "need67", "verdict", "verdict67"):
                v[f"PP2aii|{P}|{x}|{k}"] = r2[k]
            v[f"PP2aii|{P}|{x}|cp_lo"], v[f"PP2aii|{P}|{x}|cp_hi"] = lo, hi
            v[f"PP2a|{P}|{x}|verdict"] = pp2a_verdict(holds_i, r2["verdict"])
            r5 = pp5a(ol, PB, ax, vits, P, x)
            lo, hi = cp(r5["x"], r5["n"])
            for k in ("n", "x", "need", "verdict"):
                v[f"PP5a|{P}|{x}|{k}"] = r5[k]
            v[f"PP5a|{P}|{x}|cp_lo"], v[f"PP5a|{P}|{x}|cp_hi"] = lo, hi
            for r in ("0.5", "1.0"):
                Q = f"fixed_r{r}"
                trr, trx = T.tr(P, CK, Q, "raw"), T.tr(P, CK, Q, x)
                if trr is None or abs(trr) < CF.MAT:
                    desc = "not applied"
                else:
                    sr = 1 if trr > 0 else -1
                    if trx is None:
                        desc = "undefined"
                    elif sr * trx >= CF.MAT:
                        desc = "keeps sign"
                    elif abs(trx) < CF.MAT:
                        desc = "lost"
                    else:
                        desc = "reversed"
                v[f"PP3|{P}|{r}|{x}|desc"] = desc
            res[(P, x)] = {"pp2ai": per}
        v[f"PP1|{P}|holds"] = bool(hold_all)
        perf, n_f, holds_f = pp2ai(ol, PB, ax, vits, P, "s_cal", lambda s: PB.band_flat(P))
        for s in ax.sizes:
            v[f"PP2ai|{P}|s_cal|flat|in_band|{s}"] = perf[s]["in_band"]
        v[f"PP2ai|{P}|s_cal|flat|n_models_in_band"], v[f"PP2ai|{P}|s_cal|flat|holds"] = n_f, holds_f
        # ---- implementation checks (must be 0 for every outcome)
        for M in RL.METRICS:
            for Q in RL.SUBSETS:
                for x in XS:
                    nk, ns = keep_and_sign(ol, ax, vits, P, M, Q, x)
                    v[f"keepid|{P}|{M}|{Q}|{x}|nviol"] = nk
                    v[f"signrule|{P}|{M}|{Q}|{x}|nviol"] = ns
                    for k, val in sign_vit(ol, T, ax, vits, P, M, Q, x).items():
                        v[f"signvit|{P}|{M}|{Q}|{x}|{k}"] = val
        # ---- D3 deductive check (spec 5)
        rowsM, rowsTM = T.rows(P, CK, "ctm", "M"), T.rows(P, CK, "ctm", "TM")
        rM, rTM = T.parts(P, CK, "ctm", "M").rho_id(), T.parts(P, CK, "ctm", "TM").rho_id()
        nv3 = ne3 = 0
        for k, vit in enumerate(vits):
            c3 = [ol[P][s][vit][CK]["ctm"]["cM"] for s in ax.sizes]
            if not all(c3[i] < c3[i + 1] for i in range(len(c3) - 1)):
                continue
            if _tie_not_raw(rowsM[k], rowsTM[k]):
                ne3 += 1
                continue
            if rM[k] is not None and rTM[k] is not None and rM[k] - rTM[k] > 1e-12:
                nv3 += 1
        v[f"D3|{P}|nviol"], v[f"D3|{P}|nexcl_tie"] = int(nv3), int(ne3)
        # ---- plug-in (spec 5: diff on V(T - B̄, unc))
        rows = [[float(V(ol[P][s][vit], CK, "full", "raw")) - PB.bbar(P, s, vit) for s in ax.sizes] for vit in vits]
        pa = RL.TrendParts(rows, ax.n)
        dvec, _, nV = RL.diff_vec(pa, T.parts(P, CK, "full", "unc"), "common")
        v[f"plugin|{P}|tr"] = fin(pa.tr_vec()[0])
        v[f"plugin|{P}|diff"] = fin(dvec[0])
        v[f"plugin|{P}|nv"] = nV
        v[f"plugin|{P}|within"] = bool(math.isfinite(dvec[0]) and abs(dvec[0]) <= CF.MAT)
        # ---- OP-style reading (single-score trends)
        tmu = T.tr(P, "mknn", "full", "unc")
        for M in ("cka_biased", "cka_debiased"):
            trr, tru = T.tr(P, M, "full", "raw"), T.tr(P, M, "full", "unc")
            if trr is None or trr < CF.LOSS_RAW:
                rd = "not applied"
            elif tru is not None and tmu is not None and tru <= CF.LOSS_CAL and tmu >= CF.LOSS_RAW:
                rd = "loses"
            elif tru is not None and tru >= CF.LOSS_RAW:
                rd = "no loss"
            else:
                rd = "intermediate"
            v[f"readO|{P}|{M}"] = rd
        # ---- D1, D2 per size
        for M in RL.METRICS:
            for s in ax.sizes:
                d1 = [ol[P][s][vit][M]["full"]["mu"] - ol[P][s][vit][M]["entry_mu_max"] for vit in vits]
                v[f"D1|{P}|{M}|{s}"] = RL.median(d1)
                ar = [ol[P][s][vit][M]["full"]["aA"] / (ol[P][s][vit]["L_A"] - 1) for vit in vits]
                v[f"D2|{P}|{M}|{s}|argmax_rel"] = RL.median(ar)
                cc = [ol[P][s][vit][M]["full"]["c"] - ol[P][s][vit][M]["c_entry_full_argmax"] for vit in vits]
                v[f"D2|{P}|{M}|{s}|c_minus_centry"] = RL.median(cc)
        # ---- D4, D5, D6 (descriptive)
        cov = {"logparams": [math.log(PB.num_params(s)) for s in ax.sizes], "width": [PB.width(s) for s in ax.sizes],
               "depth": [PB.depth(s) for s in ax.sizes]}
        qty = {"PRstar": [PB.prstar(P, s) for s in ax.sizes],
               "Bbar": [float(np.mean([PB.bbar(P, s, vit) for vit in vits])) for s in ax.sizes],
               "calpha": [float(np.mean([float(V(ol[P][s][vit], CK, "full", "c")) for vit in vits])) for s in ax.sizes],
               "T": [float(np.mean([float(V(ol[P][s][vit], CK, "full", "raw")) for vit in vits])) for s in ax.sizes]}
        for qn, qv in qty.items():
            for cn, cvv in cov.items():
                v[f"D4|{P}|{qn}|{cn}"] = spearman(cvv, qv)
            for a, b in CF.D4_CONTRASTS:
                v[f"D4|{P}|{qn}|step|{a}|{b}"] = qv[ax.idx(b)] - qv[ax.idx(a)]
        for i in range(ax.n - 1):
            a, b = ax.sizes[i], ax.sizes[i + 1]
            v[f"D5|{P}|{a}|{b}|raw_gain"] = RL.median([float(V(ol[P][b][vit], CK, "full", "raw"))
                                                       - float(V(ol[P][a][vit], CK, "full", "raw")) for vit in vits])
            v[f"D5|{P}|{a}|{b}|thr_gain"] = RL.median([float(V(ol[P][b][vit], CK, "full", "c"))
                                                       - float(V(ol[P][a][vit], CK, "full", "c")) for vit in vits])
        a, b = ax.sizes[0], ax.sizes[-1]
        dl = math.log2(PB.num_params(b) / PB.num_params(a))
        v[f"D5|{P}|g"] = RL.median([(float(V(ol[P][b][vit], CK, "full", "raw"))
                                     - float(V(ol[P][a][vit], CK, "full", "raw"))) / dl for vit in vits])
        v[f"D5|{P}|h"] = RL.median([(float(V(ol[P][b][vit], CK, "full", "c"))
                                     - float(V(ol[P][a][vit], CK, "full", "c"))) / dl for vit in vits])
        Ls = [ol[P][s][vits[0]]["L_A"] for s in ax.sizes]
        for M, base in (("mknn", BASE_MKNN), ("cka_debiased", 0.0)):
            med = [RL.median([float(ol[P][s][vit][M]["full"]["c"] - base) for vit in vits]) for s in ax.sizes]
            for s, m in zip(ax.sizes, med):
                v[f"D6|{P}|{M}|{s}|c_minus_base"] = m
            v[f"D6|{P}|{M}|spearman_L"] = spearman(Ls, med)
            v[f"D6|{P}|{M}|step|410m|1b"] = med[ax.idx("1b")] - med[ax.idx("410m")]
    return res


# ======================================================================================================================
# section H (spec 6): the confirmatory family
# ======================================================================================================================

def boot_key(kind, M, Q, P, x, suffix=""):
    """spec 6: '<kind>|<M>|<Q>|<P>', '|unc' appended for x = unc, robustness suffix after it."""
    return CF.BOOT + f"{kind}|{M}|{Q}|{P}" + ("" if x == "s_cal" else "|unc") + suffix


def common_D(px: RL.TrendParts, pb: RL.TrendParts, sign=1):
    """(R, C, D = sign (C - R), vec of sign [Tr_V(X; s) - Tr_V(base; s)], per-ViT sign*drho on V (ViT order), n_V)."""
    Vm = px.usable & pb.usable
    if not Vm.any():
        return None, None, None, np.full(px.R.shape[0], np.nan), [], 0
    R = float(pb.R[0, Vm].mean())
    C = float(px.R[0, Vm].mean())
    vec = sign * (px.R[:, Vm].mean(axis=1) - pb.R[:, Vm].mean(axis=1))
    dv = [float(sign * (px.R[0, k] - pb.R[0, k])) for k in range(px.nv) if Vm[k]]
    return R, C, sign * (C - R), vec, dv, int(Vm.sum())


class Family:
    """Every statistic of spec section 6 for one (P, x)."""

    def __init__(self, T, PB: PartBPyt, P, x, draws_cache):
        self.P, self.x = P, x
        xm = cscore(x)

        def draws(key, dv):
            if key not in draws_cache:
                draws_cache[key] = RL.boot_draws(dv, key, CF.NB)
            return draws_cache[key]
        self.s, self.s_late = PB.s(P), PB.s_late(P)
        self.member = PB.member_H1(P)
        self.branch = PB.branch(P)
        # H1
        R, C, D, vec, dv, nV = common_D(T.parts(P, CK, "full", x), T.parts(P, CK, "full", "raw"), self.s)
        self.R, self.C, self.D1, self.nV = R, C, D, nV
        self.p_perm = RL.p_le(vec)
        self.d1_draws = draws(boot_key("H1mag", CK, "full", P, x), dv)
        self.p_mag = RL.p_mag(self.d1_draws, CF.MAT)
        self.p_mag_m = {m: RL.p_mag(self.d1_draws, m) for m in CF.MARGINS}
        self.pH1 = None if self.p_perm is None or self.p_mag is None else max(self.p_perm, self.p_mag)
        self.conf_raw = bool(self.pH1 is not None and self.pH1 <= CF.ALPHA and D is not None and D <= -CF.MAT)
        self.H1_confirmed = self.conf_raw if self.member else None
        if np.all(np.isfinite(self.d1_draws)):
            self.ci = tuple(float(z) for z in np.percentile(self.d1_draws, [2.5, 97.5]))
        else:
            self.ci = (None, None)
        if D is None:
            self.kedge = None
        else:
            lo, hi = self.ci
            self.kedge = bool(min(abs(D + CF.MAT), abs(D - CF.MAT)) <= CF.KNIFE_SWAP + TOL
                              or (lo is not None and (lo <= -CF.MAT <= hi or lo <= CF.MAT <= hi)))
        # TOST for every (M, Q) (spec 6), with base raw full
        self.tost = {}
        for M in RL.METRICS:
            for Q in ("full", "ctm"):
                X = xm if Q == "ctm" else x
                pX, pB = T.parts(P, M, Q, X), T.parts(P, M, "full", "raw")
                _, _, d, _, dvt, nVt = common_D(pX, pB, 1)
                dr = draws(boot_key("tost", M, Q, P, x), dvt)
                tp = {m: RL.tost_p(d, dr, m) for m in CF.MARGINS}
                self.tost[(M, Q)] = {"D": d, "nV": nVt if nVt else None,  # contract 6: null for all if V is empty
                                     "p_lo": tp[CF.MAT]["p_lo"], "p_hi": tp[CF.MAT]["p_hi"],
                                     "p": tp[CF.MAT]["p"], "tp": tp,
                                     "degenerate": bool(RL.degenerate(pX, pB)) if nVt else None,
                                     "ranks_identical": bool(RL.same_ranks(pX, pB)) if nVt else None}
        # H2 (both branches computed; p by branch)
        _, _, D2, vec2, dv2, nV2 = common_D(T.parts(P, CK, "ctm", xm), T.parts(P, CK, "full", "raw"), self.s_late)
        self.D2, self.nV2 = D2, nV2
        self.p_perm2 = RL.p_le(vec2)
        self.d2_draws = draws(boot_key("H2mag", CK, "ctm", P, x), dv2)
        self.p_mag2 = RL.p_mag(self.d2_draws, CF.MAT)
        self.p_mag2_m = {m: RL.p_mag(self.d2_draws, m) for m in CF.MARGINS}
        if self.branch == "H2a":
            self.p2 = None if self.p_perm2 is None or self.p_mag2 is None else max(self.p_perm2, self.p_mag2)
        else:
            self.p2 = self.tost[(CK, "ctm")]["p"]
        self.p = {"H2": self.p2, "H3": self.tost[("mknn", "full")]["p"], "H4": self.tost[("cka_debiased", "full")]["p"]}
        # gate and Holm
        if self.member and self.conf_raw:
            self.gate = "open (H1 confirmed)"
        elif not self.member:
            self.gate = "open (H1 not a member)"
        else:
            self.gate = "closed"
        self.open = self.gate != "closed"
        if self.open:
            pv = {h: (1.0 if self.p[h] is None else self.p[h]) for h in CF.HYPS}   # [R-op] a null p never rejects
            self.holm = RL.holm(pv, CF.ALPHA, order=list(CF.HYPS))
            self.holm_order = ",".join(sorted(CF.HYPS, key=lambda h: self.holm[h]["rank"]))
        else:
            self.holm, self.holm_order = None, None
        self.confirmed = {}
        for h in CF.HYPS:
            if not self.open:
                self.confirmed[h] = None
            elif h == "H2" and self.branch == "H2a":
                self.confirmed[h] = bool(self.holm[h]["reject"] and D2 is not None and D2 <= -CF.MAT)
            else:
                self.confirmed[h] = bool(self.holm[h]["reject"])


def section_H(v, fams, pres):
    for P in pres:
        for x in XS:
            F = fams[(P, x)]
            h = f"H|{P}|{x}"
            v.update({f"{h}|H1|member": F.member, f"{h}|H1|s": F.s, f"{h}|H1|D": F.D1, f"{h}|H1|nV": F.nV,
                      f"{h}|H1|R": F.R, f"{h}|H1|C": F.C, f"{h}|H1|p_perm": F.p_perm, f"{h}|H1|p_mag": F.p_mag,
                      f"{h}|H1|p": F.pH1, f"{h}|H1|confirmed": F.H1_confirmed, f"{h}|H1|ci_lo": F.ci[0],
                      f"{h}|H1|ci_hi": F.ci[1], f"{h}|H1|kedge": F.kedge})
            for m in CF.MARGINS:
                v[f"{h}|H1|p_mag|m={fmt(m)}"] = F.p_mag_m[m]
                v[f"{h}|H2|p_mag|m={fmt(m)}"] = F.p_mag2_m[m]
            t2 = F.tost[(CK, "ctm")]
            v.update({f"{h}|H2|branch": F.branch, f"{h}|H2|s_late": F.s_late, f"{h}|H2|D": F.D2,
                      f"{h}|H2|nV": F.nV2, f"{h}|H2|p_perm": F.p_perm2, f"{h}|H2|p_mag": F.p_mag2,
                      f"{h}|H2|p_lo": t2["p_lo"], f"{h}|H2|p_hi": t2["p_hi"], f"{h}|H2|p": F.p2,
                      f"{h}|H2|degenerate": t2["degenerate"], f"{h}|H2|ranks_identical": t2["ranks_identical"]})
            for m in CF.MARGINS:
                v[f"{h}|H2|p_tost|m={fmt(m)}"] = t2["tp"][m]["p"]
            for hy, key in (("H3", ("mknn", "full")), ("H4", ("cka_debiased", "full"))):
                t = F.tost[key]
                for c in ("D", "nV", "p_lo", "p_hi", "p", "degenerate", "ranks_identical"):
                    v[f"{h}|{hy}|{c}"] = t[c]
                for m in CF.MARGINS:
                    v[f"{h}|{hy}|p_tost|m={fmt(m)}"] = t["tp"][m]["p"]
            v[f"{h}|gate"], v[f"{h}|holm_order"] = F.gate, F.holm_order
            for hy in CF.HYPS:
                v[f"{h}|{hy}|tested"] = bool(F.open)
                v[f"{h}|{hy}|p_holm"] = F.holm[hy]["p_adj"] if F.open else None
                v[f"{h}|{hy}|holm_reject"] = bool(F.holm[hy]["reject"]) if F.open else None
                v[f"{h}|{hy}|confirmed"] = F.confirmed[hy]
            for (M, Q), t in F.tost.items():
                for c in ("D", "nV", "p_lo", "p_hi", "p", "degenerate", "ranks_identical"):
                    v[f"TOST|{P}|{x}|{M}|{Q}|{c}"] = t[c]


# ======================================================================================================================
# section R (spec 7): robustness
# ======================================================================================================================

def sub_parts(T, P, M, Q, X, cols, cache):
    k = (P, M, Q, X, tuple(cols))
    if k not in cache:
        rows = T.rows(P, M, Q, X)
        cache[k] = RL.TrendParts([[r[i] for i in cols] for r in rows], len(cols))
    return cache[k]


def masked_diff(a, b, mask, sign=1):
    Vm = a.usable & b.usable & mask
    if not Vm.any():
        return None
    return float(sign * (a.R[0, Vm] - b.R[0, Vm]).mean())


def section_R(v, fams, ol, T, ax, vits, pres, draws_cache):
    cache = {}
    for P in pres:
        zs = {s: [float(V(ol[P][s][vit], CK, "full", "s_cal")) for vit in vits] for s in ax.sizes}
        v[f"R|{P}|n_scal_zero"] = int(sum(z == 0.0 for s in ax.sizes for z in zs[s]))
        for s in ax.sizes:
            v[f"R|{P}|median_scal|{s}"] = RL.median(zs[s])
        for x in XS:
            F = fams[(P, x)]
            r = f"R|{P}|{x}"
            s_ = F.s
            frag = []
            for i, z in enumerate(ax.sizes):
                cols = [c for c in range(ax.n) if c != i]
                a, b = sub_parts(T, P, CK, "full", x, cols, cache), sub_parts(T, P, CK, "full", "raw", cols, cache)
                _, _, d1, vec, dv, _ = common_D(a, b, s_)
                pperm = RL.p_le(vec)
                key = boot_key("H1mag", CK, "full", P, x, f"|loo|{z}")
                if key not in draws_cache:
                    draws_cache[key] = RL.boot_draws(dv, key, CF.NB)
                pm = RL.p_mag(draws_cache[key], CF.MAT)
                p = None if pperm is None or pm is None else max(pperm, pm)
                v.update({f"{r}|loo|{z}|D1": d1, f"{r}|loo|{z}|p_perm": pperm, f"{r}|loo|{z}|p_mag": pm,
                          f"{r}|loo|{z}|p": p})
                for M, tag in (("mknn", "dtr_mknn"), ("cka_debiased", "dtr_dcka")):
                    a2, b2 = sub_parts(T, P, M, "full", x, cols, cache), sub_parts(T, P, M, "full", "raw", cols, cache)
                    vv = RL.diff_vec(a2, b2, "common")[0]
                    v[f"{r}|loo|{z}|{tag}"] = fin(vv[0])
                if ((d1 is not None and d1 <= -CF.MAT) != (F.D1 is not None and F.D1 <= -CF.MAT)
                        or (p is not None and p <= CF.ALPHA) != (F.pH1 is not None and F.pH1 <= CF.ALPHA)):
                    frag.append(z)
            v[f"{r}|fragile"], v[f"{r}|fragile_sizes"] = bool(frag), ",".join(frag)
            cols = [ax.idx(z) for z in ax.five]
            a, b = sub_parts(T, P, CK, "full", x, cols, cache), sub_parts(T, P, CK, "full", "raw", cols, cache)
            _, _, d5, vec, _, _ = common_D(a, b, s_)
            v[f"{r}|five|D1"], v[f"{r}|five|p_perm"] = d5, RL.p_le(vec)
            if d5 is None or F.D1 is None:
                v[f"{r}|five|carried"] = v[f"{r}|five|carried_D"] = None
            else:
                sd = sgnum(d5) != sgnum(F.D1)
                v[f"{r}|five|carried"] = bool(sd or ((d5 <= -CF.MAT) != (F.D1 <= -CF.MAT)))
                v[f"{r}|five|carried_D"] = bool(sd or ((d5 >= CF.MAT) != (F.D1 >= CF.MAT)))
            g = v[f"dtr|{P}|{CK}|grid5|{x}"]
            v[f"{r}|grid5|D1"] = None if g is None else s_ * g
            v[f"{r}|grid5|dtr_mknn"] = v[f"dtr|{P}|mknn|grid5|{x}"]
            v[f"{r}|grid5|dtr_dcka"] = v[f"dtr|{P}|cka_debiased|grid5|{x}"]
            for fam in CF.FAMILIES4:
                mask = np.array([CF.VIT_FAMILY4[vit] != fam for vit in vits], dtype=bool)
                v[f"{r}|lofo|{fam}|D1"] = masked_diff(T.parts(P, CK, "full", x), T.parts(P, CK, "full", "raw"), mask, s_)
                v[f"{r}|lofo|{fam}|dtr_mknn"] = masked_diff(T.parts(P, "mknn", "full", x),
                                                            T.parts(P, "mknn", "full", "raw"), mask)
                v[f"{r}|lofo|{fam}|dtr_dcka"] = masked_diff(T.parts(P, "cka_debiased", "full", x),
                                                            T.parts(P, "cka_debiased", "full", "raw"), mask)
            d0 = v[f"dtr0|{P}|{CK}|full|{x}"]
            v[f"{r}|rho0|D1"] = None if d0 is None else s_ * d0


# ======================================================================================================================
# section D (spec 8): cross-family table and 18-LM panel
# ======================================================================================================================

def section_D(v, ol, pool, num_params, ax, vits, pres):
    """pool[P][slug][vit] = PairStats (protocol full) of the 11 known LMs, this implementation's own format."""
    fams = {"BLOOMZ": list(CF.LM_FAMILIES["BLOOMZ"]), "LLaMA": list(CF.LM_FAMILIES["LLaMA"]),
            "OpenLLaMA": list(CF.LM_FAMILIES["OpenLLaMA"]), "Pythia": [ax.slug[s] for s in ax.sizes]}
    rows18 = fams["BLOOMZ"] + fams["LLaMA"] + fams["OpenLLaMA"] + fams["Pythia"]
    xlog = np.log(np.array([float(num_params[lm]) for lm in rows18]))
    for P in pres:
        def st(lm, vit):
            if lm in pool[P]:
                return pool[P][lm][vit]
            return ol[P][[z for z in ax.sizes if ax.slug[z] == lm][0]][vit]
        for M in RL.METRICS:
            parts = {}
            for X in ("raw", "s_cal", "unc"):
                G = np.array([[float(V(st(lm, vit), M, "full", X)) for vit in vits] for lm in rows18])
                r = RL.pear(G, xlog)
                lo, hi = RL.boot_r(G, xlog, CF.BOOTVIT + f"panel18|{P}|{M}|{X}")
                v[f"panel18|{P}|{M}|{X}|r"] = r if math.isfinite(r) else None
                v[f"panel18|{P}|{M}|{X}|ci_lo"], v[f"panel18|{P}|{M}|{X}|ci_hi"] = lo, hi
                for fam, lms in fams.items():
                    rows = [[V(st(lm, vit), M, "full", X) for lm in lms] for vit in vits]
                    parts[(fam, X)] = RL.TrendParts(rows, len(lms))
                    v[f"famtr|{P}|{M}|{X}|{fam}"] = fin(parts[(fam, X)].tr_vec()[0])
            for x in XS:
                for fam in fams:
                    d = RL.diff_vec(parts[(fam, x)], parts[(fam, "raw")], "common")[0]
                    v[f"famdtr|{P}|{M}|{x}|{fam}"] = fin(d[0])


# ======================================================================================================================
# section F (spec 10): P-P2b
# ======================================================================================================================

def scal_at(T, c):
    den = 1.0 - c
    return max((T - c) / den, 0.0) if den > 0 else 0.0


def forecast_one(x, Ta, Tb, Ba, Bb, ba, bb):
    """j = a (smaller), j' = b (larger); ba, bb = the bands (o_lo, o_hi); for x = unc both are mu_band."""
    if x == "s_cal":
        bmax, bmin = scal_at(Tb, Bb + bb[0]), scal_at(Tb, Bb + bb[1])
        amax, amin = scal_at(Ta, Ba + ba[0]), scal_at(Ta, Ba + ba[1])
        if bmax < amin:
            return "below"
        if bmin > amax:
            return "above"
        if bmax == 0.0 and amax == 0.0:
            return "tie"
        return "undecided"
    if Tb - Bb - bb[0] < Ta - Ba - ba[1]:
        return "below"
    if Tb - Bb - bb[1] > Ta - Ba - ba[0]:
        return "above"
    return "undecided"


def pp2b(ol, PB, ax, vits, P, x, band_of, keep=None):
    n_dec = n_ok = n_und = n_tie = n_flip = 0
    wrong = []
    for a, b in ax.pairs:
        for vit in vits:
            Ta = float(V(ol[P][a][vit], CK, "full", "raw"))
            Tb = float(V(ol[P][b][vit], CK, "full", "raw"))
            dec = forecast_one(x, Ta, Tb, PB.bbar(P, a, vit), PB.bbar(P, b, vit), band_of(a), band_of(b))
            xa, xb = V(ol[P][a][vit], CK, "full", x), V(ol[P][b][vit], CK, "full", x)
            obs = "below" if xb < xa else ("above" if xb > xa else "tie")
            ok = None if dec == "undecided" else bool(dec == obs)
            if keep is not None:
                keep[(a, b, vit)] = (dec, obs, ok)
            if dec == "undecided":
                n_und += 1
            else:
                n_dec += 1
                n_ok += bool(ok)
                n_tie += dec == "tie"
                if not ok:
                    wrong.append((a, b, vit))
            n_flip += (Tb > Ta and dec == "below")
    n_wrong = n_dec - n_ok
    vd = "fails" if n_wrong > 0 else ("not evaluable" if n_dec < CF.PP2B_MIN_DECIDED else "holds")
    return {"n_decided": n_dec, "x_correct": int(n_ok), "n_wrong": int(n_wrong), "n_undecided": n_und,
            "n_tie_fc": n_tie, "n_flip_fc_rawconc": int(n_flip), "verdict": vd, "wrong": wrong}


def section_F(v, ol, PB, ax, vits, pres):
    for P in pres:
        for x in XS:
            keep = {}
            band_of = (lambda s: PB.band(P, s)) if x == "s_cal" else (lambda s: PB.mu_band())
            r = pp2b(ol, PB, ax, vits, P, x, band_of, keep)
            for (a, b, vit), (dec, obs, ok) in keep.items():
                v[f"PP2b|{P}|{x}|dec|{a}|{b}|{vit}"] = dec
                v[f"PP2b|{P}|{x}|obs|{a}|{b}|{vit}"] = obs
                v[f"PP2b|{P}|{x}|ok|{a}|{b}|{vit}"] = ok
            for k in ("n_decided", "x_correct", "n_wrong", "n_undecided", "n_tie_fc", "n_flip_fc_rawconc", "verdict"):
                v[f"PP2b|{P}|{x}|{k}"] = r[k]
            v[f"PP2b|{P}|{x}|cp_lo"], v[f"PP2b|{P}|{x}|cp_hi"] = cp(r["x_correct"], r["n_decided"])
            if r["n_wrong"] == 0:
                v[f"PP2b|{P}|{x}|wrong_all_extrap"] = None
            else:
                v[f"PP2b|{P}|{x}|wrong_all_extrap"] = bool(all(PB.extrapolated(P, a) or PB.extrapolated(P, b)
                                                               for a, b, _ in r["wrong"]))
            ws = {z for a, b, _ in r["wrong"] for z in (a, b)}
            v[f"PP2b|{P}|{x}|wrong_sizes"] = ",".join(s for s in ax.sizes if s in ws)
        for h in CF.HS:
            if PB.band_h(P, h, ax.sizes[0]) is None:
                v[f"PP2b|{P}|s_cal|h={fmt(h)}|n_decided"] = v[f"PP2b|{P}|s_cal|h={fmt(h)}|x_correct"] = None
                continue
            r = pp2b(ol, PB, ax, vits, P, "s_cal", lambda s: PB.band_h(P, h, s))
            v[f"PP2b|{P}|s_cal|h={fmt(h)}|n_decided"] = r["n_decided"]
            v[f"PP2b|{P}|s_cal|h={fmt(h)}|x_correct"] = r["x_correct"]
        r = pp2b(ol, PB, ax, vits, P, "s_cal", lambda s: PB.band_flat(P))
        for k in ("n_decided", "x_correct", "verdict"):
            v[f"PP2b|{P}|s_cal|flat|{k}"] = r[k]


# ======================================================================================================================
# section K (spec 10): category, modifiers, readings, knife edge, curves
# ======================================================================================================================

F_REASON_PERM = "not consistent across the seven sizes"
F_REASON_MAG = "size of the drop not established"
A_FAMILY = ("A", "A-att", "A-prime")
KEPT_READINGS = ("equivalence confirmed", "kept, equivalence not established", "kept, not tested (gate closed)",
                 "no comparison changed order")


def category(member, D1, p_perm, p_mag, pp2a, pp2b_v, R, C, tau=CF.MAT, conf=None):
    """plan section 7 table in order (spec 10). conf = H1 confirmed (default: p <= 0.05 and D1 <= -tau).
    Returns (code, F_reason or None, premise or None)."""
    if not member:
        return "E", None, None
    if D1 is None:
        return None, None, None
    if D1 >= tau:
        return "D", None, None
    if abs(D1) < tau:
        return "C", None, None
    p = None if p_perm is None or p_mag is None else max(p_perm, p_mag)
    if conf is None:
        conf = p is not None and p <= CF.ALPHA and D1 <= -tau
    if not conf:
        rs = []
        if p_perm is None or p_perm > CF.ALPHA:
            rs.append(F_REASON_PERM)
        if p_mag is None or p_mag > CF.ALPHA:
            rs.append(F_REASON_MAG)
        return "F", ("; ".join(rs) if rs else None), None
    if pp2a == "fails" or pp2b_v == "fails":
        return "B", None, None
    if R is not None and R >= CF.LOSS_RAW:
        code = "A" if (C is not None and C <= CF.LOSS_CAL) else "A-att"
    else:
        code = "A-prime"
    prem = "verified" if (pp2a == "holds" and pp2b_v == "holds") else "unverified"
    return code, None, prem


def immunity(D, degenerate, tested, reject, tau=CF.MAT):
    if D is None:
        return None
    if abs(D) >= tau:
        return "moved"
    if degenerate:
        return "no comparison changed order"
    if not tested:
        return "kept, not tested (gate closed)"
    if reject:
        return "equivalence confirmed"
    return "kept, equivalence not established"


def h2_reading(F: Family, pp5a_verdict, tau=CF.MAT):
    if pp5a_verdict == "fails":
        return "premise failed"
    if F.confirmed["H2"] and F.branch == "H2a":
        return "follows the late baselines"
    if F.confirmed["H2"] and F.branch == "H2b":
        return "kept as predicted"
    if F.D2 is None:
        return None
    if F.D2 >= tau:
        r = "moved with the late baselines"
    elif abs(F.D2) < tau:
        r = "kept"
    else:
        r = "drop not confirmed"
    return r + ("" if F.open else " (not tested, gate closed)")


def section_K(v, fams, ol, PB, ax, vits, pres):
    for P in pres:
        extrap = [s for s in ax.sizes if PB.extrapolated(P, s)]
        for x in XS:
            F = fams[(P, x)]
            k = f"K|{P}|{x}"
            pp2a_v, pp2b_v = v[f"PP2a|{P}|{x}|verdict"], v[f"PP2b|{P}|{x}|verdict"]
            code, reason, prem = category(F.member, F.D1, F.p_perm, F.p_mag, pp2a_v, pp2b_v, F.R, F.C,
                                          conf=F.conf_raw if F.member else None)
            Afam = code in A_FAMILY
            v.update({f"{k}|category": code, f"{k}|premise": prem if Afam else None, f"{k}|F_reason": reason,
                      f"{k}|R": F.R, f"{k}|C": F.C, f"{k}|nV": F.nV})
            r = f"R|{P}|{x}"
            v[f"{k}|M1"], v[f"{k}|M1_sizes"] = v[f"{r}|fragile"], v[f"{r}|fragile_sizes"]
            g5 = v[f"{r}|grid5|D1"]
            v[f"{k}|M2"] = (not (g5 is not None and g5 <= -CF.MAT)) if Afam else None
            v[f"{k}|M2_x"] = g5
            h3r = immunity(F.tost[("mknn", "full")]["D"], F.tost[("mknn", "full")]["degenerate"], F.open,
                           F.holm["H3"]["reject"] if F.open else False)
            h4r = immunity(F.tost[("cka_debiased", "full")]["D"], F.tost[("cka_debiased", "full")]["degenerate"], F.open,
                           F.holm["H4"]["reject"] if F.open else False)
            gm, gd = v[f"{r}|grid5|dtr_mknn"], v[f"{r}|grid5|dtr_dcka"]
            v[f"{k}|M2_H3"] = (bool(gm is not None and abs(gm) >= CF.MAT) if h3r in KEPT_READINGS else None)
            v[f"{k}|M2_H4"] = (bool(gd is not None and abs(gd) >= CF.MAT) if h4r in KEPT_READINGS else None)
            v[f"{k}|M3"] = v[f"{r}|five|carried_D"] if code == "D" else v[f"{r}|five|carried"]
            med0 = [s for s in ax.sizes if v[f"R|{P}|median_scal|{s}"] == 0.0]
            nz = v[f"R|{P}|n_scal_zero"]
            # lead ruling R1 (2026-10-10 ~03:05Z): a key that the contract defines only for some variants is null in
            # every other variant, in every implementation. M4 is defined for x = s_cal only (contract 10), so all
            # three M4 keys (M4, M4_sizes, M4_n_zero) are null for x = unc. (Was [R-op] 2: sub-keys written for both x.)
            m4 = x == "s_cal"
            v[f"{k}|M4"] = bool(nz > (ax.n * len(vits)) / 3 or med0) if m4 else None
            v[f"{k}|M4_sizes"] = ",".join(med0) if m4 else None
            v[f"{k}|M4_n_zero"] = nz if m4 else None
            v[f"{k}|M5_extrap_sizes"] = ",".join(extrap)
            if code == "B":
                # [R-op] "every failing model of P-P2a(i)" = every model outside its band (PP2ai|P|x|failing)
                fail_i = [s for s in ax.sizes if not v[f"PP2ai|{P}|{x}|in_band|{s}"]]
                v[f"{k}|M5_B"] = bool(v[f"PP2aii|{P}|{x}|verdict"] != "fails" and all(s in extrap for s in fail_i)
                                      and v[f"PP2b|{P}|{x}|wrong_all_extrap"] is not False)
            else:
                v[f"{k}|M5_B"] = None
            held = [s for s in extrap if v[f"PP2ai|{P}|{x}|in_band|{s}"]]
            v[f"{k}|M5_A"] = bool(held) if Afam else None
            v[f"{k}|M5_A_sizes"] = ",".join(held) if Afam else None   # null outside A, A-att, A-prime (ruling R1)
            v[f"{k}|H3_reading"], v[f"{k}|H4_reading"] = h3r, h4r
            v[f"{k}|H2_reading"] = h2_reading(F, v[f"PP5a|{P}|{x}|verdict"])
            v[f"{k}|H2_same_as_H1"] = PB.same_as_H1(P)
            v[f"{k}|H2_counted"] = bool(F.confirmed["H2"] and not PB.same_as_H1(P))
            v[f"{k}|H3_counted"] = h3r == "equivalence confirmed"
            v[f"{k}|H4_counted"] = h4r == "equivalence confirmed"
            which = []
            if F.kedge:
                which.append("D1")
            if Afam and F.R is not None and abs(F.R - CF.LOSS_RAW) <= CF.KNIFE_SWAP + TOL:
                which.append("R")
            if Afam and F.C is not None and abs(F.C - CF.LOSS_CAL) <= CF.KNIFE_SWAP + TOL:
                which.append("C")
            v[f"{k}|kedge"] = bool(which) if F.kedge is not None else None
            v[f"{k}|kedge_which"] = ",".join(which)
            for tau in CF.TAUS:
                conf_t = (F.pH1 is not None and F.pH1 <= CF.ALPHA and F.D1 is not None and F.D1 <= -tau)
                v[f"{k}|curve|tau={fmt(tau)}"] = category(F.member, F.D1, F.p_perm, F.p_mag, pp2a_v, pp2b_v, F.R, F.C,
                                                          tau=tau, conf=conf_t)[0]
            for db in CF.DBS:
                mem = PB.material_db(P, db)
                v[f"{k}|curve|db={fmt(db)}"] = (None if mem is None else category(
                    mem, F.D1, F.p_perm, F.p_mag, pp2a_v, pp2b_v, F.R, F.C, conf=F.conf_raw)[0])
            # h_consistent: the H keys agree with their definitions (spec 10)
            ok = (F.pH1 == (None if F.p_perm is None or F.p_mag is None else max(F.p_perm, F.p_mag)))
            ok = ok and (F.H1_confirmed is None) == (not F.member)
            ok = ok and F.gate == ("open (H1 confirmed)" if (F.member and F.conf_raw) else
                                   ("open (H1 not a member)" if not F.member else "closed"))
            if F.open:
                pv = {h: (1.0 if F.p[h] is None else F.p[h]) for h in CF.HYPS}
                srt = sorted(CF.HYPS, key=lambda h: (pv[h], CF.HYPS.index(h)))
                go = True
                for i, h in enumerate(srt):
                    rej = go and pv[h] <= CF.ALPHA / (3 - i)
                    go = go and rej
                    ok = ok and rej == F.holm[h]["reject"]
            v[f"{k}|h_consistent"] = bool(ok)


# ======================================================================================================================
# section G (spec 9) and R's own checks (x| extras)
# ======================================================================================================================

def section_G(v, ol, PB, ax, vits, pres_run, n_known, partb_sha, clip_ok, notes):
    v["G|n_pairs_read"] = int(sum(len(ol[P][s]) for P in pres_run for s in ax.sizes))
    v["G|n_known_records"] = int(n_known)
    v["G|l2_run"] = "l2" in pres_run
    v["G|partB_sha256"] = partb_sha
    v["G|clip_equal_partB"] = clip_ok
    diffs = []
    for P in pres_run:
        rc = PB.recheck(P)
        diffs += rc["diffs"]
        v[f"x|partB_recheck|{P}"] = repr(rc["recomputed"])
    v["x|partB_recheck_diffs"] = "; ".join(diffs)[:8000]
    v["x|partB_recheck_ok"] = not diffs
    bad = []
    for s in ax.sizes:
        L = ol[pres_run[0]][s][vits[0]]["L_A"]
        if ax.name == "pythia":
            if L != ax.L_stored[s]:
                bad.append(f"{s}: L_A {L} != plan {ax.L_stored[s]}")
            for r in ("0.5", "1.0"):
                if RL.fixed_layer(L, float(r)) != CF.FIXED_DEPTH[r][s]:
                    bad.append(f"{s}: fixed r{r} != plan")
            if RL.grid_layers(L, 5) != CF.GRID5_LM[s]:
                bad.append(f"{s}: grid5 {RL.grid_layers(L, 5)} != plan {CF.GRID5_LM[s]}")
    v["x|layers_match_plan"] = not bad
    v["x|layers_issues"] = "; ".join(bad)
    v["x|grid_saturated"] = ",".join(f"{s}:grid{g}" for s in ax.sizes for g in (9, 17)
                                     if len(RL.grid_layers(ol[pres_run[0]][s][vits[0]]["L_A"], g))
                                     == ol[pres_run[0]][s][vits[0]]["L_A"])
    v["x|identity_issues"] = "; ".join(notes.get("identity_issues", []))[:4000]
