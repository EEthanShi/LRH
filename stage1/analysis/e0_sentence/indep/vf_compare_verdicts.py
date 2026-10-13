"""V-f step 5 (role e0-indep): the own P1, P2b, P2c statistics (vf_stats.json,
computed by vf_rules.py from E0's obs files) against E0's VERDICTS_E0.json and
eval tables. Rule (plan 10, V-f): every statistic within 1e-12, labels identical.
Also the secondary P1 on arms D1..D4, A and secondary P2b on arm A (same code).

Usage: python vf_compare_verdicts.py <outdir> <vf_stats.json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C  # noqa: E402
import vf_rules as R  # noqa: E402
import vf_stats as S  # noqa: E402

TOL = 1e-12
VERD = os.path.join(C.ROOT, "results", "E0_sentence", "VERDICTS_E0.json")
EVAL = os.path.join(C.ROOT, "results", "E0_sentence", "eval")


class Cmp:
    def __init__(self):
        self.num = []
        self.lab = []
        self.gaps = []

    def n(self, path, mine, e0):
        if e0 is None:
            self.gaps.append(dict(path=path, mine=mine, e0=None))
            return
        d = abs(float(mine) - float(e0))
        self.num.append(dict(path=path, mine=float(mine), e0=float(e0), absdiff=d))

    def l(self, path, mine, e0):
        self.lab.append(dict(path=path, mine=mine, e0=e0, equal=bool(mine == e0)))


def p1_block(cmp, tag, mine, e0):
    cmp.l(f"{tag}/label", mine["label"], e0["label"])
    cmp.l(f"{tag}/n_informative", mine["n_informative"], e0["n_informative"])
    cmp.n(f"{tag}/median_sum", mine["median_sum"], e0["median_sum"])
    cmp.n(f"{tag}/frac_abs_sum_le", mine["frac_le_005"], e0["frac_abs_sum_le"])
    inf_mine = sorted(n for n in S.NEW_NAMES if abs(mine["e_m"][n]) >= 0.10)
    cmp.l(f"{tag}/informative_pairs", inf_mine, sorted(e0["informative_pairs"]))
    for r in e0["per_pair"]:
        for f in ("e_k", "e_m", "sum"):
            cmp.n(f"{tag}/per_pair/{r['pair']}/{f}", mine[{"e_k": "e_k", "e_m": "e_m", "sum": "sum"}[f]][r["pair"]], r[f])
    for e, v in e0["per_encoder_median_sum"].items():
        cmp.n(f"{tag}/per_encoder_median_sum/{e}", mine["per_encoder"][e]["median_sum_informative"], v)


def p1_for_rows(rows):
    cells = R.cells_from_obs(rows)
    ek, em, sm = {}, {}, {}
    for nm in S.NEW_NAMES:
        ek[nm], em[nm], sm[nm] = R.elasticities(cells, nm, [10, 100])
    out = R.p1_label([em[n] for n in S.NEW_NAMES], [sm[n] for n in S.NEW_NAMES])
    out.update(e_k=ek, e_m=em, sum=sm)
    per = {}
    for e in C.ENCS:
        sub = [f"{v}__{e}" for v in sorted(C.VITS) if abs(em[f"{v}__{e}"]) >= 0.10]
        per[e] = dict(median_sum_informative=float(np.median([sm[n] for n in sub])) if sub else None)
    out["per_encoder"] = per
    return out


def p2b_block(cmp, tag, mine, e0, lnL_key=True):
    cmp.l(f"{tag}/label", mine["label"], e0["label"])
    cmp.n(f"{tag}/guard_threshold", R.GUARD_P2B, e0["guard_threshold"])
    cmp.n(f"{tag}/median_abs_gap", mine["median_abs_g"], e0["median_abs_gap"])
    inf_mine = sorted(e for e, v in mine["encoder_informative"].items() if v["informative"])
    e0_inf = e0["informative_encoders"]
    e0_inf = sorted(e0_inf if isinstance(e0_inf, list) else e0_inf.keys())
    cmp.l(f"{tag}/informative_encoders", inf_mine, e0_inf)
    cmp.n(f"{tag}/lambda_hat", mine["lambda_hat"], e0.get("lam"))
    for e in C.ENCS:
        v = e0.get("lam_e", {}).get(e)
        cmp.n(f"{tag}/lambda_e/{e}", mine["lambda_e"][e], None if v is None else v["lam"])
        if v is not None:
            cmp.n(f"{tag}/lambda_e/{e}/median_abs_gap", mine["encoder_informative"][e]["median_abs_g"], v["median_abs_gap"])
            cmp.l(f"{tag}/lambda_e/{e}/informative", mine["encoder_informative"][e]["informative"], v["informative"])
    for r in e0["per_pair"]:
        nm = r["pair"]
        if "L" in r and "L" in mine:
            cmp.n(f"{tag}/per_pair/{nm}/L", mine["L"][nm], r["L"])
        if "b" in mine:
            cmp.n(f"{tag}/per_pair/{nm}/b", mine["b"][nm], r["b"])
        cmp.n(f"{tag}/per_pair/{nm}/bhat_star", mine["beta_star"][nm], r["bhat_star"])
        cmp.n(f"{tag}/per_pair/{nm}/beta_M_v", R.BETA_MV[r["vit"]], r["beta_M_v"])
        cmp.n(f"{tag}/per_pair/{nm}/g", mine["g"][nm], r["g"])
        cmp.n(f"{tag}/per_pair/{nm}/y", mine["y"][nm], r["y"])
        cmp.n(f"{tag}/per_pair/{nm}/resid_vs_line", mine["residual"][nm], r["resid_vs_line"])
    lo = e0.get("leave_one_out", {})
    for e, v in lo.get("leave_encoder_out", {}).items():
        cmp.n(f"{tag}/leave_encoder_out/{e}", mine["leave_one_encoder_out"][e], v)
    for vv, v in lo.get("leave_vit_out", {}).items():
        cmp.n(f"{tag}/leave_vit_out/{vv}", mine["leave_one_vit_out"][vv], v)


def main(outdir, stats_path):
    st = json.load(open(stats_path))["from_e0_obs"]
    v = json.load(open(VERD))
    cmp = Cmp()
    # ------------- labels -------------
    for key in ("P1", "P2b", "P2c"):
        cmp.l(f"labels/{key}", st[key]["label"], v["labels"][key])
    # ------------- P1 (arm P) -------------
    p1_block(cmp, "P1", st["P1"], v["P1"])
    el = pd.read_parquet(os.path.join(EVAL, "E0_elasticity_P.parquet"))
    for r in el.itertuples(index=False):
        if r.k_set == "k10_100":
            mine_s, mine_ek, mine_em = st["P1"]["sum"][r.pair], st["P1"]["e_k"][r.pair], st["P1"]["e_m"][r.pair]
            cmp.n(f"eval/elasticity_P/{r.pair}/k10_100/e_k", mine_ek, r.e_k)
            cmp.n(f"eval/elasticity_P/{r.pair}/k10_100/e_m", mine_em, r.e_m)
        elif r.k_set == "k1_10_100":
            mine_s = st["P1"]["sum_allk"][r.pair]
        elif r.k_set == "k1_10":
            mine_s = st["P1"]["sum_k1_10"][r.pair]
        else:
            raise ValueError(r.k_set)
        cmp.n(f"eval/elasticity_P/{r.pair}/{r.k_set}/sum", mine_s, r.sum)
    # secondary P1 on arms D1..D4, A (same rule; from E0 obs)
    e0_rows, _ = S.load_e0_arm_rows()
    for arm, vk in (("D1", "D1"), ("D2", "D2"), ("D3", "D3"), ("D4", "D4"), ("D5", "A")):
        p1_block(cmp, f"secondary/P1/{vk}", p1_for_rows(e0_rows[arm]), v["secondary"]["P1"][vk])
    # ------------- P2b (arm P) -------------
    p2b_block(cmp, "P2b", st["P2b"], v["P2b"])
    cmp.n("P2b/leave_mpnet_out_lambda", st["P2b"]["leave_one_encoder_out"]["mpnet"], v["P2b"]["leave_mpnet_out_lambda"])
    for wk, mk, dk in (("vit_specific", "wilcoxon_vit", "diff_vit"), ("constant", "wilcoxon_const", "diff_const")):
        w = v["P2b"]["wilcoxon_descriptive"][wk]
        cmp.n(f"P2b/wilcoxon/{wk}/statistic", st["P2b"][mk]["W_plus"], w["statistic"])
        cmp.n(f"P2b/wilcoxon/{wk}/p_one_sided", st["P2b"][mk]["p"], w["p_one_sided"])
        b = st["P2b"]["b"]
        comp = {nm: (R.BETA_MV[nm.split("__")[0]] if wk == "vit_specific" else R.BETA_M_CONST) for nm in S.NEW_NAMES}
        med_eM = float(np.median([abs(b[nm] - comp[nm]) for nm in S.NEW_NAMES]))
        med_eL = float(np.median([abs(b[nm] - st["P2b"]["beta_star"][nm]) for nm in S.NEW_NAMES]))
        cmp.n(f"P2b/wilcoxon/{wk}/median_e_M (own reading: median |b - competitor|)", med_eM, w["median_e_M"])
        cmp.n(f"P2b/wilcoxon/{wk}/median_e_L (own reading: median |b - beta*|)", med_eL, w["median_e_L"])
    lv = pd.read_parquet(os.path.join(EVAL, "E0_levels_P.parquet"))
    for r in lv[lv["k"] == 10].itertuples(index=False):
        cmp.n(f"eval/levels_P/{r.pair}/k10/L_m1024", st["P2b"]["L"][r.pair], r.L_m1024)
    sl = pd.read_parquet(os.path.join(EVAL, "E0_slopes_P.parquet"))
    for r in sl[sl["k"] == 10].itertuples(index=False):
        cmp.n(f"eval/slopes_P/{r.pair}/k10/slope_obs", st["P2b"]["b"][r.pair], r.slope_obs)
    ds = pd.read_parquet(os.path.join(EVAL, "E0_drawslopes_P.parquet"))
    for r in ds[ds["k"] == 10].itertuples(index=False):
        cmp.n(f"eval/drawslopes_P/{r.pair}/k10/d{r.draw}", st["P2b"]["b_draws"][r.pair][int(r.draw)], r.slope_obs)
    # secondary P2b on arm A with the arm-P line
    rowsA = e0_rows["D5"]
    cellsA = R.cells_from_obs(rowsA)
    lnLA = {nm: R.level(cellsA, nm, 10, 1024)[1] for nm in S.NEW_NAMES}
    LA = {nm: R.level(cellsA, nm, 10, 1024)[0] for nm in S.NEW_NAMES}
    bA = {nm: R.mean_slope(rowsA, nm, 10)[0] for nm in S.NEW_NAMES}
    mA = R.p2b(S.NEW_PAIRS, lnLA, bA)
    mA.update(L=LA, b=bA)
    p2b_block(cmp, "secondary/P2b/A", mA, v["secondary"]["P2b"]["A"])
    # ------------- P2c -------------
    c, e = st["P2c"], v["P2c"]
    cmp.l("P2c/label", c["label"], e["label"])
    cmp.n("P2c/median_gamma", c["median_gamma"], e["median_gamma"])
    cmp.n("P2c/median_dln", c["median_delta"], e["median_dln"])
    inf_mine = sorted(k for k, x in c["per_encoder"].items() if x["informative"])
    cmp.l("P2c/informative_encoders", inf_mine, sorted(e["informative_encoders"]))
    for enc, x in e["gamma_e"].items():
        cmp.n(f"P2c/gamma_e/{enc}/median_gamma", c["per_encoder"][enc]["median_gamma"], x["median_gamma"])
        cmp.n(f"P2c/gamma_e/{enc}/median_dln", c["per_encoder"][enc]["median_delta"], x["median_dln"])
        cmp.l(f"P2c/gamma_e/{enc}/informative", c["per_encoder"][enc]["informative"], x["informative"])
    for r in e["per_pair"]:
        cmp.n(f"P2c/per_pair/{r['pair']}/gamma", c["gamma"][r["pair"]], r["gamma"])
        cmp.n(f"P2c/per_pair/{r['pair']}/dln", c["delta"][r["pair"]], r["dln"])
    for enc, x in e["leave_encoder_out_median_gamma"].items():
        cmp.n(f"P2c/leave_encoder_out_median_gamma/{enc}", c["leave_one_encoder_out"][enc]["median_gamma"], x)
    cmp.n("P2c/leave_mpnet_out_median_gamma", c["leave_one_encoder_out"]["mpnet"]["median_gamma"], e["leave_mpnet_out_median_gamma"])
    for nm, x in e["ladder"].items():
        for i in range(5):
            cmp.n(f"P2c/ladder/{nm}/L/n{i + 1}", c["L_ladder"][nm][i], x["L"][i])
            cmp.n(f"P2c/ladder/{nm}/b/n{i + 1}", c["b_ladder"][nm][i], x["b"][i])
    # P2c at k = 1 and k = 100 (descriptive; same rule applied for the label E0 records)
    for kk in ("k1", "k100"):
        x = v["secondary"]["P2c_other_k"][kk]
        # recompute the full rule at that k from E0 obs
        k = int(kk[1:])
        lnLn = {nm: [] for nm in S.NEW_NAMES}
        bn = {nm: [] for nm in S.NEW_NAMES}
        for n in range(1, 6):
            rows = e0_rows[f"D{n}"]
            cells = R.cells_from_obs(rows)
            for nm in S.NEW_NAMES:
                lnLn[nm].append(R.level(cells, nm, k, 1024)[1])
                bn[nm].append(R.mean_slope(rows, nm, k)[0])
        mk = R.p2c(S.NEW_PAIRS, lnLn, bn)
        cmp.n(f"secondary/P2c_other_k/{kk}/median_gamma", mk["median_gamma"], x["median_gamma"])
        cmp.n(f"secondary/P2c_other_k/{kk}/median_gamma (vf_stats.json)", st[f"P2c_gamma_{kk}"]["median_gamma"], x["median_gamma"])
        for enc, y in x["gamma_e"].items():
            cmp.n(f"secondary/P2c_other_k/{kk}/gamma_e/{enc}/median_gamma", mk["per_encoder"][enc]["median_gamma"], y["median_gamma"])
            cmp.n(f"secondary/P2c_other_k/{kk}/gamma_e/{enc}/median_dln", mk["per_encoder"][enc]["median_delta"], y["median_dln"])
            cmp.l(f"secondary/P2c_other_k/{kk}/gamma_e/{enc}/informative", mk["per_encoder"][enc]["informative"], y["informative"])
        if "label_if_rule_applied" in x:
            cmp.l(f"secondary/P2c_other_k/{kk}/label_if_rule_applied", mk["label"], x["label_if_rule_applied"])
    # ------------- summary -------------
    maxdiff = max(r["absdiff"] for r in cmp.num)
    worst = max(cmp.num, key=lambda r: r["absdiff"])
    n_bad = sum(r["absdiff"] > TOL for r in cmp.num)
    lab_bad = [r for r in cmp.lab if not r["equal"]]
    res = dict(verdicts=dict(path=VERD, sha256=C.sha256_file(VERD)),
               eval_files={f: C.sha256_file(os.path.join(EVAL, f)) for f in
                           ("E0_elasticity_P.parquet", "E0_levels_P.parquet", "E0_slopes_P.parquet", "E0_drawslopes_P.parquet")},
               vf_stats=dict(path=stats_path, sha256=C.sha256_file(stats_path)),
               tolerance=TOL, n_numeric=len(cmp.num), max_abs_diff=maxdiff, worst=worst,
               n_numeric_over_tol=n_bad, n_labels=len(cmp.lab), labels_differing=lab_bad,
               not_reported_by_e0=cmp.gaps, numeric=cmp.num, labels=cmp.lab)
    sha = C.write_new_json(os.path.join(outdir, "vf_compare_verdicts.json"), res)
    print(json.dumps(dict(n_numeric=len(cmp.num), max_abs_diff=maxdiff, worst=worst["path"], n_over_tol=n_bad,
                          n_labels=len(cmp.lab), labels_differing=lab_bad, gaps=cmp.gaps), indent=1))
    print("vf_compare_verdicts.json", sha)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
