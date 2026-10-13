"""F1 data layer: every number plotted in F1 is computed here from the frozen S1.2 layer-max artifacts (read-only).

Inputs (all under ~/hrh_v2_stage1_20261008/results/, opened read-only):
  S1_2_layermax/panel_layermax.parquet           per-pair scores, thresholds, argmax (main run, K = 500)
  S1_2_layermax/pairs/<pre>/<lm>__<vit>.npz      entry-wise permutation null means (cka_biased.entry_null_mean)
  S1_2_layermax/indep/diag_null_profile.json     participation ratio PR per stored layer, all 25 models
  S1_2_layermax/posthoc_null_structure.json      null level per LM layer (EXPLORATORY, post hoc)
  S1_2_layermax_llama/partB/partB_retro_checks.json  R2: per-LM null-level prediction check (retrodiction)
Nothing in the LLaMA-axis run outputs (pairs/early/indep of S1_2_layermax_llama) is opened here.
"""
import glob, hashlib, itertools, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

S1 = os.path.expanduser("~/hrh_v2_stage1_20261008/results")
LM_DIR = f"{S1}/S1_2_layermax"
M = 1024                       # WIT-1024 evaluation size
PRE = "prh_clip+l2"            # primary preprocessing
BLOOMZ = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7",
          "bigscience__bloomz-3b", "bigscience__bloomz-7b1"]
BOOT_B, BOOT_SEED = 4000, 20261009


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rho_rows(x, Y):
    """Spearman rho of the fixed vector x against every row of Y (no ties in x)."""
    rx = rankdata(x); rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    RY = np.apply_along_axis(rankdata, 1, Y)
    RY = RY - RY.mean(1, keepdims=True); RY /= np.linalg.norm(RY, axis=1, keepdims=True)
    return RY @ rx


def trend_stats(sub, col, rng):
    """Tr = mean over ViTs of Spearman(lm_params, score) within BLOOMZ; exact relabelling null; ViT bootstrap."""
    wide = sub.pivot(index="vit", columns="lm", values=col)[BLOOMZ]
    size = sub.drop_duplicates("lm").set_index("lm").loc[BLOOMZ, "lm_params"].to_numpy(float)
    Y = wide.to_numpy(float)                                         # 17 ViTs x 5 sizes
    rho = _rho_rows(size, Y)
    tr = float(rho.mean())
    # check against scipy on the observed labelling
    tr_sp = float(np.mean([spearmanr(size, y).statistic for y in Y]))
    assert abs(tr - tr_sp) < 1e-12, (tr, tr_sp)
    # exact null: the same relabelling of the 5 sizes for every ViT (120 permutations, identity included)
    null = np.array([_rho_rows(size[list(p)], Y).mean() for p in itertools.permutations(range(5))])
    p_one = float(np.mean(null >= tr - 1e-12))
    srt = np.sort(null)
    crit_hi = float(srt[-6])   # p = #(null >= v)/120 <= 0.05  <=>  v >= 6th largest
    crit_lo = float(srt[5])    # mirror for the lower tail
    # ViT bootstrap (resample the 17 ViTs; fixed seed)
    idx = rng.integers(0, len(rho), size=(BOOT_B, len(rho)))
    boot = rho[idx].mean(1)
    ci = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    return dict(Tr=tr, p_one_sided_exact=p_one, crit_lo_5pct=crit_lo, crit_hi_5pct=crit_hi,
                null_min=float(srt[0]), null_max=float(srt[-1]), vit_boot_ci95=ci, n_vit=int(len(rho)),
                per_vit_rho=dict(zip(wide.index, map(float, rho))))


def compute():
    out = {"inputs": {}, "notes": []}
    p_panel = f"{LM_DIR}/panel_layermax.parquet"
    p_diag = f"{LM_DIR}/indep/diag_null_profile.json"
    p_post = f"{LM_DIR}/posthoc_null_structure.json"
    p_r2 = f"{S1}/S1_2_layermax_llama/partB/partB_retro_checks.json"
    p_sum = f"{LM_DIR}/layermax_summary.json"
    for k, p in dict(panel_layermax=p_panel, diag_null_profile=p_diag, posthoc_null_structure=p_post,
                     partB_retro_checks=p_r2, layermax_summary=p_sum).items():
        out["inputs"][k] = {"path": p.replace(os.path.expanduser("~"), "~"), "sha256": sha(p)}

    df = pd.read_parquet(p_panel)
    b = df[(df.pre == PRE) & df.lm.isin(BLOOMZ)].copy()
    assert b.vit.nunique() == 17 and b.lm.nunique() == 5

    # ---- panel (a): within-BLOOMZ trend Tr, raw vs calibrated -------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    arms = [("cka_biased", "full"), ("cka_biased", "fixed_r1.0"), ("cka_biased", "fixed_r0.5"),
            ("mknn", "full"), ("cka_debiased", "full")]
    A = {}
    for met, sub in arms:
        x = b[(b.metric == met) & (b.subset == sub)]
        assert len(x) == 85, (met, sub, len(x))
        A[f"{met}|{sub}"] = {col: trend_stats(x, col, rng) for col in ["s_obs", "uncensored", "s_cal", "c_alpha"]}
    # frozen main-run summary: Tr, exact p and ViT-bootstrap CI as reported (plotted CIs come from here)
    SUM = json.load(open(p_sum))["per_pre"][PRE]["trends"]
    for met, sub in arms:
        for col in ["s_obs", "uncensored", "s_cal"]:
            ref = SUM[met][f"{sub}|{col}"]["tr_lm"]
            mine = A[f"{met}|{sub}"][col]
            assert abs(mine["Tr"] - ref["Tr"]) < 1e-9, (met, sub, col, mine["Tr"], ref["Tr"])
            assert abs(mine["p_one_sided_exact"] - ref["p_one"]) < 1e-9, (met, sub, col)
            mine["vit_boot_ci95_recomputed"] = mine.pop("vit_boot_ci95")
            mine["vit_boot_ci95"] = list(map(float, ref["ci"]))          # frozen summary CI
            mine["source_Tr_p_ci"] = f"layermax_summary.json per_pre/{PRE}/trends/{met}/{sub}|{col}/tr_lm"
        A[f"{met}|{sub}"]["c_alpha"]["vit_boot_ci95_recomputed"] = A[f"{met}|{sub}"]["c_alpha"].pop("vit_boot_ci95")
    out["a_trend"] = A
    out["a_crit_raw_vs_cal_max_abs_diff"] = float(max(abs(A[k]["s_obs"]["crit_hi_5pct"] - A[k]["uncensored"]["crit_hi_5pct"])
                                                      for k in A))
    out["arh_style_pooled_r_136_pairs"] = {met: {sc: SUM[met][f"full|{sc}"]["arh_r"] for sc in ["s_obs", "s_cal", "uncensored"]}
                                           for met in ["cka_biased", "mknn", "cka_debiased"]}
    out["arh_table2_reference"] = json.load(open(p_sum))["arh_table2_reference"]
    out["a_trend_meta"] = dict(pre=PRE, lm_axis=BLOOMZ, n_sizes=5, n_perm=120, boot_B=BOOT_B, boot_seed=BOOT_SEED,
                               definition="Tr = mean over 17 ViTs of Spearman(lm_params, score) across the 5 BLOOMZ sizes",
                               raw="s_obs (uncalibrated layer-pair or layer-max score T)",
                               calibrated_primary="uncensored = T - mean of the K=500 permutation scores",
                               calibrated_arh="s_cal = max((T - c_alpha)/(1 - c_alpha), 0)")

    # ---- panel (b): per BLOOMZ size, layer-max CKA and its thresholds (mean +- SE over 17 ViTs) ----------
    PRd = json.load(open(p_diag))[PRE]["PR_profile"]
    x = b[(b.metric == "cka_biased") & (b.subset == "full")]
    B = {}
    for lm in BLOOMZ:
        g = x[x.lm == lm]
        pred_max = []
        for vit in g.vit:
            pa, pb = np.asarray(PRd[lm]), np.asarray(PRd[vit])
            pred_max.append(np.sqrt(pa.max() * pb.max()) / (M - 1))   # largest predicted baseline over all layer pairs
        row = {"lm_params": int(g.lm_params.iloc[0]), "n_vit": int(len(g))}
        for name, v in [("T", g.s_obs), ("c_alpha", g.c_alpha), ("c_alpha_entry_at_argmax", g.entry_c_alpha_at_argmax),
                        ("null_mean_of_max", g.null_mean), ("uncensored", g.uncensored), ("s_cal", g.s_cal),
                        ("pred_largest_baseline", pd.Series(pred_max)), ("rel_depth_argmax_LM", g.rel_A)]:
            v = np.asarray(v, float)
            row[name] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                         "min": float(v.min()), "max": float(v.max())}
        B[lm] = row
    out["b_thresholds"] = B
    # how closely c_alpha follows the largest predicted baseline, pair by pair
    gaps = []
    for lm in BLOOMZ:
        g = x[x.lm == lm]
        for vit, ca in zip(g.vit, g.c_alpha):
            gaps.append(ca - np.sqrt(np.max(PRd[lm]) * np.max(PRd[vit])) / (M - 1))
    gaps = np.asarray(gaps)
    out["b_gap_calpha_minus_pred_largest_baseline_bloomz"] = {"n": int(gaps.size), "mean": float(gaps.mean()),
                                                              "min": float(gaps.min()), "max": float(gaps.max())}

    # ---- panel (c): CKA permutation null per LM layer vs sqrt(PR_A PR_B)/(m-1) -------------------------
    post = json.load(open(p_post))
    assert "EXPLORATORY" in post["note"]
    prof = post["per_pre"][PRE]["profile"]["cka_biased"]
    C = {}
    vits = sorted(b.vit.unique())
    for lm in BLOOMZ:
        meas_post = np.asarray(prof[lm]["null_level_by_layer"], float)
        L = len(PRd[lm])
        meas_npz = np.zeros(L); pred = np.zeros(L)
        for vit in vits:
            z = np.load(f"{LM_DIR}/pairs/{PRE}/{lm}__{vit}.npz")
            mu = z["cka_biased.entry_null_mean"]                    # L_A x L_B
            assert mu.shape == (L, len(PRd[vit]))
            meas_npz += mu.max(1)
            pred += np.sqrt(np.asarray(PRd[lm])[:, None] * np.asarray(PRd[vit])[None, :]).max(1) / (M - 1)
        meas_npz /= len(vits); pred /= len(vits)
        assert np.max(np.abs(meas_npz - meas_post)) < 1e-12, "posthoc profile != recomputed profile"
        rel = meas_npz / pred - 1
        C[lm] = {"L": L, "rel_depth": list(np.arange(L) / (L - 1)), "measured": list(map(float, meas_npz)),
                 "predicted": list(map(float, pred)), "max_abs_rel_err": float(np.abs(rel).max()),
                 "argmax_measured": int(meas_npz.argmax()), "argmax_predicted": int(pred.argmax()),
                 "PR_LM_layer0": float(PRd[lm][0]), "PR_LM_max": float(np.max(PRd[lm])),
                 "PR_LM_argmax": int(np.argmax(PRd[lm]))}
    out["c_null_profile"] = C
    out["c_observed_argmax_rel_depth_all_bloomz_pairs"] = list(map(float, x.rel_A))   # layer-max biased CKA, 85 pairs
    out["c_null_profile_meta"] = dict(
        measured="mean over 17 ViTs of max over ViT layers of the entry-wise permutation null mean (K=500); "
                 "equals posthoc_null_structure.json profile (EXPLORATORY file) to 1e-12",
        predicted="mean over 17 ViTs of max over ViT layers of sqrt(PR_A,u PR_B,v)/(m-1), Prop. 5 part 2",
        mknn_baseline=10 / (M - 1))

    # ---- all-entry check of Prop. 5 part 2 (both pre) -------------------------------------------------
    E = {}
    for pre in ["prh_clip+l2", "l2"]:
        PRp = json.load(open(p_diag))[pre]["PR_profile"]
        rel, zs = [], []
        for f in sorted(glob.glob(f"{LM_DIR}/pairs/{pre}/*.npz")):
            stem = os.path.basename(f)[:-4]; parts = stem.split("__")
            lm, vit = parts[0] + "__" + parts[1], parts[2]
            z = np.load(f)
            mu, sd = z["cka_biased.entry_null_mean"], z["cka_biased.entry_null_sd"]
            pred = np.sqrt(np.outer(PRp[lm], PRp[vit])) / (M - 1)
            rel.append(np.abs(mu / pred - 1).ravel())
            zs.append(((mu - pred) / (sd / np.sqrt(500))).ravel())      # Monte Carlo z of the K=500 null mean
        r = np.concatenate(rel); zz = np.concatenate(zs)
        E[pre] = {"n_pairs": len(rel), "n_entries": int(r.size), "median_abs_rel_err": float(np.median(r)),
                  "q95_abs_rel_err": float(np.quantile(r, .95)), "max_abs_rel_err": float(r.max()),
                  "mc_z_median_abs": float(np.median(np.abs(zz))), "mc_z_frac_abs_gt_3": float(np.mean(np.abs(zz) > 3)),
                  "mc_z_frac_abs_gt_4": float(np.mean(np.abs(zz) > 4)), "mc_z_max_abs": float(np.abs(zz).max()),
                  "mc_z_sd": float(zz.std())}
    out["entrywise_prop5_check"] = E

    # ---- R2 retrodiction (per-LM null level, prediction vs posthoc profile) ---------------------------
    R2 = json.load(open(p_r2))["per_pre"][PRE]["R2_null_level_profile"]
    out["R2_per_lm_null_level"] = {lm: {k: R2[lm][k] for k in ["pred_mean_v_Bbar", "posthoc_profile_max", "rel_err",
                                                               "pred_u_star", "posthoc_argmax_layer", "u_star_equal"]}
                                   for lm in R2}
    out["notes"].append("PR mechanism, null profile per layer and leave-layers-out readings are EXPLORATORY/post hoc "
                        "(posthoc_null_structure.json note); Prop. 5 itself is a theorem, the entrywise check is a sanity check.")
    out["notes"].append("LLaMA axis (llama-7b/13b/30b): NOT read; placeholder only.")
    return out


if __name__ == "__main__":
    o = compute()
    print(json.dumps({k: v for k, v in o.items() if k not in ("c_null_profile",)}, indent=1, default=str)[:12000])
