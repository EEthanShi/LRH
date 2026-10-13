"""F1 data layer, LLaMA-axis rows (d)-(f) and the calibrate-then-max row of panels (a) and (d).

Every number is read from frozen artifacts or recomputed from them and asserted against the frozen aggregate.
Inputs (all under ~/hrh_v2_stage1_20261008/results/, opened read-only):
  S1_2_layermax_llama/early/pairs/prh_clip+l2/<llama>__<vit>.{json,npz}  implementation E, per pair (K = 500)
  S1_2_layermax_llama/early/summary.json            E aggregate: Tr_L, exact p, ViT-bootstrap CI, means (frozen)
  S1_2_layermax_llama/indep/summary.json            implementation I aggregate (cross-check; BLOOMZ ctm)
  S1_2_layermax_llama/review/recomputed_verdicts.json   review recomputation of the Part B verdicts (cross-check)
  S1_2_layermax_llama/partB/partB_features.json     Part B, frozen before any cross-model score: PR, B-bar, num_params
  S1_2_layermax/pairs/prh_clip+l2/<bloomz>__<vit>.npz   BLOOMZ entry-wise scores and thresholds (calibrate-then-max)
  S1_2_layermax/indep/diag_null_profile.json        PR profiles used for BLOOMZ panel (c) (consistency check only)
Status: LLaMA rows test frozen Part B predictions P-L1..P-L5; the BLOOMZ calibrate-then-max row is EXPLORATORY
(protocol defined in the LLaMA plan after the BLOOMZ result was known).
"""
import hashlib, itertools, json, os
import numpy as np

S1 = os.path.expanduser("~/hrh_v2_stage1_20261008/results")
LL = f"{S1}/S1_2_layermax_llama"
PRE = "prh_clip+l2"
M = 1024
LLAMA = ["huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"]
LL_LABEL = {"huggyllama__llama-7b": "7B", "huggyllama__llama-13b": "13B", "huggyllama__llama-30b": "30B"}
BLOOMZ = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7",
          "bigscience__bloomz-3b", "bigscience__bloomz-7b1"]
BOOT_B, BOOT_SEED = 4000, 20261009
# panel rows: (key, metric, subset, raw column, calibrated column (filled), ARH s_cal column (tick))
ROWS = [("cka_biased|full", "cka_biased", "full"),
        ("cka_biased|ctm", "cka_biased", "ctm"),
        ("cka_biased|fixed_r1.0", "cka_biased", "fixed_r1.0"),
        ("cka_biased|fixed_r0.5", "cka_biased", "fixed_r0.5"),
        ("mknn|full", "mknn", "full"),
        ("cka_debiased|full", "cka_debiased", "full")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rank(v):
    from scipy.stats import rankdata
    return rankdata(v)


def _rho_rows(x, Y):
    rx = _rank(x); rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    RY = np.apply_along_axis(_rank, 1, Y)
    RY = RY - RY.mean(1, keepdims=True)
    nrm = np.linalg.norm(RY, axis=1, keepdims=True)
    assert np.all(nrm > 0), "constant row"
    return (RY / nrm) @ rx


def trend(Y, size, crit=True):
    """Tr = mean over ViTs of Spearman(size, score); exact null over all relabellings of the sizes (same for all ViTs)."""
    rho = _rho_rows(size, Y)
    tr = float(rho.mean())
    n = len(size)
    null = np.array([_rho_rows(size[list(p)], Y).mean() for p in itertools.permutations(range(n))])
    srt = np.sort(null)
    out = dict(Tr=tr, p_one_sided_exact=float(np.mean(null >= tr - 1e-12)), n_perm=int(len(null)),
               null_min=float(srt[0]), null_max=float(srt[-1]), rho=rho)
    k = int(np.floor(0.05 * len(null)))          # p = #(null >= v)/n <= 0.05  <=>  v >= k-th largest null value
    if k >= 1:                                    # 120 relabellings: k = 6, as in f1_data.trend_stats
        out.update(crit_lo_5pct=float(srt[k - 1]), crit_hi_5pct=float(srt[-k]), band="5pct_critical")
    else:                                         # 6 relabellings: no value reaches p <= 0.05; band = whole null range
        out.update(crit_lo_5pct=float(srt[0]), crit_hi_5pct=float(srt[-1]), band="full_null_range_no_5pct_value")
    return out


def boot_ci(rho, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rho), size=(BOOT_B, len(rho)))
    b = rho[idx].mean(1)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def mse(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "min": float(v.min()),
            "max": float(v.max())}


def compute(vits):
    out = {"inputs": {}, "notes": []}
    p_E, p_I = f"{LL}/early/summary.json", f"{LL}/indep/summary.json"
    p_rev, p_pb = f"{LL}/review/recomputed_verdicts.json", f"{LL}/partB/partB_features.json"
    p_diag = f"{S1}/S1_2_layermax/indep/diag_null_profile.json"
    for k, p in dict(E_summary=p_E, I_summary=p_I, review_verdicts=p_rev, partB_features=p_pb,
                     diag_null_profile=p_diag).items():
        out["inputs"][k] = {"path": p.replace(os.path.expanduser("~"), "~"), "sha256": sha(p)}
    E = json.load(open(p_E))["per_pre"][PRE]
    I = json.load(open(p_I))["per_pre"][PRE]["blocks"]
    REV = json.load(open(p_rev))
    PBall = json.load(open(p_pb)); PB = PBall["per_pre"][PRE]
    DIAG = json.load(open(p_diag))[PRE]["PR_profile"]
    assert len(vits) == 17

    # ---------------- per-pair LLaMA records (implementation E) ----------------
    rec = {}
    pair_sha = hashlib.sha256()
    for lm in LLAMA:
        for vit in vits:
            stem = f"{LL}/early/pairs/{PRE}/{lm}__{vit}"
            j = json.load(open(stem + ".json"))
            assert j["lm"] == lm and j["vit"] == vit and j["pre"] == PRE and j["K"] == 500
            pair_sha.update(sha(stem + ".json").encode()); pair_sha.update(sha(stem + ".npz").encode())
            rec[lm, vit] = j
    out["inputs"]["E_pairs_llama"] = {"path": f"~/hrh_v2_stage1_20261008/results/S1_2_layermax_llama/early/pairs/{PRE}/",
                                      "n_pairs": len(rec), "sha256_of_member_sha256s": pair_sha.hexdigest()}
    size_L = np.array([PBall["meta"]["inputs"][lm]["num_params"] for lm in LLAMA], float)

    def grid(f):
        return np.array([[f(rec[lm, v]) for lm in LLAMA] for v in vits], float)   # 17 x 3

    def col(met, sub, c):
        if sub == "ctm":
            if c == "s_obs":
                return grid(lambda j: j["summaries"][met]["full"]["s_obs"])
            return grid(lambda j: j["calib_then_max"][met]["M" if c == "s_cal" else "M_unc"])
        return grid(lambda j: j["summaries"][met][sub][c])

    # ---------------- panel (d): LLaMA trends ----------------
    D = {}
    for key, met, sub in ROWS:
        D[key] = {}
        for c in ["s_obs", "uncensored", "s_cal"]:
            t = trend(col(met, sub, c), size_L)
            if sub == "ctm" and c == "s_obs":
                ek = f"{met}|full|s_obs"
            elif sub == "ctm":
                ek = f"{met}|ctm|" + ("M" if c == "s_cal" else "M_unc")
            else:
                ek = f"{met}|{sub}|{c}"
            ref = E["trends_L"][ek]
            assert abs(t["Tr"] - ref["Tr"]) < 1e-9, (key, c, t["Tr"], ref["Tr"])
            assert abs(t["p_one_sided_exact"] - ref["p_one"]) < 1e-9, (key, c)
            t["vit_boot_ci95"] = list(map(float, ref["ci"]))
            t["source_Tr_p_ci"] = f"early/summary.json per_pre/{PRE}/trends_L/{ek}"
            t["per_vit_rho"] = dict(zip(vits, map(float, t.pop("rho"))))
            D[key][c] = t
    out["d_trend"] = D
    # cross-check with the review's independent recomputation and with implementation I
    rv = REV["verdicts"][f"{PRE}|E"]
    assert abs(D["cka_biased|full"]["s_obs"]["Tr"] - rv["P-L2"][0]) < 1e-4
    assert abs(D["cka_biased|full"]["s_cal"]["Tr"] - rv["P-L2"][1]) < 1e-4
    assert abs(D["cka_biased|full"]["uncensored"]["Tr"] - rv["P-L2"][4]) < 1e-4
    assert abs(D["cka_biased|ctm"]["s_cal"]["Tr"] - rv["P-L5"][1]) < 1e-3
    assert abs(D["cka_biased|ctm"]["uncensored"]["Tr"] - rv["P-L5"][2]) < 1e-3
    assert abs(D["cka_biased|fixed_r0.5"]["s_cal"]["Tr"] - rv["P-L3"]["fixed_r0.5"][1]) < 1e-3
    assert abs(D["cka_biased|fixed_r1.0"]["s_cal"]["Tr"] - rv["P-L3"]["fixed_r1.0"][1]) < 1e-3
    assert abs(D["mknn|full"]["s_obs"]["Tr"] - rv["P-L1"][0]) < 1e-4
    Ictm = I["cka_b"]["ctm"]["llama_axis"]
    assert abs(Ictm["s_cal"]["Tr"] - D["cka_biased|ctm"]["s_cal"]["Tr"]) < 1e-9
    assert abs(Ictm["uncensored"]["Tr"] - D["cka_biased|ctm"]["uncensored"]["Tr"]) < 1e-9
    out["d_verdicts_partB"] = {
        "P-L1 mKNN layer-max": "holds", "P-L2 CKA layer-max": "confirmed", "P-L2a thresholds track B-bar": "holds",
        "P-L3 CKA fixed r=0.5, r=1.0": "keeps sign (both)", "P-L4 debiased CKA": "holds",
        "P-L5 CKA calibrate-then-max": "reversed (predicted: keeps sign)",
        "source": "early/summary.json predictions; review/recomputed_verdicts.json (identical for E and I)"}
    out["d_trend_meta"] = dict(pre=PRE, lm_axis=LLAMA, lm_num_params=size_L.tolist(), n_sizes=3, n_perm=6,
                               note="smallest attainable one-sided exact p is 1/6; no value reaches 0.05, so the "
                                    "grey band in (d) is the full range of the 6 relabelling means",
                               ctm="calibrate-then-max: M = max_uv s_cal,uv (tick); M_unc = max_uv (S_uv - mean null_uv) "
                                   "(filled); open circle = raw layer-max T")

    # ---------------- BLOOMZ calibrate-then-max (EXPLORATORY), recomputed from frozen entry arrays ----------------
    Bm, Bmu, Bsz = [], [], []
    import pandas as pd
    df = pd.read_parquet(f"{S1}/S1_2_layermax/panel_layermax.parquet")
    b = df[(df.pre == PRE) & df.lm.isin(BLOOMZ) & (df.metric == "cka_biased") & (df.subset == "full")]
    size_B = b.drop_duplicates("lm").set_index("lm").loc[BLOOMZ, "lm_params"].to_numpy(float)
    rawB = b.pivot(index="vit", columns="lm", values="s_obs").loc[vits, BLOOMZ].to_numpy(float)
    MB = np.zeros((len(vits), 5)); MUB = np.zeros((len(vits), 5)); chk = 0.0
    for i, v in enumerate(vits):
        for k, lm in enumerate(BLOOMZ):
            z = np.load(f"{S1}/S1_2_layermax/pairs/{PRE}/{lm}__{v}.npz")
            obs, c, mu = z["cka_biased.entry_obs"], z["cka_biased.entry_c_alpha"], z["cka_biased.entry_null_mean"]
            MB[i, k] = np.maximum((obs - c) / (1.0 - c), 0.0).max()
            MUB[i, k] = (obs - mu).max()
            chk = max(chk, abs(obs.max() - rawB[i, k]))
    assert chk < 1e-12, chk                                     # entry max == stored layer-max T
    Iax = I["cka_b"]["ctm"]["bloomz_axis"]; Imean = I["cka_b"]["ctm"]["mean_by_lm"]
    for k, lm in enumerate(BLOOMZ):
        assert abs(MB[:, k].mean() - Imean["s_cal"][lm]) < 1e-12, lm
        assert abs(MUB[:, k].mean() - Imean["uncensored"][lm]) < 1e-12, lm
    A_ctm = {}
    for c, Y in [("s_obs", rawB), ("s_cal", MB), ("uncensored", MUB)]:
        t = trend(Y, size_B)
        assert abs(t["Tr"] - Iax[c]["Tr"]) < 1e-9 and abs(t["p_one_sided_exact"] - Iax[c]["p_one"]) < 1e-9, c
        t["vit_boot_ci95"] = boot_ci(t["rho"])
        t["source_Tr_p"] = f"indep/summary.json blocks/cka_b/ctm/bloomz_axis/{c} (recomputed here from entry arrays)"
        t["ci_note"] = "no frozen CI exists; ViT bootstrap recomputed here (B = 4000, seed 20261009)"
        t["per_vit_rho"] = dict(zip(vits, map(float, t.pop("rho"))))
        A_ctm[c] = t
    assert abs(A_ctm["s_cal"]["Tr"] - E["joint_BL"]["cka_biased|ctm|M"]["Tr_BLOOMZ"]) < 1e-9
    A_ctm["status"] = "EXPLORATORY: protocol defined in the LLaMA plan after the BLOOMZ result was known"
    A_ctm["means_by_size"] = {"M": dict(zip(BLOOMZ, MB.mean(0).tolist())), "M_unc": dict(zip(BLOOMZ, MUB.mean(0).tolist()))}
    out["a_trend_ctm_bloomz"] = A_ctm

    # ---------------- panel (e): thresholds by LLaMA size ----------------
    Bbar = PB["Bbar"]
    Ecol = {lm: {} for lm in LLAMA}
    for lm in LLAMA:
        f = lambda key: [rec[lm, v]["summaries"]["cka_biased"]["full"][key] for v in vits]
        row = {"num_params": int(PBall["meta"]["inputs"][lm]["num_params"]), "n_vit": len(vits),
               "T": mse(f("s_obs")), "c_alpha": mse(f("c_alpha")), "c_alpha_entry_at_argmax": mse(f("entry_c_alpha_at_argmax")),
               "null_mean_of_max": mse(f("null_mean")), "s_cal": mse(f("s_cal")), "uncensored": mse(f("uncensored")),
               "Bbar_partB": mse([Bbar[lm][v] for v in vits]),
               "rel_depth_argmax_LM": mse([rec[lm, v]["summaries"]["cka_biased"]["full"]["argmax_A"] / (rec[lm, v]["L_A"] - 1)
                                           for v in vits])}
        for k_out, k_E in [("T", "full|s_obs"), ("c_alpha", "full|c_alpha"), ("s_cal", "full|s_cal"),
                           ("uncensored", "full|uncensored"), ("null_mean_of_max", "full|null_mean")]:
            assert abs(row[k_out]["mean"] - E["means"]["cka_biased"][k_E][lm]) < 1e-12, (lm, k_out)
        assert abs(row["Bbar_partB"]["mean"] - PB["pooled"]["mean_Bbar_by_lm"][lm]) < 1e-12
        assert abs(row["rel_depth_argmax_LM"]["mean"] - E["attribution"]["selection"]["cka_biased"]["rel_A_mean_by_lm"][lm]) < 1e-12
        Ecol[lm] = row
    out["e_thresholds"] = Ecol
    gaps = np.array([rec[lm, v]["summaries"]["cka_biased"]["full"]["c_alpha"] - Bbar[lm][v] for lm in LLAMA for v in vits])
    ca = grid(lambda j: j["summaries"]["cka_biased"]["full"]["c_alpha"]); T = grid(lambda j: j["summaries"]["cka_biased"]["full"]["s_obs"])
    order = sum(int(ca[i, b_] > ca[i, a_]) for i in range(len(vits)) for a_, b_ in [(0, 1), (1, 2), (0, 2)])
    rawgap_lt = {f"{LL_LABEL[LLAMA[a_]]}->{LL_LABEL[LLAMA[b_]]}": {
        "median_raw_gap": float(np.median(T[:, b_] - T[:, a_])), "median_calpha_gap": float(np.median(ca[:, b_] - ca[:, a_])),
        "n_vit_raw_gap_lt_calpha_gap": int(np.sum((T[:, b_] - T[:, a_]) < (ca[:, b_] - ca[:, a_])))}
        for a_, b_ in [(0, 1), (1, 2), (0, 2)]}
    out["e_calpha_minus_Bbar_51_pairs"] = {"mean": float(gaps.mean()), "median_abs": float(np.median(np.abs(gaps))),
                                           "min": float(gaps.min()), "max": float(gaps.max()),
                                           "pearson": float(np.corrcoef(ca.T.ravel(), np.array([[Bbar[lm][v] for v in vits] for lm in LLAMA]).ravel())[0, 1]),
                                           "n_calpha_larger_size_higher": f"{order}/51"}
    out["e_raw_vs_threshold_gaps"] = rawgap_lt
    assert abs(rawgap_lt["7B->13B"]["median_raw_gap"] - REV["diagnostics"][PRE]["gaps"]["0-1"]["median_raw_gap"]) < 1e-12

    # ---------------- panel (f): null level per LLaMA layer vs Prop. 5 ----------------
    F = {}
    PRl = {lm: np.asarray(PB["lm"][lm]["PR"], float) for lm in LLAMA}
    PRv = {v: np.asarray(PB["vit"][v]["PR"], float) for v in vits}
    # partB PR == diag PR (source of BLOOMZ panel c) where both exist
    pr_diff = max([float(np.max(np.abs(PRv[v] - np.asarray(DIAG[v])))) for v in vits] +
                  [float(np.max(np.abs(np.asarray(PB["lm"][lm]["PR"]) - np.asarray(DIAG[lm])))) for lm in BLOOMZ + ["huggyllama__llama-7b"]])
    out["f_PR_partB_vs_diag_maxabs"] = pr_diff
    assert pr_diff < 1e-8, pr_diff
    allsel = []
    for lm in LLAMA:
        L = len(PRl[lm]); meas = np.zeros(L); pred = np.zeros(L)
        for v in vits:
            z = np.load(f"{LL}/early/pairs/{PRE}/{lm}__{v}.npz")
            mu = z["cka_biased.entry_null_mean"]; assert mu.shape == (L, len(PRv[v]))
            meas += mu.max(1)
            pred += np.sqrt(PRl[lm][:, None] * PRv[v][None, :]).max(1) / (M - 1)
            allsel.append(rec[lm, v]["summaries"]["cka_biased"]["full"]["argmax_A"] / (L - 1))
        meas /= len(vits); pred /= len(vits)
        ref = np.asarray(E["attribution"]["null_profile"]["cka_biased"][lm]["null_level_by_layer"], float)
        assert np.max(np.abs(meas - ref)) < 1e-12, lm
        rel = meas / pred - 1
        F[lm] = {"L": L, "rel_depth": list(np.arange(L) / (L - 1)), "measured": meas.tolist(), "predicted": pred.tolist(),
                 "max_abs_rel_err": float(np.abs(rel).max()), "argmax_measured": int(meas.argmax()),
                 "argmax_predicted": int(pred.argmax()), "argmax_rel": float(meas.argmax() / (L - 1)),
                 "max_measured": float(meas.max()), "PR_star": float(PRl[lm].max()), "PR_argmax": int(PRl[lm].argmax())}
    out["f_null_profile"] = F
    out["f_observed_argmax_rel_depth_all_51_pairs"] = list(map(float, allsel))
    out["f_null_profile_meta"] = dict(
        measured="mean over 17 ViTs of max over ViT layers of the entry-wise permutation null mean (K=500, E npz); "
                 "equals early/summary.json attribution/null_profile/cka_biased to 1e-12",
        predicted="mean over 17 ViTs of max over ViT layers of sqrt(PR_A,u PR_B,v)/(m-1), PR from partB_features.json",
        mknn_baseline=10 / (M - 1))
    out["notes"].append("LLaMA rows (d)-(f) read the E implementation; E and I agree on every plan-level quantity "
                        "(validation_cross.json, review). Every Tr/p equals early/summary.json to 1e-9.")
    out["notes"].append("BLOOMZ calibrate-then-max row in (a) is EXPLORATORY; LLaMA rows in (d) are frozen Part B predictions.")
    return out
