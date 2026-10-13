"""F_aggregation_app data layer, OpenLLaMA column (c, f, i). Copy of ~/hrh_v2_figs/F1/f1_llama_data.py re-pointed at
the OpenLLaMA axis and extended to the six trend rows, the thresholds by size and the baseline per language-model layer
of the full appendix figure (the F_aggregation copy keeps only four trend rows and no profile).

Every number is read from frozen artifacts or recomputed from them and asserted against the frozen aggregate.
Inputs (all under ~/hrh_v2_stage1_20261008/results/S1_2_layermax_openllama/, opened read-only):
  early/pairs/prh_clip+l2/openlm-research__open_llama_{3b,7b,13b}__<vit>.{json,npz}  implementation E, per pair (K=500)
  partB/partB_features_ol.json          Part B, frozen before any 13B cross-model score: num_params, B-bar, PR
  layermax_openllama_summary.json       merged summary (sha256 2777dadf...), embedded.E_summary.v (= I = review R)
Status: every OpenLLaMA outcome is DESCRIPTIVE (plan, author decision R.1.2); smallest attainable one-sided p is 1/6.
"""
import hashlib, itertools, json, os
import numpy as np

S1 = os.path.expanduser("~/hrh_v2_stage1_20261008/results")
OLD = f"{S1}/S1_2_layermax_openllama"
PRE = "prh_clip+l2"
M = 1024
OL = ["openlm-research__open_llama_3b", "openlm-research__open_llama_7b", "openlm-research__open_llama_13b"]
OL_SHORT = {"openlm-research__open_llama_3b": "3b", "openlm-research__open_llama_7b": "7b",
            "openlm-research__open_llama_13b": "13b"}
OL_LABEL = {"openlm-research__open_llama_3b": "3B", "openlm-research__open_llama_7b": "7B",
            "openlm-research__open_llama_13b": "13B"}
SUMMARY_SHA = "2777dadf287dfc02433eb07e84cefad0ec67381258a199b9d306b6c7d32aebaa"
# rows of the trend panel (same order as f1_llama_data.ROWS); 'ctm' = calibrate each layer pair, then take the maximum
ROWS = [("cka_biased|full", "cka_biased", "full"),
        ("cka_biased|ctm", "cka_biased", "ctm"),
        ("cka_biased|fixed_r1.0", "cka_biased", "fixed_r1.0"),
        ("cka_biased|fixed_r0.5", "cka_biased", "fixed_r0.5"),
        ("mknn|full", "mknn", "full"),
        ("cka_debiased|full", "cka_debiased", "full")]
COLS = {"s_obs": "raw", "uncensored": "unc", "s_cal": "s_cal"}       # pair-json column -> summary key suffix


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


def trend(Y, size):
    """Tr = mean over ViTs of Spearman(size, score); exact null over all relabellings of the sizes (same for all ViTs)."""
    rho = _rho_rows(size, Y)
    tr = float(rho.mean())
    null = np.array([_rho_rows(size[list(p)], Y).mean() for p in itertools.permutations(range(len(size)))])
    return dict(Tr=tr, p_one_sided_exact=float(np.mean(null >= tr - 1e-12)), n_perm=int(len(null)),
                null_min=float(null.min()), null_max=float(null.max()), rho=rho)


def mse(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "min": float(v.min()),
            "max": float(v.max())}


def compute(vits):
    p_sum, p_pb = f"{OLD}/layermax_openllama_summary.json", f"{OLD}/partB/partB_features_ol.json"
    out = {"inputs": {}, "notes": []}
    for k, p in dict(layermax_openllama_summary=p_sum, partB_features_ol=p_pb).items():
        out["inputs"][k] = {"path": p.replace(os.path.expanduser("~"), "~"), "sha256": sha(p)}
    assert out["inputs"]["layermax_openllama_summary"]["sha256"] == SUMMARY_SHA
    SUM = json.load(open(p_sum))
    V = SUM["embedded"]["E_summary"]["v"]
    PBall = json.load(open(p_pb)); PB = PBall["per_pre"][PRE]
    assert len(vits) == 17 and sorted(SUM["embedded"]["E_summary"]["panel"]["vits"]) == sorted(vits)
    src_sum = out["inputs"]["layermax_openllama_summary"]["path"]

    # ---------------- per-pair records (implementation E) ----------------
    rec, pair_sha = {}, hashlib.sha256()
    for lm in OL:
        for vit in vits:
            stem = f"{OLD}/early/pairs/{PRE}/{lm}__{vit}"
            j = json.load(open(stem + ".json"))
            assert j["lm"] == lm and j["vit"] == vit and j["pre"] == PRE and j["K"] == 500
            pair_sha.update(sha(stem + ".json").encode()); pair_sha.update(sha(stem + ".npz").encode())
            rec[lm, vit] = j
    out["inputs"]["E_pairs_openllama"] = {
        "path": f"~/hrh_v2_stage1_20261008/results/S1_2_layermax_openllama/early/pairs/{PRE}/",
        "n_pairs": len(rec), "sha256_of_member_sha256s": pair_sha.hexdigest(),
        "sha256_note": "sha256 over the member (json, npz) sha256 hex strings, OL x vits order"}
    size = np.array([PBall["num_params"][lm] for lm in OL], float)
    out["num_params"] = dict(zip(OL, map(int, size)))

    def grid(f):
        return np.array([[f(rec[lm, v]) for lm in OL] for v in vits], float)   # 17 x 3

    def col(met, sub, c):
        if sub == "ctm":
            if c == "s_obs":
                return grid(lambda j: j["summaries"][met]["full"]["s_obs"])
            return grid(lambda j: j["calib_then_max"][met]["M" if c == "s_cal" else "M_unc"])
        return grid(lambda j: j["summaries"][met][sub][c])

    # ---------------- trends (panel c) ----------------
    T = {}
    for key, met, sub in ROWS:
        T[key] = {}
        for c in ["s_obs", "uncensored", "s_cal"]:
            t = trend(col(met, sub, c), size)
            if sub == "ctm" and c == "s_obs":
                sk = f"{PRE}|{met}|full|raw"
            elif sub == "ctm":
                sk = f"{PRE}|{met}|ctm|" + ("M" if c == "s_cal" else "M_unc")
            else:
                sk = f"{PRE}|{met}|{sub}|{COLS[c]}"
            assert abs(t["Tr"] - V["tr|" + sk]) < 1e-9, (key, c, t["Tr"], V["tr|" + sk])
            assert abs(t["p_one_sided_exact"] - V["tr_pge|" + sk]) < 1e-9, (key, c)
            t["vit_boot_ci95"] = list(map(float, V["x|tr_ci|" + sk]))
            t["src"] = {"path": src_sum, "sha256": SUMMARY_SHA,
                        "key_Tr": f"embedded/E_summary/v/tr|{sk}", "key_p": f"embedded/E_summary/v/tr_pge|{sk}",
                        "key_ci": f"embedded/E_summary/v/x|tr_ci|{sk}"}
            t["per_vit_rho"] = dict(zip(vits, map(float, t.pop("rho"))))
            T[key][c] = t
    out["trend"] = T

    # ---------------- thresholds by size (panel f) ----------------
    Bbar = PB["Bbar"]
    R = {}
    for lm in OL:
        f = lambda k: [rec[lm, v]["summaries"]["cka_biased"]["full"][k] for v in vits]
        row = {"num_params": int(PBall["num_params"][lm]), "n_vit": len(vits),
               "T": mse(f("s_obs")), "c_alpha": mse(f("c_alpha")),
               "c_alpha_entry_at_argmax": mse(f("entry_c_alpha_at_argmax")),
               "null_mean_of_max": mse(f("null_mean")), "s_cal": mse(f("s_cal")), "uncensored": mse(f("uncensored")),
               "Bbar_partB": mse([Bbar[lm][v] for v in vits]),
               "rel_depth_argmax_LM": mse([rec[lm, v]["summaries"]["cka_biased"]["full"]["argmax_A"] /
                                           (rec[lm, v]["L_A"] - 1) for v in vits]),
               "per_vit": {"T": dict(zip(vits, map(float, f("s_obs")))),
                           "c_alpha": dict(zip(vits, map(float, f("c_alpha")))),
                           "Bbar_partB": {v: float(Bbar[lm][v]) for v in vits}}}
        s = OL_SHORT[lm]
        for k_out, k_sum in [("T", "raw"), ("c_alpha", "c"), ("Bbar_partB", "Bbar"), ("null_mean_of_max", "mu"),
                             ("s_cal", "s_cal"), ("uncensored", "unc")]:
            ref = V[f"x|mean|{PRE}|cka_biased|full|{k_sum}|{s}"]
            assert abs(row[k_out]["mean"] - ref) < 1e-12, (lm, k_out, row[k_out]["mean"], ref)
            row[k_out]["src_mean"] = f"embedded/E_summary/v/x|mean|{PRE}|cka_biased|full|{k_sum}|{s}"
        R[lm] = row
    rp = SUM["reviewer_posthoc"]["means"][PRE]           # second copy of the same means in the merged summary
    for i, lm in enumerate(OL):
        assert abs(R[lm]["T"]["mean"] - rp["raw"][i]) < 1e-12 and abs(R[lm]["c_alpha"]["mean"] - rp["c"][i]) < 1e-12
        assert abs(R[lm]["Bbar_partB"]["mean"] - rp["Bbar"][i]) < 1e-12
    out["thresholds"] = R
    out["thresholds_src"] = {
        "T": f"early/pairs/{PRE}/<lm>__<vit>.json summaries/cka_biased/full/s_obs",
        "c_alpha": f"early/pairs/{PRE}/<lm>__<vit>.json summaries/cka_biased/full/c_alpha",
        "c_alpha_entry_at_argmax": f"early/pairs/{PRE}/<lm>__<vit>.json summaries/cka_biased/full/entry_c_alpha_at_argmax",
        "Bbar_partB": f"partB/partB_features_ol.json per_pre/{PRE}/Bbar/<lm>/<vit>",
        "num_params": "partB/partB_features_ol.json num_params/<lm>"}

    # threshold offsets c_alpha - B-bar: all 51 pairs, and the 17 new 13B pairs (= P-O2a(i) of the frozen plan)
    gap = np.array([[R[lm]["per_vit"]["c_alpha"][v] - R[lm]["per_vit"]["Bbar_partB"][v] for lm in OL] for v in vits])
    g13 = gap[:, 2]
    ref_med = SUM["headline"][f"PO2ai|{PRE}|median"]
    ref_rng = SUM["reviewer_posthoc"]["PO2ai_range"][PRE]
    assert abs(np.median(g13) - ref_med) < 1e-12, (np.median(g13), ref_med)
    assert abs(g13.min() - ref_rng[0]) < 1e-12 and abs(g13.max() - ref_rng[1]) < 1e-12, (g13.min(), g13.max(), ref_rng)
    for i, v in enumerate(vits):
        assert abs(g13[i] - V[f"PO2ai|{PRE}|offset|{v}"]) < 1e-12, v
    ca = gap + np.array([[R[lm]["per_vit"]["Bbar_partB"][v] for lm in OL] for v in vits])
    order = sum(int(ca[i, b_] > ca[i, a_]) for i in range(len(vits)) for a_, b_ in [(0, 1), (1, 2), (0, 2)])
    out["calpha_minus_Bbar"] = {
        "all_51_pairs": {"min": float(gap.min()), "max": float(gap.max()), "median": float(np.median(gap)),
                         "mean": float(gap.mean())},
        "new_13b_17_pairs": {"min": float(g13.min()), "max": float(g13.max()), "median": float(np.median(g13)),
                             "src_median": f"{src_sum} headline/PO2ai|{PRE}|median",
                             "src_range": f"{src_sum} reviewer_posthoc/PO2ai_range/{PRE}"},
        "n_calpha_larger_size_higher_recomputed": f"{order}/51",
        "PO2aii": {"x": V[f"PO2aii|{PRE}|x"], "n": V[f"PO2aii|{PRE}|n"], "verdict": V[f"PO2aii|{PRE}|verdict"],
                   "src": f"{src_sum} embedded/E_summary/v/PO2aii|{PRE}|{{x,n,verdict}}"}}
    assert V[f"PO2aii|{PRE}|x"] == 33 and V[f"PO2aii|{PRE}|n"] == 33

    # ---------------- null level per OpenLLaMA layer vs Proposition (panel i) ----------------
    PRl = {lm: np.asarray(PB["lm"][lm]["PR"], float) for lm in OL}
    PRv = {v: np.asarray(PB["vit"][v]["PR"], float) for v in vits}
    F, allsel = {}, []
    for lm in OL:
        L = len(PRl[lm]); meas = np.zeros(L); pred = np.zeros(L)
        for v in vits:
            z = np.load(f"{OLD}/early/pairs/{PRE}/{lm}__{v}.npz")
            mu = z["cka_biased.entry_null_mean"]; assert mu.shape == (L, len(PRv[v])), (lm, v, mu.shape)
            assert rec[lm, v]["L_A"] == L
            meas += mu.max(1)
            pred += np.sqrt(PRl[lm][:, None] * PRv[v][None, :]).max(1) / (M - 1)
            allsel.append(rec[lm, v]["summaries"]["cka_biased"]["full"]["argmax_A"] / (L - 1))
        meas /= len(vits); pred /= len(vits)
        rel = meas / pred - 1
        argmax_rel = float(meas.argmax() / (L - 1))
        ref = V[f"x|nullprof|{PRE}|cka_biased|{OL_SHORT[lm]}|argmax_rel"]
        assert abs(argmax_rel - ref) < 1e-12, (lm, argmax_rel, ref)
        F[lm] = {"L": L, "rel_depth": list(np.arange(L) / (L - 1)), "measured": meas.tolist(), "predicted": pred.tolist(),
                 "max_abs_rel_err": float(np.abs(rel).max()), "argmax_measured": int(meas.argmax()),
                 "argmax_predicted": int(pred.argmax()), "argmax_rel": argmax_rel,
                 "argmax_rel_src": f"{src_sum} embedded/E_summary/v/x|nullprof|{PRE}|cka_biased|{OL_SHORT[lm]}|argmax_rel",
                 "max_measured": float(meas.max()), "PR_star": float(PRl[lm].max()), "PR_argmax": int(PRl[lm].argmax())}
    out["null_profile"] = F
    out["observed_argmax_rel_depth_all_51_pairs"] = list(map(float, allsel))
    out["null_profile_meta"] = dict(
        measured="mean over 17 ViTs of max over ViT layers of the entry-wise permutation null mean "
                 "(K=500, E npz key cka_biased.entry_null_mean)",
        predicted="mean over 17 ViTs of max over ViT layers of sqrt(PR_A,u PR_B,v)/(m-1), PR from partB_features_ol.json "
                  f"per_pre/{PRE}/lm/<lm>/PR and vit/<vit>/PR",
        check="argmax over layers equals x|nullprof|...|argmax_rel of the merged summary to 1e-12",
        mknn_baseline=10 / (M - 1))
    out["notes"].append("OpenLLaMA axis: every outcome descriptive; Tr, p and CI equal layermax_openllama_summary.json "
                        "(embedded E summary) to 1e-9; per-size means equal its x|mean keys to 1e-12.")
    return out
