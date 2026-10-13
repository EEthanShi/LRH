# ---------------------------------------------------------------- S2.1 decision table (inserted 2026-10-09, writer v1)
import glob as _glob

INPUTS["s21_summary"] = S2 + "/results/S2_1/s21_summary.json"
INPUTS["eps_s21"] = S2 + "/prereg/eps_S2_1.json"
for _k in ["s21_summary", "eps_s21"]:
    manifest["inputs"][_k] = {"path": INPUTS[_k], "sha256": sha(INPUTS[_k])}

S21 = jload("s21_summary")
assert S21["interim"] is False and S21["R"] == 64, (S21["interim"], S21["R"])
manifest["checks"]["s21_verdict"] = S21["verdict"]
T21 = S21["tables"]
R21 = [("dN_k10", r"$T_{N,10}$"), ("dO", r"$T_O$"), ("dE", r"$T_E$"), ("dMST", r"$T_{\mathrm{MST}}$"),
       ("dSOG_t0p1", r"SOG$_{0.1}$"), ("dAff", r"$T_{G/\mathrm{Aff}}$"), ("Gs", r"$G_s$"),
       ("dCKA", r"$1-\mathrm{CKA}$"), ("dIso", r"$\Delta_{\rm iso}$"), ("dPi", r"$\Pi$")]
SETS21 = [("within_A", r"InfoNCE $\tau{=}0.2$, within"), ("within_Atau", r"InfoNCE $\tau{=}0.5$, within"),
          ("within_B", "spectral, within"), ("cross_A_B", r"InfoNCE $\tau{=}0.2$ vs spectral"),
          ("cross_A_Atau", r"InfoNCE $\tau{=}0.2$ vs $\tau{=}0.5$"), ("cross_Atau_B", r"InfoNCE $\tau{=}0.5$ vs spectral"),
          ("bern_self", r"link check, self"), ("bern_peer", r"link check, peer")]
DEC21 = {"reliable": r"\textsc{r}", "inconclusive": r"\textsc{i}", "unreliable": r"\textsc{u}"}
l21 = []
for Nk, Nlab in [("50000", "50k"), ("500000", "500k")]:
    for j, (sk, slab) in enumerate(SETS21):
        row = T21[f"{sk}_{Nk}|z"]
        cells = []
        for rk, _ in R21:
            c = row.get(rk)
            if c is None:
                cells.append("--")
                continue
            cells.append(f"{c['x']}/{c['n']}\\,{DEC21[c['decision']]}")
        lead = Nlab if j == 0 else ""
        l21.append(f"{lead} & {slab} & " + " & ".join(cells) + r" \\")
    l21.append(r"\addlinespace[2pt]")
l21 = l21[:-1]
write("tab_s21_full.tex", r"$N$ & Pair set & " + " & ".join(lab for _, lab in R21) + r" \\" + "\n\\midrule\n"
      + "\n".join(l21) + "\n")

E21 = jload("eps_s21")
manifest["checks"]["s21_eps_z"] = {rk: E21["eps"]["z"][rk] for rk, _ in R21}
manifest["checks"]["s21_eps_n_pairs"] = E21["n_pairs"]
ep = S21["endpoints"]
manifest["checks"]["s21_endpoints"] = {
    "i_p_cross": {"50k": ep["i"]["p_cross_Nmin"], "500k": ep["i"]["p_cross_Nmax"]},
    "i_ceiling": ep["i"]["ceiling_Nmax"], "i_gap": ep["i"]["gap_to_ceiling_Nmax"],
    "i_improvement": ep["i"]["improvement_500k_minus_50k"], "i_holm": ep["i"]["holm"], "i_pass": ep["i"]["pass"],
    "ii_pass": ep["ii"]["pass"], "iii": ep["iii"], "iv": ep["iv"], "v_pass": ep["v"]["pass"]}
us = S21["ustat"]
manifest["checks"]["s21_ustat_500k"] = {k: us[k] for k in us if k.endswith("_500000")}
manifest["checks"]["s21_ustat_50k"] = {k: us[k] for k in us if k.endswith("_50000")}
manifest["checks"]["s21_cka_cross_minus_within"] = S21["cka_cross_minus_within"]
manifest["checks"]["s21_beta"] = {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "values"})
                                  for k, v in S21["beta"].items()}
manifest["checks"]["s21_covariates"] = S21["covariates"]

# ---------------------------------------------------------------- compute accounting (all on ds2213)
comp = {}
secs = {}
nunits = {}
for f in _glob.glob(S1 + "/results/S1_1/v11/units/*/*/*.json"):
    with open(f) as fh:
        d = json.load(fh)
    key = f"{d['config']}|{d['kind']}"
    secs[key] = secs.get(key, 0.0) + float(d["seconds"])
    nunits[key] = nunits.get(key, 0) + 1
comp["s11_v11_cpu_hours_by_config_kind"] = {k: v / 3600 for k, v in secs.items()}
comp["s11_v11_units"] = nunits
comp["s11_v11_cpu_hours_total"] = sum(secs.values()) / 3600


def runs_cost(d):
    metas = _glob.glob(d + "/*/meta.json")
    tt = []
    mt = []
    for f in metas:
        with open(f) as fh:
            m = json.load(fh)
        tt.append(float(m["t_total_s"]))
        mt.append(os.path.getmtime(f))
    return {"n_runs": len(metas), "sum_run_hours": sum(tt) / 3600,
            "median_run_s": float(np.median(tt)) if tt else None,
            "meta_mtime_span_hours": (max(mt) - min(mt)) / 3600 if mt else None}


for nm in ["S2_2_pilot", "S2_2_main", "S2_1_pilot", "S2_1_main"]:
    comp[nm] = runs_cost(S2 + "/runs/" + nm)
manifest["checks"]["compute"] = comp
print(json.dumps(comp, indent=1))
