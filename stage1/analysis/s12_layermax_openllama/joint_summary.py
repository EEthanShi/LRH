#!/usr/bin/env python
"""OpenLLaMA axis: the joint three-family summary J, EXPLORATORY (plan section 5.4; VERDICTS_SPEC.md section 7
"Section J"). New file, written at S0b by sub-role PB of oll-axis-extract.

GATED (plan F1(a)): the first action after argument parsing is guard.require_partb_frozen(out_root); before
it, nothing but STATUS.md is read. It exits 3 until the `[oll-axis-predict] Part B FROZEN` STATUS line exists.

Instance: primary pre prh_clip+l2, biased CKA, layer-max (protocol full); X in {s_cal, unc}.
Families (each read in the implementation's own format; pair readers shared with forecast_ol.py):
  B = BLOOMZ 560m < 1b1 < 1b7 < 3b < 7b1, frozen 8-LM files   (E: results/S1_2_layermax/pairs/<pre>/*.json,
                                                                 I: results/S1_2_layermax/indep/<pre>/pairs/*.npz)
  L = LLaMA-1 7b < 13b < 30b, the LLaMA-axis files               (E: results/S1_2_layermax_llama/early/pairs/<pre>/,
                                                                 I: results/S1_2_layermax_llama/indep/<pre>/pairs/)
  O = OpenLLaMA 3b < 7b < 13b, this run's files                 (E: results/S1_2_layermax_openllama/early/pairs/<pre>/,
                                                                 I: results/S1_2_layermax_openllama/indep/<pre>/pairs/)
Tr (spec section 3): rho_v = Pearson on average ranks (scipy rankdata) between the size labels and the scores of
ViT v; a ViT with all scores exactly equal (or any non-finite) is excluded; Tr = mean over the rest; the same
relabelling sigma for every ViT. dTr_f(sigma) = Tr_f(X; sigma) - Tr_f(raw T; sigma).
J (spec section 7): s_B = s_L = +1, s_O = s_O_J(prh_clip+l2) from Part B; 4320 joint relabellings enumerated;
  J_min(sigma) = max_f s_f dTr_f(sigma_f), pmin = #{J_min(sigma) <= J_min(id) + 1e-9}/4320, product cross-check
  n_B(t) n_L(t) n_O(t)/4320 (must equal pmin, else exit 6); J_mean likewise; nO; conditional size: for each of the
  6 sigma_O taken as the observed labelling (the member labelled 13b is the new one; its comparisons with the two
  others, d_raw = sign(T_new - T_a), d_X = sign(X_new - X_a); f- = #{d_X < d_raw} if s_O = +1 else #{d_X > d_raw}),
  v0's rule rejects iff J_min p (sigma_O as identity) <= 0.05 and s_O dTr_O(sigma_O) <= -0.10 and f- >= 3;
  cond_size = #rejecting / 6. Labels "EXPLORATORY" and the provenance string.
Self-check (exit 7 if it fails; B and L are never stand-ins): with B and L alone the known ranks of plan section 0 are
  reproduced: BLOOMZ dTr(s_cal) -0.912 rank 6/120, unc -1.182 rank 2/120; LLaMA -1.529 / -1.676 rank 1/6;
  BLOOMZ+LLaMA mean statistic 6/720 and 2/720.

Outputs (out_root = results/S1_2_layermax_openllama/joint/<impl>/):
  verdicts_joint.json  {"schema": "OL-VERDICTS v1", "impl", "inputs", "v"} with every key of
                       verdict_keys.required_keys("J")
  joint_detail.json    dTr over all relabellings per family, enumeration summary, per-sigma_O conditional-size table,
                       near-tie audit, self-check

RUN (step S8, after the main runs and V-O5; CPU, 1 thread, < 1 min):
  cd ~/hrh_v2_stage1_20261008
  analysis/s12_layermax_openllama/launch_ol.sh joint_E 1 -- venv/bin/python analysis/s12_layermax_openllama/joint_summary.py --impl E
  analysis/s12_layermax_openllama/launch_ol.sh joint_I 1 -- venv/bin/python analysis/s12_layermax_openllama/joint_summary.py --impl I
  venv/bin/python analysis/s12_layermax_openllama/compare_verdicts.py results/S1_2_layermax_openllama/joint/E/verdicts_joint.json results/S1_2_layermax_openllama/joint/I/verdicts_joint.json --sections J

S0B TEST MODE: as forecast_ol.py (--test-root, --partb, --out; default out ~/oll_s0b_scratch/PB/joint/<impl>).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
sys.path.insert(0, PROJ)
sys.path.insert(0, AXIS)

import guard  # noqa: E402
import forecast_ol as FO  # noqa: E402  (readers and paths only; no data loaded at import)

PRE = "prh_clip+l2"
XS = ("s_cal", "unc")
TOL = 1e-9
NREL = 4320
BLOOMZ = ["bigscience__bloomz-560m", "bigscience__bloomz-1b1", "bigscience__bloomz-1b7", "bigscience__bloomz-3b",
          "bigscience__bloomz-7b1"]
LLAMA = ["huggyllama__llama-7b", "huggyllama__llama-13b", "huggyllama__llama-30b"]
LABEL = "EXPLORATORY"
PROVENANCE = "BLOOMZ hypothesis-generating; BLOOMZ fails B4; BLOOMZ and LLaMA known before this plan"
# plan section 0 (known before this plan): (dTr rounded to 3 dp, rank) per family and X, and the 720 mean ranks
KNOWN = {"s_cal": {"B": (-0.912, 6), "L": (-1.529, 1), "BL720": 6},
         "unc": {"B": (-1.182, 2), "L": (-1.676, 1), "BL720": 2}}


def perms(n: int):
    import numpy as np
    return np.array(list(itertools.permutations(range(1, n + 1))), dtype=float)   # identity first


def tr_all(G):
    """Tr(sigma) for every relabelling sigma (identity first) of the rows of G (members in size order x ViTs)."""
    import numpy as np
    from scipy.stats import rankdata
    n, nv = G.shape
    keep = [j for j in range(nv) if np.all(np.isfinite(G[:, j])) and not np.all(G[:, j] == G[0, j])]
    if not keep:
        return np.full(len(perms(n)), np.nan), nv
    R = np.stack([rankdata(G[:, j]) for j in keep], axis=1)
    Rc = R - R.mean(axis=0, keepdims=True)
    P = perms(n)
    Pc = P - P.mean(axis=1, keepdims=True)
    num = Pc @ Rc
    den = np.sqrt((Pc ** 2).sum(axis=1))[:, None] * np.sqrt((Rc ** 2).sum(axis=0))[None, :]
    return (num / den).mean(axis=1), nv - len(keep)


def near_ties(G) -> int:
    import numpy as np
    n = 0
    for j in range(G.shape[1]):
        for a, b in itertools.combinations(range(G.shape[0]), 2):
            x, y = G[a, j], G[b, j]
            if x != y and abs(x - y) <= 1e-12 * max(abs(x), abs(y)):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=("E", "I"), required=True)
    ap.add_argument("--test-root", default=None, help="S0b test mode only")
    ap.add_argument("--partb", default=None, help="S0b test mode only")
    ap.add_argument("--out", default=None, help="S0b test mode only")
    args = ap.parse_args()
    test, out_root, test_root, partb = FO.resolve(args, "joint")
    gate = guard.require_partb_frozen(out_root, script="joint_summary.py")    # F1(a): FIRST data-related action
    # ---- nothing involving a pair file, score or Part B value is opened before this line ----
    import numpy as np

    F = json.load(open(partb))
    ols = list(F["ol_lms"])
    vits = list(F["vits"])
    if not test:
        import verdict_keys as VK
        assert ols == list(FO.OL_SLUGS), ols
        assert vits == list(VK.VITS)
    s_O = int(F["per_pre"][PRE]["s_O_J"])
    assert s_O in (1, -1)
    inputs = {os.path.relpath(partb, PROJ): FO.sha256_file(partb)}

    def grids(group, lms):
        G = {k: np.empty((len(lms), len(vits))) for k in ("T", "s_cal", "unc")}
        for i, lm in enumerate(lms):
            for j, v in enumerate(vits):
                p = FO.pair_path(args.impl, group, PRE, lm, v, test_root)
                d = FO.layermax(args.impl, p, PRE, lm, v, check_identity=not (test and group == "ol"))
                inputs[os.path.relpath(p, PROJ)] = FO.sha256_file(p)
                for k in G:
                    G[k][i, j] = d[k]
        return G

    fams = {"B": grids("bloomz", BLOOMZ), "L": grids("llama", LLAMA), "O": grids("ol", ols)}
    sgn = {"B": 1, "L": 1, "O": s_O}
    V = {"J|label": LABEL, "J|provenance": PROVENANCE}
    detail = {"impl": args.impl, "gate": gate, "label": LABEL, "provenance": PROVENANCE, "pre": PRE,
              "metric": "cka_biased", "protocol": "full", "s": sgn, "families": {"B": BLOOMZ, "L": LLAMA, "O": ols},
              "X": {}}
    selfcheck_ok = True
    for X in XS:
        d, nconst, nt = {}, {}, {}
        for f, G in fams.items():
            tx, cx = tr_all(G[X])
            tr, cr = tr_all(G["T"])
            d[f] = tx - tr
            nconst[f] = {"X": cx, "raw": cr}
            nt[f] = {"X": near_ties(G[X]), "raw": near_ties(G["T"])}
        sB, sL, sO = (sgn["B"] * d["B"], sgn["L"] * d["L"], sgn["O"] * d["O"])
        jmin = np.maximum(np.maximum(sB[:, None, None], sL[None, :, None]), sO[None, None, :])
        jmean = (sB[:, None, None] + sL[None, :, None] + sO[None, None, :]) / 3.0
        t = float(jmin[0, 0, 0])
        cnt_min = int(np.sum(jmin <= t + TOL))
        nf = {f: int(np.sum(s_ <= t + TOL)) for f, s_ in (("B", sB), ("L", sL), ("O", sO))}
        prod = nf["B"] * nf["L"] * nf["O"]
        if prod != cnt_min:
            print(f"J product cross-check FAILED for {X}: {prod} != {cnt_min} (implementation error)", file=sys.stderr)
            sys.exit(6)
        jm = float(jmean[0, 0, 0])
        cnt_mean = int(np.sum(jmean <= jm + TOL))
        nO = int(np.sum(sO <= sO[0] + TOL))
        # conditional size of v0's rule over the 6 sigma_O
        P3 = perms(3)
        GT, GX = fams["O"]["T"], fams["O"][X]
        cs_rows = []
        for j in range(6):
            lab = P3[j]                                    # size label of member i under sigma_O
            new = int(np.argmax(lab))                      # member labelled 13b
            others = [i for i in range(3) if i != new]
            fm = fp = 0
            for v in range(len(vits)):
                for a in others:
                    dr = int(np.sign(GT[new, v] - GT[a, v]))
                    dx = int(np.sign(GX[new, v] - GX[a, v]))
                    fm += int(dx < dr)
                    fp += int(dx > dr)
            fminus = fm if s_O == 1 else fp
            tj = max(float(sB[0]), float(sL[0]), float(sO[j]))
            nj = (int(np.sum(sB <= tj + TOL)), int(np.sum(sL <= tj + TOL)), int(np.sum(sO <= tj + TOL)))
            cntj = nj[0] * nj[1] * nj[2]
            rej = bool(cntj * 20 <= NREL and float(sO[j]) <= -0.10 and fminus >= 3)
            cs_rows.append({"sigma_O_labels": [int(x) for x in lab], "new_member": ols[new], "sO_dTrO": float(sO[j]),
                            "t": tj, "n_BLO": list(nj), "pmin": cntj / NREL, "fminus": fminus,
                            "fminus_down": fm, "fminus_up": fp, "rejects": rej})
        cond = sum(r["rejects"] for r in cs_rows) / 6.0
        V[f"J|{X}|sO"] = s_O
        V[f"J|{X}|Jmin"] = t
        V[f"J|{X}|pmin"] = cnt_min / NREL
        V[f"J|{X}|pmin_product"] = prod / NREL
        V[f"J|{X}|Jmean"] = jm
        V[f"J|{X}|pmean"] = cnt_mean / NREL
        V[f"J|{X}|nO"] = nO
        V[f"J|{X}|cond_size"] = cond
        for f in ("B", "L", "O"):
            V[f"J|{X}|dTr|{f}"] = float(d[f][0])
        # self-check with B and L alone (plan section 0)
        rB = int(np.sum(d["B"] <= d["B"][0] + TOL))
        rL = int(np.sum(d["L"] <= d["L"][0] + TOL))
        BL = (d["B"][:, None] + d["L"][None, :]) / 2.0
        rBL = int(np.sum(BL <= BL[0, 0] + TOL))
        kn = KNOWN[X]
        sc = {"B": {"dTr": float(d["B"][0]), "rank": rB, "of": int(d["B"].size), "plan": list(kn["B"])},
              "L": {"dTr": float(d["L"][0]), "rank": rL, "of": int(d["L"].size), "plan": list(kn["L"])},
              "BL_mean": {"rank": rBL, "of": int(BL.size), "plan": kn["BL720"]}}
        sc["pass"] = bool(round(float(d["B"][0]), 3) == kn["B"][0] and rB == kn["B"][1]
                          and round(float(d["L"][0]), 3) == kn["L"][0] and rL == kn["L"][1] and rBL == kn["BL720"])
        selfcheck_ok &= sc["pass"]
        V[f"x|J|{X}|selfcheck_pass"] = sc["pass"]
        V[f"x|J|{X}|nO_over_6"] = f"{nO}/6"
        V[f"x|J|{X}|dTr_O_equals_dTr_L"] = bool(abs(float(d["O"][0]) - float(d["L"][0])) <= 1e-12)
        detail["X"][X] = {"dTr_all": {f: [float(x) for x in d[f]] for f in d}, "n_vit_excluded": nconst,
                          "near_ties_1e-12": nt, "Jmin_id": t, "count_Jmin": cnt_min, "n_f_at_t": nf,
                          "Jmean_id": jm, "count_Jmean": cnt_mean, "nO": nO, "cond_size": cond,
                          "cond_size_rows": cs_rows, "selfcheck": sc,
                          "printing_rule": "print only with: EXPLORATORY; n_O/6; conditional size; provenance labels"}
        print(f"{args.impl} {X}: dTr B {d['B'][0]:+.4f} (rank {rB}/120) L {d['L'][0]:+.4f} (rank {rL}/6) "
              f"O {d['O'][0]:+.4f} (s_O {s_O:+d}, nO {nO}/6) | Jmin {t:+.4f} p {cnt_min}/{NREL} (product {prod}) "
              f"| Jmean {jm:+.4f} p {cnt_mean}/{NREL} | cond size {cond:.4f} | BL720 {rBL} | selfcheck {sc['pass']}",
              flush=True)
    detail["selfcheck_pass"] = selfcheck_ok
    import verdict_keys as VK
    miss = [k for k in VK.required_keys("J") if k not in V]
    if miss:
        raise SystemExit(f"internal error: required J keys missing: {miss}")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "verdicts_joint.json"), "w") as f:
        json.dump({"schema": "OL-VERDICTS v1", "impl": args.impl, "inputs": inputs, "v": V}, f, indent=1,
                  sort_keys=True)
    with open(os.path.join(out_root, "joint_detail.json"), "w") as f:
        json.dump(detail, f, indent=1)
    print(f"wrote {out_root}/verdicts_joint.json, joint_detail.json (n inputs {len(inputs)})", flush=True)
    if not selfcheck_ok:
        print("self-check against plan section 0 FAILED (BLOOMZ/LLaMA ranks)", file=sys.stderr)
        sys.exit(7)


if __name__ == "__main__":
    main()
