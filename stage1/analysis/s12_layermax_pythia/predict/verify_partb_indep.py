"""Independent re-derivation of the Part B decision quantities of the Pythia axis (role pyt-axis-predict, step S5).

Descriptive cross-check written by the predict role after partb_pyt.py ran; it is not a plan step, changes no
frozen code, and nothing it prints enters any verdict. It does NOT import partb_pyt, guard, verdict_keys, hrhv2 or
analysis/s12/common: clip thresholds, PRs, baselines, trends, materiality, membership, bands, extrapolation flags and
the P-P2a(ii)/P-P5a counts are recomputed here from the 7 Pythia feature stacks (features only) by plan section 9
(ANALYSIS_PLAN_PYTHIA.md 6de89603) and compared with results/S1_2_layermax_pythia/partB/partB_features_pyt.json.
ViT PR* are taken from the frozen LLaMA-axis Part B file (ede01275...), an independent source for the 17 ViTs.
Reads only; prints one JSON report to stdout.
"""
import hashlib, itertools, json, math, os, sys
import numpy as np

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
FEAT = os.path.join(PROJ, "features", "wit1024_pythia_axis")
PB = os.path.join(PROJ, "results", "S1_2_layermax_pythia", "partB", "partB_features_pyt.json")
LLPB = os.path.join(PROJ, "results", "S1_2_layermax_llama", "partB", "partB_features.json")
SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
SLUG = {s: f"EleutherAI__pythia-{s}" for s in SIZES}
M = 1024
DB = 0.005
BAND = {"prh_clip+l2": (0.001069, 0.012920, 0.0008, 124.5), "l2": (0.001551, 0.006843, 0.0018, 79.6)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def clip_c(st):  # mean over samples of the 0.95 quantile (linear) of |x| over all layers and coordinates
    a = np.abs(np.asarray(st, dtype=np.float64).reshape(st.shape[0], -1))
    return float(np.mean(np.quantile(a, 0.95, axis=1)))


def pr_layer(X, pre, c):
    X = np.array(X, dtype=np.float64)
    if pre == "prh_clip+l2":
        X = np.minimum(np.maximum(X, -c), c)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)  # H X: centred features give H K H = Xc Xc^T
    Kc = Xc @ Xc.T
    return float(np.trace(Kc) ** 2 / np.sum(Kc * Kc))


def spearman_size(v):  # Spearman of ranks 1..7 against v (average ranks)
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    x = np.arange(1, len(v) + 1, dtype=float)
    a, b = x - x.mean(), r - r.mean()
    return float(np.sum(a * b) / math.sqrt(np.sum(a * a) * np.sum(b * b)))


def main():
    F = json.load(open(PB))
    L = json.load(open(LLPB))
    rep = {"partB_features_pyt_sha256": sha(PB), "llama_partB_sha256": sha(LLPB), "per_pre": {}, "mismatches": []}
    vits = F["vits"]
    stacks = {s: np.load(os.path.join(FEAT, SLUG[s] + ".npy"), mmap_mode="r") for s in SIZES}
    clips = {s: clip_c(stacks[s]) for s in SIZES}
    rep["clip_equal_bitwise"] = {s: clips[s] == F["meta"]["clip_thresholds"][SLUG[s]] for s in SIZES}
    rep["clip_max_rel_diff"] = max(abs(clips[s] / F["meta"]["clip_thresholds"][SLUG[s]] - 1) for s in SIZES)
    for pre in ("prh_clip+l2", "l2"):
        Q = F["per_pre"][pre]
        prs = {s: np.array([pr_layer(stacks[s][:, l, :], pre, clips[s] if pre == "prh_clip+l2" else None)
                            for l in range(stacks[s].shape[1])]) for s in SIZES}
        relerr = max(float(np.max(np.abs(prs[s] / np.array(Q["PR"][SLUG[s]]) - 1))) for s in SIZES)
        vpr = {v: float(L["per_pre"][pre]["vit"][v]["PR_star"]) for v in vits}
        vit_rel = max(abs(vpr[v] / Q["PR_star"][v] - 1) for v in vits)
        prstar = np.array([prs[s].max() for s in SIZES])
        ustar = [int(np.argmax(prs[s])) for s in SIZES]
        late = []
        for s in SIZES:
            Ls = prs[s].size
            idx = [u for u in range(Ls) if u / (Ls - 1) >= 0.60 - 1e-12]
            late.append(max(prs[s][u] for u in idx))
        late = np.array(late)
        Bbar = np.array([[math.sqrt(p * vpr[v]) / (M - 1) for v in vits] for p in prstar])
        Blate = np.array([[math.sqrt(p * vpr[v]) / (M - 1) for v in vits] for p in late])
        tr_b, tr_l = spearman_size(prstar), spearman_size(late)
        tr5 = spearman_size(prstar[2:])
        out = {}
        for name, G, tr in (("Bbar", Bbar, tr_b), ("Blate", Blate, tr_l)):
            sB = 1 if tr >= 0 else -1
            med = float(np.median(sB * (G[-1] - G[0])))
            out[name] = {"Tr": tr, "median_end_gap": med, "material": bool(abs(tr) >= 0.75 - 1e-9 and med >= 3 * DB)}
        members = (["H1"] if out["Bbar"]["material"] else []) + ["H2a" if out["Blate"]["material"] else "H2b", "H3", "H4"]
        a, b, h, fmin = BAND[pre]
        band = {s: (a + b / math.sqrt(p) - h, a + b / math.sqrt(p) + h) for s, p in zip(SIZES, prstar)}
        extrap = {s: bool(p < fmin) for s, p in zip(SIZES, prstar)}
        comps = [(i, j, k) for i, j in itertools.combinations(range(7), 2) for k in range(len(vits))]
        n2 = sum(abs(Bbar[j, k] - Bbar[i, k]) >= DB for i, j, k in comps)
        n5 = sum(abs(Blate[j, k] - Blate[i, k]) >= DB for i, j, k in comps)
        need2, need67, need5 = -((-9 * n2) // 10), -((-2 * n2) // 3), -((-2 * n5) // 3)
        neq = int(sum(late[i] == prstar[i] for i in range(7)))
        mine = {"PR_max_rel_err_vs_json": relerr, "vit_PRstar_max_rel_vs_frozen_llama_partB": vit_rel,
                "PR_star": dict(zip(SIZES, prstar.tolist())), "u_star": dict(zip(SIZES, ustar)),
                "PR_late_star": dict(zip(SIZES, late.tolist())), "Tr_Bbar": tr_b, "Tr_Bbar_5size": tr5, "Tr_Blate": tr_l,
                "materiality": out, "members": members, "s": 1 if tr_b >= 0 else -1, "s_late": 1 if tr_l >= 0 else -1,
                "band": band, "extrapolated": extrap, "PP2aii_n": n2, "PP2aii_need": need2, "PP2aii_need67": need67,
                "PP5a_n": n5, "PP5a_need": need5, "n_Blate_equals_Bbar": neq}
        # comparisons with the JSON written by partb_pyt.py
        chk = {
            "PR": relerr <= 1e-10,
            "vit_PRstar": vit_rel <= 1e-14,
            "Tr_Bbar": abs(tr_b - Q["Tr_Bbar"]) <= 1e-12, "Tr_Bbar_5size": abs(tr5 - Q["Tr_Bbar_5size"]) <= 1e-12,
            "Tr_Blate": abs(tr_l - Q["Tr_Blate"]) <= 1e-12,
            "material_Bbar": out["Bbar"]["material"] == Q["material_Bbar"],
            "material_Blate": out["Blate"]["material"] == Q["material_Blate"],
            "median_end_gap_Bbar": abs(out["Bbar"]["median_end_gap"] - Q["materiality"]["Bbar"]["median_end_gap"]) <= 1e-12,
            "median_end_gap_Blate": abs(out["Blate"]["median_end_gap"] - Q["materiality"]["Blate"]["median_end_gap"]) <= 1e-12,
            "members": members == Q["members"], "s": mine["s"] == Q["s"], "s_late": mine["s_late"] == Q["s_late"],
            "band": all(abs(band[s][0] - Q["band"][s]["lo"]) <= 1e-12 and abs(band[s][1] - Q["band"][s]["hi"]) <= 1e-12
                        for s in SIZES),
            "extrapolated": extrap == Q["extrapolated"],
            "u_star": all(ustar[i] == Q["u_star"][SLUG[s]] for i, s in enumerate(SIZES)),
            "PP2aii": (n2, need2, need67) == (Q["qualifying"]["PP2aii_n"], Q["qualifying"]["PP2aii_need"],
                                             Q["qualifying"]["PP2aii_need67"]),
            "PP5a": (n5, need5) == (Q["qualifying"]["PP5a_n"], Q["qualifying"]["PP5a_need"]),
            "n_Blate_equals_Bbar": neq == Q["n_Blate_equals_Bbar"],
            "Bbar_cells": float(np.max(np.abs(Bbar - np.array([[Q["Bbar"][SLUG[s]][v] for v in vits] for s in SIZES])))) <= 1e-12,
            "Blate_cells": float(np.max(np.abs(Blate - np.array([[Q["Blate"][SLUG[s]][v] for v in vits] for s in SIZES])))) <= 1e-12,
        }
        mine["checks"] = chk
        rep["mismatches"] += [f"{pre}:{k}" for k, ok in chk.items() if not ok]
        rep["per_pre"][pre] = mine
    rep["pass"] = (not rep["mismatches"]) and all(rep["clip_equal_bitwise"].values())
    json.dump(rep, sys.stdout, indent=1, default=float)
    print()


if __name__ == "__main__":
    main()
