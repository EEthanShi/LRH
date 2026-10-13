import json, os, statistics as st
R = os.path.expanduser("~/hrh_v2_stage1_20261008")
pb = json.load(open(f"{R}/results/S1_2_layermax_llama/partB/partB_features.json"))
for pre in ["prh_clip+l2", "l2"]:
    pp = pb["per_pre"][pre]
    B = pp["Bbar"]
    print("==", pre)
    for lm in B:
        offs, sds, mo, nlp = [], [], [], None
        for vit, b in B[lm].items():
            f = f"{R}/results/S1_2_layermax/pairs/{pre}/{lm}__{vit}.json"
            if not os.path.exists(f):
                f = f"{R}/results/S1_2_layermax_llama/early/pairs/{pre}/{lm}__{vit}.json"
            if not os.path.exists(f): continue
            j = json.load(open(f))["summaries"]["cka_biased"]["full"]
            offs.append(j["c_alpha"] - b); sds.append(j["null_sd"]); nlp = j["n_layer_pairs"]; mo.append(j["null_mean"] - b)
        L = pp["lm"][lm]["L"]; prs = pp["lm"][lm]["PR_star"]
        if offs:
            print(f"{lm:34s} Lst={L:2d} PR*={prs:6.1f} n={len(offs):2d} off med {st.median(offs):.5f} [{min(offs):.5f},{max(offs):.5f}] null_sd med {st.median(sds):.5f} [{min(sds):.5f},{max(sds):.5f}] nullmean-B med {st.median(mo):+.6f}")
