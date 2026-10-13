import json, os, statistics as st
R = os.path.expanduser("~/hrh_v2_stage1_20261008")
pb = json.load(open(f"{R}/results/S1_2_layermax_llama/partB/partB_features.json"))
ol = json.load(open(f"{R}/results/S1_2_layermax_openllama/partB/partB_features_ol.json"))
for pre in ["prh_clip+l2", "l2"]:
    pp = pb["per_pre"][pre]; B = pp["Bbar"]
    offs, mo, sd = [], [], []
    for lm in B:
        for vit, b in B[lm].items():
            f = f"{R}/results/S1_2_layermax/pairs/{pre}/{lm}__{vit}.json"
            if not os.path.exists(f): f = f"{R}/results/S1_2_layermax_llama/early/pairs/{pre}/{lm}__{vit}.json"
            j = json.load(open(f))["summaries"]["cka_biased"]["full"]
            offs.append(j["c_alpha"]-b); mo.append(j["null_mean"]-b); sd.append(j["null_sd"])
    # add OL 13b from E pairs
    n13 = 0
    try:
        B13 = ol["per_pre"][pre]["Bbar"]["openlm-research__open_llama_13b"] if "Bbar" in ol["per_pre"][pre] else None
    except Exception as e:
        B13 = None
    if isinstance(B13, dict):
        for vit, b in B13.items():
            f = f"{R}/results/S1_2_layermax_openllama/early/pairs/{pre}/openlm-research__open_llama_13b__{vit}.json"
            if os.path.exists(f):
                j = json.load(open(f))["summaries"]["cka_biased"]["full"]
                offs.append(j["c_alpha"]-b); mo.append(j["null_mean"]-b); sd.append(j["null_sd"]); n13 += 1
    print(pre, "n", len(offs), "(13b", n13, ") offset med %.5f range [%.5f, %.5f]" % (st.median(offs), min(offs), max(offs)),
          "nullmean-B range [%.6f, %.6f]" % (min(mo), max(mo)), "null_sd range [%.5f, %.5f]" % (min(sd), max(sd)))
print(list(ol["per_pre"]["prh_clip+l2"].keys())[:20])
