#!/usr/bin/env python
"""OpenLLaMA axis, role oll-review: freeze-order audit (plan ANALYSIS_PLAN_OPENLLAMA.md 4f18d710, sections 4 F1,
9 B7, 11 steps). NEW read-only helper of the review role (not in the S0b manifest). It reads STATUS.md, the GPU
ledger, cross_order.json and file times (statx birth via coreutils `stat -c %W`, mtime via %Y); it opens no score.
Writes results/S1_2_layermax_openllama/review/role/order_audit.json.

usage: venv/bin/python -I analysis/s12_layermax_openllama/review_role/order_audit.py
exit 0 iff every ordering check passes.
"""
import calendar
import glob
import json
import os
import re
import subprocess
import sys
import time

PROJ = os.path.expanduser("~/hrh_v2_stage1_20261008")
RES = os.path.join(PROJ, "results", "S1_2_layermax_openllama")
AX = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
HF = os.path.expanduser("~/.cache/huggingface/hub/models--openlm-research--open_llama_13b")
OUT = os.path.join(RES, "review", "role")


def ep(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def utc(e):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(e)) if e is not None else None


def stat_many(paths):
    out = {}
    paths = list(paths)
    for i in range(0, len(paths), 200):
        chunk = paths[i:i + 200]
        p = subprocess.run(["stat", "-c", "%W %Y %n"] + chunk, capture_output=True, text=True)
        for line in p.stdout.splitlines():
            w, y, n = line.split(" ", 2)
            w = int(w)
            out[n] = {"birth": (w if w > 0 else None), "mtime": int(y)}
    return out


def walk(root):
    fs = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            fs.append(os.path.join(dp, f))
    return fs


def main():
    lines = open(os.path.join(PROJ, "STATUS.md"), encoding="utf-8", errors="replace").read().splitlines()
    ev = []
    for ln in lines:
        m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) \[(oll-axis-[a-z-]+)\] (.*)$", ln)
        if m:
            ev.append({"utc": m.group(1), "tag": m.group(2), "head": m.group(3)[:110]})

    def first(tag, pat):
        for e in ev:
            if e["tag"] == tag and pat in e["head"]:
                return ep(e["utc"])
        return None

    T = {
        "P0": first("oll-axis-freeze", "P0 LLaMA-axis bookkeeping DONE"),
        "S0a_partA_frozen": first("oll-axis-freeze", "S0a PART A FROZEN"),
        "S0b_code_frozen": first("oll-axis-extract", "CODE FROZEN"),
        "S1_registry": first("oll-axis-extract", "S1 registry"),
        "S2_E2_3b": first("oll-axis-extract", "E2 open_llama_3b"),
        "S2_E2_7b": first("oll-axis-extract", "E2 open_llama_7b"),
        "S3_download": first("oll-axis-extract", "S3 download"),
        "S4_extraction": first("oll-axis-extract", "S4 extraction"),
        "extraction_done": first("oll-axis-extract", "EXTRACTION DONE"),
        "predict_start": first("oll-axis-predict", "started"),
        "partB_frozen": first("oll-axis-predict", "Part B FROZEN"),
        "E_start": first("oll-axis-run-early", "started"),
        "I_start": first("oll-axis-run-indep", "started"),
        "E_VO5_verdicts_pass": first("oll-axis-run-early", "V-O5 VERDICT PART"),
        "I_final_logged": first("oll-axis-run-indep", "S8 implementation I DONE"),
        "I_cross_done": first("oll-axis-run-indep", "Cross-implementation comparison"),
        "E_report": first("oll-axis-run-early", "REPORT (E side)"),
    }
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    seq = ["P0", "S0a_partA_frozen", "S0b_code_frozen", "S1_registry", "S2_E2_3b", "S2_E2_7b", "S3_download",
           "S4_extraction", "extraction_done", "predict_start", "partB_frozen", "E_start"]
    vals = [T[k] for k in seq]
    chk("STATUS step order P0 < S0a < S0b CODE FROZEN < S1 < S2 < S3 < S4 < Part B start < Part B FROZEN < runs",
        all(v is not None for v in vals) and all(a < b for a, b in zip(vals, vals[1:])),
        {k: utc(T[k]) for k in seq})
    chk("I started after Part B FROZEN", T["I_start"] and T["I_start"] > T["partB_frozen"], utc(T["I_start"]))

    # plan file
    plan = os.path.join(AX, "ANALYSIS_PLAN_OPENLLAMA.md")
    s = stat_many([plan])[plan]
    chk("Part A plan last modified at or before the S0a freeze line", s["mtime"] <= T["S0a_partA_frozen"],
        {"birth": utc(s["birth"]), "mtime": utc(s["mtime"]), "freeze": utc(T["S0a_partA_frozen"])})

    # S0b code files
    man = json.load(open(os.path.join(AX, "S0B_CODE_MANIFEST.json")))
    cpaths = [os.path.join(PROJ, f["path"]) for f in man["files"]]
    cs = stat_many(cpaths)
    mx = max(v["mtime"] for v in cs.values())
    chk("all 106 S0b code files last modified at or before CODE FROZEN", len(cs) == 106 and mx <= T["S0b_code_frozen"],
        {"n": len(cs), "latest_mtime": utc(mx), "code_frozen": utc(T["S0b_code_frozen"])})

    # 13B bytes: HF cache and features
    hf = [HF] + walk(HF)
    hs = stat_many(hf)
    hb = [v["birth"] for v in hs.values() if v["birth"]]
    chk("no open_llama_13b HF-cache file born before CODE FROZEN (all 13B weights downloaded after code freeze)",
        len(hb) == len(hs) and min(hb) > T["S0b_code_frozen"],
        {"n": len(hs), "earliest_birth": utc(min(hb)) if hb else None, "latest_mtime": utc(max(v["mtime"] for v in hs.values()))})
    feat = glob.glob(os.path.join(PROJ, "features", "wit1024_openllama_axis", "openlm-research__open_llama_13b*"))
    fs = stat_many(feat)
    fb = [v["birth"] for v in fs.values()]
    chk("13B feature files born after the download line and before EXTRACTION DONE",
        fs and min(fb) >= T["S3_download"] - 300 and max(v["mtime"] for v in fs.values()) <= T["extraction_done"],
        {os.path.basename(k): {"birth": utc(v["birth"]), "mtime": utc(v["mtime"])} for k, v in fs.items()})

    # Part B files
    pb = glob.glob(os.path.join(RES, "partB", "*")) + [os.path.join(AX, "PARTB_PREDICTIONS_OL.md")]
    ps = stat_many(pb)
    chk("Part B files born after Part B start and last modified at or before Part B FROZEN",
        min(v["birth"] for v in ps.values()) >= T["predict_start"] - 60 and max(v["mtime"] for v in ps.values()) <= T["partB_frozen"],
        {os.path.relpath(k, PROJ): {"birth": utc(v["birth"]), "mtime": utc(v["mtime"])} for k, v in ps.items()})

    # every file of the score trees born after Part B FROZEN (independent of guard.py audit)
    trees = ["cache", "early", "indep", "review", "forecast", "joint"]
    allf = []
    for t in trees:
        allf += walk(os.path.join(RES, t))
    ss = stat_many(allf)
    before = [k for k, v in ss.items() if (v["birth"] is None or v["birth"] <= T["partB_frozen"])]
    mb = min((v["birth"], k) for k, v in ss.items() if v["birth"])
    chk("every file under cache/early/indep/review/forecast/joint born after Part B FROZEN",
        not before, {"n": len(ss), "n_before_or_unknown": len(before), "earliest": [utc(mb[0]), os.path.relpath(mb[1], PROJ)]})

    # no 13B cross-model file anywhere in the project born before Part B FROZEN
    p = subprocess.run(["find", PROJ, "-name", "*open_llama_13b__*", "-not", "-path", "*/venv/*"], capture_output=True, text=True)
    hits = [h for h in p.stdout.splitlines() if h]
    hs2 = stat_many(hits) if hits else {}
    early13 = [k for k, v in hs2.items() if v["birth"] is None or v["birth"] <= T["partB_frozen"]]
    chk("no *open_llama_13b__* file in the project born at or before Part B FROZEN", not early13,
        {"n_hits": len(hits), "n_early": len(early13),
         "earliest": utc(min(v["birth"] for v in hs2.values())) if hs2 else None})

    # 13B pair files of E and I
    pairs13 = [k for k in ss if "open_llama_13b__" in os.path.basename(k)]
    chk("13B pair files of E and I born after Part B FROZEN and after the run roles started",
        pairs13 and min(ss[k]["birth"] for k in pairs13) > T["partB_frozen"],
        {"n": len(pairs13), "earliest": utc(min(ss[k]["birth"] for k in pairs13))})

    # independence of I from E: I final files older than the first opening of E
    co = json.load(open(os.path.join(RES, "indep", "cross_order.json")))
    eopen = ep(co["E_first_opened_after_utc"])
    ifin = [os.path.join(RES, "indep", f) for f in ("summary.json", "verdicts_ol.json", "REPORT.pre_cross.md")]
    ifo = [os.path.join(RES, f) for f in ("forecast/I/verdicts_forecast.json", "joint/I/verdicts_joint.json")]
    st_i = stat_many(ifin + ifo)
    chk("I summary, verdicts, forecast, joint and pre-cross report final before I first opened E output",
        max(v["mtime"] for v in st_i.values()) < eopen,
        {**{os.path.relpath(k, RES): utc(v["mtime"]) for k, v in st_i.items()}, "E_first_opened": utc(eopen)})
    # E: summary/verdicts written before E's V-O5 pair part opened I files (E log: 19:56:02Z)
    efin = [os.path.join(RES, "early", f) for f in ("summary.json", "verdicts_ol.json")] + \
           [os.path.join(RES, f) for f in ("forecast/E/verdicts_forecast.json", "joint/E/verdicts_joint.json")]
    st_e = stat_many(efin)
    vo5p = json.load(open(os.path.join(RES, "early", "validation", "VO5_pairs.json")))
    chk("E summary, verdicts, forecast, joint written before the E-side V-O5 opened I pair files (19:56:02Z per E line)",
        max(v["mtime"] for v in st_e.values()) <= ep("2026-10-09T19:56:02Z"),
        {os.path.relpath(k, RES): utc(v["mtime"]) for k, v in st_e.items()})
    # V-O5 pass before the reports were written
    rep = stat_many([os.path.join(RES, "early", "REPORT.md"), os.path.join(RES, "indep", "REPORT.md")])
    chk("E and I REPORT.md written after the V-O5 verdict part passed (19:59:22Z)",
        min(v["mtime"] for v in rep.values()) >= T["E_VO5_verdicts_pass"],
        {os.path.relpath(k, RES): utc(v["mtime"]) for k, v in rep.items()})

    # GPU ledger
    led = [l.split("\t") for l in open(os.path.join(RES, "logs", "gpu_ledger.tsv")).read().splitlines()[1:] if l.strip()]
    tot = sum(float(r[4]) for r in led)
    pre = [r for r in led if ep(r[1]) < T["S0b_code_frozen"]]
    chk("GPU ledger total <= 60 min; GPU steps before CODE FROZEN are S0b stand-in tests only",
        tot <= 60 and all(r[0] in ("lead_smoke_extract", "oltestI_gpu_smoke", "E_gpu_pre", "G_E_validate", "G_E_main",
                                   "G_I_prh", "G_I_l2") for r in pre),
        {"total_min": round(tot, 3), "n_steps": len(led), "pre_freeze_steps": [r[0] for r in pre],
         "pre_freeze_min": round(sum(float(r[4]) for r in pre), 3)})

    res = {"role": "oll-review", "what": "freeze-order audit", "utc": utc(time.time()),
           "status_times": {k: utc(v) for k, v in T.items()}, "checks": checks,
           "pass": all(c["pass"] for c in checks), "oll_axis_status_lines": len(ev)}
    dry = "--dry" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, "order_audit.json")
    if dry:
        fn = os.devnull
    elif os.path.exists(fn):
        sys.exit("refusing to overwrite " + fn)
    with open(fn, "w") as f:
        json.dump(res, f, indent=1)
    if not dry:
        os.chmod(fn, 0o444)
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + c["check"])
        print("     ", json.dumps(c["detail"])[:600])
    print("ALL PASS" if res["pass"] else "NOT ALL PASS")
    print("output", os.path.relpath(fn, PROJ))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
