#!/usr/bin/env python3
"""s23-review-A2: order, hash and bookkeeping checks for A2/A3/A4 (stdlib + hashlib only; imports no s23/hrhv2/A1-A4
code). Reads STATUS, file birth times (births_all.tsv written by `stat -c '%W %Y %X %a %s %n'`), manifests and
ledgers. Writes order_check_A2.json next to this file."""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

HOME = os.path.expanduser("~")
S23 = os.path.join(HOME, "hrh_v2_stage2_20261008", "s23")
OUT = os.path.join(S23, "results", "stageA", "review_A2")
STATUS = os.path.join(HOME, "hrh_v2_stage1_20261008", "STATUS.md")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


res = {"checks": [], "info": {}}


def check(name, ok, **kw):
    res["checks"].append({"name": name, "pass": bool(ok), **kw})
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:600], flush=True)


# ------------------------------------------------------------------ STATUS lines
lines = open(STATUS, encoding="utf-8").read().split("\n")
lines = [l for l in lines]
def find_line(pat):
    hits = [(i + 1, l) for i, l in enumerate(lines) if re.search(pat, l)]
    return hits
L = {}
for name, pat in [("review_resume", r"\[s23-review-resume\]"), ("A2_frozen", r"\[s23-amend-A2\].*FROZEN"),
                  ("A2_start", r"\[s23-run-A2\] A2 attempt \S+ START"), ("A2_complete", r"\[s23-run-A2\].*main phase complete"),
                  ("A2_end", r"\[s23-run-A2\].*END rc=5"), ("A3_frozen", r"\[s23-amend-A3\].*FROZEN"),
                  ("A3_start", r"\[s23-run-A3\].*START"), ("nmatch", r"\[s23-run\] S2.3 Stage A N_match written"),
                  ("A3_end", r"\[s23-run-A3\].*END rc=5"), ("A4_frozen", r"\[s23-amend-A4\].*FROZEN"),
                  ("A4_start", r"\[s23-run-A4\].*START"), ("analysis_done", r"\[s23-run\] S2.3 Stage A analysis done"),
                  ("A4_end", r"\[s23-run-A4\].*END rc=0"), ("role_end", r"\[s23-run-A2\] ROLE END"),
                  ("A1_stop", r"\[s23-run-resume\].*STOPPED")]:
    h = find_line(pat)
    L[name] = {"n": len(h), "line": h[0][0] if h else None, "utc": h[0][1][:20] if h else None}
res["status_lines"] = L
for k, v in L.items():
    check(f"STATUS has exactly one '{k}' line", v["n"] == 1, line=v["line"], utc=v["utc"])
T = {k: ts(v["utc"]) for k, v in L.items() if v["utc"]}
author_decision = ts("2026-10-10T08:05:00Z")   # ~09:05 BST, in chat (approximate, minute resolution)
order = ["A1_stop", "review_resume", "A2_frozen", "A2_start", "A2_complete", "A2_end", "A3_frozen", "A3_start",
         "nmatch", "A3_end", "A4_frozen", "A4_start", "analysis_done", "A4_end", "role_end"]
check("STATUS event order A1 stop < review < A2 freeze < A2 start < completion < A2 crash < A3 freeze < A3 start < "
      "N_match < A3 crash < A4 freeze < A4 start < analysis done < A4 end < role end",
      all(T[a] <= T[b] for a, b in zip(order, order[1:])), times={k: L[k]["utc"] for k in order})
check("A2 freeze after the author decision (~08:05Z)", T["A2_frozen"] >= author_decision, a2=L["A2_frozen"]["utc"])

# ------------------------------------------------------------------ births
B = {}
for row in open(os.path.join(OUT, "births_all.tsv"), encoding="utf-8"):
    w, y, x, mode, size, name = row.rstrip("\n").split("\t", 5)
    B[name[2:] if name.startswith("./") else name] = {"birth": int(w), "mtime": int(y), "atime": int(x),
                                                      "mode": mode, "size": int(size)}
def birth(p):
    return B[p]["birth"]

# A2 / A3 / A4 manifest files
manif = {}
for tag, mf in [("A2", "resume/A2_MANIFEST.sha256"), ("A3", "resume/A3_MANIFEST.sha256"), ("A4", "resume/A4_MANIFEST.sha256")]:
    ents = []
    for row in open(os.path.join(S23, mf)):
        if row.strip() and not row.startswith("#"):
            h, p = row.split()
            for pre in ("S23/", "ROOT/", "./"):
                if p.startswith(pre):
                    p = p[len(pre):]
            ents.append((h, p))
    manif[tag] = ents
    h_mf = sha(os.path.join(S23, mf))
    fl = [l for l in lines if f"[s23-amend-{tag}]" in l and "FROZEN" in l][0]
    check(f"{tag} manifest sha256 equals the one in its FROZEN line", h_mf in fl, sha256=h_mf)
    bad = [p for h, p in ents if sha(os.path.join(S23, p)) != h or h not in fl]
    check(f"{tag} manifest: every entry hash-equal now and listed in the FROZEN line", not bad, n=len(ents), bad=bad)
    ro = [p for h, p in ents if B.get(p, {}).get("mode") != "444"] + ([mf] if B[mf]["mode"] != "444" else [])
    check(f"{tag} files chmod 444", not ro, not_444=ro)
    births = {p: iso(birth(p)) for _, p in ents if p in B}
    births[mf] = iso(birth(mf))
    lo = {"A2": max(T["review_resume"], author_decision - 3600), "A3": T["A2_end"], "A4": T["A3_end"]}[tag]
    hi = T[f"{tag}_frozen"]
    late = [p for p in births if not (birth(p) <= hi + 1)]
    early = [p for p in births if birth(p) < lo]
    res["info"][f"{tag}_births"] = births
    check(f"{tag} files born before their FROZEN line", not late, late=late, frozen=L[f"{tag}_frozen"]["utc"])
    check(f"{tag} files born after the event that required them", not early, early=early,
          after={"A2": "review 07:54:56Z / author decision", "A3": "A2 nmatch crash 09:10:16Z", "A4": "A3 main crash 09:25:17Z"}[tag],
          first=min(births.values()), last=max(births.values()))

# runs: births of the 156 A2 items and no file between the A1 stop and the A2 start
run_files = {p: v for p, v in B.items() if p.startswith("runs/")}
a1_last = ts("2026-10-10T07:37:05Z")
gap = [p for p, v in run_files.items() if a1_last + 1 < v["birth"] < T["A2_start"]]
check("no runs/ file born between the A1 stop (07:37:05Z) and the A2 start", not gap, n=len(gap), examples=gap[:5])
a2_files = sorted([p for p, v in run_files.items() if v["birth"] >= T["A2_start"]])
a2_ids = sorted({os.path.basename(p).split(".")[0] for p in a2_files})
check("files born after the A2 start are exactly 156 Rm_Ap_l22 items x 3 files", len(a2_ids) == 156 and len(a2_files) == 468
      and all(i.startswith("Rm_Ap_l22_r") for i in a2_ids), n_items=len(a2_ids), n_files=len(a2_files))
if a2_files:
    bmin = min(run_files[p]["birth"] for p in a2_files); bmax = max(run_files[p]["birth"] for p in a2_files)
    check("A2 run files born between the A2 start and 'main phase complete'", T["A2_start"] <= bmin and bmax <= T["A2_complete"] + 1,
          first=iso(bmin), last=iso(bmax))
reps = sorted(int(i.split("_r")[1]) for i in a2_ids)
exp = sorted(list(range(12, 191, 2)) + list(range(61, 192, 2)))
check("A2 items are the planned even reps r012-r190 and odd reps r061-r191", reps == exp, n=len(reps))
partial = [p for p in run_files if not re.search(r"\.(npy|json|done)$", p)]
check("no partial/tmp file under runs/", not partial, examples=partial[:5])
ndone = sum(1 for p in run_files if p.endswith(".done") and "/l" in p and not p.startswith("runs/_partial"))
res["info"]["n_done_markers_runs"] = ndone

# results/stageA: what existed at the A2 freeze; analysis outputs after completion
rs = {p: v for p, v in B.items() if p.startswith("results/stageA/") and not p.startswith("results/stageA/review_")}
pre_a2 = sorted(p for p, v in rs.items() if v["birth"] < T["A2_frozen"])
stat_like = [p for p in pre_a2 if not (p.startswith("results/stageA/cache/pilot_") or os.path.basename(p) in
             ("A1_g4_inputs.json", "eps_arm0.json", "gates_sampled.json", "g4_decision.json"))]
check("at the A2 freeze results/stageA held only gates outputs and pilot-pair caches (no main-run statistic)",
      not stat_like, n_files=len(pre_a2), other=stat_like)
cache = {p: v for p, v in rs.items() if p.startswith("results/stageA/cache/") or p.startswith("results/stageA/_crashed")}
main_cache = {p: v for p, v in cache.items() if "/pilot_" not in p}
first_main_cache = min(v["birth"] for v in main_cache.values())
check("first main-run cache file born after all 3,456 main items were complete (09:09:34Z)", first_main_cache >= T["A2_complete"],
      first=iso(first_main_cache))
for name, p, lo in [("N_match.json", "results/stageA/N_match.json", T["A3_frozen"]),
                    ("stageA_results.json", "results/stageA/stageA_results.json", T["A4_frozen"]),
                    ("REPORT_stageA.md", "results/stageA/REPORT_stageA.md", T["A4_frozen"])]:
    check(f"{name} born after the amendment that let it be written", rs[p]["birth"] >= lo, born=iso(rs[p]["birth"]), mode=rs[p]["mode"])
# which cache files were born in which window
win = {"A2_nmatch_crash": (T["A2_complete"], T["A2_end"] + 1), "A3_nmatch": (T["A3_start"], T["nmatch"] + 1),
       "A3_main_crash": (T["nmatch"] + 1, T["A3_end"] + 1), "A4_main": (T["A4_start"], T["analysis_done"] + 1)}
res["info"]["cache_births_by_window"] = {k: sorted(os.path.basename(p) for p, v in main_cache.items() if a <= v["birth"] <= b)
                                         for k, (a, b) in win.items()}
outside = [p for p, v in main_cache.items() if not any(a <= v["birth"] <= b for a, b in win.values())]
check("every main-run cache file was born inside an analysis window (A2 nmatch, A3 nmatch, A3 main, A4 main)", not outside,
      outside=outside[:10], counts={k: len(v) for k, v in res["info"]["cache_births_by_window"].items()})

# A3: crashed cache byte-equal to the regenerated one
cc = []
for row in open(os.path.join(S23, "resume/A3_crashed_cache.sha256")):
    if row.strip() and not row.startswith("#"):
        h, p = row.split()
        cc.append((h, os.path.basename(p)))
eq = []
for h, b in cc:
    moved = os.path.join(S23, "results/stageA/_crashed_cache_A2_20261010T083303Z", b)
    regen = os.path.join(S23, "results/stageA/cache", b)
    eq.append({"file": b, "moved_ok": sha(moved) == h, "regen_equal": sha(regen) == h})
check("A3: 7 crashed-attempt cache files kept unchanged and byte-equal to the regenerated ones",
      len(eq) == 7 and all(e["moved_ok"] and e["regen_equal"] for e in eq), files=eq)

# A4: cache_before listing still valid (nothing rewritten) and what A4 added
cb = {}
for row in open(os.path.join(S23, "resume/A4_attempts/20261010T093132Z/cache_before.sha256")):
    if row.strip() and not row.startswith("#"):
        h, p = row.split()
        cb[p] = h
changed = []
for p, h in cb.items():
    q = os.path.join(S23, "results/stageA", p)
    if not os.path.exists(q) or sha(q) != h:
        changed.append(p)
check("A4: the 430 cache files listed before the A4 run are unchanged", len(cb) == 430 and not changed, n=len(cb), changed=changed[:5])

# A4: the built module differs from frozen analyze_A.py only on line 705 (static re-derivation)
src = open(os.path.join(S23, "src/analyze_A.py"), encoding="utf-8").read()
old = 'and 1 / 1.5 <= pis[r_]["rho"] / mcr[r_]["rho"] <= 1.5)} for r_ in ORDER}'
new = 'and _a4_within_15(pis[r_]["rho"], mcr[r_]["rho"]))} for r_ in ORDER}'
a4 = open(os.path.join(S23, "resume/a4_analyze.py"), encoding="utf-8").read()
built = src.replace(old, new)
dl = [i + 1 for i, (a, b) in enumerate(zip(src.split("\n"), built.split("\n"))) if a != b]
check("A4: replacing the frozen expression changes exactly line 705; a4_analyze.py carries both strings",
      src.count(old) == 1 and dl == [705] and old in a4 and new in a4, changed_lines=dl)

# outputs: hashes equal the STATUS lines
for p in ("results/stageA/N_match.json", "results/stageA/stageA_results.json", "results/stageA/REPORT_stageA.md"):
    h = sha(os.path.join(S23, p))
    check(f"{os.path.basename(p)} sha256 matches STATUS", any(h in l for l in lines), sha256=h)

# frozen code and data hashes
code = []
for row in open(os.path.join(S23, "prereg/CODE_AT_PREREG.sha256")):
    if row.strip() and not row.startswith("#"):
        h, p = row.split()
        base = os.path.join(HOME, p) if p.startswith("hrh_v2") else os.path.join(S23, p)
        code.append((h, p, os.path.exists(base) and sha(base) == h))
check("37 frozen code files hash-equal to CODE_AT_PREREG.sha256", len(code) == 37 and all(c[2] for c in code),
      bad=[c[1] for c in code if not c[2]])
srcfiles = sorted(os.path.relpath(os.path.join(d, f), os.path.join(S23, "src")) for d, _, fs in os.walk(os.path.join(S23, "src"))
                  for f in fs if "__pycache__" not in d)
check("src/ holds exactly the 21 frozen files (pycache excluded)", len(srcfiles) == 21, files=srcfiles)
data = {"prereg/PREREG_S2_3_v1.0.md": "7a93b7a9519b505ee887467baab7aa724d2a0ee05635a2afdc363da280e6b5d6",
        "prereg/CODE_AT_PREREG.sha256": "f9560f0a9d3d2bc44c8407d1a9d3d440c0e8a0960765c67cc1709c7c039cf546",
        "results/prestep/frozen_predictions.json": "44d45c418da1f505bfec0c7500e1a5c2d3b5504b7d776760a0498a8b85bfb4cb",
        "results/prestep/prestep_arrays.npz": "7c560f00299c5532f687af0efd9b230f0f1f1782182e6528c76a281c73110429",
        "plans/stageA_plan.json": "dfeb2c96b80ea5cddeec5e1331206ce40e88b260b716111c3bb661ae9a17980b",
        "results/stageA/eps_arm0.json": "9d947fb2bc8c91d2564bacd2d0f1faf54ab1f54002f833668f7cb855b0a84d1e",
        "results/stageA/g4_decision.json": "2222ce763eb668368649628e37ebeef5a4f94e9b0694d32d7c0ae285c06e7d2f",
        "prereg/AMENDMENT_A1_resume.md": "8a858f5394c4be17a5162374992585f405a9cfc75fff8c38074860da9761948d"}
check("frozen data/prereg/gates files unchanged", all(sha(os.path.join(S23, p)) == h for p, h in data.items()),
      bad=[p for p, h in data.items() if sha(os.path.join(S23, p)) != h])

# GPU ledger
tot, a2s = 0.0, 0.0
for f in sorted(os.listdir(os.path.join(S23, "logs/gpu_ledger"))):
    if f.endswith(".tsv"):
        for row in open(os.path.join(S23, "logs/gpu_ledger", f)):
            parts = row.rstrip("\n").split("\t")
            try:
                s = float(parts[-1]) if len(parts) > 1 else None
            except ValueError:
                s = None
            if s is None:
                for x in parts[1:]:
                    try:
                        s = float(x); break
                    except ValueError:
                        pass
            if s is None:
                continue
            tot += s
            if f.endswith("_main.tsv") and parts[0] >= "2026-10-10T08:33":
                a2s += s
res["info"]["ledger_total_s"] = tot
res["info"]["ledger_total_h"] = tot / 3600
res["info"]["ledger_A2_attempt_s"] = a2s
check("GPU ledger total = 12.7063 h and the A2 attempt adds 4214.8 s", abs(tot / 3600 - 12.7063) < 5e-4 and abs(a2s - 4214.806) < 1.0,
      total_h=round(tot / 3600, 5), a2_s=round(a2s, 3))

res["n_fail"] = sum(1 for c in res["checks"] if not c["pass"])
with open(os.path.join(OUT, "order_check_A2.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"order checks: {len(res['checks'])} run, {res['n_fail']} failed")
