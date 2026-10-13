"""F1(d) static output-root check of the OpenLLaMA axis (ANALYSIS_PLAN_OPENLLAMA.md section 4 F1(d); S0b).

Written at S0b by role oll-axis-extract (2026-10-09); hashed in the CODE FROZEN line. Reads code only (no data).

For every python file of the axis code (analysis/s12_layermax_openllama/**.py except orig_copy/ and pb_orig_copy/,
extract/openllama_axis/*.py except orig_copy/ and superseded_draft_0949Z/):
  1. module-level constants whose name looks like an output or input root (RES, OUT, ROOT, CACHE, DIR, PARTB, ...) are
     printed with their source text (AST, no import, so no gate is triggered);
  2. the configuration modules named by the plan are imported and their roots printed: E `early/lm_core.py`
     (RES, CACHE), I `indep/lmx_common.py` (OUT), and the extraction `ol_axis_common.py` (OUT_DIR, CHECK_DIR, RES_DIR);
  3. `grep -n` equivalent for "results/", "S1_2_layermax", "features/", "os.path.join(" lists every path literal;
  4. every string literal naming a results/ or features/ subtree is classified:
       own write roots   results/S1_2_layermax_openllama/{early,indep,review,forecast,joint,cache,extraction,
                         feasibility,logs,partB} and features/wit1024_openllama_axis
       read-only trees   results/S1_2_layermax/, results/S1_2_layermax_llama/, results/S1_2/, features/wit1024/,
                         features/wit1024_llama_axis/
     and any write call (open(..., "w"/"a"), np.save*, json.dump, write_json, mkdir, os.replace, shutil) on a line that
     names a read-only tree is reported as a VIOLATION.
Output: JSON on stdout and results/S1_2_layermax_openllama/logs/f1d_outroots.json. Exit 0 if no violation, else 4.
usage: venv/bin/python analysis/s12_layermax_openllama/f1d_outroots.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys

PROJ = os.path.realpath(os.path.expanduser("~/hrh_v2_stage1_20261008"))
AXIS = os.path.join(PROJ, "analysis", "s12_layermax_openllama")
EXTR = os.path.join(PROJ, "extract", "openllama_axis")
OUT = os.path.join(PROJ, "results", "S1_2_layermax_openllama", "logs", "f1d_outroots.json")
SKIP_DIRS = {"orig_copy", "pb_orig_copy", "superseded_draft_0949Z", "__pycache__", "p0", "run"}
ROOT_NAME = re.compile(r"(RES|OUT|ROOT|CACHE|DIR|PARTB|FEAT|LOG|PAIRS|BASE|DEST|TARGET|PATH)", re.I)
PATH_LIT = re.compile(r"results/|S1_2_layermax|features/|os\.path\.join\(|wit1024")
OWN = ("results/S1_2_layermax_openllama", "features/wit1024_openllama_axis")
READ_ONLY = ("results/S1_2_layermax/", "results/S1_2_layermax_llama", "results/S1_2/", "features/wit1024/",
             "features/wit1024_llama_axis", '"S1_2_layermax"', "'S1_2_layermax'", '"S1_2_layermax_llama"',
             "'S1_2_layermax_llama'", '"wit1024"', "'wit1024'", '"wit1024_llama_axis"')
WRITE_CALL = re.compile(r"open\([^)]*[\"'](w|a|wb|ab|x)[\"']|np\.save|savez|json\.dump\(|write_json|mkdir|makedirs|"
                        r"os\.replace|shutil\.(copy|move)|\.write_text|\.write_bytes|to_parquet|chmod")


def files():
    out = []
    for base in (AXIS, EXTR):
        for d, subdirs, fs in os.walk(base):
            subdirs[:] = [s for s in subdirs if s not in SKIP_DIRS]
            out += [os.path.join(d, f) for f in sorted(fs) if f.endswith(".py")]
    return sorted(out)


def module_roots(path: str) -> list[dict]:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    rows = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and ROOT_NAME.search(t.id):
                    rows.append(dict(name=t.id, line=node.lineno, src=ast.get_source_segment(src, node.value)))
    return rows


def literal_lines(path: str) -> list[dict]:
    rows = []
    for i, ln in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        if PATH_LIT.search(ln) and not ln.lstrip().startswith("#"):
            ro = [t for t in READ_ONLY if t in ln]
            own = [t for t in OWN if t in ln]
            wr = bool(WRITE_CALL.search(ln))
            rows.append(dict(line=i, text=ln.strip()[:220], read_only_tree=ro, own_tree=own, write_call=wr,
                             violation=bool(ro and wr and not own)))
    return rows


_IMPORT_SNIPPET = r"""
import importlib.util, json, os, sys
path, names = sys.argv[1], sys.argv[2].split(",")
sys.path.insert(0, os.path.dirname(path))
spec = importlib.util.spec_from_file_location("f1d_mod", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("F1D_JSON " + json.dumps({n: str(getattr(mod, n, None)) for n in names}))
"""


def import_roots() -> dict:
    """Each configuration module is imported in its own subprocess (S0b fix: in one process, analysis/s12/common.py
    imported by lm_core shadows extract/common.py needed by ol_axis_common). An import that fails is a violation."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("OL_S0B_TEST", "OL_SENS_E2")}
    out = {}
    for label, path, names in (("E lm_core", os.path.join(AXIS, "early", "lm_core.py"),
                                ("RES", "CACHE", "SHARED_CACHE", "PARTB")),
                               ("I lmx_common", os.path.join(AXIS, "indep", "lmx_common.py"), ("OUT", "OUT_SENS")),
                               ("extract ol_axis_common", os.path.join(EXTR, "ol_axis_common.py"),
                                ("OUT_DIR", "CHECK_DIR", "E3_DIR", "E5_DIR", "RES_DIR"))):
        if not os.path.exists(path):
            out[label] = dict(error="missing")
            continue
        p = subprocess.run([sys.executable, "-c", _IMPORT_SNIPPET, path, ",".join(names)], cwd=PROJ, env=env,
                           capture_output=True, text=True, timeout=300)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("F1D_JSON ")]
        if p.returncode != 0 or not line:
            out[label] = dict(error=f"import exit {p.returncode}: {p.stderr.strip()[-300:]}")
        else:
            out[label] = json.loads(line[-1][len("F1D_JSON "):])
    return out


def main():
    rep = dict(files={}, imported=import_roots())
    viol = []
    for f in files():
        rel = os.path.relpath(f, PROJ)
        lits = literal_lines(f)
        rep["files"][rel] = dict(module_roots=module_roots(f), path_literals=lits)
        viol += [dict(file=rel, **r) for r in lits if r["violation"]]
    # imported roots must be own write roots
    for label, d in rep["imported"].items():
        if "error" in d:  # S0b fix: a configuration module that cannot be imported cannot be checked: fail closed
            viol.append(dict(file=label, line=0, text=f"import failed: {d['error']}", violation=True))
            continue
        for n, v in d.items():
            if v and v != "None" and "S1_2_layermax_openllama" not in v and "wit1024_openllama_axis" \
                    not in v and n not in ("PARTB",):
                viol.append(dict(file=label, line=0, text=f"{n} = {v}", violation=True))
    rep["violations"] = viol
    rep["pass"] = not viol
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(dict(n_files=len(rep["files"]), imported=rep["imported"], n_violations=len(viol),
                          violations=viol[:50], pass_=rep["pass"]), indent=1))
    sys.exit(0 if rep["pass"] else 4)


if __name__ == "__main__":
    main()
