"""Frozen-code guard for S2.2 v1.1 (PREREG_S2_2_v1.1.md section 14, step 5).

verify_frozen(): every file listed in the manifest results/S2_2_v11/freeze/CODE_AT_PREREG_v11.sha256 must still
have its recorded sha256, and STATUS.md must contain the [s22v11-freeze] 'PREREG v1.1 + CODE FROZEN' line quoting
the manifest's own sha256. (The manifest lives under results/S2_2_v11/ because prereg/ receives only
PREREG_S2_2_v1.1.md in v1.1; resolution R-PATH of the prereg.)
Engineering (eng_v11), pilots and main runs (train_v11 preflight), gates and the eps freeze (gates_v11), the final
analysis (analyze_v11) and the pipeline driver (pipeline_v11) call it and refuse to run otherwise.
"""
from __future__ import annotations

import hashlib
import os

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
MANIFEST = os.path.join(ROOT, "results", "S2_2_v11", "freeze", "CODE_AT_PREREG_v11.sha256")
STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")
FREEZE_TAG = "[s22v11-freeze] PREREG v1.1 + CODE FROZEN"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def verify_frozen():
    if not os.path.exists(MANIFEST):
        return False, "results/S2_2_v11/freeze/CODE_AT_PREREG_v11.sha256 absent (code not frozen)"
    msha = sha(MANIFEST)
    bad = []
    for line in open(MANIFEST):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, p = line.split(None, 1)
        p = p.strip()
        full = os.path.join(ROOT, p) if not os.path.isabs(p) else p
        if not os.path.exists(full):
            bad.append(f"missing {p}")
        elif sha(full) != h:
            bad.append(f"changed {p}")
    if bad:
        return False, "frozen files differ: " + "; ".join(bad[:10])
    try:
        txt = open(STATUS).read()
    except OSError:
        return False, "STATUS.md unreadable"
    if not any(FREEZE_TAG in ln and msha in ln for ln in txt.splitlines()):
        return False, f"no STATUS line '{FREEZE_TAG}' quoting manifest sha256 {msha}"
    return True, f"frozen code verified (manifest {msha})"


if __name__ == "__main__":
    print(verify_frozen())
