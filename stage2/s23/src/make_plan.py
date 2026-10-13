"""Stage A run list (PREREG sections 6, 8): plans/stageA_plan.json, built deterministically from
results/prestep/frozen_predictions.json and hashed into STATUS before any sample is drawn.

Phase 'pre' : arm-0 pilots (A, B; 128 each at N0/4096 and N_top), G1b (16 samples at N0/4096, N0,
              N_top; A at tau = t and t/2 on the same counts), G3 (16 pilot samples per learner at the
              two pilot N, re-solved from independent initializations), D2 (16 samples at 2^22, 2^40).
Phase 'main': arm 0 (A, B; 192 each at every grid N) and R- (A' at tau = t/2; 192 at N_br, N0/4096).
Keys: hrhv2-s23|arm=<arm>|L=<L>|log2N=<n>|rep=<r>[|pilot] + '|data' / '|init'.
"""
from __future__ import annotations

import os
import sys

import s23common as C

if __name__ == "__main__":
    C.cap_threads(1)

import cfg as CF


def _item(arm, L, log2N, rep, kind, tau, K_init, phase, pilot=False, data_key=None, tag="", extra=None):
    base = C.run_key(arm, L, log2N, rep, pilot)
    rid = f"{arm}_{L}_l{log2N}_r{rep:03d}{'_pilot' if pilot else ''}{tag}"
    it = {"id": rid, "arm": arm, "L": L, "log2N": int(log2N), "rep": int(rep), "pilot": bool(pilot), "kind": kind,
          "tau": float(tau), "K_init": K_init, "phase": phase,
          "data_key": data_key if data_key is not None else base + "|data", "init_key": base + "|init",
          "cell": f"{arm}/l{log2N}/{L}{'_pilot' if pilot else ''}"}
    if extra:
        it.update(extra)
    return it


def build(fr: dict, cfg: dict):
    t = cfg["t"]
    g = fr["grid"]
    rc = cfg["runs"]
    l0, top = g["log2N0"], g["log2N_top"]
    items = []
    # ---- phase pre
    for l2 in g["log2N_pilot"]:
        for rep in range(rc["pilot"]):
            items.append(_item("0", "A", l2, rep, "A", t, "0_A", "pre", pilot=True))
            items.append(_item("0", "B", l2, rep, "B", t, None, "pre", pilot=True))
    for off in rc["g1b_offsets"]:
        l2 = l0 + off
        for rep in range(rc["g1b"]):
            items.append(_item("G1b", "AAp", l2, rep, "G1b", t, "0_A", "pre"))
    for l2 in g["log2N_pilot"]:
        for rep in range(rc["g3"]):
            for L, kind in (("A", "G3A"), ("B", "G3B")):
                pk = C.run_key("0", L, l2, rep, True) + "|data"
                items.append(_item("G3", L, l2, rep, kind, t, "0_A" if L == "A" else None, "pre", data_key=pk,
                                   extra={"ref_id": f"0_{L}_l{l2}_r{rep:03d}_pilot"}))
    for l2 in rc["d2_log2N"]:
        for rep in range(rc["d2"]):
            items.append(_item("D2", "B", l2, rep, "D2", t, None, "pre"))
    # ---- phase main
    nondec_a = {l0 - 9, l0 + 3}          # N0/512 and 8 N0: dropped first under G4 (reduction a)
    nondec_b = {l0 - 6, l0 - 3}          # N0/64 and N0/8: 96 runs per learner under reduction b
    for l2 in g["grid"]:
        for rep in range(rc["main"]):
            for L in ("A", "B"):
                items.append(_item("0", L, l2, rep, L, t, "0_A" if L == "A" else None, "main",
                                   extra={"red_a": l2 in nondec_a, "red_b": l2 in nondec_b and rep >= rc["main"] // 2}))
    rm_points = sorted({(g["log2N_br"] if p == "br" else l0 + int(p)) for p in rc["rm_points"]})
    for l2 in rm_points:
        for rep in range(rc["rm"]):
            items.append(_item("Rm", "Ap", l2, rep, "A", t / 2.0, "Rm_A", "main",
                               extra={"red_c": rep >= rc["rm"] // 2}))   # reduction c: 96 A' per R- point
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate run ids"
    return {"version": "S2.3 Stage A plan v1.0", "grid": g, "rm_points": rm_points, "n_items": len(items),
            "nondecision_a": sorted(nondec_a), "nondecision_b": sorted(nondec_b), "items": items}


def main():
    C.verify_frozen_code(strict=True)
    fr = C.load_json(C.FROZEN_PRED)
    if not fr.get("gates_pass", False):
        raise SystemExit("pre-step gates did not pass: no plan is written (stop rule)")
    plan = build(fr, fr["cfg"])
    path = os.path.join(C.PLANS, "stageA_plan.json")
    h = C.dump_json(plan, path)
    os.chmod(path, 0o444)
    by = {}
    for it in plan["items"]:
        by[(it["phase"], it["kind"])] = by.get((it["phase"], it["kind"]), 0) + 1
    C.append_status("run", f"S2.3 Stage A plan written (before any sample): plans/stageA_plan.json sha256 {h}; "
                    f"{plan['n_items']} items " + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(by.items())))
    print(h, plan["n_items"])


if __name__ == "__main__":
    main()
