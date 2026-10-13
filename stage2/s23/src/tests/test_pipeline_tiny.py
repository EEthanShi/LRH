"""End-to-end Stage A pipeline on a tiny excluded instance (|X| = 12; keys hrhv2-s23-TEST; outputs in a
temporary S23_ROOT): pre-step -> plan -> workers (both phases) -> analysis phases gates, nmatch, main.
Checks that every phase runs, that the population limits and gates come out as the theorem says, and
that the planted controls are classified as designed."""
import os

import numpy as np
import torch

import s23common as C


def test_pipeline_tiny(tiny_cfg):
    import analyze_A as AN
    import make_plan as MP
    import prestep as PS
    import worker as WK

    dev = torch.device("cpu")
    fr, arr = PS.run(cfg=tiny_cfg, dev=dev, write=True, status=True)
    assert fr["gates_pass"], {k: fr["g1"][k] for k in fr["g1"] if k.startswith("g1_")}
    assert fr["poplim"]["0"]["errA_vs_Kstar"] <= 1e-9 and fr["poplim"]["0"]["errB_vs_KBt"] <= 1e-12
    assert fr["g1b_population"]["pass_Rm"]
    for case in ("0", "Rm", "Rp", "i_mu1", "i_mu4", "ii_p8", "ii_p4", "ii_tau", "iv", "iv_tau", "v"):
        assert case in fr["poplim"], case
    assert fr["poplim"]["0"]["ladder"]["dE"] == 0.0 and fr["poplim"]["0"]["ladder"]["dEk"] == 0.0
    assert fr["g1"]["g1_fisher_at_Kstar"]["pass"] and fr["g1b_population"]["Rm"]["applicable"]
    assert not fr["g1b_population"]["Rm"]["violation"] and fr["g1b_population"]["pass"]
    plan = MP.build(C.load_json(C.FROZEN_PRED), fr["cfg"])
    C.dump_json(plan, os.path.join(C.PLANS, "stageA_plan.json"))
    inst = WK.instance_from(C.load_json(C.FROZEN_PRED))
    fr2, arr2 = WK.load_frozen()
    gpu = WK.GpuCtx(inst, fr2, arr2, "testgpu")
    for it in plan["items"]:
        if it["phase"] == "pre":
            WK.do_item(it, inst, fr2, arr2, gpu, "testgpu")
    st = AN.Store(plan, fr2, arr2)
    os.makedirs(C.STAGEA_DIR, exist_ok=True)
    gates = AN.phase_gates(st, plan, fr2)
    assert gates["G3"]["pass"], gates["G3"]
    assert not gates["G1b"]["solver_failure"]
    assert gates["G4"]["main_allowed"], gates["G4"]
    assert not gates["G1b"]["lemmaT_violation"] and gates["G1b"]["n_applicable"] >= 1, gates["G1b"]
    for it in plan["items"]:
        if it["phase"] == "main":
            WK.do_item(it, inst, fr2, arr2, gpu, "testgpu")
    st = AN.Store(plan, fr2, arr2)
    nm = AN.phase_nmatch(st, fr2)
    assert str(nm["log2N_match"]) in nm["table"]
    res = AN.phase_main(st, fr2)
    rep = AN.write_report(res, fr2)
    assert "F0 test" in rep
    for k in ("P1_A", "P1_B", "P2", "P3a", "P3b", "P3c", "P4a", "P5_order", "P5_geom"):
        assert k in res["F0"] and "holm" in res["F0"][k]
    # calibration: link-self exact at every N, rescale-self exact
    for k, v in res["calibration"].items():
        if k.startswith("link_self") or k.startswith("rescale_self"):
            assert v["ok"], (k, v)
    assert "verdict" in res["verdict"] and "stop_rule" in res
    assert C.status_contains("PRE-STEP done") and C.status_contains("N_match written")
