from hrhv2_ref.compare import compare_ladders, tol_class


def test_tolerance_classes():
    assert tol_class("s_N@10") == "exact" and tol_class("s_T_bridge") == "exact"
    assert tol_class("quot_rowiso_expl_xfit") == "iso" and tol_class("cka_debiased") == "float"
    ref = {"s_N@10": 0.5, "cka_debiased": 0.9, "quot_iso_expl": 0.3, "pre": "l2", "x": float("nan")}
    main = {"s_N@10": 0.5, "cka_debiased": 0.9 + 1e-12, "quot_iso_expl": 0.3 + 1e-7, "x": float("nan")}
    assert all(r["ok"] for r in compare_ladders(main, ref))
    main["s_N@10"] = 0.5 + 1e-15
    main["cka_debiased"] = 0.9 + 1e-9
    bad = {r["key"] for r in compare_ladders(main, ref) if not r["ok"]}
    assert bad == {"s_N@10", "cka_debiased"}
