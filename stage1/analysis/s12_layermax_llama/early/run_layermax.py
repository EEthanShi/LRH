"""Main run: layer-max calibration for 136 LM-ViT pairs x {prh_clip+l2, l2}
(plan sections 3 and 7), K = 500, same pi for every ViT layer.

LLAMA-AXIS COPY (implementation E, role llama-axis-run-early): copied from
analysis/s12_layermax/run_layermax.py (sha256 b6490719...). Changes, marked
"LLAMA-AXIS": the pair list comes from lm_core (3 LLaMA x 17 ViTs = 51 pairs),
outputs go to results/S1_2_layermax_llama/early/pairs/<pre>/, and protocol
(iv) of ANALYSIS_PLAN_LLAMA section 3 (calibrate-then-max) is added to each
pair record under "calib_then_max" from the same entry-wise statistics (no
change to any existing field). The new subsets of section 3 (iii) come from
lm_core.subsets.

Per pair and pre:
  results/S1_2_layermax/pairs/<pre>/<lm>__<vit>.json   summaries for every
      pre-specified layer subset (full, grid n, fixed r, one-sided) and metric
  results/S1_2_layermax/pairs/<pre>/<lm>__<vit>.npz    observed L_A x L_B
      matrices (f64), entry-wise null mean / c_alpha / p (f64), per-permutation
      layer-max (f64) and argmax, and the full null tensors (CKA float32,
      mKNN counts int16) for later exploration.

Usage: python run_layermax.py [--pre PRE ...] [--pairs N]   (markers by the launcher)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lm_core as LC  # noqa: E402
from hrhv2 import calibrate as cal  # noqa: E402

METRICS = ("mknn", "cka_biased", "cka_debiased")
ORDER_IDX = cal.order_index(LC.K_PERM, LC.ALPHA)       # 476 (1-based)


def load_models(pre: str):
    clip = json.load(open(os.path.join(LC.CACHE, "clip.json")))
    models = {}
    t0 = time.time()
    for stem, _ in LC.LMS:
        tab = np.load(os.path.join(LC.CACHE, "knn_final", f"{stem}__{pre}.npy"))
        models[stem] = LC.ModelGPU(stem, pre, "A", clip[stem]["c"] if pre == "prh_clip+l2" else None, knn_tables=tab)
    for stem, _ in LC.VITS:
        tab = np.load(os.path.join(LC.CACHE, "knn_final", f"{stem}__{pre}.npy"))
        models[stem] = LC.ModelGPU(stem, pre, "B", clip[stem]["c"] if pre == "prh_clip+l2" else None, knn_tables=tab)
    print(f"[{LC.utc()}] built {len(models)} models for {pre} in {time.time() - t0:.0f}s; "
          f"GPU alloc {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    return models


def entry_stats(T: torch.Tensor) -> dict:
    """Entry-wise calibration of every layer pair: T (K+1, LA, LB), index 0 = observed."""
    obs = T[0]
    null = T[1:]
    srt, _ = torch.sort(T, dim=0)
    c_alpha = srt[ORDER_IDX - 1]
    p = (1.0 + (null >= obs[None]).sum(dim=0).double()) / (null.shape[0] + 1)
    return {"obs": obs, "null_mean": null.mean(dim=0), "c_alpha": c_alpha, "p": p,
            "null_sd": null.std(dim=0, unbiased=True)}


def subset_summary(T: torch.Tensor, ia: list[int], ib: list[int], E: dict) -> dict:
    sub = T[:, ia][:, :, ib]                                  # (K+1, a, b)
    obs = sub[0]
    LBs = len(ib)
    flat = obs.reshape(-1)
    t_obs = float(flat.max().item())
    j = int(torch.argmax(flat).item())                        # first maximum (row-major: lower LM layer first)
    a_star, b_star = ia[j // LBs], ib[j % LBs]
    n_tied = int((flat == flat.max()).sum().item())
    tmax = sub[1:].reshape(sub.shape[0] - 1, -1)
    null_max, null_arg = tmax.max(dim=1)
    null_np = null_max.double().cpu().numpy()
    s = cal.summarize(t_obs, null_np, LC.ALPHA)
    s.pop("c_alpha_arh_npquantile", None)
    s.update({"argmax_A": a_star, "argmax_B": b_star, "n_tied_max": n_tied, "n_layer_pairs": len(ia) * LBs,
              "entry_c_alpha_at_argmax": float(E["c_alpha"][a_star, b_star].item()),
              "entry_null_mean_at_argmax": float(E["null_mean"][a_star, b_star].item()),
              "entry_p_at_argmax": float(E["p"][a_star, b_star].item())})
    na = null_arg.cpu().numpy()
    s["null_argmax_A"] = [ia[int(v) // LBs] for v in na]
    s["null_argmax_B"] = [ib[int(v) % LBs] for v in na]
    return s, null_np


def calib_then_max(E: dict) -> dict:
    """LLAMA-AXIS, plan section 3 (iv): M = max_uv s_cal,uv with s_cal,uv = max((S_uv - c_uv)/(1 - c_uv), 0),
    each entry calibrated against its own c_alpha from the same 500 pi; uncensored analogue
    max_uv (S_uv - mean_k S_uv^(k)). First maximum in row-major order (lower LM layer first)."""
    obs, c, nm = E["obs"], E["c_alpha"], E["null_mean"]
    LB = obs.shape[1]
    den = 1.0 - c
    scal = torch.where(den > 0, torch.clamp((obs - c) / den, min=0.0), torch.zeros_like(obs))
    unc = obs - nm
    j = int(torch.argmax(scal.reshape(-1)).item())
    ju = int(torch.argmax(unc.reshape(-1)).item())
    M = float(scal.reshape(-1)[j].item())
    return {"M": M, "argmax_A": j // LB, "argmax_B": j % LB,
            "n_tied_max": int((scal == scal.max()).sum().item()),
            "n_entries_scal_pos": int((scal > 0).sum().item()), "n_entries": int(scal.numel()),
            "s_obs_at_argmax": float(obs.reshape(-1)[j].item()), "c_alpha_at_argmax": float(c.reshape(-1)[j].item()),
            "M_unc": float(unc.reshape(-1)[ju].item()), "M_unc_argmax_A": ju // LB, "M_unc_argmax_B": ju % LB}


def summarize_pair(R: dict, LA: int, LB: int) -> tuple[dict, dict]:
    subs = LC.subsets(LA, LB)
    js, arrays = {}, {}
    ctm = {}   # LLAMA-AXIS: protocol (iv)
    for met in METRICS:
        T = R[met]
        E = entry_stats(T)
        ctm[met] = calib_then_max(E)
        for k in ("obs", "null_mean", "c_alpha", "p", "null_sd"):
            arrays[f"{met}.entry_{k}"] = E[k].cpu().numpy()
        js[met] = {}
        for name, (ia, ib) in subs.items():
            s, null_np = subset_summary(T, ia, ib, E)
            if name != "full":
                s.pop("null_argmax_A")
                s.pop("null_argmax_B")
            else:
                arrays[f"{met}.full_null_max"] = null_np
                arrays[f"{met}.full_null_argmax_A"] = np.array(s["null_argmax_A"], dtype=np.int16)
                arrays[f"{met}.full_null_argmax_B"] = np.array(s["null_argmax_B"], dtype=np.int16)
                s.pop("null_argmax_A")
                s.pop("null_argmax_B")
            s["layers_A"], s["layers_B"] = ia, ib
            js[met][name] = s
        if met == "mknn":
            arrays[f"{met}.null_counts_int16"] = np.rint(T[1:].cpu().numpy() * LC.M_ITEMS * LC.KNN_K).astype(np.int16)
        else:
            arrays[f"{met}.null_f32"] = T[1:].float().cpu().numpy()
    js["_calib_then_max"] = ctm   # LLAMA-AXIS: moved to rec["calib_then_max"] by main()
    return js, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", nargs="*", default=list(LC.PRES))
    ap.add_argument("--pairs", type=int, default=None, help="only the first N pairs (debug)")
    args = ap.parse_args()
    LC.setup_torch(threads=6)
    t_all = time.time()
    pairs = [(lm, vit) for lm, _ in LC.LMS for vit, _ in LC.VITS]
    if args.pairs:
        pairs = pairs[:args.pairs]
    for pre in args.pre:
        outdir = os.path.join(LC.RES, "pairs", pre)
        os.makedirs(outdir, exist_ok=True)
        todo = [p for p in pairs if not os.path.exists(os.path.join(outdir, f"{p[0]}__{p[1]}.json"))]
        if not todo:
            continue
        models = load_models(pre)
        flags = {s: [int(x) for x in np.flatnonzero(models[s].boundary_tie)] for s, _ in LC.VITS}
        for n, (lm, vit) in enumerate(todo):
            t0 = time.time()
            A, B = models[lm], models[vit]
            perms = LC.perms_for(lm, vit)
            R = LC.run_pair(A, B, perms)
            torch.cuda.synchronize()
            t_gpu = time.time() - t0
            js, arrays = summarize_pair(R, A.L, B.L)
            ctm = js.pop("_calib_then_max")   # LLAMA-AXIS
            del R
            rec = {"lm": lm, "vit": vit, "pre": pre, "K": LC.K_PERM, "alpha": LC.ALPHA,
                   "perm_key": LC.perm_key(lm, vit), "L_A": A.L, "L_B": B.L,
                   "vit_layers_relabel_recomputed": flags[vit],
                   "summaries": js, "calib_then_max": ctm,   # LLAMA-AXIS: protocol (iv)
                   "timing_s": {"gpu": t_gpu, "total": time.time() - t0},
                   "finished_utc": LC.utc()}
            base = os.path.join(outdir, f"{lm}__{vit}")
            np.savez(base + ".tmp.npz", **arrays)
            os.replace(base + ".tmp.npz", base + ".npz")
            with open(base + ".json.tmp", "w") as f:
                json.dump(LC.jsonable(rec), f)
            os.replace(base + ".json.tmp", base + ".json")
            # LLAMA-AXIS: the two summary lookups for the log line were removed with it
            # LLAMA-AXIS: completion status only in the log (plan section 4: no main-run output is
            # read before V-L1..V-L7 pass)
            print(f"[{LC.utc()}] {pre} {n + 1}/{len(todo)} {lm} x {vit} ({A.L}x{B.L}) "
                  f"{time.time() - t0:.1f}s done", flush=True)
        del models
        torch.cuda.empty_cache()
        print(f"[{LC.utc()}] peak GPU alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"[{LC.utc()}] main run finished in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
