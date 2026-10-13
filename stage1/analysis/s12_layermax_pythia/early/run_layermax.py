"""Main run (plan step S7): layer-max calibration for the 119 Pythia-axis pairs (7 Pythia sizes x 17 ViTs)
x {prh_clip+l2, l2} (plan section 3), K = 500, same pi for every ViT layer. GATED (plan F1(a)).

PYTHIA-AXIS COPY (implementation E, S0b sub-role pyt-code-E): copied from
analysis/s12_layermax_openllama/early/run_layermax.py (sha256 4f690fa2...). Changes, marked "PYT-AXIS": the E
gate lm_core.gate() (own Part B check + shared axis guard, plan F1(a)) is the first action after argument
parsing; the pair list comes from lm_core (7 Pythia LMs x 17 ViTs = 119 pairs); outputs go to
results/S1_2_layermax_pythia/early/pairs/<pre>/ (lm_core.RES); --lms takes the size labels 70m..6.9b;
torch.cuda calls only on the GPU device (S0b CPU test device). The pair file format is unchanged (plan section 3:
"Pair-level code: only the model list, paths, family label and keys change"). Permutation key
S1.2-layermax|EleutherAI__pythia-<size>|<vit stem> (plan section 3).

Run (step S7, after V-P1..V-P4 and the S6 gates; GPU, <= 3 threads; prh_clip+l2 first, plan section 11):
  cd ~/hrh_v2_stage1_20261008 && analysis/s12_layermax_pythia/launch_pyt.sh pyt_E_main_prh 2 --gpu 4 -- \\
      venv/bin/python analysis/s12_layermax_pythia/early/run_layermax.py --pre prh_clip+l2
  (l2 only after the logged l2 go/no-go: ... pyt_E_main_l2 ... --pre l2. PYT-AXIS: not resumable. A non-empty
  pairs/<pre> tree is refused (exit 2); after an interruption, --restart renames the old tree to
  pairs/<pre>.interrupted-<UTC> (kept, never opened) and reruns that pre from the beginning, plan section 11)

Per pair and pre:
  <RES>/pairs/<pre>/<lm>__<vit>.json   summaries for every
      pre-specified layer subset (full, grid n, fixed r, one-sided, noLMemb, LMdeep, LMemb_only) and metric,
      plus calib_then_max (protocol iv)
  <RES>/pairs/<pre>/<lm>__<vit>.npz    observed L_A x L_B
      matrices (f64), entry-wise null mean / c_alpha / p (f64), per-permutation
      layer-max (f64) and argmax, and the full null tensors (CKA float32,
      mKNN counts int16) for later exploration.

Usage: python run_layermax.py [--pre PRE ...] [--pairs N] [--lms 70m,160m,...] [--vits STEM,...]
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

import lm_core as LC  # noqa: E402  (constants and functions only; reads no data at import)
from hrhv2 import calibrate as cal  # noqa: E402

METRICS = ("mknn", "cka_biased", "cka_debiased")
ORDER_IDX = cal.order_index(LC.K_PERM, LC.ALPHA)       # 476 (1-based)


def load_models(pre: str, stems=None):
    """Build the GPU models of the given stems (default: all 7 LMs and 17 ViTs)."""  # PYT-AXIS
    models = {}
    t0 = time.time()
    for stem, _ in LC.LMS:
        if stems is not None and stem not in stems:
            continue
        tab = np.load(os.path.join(LC.cache_dir(stem), "knn_final", f"{stem}__{pre}.npy"))
        models[stem] = LC.ModelGPU(stem, pre, "A", LC.clip_value(stem) if pre == "prh_clip+l2" else None, knn_tables=tab)
    for stem, _ in LC.VITS:
        if stems is not None and stem not in stems:
            continue
        tab = np.load(os.path.join(LC.cache_dir(stem), "knn_final", f"{stem}__{pre}.npy"))
        models[stem] = LC.ModelGPU(stem, pre, "B", LC.clip_value(stem) if pre == "prh_clip+l2" else None, knn_tables=tab)
    print(f"[{LC.utc()}] built {len(models)} models for {pre} in {time.time() - t0:.0f}s; "
          f"GPU alloc {(torch.cuda.memory_allocated() / 1e9) if LC.DEV == 'cuda' else 0.0:.2f} GB", flush=True)  # PYT-AXIS
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


def select_pairs(lms_arg, vits_arg, n_first):
    lab = {v: k for k, v in LC.SIZE_LABEL.items()}
    lms = [s for s, _ in LC.LMS]   # PYT-AXIS: no S-E2 subset on this axis
    if lms_arg:
        want = {lab.get(x.strip(), x.strip()) for x in lms_arg.split(",") if x.strip()}
        lms = [s for s in lms if s in want]
    vits = [s for s, _ in LC.VITS]
    if vits_arg:
        want = {x.strip() for x in vits_arg.split(",") if x.strip()}
        vits = [s for s in vits if s in want]
    pairs = [(lm, vit) for lm in lms for vit in vits]
    return pairs[:n_first] if n_first else pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", nargs="*", default=list(LC.PRES))
    ap.add_argument("--pairs", type=int, default=None, help="only the first N pairs (debug)")
    ap.add_argument("--lms", default=None, help="debug: comma list of LM stems or size labels 70m..6.9b")  # PYT-AXIS
    ap.add_argument("--vits", default=None, help="debug: comma list of ViT stems")
    ap.add_argument("--restart", action="store_true",  # PYT-AXIS: plan section 11 "Interrupted or deferred S7"
                    help="an existing (interrupted) pairs/<pre> tree is renamed to pairs/<pre>.interrupted-<UTC> "
                         "(kept, never opened) and that pre restarts from the beginning")  # PYT-AXIS
    args = ap.parse_args()
    LC.gate("early/run_layermax.py")   # PYT-AXIS: plan F1(a), first action
    for pre in args.pre:   # PYT-AXIS: no silent resume (plan section 11): refuse before anything is computed
        outdir = os.path.join(LC.RES, "pairs", pre)  # PYT-AXIS
        if os.path.isdir(outdir) and os.listdir(outdir) and not args.restart:  # PYT-AXIS
            print(f"[run_layermax] REFUSED: {outdir} is not empty (an earlier or interrupted run); partial outputs "
                  f"are never resumed or opened (plan section 11); rerun with --restart", file=sys.stderr, flush=True)
            sys.exit(2)  # PYT-AXIS
    LC.setup_torch(threads=2)
    t_all = time.time()
    pairs = select_pairs(args.lms, args.vits, args.pairs)
    for pre in args.pre:
        outdir = os.path.join(LC.RES, "pairs", pre)
        if os.path.isdir(outdir) and os.listdir(outdir):  # PYT-AXIS: --restart: keep the old tree, never open it
            aside = f"{outdir}.interrupted-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"  # PYT-AXIS
            os.rename(outdir, aside)  # PYT-AXIS (a rename, not a deletion: both outputs are kept)
            print(f"[{LC.utc()}] {pre}: earlier tree moved aside to {aside} (not opened)", flush=True)  # PYT-AXIS
        os.makedirs(outdir, exist_ok=True)
        todo = list(pairs)   # PYT-AXIS: every selected pair, from the beginning
        if not todo:
            continue
        models = load_models(pre, {x for p in todo for x in p})
        flags = {s: [int(x) for x in np.flatnonzero(models[s].boundary_tie)] for s, _ in LC.VITS if s in models}
        for n, (lm, vit) in enumerate(todo):
            t0 = time.time()
            A, B = models[lm], models[vit]
            perms = LC.perms_for(lm, vit)
            R = LC.run_pair(A, B, perms)
            LC.synchronize()   # PYT-AXIS
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
        if LC.DEV == "cuda":  # PYT-AXIS
            torch.cuda.empty_cache()
            print(f"[{LC.utc()}] peak GPU alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    print(f"[{LC.utc()}] main run finished in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
