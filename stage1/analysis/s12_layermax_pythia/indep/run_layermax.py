#!/usr/bin/env python
"""PYT-AXIS COPY of analysis/s12_layermax_openllama/indep/run_layermax.py (sha256 dee3a805...), sub-role pyt-code-I
(S0b, 2026-10-10). Changes (marked PYT-AXIS): the order gate (pyt_gate.gate, plan section 4 F1(a)) is the first action;
threads from PYT_THREADS (launch_pyt.sh); the model list (7 Pythia x 17 ViTs = 119 pairs, plan section 1) and paths come
from the copied lmx_common / common_pyt; option --known (plan V-P4: rerun of the 4 known pairs, output
indep/known/<pre>/pairs/); the S-E2 option is removed (plan R.4: E2 descriptive); options --device cpu and a non-default
--K are accepted only in S0b test mode (CPU smoke on known pairs); GPU memory fraction 0.30 (plan section 11).

usage (from the project root, through the launcher; one process per pre; after the gate G1 of plan section 11):
  analysis/s12_layermax_pythia/launch_pyt.sh pyt_I_main_prh 2 --gpu 4 -- venv/bin/python
      analysis/s12_layermax_pythia/indep/run_layermax.py --pre prh_clip+l2      (one line)
  then (only after V-P5 on the primary and the l2 go line in STATUS) the same with NAME pyt_I_main_l2 and --pre l2
  V-P4 (S6):      --pre P --known                 -> indep/known/<pre>/pairs/ (4 known pairs)
  smoke (debug):  --pre P --only <lm>:<vit>[,...] -> indep/smoke/<pre>/pairs/

Independent ARH layer-max runner (role arh-layermax). One preprocessing per call.

For every (LM, ViT) pair of the panel and every stored layer pair (a, b) it computes, for the identity and
K = 500 permutations pi_k (key S1.2-layermax|<lm>|<vit>, applied to the ViT rows, same pi for every ViT layer):
  * s_N(10) count  = sum_ij M_A^a[i,j] M_B^b[pi(i),pi(j)]   (exact integers)
  * biased CKA     = <Kc_A^a, P Kc_B^b P'> / (|Kc_A^a| |Kc_B^b|)
  * debiased CKA   = Song et al. HSIC_u on Kc (hrhv2 convention, no eps)
and stores the full observed and null (K, L_A, L_B) matrices.

Output: results/S1_2_layermax_pythia/indep/<pre>/pairs/<lm>__<vit>.npz       # PYT-AXIS
        results/S1_2_layermax_pythia/indep/<pre>/prep_meta.json
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True   # PYT-AXIS: a refused dry run writes nothing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyt_gate  # noqa: E402  PYT-AXIS
OUT_ROOT = pyt_gate.gate(__file__)   # PYT-AXIS F1(a): exits 3 unless the plan's gates hold (S0b test mode: test root)

_T = os.environ.get("PYT_THREADS", "1")   # PYT-AXIS
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402


import numpy as np  # noqa: E402
import torch  # noqa: E402

import lmx_common as L  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prep_models(pre: str, dev, out_meta: dict, stems):
    """CPU hrhv2 prep of every stored layer of every model -> GPU stacks."""
    m = L.M
    G = {}
    for stem in stems:
        t0 = time.time()
        S = L.load_stack64(stem)
        nL = S.shape[1]
        meta = {"L": nL, "d": int(S.shape[2]), "layers": []}
        c = None
        if pre == "prh_clip+l2":
            c_re, c_cache = L.clip_c(S, stem)
            meta.update({"clip_recomputed": c_re, "clip_s12_cache": c_cache, "clip_bitwise_equal": bool(c_re == c_cache)})
            c = c_cache
        flat = torch.empty((nL, m * m), dtype=torch.float64, device=dev)
        Mk = torch.zeros((nL, m * m), dtype=torch.float32, device=dev)
        diag = torch.empty((nL, m), dtype=torch.float64, device=dev)
        colsum = torch.empty((nL, m), dtype=torch.float64, device=dev)   # column sums of K~ (A role)
        rowsum = torch.empty((nL, m), dtype=torch.float64, device=dev)   # row sums of K~ (B role)
        nrm2 = np.empty(nL)
        hu = np.empty(nL)
        nbrs = np.empty((nL, m, L.KNN), dtype=np.int64)
        for l in range(nL):
            r = L.prep_layer(S[:, l, :], pre, c)
            Kc = r["Kc"]
            dK = np.ascontiguousarray(np.diag(Kc))
            flat[l] = torch.from_numpy(Kc.reshape(-1)).to(dev)
            diag[l] = torch.from_numpy(dK).to(dev)
            colsum[l] = torch.from_numpy(Kc.sum(axis=0) - dK).to(dev)
            rowsum[l] = torch.from_numpy(Kc.sum(axis=1) - dK).to(dev)
            nrm2[l], hu[l] = r["nrm2"], r["hu"]
            nb = r["nbr"]
            nbrs[l] = nb
            lin = (np.arange(m)[:, None] * m + nb).reshape(-1)
            Mk[l, torch.from_numpy(lin).to(dev)] = 1.0
            meta["layers"].append({"exact_boundary_ties_k10": r["exact_boundary_ties"],
                                   "near_tie_frac_k10": r["near_tie_frac"], "nrm2": r["nrm2"], "hsicu_self": r["hu"]})
        G[stem] = {"flat": flat, "M": Mk, "diag": diag, "colsum": colsum, "rowsum": rowsum,
                   "nrm2": torch.from_numpy(nrm2).to(dev), "hu": torch.from_numpy(hu).to(dev), "L": nL,
                   "nbrs": nbrs}
        meta["prep_s"] = time.time() - t0
        out_meta[stem] = meta
        log(f"prep {stem} L={nL} d={meta['d']} {meta['prep_s']:.1f}s "
            f"boundary-ties={sum(x['exact_boundary_ties_k10'] for x in meta['layers'])}"
            + (f" clip_equal={meta['clip_bitwise_equal']}" if c is not None else ""))
        del S
    return G


@torch.no_grad()
def run_pair(A, B, pis: np.ndarray, dev, batch: int):
    """pis: (P_total, m) permutations (row 0 = identity). Returns arrays
    (P_total, L_A, L_B) for cka_b, cka_d, count."""
    m = L.M
    LA, LB = A["L"], B["L"]
    Ptot = pis.shape[0]
    out_b = np.empty((Ptot, LA, LB))
    out_d = np.empty((Ptot, LA, LB))
    out_c = np.empty((Ptot, LA, LB), dtype=np.int32)
    denom_b = torch.sqrt(A["nrm2"][:, None] * B["nrm2"][None, :])          # (LA, LB)
    denom_d = torch.sqrt(A["hu"][:, None] * B["hu"][None, :])
    sA = A["colsum"].sum(dim=1)
    sB = B["rowsum"].sum(dim=1)
    t2 = (sA[:, None] * sB[None, :]) / ((m - 1) * (m - 2))
    ar = torch.arange(m, device=dev)
    max_round_err = 0.0
    for lo in range(0, Ptot, batch):
        hi = min(Ptot, lo + batch)
        P = hi - lo
        pi = torch.from_numpy(pis[lo:hi]).to(dev)                            # (P, m)
        idx = (pi[:, :, None] * m + pi[:, None, :]).reshape(-1)               # (P*m*m,)
        Bg = B["flat"].index_select(1, idx).view(LB * P, m * m)               # rows (b, p)
        H = (A["flat"] @ Bg.T).view(LA, LB, P)
        del Bg
        Mg = B["M"].index_select(1, idx).view(LB * P, m * m)
        Cn = (A["M"] @ Mg.T).view(LA, LB, P)
        del Mg
        Bd = B["diag"][:, pi]                                                 # (LB, P, m)
        Br = B["rowsum"][:, pi]
        Dd = torch.einsum("am,bpm->abp", A["diag"], Bd)
        t3 = 2.0 * torch.einsum("am,bpm->abp", A["colsum"], Br) / (m - 2)
        cka_b = H / denom_b[:, :, None]
        hu_ab = ((H - Dd) + t2[:, :, None] - t3) / (m * (m - 3))
        cka_d = hu_ab / denom_d[:, :, None]
        Cr = torch.round(Cn)
        max_round_err = max(max_round_err, float((Cn - Cr).abs().max()))
        out_b[lo:hi] = cka_b.permute(2, 0, 1).cpu().numpy()
        out_d[lo:hi] = cka_d.permute(2, 0, 1).cpu().numpy()
        out_c[lo:hi] = Cr.permute(2, 0, 1).to(torch.int32).cpu().numpy()
        del H, Cn, Dd, t3, Bd, Br, idx, pi
    del ar
    return out_b, out_d, out_c, max_round_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, choices=L.PRES)
    ap.add_argument("--K", type=int, default=L.K_PERM, help="(non-default values: S0b test mode only)")
    ap.add_argument("--mem-gb", type=float, default=4.5, help="target size of one gathered batch")
    ap.add_argument("--only", default=None, help="comma list lm:vit (debug)")
    ap.add_argument("--known", action="store_true", help="plan V-P4: the 4 known pairs -> indep/known/<pre>/pairs/")  # PYT-AXIS
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="cpu: S0b test mode only")   # PYT-AXIS
    ap.add_argument("--restart", action="store_true", help="keep an earlier (e.g. interrupted) run aside and start "
                    "from the beginning (plan 11)")   # PYT-AXIS
    args = ap.parse_args()
    assert os.path.realpath(L.OUT) == os.path.realpath(OUT_ROOT), (L.OUT, OUT_ROOT)
    if (args.device != "cuda" or args.K != L.K_PERM) and not pyt_gate.test_mode():   # PYT-AXIS
        raise SystemExit("--device cpu and a non-default --K are S0b test options")
    torch.set_num_threads(int(_T))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)   # PYT-AXIS
    if args.device == "cuda":   # PYT-AXIS
        torch.cuda.set_per_process_memory_fraction(0.30, 0)
    pairs = [(a, b) for b in L.VIT_STEMS for a in L.LM_STEMS]
    sub = None   # PYT-AXIS
    if args.known:   # PYT-AXIS: plan V-P4
        pairs = list(L.C.KNOWN_VP4)
        sub = "known"
    if args.only:
        want = {tuple(x.split(":")) for x in args.only.split(",")}
        pool = pairs + ([] if args.known else list(L.C.KNOWN_VP4))   # PYT-AXIS
        pairs = [p for p in pool if p in want]
        sub = "smoke"
    od = os.path.join(L.OUT, sub, args.pre) if sub else os.path.join(L.OUT, args.pre)   # PYT-AXIS
    if not pairs:
        raise SystemExit("no pair selected")
    # PYT-AXIS plan 11 / B6.5: an interrupted run is never resumed or opened; --restart keeps the old directory aside
    # (renamed <pre>.prev-<UTC>, never deleted) and starts from the beginning (deterministic keys)
    if os.path.isdir(os.path.join(od, "pairs")) and os.listdir(os.path.join(od, "pairs")):
        if not args.restart:
            raise SystemExit(f"{od}/pairs is not empty: an earlier run exists; rerun with --restart (it is kept aside)")
        log(f"--restart: earlier output kept aside as {L.keep_aside(od)}")
    os.makedirs(os.path.join(od, "pairs"), exist_ok=True)
    t_all = time.time()
    stems = []
    for p in pairs:   # PYT-AXIS: LMs first, then ViTs, each once, in pair order
        for s in p:
            if s not in stems:
                stems.append(s)
    meta = {}
    G = prep_models(args.pre, dev, meta, stems)
    L.write_json(os.path.join(od, "prep_meta.json"), L.jsonable(meta))   # PYT-AXIS: earlier copy kept aside
    if args.device == "cuda":
        log(f"prep done {time.time() - t_all:.0f}s; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    m = L.M
    for n, (lm, vit) in enumerate(pairs):
        fn = os.path.join(od, "pairs", f"{lm}__{vit}.npz")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        pis = np.concatenate([np.arange(m)[None, :], L.perms(lm, vit, args.K)], axis=0)
        LB = G[vit]["L"]
        batch = int(max(2, min(64, args.mem_gb * 1e9 // (LB * m * m * 8))))
        cb, cd, cc, rerr = run_pair(G[lm], G[vit], pis, dev, batch)
        if rerr != 0.0:
            raise RuntimeError(f"non-integer kNN count {rerr} for {lm} {vit}")
        np.savez(fn + ".tmp.npz", obs_cka_b=cb[0], obs_cka_d=cd[0], obs_cnt=cc[0],
                 null_cka_b=cb[1:], null_cka_d=cd[1:], null_cnt=cc[1:].astype(np.uint16),
                 K=args.K, key=L.perm_key(lm, vit), L_A=G[lm]["L"], L_B=LB, secs=time.time() - t0, batch=batch)
        os.replace(fn + ".tmp.npz", fn)
        log(f"[{n + 1}/{len(pairs)}] {lm} x {vit} ({G[lm]['L']}x{LB}, batch {batch}) {time.time() - t0:.1f}s")
    log(f"ALL DONE {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
