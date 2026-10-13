"""S1.3a analysis: pass counts, one-sided 95% Clopper-Pearson decisions, prediction verdicts.
Reads criteria/*.csv; writes ../results/S1_3a_linae_summary.json and .md. Verdict rules are frozen in
ANALYSIS_PLAN.md (written before the confirmatory runs)."""
import os, sys, json, glob, time, hashlib
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import linae_core as lc
import linae_metrics as lm

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
CRIT = os.path.join(HERE, 'criteria')
QS = ['kNN', 'Gs', 'CKA', 'proj']
NS = ['256', '4096', '65536', 'inf']
PRIMARY = [a for a, c in lc.ARMS.items() if c['primary']]
SECONDARY = [a for a, c in lc.ARMS.items() if not c['primary']]
LARGEST = '65536'


def tci(x, level=0.95):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan, len(x))
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * se
    return (float(m), float(m - h), float(m + h), int(len(x)))


def cell_decisions(df):
    out = {}
    n = len(df)
    for Q in QS:
        key = lm.DELTA_KEY[Q]
        delta = df[key].to_numpy(float) if key in df else np.full(n, np.nan)
        passes = int(np.sum(np.nan_to_num(delta, nan=np.inf) <= lm.EPS[Q]))
        dec, lo, hi = lm.decision(passes, n)
        m, clo, chi, nf = tci(delta)
        fin = delta[np.isfinite(delta)]
        curve = {f'{f:g}eps': (float(np.mean(np.nan_to_num(delta, nan=np.inf) <= f * lm.EPS[Q])) if n else np.nan)
                 for f in (0.5, 1, 2, 5, 10)}
        out[Q] = dict(n_pairs=n, passes=passes, p_hat=passes / n if n else np.nan, cp_lower=lo, cp_upper=hi,
                      decision=dec, mean=m, mean_ci95=[clo, chi], n_finite=nf,
                      median=float(np.median(fin)) if len(fin) else np.nan,
                      q10=float(np.quantile(fin, .1)) if len(fin) else np.nan,
                      q90=float(np.quantile(fin, .9)) if len(fin) else np.nan,
                      p_hat_curve=curve)
    return out


def main():
    pairs = pd.concat([pd.read_csv(f, dtype={'N': str}) for f in sorted(glob.glob(os.path.join(CRIT, '*_pairs.csv')))])
    runs = pd.concat([pd.read_csv(f, dtype={'N': str}) for f in sorted(glob.glob(os.path.join(CRIT, '*_runs.csv')))])
    S = dict(meta=dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       prereg_sha256=hashlib.sha256(open(os.path.join(ROOT, 'prereg', 'PREREG_stage1.md'), 'rb').read()).hexdigest(),
                       spec_sha256=hashlib.sha256(open(os.path.join(ROOT, 'SPEC.md'), 'rb').read()).hexdigest(),
                       code_sha256={os.path.basename(p): hashlib.sha256(open(p, 'rb').read()).hexdigest()
                                    for p in sorted(glob.glob(os.path.join(HERE, '*.py')))},
                       eps=lm.EPS, arms={a: {k: v for k, v in c.items()} for a, c in lc.ARMS.items()},
                       schedule=dict(T_CONST=lc.T_CONST, T_DECAY=lc.T_DECAY, DECAY_FACTOR=lc.DECAY_FACTOR,
                                     POLISH_LR=lc.POLISH_LR, POLISH_MAX=lc.POLISH_MAX, GD_MAX_STEPS=lc.GD_MAX_STEPS),
                       wd_lambda=lc.WD_LAMBDA, small_sigma0=lc.SMALL_SIGMA0, cert_tol=lc.CERT_TOL))
    cells = {}
    for arm in lc.ARMS:
        for N in NS:
            dfa = pairs[(pairs.arm == arm) & (pairs.N == N)]
            fin = dfa[dfa.variant == 'final'].sort_values('pair')
            c = dict(primary=cell_decisions(fin))
            if N == 'inf':
                c['first48'] = cell_decisions(fin[fin.pair < 48])
            nodiv = fin[~(fin.divA | fin.divB)]
            c['excl_diverged_pairs'] = cell_decisions(nodiv)
            pre = dfa[dfa.variant == 'prepolish'].sort_values('pair')
            if len(pre):
                c['prepolish'] = cell_decisions(pre)
            r = runs[(runs.arm == arm) & (runs.N == N)]
            okr = r[~r.diverged]
            spec = okr[[f'eigEEt_{i}' for i in range(lc.d)]].to_numpy()
            specn = spec / spec.mean(1, keepdims=True)
            pred = okr[[f'eig_fC0_{i}' for i in range(lc.d)]].to_numpy()
            c['runs'] = dict(
                n_runs=len(r), n_diverged=int(r.diverged.sum()), n_cert_ok=int(r.cert_ok.sum()),
                n_cert_fail_nondiverged=int((~r.cert_ok & ~r.diverged).sum()),
                rel_gap_max_certified=float(r.loc[r.cert_ok, 'rel_gap'].max()) if r.cert_ok.any() else np.nan,
                rel_gap_min=float(okr.rel_gap.min()) if len(okr) else np.nan,
                rel_grad_median=float(okr.rel_grad.median()) if len(okr) else np.nan,
                steps_median=float(okr.steps.median()) if len(okr) else np.nan,
                steps_max=int(okr.steps.max()) if len(okr) else -1,
                C0_norm_median=float(r.C0_norm.median()),
                Cend_norm_median=float(okr.Cend_norm.median()) if len(okr) else np.nan,
                C_drift_rel_median=float(okr.C_drift_rel.median()) if len(okr) else np.nan,
                C_drift_rel_q90=float(okr.C_drift_rel.quantile(.9)) if len(okr) else np.nan,
                C_drift_rel_max=float(okr.C_drift_rel.max()) if len(okr) else np.nan,
                kappa_q10_50_90=[float(x) for x in okr.kappa.quantile([.1, .5, .9])] if len(okr) else None,
                eigEEt_median=[float(x) for x in np.median(spec, 0)] if len(okr) else None,
                eigEEt_normalised_median=[float(x) for x in np.median(specn, 0)] if len(okr) else None,
                flow_pred_relerr_median=float(np.median(np.max(np.abs(spec - pred), 1) / pred.max(1))) if len(okr) else np.nan,
                flow_pred_relerr_q90=float(np.quantile(np.max(np.abs(spec - pred), 1) / pred.max(1), .9)) if len(okr) else np.nan,
                target_median={Q: float(okr['target_' + lm.DELTA_KEY[Q]].median()) for Q in QS} if len(okr) else None)
            if 'rel_gap_pre' in r:
                kt = sorted([cname for cname in r.columns if cname.startswith('kappa_t')], key=lambda s: int(s[7:]))
                Ct = sorted([cname for cname in r.columns if cname.startswith('Cnorm_t')], key=lambda s: int(s[7:]))
                c['runs'].update(
                    rel_gap_pre_median=float(okr.rel_gap_pre.median()) if len(okr) else np.nan,
                    Cpre_norm_median=float(okr.Cpre_norm.median()) if len(okr) else np.nan,
                    kappa_pre_q10_50_90=[float(x) for x in okr.kappa_pre.quantile([.1, .5, .9])] if len(okr) else None,
                    kappa_traj_median={kk[7:]: float(okr[kk].median()) for kk in kt},
                    Cnorm_traj_median={kk[7:]: float(okr[kk].median()) for kk in Ct})
            cells[f'{arm}|{N}'] = c
    S['cells'] = cells

    # ------------------------------------------------------------------ verdicts (rules frozen in ANALYSIS_PLAN.md)
    def combine(parts):
        if any(p == 'FALSIFIED' for p in parts):
            return 'FALSIFIED'
        if all(p == 'CONFIRMED' for p in parts):
            return 'CONFIRMED'
        return 'INCONCLUSIVE'

    def want(dec, target):
        if dec == target:
            return 'CONFIRMED'
        if dec == 'inconclusive':
            return 'INCONCLUSIVE'
        return 'FALSIFIED'

    def pred1(arm):
        parts, detail = [], []
        for Q in ['kNN', 'Gs', 'CKA']:
            for N in NS:
                dec = cells[f'{arm}|{N}']['primary'][Q]['decision']
                v = want(dec, 'unreliable'); parts.append(v)
                detail.append(dict(part='a', Q=Q, N=N, decision=dec, verdict=v))
        for Q in ['kNN', 'Gs', 'CKA']:
            big = cells[f'{arm}|{LARGEST}']['primary'][Q]
            inf = cells[f'{arm}|inf']['primary'][Q]
            inside = inf['mean_ci95'][0] <= big['mean'] <= inf['mean_ci95'][1]
            v = 'CONFIRMED' if inside else 'FALSIFIED'; parts.append(v)
            # non-confirmatory diagnostics for part (b)
            dA = pairs[(pairs.arm == arm) & (pairs.N == LARGEST) & (pairs.variant == 'final')].sort_values('pair')[lm.DELTA_KEY[Q]].to_numpy(float)
            dI = pairs[(pairs.arm == arm) & (pairs.N == 'inf') & (pairs.variant == 'final')].sort_values('pair')[lm.DELTA_KEY[Q]].to_numpy(float)
            a_, i_ = dA[np.isfinite(dA)], dI[np.isfinite(dI)]
            w = stats.ttest_ind(a_, i_, equal_var=False)
            wci = w.confidence_interval(0.95)
            dP = dA - dI[:48]
            pm, plo, phi, pn = tci(dP)
            detail.append(dict(part='b', Q=Q, mean_65536=big['mean'], n_65536=big['n_finite'], mean_inf=inf['mean'],
                               ci95_inf=inf['mean_ci95'], n_inf=inf['n_finite'], inside=bool(inside), verdict=v,
                               diag_welch_diff=float(a_.mean() - i_.mean()), diag_welch_ci95=[float(wci.low), float(wci.high)],
                               diag_welch_p=float(w.pvalue),
                               diag_paired_CRN_diff=pm, diag_paired_CRN_ci95=[plo, phi], diag_paired_n=pn,
                               diag_null_pass_prob_literal_test=float(2 * stats.norm.cdf(1.96 * np.sqrt(big['n_finite'] / inf['n_finite'])) - 1)))
        return combine(parts), detail

    V = {}
    v1, d1 = pred1('P-GD')
    V['P1'] = dict(statement='P-GD: kNN, Gs and CKA unreliable at every N; the largest-N mean is inside the 95% CI of the N=inf plateau.',
                   verdict=v1, detail=d1)
    v1s, d1s = pred1('P-GD-flow')
    V['P1_secondary_flow'] = dict(statement='(secondary, deviation arm) same as P1 for full-batch GD at lr=0.002 (gradient-flow proxy).',
                                  verdict=v1s, detail=d1s)
    parts, det = [], []
    for arm in ['WD', 'TIED', 'small-init']:
        for Q in QS:
            dec = cells[f'{arm}|{LARGEST}']['primary'][Q]['decision']
            v = want(dec, 'reliable'); parts.append(v); det.append(dict(arm=arm, Q=Q, N=LARGEST, decision=dec, verdict=v))
    V['P2'] = dict(statement='WD, TIED and small-init: reliable at the largest N (65536), all four criteria.', verdict=combine(parts), detail=det)
    parts3, det3 = [], []
    for arm in ['WD', 'TIED', 'small-init']:
        for Q in ['kNN', 'Gs', 'CKA']:
            dec = cells[f'{arm}|{LARGEST}']['primary'][Q]['decision']
            parts3.append(want(dec, 'reliable'))
    V['P2_gauge_criteria_only'] = dict(statement='(reading check) P2 restricted to kNN, Gs, CKA', verdict=combine(parts3))
    parts, det = [], []
    for arm in PRIMARY:
        dec = cells[f'{arm}|{LARGEST}']['primary']['proj']['decision']
        v = want(dec, 'reliable'); parts.append(v); det.append(dict(arm=arm, N=LARGEST, decision=dec, verdict=v))
    V['P3'] = dict(statement='Projector criterion reliable in every arm at the largest N (65536).', verdict=combine(parts), detail=det)
    det4 = {}
    for arm in ['P-SGD', 'P-Adam']:
        det4[arm] = {N: {Q: cells[f'{arm}|{N}']['primary'][Q]['decision'] for Q in QS} for N in NS}
    V['P4'] = dict(statement='P-SGD and P-Adam: no prediction (Ziyin & Chuang self-balancing question); outcome reported.',
                   verdict='NO PREDICTION (outcome reported)', decisions=det4)
    # sensitivity of verdicts to counting diverged runs as failures
    def dec_excl(arm, N, Q):
        return cells[f'{arm}|{N}']['excl_diverged_pairs'][Q]['decision']
    V['sensitivity_excluding_diverged_pairs'] = {
        'P1_part_a': {f'{Q}|{N}': dec_excl('P-GD', N, Q) for Q in ['kNN', 'Gs', 'CKA'] for N in NS},
        'P2': {f'{arm}|{Q}': dec_excl(arm, LARGEST, Q) for arm in ['WD', 'TIED', 'small-init'] for Q in QS},
        'P3': {arm: dec_excl(arm, LARGEST, 'proj') for arm in PRIMARY}}
    S['verdicts'] = V
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, 'S1_3a_linae_summary.json'), 'w') as f:
        json.dump(S, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    write_md(S)
    print(json.dumps({k: v['verdict'] for k, v in V.items() if isinstance(v, dict) and 'verdict' in v}, indent=1))


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'nan'
    if isinstance(x, float):
        return f'{x:.{nd}g}'
    return str(x)


def write_md(S):
    L = []
    L.append('# S1.3a — certified linear-AE GL instance: results summary\n')
    L.append(f"Generated {S['meta']['generated_utc']} by `linae/analyze_s13a.py` from `linae/criteria/*.csv`. "
             f"PREREG sha256 {S['meta']['prereg_sha256'][:12]}…, SPEC sha256 {S['meta']['spec_sha256'][:12]}…. "
             'Verdict rules: `linae/ANALYSIS_PLAN.md` (frozen before the confirmatory runs).\n')
    L.append('## Verdicts\n')
    for k, v in S['verdicts'].items():
        if isinstance(v, dict) and 'verdict' in v:
            L.append(f"- **{k}** — {v['statement']} → **{v['verdict']}**")
    L.append('')
    L.append('## Decision table (one-sided 95% Clopper–Pearson; reliable if lower > 0.9, unreliable if upper < 0.8)\n')
    L.append('Cells: passes/pairs, decision, mean Δ [95% CI]. N=inf uses all 500 disjoint pairs (1000 inits). Diverged runs count as failures.\n')
    L.append('| arm | N | kNN (1−s_N(10) ≤ 0.05) | Gs_rms ≤ 0.02 | 1−CKA ≤ 0.01 | proj ≤ 0.01 | diverged | cert fail |')
    L.append('|---|---|---|---|---|---|---|---|')
    for arm in lc.ARMS:
        for N in NS:
            c = S['cells'][f'{arm}|{N}']
            row = [arm + ('' if lc.ARMS[arm]['primary'] else ' (secondary)'), N]
            for Q in QS:
                q = c['primary'][Q]
                row.append(f"{q['passes']}/{q['n_pairs']} **{q['decision']}**; {fmt(q['mean'], 3)} [{fmt(q['mean_ci95'][0], 3)}, {fmt(q['mean_ci95'][1], 3)}]")
            row.append(str(c['runs']['n_diverged']))
            row.append(str(c['runs']['n_cert_fail_nondiverged']))
            L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('## Imbalance C = EEᵀ − WᵀW and latent anisotropy (spec EEᵀ = spec S)\n')
    L.append('| arm | N | median ‖C₀‖_F | median ‖C_end‖_F | ‖C_end−C₀‖/‖C₀‖ median (q90) | κ(EEᵀ) q10/q50/q90 | median normalised spec(EEᵀ) | spec vs f(spec C₀) rel.err median |')
    L.append('|---|---|---|---|---|---|---|---|')
    for arm in lc.ARMS:
        for N in NS:
            r = S['cells'][f'{arm}|{N}']['runs']
            kq = r['kappa_q10_50_90'] or [np.nan] * 3
            sp = r['eigEEt_normalised_median'] or [np.nan] * 4
            L.append(f"| {arm} | {N} | {fmt(r['C0_norm_median'], 3)} | {fmt(r['Cend_norm_median'], 3)} | {fmt(r['C_drift_rel_median'], 3)} ({fmt(r['C_drift_rel_q90'], 3)}) | "
                     f"{'/'.join(fmt(x, 3) for x in kq)} | {', '.join(fmt(x, 3) for x in sp)} | {fmt(r['flow_pred_relerr_median'], 3)} |")
    L.append('')
    L.append('## Prediction 1 part (b) detail and diagnostics\n')
    for key in ('P1', 'P1_secondary_flow'):
        for dd in S['verdicts'][key]['detail']:
            if dd['part'] == 'b':
                L.append(f"- {key} {dd['Q']}: mean(65536) = {fmt(dd['mean_65536'])} (n={dd['n_65536']}); N=inf mean {fmt(dd['mean_inf'])} CI95 [{fmt(dd['ci95_inf'][0])}, {fmt(dd['ci95_inf'][1])}] (n={dd['n_inf']}) → inside={dd['inside']} ({dd['verdict']}). "
                         f"Diagnostics: Welch diff {fmt(dd['diag_welch_diff'])} CI95 [{fmt(dd['diag_welch_ci95'][0])}, {fmt(dd['diag_welch_ci95'][1])}] p={fmt(dd['diag_welch_p'], 3)}; "
                         f"paired-CRN diff {fmt(dd['diag_paired_CRN_diff'])} CI95 [{fmt(dd['diag_paired_CRN_ci95'][0])}, {fmt(dd['diag_paired_CRN_ci95'][1])}]; "
                         f"P(literal test passes | equal means) ≈ {fmt(dd['diag_null_pass_prob_literal_test'], 3)}")
    L.append('')
    L.append('## Stochastic arms: trajectory of median κ(EEᵀ) and ‖C‖_F during the minibatch phase\n')
    for arm in ['P-SGD', 'P-Adam']:
        for N in NS:
            r = S['cells'][f'{arm}|{N}']['runs']
            if 'kappa_traj_median' in r:
                ks = r['kappa_traj_median']
                sel = [t for t in ks if int(t) % 100000 == 0 or int(t) == 25000]
                L.append(f"- {arm} N={N}: κ " + ', '.join(f't={int(t)//1000}k:{fmt(ks[t], 3)}' for t in sel)
                         + f"; ‖C‖ " + ', '.join(f"{int(t)//1000}k:{fmt(r['Cnorm_traj_median'][t], 3)}" for t in sel)
                         + f"; pre-polish κ q50 {fmt((r['kappa_pre_q10_50_90'] or [np.nan]*3)[1], 3)}, pre-polish rel gap median {fmt(r['rel_gap_pre_median'], 3)}")
    L.append('')
    L.append('## Sensitivity\n')
    sens = S['verdicts']['sensitivity_excluding_diverged_pairs']
    L.append('- Decisions excluding pairs with a diverged run: ' + json.dumps(sens))
    for arm in ['P-SGD', 'P-Adam']:
        L.append(f'- {arm} pre-polish decisions at N=65536: ' + json.dumps({Q: S['cells'][f'{arm}|65536']['prepolish'][Q]['decision'] for Q in QS}))
    L.append('')
    L.append('## Recovery vs the isotropic population-PCA target (median Δ per run)\n')
    for arm in lc.ARMS:
        L.append(f'- {arm}: ' + '; '.join(f"N={N}: " + ', '.join(f"{Q} {fmt(S['cells'][f'{arm}|{N}']['runs']['target_median'][Q], 3)}" for Q in QS)
                                          for N in NS if S['cells'][f'{arm}|{N}']['runs']['target_median']))
    with open(os.path.join(RES, 'S1_3a_linae_summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
