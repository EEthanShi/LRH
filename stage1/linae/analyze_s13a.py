"""S1.3a analysis: pass counts, one-sided 95% Clopper-Pearson decisions, prediction verdicts.
Reads criteria/*.csv; writes ../results/S1_3a_linae_summary.json and .md. Verdict rules are frozen in
ANALYSIS_PLAN.md (written before the confirmatory runs).
v1.1 (2026-10-08; ANALYSIS_PLAN_v1.1.md): the failure rule for diverged runs is explicit (a pair passes a criterion
only if neither run diverged and its discrepancy is finite and <= eps); run outcomes (certified / uncertified /
diverged, with phase) are tabulated; N=inf first-48-pair decisions are shown in the markdown."""
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


def pass_mask(df, Q, factor=1.0):
    """Conservative pass rule (ANALYSIS_PLAN item 6, v1.1 explicit): a pair passes criterion Q iff neither run
    diverged and Delta_Q is finite and <= factor*eps_Q. Diverged runs / metric errors are failures."""
    key = lm.DELTA_KEY[Q]
    n = len(df)
    delta = df[key].to_numpy(float) if key in df else np.full(n, np.nan)
    div = (df['divA'].to_numpy(bool) | df['divB'].to_numpy(bool)) if n else np.zeros(0, bool)
    return np.isfinite(delta) & (np.nan_to_num(delta, nan=np.inf) <= factor * lm.EPS[Q]) & ~div, delta, div


def cell_decisions(df):
    out = {}
    n = len(df)
    for Q in QS:
        pm, delta, div = pass_mask(df, Q)
        passes = int(pm.sum())
        dec, lo, hi = lm.decision(passes, n)
        m, clo, chi, nf = tci(delta)
        fin = delta[np.isfinite(delta)]
        curve = {f'{f:g}eps': (float(np.mean(pass_mask(df, Q, f)[0])) if n else np.nan)
                 for f in (0.5, 1, 2, 5, 10)}
        out[Q] = dict(n_pairs=n, passes=passes, n_fail_by_divergence=int(div.sum()),
                      n_metric_nan_nondiverged=int((~np.isfinite(delta) & ~div).sum()),
                      p_hat=passes / n if n else np.nan, cp_lower=lo, cp_upper=hi,
                      decision=dec, mean=m, mean_ci95=[clo, chi], n_finite=nf,
                      median=float(np.median(fin)) if len(fin) else np.nan,
                      q10=float(np.quantile(fin, .1)) if len(fin) else np.nan,
                      q90=float(np.quantile(fin, .9)) if len(fin) else np.nan,
                      p_hat_curve=curve)
    return out


def main():
    pairs = pd.concat([pd.read_csv(f, dtype={'N': str}) for f in sorted(glob.glob(os.path.join(CRIT, '*_pairs.csv')))])
    runs = pd.concat([pd.read_csv(f, dtype={'N': str}) for f in sorted(glob.glob(os.path.join(CRIT, '*_runs.csv')))])
    def sha(pth):
        return hashlib.sha256(open(pth, 'rb').read()).hexdigest() if os.path.exists(pth) else None
    S = dict(meta=dict(generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       analysis_plan_sha256={'ANALYSIS_PLAN.md': sha(os.path.join(HERE, 'ANALYSIS_PLAN.md')),
                                             'ANALYSIS_PLAN_v1.1.md': sha(os.path.join(HERE, 'ANALYSIS_PLAN_v1.1.md'))},
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
                outcome_counts={k: int((r.outcome == k).sum()) for k in ('certified', 'uncertified', 'diverged')},
                n_diverged_stochastic_phase=int((r.diverged & (r.div_phase == 1)).sum()),
                n_diverged_gd_or_polish=int((r.diverged & (r.div_phase == 2)).sum()),
                diverged_runs=[dict(run=int(x.run), phase=('stochastic' if x.div_phase == 1 else 'gd/polish'),
                                    step=int(x.div_step)) for x in r[r.diverged].itertuples()],
                n_pairs_with_diverged_run=int((fin.divA | fin.divB).sum()),
                code_versions=sorted(set(r.code_version.astype(str))),
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
            if 'rel_gap_pre' in r and r['rel_gap_pre'].notna().any():   # stochastic arms only
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
    # descriptive (non-confirmatory) reading of the P4 outcome at the largest N
    desc4 = {}
    for arm in ['P-SGD', 'P-Adam']:
        decs = [cells[f'{arm}|{LARGEST}']['primary'][Q]['decision'] for Q in ['kNN', 'Gs', 'CKA']]
        rr = cells[f'{arm}|{LARGEST}']['runs']
        lab = ('gauge criteria reliable at N=65536 (consistent with self-balancing)' if all(d_ == 'reliable' for d_ in decs)
               else 'gauge criteria unreliable at N=65536 (no self-balancing to the criterion tolerance)' if all(d_ == 'unreliable' for d_ in decs)
               else 'mixed / inconclusive at N=65536')
        desc4[arm] = dict(decisions_kNN_Gs_CKA=decs, reading=lab, C0_norm_median=rr['C0_norm_median'],
                          Cend_norm_median=rr['Cend_norm_median'], kappa_q10_50_90=rr['kappa_q10_50_90'],
                          n_diverged=rr['n_diverged'])
    V['P4'] = dict(statement='P-SGD and P-Adam: no prediction (Ziyin & Chuang self-balancing question); outcome reported.',
                   verdict='NO PREDICTION (outcome reported)', decisions=det4, descriptive_outcome=desc4)
    # sensitivity of verdicts to counting diverged runs as failures
    def dec_excl(arm, N, Q):
        return cells[f'{arm}|{N}']['excl_diverged_pairs'][Q]['decision']
    V['sensitivity_excluding_diverged_pairs'] = {
        'P1_part_a': {f'{Q}|{N}': dec_excl('P-GD', N, Q) for Q in ['kNN', 'Gs', 'CKA'] for N in NS},
        'P2': {f'{arm}|{Q}': dec_excl(arm, LARGEST, Q) for arm in ['WD', 'TIED', 'small-init'] for Q in QS},
        'P3': {arm: dec_excl(arm, LARGEST, 'proj') for arm in PRIMARY}}
    S['verdicts'] = V
    # ------------------------------------------------------------------ run integrity (v1.1 reporting)
    DIAG = os.path.join(HERE, 'diag')
    integ = dict(crash_diagnosis={}, repro_check=None)
    for pth in sorted(glob.glob(os.path.join(DIAG, 'crash_*.json'))):
        dj = json.load(open(pth))
        integ['crash_diagnosis'][f"{dj['arm']}|{dj['N']}|chunk{dj['chunk']}"] = dict(
            n_dead_stochastic=dj['n_dead_stochastic'], dead=dj['dead'], v1_polish=dj['v1_polish'])
    if os.path.exists(os.path.join(DIAG, 'repro_check.json')):
        rj = json.load(open(os.path.join(DIAG, 'repro_check.json')))
        integ['repro_check'] = dict(arm=rj['arm'], N=rj['N'], chunk=rj['chunk'], all_equal=rj['all_equal'],
                                    n_arrays_compared=len(rj['compared']),
                                    unequal=[k for k, v in rj['compared'].items() if not v])
    integ['totals'] = dict(
        n_runs=int(len(runs)), n_diverged=int(runs.diverged.sum()),
        n_uncertified_finite=int((runs.outcome == 'uncertified').sum()),
        diverged_by_cell={k: v['runs']['n_diverged'] for k, v in cells.items() if v['runs']['n_diverged']},
        pair_metric_errors=int(pairs['metric_error'].fillna('').astype(str).str.len().gt(0).sum()) if 'metric_error' in pairs else 0,
        runs_by_code_version={str(k): int(v) for k, v in runs.code_version.astype(str).value_counts().items()})
    S['integrity'] = integ
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
             'Verdict rules: `linae/ANALYSIS_PLAN.md` (frozen before the confirmatory runs) and the v1.1 addendum '
             '`linae/ANALYSIS_PLAN_v1.1.md` (crash fix; no verdict rule changed).\n')
    L.append('## Verdicts\n')
    for k, v in S['verdicts'].items():
        if isinstance(v, dict) and 'verdict' in v:
            L.append(f"- **{k}** — {v['statement']} → **{v['verdict']}**")
    L.append('')
    L.append('## Decision table (one-sided 95% Clopper–Pearson; reliable if lower > 0.9, unreliable if upper < 0.8)\n')
    L.append('Cells: passes/pairs (one-sided 95% Clopper–Pearson lower–upper), decision, mean Δ over finite pairs [95% t-CI]. '
             'Finite N: 48 disjoint pairs (96 runs). N=inf: all 500 disjoint pairs (1000 inits) are primary; the first 48 pairs are shown on the next row. '
             'A pair with a diverged run fails every criterion.\n')
    L.append('| arm | N | kNN (1−s_N(10) ≤ 0.05) | Gs_rms ≤ 0.02 | 1−CKA ≤ 0.01 | proj ≤ 0.01 | diverged | cert fail |')
    L.append('|---|---|---|---|---|---|---|---|')
    for arm in lc.ARMS:
        for N in NS:
            c = S['cells'][f'{arm}|{N}']
            row = [arm + ('' if lc.ARMS[arm]['primary'] else ' (secondary)'), N]
            for Q in QS:
                q = c['primary'][Q]
                row.append(f"{q['passes']}/{q['n_pairs']} (CP {q['cp_lower']:.3f}–{q['cp_upper']:.3f}) **{q['decision']}**; {fmt(q['mean'], 3)} [{fmt(q['mean_ci95'][0], 3)}, {fmt(q['mean_ci95'][1], 3)}]")
            row.append(str(c['runs']['n_diverged']))
            row.append(str(c['runs']['n_cert_fail_nondiverged']))
            L.append('| ' + ' | '.join(row) + ' |')
            if N == 'inf':
                row = [arm + ('' if lc.ARMS[arm]['primary'] else ' (secondary)'), 'inf, first 48 pairs']
                for Q in QS:
                    q = c['first48'][Q]
                    row.append(f"{q['passes']}/{q['n_pairs']} {q['decision']}; {fmt(q['mean'], 3)}")
                row += ['', '']
                L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('## Run outcomes and divergences (diverged runs are kept; every pair containing one fails every criterion)\n')
    L.append('| arm | N | runs | certified | uncertified (finite) | diverged (stochastic / GD-polish) | pairs hit | diverged runs (run@step) | code |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for arm in lc.ARMS:
        for N in NS:
            r = S['cells'][f'{arm}|{N}']['runs']
            oc = r['outcome_counts']
            dl = ', '.join(f"{x['run']}@{x['step']}" + ('' if x['phase'] == 'stochastic' else '(gd)') for x in r['diverged_runs'][:12])
            if len(r['diverged_runs']) > 12:
                dl += f", … (+{len(r['diverged_runs']) - 12})"
            L.append(f"| {arm} | {N} | {r['n_runs']} | {oc['certified']} | {oc['uncertified']} | {oc['diverged']} "
                     f"({r['n_diverged_stochastic_phase']} / {r['n_diverged_gd_or_polish']}) | {r['n_pairs_with_diverged_run']} | {dl} | {'+'.join(r['code_versions'])} |")
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
    L.append('## P4 (no prediction): descriptive outcome at N=65536\n')
    for arm, dd in S['verdicts']['P4']['descriptive_outcome'].items():
        L.append(f"- {arm}: kNN/Gs/CKA = {'/'.join(dd['decisions_kNN_Gs_CKA'])} → {dd['reading']}; median ‖C₀‖ {fmt(dd['C0_norm_median'], 3)} → ‖C_end‖ {fmt(dd['Cend_norm_median'], 3)}; "
                 f"κ(EEᵀ) q10/50/90 {'/'.join(fmt(x, 3) for x in (dd['kappa_q10_50_90'] or [np.nan] * 3))}; diverged runs {dd['n_diverged']}")
    L.append('')
    L.append('## Run integrity (v1.1)\n')
    ig = S['integrity']
    for k, v in ig['crash_diagnosis'].items():
        L.append(f"- v1.0 crash reproduction {k}: {v['n_dead_stochastic']} run(s) diverged in the stochastic phase "
                 + (('(' + ', '.join(f"run {x['run']} by step {x['died_by_step']}" for x in v['dead']) + ') ') if v['dead'] else '')
                 + f"; v1.0 polish → {v['v1_polish']}")
    if ig['repro_check']:
        rc = ig['repro_check']
        L.append(f"- Bitwise reproduction of a v1.0 chunk by v1.1 ({rc['arm']} N={rc['N']} chunk {rc['chunk']}): all_equal={rc['all_equal']} "
                 f"over {rc['n_arrays_compared']} stored arrays" + (f"; unequal: {rc['unequal']}" if rc['unequal'] else ''))
    t_ = ig['totals']
    L.append(f"- Totals: {t_['n_runs']} runs; diverged {t_['n_diverged']} ({json.dumps(t_['diverged_by_cell'])}); finite but uncertified {t_['n_uncertified_finite']}; "
             f"pair metric errors {t_['pair_metric_errors']}; runs by code version {json.dumps(t_['runs_by_code_version'])}")
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
