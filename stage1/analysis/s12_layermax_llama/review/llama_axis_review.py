"""Role llama-axis-review: reviewer's own recomputation of the LLaMA-axis Part B verdicts and readings,
plus skeptic diagnostics. Written from the frozen rules in ANALYSIS_PLAN_LLAMA.md sections 5, 7, 9 and
PARTB_PREDICTIONS.md section 5; no code reused from the run roles. Read-only on all inputs; CPU only, 1 thread.
Output: results/S1_2_layermax_llama/review/recomputed_verdicts.json"""
import json, glob, os, itertools, statistics as st
import numpy as np

ROOT = os.path.expanduser('~/hrh_v2_stage1_20261008')
SP = ROOT
LL = ['huggyllama__llama-7b', 'huggyllama__llama-13b', 'huggyllama__llama-30b']
POOLED = ['bigscience__bloomz-560m', 'bigscience__bloomz-1b1', 'bigscience__bloomz-1b7', 'bigscience__bloomz-3b',
          'bigscience__bloomz-7b1', 'openlm-research__open_llama_3b', 'openlm-research__open_llama_7b'] + LL
ARH = dict(zip(POOLED, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]))
BLOOMZ = POOLED[:5]
I = json.load(open(f'{ROOT}/results/S1_2_layermax_llama/indep/summary.json'))
VITS = I['vit_order']
PB = json.load(open(f'{ROOT}/results/S1_2_layermax_llama/partB/partB_features.json'))
DELTA_B = 0.005
PRES = {'prh_clip+l2': 'prh_clip+l2', 'l2': 'l2'}
MET_E = {'mknn': 'mknn', 'cka_b': 'cka_biased', 'cka_d': 'cka_debiased'}
PROT_E = {'full': 'full', 'fixed_r0.5': 'fixed_r0.5', 'fixed_r1.0': 'fixed_r1.0', 'grid17': 'grid17'}


def avgrank(x):
    x = list(x)
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = np.array(avgrank(a)), np.array(avgrank(b))
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def trend(rows, sizes=None):
    """rows: list over ViTs of score lists ordered by size. Returns Tr, p_one(increase), p_one(decrease)."""
    n = len(rows[0])
    sizes = list(range(1, n + 1)) if sizes is None else sizes
    def tr(lab):
        vals = [spearman(lab, r) for r in rows if len(set(r)) > 1]
        return float(np.mean(vals)), len(rows) - len(vals)
    obs, nconst = tr(sizes)
    perms = [tr(list(p))[0] for p in itertools.permutations(sizes)]
    p_up = sum(t >= obs - 1e-12 for t in perms) / len(perms)
    p_dn = sum(t <= obs + 1e-12 for t in perms) / len(perms)
    return obs, p_up, p_dn, nconst


def load_E(pre):
    d = {}
    for lm in LL:
        for v in VITS:
            j = json.load(open(f'{ROOT}/results/S1_2_layermax_llama/early/pairs/{pre}/{lm}__{v}.json'))
            d[(lm, v)] = j
    return d


def load_F8(pre):
    d = {}
    for f in glob.glob(f'{ROOT}/results/S1_2_layermax/pairs/{pre}/*.json'):
        j = json.load(open(f))
        d[(j['lm'], j['vit'])] = j
    return d


def getE(j, met, prot, field):
    if prot == 'ctm':
        c = j['calib_then_max'][MET_E[met]]
        return {'s_cal': c['M'], 'uncensored': c['M_unc']}[field]
    return j['summaries'][MET_E[met]][prot][field]


def getI(j, met, prot, field):
    return j[met][prot][field]


def rows_for(src, get, met, prot, field, lms=LL):
    return [[get(src[(lm, v)], met, prot, field) for lm in lms] for v in VITS]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


def verdicts(pre, impl):
    F8 = load_F8(pre)
    if impl == 'E':
        src, get = load_E(pre), getE
    else:
        src = {tuple(k.split('|')): v for k, v in I['per_pre'][pre]['pairs_llama'].items()}
        get = getI
    out = {}
    T = {}
    for met in ['mknn', 'cka_b', 'cka_d']:
        for prot in ['full', 'fixed_r0.5', 'fixed_r1.0', 'grid17']:
            for fld in ['s_obs', 's_cal', 'uncensored', 'c_alpha']:
                T[(met, prot, fld)] = trend(rows_for(src, get, met, prot, fld))
        for fld in ['s_cal', 'uncensored']:
            T[(met, 'ctm', fld)] = trend(rows_for(src, get, met, 'ctm', fld))
    tr = lambda m, p, f: T[(m, p, f)][0]
    # P-L1
    d1 = tr('mknn', 'full', 's_cal') - tr('mknn', 'full', 's_obs')
    d1u = tr('mknn', 'full', 'uncensored') - tr('mknn', 'full', 's_obs')
    out['P-L1'] = (round(tr('mknn', 'full', 's_obs'), 4), round(d1, 4), round(d1u, 4),
                   'holds' if abs(d1) < 0.10 and abs(d1u) < 0.10 else 'FAILS')
    # P-L2 (materiality from Part B)
    mat = PB['per_pre'][pre]['llama_axis']['Bbar']
    material = all(abs(abs(mat['trend']['Tr']) - 1) <= 1e-12 for _ in [0]) and all(
        g['median'] >= DELTA_B for g in mat['gaps'].values())
    s = 1 if mat['trend']['Tr'] > 0 else -1
    def four(dtr, raw):
        if not material:
            return 'no directional prediction (not material)'
        if abs(raw + s) <= 1e-12:
            return 'not testable'
        if -s * dtr >= 0.10:
            return 'confirmed'
        if s * dtr >= 0.10:
            return 'contradicted'
        return 'consistent-null'
    raw = tr('cka_b', 'full', 's_obs')
    d2 = tr('cka_b', 'full', 's_cal') - raw
    d2u = tr('cka_b', 'full', 'uncensored') - raw
    out['P-L2'] = (round(raw, 4), round(tr('cka_b', 'full', 's_cal'), 4), round(d2, 4), four(d2, raw),
                   round(tr('cka_b', 'full', 'uncensored'), 4), four(d2u, raw), 'p_down', T[('cka_b', 'full', 's_cal')][2])
    # P-L2a
    Bbar = PB['per_pre'][pre]['Bbar']
    absdiff = [abs(get(src[(lm, v)], 'cka_b', 'full', 'c_alpha') - Bbar[lm][v]) for lm in LL for v in VITS]
    nq = nag = 0
    for a, b in [(0, 1), (1, 2), (0, 2)]:
        for v in VITS:
            g = Bbar[LL[b]][v] - Bbar[LL[a]][v]
            if abs(g) >= DELTA_B:
                nq += 1
                ca, cb = (get(src[(LL[a], v)], 'cka_b', 'full', 'c_alpha'), get(src[(LL[b], v)], 'cka_b', 'full', 'c_alpha'))
                nag += (cb - ca) * g > 0
    need = int(np.ceil(2 * nq / 3))
    out['P-L2a'] = (round(st.median(absdiff), 5), f'{nag}/{nq} (need {need})',
                    'holds' if st.median(absdiff) <= 0.01 and nag >= need else 'FAILS')
    # P-L3
    p3 = {}
    for r in ['fixed_r0.5', 'fixed_r1.0']:
        rr = tr('cka_b', r, 's_obs'); sc = tr('cka_b', r, 's_cal')
        if abs(rr) < 0.10:
            p3[r] = (round(rr, 3), round(sc, 3), 'not applicable')
        else:
            sr = 1 if rr > 0 else -1
            o = 'keeps sign' if sr * sc >= 0.10 else ('reversed' if sr * sc <= -0.10 else 'lost')
            p3[r] = (round(rr, 3), round(sc, 3), o)
    out['P-L3'] = p3
    # P-L5
    m = tr('cka_b', 'ctm', 's_cal')
    s5 = 1 if raw > 0 else -1
    out['P-L5'] = (round(raw, 3), round(m, 3), round(tr('cka_b', 'ctm', 'uncensored'), 3),
                   'not applicable' if abs(raw) < 0.10 else ('keeps sign' if s5 * m >= 0.10 else ('reversed' if s5 * m <= -0.10 else 'lost')))
    # pooled 10-LM r
    def pooled(met, fld, lms=POOLED):
        xs, ys = [], []
        for lm in lms:
            for v in VITS:
                if lm in LL:
                    val = get(src[(lm, v)], met, 'full', fld)
                else:
                    val = getE(F8[(lm, v)], met, 'full', fld)
                xs.append(ARH[lm]); ys.append(val)
        return pearson(xs, ys)
    pr = {met: {f: pooled(met, f) for f in ['s_obs', 's_cal', 'uncensored']} for met in ['mknn', 'cka_b', 'cka_d']}
    # P-L4
    d4 = tr('cka_d', 'full', 's_cal') - tr('cka_d', 'full', 's_obs')
    d4u = tr('cka_d', 'full', 'uncensored') - tr('cka_d', 'full', 's_obs')
    out['P-L4'] = (round(d4, 4), round(d4u, 4), round(pr['cka_d']['s_obs'], 4), round(pr['cka_d']['s_cal'], 4),
                   'holds' if d4 >= -0.10 and d4u >= -0.10 and pr['cka_d']['s_cal'] >= pr['cka_d']['s_obs'] - 0.05 else 'FAILS')
    # readings
    def arh_reading(met):
        dc = pr[met]['s_obs'] - pr[met]['s_cal']; dm = pr['mknn']['s_obs'] - pr['mknn']['s_cal']
        gap = pr['mknn']['s_cal'] - pr[met]['s_cal']
        if dc - dm >= 0.20 and gap >= 0.20:
            v = 'reappears'
        elif dc - dm <= 0.10 or gap <= 0.10:
            v = 'absent'
        else:
            v = 'partial'
        return (round(pr[met]['s_obs'], 3), round(pr[met]['s_cal'], 3), round(pr['mknn']['s_obs'], 3),
                round(pr['mknn']['s_cal'], 3), round(dc - dm, 3), round(gap, 3), v)
    out['ARH_reading_cka_b'] = arh_reading('cka_b')
    out['ARH_reading_cka_d'] = arh_reading('cka_d')
    tu_c, tu_m = tr('cka_b', 'full', 'uncensored'), tr('mknn', 'full', 'uncensored')
    if raw < 0.60:
        lr = 'no raw trend to lose'
    elif tu_c <= 0.40 and tu_m >= 0.60:
        lr = 'loses'
    elif tu_c >= 0.60:
        lr = 'no loss'
    else:
        lr = 'intermediate'
    out['LLaMA_reading'] = (round(raw, 3), round(tu_c, 3), round(tu_m, 3), lr)
    out['grid17_cka_b'] = tuple(round(tr('cka_b', 'grid17', f), 3) for f in ['s_obs', 's_cal', 'uncensored', 'c_alpha'])
    out['Tr_calpha_full'] = round(tr('cka_b', 'full', 'c_alpha'), 3)
    # BLOOMZ from the frozen 8-LM run (consistency with the known 0.89 -> -0.02 / -0.29)
    brow = lambda f: [[getE(F8[(lm, v)], 'cka_b', 'full', f) for lm in BLOOMZ] for v in VITS]
    out['BLOOMZ_cka_b_full'] = tuple(round(trend(brow(f))[0], 3) for f in ['s_obs', 's_cal', 'uncensored', 'c_alpha'])
    return out, T



def diagnostics(pre):
    """Skeptic diagnostics (descriptive): multiplicity vs level, threshold-setting layer, PR plateau,
    raw vs threshold gaps, per-ViT rho counts, joint BLOOMZ+LLaMA delta trend (exploratory)."""
    import collections
    out = {}
    pl = I['per_pre'][pre]['pairs_llama']
    for lm in LL:
        cb = [pl[f'{lm}|{v}']['cka_b'] for v in VITS]
        mk = [pl[f'{lm}|{v}']['mknn'] for v in VITS]
        out[lm] = {
            'LA': pl[f'{lm}|{VITS[0]}']['LA'],
            'mean_n_layer_pairs': float(np.mean([pl[f'{lm}|{v}']['LA'] * pl[f'{lm}|{v}']['LB'] for v in VITS])),
            'cka_b_max_abs_calpha_minus_max_entry_c': max(abs(c['diag']['calpha_minus_max_entry_c']) for c in cb),
            'cka_b_max_abs_E0max_minus_max_entry_E0': max(abs(c['diag']['level_minus_multiplicity']) for c in cb),
            'cka_b_null_mean_argmax_LM_layers': sorted(set(c['diag']['argmax_entry_null_mean'][0] for c in cb)),
            'cka_b_null_mean_argmax_rel': sorted(set(round(c['diag']['argmax_entry_null_mean_rel_a'], 4) for c in cb)),
            'cka_b_obs_argmax_rel_mean': float(np.mean([c['full']['rel_a'] for c in cb])),
            'cka_b_ctm_argmax_rel_mean': float(np.mean([c['ctm']['rel_a'] for c in cb])),
            'cka_b_ctm_c_entry_mean': float(np.mean([c['ctm']['c_entry_at_argmax'] for c in cb])),
            'mknn_calpha_mean': float(np.mean([c['full']['c_alpha'] for c in mk])),
            'mknn_E0max_minus_max_entry_E0_mean': float(np.mean([c['diag']['level_minus_multiplicity'] for c in mk])),
        }
        pr = np.array(PB['per_pre'][pre]['lm'][lm]['PR']); L = len(pr) - 1; rel = np.arange(len(pr)) / L
        out[lm]['PR_star'] = float(pr.max()); out[lm]['PR_argmax'] = int(pr.argmax())
        out[lm]['PR_median_rel_0.70_0.95'] = float(np.median(pr[(rel >= 0.70) & (rel <= 0.95)]))
        out[lm]['PR_layers_within_2pct_of_max'] = int((pr >= 0.98 * pr.max()).sum())
    E = load_E(pre); F8 = load_F8(pre)
    rows = lambda src, lms, f: [[getE(src[(lm, v)], 'cka_b', 'full', f) for lm in lms] for v in VITS]
    out['rho_counts'] = {f: dict(collections.Counter(round(spearman([1, 2, 3], r), 2) for r in rows(E, LL, f)))
                         for f in ['s_obs', 's_cal', 'uncensored']}
    T = np.array(rows(E, LL, 's_obs')); C = np.array(rows(E, LL, 'c_alpha'))
    out['gaps'] = {f'{a}-{b}': {'median_raw_gap': float(np.median(T[:, b] - T[:, a])),
                                'median_calpha_gap': float(np.median(C[:, b] - C[:, a])),
                                'n_vit_raw_gap_lt_calpha_gap': int((T[:, b] - T[:, a] < C[:, b] - C[:, a]).sum())}
                   for a, b in [(0, 1), (1, 2), (0, 2)]}
    def trl(rs, lab):
        return float(np.mean([spearman(lab, r) for r in rs if len(set(r)) > 1]))
    joint = {}
    for f in ['s_cal', 'uncensored']:
        Br, Bs, Lr, Ls = rows(F8, BLOOMZ, 's_obs'), rows(F8, BLOOMZ, f), rows(E, LL, 's_obs'), rows(E, LL, f)
        d = lambda pb, pl_: 0.5 * (trl(Bs, pb) - trl(Br, pb)) + 0.5 * (trl(Ls, pl_) - trl(Lr, pl_))
        obs = d([1, 2, 3, 4, 5], [1, 2, 3])
        allv = [d(list(a), list(b)) for a in itertools.permutations([1, 2, 3, 4, 5]) for b in itertools.permutations([1, 2, 3])]
        joint[f] = {'dTr_BL': obs, 'p_lower_exact_720': sum(v <= obs + 1e-12 for v in allv) / 720,
                    'label': 'exploratory, not preregistered (BLOOMZ half known before the plan)'}
    out['joint_dTr_BL'] = joint
    return out


if __name__ == '__main__':
    res = {'verdicts': {}, 'diagnostics': {}}
    for pre in ['prh_clip+l2', 'l2']:
        for impl in ['E', 'I']:
            o, T = verdicts(pre, impl)
            res['verdicts'][f'{pre}|{impl}'] = o
            print(f'===== {pre} | {impl}')
            for k, v in o.items():
                print(f'  {k}: {v}')
        res['diagnostics'][pre] = diagnostics(pre)
        print(json.dumps(res['diagnostics'][pre], indent=1, default=str))
    os.makedirs(f'{ROOT}/results/S1_2_layermax_llama/review', exist_ok=True)
    json.dump(res, open(f'{ROOT}/results/S1_2_layermax_llama/review/recomputed_verdicts.json', 'w'), indent=1, default=str)
