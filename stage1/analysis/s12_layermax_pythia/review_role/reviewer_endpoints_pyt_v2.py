"""Role pyt-review (S8): the reviewer's own small recomputation of the headline endpoints of the Pythia axis.
NOT frozen code, enters no verdict; a cross-check of R, E and I written from the plan text (sections 5, 7, 9) only,
without importing any E, I or R module. Inputs: I's full observed/null matrices (indep/<P>/pairs/*.npz), E's pair
summaries (early/pairs/<P>/*.json), the frozen Part B (partB_features_pyt.json), hrhv2.util.rng_from_key.
It also measures, from the two implementations' pair files, whether the V-P5 float differences can change any order,
censoring or decision. Writes one JSON (path argv[1]); prints a short digest. CPU only."""
import itertools, json, math, os, sys, time
from fractions import Fraction
import numpy as np

PROJ = os.path.expanduser('~/hrh_v2_stage1_20261008')
sys.path.insert(0, PROJ)
from hrhv2.util import rng_from_key  # noqa: E402

RES = os.path.join(PROJ, 'results', 'S1_2_layermax_pythia')
PB = json.load(open(os.path.join(RES, 'partB', 'partB_features_pyt.json')))
SIZES, LMS, VITS = PB['sizes'], PB['pyt_lms'], PB['vits']
NS, NV = len(SIZES), len(VITS)
IDX = 475  # 476th ascending order statistic of {T} u {T_k}, K = 500 (plan section 3)
MK = 10240
PRES = ('prh_clip+l2', 'l2')


def avgrank(x):
    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def valid(row):
    row = np.asarray(row, dtype=np.float64)
    return bool(np.all(np.isfinite(row)) and len(set(row.tolist())) > 1)


def rho_perm(rows, perms):
    """rows: (nv, n) scores; perms: (np, n) size-rank vectors. Returns (np, nv) Spearman rho."""
    R = np.array([avgrank(r) for r in rows])
    Rc = R - R.mean(1, keepdims=True)
    Pc = perms - perms.mean(1, keepdims=True)
    num = Pc @ Rc.T
    den = np.sqrt((Pc ** 2).sum(1))[:, None] * np.sqrt((Rc ** 2).sum(1))[None, :]
    return num / den


def diff_stats(base, X, s, idx=None, boot_key=None, margin=0.10):
    """base, X: dict size -> (nv,) arrays. Common ViT set, D = s*mean(drho), exact p over relabellings, bootstrap."""
    idx = list(range(NS)) if idx is None else idx
    B = np.array([[base[SIZES[i]][v] for i in idx] for v in range(NV)])
    Y = np.array([[X[SIZES[i]][v] for i in idx] for v in range(NV)])
    V = [v for v in range(NV) if valid(B[v]) and valid(Y[v])]
    n = len(idx)
    perms = np.array(list(itertools.permutations(range(1, n + 1))), dtype=np.float64)
    ident = int(np.where((perms == np.arange(1, n + 1)).all(1))[0][0])
    rb, ry = rho_perm(B[V], perms), rho_perm(Y[V], perms)
    dtr = (ry - rb).mean(1)
    D = s * dtr[ident]
    p_perm = float(np.sum(s * dtr <= D + 1e-9)) / len(perms)
    dr = (ry - rb)[ident]
    out = {'nV': len(V), 'R': float(rb[ident].mean()), 'C': float(ry[ident].mean()), 'D': float(D),
           'p_perm': p_perm, 'p_perm_count': int(np.sum(s * dtr <= D + 1e-9)), 'N': len(perms),
           'degenerate': bool(np.all(dr == 0.0)), 'ranks_identical': bool(all(
               np.array_equal(avgrank(Y[v]), avgrank(B[v])) for v in V))}
    if boot_key is not None:
        rng = rng_from_key(boot_key)
        draws = np.array([s * np.mean(dr[rng.integers(0, len(V), len(V))]) for _ in range(2000)])
        out['p_mag'] = (1 + int(np.sum(draws >= -margin - 1e-9))) / 2001
        out['ci'] = [float(x) for x in np.percentile(draws, [2.5, 97.5])]
        out['p_lo'] = (1 + int(np.sum(draws <= -margin + 1e-9))) / 2001
        out['p_hi'] = (1 + int(np.sum(draws >= margin - 1e-9))) / 2001
        out['p_tost'] = max(out['p_lo'], out['p_hi']) if -margin < D * s < margin else 1.0
    return out


def scal(T, c):
    return max((T - c) / (1 - c), 0.0)


def load(P):
    """Per size: per ViT dict of quantities from I's npz (recomputed here) and from E's json."""
    I, E = {}, {}
    for s, lm in zip(SIZES, LMS):
        I[s], E[s] = [], []
        for v in VITS:
            z = np.load(os.path.join(RES, 'indep', P, 'pairs', f'{lm}__{v}.npz'))
            q = {}
            for met, ko, kn in (('cka_biased', 'obs_cka_b', 'null_cka_b'), ('cka_debiased', 'obs_cka_d', 'null_cka_d')):
                obs, nul = z[ko].astype(np.float64), z[kn].astype(np.float64)
                T = float(obs.max())
                nm = nul.reshape(nul.shape[0], -1).max(1)
                c = float(np.sort(np.concatenate([[T], nm]))[IDX])
                mu = float(nm.mean())
                q[met] = {'T': T, 'c': c, 's_cal': scal(T, c), 'mu': mu, 'unc': T - mu}
                if met == 'cka_biased':  # calibrate-then-max (entry-wise threshold, then max)
                    allv = np.concatenate([obs[None], nul], 0)
                    ce = np.sort(allv, 0)[IDX]
                    sc = np.maximum((obs - ce) / (1 - ce), 0.0)
                    a = int(np.argmax(sc))
                    q[met]['M'] = float(sc.max())
                    q[met]['cM'] = float(ce.ravel()[a])
            oc, nc = z['obs_cnt'].astype(np.int64), z['null_cnt'].astype(np.int64)
            T = int(oc.max())
            nm = nc.reshape(nc.shape[0], -1).max(1)
            c = int(np.sort(np.concatenate([[T], nm]))[IDX])
            sc = Fraction(T - c, MK - c) if T > c else Fraction(0)
            q['mknn'] = {'T': T / MK, 'c': c / MK, 's_cal': float(sc), 's_cal_exact': sc, 'Tint': T, 'cint': c,
                         'mu': float(Fraction(int(nm.sum()), 500 * MK)), 'unc': float(Fraction(T, MK) - Fraction(int(nm.sum()), 500 * MK))}
            I[s].append(q)
            r = json.load(open(os.path.join(RES, 'early', 'pairs', P, f'{lm}__{v}.json')))
            e = {}
            for met in ('mknn', 'cka_biased', 'cka_debiased'):
                f = r['summaries'][met]['full']
                e[met] = {'T': f['s_obs'], 'c': f['c_alpha'], 's_cal': f['s_cal'], 'mu': f['null_mean'], 'unc': f['uncensored']}
            E[s].append(e)
    return I, E


def vec(D, met, key):
    return {s: np.array([D[s][v][met][key] for v in range(NV)], dtype=np.float64) for s in SIZES}


def sgn(x):
    return int(x > 0) - int(x < 0)


def endpoints(P, I):
    pp = PB['per_pre'][P]
    s, s_late = pp['s'], pp['s_late']
    T = vec(I, 'cka_biased', 'T'); S = vec(I, 'cka_biased', 's_cal'); Cth = vec(I, 'cka_biased', 'c')
    Mv = vec(I, 'cka_biased', 'M')
    out = {}
    for x, key in (('s_cal', 's_cal'), ('unc', 'unc')):
        X = vec(I, 'cka_biased', key)
        suf = '' if x == 's_cal' else '|unc'
        h1 = diff_stats(T, X, s, boot_key=f'S1.2-layermax-pythia|boot|H1mag|cka_biased|full|{P}{suf}')
        h1['p'] = max(h1['p_perm'], h1['p_mag'])
        member = 'H1' in pp['members']
        h1['confirmed'] = (h1['p'] <= 0.05 and h1['D'] <= -0.10) if member else None
        h2 = diff_stats(T, Mv, s_late, boot_key=f'S1.2-layermax-pythia|boot|H2mag|cka_biased|ctm|{P}{suf}') if x == 's_cal' else None
        tost = {}
        for met in ('mknn', 'cka_debiased', 'cka_biased'):
            Tm, Xmm = vec(I, met, 'T'), vec(I, met, key)
            tost[met] = diff_stats(Tm, Xmm, 1, boot_key=f'S1.2-layermax-pythia|boot|tost|{met}|full|{P}{suf}')
        if x == 's_cal':
            tost['ctm'] = diff_stats(T, Mv, 1, boot_key=f'S1.2-layermax-pythia|boot|tost|cka_biased|ctm|{P}')
        # gate and Holm
        gate = ('open (H1 confirmed)' if h1['confirmed'] else 'closed') if member else 'open (H1 not a member)'
        holm = None
        if gate.startswith('open') and x == 's_cal':
            ph2 = max(h2['p_perm'], h2['p_mag']) if pp['H2_branch'] == 'H2a' else tost['ctm']['p_tost']
            ps = [('H2', ph2), ('H3', tost['mknn']['p_tost']), ('H4', tost['cka_debiased']['p_tost'])]
            ps.sort(key=lambda t: t[1])
            rej, go = {}, True
            for i, (nm, p) in enumerate(ps):
                go = go and p <= 0.05 / (3 - i)
                rej[nm] = go
            holm = {'order': ','.join(n for n, _ in ps), 'reject': rej}
        D1 = h1['D']
        if not member:
            cat = 'E'
        elif D1 >= 0.10:
            cat = 'D'
        elif abs(D1) < 0.10:
            cat = 'C'
        elif not h1['confirmed']:
            cat = 'F'
        else:
            cat = 'A/A-att/A-prime/B (needs P-P2)'
        out[x] = {'H1': h1, 'H2': h2, 'TOST': tost, 'gate': gate, 'holm': holm, 'category': cat}
    # P-P2a(i), (ii), P-P2b, P-P5a, M4, f-/f+, g, h, LOO, five-size, grid not recomputed here
    Bbar = {s_: np.array([pp['Bbar'][lm][v] for v in VITS]) for s_, lm in zip(SIZES, LMS)}
    Blate = {s_: np.array([pp['Blate'][lm][v] for v in VITS]) for s_, lm in zip(SIZES, LMS)}
    band = pp['band']
    med = {s_: float(np.median(Cth[s_] - Bbar[s_])) for s_ in SIZES}
    inb = {s_: band[s_]['lo'] <= med[s_] <= band[s_]['hi'] for s_ in SIZES}
    pairs = [(a, b) for a in range(NS) for b in range(a + 1, NS)]
    n2 = x2 = 0
    n5 = x5 = 0
    fc = {'below': 0, 'above': 0, 'tie': 0, 'undecided': 0, 'wrong': 0, 'correct': 0}
    fminus = fplus = 0
    cM = vec(I, 'cka_biased', 'cM')
    for a, b in pairs:
        sa, sb = SIZES[a], SIZES[b]
        for v in range(NV):
            g = Bbar[sb][v] - Bbar[sa][v]
            if abs(g) >= 0.005:
                n2 += 1
                x2 += sgn(Cth[sb][v] - Cth[sa][v]) == sgn(g)
            gl = Blate[sb][v] - Blate[sa][v]
            if abs(gl) >= 0.005:
                n5 += 1
                x5 += sgn(cM[sb][v] - cM[sa][v]) == sgn(gl)
            # P-P2b forecast (plan B6)
            lo_a, hi_a = band[sa]['lo'], band[sa]['hi']
            lo_b, hi_b = band[sb]['lo'], band[sb]['hi']
            Ta, Tb = T[sa][v], T[sb][v]
            if scal(Tb, Bbar[sb][v] + lo_b) < scal(Ta, Bbar[sa][v] + hi_a):
                dec = 'below'
            elif scal(Tb, Bbar[sb][v] + hi_b) > scal(Ta, Bbar[sa][v] + lo_a):
                dec = 'above'
            elif scal(Tb, Bbar[sb][v] + lo_b) == 0 and scal(Ta, Bbar[sa][v] + lo_a) == 0:
                dec = 'tie'
            else:
                dec = 'undecided'
            fc[dec] += 1
            if dec != 'undecided':
                obs = 'below' if S[sb][v] < S[sa][v] else ('above' if S[sb][v] > S[sa][v] else 'tie')
                fc['correct' if obs == dec else 'wrong'] += 1
            dr, dx = sgn(Tb - Ta), sgn(S[sb][v] - S[sa][v])
            fminus += dr == 1 and dx == -1
            fplus += dr == -1 and dx == 1
    nzero = int(sum(np.sum(S[s_] == 0) for s_ in SIZES))
    medscal = {s_: float(np.median(S[s_])) for s_ in SIZES}
    N = {s_: PB['num_params'][lm] for s_, lm in zip(SIZES, LMS)}
    dbl = math.log2(N['6.9b'] / N['70m'])
    gg = float(np.median((T['6.9b'] - T['70m']) / dbl))
    hh = float(np.median((Cth['6.9b'] - Cth['70m']) / dbl))
    steps = {f'{SIZES[i]}|{SIZES[i + 1]}': {'raw_gain': float(np.median(T[SIZES[i + 1]] - T[SIZES[i]])),
                                             'thr_gain': float(np.median(Cth[SIZES[i + 1]] - Cth[SIZES[i]]))}
             for i in range(NS - 1)}
    # M1 leave-one-size-out and M3 five-size check (primary x = s_cal)
    h1 = out['s_cal']['H1']
    loo = {}
    for zi, z in enumerate(SIZES):
        idx = [i for i in range(NS) if i != zi]
        r = diff_stats(T, S, s, idx=idx, boot_key=f'S1.2-layermax-pythia|boot|H1mag|cka_biased|full|{P}|loo|{z}')
        r['p'] = max(r['p_perm'], r['p_mag'])
        r['flip'] = ((r['D'] <= -0.10) != (h1['D'] <= -0.10)) or ((r['p'] <= 0.05) != (h1['p'] <= 0.05))
        loo[z] = {k: r[k] for k in ('D', 'p_perm', 'p_mag', 'p', 'flip', 'nV')}
    five = diff_stats(T, S, s, idx=[2, 3, 4, 5, 6])
    five['carried'] = (sgn(five['D']) != sgn(h1['D'])) or ((five['D'] <= -0.10) != (h1['D'] <= -0.10))
    out['common'] = {
        'PP2ai': {'median': med, 'in_band': inb, 'n_models_in_band': int(sum(inb.values())),
                  'holds': sum(inb.values()) >= 6, 'extrapolated': pp['extrapolated']},
        'PP2aii': {'n': n2, 'x': int(x2), 'need': math.ceil(Fraction(9 * n2, 10)), 'holds': x2 >= math.ceil(Fraction(9 * n2, 10))},
        'PP5a': {'n': n5, 'x': int(x5), 'need': math.ceil(Fraction(2 * n5, 3))},
        'PP2b': dict(fc, n_decided=fc['below'] + fc['above'] + fc['tie'],
                     verdict='fails' if fc['wrong'] else ('not evaluable' if fc['below'] + fc['above'] + fc['tie'] < 30 else 'holds')),
        'M4': {'n_scal_zero': nzero, 'median_scal': medscal},
        'f': {'fminus': int(fminus), 'fplus': int(fplus)},
        'g': gg, 'h': hh, 'steps': steps,
        'M1_loo': loo, 'M1_fragile': any(r['flip'] for r in loo.values()),
        'M3_five': {k: five[k] for k in ('D', 'p_perm', 'carried', 'nV')},
        'TrB': pp['Tr_Bbar'], 'members': pp['members'], 's': s, 's_late': s_late, 'H2_branch': pp['H2_branch']}
    return out


def vp5_effect(P, I, E):
    """Does the E-vs-I float difference change any order, censoring or tie, for every metric and score?"""
    rep = {}
    pairs = [(a, b) for a in range(NS) for b in range(a + 1, NS)]
    for met in ('mknn', 'cka_biased', 'cka_debiased'):
        r = {}
        for key in ('T', 'c', 's_cal', 'mu', 'unc'):
            vi, ve = vec(I, met, key), vec(E, met, key)
            ad = max(float(np.max(np.abs(vi[s] - ve[s]))) for s in SIZES)
            rd = max(float(np.max(np.abs(vi[s] - ve[s]) / np.maximum(np.maximum(np.abs(vi[s]), np.abs(ve[s])), 1e-300))) for s in SIZES)
            nfail = int(sum(np.sum(np.abs(vi[s] - ve[s]) > 1e-12 * np.maximum(np.abs(vi[s]), np.abs(ve[s])) + 1e-15) for s in SIZES))
            nord = sum(sgn(vi[SIZES[b]][v] - vi[SIZES[a]][v]) != sgn(ve[SIZES[b]][v] - ve[SIZES[a]][v])
                       for a, b in pairs for v in range(NV))
            ncens = int(sum(np.sum((vi[s] == 0) != (ve[s] == 0)) for s in SIZES)) if key == 's_cal' else None
            # smallest gap between two sizes of one ViT (decision margin of any order) vs the largest E-I difference
            mingap = min(abs(vi[SIZES[b]][v] - vi[SIZES[a]][v]) for a, b in pairs for v in range(NV)
                         if vi[SIZES[b]][v] != vi[SIZES[a]][v])
            r[key] = {'max_abs_diff': ad, 'max_rel_diff': rd, 'n_fail_vp5_tol': nfail, 'n_order_changes_357': int(nord),
                      'n_censoring_changes': ncens, 'min_nonzero_gap_between_sizes': float(mingap)}
        rep[met] = r
    return rep


def main():
    t0 = time.time()
    res = {'what': 'pyt-review reviewer endpoints (not frozen code; enters no verdict)',
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'per_pre': {}, 'vp5_effect': {}}
    for P in PRES:
        I, E = load(P)
        res['per_pre'][P] = endpoints(P, I)
        res['vp5_effect'][P] = vp5_effect(P, I, E)
    res['seconds'] = time.time() - t0
    json.dump(res, open(sys.argv[1], 'w'), indent=1, default=str)
    for P in PRES:
        o = res['per_pre'][P]['s_cal']
        print(P, 'cat', o['category'], 'R', round(o['H1']['R'], 4), 'C', round(o['H1']['C'], 4), 'D1', round(o['H1']['D'], 4),
              'p_perm', o['H1']['p_perm_count'], '/5040', 'p_mag', o['H1']['p_mag'], 'gate', o['gate'], 'holm', o['holm'])


if __name__ == '__main__':
    main()
