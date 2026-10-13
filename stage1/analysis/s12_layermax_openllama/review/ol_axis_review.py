"""OL-AXIS COPY of analysis/s12_layermax_llama/review/llama_axis_review.py (sha256 a96576195e69..., plan s3), role R.
OL-AXIS: reviewer's own recomputation of every OpenLLaMA-axis verdict (VERDICTS_SPEC.md sections A, B, C, D, F, G, J,
OL-AXIS: incl. P-O2b, the section 7 readings and J) from the pair files of E and, separately, of I, then the
OL-AXIS: comparison with E, I, forecast_ol.py and joint_summary.py (compare_verdicts.py; 0 mismatches required, plan
OL-AXIS: section 11 step S8) and the F1(c) birth-time audit. Written from the frozen plan ANALYSIS_PLAN_OPENLLAMA.md
OL-AXIS: (4f18d710...), VERDICTS_SPEC.md and the stored pair formats; no code of E, I, forecast or joint was read.
OL-AXIS: Gated (plan F1(a)): guard.require_partb_frozen(out_root) is the first action of main().
OL-AXIS: CPU only (numpy/scipy; no GPU). Threads: OL_THREADS / OMP_NUM_THREADS from launch_ol.sh (1 to 4).
OL-AXIS:
OL-AXIS: Usage (from ~/hrh_v2_stage1_20261008, through the launcher; see review/CHANGES.md):
OL-AXIS:   analysis/s12_layermax_openllama/launch_ol.sh R_review 4 -- venv/bin/python \
OL-AXIS:       analysis/s12_layermax_openllama/review/ol_axis_review.py [--impl both|E|I] [--phase all|compute|compare]
OL-AXIS: Outputs (all under results/S1_2_layermax_openllama/review/):
OL-AXIS:   E/verdicts_review.json, I/verdicts_review.json   (impl "R-E" / "R-I", every key of required_keys("ALL"))
OL-AXIS:   cmp/<a>_vs_<b>.json                              (compare_verdicts.py --out of each comparison)
OL-AXIS:   audit_F1c.json                                   (guard.py audit, exit code recorded)
OL-AXIS:   recomputed_verdicts.json, REVIEW_REPORT.md       (all comparisons, every mismatch listed in full)
OL-AXIS: Exit code: 0 = every comparison with E, I, forecast and joint passed; 1 = a mismatch or a missing target;
OL-AXIS: 3 = refused by the Part B gate (no '[oll-axis-predict] Part B FROZEN' line, or the hashes differ)."""
import json, glob, os, itertools, statistics as st
import numpy as np
import argparse  # OL-AXIS
import subprocess  # OL-AXIS
import sys  # OL-AXIS
import time  # OL-AXIS

ROOT = os.path.expanduser('~/hrh_v2_stage1_20261008')
SP = ROOT
HERE = os.path.dirname(os.path.abspath(__file__))  # OL-AXIS
AXIS = os.path.join(ROOT, 'analysis', 's12_layermax_openllama')  # OL-AXIS
RES_OL = os.path.join(ROOT, 'results', 'S1_2_layermax_openllama')  # OL-AXIS: this axis (production)
OUT_ROOT = os.path.join(RES_OL, 'review')  # OL-AXIS: the guarded output root of R (plan F1 table)
FROZEN8 = os.path.join(ROOT, 'results', 'S1_2_layermax')  # OL-AXIS: frozen 8-LM run, read-only
LLAMA_AX = os.path.join(ROOT, 'results', 'S1_2_layermax_llama')  # OL-AXIS: LLaMA axis, read-only
MERGED = os.path.join(LLAMA_AX, 'layermax_llama_summary.json')  # OL-AXIS: merged LLaMA summary (db9242f9...)
SLUG13_REAL = 'openlm-research__open_llama_13b'  # OL-AXIS
SLUG13_TEST = 'synth__ol13b'  # OL-AXIS: stand-in slug of the S0b tests (spec section 10)
PRES = ('prh_clip+l2', 'l2')  # OL-AXIS
SECTION_FILES = (('EI', 'verdicts_ol.json'), ('F', 'verdicts_forecast.json'), ('J', 'verdicts_joint.json'))  # OL-AXIS


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


def pearson(x, y):  # OL-AXIS: moved up from below rows_for(), body unchanged
    x, y = np.asarray(x, float), np.asarray(y, float)  # OL-AXIS (moved)
    x = x - x.mean(); y = y - y.mean()  # OL-AXIS (moved)
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))  # OL-AXIS (moved)


# ---------------------------------------------------------------------------------------------- OL-AXIS
# OL-AXIS: input locations. Production: the plan's paths. S0b test mode (guard.test_mode(), env OL_S0B_TEST=1
# OL-AXIS: before CODE FROZEN): --test-root <dir under ~/oll_s0b_scratch> replaces the OpenLLaMA results root
# OL-AXIS: (<root>/results/S1_2_layermax_openllama/...) and the frozen 8-LM pair roots (<root>/frozen8/...), and the
# OL-AXIS: 13b slug becomes synth__ol13b; LLaMA-axis files and the merged summary are always the real ones.
# ---------------------------------------------------------------------------------------------- OL-AXIS
class Paths:  # OL-AXIS
    def __init__(self, test_root=None):  # OL-AXIS
        self.test = test_root is not None  # OL-AXIS
        self.res_ol = os.path.join(test_root, 'results', 'S1_2_layermax_openllama') if self.test else RES_OL  # OL-AXIS
        self.frozen8 = os.path.join(test_root, 'frozen8') if self.test else FROZEN8  # OL-AXIS
        self.slug13 = SLUG13_TEST if self.test else SLUG13_REAL  # OL-AXIS

    def ol_E(self, P, lm, vit):  # OL-AXIS: E writes early/pairs/<pre>/<lm>__<vit>.{json,npz}
        b = os.path.join(self.res_ol, 'early', 'pairs', P, f'{lm}__{vit}')  # OL-AXIS
        return b + '.json', b + '.npz'  # OL-AXIS

    def ol_I(self, P, lm, vit):  # OL-AXIS: I writes indep/<pre>/pairs/<lm>__<vit>.npz
        return os.path.join(self.res_ol, 'indep', P, 'pairs', f'{lm}__{vit}.npz')  # OL-AXIS

    def frz_E(self, P, lm, vit):  # OL-AXIS: frozen 8-LM E-format files (pairs/<pre>/)
        b = os.path.join(self.frozen8, 'pairs', P, f'{lm}__{vit}')  # OL-AXIS
        return b + '.json', b + '.npz'  # OL-AXIS

    def frz_I(self, P, lm, vit):  # OL-AXIS: frozen 8-LM I-format files (indep/<pre>/pairs/)
        return os.path.join(self.frozen8, 'indep', P, 'pairs', f'{lm}__{vit}.npz')  # OL-AXIS

    def lla_E(self, P, lm, vit):  # OL-AXIS: LLaMA-axis E files (plan s3: E reads early/pairs/<pre>/)
        b = os.path.join(LLAMA_AX, 'early', 'pairs', P, f'{lm}__{vit}')  # OL-AXIS
        return b + '.json', b + '.npz'  # OL-AXIS

    def lla_I(self, P, lm, vit):  # OL-AXIS: LLaMA-axis I files (plan s3: I reads indep/<pre>/pairs/)
        return os.path.join(LLAMA_AX, 'indep', P, 'pairs', f'{lm}__{vit}.npz')  # OL-AXIS

    def partb(self):  # OL-AXIS
        return os.path.join(self.res_ol, 'partB', 'partB_features_ol.json')  # OL-AXIS

    def target(self, impl, sec):  # OL-AXIS: files of E, I, forecast_ol.py and joint_summary.py (spec section 0)
        if sec == 'EI':  # OL-AXIS
            return os.path.join(self.res_ol, 'early' if impl == 'E' else 'indep', 'verdicts_ol.json')  # OL-AXIS
        sub = 'forecast' if sec == 'F' else 'joint'  # OL-AXIS
        return os.path.join(self.res_ol, sub, impl, dict(SECTION_FILES)[sec])  # OL-AXIS


def rel(p):  # OL-AXIS
    p = os.path.abspath(p)  # OL-AXIS
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p  # OL-AXIS


def load_impl(impl, paths, VK, RL, log):  # OL-AXIS
    """OL-AXIS: read every pair file R needs for one implementation; returns the context of ol_review_sections."""
    vits = list(VK.VITS)  # OL-AXIS
    slug = {'3b': VK.OL_SLUGS['3b'], '7b': VK.OL_SLUGS['7b'], '13b': paths.slug13}  # OL-AXIS
    pooled11 = [paths.slug13 if s == SLUG13_REAL else s for s in VK.POOLED11]  # OL-AXIS
    bloomz = pooled11[:5]  # OL-AXIS
    llama_slugs = ['huggyllama__llama-7b', 'huggyllama__llama-13b', 'huggyllama__llama-30b']  # OL-AXIS
    inputs, notes = {}, {'e_checks': {}, 'identity_issues': []}  # OL-AXIS
    allow = paths.test  # OL-AXIS: missing subsets tolerated only for test stand-ins and frozen files

    def rd(kind, P, lm, vit, protocols=RL.SUBSETS, ctm=True, allow_missing=False):  # OL-AXIS
        if impl == 'E':  # OL-AXIS
            j, z = getattr(paths, kind + '_E')(P, lm, vit)  # OL-AXIS
            for f in (j, z):  # OL-AXIS
                inputs[rel(f)] = RL.sha256_file(f)  # OL-AXIS
            ps = RL.read_pair_E(j, z, protocols=protocols, want_ctm=ctm, allow_missing=allow_missing)  # OL-AXIS
            ck = ps['checks']  # OL-AXIS
            bad = {k: v for k, v in ck.items() if (k == 'max_rel_mu_f32' and v > 1e-6)  # OL-AXIS
                   or (k == 'missing_subsets' and v and kind == 'ol')  # OL-AXIS
                   or (k not in ('max_rel_mu_f32', 'missing_subsets', 'n_mknn_entry_ulp') and v)}  # OL-AXIS
            if bad:  # OL-AXIS: stored-summary vs recomputation findings, reported in REVIEW_REPORT.md
                notes['e_checks'][f'{kind}|{P}|{lm}|{vit}'] = bad  # OL-AXIS
            notes['e_mknn_entry_ulp'] = notes.get('e_mknn_entry_ulp', 0) + ck.get('n_mknn_entry_ulp', 0)  # OL-AXIS
            if not paths.test and kind == 'ol' and (ps.get('lm') != lm or ps.get('vit') != vit  # OL-AXIS
                                                     or ps.get('perm_key') != f'S1.2-layermax|{lm}|{vit}' or ps.get('K') != 500):  # OL-AXIS
                notes['identity_issues'].append(f'{P}|{lm}|{vit}: lm/vit/perm_key/K in the JSON differ')  # OL-AXIS
        else:  # OL-AXIS
            f = getattr(paths, kind + '_I')(P, lm, vit)  # OL-AXIS
            inputs[rel(f)] = RL.sha256_file(f)  # OL-AXIS
            ps = RL.read_pair_I(f, protocols=protocols, want_ctm=ctm)  # OL-AXIS
            if not paths.test and kind == 'ol':  # OL-AXIS
                z = np.load(f)  # OL-AXIS
                if str(z['key']) != f'S1.2-layermax|{lm}|{vit}' or int(z['K']) != 500:  # OL-AXIS
                    notes['identity_issues'].append(f'{P}|{lm}|{vit}: key/K in the npz differ')  # OL-AXIS
        return ps  # OL-AXIS

    ol, frozen_ol, llama, pool = {}, {}, {}, {}  # OL-AXIS
    t0 = time.time()  # OL-AXIS
    for P in PRES:  # OL-AXIS
        ol[P] = {s: {v: rd('ol', P, slug[s], v, allow_missing=allow) for v in vits} for s in ('3b', '7b', '13b')}  # OL-AXIS
        frozen_ol[P] = {s: {v: rd('frz', P, slug[s], v, allow_missing=True) for v in vits} for s in ('3b', '7b')}  # OL-AXIS
        llama[P] = {lm: {v: rd('lla', P, lm, v, protocols=('full',), ctm=False) for v in vits} for lm in llama_slugs}  # OL-AXIS
        pool[P] = {}  # OL-AXIS
        for lm in pooled11:  # OL-AXIS: sources of spec section 1
            if lm in slug.values():  # OL-AXIS: 3b/7b/13b = this implementation's reruns
                s = [k for k, v in slug.items() if v == lm][0]  # OL-AXIS
                pool[P][lm] = ol[P][s]  # OL-AXIS
            elif lm in ('huggyllama__llama-13b', 'huggyllama__llama-30b'):  # OL-AXIS: own LLaMA-axis files
                pool[P][lm] = llama[P][lm]  # OL-AXIS
            else:  # OL-AXIS: BLOOMZ and llama-7b = frozen 8-LM files of this implementation's format
                pool[P][lm] = {v: rd('frz', P, lm, v, protocols=('full',), ctm=False) for v in vits}  # OL-AXIS
        log(f'{impl} {P}: pair files read ({time.time() - t0:.0f}s)')  # OL-AXIS
    pbp = paths.partb()  # OL-AXIS
    inputs[rel(pbp)] = RL.sha256_file(pbp)  # OL-AXIS
    pb = json.load(open(pbp))  # OL-AXIS
    inputs[rel(MERGED)] = RL.sha256_file(MERGED)  # OL-AXIS
    mg = json.load(open(MERGED))['embedded']['indep_summary']['per_pre']  # OL-AXIS
    merged_pairs = {P: mg[P]['pairs_llama'] for P in PRES}  # OL-AXIS
    n_pool = sum(len(pool[P][lm]) for P in PRES for lm in pooled11)  # OL-AXIS
    return {'ol': ol, 'frozen_ol': frozen_ol, 'llama': llama, 'pool': pool, 'pb': pb, 'slug': slug,  # OL-AXIS
            'pooled11': pooled11, 'bloomz': bloomz, 'llama_slugs': llama_slugs, 'merged_pairs': merged_pairs,  # OL-AXIS
            'vits': vits, 'inputs': inputs, 'notes': notes, 'n_pool': n_pool}  # OL-AXIS


def compute(ctx, VK, RL, RS, log):  # OL-AXIS
    """OL-AXIS: every required key of verdict_keys.required_keys('ALL'), plus x| extras."""
    v, notes = {}, ctx['notes']  # OL-AXIS
    vits, ol, PB = ctx['vits'], ctx['ol'], RS.PartB(ctx['pb'], ctx['slug'])  # OL-AXIS
    T = RS.Trends(ol, vits)  # OL-AXIS
    t0 = time.time()  # OL-AXIS
    RS.section_A(v, T, PRES, RL.METRICS, VK.PROTOCOLS)  # OL-AXIS
    RS.section_B(v, ol, PB, PRES, RL.METRICS, VK.PROTOCOLS, vits)  # OL-AXIS
    log(f'sections A, B done ({time.time() - t0:.0f}s)')  # OL-AXIS
    panels = RS.panel_spec(ctx['pooled11'], ctx['slug']['13b'], ctx['pb']['num_params'])  # OL-AXIS
    D = RS.section_D(v, ctx['pool'], panels, PRES, vits)  # OL-AXIS
    log(f'section D done incl. the 11! enumeration ({time.time() - t0:.0f}s)')  # OL-AXIS
    RS.section_C(v, ol, T, PB, D, PRES, vits)  # OL-AXIS
    fnotes = {}  # OL-AXIS
    RS.section_F(v, ol, PB, PRES, vits, fnotes)  # OL-AXIS
    RS.section_J(v, ol, ctx['pool'], ctx['llama'], PB, 'prh_clip+l2', vits, ctx['bloomz'], ctx['llama_slugs'])  # OL-AXIS
    g1, g1d = RS.g_llama_vs_merged(ctx['llama'], ctx['merged_pairs'], PRES, vits, ctx['llama_slugs'][1:])  # OL-AXIS
    g2, g2d = RS.g_rerun_vs_frozen(ol, ctx['frozen_ol'], PRES, vits)  # OL-AXIS
    v['G|llama_pairs_equal_merged_summary'] = bool(g1)  # OL-AXIS
    v['G|rerun_equals_frozen_3b7b'] = bool(g2)  # OL-AXIS
    v['G|n_pairs_read'] = int(ctx['n_pool'])  # OL-AXIS
    log(f'sections C, F, J, G done ({time.time() - t0:.0f}s)')  # OL-AXIS
    # OL-AXIS: plan s3 source rule: if V-O4b fails, every OpenLLaMA-axis verdict also on the frozen 3b/7b values
    if not g2:  # OL-AXIS
        try:  # OL-AXIS
            olf = {P: {'3b': ctx['frozen_ol'][P]['3b'], '7b': ctx['frozen_ol'][P]['7b'], '13b': ol[P]['13b']} for P in PRES}  # OL-AXIS
            vf = {}  # OL-AXIS
            Tf = RS.Trends(olf, vits)  # OL-AXIS
            RS.section_A(vf, Tf, PRES, RL.METRICS, VK.PROTOCOLS)  # OL-AXIS
            RS.section_B(vf, olf, PB, PRES, RL.METRICS, VK.PROTOCOLS, vits)  # OL-AXIS
            RS.section_C(vf, olf, Tf, PB, D, PRES, vits)  # OL-AXIS
            RS.section_F(vf, olf, PB, PRES, vits, {})  # OL-AXIS
            v.update({'x|frozen|' + k: val for k, val in vf.items()})  # OL-AXIS
        except Exception as e:  # OL-AXIS: never blocks the primary verdicts; reported
            v['x|frozen|error'] = repr(e)  # OL-AXIS
    # OL-AXIS: independent cross-check of every CKA Tr with the original review's trend() (float, tol 1e-12)
    nchk = nbad = 0  # OL-AXIS
    for P in PRES:  # OL-AXIS
        for M in ('cka_biased', 'cka_debiased'):  # OL-AXIS
            for Q in VK.PROTOCOLS:  # OL-AXIS
                for X in VK.scores(Q):  # OL-AXIS
                    rows = T.rows(P, M, Q, X)  # OL-AXIS
                    if any(x is None for r in rows for x in r) or all(len(set(r)) == 1 for r in rows):  # OL-AXIS
                        continue  # OL-AXIS
                    nchk += 1  # OL-AXIS
                    if abs(trend(rows)[0] - v[f'tr|{P}|{M}|{Q}|{X}']) > 1e-12:  # OL-AXIS
                        nbad += 1  # OL-AXIS
    v['x|check|legacy_trend_n'] = nchk  # OL-AXIS
    v['x|check|legacy_trend_nbad'] = nbad  # OL-AXIS
    v['x|G|llama_vs_merged'] = json.dumps(g1d)  # OL-AXIS
    v['x|G|rerun_vs_frozen'] = json.dumps(g2d)  # OL-AXIS
    v['x|F|notes'] = json.dumps(fnotes)  # OL-AXIS
    v['x|partB|key_forms'] = ','.join(sorted(PB.key_forms))  # OL-AXIS
    v['x|E_format_checks_nonzero'] = json.dumps(notes['e_checks'])[:20000]  # OL-AXIS
    v['x|identity_issues'] = json.dumps(notes['identity_issues'])  # OL-AXIS
    v['x|E_mknn_entries_not_count_over_10240'] = int(notes.get('e_mknn_entry_ulp', 0))  # OL-AXIS: ulp-level, counts exact
    miss = [k for k in VK.required_keys('ALL') if k not in v]  # OL-AXIS
    if miss:  # OL-AXIS
        raise RuntimeError(f'{len(miss)} required keys not computed, e.g. {miss[:5]}')  # OL-AXIS
    return v  # OL-AXIS


def run_compare(a, b, sections, out_json, log):  # OL-AXIS
    """OL-AXIS: compare_verdicts.py (frozen comparator) as a subprocess; full mismatch list via its same()."""
    import compare_verdicts as CV  # OL-AXIS
    import verdict_keys as VK  # OL-AXIS
    res = {'a': rel(a), 'b': rel(b), 'sections': sections}  # OL-AXIS
    if not os.path.isfile(b) or not os.path.isfile(a):  # OL-AXIS
        res.update({'status': 'missing', 'pass': False, 'exit_code': None,  # OL-AXIS
                    'missing_file': rel(b) if not os.path.isfile(b) else rel(a)})  # OL-AXIS
        log(f'compare {rel(a)} vs {rel(b)} [{sections}]: MISSING')  # OL-AXIS
        return res  # OL-AXIS
    cmd = [sys.executable, os.path.join(AXIS, 'compare_verdicts.py'), a, b, '--sections', sections, '--out', out_json]  # OL-AXIS
    p = subprocess.run(cmd, capture_output=True, text=True)  # OL-AXIS
    res.update({'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'pass': p.returncode == 0, 'cmp_file': rel(out_json)})  # OL-AXIS
    A, B = json.load(open(a))['v'], json.load(open(b))['v']  # OL-AXIS
    req = (VK.required_keys(sections) if sections in ('EI', 'ALL', 'F', 'J')  # OL-AXIS
           else [k for s in sections.split(',') for k in VK.required_keys(s)])  # OL-AXIS
    res['mismatches_full'] = [{'key': k, 'a': A[k], 'b': B[k]} for k in sorted(set(req))  # OL-AXIS
                              if k in A and k in B and not CV.same(A[k], B[k])]  # OL-AXIS
    res['missing_in_a'] = [k for k in req if k not in A]  # OL-AXIS
    res['missing_in_b'] = [k for k in req if k not in B]  # OL-AXIS
    res['n_mismatch'] = len(res['mismatches_full'])  # OL-AXIS
    log(f'compare {rel(a)} vs {rel(b)} [{sections}]: exit {p.returncode}, {res["n_mismatch"]} mismatches')  # OL-AXIS
    return res  # OL-AXIS


CATEGORY_KEYS = ('PO1|{P}|holds', 'PO2|{P}|s_cal|cat', 'PO2|{P}|unc|cat', 'PO2|{P}|s_cal|kedge', 'PO2ai|{P}|median',  # OL-AXIS
                 'PO2ai|{P}|in_band', 'PO2aii|{P}|verdict', 'PO2b|{P}|s_cal|verdict', 'PO2b|{P}|s_cal|n_decided',  # OL-AXIS
                 'PO2b|{P}|s_cal|x_correct', 'PO2b|{P}|unc|verdict', 'PO3|{P}|0.5|s_cal|desc', 'PO3|{P}|1.0|s_cal|desc',  # OL-AXIS
                 'PO4|{P}|s_cal|holds', 'PO4|{P}|unc|holds', 'PO4|{P}|arh|holds', 'PO5a|{P}|verdict',  # OL-AXIS
                 'PO5|{P}|M|cat', 'readO|{P}|cka_biased', 'readO|{P}|cka_debiased', 'dd|11|{P}|cka_biased',  # OL-AXIS
                 'dd|11|{P}|cka_biased|ci_lo', 'dd|11|{P}|cka_biased|ci_hi', 'dd|11|{P}|cka_biased|pge',  # OL-AXIS
                 'rule7a|{P}', 'size_not_reproduced|{P}', 'fp6|{P}|cat', 'plugin|{P}|within', 'D3|{P}|nviol')  # OL-AXIS


def write_report(out_root, rec, verd, log):  # OL-AXIS
    L = ['# OpenLLaMA axis: review recomputation (R)', '',  # OL-AXIS
         f'Written {rec["utc"]} by review/ol_axis_review.py (sha256 {rec["code_sha256"]["ol_axis_review.py"]}); mode: {rec["mode"]}.', '',  # OL-AXIS
         'R recomputed every verdict of VERDICTS_SPEC.md sections A, B, C, D, F, G, J from the pair files of E and, '  # OL-AXIS
         'separately, of I (plan section 3 and section 11 step S8). All OpenLLaMA-axis results are descriptive '  # OL-AXIS
         '(no confirmatory endpoint, plan R.1.2); J is EXPLORATORY.', '',  # OL-AXIS
         '## 0-mismatch status', '', '| comparison | sections | result | mismatches |', '|---|---|---|---|']  # OL-AXIS
    for c in rec['comparisons']:  # OL-AXIS
        st_ = 'PASS' if c.get('pass') else ('MISSING' if c.get('status') == 'missing' else 'FAIL')  # OL-AXIS
        L.append(f'| {c["a"]} vs {c["b"]} | {c["sections"]} | {st_} | {c.get("n_mismatch", "-")} |')  # OL-AXIS
    L += ['', f'Required comparisons (R vs E, I, forecast, joint): **{"all pass, 0 mismatches" if rec["required_pass"] else "NOT all pass"}**.', '']  # OL-AXIS
    L += ['## Categories reached (as recomputed by R)', '', '| key | R-E | R-I |', '|---|---|---|']  # OL-AXIS
    for P in PRES:  # OL-AXIS
        for k in CATEGORY_KEYS:  # OL-AXIS
            kk = k.format(P=P)  # OL-AXIS
            L.append(f'| {kk} | {verd.get("E", {}).get(kk, "-")} | {verd.get("I", {}).get(kk, "-")} |')  # OL-AXIS
    for k in ('J|s_cal|Jmin', 'J|s_cal|pmin', 'J|s_cal|Jmean', 'J|s_cal|pmean', 'J|s_cal|nO', 'J|s_cal|cond_size',  # OL-AXIS
              'J|unc|Jmin', 'J|unc|pmin', 'J|unc|nO', 'J|unc|cond_size'):  # OL-AXIS
        L.append(f'| {k} (EXPLORATORY) | {verd.get("E", {}).get(k, "-")} | {verd.get("I", {}).get(k, "-")} |')  # OL-AXIS
    L += ['', 'J statistics carry the label "EXPLORATORY", OpenLLaMA\'s own n_O/6 and the conditional size above, and the '  # OL-AXIS
          'provenance labels "BLOOMZ hypothesis-generating; BLOOMZ fails B4; BLOOMZ and LLaMA known before this plan".', '']  # OL-AXIS
    L += ['## F1(c) birth-time audit', '', '```', json.dumps({k: rec['audit'].get(k) for k in ('exit_code', 'pass', 'partB_status_utc', 'n_files', 'n_born_before_partB', 'min_birth', 'note')}, indent=1, default=str), '```', '']  # OL-AXIS
    L += ['## Checks and deviations', '']  # OL-AXIS
    for impl in ('E', 'I'):  # OL-AXIS
        vv = verd.get(impl)  # OL-AXIS
        if not vv:  # OL-AXIS
            continue  # OL-AXIS
        L.append(f'- R-{impl}: G|llama_pairs_equal_merged_summary = {vv.get("G|llama_pairs_equal_merged_summary")}, '  # OL-AXIS
                 f'G|rerun_equals_frozen_3b7b = {vv.get("G|rerun_equals_frozen_3b7b")}, G|n_pairs_read = {vv.get("G|n_pairs_read")}; '  # OL-AXIS
                 f'legacy trend() cross-check {vv.get("x|check|legacy_trend_nbad")}/{vv.get("x|check|legacy_trend_n")} differ; '  # OL-AXIS
                 f'Part B key forms {vv.get("x|partB|key_forms")}; F notes {vv.get("x|F|notes")}; '  # OL-AXIS
                 f'E-format consistency findings {vv.get("x|E_format_checks_nonzero", "{}")[:600]}; identity issues {vv.get("x|identity_issues")}.')  # OL-AXIS
    L += ['- Operational choices of R not fixed by the plan or the spec are listed in analysis/s12_layermax_openllama/review/CHANGES.md ([R-op]).',  # OL-AXIS
          '- Deviations: none recorded by this script. Any mismatch above is listed in full in recomputed_verdicts.json.', '']  # OL-AXIS
    with open(os.path.join(out_root, 'REVIEW_REPORT.md'), 'w') as f:  # OL-AXIS
        f.write('\n'.join(L) + '\n')  # OL-AXIS


def main():  # OL-AXIS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # OL-AXIS
    ap.add_argument('--impl', choices=('both', 'E', 'I'), default='both', help='which pair files to recompute from')  # OL-AXIS
    ap.add_argument('--phase', choices=('all', 'compute', 'compare'), default='all',  # OL-AXIS
                    help='compute = write review/{E,I}/verdicts_review.json; compare = comparisons + audit + report')  # OL-AXIS
    ap.add_argument('--test-root', default=None, help='S0b test mode only (OL_S0B_TEST=1): fixture root under ~/oll_s0b_scratch')  # OL-AXIS
    ap.add_argument('--out', default=None, help='S0b test mode only: output root under ~/oll_s0b_scratch')  # OL-AXIS
    args = ap.parse_args()  # OL-AXIS
    sys.path.insert(0, AXIS)  # OL-AXIS
    import guard  # OL-AXIS
    test = guard.test_mode()  # OL-AXIS
    if test:  # OL-AXIS
        if not args.test_root:  # OL-AXIS
            sys.exit('test mode needs --test-root')  # OL-AXIS
        out_root = os.path.abspath(args.out or os.path.join(args.test_root, 'out', 'review'))  # OL-AXIS
    else:  # OL-AXIS
        if args.test_root or args.out:  # OL-AXIS
            print('[review] --test-root/--out are S0b test options; refused outside test mode', file=sys.stderr)  # OL-AXIS
            sys.exit(3)  # OL-AXIS
        out_root = OUT_ROOT  # OL-AXIS
    gate = guard.require_partb_frozen(out_root, script='review/ol_axis_review.py')  # OL-AXIS: F1(a), first action
    sys.path.insert(0, HERE)  # OL-AXIS
    import verdict_keys as VK  # OL-AXIS
    import ol_review_lib as RL  # OL-AXIS
    import ol_review_sections as RS  # OL-AXIS
    th = int(os.environ.get('OL_THREADS', '1'))  # OL-AXIS
    paths = Paths(os.path.abspath(args.test_root) if test else None)  # OL-AXIS
    os.makedirs(out_root, exist_ok=True)  # OL-AXIS
    logf = open(os.path.join(out_root, 'review_run.log'), 'a')  # OL-AXIS

    def log(msg):  # OL-AXIS
        line = f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {msg}'  # OL-AXIS
        print(line, flush=True)  # OL-AXIS
        logf.write(line + '\n')  # OL-AXIS
        logf.flush()  # OL-AXIS

    log(f'start: gate {gate["mode"]}, out_root {out_root}, impl {args.impl}, phase {args.phase}, threads {th}')  # OL-AXIS
    impls = ('E', 'I') if args.impl == 'both' else (args.impl,)  # OL-AXIS
    code = {f: RL.sha256_file(os.path.join(HERE, f)) for f in ('ol_axis_review.py', 'ol_review_lib.py', 'ol_review_sections.py')}  # OL-AXIS
    if args.phase in ('all', 'compute'):  # OL-AXIS
        for impl in impls:  # OL-AXIS
            t0 = time.time()  # OL-AXIS
            ctx = load_impl(impl, paths, VK, RL, log)  # OL-AXIS
            v = compute(ctx, VK, RL, RS, log)  # OL-AXIS
            od = os.path.join(out_root, impl)  # OL-AXIS
            os.makedirs(od, exist_ok=True)  # OL-AXIS
            doc = {'schema': 'OL-VERDICTS v1', 'impl': f'R-{impl}', 'inputs': ctx['inputs'],  # OL-AXIS
                   'code_sha256': code, 'gate': gate, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),  # OL-AXIS
                   'seconds': time.time() - t0, 'v': v}  # OL-AXIS
            tmp = os.path.join(od, 'verdicts_review.json.tmp')  # OL-AXIS
            with open(tmp, 'w') as f:  # OL-AXIS
                json.dump(doc, f, indent=0, default=str)  # OL-AXIS
            os.replace(tmp, os.path.join(od, 'verdicts_review.json'))  # OL-AXIS
            log(f'R-{impl}: {len(v)} keys written to {rel(os.path.join(od, "verdicts_review.json"))} ({time.time() - t0:.0f}s)')  # OL-AXIS
    if args.phase == 'compute':  # OL-AXIS
        return 0  # OL-AXIS
    # OL-AXIS: comparisons (spec section 0: review/E vs E, review/I vs I; forecast and joint per implementation)
    os.makedirs(os.path.join(out_root, 'cmp'), exist_ok=True)  # OL-AXIS
    comps, verd = [], {}  # OL-AXIS
    for impl in ('E', 'I'):  # OL-AXIS
        mine = os.path.join(out_root, impl, 'verdicts_review.json')  # OL-AXIS
        if os.path.isfile(mine):  # OL-AXIS
            verd[impl] = json.load(open(mine))['v']  # OL-AXIS
        for sec, _ in SECTION_FILES:  # OL-AXIS
            tgt = paths.target(impl, sec)  # OL-AXIS
            r = run_compare(mine, tgt, sec, os.path.join(out_root, 'cmp', f'R{impl}_vs_{impl}_{sec}.json'), log)  # OL-AXIS
            r['required'] = True  # OL-AXIS
            comps.append(r)  # OL-AXIS
    r = run_compare(os.path.join(out_root, 'E', 'verdicts_review.json'), os.path.join(out_root, 'I', 'verdicts_review.json'),  # OL-AXIS
                    'ALL', os.path.join(out_root, 'cmp', 'RE_vs_RI_ALL.json'), log)  # OL-AXIS
    r['required'] = False  # OL-AXIS: informational (R-level V-O5)
    comps.append(r)  # OL-AXIS
    # OL-AXIS: F1(c) birth-time audit (guard.py audit), recorded whatever the outcome
    p = subprocess.run([sys.executable, os.path.join(AXIS, 'guard.py'), 'audit'], capture_output=True, text=True)  # OL-AXIS
    try:  # OL-AXIS
        audit = json.loads(p.stdout) if p.stdout.strip() else {}  # OL-AXIS
    except ValueError:  # OL-AXIS
        audit = {'stdout': p.stdout[-4000:]}  # OL-AXIS
    audit.update({'exit_code': p.returncode, 'stderr': p.stderr[-2000:]})  # OL-AXIS
    if test:  # OL-AXIS
        audit['note'] = 'S0b test mode: no Part B FROZEN line exists yet, so the audit exits 3 by design'  # OL-AXIS
    with open(os.path.join(out_root, 'audit_F1c.json'), 'w') as f:  # OL-AXIS
        json.dump(audit, f, indent=1, default=str)  # OL-AXIS
    log(f'F1(c) audit: exit {p.returncode}, pass {audit.get("pass")}')  # OL-AXIS
    req_pass = all(c.get('pass') for c in comps if c['required'])  # OL-AXIS
    rec = {'what': 'OpenLLaMA axis R recomputation: every comparison, every mismatch in full', 'mode': gate['mode'],  # OL-AXIS
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'code_sha256': code, 'gate': gate,  # OL-AXIS
           'comparisons': comps, 'required_pass': req_pass, 'audit': audit,  # OL-AXIS
           'files': {impl: rel(os.path.join(out_root, impl, 'verdicts_review.json')) for impl in verd}}  # OL-AXIS
    with open(os.path.join(out_root, 'recomputed_verdicts.json'), 'w') as f:  # OL-AXIS
        json.dump(rec, f, indent=1, default=str)  # OL-AXIS
    write_report(out_root, rec, verd, log)  # OL-AXIS
    log(f'done: required comparisons {"PASS" if req_pass else "NOT PASSED"}')  # OL-AXIS
    return 0 if req_pass else 1  # OL-AXIS


if __name__ == '__main__':
    sys.exit(main())  # OL-AXIS
