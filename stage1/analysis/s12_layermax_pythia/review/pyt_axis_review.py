"""PYT-AXIS COPY of analysis/s12_layermax_openllama/review/ol_axis_review.py (sha256 00c98f2a..., pristine in
PYT-AXIS: orig_copy/; diff pyt_axis_review.py.diff), implementation R of the Pythia axis (plan ANALYSIS_PLAN_PYTHIA.md,
PYT-AXIS: sha256 6de89603..., sections 3 and 11 step S8): the reviewer's own recomputation of every Pythia-axis
PYT-AXIS: verdict of the contract VERDICTS_SPEC_PYT.md (sections A-D, G, H, R, F, K: the gate, every Holm decision,
PYT-AXIS: every IUT and TOST p, every category and modifier, the P-P rules incl. P-P2b, robustness, D1-D6, the cross-family
PYT-AXIS: table and the 18-LM panel) from the pair files of E and, separately, of I, plus Part B; then the comparison
PYT-AXIS: with E's and I's verdict files through the shared compare_verdicts.py (0 mismatches required) and the F1(c)
PYT-AXIS: birth-time audit. Written at S0b by role pyt-code-R (2026-10-10) from the frozen plan, the shared contract,
PYT-AXIS: the OpenLLaMA originals and the stored pair formats; no new code of the E or I writers was read (plan 3).
PYT-AXIS: Gated (plan F1(a)): pyt_review_gate.require(out_root) is the first action of main() (shared guard + R's order
PYT-AXIS: checks); without the plan's gates it exits 3 and writes nothing. CPU only. Threads: PYT_THREADS (1 to 4).
PYT-AXIS:
PYT-AXIS: Usage (from ~/hrh_v2_stage1_20261008, through the launcher, after both main runs and forecast_pyt.py):
PYT-AXIS:   analysis/s12_layermax_pythia/launch_pyt.sh pyt_R_review 4 -- env PYTHONDONTWRITEBYTECODE=1 venv/bin/python \\
PYT-AXIS:       analysis/s12_layermax_pythia/review/pyt_axis_review.py [--impl both|E|I] [--phase all|compute|compare]
PYT-AXIS: Outputs (all under results/S1_2_layermax_pythia/review/):
PYT-AXIS:   E/verdicts_review.json, I/verdicts_review.json   (schema PYT-VERDICTS v1, impl "R-E" / "R-I", every key of
PYT-AXIS:                                                     verdict_keys.required_keys("ALL") plus x| extras)
PYT-AXIS:   cmp/R<impl>_vs_<impl>_<EI|FK>.json (compare_verdicts.py, names only); cmp/R<impl>_detail.json (R's values)
PYT-AXIS:   audit_F1c.json, recomputed_verdicts.json, REVIEW_REPORT.md
PYT-AXIS: Exit code: 0 = every required comparison passed; 1 = a mismatch or a missing target; 3 = refused by the gate."""
import json, glob, os, itertools, statistics as st
import numpy as np
import argparse  # OL-AXIS
import subprocess  # OL-AXIS
import sys  # OL-AXIS
import time  # OL-AXIS
import math  # PYT-AXIS

ROOT = os.path.expanduser('~/hrh_v2_stage1_20261008')
SP = ROOT
HERE = os.path.dirname(os.path.abspath(__file__))  # OL-AXIS
AXIS = os.path.join(ROOT, 'analysis', 's12_layermax_pythia')  # PYT-AXIS
RES_PY = os.path.join(ROOT, 'results', 'S1_2_layermax_pythia')  # PYT-AXIS: this axis (production)
OUT_ROOT = os.path.join(RES_PY, 'review')  # PYT-AXIS: the guarded output root of R (plan section 4 F1)
FROZEN8 = os.path.join(ROOT, 'results', 'S1_2_layermax')  # OL-AXIS: frozen 8-LM run, read-only
LLAMA_AX = os.path.join(ROOT, 'results', 'S1_2_layermax_llama')  # OL-AXIS: LLaMA axis, read-only
OL_AX = os.path.join(ROOT, 'results', 'S1_2_layermax_openllama')  # PYT-AXIS: OpenLLaMA axis, read-only
OL_PARTB = os.path.join(OL_AX, 'partB', 'partB_features_ol.json')  # PYT-AXIS: num_params of the 11 known LMs
FEAT_PY = os.path.join(ROOT, 'features', 'wit1024_pythia_axis')  # PYT-AXIS: the 7 primary stacks (clip check)
PRES = ('prh_clip+l2', 'l2')  # OL-AXIS
TARGETS = {'EI': os.path.join('{impl_dir}', 'verdicts_pyt.json'),  # PYT-AXIS: contract section 0
           'FK': os.path.join('forecast', '{impl}', 'verdicts_forecast.json')}  # PYT-AXIS


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


# ---------------------------------------------------------------------------------------------- PYT-AXIS
# PYT-AXIS: input locations. Production: the plan's paths. S0b test mode (PYT_S0B_TEST=1 before CODE FROZEN):
# PYT-AXIS: --test-root <dir under analysis/s12_layermax_pythia/s0b_scratch/R/> replaces the Pythia results root
# PYT-AXIS: (<root>/results/S1_2_layermax_pythia/...) and the feature root (<root>/features/), and the axis becomes the
# PYT-AXIS: stand-in (slugs synth__pyt-<size>, contract section 13); known-LM pair files are always the real ones.
# ---------------------------------------------------------------------------------------------- PYT-AXIS
class Paths:  # PYT-AXIS
    def __init__(self, test_root=None):  # PYT-AXIS
        self.test = test_root is not None  # PYT-AXIS
        self.res = os.path.join(test_root, 'results', 'S1_2_layermax_pythia') if self.test else RES_PY  # PYT-AXIS
        self.feat = os.path.join(test_root, 'features') if self.test else FEAT_PY  # PYT-AXIS

    def py_E(self, P, slug, vit):  # PYT-AXIS: E pair files (OpenLLaMA E format)
        b = os.path.join(self.res, 'early', 'pairs', P, f'{slug}__{vit}')  # PYT-AXIS
        return b + '.json', b + '.npz'  # PYT-AXIS

    def py_I(self, P, slug, vit):  # PYT-AXIS: I pair files (OpenLLaMA I format)
        return os.path.join(self.res, 'indep', P, 'pairs', f'{slug}__{vit}.npz')  # PYT-AXIS

    def partb(self):  # PYT-AXIS
        return os.path.join(self.res, 'partB', 'partB_features_pyt.json')  # PYT-AXIS

    def clip(self):  # PYT-AXIS: V-P1 reference (contract section 11)
        return os.path.join(self.res, 'partB', 'clip_thresholds_partB_pyt.json')  # PYT-AXIS

    @staticmethod
    def known_E(P, lm, vit):  # PYT-AXIS: contract section 1 cross-family sources, E counterpart
        if lm.endswith('open_llama_13b'):  # PYT-AXIS
            b = os.path.join(OL_AX, 'early', 'pairs', P, f'{lm}__{vit}')  # PYT-AXIS
        elif lm.endswith('llama-13b') or lm.endswith('llama-30b'):  # PYT-AXIS
            b = os.path.join(LLAMA_AX, 'early', 'pairs', P, f'{lm}__{vit}')  # PYT-AXIS
        else:  # PYT-AXIS: BLOOMZ, llama-7b, open_llama_3b/7b: frozen 8-LM run
            b = os.path.join(FROZEN8, 'pairs', P, f'{lm}__{vit}')  # PYT-AXIS
        return b + '.json', b + '.npz'  # PYT-AXIS

    @staticmethod
    def known_I(P, lm, vit):  # PYT-AXIS: I counterpart
        if lm.endswith('open_llama_13b'):  # PYT-AXIS
            return os.path.join(OL_AX, 'indep', P, 'pairs', f'{lm}__{vit}.npz')  # PYT-AXIS
        if lm.endswith('llama-13b') or lm.endswith('llama-30b'):  # PYT-AXIS
            return os.path.join(LLAMA_AX, 'indep', P, 'pairs', f'{lm}__{vit}.npz')  # PYT-AXIS
        return os.path.join(FROZEN8, 'indep', P, 'pairs', f'{lm}__{vit}.npz')  # PYT-AXIS

    def target(self, impl, sec):  # PYT-AXIS
        t = TARGETS[sec].format(impl_dir='early' if impl == 'E' else 'indep', impl=impl)  # PYT-AXIS
        return os.path.join(self.res, t)  # PYT-AXIS


def rel(p):  # OL-AXIS
    p = os.path.abspath(p)  # OL-AXIS
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p  # OL-AXIS


def pres_run(impl, paths, ax, CF):  # PYT-AXIS
    """PYT-AXIS: contract section 9 l2_run: l2 counts as run only if all its 119 pair files of this implementation
    PYT-AXIS: exist; with none, every l2 key is null; a partial set is never opened (plan section 11)."""
    out, notes = ['prh_clip+l2'], {}  # PYT-AXIS
    f = (lambda s, v: paths.py_E('l2', ax.slug[s], v)[1]) if impl == 'E' else (lambda s, v: paths.py_I('l2', ax.slug[s], v))  # PYT-AXIS
    n = sum(os.path.isfile(f(s, v)) for s in ax.sizes for v in CF.VITS)  # PYT-AXIS
    if n == ax.n * len(CF.VITS):  # PYT-AXIS
        out.append('l2')  # PYT-AXIS
    elif n:  # PYT-AXIS
        notes['l2_partial_files'] = n  # PYT-AXIS
    return tuple(out), notes  # PYT-AXIS


def load_impl(impl, paths, ax, CF, RL, log, pres):  # PYT-AXIS
    """PYT-AXIS: read every pair file R needs for one implementation; returns the context of the sections."""
    vits = list(CF.VITS)  # PYT-AXIS
    inputs, notes = {}, {'e_checks': {}, 'identity_issues': []}  # OL-AXIS
    allow = paths.test  # PYT-AXIS: missing subsets tolerated only for the S0b stand-in (frozen 8-LM E files)

    def rd_E(j, z, lm, vit, protocols=RL.SUBSETS, ctm=True, allow_missing=False, own=True):  # PYT-AXIS
        for f in (j, z):  # OL-AXIS
            inputs[rel(f)] = RL.sha256_file(f)  # OL-AXIS
        ps = RL.read_pair_E(j, z, protocols=protocols, want_ctm=ctm, allow_missing=allow_missing)  # OL-AXIS
        ck = ps['checks']  # OL-AXIS
        bad = {k: v for k, v in ck.items() if (k == 'max_rel_mu_f32' and v > 1e-6)  # OL-AXIS
               or (k == 'missing_subsets' and v and own and not allow)  # PYT-AXIS
               or (k not in ('max_rel_mu_f32', 'missing_subsets', 'n_mknn_entry_ulp') and v)}  # OL-AXIS
        if bad:  # OL-AXIS
            notes['e_checks'][f'{lm}|{vit}'] = bad  # PYT-AXIS
        notes['e_mknn_entry_ulp'] = notes.get('e_mknn_entry_ulp', 0) + ck.get('n_mknn_entry_ulp', 0)  # OL-AXIS
        if own and not paths.test and (ps.get('lm') != lm or ps.get('vit') != vit  # PYT-AXIS
                                       or ps.get('perm_key') != f'S1.2-layermax|{lm}|{vit}' or ps.get('K') != 500):  # OL-AXIS
            notes['identity_issues'].append(f'{lm}|{vit}: lm/vit/perm_key/K in the JSON differ')  # PYT-AXIS
        return ps  # OL-AXIS

    def rd_I(f, lm, vit, protocols=RL.SUBSETS, ctm=True, own=True):  # PYT-AXIS
        inputs[rel(f)] = RL.sha256_file(f)  # OL-AXIS
        ps = RL.read_pair_I(f, protocols=protocols, want_ctm=ctm)  # OL-AXIS
        if own and not paths.test:  # PYT-AXIS
            z = np.load(f)  # OL-AXIS
            if str(z['key']) != f'S1.2-layermax|{lm}|{vit}' or int(z['K']) != 500:  # OL-AXIS
                notes['identity_issues'].append(f'{lm}|{vit}: key/K in the npz differ')  # PYT-AXIS
        return ps  # OL-AXIS

    py, pool = {}, {}  # PYT-AXIS
    t0 = time.time()  # OL-AXIS
    for P in pres:  # PYT-AXIS
        py[P] = {}  # PYT-AXIS
        for s in ax.sizes:  # PYT-AXIS
            slug = ax.slug[s]  # PYT-AXIS
            if impl == 'E':  # PYT-AXIS
                py[P][s] = {v: rd_E(*paths.py_E(P, slug, v), slug, v, allow_missing=allow) for v in vits}  # PYT-AXIS
            else:  # PYT-AXIS
                py[P][s] = {v: rd_I(paths.py_I(P, slug, v), slug, v) for v in vits}  # PYT-AXIS
        pool[P] = {}  # PYT-AXIS: contract section 8 sources, each implementation reading its own counterpart
        for lm in CF.POOLED11:  # PYT-AXIS
            if impl == 'E':  # PYT-AXIS
                pool[P][lm] = {v: rd_E(*Paths.known_E(P, lm, v), lm, v, protocols=('full',), ctm=False,  # PYT-AXIS
                                       allow_missing=True, own=False) for v in vits}  # PYT-AXIS
            else:  # PYT-AXIS
                pool[P][lm] = {v: rd_I(Paths.known_I(P, lm, v), lm, v, protocols=('full',), ctm=False,  # PYT-AXIS
                                       own=False) for v in vits}  # PYT-AXIS
        log(f'{impl} {P}: pair files read ({time.time() - t0:.0f}s)')  # OL-AXIS
    pbp = paths.partb()  # OL-AXIS
    inputs[rel(pbp)] = RL.sha256_file(pbp)  # OL-AXIS
    pb = json.load(open(pbp))  # OL-AXIS
    inputs[rel(OL_PARTB)] = RL.sha256_file(OL_PARTB)  # PYT-AXIS
    num_params = {lm: int(json.load(open(OL_PARTB))['num_params'][lm]) for lm in CF.POOLED11}  # PYT-AXIS
    for s in ax.sizes:  # PYT-AXIS: Pythia num_params from Part B (contract section 1)
        npar = pb['num_params']  # PYT-AXIS
        num_params[ax.slug[s]] = int(npar[ax.slug[s]] if ax.slug[s] in npar else npar[s])  # PYT-AXIS
    return {'py': py, 'pool': pool, 'pb': pb, 'vits': vits, 'inputs': inputs, 'notes': notes,  # PYT-AXIS
            'num_params': num_params, 'partb_sha': inputs[rel(pbp)],  # PYT-AXIS
            'n_known': sum(len(pool[P][lm]) for P in pres for lm in pool[P])}  # PYT-AXIS


def clip_check(paths, ax, RL, log):  # PYT-AXIS
    """PYT-AXIS: contract section 9 G|clip_equal_partB, R's own reading [R-op]: R recomputes the 7 clip thresholds
    PYT-AXIS: (hrhv2.prep.prh_clip_threshold on the float64 all-layer stack, plan section 3) from the primary feature
    PYT-AXIS: stacks and compares them bitwise with clip_thresholds_partB_pyt.json (an R-level V-P1)."""
    from hrhv2.prep import prh_clip_threshold  # PYT-AXIS
    ref = json.load(open(paths.clip()))['clip']  # PYT-AXIS
    rows, ok = {}, True  # PYT-AXIS
    for s in ax.sizes:  # PYT-AXIS
        slug = ax.slug[s]  # PYT-AXIS
        X = np.load(os.path.join(paths.feat, slug + '.npy'), mmap_mode='r')  # PYT-AXIS
        c = float(prh_clip_threshold(np.asarray(X, dtype=np.float64)))  # PYT-AXIS
        want = float(ref[slug]['c'])  # PYT-AXIS
        rows[s] = {'R': c, 'partB': want, 'bitwise_equal': c == want}  # PYT-AXIS
        ok = ok and c == want  # PYT-AXIS
    log(f'clip check: {sum(r["bitwise_equal"] for r in rows.values())}/{len(rows)} bitwise equal')  # PYT-AXIS
    return ok, rows  # PYT-AXIS


def compute(ctx, ax, CF, RL, RS, RP, log, pres, clip_ok, VK):  # PYT-AXIS
    """PYT-AXIS: every required key of the contract (sections A, B, C, H, R, D, G, F, K) plus x| extras."""
    v, notes = {}, ctx['notes']  # OL-AXIS
    vits, py = ctx['vits'], ctx['py']  # PYT-AXIS
    PB = RP.PartBPyt(ctx['pb'], ax, vits)  # PYT-AXIS
    T = RS.Trends(py, vits, ax.sizes, mode='common')  # PYT-AXIS: plan 5.1 common-ViT-set rule
    t0 = time.time()  # OL-AXIS
    RS.section_A(v, T, pres, RL.METRICS, RP.PROTOCOLS)  # PYT-AXIS
    RS.section_B_all(v, py, pres, RL.METRICS, RP.PROTOCOLS, vits, ax.pairs)  # PYT-AXIS
    log(f'sections A, B done ({time.time() - t0:.0f}s)')  # OL-AXIS
    RP.section_C(v, py, T, PB, ax, vits, pres)  # PYT-AXIS
    draws = {}  # PYT-AXIS: bootstrap draws by key (each key drawn once)
    fams = {(P, x): RP.Family(T, PB, P, x, draws) for P in pres for x in RP.XS}  # PYT-AXIS
    RP.section_H(v, fams, pres)  # PYT-AXIS
    RP.section_R(v, fams, py, T, ax, vits, pres, draws)  # PYT-AXIS
    log(f'sections C, H, R done ({time.time() - t0:.0f}s)')  # PYT-AXIS
    RP.section_D(v, py, ctx['pool'], ctx['num_params'], ax, vits, pres)  # PYT-AXIS
    RP.section_F(v, py, PB, ax, vits, pres)  # PYT-AXIS
    RP.section_K(v, fams, py, PB, ax, vits, pres)  # PYT-AXIS
    RP.section_G(v, py, PB, ax, vits, pres, ctx['n_known'], ctx['partb_sha'], clip_ok, notes)  # PYT-AXIS
    log(f'sections D, F, K, G done ({time.time() - t0:.0f}s)')  # PYT-AXIS
    # OL-AXIS: independent cross-check of every CKA Tr with the original review's trend() (float, tol 1e-12)
    nchk = nbad = 0  # OL-AXIS
    for P in pres:  # PYT-AXIS
        for M in ('cka_biased', 'cka_debiased'):  # OL-AXIS
            for Q in ('full', 'ctm'):  # PYT-AXIS: layer-max and calibrate-then-max only (n! python loops per call)
                for X in RS.scores(Q):  # PYT-AXIS
                    rows = T.rows(P, M, Q, X)  # OL-AXIS
                    if any(x is None for r in rows for x in r) or all(len(set(r)) == 1 for r in rows):  # OL-AXIS
                        continue  # OL-AXIS
                    nchk += 1  # OL-AXIS
                    if abs(trend(rows)[0] - v[f'tr|{P}|{M}|{Q}|{X}']) > 1e-12:  # OL-AXIS
                        nbad += 1  # OL-AXIS
    v['x|check|legacy_trend_n'] = nchk  # OL-AXIS
    v['x|check|legacy_trend_nbad'] = nbad  # OL-AXIS
    v['x|E_format_checks_nonzero'] = json.dumps(notes['e_checks'])[:20000]  # OL-AXIS
    v['x|E_mknn_entries_not_count_over_10240'] = int(notes.get('e_mknn_entry_ulp', 0))  # OL-AXIS: ulp-level
    v['x|axis'] = ax.label  # PYT-AXIS
    # PYT-AXIS: contract section 9: if l2 was not run, every key with P = l2 is null
    req = VK.required_keys('ALL')  # PYT-AXIS
    if 'l2' not in pres:  # PYT-AXIS
        for k in req:  # PYT-AXIS
            if k.split('|')[1:2] == ['l2'] and k not in v:  # PYT-AXIS
                v[k] = None  # PYT-AXIS
    miss = [k for k in req if k not in v]  # PYT-AXIS
    v['x|n_required_missing'] = len(miss)  # PYT-AXIS
    v['x|required_missing'] = ';'.join(miss[:200])  # PYT-AXIS
    extra = [k for k in v if not k.startswith('x|') and k not in set(req)]  # PYT-AXIS
    v['x|n_not_in_contract'] = len(extra)  # PYT-AXIS
    v['x|not_in_contract'] = ';'.join(extra[:200])  # PYT-AXIS
    return v  # OL-AXIS


def same(a, b):  # PYT-AXIS: the contract comparison rule (compare_verdicts.same), for R's detail file
    if a is None or b is None:  # PYT-AXIS
        return a is None and b is None  # PYT-AXIS
    if isinstance(a, bool) or isinstance(b, bool):  # PYT-AXIS
        return isinstance(a, bool) and isinstance(b, bool) and a == b  # PYT-AXIS
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):  # PYT-AXIS
        fa, fb = float(a), float(b)  # PYT-AXIS
        if math.isnan(fa) or math.isnan(fb):  # PYT-AXIS
            return math.isnan(fa) and math.isnan(fb)  # PYT-AXIS
        if math.isinf(fa) or math.isinf(fb):  # PYT-AXIS
            return fa == fb  # PYT-AXIS
        return abs(fa - fb) <= 1e-12 * max(abs(fa), abs(fb)) + 1e-15  # PYT-AXIS
    return type(a) is type(b) and a == b  # PYT-AXIS


def run_compare(a, b, sections, out_json, log):  # PYT-AXIS
    """PYT-AXIS: the shared compare_verdicts.py (names only) as a subprocess; R's detail list (values, R's own
    PYT-AXIS: output, S8) in a separate file."""
    res = {'a': rel(a), 'b': rel(b), 'sections': sections}  # OL-AXIS
    if not os.path.isfile(b) or not os.path.isfile(a):  # OL-AXIS
        res.update({'status': 'missing', 'pass': False, 'exit_code': None,  # OL-AXIS
                    'missing_file': rel(b) if not os.path.isfile(b) else rel(a)})  # OL-AXIS
        log(f'compare {rel(a)} vs {rel(b)} [{sections}]: MISSING')  # OL-AXIS
        return res  # OL-AXIS
    cmd = [sys.executable, os.path.join(AXIS, 'compare_verdicts.py'), a, b, '--sections', sections, '--out', out_json]  # PYT-AXIS
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))  # PYT-AXIS
    res.update({'exit_code': p.returncode, 'stdout': p.stdout.strip()[-2000:], 'pass': p.returncode == 0,  # PYT-AXIS
                'cmp_file': rel(out_json), 'status': 'compared'})  # PYT-AXIS
    try:  # PYT-AXIS
        rep = json.load(open(out_json))  # PYT-AXIS
        res.update({k: rep.get(k) for k in ('n_required', 'n_compared', 'n_missing_in_a', 'n_missing_in_b', 'n_mismatch')})  # PYT-AXIS
    except (OSError, ValueError):  # PYT-AXIS
        pass  # PYT-AXIS
    log(f'compare {rel(a)} vs {rel(b)} [{sections}]: exit {p.returncode}, {res.get("n_mismatch")} mismatches')  # PYT-AXIS
    return res  # OL-AXIS


def detail(a, b, VK, sections, out_json):  # PYT-AXIS: R's own value-level list of every mismatch (S8)
    A, B = json.load(open(a))['v'], json.load(open(b))['v']  # PYT-AXIS
    req = VK.required_keys(sections)  # PYT-AXIS
    mm = [{'key': k, 'R': A.get(k, '<missing>'), 'impl': B.get(k, '<missing>')} for k in req  # PYT-AXIS
          if k not in A or k not in B or not same(A[k], B[k])]  # PYT-AXIS
    with open(out_json, 'w') as f:  # PYT-AXIS
        json.dump({'a': rel(a), 'b': rel(b), 'sections': sections, 'n': len(mm), 'mismatches': mm}, f, indent=1, default=str)  # PYT-AXIS
    return len(mm)  # PYT-AXIS


CATEGORY_KEYS = ('K|{P}|{x}|category', 'K|{P}|{x}|F_reason', 'K|{P}|{x}|premise', 'H|{P}|{x}|H1|member',  # PYT-AXIS
                 'H|{P}|{x}|H1|D', 'H|{P}|{x}|H1|p_perm', 'H|{P}|{x}|H1|p_mag', 'H|{P}|{x}|H1|p',  # PYT-AXIS
                 'H|{P}|{x}|H1|confirmed', 'H|{P}|{x}|gate', 'H|{P}|{x}|holm_order', 'H|{P}|{x}|H2|confirmed',  # PYT-AXIS
                 'H|{P}|{x}|H3|confirmed', 'H|{P}|{x}|H4|confirmed', 'K|{P}|{x}|H2_reading', 'K|{P}|{x}|H3_reading',  # PYT-AXIS
                 'K|{P}|{x}|H4_reading', 'K|{P}|{x}|M1', 'K|{P}|{x}|M2', 'K|{P}|{x}|M3', 'K|{P}|{x}|M4',  # PYT-AXIS
                 'K|{P}|{x}|M5_extrap_sizes', 'K|{P}|{x}|kedge')  # PYT-AXIS
PP_KEYS = ('PP1|{P}|holds', 'PP2ai|{P}|s_cal|n_models_in_band', 'PP2ai|{P}|s_cal|holds', 'PP2aii|{P}|s_cal|verdict',  # PYT-AXIS
           'PP2a|{P}|s_cal|verdict', 'PP2b|{P}|s_cal|verdict', 'PP2b|{P}|s_cal|n_decided', 'PP2b|{P}|s_cal|x_correct',  # PYT-AXIS
           'PP5a|{P}|s_cal|verdict', 'D3|{P}|nviol')  # PYT-AXIS


def write_report(out_root, rec, verd, log):  # PYT-AXIS
    L = ['# Pythia axis: review recomputation (R)', '',  # PYT-AXIS
         f'Written {rec["utc"]} by review/pyt_axis_review.py (sha256 {rec["code_sha256"]["pyt_axis_review.py"]}); mode: {rec["mode"]}.', '',  # PYT-AXIS
         'R recomputed every verdict of VERDICTS_SPEC_PYT.md (sections A-D, G, H, R, F, K) from the pair files of E and, '  # PYT-AXIS
         'separately, of I, and Part B (plan section 3 and section 11 step S8). Confirmatory: H1 (gatekeeper) and the '  # PYT-AXIS
         'Holm family {H2, H3, H4} of the primary instance (prh_clip+l2, s_cal) only; everything else descriptive.', '',  # PYT-AXIS
         '## 0-mismatch status', '', '| R file | target | sections | status | compared | mismatches | pass |',  # PYT-AXIS
         '|---|---|---|---|---|---|---|']  # PYT-AXIS
    for c in rec['comparisons']:  # OL-AXIS
        L.append(f'| {c["a"]} | {c["b"]} | {c["sections"]} | {c.get("status")} | {c.get("n_compared", "-")} | '  # PYT-AXIS
                 f'{c.get("n_mismatch", "-")} | {c.get("pass")} |')  # PYT-AXIS
    L += ['', f'Required comparisons (R-E vs E and forecast/E, R-I vs I and forecast/I): '  # PYT-AXIS
          f'**{"all pass, 0 mismatches" if rec["required_pass"] else "NOT all pass"}**.', '']  # PYT-AXIS
    L += ['## Categories and decisions (as recomputed by R)', '', '| key | R-E | R-I |', '|---|---|---|']  # OL-AXIS
    for P in PRES:  # OL-AXIS
        for x in ('s_cal', 'unc'):  # PYT-AXIS
            for k in CATEGORY_KEYS:  # PYT-AXIS
                kk = k.format(P=P, x=x)  # PYT-AXIS
                L.append(f'| {kk} | {verd.get("E", {}).get(kk, "-")} | {verd.get("I", {}).get(kk, "-")} |')  # PYT-AXIS
        for k in PP_KEYS:  # PYT-AXIS
            kk = k.format(P=P)  # PYT-AXIS
            L.append(f'| {kk} | {verd.get("E", {}).get(kk, "-")} | {verd.get("I", {}).get(kk, "-")} |')  # PYT-AXIS
    L += ['', 'Primary instance: prh_clip+l2, biased CKA, layer-max (`full`), s_cal. l2 and uncensored rows are secondary.', '']  # PYT-AXIS
    L += ['## F1(c) birth-time audit', '', '```', json.dumps({k: rec['audit'].get(k) for k in ('pass', 'partB_status_utc', 'n_files', 'n_born_before_partB', 'n_birth_unknown', 'min_birth', 'note', 'source')}, indent=1, default=str), '```', '']  # PYT-AXIS
    L += ['## Checks and deviations', '']  # OL-AXIS
    for impl in ('E', 'I'):  # OL-AXIS
        vv = verd.get(impl)  # OL-AXIS
        if not vv:  # OL-AXIS
            continue  # OL-AXIS
        L.append(f'- R-{impl}: G|n_pairs_read = {vv.get("G|n_pairs_read")}, G|l2_run = {vv.get("G|l2_run")}, '  # PYT-AXIS
                 f'G|clip_equal_partB = {vv.get("G|clip_equal_partB")}, Part B recheck ok = {vv.get("x|partB_recheck_ok")} '  # PYT-AXIS
                 f'({vv.get("x|partB_recheck_diffs", "")[:600]}); layers vs plan {vv.get("x|layers_match_plan")}; '  # PYT-AXIS
                 f'legacy trend() cross-check {vv.get("x|check|legacy_trend_nbad")}/{vv.get("x|check|legacy_trend_n")} differ; '  # PYT-AXIS
                 f'required keys missing {vv.get("x|n_required_missing")}; E-format findings '  # PYT-AXIS
                 f'{vv.get("x|E_format_checks_nonzero", "{}")[:600]}; identity issues {vv.get("x|identity_issues")}.')  # PYT-AXIS
    L += ['- Operational choices of R not fixed by the plan or the contract are listed in '  # PYT-AXIS
          'analysis/s12_layermax_pythia/review/CHANGES.md ([R-op]).',  # PYT-AXIS
          '- Every mismatch is listed by name in cmp/*_EI.json and cmp/*_FK.json and with values in cmp/R*_detail_*.json.', '']  # PYT-AXIS
    with open(os.path.join(out_root, 'REVIEW_REPORT.md'), 'w') as f:  # OL-AXIS
        f.write('\n'.join(L) + '\n')  # OL-AXIS


def main():  # OL-AXIS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # OL-AXIS
    ap.add_argument('--impl', choices=('both', 'E', 'I'), default='both', help='which pair files to recompute from')  # OL-AXIS
    ap.add_argument('--phase', choices=('all', 'compute', 'compare'), default='all',  # OL-AXIS
                    help='compute = write review/{E,I}/verdicts_review.json; compare = comparisons + audit + report')  # OL-AXIS
    ap.add_argument('--test-root', default=None, help='S0b test mode only (PYT_S0B_TEST=1): fixture root under s0b_scratch/R/')  # PYT-AXIS
    ap.add_argument('--out', default=None, help='S0b test mode only: output root under s0b_scratch/R/')  # PYT-AXIS
    args = ap.parse_args()  # OL-AXIS
    sys.path.insert(0, HERE)  # PYT-AXIS
    import pyt_review_gate as G  # PYT-AXIS
    test = G.test_mode()  # PYT-AXIS
    if test:  # OL-AXIS
        if not args.test_root:  # OL-AXIS
            print('[review] test mode needs --test-root', file=sys.stderr)  # PYT-AXIS
            sys.exit(3)  # PYT-AXIS
        out_root = os.path.abspath(args.out or os.path.join(args.test_root, 'out', 'review'))  # OL-AXIS
    else:  # OL-AXIS
        if args.test_root or args.out:  # PYT-AXIS
            print('[review] --test-root/--out are S0b test options; refused outside test mode', file=sys.stderr)  # PYT-AXIS
            sys.exit(3)  # OL-AXIS
        out_root = OUT_ROOT  # OL-AXIS
    gate = G.require(out_root, script='review/pyt_axis_review.py')  # PYT-AXIS: F1(a), first action, writes nothing
    if test and not os.path.realpath(args.test_root).startswith(os.path.realpath(G.test_root()) + os.sep):  # PYT-AXIS
        print(f'[review] --test-root must lie under {G.test_root()}', file=sys.stderr)  # PYT-AXIS
        sys.exit(3)  # PYT-AXIS
    sys.path.insert(0, AXIS)  # PYT-AXIS: the shared contract (verdict_keys.py), after the gate
    import verdict_keys as VK  # PYT-AXIS
    import pyt_review_conf as CF  # PYT-AXIS
    import pyt_review_lib as RL  # PYT-AXIS
    import pyt_review_sections as RS  # PYT-AXIS
    import pyt_review_pythia as RP  # PYT-AXIS
    ax = CF.STANDIN if test else CF.PYTHIA  # PYT-AXIS: the S0b tests never use Pythia names
    th = int(os.environ.get('PYT_THREADS', '1'))  # PYT-AXIS
    paths = Paths(os.path.abspath(args.test_root) if test else None)  # OL-AXIS
    os.makedirs(out_root, exist_ok=True)  # OL-AXIS
    logf = open(os.path.join(out_root, 'review_run.log'), 'a')  # OL-AXIS

    def log(msg):  # OL-AXIS
        line = f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {msg}'  # OL-AXIS
        print(line, flush=True)  # OL-AXIS
        logf.write(line + '\n')  # OL-AXIS
        logf.flush()  # OL-AXIS

    log(f'start: gate {gate.get("r_gate")}, out_root {out_root}, impl {args.impl}, phase {args.phase}, threads {th}, axis {ax.label}')  # PYT-AXIS
    impls = ('E', 'I') if args.impl == 'both' else (args.impl,)  # OL-AXIS
    code = {f: RL.sha256_file(os.path.join(HERE, f)) for f in ('pyt_axis_review.py', 'pyt_review_lib.py',  # PYT-AXIS
                                                               'pyt_review_sections.py', 'pyt_review_pythia.py',  # PYT-AXIS
                                                               'pyt_review_conf.py', 'pyt_review_gate.py')}  # PYT-AXIS
    code.update({f: RL.sha256_file(os.path.join(AXIS, f)) for f in ('verdict_keys.py', 'compare_verdicts.py', 'guard.py')})  # PYT-AXIS
    if args.phase in ('all', 'compute'):  # OL-AXIS
        clip_ok, clip_rows = clip_check(paths, ax, RL, log)  # PYT-AXIS: implementation-independent, once
        for impl in impls:  # OL-AXIS
            t0 = time.time()  # OL-AXIS
            pres, pnotes = pres_run(impl, paths, ax, CF)  # PYT-AXIS
            ctx = load_impl(impl, paths, ax, CF, RL, log, pres)  # PYT-AXIS
            v = compute(ctx, ax, CF, RL, RS, RP, log, pres, clip_ok, VK)  # PYT-AXIS
            v['x|clip_rows'] = json.dumps(clip_rows)  # PYT-AXIS
            v['x|l2_notes'] = json.dumps(pnotes)  # PYT-AXIS
            od = os.path.join(out_root, impl)  # OL-AXIS
            os.makedirs(od, exist_ok=True)  # OL-AXIS
            doc = {'schema': 'PYT-VERDICTS v1', 'impl': f'R-{impl}', 'inputs': ctx['inputs'],  # PYT-AXIS
                   'code_sha256': code, 'gate': gate, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),  # OL-AXIS
                   'seconds': time.time() - t0, 'v': v}  # OL-AXIS
            tmp = os.path.join(od, 'verdicts_review.json.tmp')  # OL-AXIS
            with open(tmp, 'w') as f:  # OL-AXIS
                json.dump(doc, f, indent=0, default=str)  # OL-AXIS
            os.replace(tmp, os.path.join(od, 'verdicts_review.json'))  # OL-AXIS
            log(f'R-{impl}: {len(v)} keys ({v["x|n_required_missing"]} required missing) written to '  # PYT-AXIS
                f'{rel(os.path.join(od, "verdicts_review.json"))} ({time.time() - t0:.0f}s)')  # PYT-AXIS
    if args.phase == 'compute':  # OL-AXIS
        return 0  # OL-AXIS
    # PYT-AXIS: comparisons (contract section 0): review/E vs E (EI) and forecast/E (FK); the same for I
    os.makedirs(os.path.join(out_root, 'cmp'), exist_ok=True)  # OL-AXIS
    comps, verd = [], {}  # OL-AXIS
    for impl in ('E', 'I'):  # OL-AXIS
        mine = os.path.join(out_root, impl, 'verdicts_review.json')  # OL-AXIS
        if os.path.isfile(mine):  # OL-AXIS
            verd[impl] = json.load(open(mine))['v']  # OL-AXIS
        for sec in ('EI', 'FK'):  # PYT-AXIS
            tgt = paths.target(impl, sec)  # PYT-AXIS
            r = run_compare(mine, tgt, sec, os.path.join(out_root, 'cmp', f'R{impl}_vs_{impl}_{sec}.json'), log)  # PYT-AXIS
            r['required'] = True  # OL-AXIS
            if os.path.isfile(mine) and os.path.isfile(tgt):  # PYT-AXIS
                r['n_detail'] = detail(mine, tgt, VK, sec, os.path.join(out_root, 'cmp', f'R{impl}_detail_{sec}.json'))  # PYT-AXIS
            comps.append(r)  # OL-AXIS
    r = run_compare(os.path.join(out_root, 'E', 'verdicts_review.json'), os.path.join(out_root, 'I', 'verdicts_review.json'),  # OL-AXIS
                    'ALL', os.path.join(out_root, 'cmp', 'RE_vs_RI_ALL.json'), log)  # PYT-AXIS
    r['required'] = False  # OL-AXIS: informational (R-level V-P5)
    comps.append(r)  # OL-AXIS
    if test:  # PYT-AXIS: the shared audit exits 3 without a Part B FROZEN line, so it is not called in test mode
        audit = {'pass': None, 'note': 'S0b test mode: no Part B FROZEN line exists yet; audit not run', 'source': 'none'}  # PYT-AXIS
    else:  # PYT-AXIS
        audit = G.audit()  # PYT-AXIS
    with open(os.path.join(out_root, 'audit_F1c.json'), 'w') as f:  # OL-AXIS
        json.dump(audit, f, indent=1, default=str)  # OL-AXIS
    log(f'F1(c) audit: pass {audit.get("pass")} ({audit.get("source")})')  # PYT-AXIS
    req_pass = all(c.get('pass') for c in comps if c['required'])  # OL-AXIS
    rec = {'what': 'Pythia axis R recomputation: every comparison', 'mode': gate.get('r_gate'),  # PYT-AXIS
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
