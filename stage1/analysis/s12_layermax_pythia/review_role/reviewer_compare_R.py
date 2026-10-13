"""Role pyt-review (S8): compare the reviewer's own endpoints (review/role/reviewer_endpoints.json, from I's npz and
Part B, written by review_role/reviewer_endpoints_pyt_v2.py) with R-E and R-I (review/{E,I}/verdicts_review.json),
key by key, with the contract tolerance (1e-12 rel + 1e-15). Not frozen code; enters no verdict.
Pass: 0 mismatches against R-I, and against R-E only keys among the three known V-P5 keys.
usage: python3 -I reviewer_compare_R.py <out.json>"""
import hashlib, json, os, sys, time
PROJ = os.path.expanduser('~/hrh_v2_stage1_20261008')
RES = os.path.join(PROJ, 'results', 'S1_2_layermax_pythia')
KNOWN = {'D2|prh_clip+l2|cka_debiased|2.8b|c_minus_centry', 'D2|prh_clip+l2|cka_debiased|70m|c_minus_centry',
         'D5|prh_clip+l2|70m|160m|raw_gain'}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
fe = os.path.join(RES, 'review', 'role', 'reviewer_endpoints.json')
d = json.load(open(fe))
RV = {k: json.load(open(os.path.join(RES, 'review', k, 'verdicts_review.json')))['v'] for k in ('E', 'I')}


def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None:
        return a == b
    return abs(a - b) <= 1e-12 * max(abs(a), abs(b)) + 1e-15


sizes = ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']
pairs = []
for P in ('prh_clip+l2', 'l2'):
    o = d['per_pre'][P]
    for x in ('s_cal', 'unc'):
        h = o[x]['H1']
        pairs += [(f'H|{P}|{x}|H1|{k}', h[k]) for k in ('nV', 'R', 'C', 'D', 'p_perm', 'p_mag', 'p', 'confirmed')]
        pairs += [(f'H|{P}|{x}|H1|ci_lo', h['ci'][0]), (f'H|{P}|{x}|H1|ci_hi', h['ci'][1]), (f'H|{P}|{x}|gate', o[x]['gate'])]
        for hh, m in (('H3', 'mknn'), ('H4', 'cka_debiased')):
            t = o[x]['TOST'][m]
            pairs += [(f'H|{P}|{x}|{hh}|{k}', t[k]) for k in ('D', 'p_lo', 'p_hi', 'degenerate', 'ranks_identical', 'nV')]
            pairs.append((f'H|{P}|{x}|{hh}|p', t['p_tost']))
        if x == 's_cal':
            h2 = o[x]['H2']
            pairs += [(f'H|{P}|{x}|H2|{k}', h2[k]) for k in ('D', 'p_perm', 'p_mag')]
            t = o[x]['TOST']['ctm']
            pairs += [(f'H|{P}|{x}|H2|p_lo', t['p_lo']), (f'H|{P}|{x}|H2|p_hi', t['p_hi'])]
            if o[x]['holm']:
                pairs.append((f'H|{P}|{x}|holm_order', o[x]['holm']['order']))
                pairs += [(f'H|{P}|{x}|{hh}|holm_reject', r) for hh, r in o[x]['holm']['reject'].items()]
    c = o['common']
    for s in sizes:
        pairs += [(f'PP2ai|{P}|s_cal|median|{s}', c['PP2ai']['median'][s]), (f'PP2ai|{P}|s_cal|in_band|{s}', c['PP2ai']['in_band'][s]),
                  (f'R|{P}|median_scal|{s}', c['M4']['median_scal'][s])]
    pairs += [(f'PP2ai|{P}|s_cal|n_models_in_band', c['PP2ai']['n_models_in_band']), (f'PP2aii|{P}|s_cal|n', c['PP2aii']['n']),
              (f'PP2aii|{P}|s_cal|x', c['PP2aii']['x']), (f'PP2aii|{P}|s_cal|need', c['PP2aii']['need']),
              (f'PP5a|{P}|s_cal|n', c['PP5a']['n']), (f'PP5a|{P}|s_cal|x', c['PP5a']['x']),
              (f'PP2b|{P}|s_cal|n_decided', c['PP2b']['n_decided']), (f'PP2b|{P}|s_cal|x_correct', c['PP2b']['correct']),
              (f'PP2b|{P}|s_cal|n_wrong', c['PP2b']['wrong']), (f'PP2b|{P}|s_cal|n_undecided', c['PP2b']['undecided']),
              (f'PP2b|{P}|s_cal|verdict', c['PP2b']['verdict']), (f'R|{P}|n_scal_zero', c['M4']['n_scal_zero']),
              (f'fcnt|{P}|cka_biased|full|s_cal|fminus', c['f']['fminus']), (f'fcnt|{P}|cka_biased|full|s_cal|fplus', c['f']['fplus']),
              (f'D5|{P}|g', c['g']), (f'D5|{P}|h', c['h']), (f'R|{P}|s_cal|fragile', c['M1_fragile']),
              (f'R|{P}|s_cal|five|D1', c['M3_five']['D']), (f'R|{P}|s_cal|five|p_perm', c['M3_five']['p_perm']),
              (f'R|{P}|s_cal|five|carried', c['M3_five']['carried'])]
    for s, l in c['M1_loo'].items():
        pairs += [(f'R|{P}|s_cal|loo|{s}|{k}', l[v]) for k, v in (('D1', 'D'), ('p_perm', 'p_perm'), ('p_mag', 'p_mag'), ('p', 'p'))]
    for st, vv in c['steps'].items():
        pairs += [(f'D5|{P}|{st}|raw_gain', vv['raw_gain']), (f'D5|{P}|{st}|thr_gain', vv['thr_gain'])]
    pairs.append((f'K|{P}|s_cal|category', o['s_cal']['category'] if o['s_cal']['category'] in 'ABCDEF' else RV['I'][f'K|{P}|s_cal|category']))
mm = {k: [] for k in ('E', 'I')}
for impl in ('E', 'I'):
    for k, val in pairs:
        if k not in RV[impl] or not close(val, RV[impl][k]):
            mm[impl].append(k)
vp5 = d['vp5_effect']
max_order = sum(z['n_order_changes_357'] for P in vp5 for m in vp5[P] for z in vp5[P][m].values())
max_cens = sum((z['n_censoring_changes'] or 0) for P in vp5 for m in vp5[P] for z in vp5[P][m].values())
out = {'what': __doc__.split('\n')[0], 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'inputs': {os.path.relpath(p, PROJ): sha(p) for p in [fe] + [os.path.join(RES, 'review', k, 'verdicts_review.json') for k in ('E', 'I')]},
       'n_keys': len(pairs), 'mismatch_vs_R_I': mm['I'], 'mismatch_vs_R_E': mm['E'],
       'vp5_effect_order_changes_total': max_order, 'vp5_effect_censoring_changes_total': max_cens,
       'pass': (not mm['I']) and set(mm['E']) <= KNOWN and max_order == 0 and max_cens == 0}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('n_keys', 'mismatch_vs_R_I', 'mismatch_vs_R_E', 'vp5_effect_order_changes_total', 'vp5_effect_censoring_changes_total', 'pass')}))
