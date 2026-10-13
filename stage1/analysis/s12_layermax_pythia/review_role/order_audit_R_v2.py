"""Role pyt-review (S8), v2 (v1 dated E's copied primary-only files by birth; see the E_primary_note): freeze-order audit from STATUS.md and file birth times. READ-ONLY; not frozen code; opens no
verdict value. Usage: python3 -I order_audit_R.py > out.json. Birth times from statx (stat --format %W), mtimes %Y."""
import calendar, json, os, re, subprocess, time

PROJ = os.path.expanduser('~/hrh_v2_stage1_20261008')
AX = 'analysis/s12_layermax_pythia'
RES = 'results/S1_2_layermax_pythia'
lines = open(os.path.join(PROJ, 'STATUS.md'), encoding='utf-8').read().splitlines()


def ts(n):  # epoch of STATUS line n (1-based)
    return calendar.timegm(time.strptime(lines[n - 1][:20], '%Y-%m-%dT%H:%M:%SZ'))


def iso(e):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(e)) if e else None


def stats(paths):
    out = {}
    paths = [p for p in paths]
    for i in range(0, len(paths), 500):
        chunk = paths[i:i + 500]
        r = subprocess.run(['stat', '--format=%W %Y %n'] + chunk, capture_output=True, text=True, cwd=PROJ)
        for ln in r.stdout.splitlines():
            b, m, n = ln.split(' ', 2)
            out[n] = (int(b), int(m))
    return out


def files_under(rel, pat=None):
    res = []
    for root, dirs, fs in os.walk(os.path.join(PROJ, rel)):
        for f in fs:
            if f.startswith('._'):
                continue
            p = os.path.relpath(os.path.join(root, f), PROJ)
            if pat is None or re.search(pat, p):
                res.append(p)
    return sorted(res)


checks, fail = {}, []


def chk(name, ok, **detail):
    checks[name] = dict(ok=bool(ok), **detail)
    if not ok:
        fail.append(name)


def span(paths):
    st = stats(paths)
    if not st:
        return None, None, None, None, 0
    b = [v[0] for v in st.values()]
    m = [v[1] for v in st.values()]
    return min(b), max(b), min(m), max(m), len(st)


# 1. milestone order in STATUS (line numbers and timestamps)
L = {'plan_frozen': 216, 'code_frozen': 247, 'amend_A_G0': 249, 'g0_go': 254, 's3_go': 255, 's1': 256,
     'first_download': 259, 'downloads_done': 269, 'e1prov_decision': 267, 'g1_s4': 272, 's4_first': 273,
     'extraction_done': 282, 'partB_frozen': 285, 'E_role_start': 287, 'E_g1_s6': 297, 'E_s6_done': 301,
     'E_g1_s7': 302, 'E_s7_primary': 304, 'I_g1_s6': 310, 'I_g1_s7': 311, 'I_s7_primary': 312, 'l2_go': 313,
     'E_vp5_primary': 314, 'I_final': 315, 'E_l2_s8': 316, 'E_role_end': 318, 'I_vp5': 320}
tsl = {k: iso(ts(n)) for k, n in L.items()}
seq = ['plan_frozen', 'code_frozen', 'g0_go', 's3_go', 's1', 'first_download', 'e1prov_decision', 'downloads_done',
       'g1_s4', 's4_first', 'extraction_done', 'partB_frozen', 'E_g1_s6', 'E_s6_done', 'E_g1_s7', 'E_s7_primary',
       'I_g1_s6', 'I_g1_s7', 'I_s7_primary', 'l2_go', 'E_vp5_primary', 'I_final', 'E_l2_s8', 'E_role_end', 'I_vp5']
chk('status_milestone_order', all(ts(L[a]) <= ts(L[b]) for a, b in zip(seq, seq[1:])), times=tsl)
pyl = [i + 1 for i, l in enumerate(lines) if '[pyt-axis-' in l[:40]]
mono = all(ts(a) <= ts(b) for a, b in zip(pyl, pyl[1:]))
chk('pyt_axis_status_timestamps_monotone', mono, n_lines=len(pyl), first=pyl[0], last=pyl[-1])

# 2. plan born before its freeze line
b0, b1, m0, m1, n = span([f'{AX}/ANALYSIS_PLAN_PYTHIA.md'])
chk('plan_born_before_freeze_line', b1 <= ts(216) and m1 <= ts(216), birth=iso(b1), mtime=iso(m1), line=tsl['plan_frozen'])

# 3. every manifest file born and last modified before CODE FROZEN
man = json.load(open(os.path.join(PROJ, AX, 'S0B_CODE_MANIFEST.json')))
mp = [f['path'] for f in man['files']]
b0, b1, m0, m1, n = span(mp)
chk('code_born_and_modified_before_code_frozen', b1 <= ts(247) and m1 <= ts(247), n=n, latest_birth=iso(b1),
    latest_mtime=iso(m1), line=tsl['code_frozen'])

# 4. no Pythia weight before S3 GO; downloads after S3 GO
hf = files_under('extract/pythia_axis/hf_home')
st = stats(hf)
before = [p for p, (b, m) in st.items() if b < ts(255) and 'pythia' in p.lower()]
chk('no_pythia_download_before_S3_GO', not before, n_files=len(st), born_before=before[:20],
    min_birth=iso(min(v[0] for v in st.values())), max_birth=iso(max(v[0] for v in st.values())), line=tsl['s3_go'])

# 5. features: primary and _tok1 stacks born in the S4 window (after G1 before S4, before EXTRACTION DONE)
fp = files_under('features/wit1024_pythia_axis', r'EleutherAI__pythia-[^/]+\.npy$')
b0, b1, m0, m1, n = span(fp)
chk('features_born_in_S4_window', ts(272) <= b0 and b1 <= ts(282) and m1 <= ts(282), n=n, first=iso(b0), last=iso(b1),
    window=[tsl['g1_s4'], tsl['extraction_done']])

# 6. Part B: files born after EXTRACTION DONE and before the Part B FROZEN line
pbf = files_under(f'{RES}/partB') + [f'{AX}/PARTB_PREDICTIONS_PYT.md']
b0, b1, m0, m1, n = span(pbf)
chk('partB_born_between_extraction_and_freeze', ts(282) <= b0 and b1 <= ts(285) and m1 <= ts(285), n=n, first=iso(b0),
    last=iso(b1), latest_mtime=iso(m1), window=[tsl['extraction_done'], tsl['partB_frozen']])

# 7. every score file born after Part B FROZEN (F1(c), restated) and the main pair files after their G1 / l2 GO lines
sc = []
for d in ('cache', 'early', 'indep', 'forecast', 'review'):
    sc += files_under(f'{RES}/{d}')
b0, b1, m0, m1, n = span(sc)
chk('score_files_born_after_partB_frozen', b0 > ts(285), n=n, first=iso(b0), line=tsl['partB_frozen'])
win = {'early/pairs/prh_clip+l2': (302, 304), 'early/pairs/l2': (313, 316),
       'indep/prh_clip+l2/pairs': (311, 312), 'indep/l2/pairs': (313, 315)}
for d, (a, z) in win.items():
    b0, b1, m0, m1, n = span(files_under(f'{RES}/{d}'))
    chk(f'pairs_born_in_window|{d}', ts(a) <= b0 and b1 <= ts(z) and m1 <= ts(z), n=n, first=iso(b0), last=iso(b1),
        latest_mtime=iso(m1), window=[iso(ts(a)), iso(ts(z))])


def birth(p):
    s = stats([p])
    return s[p][0] if p in s else None


# 8. independence order of the V-P5 exchange (birth times of the files each side wrote)
bt = {k: birth(f'{RES}/{p}') for k, p in {
    'I_primary_summary': 'indep/summary.json.prev-20261010T095714Z',
    'I_order_file_primary': 'indep/cross_order_1_prh_clip+l2.json',
    'E_primary_verdicts': 'early/verdicts_pyt.json.prev-20261010T095532Z',
    'E_vp5_primary': 'early/validation/VP5_E_vs_I.prh_clip+l2.json',
    'I_final_summary': 'indep/summary.json', 'I_final_verdicts': 'indep/verdicts_pyt.json',
    'forecast_I': 'forecast/I/verdicts_forecast.json', 'I_order_file_all': 'indep/cross_order_1-2-3_prh_clip+l2+l2.json',
    'E_final_verdicts': 'early/verdicts_pyt.json', 'forecast_E': 'forecast/E/verdicts_forecast.json',
    'E_vp5_all': 'early/validation/VP5_E_vs_I.json', 'I_vp5_all': 'indep/validation_cross_1-2-3_prh_clip+l2+l2.json',
    'R_first': 'review/E/verdicts_review.json'}.items()}
order = [('I_primary_summary', 'I_order_file_primary'), ('E_primary_verdicts', 'E_vp5_primary'),
         ('I_final_summary', 'I_order_file_all'), ('forecast_I', 'I_order_file_all'),
         ('E_final_verdicts', 'forecast_E'), ('forecast_E', 'E_vp5_all'), ('I_final_verdicts', 'E_vp5_all'),
         ('forecast_I', 'E_vp5_all'), ('E_final_verdicts', 'I_vp5_all'), ('forecast_E', 'R_first'),
         ('forecast_I', 'R_first'), ('I_vp5_all', 'R_first'), ('E_vp5_all', 'R_first')]
# v2: E's primary-only files were COPIED aside at 09:55:32Z (new inode, birth = copy time, mtime preserved), so their
# birth cannot date them; they are dated by their mtime and by the STATUS line that logged their hashes (line 304).
pv = f'{RES}/early/verdicts_pyt.json.prev-20261010T095532Z'
ps_ = f'{RES}/early/summary.json.prev-20261010T095532Z'
st2 = stats([pv, ps_])
import hashlib
def _sha(p):
    h = hashlib.sha256(); f = open(os.path.join(PROJ, p), 'rb')
    for c in iter(lambda: f.read(1 << 22), b''):
        h.update(c)
    return h.hexdigest()
e_prim = {'verdicts_sha256': _sha(pv), 'summary_sha256': _sha(ps_), 'verdicts_mtime': iso(st2[pv][1]),
          'summary_mtime': iso(st2[ps_][1]), 'copy_birth': iso(st2[pv][0]),
          'hashes_in_line_304': _sha(pv) in lines[303] and _sha(ps_) in lines[303], 'line_304_utc': iso(ts(304))}
first_I_pair = span(files_under(f'{RES}/indep/prh_clip+l2/pairs'))[0]
e_prim['first_I_pair_birth'] = iso(first_I_pair)
e_ok = e_prim['hashes_in_line_304'] and ts(304) < first_I_pair and st2[pv][1] <= ts(304)
res = {f'{a} <= {b}': (bt[a] is not None and bt[b] is not None and bt[a] <= bt[b]) for a, b in order
       if a != 'E_primary_verdicts'}
res['E_primary_verdicts (mtime and STATUS line 304) before any I pair file and before E_vp5_primary'] = bool(
    e_ok and st2[pv][1] <= bt['E_vp5_primary'])
chk('vp5_exchange_order_by_birth', all(res.values()), pairs=res, births={k: iso(v) for k, v in bt.items()},
    E_primary_note=e_prim)

# 9. GPU: ledger total and no Pythia GPU step inside an S2.2 / S2.3 GPU window (from STATUS lines)
led = [l.split('\t') for l in open(os.path.join(PROJ, RES, 'logs', 'gpu_ledger.tsv')).read().splitlines()[1:]]
tot = float(led[-1][5])
ep = lambda s: calendar.timegm(time.strptime(s, '%Y-%m-%dT%H:%M:%SZ'))
s23 = {'S2.3 A1 resume (GPU)': ('2026-10-10T03:12:54Z', '2026-10-10T07:37:07Z'),
       'S2.3 A2 completion (GPU workers)': ('2026-10-10T08:33:03Z', '2026-10-10T09:10:16Z'),
       'S2.3 A3 analysis (declared CPU only)': ('2026-10-10T09:20:47Z', '2026-10-10T09:25:17Z'),
       'S2.3 A4 analysis (declared CPU only)': ('2026-10-10T09:31:32Z', '2026-10-10T09:31:58Z')}
ov = []
for r in led:
    a, z = ep(r[1]), ep(r[2])
    for k, (s0, s1) in s23.items():
        if a < ep(s1) and z > ep(s0):
            ov.append({'step': r[0], 'start': r[1], 'end': r[2], 'overlaps': k})
gpu_ov = [o for o in ov if 'GPU' in o['overlaps']]
# 10-minute idle after the end of each S2.3 GPU window before the next Pythia GPU step
idle = {}
for k, (s0, s1) in s23.items():
    if 'GPU' not in k:
        continue
    nxt = min((ep(r[1]) for r in led if ep(r[1]) > ep(s1)), default=None)
    idle[k] = {'s23_end': s1, 'next_pythia_gpu_start': iso(nxt), 'gap_min': None if nxt is None else (nxt - ep(s1)) / 60}
chk('gpu_ledger_within_60', tot <= 60, total_min=tot, n_steps=len(led))
chk('no_pythia_gpu_step_inside_s23_gpu_window', not gpu_ov, overlaps_gpu=gpu_ov, overlaps_cpu_only=[o for o in ov if 'GPU' not in o['overlaps']],
    gap_after_s23_gpu=idle)
chk('gap_after_each_s23_gpu_window_ge_10min', all(v['gap_min'] is None or v['gap_min'] >= 10 for v in idle.values()), gaps=idle)

print(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'status_lines': len(lines),
                  'pass': not fail, 'fail': fail, 'checks': checks}, indent=1, default=str))
