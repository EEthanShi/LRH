"""Role pyt-review (S8), entry/end check, READ-ONLY. Not frozen code; computes no score; opens no verdict value.
Checks: plan hash; S0B manifest hash and its 189 files (hash-equal, read-only); exactly one pyt-axis CODE FROZEN line;
Part B FROZEN line hashes; 7 primary + 7 _tok1 stacks vs the S4 lines; E and I outputs vs their STATUS lines;
pair-file digests (sha256 of `sha256sum * | sort -k2`) vs STATUS. Prints one JSON object to stdout."""
import hashlib, json, os, re, subprocess, sys, time

PROJ = os.path.expanduser('~/hrh_v2_stage1_20261008')
AX = os.path.join(PROJ, 'analysis', 's12_layermax_pythia')
RES = os.path.join(PROJ, 'results', 'S1_2_layermax_pythia')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


out = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'checks': {}, 'fail': []}
C = out['checks']


def chk(name, ok, detail=None):
    C[name] = {'ok': bool(ok), 'detail': detail}
    if not ok:
        out['fail'].append(name)


status = open(os.path.join(PROJ, 'STATUS.md'), encoding='utf-8').read()
lines = status.split('\n')
if lines and lines[-1] == '':
    lines = lines[:-1]
out['status_lines'] = len(lines)
out['status_sha256'] = hashlib.sha256(status.encode('utf-8')).hexdigest()

chk('plan', sha(os.path.join(AX, 'ANALYSIS_PLAN_PYTHIA.md')) == '6de896036146993a5c79cd1bc583201cf0c38d240752a56871ca914f47879e7b')
ts = r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ '
cf = [i + 1 for i, l in enumerate(lines) if re.match(ts + r'\[pyt-axis-[A-Za-z0-9-]+\] CODE FROZEN', l)]
chk('code_frozen_only_line_247', cf == [247], cf)
pb = [i + 1 for i, l in enumerate(lines) if re.match(ts + re.escape('[pyt-axis-predict] Part B FROZEN'), l)]
chk('partB_only_line_285', pb == [285], pb)
s3 = [i + 1 for i, l in enumerate(lines) if re.match(ts + r'\[pyt-axis-[A-Za-z0-9-]+\] S3 GO', l)]
chk('s3go_only_line_255', s3 == [255], s3)
g0 = [i + 1 for i, l in enumerate(lines) if re.match(ts + r'\[pyt-axis-[A-Za-z0-9-]+\] G0 GO', l)]
chk('g0go_line_254', g0 == [254], g0)
dfr = [i + 1 for i, l in enumerate(lines) if re.match(ts + r'\[pyt-axis-[A-Za-z0-9-]+\] DEFERRED', l)]
chk('no_deferred_line', dfr == [], dfr)

man_p = os.path.join(AX, 'S0B_CODE_MANIFEST.json')
man_sha = sha(man_p)
chk('manifest_sha', man_sha == 'ca600c80b94277328da6f9c2ee8e26785776bbfd0d80f8669b0a050098dcb237', man_sha)
man = json.load(open(man_p))
bad, writable = [], []
for f in man['files']:
    p = os.path.join(PROJ, f['path'])
    if not os.path.isfile(p) or sha(p) != f['sha256']:
        bad.append(f['path'])
    elif os.stat(p).st_mode & 0o222:
        writable.append(f['path'])
chk('manifest_files_equal', not bad, {'n': len(man['files']), 'bad': bad})
chk('manifest_files_readonly', not writable, writable)
chk('manifest_in_cf_line', man_sha in lines[246], None)

L285 = lines[284]
pbfiles = {'analysis/s12_layermax_pythia/PARTB_PREDICTIONS_PYT.md': None,
           'results/S1_2_layermax_pythia/partB/partB_features_pyt.json': None,
           'analysis/s12_layermax_pythia/partb_pyt.py': None}
for rp in pbfiles:
    h = sha(os.path.join(PROJ, rp))
    pbfiles[rp] = h
    chk('partB_hash_in_line|' + rp, h in L285, h)
h = sha(os.path.join(RES, 'partB', 'clip_thresholds_partB_pyt.json'))
chk('partB_clip_hash_prefix_in_line', h[:6] in L285, h)

# features vs S4 lines 273-279 (full sha256 in each line)
sizes = ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']
feat = {}
for k, s in enumerate(sizes):
    ln = lines[272 + k]
    shas = re.findall(r'[0-9a-f]{64}', ln)
    p = os.path.join(PROJ, 'features', 'wit1024_pythia_axis', f'EleutherAI__pythia-{s}.npy')
    pt = os.path.join(PROJ, 'features', 'wit1024_pythia_axis', '_tok1', f'EleutherAI__pythia-{s}.npy')
    h, ht = sha(p), sha(pt)
    feat[s] = {'primary': h[:16], 'tok1': ht[:16], 'primary_in_line': h in shas, 'tok1_in_line': ht in shas}
chk('features_equal_S4', all(v['primary_in_line'] and v['tok1_in_line'] for v in feat.values()), feat)

want = {  # file -> (sha256 prefix, STATUS line where it was logged)
    'early/summary.json': ('abdc0d069aeb471fab67edccc90e36a567afb028c3e5791e412513c5a6b76143', 318),
    'early/verdicts_pyt.json': ('99e0c54c84b9f439f45ad13f04d3f521ba11b97c2b40360b202fea5f66f626cb', 318),
    'early/REPORT.md': ('61cf2a6ad3787982df9056a67109feee5c3773a4cd15c8ee5df7b9037cb6d93f', 318),
    'early/hashes.json': ('8c374def6168c77362e691b0602bd662837e8c816267d2163d5a2e34bce1cac9', 318),
    'early/validation/VP5_E_vs_I.json': ('f97aa474b5807dea65b58dddf7b2d50faf48f6658e681ef3b94050e4fabf174c', 318),
    'forecast/E/verdicts_forecast.json': ('d0f7146bf5429aebaf5532683c44b736005b97daf6c67044b0425bda6839516b', 316),
    'forecast/E/forecast_detail.json': ('123546671236f5c469d009636805ffe1b3df8bbfa3d69badd4ddd3863dc11596', 316),
    'indep/summary.json': ('8733c8c1e699988cfaa1539de628142d36893b1b530768ceae96122cb32153bf', 315),
    'indep/verdicts_pyt.json': ('797934c2a80d84ad0b1d262f6bb9864496468a93e0f8d5cfbab243895cbcf4d9', 315),
    'indep/REPORT.md': ('c637dd82b8e431583d48a7698dc379918de0a548604ee45ffcd0cce756b410d0', 320),
    'indep/validation_cross_1-2-3_prh_clip+l2+l2.json': ('0632dbe68791fff5ef262151b2b04637d9dd7aacbb935ccc68ba4a10627a9786', 320),
    'indep/validation_cross.json': ('a229e77679ffd95fe7b194761db2a468e74a0f3abf6b8ba37d94a75d93040ae0', 320),
    'forecast/I/verdicts_forecast.json': ('d78d0b2b10d71b0cdb0156f28394cc90973990ebe7a4a99755f3d836b585b1d7', 320),
    'cache/clip_thresholds.json': ('a778bf56eff6cd9aff1f095346e306e7be42ab054623618818c8bfd312653a49', 301),
}
outs = {}
for rp, (w, ln) in want.items():
    h = sha(os.path.join(RES, rp))
    outs[rp] = {'sha256': h, 'ok': h == w, 'status_line': ln, 'in_line': w in lines[ln - 1]}
chk('E_I_outputs_equal_status', all(v['ok'] and v['in_line'] for v in outs.values()), outs)

dig = {}
for d, w in (('early/pairs/prh_clip+l2', '032639c8369c82d0ebfb872eae1ace4ba6da79d2490de0e99c4f3c7a2cf6f4e3'),
             ('early/pairs/l2', '18dd211db9815df49144230742d2bbaf07c16ccf58968f3d8f0d639905139cf8'),
             ('indep/prh_clip+l2/pairs', 'a59afc02921b5fb0926793221acf461d06b6f0c66be1c1805f36f17adc01d0f9'),
             ('indep/l2/pairs', '9bf46b3495c5fade0c8cebaf5485705c00f82f89d4c9eddb1563e80c9f0a4745')):
    full = os.path.join(RES, d)
    # the roles' digest: sha256 of `sha256sum * | sort -k2` in the directory (shell locale order)
    h = subprocess.run("sha256sum * | sort -k2 | sha256sum", shell=True, cwd=full, capture_output=True,
                       text=True).stdout.split()[0]
    dig[d] = {'n': len(os.listdir(full)), 'digest': h, 'ok': h == w}
chk('pair_digests_equal_status', all(v['ok'] for v in dig.values()), dig)
chk('review_root_absent_or_empty', not os.path.exists(os.path.join(RES, 'review')) or not os.listdir(os.path.join(RES, 'review')),
    sorted(os.listdir(RES)))
p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True)
out['gpu_compute_apps'] = p.stdout.strip()
out['pass'] = not out['fail']
print(json.dumps(out, indent=1))
