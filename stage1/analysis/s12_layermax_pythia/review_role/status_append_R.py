"""Role pyt-review: append ONE line to STATUS.md (append-only). Usage: python3 -I status_append_R.py < body.txt
The body is one line without the timestamp; this script prefixes '<UTC>Z [pyt-axis-review] ', appends the hash and
line count of STATUS.md before the append, refuses gate strings, and appends under flock. Prints the new line count."""
import fcntl, hashlib, os, re, sys, time

P = os.path.expanduser('~/hrh_v2_stage1_20261008/STATUS.md')
body = sys.stdin.read().strip()
if '\n' in body or not body:
    sys.exit('body must be one non-empty line')
for bad in ('CODE FROZEN', 'S3 GO', 'G0 GO', 'DEFERRED', 'Part B FROZEN', 'G1 GO'):
    if body.startswith(bad):
        sys.exit('body starts with a gate string: ' + bad)
with open(P, 'a+', encoding='utf-8') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0)
    old = f.read()
    n = old.count('\n')
    if not old.endswith('\n'):
        sys.exit('STATUS.md does not end with a newline; refusing')
    h = hashlib.sha256(old.encode('utf-8')).hexdigest()
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [pyt-axis-review] {body} STATUS.md before this line sha256 {h} ({n} lines)\n"
    f.seek(0, 2)
    f.write(line)
    f.flush()
    os.fsync(f.fileno())
    fcntl.flock(f, fcntl.LOCK_UN)
new = open(P, encoding='utf-8').read()
print(n + 1, hashlib.sha256(new.encode('utf-8')).hexdigest())
