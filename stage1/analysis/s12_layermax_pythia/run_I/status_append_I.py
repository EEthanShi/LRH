"""Role pyt-run-I: append ONE line to STATUS.md (append-only, flock), tag [pyt-axis-run-I].

usage: python3 -B status_append_I.py <file with the message text (one line, no newline needed)>
Prints the STATUS sha256 and line count before and after. Refuses a message with a newline or a missing UTC prefix.
"""
import fcntl, hashlib, os, re, sys, time

STATUS = os.path.expanduser("~/hrh_v2_stage1_20261008/STATUS.md")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    msg = open(sys.argv[1], encoding="utf-8").read().strip("\n")
    assert "\n" not in msg and msg.strip(), "one non-empty line only"
    with open(STATUS, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        before = f.read()
        b_sha, b_n = hashlib.sha256(before.encode()).hexdigest(), before.count("\n")
        assert before.endswith("\n")
        utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"{utc} [pyt-axis-run-I] {msg.replace('{STATUS_BEFORE}', f'STATUS.md before this line sha256 {b_sha} ({b_n} lines)')}"
        assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \[pyt-axis-run-I\] ", line)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)
    a = open(STATUS, encoding="utf-8").read()
    print(f"appended line {a.count(chr(10))} at {utc}; before sha256 {b_sha} ({b_n} lines); after sha256 {sha(STATUS)} "
          f"({a.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
