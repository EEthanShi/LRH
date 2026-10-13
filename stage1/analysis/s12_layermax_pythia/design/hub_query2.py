import json, os, sys
for k in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(k, None)
from huggingface_hub import HfApi
api = HfApi()
base = json.loads(sys.stdin.readline())
out = {}
for m, step in base.items():
    r = {}
    try:
        info = api.model_info(m, revision=step, token=False, files_metadata=True)
        r["step_sha"] = info.sha
        r["step_siblings"] = [(s.rfilename, s.size, (s.lfs.sha256 if s.lfs else s.blob_id)) for s in info.siblings]
    except Exception as e:
        r["step_err"] = repr(e)
    try:
        cs = api.list_repo_commits(m, token=False)
        r["commits"] = [(c.commit_id[:12], str(c.created_at), c.title[:90]) for c in cs[:12]]
        r["n_commits"] = len(cs)
    except Exception as e:
        r["commits_err"] = repr(e)
    out[m] = r
json.dump(out, sys.stdout, indent=1, default=str)
