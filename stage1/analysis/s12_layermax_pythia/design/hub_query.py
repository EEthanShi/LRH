import json, os, sys
for k in ("HF_TOKEN","HUGGING_FACE_HUB_TOKEN"): os.environ.pop(k, None)
from huggingface_hub import HfApi
import huggingface_hub
api = HfApi()
models = ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b",
          "EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b"]
out = {"hub_version": huggingface_hub.__version__}
for m in models:
    r = {}
    try:
        info = api.model_info(m, token=False, files_metadata=True)
        r["sha"] = info.sha; r["gated"] = info.gated; r["private"] = info.private; r["disabled"] = getattr(info, "disabled", None)
        r["lastModified"] = str(info.last_modified)
        cd = info.card_data.to_dict() if info.card_data else {}
        r["license"] = cd.get("license"); r["datasets"] = cd.get("datasets")
        r["tags"] = [t for t in (info.tags or []) if t.startswith("license") or t in ("safetensors","pytorch","gpt_neox")]
        r["config"] = info.config
        sib = []
        for s in info.siblings:
            d = {"f": s.rfilename, "size": s.size, "blob": s.blob_id}
            if s.lfs: d["lfs_sha256"] = s.lfs.sha256 if hasattr(s.lfs, "sha256") else s.lfs.get("sha256")
            sib.append(d)
        r["siblings"] = sib
        r["safetensors_meta"] = (info.safetensors.parameters if info.safetensors else None)
        try:
            refs = api.list_repo_refs(m, token=False)
            br = {b.name: b.target_commit for b in refs.branches}
            r["n_branches"] = len(br); r["main"] = br.get("main"); r["step143000"] = br.get("step143000")
            r["converts"] = [c.ref for c in (refs.converts or [])]
        except Exception as e:
            r["refs_error"] = repr(e)
    except Exception as e:
        r["error"] = repr(e)
    out[m] = r
json.dump(out, sys.stdout, indent=1, default=str)
