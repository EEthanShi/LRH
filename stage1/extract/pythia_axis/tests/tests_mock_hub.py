"""S0b test helpers (role pyt-axis-extract, 2026-10-10): a mock HfApi built from the designer's Hub metadata JSON, and
a fake downloader that copies synthetic repos into a hub-cache layout. No network, no Pythia weight.

Used only in S0b test mode (PYT_S0B_TEST=1, before the CODE FROZEN line); hashed in the CODE FROZEN line.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace as NS


class GatedRepoErrorMock(Exception):
    pass


def _is_lfs(name: str) -> bool:
    return name.endswith((".bin", ".safetensors", ".pt"))


class MockApi:
    """model_info / list_repo_refs from design/hub_pythia_main.json + design/hub_pythia_step143000.json.
    faults: None | gated | lfs | step_differs | branches | license | converts | auth | f16."""

    def __init__(self, design_dir: str, faults: str | None = None):
        d = Path(design_dir)
        self.main = json.load(open(d / "hub_pythia_main.json"))
        self.step = json.load(open(d / "hub_pythia_step143000.json"))
        self.faults = faults
        self.calls = []

    def _sib(self, rows):
        out = []
        for f, size, ident in rows:
            if _is_lfs(f):
                out.append(NS(rfilename=f, size=size, blob_id="x" * 40, lfs=NS(sha256=ident)))
            else:
                out.append(NS(rfilename=f, size=size, blob_id=ident, lfs=None))
        return out

    def model_info(self, model, revision=None, token=None, files_metadata=False):
        assert token is False, "token must be False"
        self.calls.append(("model_info", model, revision))
        r = copy.deepcopy(self.main[model])
        if self.faults == "auth" and model.endswith("pythia-70m"):
            raise GatedRepoErrorMock("403 Client Error: gated repo, access requires accepting the licence terms")
        step_commit = r["step143000"]
        if revision is not None and revision == step_commit:
            st = self.step[model]
            rows = [tuple(x) for x in st["step_siblings"]]
            if self.faults == "step_differs" and model.endswith("pythia-1b"):
                rows = [(f, s, ("0" * 64 if f == "pytorch_model.bin" else i)) for f, s, i in rows]
            sha = st["step_sha"]
        else:
            rows = [(s["f"], s["size"], s.get("lfs_sha256") or s["blob"]) for s in r["siblings"]]
            if self.faults == "lfs" and model.endswith("pythia-410m"):
                rows = [(f, s, ("1" * 64 if f == "pytorch_model.bin" else i)) for f, s, i in rows]
            sha = r["sha"] if revision is None or revision == r["sha"] else revision
        gated = "auto" if (self.faults == "gated" and model.endswith("pythia-160m")) else r["gated"]
        lic = "mit" if (self.faults == "license" and model.endswith("pythia-2.8b")) else r["license"]
        f16 = dict(r.get("safetensors_meta") or {})
        if self.faults == "f16" and model.endswith("pythia-6.9b"):
            f16["F16"] = f16["F16"] + 1
        card = NS(to_dict=lambda lic=lic, ds=r.get("datasets"): dict(license=lic, datasets=ds))
        return NS(sha=sha, gated=gated, private=r["private"], disabled=r["disabled"], last_modified=r["lastModified"],
                  card_data=card, siblings=self._sib(rows), safetensors=NS(parameters=f16))

    def list_repo_refs(self, model, token=None):
        assert token is False
        r = self.main[model]
        n = r["n_branches"] - (1 if self.faults == "branches" and model.endswith("pythia-1.4b") else 0)
        br = [NS(name="main", target_commit=r["main"]), NS(name="step143000", target_commit=r["step143000"])]
        br += [NS(name=f"step{k}", target_commit=f"{k:040d}") for k in range(n - 2)]
        conv = [NS(ref="refs/pr/1")] if (self.faults == "converts" and model.endswith("pythia-70m")) else []
        return NS(branches=br, converts=conv, tags=[])

    def fetch_config(self, model, rev):  # config.json content is not in the designer's metadata
        return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fake_downloader(fake_root: str, hub: Path):
    """dl(item, max_workers) copies fake_root/<repo dir>/<revision>/<allow-listed files> into the HF hub-cache layout
    hub/models--A--B/{blobs/<id>, snapshots/<rev>/<file> -> ../../blobs/<id>} and returns the snapshot path.
    redl(item, fname) re-copies one file (the force_download stand-in)."""
    fake_root = Path(fake_root)

    def _place(item, fname):
        src = fake_root / item["model"].replace("/", "--") / item["revision"] / fname
        repo = hub / ("models--" + item["model"].replace("/", "--"))
        (repo / "blobs").mkdir(parents=True, exist_ok=True)
        snap = repo / "snapshots" / item["revision"]
        snap.mkdir(parents=True, exist_ok=True)
        bid = _sha256(src) if _is_lfs(fname) else hashlib.sha1(b"blob %d\0" % src.stat().st_size + src.read_bytes()).hexdigest()
        blob = repo / "blobs" / bid
        tmp = blob.with_name(blob.name + ".incomplete")
        shutil.copyfile(src, tmp)
        os.replace(tmp, blob)
        link = snap / fname
        if not link.is_symlink():
            os.symlink(os.path.relpath(blob, snap), link)
        return snap

    def dl(item, max_workers):
        snap = None
        for f in item["allow"]:
            snap = _place(item, f)
        return snap

    def redl(item, fname):
        _place(item, fname)

    return dl, redl
