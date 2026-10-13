"""Stage 2 data preparation (CIFAR-5m, Nakkiran et al. 2021).

CIFAR-5m parts are stored (uncompressed) .npz files, so the X/Y members are
memory-mapped in place: we record the byte offset of each member's array data.

Pools
  train pool  = parts 0-4 (all images; global index = part offset + local index)
  held-out    = part 5 only; never used for training. From part 5, items are
                ordered by sha256("hrhv2-stage2|heldout|part5:<i>") and taken as
                  S     = first 2048   (evaluation set for all criteria)
                  bank  = next 10000   (labelled kNN-accuracy bank)
                  hold  = next 4096    (held-out loss / generalization gap)
                  (rest of part 5 unused)
Outputs (data/): c5m_index.json, S_X.npy, S_Y.npy, S_idx.npy, bank_*.npy, hold_*.npy
"""
import hashlib
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.expanduser("~/hrh_v2_stage2_20261008")
C5M = os.path.join(ROOT, "data", "cifar5m")
OUT = os.path.join(ROOT, "data")


def member_offset(path, name):
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        if info.compress_type != 0:
            raise RuntimeError(f"{path}:{name} is compressed")
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        lh = f.read(30)
        if lh[:4] != b"PK\x03\x04":
            raise RuntimeError("bad local header")
        n_name = int.from_bytes(lh[26:28], "little")
        n_extra = int.from_bytes(lh[28:30], "little")
        start = info.header_offset + 30 + n_name + n_extra
        f.seek(start)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        data_off = f.tell()
    if fortran:
        raise RuntimeError("fortran order")
    return {"offset": int(data_off), "shape": list(shape), "dtype": str(dtype)}


def main():
    idx = {"parts": []}
    for p in range(6):
        path = os.path.join(C5M, f"part{p}.npz")
        if not os.path.exists(path + ".ok"):
            sys.exit(f"part{p} not downloaded/verified")
        X = member_offset(path, "X.npy")
        Y = member_offset(path, "Y.npy")
        assert X["dtype"] == "uint8" and X["shape"][1:] == [32, 32, 3], X
        assert X["shape"][0] == Y["shape"][0]
        idx["parts"].append({"path": path, "X": X, "Y": Y, "n": X["shape"][0]})
    sizes = [q["n"] for q in idx["parts"][:5]]
    idx["train_parts"] = [0, 1, 2, 3, 4]
    idx["train_sizes"] = sizes
    idx["train_total"] = int(sum(sizes))
    idx["heldout_part"] = 5
    # held-out selections from part 5
    n5 = idx["parts"][5]["n"]
    h = np.array([int.from_bytes(hashlib.sha256(f"hrhv2-stage2|heldout|part5:{i}".encode()).digest()[:8], "little")
                  for i in range(n5)], dtype=np.uint64)
    order = np.argsort(h, kind="stable")
    sel = {"S": order[:2048], "bank": order[2048:12048], "hold": order[12048:16144]}
    q = idx["parts"][5]
    X5 = np.memmap(q["path"], dtype=np.uint8, mode="r", offset=q["X"]["offset"], shape=tuple(q["X"]["shape"]))
    Y5 = np.memmap(q["path"], dtype=np.dtype(q["Y"]["dtype"]), mode="r", offset=q["Y"]["offset"], shape=tuple(q["Y"]["shape"]))
    for name, ii in sel.items():
        np.save(os.path.join(OUT, f"{name}_idx.npy"), ii.astype(np.int64))
        np.save(os.path.join(OUT, f"{name}_X.npy"), np.ascontiguousarray(X5[ii]))
        np.save(os.path.join(OUT, f"{name}_Y.npy"), np.asarray(Y5[ii]).astype(np.int64))
        idx[f"{name}_n"] = int(len(ii))
        idx[f"{name}_label_counts"] = np.bincount(np.asarray(Y5[ii]).astype(np.int64), minlength=10).tolist()
    # sanity: label range over a sample of each part
    for p, q in enumerate(idx["parts"]):
        Y = np.memmap(q["path"], dtype=np.dtype(q["Y"]["dtype"]), mode="r", offset=q["Y"]["offset"], shape=tuple(q["Y"]["shape"]))
        y = np.asarray(Y[:: 997])
        q["label_min"], q["label_max"] = int(y.min()), int(y.max())
    with open(os.path.join(OUT, "c5m_index.json"), "w") as f:
        json.dump(idx, f, indent=1)
    S = np.load(os.path.join(OUT, "S_X.npy"))
    print(json.dumps({k: v for k, v in idx.items() if k != "parts"}, indent=1))
    print("S", S.shape, S.dtype, S.mean())


if __name__ == "__main__":
    main()
