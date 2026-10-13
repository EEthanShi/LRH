"""Shared helpers: deterministic sha256 seeds, hash splits, upper-triangle
extraction, and a fork-based row-parallel map.

No function in this package touches numpy's global RNG. Every random draw is
made from ``rng_from_key(key)`` where ``key`` is a human-readable string such
as ``"G0|P-Gauss|p=4|rep=3|noise"``; the seed is the first 8 bytes of
sha256(key) read little-endian.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from typing import Callable, Iterable, Sequence

import numpy as np


# ----------------------------------------------------------------------------
# Seeds
# ----------------------------------------------------------------------------

def seed_from_key(key: str) -> int:
    """64-bit seed = first 8 bytes (little endian) of sha256(key)."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def rng_from_key(key: str) -> np.random.Generator:
    """A fresh PCG64 generator seeded from sha256(key)."""
    return np.random.default_rng(seed_from_key(key))


def item_hash(salt: str, item_id) -> int:
    """64-bit integer hash of one item id under a salt."""
    return int.from_bytes(hashlib.sha256(f"{salt}|{item_id}".encode("utf-8")).digest()[:8], "little")


def hash_half(item_ids: Iterable, salt: str = "hrhv2-halves") -> np.ndarray:
    """Boolean mask, True = item in H1.

    Membership is decided per item (lowest bit of sha256(salt|id)), so the
    split is consistent across nested pools: an item keeps its half when the
    pool grows. Halves are therefore only approximately equal in size.
    """
    return np.array([(item_hash(salt, i) & 1) == 0 for i in item_ids], dtype=bool)


def hash_order(item_ids: Sequence, salt: str) -> np.ndarray:
    """Permutation that sorts items by sha256(salt|id) (nested-pool order)."""
    h = np.array([item_hash(salt, i) for i in item_ids], dtype=np.uint64)
    return np.argsort(h, kind="stable")


# ----------------------------------------------------------------------------
# Upper-triangle helpers (row-major order: (0,1),(0,2),...,(0,m-1),(1,2),...)
# ----------------------------------------------------------------------------

def n_pairs(m: int) -> int:
    return m * (m - 1) // 2


def upper_values(M: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Strict upper triangle of a square matrix, row-major, without
    materializing index arrays (memory ~ one float vector of m(m-1)/2)."""
    m = M.shape[0]
    out = np.empty(n_pairs(m), dtype=dtype)
    pos = 0
    for i in range(m - 1):
        n = m - 1 - i
        out[pos:pos + n] = M[i, i + 1:]
        pos += n
    return out


def upper_values_masked(M: np.ndarray, mask: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Strict upper triangle restricted to pairs with BOTH endpoints in mask."""
    idx = np.flatnonzero(mask)
    sub = M[np.ix_(idx, idx)]
    return upper_values(sub, dtype=dtype)


def upper_row_index(m: int) -> np.ndarray:
    """Row index i of each upper-triangle pair (row-major). int32 if m < 2**31."""
    dt = np.int32 if m < 2**31 else np.int64
    return np.repeat(np.arange(m - 1, dtype=dt), np.arange(m - 1, 0, -1))


# ----------------------------------------------------------------------------
# Fork-based row-parallel map
# ----------------------------------------------------------------------------

_SHARED: dict = {}


def shared() -> dict:
    """Arrays published to forked workers by ``map_rows`` (read-only)."""
    return _SHARED


def default_jobs() -> int:
    v = os.environ.get("HRHV2_JOBS")
    if v:
        return max(1, int(v))
    return 1


def map_rows(fn: Callable[[int, int], object], n: int, *, n_jobs: int | None = None,
             share: dict | None = None, chunk: int | None = None) -> list:
    """Apply ``fn(lo, hi)`` over row blocks [lo, hi) of range(n).

    ``share`` is published as a module global *before* forking, so workers
    read the parent's arrays copy-on-write (nothing is pickled). ``fn`` must
    be a module-level function that reads its inputs via ``util.shared()``.
    Results are returned in block order. Workers must not call BLAS.
    """
    global _SHARED
    if share is not None:
        _SHARED = share
    n_jobs = default_jobs() if n_jobs is None else max(1, int(n_jobs))
    if chunk is None:
        chunk = max(1, -(-n // (n_jobs * 4)))
    ranges = [(lo, min(n, lo + chunk)) for lo in range(0, n, chunk)]
    if n_jobs <= 1 or len(ranges) <= 1:
        return [fn(lo, hi) for lo, hi in ranges]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(n_jobs, len(ranges))) as pool:
        out = pool.starmap(fn, ranges)
    return out


class Timer:
    """Tiny wall-clock accumulator: ``with T('block'): ...`` then ``T.times``."""

    def __init__(self):
        self.times: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *exc):
                timer.times[name] = timer.times.get(name, 0.0) + time.perf_counter() - self_inner.t0
                return False

        return _Ctx()
