"""Prototype of the proposed serve-side PLE overlay (to live at minisgl/ple/overlay.py).

Pure numpy, torch-free. Holds per-edit DELTAS keyed by global n-gram row id, compiles them into one
sorted (ids, vals) pair, and applies them in place to the fp32 host staging rows right before the H2D
(`PLEEmbeddingSource.finish_rows`). Swapping the compiled pair is a single attribute assignment, so a
reader never sees a half-built overlay.
"""
from __future__ import annotations
import threading
import numpy as np


class PLEOverlay:
    def __init__(self, row_elems: int = 160, n_rows: int | None = None) -> None:
        self.row_elems = int(row_elems)
        self.n_rows = n_rows
        self._edits: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # edit_id -> (ids (K,), delta (K,E) f32)
        self._lock = threading.Lock()
        self.epoch = 0
        self._compiled = (np.empty(0, np.int64), np.empty((0, self.row_elems), np.float32))

    # -- control plane -------------------------------------------------------------------------
    def put(self, edit_id: str, ids, delta) -> np.ndarray:
        ids = np.asarray(ids, np.int64).reshape(-1)
        delta = np.asarray(delta, np.float32).reshape(ids.size, self.row_elems)
        if ids.size and (ids.min() < 0 or (self.n_rows is not None and ids.max() >= self.n_rows)):
            raise IndexError("overlay row id out of table range")
        with self._lock:
            self._edits[edit_id] = (ids.copy(), delta.copy())
            self._recompile()
        return ids

    def remove(self, edit_id: str) -> np.ndarray:
        with self._lock:
            ids, _ = self._edits.pop(edit_id, (np.empty(0, np.int64), None))
            self._recompile()
        return ids

    def _recompile(self) -> None:
        if not self._edits:
            comp = (np.empty(0, np.int64), np.empty((0, self.row_elems), np.float32))
        else:
            ids = np.concatenate([i for i, _ in self._edits.values()])
            vals = np.concatenate([v for _, v in self._edits.values()])
            u, inv = np.unique(ids, return_inverse=True)          # several edits on one row SUM
            acc = np.zeros((u.size, self.row_elems), np.float32)
            np.add.at(acc, inv, vals)
            comp = (u, acc)
        self._compiled = comp          # atomic publish
        self.epoch += 1

    # -- data plane (hot path) -----------------------------------------------------------------
    def apply(self, flat_ids: np.ndarray, host_rows: np.ndarray) -> int:
        """host_rows: (n, row_elems) f32 decoded rows for flat_ids (n,). In place. Returns #hits."""
        ids, vals = self._compiled
        if ids.size == 0 or flat_ids.size == 0:
            return 0
        pos = np.searchsorted(ids, flat_ids)
        pos[pos == ids.size] = 0
        hit = ids[pos] == flat_ids
        if not hit.any():
            return 0
        host_rows[hit] += vals[pos[hit]]
        return int(hit.sum())


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    N_ROWS = 320_001_536
    for K in (512, 10_000, 200_000):
        ov = PLEOverlay(160, N_ROWS)
        ov.put("e1", rng.choice(N_ROWS, K, replace=False), rng.standard_normal((K, 160)) * 1e-3)
        for n_tok in (1, 8, 2048):
            flat = rng.integers(0, N_ROWS, n_tok * 16)
            flat[: min(4, flat.size)] = ov._compiled[0][:min(4, flat.size)]   # force a few hits
            rows = np.zeros((flat.size, 160), np.float32)
            reps = 2000 if n_tok < 100 else 50
            t = time.perf_counter()
            for _ in range(reps):
                ov.apply(flat, rows)
            dt = (time.perf_counter() - t) / reps
            print(f"K={K:>7} tokens={n_tok:>5} rows={flat.size:>6}: apply {dt*1e6:8.1f} us")
    # correctness: two edits on one row sum, rollback restores
    ov = PLEOverlay(4); ov.put("a", [5, 9], np.ones((2, 4))); ov.put("b", [9], 2 * np.ones((1, 4)))
    r = np.zeros((3, 4), np.float32); ov.apply(np.array([9, 1, 5]), r)
    assert r[0, 0] == 3 and r[1, 0] == 0 and r[2, 0] == 1
    ov.remove("b"); r[:] = 0; ov.apply(np.array([9, 1, 5]), r); assert r[0, 0] == 1
    print("overlay semantics OK, epoch", ov.epoch)
