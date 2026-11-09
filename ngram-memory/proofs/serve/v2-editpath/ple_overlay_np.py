"""Torch-free PLE row overlay for minisgl df29117 (prototype; NOT installed in the fork).

Hook point: `PLEEmbeddingSource.finish_rows`, directly after source.py:243
    self._host_rows[:flat_n] = self.table.decode_raw(raw)
`_host_rows` is a numpy view of the pinned fp32 staging tensor (source.py:114-117), so the overlay
touches numpy only. Base decode for the deployed table is FusedScaleRowTable.decode_raw ->
dequant_int6_g32 (row_table.py:880-884, 109-148): fp32 out. The device then casts fp32 -> bf16
(source.py:245), so the overlay may store either
  * "delta": fp32 deltas, served = bf16(base + delta)                       (v1 plan, 648 B/row)
  * "bf16":  replacement rows as bf16 bits, served = row                    (328 B/row)
and the two are bit-identical on the device when row = bf16(base + delta) (requant_delta.py asserts
it). Compiled arrays can live in a file-backed mmap (e.g. /dev/shm) so both TP ranks share ONE copy.

Self-test (run with torch blocked): builds a tiny int6 packed+scale checkpoint, opens it with the
fork's own `open_qwen4exp_ngram_table`, runs the same begin/wait/decode sequence as source.py
:229/:241/:243, applies the overlay and checks the result.
"""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import tempfile

import numpy as np


def bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class PLEOverlayNP:
    """Sorted (ids, payload) pair; published by ONE attribute assignment; epoch bumps per change."""

    def __init__(self, row_elems: int, n_rows: int, mode: str = "bf16") -> None:
        assert mode in ("bf16", "delta")
        self.row_elems, self.n_rows, self.mode = int(row_elems), int(n_rows), mode
        self.epoch = 0
        self._compiled = (np.empty(0, np.int64), self._empty_payload(0))

    def _empty_payload(self, k):
        return np.empty((k, self.row_elems), np.uint16 if self.mode == "bf16" else np.float32)

    def publish(self, ids: np.ndarray, payload: np.ndarray, shm_path: str | None = None) -> None:
        """Install a compiled set. `ids` unique; `payload` bf16 bits (uint16) or fp32 deltas.
        With shm_path the arrays are written to a file and re-opened as read-only memmaps, which is
        how two rank processes would share one copy (the second rank only opens it)."""
        o = np.argsort(ids, kind="stable")
        ids, payload = np.ascontiguousarray(ids[o], np.int64), np.ascontiguousarray(payload[o])
        if ids.size and (ids[0] < 0 or ids[-1] >= self.n_rows or (np.diff(ids) == 0).any()):
            raise ValueError("overlay ids must be unique and in range")
        if shm_path:
            np.save(shm_path + ".ids.npy", ids); np.save(shm_path + ".rows.npy", payload)
            ids = np.load(shm_path + ".ids.npy", mmap_mode="r")
            payload = np.load(shm_path + ".rows.npy", mmap_mode="r")
        self._compiled = (ids, payload)   # single assignment = atomic publish w.r.t. the hook
        self.epoch += 1

    def apply(self, flat_ids: np.ndarray, host_rows: np.ndarray) -> int:
        """In place on the fp32 host rows. Returns hits. Pure numpy."""
        u, pay = self._compiled
        if u.size == 0:
            return 0
        pos = np.searchsorted(u, flat_ids)
        pos[pos == u.size] = 0
        hit = u[pos] == flat_ids
        if not hit.any():
            return 0
        sel = np.flatnonzero(hit)
        if self.mode == "bf16":
            host_rows[sel] = bits_to_f32(pay[pos[sel]])
        else:
            host_rows[sel] += pay[pos[sel]]
        return int(sel.size)

    def nbytes_per_row(self) -> int:
        return 8 + self.row_elems * (2 if self.mode == "bf16" else 4)


# ----------------------------------------------------------------------------- self-test
def _write_safetensors(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name, (dt, arr) in tensors.items():
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b); blobs.append(b)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in blobs:
            f.write(b)


def _selftest():
    blocked = sys.modules.get("torch", "absent") is None
    repo = "/home/user/patcarter883/minisglang-rdna4/python/minisgl/weights/row_table.py"
    spec = importlib.util.spec_from_file_location("row_table_df29117", repo)
    rt = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rt; spec.loader.exec_module(rt)
    rng = np.random.default_rng(1)
    H, NS, D = 64, 2, 160
    pre = rt.PLE_PREFIX
    tens = {}
    for s in range(NS):
        tens[f"{pre}.ngram_embedding.shard_{s}.weight_packed"] = ("U8", rng.integers(0, 256, (H, 120), dtype=np.uint8))
        tens[f"{pre}.ngram_embedding.shard_{s}.weight_scale"] = ("F16", (rng.random((H, 5)) * 1e-3 + 1e-4).astype(np.float16))
    tens[f"{pre}.ngram_heads_vocab_sizes"] = ("I64", np.full(16, 8, np.int64))
    tens[f"{pre}.ngram_heads_offsets"] = ("I64", np.arange(16, dtype=np.int64) * 8)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model-00001-of-00001.safetensors")
        _write_safetensors(p, tens)
        table, heads = rt.open_qwen4exp_ngram_table([p], workers=16)
        assert type(table).__name__ == "FusedScaleRowTable" and table.row_bytes == 130 and table.row_elems == 160
        T = 4                                   # tokens; 16 heads each -> 64 rows (>=16 -> threaded pool)
        flat = rng.integers(0, table.n_rows, T * 16).astype(np.int64)
        raw_buf = np.empty((flat.size, table.row_bytes), np.uint8)       # source._raw
        host_rows = np.empty((flat.size, D), np.float32)                 # source._host_rows
        h = table.gather_raw_into_async(flat, raw_buf)                   # source.py:229
        raw = table.gather_wait(h)                                       # source.py:241
        host_rows[:] = table.decode_raw(raw)                             # source.py:243 (dequant, fp32)
        base = host_rows.copy()
        # an edit on 5 distinct rows that the batch reads (some repeatedly)
        eids = np.unique(flat)[:5]
        delta = (rng.standard_normal((5, D)) * 1e-2).astype(np.float32)
        # base rows of the edit, read through the same table (what a compile step would do)
        ebase = table.gather(eids)
        ov_b = PLEOverlayNP(table.row_elems, table.n_rows, "bf16")
        ov_d = PLEOverlayNP(table.row_elems, table.n_rows, "delta")
        shm = os.path.join(td, "ov")
        ov_b.publish(eids, f32_to_bf16_bits(ebase + delta), shm_path=shm)
        ov_d.publish(eids, delta)
        hb, hd = base.copy(), base.copy()
        nb, nd = ov_b.apply(flat, hb), ov_d.apply(flat, hd)
        exp_hits = int(np.isin(flat, eids).sum())
        assert nb == nd == exp_hits
        # device cast emulation: bf16(host)  -> both modes must be bit-identical on the device
        assert np.array_equal(f32_to_bf16_bits(hb), f32_to_bf16_bits(hd))
        untouched = ~np.isin(flat, eids)
        assert np.array_equal(hb[untouched], base[untouched])
        table.close()
    print(f"selftest ok: torch blocked={blocked}; fused int6 table via fork opener; {exp_hits} hits; "
          f"bf16-replacement == fp32-delta after device cast; untouched rows bit-identical; "
          f"bytes/row bf16={ov_b.nbytes_per_row()} delta={ov_d.nbytes_per_row()}")


if __name__ == "__main__":
    if "--block-torch" in sys.argv:
        sys.modules["torch"] = None
    _selftest()
