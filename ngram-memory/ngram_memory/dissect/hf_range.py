"""HTTP range reads of individual tensors / rows from a public safetensors checkpoint on the HF hub (§D).

Pattern of `P/t1/fetch_*.py` and `GF/dissect/range_probe.py`, with two changes that matter for 10^5 requests:

* the `resolve/main/<file>` redirect is followed ONCE per file and the signed CDN URL is reused;
* each worker thread keeps one persistent HTTPS connection (a CONNECT tunnel through the agent proxy when
  HTTPS_PROXY is set), so a 320-byte row costs one round trip instead of a TLS handshake
  (measured 0.10 s serial vs 0.20 s, `scratchpad/nm/m0/dissect/ka_probe.py`).

Multi-range requests are refused by the CDN (HTTP 416, `GF/dissect/range_probe.out`), so every row is one request.
Byte and request counters (`stats`) make the download budget auditable. Torch-free (numpy only).
"""
from __future__ import annotations

import concurrent.futures as cf
import http.client
import json
import os
import ssl
import struct
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_CAFILE = "/root/.ccr/ca-bundle.crt"
HF = "https://huggingface.co"


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """uint16 (bf16 bit pattern) -> float32, exact."""
    return (np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


@dataclass
class TensorInfo:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    start: int          # absolute byte offset of the tensor's data in the file
    nbytes: int

    @property
    def itemsize(self) -> int:
        return {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "F8_E4M3": 1}[self.dtype]


@dataclass
class RangeStats:
    requests: int = 0
    bytes: int = 0
    seconds: float = 0.0
    retries: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, n: int, dt: float, retries: int = 0) -> None:
        with self.lock:
            self.requests += 1
            self.bytes += n
            self.seconds += dt
            self.retries += retries

    def to_dict(self) -> dict:
        return dict(requests=self.requests, bytes=self.bytes, request_seconds=round(self.seconds, 1),
                    retries=self.retries)


class HFRangeReader:
    """Range reader for `https://huggingface.co/<repo>/resolve/<revision>/<file>`."""

    def __init__(self, repo: str, revision: str = "main", cafile: str | None = DEFAULT_CAFILE,
                 index_path: str | Path | None = None, timeout: float = 60.0, max_retries: int = 6):
        self.repo = repo
        self.revision = revision
        self.timeout = timeout
        self.max_retries = max_retries
        self.ctx = ssl.create_default_context(cafile=cafile if cafile and os.path.exists(cafile) else None)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler(),
                                                   urllib.request.HTTPSHandler(context=self.ctx))
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        self.proxy = urllib.parse.urlsplit(proxy) if proxy else None
        self._resolved: dict[str, tuple[str, str]] = {}   # file -> (host, path?query)
        self._headers: dict[str, tuple[int, dict]] = {}
        self._hl_cache: dict[str, int] = {}
        self._lock = threading.Lock()
        self._tl = threading.local()
        self.stats = RangeStats()
        self._index: dict[str, str] | None = None
        if index_path is not None:
            self._index = json.loads(Path(index_path).read_text())["weight_map"]

    # ------------------------------------------------------------------------------------------- plumbing
    def url(self, file: str) -> str:
        return f"{HF}/{self.repo}/resolve/{self.revision}/{file}"

    def _resolve(self, file: str) -> tuple[str, str]:
        with self._lock:
            if file in self._resolved:
                return self._resolved[file]
        r = self._opener.open(urllib.request.Request(self.url(file), headers={"Range": "bytes=0-7"}),
                              timeout=self.timeout)
        head = r.read()
        u = urllib.parse.urlsplit(r.geturl())
        res = (u.hostname or "", u.path + ("?" + u.query if u.query else ""))
        with self._lock:
            self._resolved[file] = res
            self._hl_cache[file] = struct.unpack("<Q", head)[0]
        return res

    def _conn(self, host: str) -> http.client.HTTPSConnection:
        conns = getattr(self._tl, "conns", None)
        if conns is None:
            conns = self._tl.conns = {}
        c = conns.get(host)
        if c is None:
            if self.proxy is not None:
                c = http.client.HTTPSConnection(self.proxy.hostname, self.proxy.port or 80, context=self.ctx,
                                                timeout=self.timeout)
                c.set_tunnel(host, 443)
            else:
                c = http.client.HTTPSConnection(host, 443, context=self.ctx, timeout=self.timeout)
            conns[host] = c
        return c

    def _drop_conn(self, host: str) -> None:
        conns = getattr(self._tl, "conns", {})
        c = conns.pop(host, None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass

    def read(self, file: str, a: int, b: int) -> bytes:
        """Bytes [a, b] inclusive of `file`."""
        host, path = self._resolve(file)
        want = b - a + 1
        err: Exception | None = None
        t0 = time.time()
        for attempt in range(self.max_retries):
            try:
                c = self._conn(host)
                c.request("GET", path, headers={"Range": f"bytes={a}-{b}"})
                resp = c.getresponse()
                data = resp.read()
                if resp.status == 206 and len(data) == want:
                    self.stats.add(len(data), time.time() - t0, attempt)
                    return data
                if resp.status in (403, 410):  # signed URL expired: re-resolve once
                    with self._lock:
                        self._resolved.pop(file, None)
                    host, path = self._resolve(file)
                err = IOError(f"HTTP {resp.status}, {len(data)} of {want} bytes")
                if resp.status == 429:
                    time.sleep(min(30.0, 2.0 ** attempt))
            except Exception as e:  # connection reset, timeout, ...
                err = e
                self._drop_conn(host)
            time.sleep(0.2 * (attempt + 1))
        raise IOError(f"range read failed {file} [{a},{b}]: {err}")

    # ------------------------------------------------------------------------------------------- metadata
    def header(self, file: str) -> tuple[int, dict]:
        """(header_len, header_json) of a safetensors file."""
        with self._lock:
            if file in self._headers:
                return self._headers[file]
        self._resolve(file)
        hl = self._hl_cache[file]
        hdr = json.loads(self.read(file, 8, 8 + hl - 1))
        with self._lock:
            self._headers[file] = (hl, hdr)
        return hl, hdr

    def index(self) -> dict[str, str]:
        if self._index is None:
            r = self._opener.open(self.url("model.safetensors.index.json"), timeout=self.timeout)
            self._index = json.loads(r.read())["weight_map"]
        return self._index

    def tensor(self, name: str, file: str | None = None) -> TensorInfo:
        file = file or self.index()[name]
        hl, hdr = self.header(file)
        v = hdr[name]
        a, b = v["data_offsets"]
        return TensorInfo(name, file, v["dtype"], tuple(v["shape"]), 8 + hl + a, b - a)

    # ----------------------------------------------------------------------------------------------- reads
    def read_tensor_bytes(self, info: TensorInfo, chunk: int = 32 << 20, threads: int = 4,
                          out_path: str | Path | None = None) -> bytes | Path:
        """Whole tensor, fetched in parallel chunks. With out_path, streamed to that file (returns the path)."""
        spans = [(o, min(o + chunk, info.nbytes)) for o in range(0, info.nbytes, chunk)]
        if out_path is None:
            buf = bytearray(info.nbytes)

            def one(s):
                buf[s[0]:s[1]] = self.read(info.file, info.start + s[0], info.start + s[1] - 1)
            with cf.ThreadPoolExecutor(threads) as ex:
                list(ex.map(one, spans))
            return bytes(buf)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.truncate(info.nbytes)
        fd = os.open(out_path, os.O_WRONLY)
        try:
            def one(s):
                os.pwrite(fd, self.read(info.file, info.start + s[0], info.start + s[1] - 1), s[0])
            with cf.ThreadPoolExecutor(threads) as ex:
                list(ex.map(one, spans))
        finally:
            os.close(fd)
        return out_path

    def read_tensor(self, info: TensorInfo, **kw) -> np.ndarray:
        """Whole tensor as float32 (BF16/F32) or its native dtype."""
        raw = self.read_tensor_bytes(info, **kw)
        return decode(raw, info)

    def read_rows(self, info: TensorInfo, rows: Sequence[int], threads: int = 48) -> np.ndarray:
        """Rows (first-axis slices) of a 2-D tensor; one request per row. Returns raw bytes as uint8 [n, row_bytes]."""
        row_bytes = int(np.prod(info.shape[1:])) * info.itemsize
        rows = [int(r) for r in rows]
        out = np.empty((len(rows), row_bytes), dtype=np.uint8)

        def one(i):
            a = info.start + rows[i] * row_bytes
            out[i] = np.frombuffer(self.read(info.file, a, a + row_bytes - 1), dtype=np.uint8)
        with cf.ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, range(len(rows))))
        return out

    def read_blocks(self, info: TensorInfo, starts: Sequence[int], count: int, threads: int = 8) -> np.ndarray:
        """Contiguous blocks of `count` rows from each start: uint8 [len(starts), count, row_bytes]."""
        row_bytes = int(np.prod(info.shape[1:])) * info.itemsize
        out = np.empty((len(starts), count, row_bytes), dtype=np.uint8)

        def one(i):
            a = info.start + int(starts[i]) * row_bytes
            out[i] = np.frombuffer(self.read(info.file, a, a + count * row_bytes - 1), dtype=np.uint8).reshape(
                count, row_bytes)
        with cf.ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, range(len(starts))))
        return out


def decode(raw: bytes | np.ndarray, info: TensorInfo) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) else raw
    if info.dtype == "BF16":
        return bf16_to_f32(a.view(np.uint16)).reshape(info.shape)
    if info.dtype == "F32":
        return a.view(np.float32).reshape(info.shape)
    dt = {"F16": np.float16, "I64": np.int64, "I32": np.int32, "U8": np.uint8}[info.dtype]
    return a.view(dt).reshape(info.shape)


# ------------------------------------------------------------------------------------------ Flash-Next rows
FLASH_NEXT_REPO = "Qwen/Qwen3.8-Flash-Next"
FLASH_NEXT_PLE_PREFIX = "model.language_model.layers.1.ple."


class FlashNextRowFetcher:
    """Global n-gram table row ids -> bf16 rows (as uint16 [n, 160]) via range reads of the shard tensors.

    Rows already fetched are cached in memory and, with `cache_path`, in an .npz (row_ids int64, rows uint16) so a
    re-run does not re-download. Shard addressing follows TableSpec.shard_of (§2.1b)."""

    def __init__(self, reader: HFRangeReader, cache_path: str | Path | None = None, threads: int = 48):
        from ..spec import flash_next_format
        self.reader = reader
        self.fmt = flash_next_format()
        self.key, self.table = self.fmt.key, self.fmt.table
        self.width = self.table.row_width
        self.threads = threads
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[int, np.ndarray] = {}
        if self.cache_path is not None and self.cache_path.exists():
            z = np.load(self.cache_path)
            for r, v in zip(z["row_ids"], z["rows"]):
                self.cache[int(r)] = v

    def shard_info(self, shard: int) -> TensorInfo:
        return self.reader.tensor(self.table.shard_tensor_name(shard))

    def fetch(self, row_ids: Iterable[int], log_every: int = 0) -> int:
        """Fetch every row not yet cached; returns the number fetched now."""
        need = sorted({int(r) for r in row_ids} - set(self.cache))
        if not need:
            return 0
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for r in need:
            s, l = self.table.shard_of(self.key, r)
            by_shard.setdefault(s, []).append((r, l))
        infos = {s: self.shard_info(s) for s in sorted(by_shard)}   # headers (one per file) first
        jobs = [(r, infos[s], l) for s, lst in by_shard.items() for r, l in lst]
        rb = self.width * 2
        done = [0]
        lock = threading.Lock()
        t0 = time.time()

        def one(j):
            r, info, l = j
            a = info.start + l * rb
            v = np.frombuffer(self.reader.read(info.file, a, a + rb - 1), dtype=np.uint16).copy()
            with lock:
                self.cache[r] = v
                done[0] += 1
                if log_every and done[0] % log_every == 0:
                    dt = time.time() - t0
                    print(f"  rows {done[0]}/{len(jobs)} {dt:.0f}s ({done[0] / dt:.0f}/s)", flush=True)
        with cf.ThreadPoolExecutor(self.threads) as ex:
            list(ex.map(one, jobs))
        return len(jobs)

    def get(self, row_ids) -> np.ndarray:
        """float32 [..., 160] for cached rows (KeyError if one is missing)."""
        a = np.asarray(row_ids, dtype=np.int64)
        flat = a.reshape(-1)
        out = np.stack([self.cache[int(r)] for r in flat]) if flat.size else np.zeros((0, self.width), np.uint16)
        return bf16_to_f32(out).reshape(*a.shape, self.width)

    def save(self) -> None:
        if self.cache_path is None:
            return
        ids = np.fromiter(self.cache.keys(), dtype=np.int64, count=len(self.cache))
        order = np.argsort(ids)
        rows = np.stack([self.cache[int(i)] for i in ids[order]]) if len(ids) else np.zeros((0, self.width),
                                                                                              np.uint16)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache_path, row_ids=ids[order], rows=rows)


__all__ = ["HFRangeReader", "TensorInfo", "RangeStats", "FlashNextRowFetcher", "bf16_to_f32", "decode",
           "FLASH_NEXT_REPO", "FLASH_NEXT_PLE_PREFIX"]
