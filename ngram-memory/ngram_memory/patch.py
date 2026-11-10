"""`ngram-edit/v1` patches (DESIGN.md §3.4): a JSON manifest plus a safetensors payload.

Payload tensors:
  row_ids        int64 [R]      sorted, unique, in [0, padded_rows)
  values         fp32  [R, W]   'delta' mode (native: added to the decoded base row) or
                                'absolute' mode (grafts: base rows are 0, so value == delta from 0)
  reset_row_ids  int64 [Z]      optional; rows set back to base BEFORE this patch's values are added
                                (Stage-2 increments: dead rows deleted, dirty rows reset and refit, §I.6)

Served row = engine_cast(codec_decode(base[r]) + delta[r])  -- the overlay rule of §3.3; edits are never
re-quantised into a quantised base table.

The manifest is stored twice: as JSON in the safetensors `__metadata__["ngram_edit_manifest"]` and, by default,
as a `<file>.json` sidecar. Readers check that both agree and that the payload hash matches.

numpy only (no torch): the safetensors reader/writer here is a small implementation of the published format
(8-byte LE header length, JSON header, raw little-endian data), checked against the `safetensors` package in tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .spec import FormatSpec, KeySpec, ModuleSpec, TableSpec

PATCH_FORMAT = "ngram-edit/v1"
MANIFEST_KEY = "ngram_edit_manifest"
MODES = ("delta", "absolute")

# ------------------------------------------------------------------------------------------- safetensors I/O
#: safetensors dtype -> (numpy storage dtype, element bytes). BF16 is carried as its uint16 bit pattern.
ST_DTYPES: dict[str, tuple[str, int]] = {
    "I64": ("<i8", 8), "I32": ("<i4", 4), "U8": ("u1", 1), "I8": ("i1", 1), "U16": ("<u2", 2),
    "F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2), "F64": ("<f8", 8),
}
_NP_TO_ST = {np.dtype("<i8"): "I64", np.dtype("<i4"): "I32", np.dtype("u1"): "U8", np.dtype("i1"): "I8",
             np.dtype("<f4"): "F32", np.dtype("<f2"): "F16", np.dtype("<f8"): "F64", np.dtype("<u2"): "U16"}


class BF16Bits:
    """Marks a uint16 array as bf16 bit patterns for `write_safetensors` (dtype BF16 on disk)."""

    def __init__(self, bits: np.ndarray):
        bits = np.ascontiguousarray(bits)
        if bits.dtype != np.uint16:
            raise TypeError("BF16Bits needs a uint16 array of bf16 bit patterns")
        self.bits = bits


def write_safetensors(path: str, tensors: Mapping[str, np.ndarray | BF16Bits],
                      metadata: Mapping[str, str] | None = None) -> None:
    """Write numpy arrays as a safetensors file (atomic: temp file + rename)."""
    hdr: dict[str, Any] = {}
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs, off = [], 0
    for name in sorted(tensors):
        t = tensors[name]
        if isinstance(t, BF16Bits):
            arr, dt = t.bits, "BF16"
        else:
            arr = np.ascontiguousarray(t)
            if arr.dtype.byteorder == ">":
                arr = arr.astype(arr.dtype.newbyteorder("<"))
            dt = _NP_TO_ST.get(arr.dtype)
            if dt is None:
                raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
        b = arr.tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)  # align the data section to 8 bytes, as the reference writer does
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


def read_safetensors(path: str, names: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """(tensors, metadata). BF16 tensors come back as uint16 bit patterns (use bf16_bits_to_f32)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        start = 8 + n
        meta = hdr.pop("__metadata__", {}) or {}
        out: dict[str, np.ndarray] = {}
        for name, t in hdr.items():
            if names is not None and name not in names:
                continue
            npd, nb = ST_DTYPES[t["dtype"]]
            a, b = t["data_offsets"]
            f.seek(start + a)
            buf = f.read(b - a)
            if len(buf) != b - a:
                raise IOError(f"{path}: short read of {name}")
            out[name] = np.frombuffer(buf, dtype=npd).reshape(t["shape"]).copy()
    return out, meta


def read_safetensors_dtypes(path: str) -> dict[str, str]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return {k: v["dtype"] for k, v in hdr.items()}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_sha256(row_ids: np.ndarray, values: np.ndarray, reset_row_ids: np.ndarray | None = None) -> str:
    """Hash of the payload content (independent of file layout): ids int64 LE, values fp32 LE, resets."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(row_ids, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
    if reset_row_ids is not None and len(reset_row_ids):
        h.update(b"reset")
        h.update(np.ascontiguousarray(reset_row_ids, dtype="<i8").tobytes())
    return h.hexdigest()


def rows_fingerprint(row_ids: np.ndarray, rows_f32: np.ndarray) -> str:
    """Fingerprint of dequantised base rows at the touched ids (fp32 LE bytes after the ids)."""
    h = hashlib.sha256(b"base-rows/v1")
    h.update(np.ascontiguousarray(row_ids, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(rows_f32, dtype="<f4").tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- manifest
@dataclass
class PatchManifest:
    """The `ngram-edit/v1` manifest (§3.4). `cartridge` holds the §3.5 fields when the patch is a cartridge,
    an increment or a fact patch inside a namespace (see ngram_memory.cartridge.CartridgeInfo)."""

    mode: str
    key_spec: dict
    key_fingerprint: str
    table_spec: dict
    table_fingerprint: str
    row_width: int
    padded_rows: int
    rows_per_shard: int
    n_rows: int = 0
    n_reset: int = 0
    payload_sha256: str = ""
    module_spec: dict | None = None
    module_fingerprint: str | None = None
    format_fingerprint: str | None = None
    model_repo: str | None = None
    model_commit: str | None = None
    base_fingerprint: str | None = None
    edits: list[dict] = field(default_factory=list)
    cartridge: dict | None = None
    created: float = field(default_factory=time.time)
    format: str = PATCH_FORMAT

    def __post_init__(self) -> None:
        if self.format != PATCH_FORMAT:
            raise ValueError(f"not an {PATCH_FORMAT} manifest: {self.format!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @classmethod
    def for_format(cls, fmt: FormatSpec, mode: str | None = None, **kw) -> "PatchManifest":
        """Manifest skeleton for a FormatSpec; mode defaults to 'absolute' for grafts (base_codec 'none') and
        'delta' otherwise."""
        if mode is None:
            mode = "absolute" if fmt.table.base_codec == "none" else "delta"
        if mode == "absolute" and fmt.table.base_codec != "none":
            raise ValueError("'absolute' patches are only defined on tables without a base (grafts)")
        return cls(mode=mode, key_spec=fmt.key.to_dict(), key_fingerprint=fmt.key.fingerprint(),
                   table_spec=fmt.table.to_dict(), table_fingerprint=fmt.table.fingerprint(),
                   row_width=fmt.row_width, padded_rows=fmt.padded_rows,
                   rows_per_shard=fmt.table.rows_per_shard(fmt.key), module_spec=fmt.module.to_dict(),
                   module_fingerprint=fmt.module.fingerprint(), format_fingerprint=fmt.fingerprint(), **kw)

    def key(self) -> KeySpec:
        k = KeySpec.from_dict(self.key_spec)
        if k.fingerprint() != self.key_fingerprint:
            raise ValueError("manifest key_spec does not match its key_fingerprint")
        return k

    def table(self) -> TableSpec:
        t = TableSpec.from_dict(self.table_spec)
        if t.fingerprint() != self.table_fingerprint:
            raise ValueError("manifest table_spec does not match its table_fingerprint")
        return t

    def module(self) -> ModuleSpec | None:
        if self.module_spec is None:
            return None
        m = ModuleSpec.from_dict(self.module_spec)
        if m.fingerprint() != self.module_fingerprint:
            raise ValueError("manifest module_spec does not match its module_fingerprint")
        return m

    def format_spec(self) -> FormatSpec:
        m = self.module()
        if m is None:
            raise ValueError("manifest has no module_spec")
        f = FormatSpec(self.key(), self.table(), m)
        if self.format_fingerprint is not None and f.fingerprint() != self.format_fingerprint:
            raise ValueError("manifest specs do not match its format_fingerprint")
        return f

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "PatchManifest":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ------------------------------------------------------------------------------------------------ patch
def rows2d(x, n: int, dtype=np.float32, width: int | None = None) -> np.ndarray:
    """x as a C-contiguous [n, w] array; works for n == 0 (reshape(0, -1) is ambiguous in numpy)."""
    x = np.ascontiguousarray(x, dtype=dtype)
    if width is None:
        width = (x.shape[-1] if x.ndim >= 2 else 0) if x.size == 0 else x.size // max(n, 1)
    return x.reshape(n, width)


@dataclass
class Patch:
    manifest: PatchManifest
    row_ids: np.ndarray  # int64 [R]
    values: np.ndarray  # fp32 [R, W]
    reset_row_ids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))

    def __post_init__(self) -> None:
        self.row_ids = np.ascontiguousarray(self.row_ids, dtype=np.int64).reshape(-1)
        self.values = rows2d(self.values, self.row_ids.size, np.float32)
        self.reset_row_ids = np.ascontiguousarray(self.reset_row_ids, dtype=np.int64).reshape(-1)

    def check(self) -> None:
        """Structural checks (no spec needed)."""
        m = self.manifest
        _check_ids(self.row_ids, m.padded_rows, "row_ids")
        _check_ids(self.reset_row_ids, m.padded_rows, "reset_row_ids")
        if self.values.shape != (self.row_ids.size, m.row_width):
            raise ValueError(f"values {self.values.shape} != ({self.row_ids.size}, {m.row_width})")
        if not np.isfinite(self.values).all():
            raise ValueError("non-finite values in patch")
        if m.n_rows != self.row_ids.size or m.n_reset != self.reset_row_ids.size:
            raise ValueError("manifest row counts disagree with the payload")
        if m.payload_sha256 != payload_sha256(self.row_ids, self.values, self.reset_row_ids):
            raise ValueError("payload hash mismatch")

    def finalize(self) -> "Patch":
        """Fill the manifest's payload fields from the arrays (call before writing)."""
        m = self.manifest
        order = np.argsort(self.row_ids, kind="stable")
        self.row_ids, self.values = self.row_ids[order], self.values[order]
        self.reset_row_ids = np.unique(self.reset_row_ids)
        m.n_rows, m.n_reset = int(self.row_ids.size), int(self.reset_row_ids.size)
        m.payload_sha256 = payload_sha256(self.row_ids, self.values, self.reset_row_ids)
        self.check()
        return self


def _check_ids(ids: np.ndarray, n: int, what: str) -> None:
    if ids.size == 0:
        return
    if (np.diff(ids) <= 0).any():
        raise ValueError(f"{what} must be sorted and unique")
    if ids[0] < 0 or ids[-1] >= n:
        raise IndexError(f"{what} outside [0, {n})")


def make_patch(fmt: FormatSpec, row_ids, values, *, mode: str | None = None, reset_row_ids=None,
               base_rows: np.ndarray | None = None, **manifest_kw) -> Patch:
    """Build a finalised patch. `base_rows` (fp32 decoded base rows at the sorted row_ids) sets base_fingerprint."""
    man = PatchManifest.for_format(fmt, mode=mode, **manifest_kw)
    ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    vals = np.asarray(values, dtype=np.float32).reshape(ids.size, fmt.row_width)
    if np.unique(ids).size != ids.size:
        raise ValueError("row_ids must be unique (sum duplicates before making a patch)")
    p = Patch(man, ids, vals, np.empty(0, np.int64) if reset_row_ids is None else reset_row_ids).finalize()
    if base_rows is not None:
        if man.mode != "delta":
            raise ValueError("base_fingerprint only applies to 'delta' patches")
        order = np.argsort(ids, kind="stable")
        p.manifest.base_fingerprint = rows_fingerprint(p.row_ids, np.asarray(base_rows, np.float32)[order])
    return p


def write_patch(path: str, patch: Patch, sidecar: bool = True) -> str:
    """Write `<path>` (safetensors) and, by default, `<path>.json`. Returns the payload file sha256."""
    patch.check()
    tensors: dict[str, np.ndarray] = {"row_ids": patch.row_ids, "values": patch.values}
    if patch.reset_row_ids.size:
        tensors["reset_row_ids"] = patch.reset_row_ids
    js = patch.manifest.to_json(indent=None)
    write_safetensors(path, tensors, {MANIFEST_KEY: js})
    if sidecar:
        with open(path + ".json", "w") as f:
            f.write(patch.manifest.to_json())
    return sha256_file(path)


def read_patch(path: str, check_sidecar: bool = True) -> Patch:
    t, meta = read_safetensors(path)
    if MANIFEST_KEY not in meta:
        raise ValueError(f"{path}: no {MANIFEST_KEY} in safetensors metadata")
    man = PatchManifest.from_dict(json.loads(meta[MANIFEST_KEY]))
    if check_sidecar and os.path.exists(path + ".json"):
        with open(path + ".json") as f:
            side = PatchManifest.from_dict(json.load(f))
        if side.to_dict() != man.to_dict():
            raise ValueError(f"{path}: sidecar manifest differs from the embedded one")
    p = Patch(man, t["row_ids"], t["values"], t.get("reset_row_ids", np.empty(0, np.int64)))
    p.check()
    return p


def verify_patch(patch: Patch, fmt: FormatSpec, base_rows_fn=None) -> list[str]:
    """Refuse on any mismatch with `fmt` (raises ValueError). `base_rows_fn(ids) -> fp32 [n, W]` (decoded base)
    re-derives base_fingerprint for 'delta' patches that carry one. Returns the list of checks performed."""
    patch.check()
    m, done = patch.manifest, ["structure", "payload_sha256"]
    if m.key_fingerprint != fmt.key.fingerprint():
        raise ValueError("KeySpec fingerprint mismatch")
    if m.table_fingerprint != fmt.table.fingerprint():
        raise ValueError("TableSpec fingerprint mismatch")
    done += ["key_fingerprint", "table_fingerprint"]
    # a row's meaning depends on the module (W_v seed/scale, arm, gate/conv), so the ModuleSpec is checked always:
    # a manifest without module_fingerprint is refused (VF-7); a module_spec dict, if present, must agree too
    if m.module_fingerprint is None:
        raise ValueError("patch manifest has no module_fingerprint: cannot check the ModuleSpec")
    if m.module_fingerprint != fmt.module.fingerprint():
        raise ValueError("ModuleSpec fingerprint mismatch")
    done.append("module_fingerprint")
    ms = getattr(m, "module_spec", None)
    if ms is not None:
        from .spec import ModuleSpec
        if ModuleSpec.from_dict(ms).fingerprint() != fmt.module.fingerprint():
            raise ValueError("manifest module_spec does not match the FormatSpec's ModuleSpec")
    if m.format_fingerprint is not None and m.format_fingerprint != fmt.fingerprint():
        raise ValueError("FormatSpec fingerprint mismatch")
    if (m.row_width, m.padded_rows) != (fmt.row_width, fmt.padded_rows):
        raise ValueError("row width / padded rows mismatch")
    if (m.mode == "absolute") != (fmt.table.base_codec == "none"):
        raise ValueError(f"mode {m.mode!r} does not fit base codec {fmt.table.base_codec!r}")
    done.append("mode")
    if m.base_fingerprint is not None and base_rows_fn is not None:
        got = rows_fingerprint(patch.row_ids, np.asarray(base_rows_fn(patch.row_ids), np.float32))
        if got != m.base_fingerprint:
            raise ValueError("base rows changed since the patch was made (base_fingerprint mismatch)")
        done.append("base_fingerprint")
    return done


__all__ = ["PATCH_FORMAT", "PatchManifest", "Patch", "make_patch", "write_patch", "read_patch", "verify_patch",
           "write_safetensors", "read_safetensors", "read_safetensors_dtypes", "BF16Bits", "payload_sha256",
           "rows_fingerprint", "sha256_file"]
