"""Project cartridges (DESIGN.md §3.5): namespaced, versioned `ngram-edit/v1` patches, the graft file they are
trained against, layering/compilation to served rows, and the per-file reader index sidecar. numpy only.

Artifacts
---------
* graft file  `graft-<host>.safetensors`: value_proj bf16 [d, E] (+ G-dir conv1d fp32) with a manifest holding the
  KeySpec / TableSpec / ModuleSpec fingerprints, the host config sha256 and the W_v sha256.
* cartridge patch (training side, full precision): an ngram-edit/v1 patch whose manifest carries a `cartridge`
  section (namespace, version, parent_version, kind, source, row_policy, train_config, prune_floor, arm, graft,
  reader_index). Increments are child patches (`parent_version`), optionally with `reset_row_ids`.
* served cartridge `cartridge-<name>-v<k>.safetensors`: `row_ids` int64 sorted [U] + `rows` bf16 [U, w], the
  compiled `cast(decode(base) + delta_cart + sum increments + sum facts)` (one fp32 sum, one cast), plus the
  manifest of the graft it was trained against. Loaders refuse on any fingerprint mismatch.

Layering (fixed order, fp32 sum, one cast): chain order = cartridge, then increments by version, then facts.
`reset_row_ids` of a patch set those rows back to base before its values are added, so deleting a row returns it
exactly to base.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import codecs
from .patch import (BF16Bits, Patch, PatchManifest, make_patch, read_patch, read_safetensors, sha256_file,
                    verify_patch, write_patch, write_safetensors)
from .spec import FormatSpec

GRAFT_FILE_FORMAT = "ngram-graft/v1"
SERVED_FORMAT = "ngram-cartridge-served/v1"
READER_INDEX_FORMAT = "ngram-reader-index/v1"
KINDS = ("cartridge", "increment", "fact")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return sha256_bytes(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())


def config_sha256(cfg: Mapping) -> str:
    """sha256 of a canonical JSON of a host config dict."""
    return sha256_bytes(json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode())


# ------------------------------------------------------------------------------------------- graft file
@dataclass
class GraftManifest:
    host: str
    format_spec: dict  # FormatSpec.to_dict()
    key_fingerprint: str
    table_fingerprint: str
    module_fingerprint: str
    format_fingerprint: str
    value_proj_sha256: str  # over the bf16 bits as stored
    host_config_sha256: str | None = None
    extra_sha256: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    format: str = GRAFT_FILE_FORMAT

    def fingerprint(self) -> str:
        """What a cartridge pins: everything that changes the module's output for given rows."""
        return sha256_bytes(json.dumps(dict(format=self.format_fingerprint, wv=self.value_proj_sha256,
                                            host=self.host_config_sha256, extra=self.extra_sha256),
                                       sort_keys=True).encode())

    def format_spec_obj(self) -> FormatSpec:
        return FormatSpec.from_dict(self.format_spec)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["fingerprint"] = self.fingerprint()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GraftManifest":
        m = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if m.format != GRAFT_FILE_FORMAT:
            raise ValueError(f"not a {GRAFT_FILE_FORMAT} manifest")
        if "fingerprint" in d and d["fingerprint"] != m.fingerprint():
            raise ValueError("graft manifest fingerprint mismatch")
        return m


def write_graft_file(path: str, fmt: FormatSpec, value_proj: np.ndarray, *, host: str,
                     host_config: Mapping | None = None, extra: Mapping[str, np.ndarray] | None = None
                     ) -> GraftManifest:
    """`value_proj` fp32 [d, E] (stored bf16). For G-lin the 0.5 gate is NOT folded in (the serve's linear mode
    may fold it: 0.5 is a power of two, so folding is bit-exact, GF/serve-port/parity_hc1.out)."""
    if fmt.module.kind != "graft":
        raise ValueError("graft files hold graft ModuleSpecs")
    wv = np.asarray(value_proj, dtype=np.float32)
    if wv.shape != (fmt.module.hidden_size * fmt.module.hc_count, fmt.module.ple_embed_dim):
        raise ValueError(f"value_proj {wv.shape} does not fit the module spec")
    bits = codecs.bf16_bits(wv)
    tensors: dict = {"value_proj": BF16Bits(bits)}
    ex = {}
    for k, v in (extra or {}).items():
        tensors[k] = np.ascontiguousarray(v, dtype=np.float32)
        ex[k] = array_sha256(tensors[k])
    man = GraftManifest(host=host, format_spec=fmt.to_dict(), key_fingerprint=fmt.key.fingerprint(),
                        table_fingerprint=fmt.table.fingerprint(), module_fingerprint=fmt.module.fingerprint(),
                        format_fingerprint=fmt.fingerprint(), value_proj_sha256=array_sha256(bits),
                        host_config_sha256=None if host_config is None else config_sha256(host_config),
                        extra_sha256=ex)
    write_safetensors(path, tensors, {"ngram_graft_manifest": json.dumps(man.to_dict(), sort_keys=True)})
    return man


def read_graft_file(path: str) -> tuple[GraftManifest, dict[str, np.ndarray]]:
    """(manifest, tensors) with value_proj decoded to fp32. Refuses if the stored W_v hash mismatches."""
    t, meta = read_safetensors(path)
    man = GraftManifest.from_dict(json.loads(meta["ngram_graft_manifest"]))
    if array_sha256(t["value_proj"]) != man.value_proj_sha256:
        raise ValueError("value_proj sha256 mismatch")
    for k, h in man.extra_sha256.items():
        if array_sha256(t[k]) != h:
            raise ValueError(f"{k} sha256 mismatch")
    t["value_proj"] = codecs.bf16_bits_to_f32(t["value_proj"])
    man.format_spec_obj()  # re-validates the stored FormatSpec fingerprint
    return man, t


# -------------------------------------------------------------------------------------- cartridge info
@dataclass
class CartridgeInfo:
    """The §3.5 manifest fields of a cartridge / increment / fact patch."""

    namespace: str
    version: int
    kind: str = "cartridge"
    parent_version: int | None = None
    source: dict = field(default_factory=dict)  # repo, commit | commit_range, file_list_sha256, tokenizer_sha256, framings
    row_policy: str | None = None
    train_config: dict = field(default_factory=dict)
    prune_floor: float | None = None
    arm: str | None = None  # G-lin | G-dir (grafts)
    graft: dict | None = None  # GraftManifest.to_dict() subset: fingerprint + component fingerprints
    reader_index: dict | None = None  # {"path": ..., "sha256": ...}

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not self.namespace or not isinstance(self.version, int) or self.version < 0:
            raise ValueError("namespace must be non-empty and version a non-negative int")
        if self.kind == "cartridge" and self.parent_version is not None:
            raise ValueError("a root cartridge has no parent_version")
        if self.kind != "cartridge" and self.parent_version is None:
            raise ValueError(f"a {self.kind} patch needs parent_version")
        if self.parent_version is not None and self.parent_version >= self.version:
            raise ValueError("parent_version must precede version")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "CartridgeInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def graft_pin(gm: GraftManifest) -> dict:
    return dict(fingerprint=gm.fingerprint(), format_fingerprint=gm.format_fingerprint,
                value_proj_sha256=gm.value_proj_sha256, host_config_sha256=gm.host_config_sha256, host=gm.host)


def make_cartridge_patch(fmt: FormatSpec, row_ids, values, info: CartridgeInfo, *, reset_row_ids=None,
                         graft: GraftManifest | None = None, base_rows=None, **manifest_kw) -> Patch:
    if graft is not None:
        if graft.format_fingerprint != fmt.fingerprint():
            raise ValueError("graft file was made for a different FormatSpec")
        info.graft = graft_pin(graft)
    if fmt.module.kind == "graft" and info.arm is None:
        info.arm = fmt.module.arm
    return make_patch(fmt, row_ids, values, reset_row_ids=reset_row_ids, base_rows=base_rows,
                      cartridge=info.to_dict(), **manifest_kw)


def cartridge_info(p: Patch) -> CartridgeInfo:
    if p.manifest.cartridge is None:
        raise ValueError("patch has no cartridge section")
    return CartridgeInfo.from_dict(p.manifest.cartridge)


def check_against_graft(p: Patch, graft: GraftManifest) -> None:
    """Refuse a cartridge trained against another graft (format, W_v or host config)."""
    info = cartridge_info(p)
    if p.manifest.format_fingerprint != graft.format_fingerprint:
        raise ValueError("cartridge FormatSpec fingerprint != graft file's")
    if info.graft is None or info.graft.get("fingerprint") != graft.fingerprint():
        raise ValueError("cartridge was trained against a different graft file (W_v / host config)")


# --------------------------------------------------------------------------------------------- chains
class CartridgeChain:
    """A root cartridge plus child patches, validated: one namespace, one format, one graft pin, contiguous
    parent_version links, kinds cartridge -> increment/fact."""

    def __init__(self, patches: Sequence[Patch]):
        if not patches:
            raise ValueError("empty chain")
        self.patches = list(patches)
        infos = [cartridge_info(p) for p in self.patches]
        root = infos[0]
        if root.kind != "cartridge":
            raise ValueError("a chain starts with a kind='cartridge' patch")
        fp = self.patches[0].manifest.format_fingerprint
        for prev, (p, inf) in zip(infos, zip(self.patches[1:], infos[1:])):
            if inf.kind == "cartridge":
                raise ValueError("only the first patch of a chain may be a cartridge")
            if inf.namespace != root.namespace:
                raise ValueError(f"namespace {inf.namespace!r} != {root.namespace!r}")
            if inf.parent_version != prev.version:
                raise ValueError(f"v{inf.version}: parent_version {inf.parent_version} != previous v{prev.version}")
            if p.manifest.format_fingerprint != fp:
                raise ValueError(f"v{inf.version}: format fingerprint differs from the root's")
            if (inf.graft or {}).get("fingerprint") != (root.graft or {}).get("fingerprint"):
                raise ValueError(f"v{inf.version}: graft pin differs from the root's")
            if p.manifest.mode != self.patches[0].manifest.mode:
                raise ValueError("mixed modes in a chain")
        self.infos = infos

    @property
    def namespace(self) -> str:
        return self.infos[0].namespace

    @property
    def version(self) -> int:
        return self.infos[-1].version

    @property
    def manifest(self) -> PatchManifest:
        return self.patches[0].manifest

    def upto(self, version: int) -> "CartridgeChain":
        """Rollback: the chain truncated at `version` (drops later patches)."""
        k = [i.version for i in self.infos].index(version)
        return CartridgeChain(self.patches[:k + 1])

    def compose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(ids, fp32 delta sum, reset_mask): fixed chain order; a patch's reset rows go back to base (delta 0,
        and the row leaves the overlay unless a later patch writes it). reset_mask marks rows present only because
        they were reset (they serve base)."""
        return compose_patches(self.patches)


def compose_patches(patches: Sequence[Patch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = patches[0].manifest.row_width
    all_ids = np.unique(np.concatenate([np.concatenate([p.row_ids, p.reset_row_ids]) for p in patches]))
    acc = np.zeros((all_ids.size, w), np.float32)
    written = np.zeros(all_ids.size, bool)
    for p in patches:
        if p.reset_row_ids.size:
            r = np.searchsorted(all_ids, p.reset_row_ids)
            acc[r] = 0.0
            written[r] = False
        if p.row_ids.size:
            q = np.searchsorted(all_ids, p.row_ids)
            acc[q] = acc[q] + p.values
            written[q] = True
    return all_ids[written], acc[written], all_ids[~written]


# ------------------------------------------------------------------------------------------ compile
def compile_served(ids: np.ndarray, delta: np.ndarray, base_rows_fn=None, codec: str = "bf16"
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Served rows for an overlay: codec(decode(base) + delta), ONE fp32 sum and ONE cast.

    codec 'bf16' -> uint16 bf16 bits [U, w] (the served cartridge / replacement-row overlay);
    'int6_g32'   -> fused uint8 rows [U, 130 at w=160] (the lower-RAM raw-row overlay; lossy, see roundtrip_stats);
    'fp32'       -> fp32 [U, w]."""
    ids = np.asarray(ids, dtype=np.int64)
    acc = np.asarray(delta, dtype=np.float32)
    if base_rows_fn is not None:
        acc = np.asarray(base_rows_fn(ids), dtype=np.float32) + acc
    if codec == "bf16":
        return ids, codecs.bf16_bits(acc)
    if codec == "int6_g32":
        return ids, codecs.int6_encode_fused(acc)
    if codec == "fp32":
        return ids, acc.copy()
    raise ValueError(f"unknown served codec {codec!r}")


@dataclass
class ServedCartridge:
    namespace: str
    version: int
    row_ids: np.ndarray  # int64 [U] sorted
    rows_bits: np.ndarray  # uint16 bf16 bits [U, w]
    manifest: dict  # chain manifest: format/key/table/module fingerprints, graft pin, versions, payload hashes

    def rows_f32(self) -> np.ndarray:
        return codecs.bf16_bits_to_f32(self.rows_bits)


def compile_chain(chain: CartridgeChain, base_rows_fn=None) -> ServedCartridge:
    ids, d, _ = chain.compose()
    ids, bits = compile_served(ids, d, base_rows_fn, "bf16")
    m = chain.manifest
    man = dict(format=SERVED_FORMAT, namespace=chain.namespace, version=chain.version,
               versions=[i.version for i in chain.infos], kinds=[i.kind for i in chain.infos],
               payload_sha256=[p.manifest.payload_sha256 for p in chain.patches], mode=m.mode,
               key_fingerprint=m.key_fingerprint, table_fingerprint=m.table_fingerprint,
               module_fingerprint=m.module_fingerprint, format_fingerprint=m.format_fingerprint,
               row_width=m.row_width, padded_rows=m.padded_rows, graft=chain.infos[0].graft,
               base_fingerprint=None if base_rows_fn is None else
               _base_fp(ids, base_rows_fn), rows_sha256=array_sha256(bits), n_rows=int(ids.size))
    return ServedCartridge(chain.namespace, chain.version, ids, bits, man)


def _base_fp(ids, fn) -> str:
    from .patch import rows_fingerprint
    return rows_fingerprint(ids, np.asarray(fn(ids), np.float32))


def write_served(path: str, sc: ServedCartridge) -> str:
    write_safetensors(path, {"row_ids": sc.row_ids, "rows": BF16Bits(sc.rows_bits)},
                      {"ngram_cartridge_manifest": json.dumps(sc.manifest, sort_keys=True)})
    return sha256_file(path)


def read_served(path: str, graft: GraftManifest | None = None, fmt: FormatSpec | None = None) -> ServedCartridge:
    """Load a served cartridge; refuses on a rows hash, graft pin or format fingerprint mismatch."""
    t, meta = read_safetensors(path)
    man = json.loads(meta["ngram_cartridge_manifest"])
    if man.get("format") != SERVED_FORMAT:
        raise ValueError("not a served cartridge")
    ids, bits = t["row_ids"], t["rows"]
    if array_sha256(bits) != man["rows_sha256"]:
        raise ValueError("served rows sha256 mismatch")
    if ids.size and ((np.diff(ids) <= 0).any() or ids[0] < 0 or ids[-1] >= man["padded_rows"]):
        raise ValueError("served row ids must be sorted, unique and in range")
    if fmt is not None and man["format_fingerprint"] != fmt.fingerprint():
        raise ValueError("served cartridge format fingerprint != the given FormatSpec")
    if graft is not None:
        if man["format_fingerprint"] != graft.format_fingerprint:
            raise ValueError("served cartridge format fingerprint != graft file's")
        if (man.get("graft") or {}).get("fingerprint") != graft.fingerprint():
            raise ValueError("served cartridge was compiled for a different graft file")
    return ServedCartridge(man["namespace"], int(man["version"]), ids, bits, man)


# --------------------------------------------------------------------------------- numpy overlay (serve)
class CompiledOverlayNP:
    """Torch-free row overlay, as the serve would hold it after `decode_raw` (port of
    P/serve/v2-editpath/ple_overlay_np.py). Sorted (ids, payload), published by one attribute assignment; epoch
    bumps per publish. mode 'bf16': payload = bf16 replacement bits, served = row; mode 'delta': fp32 deltas,
    served = bf16(base + delta) after the device cast. Both are bit-identical on the device when
    row = bf16(base + delta)."""

    def __init__(self, row_width: int, n_rows: int, mode: str = "bf16"):
        if mode not in ("bf16", "delta"):
            raise ValueError(mode)
        self.row_width, self.n_rows, self.mode = int(row_width), int(n_rows), mode
        self.epoch = 0
        self._compiled: tuple[np.ndarray, np.ndarray] = (np.empty(0, np.int64), self._empty(0))

    def _empty(self, k: int) -> np.ndarray:
        return np.empty((k, self.row_width), np.uint16 if self.mode == "bf16" else np.float32)

    def publish(self, ids: np.ndarray, payload: np.ndarray) -> int:
        o = np.argsort(ids, kind="stable")
        ids = np.ascontiguousarray(np.asarray(ids)[o], np.int64)
        payload = np.ascontiguousarray(np.asarray(payload)[o])
        if ids.size and (ids[0] < 0 or ids[-1] >= self.n_rows or (np.diff(ids) == 0).any()):
            raise ValueError("overlay ids must be unique and in range")
        want = np.uint16 if self.mode == "bf16" else np.float32
        if payload.dtype != want or payload.shape != (ids.size, self.row_width):
            raise ValueError(f"payload must be {want.__name__} [{ids.size}, {self.row_width}]")
        self._compiled = (ids, payload)
        self.epoch += 1
        return self.epoch

    def apply(self, flat_ids: np.ndarray, host_rows: np.ndarray) -> int:
        """In place on fp32 host rows [n, w] read for `flat_ids` [n]. Returns the number of hits."""
        u, pay = self._compiled
        if u.size == 0:
            return 0
        flat_ids = np.asarray(flat_ids, dtype=np.int64)
        pos = np.searchsorted(u, flat_ids)
        pos[pos == u.size] = 0
        hit = u[pos] == flat_ids
        if not hit.any():
            return 0
        sel = np.flatnonzero(hit)
        if self.mode == "bf16":
            host_rows[sel] = codecs.bf16_bits_to_f32(pay[pos[sel]])
        else:
            host_rows[sel] += pay[pos[sel]]
        return int(sel.size)


# ------------------------------------------------------------------------------------------ reader index
@dataclass
class ReaderIndex:
    """Per-file sparse row counts: which rows each file of the corpus reads, and how often (§3.5, §I.6).

    Stored as a sidecar safetensors: `file_offsets` int64 [F+1], `row_ids` int64 [N], `counts` int64 [N]
    (rows sorted within each file), file names in the metadata."""

    files: list[str]
    file_offsets: np.ndarray
    row_ids: np.ndarray
    counts: np.ndarray
    key_fingerprint: str

    @classmethod
    def build(cls, key, files: Mapping[str, Sequence[int] | np.ndarray]) -> "ReaderIndex":
        """`files`: name -> token ids (each file hashed from a fresh, eos-seeded history)."""
        from .hashing import hasher_for
        h = hasher_for(key)
        names = sorted(files)
        offs, ids_l, cnt_l = [0], [], []
        for n in names:
            toks = np.asarray(files[n], dtype=np.int64).reshape(-1)
            r = h.rows_for(toks).reshape(-1) if toks.size else np.empty(0, np.int64)
            u, c = np.unique(r, return_counts=True)
            ids_l.append(u)
            cnt_l.append(c.astype(np.int64))
            offs.append(offs[-1] + u.size)
        cat = (lambda xs: np.concatenate(xs) if xs else np.empty(0, np.int64))
        return cls(names, np.asarray(offs, np.int64), cat(ids_l), cat(cnt_l), key.fingerprint())

    def rows_of(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.files.index(name)
        a, b = self.file_offsets[i], self.file_offsets[i + 1]
        return self.row_ids[a:b], self.counts[a:b]

    def totals(self, exclude: Iterable[str] = ()) -> tuple[np.ndarray, np.ndarray]:
        """(rows, total count) over all files except `exclude`."""
        ex = set(exclude)
        parts = [self.rows_of(f) for f in self.files if f not in ex]
        if not parts:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        ids = np.concatenate([p[0] for p in parts])
        c = np.concatenate([p[1] for p in parts])
        u, inv = np.unique(ids, return_inverse=True)
        return u, np.bincount(inv, weights=c).astype(np.int64)

    def dirty_rows(self, changed: Iterable[str]) -> np.ndarray:
        """Rows read by any changed file (old version): candidates for reset-and-refit."""
        parts = [self.rows_of(f)[0] for f in changed if f in self.files]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)

    def dead_rows(self, removed: Iterable[str], new: "ReaderIndex | None" = None) -> np.ndarray:
        """Rows read before but by no file after removing `removed` (and, with `new`, after the new versions)."""
        removed = list(removed)
        before = self.totals()[0]
        after = self.totals(exclude=removed)[0]
        if new is not None:
            after = np.union1d(after, new.totals()[0])
        return np.setdiff1d(before, after, assume_unique=True)

    def save(self, path: str) -> str:
        write_safetensors(path, {"file_offsets": self.file_offsets, "row_ids": self.row_ids, "counts": self.counts},
                          {"ngram_reader_index": json.dumps(dict(format=READER_INDEX_FORMAT, files=self.files,
                                                                 key_fingerprint=self.key_fingerprint))})
        return sha256_file(path)

    @classmethod
    def load(cls, path: str, sha256: str | None = None) -> "ReaderIndex":
        if sha256 is not None and sha256_file(path) != sha256:
            raise ValueError("reader index sha256 mismatch")
        t, meta = read_safetensors(path)
        m = json.loads(meta["ngram_reader_index"])
        if m.get("format") != READER_INDEX_FORMAT:
            raise ValueError("not a reader index")
        return cls(list(m["files"]), t["file_offsets"], t["row_ids"], t["counts"], m["key_fingerprint"])


# --------------------------------------------------------------------------------------- torch bridge
def row_overlay_from(ids: np.ndarray, values_f32: np.ndarray, num_rows: int, *, engine_cast=None,
                     trainable: bool = False):
    """A serve.hf_adapter.RowOverlay holding fp32 `values_f32` at `ids` (lazy torch import)."""
    import torch

    from .serve.hf_adapter import RowOverlay
    from .tables import SparseRowStore
    st = SparseRowStore(num_rows, values_f32.shape[1], torch.from_numpy(np.ascontiguousarray(ids, np.int64)),
                        torch.from_numpy(np.ascontiguousarray(values_f32, np.float32)))
    return RowOverlay(st, trainable=trainable, engine_cast=engine_cast)


__all__ = ["GraftManifest", "write_graft_file", "read_graft_file", "CartridgeInfo", "make_cartridge_patch",
           "cartridge_info", "check_against_graft", "graft_pin", "CartridgeChain", "compose_patches",
           "compile_served", "compile_chain", "ServedCartridge", "write_served", "read_served", "CompiledOverlayNP",
           "ReaderIndex", "row_overlay_from", "config_sha256", "array_sha256", "verify_patch", "read_patch",
           "write_patch"]
