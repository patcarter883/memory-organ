"""Edit records and the row ledger (DESIGN.md §3.3). numpy only.

* `EditRecord`: one edit (a fact, a cartridge or an increment): provenance, the rows it touches
  (tensor, row ids, col0, ncols) with each row's `base_bits_sha256`, and its fp32 `delta` [R, ncols].
* `RowLedger`: row -> [(edit_id, fp32 delta)], with each row's ORIGINAL bits saved on first write.

Materialisation is an overlay, the base bits are never rewritten:

    served[r] = engine_cast(decode(orig_bits[r]) + (sum_{active e, canonical order} delta_e[r]))

The canonical order is sorted edit id, so the result does not depend on the order edits were applied or removed;
removal is re-materialisation from the saved bits (bit-exact), never in-place subtraction (which left residue on
50/64 rows in P/solver/v5.out). Conflict sets: the active edits that share >= 1 row with an edit.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

from . import codecs
from .patch import Patch, PatchManifest, make_patch, read_safetensors, rows2d, write_safetensors

LEDGER_FORMAT = "ngram-ledger/v1"


# -------------------------------------------------------------------------------------------- base rows
class BaseRows(Protocol):
    """Where a ledger reads a row's ORIGINAL bits and how it decodes them."""

    codec: str
    row_width: int

    def gather_bits(self, ids: np.ndarray) -> np.ndarray: ...  # uint8 [n, row_bytes]


class ZeroBase:
    """Grafts: every base row is 0 (codec 'none', no bits)."""

    codec = "none"

    def __init__(self, row_width: int):
        self.row_width = int(row_width)

    def gather_bits(self, ids: np.ndarray) -> np.ndarray:
        return np.zeros((np.asarray(ids).size, 0), np.uint8)


class ArrayBase:
    """A table held as raw row bytes [N, row_bytes] (fp32 / bf16 / int6 fused) or as an fp32 / bf16-bits array."""

    def __init__(self, table: np.ndarray, codec: str, row_width: int | None = None):
        self.codec = codec
        t = np.ascontiguousarray(table)
        if codec == "fp32" and t.dtype == np.float32:
            self.row_width, t = t.shape[1], t.view(np.uint8).reshape(t.shape[0], -1)
        elif codec == "bf16" and t.dtype == np.uint16:
            self.row_width, t = t.shape[1], t.view(np.uint8).reshape(t.shape[0], -1)
        else:
            if row_width is None:
                raise ValueError("row_width is needed for raw byte tables")
            self.row_width = int(row_width)
        if t.dtype != np.uint8 or t.shape[1] != codecs.row_bytes(codec, self.row_width):
            raise ValueError(f"table {t.dtype} {t.shape} does not hold {codec} rows of width {self.row_width}")
        self.raw = t

    def gather_bits(self, ids: np.ndarray) -> np.ndarray:
        return self.raw[np.asarray(ids, dtype=np.int64)]


class CallableBase:
    """Adapter for any store: `fn(ids) -> uint8 [n, row_bytes]` (e.g. ShardMmapStore.gather_raw)."""

    def __init__(self, fn, codec: str, row_width: int):
        self.fn, self.codec, self.row_width = fn, codec, int(row_width)

    def gather_bits(self, ids: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.fn(np.asarray(ids, dtype=np.int64)), dtype=np.uint8)


def row_bits_sha256(bits: np.ndarray) -> list[str]:
    return [hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() for b in bits]


# ------------------------------------------------------------------------------------------- EditRecord
@dataclass
class EditRecord:
    """One edit. `delta` is fp32 [R, ncols] for columns [col0, col0+ncols) of rows `row_ids` of `tensor`."""

    edit_id: str
    row_ids: np.ndarray
    delta: np.ndarray
    spec_fingerprint: str | None = None
    tensor: str = ""
    col0: int = 0
    ncols: int | None = None
    kind: str = "fact"  # fact | cartridge | increment
    mode: str = "delta"  # delta (native) | absolute (graft; base 0)
    namespace: str | None = None
    created: float = field(default_factory=time.time)
    author: str | None = None
    model_repo: str | None = None
    model_commit: str | None = None
    fact: dict | None = None  # (s, r, o_true, o_new)
    prompts: dict | None = None  # prompt sets + template train/held-out split
    keys: list | None = None  # token-id n-grams with tier
    solver: dict | None = None
    metrics: dict | None = None
    # filled by RowLedger.add
    base_bits_sha256: list[str] | None = None
    parent_epoch: int | None = None
    conflict_set: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = np.asarray(self.row_ids, dtype=np.int64).reshape(-1)
        d = rows2d(self.delta, ids.size, np.float32, self.ncols if ids.size == 0 and self.ncols is not None else None)
        order = np.argsort(ids, kind="stable")
        ids, d = ids[order], d[order]
        if ids.size and (np.diff(ids) == 0).any():
            raise ValueError(f"edit {self.edit_id}: duplicate row ids (sum them first)")
        if not np.isfinite(d).all():
            raise ValueError(f"edit {self.edit_id}: non-finite delta")
        self.row_ids, self.delta = np.ascontiguousarray(ids), np.ascontiguousarray(d)
        if self.ncols is None:
            self.ncols = int(d.shape[1])
        if d.shape[1] != self.ncols:
            raise ValueError(f"edit {self.edit_id}: delta has {d.shape[1]} columns, ncols={self.ncols}")
        if self.mode not in ("delta", "absolute"):
            raise ValueError(f"edit {self.edit_id}: mode {self.mode!r}")

    META_FIELDS = ("edit_id", "spec_fingerprint", "tensor", "col0", "ncols", "kind", "mode", "namespace", "created",
                   "author", "model_repo", "model_commit", "fact", "prompts", "keys", "solver", "metrics",
                   "base_bits_sha256", "parent_epoch", "conflict_set")

    def meta(self) -> dict:
        return {k: getattr(self, k) for k in self.META_FIELDS}

    @classmethod
    def from_meta(cls, meta: dict, row_ids: np.ndarray, delta: np.ndarray) -> "EditRecord":
        return cls(row_ids=row_ids, delta=delta, **{k: meta[k] for k in cls.META_FIELDS if k in meta})

    @classmethod
    def from_patch(cls, patch: Patch, edit_id: str | None = None, **kw) -> "EditRecord":
        """A whole patch as one edit (its reset rows are not representable here; use cartridge.compose_chain)."""
        if patch.reset_row_ids.size:
            raise ValueError("patches with reset_row_ids compose through cartridge chains, not ledger edits")
        m = patch.manifest
        prov = m.edits[0] if len(m.edits) == 1 else {}
        eid = edit_id or prov.get("edit_id") or m.payload_sha256[:16]
        kind = (m.cartridge or {}).get("kind", prov.get("kind", "fact"))
        return cls(edit_id=eid, row_ids=patch.row_ids, delta=patch.values, spec_fingerprint=m.format_fingerprint,
                   tensor=m.table_spec.get("tensor_name", ""), kind=kind, mode=m.mode,
                   namespace=(m.cartridge or {}).get("namespace"), model_repo=m.model_repo,
                   model_commit=m.model_commit, **kw)


# --------------------------------------------------------------------------------------------- RowLedger
class RowLedger:
    """Row -> [(edit_id, fp32 delta)] with original bits saved on first write. `epoch` counts changes."""

    def __init__(self, num_rows: int, row_width: int, base: BaseRows | None = None,
                 spec_fingerprint: str | None = None, tensor: str = ""):
        self.num_rows, self.row_width = int(num_rows), int(row_width)
        self.base = base if base is not None else ZeroBase(row_width)
        if self.base.row_width != self.row_width:
            raise ValueError("base row width differs from the ledger's")
        self.codec = self.base.codec
        self.spec_fingerprint, self.tensor = spec_fingerprint, tensor
        self.records: dict[str, EditRecord] = {}
        self.active: set[str] = set()
        self.epoch = 0
        self.log: list[tuple[int, str, str]] = []  # (epoch, op, edit_id)
        self._orig_ids = np.empty(0, np.int64)
        self._orig_bits = np.zeros((0, codecs.row_bytes(self.codec, self.row_width)), np.uint8)
        self._orig_f32 = np.zeros((0, self.row_width), np.float32)

    # ---------------------------------------------------------------------------------------- orig bits
    def _save_orig(self, ids: np.ndarray) -> np.ndarray:
        """Save the original bits of rows not seen before; return the bits of `ids`."""
        new = ids[~np.isin(ids, self._orig_ids)]
        if new.size:
            bits = self.base.gather_bits(new)
            f32 = codecs.decode_rows(bits, self.codec, self.row_width)
            allid = np.concatenate([self._orig_ids, new])
            o = np.argsort(allid, kind="stable")
            self._orig_ids = allid[o]
            self._orig_bits = np.concatenate([self._orig_bits, bits])[o]
            self._orig_f32 = np.concatenate([self._orig_f32, f32])[o]
        return self._orig_bits[np.searchsorted(self._orig_ids, ids)]

    def orig_rows(self, ids) -> np.ndarray:
        """fp32 decode of the saved original bits (rows never written read through the base)."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        out = np.empty((ids.size, self.row_width), np.float32)
        if self._orig_ids.size:
            pos = np.searchsorted(self._orig_ids, ids).clip(max=self._orig_ids.size - 1)
            have = self._orig_ids[pos] == ids
        else:
            pos, have = np.zeros(ids.size, np.int64), np.zeros(ids.size, bool)
        if have.any():
            out[have] = self._orig_f32[pos[have]]
        if (~have).any():
            out[~have] = codecs.decode_rows(self.base.gather_bits(ids[~have]), self.codec, self.row_width)
        return out

    def orig_bits(self, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if not np.isin(ids, self._orig_ids).all():
            raise KeyError("orig_bits: some rows were never written")
        return self._orig_bits[np.searchsorted(self._orig_ids, ids)]

    # --------------------------------------------------------------------------------------- mutation
    def add(self, rec: EditRecord) -> int:
        if rec.edit_id in self.records:
            raise KeyError(f"edit {rec.edit_id} already in the ledger")
        if self.spec_fingerprint is not None and rec.spec_fingerprint not in (None, self.spec_fingerprint):
            raise ValueError(f"edit {rec.edit_id}: spec fingerprint {rec.spec_fingerprint} != ledger "
                             f"{self.spec_fingerprint}")
        if rec.mode == "absolute" and self.codec != "none":
            raise ValueError("absolute edits need a zero base (graft)")
        if rec.col0 < 0 or rec.col0 + rec.ncols > self.row_width:
            raise ValueError(f"edit {rec.edit_id}: columns [{rec.col0}, {rec.col0 + rec.ncols}) outside the row")
        if rec.row_ids.size and (rec.row_ids[0] < 0 or rec.row_ids[-1] >= self.num_rows):
            raise IndexError(f"edit {rec.edit_id}: row id out of range")
        bits = self._save_orig(rec.row_ids)
        rec.base_bits_sha256 = row_bits_sha256(bits) if self.codec != "none" else None
        rec.parent_epoch = self.epoch
        rec.conflict_set = sorted(self._sharing(rec.row_ids, exclude=rec.edit_id))
        self.records[rec.edit_id] = rec
        self.active.add(rec.edit_id)
        self.epoch += 1
        self.log.append((self.epoch, "add", rec.edit_id))
        return self.epoch

    def remove(self, edit_id: str) -> int:
        """Deactivate an edit (the record is kept for provenance). Re-validate `dependents(edit_id)`."""
        if edit_id not in self.active:
            raise KeyError(f"edit {edit_id} is not active")
        self.active.discard(edit_id)
        self.epoch += 1
        self.log.append((self.epoch, "remove", edit_id))
        return self.epoch

    def reactivate(self, edit_id: str) -> int:
        if edit_id not in self.records or edit_id in self.active:
            raise KeyError(f"edit {edit_id} is unknown or already active")
        self.active.add(edit_id)
        self.epoch += 1
        self.log.append((self.epoch, "add", edit_id))
        return self.epoch

    # ---------------------------------------------------------------------------------------- queries
    def order(self) -> list[str]:
        """Active edits in canonical (sorted id) order: the fp32 summation order."""
        return sorted(self.active)

    def _sharing(self, ids: np.ndarray, exclude: str | None = None) -> set[str]:
        return {e for e in self.active if e != exclude and np.isin(self.records[e].row_ids, ids,
                                                                    assume_unique=True).any()}

    def conflict_set(self, edit_id: str) -> list[str]:
        """Active edits sharing >= 1 row with `edit_id` (now, not at add time)."""
        return sorted(self._sharing(self.records[edit_id].row_ids, exclude=edit_id))

    def conflict_sets(self) -> dict[str, list[str]]:
        return {e: self.conflict_set(e) for e in self.order()}

    def dependents(self, edit_id: str) -> list[str]:
        """Active edits added after `edit_id` that share rows with it: solved conditioned on it, so re-validate them
        when it is removed (§3.3)."""
        r = self.records[edit_id]
        added = {e: ep for ep, op, e in self.log if op == "add"}
        t0 = added.get(edit_id, 0)
        return sorted(e for e in self._sharing(r.row_ids, exclude=edit_id) if added.get(e, 0) > t0)

    def edits_touching(self, ids) -> list[str]:
        return sorted(self._sharing(np.asarray(ids, dtype=np.int64).reshape(-1)))

    def active_rows(self) -> np.ndarray:
        """Rows with >= 1 active edit (the overlay set), sorted."""
        if not self.active:
            return np.empty(0, np.int64)
        return np.unique(np.concatenate([self.records[e].row_ids for e in self.active]))

    def touched_rows(self) -> np.ndarray:
        """Every row ever written (needed to restore base bits in offline in-place mode)."""
        return self._orig_ids.copy()

    def rows_changed_since(self, epoch: int) -> np.ndarray:
        """Rows whose served value may differ from what it was at `epoch` (cache invalidation, §3.4)."""
        ids = [self.records[e].row_ids for ep, _, e in self.log if ep > epoch]
        return np.unique(np.concatenate(ids)) if ids else np.empty(0, np.int64)

    # ----------------------------------------------------------------------------------- materialise
    def delta_sum(self, ids=None, edits: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(ids, fp32 sum of active deltas) in canonical order; ids default to `active_rows()`."""
        order = sorted(edits) if edits is not None else self.order()
        ids = (np.unique(np.concatenate([self.records[e].row_ids for e in order])) if order
               else np.empty(0, np.int64)) if ids is None else np.asarray(ids, dtype=np.int64).reshape(-1)
        acc = np.zeros((ids.size, self.row_width), np.float32)
        self._accumulate(acc, ids, order)
        return ids, acc

    def _accumulate(self, acc: np.ndarray, ids: np.ndarray, order: Iterable[str]) -> None:
        if ids.size == 0:
            return
        sid = np.argsort(ids, kind="stable")
        sorted_ids = ids[sid]
        for e in order:
            r = self.records[e]
            p = np.searchsorted(sorted_ids, r.row_ids).clip(max=sorted_ids.size - 1)
            hit = sorted_ids[p] == r.row_ids
            if not hit.any():
                continue
            dst = sid[p[hit]]
            c0, c1 = r.col0, r.col0 + r.ncols
            acc[dst, c0:c1] = acc[dst, c0:c1] + r.delta[hit]  # dst unique per edit: no lost updates

    def materialise(self, ids=None, cast: str | None = "bf16", edits: Sequence[str] | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
        """(ids, served rows) with served = cast(decode(orig) + sum active deltas), summed in canonical order.

        `cast` 'bf16' returns fp32 holding bf16 values (the engine cast), None/'fp32' the fp32 sum. `ids` default
        to the active overlay rows; rows with no active edit come back as cast(decode(orig))."""
        order = sorted(edits) if edits is not None else self.order()
        if ids is None:
            ids = (np.unique(np.concatenate([self.records[e].row_ids for e in order])) if order
                   else np.empty(0, np.int64))
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        # base + (sum of deltas): the same association as the patch / serve compile (compile_served), so the
        # ledger's view and the served bits agree bit-for-bit (VF-1)
        dsum = np.zeros((ids.size, self.row_width), np.float32)
        self._accumulate(dsum, ids, order)
        acc = self.orig_rows(ids) + dsum
        if cast == "bf16":
            acc = codecs.bf16_round(acc)
        elif cast not in (None, "fp32"):
            raise ValueError(f"unknown cast {cast!r}")
        return ids, acc

    def compiled_overlay(self, cast: str = "bf16") -> tuple[np.ndarray, np.ndarray]:
        """Overlay payload for a replacement-row engine hook: (sorted ids, bf16 bits uint16 [U, w])."""
        ids, rows = self.materialise(cast=None)
        if cast != "bf16":
            raise ValueError("compiled overlays are bf16 replacement rows (INTEGRATION §2.3)")
        return ids, codecs.bf16_bits(rows)

    def to_patch(self, fmt, edits: Sequence[str] | None = None, **manifest_kw) -> Patch:
        """The summed active deltas as one `ngram-edit/v1` patch for `fmt` (mode from the format)."""
        if self.spec_fingerprint is not None and fmt.fingerprint() != self.spec_fingerprint:
            raise ValueError("format fingerprint differs from the ledger's")
        ids, d = self.delta_sum(edits=edits)
        order = sorted(edits) if edits is not None else self.order()
        prov = [dict(edit_id=e, kind=self.records[e].kind, n_rows=int(self.records[e].row_ids.size),
                     parent_epoch=self.records[e].parent_epoch, conflict_set=self.records[e].conflict_set)
                for e in order]
        base = None if fmt.table.base_codec == "none" else self.orig_rows(ids)
        return make_patch(fmt, ids, d, base_rows=base, edits=prov, **manifest_kw)

    # ------------------------------------------------------------------------------------- save/load
    def save(self, directory: str) -> None:
        os.makedirs(os.path.join(directory, "edits"), exist_ok=True)
        for e, r in self.records.items():
            p = os.path.join(directory, "edits", f"{_safe(e)}.safetensors")
            # always (re)written, atomically: a stale file from another ledger with the same edit id must never
            # be read back as this ledger's delta (VF-8)
            write_safetensors(p + ".tmp", {"row_ids": r.row_ids, "delta": r.delta},
                              {"edit_record": json.dumps(r.meta(), sort_keys=True)})
            os.replace(p + ".tmp", p)
        write_safetensors(os.path.join(directory, "orig.safetensors"),
                          {"orig_ids": self._orig_ids, "orig_bits": self._orig_bits})
        meta = dict(format=LEDGER_FORMAT, num_rows=self.num_rows, row_width=self.row_width, codec=self.codec,
                    spec_fingerprint=self.spec_fingerprint, tensor=self.tensor, epoch=self.epoch,
                    active=sorted(self.active), records=sorted(self.records), log=self.log)
        tmp = os.path.join(directory, "ledger.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(directory, "ledger.json"))

    @classmethod
    def load(cls, directory: str, base: BaseRows | None = None) -> "RowLedger":
        """Reload; `base` is only consulted for rows never written (saved original bits are authoritative)."""
        with open(os.path.join(directory, "ledger.json")) as f:
            meta = json.load(f)
        if meta.get("format") != LEDGER_FORMAT:
            raise ValueError("not a ledger directory")
        if base is None:
            base = ZeroBase(meta["row_width"]) if meta["codec"] == "none" else _NoBase(meta["codec"], meta["row_width"])
        if base.codec != meta["codec"]:
            raise ValueError(f"base codec {base.codec} != saved {meta['codec']}")
        led = cls(meta["num_rows"], meta["row_width"], base, meta["spec_fingerprint"], meta.get("tensor", ""))
        t, _ = read_safetensors(os.path.join(directory, "orig.safetensors"))
        rb = codecs.row_bytes(led.codec, led.row_width)
        led._orig_ids, led._orig_bits = t["orig_ids"], t["orig_bits"].reshape(t["orig_ids"].size, rb)
        led._orig_f32 = codecs.decode_rows(led._orig_bits, led.codec, led.row_width)
        for e in meta["records"]:
            tt, m = read_safetensors(os.path.join(directory, "edits", f"{_safe(e)}.safetensors"))
            led.records[e] = EditRecord.from_meta(json.loads(m["edit_record"]), tt["row_ids"], tt["delta"])
        led.active = set(meta["active"])
        led.epoch = int(meta["epoch"])
        led.log = [tuple(x) for x in meta["log"]]
        return led


class _NoBase:
    def __init__(self, codec: str, row_width: int):
        self.codec, self.row_width = codec, row_width

    def gather_bits(self, ids):
        raise RuntimeError("ledger loaded without a base: only rows with saved original bits are readable")


def _safe(edit_id: str) -> str:
    if not edit_id or any(c in edit_id for c in "/\\\0") or edit_id.startswith("."):
        raise ValueError(f"edit id {edit_id!r} is not file-safe")
    return edit_id


__all__ = ["EditRecord", "RowLedger", "ZeroBase", "ArrayBase", "CallableBase", "BaseRows", "row_bits_sha256",
           "LEDGER_FORMAT"]
