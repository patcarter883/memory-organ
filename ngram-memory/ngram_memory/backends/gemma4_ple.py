"""Gemma-PLE backend (DESIGN.md §2.5, §4.2, §4.4 "Alternative format"): token-keyed per-layer embeddings.

Key = (token t, layer l); row = `W[t, D*l : D*(l+1)]` of `embed_tokens_per_layer.weight` [V, L*D]. The editor
treats a table row (one token, all layers, width L*D) as the ledger row and a (t, l) slot as a column range.

* `GemmaPLESpec`: the token-keyed table format (its own fingerprint; there is no hashing, so it is not a KeySpec).
* `Gemma4PLEBackend`: Backend protocol over a Gemma4ForCausalLM whose PLE is enabled. Applied edits are a
  `RowLedger` overlay hooked on the table module; served value = `cast(W[t] + sum delta) * embed_scale` exactly as
  `Gemma4TextScaledWordEmbedding` computes it (g4:1444), and rows without a delta are left bit-identical.
  Row 0 (pad / multimodal placeholder, g4:2316-2320) is never edited.
* `editor_init_ple_graft`: a host with PLE disabled (26B/31B) re-built with PLE enabled at the editor init that
  gives an exact no-op: table 0, `per_layer_model_projection` 0 (optionally `per_layer_projection_norm` 0),
  `post_per_layer_input_norm.weight` NONZERO (Gemma4RMSNorm scales by `weight`, not 1+w, g4:222: weight 0 would
  silence the branch and zero the gradient, P/redteam-correctness/checks.out [3]); gate G_l and projection P_l
  keep a seeded random init.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ledger import CallableBase, EditRecord, RowLedger
from ..tables import SparseRowEmbedding, SparseRowStore
from .base import Key, ParityReport, RowAddr, RowDiff, StateSpec

GEMMA_PLE_TENSOR = "model.language_model.embed_tokens_per_layer.weight"


@dataclass(frozen=True)
class GemmaPLESpec:
    vocab_size: int
    n_layers: int
    slot_width: int  # D = hidden_size_per_layer_input
    pad_token_id: int = 0
    tensor_name: str = GEMMA_PLE_TENSOR
    tokenizer_sha256: str | None = None
    kind: str = "unigram"

    @property
    def row_width(self) -> int:
        return self.n_layers * self.slot_width

    def slot(self, token: int, layer: int) -> tuple[int, int, int]:
        """(row, col0, ncols) of key (token, layer)."""
        if not (0 <= token < self.vocab_size and 0 <= layer < self.n_layers):
            raise IndexError((token, layer))
        return int(token), int(layer) * self.slot_width, self.slot_width

    def byte_range(self, token: int, layer: int, dtype_bytes: int = 2) -> tuple[int, int]:
        """Byte range of a slot relative to the tensor's data start (row-major [V, L*D]; P/t2-gemma4-ple)."""
        s = (token * self.row_width + layer * self.slot_width) * dtype_bytes
        return s, s + self.slot_width * dtype_bytes

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(dict(schema="GemmaPLESpec/v1", **asdict(self)), sort_keys=True).encode()
                              ).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GemmaPLESpec":
        return cls(**d)

    @classmethod
    def from_config(cls, cfg, tokenizer_sha256: str | None = None) -> "GemmaPLESpec":
        t = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        if not t.hidden_size_per_layer_input:
            raise ValueError("config has no per-layer embeddings (hidden_size_per_layer_input = 0)")
        return cls(vocab_size=t.vocab_size_per_layer_input, n_layers=t.num_hidden_layers,
                   slot_width=t.hidden_size_per_layer_input, pad_token_id=t.pad_token_id or 0,
                   tokenizer_sha256=tokenizer_sha256)


def gemma_text_model(model: nn.Module) -> nn.Module:
    m = model
    for name in ("model", "language_model"):
        if hasattr(m, name) and not hasattr(m, "embed_tokens_per_layer"):
            m = getattr(m, name)
    if not hasattr(m, "embed_tokens_per_layer"):
        raise TypeError("model has no embed_tokens_per_layer (PLE disabled?)")
    return m


class ScaledRowOverlay(nn.Module):
    """Forward hook on a `Gemma4TextScaledWordEmbedding`: rows with a delta are replaced by
    `cast(W[t] + delta[t]) * embed_scale` (the module's own formula), rows without one are left untouched."""

    def __init__(self, delta: SparseRowStore, trainable: bool = False, engine_cast: torch.dtype | None = None,
                 mode: str = "replace"):
        """mode 'replace' (applied edits: served rows recomputed from the base weight) or 'add' (a trainable leaf
        stacked on top: output + delta * embed_scale, so it adds to whatever is served)."""
        super().__init__()
        if mode not in ("replace", "add"):
            raise ValueError(mode)
        self.delta = SparseRowEmbedding(delta.copy(), sparse_grad=False, requires_grad=trainable)
        self.engine_cast, self.mode = engine_cast, mode
        self._handle = None

    def _hook(self, module, inputs, output):
        ids = inputs[0]
        if self.delta.row_ids.numel() == 0:
            return output
        if self.mode == "add":
            return output + (self.delta(ids) * module.embed_scale.to(torch.float32)).to(output.dtype)
        hit = torch.isin(ids, self.delta.row_ids.to(ids.device))
        if not bool(hit.any()):
            return output
        base = F.embedding(ids, module.weight).to(torch.float32)
        rows = base + self.delta(ids).to(device=base.device, dtype=torch.float32)
        rows = rows.to(self.engine_cast or module.weight.dtype).to(module.weight.dtype)
        new = rows * module.embed_scale.to(module.weight.dtype)
        return torch.where(hit.unsqueeze(-1), new.to(output.dtype), output)

    def install(self, table: nn.Module) -> "ScaledRowOverlay":
        if self._handle is not None:
            raise RuntimeError("already installed")
        if (table.num_embeddings, table.embedding_dim) != (self.delta.num_embeddings, self.delta.embedding_dim):
            raise ValueError("overlay shape does not match the table")
        self._handle = table.register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class Gemma4PLEBackend:
    """Backend protocol (§3.2) over a Gemma4 model with per-layer embeddings (E4B natively, or an
    `editor_init_ple_graft` host)."""

    def __init__(self, model: nn.Module, spec: GemmaPLESpec | None = None, engine_cast: torch.dtype | None = None):
        self.model = model
        self.tm = gemma_text_model(model)
        self.spec = spec or GemmaPLESpec.from_config(model.config)
        self.table = self.tm.embed_tokens_per_layer
        if (self.table.num_embeddings, self.table.embedding_dim) != (self.spec.vocab_size, self.spec.row_width):
            raise ValueError("table shape does not match the spec")
        W = self.table.weight

        def bits(ids: np.ndarray) -> np.ndarray:
            r = W.detach()[torch.from_numpy(ids)].cpu()
            if r.dtype == torch.bfloat16:
                return r.view(torch.int16).numpy().view(np.uint8).reshape(ids.size, -1)
            return r.to(torch.float32).numpy().view(np.uint8).reshape(ids.size, -1)

        codec = "bf16" if W.dtype == torch.bfloat16 else "fp32"
        self.ledger = RowLedger(self.spec.vocab_size, self.spec.row_width, CallableBase(bits, codec, self.spec.row_width),
                                spec_fingerprint=self.spec.fingerprint(), tensor=self.spec.tensor_name)
        self.overlay = ScaledRowOverlay(SparseRowStore.empty(self.spec.vocab_size, self.spec.row_width),
                                        engine_cast=engine_cast).install(self.table)
        self._ids = itertools.count()

    # ------------------------------------------------------------------------------------------- keys
    def keys(self, ids, positions: Sequence[int]) -> list[Key]:
        x = np.asarray(ids.detach().cpu() if isinstance(ids, torch.Tensor) else ids, dtype=np.int64).reshape(-1)
        return [(int(x[p]), l) for p in positions for l in range(self.spec.n_layers)]

    def rows_for(self, keys: Sequence[Key]) -> RowAddr:
        s = [self.spec.slot(t, l) for t, l in keys]
        return RowAddr(self.spec.tensor_name, torch.tensor([a[0] for a in s], dtype=torch.long),
                       torch.tensor([a[1] for a in s], dtype=torch.long), self.spec.slot_width)

    def read_rows(self, addr: RowAddr) -> torch.Tensor:
        """[N, D] fp32 slots as served (base + active deltas, before embed_scale)."""
        _, full = self.ledger.materialise(addr.rows.numpy(), cast=None)
        c = addr.col0.numpy()
        return torch.from_numpy(np.stack([full[i, c[i]:c[i] + addr.ncols] for i in range(len(c))])
                                if len(c) else np.zeros((0, addr.ncols), np.float32))

    @torch.no_grad()
    def effective_ple(self, token: int) -> torch.Tensor:
        """[L, D] per_layer_input the decoder layers receive for `token` (a function of the token alone)."""
        ids = torch.tensor([[token]])
        return self.tm.project_per_layer_inputs(self.tm.embed_tokens(ids), self.tm.get_per_layer_inputs(ids))[0, 0]

    # --------------------------------------------------------------------------------------- forward
    def delta_leaf(self, tokens) -> ScaledRowOverlay:
        t = torch.as_tensor(tokens).reshape(-1)
        if bool((t == self.spec.pad_token_id).any()):
            raise ValueError("row 0 / pad is never edited")
        return ScaledRowOverlay(SparseRowStore.reserved(self.spec.vocab_size, self.spec.row_width, t), trainable=True,
                                mode="add")

    def forward(self, batch: dict, delta: ScaledRowOverlay | None = None):
        """Frozen forward with a trainable leaf. The leaf hooks the same table AFTER the applied-edit overlay, so
        it adds to the served rows (delta in table units, before embed_scale)."""
        if delta is None:
            return self.model(**batch)
        delta.install(self.table)
        try:
            return self.model(**batch)
        finally:
            delta.remove()

    # ----------------------------------------------------------------------------------------- edits
    def _publish(self) -> None:
        ids, d = self.ledger.delta_sum()
        self.overlay.delta.load_store(SparseRowStore(self.spec.vocab_size, self.spec.row_width, torch.from_numpy(ids),
                                                     torch.from_numpy(d)))

    def apply(self, diff: RowDiff | EditRecord) -> int:
        if isinstance(diff, RowDiff):
            if diff.mode != "delta":
                raise ValueError("the Gemma PLE table takes 'delta' diffs")
            rec = EditRecord(diff.edit_id or f"edit{next(self._ids):06d}", diff.row_ids.cpu().numpy(),
                             diff.values.detach().cpu().numpy(), spec_fingerprint=self.spec.fingerprint(),
                             tensor=self.spec.tensor_name)
        else:
            rec = diff
        if (rec.row_ids == self.spec.pad_token_id).any():
            raise ValueError("row 0 / pad is never edited")
        ep = self.ledger.add(rec)
        self._publish()
        return ep

    def apply_slot(self, token: int, layer: int, values: torch.Tensor, edit_id: str | None = None) -> int:
        row, c0, n = self.spec.slot(token, layer)
        return self.apply(EditRecord(edit_id or f"edit{next(self._ids):06d}", [row],
                                     values.detach().reshape(1, n).cpu().numpy(), col0=c0, ncols=n,
                                     spec_fingerprint=self.spec.fingerprint(), tensor=self.spec.tensor_name))

    def revert(self, edit_id: str) -> int:
        ep = self.ledger.remove(edit_id)
        self._publish()
        return ep

    # ------------------------------------------------------------------------------------ invariants
    def stale(self, token_seqs: Sequence[torch.Tensor], row_ids) -> torch.Tensor:
        rid = torch.as_tensor(row_ids).reshape(-1)
        return torch.tensor([bool(torch.isin(s.reshape(-1).cpu(), rid).any()) for s in token_seqs])

    def decode_state(self) -> StateSpec:
        """No extra state: per_layer_inputs are recomputed from each step's ids (g4:1637-1640)."""
        return StateSpec(layer_idx=-1, slots={})

    @torch.no_grad()
    def parity(self, ids: torch.Tensor | None = None, prefill: int | None = None) -> ParityReport:
        if ids is None:
            ids = torch.randint(1, self.spec.vocab_size, (1, 24), generator=torch.Generator().manual_seed(0))
        full = self.model(input_ids=ids, use_cache=False).logits.float()
        P = prefill or ids.shape[1] // 2
        out = self.model(input_ids=ids[:, :P], use_cache=True)
        d = [(out.logits.float() - full[:, :P]).abs().max().item()]
        cache = out.past_key_values
        for t in range(P, ids.shape[1]):
            out = self.model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            d.append((out.logits.float()[:, -1] - full[:, t]).abs().max().item())
        return ParityReport(rows_for_matches_model=True, cached_vs_full_max_abs=max(d),
                            notes=["token-keyed: the row id is the token id"])

    def remove(self) -> None:
        self.overlay.remove()


# ------------------------------------------------------------------------------------------- PLE graft
PLE_GRAFT_KEYS = ("embed_tokens_per_layer", "per_layer_input_gate", "per_layer_projection",
                  "post_per_layer_input_norm", "per_layer_model_projection", "per_layer_projection_norm")


@torch.no_grad()
def editor_init_ple_graft(host: nn.Module, slot_width: int, *, seed: int = 0, post_norm_weight: float = 1.0,
                          zero_projection_norm: bool = True) -> nn.Module:
    """A copy of a gemma4 host (PLE disabled) with Gemma's own PLE enabled at the editor init (exact no-op)."""
    if post_norm_weight == 0:
        raise ValueError("post_per_layer_input_norm.weight = 0 silences the branch: writes would be no-ops with "
                         "zero gradient (Gemma4RMSNorm scales by weight, g4:222)")
    cfg = host.config.__class__.from_dict(host.config.to_dict())
    cfg._attn_implementation = host.config._attn_implementation
    if getattr(cfg, "hidden_size_per_layer_input", 0):
        raise ValueError("host already has per-layer embeddings")
    cfg.hidden_size_per_layer_input = slot_width
    cfg.vocab_size_per_layer_input = cfg.vocab_size
    torch.manual_seed(seed)
    g = type(host)(cfg).eval().to(next(host.parameters()).dtype)
    missing, unexpected = g.load_state_dict(host.state_dict(), strict=False)
    bad = [k for k in missing if not any(s in k for s in PLE_GRAFT_KEYS)]
    if bad or unexpected:
        raise RuntimeError(f"unexpected state-dict mismatch: missing {bad}, unexpected {list(unexpected)}")
    tm = gemma_text_model(g)
    tm.embed_tokens_per_layer.weight.zero_()
    tm.per_layer_model_projection.weight.zero_()
    if zero_projection_norm:
        tm.per_layer_projection_norm.weight.zero_()
    for layer in tm.layers:
        layer.post_per_layer_input_norm.weight.fill_(post_norm_weight)
    g._ple_graft_missing_keys = sorted(missing)
    return g


__all__ = ["GemmaPLESpec", "Gemma4PLEBackend", "ScaledRowOverlay", "editor_init_ple_graft", "gemma_text_model",
           "GEMMA_PLE_TENSOR"]
