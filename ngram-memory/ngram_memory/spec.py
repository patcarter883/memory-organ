"""Format spec: KeySpec (row ids), TableSpec (physical layout), ModuleSpec (module math) -- DESIGN.md §2.1-§2.4.

Pure python (no numpy/torch/transformers), so the engine side and offline tools can import it.

Conventions
-----------
* ``KeySpec.orders`` is the HEAD ORDER: order ``orders[i]`` owns heads ``[i*hpn, (i+1)*hpn)``. Flash-Next is
  ``(2, 3)``; the graft-only unigram extension is ``(2, 3, 1)`` so heads 0-15 keep Flash-Next's row ids (§2.4).
* Head ``h`` uses the k-th prime after ``base - 1`` with ``k = n_heads * ple_layer_index + h + 1`` (q4:1091-1096).
* Multipliers: ``ngram_size = max(orders)`` of them, q4:1033-1050.
* Fingerprints are sha256 over a canonical JSON of the fields that decide the contract; they are stable across
  python versions and process runs (no ``hash()``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, Sequence

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_1 = 10007
DEFAULT_SEED = 1234

# Flash-Next (Qwen3.8-Flash-Next, qwen4_exp) values; checked against the real checkpoint buffers in tests.
FLASH_NEXT_VOCAB = 248320
FLASH_NEXT_EOS = 248044
FLASH_NEXT_TENSOR = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"
FLASH_NEXT_SHARD_TEMPLATE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight"

# Qwen2.5 proxy (P/stage1/experiment/ingest.py: V=151936, eos 151643, base 200_000 -> primes from 200003).
QWEN25_VOCAB = 151936
QWEN25_EOS = 151643


def splitmix64(x: int) -> int:
    x = (x + SPLITMIX_GAMMA) & MASK64
    x = ((x ^ (x >> 30)) * SPLITMIX_M1) & MASK64
    x = ((x ^ (x >> 27)) * SPLITMIX_M2) & MASK64
    return (x ^ (x >> 31)) & MASK64


def layer_multipliers(vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
    """q4 `_build_layer_multipliers`: odd, bounded by (2^63-1)//V so x*m never overflows int64."""
    max_long = (1 << 63) - 1
    half = max(1, (max_long // max(vocab_size, 1)) // 2)
    base = seed + PRIME_1 * ple_layer_index
    return [2 * (splitmix64((base + SPLITMIX_GAMMA * (i + 1)) & MASK64) % half) + 1 for i in range(ngram_size)]


def is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


def _consecutive_primes_after(start: int, skip: int, n: int) -> list[int]:
    """The (skip+1)-th .. (skip+n)-th primes after `start` (one pass, same values as nth_prime_after)."""
    out: list[int] = []
    p, seen = start, 0
    while len(out) < n:
        p += 1
        if is_prime(p):
            seen += 1
            if seen > skip:
                out.append(p)
    return out


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------- KeySpec
@dataclass(frozen=True)
class KeySpec:
    """Decides row ids (§2.1). Hidden size and row width play no part."""

    vocab_size: int
    eos_token_id: int
    seed: int = DEFAULT_SEED
    ple_layer_index: int = 0
    ngram_size: int = 3
    heads_per_ngram: int = 8
    base: int = 20_000_000
    orders: tuple[int, ...] = (2, 3)
    tokenizer_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", tuple(int(o) for o in self.orders))
        if not self.orders or len(set(self.orders)) != len(self.orders):
            raise ValueError(f"orders must be non-empty and unique, got {self.orders}")
        if min(self.orders) < 1 or max(self.orders) > self.ngram_size:
            raise ValueError(f"orders {self.orders} must lie in [1, ngram_size={self.ngram_size}]")
        if self.heads_per_ngram < 1 or self.vocab_size < 1:
            raise ValueError("heads_per_ngram and vocab_size must be >= 1")

    # ---- derived
    @property
    def n_heads(self) -> int:
        return len(self.orders) * self.heads_per_ngram

    @property
    def context_len(self) -> int:
        """Tokens of history a decode step needs (x_1 .. x_{n-1})."""
        return self.ngram_size - 1

    @cached_property
    def multipliers(self) -> tuple[int, ...]:
        return tuple(layer_multipliers(self.vocab_size, self.ngram_size, self.ple_layer_index, self.seed))

    @cached_property
    def head_sizes(self) -> tuple[int, ...]:
        return tuple(_consecutive_primes_after(self.base - 1, self.n_heads * self.ple_layer_index, self.n_heads))

    @cached_property
    def head_offsets(self) -> tuple[int, ...]:
        offs, acc = [], 0
        for s in self.head_sizes:
            offs.append(acc)
            acc += s
        return tuple(offs)

    @property
    def total_rows(self) -> int:
        return sum(self.head_sizes)

    @property
    def head_orders(self) -> tuple[int, ...]:
        """n-gram order of each head, in head order."""
        return tuple(o for o in self.orders for _ in range(self.heads_per_ngram))

    def heads_of_order(self, n: int) -> tuple[int, ...]:
        i = self.orders.index(n)
        return tuple(range(i * self.heads_per_ngram, (i + 1) * self.heads_per_ngram))

    def padded_rows(self, pad_divisor: int) -> int:
        return math.ceil(self.total_rows / pad_divisor) * pad_divisor

    # ---- contract
    def fingerprint(self) -> str:
        return _sha(dict(kind="KeySpec/v1", V=self.vocab_size, eos=self.eos_token_id, seed=self.seed,
                         ple_layer_index=self.ple_layer_index, multipliers=list(self.multipliers),
                         head_sizes=list(self.head_sizes), orders=list(self.orders),
                         head_orders=list(self.head_orders), tokenizer_sha256=self.tokenizer_sha256))

    def to_dict(self) -> dict:
        return dict(vocab_size=self.vocab_size, eos_token_id=self.eos_token_id, seed=self.seed,
                    ple_layer_index=self.ple_layer_index, ngram_size=self.ngram_size,
                    heads_per_ngram=self.heads_per_ngram, base=self.base, orders=list(self.orders),
                    tokenizer_sha256=self.tokenizer_sha256)

    @classmethod
    def from_dict(cls, d: dict) -> "KeySpec":
        d = dict(d)
        d["orders"] = tuple(d["orders"])
        return cls(**d)


# ----------------------------------------------------------------------------------------------------- TableSpec
BaseCodec = Literal["bf16", "fp32", "fp8_e4m3_global", "int6_g32", "none"]


@dataclass(frozen=True)
class TableSpec:
    """Decides physical layout, not row ids (§2.1b). `base_codec="none"` = graft (base rows are 0)."""

    tensor_name: str
    row_width: int
    pad_divisor: int = 128
    n_shards: int = 1
    shard_name_template: str | None = None
    base_codec: str = "bf16"

    def __post_init__(self) -> None:
        if self.row_width < 1 or self.pad_divisor < 1 or self.n_shards < 1:
            raise ValueError("row_width, pad_divisor, n_shards must be >= 1")
        if self.base_codec not in ("bf16", "fp32", "fp8_e4m3_global", "int6_g32", "none"):
            raise ValueError(f"unknown base_codec {self.base_codec!r}")

    def padded_rows(self, key: KeySpec) -> int:
        return key.padded_rows(self.pad_divisor)

    def rows_per_shard(self, key: KeySpec) -> int:
        """torch.chunk semantics (ceil); equal shards on Flash-Next (320,001,536 / 128 = 2,500,012)."""
        return math.ceil(self.padded_rows(key) / self.n_shards)

    def shard_of(self, key: KeySpec, row: int) -> tuple[int, int]:
        """(shard index, local row) of a global row id (P/t1/src/ngram_format.py checkpoint_loc)."""
        if not 0 <= row < self.padded_rows(key):
            raise IndexError(f"row {row} outside [0, {self.padded_rows(key)})")
        return divmod(int(row), self.rows_per_shard(key))

    def shard_tensor_name(self, shard: int) -> str:
        if self.shard_name_template is None:
            if self.n_shards != 1:
                raise ValueError("n_shards > 1 needs shard_name_template")
            return self.tensor_name
        return self.shard_name_template.format(s=shard)

    def fingerprint(self) -> str:
        return _sha(dict(schema="TableSpec/v1", **self.to_dict()))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TableSpec":
        return cls(**d)


# ---------------------------------------------------------------------------------------------------- ModuleSpec
Arm = Literal["G-lin", "G-dir"]


@dataclass(frozen=True)
class ModuleSpec:
    """Module math (§2.2) plus, for grafts, the fixed editor init (§2.4).

    kind="native": read from a qwen4_exp config (hc_count > 1 allowed).
    kind="graft":  hc_count == 1, W_k = 0, norms 0 (scale 1), table 0, W_v = value_scale * seeded orthogonal;
                   arm "G-lin": conv1d = 0;  arm "G-dir": conv1d[:, 0, j] = gdir_alpha * gdir_taps[j]
                   (tap j multiplies position t - dilation*(kernel-1-j); taps (0,0,0,1) = current position only).
    """

    hidden_size: int
    ple_embed_dim: int
    hc_count: int = 1
    conv_kernel_size: int = 4
    conv_dilation: int = 3
    rms_norm_eps: float = 1e-6
    layer_idx: int = 1  # 0-based decoder index whose INPUT receives the output (Flash-Next ple_layer_ids=[2])
    kind: str = "graft"
    arm: str | None = "G-lin"
    value_seed: int = 0
    value_scale: float = 1.0
    gdir_alpha: float = 1.0
    gdir_taps: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gdir_taps", tuple(float(t) for t in self.gdir_taps))
        if self.kind not in ("native", "graft"):
            raise ValueError(f"kind must be native|graft, got {self.kind!r}")
        if self.kind == "graft":
            if self.hc_count != 1:
                raise ValueError("graft profile requires hc_count == 1 (§2.4)")
            if self.arm not in ("G-lin", "G-dir"):
                raise ValueError(f"graft arm must be G-lin|G-dir, got {self.arm!r}")
            if len(self.gdir_taps) != self.conv_kernel_size:
                raise ValueError("gdir_taps must have conv_kernel_size entries")
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be >= 0")

    @property
    def conv_state_len(self) -> int:
        """Positions of N_conv(u) the decode state carries: (kernel-1)*dilation = 9 on Flash-Next (§2.3)."""
        return (self.conv_kernel_size - 1) * self.conv_dilation

    def row_width(self, key: KeySpec) -> int:
        if self.ple_embed_dim % key.n_heads:
            raise ValueError(f"ple_embed_dim {self.ple_embed_dim} not divisible by n_heads {key.n_heads}")
        return self.ple_embed_dim // key.n_heads

    def fingerprint(self) -> str:
        return _sha(dict(schema="ModuleSpec/v1", **self.to_dict()))

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["gdir_taps"] = list(self.gdir_taps)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModuleSpec":
        d = dict(d)
        d["gdir_taps"] = tuple(d.get("gdir_taps", (0.0, 0.0, 0.0, 1.0)))
        return cls(**d)

    @classmethod
    def from_qwen4exp_config(cls, cfg: dict, ple_layer_index: int = 0) -> "ModuleSpec":
        """Native ModuleSpec from a qwen4_exp config dict (top-level or its `text_config`)."""
        t = cfg.get("text_config", cfg)
        ids = t.get("ple_layer_ids", [2])
        return cls(hidden_size=int(t["hidden_size"]), ple_embed_dim=int(t["ple_embed_dim"]),
                   hc_count=int(t["hc_count"]), conv_kernel_size=int(t.get("ple_conv_kernel_size", 4)),
                   conv_dilation=int(t.get("ngram_size", 3)), rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)),
                   layer_idx=int(ids[ple_layer_index]) - 1, kind="native", arm=None)


# ---------------------------------------------------------------------------------------------------- FormatSpec
@dataclass(frozen=True)
class FormatSpec:
    """KeySpec + TableSpec + ModuleSpec with one combined fingerprint (what patches and edit records carry)."""

    key: KeySpec
    table: TableSpec
    module: ModuleSpec

    def __post_init__(self) -> None:
        if self.module.row_width(self.key) != self.table.row_width:
            raise ValueError(f"row width mismatch: module gives {self.module.row_width(self.key)}, "
                             f"table says {self.table.row_width}")

    @property
    def row_width(self) -> int:
        return self.table.row_width

    @property
    def padded_rows(self) -> int:
        return self.table.padded_rows(self.key)

    def fingerprint(self) -> str:
        return _sha(dict(kind="FormatSpec/v1", key=self.key.fingerprint(), table=self.table.fingerprint(),
                         module=self.module.fingerprint()))

    def to_dict(self) -> dict:
        return dict(format="ngram-memory/FormatSpec/v1", key=self.key.to_dict(), table=self.table.to_dict(),
                    module=self.module.to_dict(), fingerprint=self.fingerprint())

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "FormatSpec":
        f = cls(KeySpec.from_dict(d["key"]), TableSpec.from_dict(d["table"]), ModuleSpec.from_dict(d["module"]))
        if "fingerprint" in d and d["fingerprint"] != f.fingerprint():
            raise ValueError(f"fingerprint mismatch: stored {d['fingerprint']} vs computed {f.fingerprint()}")
        return f

    @classmethod
    def from_json(cls, s: str) -> "FormatSpec":
        return cls.from_dict(json.loads(s))


# -------------------------------------------------------------------------------------------------- constructors
def flash_next_key_spec(tokenizer_sha256: str | None = None) -> KeySpec:
    return KeySpec(vocab_size=FLASH_NEXT_VOCAB, eos_token_id=FLASH_NEXT_EOS, seed=DEFAULT_SEED, ple_layer_index=0,
                   ngram_size=3, heads_per_ngram=8, base=20_000_000, orders=(2, 3),
                   tokenizer_sha256=tokenizer_sha256)


def key_spec_from_qwen4exp_config(cfg: dict, ple_layer_index: int = 0,
                                  tokenizer_sha256: str | None = None) -> KeySpec:
    t = cfg.get("text_config", cfg)
    eos = t["eos_token_id"]
    eos = eos[0] if isinstance(eos, list) else eos
    seed = t.get("seed")
    return KeySpec(vocab_size=int(t["vocab_size"]), eos_token_id=int(eos),
                   seed=DEFAULT_SEED if seed is None else int(seed),  # shipping config omits seed -> q4c default
                   ple_layer_index=ple_layer_index, ngram_size=int(t.get("ngram_size", 3)),
                   heads_per_ngram=int(t.get("heads_per_ngram", 8)),
                   base=int(t.get("ngram_vocab_size_base", 20_000_000)), orders=(2, 3),
                   tokenizer_sha256=tokenizer_sha256)


def flash_next_format(config: dict | None = None, tokenizer_sha256: str | None = None) -> FormatSpec:
    """Flash-Next's native format. With `config` (the qwen4_exp config.json dict) values are read from it,
    otherwise the known values are used (identical; tests check both against the checkpoint buffers)."""
    if config is None:
        config = dict(vocab_size=FLASH_NEXT_VOCAB, eos_token_id=FLASH_NEXT_EOS, hidden_size=2560, hc_count=4,
                      ple_embed_dim=2560, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=8,
                      ngram_vocab_size_base=20_000_000, make_ngram_vocab_size_divisible_by=128,
                      ple_layer_ids=[2], split_ngram_parts=128, rms_norm_eps=1e-6)
    t = config.get("text_config", config)
    key = key_spec_from_qwen4exp_config(t, 0, tokenizer_sha256)
    module = ModuleSpec.from_qwen4exp_config(t)
    table = TableSpec(tensor_name=FLASH_NEXT_TENSOR, row_width=module.row_width(key),
                      pad_divisor=int(t.get("make_ngram_vocab_size_divisible_by", 128)),
                      n_shards=int(t.get("split_ngram_parts", 512)), shard_name_template=FLASH_NEXT_SHARD_TEMPLATE,
                      base_codec="bf16")
    return FormatSpec(key, table, module)


def graft_key_spec(vocab_size: int, eos_token_id: int, base: int = 20_000_000, orders: Sequence[int] = (2, 3),
                   seed: int = DEFAULT_SEED, tokenizer_sha256: str | None = None) -> KeySpec:
    return KeySpec(vocab_size=vocab_size, eos_token_id=eos_token_id, seed=seed, base=base, orders=tuple(orders),
                   ngram_size=max(3, max(orders)), tokenizer_sha256=tokenizer_sha256)


def qwen25_proxy_key_spec(orders: Sequence[int] = (2, 3)) -> KeySpec:
    """The Stage-1 0.5B experiment's keys: Qwen2.5 vocab, eos 151643, 16 primes from 200003."""
    return graft_key_spec(QWEN25_VOCAB, QWEN25_EOS, base=200_000, orders=orders)


def graft_format(hidden_size: int, key: KeySpec, *, arm: str = "G-lin", layer_idx: int = 1, value_seed: int = 0,
                 value_scale: float = 1.0, gdir_alpha: float = 1.0,
                 gdir_taps: Sequence[float] = (0.0, 0.0, 0.0, 1.0), row_width: int | None = None,
                 rms_norm_eps: float = 1e-6, tensor_name: str = "graft.ple.ple_embedding.ngram_embedding.weight",
                 pad_divisor: int = 128) -> FormatSpec:
    """Graft profile (§2.4): row width w = d/16 by default (independent of the number of orders, so the unigram
    extension has ple_embed_dim = 24w), hc_count 1, editor init."""
    w = row_width if row_width is not None else hidden_size // 16
    if row_width is None and hidden_size % 16:
        raise ValueError(f"hidden_size {hidden_size} not divisible by 16; pass row_width")
    module = ModuleSpec(hidden_size=hidden_size, ple_embed_dim=key.n_heads * w, hc_count=1, conv_kernel_size=4,
                        conv_dilation=key.ngram_size, rms_norm_eps=rms_norm_eps, layer_idx=layer_idx, kind="graft",
                        arm=arm, value_seed=value_seed, value_scale=value_scale, gdir_alpha=gdir_alpha,
                        gdir_taps=tuple(gdir_taps))
    table = TableSpec(tensor_name=tensor_name, row_width=w, pad_divisor=pad_divisor, n_shards=1,
                      shard_name_template=None, base_codec="none")
    return FormatSpec(key, table, module)


__all__ = [
    "KeySpec", "TableSpec", "ModuleSpec", "FormatSpec", "flash_next_key_spec", "flash_next_format",
    "key_spec_from_qwen4exp_config", "graft_key_spec", "qwen25_proxy_key_spec", "graft_format",
    "layer_multipliers", "nth_prime_after", "splitmix64", "FLASH_NEXT_VOCAB", "FLASH_NEXT_EOS", "QWEN25_VOCAB",
    "QWEN25_EOS",
]
