# Vendored from minisglang-rdna4 python/minisgl/ple/hashing.py at commit df29117 (2026-11-06), unmodified below this header.
# Used only by tests (parity of ngram_memory.hashing against the serve-side implementation).
"""Qwen4-Exp n-gram hashing — the function that turns recent token ids into table row ids.

STATUS: **VERIFIED**, not inferred. Every line below is a transcription of
`transformers/models/qwen4_exp/modeling_qwen4_exp.py` @ `huggingface/transformers` `main`
(`Qwen4ExpTextNGramEmbedding.forward`, `_shift_right_ignore_eos`, `_build_layer_multipliers`,
`_splitmix64`, `_find_nth_prime_after`) — the reference implementation of the architecture, fetched
2026-09-03. Two independent checks tie the transcription to the SHIPPING checkpoint rather than to
the source alone (both are asserted in `tests/qwen4exp_ple_hash_test.py`):

  * `layer_multipliers` derived here from `(vocab_size=248320, ngram_size=3, ple_layer_index=0,
    seed=1234)` equals `RadixArk/Qwen3.8-Flash-Next-NVFP4`'s
    `...ple.ple_embedding.layer_multipliers` EXACTLY: `[23703573157769, 20109073645365,
    8052911324071]`.
  * the 16 head vocab sizes derived here (the first 16 primes > 20,000,000 - 1) equal the
    checkpoint's `ngram_heads_vocab_sizes`, and their prefix sum equals `ngram_heads_offsets`.

WHY THIS FILE IS SO CAREFUL
---------------------------
Nothing here can fail loudly. A wrong multiplier, a wrong shift, a wrong head->n-gram assignment or
a wrong EOS fill still produces a valid row id in the right head's band, so the model reads a real
embedding — just the wrong one, for every token, forever. The only symptom is quality. Hence: one
named function per piece of the arithmetic, the derivation kept next to the checkpoint value it must
reproduce, and a test that asserts the equality instead of trusting it.

THE ARITHMETIC
--------------
`ngram_size = 3`, `heads_per_ngram = 8` -> `ngram_heads = (3 - 1) * 8 = 16` hash heads, which is
exactly the 16 bands `row_table.NgramHeads` addresses (embedding width 2560 = 16 x 160).

For a token at position `p` with history `t`:

    s_k[p] = t[p - k]  , or EOS when p - k falls before the current EOS-delimited segment  (k=0,1,2)
    mixed_n[p] = XOR_{k<n} ( s_k[p] * layer_multipliers[k] )      for n in {2, 3}
    head h in [0, 8)   uses mixed_2 ;  head h in [8, 16) uses mixed_3
    row[p, h] = offsets[h] + mixed_{n(h)}[p] mod vocab_sizes[h]

`layer_multipliers[k]` is odd and bounded by `(2**63 - 1) // vocab_size`, so `s_k * m_k` never
overflows int64 and `mixed_n` is always non-negative — `np.mod`/`torch.remainder` therefore agree,
and neither can produce a negative row id.

Everything here is numpy and torch-free: the gather it feeds is an NVMe/page-cache read on the host
(`weights/row_table.py`), so the hash belongs on the host too, next to it.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# --- splitmix64, verbatim from the reference (`_MASK64`, `_SPLITMIX_*`, `_PRIME_1`) -------------
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007

#: `Qwen4ExpTextConfig.seed` default. The shipping config.json does NOT set `seed`, so the whole
#: hash — every multiplier, hence every row id — rests on this default being right. It is asserted
#: against the checkpoint's own `layer_multipliers` tensor rather than trusted.
DEFAULT_NGRAM_SEED = 1234


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int = 0, seed: int = DEFAULT_NGRAM_SEED
) -> np.ndarray:
    """The `ngram_size` odd int64 multipliers the n-gram key is mixed with.

    The checkpoint SHIPS this as `...ple.ple_embedding.layer_multipliers`, and the model loads that
    tensor — this function exists so the loaded value can be CHECKED, and so a checkpoint that omits
    it can still be served. The two must agree; `Qwen4ExpNGramHasher.from_checkpoint_multipliers`
    is the one place that compares them.

    `multiplier_max = (2**63 - 1) // vocab_size` is what makes `token_id * multiplier` overflow-free
    in int64 for every legal token id — do not "simplify" the bound away.
    """
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(int(unigram_vocab_size), 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = int(seed) + _PRIME_1 * int(ple_layer_index)
    out = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(int(ngram_size))
    ]
    return np.array(out, dtype=np.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def build_head_vocab_sizes(
    ngram_vocab_size_base: int, ngram_heads: int, ple_layer_index: int = 0
) -> np.ndarray:
    """Per-head table band sizes: consecutive primes above `ngram_vocab_size_base - 1`.

    Head `h` of PLE layer `p` takes the `(p * ngram_heads + h + 1)`-th prime, so a model with two
    PLE layers gives them DISJOINT primes. Qwen3.8-Flash-Next has one PLE layer, so `ple_layer_index`
    is 0 and the 16 sizes are the first 16 primes > 19,999,999.

    Recomputing these is ~1 s of trial division, so the serving path reads the checkpoint's
    `ngram_heads_vocab_sizes` (via `row_table.NgramHeads`) and this is the checker.
    """
    sizes = [
        _find_nth_prime_after(int(ngram_vocab_size_base) - 1, ple_layer_index * ngram_heads + h + 1)
        for h in range(int(ngram_heads))
    ]
    return np.array(sizes, dtype=np.int64)


def shift_right_ignore_eos(tokens: np.ndarray, shift: int, eos_token_id: int) -> np.ndarray:
    """`token[p - shift]`, but never reaching back across an EOS boundary.

    `tokens` is 1-D (ONE sequence's token history, oldest first). Positions whose source falls
    before the start of the current EOS-delimited segment get `eos_token_id` instead — that is what
    stops an n-gram feature from spanning two concatenated documents (or two turns).

    Transcribed from `Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos`. Note the reference's
    `previous_eos` is the cummax SHIFTED RIGHT BY ONE, i.e. a token that IS the EOS still belongs to
    the segment that ends with it; the next token starts the new segment. Getting that off by one
    silently changes the fill for the first 2 tokens of every segment.
    """
    t = np.asarray(tokens, dtype=np.int64)
    if t.ndim != 1:
        raise ValueError(f"shift_right_ignore_eos expects one sequence (1-D), got {t.shape}")
    if shift == 0:
        return t
    n = t.size
    positions = np.arange(n, dtype=np.int64)
    eos_positions = np.where(t == eos_token_id, positions, np.int64(-1))
    previous_eos_inclusive = np.maximum.accumulate(eos_positions)
    previous_eos = np.concatenate([np.array([-1], dtype=np.int64), previous_eos_inclusive[:-1]])
    position_in_segment = positions - (previous_eos + 1)
    source_positions = positions - shift
    shifted = t[np.maximum(source_positions, 0)]
    valid = (position_in_segment >= shift) & (source_positions >= 0)
    return np.where(valid, shifted, np.int64(eos_token_id))


class Qwen4ExpNGramHasher:
    """Recent-token-ids -> per-head n-gram table row ids, for ONE sequence at a time.

    Stateless with respect to the sequence: the caller supplies the `context_len = ngram_size - 1`
    tokens that preceded this chunk (`PLEStateCache` keeps them per slot), so the same object serves
    every request and nothing has to be reset.
    """

    def __init__(
        self,
        *,
        ngram_size: int,
        heads_per_ngram: int,
        layer_multipliers: Sequence[int] | np.ndarray,
        eos_token_id: int,
    ) -> None:
        self.ngram_size = int(ngram_size)
        self.heads_per_ngram = int(heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.context_len = self.ngram_size - 1
        self.eos_token_id = int(eos_token_id)
        m = np.asarray(layer_multipliers, dtype=np.int64).reshape(-1)
        if m.size != self.ngram_size:
            raise ValueError(f"layer_multipliers has {m.size} entries, need ngram_size={self.ngram_size}")
        if not (m % 2 == 1).all():
            raise ValueError(
                f"layer_multipliers must all be ODD (2*x+1 by construction); got {m.tolist()}. An "
                f"even multiplier means the tensor is not the one this hash was built from."
            )
        self.layer_multipliers = m

    @classmethod
    def from_checkpoint_multipliers(
        cls,
        *,
        ngram_size: int,
        heads_per_ngram: int,
        eos_token_id: int,
        checkpoint_multipliers: Sequence[int] | np.ndarray | None,
        vocab_size: int,
        ple_layer_index: int = 0,
        seed: int = DEFAULT_NGRAM_SEED,
    ) -> "Qwen4ExpNGramHasher":
        """Prefer the checkpoint's `layer_multipliers`; CHECK it against the derivation; derive it
        when the checkpoint omits it.

        A disagreement raises. It means either the seed is not the default or the derivation is
        wrong, and both make every row id wrong — the one failure mode that has no other symptom.
        """
        derived = build_layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed)
        if checkpoint_multipliers is None:
            return cls(
                ngram_size=ngram_size,
                heads_per_ngram=heads_per_ngram,
                layer_multipliers=derived,
                eos_token_id=eos_token_id,
            )
        ckpt = np.asarray(checkpoint_multipliers, dtype=np.int64).reshape(-1)
        if not np.array_equal(ckpt, derived):
            raise ValueError(
                f"qwen4_exp n-gram layer_multipliers mismatch: checkpoint {ckpt.tolist()} vs "
                f"derived {derived.tolist()} from (vocab_size={vocab_size}, ngram_size={ngram_size}, "
                f"ple_layer_index={ple_layer_index}, seed={seed}). The checkpoint tensor is "
                f"authoritative and is what will be used, but the disagreement means one of those "
                f"four inputs is wrong somewhere else too — fix it before serving."
            )
        return cls(
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            layer_multipliers=ckpt,
            eos_token_id=eos_token_id,
        )

    # -- the hash ----------------------------------------------------------

    def mixed_ids(self, history: np.ndarray) -> np.ndarray:
        """(L,) token history -> (L, ngram_size - 1) int64 mixed keys, one column per n-gram order.

        Column `j` is the key for n-gram order `n = j + 2`. Exposed separately from `row_ids` so a
        test can pin the mixing without the head arithmetic on top of it.
        """
        h = np.asarray(history, dtype=np.int64)
        shifted = [shift_right_ignore_eos(h, k, self.eos_token_id) for k in range(self.ngram_size)]
        out = np.empty((h.size, self.ngram_size - 1), dtype=np.int64)
        for n in range(2, self.ngram_size + 1):
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, n):
                mixed = np.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            out[:, n - 2] = mixed
        return out

    def hashes(self, history: np.ndarray, n_new: int) -> np.ndarray:
        """(L,) history -> (n_new, ngram_heads) int64 hashes for the LAST `n_new` positions.

        `history` must be `previous_context ++ new_tokens`; only the tail is returned, exactly as
        the reference's `[:, -input_ids.shape[1]:]` slice does. Head `h` carries the key of n-gram
        order `h // heads_per_ngram + 2` — the head blocks are CONTIGUOUS and ordered by n, matching
        `ngram_heads_vocab_sizes`' own ordering.
        """
        mixed = self.mixed_ids(history)
        if n_new > mixed.shape[0]:
            raise ValueError(f"asked for {n_new} positions from a history of {mixed.shape[0]}")
        tail = mixed[mixed.shape[0] - n_new :]  # (n_new, ngram_size-1)
        return np.repeat(tail, self.heads_per_ngram, axis=1)  # (n_new, ngram_heads)

    def row_ids(self, heads, history: np.ndarray, n_new: int) -> np.ndarray:
        """(n_new, ngram_heads) GLOBAL table row ids. `heads` is a `row_table.NgramHeads`."""
        return heads.row_ids(self.hashes(history, n_new))


__all__ = [
    "DEFAULT_NGRAM_SEED",
    "Qwen4ExpNGramHasher",
    "build_head_vocab_sizes",
    "build_layer_multipliers",
    "shift_right_ignore_eos",
]
