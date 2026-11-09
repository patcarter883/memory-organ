"""Independent (transformers-free) re-implementation of the Qwen4Exp / Qwen3.8-Flash-Next PLE n-gram hashing + layout.

Mirrors transformers/models/qwen4_exp/modeling_qwen4_exp.py:1033-1174 (hash) and conversion_mapping.py:2131-2140 (shards).
Pure python ints, so it can run inside any serving engine / offline editor without torch.
"""
import math
from dataclasses import dataclass, field

MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB
PRIME_1 = 10007


def splitmix64(x):
    x = (x + GAMMA) & MASK64
    x = ((x ^ (x >> 30)) * M1) & MASK64
    x = ((x ^ (x >> 27)) * M2) & MASK64
    return (x ^ (x >> 31)) & MASK64


def layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    half = max(1, (max_long // max(vocab_size, 1)) // 2)
    base = seed + PRIME_1 * ple_layer_index
    return [2 * (splitmix64((base + GAMMA * (i + 1)) & MASK64) % half) + 1 for i in range(ngram_size)]


def is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


@dataclass
class NGramTableFormat:
    vocab_size: int            # unigram (tokenizer) vocab size -> enters the multipliers
    eos_token_id: int          # n-gram history reset/pad token
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    divisible_by: int = 128
    seed: int = 1234
    ple_layer_index: int = 0   # index into ple_layer_ids (NOT the decoder layer index)
    embed_dim: int = 2560      # ple_embed_dim = concat of all head rows
    split_parts: int = 128     # checkpoint shards
    sizes: list = field(init=False)
    offsets: list = field(init=False)
    mult: list = field(init=False)

    def __post_init__(self):
        H = self.n_heads
        self.sizes = [nth_prime_after(self.ngram_vocab_size_base - 1, self.ple_layer_index * H + h + 1) for h in range(H)]
        self.offsets = [sum(self.sizes[:h]) for h in range(H)]
        self.mult = layer_multipliers(self.vocab_size, self.ngram_size, self.ple_layer_index, self.seed)

    @property
    def n_heads(self):
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def row_dim(self):
        return self.embed_dim // self.n_heads

    @property
    def total_rows(self):
        return sum(self.sizes)

    @property
    def padded_rows(self):
        return math.ceil(self.total_rows / self.divisible_by) * self.divisible_by

    @property
    def rows_per_shard(self):
        return math.ceil(self.padded_rows / self.split_parts)  # torch.chunk semantics

    def history(self, ids, pos):
        """tokens x_{pos-k}, k=0..n-1, with the eos-segment rule of _shift_right_ignore_eos (modeling:1132-1145).
        `ids` must already have padding replaced by eos (modeling:1467-1470)."""
        out = []
        seg_start = 0
        for j in range(pos - 1, -1, -1):          # last eos strictly before pos
            if ids[j] == self.eos_token_id:
                seg_start = j + 1
                break
        for k in range(self.ngram_size):
            src = pos - k
            out.append(ids[src] if (k == 0 or (src >= 0 and pos - seg_start >= k)) else self.eos_token_id)
        return out

    def rows_for(self, ids, pos):
        """[(head, global_row)] read at position `pos` of token list `ids` (prompt start = fresh history)."""
        x = self.history(ids, pos)
        res = []
        for n in range(2, self.ngram_size + 1):
            mixed = x[0] * self.mult[0]
            for k in range(1, n):
                mixed ^= x[k] * self.mult[k]
            for j in range(self.heads_per_ngram):
                h = (n - 2) * self.heads_per_ngram + j
                res.append((h, self.offsets[h] + mixed % self.sizes[h]))
        return res

    def rows_for_ngram(self, toks):
        """rows for an explicit n-gram given as (x_{t-n+1},...,x_t); only that order's 8 heads."""
        x = list(reversed(toks))
        mixed = x[0] * self.mult[0]
        for k in range(1, len(x)):
            mixed ^= x[k] * self.mult[k]
        n = len(x)
        return [((n - 2) * self.heads_per_ngram + j, self.offsets[(n - 2) * self.heads_per_ngram + j]
                 + mixed % self.sizes[(n - 2) * self.heads_per_ngram + j]) for j in range(self.heads_per_ngram)]

    def checkpoint_loc(self, global_row):
        """(shard index, local row) in `...ple_embedding.ngram_embedding.shard_{i}.weight`."""
        return divmod(global_row, self.rows_per_shard)
