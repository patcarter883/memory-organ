"""KVAdapter — the UNCOMPRESSED key→value control (upper bound) for the bind-only capacity ladder.

RESULTS.md §1/§2 cite an "uncompressed KV control → 1.000" that proved the naive store's M-wall is
its *compression*, not the task: a store that keeps every binding's key/value embedding verbatim has
no capacity mechanism to wall out. The original control was a one-off falsifier that never landed;
this is its reimplementation as `bind_msweep --store kv` (REPRODUCING.md §1 known-gap).

Mechanism: per doc, the M (key → value) mem-space embeddings are kept AS-IS (no slots, no recurrent
state — storage is exact by construction). The read is softmax attention from the QA query tokens
over the M stored keys → weighted value mix, then the SAME periphery every other adapter uses
(in_proj+LayerNorm in, readout_q attention-pool to K slots, out_proj, tied-unembed direct loss), so
the carry-vs-M comparison isolates the storage mechanism exactly as the bolt/pk comparison does.
Trainable surface: in_proj/norm, readout_q, out_proj — there are NO store parameters to train.

Single-token phrasings only (dict / natural / counterfactual / varied — the layouts with one value
token per binding); the multi-token ladder has its own controls.
"""
import torch
import torch.nn as nn


class KVAdapter(nn.Module):
    """Uncompressed per-doc KV memory behind the standard bind-adapter contract
    (.inject / .direct_logits, constructor positionally matching BoltAdapter/PKStoreAdapter)."""

    def __init__(self, embed_weight, base_hidden, mem_dim, heads, chunk, expansion, k):
        super().__init__()
        # heads/chunk/expansion accepted for constructor signature-compatibility; an exact store has
        # no capacity knobs — that is the point of the control.
        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)   # FROZEN, tied table
        self.in_proj = nn.Linear(base_hidden, mem_dim, bias=False)             # same in-path as bolt/pk
        self.norm = nn.LayerNorm(mem_dim)
        self.readout_q = nn.Parameter(torch.randn(k, mem_dim) * 0.02)          # same K-slot pooling
        self.out_proj = nn.Linear(mem_dim, base_hidden, bias=False)
        self.register_buffer("unembed", embed_weight.t().contiguous())         # tied readout
        self.mem_dim = mem_dim
        self.k = k
        self.builder = None

    def set_builder(self, builder):
        self.builder = builder
        return self

    def _e(self, ids):
        return self.norm(self.in_proj(self.embed(ids).float()))

    def _kv_positions(self):
        """Absolute (key_pos list, val_pos list) of the M bindings, from the builder's layout —
        the same offsets the pk adapter reads (constant bind_len, or per-binding for varied)."""
        b = self.builder
        assert b is not None, "KVAdapter.set_builder(builder) must be called before inject"
        assert not getattr(b, "multitoken", False), "KV control is single-token only"
        hstart = len(b.bos) + len(b.header)
        if getattr(b, "phrasing", None) == "varied":
            return b.binding_positions(hstart)
        koff = getattr(b, "key_off", 0)
        voff = b.val_off if hasattr(b, "val_off") else (1 + len(b.colon))
        keys = [hstart + m * b.bind_len + koff for m in range(b.M)]
        vals = [hstart + m * b.bind_len + voff for m in range(b.M)]
        return keys, vals

    def retrieve(self, emb, qa_start, answer_pos):
        """QA-query attention over the M verbatim (key, value) embeds.
        -> (retrieved [B,Lq,mem_dim], attn [B,Lq,M]). Exposed for the retrieval-correctness test."""
        kpos, vpos = self._kv_positions()
        keys, vals = emb[:, kpos], emb[:, vpos]                      # [B,M,mem_dim] — stored VERBATIM
        q = emb[:, qa_start:answer_pos]                              # leak-free query tokens
        attn = torch.softmax(q @ keys.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)
        return attn @ vals, attn

    def inject(self, ids, seg_len, qa_start, answer_pos, carry=True):
        """-> K prefix vectors [B,K,base_hidden]. carry=False reads nothing (the ablated floor:
        the retrieved mix is zero, so the prefix is exactly out_proj(0) = 0)."""
        emb = self._e(ids)
        B = emb.shape[0]
        if carry:
            retrieved, _ = self.retrieve(emb, qa_start, answer_pos)
        else:
            retrieved = emb.new_zeros(B, answer_pos - qa_start, self.mem_dim)
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)
        attn = torch.softmax(pq @ retrieved.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)
        return self.out_proj(attn @ retrieved)                       # [B,K,base_hidden]

    def direct_logits(self, prefix):
        return prefix.mean(dim=1) @ self.unembed                    # [B,vocab], single-token readout
