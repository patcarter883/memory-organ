"""PKStoreAdapter — drive the product-key sparse store (pk_store.ProductKeyStore) as a drop-in
BoltAdapter replacement for the bind-only M-ceiling ladder (bind_msweep.py), DECOUPLED from the
canonical-Z committee hub.

WHY (the port): the v0 DeepMemory store (recall_boltA.BoltAdapter) has an associative-capacity wall
between M=8 (carry ~0.84) and M=16 (collapse to chance). ProductKeyStore was built as the
higher-capacity alternative (sparse top-k product-key addressing + multi-head reads) — but it was
coupled to the canonical-Z hub: it operated in d_hub=4096, with sub-codebooks SEEDED from the atlas
anchor keys (anchor_keys=Z) and trained by train_mem_canonical.py. This adapter ports it OFF the hub.

WHAT WAS DECOUPLED FROM THE HUB:
  1. anchor_keys = None  -> product-key sub-codebooks are RANDOM-init (no committee-Z anchoring), and
     they train freely as ordinary parameters (no fidelity pull, which lived in train_mem_canonical).
  2. d_hub = mem_dim  -> the store operates in the adapter's own mem_dim space (default 512), NOT the
     committee's d=4096 hub. We project base-1 (Qwen3.5-4B) embeddings into mem_dim with a learned
     in-projection (in_proj + LayerNorm, exactly as BoltAdapter does — the diagnostic-D-necessary
     input normalization), so the store addresses in a base-derived, hub-free geometry.
  3. The MAG tap / GatedMemoryTap is NOT used: this is the BIND-ONLY ladder (Stage-1 only). Training is
     the SAME direct tied-unembed loss BoltAdapter uses (no base in the loop), so the only thing that
     changes vs the BoltAdapter baseline is the MEMORY MECHANISM (DeepMemory linear-attn state -> a
     product-key sparse store). Everything else (in_proj/norm, readout pooling, out_proj, tied
     unembed, direct loss, eval) is held byte-for-byte identical so the carry-vs-M comparison is clean.

BIND-TASK -> PK_STORE API MAPPING (mirrors train_mem_canonical.CanonicalMemoryFrontEnd, hub stripped):
  - per doc, the M (name->cargo) bindings are written into a FRESH per-episode value bank:
        key   = cargo-token mem embed   (dict layout: cargo at offset 0 of each binding)
        value = name-token  mem embed   (name at offset 1+len(colon) of each binding)
    store.write(V, keys, vals) does the error-correcting delta-update into the top-k addressed slots.
  - the read query = the QA cargo tokens (leak-free, strictly before the answer): store.read(V, q)
    runs the multi-head product-key read; we attn-pool the per-token read into K learned slots
    (readout_q), then out_proj -> K base-embedding-space prefix vectors. direct_logits = tied unembed.

CONTRACT (what bind_msweep/recall_mag/eval_direct require): an nn.Module exposing
    .inject(ids, seg_len, qa_start, answer_pos, carry=True) -> [B, K, base_hidden]
    .direct_logits(prefix) -> [B, vocab]
This adapter satisfies exactly that, so recall_mag.bind_adapter and recall_boltA.eval_direct drive it
UNCHANGED. The bind block positions need the builder (bind_len/colon/header/bos/M); we hold a builder
ref (set_builder, called by bind_msweep right after construction). The 'carry' flag selects writing the
M bindings (carry=True) vs an empty store (carry=False = the ablated floor), mirroring BoltAdapter's
carry/ablated branch.
"""
import os
import sys

import torch
import torch.nn as nn

# flat package: sibling imports resolve relatively when imported as cam.X and fall back to a
# path-hacked absolute import when run as a file.
try:
    from .pk_store import ProductKeyStore
except ImportError:
    if __package__:  # real ImportError inside a sibling, not "run as a file" — don't mask it
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from pk_store import ProductKeyStore  # noqa: E402


class PKStoreAdapter(nn.Module):
    """Product-key sparse store driven as a BoltAdapter-compatible bind adapter, hub-free.

    Constructor signature MATCHES BoltAdapter(embed_weight, base_hidden, mem_dim, heads, chunk,
    expansion, k) positionally so bind_msweep.bind_at_M can construct it the same way. 'heads' maps to
    the store's read-head count (n_heads); 'chunk'/'expansion' are accepted for signature-compatibility
    and unused (they were DeepMemory linear-attn knobs with no product-key analog). Extra pk-only knobs
    (n_sub/topk/sub_topk) take train_mem_canonical's smoke defaults and can be overridden by kwarg."""

    def __init__(self, embed_weight, base_hidden, mem_dim, heads, chunk, expansion, k,
                 n_sub=32, topk=8, sub_topk=4, write_beta=1.0, addr_sup_weight=0.0, read_heads=None,
                 mt_value="mean", mt_positions=4, readout="linear", dec_layers=2, dec_heads=4,
                 dec_dim=256, perpos_key="additive"):
        super().__init__()
        assert mem_dim % 2 == 0, "product-key splits the mem query in half"
        # MULTI-TOKEN value mode: 'mean' = store ONE value = mean of the K cargo-token embeds (the
        # faithful single-store-slot extension; the K-slot sequence head must DISENTANGLE the phrase
        # from one mixed value). 'perpos' = store the K cargo tokens as K SEPARATE position-tagged
        # associations (key = name + learned pos_tag[t], value = cargo_t); read slot t with the same
        # pos-tagged query so each answer position retrieves its OWN value. perpos is the natural
        # sequence-store fix for the mean bottleneck; 'mean' stays the default (byte-preserved).
        self.mt_value = mt_value
        self.mt_positions = mt_positions
        self.pos_tag = nn.Parameter(torch.randn(mt_positions, mem_dim) * 0.02)  # learned position codes
        # ---- PER-POSITION KEY conditioning (Thrust 1 #3) ---------------------------------------
        # exp#2's diagnosis: the per-position ADDRESS never resolved (per-position addr-sup InfoNCE
        # stalled at ~1.7 vs ~0.05 for the binding-level) because a learned ADDITIVE position code on a
        # SHARED name key (key_p = name + pos_tag[t]) is too weak a discriminator for the product-key
        # top-k addressing to separate position 0 from position 1 of the SAME binding. perpos_key picks
        # how the position is folded into the per-position key/query:
        #   'additive' (default, byte-preserved): key_p = name + pos_tag[t]. The weak code exp#2 ruled out.
        #   'gated':    key_p = name * pos_gate[t] + pos_tag[t]. A learned per-position ELEMENTWISE gate
        #               (init 1) before the additive shift, so each HALF the sub-codebooks score is already
        #               position-scaled — a position-distinct key without a full projection.
        #   'codebook': key_p = pos_proj[t](name) + pos_tag[t]. A learned per-position LINEAR map (mem_dim
        #               -> mem_dim, init ~identity) projects the name into a position-specific subspace
        #               BEFORE the product-key half-split, so position 0 and position 1 address into
        #               genuinely different codebook regions (the strongest separation). This is the
        #               "position-conditioned sub-codebook" realized as a per-position key transform — it
        #               makes EACH half-query the shared sub-codebooks score position-distinct, which the
        #               additive code could not.
        #   'disjoint' (Thrust 1 #4): each answer position t owns an ENTIRELY SEPARATE ProductKeyStore
        #               (its own codebook pair, write/read/head projections, value bank). Position 0 and
        #               position 1 do top-k addressing over DISJOINT codebook banks — not a shared codebook
        #               reached via a per-position key transform (exp#3 codebook only HALF-collapsed the
        #               InfoNCE to ~0.75). With physically separate sub-codebooks, position t's retrieval
        #               CANNOT be contaminated by the other position's slots, so the per-position address
        #               can fully resolve. The per-position KEY into store t is L2-normalized name +
        #               pos_tag[t] (the L2-norm before the product-key half-split sharpens the codebook
        #               match; the additive tag still distinguishes the small mt_positions banks).
        # additive/gated/codebook reuse the SHARED product-key store (codebooks/heads) — only the
        # per-position KEY that enters the store changes. disjoint instead routes each position to its OWN
        # store. The per-position key transform (additive/gated/codebook) or the store selection (disjoint)
        # is applied IDENTICALLY at WRITE (key_p) and READ (query_p) and to the per-position addr-sup
        # write-address targets, so train/eval/supervision stay consistent.
        self.perpos_key = perpos_key
        if perpos_key == "gated":
            self.pos_gate = nn.Parameter(torch.ones(mt_positions, mem_dim))      # elementwise, init 1
        elif perpos_key == "codebook":
            # per-position linear key map, init near identity so step-0 ~= additive (clean warm start).
            self.pos_proj = nn.Parameter(
                torch.eye(mem_dim).unsqueeze(0).repeat(mt_positions, 1, 1)
                + torch.randn(mt_positions, mem_dim, mem_dim) * 0.02)            # [P, mem_dim, mem_dim]
        self.embed = nn.Embedding.from_pretrained(embed_weight, freeze=True)   # FROZEN, tied table
        # base-embed -> mem_dim: the hub-free "into-store" projection. SAME in_proj+norm BoltAdapter
        # uses (the diagnostic-D-necessary input normalization). key/value/query share this in-proj.
        self.in_proj = nn.Linear(base_hidden, mem_dim, bias=False)
        self.norm = nn.LayerNorm(mem_dim)
        # the product-key store, in mem_dim space, with NO hub anchoring (anchor_keys=None).
        # read_heads overrides the ladder's --heads as the store's read-head count (the pk store is
        # designed for more retrieval-mode heads than the 4 the bolt path reuses); default = heads.
        nh = read_heads if read_heads is not None else heads
        self.store = ProductKeyStore(mem_dim, n_sub=n_sub, topk=topk, sub_topk=sub_topk,
                                     n_heads=nh, anchor_keys=None, write_beta=write_beta)
        # DISJOINT per-position sub-codebooks (Thrust 1 #4): one INDEPENDENT ProductKeyStore per answer
        # position, so position t addresses over its OWN codebook bank (no shared-codebook contamination).
        # self.store above stays constructed (byte-preserved for additive/gated/codebook + single-token),
        # but in disjoint mode all write/read/addr-sup route through self.stores[t] instead.
        if perpos_key == "disjoint":
            self.stores = nn.ModuleList([
                ProductKeyStore(mem_dim, n_sub=n_sub, topk=topk, sub_topk=sub_topk,
                                n_heads=nh, anchor_keys=None, write_beta=write_beta)
                for _ in range(mt_positions)])
        # STRONGER SUBJECT-KEY ENCODER (Track 4 #19 incr#2): the pooled subject key (CAM_POOLED_SUBJ_KEY)
        # mean-pools the subject-span token embeds, which weights every token equally — so multi-token NAMES
        # that share common tokens (first names, particles) don't separate. A learned ATTENTION pool weights
        # the discriminative tokens instead. subj_pool_q attends over the span; used symmetrically at WRITE
        # (key) and READ (query) so addressing stays consistent. Opt-in via CAM_LEARNED_KEY_POOL=1 (else the
        # mean, byte-preserved).
        # MULTI-VECTOR KEYS (#19 capacity probe -> ceiling is ADDRESSING): CAM_KEY_HEADS=H>1 gives each
        # subject H DISTINCT learned key vectors (H attention heads over the span) written to H store slots
        # pointing at the SAME value, so a subject occupies a richer, more separable address (any head can
        # retrieve it; the joint address separates similar names the single pool collapses). subj_pool_q is
        # [H, mem_dim]; head 0 is the addr-supervised primary, heads 1..H-1 train via the edit loss. H read
        # at init so the param is in the state dict. Always constructed (H=1 -> byte-identical to incr#2).
        self.n_key_heads = max(1, int(os.environ.get("CAM_KEY_HEADS", "1")))
        self.subj_pool_q = nn.Parameter(torch.randn(self.n_key_heads, mem_dim) * 0.02)
        # DECOUPLED MEMORY-BASE PROBE (#19): key/query the store from a SEPARATE encoder (GTE-ModernColBERT)
        # instead of the served model's input embeddings. CAM_GTE_KEYS=1 loads a {subject_tids -> [gte_dim]}
        # table (CAM_GTE_KEYS_FILE) and a learned gte_proj into mem_dim; the KEY/QUERY addressing then comes
        # from GTE, VALUES stay in base space. Tests whether a purpose-built retrieval encoder addresses more
        # separably than the input embeddings. (Single-vector/pooled mode — GTE's weak mode; MaxSim needs the
        # late-interaction store redesign.)
        self._gte_keys = None
        if os.environ.get("CAM_GTE_KEYS") == "1":
            import pickle as _pk
            _f = os.environ.get("CAM_GTE_KEYS_FILE", "")
            with open(_f, "rb") as _fh:
                raw = _pk.load(_fh)
            self._gte_keys = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
            gte_dim = next(iter(self._gte_keys.values())).shape[0]
            self.gte_proj = nn.Linear(gte_dim, mem_dim, bias=False)
            self._gte_dim = gte_dim
        # pool the per-token store read into K learned slots (mirrors BoltAdapter.readout_q), then
        # out_proj back to base-embedding space for the tied-unembed direct readout.
        self.readout_q = nn.Parameter(torch.randn(k, mem_dim) * 0.02)
        self.out_proj = nn.Linear(mem_dim, base_hidden, bias=False)
        self.mem_dim = mem_dim
        self.k = k
        self.register_buffer("unembed", embed_weight.t().contiguous())        # [base_hidden, vocab] tied
        # ---- READOUT HEAD selection (Thrust 1 #1) ----------------------------------------------
        # 'linear' (default, byte-preserved): direct_logits maps slot t -> answer token t in ONE
        # projection (prefix[:, :Kc] @ unembed). This cannot express a token SEQUENCE — each answer
        # position is an independent argmax over an independently-pooled slot, so a 2-token answer
        # collapses to ~one recoverable token (the Stage-1 ceiling we are attacking).
        # 'decoder': a tiny AR transformer-decoder head. The K retrieved store slots (the prefix,
        # base_hidden) are the cross-attention MEMORY; the decoder reads the answer token sequence
        # AUTOREGRESSIVELY (teacher-forced gold-token embeds in, causal self-attn over answer
        # positions + cross-attn to the K slots) -> [B,Kc,vocab]. This gives the value path real
        # SEQUENCE capacity instead of K independent projections.
        self.readout = readout
        self.vocab = embed_weight.shape[0]
        if readout == "decoder":
            self.dec_dim = dec_dim
            # project the K base_hidden store-slot prefix vectors -> decoder model dim (cross memory)
            self.dec_mem_proj = nn.Linear(base_hidden, dec_dim, bias=False)
            self.dec_mem_norm = nn.LayerNorm(dec_dim)
            # answer-token embeddings into the decoder: reuse the FROZEN tied table (base_hidden) and
            # project to dec_dim. A learned BOS starts the autoregressive sequence.
            self.dec_tok_proj = nn.Linear(base_hidden, dec_dim, bias=False)
            self.dec_bos = nn.Parameter(torch.randn(dec_dim) * 0.02)
            self.dec_pos = nn.Parameter(torch.randn(mt_positions, dec_dim) * 0.02)  # answer-pos codes
            layer = nn.TransformerDecoderLayer(d_model=dec_dim, nhead=dec_heads,
                                               dim_feedforward=4 * dec_dim, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, num_layers=dec_layers)
            self.dec_out_norm = nn.LayerNorm(dec_dim)
            self.dec_logit = nn.Linear(dec_dim, self.vocab, bias=False)
        self.builder = None   # set by bind_msweep right after construction (bind-block positions)
        # ADDRESSING SUPERVISION (the dropped write->read alignment loss, ported off the hub into
        # mem_dim space). addr_sup_weight>0 turns it on; inject() then stashes the loss in
        # self._last_aux_loss for bind_adapter to add to the direct CE. See aux_loss() below.
        self.addr_sup_weight = float(addr_sup_weight)
        self._last_aux_loss = None
        # per-write cache (set in _write_episode, consumed in inject when carry=True): the queried-
        # binding cargo token ids and the hub keys/vals needed for the InfoNCE address/value targets.
        self._cargo_ids = None      # [B,M] binding cargo token ids (to find which binding was queried)

    def set_builder(self, builder):
        """bind_msweep hands us the DocBuilder so inject can locate the per-doc binding tokens."""
        self.builder = builder
        return self

    def _e(self, ids):
        """frozen embed -> in_proj -> LayerNorm: base ids -> [B,L,mem_dim] normalized mem embeds."""
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(self.embed.weight.shape[0])):
            # a negative/OOB token id (e.g. the -1 multi-token sentinel) is an out-of-bounds embedding
            # gather -> an unrecoverable HSA_STATUS_ERROR_EXCEPTION 0x1016 GPU abort. Fail loud in Python.
            raise ValueError(f"PKStoreAdapter._e: token id out of range [0,{self.embed.weight.shape[0]}) "
                             f"(min={int(ids.min())}, max={int(ids.max())}) — likely an unresolved "
                             f"multi-token object (new_tid=-1); resolve to new_ids[0] before writing.")
        return self.norm(self.in_proj(self.embed(ids).float()))

    def _e_val(self, ids):
        """VALUE-path embed. Values are the object tokens the store must ROUND-TRIP (mt-recon delivery),
        NOT addressing keys. CAM_MT_VALUE_NO_NORM=1 drops the shared LayerNorm on the value branch so the
        stored code keeps its per-token magnitude/direction spread instead of being flattened onto the
        mem_dim unit sphere — the value-capacity lever for distinguishing ~100k token identities. Applied
        IDENTICALLY at train (mt-recon) and delivery (persistent write) so the round-trip stays consistent.
        Default (env unset) is byte-identical to _e (keeps the norm)."""
        if os.environ.get("CAM_MT_VALUE_NO_NORM") != "1":
            return self._e(ids)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(self.embed.weight.shape[0])):
            raise ValueError(f"PKStoreAdapter._e_val: token id out of range "
                             f"[0,{self.embed.weight.shape[0]}) (min={int(ids.min())}, max={int(ids.max())}).")
        return self.in_proj(self.embed(ids).float())

    def _pool_subject(self, span, keepdim=False):
        """Pool a subject-span embed [B,L,mem_dim] -> subject key/query vector(s). Returns [B,mem_dim]
        (or [B,1,mem_dim] if keepdim) for the single-key path, OR [B,H,mem_dim] when CAM_KEY_HEADS=H>1
        (multi-vector keys — H distinct learned attention heads). Used symmetrically at write-key and
        read-query for consistent addressing (#19 incr#2/multi-vector). CAM_LEARNED_KEY_POOL off -> the
        uniform mean, single vector (byte-preserved)."""
        if os.environ.get("CAM_LEARNED_KEY_POOL") == "1":
            # subj_pool_q [H,mem] -> per-head attention weights [B,H,L] -> H pooled vectors [B,H,mem]
            scores = torch.einsum("blm,hm->bhl", span, self.subj_pool_q) / (self.mem_dim ** 0.5)  # [B,H,L]
            w = torch.softmax(scores, dim=-1)                                             # [B,H,L]
            pooled = torch.einsum("bhl,blm->bhm", w, span)                                # [B,H,mem]
            if self.n_key_heads > 1:
                return pooled                                                             # [B,H,mem] multi-vector
            pooled = pooled[:, 0]                                                          # [B,mem] single head
        else:
            pooled = span.mean(dim=1)                                                      # [B,mem]
        return pooled.unsqueeze(1) if keepdim else pooled

    def _maxsim_reduce(self, read):
        """MULTI-VECTOR READ RULE (Phase B, #19). read [B,H,mem] = the H per-head store retrievals for one
        subject's H key vectors. Phase A showed pooling the H reads collides at scale; ColBERT's MaxSim
        instead takes the SINGLE best-matching key. Soft-MaxSim: weight heads by retrieval STRENGTH (the
        read-value norm — a clean, confident match has larger norm; a collided/averaged one is smaller) at
        a low temperature so the discriminative head dominates. Differentiable (train + eval consistent).
        No-op unless CAM_KEY_MAXSIM=1 and H>1."""
        if read.shape[1] <= 1 or os.environ.get("CAM_KEY_MAXSIM") != "1":
            return read
        temp = float(os.environ.get("CAM_KEY_MAXSIM_TEMP", "0.1"))
        w = torch.softmax(read.norm(dim=-1) / temp, dim=1)          # [B,H] concentrate on the best head
        return (w.unsqueeze(-1) * read).sum(dim=1, keepdim=True)    # [B,1,mem]

    def _gte_key(self, ids_span):
        """[B,L] subject token-ids -> [B,mem_dim] key/query from the precomputed GTE table + learned proj +
        norm. Per-row tuple lookup; a subject not in the table (padding/OOV) falls back to a zero vector."""
        dev = self.gte_proj.weight.device
        rows = []
        for b in range(ids_span.shape[0]):
            key = tuple(int(x) for x in ids_span[b].tolist())
            v = self._gte_keys.get(key)
            rows.append(v if v is not None else torch.zeros(self._gte_dim))
        g = torch.stack(rows).to(dev)                          # [B, gte_dim]
        return self.norm(self.gte_proj(g))                     # [B, mem_dim]

    def _pos_key(self, name_key, t):
        """Fold answer position t into the name key -> the per-position store key/query [..., mem_dim].
        name_key: [..., mem_dim] (the binding's name-token mem embed). Applied IDENTICALLY at write
        (key) and read (query) and to the addr-sup write-address targets. perpos_key selects strength:
        additive (weak, byte-preserved) / gated (elementwise scale) / codebook (per-position linear map).
        The codebook map is einsum'd over the trailing mem_dim so it works for [B,mem_dim] and any
        leading-batch shape."""
        if self.perpos_key == "gated":
            return name_key * self.pos_gate[t] + self.pos_tag[t]
        if self.perpos_key == "codebook":
            proj = torch.einsum("...d,ed->...e", name_key, self.pos_proj[t])     # [..., mem_dim]
            return proj + self.pos_tag[t]
        if self.perpos_key == "disjoint":
            # L2-normalize the name before the product-key half-split (sharpens the codebook match),
            # then add the position tag. Each position addresses its OWN store (self.stores[t]), so the
            # tag only has to distinguish the mt_positions banks, not fully separate the address.
            import torch.nn.functional as F
            return F.normalize(name_key, dim=-1) + self.pos_tag[t]
        return name_key + self.pos_tag[t]                                        # additive (default)

    # ---- write the M (key->value) bindings of each doc into a fresh store ----
    def _write_episode(self, mem_emb, ids=None):
        """mem_emb:[B,S,mem_dim] -> value bank V:[B,N,mem_dim] holding the M bindings.

        single-token dict layout per binding: [cargo, ':', name, '\\n'] -> KEY=cargo (offset 0),
        VALUE=name (offset 1+len(colon)). multi-token (role-swapped) layout: [name, ':', <K cargo>, '\\n']
        -> KEY=name (offset 0), VALUE = MEAN of the K cargo-token mem embeds (offsets 1+len(colon)..+K).
        The store value slot is a single mem_dim vector either way; the multi-token answer phrase is
        DECODED from the K-slot pooled read (direct_logits returns [B,K,vocab]), the value just has to
        carry enough of the phrase's content to address+retrieve it. (mirrors
        CanonicalMemoryFrontEnd.write_episode.) When addressing supervision is on (+ids) also caches the
        per-binding write keys/vals and KEY token ids so inject() can build the write->read InfoNCE."""
        b = self.builder
        assert b is not None, "PKStoreAdapter.set_builder(builder) must be called before inject"
        B = mem_emb.shape[0]
        hstart = len(b.bos) + len(b.header)
        mt = getattr(b, "multitoken", False)
        K = getattr(b, "cargo_tokens", 1)
        # KEY / VALUE offsets WITHIN a binding block. Builders now declare these explicitly (key_off,
        # val_off); fall back to the historic dict/manifest values (key@0, value@1+len(colon)) for any
        # older builder that predates the attributes, so this path stays byte-identical for dict.
        koff = getattr(b, "key_off", 0)
        # value block offset within a binding. Builders now declare val_off; fall back to the historic
        # dict/manifest value (1+len(colon)) ONLY for a builder that predates it (guard: natural phrasing
        # has no .colon, so evaluate the fallback lazily rather than in getattr's default expression).
        voff = b.val_off if hasattr(b, "val_off") else (1 + len(b.colon))
        keys_pos = []
        val_pos_list = []                                # per-binding list of value token offsets
        if getattr(b, "phrasing", None) in ("varied", "counterfactual_multi"):
            # VARIED / MULTI-RELATION counterfactual: bind blocks have DIFFERENT lengths (per-relation),
            # so the constant bind_len arithmetic does not hold. The builder hands us the exact per-binding
            # KEY (subject) and VALUE (object) absolute positions; single-token only (mt is False here).
            assert not mt, "per-binding-position phrasings are single-token only"
            vk_pos, vv_pos = b.binding_positions(hstart)
            keys_pos = list(vk_pos)
            val_pos_list = [[vp] for vp in vv_pos]
        else:
            for m in range(b.M):
                base = hstart + m * b.bind_len
                keys_pos.append(base + koff)             # KEY token (cargo single / name multi / subj nat)
                if mt:
                    val_pos_list.append(list(range(base + voff, base + voff + K)))  # K cargo tokens
                else:
                    val_pos_list.append([base + voff])   # single name token
        # RICHER SUBJECT KEY (N-scaling key-separation fix): the default key is one token (subject LAST
        # token). For multi-token subjects that don't separate at high M, POOL the full subject span
        # (mean) into the key — more separable, so bind carry recovers. Opt-in via CAM_POOLED_SUBJ_KEY=1.
        keys_multi = None                               # [B,M,H,mem] when multi-vector keys (H>1), else None
        if self._gte_keys is not None and hasattr(b, "binding_key_spans") and ids is not None:
            # DECOUPLED: key each binding from GTE (subject token-ids -> GTE table -> proj), not base embeds.
            spans = b.binding_key_spans(hstart)
            keys = torch.stack([self._gte_key(ids[:, s:s + L]) for (s, L) in spans], dim=1)  # [B,M,mem]
        elif os.environ.get("CAM_POOLED_SUBJ_KEY") == "1" and hasattr(b, "binding_key_spans"):
            spans = b.binding_key_spans(hstart)
            # mean, or a learned attention pool (CAM_LEARNED_KEY_POOL) — the stronger multi-token key encoder.
            pooled = [self._pool_subject(mem_emb[:, s:s + L]) for (s, L) in spans]   # each [B,mem] or [B,H,mem]
            multi = self.n_key_heads > 1 and os.environ.get("CAM_LEARNED_KEY_POOL") == "1"
            if multi:
                keys_multi = torch.stack(pooled, dim=1)  # [B,M,H,mem] — the H per-subject key vectors
                keys = keys_multi[:, :, 0]               # [B,M,mem] PRIMARY head (addr-supervised)
            else:
                keys = torch.stack(pooled, dim=1)        # [B,M,mem_dim]
        else:
            keys = mem_emb[:, keys_pos]                 # [B,M,mem_dim] (name in multi, cargo in single)
        if mt and self.mt_value == "perpos" and self.perpos_key == "disjoint":
            # DISJOINT (Thrust 1 #4): position t writes its M associations into ITS OWN store
            # self.stores[t]. For binding m, position t: key = _pos_key(name_m, t) [L2(name)+pos_tag[t]],
            # value = cargo_{m,t}. -> a SEPARATE bank per position over a SEPARATE codebook, so position 0
            # and position 1 address disjoint slot spaces (no shared-codebook contamination).
            Vs = []
            wk_pp, wv_pp = [], []
            want = self.addr_sup_weight > 0 and ids is not None
            for t in range(K):
                store_t = self.stores[t]
                keys_t = torch.stack([self._pos_key(keys[:, m], t) for m in range(b.M)], dim=1)  # [B,M,d]
                vals_t = torch.stack([mem_emb[:, val_pos_list[m][t]] for m in range(b.M)], dim=1)  # [B,M,d]
                Vt = store_t.init_state(B, mem_emb.device, dtype=torch.float32)
                Vs.append(store_t.write(Vt, keys_t, vals_t))
                if want:
                    wkt, wvt = store_t.write_addr_val(keys_t, vals_t)   # [B,M,d] per-position targets
                    wk_pp.append(wkt)
                    wv_pp.append(wvt)
            if want:
                # per-position addr-sup targets kept POSITION-MAJOR as a list of [B,M,d]; the read-query at
                # position t supervises against store t's OWN M write-addresses (the negatives are the M-1
                # OTHER bindings of the SAME position — disjoint stores make cross-position negatives
                # meaningless since each position addresses a separate codebook).
                self._wk_pp_dj, self._wv_pp_dj = wk_pp, wv_pp
                self._cargo_ids = ids[:, keys_pos]
                # binding-level targets (unused in disjoint read, kept None-safe)
                self._wk = self._wv = None
            return Vs
        V = self.store.init_state(B, mem_emb.device, dtype=torch.float32)
        if mt and self.mt_value == "perpos":
            # PER-POSITION: store K SEPARATE associations per binding. For binding m, position t:
            # key = name_m + pos_tag[t], value = cargo_{m,t}. -> M*K associations. The read (memory_bank)
            # queries name_q + pos_tag[t] for each answer slot t, so each slot retrieves its OWN token.
            pk_keys, pk_vals = [], []
            for m in range(b.M):
                vp = val_pos_list[m]                    # K value token positions
                for t in range(K):
                    pk_keys.append(self._pos_key(keys[:, m], t))     # [B,mem_dim] per-position key
                    pk_vals.append(mem_emb[:, vp[t]])                # [B,mem_dim] the t-th cargo token
            keys_w = torch.stack(pk_keys, dim=1)        # [B,M*K,mem_dim]
            vals_w = torch.stack(pk_vals, dim=1)
        elif mt:
            # MEAN: VALUE = mean over the K cargo-token mem embeds per binding -> [B,M,mem_dim]
            keys_w = keys
            vals_w = torch.stack([mem_emb[:, vp].mean(dim=1) for vp in val_pos_list], dim=1)
        else:
            keys_w = keys
            vals_w = mem_emb[:, [vp[0] for vp in val_pos_list]]
        if self.addr_sup_weight > 0 and ids is not None:
            # wk = to_wkey(key) is the address the binding wrote to; wv = to_wval(value) is what was
            # stored. These are the address/value InfoNCE targets (mirrors write_episode return_assoc).
            # addr-sup uses the BINDING-level keys/vals (perpos: the name key + the mean value) so the
            # write->read alignment term still selects the right binding.
            sup_vals = (torch.stack([mem_emb[:, vp].mean(dim=1) for vp in val_pos_list], dim=1)
                        if mt else vals_w)
            self._wk, self._wv = self.store.write_addr_val(keys, sup_vals)   # [B,M,mem_dim]
            self._cargo_ids = ids[:, keys_pos]          # [B,M] binding KEY token ids (cargo/name)
            if mt and self.mt_value == "perpos":
                # PER-POSITION addressing-supervision targets (the untested lever): for EACH binding m
                # and answer position t, the write-address is to_wkey(name_m + pos_tag[t]) and the
                # stored value is to_wval(cargo_{m,t}). These are the per-(binding,position) address/value
                # targets the per-position read-query must align to — supervising position t's read-query
                # against position t's write-address, not just the binding-level address. keys_w/vals_w
                # are already the M*K per-position (key,value) pairs in binding-major order.
                self._wk_pp, self._wv_pp = self.store.write_addr_val(keys_w, vals_w)  # [B,M*K,mem_dim]
        if keys_multi is not None:
            # MULTI-VECTOR KEYS: write the H per-subject key vectors to H slots, all pointing at the SAME
            # value. keys [B,M,H,mem] -> [B,M*H,mem] (binding-major, head-minor); values repeated per head.
            Bk, M, H, d = keys_multi.shape
            keys_w = keys_multi.reshape(Bk, M * H, d)
            vals_w = vals_w.unsqueeze(2).expand(Bk, M, H, d).reshape(Bk, M * H, d)
        return self.store.write(V, keys_w, vals_w)

    def memory_bank(self, ids, seg_len, qa_start, answer_pos, carry=True):
        """-> the K-slot pooled store read [B,K,mem_dim] (PRE out_proj). This is the pk-store analog of
        recall_mag.memory_bank() (the BoltAdapter attn-pooled retrieval): the base-AGNOSTIC mem_dim bank
        fed to the MAG tap / translator at Stage-2 and v1. inject() = out_proj(memory_bank(...)).
        Addressing supervision is computed here too when training+carry (so the bind block still sees
        the aux loss via inject), but at Stage-2/eval the adapter is frozen+eval so want_sup is False."""
        emb = self._e(ids)                              # [B,S,mem_dim]
        B = emb.shape[0]
        self._last_aux_loss = None
        self._last_conf = None                          # per-example store-confidence scalar (MAG conf gate)
        self._last_relidx = getattr(self.builder, "q_relidx", None)   # queried relation (per-relation EMA)
        want_sup = self.training and carry and self.addr_sup_weight > 0
        if carry:
            # GTE keying needs ids for the subject-tid lookup, so always pass ids when the GTE table is on.
            V = self._write_episode(emb, ids=ids if (want_sup or self._gte_keys is not None) else None)
        elif self.perpos_key == "disjoint" and getattr(self.builder, "multitoken", False) \
                and self.mt_value == "perpos":
            # ablated floor for disjoint: an EMPTY bank per position store (read block indexes V[t])
            K = self.builder.cargo_tokens
            V = [self.stores[t].init_state(B, emb.device, dtype=torch.float32) for t in range(K)]
        else:
            V = self.store.init_state(B, emb.device, dtype=torch.float32)   # empty store -> ablated floor
        b = self.builder
        # N-scaling key-separation fix (opt-in CAM_SUBJ_ONLY_QUERY=1): narrow the read query to just the
        # SUBJECT span, dropping the relation PREFIX (noise the subject write-key lacks) so the read
        # addresses by the subject, not the prompt. counterfactual_multi only (exposes q_subj_off/q_key_off).
        if os.environ.get("CAM_SUBJ_ONLY_QUERY") == "1" and getattr(b, "phrasing", None) == "counterfactual_multi":
            qs = qa_start + b.q_subj_off
            if self._gte_keys is not None:               # DECOUPLED: query from GTE (subject token-ids), one vector
                q = self._gte_key(ids[:, qs:qa_start + b.q_key_off + 1]).unsqueeze(1)   # [B,1,mem_dim]
            else:
                q = emb[:, qs:qa_start + b.q_key_off + 1]    # [B, subj_len, mem_dim] subject span only
            # symmetric with the write key: when the learned pool is on, address with the SAME pooled
            # subject vector (one query position) instead of per-token, so read matches write exactly.
            if os.environ.get("CAM_LEARNED_KEY_POOL") == "1":
                q = self._pool_subject(q, keepdim=True)   # [B,1,mem_dim]
        else:
            q = emb[:, qa_start:answer_pos]             # query tokens (leak-free): 'cargo' / 'name :'
        if getattr(b, "multitoken", False) and self.mt_value == "perpos":
            # PER-POSITION read (consistent train + eval): query name+pos_tag[t] for each answer slot t
            # -> the t-th cargo token's value. name query = the KEY-token mem embed (binding key pos 0).
            # This factorizes the K-token answer into K single-token bindings keyed by (name, position),
            # so each answer slot retrieves its OWN value. When training+carry+addr_sup, supervise EACH
            # position's read-query against THAT position's write-address (per-position addressing
            # supervision — the untested lever the single-slot perpos store lacked).
            Kc = b.cargo_tokens
            name_q = q[:, 0]                            # [B,mem_dim] the name token (query key)
            disjoint = self.perpos_key == "disjoint"
            slots = []
            rqs, ctxs_pp = [], []
            for t in range(Kc):
                store_t = self.stores[t] if disjoint else self.store   # own store per position (disjoint)
                Vt = V[t] if disjoint else V                           # own bank per position (disjoint)
                qt = self._pos_key(name_q, t).unsqueeze(1)            # [B,1,mem_dim] per-position query
                if want_sup:
                    rt, _hn, ctxs_t = store_t.read(Vt, qt, return_ctx=True)
                    rqs.append(store_t.head_query(qt, h=0)[:, 0])     # [B,mem_dim] factual read query
                    ctxs_pp.append(ctxs_t[0][:, 0])                   # [B,mem_dim] factual value mix
                else:
                    rt, _ = store_t.read(Vt, qt)                      # [B,1,mem_dim]
                slots.append(rt[:, 0])
            if want_sup:
                if disjoint:
                    self._compute_addr_sup_disjoint(rqs, ctxs_pp, ids, qa_start, Kc)
                else:
                    self._compute_addr_sup_perpos(rqs, ctxs_pp, ids, qa_start, Kc)
            return torch.stack(slots, dim=1)           # [B,Kc,mem_dim] one slot per answer position
        if want_sup:
            read, _head_norms, ctxs = self.store.read(V, q, return_ctx=True)  # +per-head value-mix ctx
            self._compute_addr_sup(q, ctxs, ids, qa_start)
        else:
            # frozen Stage-2/eval: also pull the store-confidence scalar (factual-head pre-norm retrieval
            # magnitude) so the MAG tap's confidence gate can fire in proportion to retrieval strength.
            read, _head_norms, self._last_conf = self.store.read(V, q, return_conf=True)
        read = self._maxsim_reduce(read)                # multi-vector MaxSim: best head, not pool (Phase B)
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)
        attn = torch.softmax(pq @ read.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)
        return attn @ read                              # [B,K,mem_dim] pooled (PRE out_proj)

    # ---- Track 4: PERSISTENT / online store (doc-independent) --------------------------------------
    def persistent_write(self, V, keys, vals, store=None):
        """Write A associations (keys/vals [B,A,mem_dim]) into a PERSISTENT value bank V (error-correcting
        delta write). Track 4 online binding: call once per edit on the SAME V to accumulate a standing
        memory — no episodic doc, no reset. Returns the updated V.

        `store` selects WHICH ProductKeyStore to write through (default self.store). The perpos-key=disjoint
        persistent path passes self.stores[t] so answer position t writes into its OWN per-position codebook
        (mirrors _write_episode's disjoint branch); byte-identical to before when store is None.

        Phase K1 (write-where-you-read, now the DEFAULT — CAM_WRITE_AT_READ=0 to disable): address the
        write with the READ query head_query(key)=read_q[0](key)+head_bias[0] instead of to_wkey(key), so
        the value lands at the exact slot the read selects → closes the write↔read addressing gap for
        boundary subjects. Validated (§3.16, n=3): below-gate 15-25→0, delivery 0.72→~0.90 at held
        locality, persistent-path only. Made the default 2026-07-04 given the strict, structural win.

        Phase K2 (CAM_WRITE_REDUNDANT=1): write the value to BOTH the to_wkey slots AND the read-query
        slots (two delta writes into the SAME V) — a softer K1 that keeps the trained write-address too, so
        a boundary subject is covered whichever cell the read lands in. Fallback if K1's pure relocation
        regresses anything."""
        store = store if store is not None else self.store
        if os.environ.get("CAM_WRITE_REDUNDANT") == "1":      # K2: to_wkey slots AND read-query slots
            V = store.write(V, keys, vals)                    # (a) the trained write-address
            return store.write(V, keys, vals, addr=store.head_query(keys, 0))  # (b) the read-address
        addr = None
        if os.environ.get("CAM_WRITE_AT_READ", "1") != "0":   # K1 DEFAULT-ON (opt out with =0)
            addr = store.head_query(keys, 0)                  # read-space write address (K1)
        return store.write(V, keys, vals, addr=addr)

    def persistent_bank(self, V, q, store=None):
        """Read a PERSISTENT bank V with a subject query q [B,Lq,mem_dim] -> pooled [B,K,mem_dim] bank +
        the store-confidence scalar (self._last_conf). A doc-independent mirror of memory_bank's
        read+readout — the query is just the subject, the bank is the standing store. `store` selects the
        ProductKeyStore (default self.store); the perpos-key=disjoint path reads position t from
        self.stores[t] (its own codebook). The readout (maxsim/readout_q/out_proj) is store-agnostic."""
        store = store if store is not None else self.store
        read, _hn, self._last_conf = store.read(V, q, return_conf=True)
        read = self._maxsim_reduce(read)                # multi-vector MaxSim: best head, not pool (Phase B)
        B = q.shape[0]
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)
        attn = torch.softmax(pq @ read.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)
        return attn @ read

    def inject(self, ids, seg_len, qa_start, answer_pos, carry=True):
        """-> K prefix vectors [B,K,base_hidden]. carry=True writes the M bindings then reads with the
        QA cargo query; carry=False reads an EMPTY store (the ablated floor). Same contract as
        BoltAdapter.inject (seg_len is accepted for signature-compatibility; the store ingests the whole
        binding block at once via _write_episode, not segment-by-segment)."""
        pooled = self.memory_bank(ids, seg_len, qa_start, answer_pos, carry=carry)  # [B,K,mem_dim]
        return self.out_proj(pooled)                    # [B,K,base_hidden]

    # ---- addressing supervision (the dropped write->read alignment loss, hub-free) ----
    def _compute_addr_sup(self, q, ctxs, ids, qa_start):
        """Two InfoNCE terms over the M bindings, computed in mem_dim space (NO hub geometry):
          (a) ADDRESS: factual read query read_q[0](q)+head_bias[0] (pooled over QA tokens) close to the
              QUERIED binding's write-address wk=to_wkey(cargo), far from the other M-1 bindings.
          (b) VALUE:   the factual head's retrieved value-mix ctx[0] close to the queried binding's
              STORED value wv=to_wval(name), far from the others.
        The queried binding is the one whose cargo token id == the QA query token id (ids[:,qa_start]).
        Identical formulation to train_mem_canonical.run_step, but on base-derived random codebooks in
        mem_dim — needs no canonical-Z target geometry; the targets ARE the store's own write projns."""
        import torch.nn.functional as F
        # which binding was queried? counterfactual_multi tells us the queried binding SLOT directly
        # (q_bind_idx, batch-uniform) — robust to multi-token subjects that may share a last token. Other
        # phrasings match the QA subject token (at qa_start, or qa_start+q_key_off) to the M binding KEY ids.
        b = self.builder
        if getattr(b, "phrasing", None) == "counterfactual_multi":
            tgt = q.new_full((q.shape[0],), int(getattr(b, "q_bind_idx", 0)), dtype=torch.long)
        else:
            qoff = getattr(b, "q_key_off", getattr(b, "q_subj_off", 0))
            q_tok = ids[:, qa_start + qoff].unsqueeze(1)               # [B,1]
            tgt = (self._cargo_ids == q_tok).float().argmax(dim=1)     # [B] queried binding index
        rq = self.store.head_query(q, h=0).mean(dim=1)             # [B,mem_dim] factual read query
        ctx_fac = ctxs[0].mean(dim=1)                              # [B,mem_dim] factual retrieved mix
        sa = torch.einsum("bd,bmd->bm", F.normalize(rq, dim=-1),
                          F.normalize(self._wk, dim=-1)) / 0.1     # address scores
        sv = torch.einsum("bd,bmd->bm", F.normalize(ctx_fac, dim=-1),
                          F.normalize(self._wv, dim=-1)) / 0.1     # value scores
        self._last_aux_loss = F.cross_entropy(sa, tgt) + F.cross_entropy(sv, tgt)
        # KEY-SEPARATION REPULSION (#19: ceiling is ADDRESSING). Push the M distinct subjects' WRITE
        # ADDRESSES (what the product-key top-k actually scores) toward mutual orthogonality, so similar/
        # multi-token names don't alias to the same slots as N grows. Off-diagonal Gram penalty on the
        # normalized write-addresses; backprops into the key encoder + to_wkey. Opt-in CAM_KEY_REPULSION>0.
        rep_w = float(os.environ.get("CAM_KEY_REPULSION", "0"))
        if rep_w > 0 and self._wk.shape[1] > 1:
            wk_n = F.normalize(self._wk, dim=-1)                   # [B,M,mem_dim]
            gram = torch.einsum("bmd,bnd->bmn", wk_n, wk_n)        # [B,M,M] pairwise cosine
            M = gram.shape[1]
            off = gram - torch.eye(M, device=gram.device, dtype=gram.dtype).unsqueeze(0)  # zero self-sim
            rep = (off ** 2).sum() / (gram.shape[0] * M * (M - 1))  # mean squared off-diagonal cosine
            self._last_aux_loss = self._last_aux_loss + rep_w * rep
            # GLOBAL repulsion (Phase A.2): the per-doc term only separates the M=8 in-doc subjects. To
            # attack the N~137 ceiling, ALSO repel this doc's keys against a running FIFO buffer of keys
            # written in RECENT docs (MoCo-style memory bank; detached, no grad through the buffer) — so
            # the separation objective sees the whole standing population, not just one doc. Opt-in
            # CAM_KEY_REPULSION_GLOBAL=1; buffer size CAM_KEY_REPULSION_BUFSIZE (default 256).
            if os.environ.get("CAM_KEY_REPULSION_GLOBAL") == "1":
                cur = wk_n.reshape(-1, wk_n.shape[-1])            # [B*M, mem_dim] this doc's normalized keys
                buf = getattr(self, "_key_buf", None)
                if buf is not None and buf.shape[0] > 0:
                    sims = cur @ buf.to(cur.device, cur.dtype).t()   # [B*M, Nbuf] cosine vs buffered keys
                    self._last_aux_loss = self._last_aux_loss + rep_w * (sims ** 2).mean()
                bufsize = int(os.environ.get("CAM_KEY_REPULSION_BUFSIZE", "256"))
                newk = cur.detach()
                self._key_buf = newk if buf is None else torch.cat([buf, newk], dim=0)
                if self._key_buf.shape[0] > bufsize:
                    self._key_buf = self._key_buf[-bufsize:]      # FIFO cap

    def _compute_addr_sup_perpos(self, rqs, ctxs_pp, ids, qa_start, Kc):
        """PER-POSITION addressing supervision (the untested lever). For each answer position t:
          (a) ADDRESS: the position-t read-query rqs[t] = read_q[0](name_q + pos_tag[t]) close to the
              QUERIED binding's position-t write-address wk_pp[:, m*Kc+t], far from EVERY other
              (binding, position) write-address. This supervises position t's read-query against position
              t's write-address — earlier perpos work supervised only the binding-level address.
          (b) VALUE:   the position-t factual value-mix ctxs_pp[t] close to the queried binding's
              position-t stored value wv_pp[:, m*Kc+t], far from the others.
        Targets span all M*Kc per-position associations (binding-major: assoc index m*Kc + t), so the
        per-position read must disambiguate BOTH which binding AND which position. The queried binding m
        is the one whose KEY (name) token id == the QA query token id (ids[:,qa_start])."""
        import torch.nn.functional as F
        q_tok = ids[:, qa_start].unsqueeze(1)                      # [B,1] queried name token id
        m = (self._cargo_ids == q_tok).float().argmax(dim=1)      # [B] queried binding index
        wk = F.normalize(self._wk_pp, dim=-1)                     # [B,M*Kc,mem_dim]
        wv = F.normalize(self._wv_pp, dim=-1)
        loss = rqs[0].new_zeros(())
        for t in range(Kc):
            tgt = m * Kc + t                                       # [B] queried (binding,position) assoc
            sa = torch.einsum("bd,bnd->bn", F.normalize(rqs[t], dim=-1), wk) / 0.1     # [B,M*Kc]
            sv = torch.einsum("bd,bnd->bn", F.normalize(ctxs_pp[t], dim=-1), wv) / 0.1
            loss = loss + F.cross_entropy(sa, tgt) + F.cross_entropy(sv, tgt)
        self._last_aux_loss = loss / Kc

    def _compute_addr_sup_disjoint(self, rqs, ctxs_pp, ids, qa_start, Kc):
        """DISJOINT per-position addressing supervision (Thrust 1 #4). Position t addresses its OWN store
        (self.stores[t]) over its OWN M write-addresses self._wk_pp_dj[t] [B,M,mem_dim]. So position t's
        InfoNCE is M-way (which BINDING), NOT the M*Kc-way of the shared-codebook perpos: cross-position
        negatives are meaningless because each position lives in a physically separate codebook, so there
        is nothing for position t's read to be contaminated BY. This is exactly the separation the shared
        codebook (exp#3 codebook) could only approximate via a per-position projection into a shared bank.
          (a) ADDRESS: rqs[t] = head_query_t(name_q + pos_tag[t]) close to store t's queried-binding
              write-address, far from store t's other M-1 bindings.
          (b) VALUE:   ctxs_pp[t] close to store t's queried-binding stored value, far from the others.
        The queried binding m is the one whose KEY (name) token id == the QA query token id."""
        import torch.nn.functional as F
        q_tok = ids[:, qa_start].unsqueeze(1)                      # [B,1] queried name token id
        m = (self._cargo_ids == q_tok).float().argmax(dim=1)      # [B] queried binding index (target)
        loss = rqs[0].new_zeros(())
        for t in range(Kc):
            wk = F.normalize(self._wk_pp_dj[t], dim=-1)           # [B,M,mem_dim] store-t write addresses
            wv = F.normalize(self._wv_pp_dj[t], dim=-1)
            sa = torch.einsum("bd,bmd->bm", F.normalize(rqs[t], dim=-1), wk) / 0.1     # [B,M]
            sv = torch.einsum("bd,bmd->bm", F.normalize(ctxs_pp[t], dim=-1), wv) / 0.1
            loss = loss + F.cross_entropy(sa, m) + F.cross_entropy(sv, m)
        self._last_aux_loss = loss / Kc

    def aux_loss(self):
        """The weighted addressing-supervision loss from the LAST training inject() (or None). bind_adapter
        adds this to the direct CE only for adapters that expose it, so BoltAdapter's path is untouched."""
        if self._last_aux_loss is None:
            return None
        return self.addr_sup_weight * self._last_aux_loss

    def direct_logits(self, prefix, ans=None):
        """tied-unembed readout of the injected prefix -> vocab logits. Training signal.

        single-token: mean over the K injected slots -> [B,vocab] (byte-identical to BoltAdapter).
        multi-token, readout='linear' (default, byte-preserved): decode answer-token t from injected
          slot t -> [B,Kc,vocab]; K-slot pooled read as a sequence head, slot t predicts answer token t.
        multi-token, readout='decoder': AR transformer-decoder head. The K prefix slots are the
          cross-attention MEMORY; the answer sequence is decoded autoregressively, TEACHER-FORCED on
          the gold answer ids (`ans` [B,Kc]) shifted right by a learned BOS, with a causal mask over
          answer positions. -> [B,Kc,vocab]. `ans` is required in this mode (training + teacher-forced
          eval both pass it)."""
        b = self.builder
        mt = getattr(b, "multitoken", False)
        if mt and self.readout == "decoder":
            return self._decoder_logits(prefix, ans, b.cargo_tokens)
        if mt:
            Kc = b.cargo_tokens
            return prefix[:, :Kc] @ self.unembed        # [B,Kc,vocab]
        return prefix.mean(dim=1) @ self.unembed        # [B,vocab]

    def _decoder_logits(self, prefix, ans, Kc):
        """AR decoder readout. prefix:[B,K,base_hidden] = K retrieved store slots (cross memory).
        ans:[B,Kc] gold answer ids (teacher forcing). -> [B,Kc,vocab]."""
        assert ans is not None, "decoder readout needs gold answer ids (teacher forcing)"
        B = prefix.shape[0]
        mem = self.dec_mem_norm(self.dec_mem_proj(prefix))             # [B,K,dec_dim] cross memory
        # teacher-forced inputs: <BOS>, emb(ans_0), ..., emb(ans_{Kc-2}) -> predict ans_0..ans_{Kc-1}
        ans_emb = self.dec_tok_proj(self.embed(ans).float())          # [B,Kc,dec_dim]
        bos = self.dec_bos.view(1, 1, -1).expand(B, 1, -1)            # [B,1,dec_dim]
        tgt = torch.cat([bos, ans_emb[:, :Kc - 1]], dim=1)           # [B,Kc,dec_dim] shifted right
        tgt = tgt + self.dec_pos[:Kc].unsqueeze(0)                    # add answer-position codes
        cmask = torch.triu(torch.ones(Kc, Kc, device=prefix.device, dtype=torch.bool), diagonal=1)
        out = self.dec(tgt, mem, tgt_mask=cmask)                     # [B,Kc,dec_dim]
        out = self.dec_out_norm(out)
        return self.dec_logit(out)                                   # [B,Kc,vocab]
