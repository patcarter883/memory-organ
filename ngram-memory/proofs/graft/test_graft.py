"""Proofs for graft_proto.NGramGraft on tiny random hosts built from the REAL config classes / shrunk real configs.

usage: python test_graft.py <models_dir> <out_json>
"""
import json, math, sys, time, copy
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3_5 import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
from transformers.models.gemma4 import Gemma4TextConfig, Gemma4ForCausalLM

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from graft_proto import NGramGraft, FLASH_NEXT_KEY_SPEC, SparseRows  # noqa

torch.set_grad_enabled(False)
torch.set_num_threads(1)
MD, OUT = sys.argv[1], sys.argv[2]
V = 512  # tiny host vocab
EOS, PAD = 1, 0
RESULTS = {}


def load_text_cfg(path):
    return json.load(open(f"{MD}/{path}/config.json"))["text_config"]


def tiny_qwen(path, moe):
    t = load_text_cfg(path)
    keep = {k: t[k] for k in ("attn_output_gate", "hidden_act", "rms_norm_eps", "full_attention_interval",
                              "partial_rotary_factor", "linear_conv_kernel_dim", "output_gate_type") if k in t}
    rope = dict(t["rope_parameters"]); rope["mrope_section"] = [2, 1, 1]  # head_dim 32 * 0.25 / 2 = 4
    kw = dict(keep, vocab_size=V, hidden_size=64, num_hidden_layers=4, layer_types=t["layer_types"][:4],
              num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_key_head_dim=16,
              linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, rope_parameters=rope,
              eos_token_id=EOS, bos_token_id=EOS, pad_token_id=PAD, tie_word_embeddings=False)
    if moe:
        kw.update(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
        cfg = Qwen3_5MoeTextConfig(**kw); cls = Qwen3_5MoeForCausalLM
    else:
        kw.update(intermediate_size=128)
        cfg = Qwen3_5TextConfig(**kw); cls = Qwen3_5ForCausalLM
    cfg._attn_implementation = "eager"
    return cls(cfg).eval(), cfg


def tiny_gemma(path):
    t = load_text_cfg(path)
    keep = {k: t[k] for k in ("hidden_activation", "rms_norm_eps", "final_logit_softcapping", "attention_k_eq_v",
                              "enable_moe_block", "use_double_wide_mlp", "rope_parameters", "num_kv_shared_layers",
                              "hidden_size_per_layer_input") if k in t}
    lt = t["layer_types"][:6]  # 5 sliding + 1 full, as in the real pattern
    kw = dict(keep, vocab_size=V, vocab_size_per_layer_input=V, hidden_size=64, num_hidden_layers=6, layer_types=lt,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
              num_global_key_value_heads=1, intermediate_size=96, sliding_window=8,
              eos_token_id=EOS, bos_token_id=2, pad_token_id=PAD, tie_word_embeddings=True)
    if t.get("enable_moe_block"):
        kw.update(num_experts=8, top_k_experts=2, moe_intermediate_size=32)
    cfg = Gemma4TextConfig(**kw)
    cfg._attn_implementation = "eager"
    m = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():  # real checkpoints carry non-unit layer_scalar buffers; exercise that path
        for i, l in enumerate(m.model.layers):
            l.layer_scalar.fill_(0.7 + 0.1 * i)
    return m, cfg


TINY_KEY_SPEC = dict(FLASH_NEXT_KEY_SPEC, ngram_vocab_size_base=1000, eos_token_id=EOS)


def rand_ids(g, B, T, lo=3):
    return torch.randint(lo, V, (B, T), generator=g)


def logits(model, ids, **kw):
    return model(input_ids=ids, use_cache=False, **kw).logits.float()


def maxdiff(a, b):
    return (a - b).abs().max().item()


def decode_parity(model, ids, P, mask=None, pos=None):
    """prefill ids[:, :P] with cache, then feed ids[:, P:] one token at a time; compare to one full forward."""
    full = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
    out = model(input_ids=ids[:, :P], attention_mask=None if mask is None else mask[:, :P],
                position_ids=None if pos is None else pos[:, :P], use_cache=True)
    pre = (out.logits.float() - full[:, :P]).abs()
    if mask is not None:  # logits at left-pad positions are meaningless; compare valid positions only
        pre = pre * mask[:, :P, None]
    cache, d = out.past_key_values, [pre.max().item()]
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], past_key_values=cache,
                  attention_mask=None if mask is None else mask[:, :t + 1],
                  position_ids=None if pos is None else pos[:, t:t + 1], use_cache=True)
        d.append(maxdiff(o.logits.float()[:, -1], full[:, t]))
    return max(d), full


def run_host(name, model, cfg):
    t0 = time.time()
    r = {}
    g = torch.Generator().manual_seed(0)
    H = cfg.hidden_size
    ids = rand_ids(g, 3, 24)
    ids[1, 10] = EOS  # an in-sequence EOS resets n-gram history (Flash-Next semantics)
    base = logits(model, ids)

    # ---------------- 1. exact no-op when empty (dense + sparse, editor + random projections)
    noop = {}
    for table in ("dense", "sparse"):
        for init in ("editor", "random"):
            gr = NGramGraft(H, TINY_KEY_SPEC, table=table, init=init).install(model)
            l = logits(model, ids)
            # also cached decode with empty table
            dp, _ = decode_parity(model, ids, 16)
            gr.remove()
            noop[f"{table}/{init}"] = dict(max_abs_dlogit=maxdiff(l, base), bitwise_equal=bool(torch.equal(l, base)))
    r["noop_empty_table"] = noop

    # ---------------- 2. an edited row fires only on its n-gram (module output + logits)
    gr = NGramGraft(H, TINY_KEY_SPEC, table="sparse", init="editor").install(model)
    a, b = int(ids[0, 5]), int(ids[0, 6])
    big = rand_ids(torch.Generator().manual_seed(1), 16, 64)
    big[3, 20:22] = torch.tensor([a, b]); big[7, 40:42] = torch.tensor([a, b]); big[7, 2:4] = torch.tensor([a, b])
    delta = torch.randn(H, generator=g)
    W = gr.ple.value_proj.weight  # (H, E) orthonormal; out = 0.5 * W e
    e = 2 * W.T @ delta  # min-norm e with 0.5 W e = delta
    gr.write_ngram([a, b], e)
    logits(model, big)
    out = gr.debug_last["out"]  # (B, T, H)
    is_bigram = torch.zeros(big.shape, dtype=torch.bool)
    is_bigram[:, 1:] = (big[:, :-1] == a) & (big[:, 1:] == b)
    nz = out.norm(dim=-1) > 0
    E2 = W.shape[1] // 2  # 2-gram heads are the first half of the concatenated embedding
    expected = 0.5 * W[:, :E2] @ e[:E2]  # a 2-gram-only write reaches only an E/2-dim subspace of the residual
    full_fire = (out - expected).norm(dim=-1) < 1e-4
    # per-head partial collisions: positions that hit >=1 of the 8 written rows but are not the bigram
    rid = gr.row_ids(big)[..., :8]
    written = gr.ple.ple_embedding.ngram_embedding.row_ids
    hits = torch.isin(rid, written).sum(-1)
    r["fires_only_on_ngram"] = dict(
        bigram_positions=int(is_bigram.sum()), nonzero_output_positions=int(nz.sum()),
        nonzero_exactly_at_bigram=bool(torch.equal(nz & ~(hits > 0), torch.zeros_like(nz)) and torch.equal(nz[is_bigram], torch.ones(int(is_bigram.sum()), dtype=torch.bool))),
        full_8_head_fire_at_bigram=bool(full_fire[is_bigram].all()),
        full_8_head_fire_elsewhere=int((full_fire & ~is_bigram).sum()),
        partial_collision_positions=int(((hits > 0) & ~is_bigram).sum()),
        partial_collision_max_heads=int(hits[~is_bigram].max()),
        partial_collision_rel_norm_max=float((out.norm(dim=-1)[(hits > 0) & ~is_bigram] / delta.norm() * 2).max()) if ((hits > 0) & ~is_bigram).any() else 0.0,
        total_positions=int(big.numel()),
        written_value_reaches_residual_exactly=maxdiff(out[is_bigram], expected.expand(int(is_bigram.sum()), -1)),
        fraction_of_target_delta_reached_by_2gram_only=float(expected @ delta / delta.norm() ** 2),
    )
    # logit level: sequences without the bigram are bit-identical; with it, logits before first occurrence identical
    gr.remove()
    base_big = logits(model, big)
    gr.install(model)
    lb = logits(model, big)
    rows_without = [i for i in range(big.shape[0]) if not is_bigram[i].any() and not (hits[i] > 0).any()]
    first = int(is_bigram[3].nonzero()[0])
    r["fires_only_on_ngram"].update(
        rows_without_any_hit_bitwise_equal=bool(torch.equal(lb[rows_without], base_big[rows_without])),
        n_rows_without=len(rows_without),
        row3_prefix_before_bigram_bitwise_equal=bool(torch.equal(lb[3, :first], base_big[3, :first])),
        row3_dlogit_at_bigram=maxdiff(lb[3, first], base_big[3, first]),
    )
    gr.remove()

    # ---------------- 3. dense == sparse with identical written rows
    gd = NGramGraft(H, TINY_KEY_SPEC, table="dense", init="editor")
    gd.ple.load_state_dict({k: v for k, v in gr.ple.state_dict().items() if "ngram_embedding" not in k}, strict=False)
    gd.ple.ple_embedding.ngram_embedding.weight.data.copy_(gr.ple.ple_embedding.ngram_embedding.dense())
    gd.install(model); ld = logits(model, big); gd.remove()
    r["dense_equals_sparse_maxdiff"] = maxdiff(ld, lb)

    # ---------------- 4. decode parity (cached vs full) -- random projections + random dense table (everything fires)
    dec = {}
    for init in ("editor", "random"):
        gq = NGramGraft(H, TINY_KEY_SPEC, table="dense", init=init)
        gq.ple.ple_embedding.ngram_embedding.weight.data.normal_(0, 1.0, generator=torch.Generator().manual_seed(3))
        gq.install(model)
        d, full = decode_parity(model, ids, 7)
        dec[init] = dict(maxdiff=d, graft_effect_on_logits=maxdiff(full, base))
        # beam-style cache reorder
        o = model(input_ids=ids[:, :7], use_cache=True)
        perm = torch.tensor([2, 0, 1])
        o.past_key_values.reorder_cache(perm)
        o2 = model(input_ids=ids[perm, 7:8], past_key_values=o.past_key_values, use_cache=True)
        dec[init]["reorder_cache_maxdiff"] = maxdiff(o2.logits.float()[:, -1], full[perm, 7])
        # generate() greedy == manual greedy, with the graft active
        gen = model.generate(ids[:, :7], max_new_tokens=6, do_sample=False, pad_token_id=PAD)
        cur = ids[:, :7]
        for _ in range(6):
            nxt = logits(model, cur)[:, -1].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], 1)
        dec[init]["generate_equals_full_forward_greedy"] = bool(torch.equal(gen, cur))
        gq.remove()
        dec[init]["host_without_graft_decode_maxdiff"] = decode_parity(model, ids, 7)[0]
    r["decode_parity"] = dec

    # ---------------- 5. batch + left padding
    gq = NGramGraft(H, TINY_KEY_SPEC, table="dense", init="random")
    gq.ple.ple_embedding.ngram_embedding.weight.data.normal_(0, 1.0, generator=torch.Generator().manual_seed(4))
    gq.install(model)
    L = [14, 9, 5]
    seqs = [rand_ids(torch.Generator().manual_seed(10 + i), 1, n)[0] for i, n in enumerate(L)]
    Tm = max(L)
    batch = torch.full((3, Tm), PAD); mask = torch.zeros(3, Tm, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, Tm - len(s):] = s; mask[i, Tm - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    lpad = model(input_ids=batch, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
    lm = gq.debug_last.get("layer_mask")
    same_mask = None
    if isinstance(lm, torch.Tensor) and lm.ndim == 2:
        same_mask = bool(torch.equal(lm, gq.debug_last["conv_mask"]))
    pad_diff, pad_diff_nograft = [], []
    singles = [logits(model, s[None]) for s in seqs]
    for i, s in enumerate(seqs):
        pad_diff.append(maxdiff(lpad[i, Tm - len(s):], singles[i][0]))
    # graft conv-mask equals the host's own linear-attention mask on qwen hosts
    # padded batched decode parity: extend each sequence by 4 tokens
    ext = rand_ids(torch.Generator().manual_seed(20), 3, 4)
    ids2 = torch.cat([batch, ext], 1); mask2 = torch.cat([mask, torch.ones(3, 4, dtype=torch.long)], 1)
    pos2 = (mask2.cumsum(-1) - 1).clamp_min(0)
    dpp, full2 = decode_parity(model, ids2, Tm, mask=mask2, pos=pos2)
    single_ext = [logits(model, torch.cat([s, ext[i]])[None]) for i, s in enumerate(seqs)]
    pad_dec_vs_single = max(maxdiff(full2[i, Tm:], single_ext[i][0, -4:]) for i in range(3))
    gq.remove()
    lpad0 = model(input_ids=batch, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
    for i, s in enumerate(seqs):
        pad_diff_nograft.append(maxdiff(lpad0[i, Tm - len(s):], logits(model, s[None])[0]))
    r["left_pad"] = dict(graft_padded_vs_single_maxdiff=max(pad_diff), host_only_padded_vs_single_maxdiff=max(pad_diff_nograft),
                         graft_conv_mask_equals_host_linear_mask=same_mask, padded_cached_decode_vs_full=dpp,
                         padded_batch_ext_vs_single_ext=pad_dec_vs_single)

    # ---------------- 6. final-NLL gradient reaches the rows (and only the rows that were read)
    with torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(False)
        gg = NGramGraft(H, TINY_KEY_SPEC, table="dense", init="editor").install(model)
        tab = gg.ple.ple_embedding.ngram_embedding.weight
        tab.requires_grad_(True)
        gg.ple.key_proj.weight.requires_grad_(True); gg.ple.conv1d.weight.requires_grad_(True)
        x = ids[:1, :12]; tgt = torch.tensor([7])
        loss = F.cross_entropy(model(input_ids=x, use_cache=False).logits[:, -1].float(), tgt)
        loss.backward()
        read = gg.row_ids(x).reshape(-1).unique()
        gnorm = tab.grad.norm(dim=-1)
        last_rows = gg.row_ids(x)[0, -1]
        grad = dict(loss=loss.item(), rows_read=int(read.numel()), rows_with_grad=int((gnorm > 0).sum()),
                    grad_only_on_read_rows=bool(set((gnorm > 0).nonzero().flatten().tolist()) <= set(read.tolist())),
                    last_position_rows_grad_norm=float(gnorm[last_rows].min()),
                    all_finite=bool(torch.isfinite(tab.grad).all()),
                    key_proj_grad_absmax_at_zero_init=float(gg.ple.key_proj.weight.grad.abs().max()),
                    conv1d_grad_absmax_at_zero_init=float(gg.ple.conv1d.weight.grad.abs().max()))
        # tiny one-fact edit: optimise ONLY the 16 rows of the last position (2-gram + 3-gram keys)
        gg.remove()
        gs = NGramGraft(H, TINY_KEY_SPEC, table="sparse", init="editor")
        sl = gs.head_slices()
        gs.write_ngram([int(x[0, -2]), int(x[0, -1])], torch.zeros(gs.cfg.ple_embed_dim))
        gs.write_ngram([int(x[0, -3]), int(x[0, -2]), int(x[0, -1])], torch.zeros(gs.cfg.ple_embed_dim))
        gs.install(model)
        rows = gs.ple.ple_embedding.ngram_embedding.rows
        rows.requires_grad_(True)
        opt = torch.optim.Adam([rows], lr=0.3)
        p0 = None
        for step in range(80):
            lp = model(input_ids=x, use_cache=False).logits[:, -1].float().log_softmax(-1)
            if p0 is None: p0 = lp[0, 7].exp().item()
            l = -lp[0, 7]; opt.zero_grad(); l.backward(); opt.step()
        p1 = model(input_ids=x, use_cache=False).logits[:, -1].float().softmax(-1)[0, 7].item()
        grad.update(edit_target_prob_before=p0, edit_target_prob_after=p1, edit_rows=int(rows.shape[0]),
                    edit_row_norm=float(rows.detach().norm(dim=-1).mean()))
        gs.remove()
    r["gradient"] = grad
    r["seconds"] = round(time.time() - t0, 1)
    return r


def key_compat():
    """Graft keys (built by the SAME Qwen4ExpTextNGramEmbedding) vs the real Flash-Next checkpoint hash buffers."""
    gr = NGramGraft(5120, FLASH_NEXT_KEY_SPEC, table="sparse")  # 27B-sized host, real 320M-row key space, no alloc
    e = gr.ple.ple_embedding
    ck = json.load(open(__file__.rsplit("/", 1)[0] + "/flashnext_hash_buffers.json"))
    res = dict(layer_multipliers_match=e.layer_multipliers.tolist() == ck["layer_multipliers"],
               head_sizes_match=e.ngram_heads_vocab_sizes.tolist() == ck["ngram_heads_vocab_sizes"],
               head_offsets_match=e.ngram_heads_offsets.tolist() == ck["ngram_heads_offsets"],
               padded_rows=e.ngram_embedding.num_embeddings, row_width_27B=e.ngram_embedding.embedding_dim,
               flashnext_rows_per_shard=e.ngram_embedding.num_embeddings // 128)
    # different host hidden sizes -> identical row ids (keys don't depend on hidden / ple_embed_dim)
    g2 = NGramGraft(2048, FLASH_NEXT_KEY_SPEC, table="sparse")
    toks = torch.randint(0, 248320, (2, 50), generator=torch.Generator().manual_seed(0)); toks[0, 20] = 248044
    res["row_ids_identical_across_hosts_5120_vs_2048"] = bool(torch.equal(gr.row_ids(toks), g2.row_ids(toks)))
    # gotcha: key vocab matters -- the gemma vocab (262144) gives different multipliers
    g3 = NGramGraft(2816, dict(FLASH_NEXT_KEY_SPEC, vocab_size=262144, eos_token_id=1), table="sparse")
    res["gemma_vocab_multipliers"] = g3.ple.ple_embedding.layer_multipliers.tolist()
    res["gemma_vocab_multipliers_differ"] = g3.ple.ple_embedding.layer_multipliers.tolist() != ck["layer_multipliers"]
    return res


if __name__ == "__main__":
    torch.manual_seed(0)
    RESULTS["key_compat"] = key_compat()
    print(json.dumps(RESULTS["key_compat"]), flush=True)
    hosts = [("Qwen3.8-27B (qwen3_5)", lambda: tiny_qwen("Qwen/Qwen3.8-27B", False)),
             ("Qwen3.6-35B-A3B (qwen3_5_moe)", lambda: tiny_qwen("Qwen/Qwen3.6-35B-A3B", True)),
             ("gemma-4-26B-A4B-it (gemma4 moe)", lambda: tiny_gemma("google/gemma-4-26B-A4B-it")),
             ("gemma-4-31B-it (gemma4 dense)", lambda: tiny_gemma("google/gemma-4-31B-it"))]
    only = sys.argv[3:]
    for i, (n, mk) in enumerate(hosts):
        if only and str(i) not in only:
            continue
        torch.manual_seed(0)
        model, cfg = mk()
        RESULTS[n] = run_host(n, model, cfg)
        print(n, json.dumps(RESULTS[n], indent=1), flush=True)
    json.dump(RESULTS, open(OUT, "w"), indent=1)
