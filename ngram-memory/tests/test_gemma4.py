"""Gemma 4: graft on tiny gemma4 hosts (26B-like MoE + k_eq_v, 31B-like dense; P/graft/test_graft.py), the
Gemma-PLE native backend on a tiny E4B-like model (P/t2-gemma4-ple/tiny_ple.py a-e), and the Gemma-PLE editor-init
graft (proof g; P/redteam-correctness/checks.py [3])."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ngram_memory.backends.base import RowDiff
from ngram_memory.backends.gemma4_ple import Gemma4PLEBackend, GemmaPLESpec, editor_init_ple_graft
from ngram_memory.backends.graft import NGramGraft
from ngram_memory.backends.graft.gemma4 import gemma_graft_key_spec, install_graft_gemma4
from ngram_memory.spec import FLASH_NEXT_VOCAB, flash_next_key_spec

from conftest import EOS, PAD, decode_parity, fill_random_rows, logits, maxdiff, rand_ids, tiny_fmt
from tiny_hosts import record, tiny_gemma, tiny_gemma_cfg

GEMMA_HOSTS = {"26b": "gemma-4-26b-a4b-it.config.json", "31b": "gemma-4-31b-it.config.json"}
_CACHE: dict = {}


def gemma_host(kind: str):
    if kind not in _CACHE:
        m = tiny_gemma(GEMMA_HOSTS[kind])
        with torch.no_grad():  # real checkpoints carry non-unit layer_scalar buffers; exercise that path
            for i, l in enumerate(m.model.layers):
                l.layer_scalar.fill_(0.7 + 0.1 * i)
        _CACHE[kind] = m
    return _CACHE[kind]


def test_gemma_key_instance_differs_from_flash_next():
    k = gemma_graft_key_spec()
    assert (k.vocab_size, k.eos_token_id) == (262144, 1)
    assert k.multipliers != flash_next_key_spec().multipliers and k.head_sizes == flash_next_key_spec().head_sizes
    assert FLASH_NEXT_VOCAB != k.vocab_size


@pytest.mark.parametrize("kind", ["26b", "31b"])
@pytest.mark.parametrize("arm", ["G-lin", "G-dir"])
def test_gemma_graft_noop_decode_padding_gradient(kind, arm):
    host = gemma_host(kind)
    H = host.config.hidden_size
    assert host.config.enable_moe_block == (kind == "26b")
    ids = rand_ids(0, 3, 24)
    ids[1, 10] = EOS
    base = logits(host, ids)
    host_dec, _ = decode_parity(host, ids, 7)
    # ---- no-op (empty table), sparse and dense
    for table in ("sparse", "dense"):
        g = NGramGraft(tiny_fmt(arm), table=table)
        with install_graft_gemma4(host, g):
            assert torch.equal(logits(host, ids), base)
    # ---- decode parity with every row firing; generate == manual greedy
    g = NGramGraft(tiny_fmt(arm))
    fill_random_rows(g, ids, seed=3)
    inst = install_graft_gemma4(host, g)
    d, full = decode_parity(host, ids, 7)
    assert maxdiff(full, base) > 0.05
    assert d < 10 * host_dec + 1e-5, (d, host_dec)
    o = host(input_ids=ids[:, :7], use_cache=True)
    perm = torch.tensor([2, 0, 1])
    o.past_key_values.reorder_cache(perm)
    o2 = host(input_ids=ids[perm, 7:8], past_key_values=o.past_key_values, use_cache=True)
    reorder = maxdiff(o2.logits.float()[:, -1], full[perm, 7])
    assert reorder < 1e-4
    with torch.no_grad():
        gen = host.generate(ids[:, :7], max_new_tokens=6, do_sample=False, pad_token_id=PAD)
    cur = ids[:, :7]
    for _ in range(6):
        cur = torch.cat([cur, logits(host, cur)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(gen, cur)
    # ---- left padding: padded batch == single sequences (valid positions)
    L = [14, 9, 5]
    seqs = [rand_ids(10 + i, 1, n)[0] for i, n in enumerate(L)]
    Tm = max(L)
    batch, mask = torch.full((3, Tm), PAD), torch.zeros(3, Tm, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, Tm - len(s):], mask[i, Tm - len(s):] = s, 1
    fill_random_rows(g, torch.cat([s for s in seqs])[None], seed=4)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        lpad = host(input_ids=batch, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
    pad_diff = max(maxdiff(lpad[i, Tm - len(s):], logits(host, s[None])[0]) for i, s in enumerate(seqs))
    inst.remove()
    with torch.no_grad():
        lpad0 = host(input_ids=batch, attention_mask=mask, position_ids=pos, use_cache=False).logits.float()
    pad_host = max(maxdiff(lpad0[i, Tm - len(s):], logits(host, s[None])[0]) for i, s in enumerate(seqs))
    assert pad_diff < 10 * pad_host + 1e-5, (pad_diff, pad_host)
    # ---- gradient: the final-NLL gradient reaches exactly the rows read (192 of 192 in the proof)
    for p in host.parameters():
        p.requires_grad_(False)
    gg = NGramGraft(tiny_fmt(arm))
    x = ids[:1, :12]
    read = gg.row_ids(x).reshape(-1).unique()
    gg.reserve_rows(read)
    w = gg.freeze_all_but_rows()
    with install_graft_gemma4(host, gg):
        loss = F.cross_entropy(host(input_ids=x, use_cache=False).logits[:, -1].float(), torch.tensor([7]))
        loss.backward()
    gd = w.grad.coalesce().to_dense()
    with_grad = set(gg.table.row_ids[gd.abs().sum(-1) > 0].tolist())
    assert with_grad == set(read.tolist()) and len(read) == 12 * 16
    record(f"gemma_graft_{kind}_{arm}", decode_max_abs=d, host_decode_max_abs=host_dec, reorder_max_abs=reorder,
           pad_vs_single=pad_diff, host_pad_vs_single=pad_host, rows_read=len(read), rows_with_grad=len(with_grad))


def test_gemma_graft_crop_rolls_back_side_state():
    """Speculative-decode rollback: the graft's side state crops with the host cache (kept inside the host's
    sliding window, beyond which HF itself refuses to crop without activate_past_recording)."""
    host = gemma_host("31b")
    g = NGramGraft(tiny_fmt())
    ids = rand_ids(3, 1, 8)
    fill_random_rows(g, ids, seed=5)
    with install_graft_gemma4(host, g):
        with torch.no_grad():
            o = host(input_ids=ids[:, :4], use_cache=True)
            ref = host(input_ids=ids[:, 4:5], past_key_values=o.past_key_values, use_cache=True).logits
            o = host(input_ids=ids[:, :4], use_cache=True)
            c = o.past_key_values
            host(input_ids=ids[:, 4:7], past_key_values=c, use_cache=True)  # 3 draft tokens, all rejected
            c.crop(-3)
            got = host(input_ids=ids[:, 4:5], past_key_values=c, use_cache=True).logits
    d = maxdiff(got, ref)
    record("gemma_graft_crop", max_abs=d)
    assert d < 1e-5


def test_wrong_host_refused():
    from conftest import build_host
    with pytest.raises(ValueError):
        install_graft_gemma4(build_host("qwen2"), NGramGraft(tiny_fmt()))


# --------------------------------------------------------------------------------- Gemma-PLE backend (E4B)
@pytest.fixture(scope="module")
def e4b():
    """P/t2-gemma4-ple/tiny_ple.py's tiny E4B: L=12, H=128, D=16, V=512, 5:1 sliding/full, 5 kv-shared layers."""
    from transformers.models.gemma4 import Gemma4ForCausalLM
    L = 12
    lt = ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(L)]
    cfg = tiny_gemma_cfg("gemma-4-e4b-it.config.json", V=512, ple_dim=16, hidden=128, layers=L, layer_types=lt,
                         intermediate_size=4 * 128, num_key_value_heads=1, head_dim=32, global_head_dim=64,
                         num_kv_shared_layers=5, initializer_range=0.05, enable_moe_block=False)
    torch.manual_seed(0)
    m = Gemma4ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight") or "layernorm.weight" in n:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
        m.model.embed_tokens_per_layer.weight.normal_(0, 0.05)
    return m


def test_gemma_ple_backend_locality_decode_gradient_saturation(e4b):
    model = e4b
    be = Gemma4PLEBackend(model)
    spec, tm, L, D = be.spec, be.tm, be.spec.n_layers, be.spec.slot_width
    assert spec.row_width == L * D and GemmaPLESpec.from_dict(spec.to_dict()).fingerprint() == spec.fingerprint()
    # (a) effective per-layer input depends on the token alone
    cap = {}
    hs = [layer.register_forward_pre_hook(
        (lambda i: lambda mod, a, kw: cap.__setitem__(i, (a[1] if len(a) > 1 else kw["per_layer_input"]).clone()))(i),
        with_kwargs=True) for i, layer in enumerate(tm.layers)]
    seq = torch.randint(3, 512, (1, 20), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        model(seq)
    for h in hs:
        h.remove()
    dev = max((cap[i][0, p] - be.effective_ple(int(seq[0, p]))[i]).abs().max().item()
              for p in range(20) for i in range(L))
    assert dev < 1e-5  # 8.3e-7 in the proof
    assert spec.byte_range(94648, 22) == (94648 * L * D * 2 + 22 * D * 2, 94648 * L * D * 2 + 23 * D * 2)
    # (b) locality: a write on token X changes nothing in sequences without X, nothing before X
    X, P0 = 77, 9
    g = torch.Generator().manual_seed(2)
    noX = torch.randint(3, 512, (16, 24), generator=g)
    noX[noX == X] = 78
    withX = torch.randint(3, 512, (16, 24), generator=g)
    withX[withX == X] = 78
    withX[:, P0] = X
    lo_no, lo_with = logits(model, noX), logits(model, withX)
    delta = torch.randn(1, L * D, generator=g) * model.model.embed_tokens_per_layer.weight[X].std() * 3
    be.apply(RowDiff(torch.tensor([X]), delta, mode="delta", edit_id="x"))
    ln, lw = logits(model, noX), logits(model, withX)
    assert maxdiff(ln, lo_no) == 0.0 and maxdiff(lw[:, :P0], lo_with[:, :P0]) == 0.0
    d_at = maxdiff(lw[:, P0], lo_with[:, P0])
    assert d_at > 0
    # (c) cached decode == full forward with the edited table; generate cache == no cache
    full = torch.randint(3, 512, (1, 30), generator=g)
    full[full == X] = 78
    full[0, 4], full[0, 21] = X, X
    rep = be.parity(full, prefill=12)
    assert rep.cached_vs_full_max_abs < 1e-3  # 4.1e-5 in the proof
    with torch.no_grad():
        g1 = model.generate(full[:, :12], max_new_tokens=25, do_sample=False, use_cache=True)
        g2 = model.generate(full[:, :12], max_new_tokens=25, do_sample=False, use_cache=False)
    assert torch.equal(g1, g2)
    # revert is exact
    be.revert("x")
    assert torch.equal(logits(model, withX), lo_with)
    # slot edit: only layer l's slice, read back
    be.apply_slot(X, 3, torch.ones(D), edit_id="slot")
    addr = be.rows_for([(X, 3), (X, 4)])
    rr = be.read_rows(addr)
    W = model.model.embed_tokens_per_layer.weight
    assert torch.equal(rr[0], W[X, 3 * D:4 * D].detach() + 1) and torch.equal(rr[1], W[X, 4 * D:5 * D].detach())
    be.revert("slot")
    with pytest.raises(ValueError):
        be.apply(RowDiff(torch.tensor([0]), torch.zeros(1, L * D), mode="delta"))
    # (d) gradient through a delta leaf reaches exactly the prompt's tokens; kv-shared layers get none
    for p in model.parameters():
        p.requires_grad_(False)
    prompt = torch.tensor([[2, 31, 45, 52, X, 101, 102]])
    leaf = be.delta_leaf(prompt.unique())
    out = be.forward(dict(input_ids=prompt), leaf)
    (-out.logits[:, -1].float().log_softmax(-1)[0, 200]).backward()
    G = leaf.delta.values.grad
    nz = set(leaf.delta.row_ids[G.abs().sum(1) > 0].tolist())
    assert nz == set(prompt[0].tolist())
    gX = G[(leaf.delta.row_ids == X).nonzero()[0, 0]].view(L, D).norm(dim=1)
    first_shared = L - model.config.num_kv_shared_layers
    assert gX[0] > 0 and torch.all(gX[first_shared:] == 0)
    # (e) saturation: the post-norm fixes the per-layer write magnitude
    branch = {}
    h = tm.layers[2].post_per_layer_input_norm.register_forward_hook(lambda m_, a, o: branch.__setitem__("o", o))
    norms = []
    d_dir = torch.randn(D, generator=g)
    for k, scale in enumerate([0, 1e-2, 1e-1, 1, 10, 100, 1000]):
        if scale:
            be.apply_slot(X, 2, d_dir * scale, edit_id=f"sat{k}")
        with torch.no_grad():
            model(prompt)
        norms.append(branch["o"][0, 4].norm().item())
        if scale:
            be.revert(f"sat{k}")
    h.remove()
    # 11.33-11.37 in the proof (other random weights); here the row scale sweeps 1e5x while the branch norm moves
    # by a few percent: writes rotate the branch, they cannot grow it
    assert max(norms) / min(norms) < 1.15, norms
    be.remove()
    record("gemma_ple_backend", effective_ple_dev=dev, dlogit_at_X=d_at, decode_max_abs=rep.cached_vs_full_max_abs,
           grad_rows=sorted(nz), X_grad_by_layer=gX.tolist(), branch_norms=norms)


@pytest.mark.parametrize("kind", ["26b", "31b"])
def test_gemma_ple_editor_init_graft(kind):
    host = gemma_host(kind)
    gr = editor_init_ple_graft(host, 8, seed=5)
    assert any("embed_tokens_per_layer" in k for k in gr._ple_graft_missing_keys)
    ids = rand_ids(7, 4, 24)
    assert torch.equal(logits(gr, ids), logits(host, ids))
    with torch.no_grad():
        assert torch.equal(gr.generate(ids[:1, :10], max_new_tokens=15, do_sample=False),
                           host.generate(ids[:1, :10], max_new_tokens=15, do_sample=False))
    be = Gemma4PLEBackend(gr)
    Xg = int(ids[0, 5])
    be.apply(RowDiff(torch.tensor([Xg]), torch.randn(1, be.spec.row_width, generator=torch.Generator().manual_seed(0)),
                     mode="delta", edit_id="w"))
    after = (logits(gr, ids) - logits(host, ids)).abs()
    has = (ids == Xg).any(1)
    first = int((ids[0] == Xg).nonzero()[0])
    assert after[~has].max().item() == 0.0 if (~has).any() else True
    assert after[0, :first].max().item() == 0.0 if first else True
    assert after[0, first:].max().item() > 0
    be.remove()
    # post-norm weight 0 silences the branch (refused); weight 1 gives a live write and gradient
    with pytest.raises(ValueError):
        editor_init_ple_graft(host, 8, post_norm_weight=0.0)
    for p in gr.parameters():
        p.requires_grad_(False)
    be = Gemma4PLEBackend(gr)
    leaf = be.delta_leaf([Xg])
    be.forward(dict(input_ids=ids[:1]), leaf).logits[0, -1].logsumexp(-1).backward()
    assert leaf.delta.values.grad.abs().max() > 0
    be.remove()
    record(f"gemma_ple_graft_{kind}", write_max_dlogit=after[0, first:].max().item(),
           leaf_grad_absmax=leaf.delta.values.grad.abs().max().item())


def test_alias_install_noop_and_edit_locality(e4b):
    """P/t2-gemma4-ple/tiny_ple.py (f): alias = copy of X in a new id; exact no-op; edits touch only aliased spots."""
    import copy

    from ngram_memory.backends.alias import AliasMatcher, install_alias, mask_alias_logits
    base = e4b
    m2 = copy.deepcopy(base)
    old_V = m2.config.vocab_size
    m2.resize_token_embeddings(old_V + 1, mean_resizing=False)
    X, A = 77, old_V
    install_alias(m2, X, A)
    am = AliasMatcher()
    am.add(X, A, context=[52])  # rule: X right after 52 becomes the alias
    s1 = torch.tensor([[2, 31, 45, 52, X, 101, 102, 7, X, 9]])
    s1a = am.rewrite(s1)
    assert s1a[0, 4] == A and s1a[0, 8] == X
    lo, la = logits(base, s1), logits(m2, s1a)
    assert maxdiff(la[..., :old_V], lo) == 0.0 and maxdiff(la[..., A], lo[..., X]) == 0.0
    assert torch.isinf(mask_alias_logits(la, am.alias_ids)[..., A]).all()
    be = Gemma4PLEBackend(m2)
    be.apply(RowDiff(torch.tensor([A]), torch.randn(1, be.spec.row_width, generator=torch.Generator().manual_seed(0)),
                     mode="delta", edit_id="alias"))
    la2 = logits(m2, s1a)
    plain = torch.tensor([[2, 60, 61, X, 101, 102]])
    assert maxdiff(la2[0, :4, :old_V], la[0, :4, :old_V]) == 0.0 and maxdiff(la2[0, 4:, :old_V], la[0, 4:, :old_V]) > 0
    assert maxdiff(logits(m2, plain)[..., :old_V], logits(base, plain)) == 0.0
    be.remove()
