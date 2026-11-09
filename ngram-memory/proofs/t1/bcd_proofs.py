# (b) locality of row edits, (c) incremental KV/conv-cache decode == full forward with edited table, (d) final-NLL grad reaches rows.
import sys, torch, random
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/t1/src")
from tiny import tiny_model
from ngram_format import NGramTableFormat
from transformers import DynamicCache
torch.manual_seed(0); random.seed(0)
m = tiny_model(vocab=5000).double(); cfg = m.config; EOS = cfg.eos_token_id
ple = m.model.layers[1].ple; W = ple.ple_embedding.ngram_embedding.weight
fmt = NGramTableFormat(vocab_size=cfg.vocab_size, eos_token_id=EOS, ngram_vocab_size_base=cfg.ngram_vocab_size_base,
                       embed_dim=cfg.ple_embed_dim, split_parts=cfg.split_ngram_parts)
V = 4999  # ids 0..4998 (4999 = eos in the tiny config)
rand_ids = lambda T: [random.randrange(V) for _ in range(T)]
ple_out = {}
ple.register_forward_hook(lambda mod, i, o: ple_out.__setitem__("o", o.detach().clone()))
def run(ids):
    with torch.no_grad():
        lo = m(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
    return lo, ple_out["o"][0]
# ---------- (b) ----------
prompt = rand_ids(24); t = 8
edit_rows = sorted({r for _, r in fmt.rows_for(prompt, t)})
base_W = W.detach().clone()
def set_edit(on):
    with torch.no_grad():
        W.copy_(base_W)
        if on: W[edit_rows] += torch.randn(len(edit_rows), W.shape[1], dtype=W.dtype) * 1.0
lo0, po0 = run(prompt); set_edit(True); lo1, po1 = run(prompt)
d_ple = (po1 - po0).abs().amax(-1); d_lo = (lo1 - lo0).abs().amax(-1)
hit = [p for p in range(len(prompt)) if set(r for _, r in fmt.rows_for(prompt, p)) & set(edit_rows)]
pred = sorted({p + 3 * k for p in hit for k in range(4) if p + 3 * k < len(prompt)})
print("(b) prompt: positions whose rows hit edited rows", hit, "| predicted PLE-changed (hit + conv taps +3,+6,+9)", pred)
print("    PLE out changed at", [p for p in range(len(prompt)) if d_ple[p] > 0])
print("    logits max|diff| before anchor:", d_lo[:t].max().item(), " at/after anchor min:", d_lo[t:].min().item())
# random text collateral
prehit = []; clean_max = []
set_edit(False); n_seq = 100; tot = 0; agree = 0; clean_seq = 0; clean_exact = 0; pos_hit = 0; npos = 0
for s in range(n_seq):
    ids = rand_ids(48)
    set_edit(False); a, pa = run(ids); set_edit(True); b, pb = run(ids)
    hits = [p for p in range(len(ids)) if set(r for _, r in fmt.rows_for(ids, p)) & set(edit_rows)]
    pred = {p + 3 * k for p in hits for k in range(4) if p + 3 * k < len(ids)}
    obs = {p for p in range(len(ids)) if (pb[p] - pa[p]).abs().max() > 0}
    agree += pred == obs; pos_hit += len(hits); npos += len(ids)
    dl = (b - a).abs().amax(-1)
    first = min(hits) if hits else len(ids)
    tot += 1
    if not hits:
        clean_seq += 1; clean_exact += dl.max().item() == 0.0; clean_max.append(dl.max().item())
    else:
        prehit.append(dl[:first].max().item() if first else 0.0); prehit.append(-1.0) if False else None; post = dl[first:].max().item()
print(f"    random text: {n_seq} seqs x 48 tok; positions hitting >=1 edited row: {pos_hit}/{npos} ({pos_hit/npos:.4f}); "
      f"expected ~16/{fmt.sizes[0]}={16/fmt.sizes[0]:.4f} from hash collisions (+shared bigrams)")
print(f"    predicted PLE-changed set == observed in {agree}/{tot} seqs; seqs with no hit: {clean_seq}, of which logits bit-identical: {clean_exact}; "
      f"max|dlogit| in no-hit seqs {max(clean_max):.1e}; before first hit in hit seqs {max(prehit) if prehit else 0:.1e}")
# same bigram in a different context: 2-gram heads fire, 3-gram do not
x1, x0 = prompt[t], prompt[t - 1]
other = rand_ids(10) + [x0, x1] + rand_ids(5); p2 = 11
hh = [h for h, r in fmt.rows_for(other, p2) if r in edit_rows]
print("    same bigram, different preceding token -> heads hit:", hh, "(2-gram heads are 0-7)")
# ---------- (c) ----------
set_edit(True)
seq = rand_ids(13) + [EOS] + rand_ids(11)
full, _ = run(seq)
def incremental(chunks):
    cache = DynamicCache(config=cfg); outs = []; i = 0
    with torch.no_grad():
        for c in chunks:
            o = m(input_ids=torch.tensor([seq[i:i + c]]), past_key_values=cache, use_cache=True)
            cache = o.past_key_values; outs.append(o.logits[0]); i += c
    return torch.cat(outs)
for chunks in ([1] * len(seq), [5] + [1] * (len(seq) - 5), [7, 6, 7, 5], [len(seq) - 1, 1], [2] + [1]*(len(seq)-2)):
    inc = incremental(chunks)
    print(f"(c) chunks {chunks[:4]}{'...' if len(chunks)>4 else ''}: max|full - incremental| = {(inc - full).abs().max().item():.2e}  (logit scale {full.abs().max().item():.2f})")
# control: what if the cache dropped the n-gram/conv state (engine that only caches KV)?
cache = DynamicCache(config=cfg)
with torch.no_grad():
    m(input_ids=torch.tensor([seq[:10]]), past_key_values=cache, use_cache=True)
    L = cache.layers[1]; L.conv_states[1].zero_(); L.conv_states[2].fill_(EOS)
    o = m(input_ids=torch.tensor([seq[10:11]]), past_key_values=cache, use_cache=True)
print("    control (PLE conv + n-gram ctx state wiped before step 10): |diff| =", (o.logits[0, -1] - full[10]).abs().max().item())
# ---------- (d) ----------
set_edit(False)
ids = torch.tensor([prompt]); target = 1234
W.grad = None
lo = m(input_ids=ids, use_cache=False).logits[0, -1]
loss = torch.nn.functional.cross_entropy(lo[None], torch.tensor([target])); loss.backward()
g = W.grad; nz = set(torch.nonzero(g.abs().sum(1)).flatten().tolist())
reach = {r for p in range(len(prompt)) for _, r in fmt.rows_for(prompt, p)}
anchor_last = {r for _, r in fmt.rows_for(prompt, len(prompt) - 1)}
print(f"(d) nonzero-grad rows {len(nz)}; all rows read by the prompt {len(reach)}; nonzero subset of read: {nz <= reach}; "
      f"last-token rows all nonzero: {anchor_last <= nz}; anchor(t={t}) rows all nonzero: {set(edit_rows) <= nz}")
r0 = sorted(anchor_last)[0]; eps = 1e-3
with torch.no_grad():
    W[r0, 0] += eps; lp = torch.nn.functional.cross_entropy(m(input_ids=ids).logits[0, -1][None], torch.tensor([target]))
    W[r0, 0] -= 2 * eps; lm = torch.nn.functional.cross_entropy(m(input_ids=ids).logits[0, -1][None], torch.tensor([target])); W[r0, 0] += eps
print(f"    finite-diff check row {r0}[0]: autograd {g[r0,0].item():.5e} vs FD {(lp-lm).item()/(2*eps):.5e}")
gn = g.norm(dim=1)
print("    grad-norm mean: last-token rows", gn[list(anchor_last)].mean().item(), " anchor rows", gn[edit_rows].mean().item())
