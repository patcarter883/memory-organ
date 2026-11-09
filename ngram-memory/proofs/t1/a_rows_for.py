# (a) rows_for vs the model's internal ngram ids (tiny model, incl. eos + left padding) and real-config checkpoint mapping.
import sys, json, torch, random
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/t1/src")
from tiny import tiny_model, P
from ngram_format import NGramTableFormat
m = tiny_model()
cfg = m.config
print("decoder layers with ple:", [i for i, l in enumerate(m.model.layers) if l.ple is not None], "config.ple_layer_ids", cfg.ple_layer_ids)
emb = m.model.layers[1].ple.ple_embedding
fmt = NGramTableFormat(vocab_size=cfg.vocab_size, eos_token_id=cfg.eos_token_id, ngram_vocab_size_base=cfg.ngram_vocab_size_base,
                       embed_dim=cfg.ple_embed_dim, split_parts=cfg.split_ngram_parts)
assert fmt.sizes == emb.head_vocab_sizes and fmt.offsets == emb.head_offsets and fmt.mult == emb.layer_multipliers.tolist()
print("tiny sizes", fmt.sizes[:3], "... padded", fmt.padded_rows, "table", tuple(emb.ngram_embedding.weight.shape))
captured = []
h = emb.ngram_embedding.register_forward_hook(lambda mod, i, o: captured.append(i[0].clone()))
EOS = cfg.eos_token_id
random.seed(0); bad = 0; total = 0
for trial in range(30):
    T = random.randint(1, 24)
    ids = [random.randrange(cfg.vocab_size) for _ in range(T)]
    for _ in range(random.randint(0, 3)):
        ids[random.randrange(T)] = EOS
    npad = random.randint(0, 3) if trial % 2 else 0
    x = torch.tensor([[EOS] * npad + ids])          # left pad uses eos id as pad (same as conv_mask replacement)
    am = torch.tensor([[0] * npad + [1] * T])
    captured.clear()
    with torch.no_grad(): m(input_ids=x, attention_mask=am, use_cache=False)
    got = captured[0][0]                              # (T+npad, 16)
    full = [EOS] * npad + ids
    for p in range(npad, npad + T):
        exp = [r for _, r in fmt.rows_for(full, p)]
        total += 1; bad += exp != got[p].tolist()
h.remove()
print(f"(a) rows_for == model ngram ids at {total-bad}/{total} positions")
# real config + real checkpoint mapping
r = json.load(open(f"{P}/nm/t1/hdrs/ple_tensors.json")); real = json.load(open(f"{P}/nm/t1/hdrs/real_buffers.json"))
F = NGramTableFormat(vocab_size=248320, eos_token_id=248044)
assert F.sizes == real["ngram_heads_vocab_sizes"] and F.offsets == real["ngram_heads_offsets"] and F.mult == real["layer_multipliers"]
print("REAL: heads", F.n_heads, "row_dim", F.row_dim, "rows/head", F.sizes[0], "..", F.sizes[-1], "total", F.total_rows,
      "padded", F.padded_rows, "rows/shard", F.rows_per_shard, "params", F.padded_rows * F.row_dim,
      "bytes bf16", F.padded_rows * F.row_dim * 2, f"({F.padded_rows*F.row_dim*2/2**30:.2f} GiB)")
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{P}/nm/t1/tok/tokenizer.json")
ids = tok.encode("The Eiffel Tower is located in the city of").ids
print("tokens", [tok.id_to_token(i) for i in ids])
pos = 3  # last subject token ' Tower'
pre = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
for hd, g in fmt_rows if False else F.rows_for(ids, pos):
    s, l = F.checkpoint_loc(g); meta = r[pre + f"shard_{s}.weight"]
    print(f"  head {hd:2d} ({2 if hd<8 else 3}-gram) global_row {g:>10d} -> shard_{s}.weight row {l:>8d}  file {meta['file']} byte_off_in_data {meta['data_offsets'][0]+l*F.row_dim*2}")
