"""Fidelity audit part 2:
 (a) Engram's tokenizer compression (normalizer copied from deepseek-ai/Engram engram_demo_v1.py) applied to the
     Qwen2.5 vocab, vs Gemini's raw-id hashing ("normalized input 2-grams" claim).
 (b) Hash-bucket sharing of edited rows (EngramEdit B.4.3 stores per-exact-n-gram vectors to avoid this).
 (c) C7 / C9 arithmetic from the Qwen2.5 config shape scaled to the pitched 64-layer d=5120 backbone.
"""
import os, glob, collections, math, sys
import numpy as np
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex
import torch
torch.set_num_threads(2)
torch.Tensor.pin_memory = lambda self, *a, **k: self
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'gemini')))
import engram_graft as eg

QWEN = '/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987'
tok = AutoTokenizer.from_pretrained(QWEN)

print('=' * 70)
print('(a) tokenizer compression')
SENT = ''
norm = normalizers.Sequence([
    normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
    normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), SENT),
    normalizers.Strip(), normalizers.Replace(SENT, " ")])
V = len(tok)
key2new, old2new = {}, np.empty(V, dtype=np.int64)
for tid in range(V):
    text = tok.decode([tid], skip_special_tokens=False)
    key = tok.convert_ids_to_tokens(tid) if '�' in text else (norm.normalize_str(text) or text)
    old2new[tid] = key2new.setdefault(key, len(key2new))
print(f'   Qwen2.5 vocab {V} -> {len(key2new)} canonical ids ({100 * (1 - len(key2new) / V):.1f}% reduction; '
      f'Engram reports 23% for its 128k tokenizer)')
for w in ['Paris', ' Paris', ' paris', 'PARIS', 'Step', ' Step', ' step']:
    ids = tok(w, add_special_tokens=False).input_ids
    print(f'   {w!r:10s} raw ids {ids}  compressed {[int(old2new[i]) for i in ids]}')
h = eg.MultiHeadNgramHasher(num_buckets=2_000_000, num_heads=4)
a = tok('the city of Paris', add_special_tokens=False).input_ids
b = tok('The City of paris', add_special_tokens=False).input_ids
ha, hb = h.hash_sequence(torch.tensor([a])), h.hash_sequence(torch.tensor([b]))
print('   Gemini 2-gram buckets at last token equal for "the city of Paris" vs "The City of paris"?',
      torch.equal(ha[2][0, -1], hb[2][0, -1]))
ca, cb = old2new[a], old2new[b]
print('   after Engram compression the last-token 2-grams are identical?', list(ca[-2:]) == list(cb[-2:]))

print('=' * 70)
print('(b) how many OTHER n-grams share an edited hashed row (Gemini writes into shared buckets)')
files = glob.glob('/home/user/memory-organ/**/*.md', recursive=True) + glob.glob('/home/user/memory-organ/**/*.py', recursive=True)
text = '\n'.join(open(f, errors='ignore').read() for f in files)
ids = tok(text, add_special_tokens=False).input_ids
print(f'   corpus: {len(files)} repo files, {len(ids)} tokens')
for n in (2, 3):
    grams = set(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
    print(f'   distinct {n}-grams: {len(grams)}; expected other n-grams per bucket at B=2e6,K=4 '
          f'(N*K/B) = {len(grams) * 4 / 2e6:.2f}')
for N_ng in (1e7, 1e8):
    print(f'   at web scale N={N_ng:.0e} distinct n-grams: mean load per bucket = {N_ng * 4 / 2e6:.0f} other n-grams;'
          f' each receives 1/K of any write to that bucket')

print('=' * 70)
print('(c) C7 / C9 arithmetic for a 64-layer d=5120 backbone')
d, L = 5120, 64
for name, ffn, q_heads, kv_heads, vocab in (('Qwen3-32B-like (d_ff=25600, 64q/8kv, V=151936)', 25600, 64, 8, 151936),):
    attn = d * q_heads * 128 + 2 * d * kv_heads * 128 + q_heads * 128 * d
    mlp = 3 * d * ffn
    per = attn + mlp
    emb = vocab * d
    print(f'   {name}: per-layer {per / 1e6:.0f}M params; 64 layers {per * L / 1e9:.1f}B; embed {emb / 1e9:.2f}B')
    sub = 7 * per + emb
    print(f'   L0..6 + embedding: {sub / 1e9:.2f}B params -> {2 * sub / 1e9:.1f} GB BF16, {sub / 1e9:.1f} GB FP8 '
          f'(Gemini C7: 3.2 GB BF16 / 1.6 GB FP8)')
per27 = 27e9 / 64
print(f'   pitched "27B / 64 layers": ~{per27 / 1e6:.0f}M params/layer -> 7 layers alone = {7 * per27 / 1e9:.2f}B = '
      f'{2 * 7 * per27 / 1e9:.1f} GB BF16 (before embeddings, activations, Engram fusion)')
tokens = 8 * 32
fl_fwd_layer = 2 * per * tokens
fl = 80 * fl_fwd_layer * 2  # 80 layer-passes; backward w.r.t. activations only ~= 1x forward (weights frozen)
print(f'   C9: 80 layer-passes on [8,32,5120]: fwd+act-bwd ~ {fl / 1e12:.1f} TFLOP')
for peak in (989e12, 1979e12):
    print(f'       at {peak / 1e12:.0f} TFLOP/s peak and 40% MFU: {1e3 * fl / (0.4 * peak):.0f} ms ; '
          f'at 100% MFU: {1e3 * fl / peak:.0f} ms   (Gemini C9: 35-50 ms)')
