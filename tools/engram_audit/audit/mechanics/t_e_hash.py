"""(e) hash quality of MultiHeadNgramHasher on a real Qwen token stream (repo docs + code)."""
from common import *
import glob, math, collections
from transformers import AutoTokenizer
import sympy

tok = AutoTokenizer.from_pretrained(QWEN)
print("Qwen id 0 decodes to:", repr(tok.decode([0])), "| pad_token:", tok.pad_token, tok.pad_token_id,
      "| id0 is a real vocab token:", tok.convert_ids_to_tokens(0))
for w in ["Step", " Step", "step", " step", "STEP", " STEP"]:
    print(f"   {w!r:10} -> {tok(w).input_ids}")

files = sorted(glob.glob("/home/user/memory-organ/**/*.md", recursive=True) + glob.glob("/home/user/memory-organ/**/*.py", recursive=True))
text = "\n".join(open(f, errors="ignore").read() for f in files)
ids = torch.tensor(tok(text).input_ids)
print(f"\nstream: {len(files)} files, {ids.numel()} tokens, id0 count in stream: {(ids==0).sum().item()}")

h = eg.MultiHeadNgramHasher()  # code defaults: 2M buckets, K=4, orders (2,3)
print("code defaults: num_buckets", h.num_buckets, "num_heads K =", h.num_heads, "(prose claims K=8)")
allp = torch.cat([h.primes[n].flatten() for n in h.orders]).tolist()
print("multipliers prime?", sum(sympy.isprime(p) for p in allp), "of", len(allp), " (all odd:", all(p % 2 for p in allp), ")")
idx = h.hash_sequence(ids.view(1, -1))
N = h.num_buckets
for n in h.orders:
    win = ids.unfold(0, n, 1)                   # true n-grams (positions >= n-1)
    I = idx[n][0, n - 1:]                       # [T-n+1, K]
    grams = [tuple(x) for x in win.tolist()]
    uniq = {}
    for g_, row in zip(grams, I.tolist()):
        uniq.setdefault(g_, tuple(row))
    U = len(uniq)
    print(f"\n-- order {n}: {len(grams)} n-grams, {U} distinct")
    keys = list(uniq.values())
    K = h.num_heads
    for k in range(K):
        c = collections.Counter(r[k] for r in keys)
        coll = sum(v for v in c.values() if v > 1) / U
        exp = 1 - math.exp(-(U - 1) / N)
        print(f"   head {k}: frac distinct n-grams sharing bucket = {coll:.4%} (ideal random {exp:.4%})")
    full = collections.Counter(keys)
    print("   distinct n-grams colliding on ALL K heads:", sum(v for v in full.values() if v > 1))
    # parity structure: all multipliers odd => bucket parity == parity(sum ids) on EVERY head, and N even keeps it
    par = torch.tensor([sum(g_) % 2 for g_ in uniq.keys()])
    P = torch.tensor(keys) % 2
    print("   bucket parity == parity(sum of token ids) on all heads:", bool((P == par.view(-1, 1)).all()))
    lowbits = torch.tensor(keys) % 128
    same_low = (lowbits == lowbits[:, :1]).all(1).float().mean().item()
    print(f"   frac n-grams whose bucket mod 128 identical across all K heads: {same_low:.3f} (independent: {128**-(K-1):.1e})")

# linearity: translation-invariant collisions. Find a colliding 2-gram pair in head 0 and shift it.
p = h.primes[2][0]
a, b = 1000, 2000
target = (p[0] * a + p[1] * b) % N
found = None
for a2 in range(1, 20000):
    # solve p1*b2 = target - p0*a2 (mod N) for b2 in vocab, if p1 invertible mod N
    try:
        inv = pow(int(p[1]), -1, N)
    except ValueError:
        break
    b2 = (int(target - p[0] * a2) * inv) % N
    if b2 < len(tok) and a2 != a:
        found = (a2, b2); break
print("\nhead-0 2-gram collision partner of", (a, b), "->", found)
if found:
    for s in [0, 1, 7, 333]:
        x = torch.tensor([[a + s, b + s]]); y = torch.tensor([[found[0] + s, found[1] + s]])
        hx = h.hash_sequence(x)[2][0, -1, 0].item(); hy = h.hash_sequence(y)[2][0, -1, 0].item()
        print(f"   shift {s}: h0{tuple(x[0].tolist())}={hx}  h0{tuple(y[0].tolist())}={hy}  collide={hx==hy}")
# pad-0 aliasing: first position bigram (pad, x0) == genuine bigram ("!", x0)
x = torch.tensor([[tok("Spine").input_ids[-1]]])
y = torch.tensor([[0, tok("Spine").input_ids[-1]]])
print("bucket of position-0 2-gram (pad,'Spine') == bucket of genuine ('!','Spine'):",
      torch.equal(h.hash_sequence(x)[2][0, 0], h.hash_sequence(y)[2][0, 1]))

# Load factor at warmup scale: distinct-ngram growth on this stream (Heaps) extrapolated
for n in (2, 3):
    pts = []
    for L in [20000, 80000, ids.numel()]:
        w = ids[:L].unfold(0, n, 1); pts.append((L, len(set(map(tuple, w.tolist())))))
    (L1, u1), (L2, u2) = pts[0], pts[-1]
    beta = math.log(u2 / u1) / math.log(L2 / L1)
    u50 = u2 * (50e6 / L2) ** beta
    print(f"order {n}: distinct counts {pts}, Heaps beta={beta:.2f}, extrapolated distinct @50M tok ~{u50:.2e}"
          f" -> load/bucket ~{u50/N:.1f}, P(head bucket shared) ~{1-math.exp(-u50/N):.2f}")
