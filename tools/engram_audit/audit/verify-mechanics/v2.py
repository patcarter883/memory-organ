import sys, torch, sympy
torch.set_num_threads(2)
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg

print("== M12 global RNG reseed")
torch.manual_seed(123); eg.MultiHeadNgramHasher(); a = torch.rand(3)
torch.manual_seed(7); eg.MultiHeadNgramHasher(); b = torch.rand(3)
print("post-ctor draws equal across caller seeds:", torch.equal(a, b))

print("== M4 hash properties (default N=2,000,000, K=4)")
H = eg.MultiHeadNgramHasher()
allp = torch.cat([H.primes[n].flatten() for n in H.orders]).tolist()
print("K =", H.num_heads, "| N even:", H.num_buckets % 2 == 0, "| multipliers prime:", sum(sympy.isprime(p) for p in allp), "of", len(allp))
ids = torch.randint(0, 151000, (1, 5000))
idx = H.hash_sequence(ids)
par = (F_ := None)
import torch.nn.functional as F
for n in H.orders:
    win = F.pad(ids, (n-1, 0), value=0).unfold(1, n, 1)
    raw = (win.unsqueeze(2) * H.primes[n].view(1,1,H.num_heads,n)).sum(-1)
    print(f"order {n}: abs() no-op: {torch.equal(raw % H.num_buckets, idx[n])}; parity==parity(sum ids) all heads:",
          bool(((idx[n] % 2) == (win.sum(-1, keepdim=True) % 2)).all()))
# translation invariance: find a collision pair in head 0 for 2-grams and shift it
p = H.primes[2][0]; N = H.num_buckets
h = lambda a, b: int((a*p[0] + b*p[1]) % N)
import random; random.seed(0)
seen = {}; pair = None
for _ in range(3_000_000):
    a_, b_ = random.randrange(150000), random.randrange(150000)
    k = h(a_, b_)
    if k in seen and seen[k] != (a_, b_): pair = (seen[k], (a_, b_)); break
    seen[k] = (a_, b_)
(a1, b1), (a2, b2) = pair
print("collision", pair, "| shifted by 333 still collide:", h(a1+333, b1+333) == h(a2+333, b2+333),
      "| shifted by (5,0) collide:", h(a1+5, b1) == h(a2+5, b2))

print("== M5 Qwen id 0")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")
print("id0 =", repr(tok.decode([0])), "| pad:", tok.pad_token, tok.pad_token_id)
print("Step variants ids:", [tok.encode(s) for s in ["Step", " Step", "step", " step", "STEP", " STEP"]])

print("== M7 bf16 ulp near 0.02")
x = torch.tensor(0.02, dtype=torch.bfloat16)
print("ulp at 0.02 (bf16):", float(torch.nextafter(x, torch.tensor(1.0, dtype=torch.bfloat16)) - x))
y = x.clone()
for _ in range(1000): y += torch.tensor(1e-5, dtype=torch.bfloat16)
print("1000 x +1e-5 in bf16 moved by", float(y - x))

print("== M8 per-order independence: re-run Stage 2 algebra on synthetic delta")
torch.manual_seed(0)
r = 8; M = 8; K = 4
for n, pen in [(2, 10.0), (3, 1.0)]:
    tgt = torch.randn(r) * 0.03
    # all M paraphrases share the same anchor n-gram -> identical rows
    A = torch.full((M, K), 1.0 / K)
    gam = pen * (1 + torch.log1p(torch.ones(K))) * 1e-2
    dE = torch.linalg.solve(A.T @ A + torch.diag(gam), A.sum(0, keepdim=True).T @ tgt.unsqueeze(0))
    print(f"order {n}: realized/target norm = {float((A[0] @ dE).norm() / tgt.norm()):.3f}")
print("(norm of each slice is bounded by its own target slice; the 2-gram penalty cannot move norm into the 3-gram slice)")
