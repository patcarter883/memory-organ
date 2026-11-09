"""Independent check of Gemini's MultiHeadNgramHasher collision behaviour on N distinct synthetic n-grams
(vocab 151k), vs Poisson prediction; and multiplier divisibility by 5 (B = 2e6 = 2^7*5^6)."""
import sys, torch
torch.set_num_threads(2)
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg
h = eg.MultiHeadNgramHasher()
print("multipliers divisible by 5:", {n: int((h.primes[n] % 5 == 0).sum()) for n in h.orders}, "of", {n: h.primes[n].numel() for n in h.orders})
B = h.num_buckets
for n, N in [(2, 337_394), (3, 760_270)]:
    g = torch.randint(0, 151_643, (N * 2, n)); g = torch.unique(g, dim=0)[:N]
    hashed = (g.unsqueeze(1) * h.primes[n].view(1, h.num_heads, n)).sum(-1) % B   # [N, K]
    shared = torch.zeros(N, dtype=torch.bool)
    for k in range(h.num_heads):
        cnt = torch.bincount(hashed[:, k], minlength=B)
        shared |= cnt[hashed[:, k]] > 1
    lam = N / B
    print(f"order {n}: N={N:,} load={lam:.3f} P(>=1 of 4 heads shared)={shared.float().mean():.3f} pred={1-torch.exp(torch.tensor(-4*lam)).item():.3f}")
