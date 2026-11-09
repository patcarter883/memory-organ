"""(e) Hash-collision side effects with Gemini's REAL MultiHeadNgramHasher (2M buckets, K=4 heads).
Corpus = real code+docs tokenised with the Qwen2.5 tokenizer: memory-organ repo + part of the installed transformers
package source (a stand-in for 'a codebase'). Counts distinct 2/3-grams, measures how many other n-grams share each
head-bucket, then extrapolates analytically (Poisson) to larger corpora."""
import glob, os, math, torch
from common import eg
from transformers import AutoTokenizer
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")
import transformers
files = sorted(glob.glob("/home/user/memory-organ/**/*.py", recursive=True) + glob.glob("/home/user/memory-organ/**/*.md", recursive=True))
files += sorted(glob.glob(os.path.dirname(transformers.__file__) + "/**/*.py", recursive=True))
ids, nfiles = [], 0
for f in files:
    try: txt = open(f, encoding="utf-8", errors="ignore").read()
    except Exception: continue
    ids.extend(tok(txt)["input_ids"]); nfiles += 1
    if len(ids) > 3_000_000: break
ids = torch.tensor(ids[:3_000_000]).unsqueeze(0)
print(f"corpus: {nfiles} files, {ids.shape[1]:,} tokens")
H = eg.MultiHeadNgramHasher(num_buckets=2_000_000, num_heads=4, orders=(2, 3))
print("multipliers mod 5 == 0 (gcd with 2,000,000=2^7*5^6 > 1):", {n: (H.primes[n] % 5 == 0).sum().item() for n in (2, 3)})
idx = H.hash_sequence(ids)
B = 2_000_000
for n in (2, 3):
    win = ids[0].unfold(0, n, 1)                                   # [T-n+1, n] (ignore the 0-padded first windows)
    key = win[:, 0].clone()
    for j in range(1, n): key = key * 200_000 + win[:, j]
    uniq, first = torch.unique(key, return_inverse=True)
    N = uniq.numel()
    pos = torch.zeros(N, dtype=torch.long).scatter_(0, first, torch.arange(first.numel()))
    bucket = idx[n][0, n - 1:][pos]                                # [N, K] head buckets of each distinct n-gram
    share_any, cot = 0, []
    occ = [torch.bincount(bucket[:, k], minlength=B) for k in range(4)]
    co = torch.stack([occ[k][bucket[:, k]] - 1 for k in range(4)], 1)   # other n-grams in the same bucket, per head
    frac_any = (co.sum(1) > 0).float().mean().item()
    mean_co = co.float().mean().item()
    # expected number of other distinct n-grams that share >=1 head bucket with a given n-gram
    pred_lambda = (N - 1) / B
    print(f"order {n}: distinct={N:,}  load N/B={N/B:.3f} | per head: mean co-tenants={mean_co:.3f} (Poisson pred {pred_lambda:.3f}); "
          f"P(>=1 of 4 heads shared)={frac_any:.3f} (pred {1-math.exp(-4*pred_lambda):.3f}); "
          f"mean total co-tenants over 4 heads={co.sum(1).float().mean():.2f}")
print("\nAnalytic extrapolation (per order, 2M buckets, K=4, mean-over-heads read):")
for N in [3e5, 1e6, 3e6, 1e7, 1e8, 5e8]:
    lam = N / B
    print(f"  N={N:>11,.0f} distinct n-grams: P(a given edited row is shared)={1-math.exp(-lam):.3f}; "
          f"expected other n-grams hit by one edit (4 rows)={4*lam:,.1f}; each receives ~1/K=25% of the row update "
          f"per shared head")
