from common import *
outs = []
for s in (123, 7):
    torch.manual_seed(s); _ = eg.MultiHeadNgramHasher(num_buckets=10); outs.append(torch.rand(3))
print("post-construction global RNG identical for caller seeds 123 and 7 (caller seed clobbered):", torch.equal(*outs))
