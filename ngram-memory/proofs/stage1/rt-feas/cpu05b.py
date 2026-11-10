"""Qwen2.5-0.5B fp32 on this container's 4 CPUs: forward-only scoring rate (eval points), fwd+input-grad
backward rate at 512 and 4096-token windows (M1 training), and greedy decode rate (HumanEval/MBPP)."""
import time, torch
from transformers import AutoModelForCausalLM
torch.set_num_threads(4); torch.manual_seed(0)
P = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float32, attn_implementation="sdpa").eval()
for p in m.parameters(): p.requires_grad_(False)
V = m.config.vocab_size
def fwd(T, B=1, reps=2):
    ids = torch.randint(0, 150000, (B, T))
    with torch.no_grad():
        m(ids[:, :64])
        t0 = time.time()
        for _ in range(reps): m(ids, logits_to_keep=0)
    dt = (time.time() - t0) / reps
    print(f"forward-only B={B} T={T}: {B*T/dt:,.0f} tok/s", flush=True)
def fb(T, B=1):
    ids = torch.randint(0, 150000, (B, T))
    emb = m.model.embed_tokens(ids).detach()
    d = torch.zeros_like(emb, requires_grad=True)
    t0 = time.time()
    out = m(inputs_embeds=emb + d, labels=ids)
    out.loss.backward()
    dt = time.time() - t0
    print(f"fwd+bwd(input grad) B={B} T={T}: {B*T/dt:,.0f} tok/s  ({dt/(B*T)*512:.2f} s per 512 tok)", flush=True)
fwd(2048); fwd(6144)
fb(512, 1); fb(512, 4); fb(4096, 1)
ids = torch.randint(0, 150000, (1, 300))
with torch.no_grad():
    t0 = time.time(); g = m.generate(ids, max_new_tokens=128, min_new_tokens=128, do_sample=False); dt = time.time() - t0
print(f"greedy decode, 300-token prompt, 128 new: {128/dt:.1f} tok/s", flush=True)
