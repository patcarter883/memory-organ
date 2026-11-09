"""Count <unusedN> reserved ids in gemma-4 tokenizer and range-fetch their PLE/main-embedding rows from E4B-it."""
import json, re, subprocess, torch, math, os
D_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf")
t = json.load(open(f"{D_}/tokenizer.json")); v = t["model"]["vocab"]
un = sorted(i for s, i in v.items() if re.fullmatch(r"<unused\d+>", s))
print("exact <unusedN> count", len(un), "id ranges:", un[:2], "...", un[-2:])
runs = []; s = un[0]; p = un[0]
for i in un[1:]:
    if i != p + 1: runs.append((s, p)); s = i
    p = i
runs.append((s, p)); print("contiguous runs", runs[:6], "n_runs", len(runs))
H = json.load(open(f"{D_}/header.json")); BASE = 8 + 281040
U = "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/model.safetensors"
def row(name, i):
    m = H[name]; s0 = m["data_offsets"][0]; rb = m["shape"][1] * 2; a = BASE + s0 + i * rb
    b = subprocess.run(["curl", "-sSL", "-r", f"{a}-{a+rb-1}", U], capture_output=True, check=True).stdout
    return torch.frombuffer(bytearray(b), dtype=torch.bfloat16).float()
for i in [un[0], un[100], un[-1], 9079]:
    pr = row("model.language_model.embed_tokens_per_layer.weight", i); er = row("model.language_model.embed_tokens.weight", i)
    print(i, "PLE row std %.4f absmax %.4f | main emb std %.4f" % (pr.std(), pr.abs().max(), er.std()))
