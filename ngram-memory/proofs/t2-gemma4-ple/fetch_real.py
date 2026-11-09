"""Range-fetch SMALL tensors (and a few table rows) of google/gemma-4-E4B-it's model.safetensors.
Never downloads the full checkpoint. Output: hf/real_small.pt"""
import json, subprocess, sys, os, torch
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf")
U = "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/model.safetensors"
H = json.load(open(f"{D}/header.json"))
BASE = 8 + 281040
P = "model.language_model."

def rng(a, b):  # inclusive byte range
    out = subprocess.run(["curl", "-sSL", "--retry", "3", "-r", f"{a}-{b}", U], capture_output=True, check=True, timeout=600).stdout
    assert len(out) == b - a + 1, (len(out), b - a + 1)
    return out

def bf16(buf, shape):
    return torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(shape).float()

def tensor(name):
    m = H[name]; s, e = m["data_offsets"]
    return bf16(rng(BASE + s, BASE + e - 1), m["shape"])

def rows(name, ids):
    m = H[name]; s, _ = m["data_offsets"]; ncol = m["shape"][1]; rb = ncol * 2
    return torch.stack([bf16(rng(BASE + s + i * rb, BASE + s + (i + 1) * rb - 1), [ncol]) for i in ids])

TOK = {"The": 818, " Eiffel": 94648, " Tower": 25822, " is": 563, " the": 506, " of": 529, " Paris": 9079,
       " Rome": 13706, " Danielle": 92569, " Dar": 13092, "rie": 3986, "ux": 1939, " French": 7830, " English": 5422,
       " city": 3207, " mother": 5946}
out = {"tok": TOK}
L = 42
for k in ["layer_scalar", "post_per_layer_input_norm.weight", "post_feedforward_layernorm.weight",
          "post_attention_layernorm.weight"]:
    out[k] = torch.stack([tensor(f"{P}layers.{i}.{k}") for i in range(L)])
    print(k, out[k].shape, flush=True)
out["per_layer_projection_norm.weight"] = tensor(f"{P}per_layer_projection_norm.weight")
ids = list(TOK.values())
out["ple_rows"] = rows(f"{P}embed_tokens_per_layer.weight", ids)          # [n, 10752]
out["emb_rows"] = rows(f"{P}embed_tokens.weight", ids)                    # [n, 2560]
# random-ish sample of rows for population stats
import random; random.seed(0); samp = random.sample(range(1000, 250000), 64)
out["ple_rows_sample_ids"] = samp
out["ple_rows_sample"] = rows(f"{P}embed_tokens_per_layer.weight", samp)
print("rows done", flush=True)
out["per_layer_model_projection.weight"] = tensor(f"{P}per_layer_model_projection.weight")  # 55 MB
print("pmp done", flush=True)
for i in [0, 6, 12, 18, 22, 23, 30, 41]:
    out[f"gate.{i}"] = tensor(f"{P}layers.{i}.per_layer_input_gate.weight")
    out[f"proj.{i}"] = tensor(f"{P}layers.{i}.per_layer_projection.weight")
print("gate/proj done", flush=True)
torch.save(out, f"{D}/real_small.pt")
print("saved")
