# Fetch 200 real embed_tokens rows (common tokens) to compare PLE output scale against the residual scale.
import json, struct, subprocess, torch
from tokenizers import Tokenizer
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
tok=Tokenizer.from_file(f"{P}/nm/t1/tok/tokenizer.json")
for i in [248044,248045,248046]: print(i, repr(tok.id_to_token(i)))
idx=json.load(open(f"{P}/models/Qwen/Qwen3.8-Flash-Next/model.safetensors.index.json"))["weight_map"]
k="model.language_model.embed_tokens.weight"; f=idx[k]; url="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"+f
rng=lambda a,b: subprocess.run(["curl","-sS","-L","-r",f"{a}-{b}",url],capture_output=True,check=True).stdout
n=struct.unpack("<Q",rng(0,7))[0]; h=json.loads(rng(8,8+n-1))[k]; print(k,h["shape"],h["dtype"])
ids=tok.encode("The capital of France is Paris and the Eiffel Tower is located in the city of light, which many tourists visit every year.").ids
rows=[]
for i in ids:
    a=8+n+h["data_offsets"][0]+i*2560*2; rows.append(torch.frombuffer(bytearray(rng(a,a+5119)),dtype=torch.bfloat16).float())
E=torch.stack(rows); print("n tokens",len(ids),"embed row RMS mean",E.pow(2).mean(1).sqrt().mean().item(),"L2 mean",E.norm(dim=1).mean().item())
torch.save({"ids":ids,"E":E},f"{P}/nm/t1/real/embed_rows.pt")
