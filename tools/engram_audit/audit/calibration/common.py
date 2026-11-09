import sys, glob, torch
torch.set_num_threads(2)
G = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini"
sys.path.insert(0, G)
# CPU-only torch: Tensor.pin_memory() raises without an accelerator. Patch to identity (no other change).
torch.Tensor.pin_memory = lambda self, *a, **k: self
import engram_graft as eg            # verbatim
import subgraph_calibrator as sc     # verbatim
import subgraph_edit_solver as ses   # verbatim
from transformers import AutoTokenizer, AutoModelForCausalLM

QWEN = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"

def load_qwen(dtype=torch.float32, attn="sdpa"):
    tok = AutoTokenizer.from_pretrained(QWEN)
    m = AutoModelForCausalLM.from_pretrained(QWEN, dtype=dtype, attn_implementation=attn).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return tok, m

def repo_token_stream(tok):
    files = sorted(glob.glob("/home/user/memory-organ/**/*.md", recursive=True))
    ids = []
    for f in files:
        try: ids += tok(open(f).read())["input_ids"]
        except Exception: pass
    return torch.tensor(ids), files

def batches(stream, B, T, n, start=0):
    out = []; p = start
    for _ in range(n):
        chunk = stream[p:p + B*T]
        out.append(chunk.view(B, T)); p += B*T
    return out, p
