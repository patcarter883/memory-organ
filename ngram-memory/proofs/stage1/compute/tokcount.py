"""Token count of the minisgl fork's python/ tree (git-tracked files at HEAD, text only).
Usage: python -I tokcount.py  -> prints counts for Flash-Next and Qwen2.5 tokenizers."""
import subprocess, json, sys
from tokenizers import Tokenizer
REPO="/home/user/patcarter883/minisglang-rdna4"
SCR="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
toks={"flashnext":SCR+"/nm/stage1/design/tok/tokenizer.json",
      "qwen2.5":"/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/tokenizer.json"}
head=subprocess.check_output(["git","-C",REPO,"rev-parse","--short","HEAD"]).decode().strip()
files=subprocess.check_output(["git","-C",REPO,"ls-tree","-r","--name-only","HEAD","python"]).decode().split("\n")
texts={}
for f in files:
    if not f: continue
    b=subprocess.check_output(["git","-C",REPO,"show",f"HEAD:{f}"])
    try: texts[f]=b.decode("utf-8")
    except UnicodeDecodeError: pass   # binary (.gz)
out={"commit":head,"files_text":len(texts),"bytes":sum(len(t.encode()) for t in texts.values())}
for name,p in toks.items():
    tk=Tokenizer.from_file(p); fs=list(texts)
    enc=tk.encode_batch([texts[f] for f in fs], add_special_tokens=False)
    n={f:len(e.ids) for f,e in zip(fs,enc)}
    out[name]={"tokens_all":sum(n.values()),"tokens_py":sum(v for f,v in n.items() if f.endswith(".py")),
               "files_py":sum(f.endswith(".py") for f in fs),
               "max_file":max(n.values()),"files_gt_4096":sum(v>4096 for v in n.values()),
               "files_gt_16384":sum(v>16384 for v in n.values())}
print(json.dumps(out,indent=1))
