"""Probe: does MANUAL KV-cache incremental decode work for Qwen3.5-4B on titans:dev (torch-GDN)?

Compares the historical no-cache full-sequence re-prefill loop against a cached incremental loop on
the same prompt, and checks the cache actually grows (seq_length) — i.e. the GDN mixer state is
being carried. Green here = switching the harness generation loops to cached decode is safe.

Run (lean... no — titans:dev + torch-GDN sitecustomize):
  docker run ... -e PYTHONPATH=/engine/tools/torchgdn_site titans:dev python tools/gen_cache_probe.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cam"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from m2_adapter import load_frozen_base, DEV   # noqa: E402


def no_cache_gen(base, tok, prompt_ids, n=12):
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=DEV)
    out = []
    with torch.no_grad():
        for _ in range(n):
            logits = base(input_ids=cur).logits[:, -1].float()
            nxt = logits.argmax(-1)
            t = int(nxt.item())
            out.append(t)
            cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
    return out


def cache_gen(base, tok, prompt_ids, n=12):
    step = torch.tensor([prompt_ids], dtype=torch.long, device=DEV)
    out = []
    past = None
    with torch.no_grad():
        for i in range(n):
            outp = base(input_ids=step, past_key_values=past, use_cache=True)
            past = outp.past_key_values
            if i == 0:
                print(f"[cacheprobe] seq_length after prefill: {past.get_seq_length()}", flush=True)
            logits = outp.logits[:, -1].float()
            nxt = logits.argmax(-1)
            t = int(nxt.item())
            out.append(t)
            step = nxt.view(1, 1)
    print(f"[cacheprobe] seq_length after {n} steps: {past.get_seq_length()}", flush=True)
    return out


def main():
    base, tok = load_frozen_base("Qwen/Qwen3.5-4B")
    base.eval()
    base.config.use_cache = True                      # the harness disables it; per-call arg overrides
    prompt = "The mother tongue of Raymond Triboulet is"
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + tok(prompt, add_special_tokens=False).input_ids
    print(f"[cacheprobe] prompt: {prompt!r} ({len(ids)} ids)", flush=True)
    a = no_cache_gen(base, tok, ids)
    b = cache_gen(base, tok, ids)
    print(f"[cacheprobe] no-cache : {tok.decode(a)!r}", flush=True)
    print(f"[cacheprobe] cached   : {tok.decode(b)!r}", flush=True)
    same = a == b
    print(f"[cacheprobe] token-exact match: {same}  (allowing first-step argmax near-tie drift)",
          flush=True)


if __name__ == "__main__":
    main()
