"""M2 — the Titans memory ADAPTER (bolt-on, embeddings input, deep-MLP surprise memory) + LM smoke.

Architecture (frozen base, only the adapter trains):
  in_proj: base-embed(2560) -> mem_dim
  NeuralMemory: deep-MLP (depth-2) test-time memory (surprise + momentum + decay), RDNA4 flags
  readout: K learned query vectors probe the current memory state -> K vectors
  out_proj: mem_dim -> base-embed(2560)  (ZERO-INIT: memory tokens start ~0, clean step-0)

Per segment t over a long doc:
  READ  : mem_tokens = out_proj(retrieve(readout_q, memory_weights_{t-1}))   # summarize the past
  INJECT: frozen base on inputs_embeds=[mem_tokens ; segment_t embeds] -> LM loss on segment_t
  INGEST: state_t = memory.forward(segment_t embeds, state_{t-1})            # write current segment
Cross-segment info reaches the base ONLY through the K injected tokens (base is called per-segment,
so it has no context window across segments) -> isolates the memory's contribution.

Smoke success = LM loss DROPS over the run (memory learns to carry cross-segment info).

Entry:
  python -m cam.m2_adapter --steps 100
"""
import argparse, os, sys, time, torch, torch.nn as nn, torch.nn.functional as F

# flat package: the sibling import (deep_memory) is lazy, inside TitansMemoryAdapter.__init__ —
# it resolves relatively when imported as cam.X and falls back to a path-hacked absolute import
# when run as a file (`python cam/m2_adapter.py`).

# MODEL is the frozen base. Default is the production Qwen3.5-4B; CAM_BASE_MODEL overrides it (used by
# CPU wiring smokes on a tiny model like Qwen/Qwen3-0.6B — no GPU, just proves the code path).
MODEL = os.environ.get("CAM_BASE_MODEL", "Qwen/Qwen3.5-4B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class StageCost:
    """Wall-clock + peak-GPU-memory logger for a pipeline stage (issue #11: put a MEASURED cost line
    on record for every stage instead of step-count arithmetic). Usage:

        with StageCost("stage-1 bind (M=8, 6000 steps)"):
            ...

    Prints `[cost] <tag>: <sec>s | peak GPU <GiB> GiB` on exit (time-only off-GPU). Peak memory is
    reset on entry, so the number is THIS stage's peak, not the process high-water mark."""

    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        mem = ""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem = f" | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        print(f"[cost] {self.tag}: {dt:.0f}s{mem}", flush=True)
        return False

# distinct, structured-ish long passage -> several segments
TEXT = (
    "In 1969 the river port of Calmwater shipped grain north to the city of Auberon. "
    "The harbormaster, a woman named Iris Vale, kept a ledger of every vessel by name and cargo. "
    "The barge Northwind carried wheat; the cutter Gull carried salt; the steamer Meridian carried iron. "
    "Years later a historian asked which vessel had carried the iron, and the ledger answered: the Meridian. "
    "The same ledger recorded that the wheat had gone aboard the Northwind and the salt aboard the Gull. "
    "Memory of this kind is not attention over a window but a persistent store consulted on demand. "
    "When the question came about salt, the store returned the Gull, though many tokens had passed between. "
    "Thus the port remembered its manifest long after the boats had sailed beyond the bend of the river. "
)  # ~3 segments @ seg_len 64 — pure-torch deep memory retains the full multi-segment graph (mem-hungry)


class TitansMemoryAdapter(nn.Module):
    """memory='deepmem' (default): our graph-free deep Titans memory (deep_mem/deep_memory.py) — the
    OOM fix, scales to many segments. memory='lucidrains': the original NeuralMemory (OOMs ~6 segs)."""
    def __init__(self, base_hidden=2560, mem_dim=512, n_mem_tokens=8,
                 mem_chunk=16, mem_heads=4, mem_depth=2, mem_expansion=4.0, memory="deepmem"):
        super().__init__()
        self.K = n_mem_tokens
        self.mem_dim = mem_dim
        self.kind = memory
        self.in_proj = nn.Linear(base_hidden, mem_dim, bias=False)
        # normalize the memory's input: frozen base embeddings aren't std~0.02, so unnormalized in_proj
        # output saturates the surprise-MLP gelu -> ~zero gradient (diagnostic D: flat-at-chance without
        # this; binds frozen Qwen embeddings WITH it). Applied to both ingest and retrieve-query paths.
        self.norm = nn.LayerNorm(mem_dim)
        if memory == "deepmem":
            try:
                from .deep_memory import DeepMemory  # graph-free, depth-2 (mem_depth ignored)
            except ImportError:
                if __package__:  # real ImportError inside deep_memory, not "run as a file"
                    raise
                _here = os.path.dirname(os.path.abspath(__file__))
                if _here not in sys.path:
                    sys.path.insert(0, _here)
                from deep_memory import DeepMemory
            self.mem = DeepMemory(dim=mem_dim, chunk_size=mem_chunk, heads=mem_heads,
                                  expansion=mem_expansion)
        elif memory == "lucidrains":
            from titans_pytorch.neural_memory import NeuralMemory
            self.mem = NeuralMemory(
                dim=mem_dim, chunk_size=mem_chunk, heads=mem_heads,
                use_accelerated_scan=False,  # RDNA4: CUDA accelerated_scan not buildable
                default_model_kwargs=dict(depth=mem_depth, expansion_factor=mem_expansion),
            )
        else:
            raise ValueError(f"unknown memory backend {memory!r}")
        self.readout_q = nn.Parameter(torch.randn(n_mem_tokens, mem_dim) * 0.02)
        self.out_proj = nn.Linear(mem_dim, base_hidden, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # memory tokens start ~0 -> clean step-0

    def read(self, state, query_emb, out_dtype):
        """QUERY-CONDITIONED retrieval (option C): query the memory with the current context tokens
        `query_emb` ([B,Lq,base_hidden]) — so the memory surfaces the VALUE bound to what's being asked
        (mirrors the MQAR retrieve(key)->value that validated the core) — then attention-pool the
        per-token surfaced values down to K prepend tokens (readout_q = pool queries).
        CAUSALITY: `query_emb` must contain only tokens the consumer position may legally see (the caller
        excludes the teacher-forced answer); a query-conditioned prefix over future tokens would leak."""
        B = query_emb.shape[0]
        qv = self.norm(self.in_proj(query_emb.float()))             # [B,Lq,mem_dim] normalized queries
        if self.kind == "deepmem":
            st = self.mem.init_state(B) if state is None else state
            retrieved = self.mem.retrieve(qv, st)                    # [B,Lq,mem_dim] surfaced values
        else:
            weights = self.mem.init_weights(B) if state is None else state.weights
            retrieved = self.mem.retrieve_memories(qv, weights)
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)           # [B,K,mem_dim] pool queries
        attn = torch.softmax(pq @ retrieved.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)  # [B,K,Lq]
        pooled = attn @ retrieved                                    # [B,K,mem_dim] pooled surfaced value
        return self.out_proj(pooled).to(out_dtype)                  # [B,K,base_hidden] -> base dtype

    def ingest(self, seg_embeds, state):
        # memory runs in fp32 (surprise grad through the MLP needs precision); base embeds are bf16
        x = self.norm(self.in_proj(seg_embeds.float()))
        if self.kind == "deepmem":
            return self.mem(x, self.mem.init_state(x.shape[0]) if state is None else state)
        _, state = self.mem(x, state=state)
        return state

    def detach_state(self, state):
        if state is None:
            return None
        if self.kind == "deepmem":
            return state.detach()
        from titans_pytorch.neural_memory import mem_state_detach
        return mem_state_detach(state)


def lm_loss_segmented(base, adapter, ids, embeds, seg_len, detach_every=0):
    """Per-segment inject+forward+ingest; returns mean LM loss across segments."""
    B, S, H = embeds.shape
    V = base.config.get_text_config().vocab_size
    K = adapter.K
    state, total, nseg, prev_seg = None, 0.0, 0, None
    for si, s in enumerate(range(0, S, seg_len)):
        seg_emb = embeds[:, s:s + seg_len]
        seg_ids = ids[:, s:s + seg_len]
        L = seg_emb.shape[1]
        if L < 2:
            break
        # query-conditioned read on the PREVIOUS segment (leak-free for the LM objective: the prefix
        # never depends on tokens this segment is predicting); first segment has no past -> zero prefix
        if prev_seg is None:
            mem_tokens = torch.zeros(B, K, H, dtype=embeds.dtype, device=embeds.device)
        else:
            mem_tokens = adapter.read(state, prev_seg, embeds.dtype)  # [B,K,H]
        inp = torch.cat([mem_tokens, seg_emb], dim=1)                # [B,K+L,H]
        logits = base(inputs_embeds=inp).logits[:, K:]              # [B,L,V] (drop memory positions)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), seg_ids[:, 1:].reshape(-1))
        total = total + loss
        nseg += 1
        state = adapter.ingest(seg_emb, state)                       # write current segment
        prev_seg = seg_emb
        if detach_every and (si + 1) % detach_every == 0:
            state = adapter.detach_state(state)                     # truncated BPTT
    return total / max(nseg, 1), nseg


def _maybe_patch_native_gdn(model):
    """OPT-IN RDNA4 acceleration: swap the frozen base's HuggingFace Qwen3.5 gated-delta-net (fla) layers
    for minisgl's native Triton-free `gdn_hip` path. Enabled ONLY when `CAM_NATIVE_GDN=1` — a no-op
    otherwise, so this repo carries NO hard dependency on minisgl (the import is lazy + guarded, and
    external reviewers see an optional accel path, not a coupling). On gfx1201, fla's GDN kernel HANGS as
    the tap gate opens and isn't in the serve image; `gdn_hip` is the differentiable Triton-free
    replacement (forward matches fla ~1e-3, dL/d_hidden cos>0.9999). The run must mount the minisgl
    checkout and put it on PYTHONPATH (see tools/run_cam_native_gdn.sh). Prints the layers patched.
    Fails LOUD if opted in but minisgl can't be imported — a silent fla fallback would hang, not error."""
    if os.environ.get("CAM_NATIVE_GDN") != "1":
        return model
    from minisgl.gdn.hf_patch import patch_qwen3_5_gdn   # lazy: only imported when opted in
    n = patch_qwen3_5_gdn(model)
    print(f"[cam] CAM_NATIVE_GDN=1 -> patched {n} Qwen3.5 gated-delta-net layer(s) to native gdn_hip",
          flush=True)
    if n == 0:
        print("[cam] WARNING: CAM_NATIVE_GDN=1 but 0 GDN layers found — base is running on fla "
              "(will HANG on RDNA4 once the gate opens). Is this a GDN-hybrid model?", flush=True)
    return model


def load_frozen_base(model_id=None):
    """Load + freeze the donor base. model_id=None keeps the historical default (MODEL, Qwen3.5-4B);
    drivers expose it as --base1 so reproducers can swap donors / smaller cards without editing code.

    MODEL-PARALLEL (CAM_DEVICE_MAP=auto): a base too big for one 16GB card (e.g. Qwen3.5-9B, ~18GB bf16)
    is sharded across the visible GPUs with device_map='auto'. The base is FROZEN and CAM trains only the
    tap by backprop THROUGH it — cross-device autograd handles the grad flow. The tap must sit on the
    device of ITS layer and cast the bank/conf to h.device (see gated_tap); the store stays on cuda:0.
    Requires a 2-card lease (gpu-lease -n 2). Unset -> single-card .to(cuda) (byte-identical default)."""
    model_id = model_id or MODEL
    device_map = os.environ.get("CAM_DEVICE_MAP")               # 'auto' -> shard across visible GPUs
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    m, last = None, None
    for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            if device_map:
                # 'auto' front-loads cuda:0, but cuda:0 ALSO holds the store/tap/embeddings + the
                # activations — so reserve headroom there (the 9B OOM'd on plain 'auto'). Cap each GPU at
                # capacity minus ~5GiB on cuda:0 / ~1GiB elsewhere.
                mm = None
                if device_map == "auto" and torch.cuda.is_available():
                    mm = {}
                    for i in range(torch.cuda.device_count()):
                        cap = torch.cuda.get_device_properties(i).total_memory / 1e9
                        mm[i] = f"{max(1, int(cap - (5 if i == 0 else 1)))}GiB"
                m = loader.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                           device_map=device_map, max_memory=mm).eval()
            else:
                m = loader.from_pretrained(model_id, dtype=torch.bfloat16,
                                           low_cpu_mem_usage=True).to(DEV).eval()
            break
        except Exception as e:  # noqa
            last = e
    if m is None:
        raise last
    for p in m.parameters():
        p.requires_grad_(False)
    m.config.use_cache = False
    # opt-in RDNA4 accel — OUTSIDE the loader try/except so a patch/import failure stays loud (a silent
    # fla fallback would HANG on RDNA4, not error).
    m = _maybe_patch_native_gdn(m)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seg-len", type=int, default=64)
    ap.add_argument("--mem-dim", type=int, default=512)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--memory", choices=["deepmem", "lucidrains"], default="deepmem")
    ap.add_argument("--repeat", type=int, default=1, help="repeat TEXT to scale segment count (OOM test)")
    ap.add_argument("--detach-every", type=int, default=0, help="truncated BPTT: detach state every N segments")
    args = ap.parse_args()

    print(f"[m2] device={DEV} memory={args.memory} steps={args.steps} seg_len={args.seg_len} "
          f"mem_dim={args.mem_dim} K={args.k} repeat={args.repeat} detach_every={args.detach_every}")
    base, tok = load_frozen_base()
    H = base.config.get_text_config().hidden_size
    ids = tok(TEXT * args.repeat, return_tensors="pt").input_ids.to(DEV)
    embeds = base.get_input_embeddings()(ids).detach()
    print(f"[m2] frozen base hidden={H}; seq_len={ids.shape[1]} -> ~{ids.shape[1]//args.seg_len} segments")

    adapter = TitansMemoryAdapter(base_hidden=H, mem_dim=args.mem_dim, n_mem_tokens=args.k,
                                  memory=args.memory).to(DEV)  # fp32 memory
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"[m2] adapter trainable params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    first = None
    for step in range(args.steps):
        opt.zero_grad()
        loss, nseg = lm_loss_segmented(base, adapter, ids, embeds, args.seg_len, args.detach_every)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        if first is None:
            first = loss.item()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"[m2] step {step:3d} loss {loss.item():.4f} (nseg={nseg})")
    last = loss.item()
    drop = first - last
    if DEV == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[m2] peak GPU mem {peak:.2f} GB over {ids.shape[1]//args.seg_len} segments "
              f"(memory={args.memory})")
    print(f"[m2] {'PASS' if drop > 0.05 else 'CHECK'}: loss {first:.4f} -> {last:.4f} (Δ={drop:+.4f}); "
          f"{'memory adapter is learning to carry cross-segment info.' if drop > 0.05 else 'no clear drop — inspect.'}")


if __name__ == "__main__":
    main()
