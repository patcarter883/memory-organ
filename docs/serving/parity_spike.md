# Train/Serve Parity Spike — does minisgl's layer-24 residual match HF's?

**Status:** PLAN + probe skeleton only. **Nothing here has been run.** The probe MUST be launched
under `gpu-lease -n 1` inside `titans:dev` (recipe in §5). Do not execute from a bare shell.

## 0. The one question this spike answers

The knowledge-editing **tap** (`cam/gated_tap.py :: GatedMemoryTap`, driven by
`cam/recall_mag.py`) was trained against **HuggingFace transformers'** `Qwen/Qwen3.5-4B`. It reads
and injects into the **residual stream at the output of decoder layer 24**. Concretely, at train
time the tap is a forward-hook on `base.model.layers[24]` that rewrites `output[0]`
(`MAGInjector._hook` / `attach` in `cam/gated_tap.py:177-193`), i.e. it operates on **the HF decoder
layer's output hidden state after layer index 24**.

Production serving uses `minisgl-rdna4`'s own Qwen3.5 impl
(`python/minisgl/models/qwen3_5.py`, registered as `Qwen3_5ForConditionalGeneration` in
`models/register.py:11`). Different kernels/fusions (`tail_hip` fused RMSNorm, HIP attention,
native `gdn_hip`, graph capture), all bf16.

**If minisgl's residual at layer 24 does not numerically match HF's for the same tokens, the learned
injection lands in the wrong basis and served edits silently fail (produce garbage or no-ops).** This
spike measures that residual, per layer, and decides whether the tap transfers as-is.

### Why there is real hope (and real risk)
- **Hope:** the CAM HF path already runs `CAM_NATIVE_GDN=1` →
  `minisgl.gdn.hf_patch.patch_qwen3_5_gdn` (`cam/m2_adapter.py:181-199`,
  `python/minisgl/gdn/hf_patch.py`). So **HF's GDN layers already use minisgl's `gdn_hip` compute**
  (weights copied 1:1, `_build_native_from_hf`). The linear-attention mixer — the biggest structural
  risk in a GDN-hybrid — is already shared code on both sides. That removes one whole family of
  divergence *for the layers CAM was trained through*.
- **Risk:** everything else still differs — **full-attention** layers (`Qwen3_5Attn`, HIP paged
  attention vs HF SDPA/eager), **RMSNorm** (minisgl `tail_hip` fused fp32-internal vs HF), **RoPE**
  (partial rotary_dim=64), **MLP** (`GatedMLP` SwiGLU), dtype accumulation order, and the residual
  bookkeeping (§3 has a specific off-by-one hazard). bf16 makes small drifts inevitable; the question
  is whether they stay benign through 24 layers.

## 1. Exact comparison

**Inputs.** A small fixed prompt set, tokenized **identically** by the same tokenizer both sides
(`AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")`, the tokenizer `load_frozen_base` already uses,
`cam/m2_adapter.py:214`). Suggested set:
- The `m2_adapter.TEXT` ledger passage (already in-repo, `cam/m2_adapter.py:63-72`) — long enough to
  exercise the GDN recurrent state over many positions.
- 2–3 CounterFact-style edit prompts ("The capital of France is", etc.) — the actual serving
  distribution the tap fires on.
- One BOS-only and one single-token prompt as degenerate controls.

**Tokenization/BOS discipline (must match exactly):**
- HF: `tok(text, return_tensors="pt").input_ids` — this **adds BOS** per Qwen tokenizer default.
  minisgl's `LLM._tokenize_one` must be checked for whether it prepends BOS; align them. The probe
  asserts the two id tensors are **byte-identical** before comparing hidden states — any mismatch here
  invalidates everything. (Note `recall_mag` builds ids with an explicit `bos` list; the probe should
  standardize on one tokenization and feed the *same integer ids* to both engines.)
- **Prefill only. No sampling, no KV-cache decode.** Compare a single full-sequence forward. HF:
  `use_cache=False` (already set in `load_frozen_base`, `m2_adapter.py:240`). minisgl: a single
  prefill batch, **`--graph 0` (eager)** — graph capture is explicitly skipped for `return_hidden`
  (`engine/engine.py:476` "Never a CUDA graph (return_hidden…)"), so capture is off the table here
  anyway and must not be forced on.

**What we capture, both sides:** the residual stream `h_L` at the **output of each decoder layer L**,
for L in `0..num_layers-1` (35 layers for the 4B; layer 24 is the tap site). Shape `[T, H]`
(`H=2560`), fp32-upcast for the metric.

**Metric, per layer L, per position t:**
- **Cosine similarity** `cos(h_L^minisgl[t], h_L^HF[t])` — the primary metric (the tap is a
  cross-attention in this basis; direction is what matters).
- **Relative L2** `‖h_L^minisgl[t] − h_L^HF[t]‖₂ / ‖h_L^HF[t]‖₂` — catches scale drift cosine hides.
- Report **per-layer**: `min` and `mean` cosine over positions, `max` and `mean` relL2. The `min`
  over positions is the honest number (one bad position breaks an edit).

**Alignment:** same token ids ⇒ position t aligns 1:1. Upcast **both** to fp32 before the metric
(don't compare in bf16). If minisgl returns `[num_layers, T, H]` stacked and HF returns a list, index
both by absolute layer id.

**PASS threshold (layer 24, the tap site):**
- **PASS:** `min-over-positions cos ≥ 0.999` **and** `mean relL2 ≤ 0.02` at L=24.
- **MARGINAL:** `0.99 ≤ cos < 0.999` — tap probably transfers but validate end-to-end (§4).
- **FAIL:** `cos < 0.99` at L=24 → tap does not transfer as-is; localize (§4).

Rationale: the tap's gate is `tanh`-gated cross-attention; a 1e-3 directional error in a bf16 residual
is within the noise the tap already tolerated at train time (GDN fwd parity itself is only ~1e-3,
`hf_patch.py` docstring). Below 0.99 the injected key/value alignment degrades enough to matter.

## 2. How to extract per-layer hidden states — from each side

### minisgl (NO source modification needed — the hook already exists)
`Qwen3_5Model.forward(input_ids, return_hidden=True)` already supports per-layer capture:
- `set_capture_layers(ids)` programs `self._capture_layer_ids`
  (`models/qwen3_5.py:296`, exposed on the top model via
  `Qwen3_5ForConditionalGeneration.set_capture_layers`, `qwen3_5.py:449`).
- In the forward loop, for each captured `lid` it does `grabbed[lid] = residual.clone()`
  (`qwen3_5.py:310-312`) and returns `aux_stack = torch.stack([grabbed[i] for i in cap], dim=0)`
  as the 3rd return value (`qwen3_5.py:319-320`).
- The engine plumbs this: `engine.forward(..., return_hidden=True)` →
  `model.forward(return_hidden=True)` → `(logits, last_hidden, aux_hidden)` (`engine/engine.py:467-505`).

**So the extraction path is: `set_capture_layers(list(range(num_layers)))`, run one eager prefill with
`return_hidden=True`, read `aux_hidden` ([num_layers, T, H]).** This is the mechanism the EAGLE3/MTP
proposers already use — it is a supported, tested path, not new instrumentation.

**⚠️ Residual-convention hazard (READ THIS — it is the #1 way to get a wrong answer):**
`grabbed[lid] = residual.clone()` captures minisgl's **fused-norm running residual**, which is NOT the
same tensor as HF's `layer.output[0]`. minisgl uses `RMSNormFused` (`layers/norm.py:36-57`) where
`residual ← x + residual` is folded at the *next* norm. Trace `Qwen3_5DecoderLayer.forward`
(`qwen3_5.py:262-271`):
```
x, residual = input_layernorm(x, residual)      # residual ← x_in + residual_in ; x = norm(...)
x = mixer(x)
x, residual = post_attention_layernorm(x, residual)  # residual ← x(mixer_out) + residual ; x = norm(...)
x = mlp(x)                                       # mlp output; NOT yet added to residual
return x, residual                               # residual == pre-MLP-add sum; x == mlp_out
```
The **true residual stream after layer L** (== HF's `hidden_states` output of layer L) is
`x + residual` — the MLP is only folded into `residual` by the *next* layer's `input_layernorm`.
minisgl itself computes this correct quantity for the MTP seed: `pre_norm = (x + residual).clone()`
(`qwen3_5.py:315`). **But the per-layer capture at line 311 stores `residual.clone()` — the
pre-MLP-add sum — which is off by the layer's own MLP contribution.** For the tap-parity comparison we
need the post-MLP residual = HF's layer output.

Two clean options (pick one; do NOT compare `residual.clone()` directly to HF):
1. **Non-invasive: capture at the top of the NEXT layer.** The value fed into layer `L+1`'s
   `input_layernorm` as `residual` *is* the post-MLP residual of layer L. So capturing "input residual
   of layer L+1" == "output hidden of layer L". Programming capture ids `{25}` and reading it gives
   HF's layer-24 output. But the existing hook captures the *output* residual, not the *input*, so
   this still needs care — see option 2.
2. **Preferred: add a tiny, throwaway capture that stores `(x + residual)`** at the tap layer. This is
   a **read-only debug capture in a scratch fork/branch of the model file, not a production edit** —
   e.g. in a probe-only subclass or a monkeypatch in the probe script that wraps the layer loop and
   records `x + residual` after `layer.forward`. Since we already reconstruct `x + residual` for the
   MTP seed, this is trivially correct and matches HF exactly. **The plan's probe (§5) monkeypatches
   the capture in the probe script — no source file is modified.**

Concretely the probe monkeypatches `Qwen3_5Model.forward` (or wraps each `layer.forward`) to record
`(x + residual).detach().float()` after every layer — the exact post-layer residual stream. This is
non-invasive (lives in the probe module) and sidesteps the line-311 off-by-one entirely.

### HF (transformers) — the reference the tap was trained on
`load_frozen_base()` (`cam/m2_adapter.py:202-245`) already gives `(model, tok)` frozen bf16 with
`CAM_NATIVE_GDN=1` applied (so GDN==minisgl). Two equally valid ways to get per-layer output hidden:
- `model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states` →
  tuple of `num_layers+1` tensors; `hidden_states[L+1]` is the output of decoder layer L
  (index 0 is the embedding output). **This is the exact quantity the tap's forward-hook on
  `layers[L]` sees** as `output[0]`. Use `hidden_states[25]` for the layer-24-output tap site.
- Or replicate the tap's own mechanism: register forward-hooks on `decoder_layers(base)[L]`
  (`cam/gated_tap.py:21-34, 177-193`) and record `out[0]`. This is the most faithful — it captures
  literally what the trained tap consumed. **Do both and assert they agree** (they must); it validates
  the `output_hidden_states` indexing.

**Both engines must run the SAME frozen weights.** minisgl loads from the HF checkpoint via its own
loader; confirm `Qwen/Qwen3.5-4B` resolves to the same snapshot in `~/.cache/huggingface` (mounted
`HF_HUB_OFFLINE=1`). If minisgl and HF disagree on a weight remap the parity will fail for a boring
reason — the probe prints `‖W‖` of a couple of shared tensors (e.g. `embed_tokens`, `layers.24.mlp`)
on both sides as a sanity gate before comparing activations.

## 3. Controls / confounders — benign vs disqualifying

| Confounder | Benign (expected) | Disqualifying (a real port bug) |
|---|---|---|
| **bf16 nondeterminism / accumulation order** | per-layer cos drifts down slowly, ~1e-4/layer; L24 cos ≥ 0.999 | cos falls off a cliff at one specific layer → that sublayer diverges |
| **RMSNorm eps / convention** | identical if both use `plus_one=True` + same `rms_norm_eps` | wrong eps or missing `(1+weight)` gain → early, large, uniform drift. **Verify:** minisgl uses `plus_one=True` everywhere (`qwen3_5.py:116,253-258,291`); HF Qwen3.5 config must match. |
| **RoPE (partial rotary_dim=64)** | matching θ base + partial-dim slice → identical | off-by rotary_dim, wrong base, or full-vs-partial mismatch → attention-layer residuals diverge, GDN layers stay fine (a very diagnostic signature) |
| **GDN native path** | **already shared** (both use `gdn_hip`); expect ~1e-3, non-accumulating | if GDN layers diverge, the weight copy in `_build_native_from_hf` mismatched — check in/out_proj concat order |
| **Attention backend** | HIP paged attn vs HF eager: small bf16 drift at full-attn layers | large jump only at full-attention layer ids → attention impl bug (masking, scale, gate `sigmoid`) |
| **KV-cache vs full forward** | eliminated by design: **prefill-only, no cache both sides** | if someone runs minisgl with a decode step, positions misalign — forbidden |
| **Sequence length / batch** | batch=1, T = prompt length; compare all positions | — |
| **BOS handling** | both prepend (or both omit) — asserted byte-identical ids | ids differ → every downstream number meaningless; hard-fail the probe |
| **Gate/output-gate (`Qwen3_5Attn` sigmoid gate)** | applied both sides identically | present one side only → attention-layer divergence |

**Tolerance stance:** treat cos ≥ 0.999 with slow monotone decay as *benign bf16 noise*. Treat any
**step change** at a specific layer as a *localized port defect* to be named (which sublayer) even if
L24 still passes — because it tells us the port is fragile.

## 4. Decision tree

```
Run probe → per-layer cosine table.

├─ L24 cos ≥ 0.999 AND relL2 ≤ 0.02
│    → TAP TRANSFERS. Integration is plumbing:
│      wire the trained tap into minisgl serve as a forward-hook / injection at
│      layer-24 output residual (the x+residual point), feeding the same [B,K,mem_dim] bank.
│      Confirm with ONE end-to-end served edit (edited fact answered, neighbor unchanged).
│
├─ 0.99 ≤ L24 cos < 0.999  (MARGINAL)
│    → Likely transfers. Do the end-to-end served-edit check BEFORE trusting it.
│      If edits fire correctly → ship. If not → treat as FAIL below.
│
└─ L24 cos < 0.99  (FAIL) → LOCALIZE, then choose a remedy:
     1. Read the per-layer table top-down. Find the FIRST layer where cos drops below ~0.9995
        and the WIDTH of the drop:
          • uniform slow decay from layer 0 → global dtype/accumulation; not a single bug.
          • step at full-attention layer ids only → attention/RoPE/gate port bug (fix the kernel;
            cheapest, keeps the tap).
          • step at GDN layer ids → gdn_hip weight-copy / state bug (fix _build_native_from_hf).
          • step at ALL layers uniformly from L0 → RMSNorm eps/convention or embedding mismatch.
     2. Remedy options, cheapest first:
          (a) FIX THE LOCALIZED SUBLAYER so parity returns — best outcome, tap untouched.
          (b) PICK A BETTER LAYER: if some layer L' has cos ≥ 0.999 and the tap can be
              re-placed there, retrain the tap at L' (cheap — tap is tiny, recall_mag --tap-layers L').
          (c) THIN ADAPTER: learn a small linear map minisgl_resid → HF_resid at L24 (a frozen
              basis-align) and inject through it. Only if (a)/(b) fail.
          (d) RETRAIN THE TAP against minisgl's residual directly — but CAM trains by backprop
              THROUGH the frozen base (recall_mag train_taps), which needs a differentiable minisgl
              forward. minisgl serve forward is inference-only, so this is the most expensive option
              (would require the differentiable gdn_hip train path + an HF-shaped wrapper). Last resort.
```

## 5. Concrete probe script skeleton

**Marked NEEDS-GPU. Run only under `gpu-lease -n 1` in `titans:dev`. Do not run from this session.**

Container recipe mirrors `memory-organ/tools/run_cam_native_gdn.sh` (the validated CAM-native recipe):
titans:dev image, minisgl WORKTREE mounted read-only (with a current `gdn_hip` .so carrying the
kernels), memory-organ WORKTREE mounted, HF cache, the full ROCm device passthrough.

### Isolation (do FIRST, per CLAUDE.md worktree rule)
```bash
# minisgl worktree off the serving commit; COPY the vendored gitignored .so in (worktree lacks them)
git -C /home/pat/code/minisgl-rdna4 worktree add -b parity-spike \
    /home/pat/code/minisgl-rdna4-parity rdna4
cp /home/pat/code/minisgl-rdna4/gdn_hip/*.so /home/pat/code/minisgl-rdna4-parity/gdn_hip/   # +the 11 other vendored HIP .so
# (repeat for attn_hip/attn_decode/attn_prefill_paged/tail_hip/mla_hip/... — copy ALL vendored .so,
#  else boot crashes; see MEMORY "Worktree validation needs .so copied")
# memory-organ worktree (read-only in container)
git -C /home/pat/code/memory-organ worktree add -b parity-spike-organ \
    /home/pat/code/memory-organ-parity HEAD
```

### Probe script — `tools/parity_probe.py` (lives in the memory-organ worktree)
```python
#!/usr/bin/env python
"""Train/serve residual-parity probe. NEEDS A GPU LEASE. Compares minisgl's Qwen3.5-4B per-layer
output residual stream to HF's, for identical token ids, prefill-only, eager. Prints a per-layer
cosine / relL2 table and PASS/FAIL at layer 24. Modifies NO source (minisgl capture is monkeypatched)."""
import os, torch, torch.nn.functional as F
os.environ.setdefault("CAM_NATIVE_GDN", "1")   # HF path already uses minisgl gdn_hip (removes GDN as a variable)
TAP_LAYER = 24
PROMPTS = [
    "The capital of France is",
    ("In 1969 the river port of Calmwater shipped grain north to the city of Auberon. "
     "The barge Northwind carried wheat; the cutter Gull carried salt; the steamer Meridian carried iron."),
]

# ---------- HF reference (exactly as recall_mag loads it) ----------
import sys; sys.path.insert(0, "/engine")            # memory-organ worktree
from cam.m2_adapter import load_frozen_base, DEV     # CAM_NATIVE_GDN patch applied inside
hf, tok = load_frozen_base("Qwen/Qwen3.5-4B")        # frozen bf16, use_cache=False, gdn_hip-patched

def hf_resids(ids):
    with torch.no_grad():
        hs = hf(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    # hs[L+1] == output hidden of decoder layer L (index 0 = embedding). Return [num_layers, T, H] fp32.
    return torch.stack([h[0].float() for h in hs[1:]], dim=0)   # drop embedding row

# ---------- minisgl (native serve model), same ids ----------
# Bring up the engine on the parity worktree (PYTHONPATH=/minisgl/python:/minisgl), eager (graph off).
# Use the LLM/engine offline path; program full-layer capture; monkeypatch to record x+residual.
from minisgl.models.qwen3_5 import Qwen3_5Model
_orig = Qwen3_5Model.forward
_CAPTURE = {}
def _patched(self, input_ids, return_hidden=False):
    # replicate forward but record the TRUE post-layer residual (x+residual) at every layer — sidesteps
    # the line-311 pre-MLP-add off-by-one. Read-only; probe-local; no source file touched.
    x = self.embed_tokens.forward(input_ids); residual=None; rec=[]
    for layer in self.layers.op_list:
        x, residual = layer.forward(x, residual)
        rec.append((x + residual).detach().float().clone())   # == HF layer output hidden
    _CAPTURE["resids"] = torch.stack(rec, dim=0)               # [num_layers, T, H]
    return self.norm.forward(x, residual)[0]
Qwen3_5Model.forward = _patched

def minisgl_bringup():
    # instantiate the serving engine on Qwen3.5-4B, eager, TP=1, bf16 — reuse the repo's offline LLM
    # entry (python/minisgl/llm/llm.py) or engine.Engine directly. Exact bring-up = the serve harness
    # minus the API server; --graph 0. (Fill in from tools/*serve*/run_bench_window.sh conventions.)
    from minisgl.llm import LLM
    return LLM("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, graph=0, tp_size=1)  # signature per llm.py

eng = minisgl_bringup()

def ms_resids(ids):
    # drive ONE prefill of these exact ids through the engine; _patched fills _CAPTURE["resids"].
    _ = eng.forward_prefill(ids)          # exact call per engine API; return_hidden not needed (monkeypatch grabs it)
    return _CAPTURE["resids"]

# ---------- compare ----------
def table(ids):
    ids = ids.to(DEV)
    A = ms_resids(ids); B = hf_resids(ids)
    assert A.shape == B.shape, (A.shape, B.shape)
    print(f"{'L':>3} {'min_cos':>9} {'mean_cos':>9} {'mean_relL2':>11} {'max_relL2':>10}")
    for L in range(A.shape[0]):
        a, b = A[L], B[L]                                   # [T,H]
        cos = F.cosine_similarity(a, b, dim=-1)             # [T]
        rel = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)
        mark = "  <-- TAP" if L == TAP_LAYER else ""
        print(f"{L:>3} {cos.min():9.5f} {cos.mean():9.5f} {rel.mean():11.5f} {rel.max():10.5f}{mark}")
    L = TAP_LAYER; a,b = A[L],B[L]
    cmin = F.cosine_similarity(a,b,dim=-1).min().item()
    r = ((a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-6)).mean().item()
    verdict = "PASS" if (cmin>=0.999 and r<=0.02) else ("MARGINAL" if cmin>=0.99 else "FAIL")
    print(f"\n[parity] L{L}: min_cos={cmin:.5f} mean_relL2={r:.5f} -> {verdict}")

for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids            # BOS per Qwen default
    # ASSERT minisgl tokenizes p to the SAME ids before trusting the comparison (align BOS!).
    table(ids)
```
> Skeleton. The two `# fill in` spots — the exact minisgl `LLM`/`Engine` bring-up call and the
> single-prefill drive (`eng.forward_prefill`) — must be matched to `python/minisgl/llm/llm.py` /
> `engine/engine.py:467` signatures when wiring it up. `forward` with `return_hidden` and the
> `set_capture_layers` API is the supported fallback if the monkeypatch is undesirable, but the
> monkeypatch is preferred because it captures `x+residual` (HF's exact quantity) rather than the
> pre-MLP-add `residual.clone()`.

### Run (later, with a lease)
```bash
gpu-lease -n 1 -- bash -c '
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e CAM_NATIVE_GDN=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/minisgl/python:/minisgl \
    -v /home/pat/code/minisgl-rdna4-parity:/minisgl:ro \
    -v /home/pat/code/memory-organ-parity:/engine:ro \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash titans:dev -lc "source /app/.venv/bin/activate && python /engine/tools/parity_probe.py"
'
```
(Reuse a warm Triton cache mount `-v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton`
if a cold GDN autotune is unacceptable; a throwaway copy for a one-shot probe.)

## 6. Effort estimate & gating

- **Wire-up:** ~0.5 day — the monkeypatch capture and HF `output_hidden_states` are both trivial; the
  only real work is the minisgl offline bring-up call (`LLM`/`Engine` signature) + the single-prefill
  drive, and the tokenization-alignment assert.
- **Run:** one `gpu-lease -n 1` window, minutes of compute (two prefills, 4B, eager). One warm-cache
  boot.
- **Analysis:** the per-layer table is self-interpreting via §4.

**What it gates:**
- **PASS at L24 → unblocks integration.** The tap port becomes pure plumbing: hook the trained tap on
  minisgl's layer-24 output residual (`x+residual`) at serve time, feed the same bank. Green-light the
  serve-side integration issue.
- **FAIL → blocks integration and hands back a *specific* target** (which sublayer, how big) so the fix
  is a named kernel/parity bug, a layer re-selection, or a thin adapter — not a blind "retrain
  everything." The table itself is the deliverable that turns an open-ended risk into a scoped task.

**Cleanup:** `git worktree remove` both worktrees when done.
