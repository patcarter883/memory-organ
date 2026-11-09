"""Shared helpers: import Gemini's code UNMODIFIED (only pin_memory is patched because this box has no CUDA),
and extract the VERBATIM Stage-2 block of EngramEditSolver.apply_edit so it can be run with a controlled delta*."""
import sys, inspect, textwrap, torch
torch.set_num_threads(2)
GEM = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini"
sys.path.insert(0, GEM)
_orig_pin = torch.Tensor.pin_memory
torch.Tensor.pin_memory = lambda self, *a, **k: self  # CPU-only box: pinning is a no-op for the math
import engram_graft as eg  # noqa: E402

src = inspect.getsource(eg.EngramEditSolver.apply_edit)
_s = src.index("        idx_map = self.engram.hasher.hash_sequence(prompt_ids_batch)")
_e = src.index("        return float(nll_loss.item())")
STAGE2_SRC = textwrap.dedent(src[_s:_e])
STAGE2_CODE = compile(STAGE2_SRC, "<gemini-stage2-verbatim>", "exec")


def run_stage2(solver, prompt_ids_batch, anchor_positions, delta_star, order_penalties={2: 10.0, 3: 1.0}):
    ns = dict(self=solver, prompt_ids_batch=prompt_ids_batch, anchor_positions=anchor_positions,
              delta_star=delta_star, order_penalties=order_penalties, torch=torch)
    exec(STAGE2_CODE, ns)
    return ns


def make_engram(d_model=64, r_mem=32, num_buckets=2_000_000, num_heads=4, dtype=torch.float32):
    return eg.GraftedEngramModule(d_model=d_model, r_mem=r_mem, num_buckets=num_buckets, num_heads=num_heads,
                                  device="cpu", dtype=dtype)


def read_e(engram, ids):
    engram.prefetch_from_host(ids)
    return engram._prefetched_e.detach().float().clone()
