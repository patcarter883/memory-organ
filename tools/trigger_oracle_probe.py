"""Probe: what does the FROZEN base answer for the E1 2-hop oracle clozes — with and without the
TriggerInjector attached? This discriminates "the harness harness itself changes base outputs"
(the attach path not being a no-op when bank=None) from "the base genuinely collides".

Run under the lean stack (GPU):
  PYTHONPATH=/opt/kernels:/minisgl/python:/minisgl python tools/trigger_oracle_probe.py [--attach] [langs...]
Prints the raw _oracle span for every (hop, language) — these exact strings become gold_true/gold_cf.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cam"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from m2_adapter import load_frozen_base              # noqa: E402
from mquake_ripple import _oracle, _TWOHOPS         # noqa: E402


class _NoTap:
    def set_bank(self, *a, **k):
        return None

    def set_trigger_ids(self, *a, **k):
        return None


def main():
    args = [a for a in sys.argv[1:] if a != "--attach"]
    attach = "--attach" in sys.argv
    langs = args or ["French", "Dutch", "Russian", "English"]
    base, tok = load_frozen_base("Qwen/Qwen3.5-4B")
    base.eval()
    torch.manual_seed(0)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    be = base.get_input_embeddings()
    lm = base.get_output_embeddings().weight
    n = int(os.environ.get("CAM_INDIST_ORACLE_LEN", "8"))

    injector = _NoTap()
    if attach:
        from gated_tap import TriggerInjector, build_trigger_keys
        print("[probe] creating TriggerInjector (fresh, apply_mask=True)...", flush=True)
        injector = TriggerInjector(base, 12, 512, n_heads=8, conf_gate=True, n_rel=1,
                                   orders=(2, 3), apply_mask=True).to("cuda")
        injector.attach()
        print("[probe] attached.", flush=True)
        subj_ids = tok("Raymond Triboulet", add_special_tokens=False).input_ids
        keys, used = build_trigger_keys(subj_ids, orders=(2, 3))
        injector.set_trigger_keys(subj_ids)
        print(f"[probe] keys for 'Raymond Triboulet' = {len(keys)} orders={used}", flush=True)

    for name, (cloze_t, _q) in _TWOHOPS.items():
        print(f"[probe] oracle hop {name!r}...", flush=True)
        row = []
        for L in langs:
            g = _oracle(base, None, tok, injector, be, lm, cloze_t.format(L=L), bos, n)
            row.append(f"{L}={g!r}")
        print(f"[probe] {name:8s} | " + " | ".join(row), flush=True)
    if attach:
        injector.detach()


if __name__ == "__main__":
    main()
