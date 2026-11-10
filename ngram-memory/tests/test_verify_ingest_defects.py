"""[Integration status: every defect below is FIXED; the former strict-xfail tests are regular regression tests.]
Adversarial-verifier tests for the ingest piece. Each test demonstrates one defect and is marked
xfail(strict=True) with its defect id; it should start passing (and then be un-marked) once the defect is fixed.

VI-1  the 'key-shuffle' arm (C1 control, §6.A.4) is the base arm relabelled: cartridge() un-permutes the slots,
      so the served cartridge is the G-lin cartridge up to float noise.
VI-2  assign_split_a(frozen=...) skips the §6.A.1 dedupe gate and the near-dup cluster rule for NEW files, so a
      verbatim copy of a train file added later (corpus v1) lands in test.
VI-3  the generic secret-assignment rule anchors on \\b before the key name, so prefixed snake/upper-case names
      (OPENAI_API_KEY, db_password, aws_secret_access_key, GITHUB_TOKEN) are never flagged.
"""
from __future__ import annotations

import pytest
import torch

from ngram_memory.ingest import train as TR
from ngram_memory.ingest.corpus import assign_split_a, scan_secrets, split_a_hash
from test_ingest_train import _setup


def test_vi1_key_shuffle_cartridge_differs_from_base_cartridge():
    carts = {}
    for arm in ("G-lin", "key-shuffle"):
        host, fmt, store, ws, anc, cfg, ix = _setup(arm=arm, max_steps=8)
        tr = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc)
        tr.train()
        tr.close()
        carts[arm] = tr.cartridge(prune_zero=False)
    a, b = carts["G-lin"].values, carts["key-shuffle"].values
    assert torch.equal(carts["G-lin"].row_ids, carts["key-shuffle"].row_ids)
    # a content-free control must not ship (nearly) the same rows as the real arm; measured 0.0042 today
    assert float((a - b).norm() / a.norm()) > 0.5


def test_vi2_frozen_split_applies_dedupe_gate_to_new_files():
    body = "\n".join(f"def function_number_{i}(alpha, beta):\n    return alpha * {i} + beta  # line {i}"
                     for i in range(40))
    files = {"pkg/a.py": body, "pkg/b.py": "x = 1\n" * 100}
    frozen = assign_split_a(files)
    assert frozen.train == ["pkg/a.py", "pkg/b.py"]
    i = 0
    while not split_a_hash(f"tools/copy_{i}.py") < 0.15:
        i += 1
    new = f"tools/copy_{i}.py"
    assert assign_split_a({**files, new: body}).assignment()[new] == "neardup_diag"   # unfrozen: gated
    again = assign_split_a({**files, new: body}, frozen=frozen)
    assert again.dup_frac[new] == 1.0
    assert again.assignment()[new] != "test"


def test_vi3_secret_scan_flags_prefixed_key_names():
    # synthetic random-looking values, not real credentials
    lines = ['OPENAI_API_KEY = "Qm9yZW1pcHN1bWRvbG9yc2l0YW1ldA"',
             'db_password = "Zq8xL2pR9vT4mN7kW3"',
             'aws_secret_access_key = "Ab3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5bC7d"',
             'GITHUB_TOKEN = "Ab3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1"']
    missed = [ln.split("=")[0].strip() for ln in lines if not scan_secrets("f.py", ln)]
    assert not missed, f"not flagged: {missed}"


def test_vi3_secret_scan_extra_forms_and_no_false_positive_on_code():
    hit = ['MY_SECRET = "Ab3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1"',
           'k = os.environ.get("SERVICE_TOKEN", "Ab3dE5fG7hJ9kL1mN3pQ5rS7")',
           'x = "sk_live_' + 'Ab3dE5fG7hJ9kL1mN3pQ5"']
    miss = ['_R + "patch_generator.cls_token.token": "vision.cls_token",',   # radio_vision.py at df29117
            'tokenizer_path = "Qwen/Qwen2.5-0.5B-Instruct-long"', 'pad_token = "<|endoftext|>"',
            'max_tokens = 4096']
    assert all(scan_secrets("f.py", ln) for ln in hit)
    assert not any(scan_secrets("f.py", ln) for ln in miss)
