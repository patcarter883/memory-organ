"""Stage-1 evaluation (§6.A): splits (reader of the ingest split files), teacher-forced scoring at a fixed in-file
context, token classes, repository QA, R3 capability, eval sets, and baselines/controls (baselines/).

Entry point: scripts/evaluate.py. Acceptance checks: scripts/repro_ctx_len.py (S1/rt-science/ctx_len.py) and
scripts/repro_tokdelta.py (S1/evalproto2/tokdelta.py)."""
