"""cam — Canonical Associative Memory research harness.

A flat, importable package consolidating the bind/recall/transfer experiments:

  * ``cam.bind_msweep``  — capacity ladder: fresh-bind DeepMemory at increasing M,
    report held-out carry vs chance (is the M ceiling architectural or under-training?).
  * ``cam.recall_mag``   — Stage-1 bind + Stage-2 Memory-as-Gate (MAG) delivery through a
    frozen base.
  * ``cam.recall_v1``    — cross-base transfer: one frozen memory served to a second base
    through a tiny learned translator.

The driver scripts import a small set of building blocks (adapters, the product-key store,
the graph-free deep Titans memory, gated taps, translators). Everything lives flat in this
package; importing a module never loads a model or requires a GPU — model loading is guarded
under ``if __name__ == "__main__"`` / explicit calls.

Run a driver either as a module or as a file from the repo root::

    python -m cam.recall_mag --store pk --addr-sup-weight 1.0 --M 8
    python cam/recall_mag.py  --store pk --addr-sup-weight 1.0 --M 8
"""

__all__ = [
    "bind_msweep",
    "recall_mag",
    "recall_v1",
    "recall_boltA",
    "recall_deepmem",
    "m2_adapter",
    "pk_store",
    "pk_store_adapter",
    "gated_tap",
    "translator",
    "deep_memory",
    "deep_mem_analytic",
    "store_recurrence",
]
