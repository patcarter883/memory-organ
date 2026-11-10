"""ngram_memory: portable row writer for native and grafted n-gram tables (DESIGN.md)."""
from .spec import (FormatSpec, KeySpec, ModuleSpec, TableSpec, flash_next_format, flash_next_key_spec,
                   graft_format, graft_key_spec, qwen25_proxy_key_spec)

__all__ = ["FormatSpec", "KeySpec", "ModuleSpec", "TableSpec", "flash_next_format", "flash_next_key_spec",
           "graft_format", "graft_key_spec", "qwen25_proxy_key_spec"]
