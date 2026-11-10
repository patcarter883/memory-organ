"""Import the fork's Qwen4ExpPLE on a CPU box by stubbing third-party modules it never uses on the compute path."""
import sys, types, importlib.abc, importlib.machinery
from unittest import mock
STUB = {"msgpack", "zmq", "uvloop", "fastapi", "uvicorn", "pydantic", "sentencepiece", "tokenizers", "triton", "aiter", "hip", "flashinfer", "sgl_kernel"}
class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUB:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; return m
    def exec_module(self, m): pass
sys.meta_path.insert(0, _F())
