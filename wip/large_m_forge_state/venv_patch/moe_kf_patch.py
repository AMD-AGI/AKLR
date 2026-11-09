"""Campaign-local patch (this venv only): KernelForge hard-codes a 300 s ceiling on one driver bench, and this
MoE task with a slow torch starting kernel needs longer. Raises the default bench timeout of
IterationConfig and GateMeasurement once their modules are imported; explicit values still win."""

import importlib.abc
import importlib.machinery
import os
import sys

BENCH_TIMEOUT_SEC = int(os.environ.get("MOE_BENCH_TIMEOUT_SEC", "1800"))
_TARGETS = {
    "kernelforge.loop.runner": "IterationConfig",
    "kernelforge.loop.gate_settings": "GateMeasurement",
}


def _patch(module):
    cls = getattr(module, _TARGETS[module.__name__], None)
    if cls is None or getattr(cls, "_moe_patched", False):
        return
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("bench_timeout_sec", BENCH_TIMEOUT_SEC)
        original(self, *args, **kwargs)

    cls.__init__ = __init__
    cls._moe_patched = True


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
