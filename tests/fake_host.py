# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A fake host for the tests of ``Plugin`` and of the aiter provider: its families under a temporary
``aklr/plugins/<host>/``, and its modules, which exist only in ``sys.modules``."""

import importlib
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aklr.core.contract.op import OpContract
from aklr.core.plugin import Plugin
from aklr.plugins.sglang.kimi_k3_tiny_gemm.contract import KIMI_K3_TINY_GEMM

# a second contract of the same calls, as a family's contract.py below writes it; not a model_copy,
# which would carry the first one's cached domain
SECOND = OpContract(
    **{
        **{name: getattr(KIMI_K3_TINY_GEMM, name) for name in OpContract.model_fields},
        "contract_id": "second_gemm@1.0",
    }
)
# a family's contract.py as far as its statements: {host} is the host's name
_CONTRACT_PY = """\
from aklr.core.contract.op import OpContract
from aklr.plugins.{host} import plugin
from aklr.plugins.sglang.kimi_k3_tiny_gemm.contract import KIMI_K3_TINY_GEMM as FIRST

SECOND = OpContract(
    **{{**{{name: getattr(FIRST, name) for name in OpContract.model_fields}}, "contract_id": "second_gemm@1.0"}}
)
CONTRACTS = (FIRST, SECOND)

"""
# the line of a family's first statement in its contract.py
FIRST_STATEMENT_LINE = _CONTRACT_PY.count("\n") + 1


class FakeHost:
    """A fresh ``Plugin`` over a temporary ``aklr/plugins/<name>/``, switched on, in place of the installed
    one; and the host's own modules, each given as its dotted name and members."""

    def __init__(
        self,
        name: str,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        members_by_module: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.name, self.root, self.plugin = name, root / name, Plugin(name)
        self.root.mkdir()
        package = importlib.import_module(f"aklr.plugins.{name}")
        monkeypatch.setattr(package, "__path__", [str(self.root)])
        monkeypatch.setattr(package, "plugin", self.plugin)
        monkeypatch.setenv(f"AKLR_{name.upper()}_PLUGIN", "1")
        self.modules: dict[str, types.ModuleType] = {}
        for module_path, members in members_by_module.items():
            parts = module_path.split(".")
            for depth in range(1, len(parts) + 1):
                module_name = ".".join(parts[:depth])
                if module_name in self.modules:
                    continue
                module = self.modules[module_name] = types.ModuleType(module_name)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, module_name, module)
                if depth > 1:
                    setattr(self.modules[".".join(parts[: depth - 1])], parts[depth - 1], module)
            vars(self.modules[module_path]).update(members)

    def family(self, name: str, declarations: str, *, schemas: Mapping[str, str] | None = None) -> Path:
        """Write the family ``name``: a contract.py holding ``FIRST`` (Kimi-K3's tiny gemm contract) and
        ``SECOND``, with the statements ``declarations`` at its bottom, and a schema file holding
        ``contract_id`` for each ``impls/...`` path of ``schemas``. Returns its contract.py."""
        directory = self.root / name
        directory.mkdir()
        (directory / "__init__.py").touch()
        for path, contract_id in (schemas or {}).items():
            (directory / path).parent.mkdir(parents=True)
            (directory / path).write_text(f'{{"contract_id": "{contract_id}"}}')
        contract_py = directory / "contract.py"
        contract_py.write_text(_CONTRACT_PY.format(host=self.name) + f"{declarations}\n")
        return contract_py
