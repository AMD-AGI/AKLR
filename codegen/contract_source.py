# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contracts written as the python source that builds them."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel

from aklr.core.contract.op import OpContract

__all__ = ["ContractSource"]

_WIDTH = 110  # the repository's line length
_BRANCHES = ("distribution=", "error_bounds=")  # a tensor's fields of (predicate, item) branches


class ContractSource:
    """The python source building ``contracts``, each under its name, as a contract.py writes them: the
    constructors of the public names, nothing a default already says, a call that fits on one line."""

    def __init__(self, contracts: Mapping[str, OpContract]) -> None:
        self.contracts = dict(contracts)
        self._imports: dict[str, set[str]] = {}  # each public package -> the names the source takes of it

    def text(self) -> str:
        """The imports, an assignment per contract and ``CONTRACTS``."""
        self._imports = {}
        assignments = "".join(
            f"\n{name} = {self._written(contract, _WIDTH, len(name) + 3)}\n"
            for name, contract in self.contracts.items()
        )
        imports = "".join(
            f"from {package} import {', '.join(sorted(names))}\n"
            for package, names in sorted(self._imports.items())
        )
        listed = ", ".join(self.contracts) + ("," if len(self.contracts) == 1 else "")
        return f"{imports}{assignments}\nCONTRACTS = ({listed})\n"

    def _written(self, value: Any, room: int, taken: int = 0) -> str:
        """``value``, a part of a contract, as the expression that builds it: on one line where the
        line, ``room`` columns wide with ``taken`` of them before the expression, holds it, else an item a
        line, each written the same way."""
        if isinstance(value, Enum):
            # an enum the public packages do not export is written as its value, which the field takes
            is_named = self._home(type(value)) is not None
            return f"{type(value).__name__}.{value.name}" if is_named else self._written(value.value, room)
        if isinstance(value, str):
            text = repr(value)
            return f'"{text[1:-1]}"' if text[0] == "'" and '"' not in value else text
        if isinstance(value, BaseModel):
            if self._home(type(value)) is None:
                raise ValueError(f"{type(value).__name__} has no public name to be written by")
            fields = type(value).model_fields
            items = [
                (f"{name}=", getattr(value, name))
                for name, field in fields.items()
                if getattr(value, name) != field.get_default(call_default_factory=True)
            ]
            # a tensor's one unconditional branch is written as the item alone, as its field takes it
            items = [
                (
                    prefix,
                    item[0][1] if prefix in _BRANCHES and len(item) == 1 and item[0][0] is None else item,
                )
                for prefix, item in items
            ]
            opening, closing = f"{type(value).__name__}(", ")"
        elif isinstance(value, Mapping):
            items = [(f"{self._written(key, room)}: ", item) for key, item in value.items()]
            opening, closing = "{", "}"
        elif isinstance(value, tuple | list):
            items = [("", item) for item in value]
            opening, closing = "(", ",)" if len(value) == 1 else ")"
        else:
            return repr(value)
        flat = opening + ", ".join(prefix + self._written(item, 10**6) for prefix, item in items) + closing
        if taken + len(flat) < room or not items:  # a column is left for the comma after it
            return flat
        lines = [opening]
        for prefix, item in items:
            written = prefix + self._written(item, room - 4, len(prefix)) + ","
            lines += [f"    {line}" for line in written.split("\n")]
        return "\n".join([*lines, closing.lstrip(",")])

    def _home(self, cls: type) -> str | None:
        """The public package exporting ``cls``, noted as an import of the source; ``None`` when none does."""
        package = cls.__module__.partition("._")[0]
        if getattr(importlib.import_module(package), cls.__name__, None) is not cls:
            return None
        self._imports.setdefault(package, set()).add(cls.__name__)
        return package
