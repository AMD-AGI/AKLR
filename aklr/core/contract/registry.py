# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import itertools
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from aklr.core.contract import expr, locate
from aklr.core.contract.op import OpContract, api_name
from aklr.core.contract.spec import ContractId
from aklr.core.contract.spec._sources import decode_text

__all__ = ["ContractRegistry", "UnknownContractError"]


class UnknownContractError(KeyError):
    pass


class ContractRegistry:
    """Read-only after construction."""

    def __init__(self, contracts: Iterable[OpContract]) -> None:
        by_id, by_api = {}, {}
        for c in sorted(contracts, key=lambda c: c.contract_id):
            for by_key, key in ((by_id, c.contract_id), (by_api, c.api_name)):
                if key in by_key:
                    clash = f"{by_key[key].contract_id} and {c.contract_id}"
                    raise ValueError(f"{key!r} declared by {clash}")
                by_key[key] = c
        self._by_id = MappingProxyType(by_id)
        self._by_api = MappingProxyType(by_api)

    @classmethod
    @functools.cache
    def default(cls) -> ContractRegistry:
        """The ``CONTRACTS`` of every op package's ``contract`` module."""
        return cls(c for pkg in locate.op_packages() for c in locate.op_module(pkg, "contract").CONTRACTS)

    def get(self, contract_id: ContractId) -> OpContract:
        """The contract of ``contract_id``; else ``UnknownContractError``."""
        try:
            return self._by_id[contract_id]
        except KeyError:
            raise UnknownContractError(contract_id) from None

    def for_api(self, fn: Callable) -> OpContract:
        """The contract of the api function ``fn``; else ``UnknownContractError``."""
        name = api_name(fn)
        try:
            return self._by_api[name]
        except KeyError:
            raise UnknownContractError(name) from None

    def for_schema_json(self, data: Any) -> OpContract:
        """The contract the schema json ``data`` names; else ``ValueError``."""
        contract_id = data.get("contract_id") if isinstance(data, dict) else None
        try:
            return self.get(contract_id)
        # a hand-written contract_id may be a list or dict: unhashable
        except (UnknownContractError, TypeError):
            raise ValueError(f"unknown contract_id {contract_id!r}") from None

    def parse_schema(self, data: bytes, where: str) -> OpContract:
        """The schema whose ``schema_text`` is exactly ``data``, narrowed at its first
        workload's Point. Raises one ``ValueError`` listing every problem, each
        prefixed by ``where``."""
        text = decode_text(where, data)
        try:
            parsed = json.loads(text, parse_constant=expr.reject_json_constant)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}:{e.lineno}: not JSON: {e.msg}") from None
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        try:
            contract = self.for_schema_json(parsed)
            schema = contract.narrow(
                contract.point_from_json(parsed, parsed["workloads"][0]["axes"]),
                workloads=parsed["workloads"],
                kernel=parsed["kernel"],
                models=parsed.get("models", ()),
            )
            expected_text = schema.schema_text()
        except ValidationError as e:
            lines = OpContract.error_lines(where, parsed, e)
            raise ValueError("\n".join(lines)) from None
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            raise ValueError(f"{where}: not a schema: {e}") from None
        if parsed.get("name") != schema.name:
            raise ValueError(
                f"{where}: name {parsed.get('name')!r} is not {schema.name!r}: the "
                "naming rule (aklr/ops/README.md, Names) or a declaration order changed"
            )
        if text != expected_text:
            pairs = itertools.zip_longest(text.split("\n"), expected_text.split("\n"))
            line = next(n for n, (a, b) in enumerate(pairs, 1) if a != b)
            raise ValueError(
                f"{where}:{line}: not the schema text of {contract.contract_id} (its "
                "values at the Point, schema key order, indent 2, number lists on one "
                "line, one final newline)"
            )
        return schema

    def contracts(self) -> tuple[OpContract, ...]:
        """Sorted by ``contract_id``."""
        return tuple(self._by_id.values())

    def read_schemas(
        self,
        op_package: str,
        *,
        skipped_contract_ids: frozenset[ContractId] = frozenset(),
        measured_dir: Path | None = None,
    ) -> tuple[list[tuple[Path, OpContract]], list[str]]:
        """``(schemas passing every check, error lines)`` of an op's committed schema
        files, each read from ``measured_dir`` instead where that holds its repository
        path; skipped contracts' schemas are neither checked nor returned. Never
        raises."""
        passed: list[tuple[Path, OpContract]] = []
        errors: list[str] = []
        op_dir = locate.op_dir(op_package)
        for path in locate.committed_schema_paths(op_dir):
            if measured_dir is not None:
                measured = measured_dir.joinpath(*op_package.split("."), path.relative_to(op_dir))
                path = measured if measured.is_file() else path
            try:
                data = path.read_bytes()
                schema = self.parse_schema(data, str(path))
            except OSError as e:
                errors.append(f"{path}: {e.strerror or e}")
                continue
            except ValueError as e:
                if _raw_contract_id(data) not in skipped_contract_ids:
                    errors.append(str(e))
                continue
            if schema.contract_id in skipped_contract_ids:
                continue
            found = self.get(schema.contract_id).schema_errors(path, schema)
            if found:
                errors += found
            else:
                passed.append((path, schema))
        return passed, errors


def _raw_contract_id(data: bytes) -> str | None:
    """The ``contract_id`` string an unparsable schema file names, else None."""
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    contract_id = parsed.get("contract_id") if isinstance(parsed, dict) else None
    return contract_id if isinstance(contract_id, str) else None
