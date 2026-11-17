# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Print, as JSON, the keyword arguments AKLR calls a schema's entry with (``params``), the outputs among them in
the order the contract returns them (``outputs``), and its contract's baseline function as ``[module, name]``, or
null when the contract has none. argv: the AKLR checkout, then the
schema file."""

import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

from aklr.core.contract.registry import ContractRegistry

raw = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
contract = ContractRegistry.default().for_schema_json(raw)
# The builder binds the entry with the narrowed schema's call keys, a subset of the contract's api params.
schema = contract.narrow(
    contract.point_from_json(raw, raw["workloads"][0]["axes"]),
    workloads=raw["workloads"],
    kernel=raw["kernel"],
)
baseline = contract.baseline()
print(
    json.dumps(
        {
            "params": list(schema.kernel_call_keys),
            "outputs": list(contract.outputs),
            "baseline": baseline and [baseline.__module__, baseline.__name__],
        }
    )
)
