# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Print, as JSON, the keyword arguments AKLR calls a schema's entry with (``params``) and its contract's
baseline function as ``[module, name]``, or null when the contract has none. argv: the AKLR checkout, then the
schema file."""

import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

from aklr.core.contract.registry import ContractRegistry

contract = ContractRegistry.default().for_schema_json(
    json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
)
baseline = contract.baseline()
print(
    json.dumps(
        {
            "params": [param.name for param in contract.api_params],
            "baseline": baseline and [baseline.__module__, baseline.__name__],
        }
    )
)
