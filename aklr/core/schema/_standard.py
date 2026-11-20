###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from aklr.core.schema._check import default_check
from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.schema._function_source import FunctionSource


@frozen_dataclass
class Standard:
    reference: Callable[..., Any]
    init: Optional[Callable[..., Dict[str, Any]]] = None
    check: Callable[..., bool] = default_check
    baseline: Optional[Callable[..., Optional[Callable[[], Any]]]] = None

    def save(self, directory: Path) -> None:
        preambles = {
            "init": "from __future__ import annotations\nimport math\nimport torch\n",
            "reference": (
                "from __future__ import annotations\nimport math\nimport torch\nimport torch.nn.functional as F\n"
            ),
            "check": "from __future__ import annotations\nimport torch\n",
            "baseline": "from __future__ import annotations\nimport os\nimport torch\n",
        }
        directory.mkdir(parents=True, exist_ok=True)
        for name, preamble in preambles.items():
            fn = getattr(self, name)
            path = directory / f"{name}.py"
            if fn is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(FunctionSource(preamble).render(fn), encoding="utf-8")
