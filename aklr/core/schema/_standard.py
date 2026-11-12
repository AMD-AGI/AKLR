###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A schema's test standard, defined with the schema in code and never written to its json."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from aklr.core.schema._check import default_check
from aklr.core.schema._frozen_model import frozen_dataclass
from aklr.core.schema._function_source import FunctionSource


@frozen_dataclass
class Standard:
    """What a kernel of the schema is held to: init draws its inputs, reference and check judge its outputs, and
    baseline is the backend call it must beat. Each is called with the schema's names as keywords."""

    reference: Callable[..., Any]
    init: Optional[Callable[..., Dict[str, Any]]] = None
    check: Callable[..., bool] = default_check
    # the backend's call on the inputs and outputs, ready to time (its set-up done), or None where it declines them
    baseline: Optional[Callable[..., Optional[Callable[[], Any]]]] = None

    def save(self, directory: Path) -> None:
        """Each function as a standalone module directory/<name>.py binding it to ``run``: torch and, for the
        baseline, its backend are all it imports."""
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
