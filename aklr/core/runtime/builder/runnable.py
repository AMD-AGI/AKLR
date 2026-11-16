###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from pydantic import Field

from aklr.core.schema import Schema, frozen_dataclass

__all__ = ["Runnable", "RunnableMetadata"]


@frozen_dataclass
class RunnableMetadata:
    build_type: str
    op_type: str
    schema_name: str
    kernel_name: str
    destination_passing_style: bool = True
    schema: Optional[Schema] = None
    misc: dict[str, Any] = Field(default_factory=dict)


class Runnable:
    metadata: RunnableMetadata

    _callable: Callable[..., Any]
    _cleaner: Optional[Callable[[], None]]

    def __init__(
        self,
        callable: Callable[..., Any],
        metadata: RunnableMetadata,
        cleaner: Optional[Callable[[], None]] = None,
    ) -> None:
        self._callable = callable
        self.metadata = metadata
        self._cleaner = cleaner

    def __call__(self, *args: Any) -> Any:
        """Call the kernel; a one-element tuple result is unwrapped and an empty one becomes None."""
        ret = self._callable(*args)
        if not isinstance(ret, tuple) or len(ret) > 1:
            return ret
        return ret[0] if ret else None

    def cleanup(self) -> None:
        if self._cleaner:
            try:
                self._cleaner()
            finally:
                self._cleaner = None
