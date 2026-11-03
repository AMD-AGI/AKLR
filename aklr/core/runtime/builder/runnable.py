###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Source: flashinfer_bench/compile/runnable.py
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
# Modified for aklr/core/runtime/builder
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################


from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["Runnable", "RunnableMetadata"]


class RunnableMetadata(BaseModel):
    """How a runnable was built and what it implements."""

    build_type: str
    """The language of the builder that produced it."""
    contract_id: str
    """The contract whose call keys it takes."""
    schema_name: str
    """The name of the first schema it was built for."""
    content_hash: str
    """``kernel.content_hash``: the registry's cache key."""
    misc: dict[str, Any] = Field(default_factory=dict)
    """Builder-specific extras: ``module_name``, ``entry_symbol``."""


class Runnable:
    """A schema's imported kernel, called with one call's api params (``api_params`` names) as keywords; it fills the outputs."""

    metadata: RunnableMetadata

    _callable: Callable[..., Any]
    _cleaner: Callable[[], None] | None

    def __init__(
        self,
        fn: Callable[..., Any],
        metadata: RunnableMetadata,
        cleaner: Callable[[], None] | None = None,
    ) -> None:
        self._callable = fn
        self.metadata = metadata
        self._cleaner = cleaner

    def __call__(self, **call: Any) -> Any:
        return self._callable(**call)

    def cleanup(self) -> None:
        """Run the cleaner once (the kernel's package leaves ``sys.modules``); safe to call more than once."""
        if self._cleaner:
            try:
                self._cleaner()
            finally:
                self._cleaner = None
