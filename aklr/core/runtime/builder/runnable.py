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
# not the MIT license that covers the rest of aiter (see LICENSE).
###############################################################################


from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from pydantic import Field

from aklr.core.schema import Schema, frozen_dataclass

__all__ = ["Runnable", "RunnableMetadata"]


@frozen_dataclass
class RunnableMetadata:
    """How a runnable was built and what it implements."""

    build_type: str
    """The language of the builder that produced it."""
    op_type: str
    """The op whose call keys it takes."""
    schema_name: str
    """Name of the kernel that was compiled into this runnable."""
    kernel_name: str
    """The name of the first schema it was built for."""
    destination_passing_style: bool = True
    """Whether the kernel uses destination-passing style."""
    schema: Optional[Schema] = None
    """``kernel.content_hash``: the registry's cache key."""
    misc: dict[str, Any] = Field(default_factory=dict)
    """Builder-specific extras: ``module_name``, ``entry_symbol``."""


class Runnable:
    """An executable wrapper around a compiled solution.

    A Runnable encapsulates a callable function along with metadata about how it was built
    and a cleanup function to release resources. It provides a uniform interface for
    executing solutions regardless of the build system or language used.
    """

    metadata: RunnableMetadata
    """Metadata about the build process and source solution."""

    _callable: Callable[..., Any]
    """The underlying callable function."""
    _cleaner: Optional[Callable[[], None]]
    """Optional cleanup function to release build artifacts and resources."""

    def __init__(
        self,
        callable: Callable[..., Any],
        metadata: RunnableMetadata,
        cleaner: Optional[Callable[[], None]] = None,
    ) -> None:
        """Constructor for the Runnable class.

        Parameters
        ----------
        callable : Callable[..., Any]
            The callable that is wrapped by the runnable.
        metadata : RunnableMetadata
            The metadata for the runnable.
        cleaner : Optional[Callable[[], None]]
            The cleaner function for the runnable. It will clean up the build artifacts/resources.
        """
        self._callable = callable
        self.metadata = metadata
        self._cleaner = cleaner

    def __call__(self, *args: Any) -> Any:
        """Execute the runnable with positional arguments.

        This method calls the underlying compiled function with the provided inputs.
        If the function returns a single-element tuple, it is automatically unpacked
        to a scalar value for convenience.

        Parameters
        ----------
        args : Any
            Positional arguments for the underlying function.

        Returns
        -------
        Any
            The result of the underlying function. Single-element tuples are unpacked
            to scalar values.
        """
        ret = self._callable(*args)
        if not isinstance(ret, tuple) or len(ret) > 1:
            return ret
        return ret[0] if ret else None

    def cleanup(self) -> None:
        """Clean up build artifacts and release resources.

        This method calls the cleaner function if one was provided during construction.
        It is idempotent: calling it multiple times is safe and has no additional effect
        after the first call.
        """
        if self._cleaner:
            try:
                self._cleaner()
            finally:
                self._cleaner = None
