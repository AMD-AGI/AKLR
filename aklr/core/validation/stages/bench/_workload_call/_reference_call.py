###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""A schema's Reference, called the way the bench calls it."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from aklr.core.schema import Schema

__all__ = ["ReferenceCall"]


class ReferenceCall:
    def __init__(self, schema: Schema) -> None:
        self.reference = schema.standard.reference
        self.declared = inspect.signature(self.reference).parameters

    def outputs(self, input_sets: Sequence[Mapping[str, Any]], rank: int) -> list[Any]:
        """The Reference's outputs on input_sets[rank], every rank's input sets given as group_calls when it
        declares them."""
        given = {**input_sets[rank], "group_calls": input_sets}
        expected = self.reference(**{n: v for n, v in given.items() if n in self.declared})
        return list(expected) if isinstance(expected, (tuple, list)) else [expected]
