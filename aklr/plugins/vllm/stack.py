# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Room on CPython's frame stack for a vLLM step.

CPython 3.11+ keeps frames in chunks: a call that does not fit in the current chunk maps a
new one and its return unmaps it at once, so a call repeated at a chunk boundary maps and
unmaps memory each time (about 2 us, more with many threads). vLLM's eager steps make
thousands of such calls (aiter's MoE launches alone), and a route's frame moves the boundary
onto them. ``roomy`` runs a function from a fresh chunk with room for a whole step."""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["roomy"]

# frames past one chunk's room (16 KiB) force a chunk of their own; CPython sizes it to the
# frame plus 1000 words, rounded up to a power of two: here 64 KiB, about 30 KiB to spare
_FRAME_WORDS = 4200


def _make() -> Callable:
    body = "\n".join(f"        _{i} = None" for i in range(_FRAME_WORDS))
    source = f"def roomy(fn, *args, **kwargs):\n    if False:\n{body}\n    return fn(*args, **kwargs)\n"
    scope: dict = {}
    exec(compile(source, "<aklr roomy>", "exec"), scope)
    roomy = scope["roomy"]
    roomy.__doc__ = "``fn(*args, **kwargs)``, its frames in a chunk of their own."
    return roomy


roomy = _make()
