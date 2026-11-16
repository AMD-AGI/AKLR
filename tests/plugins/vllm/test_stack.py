# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A vLLM step's frames in a chunk of their own."""

from aklr.plugins.vllm.stack import _FRAME_WORDS, roomy

CHUNK_BYTES = 16 * 1024


def test_roomy_calls_through():
    assert roomy(lambda *args, **kwargs: (args, kwargs), 1, 2, k=3) == ((1, 2), {"k": 3})


def test_roomy_frame_does_not_fit_a_chunk():
    assert roomy.__code__.co_nlocals * 8 > CHUNK_BYTES
    assert roomy.__code__.co_nlocals >= _FRAME_WORDS
