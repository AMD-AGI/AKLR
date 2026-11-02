# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A long prompt's tokens from cached encodes of the text between its special tokens, which
a fast tokenizer splits out first; the first calls are checked against the tokenizer's own
encode. ``AKLR_DSV41_TOKEN_CACHE=0`` keeps sglang's encode."""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import MODEL_TYPE
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

_LOGGER = logging.getLogger(__name__)

ENV = "AKLR_DSV41_TOKEN_CACHE"
MANAGER = "sglang.srt.managers.tokenizer_manager.TokenizerManager"
# shorter texts gain nothing worth the split
MIN_CHARS = 1 << 14
# the cached segments' characters at most
CACHE_CHARS = 1 << 28
# the first calls of at least MIN_CHARS checked against the tokenizer's own encode
CHECKS = 3

SGLANG_READS = {
    "srt/managers/tokenizer_manager.py": (
        "TokenizerManager",
        "tokenizer",
        "model_config",
    ),
}


def _enabled() -> bool:
    value = os.environ.get(ENV) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV}={value!r}: expected '1' or '0'")
    return value == "1"


class _SegmentEncode:
    """``tokenizer.encode`` for plain text: the special tokens' ids, and each run of text
    between them encoded once and kept (LRU by characters)."""

    def __init__(self, tokenizer, encode):
        self._encode = encode
        added = tokenizer.added_tokens_decoder
        specials = {t.content: i for i, t in added.items() if t.special}
        self._ids = specials
        self._split = re.compile(
            "(" + "|".join(map(re.escape, sorted(specials, key=len, reverse=True))) + ")"
        )
        self._prefix = list(encode(""))
        self._cache: OrderedDict[str, list[int]] = OrderedDict()
        self._chars = 0
        self._checks = CHECKS
        self.off = None  # why the cache is off

    def __call__(self, text=None, *args, **kwargs):
        if (
            self.off is not None
            or args
            or not isinstance(text, str)
            or len(text) < MIN_CHARS
            or kwargs.get("add_special_tokens", True) is not True
            or set(kwargs) - {"add_special_tokens"}
        ):
            return self._encode(text, *args, **kwargs)
        ids = self._segments(text)
        if self._checks > 0:
            self._checks -= 1
            want = self._encode(text, **kwargs)
            if ids != list(want):
                self.off = f"segment encode != encode for a text of {len(text)} chars"
                stats.set_info("token_cache", self.off)
                _LOGGER.warning("DeepSeek-V4.1 token cache off: %s", self.off)
                return want
            stats.set_info("token_cache", "segments (checked)")
        return ids

    def _segments(self, text: str) -> list[int]:
        out = list(self._prefix)
        cache, ids = self._cache, self._ids
        for part in self._split.split(text):
            if not part:
                continue
            tok = ids.get(part)
            if tok is not None:
                out.append(tok)
                continue
            hit = cache.get(part)
            if hit is None:
                hit = self._encode(part, add_special_tokens=False)
                cache[part] = hit
                self._chars += len(part)
                while self._chars > CACHE_CHARS and cache:
                    old, _ = cache.popitem(last=False)
                    self._chars -= len(old)
            else:
                cache.move_to_end(part)
            out.extend(hit)
        return out


@sglang_route(f"{MANAGER}.__init__", versions=VERSIONS, around=True)
def _init(orig, self, *args, **kwargs):
    orig(self, *args, **kwargs)
    tokenizer = getattr(self, "tokenizer", None)
    hf_config = getattr(self.model_config, "hf_config", None)
    if (
        not _enabled()
        or tokenizer is None
        or not getattr(tokenizer, "is_fast", False)
        # DeepSeek-V4 has the same architecture
        or getattr(hf_config, "model_type", None) != MODEL_TYPE
    ):
        return
    try:
        tokenizer.encode = _SegmentEncode(tokenizer, tokenizer.encode)
    except Exception as e:  # noqa: BLE001
        stats.set_info("token_cache", f"off: {type(e).__name__}: {e}")
        _LOGGER.warning("DeepSeek-V4.1 token cache off: %s", e)
        return
    stats.set_info("token_cache", "segments")
    _LOGGER.info("DeepSeek-V4.1: long prompts encoded from cached segments (%s)", ENV)
