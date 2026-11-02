# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every route still matches the sglang method it hooks, on the installed sglang."""

import importlib
import inspect
import sys
from importlib.metadata import version

import pytest
from packaging.version import Version

from aklr.plugins.sglang import hook

DSV4 = "aklr.plugins.sglang.models.deepseek_v4"
COMMON = (
    "routes",
    "compressor_routes",
    "dp_routes",
    "engram_routes",
    "metadata_routes",
    "proposal",
    "accept",
    "inject",
    "radix_routes",
    "tokenize_routes",
)
_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _resolve(target: str):
    """The object at a dotted path: the longest importable module, then attributes."""
    parts = target.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for name in parts[i:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(target)


def _mismatch(route, method, around: bool):
    """Why sglang's call of ``method`` would not bind to ``route`` as it does to the
    method, or None."""
    params = list(inspect.signature(route).parameters.values())
    if around:
        if not params or params[0].name != "orig":
            return "an around route takes sglang's method first, as orig"
        params = params[1:]
    expected = list(inspect.signature(method).parameters.values())
    kinds = {p.kind for p in params}
    positional = [p.name for p in params if p.kind in _POSITIONAL]
    method_positional = [p.name for p in expected if p.kind in _POSITIONAL]
    if inspect.Parameter.VAR_POSITIONAL in kinds:
        method_positional = method_positional[: len(positional)]
    if positional != method_positional:
        return f"positional {positional}, sglang's {method_positional}"
    names = {p.name for p in expected}
    keyword = {p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
    if not keyword <= names:
        return f"keywords {sorted(keyword - names)} sglang does not pass"
    if inspect.Parameter.VAR_KEYWORD not in kinds:
        method_keyword = {p.name for p in expected if p.kind is inspect.Parameter.KEYWORD_ONLY}
        if not method_keyword <= keyword:
            return f"sglang's keywords {sorted(method_keyword - keyword)} are not taken"
    return None


@pytest.mark.gpu  # sglang's models initialize the GPU at import
def test_each_route_matches_the_sglang_method(monkeypatch):
    """On the installed sglang every target of the DeepSeek-V4.1 routes exists, and its
    route binds the arguments sglang calls the method with."""
    pytest.importorskip("sglang.srt.models.deepseek_v4")
    monkeypatch.setattr(hook, "_ROUTES", [])
    package = importlib.import_module(DSV4)
    every = ("compat", "runtime_routes") + COMMON
    # imported afresh: compat sees the installed sglang (not another test's stand-ins),
    # and the routes declare into this test's _ROUTES
    for name in every:
        monkeypatch.delitem(sys.modules, f"{DSV4}.{name}", raising=False)
        monkeypatch.delattr(package, name, raising=False)
    pytest.importorskip("sglang.srt.monokernels.spec")
    for name in COMMON + ("runtime_routes",):
        importlib.import_module(f"{DSV4}.{name}")
    running = Version(version("sglang")).base_version
    routes = [r for r in hook._ROUTES if r[1].contains(running)]
    assert routes, f"no route for sglang {running}"
    bad = {}
    for target, _, route, around in routes:
        try:
            method = _resolve(target)
        except (ImportError, AttributeError) as e:
            bad[target] = f"not in sglang {running}: {e}"
            continue
        if (why := _mismatch(route, method, around)) is not None:
            bad[target] = why
    assert bad == {}
