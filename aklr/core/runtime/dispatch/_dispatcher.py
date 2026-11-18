# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Created by OpenAI

from __future__ import annotations

import ast
import functools
import inspect
import logging
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

import torch

from aklr.core.contract.closure import (
    compile_closure,
    fill_expr,
    fill_statements,
)
from aklr.core.contract.fastkey import (
    AllocOutputsFnGenerator,
    CallArgs,
    FastKey,
    FastKeyCodec,
    FastKeyFnGenerator,
)
from aklr.core.contract.op import OpContract
from aklr.core.contract.spec import TOOLCHAINS, WorkloadSpec, axes_text
from aklr.core.platform.env import ENV_DISABLE
from aklr.core.platform.toolchain import toolchain_version
from aklr.core.runtime.builder import Runnable, RunnableMetadata
from aklr.core.runtime.dispatch._trace import trace

if TYPE_CHECKING:
    from aklr.core.runtime.dispatch._recorder import CallRecorder

__all__ = ["DispatchTable", "FallbackReason", "KernelDispatcher", "kernel_failure_hint"]

_LOGGER = logging.getLogger(__name__)


def kernel_failure_hint(contract_id: str, language: str) -> str:
    """What a failing kernel's message adds: its toolchain here, and the switch that
    turns its op off."""
    toolchain = TOOLCHAINS[language]
    version = toolchain_version(toolchain) or "not installed"
    return f"{toolchain} {version} here; {ENV_DISABLE}={contract_id} turns this op off"


class FallbackReason(str, Enum):
    """Why an api call did not run a AKLR kernel; the caller then runs its own."""

    NO_SCHEMA = "no_schema"
    # AKLR_DISABLE
    DISABLED = "disabled"
    # fast_key returned None: no Point describes the call
    OUT_OF_CONTRACT = "out_of_contract"
    # keyed, but no loaded schema has a workload of this Point
    NO_WORKLOAD = "no_workload"
    # keyed, but no candidate has a speedup
    NOT_MEASURED = "not_measured"
    INACCURATE = "inaccurate"
    # measured passed, but none reaches AKLR_MIN_SPEEDUP
    NOT_QUALIFIED = "not_qualified"
    # only inside a graph capture: its Point never ran outside one
    NOT_WARMED = "not_warmed"


_FALLBACK_PRECEDENCE: Final = (
    FallbackReason.NOT_QUALIFIED,
    FallbackReason.INACCURATE,
    FallbackReason.NOT_MEASURED,
)


def _is_range_safe(runnable: Runnable) -> bool:
    """Whether the kernel's module sets ``RANGE_SAFE = True``: it builds no variant per value of a bucketed axis
    (a launch cached per token count compiles on every new one while serving), so a range's unmeasured values cost
    no more than its measured ones."""
    metadata = getattr(runnable, "metadata", None)
    name = None if metadata is None else metadata.misc.get("module_name")
    return bool(getattr(sys.modules.get(name) if name else None, "RANGE_SAFE", False))


@dataclass(frozen=True, slots=True, eq=False)
class DispatchTable:
    """One contract's committed kernels on one device, keyed by each Point's FastKey as
    ``match_key`` (``FastKeyCodec.match_key``) leaves it."""

    # the closures FastKeyFnGenerator and AllocOutputsFnGenerator make
    fast_key: Callable[[CallArgs], FastKey | None]
    alloc_outputs: Callable[[FastKey, CallArgs], Any]
    runnables: dict[FastKey, Runnable]
    warmed_keys: set[FastKey]
    # keys with candidates but no qualified one; a plain dict, as runnables
    fallback_reasons: dict[FastKey, FallbackReason]
    match_key: Callable[[FastKey], FastKey]
    # a contract with a bucketed axis: a call's exact key (``FastKeyCodec.exact_key``), served first where a
    # workload measured there qualifies; None without one
    exact_key: Callable[[FastKey], FastKey] | None = None
    # a served range key -> per bucketed axis, the lowest and highest value its schema measured in that range: an
    # unmeasured value is served only between them (a kernel's tile classes end at its last measured value, past which
    # it may build per value or not run at all)
    spans: dict[FastKey, tuple[tuple[int, int], ...]] = field(default_factory=dict)
    bucketed_indices: tuple[int, ...] = ()

    @classmethod
    def from_schemas(
        cls,
        contract: OpContract,
        device: torch.device,
        schema_runnables: Iterable[tuple[OpContract, Runnable, Iterable[WorkloadSpec]]],
        *,
        min_speedup: float,
    ) -> DispatchTable:
        """The table serving, per Point of the given workloads."""
        codec = FastKeyCodec.from_contract(contract)
        best: dict[FastKey, tuple[tuple[bool, float, str], Runnable]] = {}
        spans: dict[FastKey, tuple[tuple[int, int], ...]] = {}
        fallback_reasons: dict[FastKey, FallbackReason] = {}
        bucketed = bool(codec.bucketed_indices)
        for schema, runnable, workloads in schema_runnables:
            # workloads differing in any_size axes alone share a key: the schema serves it
            # only if each of them qualifies, at the lowest rank among them; with a bucketed
            # axis, each workload's own value is a key, served before its range's, which a
            # kernel serves only if it declares RANGE_SAFE (``_is_range_safe``)
            ranges = not bucketed or _is_range_safe(runnable)
            keyed: dict[FastKey, list[WorkloadSpec]] = {}
            measured: dict[FastKey, list[FastKey]] = {}  # a range key -> the measured keys in it
            for workload in workloads:
                fast_key = codec.to_fast_key(schema.point(workload.axes))
                if ranges:
                    keyed.setdefault(codec.match_key(fast_key), []).append(workload)
                    if bucketed:
                        measured.setdefault(codec.match_key(fast_key), []).append(fast_key)
                if bucketed:
                    keyed.setdefault(codec.exact_key(fast_key), []).append(workload)
            for key, same in keyed.items():
                failing = [w for w in same if not w.qualifies(min_speedup)]
                if not failing:
                    rank = min(map(schema.rank_key, same))
                    # strict: an equal rank shares a content hash, i.e.
                    if key not in best or rank > best[key][0]:
                        best[key] = (rank, runnable)
                        if key in measured:
                            spans[key] = tuple(
                                (min(k[i] for k in measured[key]), max(k[i] for k in measured[key]))
                                for i in codec.bucketed_indices
                            )
                for workload in failing:
                    if workload.accuracy == "failed":
                        reason = FallbackReason.INACCURATE
                    elif workload.speedup is None:
                        reason = FallbackReason.NOT_MEASURED
                    else:
                        reason = FallbackReason.NOT_QUALIFIED
                    fallback_reasons[key] = min(
                        fallback_reasons.get(key, reason),
                        reason,
                        key=_FALLBACK_PRECEDENCE.index,
                    )
        return cls(
            FastKeyFnGenerator(codec, device).generate(),
            AllocOutputsFnGenerator(codec, device).generate(),
            {key: runnable for key, (_, runnable) in best.items()},
            set(),
            {key: reason for key, reason in fallback_reasons.items() if key not in best},
            codec.match_key,
            codec.exact_key if bucketed else None,
            {key: span for key, span in spans.items() if key in best},
            codec.bucketed_indices,
        )

    @property
    def closed_reason(self) -> FallbackReason:
        """Why a dispatcher given this table closes instead (no runnable, no recorder)."""
        return min(
            self.fallback_reasons.values(),
            key=_FALLBACK_PRECEDENCE.index,
            default=FallbackReason.NOT_MEASURED,
        )

    def lookup(self, call: CallArgs) -> tuple[FastKey | None, FastKey | None, Runnable | FallbackReason]:
        """``call``'s key, its match and the kernel serving it, else why none does."""
        key = self.fast_key(call)
        if key is None:
            return None, None, FallbackReason.OUT_OF_CONTRACT
        runnable = exact = None
        if self.exact_key is not None:
            match = exact = self.exact_key(key)
            runnable = self.runnables.get(match)
        if runnable is None:
            match = self.match_key(key)
            runnable = self.runnables.get(match)
            span = self.spans.get(match) if runnable is not None else None
            if span is not None and not all(
                lo <= key[i] <= hi for i, (lo, hi) in zip(self.bucketed_indices, span, strict=True)
            ):
                runnable = None  # outside what its schema measured in the range
        if runnable is None:
            reason = (
                self.fallback_reasons.get(match)
                or self.fallback_reasons.get(exact)
                or (
                    FallbackReason.NO_WORKLOAD
                    if self.runnables or self.fallback_reasons
                    else FallbackReason.NO_SCHEMA
                )
            )
            return key, match, reason
        if match not in self.warmed_keys and torch.cuda.is_current_stream_capturing():
            return key, match, FallbackReason.NOT_WARMED
        return key, match, runnable


class KernelDispatcher:
    """One contract's dispatch: the api function (``make_api_fn``) calls ``dispatch``."""

    __slots__ = (
        "closed_reason",
        "contract",
        "has_warned_not_warmed",
        "recorder",
        "table",
    )
    contract: OpContract
    table: DispatchTable | None  # None: every call returns closed_reason, unkeyed
    closed_reason: FallbackReason  # read only while table is None
    recorder: CallRecorder | None  # export on; None whenever table is None
    has_warned_not_warmed: bool

    def __init__(self, contract: OpContract) -> None:
        self.contract = contract
        self.closed_reason = FallbackReason.NO_SCHEMA
        self.has_warned_not_warmed = False
        self.recorder = None
        self.table = None

    def dispatch(self, call: CallArgs) -> Any:
        """The api's result, or a ``FallbackReason``.

        The result is the outputs: one alone, else a tuple in ``contract.outputs`` order,
        the caller's buffers where ``call`` holds them, else allocated first. The kernel
        fills them.
        """
        table = self.table
        if table is None:
            if (tracer := trace()) is not None and tracer.is_op:
                tracer.op(self.contract, None, self.closed_reason)
            return self.closed_reason
        key, match, runnable = table.lookup(call)
        if isinstance(runnable, FallbackReason):
            result = runnable
            if result is FallbackReason.NOT_WARMED and not self.has_warned_not_warmed:
                self.has_warned_not_warmed = True  # one warning per dispatcher
                _LOGGER.warning(
                    "%s: %s was first called inside a graph capture: it falls "
                    "back there and in every replay; call each Point once "
                    "outside a capture first (later Points are not named)",
                    self.contract.contract_id,
                    axes_text(FastKeyCodec.from_contract(self.contract).to_point(key).axes),
                )
        else:  # a warmed hit, or the first eager call (it may compile)
            result = table.alloc_outputs(key, call)
            try:
                runnable(**call)
            except Exception:
                point = FastKeyCodec.from_contract(self.contract).to_point(key)
                _LOGGER.error(
                    "%s: the kernel serving %s at %s failed (%s)",
                    self.contract.contract_id,
                    self.contract.name_at(point),
                    axes_text(point.axes),
                    kernel_failure_hint(self.contract.contract_id, runnable.metadata.build_type),
                )
                raise
            table.warmed_keys.add(match)
        if (tracer := trace()) is not None and tracer.is_op:
            tracer.op(self.contract, key, runnable)
        if self.recorder is not None:
            self.recorder.count(
                self.contract,
                key,
                call,
                result if isinstance(result, FallbackReason) else None,
            )
        return result

    def fallback_reason(self, call: CallArgs) -> FallbackReason | None:
        """``None`` if ``dispatch`` would serve ``call``, else its reason, counted as
        ``dispatch`` counts it; nothing runs."""
        served = self.serving(call)
        return served if isinstance(served, FallbackReason) else None

    def serving(self, call: CallArgs) -> RunnableMetadata | FallbackReason:
        """The kernel ``dispatch`` would serve ``call`` with, else its reason, counted as
        ``dispatch`` counts it; nothing runs."""
        table = self.table
        if table is None:
            if (tracer := trace()) is not None and tracer.is_op:
                tracer.op(self.contract, None, self.closed_reason)
            return self.closed_reason
        key, _, runnable = table.lookup(call)
        if not isinstance(runnable, FallbackReason):
            return runnable.metadata
        if (tracer := trace()) is not None and tracer.is_op:
            tracer.op(self.contract, key, runnable)
        if self.recorder is not None:
            self.recorder.count(self.contract, key, call, runnable)
        return runnable

    def set_table(self, table: DispatchTable, recorder: CallRecorder | None = None) -> None:
        """Serve by ``table``, counting into ``recorder`` unless ``None``."""
        self.recorder = recorder
        self.table = table

    def close(self, reason: FallbackReason) -> None:
        """Return ``reason`` for every call, uncounted even while exporting."""
        self.closed_reason = reason
        self.recorder = None
        self.table = None

    def make_api_fn(self, fn: Callable) -> Callable:
        """``fn`` checked against ``contract`` (``check_api_signature``), and the api function
        replacing it, whose ``fallback_reason`` and ``serving`` ask without serving."""
        self.contract.check_api_signature(fn)
        api_fn = self._generate(fn, "dispatch")
        api_fn.fallback_reason = self._generate(fn, "fallback_reason")
        api_fn.serving = self._generate(fn, "serving")
        return api_fn

    def _generate(self, fn: Callable, method: str) -> Callable:
        """A function of ``fn``'s signature calling ``method`` on the call it forms."""
        code = fn.__code__
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        args, kwonly = names[: code.co_argcount], names[code.co_argcount :]
        params = inspect.signature(fn).parameters
        contract_names = {p.name for p in self.contract.api_params}
        # Defaults go through scope, not as constants: an enum is no constant.
        defaults = {p.name: p.default for p in self.contract.api_params}
        defaults |= {n: params[n].default for n in names if n not in contract_names}
        module = ast.parse(f"def f():\n    return _dispatcher.{method}()")
        node = module.body[0]
        node.name = fn.__name__
        node.args.args = [ast.arg(name) for name in args]
        node.args.kwonlyargs = [ast.arg(name) for name in kwonly]
        node.args.kw_defaults = [None] * len(kwonly)
        guards = [
            fill_statements(
                "if P is not D and (D is None or P != D):\n    return _out_of_contract",
                P=name,
                D=fill_expr("_d[N]", N=ast.Constant(name)),
            )[0]
            for name in names
            if name not in contract_names
        ]
        call = ast.Dict(
            [ast.Constant(p.name) for p in self.contract.api_params],
            [
                (
                    ast.Name(p.name, ast.Load())
                    if p.name in names
                    else fill_expr("_d[N]", N=ast.Constant(p.name))
                )
                for p in self.contract.api_params
            ],
        )
        node.body[0].value.args = [call]
        node.body[:0] = guards
        scope = {
            "__builtins__": {},
            "_dispatcher": self,
            "_d": defaults,
            "_out_of_contract": FallbackReason.OUT_OF_CONTRACT,
        }
        origin = f"<api:{self.contract.contract_id}>"
        generated = compile_closure(module, fn.__name__, scope, origin)
        functools.update_wrapper(generated, fn)
        generated.__defaults__ = fn.__defaults__
        generated.__kwdefaults__ = fn.__kwdefaults__
        return generated
