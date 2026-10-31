# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from aklr.core.contract.export import CallRecord, write_calls
from aklr.core.contract.fastkey import CallArgs, FastKey, FastKeyCodec
from aklr.core.contract.op import OpContract
from aklr.core.contract.spec import ContractId, PointSpec, ScalarSpec
from aklr.core.platform.device import DeviceTarget

if TYPE_CHECKING:
    from aklr.core.runtime.dispatch._dispatcher import FallbackReason

__all__ = ["CallRecorder"]

_LOGGER = logging.getLogger(__name__)


class _RecordKey(NamedTuple):
    contract_id: ContractId
    fast_key: FastKey | None
    scalars: tuple[tuple[str, bool | int | float], ...]
    reason: FallbackReason | None

    @classmethod
    def from_call(
        cls,
        contract: OpContract,
        fast_key: FastKey | None,
        call: CallArgs,
        reason: FallbackReason | None,
    ) -> _RecordKey:
        if fast_key is None:
            scalars = ()
        else:
            scalars = tuple(
                (name, spec.normalize(call[name]))
                for name, spec in contract.inputs.items()
                if isinstance(spec, ScalarSpec) and call[name] is not None
            )
        return cls(contract.contract_id, fast_key, scalars, reason)

    def to_call_record(self, point: PointSpec | None, target: DeviceTarget, n_calls: int) -> CallRecord:
        return CallRecord(
            contract_id=self.contract_id,
            point=point,
            scalars=dict(self.scalars),
            fallback_reason=None if self.reason is None else self.reason.value,
            target=target,
            n_calls=n_calls,
        )


class CallRecorder:
    """Counts this process's calls by record key while exporting, and keeps its call file."""

    __slots__ = (
        "_codecs",
        "_has_unsaved_key",
        "_has_warned_save_failure",
        "_saved_at",
        "_warned_scalar_contract_ids",
        "calls",
        "export_dir",
        "target",
    )
    calls: Counter[_RecordKey]
    export_dir: Path
    target: DeviceTarget | None
    _codecs: dict[ContractId, FastKeyCodec]
    _has_unsaved_key: bool  # a record key counted since the last save began
    _has_warned_save_failure: bool
    _saved_at: float  # time.monotonic() when the last save began
    _warned_scalar_contract_ids: set[ContractId]

    def __init__(
        self,
        export_dir: Path,
        target: DeviceTarget | None,
        previous_recorder: CallRecorder | None = None,
    ) -> None:
        self.export_dir = export_dir
        if previous_recorder is None:
            self.calls = Counter()
            self._codecs = {}
            self.target = target
        else:
            self.calls = previous_recorder.calls
            self._codecs = previous_recorder._codecs
            self.target = previous_recorder.target if target is None else target
        self._has_warned_save_failure = False
        # -inf: the first new key saves at once
        self._has_unsaved_key, self._saved_at = False, float("-inf")
        self._warned_scalar_contract_ids = set()

    def count(
        self,
        contract: OpContract,
        fast_key: FastKey | None,
        call: CallArgs,
        reason: FallbackReason | None,
    ) -> None:
        """Count one call; the first new key, and then at most one per second, saves the file."""
        contract_id = contract.contract_id
        if fast_key is not None and contract_id not in self._codecs:
            self._codecs[contract_id] = FastKeyCodec.from_contract(contract)
        try:
            record_key = _RecordKey.from_call(contract, fast_key, call, reason)
        except ValueError:
            if contract_id not in self._warned_scalar_contract_ids:
                self._warned_scalar_contract_ids.add(contract_id)
                _LOGGER.warning(
                    "%s: scalars %r cannot be normalized; calls like this one are served but not recorded",
                    contract_id,
                    {
                        name: call[name]
                        for name, spec in contract.inputs.items()
                        if isinstance(spec, ScalarSpec)
                    },
                )
            return
        if record_key not in self.calls:
            self._has_unsaved_key = True
        self.calls[record_key] += 1
        # a save rewrites every key, so one per new key is O(N^2) over a token-count sweep;
        # checked on every call so a held key does not wait for exit (killed workers never
        # reach atexit)
        # shortcut: a fixed 1 s; past ~100k keys a save itself nears 1 s, scale the interval
        # by the save's cost then
        if self._has_unsaved_key and time.monotonic() - self._saved_at >= 1.0:
            self.save_or_warn()

    def save(self) -> None:
        """Rewrite this process's call file from the counted calls; an ``OSError`` propagates."""
        # before the snapshot: a key counted meanwhile stays unsaved
        self._has_unsaved_key, self._saved_at = False, time.monotonic()
        write_calls(
            self.export_dir,
            [
                record_key.to_call_record(
                    (
                        None
                        if record_key.fast_key is None
                        else self._codecs[record_key.contract_id].to_point(record_key.fast_key)
                    ),
                    self.target,
                    n_calls,
                )
                for record_key, n_calls in list(self.calls.items())
            ],
        )

    def save_or_warn(self) -> None:
        """``save``, but an ``OSError`` is only warned about, once."""
        try:
            self.save()
        except OSError as e:
            if not self._has_warned_save_failure:
                self._has_warned_save_failure = True
                _LOGGER.warning("cannot write call records, serving on: %s", e)
