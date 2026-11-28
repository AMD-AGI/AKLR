# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, Final

from aklr.core.validation.validators.verify._verdict import Verdict

if TYPE_CHECKING:
    from aklr.core.validation.validators.verify._extraction import Extraction
    from aklr.core.validation.validators.verify._volta import VoltaCheck, VoltaRun

__all__ = ["ISA_RULES", "IsaRule"]


class IsaRule:
    number: ClassVar[int]
    name: ClassVar[str]

    def judge(self, extraction: Extraction, run: VoltaRun | None) -> Verdict:
        if run is None:
            return self.concluded("unchecked", extraction.failure)
        if run.failure:
            return self.concluded("unchecked", run.failure)
        return self.verdict(run)

    def verdict(self, run: VoltaRun) -> Verdict:
        raise NotImplementedError

    def concluded(self, status: str, detail: str = "") -> Verdict:
        return Verdict(rule=self.name, status=status, detail=detail)


class _GraphRule(IsaRule):
    check: ClassVar[str]

    def verdict(self, run: VoltaRun) -> Verdict:
        check = run.checks.get(self.check)
        if check is None:
            raise RuntimeError(f"volta did not run its check of {self.check}")
        aborted = f"; the run aborted at {run.abort}" if run.abort else ""
        if check.status == "FAIL" and check.witnesses:
            status, witness = self.rejection(check)
            return self.concluded(status, witness + aborted)
        stopped = self.stopped(run)
        if stopped:
            return self.concluded("rejected", stopped)
        if run.incomplete:
            return self.concluded("unchecked", run.incomplete)
        if check.status == "UNKNOWN":
            reasons = "; ".join(text for text, _ in check.witnesses)
            return self.concluded("unchecked", f"unknown: {reasons}")
        return self.concluded("proved")

    def rejection(self, check: VoltaCheck) -> tuple[str, str]:
        raise NotImplementedError

    def stopped(self, run: VoltaRun) -> str:
        return ""


class _LdsRaceRule(_GraphRule):
    number = 1
    name = "lds_race"
    check = "memory-conflict freedom"

    _CONFLICT: ClassVar[re.Pattern[str]] = re.compile(
        r"MemoryConflict \{ kind: (\w+), location: ByteLocation \{ space: (\w+), addr: (\d+) \}"
    )

    def rejection(self, check: VoltaCheck) -> tuple[str, str]:
        text, events = check.witnesses[0]
        conflict = self._CONFLICT.match(text)
        if conflict is None or len(events) != 2:
            return "rejected", text
        witness = (
            f"{conflict[1]} conflict at {conflict[2]} 0x{int(conflict[3]):x}"
            f" between `{events[0]}` and `{events[1]}`"
        )
        if self._only_duplicate_loads(check):
            return "unchecked", f"duplicate-load (ptx_verifier#48): {witness}"
        return "rejected", witness

    def stopped(self, run: VoltaRun) -> str:
        return run.abort if re.search(r": LDS race at ", run.abort) else ""

    def _only_duplicate_loads(self, check: VoltaCheck) -> bool:
        return bool(check.witnesses) and all(
            (conflict := self._CONFLICT.match(text)) is not None
            and conflict[1] == "WriteWrite"
            and len(events) == 2
            and events[0] == events[1]
            and re.search(r": buffer_load_\w+ .* lds$", events[0])
            for text, events in check.witnesses
        )


class _BarrierEpochRule(_GraphRule):
    number = 2
    name = "barrier_epoch"
    check = "barrier epochs (named barriers)"

    _VIOLATION: ClassVar[re.Pattern[str]] = re.compile(
        r"EpochViolation \{ barrier: SyncObjectId\((\d+)\), generation: GenerationId\((\d+)\),"
        r" next: GenerationId\((\d+)\)"
    )

    def rejection(self, check: VoltaCheck) -> tuple[str, str]:
        text, events = check.witnesses[0]
        violation = self._VIOLATION.match(text)
        if violation is not None:
            text = (
                f"generation {violation[2]} of barrier {violation[1]}"
                f" is not separated from generation {violation[3]}"
            )
        return "rejected", text + (": " + ", ".join(f"`{e}`" for e in events) if events else "")

    def stopped(self, run: VoltaRun) -> str:
        return run.deadlock


class _AbortRule(IsaRule):
    def verdict(self, run: VoltaRun) -> Verdict:
        if run.abort and self.is_defect(run):
            return self.concluded("rejected", run.abort)
        if run.incomplete:
            return self.concluded("unchecked", run.incomplete)
        return self.concluded("proved")

    def is_defect(self, run: VoltaRun) -> bool:
        raise NotImplementedError


class _PendingReadRule(_AbortRule):
    number = 3
    name = "pending_read"

    def is_defect(self, run: VoltaRun) -> bool:
        return (
            re.search(r": (?:scalar|vector) register \S+ (?:in lane \d+ )?has a pending load$", run.abort)
            is not None
        )


class _LdsBoundsRule(_AbortRule):
    number = 4
    name = "lds_bounds"

    def is_defect(self, run: VoltaRun) -> bool:
        access = re.search(
            r": LDS access at 0x([0-9a-f]+) is misaligned or outside the (\d+)-byte allocation$", run.abort
        )
        if access is None:
            return False
        address = int(access[1], 16)
        return address % 4 == 0 or address >= int(access[2])


class _LdsUninitializedReadRule(IsaRule):
    number = 5
    name = "lds_uninitialized_read"

    def verdict(self, run: VoltaRun) -> Verdict:
        if re.search(r": LDS read at 0x[0-9a-f]+ of a word no wave has written$", run.abort):
            return self.concluded("unchecked", f"uninitialized-read: {run.abort}")
        if run.incomplete:
            return self.concluded("unchecked", run.incomplete)
        return self.concluded("proved")


class _GlobalBoundsRule(_AbortRule):
    number = 6
    name = "global_bounds"

    def is_defect(self, run: VoltaRun) -> bool:
        access = re.search(
            r": 4-byte (?:read|write) at 0x([0-9a-f]+) is not an element of an (?:input|output) array$",
            run.abort,
        )
        return access is not None and run.is_past_an_array(int(access[1], 16))


ISA_RULES: Final[tuple[IsaRule, ...]] = (
    _LdsRaceRule(),
    _BarrierEpochRule(),
    _PendingReadRule(),
    _LdsBoundsRule(),
    _LdsUninitializedReadRule(),
    _GlobalBoundsRule(),
)
