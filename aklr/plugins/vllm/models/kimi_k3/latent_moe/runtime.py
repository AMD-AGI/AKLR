###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright contributors to the vLLM project
#
# Adapted from vLLM (https://github.com/vllm-project/vllm)
# Source: vllm/models/kimi_k3/amd/specialized_decode/runtime/peer.py (PeerBuffer, tp_agree; itself adapted from
# ROCm/ATOM at a526f0d, MIT License) and runtime/guards.py (FailStop), as in vLLM 0.32.1rc1.dev276
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
# Modified for aklr/plugins/vllm/models/kimi_k3/latent_moe: HIP's IPC calls made here (as aiter's MIT-licensed
# UncachedIpcHeap makes them) instead of through aiter; per-workgroup launch counters in place of the step epoch;
# the fail-stop watch run from the vLLM plugin's step hook
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

"""What the latent-MoE tail kernel keeps per rank between calls: an uncached buffer every rank of the group has
mapped (``peer_table``, the ranks' base addresses) with the group's FAILED word at its end, a device scratch, one
launch counter per workgroup and the health words.

Every launch of the kernel reads its workgroup's counter as its mailbox tag and bumps it as it ends, so every rank's
counters advance together as long as every rank launches the kernel the same number of times: the route declines
alike on every rank, and nothing here is ever zeroed between calls.

A rank whose in-kernel wait gives up (``WAIT_SECONDS``) records the wait in its health words, sets its FAILED word,
and sets the group's (``GROUP_FAILED``) in every rank's buffer before it stores anything more; a wait that ends
because the group's word is set (they all do, soon: so no rank hangs) sets it on every rank the same way. Both are
sticky. So whatever data any rank reads from a wait that gave up, its own group word was set first: a rank's group
word, read after its step's launches, says whether its step saw one.

``after_step`` (the vLLM plugin's step watch) queues a copy of this rank's group word to pinned memory once a step,
on a stream of its own after the step's launches: no collective, nothing waits. Its check runs before the step's
output leaves the worker (as vLLM's output thread reads it) and raises once the word is set.
"""

from __future__ import annotations

import ctypes
import logging
import os
import zlib
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "FAILED",
    "GRID",
    "GROUP_FAILED",
    "HEALTH_WORDS",
    "PEER_BYTES",
    "RECORD_WORDS",
    "SCRATCH_BYTES",
    "WAIT_SECONDS",
    "TailGroup",
    "after_step",
    "group_of",
    "region_id",
    "setup",
    "tp_agree",
]

_LOGGER = logging.getLogger(__name__)

GRID = 256  # the kernel's workgroups, one a CU, every one resident at once (a wait spins on the others)
PEER_BYTES = 16 << 20  # a rank's peer buffer, where the kernel lays out its mailbox regions
SCRATCH_BYTES = 2 << 20  # a rank's device scratch (hand-offs between its workgroups)
RECORD_WORDS = 8  # a health record: given up, region, byte offset, tag, seen tag, workgroup, clock, 0
FAILED = 0  # the health word set once a wait of this rank gave up
# the byte offset in every rank's peer buffer of the group's FAILED word, past the kernel's mailbox regions: set on
# every rank by any rank's wait that gives up or ends because it is set
GROUP_FAILED = PEER_BYTES - 64
HEALTH_WORDS = (GRID + 1) * RECORD_WORDS  # FAILED's record, then one a workgroup
WAIT_SECONDS = 10  # an in-kernel wait gives up after this long

_GROUPS: dict[torch.device, TailGroup] = {}


def region_id(name: str) -> int:
    """A mailbox region's id in a health record: its name's CRC-32 (31 bits)."""
    return zlib.crc32(name.encode()) & 0x7FFFFFFF


def tp_agree(ok: bool, group: Any) -> bool:
    """True when every rank of ``group`` (a CPU process group, or None for one rank) passed ``ok``; every rank must
    call it at the same point."""
    if group is None:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


class _Hip:
    """The HIP runtime calls of uncached IPC memory, through the runtime torch has loaded."""

    HANDLE_BYTES = 64
    UNCACHED = 0x3  # hipExtMallocWithFlags: hipDeviceMallocUncached
    LAZY_PEER_ACCESS = 0x1  # hipIpcOpenMemHandle: hipIpcMemLazyEnablePeerAccess
    _lib = None

    class Handle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_byte * 64)]

    @classmethod
    def lib(cls):
        if cls._lib is None:
            for name in ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"):
                try:
                    lib = ctypes.CDLL(name, mode=os.RTLD_NOLOAD)
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("the HIP runtime is not loaded")
            void_p, uint = ctypes.c_void_p, ctypes.c_uint
            for fn, args in (
                ("hipExtMallocWithFlags", [ctypes.POINTER(void_p), ctypes.c_size_t, uint]),
                ("hipMemset", [void_p, ctypes.c_int, ctypes.c_size_t]),
                ("hipMemcpyAsync", [void_p, void_p, ctypes.c_size_t, ctypes.c_int, void_p]),
                ("hipDeviceSynchronize", []),
                ("hipIpcGetMemHandle", [ctypes.POINTER(cls.Handle), void_p]),
                ("hipIpcOpenMemHandle", [ctypes.POINTER(void_p), cls.Handle, uint]),
                ("hipIpcCloseMemHandle", [void_p]),
                ("hipFree", [void_p]),
            ):
                getattr(lib, fn).restype = ctypes.c_int
                getattr(lib, fn).argtypes = args
            lib.hipGetErrorString.restype = ctypes.c_char_p
            lib.hipGetErrorString.argtypes = [ctypes.c_int]
            cls._lib = lib
        return cls._lib

    @classmethod
    def check(cls, err: int, what: str) -> None:
        if err:
            raise RuntimeError(f"{what}: {cls.lib().hipGetErrorString(err).decode(errors='replace')}")

    @staticmethod
    def compute_units(device: torch.device) -> int:
        return torch.cuda.get_device_properties(device).multi_processor_count

    @classmethod
    def alloc_uncached(cls, nbytes: int) -> int:
        """Zeroed uncached device memory on the current device, whole pages (IPC handles of uncached memory need
        them); the zeros land before a peer can map it."""
        lib, ptr = cls.lib(), ctypes.c_void_p()
        nbytes = -(-nbytes // 4096) * 4096
        cls.check(lib.hipExtMallocWithFlags(ctypes.byref(ptr), nbytes, cls.UNCACHED), "hipExtMallocWithFlags")
        cls.check(lib.hipMemset(ptr, 0, nbytes), "hipMemset")
        cls.check(lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
        return int(ptr.value)

    @classmethod
    def zero(cls, ptr: int, nbytes: int) -> None:
        """Zero device memory, the device idle after."""
        cls.check(cls.lib().hipMemset(ctypes.c_void_p(ptr), 0, nbytes), "hipMemset")
        cls.check(cls.lib().hipDeviceSynchronize(), "hipDeviceSynchronize")

    @classmethod
    def copy_word(cls, host: torch.Tensor, address: int, stream: int) -> None:
        """The int32 at device ``address`` into ``host`` (one pinned int32), queued on ``stream``."""
        lib, d2h = cls.lib(), 2  # hipMemcpyDeviceToHost
        err = lib.hipMemcpyAsync(ctypes.c_void_p(host.data_ptr()), ctypes.c_void_p(address), 4, d2h,
                                 ctypes.c_void_p(stream))  # fmt: skip
        cls.check(err, "hipMemcpyAsync")

    @classmethod
    def handle(cls, ptr: int) -> bytes:
        h = cls.Handle()
        cls.check(cls.lib().hipIpcGetMemHandle(ctypes.byref(h), ctypes.c_void_p(ptr)), "hipIpcGetMemHandle")
        return bytes(h.reserved)

    @classmethod
    def open(cls, handle: bytes) -> int:
        h, ptr = cls.Handle(), ctypes.c_void_p()
        ctypes.memmove(ctypes.byref(h), handle, cls.HANDLE_BYTES)
        cls.check(
            cls.lib().hipIpcOpenMemHandle(ctypes.byref(ptr), h, cls.LAZY_PEER_ACCESS), "hipIpcOpenMemHandle"
        )
        return int(ptr.value)

    @classmethod
    def close(cls, ptr: int) -> None:
        cls.check(cls.lib().hipIpcCloseMemHandle(ctypes.c_void_p(ptr)), "hipIpcCloseMemHandle")

    @classmethod
    def free(cls, ptr: int) -> None:
        cls.check(cls.lib().hipFree(ctypes.c_void_p(ptr)), "hipFree")


def _all_gather_objects(obj: Any, group: Any) -> list[Any]:
    """``obj`` of every rank of ``group``, in group-rank order (CPU broadcasts)."""
    ranks = dist.get_process_group_ranks(group)
    objs = [[None] for _ in ranks]
    objs[dist.get_rank(group)][0] = obj
    for i, global_rank in enumerate(ranks):
        dist.broadcast_object_list(objs[i], src=global_rank, group=group, device="cpu")
    return [o[0] for o in objs]


class TailGroup:
    """One rank's state of the kernel's group; built by ``setup``, collectively."""

    def __init__(self, cpu_group: Any, rank: int, world: int, device: torch.device, local: int, opened: list[int],
                 addresses: Sequence[int], hip: Any = _Hip) -> None:  # fmt: skip
        self.cpu_group, self.rank, self.world, self.device = cpu_group, rank, world, device
        self._local, self._opened, self._hip = local, opened, hip
        self.peer_table = torch.tensor(list(addresses), dtype=torch.int64, device=device)
        self.scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self.counters = torch.zeros(GRID, dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
        # the kernel builds this group launches: its wait timeout (tests shorten it) and which are compiled
        self.wait_seconds = WAIT_SECONDS
        self.warmed: set[Any] = set()
        self._failstop = _FailStop(hip, local + GROUP_FAILED, device)

    def barrier(self) -> None:
        """Every rank of the group here (the CPU group)."""
        tp_agree(True, self.cpu_group)

    def failed(self) -> bool:
        """Whether a wait of this rank gave up (a device read: it waits for the stream)."""
        return bool(self.health[FAILED].item())

    def clear_failed(self) -> None:
        """This rank's health words and its group FAILED word cleared: a tool's, between its cases, with no launch in
        flight on any rank (the device idle after)."""
        self.health.zero_()
        self._hip.zero(self._local + GROUP_FAILED, 4)

    def given_up_waits(self, regions: Sequence[str]) -> list[str]:
        """Each workgroup's recorded wait that gave up; ``regions`` names the ids a record may hold."""
        names = {region_id(n): n for n in regions}
        records = self.health.view(GRID + 1, RECORD_WORDS)[1:].tolist()
        return [
            f"workgroup {cta}: {names.get(rid, rid)} at byte {offset} waited for tag {tag}, saw {seen}"
            for flag, rid, offset, tag, seen, cta, _, _ in records
            if flag
        ]

    def after_step(self) -> Callable[[], None] | None:
        """This step's check (see ``after_step``); every rank calls it at the same point."""
        return self._failstop.after_step()

    def close(self) -> None:
        """Unmap the peers and free this rank's buffer: only once no rank's kernel can still touch it."""
        opened, self._opened = self._opened, []
        for base in opened:
            self._hip.close(base)
        if self._local:
            self._hip.free(self._local)
            self._local = 0
        _GROUPS.pop(self.device, None)


def setup(cpu_group: Any, rank: int, world: int, device: torch.device, hip: Any = _Hip) -> TailGroup | None:
    """This rank's group state, made collectively: every rank of ``cpu_group`` (a CPU process group of the ``world``
    ranks, ``rank`` this one's) must call it at the same point. None on every rank when any rank cannot take part
    (a GPU with fewer CUs than workgroups, a failed allocation, export or mapping); an earlier group of the device is
    kept. ``hip`` makes the HIP calls (``_Hip``'s).

    The ranks agree after each step that can fail on one rank alone, before any of them goes on to the next
    collective: a rank that skipped one the others entered would leave them waiting in it for ever."""
    device = torch.device(device)
    if device in _GROUPS:
        return _GROUPS[device]
    failure = None
    try:
        if hip.compute_units(device) < GRID:
            failure = RuntimeError(f"a GPU of fewer than {GRID} CUs")
    except Exception as e:  # every rank must still reach the agreement
        failure = e
    if not _agreed(failure, cpu_group, "the GPU check"):
        return None
    local = handle = None
    try:
        local = hip.alloc_uncached(PEER_BYTES)
        handle = hip.handle(local) if world > 1 else None
    except Exception as e:
        failure = e
    if not _agreed(failure, cpu_group, "allocating or exporting the peer buffer"):
        _release(hip, [], local)
        return None
    handles = _all_gather_objects(handle, cpu_group) if world > 1 else [None]
    opened, group = [], None
    try:
        addresses = []
        for peer, peer_handle in enumerate(handles):
            if peer == rank:
                addresses.append(local)
            else:
                opened.append(hip.open(peer_handle))
                addresses.append(opened[-1])
        group = TailGroup(cpu_group, rank, world, device, local, opened, addresses, hip)
    except Exception as e:
        failure = e
    if not _agreed(failure, cpu_group, "mapping the peers' buffers"):
        _release(hip, opened, local)
        return None
    _GROUPS[device] = group
    _register_after_step()
    return group


def _release(hip: Any, opened: list[int], local: int | None) -> None:
    """Unmap ``opened`` and free ``local`` after the ranks agreed to give up: a failure here is only logged, as no
    rank waits on this one any more."""
    try:
        for base in opened:
            hip.close(base)
        if local:
            hip.free(local)
    except Exception as e:
        _LOGGER.warning("latent-MoE tail: releasing the peer memory failed (%r)", e)


def _agreed(failure: BaseException | None, cpu_group: Any, what: str) -> bool:
    """Whether no rank's ``failure`` is set (every rank calls it at the same point); the warning where one is."""
    if tp_agree(failure is None, cpu_group):
        return True
    _LOGGER.warning("latent-MoE tail: %s failed on %s rank (%r); the kernel is off",
                    what, "this" if failure else "another", failure)  # fmt: skip
    return False


def group_of(device: torch.device) -> TailGroup:
    """The group state ``setup`` made for ``device``; ``RuntimeError`` without one."""
    group = _GROUPS.get(torch.device(device))
    if group is None:
        raise RuntimeError(f"latent-MoE tail: no group on {device}: setup() runs on every rank first")
    return group


def after_step() -> Callable[[], None] | None:
    """The fail-stop watch of every group of this process, the vLLM plugin's step watch: once a step, on every rank,
    after the step's launches. Its check, run before the step's output leaves the worker, raises once the group's
    FAILED word was set in that step or before; None inside a graph capture (no host call there) or without a
    group."""
    checks = [
        check for check in (group.after_step() for group in list(_GROUPS.values())) if check is not None
    ]
    if not checks:
        return None

    def check() -> None:
        for one in checks:
            one()

    return check


def _register_after_step() -> None:
    from aklr.plugins import vllm as plugin

    if after_step not in plugin.AFTER_STEP:
        plugin.AFTER_STEP.append(after_step)


class _FailStop:
    """A step's verdict: this rank's group FAILED word as the step's launches left it, copied to a pinned slot on a
    stream of its own once the step's launches are done (the step's next kernels do not wait for the copy, nor does
    the host: no collective, no device read). Its check waits for that copy alone, in vLLM's output thread, and
    raises where the word is set. A slot is reused RING steps on, long after vLLM has read the step's output. The
    word is sticky, so a copy never reads it clearer than the step left it."""

    RING = 64

    def __init__(self, hip: Any, word: int, device: torch.device) -> None:
        self.hip, self.word = hip, word
        cuda = device.type == "cuda"
        self.host = torch.zeros(self.RING, dtype=torch.int32, pin_memory=cuda)
        self.stream = torch.cuda.Stream(device) if cuda else None
        # each slot's events: the step's launches done (on the step's stream), its copy landed
        self.ready = [torch.cuda.Event() for _ in range(self.RING)] if cuda else None
        self.copied = [torch.cuda.Event(blocking=True) for _ in range(self.RING)] if cuda else None
        self.steps = 0

    def after_step(self) -> Callable[[], None] | None:
        if torch.cuda.is_current_stream_capturing():
            return None
        slot = self.steps % self.RING
        self.steps += 1
        host, copied = self.host[slot : slot + 1], None
        if self.stream is None:
            self.hip.copy_word(host, self.word, None)
        else:
            self.ready[slot].record(torch.cuda.current_stream(self.stream.device))
            self.stream.wait_event(self.ready[slot])
            self.hip.copy_word(host, self.word, self.stream.cuda_stream)
            copied = self.copied[slot]
            copied.record(self.stream)

        def check() -> None:
            if copied is not None:
                copied.synchronize()
            if host.item():
                raise RuntimeError(
                    "Kimi-K3 latent-MoE tail: a TP rank gave up an in-kernel wait in this step or before (each rank's "
                    "TailGroup.given_up_waits lists its own); stopping (AKLR_VLLM_MOE_TAIL=0 runs vLLM's tail)"
                )

        return check
