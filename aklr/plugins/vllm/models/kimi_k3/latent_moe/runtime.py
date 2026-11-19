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
mapped (``peer_table``, the ranks' base addresses), a device scratch, one launch counter per workgroup and the health
words.

Every launch of the kernel reads its workgroup's counter as its mailbox tag and bumps it as it ends, so every rank's
counters advance together as long as every rank launches the kernel the same number of times: the route declines
alike on every rank, and nothing here is ever zeroed between calls.

A rank whose in-kernel wait gives up (``WAIT_SECONDS``) records the wait in its health words and sets FAILED, which
is sticky: its later waits end at once, so no rank hangs. ``after_step`` stops every rank at the same step once any
of them failed.
"""

from __future__ import annotations

import ctypes
import logging
import os
import zlib
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "FAILED",
    "GRID",
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
                 addresses: Sequence[int]) -> None:  # fmt: skip
        self.cpu_group, self.rank, self.world, self.device = cpu_group, rank, world, device
        self._local, self._opened = local, opened
        self.peer_table = torch.tensor(list(addresses), dtype=torch.int64, device=device)
        self.scratch = torch.zeros(SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self.counters = torch.zeros(GRID, dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
        # the kernel builds this group launches: its wait timeout (tests shorten it) and which are compiled
        self.wait_seconds = WAIT_SECONDS
        self.warmed: set[Any] = set()
        self._failstop = _FailStop(self.health, rank, world, cpu_group)

    def barrier(self) -> None:
        """Every rank of the group here (the CPU group)."""
        tp_agree(True, self.cpu_group)

    def failed(self) -> bool:
        """Whether a wait of this rank gave up (a device read: it waits for the stream)."""
        return bool(self.health[FAILED].item())

    def given_up_waits(self, regions: Sequence[str]) -> list[str]:
        """Each workgroup's recorded wait that gave up; ``regions`` names the ids a record may hold."""
        names = {region_id(n): n for n in regions}
        records = self.health.view(GRID + 1, RECORD_WORDS)[1:].tolist()
        return [
            f"workgroup {cta}: {names.get(rid, rid)} at byte {offset} waited for tag {tag}, saw {seen}"
            for flag, rid, offset, tag, seen, cta, _, _ in records
            if flag
        ]

    def after_step(self) -> None:
        self._failstop.after_step()

    def close(self) -> None:
        """Unmap the peers and free this rank's buffer: only once no rank's kernel can still touch it."""
        opened, self._opened = self._opened, []
        for base in opened:
            _Hip.close(base)
        if self._local:
            _Hip.free(self._local)
            self._local = 0
        _GROUPS.pop(self.device, None)


def setup(cpu_group: Any, rank: int, world: int, device: torch.device) -> TailGroup | None:
    """This rank's group state, made collectively: every rank of ``cpu_group`` (a CPU process group of the ``world``
    ranks, ``rank`` this one's) must call it at the same point. None on every rank when any rank cannot take part
    (a GPU with fewer CUs than workgroups, a failed mapping); an earlier group of the device is kept."""
    device = torch.device(device)
    if device in _GROUPS:
        return _GROUPS[device]
    ok = torch.cuda.get_device_properties(device).multi_processor_count >= GRID
    if not tp_agree(ok, cpu_group):
        _LOGGER.warning("latent-MoE tail: a rank's GPU has fewer than %d CUs; the kernel is off", GRID)
        return None
    local, opened, addresses, failure = 0, [], [], None
    try:
        local = _Hip.alloc_uncached(PEER_BYTES)
        handles = _all_gather_objects(_Hip.handle(local), cpu_group) if world > 1 else [None]
        for peer, handle in enumerate(handles):
            if peer == rank:
                addresses.append(local)
            else:
                opened.append(_Hip.open(handle))
                addresses.append(opened[-1])
    except RuntimeError as e:  # every rank must still reach the agreement
        failure = e
    if not tp_agree(failure is None, cpu_group):
        for base in opened:
            _Hip.close(base)
        if local:
            _Hip.free(local)
        _LOGGER.warning("latent-MoE tail: the peer memory handshake failed on %s rank (%r); the kernel is off",
                        "this" if failure else "another", failure)  # fmt: skip
        return None
    group = _GROUPS[device] = TailGroup(cpu_group, rank, world, device, local, opened, addresses)
    _register_after_step()
    return group


def group_of(device: torch.device) -> TailGroup:
    """The group state ``setup`` made for ``device``; ``RuntimeError`` without one."""
    group = _GROUPS.get(torch.device(device))
    if group is None:
        raise RuntimeError(f"latent-MoE tail: no group on {device}: setup() runs on every rank first")
    return group


def after_step() -> None:
    """The fail-stop watch of every group of this process; once a step, on every rank (the vLLM plugin's step hook)."""
    for group in list(_GROUPS.values()):
        group.after_step()


def _register_after_step() -> None:
    from aklr.plugins import vllm as plugin

    if after_step not in plugin.AFTER_STEP:
        plugin.AFTER_STEP.append(after_step)


class _FailStop:
    """Stops every rank at the same step once any of them gave up an in-kernel wait (its FAILED word). Each step
    queues a copy of this rank's word to the host and keeps the newest that has landed, without blocking; every
    CHECK_EVERY steps the ranks agree on theirs over the CPU group and all raise together (a rank raising alone would
    leave the others in their next collective for ever)."""

    CHECK_EVERY = 16

    def __init__(self, health: torch.Tensor, rank: int, world: int, cpu_group: Any) -> None:
        self.failed = health[FAILED : FAILED + 1]
        self.rank, self.world, self.cpu_group = rank, world, cpu_group
        self.host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self.copied: torch.cuda.Event | None = None
        self.seen = False
        self.steps = 0

    def after_step(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        if self.copied is None or self.copied.query():
            if self.copied is not None:
                self.seen |= bool(self.host[0])
            self.host.copy_(self.failed, non_blocking=True)
            self.copied = torch.cuda.Event()
            self.copied.record()
        self.steps += 1
        if self.steps % self.CHECK_EVERY:
            return
        votes = torch.zeros(self.world, dtype=torch.int32)
        votes[self.rank] = int(self.seen)
        dist.all_reduce(votes, group=self.cpu_group)
        if votes.any():
            ranks = [r for r, v in enumerate(votes.tolist()) if v]
            raise RuntimeError(
                f"Kimi-K3 latent-MoE tail: TP ranks {ranks} gave up an in-kernel wait; stopping "
                "(AKLR_VLLM_MOE_TAIL=0 runs vLLM's tail)"
            )
