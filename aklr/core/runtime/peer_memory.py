# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Peer memory (``KernelSpec.peer_bytes``): bytes on every rank of a process group, each rank's
mapped into the others' address space, through which a kernel exchanges with its peers inside a
call (the receive buffers of an all-reduce). The integration that knows the ranks names the
group (``set_group``); without one a process is a group of one."""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

__all__ = ["PeerMemory", "open_peer_memory", "set_group"]

_HIP_DEVICE_MALLOC_UNCACHED = 0x3  # other GPUs write it while this one polls
_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
_HANDLE_BYTES = 64


class PeerMemory(NamedTuple):
    """One kernel's peer memory on one device: every rank's buffer address (int64, on the
    host), this rank's at ``rank``; ``owner`` keeps a group of one's buffer alive."""

    addresses: torch.Tensor
    rank: int
    owner: Any = None


_group: Callable[[], torch.distributed.ProcessGroup] | None = None


def set_group(group: Callable[[], torch.distributed.ProcessGroup] | None) -> None:
    """Open every later kernel's peer memory across the process group ``group()`` returns, at
    the kernel's first call on a device. Opening is collective: every rank must call the
    kernels in the same order. ``None``: a group of one.

    The group swaps the buffers' IPC handles as host tensors, so it is a gloo group: a first
    call can come just before a graph capture, and an NCCL collective issued there can end
    the process from NCCL's watchdog thread."""
    global _group
    _group = group


def open_peer_memory(nbytes: int, device: torch.device) -> PeerMemory:
    """``nbytes`` of zeroed peer memory on ``device`` on every rank of the group: uncached
    device memory mapped through HIP IPC handles swapped over the group, kept for the
    process's life, or a group of one's tensor."""
    group = None if _group is None else _group()
    if group is None:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError("peer memory in a process of several ranks needs set_group")
        local = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        return PeerMemory(torch.tensor([local.data_ptr()], dtype=torch.int64), 0, local)
    hip = _hip()
    rank = torch.distributed.get_rank(group)
    with torch.cuda.device(device):
        local = ctypes.c_void_p()
        _check(
            hip.hipExtMallocWithFlags(ctypes.byref(local), nbytes, _HIP_DEVICE_MALLOC_UNCACHED),
            "hipExtMallocWithFlags",
        )
        _check(hip.hipMemset(local, 0, nbytes), "hipMemset")
        # zeroed before any other rank can write it
        _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        handle = _IpcHandle()
        _check(hip.hipIpcGetMemHandle(ctypes.byref(handle), local), "hipIpcGetMemHandle")
        mine = torch.frombuffer(bytearray(bytes(handle)), dtype=torch.uint8)
        n_ranks = torch.distributed.get_world_size(group)
        handles = [torch.empty_like(mine) for _ in range(n_ranks)]
        torch.distributed.all_gather(handles, mine, group=group)
        addresses = []
        for peer, raw in enumerate(h.numpy() for h in handles):
            if peer == rank:
                addresses.append(local.value)
                continue
            mapped = ctypes.c_void_p()
            _check(
                hip.hipIpcOpenMemHandle(
                    ctypes.byref(mapped),
                    _IpcHandle.from_buffer_copy(raw.tobytes()),
                    _HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
                ),
                "hipIpcOpenMemHandle",
            )
            addresses.append(mapped.value)
    return PeerMemory(torch.tensor(addresses, dtype=torch.int64), rank)


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * _HANDLE_BYTES)]


@functools.cache
def _hip() -> ctypes.CDLL:
    """The HIP runtime torch loaded, not another copy: device pointers and IPC handles belong
    to one runtime."""
    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        paths = {
            fields[5].strip()
            for fields in (line.split(maxsplit=5) for line in maps)
            if len(fields) == 6 and "/libamdhip64.so" in fields[5]
        }
    if len(paths) != 1:
        raise RuntimeError(f"expected one HIP runtime in this process, found {sorted(paths)}")
    hip = ctypes.CDLL(paths.pop())
    void_pp, size, flags = (
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    )
    signatures = {
        "hipExtMallocWithFlags": (void_pp, size, flags),
        "hipMemset": (ctypes.c_void_p, ctypes.c_int, size),
        "hipDeviceSynchronize": (),
        "hipIpcGetMemHandle": (ctypes.POINTER(_IpcHandle), ctypes.c_void_p),
        "hipIpcOpenMemHandle": (void_pp, _IpcHandle, flags),
    }
    for name, argtypes in signatures.items():
        fn = getattr(hip, name)
        fn.argtypes, fn.restype = argtypes, ctypes.c_int
    return hip


def _check(status: int, call: str) -> None:
    if status != 0:
        raise RuntimeError(f"{call} failed: hipError_t {status}")
