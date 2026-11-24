# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's latent-MoE tail: the committed kernel on 8 gfx950 GPUs, 8 ranks of one torch.distributed group (a
process each, spawned): its out against the reference on every rank, its in-place write of the shared partial, graph
replays, dispatch through the route and the state the route sets up, a late rank and the fail-stop (the kernel's own
waits giving up, no fake HIP), and a setup that fails on one rank. A test needs 8 gfx950 GPUs and skips on fewer."""

from __future__ import annotations

import contextlib
import datetime
import importlib
import multiprocessing
import os
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from aklr.core.contract.locate import committed_schema_paths, op_dir
from aklr.core.contract.registry import ContractRegistry
from aklr.core.runtime.builder import FlyDSLBuilder
from aklr.core.runtime.dispatch import DispatcherRegistry, DispatchTable, FallbackReason
from aklr.plugins import vllm as plugin
from aklr.plugins.vllm.models.kimi_k3.latent_moe import api, failstop
from aklr.plugins.vllm.models.kimi_k3.latent_moe.contract import KIMI_K3_LATENT_MOE_TAIL

TP, HIDDEN, LATENT = 8, 7168, 3584
EPS = 1e-5
# below 28 rows kernel_small.py serves, from 28 kernel.py; every one a workload of the committed schema
ROWS = (1, 10, 20, 32, 40, 120, 128)


def on_gpus(fn, tmp_path, *args, timeout=600):
    """``fn(rank, *args)`` on 8 spawned processes, rank r on GPU r, in one torch.distributed group (gloo for the CPU,
    NCCL for the GPUs); each rank's result in rank order. A rank not done within ``timeout`` s fails the test."""
    if torch.cuda.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    store = (tmp_path / "store").as_uri()
    with multiprocessing.get_context("spawn").Pool(TP) as pool:
        jobs = [pool.apply_async(_rank_main, (fn, rank, store, args)) for rank in range(TP)]
        return [job.get(timeout=timeout) for job in jobs]


def _rank_main(fn, rank, store, args):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=store,
        rank=rank,
        world_size=TP,
        timeout=datetime.timedelta(seconds=300),
        device_id=torch.device("cuda", rank),
    )
    try:
        return fn(rank, *args)
    finally:
        dist.destroy_process_group()


def _kernel():
    """(schema, runnable, entry module, runtime module) of the committed kernel, built as dispatch builds it."""
    (path,) = committed_schema_paths(op_dir(KIMI_K3_LATENT_MOE_TAIL.op_package))
    schema = ContractRegistry.default().parse_schema(path.read_bytes(), str(path))
    runnable = FlyDSLBuilder().build(schema)
    name = runnable.metadata.misc["module_name"]
    return schema, runnable, sys.modules[name], importlib.import_module(f"{name.rpartition('.')[0]}.runtime")


def _draw(m, rank, device, seed):
    """This rank's call as the contract draws it: N(0, 1) partials and norm weight, LeCun up-projection rows."""
    gen = torch.Generator(device).manual_seed(seed * 1000 + rank)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(torch.bfloat16)

    return dict(
        routed=randn(m, LATENT),
        shared=randn(m, HIDDEN),
        up_weight=randn(HIDDEN // TP, LATENT, scale=LATENT**-0.5),
        norm_weight=randn(LATENT),
        tp_size=TP,
        eps=EPS,
        out_dtype=None,
    )


def _gather(tensor):
    parts = [torch.empty_like(tensor) for _ in range(TP)]
    dist.all_gather(parts, tensor.contiguous())
    return parts


def _all(ok, device):
    """Whether ``ok`` holds on every rank."""
    flag = torch.tensor([int(bool(ok))], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _barrier():
    """Returns once every rank has reached it: a CPU collective, which allocates nothing on a GPU."""
    dist.all_reduce(torch.ones(1))


def _computes(rank, rows, replays):
    device = torch.device("cuda", rank)
    _, _, kernel, runtime = _kernel()
    results = {}
    for m in rows:
        call = _draw(m, rank, device, seed=m)
        given = call["shared"].clone()
        out = torch.empty(m, HIDDEN, dtype=torch.bfloat16, device=device)
        # the first call sets the group up over the default group (no caller did), as validation's ranks
        kernel.run(**call, out=out)
        torch.cuda.synchronize()
        after = call["shared"].clone()
        outs = _gather(out)
        names = ("routed", "shared", "up_weight", "norm_weight")
        parts = {name: _gather(given if name == "shared" else call[name]) for name in names}
        calls = [{name: parts[name][r] for name in names} | {"eps": EPS} for r in range(TP)]
        expected = KIMI_K3_LATENT_MOE_TAIL.reference.reference(calls, rank)
        out_ok, out_why = (
            KIMI_K3_LATENT_MOE_TAIL.outputs["out"].error_bounds[0][1].compare(out, expected["out"])
        )
        shared_ok, shared_why = (
            KIMI_K3_LATENT_MOE_TAIL.inputs["shared"].error_bounds[0][1].compare(after, expected["shared"])
        )
        own = torch.zeros(HIDDEN, dtype=torch.bool, device=device)
        own[rank * (HIDDEN // TP) : (rank + 1) * (HIDDEN // TP)] = True
        # replays of a captured call, the shared partial put back before each: each the eager bits
        replayed = torch.empty_like(out)
        call["shared"].copy_(given)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel.run(**call, out=replayed)
        differ = 0
        for _ in range(replays):
            call["shared"].copy_(given)
            replayed.zero_()
            graph.replay()
            differ += not (torch.equal(replayed, out) and torch.equal(call["shared"], after))
        torch.cuda.synchronize()
        group = runtime.GROUPS[device]
        results[m] = {
            "the same bits on every rank": all(torch.equal(o, outs[0]) for o in outs),
            f"out within its bound ({out_why})": out_ok,
            f"shared's own columns within their bound ({shared_why})": shared_ok,
            "shared's other columns untouched": torch.equal(after[:, ~own], given[:, ~own]),
            f"{replays} graph replays the eager bits": differ == 0,
            "no wait gave up": not bool(group.health[runtime.FAILED].item()),
        }
    return results


@pytest.mark.gfx950
def test_the_kernel_computes_the_op_on_every_rank(tmp_path):
    """At every row count of both builds: the out the same bits on every rank and within its bound of the reference
    (over every rank's call), this rank's own columns of the shared partial written in place within theirs and the
    other columns untouched, 100 graph replays the eager bits, no wait given up."""
    for rank, results in enumerate(on_gpus(_computes, tmp_path, ROWS, 100)):
        for m, checks in results.items():
            assert all(checks.values()), (rank, m, {k: v for k, v in checks.items() if not v})


class _TpGroup(SimpleNamespace):
    """vLLM's TP group as the route reads it, over torch.distributed's default group."""


def _fake_vllm(rank):
    """vLLM's distributed module as the route reads it: TP 8 over the default group, one pipeline stage."""
    tp = _TpGroup(rank_in_group=rank, world_size=TP, cpu_group=dist.group.WORLD)
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: tp
    distributed.get_tensor_model_parallel_rank = lambda: rank
    distributed.get_tensor_model_parallel_world_size = lambda: TP
    distributed.get_pp_group = lambda: SimpleNamespace(world_size=1)
    sys.modules["vllm"], sys.modules["vllm.distributed"] = types.ModuleType("vllm"), distributed


def _runner(rank, device):
    gen = torch.Generator(device).manual_seed(7 + rank)
    bf16 = torch.bfloat16
    return SimpleNamespace(
        routed_output_transform=SimpleNamespace(
            norm=SimpleNamespace(
                weight=torch.randn(LATENT, generator=gen, device=device).to(bf16), variance_epsilon=EPS
            ),
            up_proj=SimpleNamespace(
                weight=(torch.randn(HIDDEN // TP, LATENT, generator=gen, device=device) * LATENT**-0.5).to(
                    bf16
                )
            ),
        ),
        _up_proj_preshard=True,
        _up_proj_shard_size=HIDDEN // TP,
        _fused_output_is_reduced=False,
    )


@contextlib.contextmanager
def _served_by_the_committed_schema(device, schema, runnable):
    """The op's dispatcher serving the committed schema alone, its workloads unmeasured."""
    dispatcher = DispatcherRegistry.default().dispatcher(KIMI_K3_LATENT_MOE_TAIL)
    table, closed = dispatcher.table, dispatcher.closed_reason
    dispatcher.set_table(
        DispatchTable.from_schemas(
            KIMI_K3_LATENT_MOE_TAIL, device, [(schema, runnable, schema.workloads)], min_speedup=0
        )
    )
    try:
        yield dispatcher
    finally:
        if table is None:
            dispatcher.close(closed)
        else:
            dispatcher.set_table(table)


def _routed(rank, rows):
    device = torch.device("cuda", rank)
    schema, runnable, kernel, runtime = _kernel()
    _fake_vllm(rank)
    api._CHECK = False  # the first-call check runs vLLM's own tail, which needs vLLM
    runner = _runner(rank, device)
    found = {}
    with _served_by_the_committed_schema(device, schema, runnable):
        for m in rows:
            call = _draw(m, rank, device, seed=m)
            out = api._shard_up_proj_tail(runner, call["routed"], call["shared"], None)
            torch.cuda.synchronize()
            served = isinstance(out, torch.Tensor)
            outs = _gather(out if served else torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=device))
            found[m] = {
                "served": served,
                "the same bits on every rank": all(torch.equal(o, outs[0]) for o in outs),
            }
    (key,) = [k for k in api._GROUPS if k[0] == device]
    group = api._GROUPS[key]
    return found | {
        # the route set up the state of the kernel that served it, through that kernel's package
        "the route's group is the kernel's": group is runtime.GROUPS[device]
        and key[1] == kernel.__name__.rpartition(".")[0],
        "the kernel set no group of its own up": runtime.default_group.cache_info().currsize == 0,
        "the route watches its FAILED word": failstop.after_step in plugin.AFTER_STEP,
    }


@pytest.mark.gfx950
def test_the_route_sets_up_the_state_the_kernel_reads(tmp_path):
    """Through the route and AKLR's dispatcher, its table the committed schema: every rank serves every row count, the
    same bits; the group the route set up (through the runtime of the kernel the dispatcher serves with) is the one
    the kernel's launches read, the kernel set none up itself, and the fail-stop watches it."""
    for rank, found in enumerate(on_gpus(_routed, tmp_path, ROWS)):
        for key, value in found.items():
            assert value if not isinstance(value, dict) else all(value.values()), (rank, key, value)


class _AsyncOutput:
    """vLLM's async model runner output of a step: read by its output thread once the step is done."""

    def __init__(self, step):
        self.step = step

    def get_output(self):
        return self.step


@contextlib.contextmanager
def _watchdog(seconds, rank, case):
    """Ends the process when ``case`` is not done within ``seconds``: a hang is what the stall test looks for."""
    timer = threading.Timer(seconds, lambda: (print(f"HANG: rank {rank}, {case}", flush=True), os._exit(2)))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def _late(rank, late_rank, rows, wait, short, long):
    device = torch.device("cuda", rank)
    _, _, kernel, runtime = _kernel()
    group = runtime.setup(dist.group.WORLD, rank, TP, device)
    group.wait_seconds = wait  # a build of its own, compiled at its first call
    failstop.watch(group)
    late = rank == late_rank
    outcomes = {}

    def step(forward, n):
        """One step through the vLLM plugin's hooks: its output as vLLM's output thread reads it, or the error."""
        plugin._execute(forward)
        output = plugin._sample(lambda: _AsyncOutput(n))
        try:
            return output.get_output()
        except RuntimeError as e:
            return str(e)

    call = _draw(rows, rank, device, seed=rows)
    given = call["shared"].clone()

    def launch(out, delay=0.0):
        def forward():
            if late and delay:
                time.sleep(delay)
            call["shared"].copy_(given)
            kernel.run(**call, out=out)

        return forward

    expected = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    launch(expected)()  # compiles; nobody late
    torch.cuda.synchronize()
    dist.barrier()
    out = torch.empty_like(expected)
    with _watchdog(4 * (long + wait), rank, "short"):
        read = step(launch(out, short), 0)
        torch.cuda.synchronize()
    outcomes["late below the timeout: the out exact, the output handed over"] = _all(
        torch.equal(out, expected) and read == 0 and not group.health[runtime.FAILED].item(), device
    )
    dist.barrier()
    with _watchdog(4 * (long + wait), rank, "long"):
        first = step(launch(out, long), 1)
        second = step(launch(out), 2)
        torch.cuda.synchronize()
    failed = [bool(f.item()) for f in _gather(group.health[runtime.FAILED : runtime.FAILED + 1].clone())]
    outcomes["late past the timeout: a waiting rank gave up"] = any(failed)
    outcomes["that step's output and the next's raise on every rank"] = _all(
        isinstance(first, str) and isinstance(second, str) and "gave up an in-kernel wait" in first, device
    )
    return outcomes


@pytest.mark.gfx950
@pytest.mark.parametrize("rows", [10, 40], ids=["kernel_small", "kernel"])
def test_a_late_rank_stops_every_rank_at_that_step(tmp_path, rows):
    """The kernel built with a 2 s in-kernel wait, rank 3 late. 0.5 s late: every rank finishes with the out of a
    launch nobody was late for, and the step's output leaves the worker. 6 s late: every rank finishes (the others'
    waits give up, set their FAILED and the group's word in every rank's buffer: nothing hangs), and that step's
    output, then the next's, raise on every rank, through the vLLM plugin's step hooks and the fail-stop's check of
    the group word the kernel wrote."""
    for rank, outcomes in enumerate(on_gpus(_late, tmp_path, 3, rows, 2.0, 0.5, 6.0)):
        assert all(outcomes.values()), (rank, {k: v for k, v in outcomes.items() if not v})


def _injected(call, real_open, failing):
    """HIP's ``call`` raising (``open``: at the second peer's buffer)."""

    def fail(*args):
        if call == "open":
            failing.opened += 1
            if failing.opened < 2:
                return real_open(*args)
        raise RuntimeError(f"{call}: injected")

    return fail


def _failing_setup(rank, failing_rank):
    device = torch.device("cuda", rank)
    *_, runtime = _kernel()
    # a process's first GPU tensors cost it 2 MiB of the device once, which HIP keeps (measured: only the first
    # setup that gets as far as the group's tensors loses them): made here, before anything is measured
    torch.tensor([0], dtype=torch.int64, device=device)
    torch.zeros(runtime.SCRATCH_BYTES, dtype=torch.uint8, device=device)
    torch.zeros(runtime.GRID, dtype=torch.int32, device=device)
    outcomes = {}
    for call in ("alloc_uncached", "handle", "open"):

        class Failing(runtime.Hip):
            opened = 0

        if rank == failing_rank:
            setattr(Failing, call, staticmethod(_injected(call, runtime.Hip.open, Failing)))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(device)[0]
        group = runtime.setup(dist.group.WORLD, rank, TP, device, hip=Failing)
        none = group is None
        del group
        # a peer that still maps this rank's buffer keeps its memory: measured once every rank has released
        _barrier()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        leaked = free - torch.cuda.mem_get_info(device)[0]
        outcomes[f"{call} failing on rank {failing_rank}: None on every rank, nothing left"] = _all(
            none and abs(leaked) < (1 << 20), device
        )
    outcomes["a clean setup after them"] = _all(
        runtime.setup(dist.group.WORLD, rank, TP, device) is not None, device
    )
    return outcomes


@pytest.mark.gfx950
def test_a_setup_failing_on_one_rank_fails_on_every_rank(tmp_path):
    """HIP's allocation, IPC export or mapping of the second peer failing on rank 2 alone: every rank's setup returns
    None at the same point, nothing is left allocated or mapped, and a clean setup after them succeeds."""
    for rank, outcomes in enumerate(on_gpus(_failing_setup, tmp_path, 2)):
        assert all(outcomes.values()), (rank, {k: v for k, v in outcomes.items() if not v})


def _one_rank_apart(rank, failing_rank, rows):
    device = torch.device("cuda", rank)
    schema, runnable, *_ = _kernel()
    _fake_vllm(rank)
    api._CHECK = False
    runner = _runner(rank, device)
    outcomes = {}
    with _served_by_the_committed_schema(device, schema, runnable) as dispatcher:
        table = dispatcher.table
        for m in rows:
            gen = torch.Generator(device).manual_seed(1000 + rank)
            fused = torch.randn(m, LATENT, generator=gen, device=device).to(torch.bfloat16)
            shared = torch.randn(m, HIDDEN, generator=gen, device=device).to(torch.bfloat16)
            shared.view(-1)[: m * LATENT] = fused.view(-1)  # the aliased case's views hold these very values
            reference = None
            for case in ("aligned", "off alignment", "aliased", "no schema"):
                api._VERDICTS.clear()  # the ranks agree on each case afresh
                dispatcher.set_table(table)
                routed, partial = fused, shared.clone()
                if rank == failing_rank:
                    if case == "off alignment":
                        routed = torch.empty(fused.numel() + 1, dtype=fused.dtype, device=device)[1:].view(
                            fused.shape
                        )
                        routed.copy_(fused)
                    elif case == "aliased":
                        buffer = shared.clone().view(-1)
                        routed, partial = buffer[: m * LATENT].view(m, LATENT), buffer.view(m, HIDDEN)
                    elif case == "no schema":
                        dispatcher.close(FallbackReason.NO_SCHEMA)
                with _watchdog(120, rank, f"{case}, {m} rows"):
                    result = api._shard_up_proj_tail(runner, routed, partial, None)
                    torch.cuda.synchronize()
                served = isinstance(result, torch.Tensor)
                if case == "no schema":
                    outcomes[f"{case}, {m} rows"] = _all(result is FallbackReason.NO_SCHEMA, device)
                    continue
                outs = _gather(
                    result if served else torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=device)
                )
                if case == "aligned":
                    reference = result
                outcomes[f"{case}, {m} rows"] = _all(
                    served and all(torch.equal(o, outs[0]) for o in outs) and torch.equal(result, reference),
                    device,
                )
    return outcomes


@pytest.mark.gfx950
def test_what_one_rank_alone_sees_never_splits_the_group(tmp_path):
    """Through the route and the committed schema: rank 2's routed partial off the dispatcher's alignment or in the
    first bytes of its shared partial, every rank still serves the aligned run's bits; rank 2's dispatcher without
    the schema, every rank goes back to vLLM with no_schema, none launching."""
    for rank, outcomes in enumerate(on_gpus(_one_rank_apart, tmp_path, 2, (10, 40))):
        assert all(outcomes.values()), (rank, {k: v for k, v in outcomes.items() if not v})
