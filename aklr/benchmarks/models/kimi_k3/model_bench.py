###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Portions of this file are derived from SGLang
# (https://github.com/sgl-project/sglang),
# licensed under the Apache License, Version 2.0 (see LICENSE-APACHE).
# Original copyright: Copyright 2023-2024 SGLang Team.
# Modifications to the imported portions have been made by the AKLR team.
#
# Those portions remain under Apache-2.0, not the MIT license that covers the
# rest of AKLR (see LICENSE).
###############################################################################

import argparse
import contextlib
import dataclasses
import inspect
import multiprocessing
import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch
from sglang_bench import (
    GRAPH_REPS,
    A2aImpl,
    PrefillStep,
    _balanced_forward,
    _partitions,
    agreed,
    local_conc,
    moe_parallel,
    publish_once,
    require_aiter,
    table,
)

DEFAULT_MODEL = "moonshotai/Kimi-K3"

MODEL_TAG = "kimi_k3_mxfp4"


def make_parser() -> argparse.ArgumentParser:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    # ServerArgs marks --model-path required; default it to the hub id instead.
    for action in parser._actions:
        if action.dest == "model_path":
            action.required = False
            action.default = DEFAULT_MODEL
    parser.add_argument("--ISL", type=int, default=4096, help="input length per request")
    parser.add_argument("--reps", type=int, default=50, help="graph replays per op")
    parser.add_argument("--repeat", type=int, default=1, help="timing passes per op, fastest one wins")
    parser.add_argument(
        "--full-step",
        type=int,
        default=0,
        metavar="N",
        help="time N whole steps under sglang's own cuda graphs and exit, "
        "instead of profiling per op; the only server-comparable step time. "
        "Read by build_server_args, so it lives here rather than with the "
        "decode profile that acts on it.",
    )
    parser.add_argument("--out", help="markdown report path (default: auto-named)")
    return parser


DUMMY_WARNING = (
    "--load-format dummy. Operators whose cost depends on values are mistimed. "
    "On K3 the router is the one that matters: dummy weights make every token "
    "pick the same 16 of 896 experts, so `mlp.experts` reads a fraction of the "
    "expert weights the real model streams. balance_moe_routing() overrides the "
    "top-k to compensate, which restores the byte count but not the "
    "distribution. Treat the MoE rows as an estimate."
)


def warn_if_dummy_weights(load_format: str) -> None:
    if load_format != "dummy":
        return
    print(f"[BENCH] WARNING: {DUMMY_WARNING}", flush=True)


def dummy_banner(server_args) -> str:
    if server_args.load_format != "dummy":
        return ""
    return f"\n> **WARNING: {DUMMY_WARNING}**\n"


def parallel_tags(model_runner, layers: "Layers") -> list[str]:
    ps, moe = model_runner.ps, moe_parallel(model_runner)
    tags = [f"tp{ps.tp_size}"]
    if ps.attn_dp_size > 1:
        tags.append(f"dp{ps.attn_dp_size}")
    if dcp_size(model_runner) > 1:
        tags.append(f"dcp{dcp_size(model_runner)}")
    if moe.ep_size > 1:
        tags.append(f"ep{moe.ep_size}")
    if moe.a2a:
        tags += [moe.a2a, cfg(model_runner.server_args).deepep_mode]
        impl = a2a_impl(model_runner, layers)
        if impl is not None:
            tags.append(f"cap{impl.cap}")
    return tags


def dcp_size(model_runner) -> int:
    return max(1, getattr(model_runner.ps, "attn_dcp_size", 1) or 1)


def parallel_line(model_runner) -> str:
    ps, moe = model_runner.ps, moe_parallel(model_runner)
    line = f"TP {ps.tp_size}, attn-DP {ps.attn_dp_size}"
    if dcp_size(model_runner) > 1:
        line += f", DCP {dcp_size(model_runner)} ({cfg(model_runner.server_args).dcp_comm_backend})"
    line += f", MoE-EP {moe.ep_size}"
    if moe.ep_size > 1 and moe.tp_size > 1:
        line += f" x TP {moe.tp_size}"
    if moe.a2a:
        line += f", MoE tokens {moe.a2a} all-to-all"
    return line


def a2a_impl(model_runner, layers: "Layers") -> A2aImpl | None:
    if moe_parallel(model_runner).a2a is None:
        return None
    experts = next(
        (layer.mlp.experts for layer in layers.layers if hasattr(layer.mlp, "experts")),
        None,
    )
    try:
        dispatcher = experts.dispatcher._inners[0]
        impl = (
            dispatcher._normal_dispatcher
            if cfg(model_runner.server_args).deepep_mode == "normal"
            else dispatcher._low_latency_dispatcher
        )
        return A2aImpl(
            cap=int(impl.num_max_dispatch_tokens_per_rank),
            dispatch_dtype=impl.dispatch_dtype.name,
            combine_dtype=impl.combine_dtype.name,
        )
    except (AttributeError, IndexError):
        return None


def moe_line(model_runner, layers: "Layers") -> str | None:
    moe = moe_parallel(model_runner)
    if moe.a2a is None:
        return None
    impl = a2a_impl(model_runner, layers)
    # The mode picks the kernel; the wrong one made decode 5x slower.
    mode = cfg(model_runner.server_args).deepep_mode
    if impl is None:
        return (
            f"{moe.a2a} all-to-all, mode {mode}; dispatch cap and dtypes not found"
            " on the dispatcher, see the `[MORI init]` line in the log"
        )
    knob = " (SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK)" if moe.a2a == "mori" else ""
    return (
        f"{moe.a2a} all-to-all, mode {mode}, dispatch {impl.dispatch_dtype}, combine"
        f" {impl.combine_dtype}, dispatch cap {impl.cap:,} tokens/rank{knob}:"
        f" `mlp.experts` is handed up to {impl.cap * moe.ep_size:,} rows per call."
        " An a2a backend also takes the MoE off its fused front: the gate, the"
        " latent down-proj and the shared experts run as their own modules and"
        " so get their own rows, which they do not have under plain TP"
    )


def cfg(server_args):
    """The ServerArgs values sglang's resolution settled on; the raw fields can say otherwise."""
    try:
        from sglang.srt.arg_groups.overrides import resolving_view
    except ImportError:
        return server_args

    return resolving_view(server_args)


def build_server_args(args: argparse.Namespace, cuda_graph: bool = False):
    from sglang.srt.server_args import ServerArgs

    require_aiter()

    warn_if_dummy_weights(args.load_format)
    if args.speculative_algorithm is not None:
        raise SystemExit(
            "this bench does not model speculative decoding on K3. Drop"
            " --speculative-algorithm; the rows below would cover the target"
            " stack only while the draft's share of the step went unmeasured."
        )
    # Per-op profiling needs eager: with sglang's cuda graph on, record_inputs() pre-hooks never fire.
    args.disable_cuda_graph = not (args.full_step or cuda_graph)
    # `auto` is bf16, doubling attention bytes per token; production runs fp8.
    if args.kv_cache_dtype == "auto":
        args.kv_cache_dtype = "fp8_e4m3"
    # The K3 checkpoint ships its config and processor as code; the tokenizer refuses to load without this.
    args.trust_remote_code = True
    # Unset, sglang sizes the request and KDA state pools for a huge serving batch.
    if args.max_running_requests is None:
        args.max_running_requests = pool_rows(args)

    server_args = ServerArgs.from_cli_args(args)
    # Resolve as one_batch.main does: unresolved, mori gets deepep_mode=auto and picks the wrong kernel.
    if hasattr(server_args, "resolve_once"):
        server_args.resolve_once()
    warn_if_mem_fraction_scaled(args, server_args)
    if cfg(server_args).enable_dp_attention and args.CONC % server_args.dp_size:
        raise ValueError(
            f"--CONC {args.CONC} is the global batch and must divide evenly "
            f"across --dp-size {server_args.dp_size}"
        )
    return server_args


def warn_if_mem_fraction_scaled(args, server_args) -> None:
    asked = args.mem_fraction_static
    got = cfg(server_args).mem_fraction_static
    if asked is None or got is None or abs(asked - got) < 1e-6:
        return
    print(
        f"[k3 bench] --mem-fraction-static {asked} resolved to {got:.4f}. sglang"
        " scales it under the aiter attention backend at long context; set"
        " SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 to keep the value given.",
        flush=True,
    )


def attn_tp_size(args) -> int:
    if args.enable_dp_attention:
        return max(1, args.tp_size // max(1, args.dp_size))
    return max(1, args.tp_size)


def pool_rows(args) -> int:
    dp = max(1, args.dp_size) if args.enable_dp_attention else 1
    tp = attn_tp_size(args)
    local = max(1, args.CONC // dp)
    return (-(-local // tp) * tp + 1) * dp


def claim_out_dir(out: str | None) -> str | None:
    if not out:
        return out
    out = str(Path(out).expanduser())
    parent = Path(out).parent
    parent.mkdir(parents=True, exist_ok=True)
    if not os.access(parent, os.W_OK):
        raise ValueError(f"--out directory is not writable: {parent}")
    return out


def run(work_func: Callable, server_args, args) -> None:
    """Call `work_func` on each rank as sglang's one_batch.main does, one process per rank when TP > 1."""
    if hasattr(args, "out"):
        args.out = claim_out_dir(args.out)

    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs
    from sglang.srt.utils import kill_process_tree, maybe_reindex_device_id

    # The default of 32 per process times tp_size is a few hundred compile workers.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, args, 0, 0)
        return

    nranks_per_node = server_args.tp_size // server_args.nnodes
    lo = server_args.node_rank * nranks_per_node
    procs = []
    try:
        for tp_rank in range(lo, lo + nranks_per_node):
            with maybe_reindex_device_id(tp_rank - lo) as gpu_id:
                p = multiprocessing.Process(
                    target=work_func,
                    args=(server_args, port_args, args, gpu_id, tp_rank),
                )
                p.start()
                procs.append(p)
        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode]
        if failed:
            raise SystemExit(f"{len(failed)} of {len(procs)} ranks exited non-zero: {failed}")
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def setup(server_args, port_args, gpu_id: int, tp_rank: int, tag: str = "BENCH"):
    """Load the model on this rank; returns (model_runner, rank_print), rank_print a no-op off rank 0."""
    from sglang.benchmark.one_batch import load_model
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import configure_logger

    # Dispatch on the signature: sglang trees declare these differently and none agree on a version.
    if not inspect.signature(initialize_moe_config).parameters:
        publish_once(server_args, gpu_id, tp_rank)
    for fn in (
        initialize_moe_config,
        initialize_fp8_gemm_config,
        initialize_fp4_gemm_config,
    ):
        fn(server_args) if inspect.signature(fn).parameters else fn()
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    # As the scheduler process does: AKLR's dispatch attaches through sglang.srt.plugins hooks.
    load_plugins()

    model_runner = load_model(server_args, port_args, gpu_id, tp_rank)[0].torch_runner

    if tp_rank != 0:
        return model_runner, lambda *a, **k: None

    def rank_print(msg=""):
        print("\n".join(f"[{tag}] {ln}" for ln in str(msg).split("\n")), flush=True)

    return model_runner, rank_print


def make_reqs(args, model_runner, factor: int = 1, decode_tokens: int = 0):
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    conc = local_conc(args, model_runner)
    page, have = kv_alloc_capacity(model_runner)
    # `factor` is the peak: prefill holds its context and every chunk allocates on top of it.
    want = -(-(args.ISL + decode_tokens) // page) * page * conc * factor
    if want > have:
        # Fail here: sglang's own out-of-memory path raises while formatting its message.
        raise ValueError(
            f"--ISL {args.ISL:,} x --CONC {args.CONC}"
            + (
                f" x {factor} (prefill holds its context while the chunks allocate over it)"
                if factor > 1
                else ""
            )
            + (f" plus {decode_tokens} decode token" if decode_tokens else "")
            + f" needs {want:,} KV tokens "
            f"per rank, and the MLA KV pool holds {have:,}"
            + (
                f" (its rows x --dcp-size {dcp_size(model_runner)}: each rank"
                f" stores one token in {dcp_size(model_runner)})"
                if dcp_size(model_runner) > 1
                else ""
            )
            + ". Lower one of them, raise --mem-fraction-static, or give the"
            " KV side more of the split with a lower --mamba-full-memory-ratio."
        )

    # The KDA state pool is a second ceiling, binding at high concurrency: a state per request, any length.
    cap = state_request_cap(model_runner)
    if cap is not None and conc > cap:
        raise ValueError(
            f"--CONC {args.CONC} is {conc} requests on this rank and the KDA "
            f"state pool serves {cap} at once. Lower it, or give the state "
            f"side more of the split with a higher --mamba-full-memory-ratio."
        )

    return prepare_synthetic_inputs_for_latency_test(conc, args.ISL)


def kv_alloc_capacity(model_runner) -> tuple[int, int]:
    """(page, tokens) the KV allocator hands out; under DCP, dcp_size times the rows a rank stores."""
    alloc = model_runner.token_to_kv_pool_allocator
    page = getattr(alloc, "page_size", None) or cfg(model_runner.server_args).page_size
    size = getattr(alloc, "size", None)
    if size is None:
        size = model_runner.max_total_num_tokens * dcp_size(model_runner)
    return page, size


def runner_batch(saved: dict, fallback):
    """The ForwardBatch the model actually ran on in the recording forward, else `fallback`."""
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    for entry in saved.values():
        for value in (*entry.args, *entry.kwargs.values()):
            if isinstance(value, ForwardBatch):
                return value
    return fallback


def make_extend_batch(reqs, model_runner, reset: bool = True):
    from sglang.benchmark.one_batch import (
        TreeCacheNamespace,
        _maybe_prepare_mlp_sync_batch,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if reset:
        model_runner.token_to_kv_pool_allocator.clear()
        model_runner.req_to_token_pool.clear()

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=TreeCacheNamespace(
            # The allocator's page, which DCP widens by dcp_size, so eviction and allocation agree.
            page_size=kv_alloc_capacity(model_runner)[0],
            device=model_runner.device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        ),
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        batch.prefill_input_ids_cpu = None
    return batch


def randomize_kv(model_runner) -> None:
    """Fill the KV and KDA state pools with random values, standing in for a prefill forward."""
    _randomize_pool(kv_pool(model_runner))
    randomize_state(model_runner)


def randomize_state(model_runner) -> None:
    for buf in state_buffers(model_runner):
        _fill(buf)


def _fill(buf) -> None:
    if buf.element_size() == 1:
        # Bytes below 0x7C are finite and non-negative in every fp8 format.
        buf.view(torch.uint8).random_(0, 0x7C)
    else:
        buf.uniform_(-1, 1)


def kv_pool(model_runner):
    pool = model_runner.token_to_kv_pool
    return getattr(pool, "full_kv_pool", pool)


def _randomize_pool(pool) -> None:
    for buf in pool.kv_buffer:
        _fill(buf)


def mamba_pool(model_runner):
    return getattr(model_runner.req_to_token_pool, "mamba_pool", None)


def mamba_cache(model_runner):
    return getattr(mamba_pool(model_runner), "mamba_cache", None)


def state_buffers(model_runner) -> list:
    cache = mamba_cache(model_runner)
    if cache is None:
        return []
    names = [f.name for f in dataclasses.fields(cache)] if dataclasses.is_dataclass(cache) else []
    out, seen = [], set()
    for field in names:
        for buf in _tensors(getattr(cache, field, None)):
            key = (buf.data_ptr(), tuple(buf.shape), tuple(buf.stride()))
            if key in seen or not buf.numel():
                continue
            seen.add(key)
            out.append(buf)
    return out


def _tensors(value) -> list:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _tensors(v)]
    return []


def state_bytes(model_runner) -> int:
    pool = mamba_pool(model_runner)
    return int(pool.get_kv_size_bytes()) if pool is not None else 0


def state_slot_of(req):
    held = getattr(getattr(req, "kv", None), "mamba_pool_idx", None)
    return held if held is not None else getattr(req, "mamba_pool_idx", None)


def state_slots(model_runner) -> int | None:
    pool = mamba_pool(model_runner)
    if pool is None:
        return None
    size = getattr(pool, "size", None)
    if not isinstance(size, int):
        raise TypeError(
            "the KDA state pool does not report a slot count under `.size`"
            f" (found {type(size).__name__}); without it the state row cannot"
            " be priced"
        )
    return size


def state_request_cap(model_runner) -> int | None:
    if mamba_pool(model_runner) is None:
        return None
    size = getattr(model_runner.req_to_token_pool, "size", None)
    if not isinstance(size, int):
        raise TypeError(
            "the request pool does not report a request count under `.size`"
            f" (found {type(size).__name__}); without it a batch larger than the"
            " KDA state pool is admitted and silently comes up short of requests"
        )
    return size


def chunk_tokens(args, model_runner) -> int:
    size = cfg(model_runner.server_args).chunked_prefill_size
    return size if size > 0 else args.ISL


def chunk_flag(args, model_runner) -> int:
    """The --chunked-prefill-size flag value: `chunk_tokens` before sglang divides it across DP ranks."""
    if args.chunked_prefill_size:
        return args.chunked_prefill_size
    return chunk_tokens(args, model_runner) * model_runner.ps.attn_dp_size


def plan_prefill(args, model_runner) -> list[PrefillStep]:
    page = kv_alloc_capacity(model_runner)[0]
    cached = int(args.ISL * args.prefix_cache_hit) // page * page
    budget = chunk_tokens(args, model_runner)
    steps, prefix = [], cached
    while prefix < args.ISL:
        extend = min(budget, args.ISL - prefix)
        steps.append(PrefillStep(prefix, extend))
        prefix += extend
    if not steps:
        raise ValueError(
            f"--prefix-cache-hit {args.prefix_cache_hit} leaves nothing to prefill at --ISL {args.ISL}"
        )
    return steps


def make_prefill_context(args, model_runner) -> list[torch.Tensor]:
    """KV slots holding each request's context; each prefill step takes its prefix from these."""
    batch = make_extend_batch(make_reqs(args, model_runner, factor=2), model_runner)
    randomize_kv(model_runner)
    table = model_runner.req_to_token_pool.req_to_token
    # Clone: a table row is a view and every extend rewrites the table.
    context = [table[idx, : args.ISL].clone().to(torch.int64) for idx in batch.req_pool_indices_cpu.tolist()]
    free_req_rows(batch, model_runner)
    return context


def make_prefill_batch(model_runner, context, step: PrefillStep):
    from sglang.benchmark.one_batch import prepare_synthetic_inputs_for_latency_test

    reqs = prepare_synthetic_inputs_for_latency_test(len(context), step.prefix + step.extend)
    for req, slots in zip(reqs, context):
        req.prefix_indices = slots[: step.prefix]
        req.set_extend_range(step.prefix, step.prefix + step.extend)
    return make_extend_batch(reqs, model_runner, reset=False)


def free_req_rows(batch, model_runner) -> None:
    """Give a batch's page-table rows and KDA state slots back; its KV slots stay allocated."""
    pool = model_runner.req_to_token_pool
    free_mamba = getattr(pool, "free_mamba_cache", None)
    if free_mamba is not None:
        for req in batch.reqs:
            if state_slot_of(req) is not None:
                free_mamba(req)
    if hasattr(pool, "free_rows"):
        pool.free_rows(batch.req_pool_indices_cpu.tolist())
        return
    for req in batch.reqs:
        if getattr(req, "req_pool_idx", None) is not None:
            pool.free(req)


MLP_FLAGS = ("fuse_mlp_allreduce", "mlp_reduce_scatter")


def mlp_flags(fwd) -> dict[str, bool]:
    return {f: getattr(fwd, f) for f in MLP_FLAGS if hasattr(fwd, f)}


# Attributes stashed on a module between recording and calling it, which it reads to pick a kernel.
STASH_ATTRS = ("_k3_onorm_gate", "_k3_onorm_consumed", "_k3_deferred_f_b")


def module_stash(mod) -> dict:
    return {a: getattr(mod, a) for a in STASH_ATTRS if hasattr(mod, a)}


def attn_res_banks(args, kwargs) -> list[tuple]:
    return [
        (a, a.num_valid_blocks)
        for a in (*args, *kwargs.values())
        if hasattr(a, "num_valid_blocks") and hasattr(a, "block_residual")
    ]


class Recorded(NamedTuple):
    """One op's recorded call: its arguments plus the ambient forward state it reads."""

    args: tuple
    kwargs: dict
    attn_inputs: object
    mlp_flags: dict[str, bool]
    stash: dict
    banks: list[tuple]


def residual_owner(name: str) -> str:
    kind, _, op = name.rpartition("|")
    kind = kind.partition("#")[0]
    if not kind:
        return ""
    if op == "layer":
        return ""
    parent = op.rpartition(".")[0] or "layer"
    if parent not in RESIDUALS:
        return ""
    return f"{kind}|{parent}.<rest>"


def skipped_note(skipped: dict, gaps: dict | None = None) -> str:
    """Ops that fired but could not be replayed on their own, and the row their cost now sits in."""
    if not skipped:
        return ""
    rows = []
    for name, err in sorted(skipped.items()):
        kind, _, op = name.rpartition("|")
        depth_keyed = "#" in kind
        kind = kind.partition("#")[0]
        owner = residual_owner(name)
        if depth_keyed:
            where = "nowhere -- the kind is summed over fewer layers"
        elif op in RESIDUALS:
            where = "`layer.<rest>`; its children leave the roll-up too"
        elif owner:
            where = f"`{owner.split('|', 1)[1]}`"
        else:
            where = "nowhere -- the roll-up is short by it"
        rows.append((kind or "--", f"`{op}`", where, err))
    note = table(rows, ("layer kind", "op", "its ms now sits in", "why it could not be replayed")) + (
        "\n\nThese ops ran in the recording forward, so their cost is real and is"
        " in the parent row; what is missing is the row that would separate it"
        " out. Read the named `<rest>` row as an upper bound, not a measurement."
    )
    if gap := gap_note(gaps or {}):
        note += f"\n\n{gap}"
    return note


FUSION_NOTES = (
    (
        ("input_layernorm", "post_attention_layernorm"),
        (
            "the attention-residual aggregation is one kernel that reads these"
            " two as weights rather than calling them. The exception is layer"
            " 0's pre-attention point, which has no banked rows to mix yet and"
            " applies its norm directly"
        ),
    ),
    (
        (
            "self_attention_res_norm",
            "self_attention_res_proj",
            "mlp_res_norm",
            "mlp_res_proj",
        ),
        ("the same aggregation kernel reads the score projection and its norm as one precomputed product"),
    ),
    (
        ("attn.b_proj", "attn.f_a_proj", "attn.f_b_proj"),
        (
            "where the attention TP group is the whole TP group, weight loading"
            " merges these into one tensor and a tiny GEMM replaces the modules"
        ),
    ),
    (
        ("mlp.gate", "mlp.routed_expert_down_proj", "mlp.shared_experts"),
        (
            "under plain TP the MoE front is one merged GEMM over the router"
            " gate, the latent down-projection and the shared experts' gate_up"
        ),
    ),
    (
        ("output_attn_res_proj", "output_attn_res_norm", "norm"),
        (
            "the final aggregation reads these as weights; it is timed as"
            " `attn_res_out`, which is their parent"
        ),
    ),
)


def coverage_note(ops: dict, saved: dict) -> str:
    """The hooked ops that never fired in the recording forward, so no table has a row for them."""
    missing = sorted(n for n in ops if n not in saved)
    if not missing:
        return ""
    by_op: dict[str, list[str]] = {}
    for name in missing:
        kind, _, op = name.rpartition("|")
        by_op.setdefault(op, []).append(kind.partition("#")[0] or "--")
    lines = []
    for group, why in FUSION_NOTES:
        hit = [op for op in group if op in by_op]
        if hit:
            lines.append(f"- {', '.join(f'`{op}`' for op in hit)} -- {why}.")
    named = {op for group, _ in FUSION_NOTES for op in group}
    if "attn.attn_mqa" in by_op and any(n.endswith("|attn.attn_mqa_for_dcp_decode") for n in saved):
        lines.append(
            "- `attn.attn_mqa` -- under DCP a decode step runs"
            " `attn_mqa_for_dcp_decode` in its place, over this rank's share of"
            " the context."
        )
        named.add("attn.attn_mqa")
    rest = [op for op in by_op if op not in named]
    if rest:
        lines.append(
            "- "
            + ", ".join(f"`{op}`" for op in sorted(rest))
            + " -- this layer kind does not have the op, or a fused kernel does"
            " its work without going through the module."
        )
    return (
        "Hooked but never called during the recording forward, so absent from"
        " every table and folded into whichever `<rest>` row contains them.\n\n" + "\n".join(lines)
    )


@contextlib.contextmanager
def record_inputs(ops: dict, into: dict):
    from sglang.srt.runtime_context import get_forward

    handles = []
    for name, mod in ops.items():
        if not isinstance(mod, torch.nn.Module):
            continue

        def hook(_m, args, kwargs, _n=name):
            if _n not in into:
                fwd = get_forward()
                into[_n] = Recorded(
                    args,
                    kwargs,
                    getattr(fwd, "attn_inputs", None),
                    mlp_flags(fwd),
                    module_stash(_m),
                    attn_res_banks(args, kwargs),
                )

        handles.append(mod.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield into
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def record_attn_res_out(model, ops: dict, into: dict):
    """Record the model-level attention-residual aggregation as op `attn_res_out`."""
    try:
        from sglang.srt.layers.attn_residual import AttnResidual
    except ImportError:
        yield into
        return

    out_proj = getattr(backbone(model), "output_attn_res_proj", None)
    if out_proj is None:
        yield into
        return
    name = "attn_res_out"
    patched = {}

    def patch(method_name):
        original = getattr(AttnResidual, method_name, None)
        if original is None:
            return

        def wrapper(self, *args, **kwargs):
            score_proj = kwargs.get("score_proj", args[2] if len(args) > 2 else None)
            mine = score_proj is out_proj and name not in into
            # Capture before the call: the replay must start from the count this call started from.
            depth = self.num_valid_blocks if mine else None
            out = original(self, *args, **kwargs)
            if mine and out is not None:
                # Bind the real method, not this wrapper, so the replay does not re-enter the recorder.
                ops[name] = original.__get__(self)
                into[name] = Recorded(args, kwargs, None, {}, {}, [(self, depth)])
            return out

        patched[method_name] = original
        setattr(AttnResidual, method_name, wrapper)

    for method_name in ("forward", "forward_sp_all_gather"):
        patch(method_name)
    try:
        yield into
    finally:
        for method_name, original in patched.items():
            setattr(AttnResidual, method_name, original)


def make_call(mod, entry) -> Callable:
    from sglang.srt.layers.communicator import ATTN_TP_CONTEXT, AttentionInputs
    from sglang.srt.runtime_context import get_forward

    src = entry.attn_inputs

    def call():
        if src is not None:
            ATTN_TP_CONTEXT.set_attn_inputs(
                AttentionInputs(src.hidden_states_local, src.forward_batch, src.qkv_latent_func)
            )
        for attr, value in entry.stash.items():
            setattr(mod, attr, value)
        for bank, depth in entry.banks:
            bank.num_valid_blocks = depth
        with get_forward().scoped(**entry.mlp_flags):
            return mod(*entry.args, **entry.kwargs)

    return call


MALL_BYTES = 256 << 20  # MI355X's last-level cache
ROTATION_BYTES = 2 * MALL_BYTES  # twice the cache, so a slot is evicted before its next read
MAX_ROTATION_SLOTS = 256


@contextlib.contextmanager
def rotated_calls(mod, entry):
    """Yield (fns, inner) for bench(), rotating a weight that fits the MALL so no call reads it cache-hot."""
    call = make_call(mod, entry)
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        yield [call], GRAPH_REPS
        return

    nbytes = w.numel() * w.element_size()
    slots = min(ROTATION_BYTES // max(nbytes, 1), MAX_ROTATION_SLOTS)
    if nbytes >= MALL_BYTES or slots < 2 or slots * nbytes <= MALL_BYTES:
        yield [call], GRAPH_REPS
        return

    try:
        ring = torch.empty((slots, *w.shape), dtype=w.dtype, device=w.device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        ring = None
    # Agreed across ranks: a rank falling back alone would issue mismatched collectives and wedge the world.
    if not agreed(ring is not None):
        if ring is not None:
            del ring
            torch.cuda.empty_cache()
        yield [call], GRAPH_REPS
        return
    ring.copy_(w)

    def rotate_to(slot):
        def fn():
            mod.weight = slot
            return call()

        return fn

    params = [torch.nn.Parameter(ring[i], requires_grad=False) for i in range(slots)]
    try:
        # The assignment runs during capture, so each launch records its own slot's pointer.
        yield [rotate_to(p) for p in params], max(GRAPH_REPS, slots)
    finally:
        mod.weight = w
        del params, ring
        torch.cuda.empty_cache()


def balance_moe_routing(layers, load_format: str) -> None:
    if load_format != "dummy":
        return
    balance_modules(layers.layers)


def balance_modules(modules) -> None:
    for layer in modules:
        experts = getattr(layer.mlp, "experts", None)
        if experts is not None:
            experts.forward = _balanced_forward(experts)


def language_model(model):
    return getattr(model, "language_model", None) or model


def backbone(model):
    return language_model(model).model


class Layers:
    """Decoder layers by kind and residual depth, so the roll-up multiplies each row by the right count."""

    def __init__(self, model_runner):
        self.layers = backbone(model_runner.model).layers
        self.n_all = len(self.layers)

        self.kinds: dict[str, list[int]] = {}
        self.depths: dict[str, dict[tuple[int, bool], list[int]]] = {}
        for i, layer in enumerate(self.layers):
            kind = "moe" if hasattr(layer.mlp, "experts") else "dense"
            kind += "+kda" if is_kda(layer) else "+mla"
            self.kinds.setdefault(kind, []).append(i)
            self.depths.setdefault(kind, {}).setdefault(depth_of(layer), []).append(i)

    @property
    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.kinds.items()}

    @property
    def depth_counts(self) -> dict[tuple[str, tuple[int, bool]], int]:
        return {(kind, depth): len(ids) for kind, per in self.depths.items() for depth, ids in per.items()}

    def describe(self) -> str:
        kinds = ", ".join(f"{k} {len(v)}" for k, v in self.kinds.items())
        nvb = sorted({d[0] for per in self.depths.values() for d in per})
        writes = sum(len(ids) for per in self.depths.values() for d, ids in per.items() if d[1])
        return (
            f"{self.n_all} layers -- {kinds}; attention-residual depth"
            f" {nvb[0]}..{nvb[-1]}, {writes} of them banking a new row"
        )

    def experts_per_rank(self) -> int:
        for layer in self.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None:
                return int(experts.num_local_experts - getattr(experts, "num_fused_shared_experts", 0))
        return 0


def depth_of(layer) -> tuple[int, bool]:
    """The layer's residual depth: (banked rows it mixes, whether it banks one); cost grows with depth."""
    return (
        int(getattr(layer, "prev_valid_blocks", 0) or 0),
        bool(getattr(layer, "is_block_write_layer", False)),
    )


def depth_tag(depth: tuple[int, bool]) -> str:
    nvb, write = depth
    return f"{nvb}w" if write else str(nvb)


def is_kda(layer) -> bool:
    return hasattr(layer.self_attn, "o_norm")


TREE = [
    ("layer", 0),
    ("input_layernorm", 1),
    ("attn", 1),
    ("attn.fused_qkv_a_proj_with_mqa", 2),
    ("attn.q_b_proj", 2),
    ("attn.kv_b_proj", 2),
    ("attn.fused_qkvg_proj", 2),
    ("attn.qkv_proj", 2),
    ("attn.b_proj", 2),
    ("attn.f_a_proj", 2),
    ("attn.f_b_proj", 2),
    ("attn.g_proj", 2),
    ("attn.attn", 2),
    ("attn.attn_mqa", 2),
    ("attn.attn_mqa_for_dcp_decode", 2),
    ("attn.attn_mha", 2),
    ("attn.o_norm", 2),
    ("attn.o_proj", 2),
    ("attn.<rest>", 2),
    ("post_attention_layernorm", 1),
    ("mlp", 1),
    ("mlp.gate", 2),
    ("mlp.topk", 2),
    ("mlp.routed_expert_down_proj", 2),
    ("mlp.experts", 2),
    ("mlp.routed_expert_norm", 2),
    ("mlp.routed_expert_up_proj", 2),
    ("mlp.shared_experts", 2),
    ("mlp.gate_up_proj", 2),
    ("mlp.act_fn", 2),
    ("mlp.down_proj", 2),
    ("mlp.<rest>", 2),
    ("self_attention_res_norm", 1),
    ("self_attention_res_proj", 1),
    ("mlp_res_norm", 1),
    ("mlp_res_proj", 1),
    ("layer.<rest>", 1),
]

# parent -> the children subtracted from it to get its `<rest>` row.
RESIDUALS = {
    parent: [n for n, d in TREE if d == 2 and n.startswith(f"{parent}.") and not n.endswith("<rest>")]
    for parent in ("attn", "mlp")
}
RESIDUALS["layer"] = [n for n, d in TREE if d == 1 and not n.endswith("<rest>")]

# Rows that partition a layer without overlap; their sum is the roll-up exactly.
LEAVES = [n for n, d in TREE if d == 2] + [n for n, d in TREE if d == 1 and n not in RESIDUALS]

_ATTR = {"attn": "self_attn"}


def layer_ops(layer) -> dict[str, torch.nn.Module]:
    owners = {"": layer, "attn": layer.self_attn, "mlp": layer.mlp}
    ops = {"layer": layer}
    for name, _ in TREE:
        if name in ops or name.endswith("<rest>"):
            continue
        parent, _, attr = name.rpartition(".")
        mod = getattr(owners[parent], _ATTR.get(name, attr), None)
        if isinstance(mod, torch.nn.Module):
            ops[name] = mod
    return ops


GLOBAL_OPS = (
    "embed_tokens",
    "attn_res_out",
    "output_attn_res_proj",
    "output_attn_res_norm",
    "norm",
    "logits",
)

# The ops `attn_res_out` subsumes; on fused paths it reads their weights rather than calling them.
ATTN_RES_OUT_CHILDREN = ("output_attn_res_proj", "output_attn_res_norm", "norm")

# Ops whose cost depends on residual depth, timed per (kind, depth) rather than per kind.
DEPTH_OPS = ("layer",)


def op_key(kind: str, depth: tuple[int, bool], name: str) -> str:
    """`kind|op`, or `kind#depth|op` for an op timed per depth."""
    if name not in DEPTH_OPS:
        return f"{kind}|{name}"
    return f"{kind}#{depth_tag(depth)}|{name}"


def collect_ops(layers: "Layers", model) -> dict:
    ops = {}
    for kind, per_depth in layers.depths.items():
        for depth, ids in sorted(per_depth.items()):
            layer = layers.layers[ids[0]]
            for name in DEPTH_OPS:
                ops[op_key(kind, depth, name)] = layer_ops(layer)[name]
    for kind, ids in layers.kinds.items():
        for name, mod in layer_ops(layers.layers[ids[0]]).items():
            if name not in DEPTH_OPS:
                ops[f"{kind}|{name}"] = mod
    inner, lm = backbone(model), language_model(model)
    for name in GLOBAL_OPS:
        # `norm` is a PPMissingLayer on a rank that does not own the tail.
        mod = getattr(inner, name, None)
        if isinstance(mod, torch.nn.Module):
            ops[name] = mod
    # LogitsProcessor, not lm_head: it matmuls the head's weight directly, so lm_head hooks never fire.
    ops["logits"] = lm.logits_processor
    return ops


def fold_depths(times: dict, layers: "Layers") -> tuple[dict, dict]:
    """Fold the per-depth rows into one row per kind; counts drop to the layers actually measured."""
    out = {k: v for k, v in times.items() if "#" not in k.partition("|")[0]}
    per_depth = layers.depth_counts
    counts = dict(layers.counts)
    for name in DEPTH_OPS:
        for kind in layers.kinds:
            got = [
                (times[op_key(kind, depth, name)], per_depth[(kind, depth)])
                for depth in sorted(layers.depths.get(kind, {}))
                if op_key(kind, depth, name) in times
            ]
            measured = sum(n for _, n in got)
            if got:
                out[f"{kind}|{name}"] = sum(ms * n for ms, n in got) / measured
            counts[kind] = min(counts[kind], measured)
    return out, counts


def coverage_gaps(counts: dict, layers: "Layers") -> dict[str, tuple[int, int]]:
    return {kind: (counts[kind], n) for kind, n in layers.counts.items() if counts.get(kind, n) < n}


def gap_note(gaps: dict) -> str:
    if not gaps:
        return ""
    parts = ", ".join(f"`{k}` {got} of {all_}" for k, (got, all_) in sorted(gaps.items()))
    return (
        f"A `layer` replay failed at one or more residual depths, so these"
        f" kinds are summed over the layers that were measured rather than over"
        f" the whole kind: {parts}. The roll-up is short by the rest, which is"
        f" what the ratio will show."
    )


def depth_rows(times: dict, layers: "Layers") -> list[tuple]:
    counts = layers.depth_counts
    rows = []
    for name in DEPTH_OPS:
        for kind in layers.kinds:
            for depth in sorted(layers.depths.get(kind, {})):
                key = op_key(kind, depth, name)
                if key in times:
                    n = counts[(kind, depth)]
                    rows.append((kind, name, depth_tag(depth), n, times[key], times[key] * n))
    return rows


def add_residuals(times: dict, counts: dict) -> None:
    """Give every parent a `<rest>` row: itself minus its measured children."""
    for kind in counts:
        for parent, kids in RESIDUALS.items():
            if f"{kind}|{parent}" in times:
                got = sum(times[f"{kind}|{k}"] for k in kids if f"{kind}|{k}" in times)
                times[f"{kind}|{parent}.<rest>"] = times[f"{kind}|{parent}"] - got


def contradictions(times: dict, counts: dict) -> list[tuple]:
    """Parents measured at under half their children's sum, so one of the timings was disturbed."""
    out = []
    for kind in counts:
        for parent, kids in RESIDUALS.items():
            key = f"{kind}|{parent}"
            if key not in times:
                continue
            got = sum(times[f"{kind}|{k}"] for k in kids if f"{kind}|{k}" in times)
            if got > 2 * times[key]:
                out.append((kind, parent, times[key], got))
    return out


def contradiction_banner(bad: list[tuple]) -> str:
    if not bad:
        return ""
    rows = ", ".join(
        f"`{kind}|{parent}` measured {ms:.3f} ms against {got:.3f} ms of children"
        for kind, parent, ms, got in bad
    )
    return (
        "\n> **WARNING: a measurement is not trustworthy.** "
        + rows
        + ". A parent runs its children, so it cannot take less time than they"
        " do; one of these timings was disturbed while it ran -- another"
        " process on the device is the usual cause. The `<rest>` row derived"
        " from it carries the error at full size and reversed sign, and every"
        " table below inherits it. The roll-up ratio does not catch this: the"
        " parent and its residual cancel. Re-run, and on a shared device raise"
        " `--repeat` so each row is the fastest of several passes.\n"
    )


def leaf_totals(times: dict, counts: dict) -> dict:
    totals = {}
    for name in LEAVES:
        ms = sum(
            times[f"{k}|{name}"] * n
            for k, n in counts.items()
            if f"{k}|{name}" in times and _partitions(times, k, name)
        )
        if ms:
            totals[name] = ms
    totals.update({n: times[n] for n in global_leaves(times)})
    return totals


def kernel_children(label: str, ops: dict, times: dict) -> list:
    """The timed child modules a parent's kernel profile leaves out, so its kernels are its own."""
    head, _, name = label.rpartition("|")
    if not head or name not in RESIDUALS:
        return []
    kind = head.partition("#")[0]
    mod = ops[label]
    owned = layer_ops(mod) if name == "layer" else {}
    kids = []
    for child in RESIDUALS[name]:
        if f"{kind}|{child}" not in times:
            continue
        if name == "layer":
            m = owned.get(child)
        else:
            attr = child.rpartition(".")[2]
            m = getattr(mod, _ATTR.get(child, attr), None)
        if isinstance(m, torch.nn.Module):
            kids.append(m)
    return kids


def kernel_totals(profiles: dict, times: dict, layers: "Layers") -> dict:
    """Kernel samples rolled up onto the hotspot rows, multiplied by the roll-up's layer counts."""
    folded, counts = fold_depths(times, layers)
    depth_of_key = {
        op_key(kind, depth, name): (kind, depth)
        for name in DEPTH_OPS
        for kind in layers.kinds
        for depth in layers.depths.get(kind, {})
    }
    leaves = set(global_leaves(times))
    totals = {}
    for label, kernels in profiles.items():
        if label not in times:
            continue
        head, _, name = label.rpartition("|")
        if not head:
            if label not in leaves:
                continue
            row, mult = label, 1
        elif label in depth_of_key:
            kind, depth = depth_of_key[label]
            if kind not in counts:
                continue
            row, mult = "layer.<rest>", layers.depth_counts[(kind, depth)]
        else:
            if head not in counts or not _partitions(folded, head, name):
                continue
            row = name + ".<rest>" if name in RESIDUALS else name
            mult = counts[head]
        dest = totals.setdefault(row, {})
        for kernel, (ms, calls, _) in kernels.items():
            old_ms, old_calls = dest.get(kernel, (0.0, 0))
            dest[kernel] = (old_ms + ms * mult, old_calls + calls * mult)
    return totals


def global_leaves(times: dict) -> list[str]:
    names = [n for n in GLOBAL_OPS if n in times]
    if "attn_res_out" in times:
        names = [n for n in names if n not in ATTN_RES_OUT_CHILDREN]
    return names


# Environment that selects a kernel rather than tuning one, so the report says what was measured.
KERNEL_ENV = (
    "SGLANG_USE_AITER",
    "SGLANG_AITER_K3_OPT",
    "AITER_FLYDSL_FORCE",
    "AITER_SITUV2_A8W4",
    "AITER_SITUV2_A4W4",
    "SGLANG_K3_KDA_FUSED_BACKEND",
    "SGLANG_K3_FLYDSL_SOURCE",
    "SGLANG_K3_FUSED_FRONT",
    "SGLANG_K3_SP_ATTN_RES",
    "SGLANG_MLA_DECODE_TUNE",
    "SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION",
    "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "MORI_SHMEM_MODE",
)


def kernel_env() -> list[tuple]:
    return [(name, os.environ.get(name) or "unset") for name in KERNEL_ENV]


GB = 1 << 30


def memory_section(args, model_runner, layers, kv_factor: int = 1) -> str:
    mr = model_runner
    kv_total = mr.token_to_kv_pool.get_kv_size_bytes()
    _, cap = kv_alloc_capacity(mr)
    dcp = dcp_size(mr)
    conc = local_conc(args, mr)
    used = conc * args.ISL * kv_factor
    dp = mr.ps.attn_dp_size
    n_mla = sum(len(v) for k, v in layers.kinds.items() if k.endswith("+mla"))
    n_kda = sum(len(v) for k, v in layers.kinds.items() if k.endswith("+kda"))

    state_total = state_bytes(mr)
    slots = state_slots(mr)
    state_cap = state_request_cap(mr)

    def row(name, n_layers, nbytes, per, unit):
        each = nbytes / per if per else 0
        return (
            name,
            n_layers,
            unit,
            round(each / max(n_layers, 1)),
            round(each),
            nbytes / GB,
        )

    # Summed, since weight_load_mem_usage is a free-memory delta that swings between runs.
    weights = sum(t.numel() * t.element_size() for t in (*mr.model.parameters(), *mr.model.buffers()))
    hbm = torch.cuda.mem_get_info()[1]
    frac = cfg(mr.server_args).mem_fraction_static
    rows = [row("MLA kv", n_mla, kv_total, cap, "token")]
    if state_total:
        rows.append(row("KDA state", n_kda, state_total, slots or 0, "slot"))
    rows += [
        ("**pools**", "", "", "", "", (kv_total + state_total) / GB),
        ("weights", "", "", "", "", weights / GB),
        ("**HBM**", "", "", "", "", hbm / GB),
    ]
    if dp > 1:
        capacity = (
            f"**MLA KV capacity {cap * dp:,} tokens cluster-wide** = {cap:,} per"
            f" rank x {dp} DP ranks = {cap * dp // args.ISL:,} requests at ISL"
            f" {args.ISL:,}"
        )
    elif dcp > 1:
        capacity = (
            f"**MLA KV capacity {cap:,} tokens** = {cap // args.ISL:,} requests"
            f" at ISL {args.ISL:,}; DCP {dcp} shards the latent along the"
            f" sequence, so each rank stores {cap // dcp:,} rows and a token of"
            f" context costs it 1/{dcp} of a row"
        )
    else:
        capacity = (
            f"**MLA KV capacity {cap:,} tokens** = {cap // args.ISL:,} requests"
            f" at ISL {args.ISL:,}; the MLA latent is replicated on every TP"
            " rank, so the cluster holds these same tokens, not tp_size times"
            " them"
        )
    notes = [capacity]
    if slots:
        notes.append(
            f"**KDA state pool {slots:,} slots per rank**, the same number at"
            f" any ISL, priced per slot above. sglang reserves several slots"
            f" per live request, so the requests it admits are fewer: this run"
            f" was given {state_cap:,} request rows. Neither DP, EP nor DCP"
            f" shards this pool: only the attention-TP width,"
            f" `--mamba-ssm-dtype` and `--mamba-radix-cache-strategy` change"
            f" what a rank pays per slot, which is why it is the ceiling a"
            f" concurrency sweep runs into first"
        )
    notes += [
        (
            f"`--mem-fraction-static {frac}` caps weights + pools at"
            f" {frac * hbm / GB:.0f} GB of the {hbm / GB:.0f} GB HBM; this run"
            f" lands at {(weights + kv_total + state_total) / GB:.0f} GB, split"
            " between the two pools by `--mamba-full-memory-ratio"
            f" {getattr(cfg(mr.server_args), 'mamba_full_memory_ratio', None)}`"
        ),
        f"this run uses {used:,} KV tokens, {used / cap * 100:.1f}% of this"
        " rank's KV pool"
        + (f", and {conc} state slots, {conc / slots * 100:.1f}% of its state pool" if slots else ""),
    ]
    moe = moe_parallel(mr)
    if moe.ep_size > 1:
        notes.append(
            f"EP {moe.ep_size}: this rank holds {layers.experts_per_rank()} routed"
            " experts"
            + (
                f", each with its intermediate split {moe.tp_size} ways"
                if moe.tp_size > 1
                else " at full intermediate width"
            )
            + (
                f"; under {moe.a2a} the two shared experts are replicated rather"
                " than sharded, and the all-to-all buffers (sized by the"
                " dispatch cap, allocated outside torch) are in neither this"
                " table nor the `--mem-fraction-static` budget"
                if moe.a2a
                else ""
            )
        )
    if not state_total:
        notes.append(
            "the KDA state pool was not found on `req_to_token_pool.mamba_pool`,"
            " so its bytes are missing from this table; the `pools` row is the"
            " MLA KV alone"
        )
    return "\n".join(
        [
            table(
                rows,
                ("component", "layers", "per", "B/unit/layer", "B/unit", "GB"),
                fmt=".2f",
            ),
            "",
            *(f"- {n}" for n in notes),
        ]
    )


RESIDUAL_CONTENTS = {
    "attn.<rest>": "attn internals: see the per-kind tables",
    "mlp.<rest>": "gather/scatter+prefix add (dense); fused front+tail (moe)",
    "layer.<rest>": "attn-residual aggregation (both layernorms fold into it)+attn reduce+residual adds",
}

KDA_ATTN_REST = "qkv/gate split+beta+forget gate+merged [f_a|b] GEMMs"
MLA_ATTN_REST = "qk_norm+absorb_bmm+output gate (kernel is attn_mqa)"

DCP_ATTN_REST = (
    "qk_norm+absorb_bmm+output gate+DCP partial exchange and LSE combine (kernel is attn_mqa_for_dcp_decode)"
)

MHA_ATTN_REST = "qk_norm+k/v split+kv_write (kernel is attn_mha)"


def attn_path(saved) -> str:
    if any(n.endswith("|attn.kv_b_proj") for n in saved):
        return "mha"
    if any(n.endswith("|attn.attn_mqa_for_dcp_decode") for n in saved):
        return "dcp"
    return "mla"


def residual_contents(model_runner, path: str) -> dict[str, str]:
    contents = dict(RESIDUAL_CONTENTS)
    a2a = moe_parallel(model_runner).a2a
    if a2a is None:
        return contents
    contents["mlp.experts"] = f"{a2a} dispatch+experts+combine"
    contents["mlp.<rest>"] = "gather/scatter+prefix add (dense); tail add (moe)"
    if model_runner.ps.attn_tp_size > 1:
        contents["layer.<rest>"] = (
            "attn-residual aggregation (both layernorms fold into it)+reduce-scatter+all-gather"
        )
    return contents


def kind_contents(model_runner, kind: str, path: str) -> dict[str, str]:
    contents = dict(residual_contents(model_runner, path))
    if kind.endswith("+kda"):
        contents["attn.<rest>"] = KDA_ATTN_REST
    else:
        contents["attn.<rest>"] = {"mha": MHA_ATTN_REST, "dcp": DCP_ATTN_REST}.get(path, MLA_ATTN_REST)
    return contents


def attn_line(model_runner, layers: "Layers") -> str:
    counts = layers.counts
    n_kda = sum(v for k, v in counts.items() if k.endswith("+kda"))
    n_mla = sum(v for k, v in counts.items() if k.endswith("+mla"))
    return (
        f"hybrid: {n_kda} KDA (linear, recurrent state), {n_mla} MLA (paged KV);"
        f" {attn_backends(model_runner)}"
    )


def attn_backends(model_runner) -> str:
    parts = []

    def walk(backend, role):
        children = [
            (attr, getattr(backend, attr, None))
            for attr in (
                "prefill_backend",
                "decode_backend",
                "full_attn_backend",
                "linear_attn_backend",
            )
        ]
        children = [(a, b) for a, b in children if b is not None]
        if not children:
            name = f"`{type(backend).__name__}`"
            parts.append(f"{role} {name}" if role else name)
            return
        for attr, child in children:
            sub = attr.removesuffix("_backend").removesuffix("_attn")
            walk(child, f"{role} {sub}".strip())

    walk(getattr(model_runner, "attn_backend", None), "")
    return ", ".join(parts)


def tree_name(name: str, depth: int, contents: dict[str, str] | None = None) -> str:
    contents = (RESIDUAL_CONTENTS if contents is None else contents).get(name)
    label = f"{name} :: {contents}" if contents else name
    return f"`{'  ' * depth}{label}`"
