###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

import sys
import time
from pathlib import Path

import torch

# sglang_bench.py and kernel_breakdown.py live one directory up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import kernel_breakdown
import model_bench
import sglang_bench


def work(server_args, port_args, args, gpu_id, tp_rank):
    mr, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    layers = model_bench.Layers(mr)
    model_bench.balance_moe_routing(layers, mr.server_args.load_format)

    with sglang_bench.forward_ctx(mr):

        def new_batch():
            b = model_bench.make_extend_batch(model_bench.make_reqs(args, mr, decode_tokens=1), mr)
            sglang_bench.prepare_decode(b, mr)
            return b

        if args.full_step:
            n = args.full_step
            model_bench.randomize_kv(mr)

            # Synced per step: all CPU work sits on the critical path.
            ms = []
            for _ in range(n + 2):
                # A step advances the batch it is given, so each needs its own.
                b = new_batch()
                torch.cuda.synchronize()
                t = time.perf_counter()
                mr.forward(sglang_bench.forward_batch_of(b, mr))
                torch.cuda.synchronize()
                ms.append((time.perf_counter() - t) * 1000.0)

            rank_print("synced per step ms: " + " ".join(f"{m:.3f}" for m in ms))
            rank_print(f"step: synced {min(ms):.3f} ms")
            return

        batch = new_batch()
        model_bench.randomize_kv(mr)
        fb = sglang_bench.forward_batch_of(batch, mr)

        ops = model_bench.collect_ops(layers, mr.model)
        saved = {}
        if args.trace:
            sglang_bench.trace_step(args, fb, mr, rank_print, tp_rank)
            return

        with (
            model_bench.record_inputs(ops, saved),
            model_bench.record_attn_res_out(mr.model, ops, saved),
        ):
            mr.forward(fb)
        torch.cuda.synchronize()
        sglang_bench.assert_lockstep(saved)

        # The roll-up's reference, graph-captured so it is launch-free like the per-op rows.
        rank_print("timing: full forward")
        model = model_bench.language_model(mr.model)
        full_ms = sglang_bench.bench(
            lambda: model.forward(fb.input_ids, fb.positions, fb),
            args.reps,
            inner=1,
            rounds=args.repeat,
        )

        times = {}
        skipped = {}
        for name, entry in saved.items():
            rank_print(f"timing: {name}")
            try:
                with model_bench.rotated_calls(ops[name], entry) as (fns, inner):
                    times[name] = sglang_bench.bench(fns, args.reps, inner=inner, rounds=args.repeat)
            except torch.cuda.OutOfMemoryError:
                # Not a shape mismatch: every later op would be skipped too, so abort, not a partial report.
                raise
            except RuntimeError as e:
                # Outside its own forward a module can fail a check on ambient state; lose only that row.
                first = str(e).split("\n")[0]
                if "out of memory" in first.lower():
                    raise
                skipped[name] = first
                rank_print(f"  skipped: {skipped[name]}")
            # Agreed across ranks: a rank that kept a row the others dropped would unpair their collectives.
            if not sglang_bench.agreed(name in times):
                times.pop(name, None)
                skipped.setdefault(name, "skipped on another rank")

        # After all timing, so the profiler never touches a clock; every rank replays to pair collectives.
        profiles, profile_errors = {}, {}
        if not args.no_kernels:
            profiles, profile_errors = profile_saved_ops(ops, saved, times, rank_print)

    if tp_rank != 0:
        return

    if skipped:
        rank_print(f"{len(skipped)} op(s) could not be replayed standalone:")
        for name, err in skipped.items():
            rank_print(f"  {name}: {err}")

    md, headline = report(
        args,
        mr,
        layers,
        times,
        full_ms,
        model_bench.coverage_note(ops, saved),
        skipped=skipped,
        path=model_bench.attn_path(saved),
        kernels=(None if args.no_kernels else model_bench.kernel_totals(profiles, times, layers)),
        kernel_errors=profile_errors,
    )

    out = Path(args.out or default_out(args, mr, layers))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    rank_print(headline)
    rank_print(f"report -> {out.resolve()}")


def profile_saved_ops(ops, saved, times, rank_print):
    profiles, errors = {}, {}
    for label in times:
        rank_print(f"kernels: {label}")
        error = ""
        try:
            profiles[label] = kernel_breakdown.profile_call(
                model_bench.make_call(ops[label], saved[label]),
                model_bench.kernel_children(label, ops, times),
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            error = str(exc).split("\n")[0]
            if "out of memory" in error.lower():
                raise
        # Agreed across ranks: a rank that dropped out would leave its peers' next collective unpaired.
        if not sglang_bench.agreed(label in profiles):
            profiles.pop(label, None)
            error = error or "kernel profiling failed on another rank"
        if error:
            errors.setdefault(label, []).append(error)
            rank_print(f"  kernel breakdown unavailable: {error}")
    return profiles, errors


def default_out(args, mr, layers) -> str:
    parts = [
        model_bench.MODEL_TAG,
        "decode",
        f"isl{args.ISL}",
        f"conc{args.CONC}",
        *model_bench.parallel_tags(mr, layers),
    ]
    parts.append(mr.server_args.kv_cache_dtype)
    # Keeps a dummy run from overwriting the real run of the same shape.
    parts += filter(None, [sglang_bench.load_tag(mr.server_args)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def config_table(args, mr, layers) -> str:
    sa = mr.server_args
    rows = [
        ("model", f"`{type(mr.model).__name__}`  {sa.model_path}"),
        ("layers", str(layers)),
        ("attention", model_bench.attn_line(mr, layers)),
        ("parallel", model_bench.parallel_line(mr)),
        *([("moe", line)] if (line := model_bench.moe_line(mr, layers)) else []),
        (
            "batch",
            f"CONC {args.CONC}, {sglang_bench.local_conc(args, mr)} requests per rank",
        ),
        ("context", f"ISL {args.ISL} per request"),
        ("kv dtype", f"`{sa.kv_cache_dtype}`"),
        ("weights", f"`{sa.load_format}` load, MXFP4 routed experts, BF16 elsewhere"),
        (
            "measurement",
            f"{args.reps} graph replays per op; {sglang_bench.GRAPH_REPS} calls per"
            f" graph, more for a weight small enough to rotate over the"
            f" {model_bench.ROTATION_BYTES >> 20} MB buffer"
            + (f", fastest of {args.repeat} passes" if args.repeat > 1 else ""),
        ),
    ]
    return sglang_bench.table(rows, ("item", "value"))


def summary_table(args, mr, full_ms, rollup) -> str:
    a2a = sglang_bench.moe_parallel(mr).a2a
    rows = [
        (
            "step time",
            f"{full_ms:.3f} ms",
            "whole model in one graph, the way the server replays decode",
        ),
        ("roll-up", f"{rollup:.3f} ms", "sum of the per-op leaf rows below"),
        (
            "ratio",
            f"{rollup / full_ms:.3f}",
            (
                "roll-up / step time. Below 1.0 is work no row covers. Above 1.0"
                " is a row costing more alone than in the step, which the"
                f" {a2a} all-to-all inside `mlp.experts` does: replayed on its"
                " own it is the only thing every rank is running, so each"
                " exchange waits out the slowest peer with no neighbouring"
                " compute to overlap. Read that row as an upper bound"
                if a2a
                else "roll-up / step time; 1.0 means nothing was missed"
            ),
        ),
        (
            "throughput",
            f"{args.CONC / full_ms * 1000:.1f} tok/s",
            f"at CONC {args.CONC}",
        ),
    ]
    return sglang_bench.table(rows, ("metric", "value", "note"))


def hotspot_section(totals, rollup, contents, kernels=None, kernel_errors=None):
    rows = [
        (model_bench.tree_name(name, 0, contents), ms, f"{ms / rollup * 100:.1f}%")
        for name, ms in sorted(totals.items(), key=lambda kv: -kv[1])
    ]
    if kernels is not None:
        return kernel_hotspot_table(totals, rollup, contents, kernels, kernel_errors)
    note = (
        "\n\nLeaf rows only, so the shares add to 100%. A `<rest>` row is its parent"
        " minus the children listed under it, and the `::` tag names the work that"
        " lands there -- inline code and fused kernels with no module to hook. It is"
        " a difference of two measurements rather than a measurement, so any error"
        " in the parent or in any child lands in it at full size and reversed sign:"
        " a `<rest>` small next to its parent is mostly that error, and can read"
        " negative. `attn.<rest>` here spans both attentions; the per-kind tables"
        " below name each one's contents separately. "
    )
    if "mlp.experts" in contents:
        note += "`mlp.experts` is labelled too: under an all-to-all the dispatch and combine run inside it."
    else:
        note += (
            "Under plain TP `mlp.<rest>` holds most of the MoE front: the router"
            " gate, the latent down-projection and the shared experts' gate_up"
            " are one merged GEMM, so none of the three has a row of its own."
        )
    return sglang_bench.table(rows, ("op", "ms/step", "share")) + note


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


COLLECTIVE_MARKS = (
    "all_reduce",
    "allreduce",
    "cross_device_reduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "all_to_all",
    "alltoall",
    "nccl",
    "rccl",
    "mori",
)


def is_collective(kernel: str, api: str) -> bool:
    text = f"{kernel} {api}".lower()
    return any(m in text for m in COLLECTIVE_MARKS)


def kernel_hotspot_table(totals, rollup, contents, kernels, errors) -> str:
    headers = ("op / kernel", "ms/step", "share", "calls/step", "GPU ms/step", "API")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|--:|--:|--:|--:|---|",
    ]
    for name, ms in sorted(totals.items(), key=lambda kv: -kv[1]):
        items = kernels.get(name)
        gpu = f"{sum(v[0] for v in items.values()):.3f}" if items else ""
        lines.append(
            f"| {_cell(model_bench.tree_name(name, 0, contents))} | {ms:.3f}"
            f" | {ms / rollup * 100:.1f}% |  | {gpu} |  |"
        )
        if items is None:
            lines.append("|   ↳ kernel profile unavailable |  |  |  |  |  |")
            continue
        if not items:
            lines.append("|   ↳ no kernels of its own (all in child rows) |  |  |  |  |  |")
        for (kernel, api), (gpu_ms, calls) in sorted(items.items(), key=lambda kv: (-kv[1][0], kv[0])):
            mark = " ⧗" if is_collective(kernel, api) else ""
            lines.append(f"|   ↳ `{_cell(kernel)}`{mark} |  |  | {calls} | {gpu_ms:.3f} | `{_cell(api)}` |")
    note = (
        "\n\nUnder each row, every GPU kernel and copy its standalone replay"
        " launched, from one eager replay per layer kind (per depth for `layer`)"
        " on rank 0, multiplied by the layers it stands for. A parent's replay"
        " leaves out the kernels of its timed children, so what lands on its"
        " `<rest>` is its own. `GPU ms/step` on a hotspot row is its kernels'"
        " sum: eager and profiled, so it runs above the graph-timed `ms/step`"
        " where launches or a collective's wait for its peers fall inside a"
        " kernel. It says what a row is made of, not how long it takes; the"
        " `ms/step` column is still the measurement. API is the nearest"
        " Python frame and dispatcher op Kineto links to the launch."
        "\n\n⧗ marks a collective. Its GPU time is not its cost: the replay runs"
        " eagerly on every rank, so the kernel also holds the wait for the"
        " slowest peer to arrive, and outside a graph the custom all-reduce"
        " copies into its IPC buffer first -- the `Memcpy DtoD` beside it. The"
        " row's graph-timed `ms/step` is what the collective costs in the step."
    )
    roots = kernel_breakdown.package_roots(kernels.values())
    if roots:
        listed = ", ".join(f"`{n}` = `{p}`" for n, p in roots.items())
        note += f" Packages the APIs were loaded from: {listed}."
    if errors:
        failed = ", ".join(f"`{label}`" for label in errors)
        note += f"\n\nNo kernel profile for {failed}: " + "; ".join(
            sorted({e for errs in errors.values() for e in errs})
        )
    return "\n".join(lines) + note


def kind_sections(mr, times, counts, path) -> str:
    out = []
    for kind, n in counts.items():
        contents = model_bench.kind_contents(mr, kind, path)
        rows = [
            (
                model_bench.tree_name(name, depth, contents),
                times[f"{kind}|{name}"],
                times[f"{kind}|{name}"] * n,
            )
            for name, depth in model_bench.TREE
            if f"{kind}|{name}" in times
        ]
        table = sglang_bench.table(rows, ("op", "ms/call", "ms/step"))
        out.append(f"### {kind} -- x{n} layers\n\n{table}")
    return "\n\n".join(out)


def rollup_table(times, counts) -> str:
    rows = [
        (f"{kind} layer", n, times[f"{kind}|layer"], times[f"{kind}|layer"] * n)
        for kind, n in counts.items()
        if f"{kind}|layer" in times
    ]
    rows += [(name, 1, times[name], times[name]) for name in model_bench.global_leaves(times)]
    return sglang_bench.table(rows, ("op", "x", "ms/call", "ms/step")) + (
        "\n\nA `layer` row is the mean over that kind's residual depths, weighted"
        " by how many layers sit at each, so multiplying it by `x` reproduces"
        " the sum over those layers. `x` is the whole kind unless a depth could"
        " not be replayed, in which case it is the layers that were measured"
        " and `Not replayable` says so. The depths are broken out above."
    )


def depth_section(rows, gaps=None) -> str:
    table = sglang_bench.table(
        [(kind, depth, n, ms, total) for kind, _, depth, n, ms, total in rows],
        ("layer kind", "depth", "layers", "ms/call", "ms/step"),
    )
    note = table + (
        "\n\n`depth` is how many snapshot rows of the residual stream the"
        " layer's two aggregation points mix in; a fresh row is banked every 12"
        " layers. A `w` marks a layer that banks one: it snapshots at its"
        " pre-attention point, so its second aggregation mixes one row more"
        " than its first. `ms/step` is this group's whole contribution, so the"
        " column sums to what the `layer` rows of the roll-up come to."
    )
    if gap := model_bench.gap_note(gaps or {}):
        note += f"\n\n{gap}"
    return note


def report(
    args,
    mr,
    layers,
    times,
    full_ms,
    coverage="",
    skipped=None,
    path="mla",
    kernels=None,
    kernel_errors=None,
):
    depths = model_bench.depth_rows(times, layers)
    times, counts = model_bench.fold_depths(times, layers)
    gaps = model_bench.coverage_gaps(counts, layers)
    model_bench.add_residuals(times, counts)
    bad = model_bench.contradictions(times, counts)
    totals = model_bench.leaf_totals(times, counts)
    rollup = sum(totals.values())
    contents = model_bench.residual_contents(mr, path)

    sections = [
        ("Summary", summary_table(args, mr, full_ms, rollup)),
        ("Config", config_table(args, mr, layers)),
        (
            "Versions",
            sglang_bench.table(sglang_bench.versions(), ("component", "version")),
        ),
        (
            "Kernel environment",
            sglang_bench.table(model_bench.kernel_env(), ("variable", "value"))
            + (
                "\n\naiter reads the MoE activation dtype off this environment"
                " and the call itself does not distinguish the results, so a"
                " report without these rows does not say which arithmetic the"
                " `mlp.experts` row measured."
            ),
        ),
        ("Memory, per rank", model_bench.memory_section(args, mr, layers)),
        (
            "Hotspots -- one step, all layers",
            hotspot_section(totals, rollup, contents, kernels, kernel_errors),
        ),
        ("Per layer kind -- one step", kind_sections(mr, times, counts, path)),
        *(
            [("Per residual depth -- one step", depth_section(depths, gaps))]
            if len(depths) > len(counts)
            else []
        ),
        ("Roll-up -- one step", rollup_table(times, counts)),
        *([("Coverage", coverage)] if coverage else []),
        *([("Not replayable", note)] if (note := model_bench.skipped_note(skipped or {}, gaps)) else []),
    ]
    md = "\n\n".join(
        [
            "# Kimi-K3 decode profile",
            time.strftime("%Y-%m-%d %H:%M %Z"),
            *filter(
                None,
                [
                    model_bench.dummy_banner(mr.server_args),
                    model_bench.contradiction_banner(bad),
                ],
            ),
        ]
        + [f"## {title}\n\n{body}" for title, body in sections]
    )
    headline = (
        f"step {full_ms:.3f} ms  "
        f"roll-up {rollup:.3f} ms  ratio {rollup / full_ms:.3f}  "
        f"{args.CONC / full_ms * 1000:.1f} tok/s"
    )
    return md + "\n", headline


def make_parser():
    parser = model_bench.make_parser()
    parser.add_argument(
        "--trace",
        action="store_true",
        help="profile one step and print its kernels, instead of timing ops; "
        "eager, so kernel durations are real but the gaps between them are not",
    )
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="skip the per-kernel breakdown under each hotspot row",
    )
    parser.add_argument(
        "--CONC",
        type=int,
        default=1,
        help="concurrent requests, global across DP ranks",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    model_bench.run(work, model_bench.build_server_args(args), args)
