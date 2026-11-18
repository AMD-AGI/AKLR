###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

import sys
import time
from pathlib import Path

import torch

# sglang_bench.py lives one directory up.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import model_bench
import sglang_bench


class StepResult:
    """One prefill chunk's timings, folded and rolled up for the report."""

    def __init__(self, step, times, full_ms, layers, path, contents, skipped=None, fired=()):
        self.step = step
        self.skipped = skipped or {}
        self.full_ms = full_ms
        self.path = path
        # Coverage needs every op that fired, under the per-depth names folding replaces.
        self.fired = dict.fromkeys(fired, True)
        self.depths = model_bench.depth_rows(times, layers)
        times, counts = model_bench.fold_depths(times, layers)
        self.counts = counts
        self.gaps = model_bench.coverage_gaps(counts, layers)
        model_bench.add_residuals(times, counts)
        self.bad = model_bench.contradictions(times, counts)
        self.times = times
        self.totals = model_bench.leaf_totals(times, counts)
        self.rollup = sum(self.totals.values())
        self.contents = contents


def measure_step(args, mr, ops, context, step, layers, rank_print) -> StepResult:
    batch = model_bench.make_prefill_batch(mr, context, step)
    fb = sglang_bench.forward_batch_of(batch, mr)

    saved = {}
    with (
        model_bench.record_inputs(ops, saved),
        model_bench.record_attn_res_out(mr.model, ops, saved),
    ):
        mr.forward(fb)
    torch.cuda.synchronize()
    sglang_bench.assert_lockstep(saved)
    path = model_bench.attn_path(saved)
    # `saved` is emptied op by op to release inputs, so take its keys for Coverage now.
    fired = list(saved)
    fb = model_bench.runner_batch(saved, fb)
    # After the recording forward: it allocates the KDA state slot and marks it for a deferred clear.
    model_bench.randomize_state(mr)

    # Eager, as the server runs a chunk: the roll-up is checked against this and TTFT is summed from it.
    rank_print("  timing: full forward")
    model = model_bench.language_model(mr.model)
    full_ms = sglang_bench.bench_eager(
        lambda: model.forward(fb.input_ids, fb.positions, fb),
        args.reps,
        rounds=args.repeat,
    )

    inner = sglang_bench.graph_reps_for(step.extend)
    times = {}
    skipped = {}
    eager = set()
    # Popped, not iterated: a recorded entry pins that op's chunk-sized input tensors.
    for name in list(saved):
        rank_print(f"  timing: {name}")
        fn = model_bench.make_call(ops[name], saved.pop(name))
        try:
            times[name] = sglang_bench.bench(fn, args.reps, inner=inner, rounds=args.repeat)
        except torch.cuda.OutOfMemoryError:
            # Not a shape mismatch: every later op would be skipped too, so abort, not a partial report.
            raise
        except RuntimeError as e:
            # Only a deterministic failure may be skipped, since every rank must skip the same ops.
            first = str(e).split("\n")[0]
            if "out of memory" in first.lower():
                raise
            if "stream is capturing" in first:
                # A call a graph cannot hold (e.g. KDA's chunked prefill kernel): time it with launches in.
                times[name] = sglang_bench.bench_eager(fn, args.reps, rounds=args.repeat)
                eager.add(name)
                rank_print("    not capturable; timed eagerly")
            else:
                skipped[name] = first
                rank_print(f"    skipped: {first}")
        del fn
        if not sglang_bench.agreed(name in times):
            times.pop(name, None)
            skipped.setdefault(name, "skipped on another rank")
    model_bench.free_req_rows(batch, mr)
    # The recorded `attn_res_out` binds this chunk's AttnResidual bank, gigabytes at a long chunk; drop it.
    ops.pop("attn_res_out", None)
    result = StepResult(
        step,
        times,
        full_ms,
        layers,
        path,
        model_bench.residual_contents(mr, path),
        skipped,
        fired,
    )
    result.eager = eager
    return result


def work(server_args, port_args, args, gpu_id, tp_rank):
    mr, rank_print = model_bench.setup(server_args, port_args, gpu_id, tp_rank)
    layers = model_bench.Layers(mr)
    model_bench.balance_moe_routing(layers, mr.server_args.load_format)

    with sglang_bench.forward_ctx(mr):
        context = model_bench.make_prefill_context(args, mr)
        steps = model_bench.plan_prefill(args, mr)
        ops = model_bench.collect_ops(layers, mr.model)

        results = []
        for i, step in enumerate(steps):
            rank_print(f"chunk {i + 1}/{len(steps)}: {step.extend} new on {step.prefix}")
            results.append(measure_step(args, mr, ops, context, step, layers, rank_print))

    if tp_rank != 0:
        return
    md, headline = report(args, mr, layers, results, ops)
    path = Path(args.out or default_out(args, mr, layers))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    rank_print(headline)
    rank_print(f"report -> {path.resolve()}")


def default_out(args, mr, layers) -> str:
    parts = [
        model_bench.MODEL_TAG,
        "prefill",
        f"isl{args.ISL}",
        f"chunk{model_bench.chunk_flag(args, mr)}",
        f"hit{int(args.prefix_cache_hit * 100)}",
        *model_bench.parallel_tags(mr, layers),
    ]
    parts.append(mr.server_args.kv_cache_dtype)
    # Keeps a dummy run from overwriting the real run of the same shape.
    parts += filter(None, [sglang_bench.load_tag(mr.server_args)])
    parts.append(time.strftime("%Y%m%d"))
    return "_".join(parts) + ".md"


def ttft_ms(results) -> float:
    return sum(r.full_ms for r in results)


def summary_table(args, results) -> str:
    ttft = ttft_ms(results)
    new_tokens = sum(r.step.extend for r in results)
    rollup = sum(r.rollup for r in results)
    rows = [
        ("TTFT", f"{ttft:.1f} ms", f"{len(results)} chunks, one request per rank"),
        ("roll-up", f"{rollup:.1f} ms", "sum of the per-op leaf rows below"),
        (
            "ratio",
            f"{rollup / ttft:.3f}",
            "roll-up / TTFT; 1.0 means nothing was missed",
        ),
        (
            "computed",
            f"{new_tokens:,} tokens",
            f"of {args.ISL:,}; the other {args.ISL - new_tokens:,} came from cache",
        ),
        (
            "prompt rate",
            f"{args.ISL / ttft * 1000:,.0f} tok/s",
            "the whole prompt over TTFT, per request; what a client sees",
        ),
        (
            "compute rate",
            f"{new_tokens / ttft * 1000:,.0f} tok/s",
            "only the tokens forwarded; the same number at a 0% hit",
        ),
    ]
    return sglang_bench.table(rows, ("metric", "value", "note"))


def config_table(args, mr, layers) -> str:
    ps, sa = mr.ps, mr.server_args
    rows = [
        ("model", f"`{type(mr.model).__name__}`  {sa.model_path}"),
        ("layers", str(layers)),
        ("attention", model_bench.attn_line(mr, layers)),
        ("parallel", model_bench.parallel_line(mr)),
        *([("moe", line)] if (line := model_bench.moe_line(mr, layers)) else []),
        ("context", f"ISL {args.ISL:,} per request"),
        (
            "prefix cache",
            (
                f"{args.prefix_cache_hit:.0%} hit, rounded down to the"
                f" {model_bench.kv_alloc_capacity(mr)[0]}-token page. Modelled as MLA KV"
                " already in the pool; a KDA layer's own cache hit is a state"
                " restore this bench does not run, which changes no chunk's"
                " shape and so no chunk's time"
            ),
        ),
        (
            "chunk",
            f"--chunked-prefill-size {model_bench.chunk_flag(args, mr):,}"
            + (f" / {ps.attn_dp_size} DP ranks = " if ps.attn_dp_size > 1 else " = ")
            + f"{model_bench.chunk_tokens(args, mr):,} new tokens per forward",
        ),
        ("batch", f"1 request per attention rank, {args.CONC} in total"),
        ("kv dtype", f"`{sa.kv_cache_dtype}`"),
        ("weights", f"`{sa.load_format}` load, MXFP4 routed experts, BF16 elsewhere"),
        (
            "measurement",
            f"{args.reps} graph replays per op;"
            " the full forward eager, the way the server runs a chunk"
            + (f", fastest of {args.repeat} passes" if args.repeat > 1 else ""),
        ),
    ]
    return sglang_bench.table(rows, ("item", "value"))


def timeline_table(results) -> str:
    rows = [
        (
            i + 1,
            f"{r.step.prefix:,}",
            f"{r.step.extend:,}",
            r.path,
            r.full_ms,
            r.rollup,
            f"{r.rollup / r.full_ms:.3f}",
        )
        for i, r in enumerate(results)
    ]
    return sglang_bench.table(
        rows,
        ("chunk", "prefix", "new", "mla path", "ms", "roll-up", "ratio"),
        fmt=".2f",
    ) + (
        "\n\n`prefix` is what is already in the KV pool when the chunk starts --"
        " a cache hit and the chunks before it are the same thing here."
        " `mla path` is the path the 24 MLA layers dispatched to, read off"
        " whether `kv_b_proj` was called: `mha` is dense and projects the latent"
        " back out to real k and v, `mla` is the absorbed path that folds the"
        " same weights into bmms. The 69 KDA layers have one path at every"
        " depth. `ratio` is the roll-up over the full forward, the only"
        " self-check the decomposition has -- the forward eager and the ops in"
        " graphs, so anything an eager chunk spends between kernels lands here"
        " as a shortfall."
    )


def growth_table(results) -> str:
    totals = {}
    for r in results:
        for name, ms in r.totals.items():
            totals[name] = totals.get(name, 0.0) + ms
    names = sorted(totals, key=lambda n: -totals[n])
    rollup = sum(totals.values())
    # Exact prefixes, not a rounded `8k`, which would collide for chunks under 1000 tokens.
    headers = ("op",) + tuple(f"{r.step.prefix:,}" for r in results) + ("total", "share")
    rows = [
        (model_bench.tree_name(name, 0, results[-1].contents),)
        + tuple(r.totals.get(name, 0.0) for r in results)
        + (totals[name], f"{totals[name] / rollup * 100:.1f}%")
        for name in names
    ]
    note = (
        "\n\nms per chunk, columns labelled by the prefix each chunk starts on."
        " Leaf rows only, so a column sums to that chunk's roll-up and `share`,"
        " the row's cut of the whole TTFT, adds to 100%. A zero is an op that"
        " never ran in that chunk. Rows that climb across the table are reading"
        " the context; rows that stay flat only ever see the chunk."
        "\n\nEvery row here sums one op over both attentions, so a climbing"
        " attention row is the 24 MLA layers moving under 69 flat KDA ones: a"
        " KDA layer's conv and recurrence are a function of the chunk's own"
        " token count, whatever the prefix behind it. The per-kind table below"
        " separates them."
    )
    if len({r.path for r in results}) > 1:
        note += (
            " The `attn.<rest>` row spans both MLA paths and is named for the"
            f" last chunk's ({results[-1].path}); see the timeline for where the"
            " switch falls."
        )
    return sglang_bench.table(rows, headers, fmt=".2f") + note


def kind_sections(mr, result, counts) -> str:
    out = []
    for kind, n in counts.items():
        contents = model_bench.kind_contents(mr, kind, result.path)
        rows = [
            (
                model_bench.tree_name(name, depth, contents),
                result.times[f"{kind}|{name}"],
                result.times[f"{kind}|{name}"] * n,
            )
            for name, depth in model_bench.TREE
            if f"{kind}|{name}" in result.times
        ]
        table = sglang_bench.table(rows, ("op", "ms/call", "ms/chunk"))
        out.append(f"### {kind} -- x{n} layers\n\n{table}")
    return "\n\n".join(out)


def depth_section(result, gaps=None) -> str:
    table = sglang_bench.table(
        [(kind, depth, n, ms, total) for kind, _, depth, n, ms, total in result.depths],
        ("layer kind", "depth", "layers", "ms/call", "ms/chunk"),
    )
    note = table + (
        "\n\n`depth` is how many snapshot rows of the residual stream the"
        " layer's two aggregation points mix in; a fresh row is banked every 12"
        " layers. A `w` marks a layer that banks one: it snapshots at its"
        " pre-attention point, so its second aggregation mixes one row more"
        " than its first. `ms/chunk` is this group's whole contribution to the"
        " chunk."
    )
    if gap := model_bench.gap_note(gaps or {}):
        note += f"\n\n{gap}"
    return note


def eager_note(results) -> str:
    by_op: dict[str, set[str]] = {}
    for name in {n for r in results for n in r.eager}:
        kind, _, op = name.rpartition("|")
        by_op.setdefault(op, set()).add(kind.partition("#")[0])
    if not by_op:
        return ""
    return (
        "A CUDA graph could not capture these ops, so each was timed eagerly,"
        " launches included, the way the server runs a prefill chunk:\n\n"
        + "\n".join(
            f"- `{op}` on {', '.join(f'`{k}`' for k in sorted(kinds))}" for op, kinds in sorted(by_op.items())
        )
    )


def report(args, mr, layers, results, ops):
    last = results[-1]
    sections = [
        ("Summary", summary_table(args, results)),
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
        ("Memory, per rank", model_bench.memory_section(args, mr, layers, kv_factor=2)),
        ("Chunk timeline", timeline_table(results)),
        ("Hotspots -- one TTFT, all layers", growth_table(results)),
        (
            f"Per layer kind -- deepest chunk, prefix {last.step.prefix:,}",
            kind_sections(mr, last, last.counts),
        ),
        *(
            [
                (
                    (f"Per residual depth -- deepest chunk, prefix {last.step.prefix:,}"),
                    depth_section(last, last.gaps),
                )
            ]
            if len(last.depths) > len(layers.counts)
            else []
        ),
        *([("Coverage", coverage)] if (coverage := model_bench.coverage_note(ops, last.fired)) else []),
        *(
            [("Not replayable", note)]
            if (note := model_bench.skipped_note(sglang_bench.merged_skips(results), last.gaps))
            else []
        ),
        *([("Timed eagerly", note)] if (note := eager_note(results)) else []),
    ]
    md = "\n\n".join(
        [
            "# Kimi-K3 prefill profile",
            time.strftime("%Y-%m-%d %H:%M %Z"),
            *filter(
                None,
                [
                    model_bench.dummy_banner(mr.server_args),
                    model_bench.contradiction_banner([b for r in results for b in r.bad]),
                ],
            ),
        ]
        + [f"## {title}\n\n{body}" for title, body in sections]
    )
    ttft = ttft_ms(results)
    new_tokens = sum(r.step.extend for r in results)
    headline = (
        f"TTFT {ttft:.1f} ms over {len(results)} chunks  "
        f"{args.ISL / ttft * 1000:,.0f} tok/s prompt, "
        f"{new_tokens / ttft * 1000:,.0f} tok/s computed"
    )
    return md + "\n", headline


def make_parser():
    parser = model_bench.make_parser()
    parser.add_argument(
        "--prefix-cache-hit",
        type=float,
        default=0.0,
        help="fraction of ISL already in the KV cache, 0..1",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    # Not a flag: chunked prefill gives a long request a whole forward, and each DP rank runs its own.
    args.CONC = sglang_bench.prefill_conc(args)
    model_bench.run(work, model_bench.build_server_args(args), args)
