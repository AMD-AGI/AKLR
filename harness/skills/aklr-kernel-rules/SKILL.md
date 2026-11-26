---
name: aklr-kernel-rules
description: AKLR's side of kernel work, read beside the kernel optimization book, which cites nothing of AKLR. What a kernel must do to be served by AKLR, how AKLR serves and times it, where its kernels are, and how to validate, A/B, reproduce and roll back one. Load it when you write, port, optimize, debug or review a kernel for an AKLR definition file (schema), or read a speedup AKLR stored. KernelForge runs set up by forge_task load it with the book.
---

# Writing a kernel for AKLR

The kernel optimization book ([SKILL.md](../kernel-optimization-book/SKILL.md)) says what makes a kernel fast on AMD
GPUs. It names kernels by op and shapes and cites nothing of AKLR. This skill is AKLR's side: what AKLR rules out,
how it serves and times a kernel, where its kernels are, and how one gets in and out. Paths starting with `aklr/`,
`docs/` or `harness/` are relative to the repository root. `aklr/plugins/aiter/README.md` ("the README") is the
authority: where this skill disagrees with it, the README wins.

AKLR's words (README § What an op is): a **contract** defines an op. A **Point** is one call's shapes, dtypes, attrs
and layout. A **definition file** ("schema", `<uuid>.json`) holds one scenario (a Point without the axes that vary
per call), one kernel's sources, and its **workloads**: the Points it serves. The book's "size" is a workload.

## Stop: what an AKLR kernel cannot do

Check a plan against this table before writing code. Rules: README § The kernel section and ABI; SA0–SA5 are its
audit rules.

| Rule | What it rules out here |
|---|---|
| `run(*, <kernel_call_keys>)` writes every element of every preallocated output and returns nothing (SA4). Bench fills the outputs with NaN, or `0xFF` bytes, before each checked call | Allocating or keeping an output; leaving a row unwritten (an empty group writes zeros) |
| Outputs are within the contract's error bounds. Under `compare.determinism: bitwise`, the default, two runs give the same bytes | Reducing in arrival order (atomic split-K, stream-K fix-ups): reduce in a fixed order (`../kernel-optimization-book/gemm/correctness.md` §8, `../kernel-optimization-book/gemm/grid_pricing.md` §2). A `bounded` contract compares no two runs |
| Launch code (`run` and what it calls) does not synchronize, call `.item()` / `.cpu()` / `.tolist()`, print, do I/O, read a clock or the environment, or change torch state (SA3) | Timing or racing configurations in `run`; a host branch or grid size taken from device data |
| Compilation and the choice of configuration happen in a module-level `@functools.cache` function, on a Point's first call, for every variant a scalar selects. Later calls may be inside a CUDA graph capture, where a compile is `capture_unsafe` | The first-call race of `../kernel-optimization-book/gemm/autotune.md` §1 as written: choose from the shape, with a table raced offline. A compile-time constant taken from a scalar's value or from an `any_size` axis |
| Inputs other than the mutable ones are read-only (SA2). Nothing derived from an argument is kept across calls. No `global` or `nonlocal` (SA1) | Caching a repacked weight, also by address or `_version`: repack on every call, or serve the contract's shuffled layout variant where it has one |
| Scratch is allocated in the call (`torch.empty`: contents undefined) or once per shape in the compile function and rewritten before it is read. No call depends on what an earlier call left | Counters that live across calls: zero workspaces in each call before any read (`../kernel-optimization-book/gemm/small_m.md` §4), or stamp slots with this launch's tag (`../kernel-optimization-book/gemm/wait_free_fold.md` § Folding without a counter) |
| The kernel is right for every value of a scalar's declared range and every size of an `any_size` axis, not only the ones bench draws | Hard-coding a ranged scalar such as an epsilon; assuming the bench's pool size |
| Sources import nothing of `aklr` but `aklr.core.jit` (SA5) and import sibling files relatively. The entry file's docstring describes the kernel, with no history or speedup (SA4) | Calling the baseline or the reference as a fallback; importing another definition's helper (copy it) |
| Not in the audit; `docs/forge_task.md` (step 5) requires it before a kernel is delivered: no workgroup waits for another without a way out. Validate gives a kernel the GPU alone, a deployment does not (`../kernel-optimization-book/attention/split_kv.md` §2, Bounded wait) | A split-K fold or split-KV merge whose owner polls its partners with no bound, or gives up and writes. Check: [validate.md](validate.md) § Stress |

AKLR is a caller that compiles one kernel per Point, may capture any later call, overwrites weights in place and
checks every output element. Where a book page assumes a caller that tunes at run time or owns its weights, read
it as `../kernel-optimization-book/moe/INDEX.md` § Callers that compile per token count and capture the call says.

## How AKLR serves a kernel

- **Per Point.** Dispatch reads the call's Point and runs the file with the best stored `speedup` there, from
  `AKLR_MIN_SPEEDUP` (default 1.05) up, never one with `accuracy: failed`. Otherwise the api returns a
  `FallbackReason` and the caller runs its own code (`docs/ops_tutorial.md` § 5 lists the reasons). So judge a kernel
  per Point, against the faster of the file served there and the baseline, never on a mean.
- **Sizes match exactly.** A var axis is matched to the workloads' values; only an axis the contract declares
  `any_size` is free. A size no file lists falls back with `no_workload`: list the sizes the model sends
  (README § Workflow, export).
- **What `run` may assume.** Dispatch calls it only with tensors in the stored layout, not negated, 16-byte aligned,
  on the table's device, no output overlapping an input, attrs and scalars inside the contract
  (`aklr/core/contract/fastkey/`). Any other call falls back with `out_of_contract`.
- **`run` does not fall back.** An exception from `run` reaches the caller. To serve fewer Points, edit `workloads`.
- **First call.** A Point's first call compiles. First called inside a capture, it falls back with `not_warmed`,
  and the graph keeps the fallback.
- **Only CI's numbers serve.** CI writes `accuracy`, `speedup` and `microseconds` into the file; a new or edited
  file has none and is not served. To run one before that: `AKLR_MEASURED_DIR=<validate --out>/schemas`, or
  `AKLR_MIN_SPEEDUP=0` (README § Environment variables).
- **How a model reaches it.** Through AKLR's provider for aiter's routed ops (`AKLR_AITER_PLUGIN=1`), a plugin route
  (`AKLR_SGLANG_PLUGIN=1`, `AKLR_VLLM_PLUGIN=1`), or a host that calls the op's api (`aklr/plugins/README.md`).
  `AKLR_TRACE=op` logs each distinct call with the kernel that served it or the fallback reason, and `AKLR_EXPORT=1`
  records every call's Point with its `fallback_reason`: read them before you read an end-to-end time.
- **The baseline** is the contract's baseline class (`<op>/baseline.py`): for `aklr/plugins/aiter`, mostly ROCm/aiter
  at the commit `aiter` in `stack.json`; for an op of the sglang or vLLM plugin, the engine's own code. AKLR does
  not ship or patch aiter.
  The book cites `aiter/` paths at the revision its own SKILL.md names, which need not be that commit.

## How AKLR times a kernel

- Each call is replayed in a CUDA graph behind a 1 GiB read that empties L2 and MALL. A call's time is the sum of
  its kernels' device times; host time and the gaps between kernels are not in it.
- `speedup` is a one-sided lower bound of baseline time over kernel time, over 16 or 32 rounds, per Point.
- Every kernel of a call reads at least about 4.4 µs on this timer. A short kernel and its baseline can read the
  same, and each extra launch costs one floor ([timing.md](timing.md) § The floor).
- A local `python -m aklr validate` runs the same ruler. Its result does not count: CI measures.
- More: [timing.md](timing.md); thresholds and judging: `harness/skills/read-validation/references/ruler.md`.

## Where the kernels are

```
aklr/plugins/aiter/<op>/impls/<language>/<name>/<uuid>.json                 the ops aiter's functions reach
aklr/plugins/<sglang|vllm>/.../<package>/impls/<language>/<name>/<uuid>.json  an engine plugin's own ops
```

- `<name>` is the contract, its non-default variants and its fixed shapes (README § Names):
  `aklr/plugins/aiter/gemm/impls/flydsl/gemm_n6144_k4096/` is the bf16 GEMM at N 6144 × K 4096. A case page of the
  book names its kernel by op, shapes, dtypes and sizes: that is one `<name>` directory and the workloads of one
  file in it.
- A directory may hold several files: kernels for different sizes, and a definition with `"sources": []` (no
  kernel, never served). Another language's files for the same Points are under `impls/<other language>/<name>/`.
- A file's kernel is `kernel.sources[*].content`, each a whole source file; `kernel.entry_point` names the entry.
  `workloads[*]` holds each Point's `axes` and CI's `accuracy`, `speedup` and `microseconds`.

```bash
find aklr -type d -path '*/impls/*' -name '*n6144_k4096*'        # the directories of a shape
python3 -c '
import json, pathlib, sys
for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):      # each file: its sources, its Points, CI numbers
    s = json.loads(p.read_text())
    print(p.name, [f["path"] for f in s["kernel"]["sources"]] or "no kernel")
    for w in s["workloads"]:
        print("  ", w["axes"], w.get("accuracy", "-"), w.get("speedup", "-"), w.get("microseconds", "-"))
' aklr/plugins/aiter/gemm/impls/flydsl/gemm_n6144_k4096
python3 -c '
import json, pathlib, sys
for f in json.load(open(sys.argv[1]))["kernel"]["sources"]:    # write the sources of one file into a directory
    path = pathlib.Path(sys.argv[2], f["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f["content"])
' <file>.json /tmp/kernel_src
```

## Workflow

1. Find the contract (`<op>/contract.py`) and the Points the model sends (README § Workflow, steps 1–2).
2. Read the book for the op and regime. Check the plan against the Stop table.
3. Write the kernel and put it into a definition file (`docs/ops_tutorial.md` § 2–3; in a forge_task run, `publish`).
4. `python -m aklr export check`, then validate with `--base origin/main` on one idle GPU, then the stress check
   ([validate.md](validate.md)). Fix or remove every Point it demotes.
5. Open a PR with the `ci:gpu` label. CI measures. After the merge it measures on main, and a bot PR writes the
   numbers into the file: dispatch serves the kernel once that PR merges (`docs/ops_tutorial.md` § 5).

## Pages

| Need | Read |
|---|---|
| Validate, A/B a change, reproduce a number or a failing step, the stress check, roll back | [validate.md](validate.md) |
| What the timer measures and what it leaves out: the floor, noise, inputs, the baseline's build | [timing.md](timing.md) |
| What is specific to FlyDSL, Triton or Gluon, and HIP kernels in AKLR | [languages.md](languages.md) |
| The rules in full, the definition file's fields, commands, environment variables | `aklr/plugins/aiter/README.md` |
| A walk-through of adding a kernel; creating an op | `docs/ops_tutorial.md` |
| Routes, the aiter provider, running a model with a plugin | `aklr/plugins/README.md`, `docs/sglang_plugin_tutorial.md`, `docs/vllm_plugin_tutorial.md` |
| Reading a validation report, CI's jobs and artifacts | `harness/skills/read-validation/SKILL.md` |
| An optimization campaign with KernelForge | `harness/scripts/forge_task/README.md`, `harness/skills/forge-loop-campaign/SKILL.md` |
| What a PR must show: 5% at a Point, 2% end to end for a new route | `docs/acceptance_criteria.md`, `harness/skills/open-pr/SKILL.md` |
