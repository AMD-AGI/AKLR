---
title: "The code you timed is not the code you edited"
kind: pitfall
scope: [remote or synced campaign harnesses, servers and benches that import another checkout, FlyDSL's compile cache, several venvs or FlyDSL versions, AKLR validate and e2e runs]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_041317/campaign_remote.py:13"
  - "AMD-AGI/AKLR@0cef4f2e3:aklr/ops/README.md"
  - "AMD-AGI/AKLR@0cef4f2e3:aklr/core/runtime/dispatch/_dispatcher.py:50"
  - "ROCm/aiter@e7d2453f2:requirements.txt:24"
---

# Measured a different build

## Symptom

Arms that should differ read the same, a fix seems to take with no effect, or a variant that cannot work is
reported correct. Nothing fails: every number looks like a normal reading.

- Six reads meant as three versions of an MXFP4 dense GEMM all fell in 0.985–0.991 because a harness synced the
  local tree over the remote one before each run (`../gemm/measurement.md` §1, its E2). Run without
  the sync, main read 0.9601 / 0.9597 and the branch head 0.9724: +1.30% *(field, E1)*.
- A `BLOCK_N=128` variant of the same kernel was recorded as correct; with the cache cleared it failed an
  assertion, and the earlier pass had come from the cached `BLOCK_N=256` binary (E1).
- A server-side change was judged to crash a deployment while the server imported another sglang checkout: the
  change had not loaded, and the crash came from an unrelated change (S5).

## Root cause

Each channel puts a different build under the ruler.

1. **The harness overwrites the tree.** A remote-run wrapper rsynced the local repository over the remote one on
   every call, with an exclude list that does not exclude `*.py` (S1): edits made on the remote machine were replaced by the
   local HEAD before each bench.
2. **The process imports another tree.** An editable install whose finder maps the package to another checkout,
   or a `PYTHONPATH` order, decides what loads; `import aiter` loads the first `flydsl` on the path (S2). The
   absence of a message your new code would print shows nothing: code that did not load prints nothing either.
3. **The compile cache returns an earlier binary.** What FlyDSL's cache key does and does not see is in
   `../flydsl/debug-flydsl-kernel.md` § What the cache key sees.
4. **Another venv or FlyDSL version.** Several changes between versions alter the result without an error
   (`../flydsl/language/version_notes.md` § Behaviour that changes without an error). AKLR's CI installs the checkout
   without its dependencies and runs the FlyDSL of its CI image, not the `flydsl==0.3.2` of the repository's
   `requirements.txt` (S6): 0.3.4.1 in the image of the runs from 2026-10-24 on, 0.3.1 in the one before, each
   the pin of the aiter commit the image was built from (S7, S8). The `speedup` CI writes into a schema was
   measured on that FlyDSL, and the run's commit message names the image, not the version.
5. **AKLR runs the schema, not your file.** Validation and dispatch run the `content` strings of the schema's
   `kernel.sources`, imported as a package `_aklr_<hash16>` (`aklr/ops/README.md` § Workflow and § The kernel
   section and ABI). A model reaches an `aklr/ops` kernel only where the engine calls the op's api: AKLR's plugins
   route models to plugin ops (`aklr/plugins/README.md`). The api returns a reason instead of running the kernel
   (S3) when for example AKLR is disabled, the Point has no workload or no qualifying measured speedup, or it is
   first called inside a graph capture; a toolchain outside the schema's `requires` skips the schema with an INFO
   log (`aklr/ops/README.md` § The kernel section and ABI). A schema you just edited has no `speedup` until CI
   writes one (§ Workflow), so it is not served at its Points unless `AKLR_MIN_SPEEDUP` allows unmeasured
   workloads (§ Environment variables).

## How to detect

- **Ask the process that ran.** In the same interpreter and environment as the measurement, print at startup,
  unconditionally: `sys.executable`, `flydsl.__version__`, `flydsl.__file__`, and the `__file__` of the module
  that defines your kernel.
- **On a remote machine, hash the source there right after the run** and compare it with the file you meant to
  time.
- **Read back the binary.** Take the kernel names from a kernel trace of the timed run (register counts there are
  half the true value, `../gemm/attribution.md` §9) and compare them with an ISA dump of the source you meant (`FLYDSL_DUMP_IR=1`, which skips the disk cache,
  `../gemm/mainloop.md` §10) made in a separate run: do not time with the dump on
  (`../flydsl/gemm.md` § measurement §6). A compile-time switch that shows in the kernel name makes the arms
  tell apart in a trace (`../gemm/attribution.md` §3). Break the kernel on purpose once
  (`../flydsl/debug-flydsl-kernel.md` § What the cache key sees).
- **In AKLR, check what was served.** Compare the schema's `content` with your file. Call the op's api with
  `AKLR_EXPORT=1`: every call record carries `fallback_reason`, empty when the kernel served it (S4).

## Fix or workaround

- With a wrapper that syncs a tree, edit only the side it syncs from, or call the bench without the wrapper.
- Point the editable install or the front of `PYTHONPATH` at the edited tree.
- Clear the compile cache before believing a result (`../flydsl/SKILL.md` § Rules that hold everywhere).
- Pin one FlyDSL per venv and print it with every result.
- In AKLR, copy the file into the schema's `content` before `python -m aklr validate`, and read the fallback
  reasons of an e2e run before reading its time.

## Scope

- Any harness, server or bench that loads code from a path other than the one edited; any FlyDSL version (the
  cache rules cited are 0.3.2's).
- The numbers above are from one MXFP4 dense-GEMM campaign on MI355X; the server case is a GLM-5.2 sglang run.

## Evidence

- E1 [2026-09, gfx950, FlyDSL ?, N=3, scored bench run without the rsync wrapper (the bench script called directly)] main read 0.9601 / 0.9597 and HEAD 0.9724, +1.30%, where the wrapper had given six reads of the same local HEAD in 0.985–0.991; the `BLOCK_N=128` arm failed `assert not b_ilv or b_ilv == nd` once the cache was off (MXFP4 dense GEMM, Llama training shapes; KB note; local:/workspace/code/gpt_oss_docker/.claude/memory/project_mxfp4_llama_gemm_b200_campaign.md:327; :127; :322)
- S1 local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260910_041317/campaign_remote.py:13 `sync()` rsyncs `REPO_LOCAL` over the remote tree and runs at :20, before every command; the exclude list `/workspace/code/gpt_oss_docker/sync/.rsync-exclude` keeps `*.py`
- S2 local:/workspace/code/gpt_oss_docker/work/gemm_h2h/code/h2h_adapter_vendor.py:56 `import aiter` imports whatever flydsl is first on `PYTHONPATH`; aiter's FlyDSL a8w8 kernel then failed under 0.2.4 and 0.4.0
- S3 aklr/core/runtime/dispatch/_dispatcher.py:50 at 0cef4f2e3: `FallbackReason` (`disabled`, `out_of_contract`, `no_workload`, `not_measured`, `inaccurate`, `not_qualified`, `not_warmed`)
- S4 aklr/core/runtime/dispatch/_recorder.py:58 at 0cef4f2e3: the call record's `fallback_reason`, `None` when served
- S5 local:/workspace/code/gpt_oss_docker/.claude/memory/reference_sglang_e2e_ab_pitfalls.md:49 the server's editable install mapped `sglang` to `/sgl-workspace/sglang_kyle/python/sglang` while the edits went to another checkout; both conclusions drawn from those runs were withdrawn (KB note, 2026-09)
- S6 .github/workflows/ci.yaml:404 at AMD-AGI/AKLR 0cef4f2e3: `pip install --no-build-isolation --no-deps -e .` inside `AKLR_CI_IMAGE` (:5); `requirements.txt:25` pins `flydsl==0.3.2`
- S7 ROCm/aiter@e7d2453f2:requirements.txt:24 `flydsl==0.3.4.1`; AKLR's CI image `lmsysorg/sglang:v0.5.22-rocm10-mi35x@sha256:1438dd8b9000` (`.github/workflows/ci.yaml:5`; CI runs of cca5b12df, 102877ca0, 5571f4589) is built with `AITER_COMMIT=e7d2453f2`, and that build step runs `pip install -r requirements.txt` in the aiter checkout (image config, history step 94); no later step installs FlyDSL, and sglang v0.5.22 does not depend on it
- S8 ROCm/aiter@c16d44b93:requirements.txt:24 `flydsl==0.3.1`; the image of CI run 39908769447 (cc6e3b96c), `lmsysorg/sglang:v0.5.19-rocm724-mi35x@sha256:424411dacf57`, is built with `AITER_COMMIT=c16d44b93`, whose build step installs that file (image config, history step 92)
