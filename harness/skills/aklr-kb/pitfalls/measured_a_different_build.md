---
title: "The code you timed is not the code you edited"
kind: pitfall
scope: [remote or synced campaign harnesses, servers and benches that import another checkout, FlyDSL's compile cache, several venvs or FlyDSL versions, AKLR validate and e2e runs]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "aklr/ops/README.md"
  - "aklr/core/runtime/dispatch/_dispatcher.py:51"
  - "ROCm/aiter@e7d2453f2:requirements.txt:24"
---

# Measured a different build

## Symptom

Arms that should differ read the same, a fix seems to take with no effect, or a variant that cannot work is
reported correct. Nothing fails: every number looks like a normal reading.

- In an MXFP4 dense-GEMM campaign, a harness synced the local tree over the remote one before each run (S1), so
  edits made on the remote side did not reach the bench.
- A GLM-5.2 server imported its sglang from another checkout than the one the campaign synced its edits to, so
  sglang-side edits did not load (S5).

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
   (`../flydsl/language/version_notes.md` § Behaviour that changes without an error). AKLR's CI installs the
   baseline aiter (`AKLR_BASELINE_AITER`) over its image and runs the FlyDSL that aiter pins (S9, S10; read from
   the workflow, not checked in a run's log), not necessarily the `flydsl>=0.3.2` of `pyproject.toml`. From this
   repository's first commit (2026-10-31) to 2026-11-03 it installed the checkout with its dependencies, whose
   `pyproject.toml` pinned `flydsl==0.3.2` (S11). The repository the first schemas were measured in, whose history
   this one does not include, installed the checkout without its dependencies (S6), but that editable install ran
   the checkout's `setup.py`, which replaced any other FlyDSL with its pin 0.3.2 (S12): its runs used 0.3.2, not
   the FlyDSL of their image, 0.3.4.1 from 2026-10-24 on and 0.3.1 before, each the pin of the aiter commit the
   image was built from (S7, S8; read from the workflow and `setup.py`, not checked in a run's log). `--no-deps`
   does not stop a `setup.py` that installs a package itself. The `speedup` CI writes into a schema was measured
   on the FlyDSL of its run, and the run's commit message names the image, not the version.
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
- **Read back the binary.** Take the kernel names from a kernel trace of the timed run (its register counts are
  wrong above 128 architectural VGPRs, `../gemm/attribution.md` §9) and compare them with an ISA dump of the source you meant (`FLYDSL_DUMP_IR=1`, which skips the disk cache,
  `../flydsl/gemm.md` § mainloop §10) made in a separate run: do not time with the dump on
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
- The harness case is from one MXFP4 dense-GEMM campaign on MI355X; the server case is a GLM-5.2 sglang run.

## Evidence

- S1 `sync()` rsyncs `REPO_LOCAL` over the remote tree and runs before every command; the exclude list keeps `*.py`
- S2 `import aiter` imports whatever flydsl is first on `PYTHONPATH`; aiter's FlyDSL a8w8 kernel then failed under 0.2.4 and 0.4.0
- S3 aklr/core/runtime/dispatch/_dispatcher.py:51: `FallbackReason` (`no_schema`, `disabled`, `out_of_contract`, `no_workload`, `not_measured`, `inaccurate`, `not_qualified`, `not_warmed`)
- S4 aklr/core/runtime/dispatch/_recorder.py:55: the call record's `fallback_reason`, `None` when served
- S5 the server imports `sglang` from a separate checkout, not from the tree the campaign rsyncs its edits to (2026-09)
- S6 .github/workflows/ci.yaml, in the repository before this one (not in this one's history): `pip install --no-build-isolation --no-deps -e .` inside `AKLR_CI_IMAGE`; `requirements.txt` pins `flydsl==0.3.2`
- S7 ROCm/aiter@e7d2453f2:requirements.txt:24 `flydsl==0.3.4.1`; AKLR's CI image `lmsysorg/sglang:v0.5.22-rocm10-mi35x` (`.github/workflows/ci.yaml:5`) is built with `AITER_COMMIT=e7d2453f2`, and that build step runs `pip install -r requirements.txt` in the aiter checkout (image config, history step 94); no later step installs FlyDSL, and sglang v0.5.22 does not depend on it
- S8 ROCm/aiter@c16d44b93:requirements.txt:24 `flydsl==0.3.1`; the CI image before it, `lmsysorg/sglang:v0.5.19-rocm724-mi35x`, is built with `AITER_COMMIT=c16d44b93`, whose build step installs that file (image config, history step 92)
- S9 AMD-AGI/AKLR@f092828ea:.github/workflows/ci.yaml:285 `pip install "amd-aiter @ git+https://github.com/ROCm/aiter.git@$AKLR_BASELINE_AITER"`, over the image's aiter (:283); the commit is `AKLR_BASELINE_AITER` at :7
- S10 ROCm/aiter@dbd8bf5:setup.py:17 `FLYDSL_VERSION = "flydsl==0.3.2"`, in `install_requires` at :478
- S11 AMD-AGI/AKLR@9a29616:.github/workflows/ci.yaml:334 `pip install -e . -v` ("With its dependencies: aiter's kernels need its pinned FlyDSL, not the image's.", :332); `pyproject.toml:27` pins `flydsl==0.3.2`
- S12 setup.py, in the repository before this one (not in this one's history): an editable install (`is_develop_mode`) without `AITER_TRITON_ONLY` runs `pip install flydsl==0.3.2` (`FLYDSL_VERSION`) whenever another FlyDSL version is installed; the CI install of S6 is editable and sets no `AITER_TRITON_ONLY`; the parents of all seven CI measurement runs there (2026-10-16 to 2026-10-28) carry the same block and pin
