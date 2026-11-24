---
title: "The code you timed is not the code you edited"
kind: pitfall
scope: [remote or synced bench harnesses, servers and benches that import another checkout, FlyDSL's compile cache, several venvs or FlyDSL versions, layers that run a stored copy of a kernel or fall back to another implementation]
gens: [gfx950]
flydsl: 0.3.2
basis: single-observation
sources:
  - "ROCm/aiter@e7d2453f2:requirements.txt:24"
---

# Measured a different build

## Symptom

Arms that should differ read the same, a fix seems to take with no effect, or a variant that cannot work is
reported correct. Nothing fails: every number looks like a normal reading.

- While an MXFP4 dense GEMM was being optimized, a harness synced the local tree over the remote one before each run (S1), so
  edits made on the remote side did not reach the bench.
- A GLM-5.2 server imported its sglang from another checkout than the one the edits were synced to, so
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
   (`../flydsl/language/version_notes.md` § Behaviour that changes without an error). The FlyDSL under the
   ruler is the one the last install step left, not necessarily the one a project file asks for. Installing aiter
   brings the FlyDSL that aiter pins, 0.3.2 at aiter `dbd8bf5` (S10); an image built from an aiter commit carries
   that commit's pin, 0.3.4.1 in one sglang image and 0.3.1 in the one before it (S7, S8). `--no-deps` does not
   stop a `setup.py` that installs a package itself: aiter's, in an editable install, runs `pip install` of its
   FlyDSL pin whenever it finds another version (S12), so `pip install --no-build-isolation --no-deps -e .` on an aiter checkout at
   `dbd8bf5` in the 0.3.4.1 image leaves 0.3.2 (read from `setup.py` and the image's build steps, not checked in a
   run's log). Record the FlyDSL version with every number: the image's name does not give it.
5. **A layer between the caller and the kernel runs something else.** A runtime that keeps its own copy of a
   kernel's source runs that copy, not the file you edited. A dispatch layer that can fall back to another
   implementation (a library call, the caller's own path) does so without an error. Either way the reading is a
   normal number for code you did not mean to time.

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
- **Where a layer picks the implementation, check which one ran.** Compare any stored copy of the source with
  your file, and take the kernel names from a trace of the timed run (above): a fallback shows other names.

## Fix or workaround

- With a wrapper that syncs a tree, edit only the side it syncs from, or call the bench without the wrapper.
- Point the editable install or the front of `PYTHONPATH` at the edited tree.
- Clear the compile cache before believing a result (`../flydsl/SKILL.md` § Rules that hold everywhere).
- Pin one FlyDSL per venv and print it with every result.
- Where a layer runs a stored copy, update the copy before measuring; where it can fall back, confirm from a
  trace that the kernel ran before reading the time.

## Scope

- Any harness, server or bench that loads code from a path other than the one edited; any FlyDSL version (the
  cache rules cited are 0.3.2's).
- The harness case is from the optimization of one MXFP4 dense GEMM on MI355X; the server case is a GLM-5.2 sglang run.

## Evidence

- S1 `sync()` rsyncs `REPO_LOCAL` over the remote tree and runs before every command; the exclude list keeps `*.py`
- S2 `import aiter` imports whatever flydsl is first on `PYTHONPATH`; aiter's FlyDSL a8w8 kernel then failed under 0.2.4 and 0.4.0
- S5 the server imports `sglang` from a separate checkout, not from the tree the edits are rsynced to (2026-09)
- S7 ROCm/aiter@e7d2453f2:requirements.txt:24 `flydsl==0.3.4.1`; the image `lmsysorg/sglang:v0.5.22-rocm10-mi35x` is built with `AITER_COMMIT=e7d2453f2`, and that build step runs `pip install -r requirements.txt` in the aiter checkout (image config, history step 94); no later step installs FlyDSL, and sglang v0.5.22 does not depend on it
- S8 ROCm/aiter@c16d44b93:requirements.txt:24 `flydsl==0.3.1`; the image before it, `lmsysorg/sglang:v0.5.19-rocm724-mi35x`, is built with `AITER_COMMIT=c16d44b93`, whose build step installs that file (image config, history step 92)
- S10 ROCm/aiter@dbd8bf5:setup.py:17 `FLYDSL_VERSION = "flydsl==0.3.2"`, in `install_requires` at :478
- S12 ROCm/aiter@dbd8bf5:setup.py:57 an editable install (`is_develop_mode`) without `AITER_TRITON_ONLY` runs `pip install flydsl==0.3.2` (`FLYDSL_VERSION`, :66-72) whenever another FlyDSL version is installed (:63)
