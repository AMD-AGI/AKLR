# Reproducing the AITER provider-interface run

This is the recipe behind the result cited in AITER's RFC ROCm/aiter#7240: AKLR,
discovered through packaging metadata and enabled by the framework, serving
AITER's routed contracts on an MI355X.

It is written so somebody who was not involved can run it. Where a step is not
yet possible from released software, it says so rather than pretending.

## What this does and does not claim

It checks **routing and correctness**: that AITER finds AKLR, asks it per call,
runs its kernel when it accepts, and reaches AITER's own kernel when it
declines — with the decision made before anything executes. It makes **no
performance claim**; the pilot never compared wall-clock against AITER.

## Prerequisites

- An MI355X (gfx950). Other targets will not match the kernels used here, and
  the recipe will tell you so rather than silently serving nothing.
- ROCm with a working `torch`, and `triton>=3.8` if you want the triton-backed
  kernels; with 3.7 they are skipped and reported, which is correct behaviour,
  not a failure.
- A `flydsl` matching the one the kernels were built against. A mismatch is
  visible: see *Known rough edges*.

## 1. AITER with the provider interface

**Not yet released.** The interface lives in draft PRs; `aiter.plugin` does not
exist on `ROCm/aiter` main yet, so AKLR's entry point resolves to nothing on a
released AITER and this step cannot be skipped:

```bash
git clone https://github.com/ROCm/aiter.git && cd aiter
gh pr checkout 7484          # the routed ops; builds on #7466 and #7465
python3 -m pip install -e .
```

Once those merge this becomes `pip install aiter` and the rest is unchanged.

## 2. AKLR

```bash
pip install -e .             # from this repository
```

Installation makes AKLR *available*, never active. AITER is default-deny: a
wheel appearing on the path must not change which kernels run.

## 3. Enable it, the way a framework would

```python
import aiter
from aiter import plugin

plugin.configure(enable=["aklr"], source="reproduction recipe")
```

`configure()` is the only way to enable a provider, and `source` is recorded in
every routing decision, so "why is this routed" always names an owner. Nothing
in the environment can enable one; `AITER_PLUGINS=0` only turns routing off.

Check it was found before going further:

```python
print([p["name"] for p in plugin.status()["providers"]])    # ['aklr']
print(sorted(plugin.status()["routed_ops"]))
```

An empty provider list means the entry point was not installed — usually an
editable install that predates the `aiter.plugins` entry in `pyproject.toml`.
Reinstall.

## 4. Run an op and read the decision

```python
import torch

weight = torch.rand(6144, dtype=torch.bfloat16, device="cuda")
value = torch.rand((64, 6144), dtype=torch.bfloat16, device="cuda")
out = aiter.rmsnorm2d_fwd(value, weight, 1e-6)
print(plugin.last_decision())
```

Expect `route='provider'`, `provider='aklr'`, and a `kernel_id` naming the
kernel that ran. Serving is **per Point**: some token counts are served and
others declined, by AKLR's own measurement, and both are ordinary operation.
`plugin.explain("rmsnorm", ...)` reports a decline without running anything.

To take a baseline, empty the enable list and call again:

```python
plugin.configure(enable=[], source="baseline")
assert plugin.last_decision().route == "native"
```

Assert it, do not assume it. An earlier version of our own script compared AKLR
against itself and reported a meaningless "max diff 0".

## Known rough edges

These are AKLR-side and will bite in this order.

### The device name decides whether any kernel loads

AKLR's schemas declare `hardware: MI355X`, and the match requires every field a
schema states. On some ROCm builds the card reports as **"AMD Radeon
Graphics"**, which normalizes to a model id that matches nothing — so every
schema is skipped on hardware they were measured on. The symptom is a single
warning and a provider that declines everything:

```
WARNING aklr...: loaded 0 schemas, 0 points; skipped 324 schemas of other
targets; device target gfx950/RADEON GRAPHICS
```

Until the matching is relaxed, state the model before importing AKLR:

```python
import torch
from aklr.core.platform.device import DeviceTarget

DeviceTarget._target_by_device_idx[torch.cuda.current_device()] = (
    DeviceTarget.from_names("gfx950", "MI355X")
)
```

This is a workaround for a detection failure on hardware that *is* an MI355X,
not a way to run kernels on a card they were not measured on. Check the line
that follows it reports `loaded N schemas` with N > 0.

### One unloadable kernel disables every op

Schema loading is all-or-nothing per process: if any kernel fails to import,
`init_once` raises and every contract stays closed at `NO_SCHEMA` — including
the ones that were fine. With a mismatched flydsl this appears as:

```
cannot import kernel.py::run: ModuleNotFoundError: No module named 'flydsl.extension'
```

Turn off the contracts that cannot load, naming them explicitly:

```bash
export AKLR_DISABLE=fused_moe@1.0,mega_moe@1.0,bmm@1.0        # extend as needed
```

The message names the contract to disable. Repeat until `init_once` succeeds.
Do not disable `gemm_a8w8_blockwise_scaled@1.0` if you want the GEMM pilot.

### A kernel can fail after being selected

A kernel built against a different flydsl may pass `select()` and then fail
inside `run()`. AITER does **not** fall back after execution begins, by design,
so this surfaces as an exception rather than a silent native result. That is
the failure boundary working; the fix is a matching flydsl, not a retry.

## The GEMM pilot

Same shape, with `gemm_a8w8_blockscale_bpreshuffle`, fp8 `e4m3fn` (the OCP
variant gfx950 and these kernels use — `e4m3fnuz` is out of contract and will
be declined), per-128-block fp32 scales, and `shuffle_weight(w, layout=(16,16))`
for the preshuffled operand. On this host only the preshuffled layout variant
loaded, because the non-shuffled kernels need `triton>=3.8`.

Covered points serve and uncovered ones decline before executing:

| m | n | k | expected |
|---|---|---|---|
| 1, 4 | 4096 | 4096 | served |
| 8 | 4096 | 8192 | served where the kernel loads |
| any | 5120 | 4096 | declined, native |

`aklr.ops.gemm.gemm_a8w8_blockwise_scaled.fallback_reason(...)` answers why a
call would be declined without running it, which is the quickest way to tell a
dtype mismatch (`OUT_OF_CONTRACT`) from an unmeasured shape (`NO_WORKLOAD`) from
a provider that loaded nothing at all (`NO_SCHEMA`).
