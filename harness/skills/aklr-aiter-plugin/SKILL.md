---
name: aklr-aiter-plugin
description: AKLR's AITER kernel plugin — the out-of-tree provider in aklr/plugins/aiter serving AITER's routed rmsnorm, rmsnorm_with_add and blockscale GEMM contracts. Covers what this port serves, the four environment pitfalls that bite before any kernel runs, and how to verify it. Use when changing aklr/plugins/aiter/, the contracts AKLR serves, or the pinned aiter branch.
argument-hint: [what you are changing — adapter, a served contract, or the environment]
---

# AKLR as an AITER kernel plugin

**AITER is the authority.** The procedure every plugin follows — contracts,
entry points, what `select()` owes, provenance, versions, the kill switch —
lives in aiter's `aiter-kernel-plugin` skill and `aiter/plugin/README.md`. Read
that first, and when this file and aiter disagree, aiter wins and this file is
the bug.

This file holds only what is specific to **this** port.

## What AKLR serves

Four AITER contracts, mapped onto AKLR's own ops:

| AITER contract | AKLR op |
|---|---|
| `rmsnorm` | `aklr.ops.rms_norm.api.rms_norm` |
| `rmsnorm_with_add` | `add_rms_norm` |
| `gemm_a8w8_blockscale` | `gemm_a8w8_blockwise_scaled`, `operand_format` by flag |
| `gemm_a8w8_blockscale_bpreshuffle` | same op, preshuffled operand format |

Unlike AKLR's sglang and vllm plugins this one **replaces nothing**: AITER keeps
the call and asks per call whether we serve it.

```toml
[project.entry-points."aiter.plugins"]
aklr = "aklr.plugins.aiter:provider"
```

One selector per contract, each unpacking AITER's canonical argument list **by
name**. AITER applies the contract's defaults before asking, so the arity is the
contract: a contract that grows a parameter fails loudly here rather than
reading the wrong index quietly.

**The two GEMM entries share one AKLR contract but keep separate `run`
functions.** That is not redundancy. The shuffled layout is a flag inside
`gemm_a8w8_blockscale` *and* a separate public entry of its own, with different
arities — seven parameters against six. Binding `run` to the layout instead of
the contract is the defect this port shipped first; AITER's candidate check
refused it (`reason="contract-mismatch"`) and the call went native.

## Four things that bite before a kernel runs

In the order they will hit you. All four are AKLR-side, none is a property of
the interface.

**1. The device name decides whether anything loads.** On MI355X under
`rocm/atom-dev` with torch 2.10.0+rocm7.2.4:

```
torch.cuda.get_device_name(0)  ->  'AMD Radeon Graphics'
gcnArchName                    ->  'gfx950:sramecc+:xnack-'
```

AKLR's schemas declare `hardware: MI355X` and match on every field they state,
so the name mismatch skips **every** schema: the provider declines every call
and the interface looks broken while behaving exactly as specified. Override
`DeviceTarget` before importing AKLR, or relax the match to `gfx950` — the
information is there, it is the *name* that is unreliable.

**2. One unloadable kernel closes every op.** Schema loading is all-or-nothing
per process: if any kernel fails to import, `init_once` raises and every
contract stays at `NO_SCHEMA`, including the ones that were fine. Disable the
ones that cannot load, by name: `AKLR_DISABLE=fused_moe@1.0,...`

**3. flydsl must match the kernels.** A mismatch can pass `select()` and fail
inside `run()`. AITER does not fall back after execution begins — by design —
so this surfaces as an exception, not a silent native result. The fix is a
matching flydsl, not a retry.

**4. triton >= 3.8 for the non-shuffled GEMM.** With 3.7 those kernels are
skipped and reported; the preshuffled variant still serves. Correct behaviour,
not a failure.

## Known gap: `kernel_id` does not name a kernel

`_candidate()` fills `kernel_id` with the **contract** id (`rms_norm@1.0`) and
`artifact_digest` with a constant, because AKLR's dispatcher knows which kernel
file served a Point — the uuid hashes it — but exposes no accessor.

That makes the provenance identical for every kernel that ever serves the
contract, which is worthless exactly when a bug needs tracing to a binary. The
fix is a small read-only accessor on the AKLR side; until then the TODO in the
source says what is missing, and it should stay there.

## Verifying the port

Without a GPU:

```python
from aiter import plugin
status = plugin.status()
assert "aklr" in [p["name"] for p in status["providers"]]   # discovered
assert status["enabled"] == ()                              # and not enabled
plugin.configure(source="port check", enable=["aklr"])
plugin.explain("rmsnorm", out, x, w, 1e-6)                  # dry run, no kernel
```

On MI355X, `aklr/plugins/aiter/REPRODUCING.md` is the recipe, written so
somebody who was not involved can repeat it. Verified there: `rmsnorm` 64x6144
served to 3.9e-3 (bf16), 7x777 declined to native; `rmsnorm_with_add` served
with the residual output exact; the preshuffled GEMM served at m1/m8 and
declined n5120; `gemma_norm=True` and `split_k` declined as outside the
contract. **Both paths on every contract** — serving and declining — is what
makes it a verification rather than a demo.
