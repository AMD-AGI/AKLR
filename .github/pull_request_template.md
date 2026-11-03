<!-- A one-minute briefing for a reviewer who has the diff. Drop a section with nothing to say.
What each kind of change needs (the review-tier check finds the kinds from the diff):
- contract (schema, empty kernel.sources): Why, Scope, Verification (export), Risk. No GPU run.
- kernel (schema with kernel.sources): every section. GPU validation (ci:gpu).
- code, aklr/core above all: every section. CI validation of every schema and a code owner's approval.
- withdrawn kernel (deleted schema): Why, Scope, Risk.
- document (docs/, harness/, .claude/, *.md: docs and agent skills): Why only.
Kernels and contracts in a PR of their own merge on their evidence alone; mixed with code they
wait for the code. -->

## Why

<!-- Required, every kind. The problem and the approach, 1–3 short sentences. -->

## Scope

<!-- Required for every kind but documents. 1–3 items: what this covers (models, ops, shapes) and what it leaves on AITER or defers. -->

## Tradeoffs

<!-- Optional, any kind: only an alternative a reviewer would ask about (another baseline, language, bound). -->

## Verification

<!-- Required when the PR has contracts, kernels or code; delete it for documents and withdrawn
kernels only. Everything below ran on this PR's head. Keep only the subsections of the kinds
this PR has; delete the others. -->

<!-- Required: both lines. -->
**Environment**: <!-- head commit, GPU, image, ROCm / FlyDSL / Triton / AITER versions -->
**Gate overrides**: <!-- AKLR_* or AITER_* variables set for the run that loosen a check, or "none" -->

### Contracts

<!-- Required when the PR adds or changes a contract; delete otherwise.
Where the workloads come from: the model, its serving config, the AKLR_EXPORT run. -->

```text
$ python -m aklr export check
<output>
```

### Kernels

<!-- Required when the PR adds or changes a kernel; delete otherwise.
One row per contract. Measured: workloads benched / all (a timeout is not measured).
Accuracy: passing / measured, and the bound. Speedup: lower bound over the baseline, lowest – highest. -->

| contract | baseline | measured | accuracy (bound) | speedup |
|---|---|---|---|---|
| | | n/n | n/n (SqnrBound 30 dB) | x – x |

<!-- Required: the CI run of this head; without one, the local log below. -->
<details><summary>Validation log</summary>

```text
$ python -m aklr validate --base origin/main --stages audit,bench <schemas>
<output>
```

</details>

### Code

<!-- Required when the PR changes code (any file but a schema or a document); delete otherwise.
Required: tests run and their result, new tests added; and the CI run whose `validation` job
passed on this head (it validates every schema). -->

<!-- Required when aklr/core changes, delete otherwise: does a schema or measurement version change? Which kernels changed result
against main (fell below the baseline, now fail the audit or accuracy)? "None" is an answer. -->

## Not run

<!-- Required for contracts, kernels and code. What was not run or not measured, and why. "Nothing" only when every check its kinds
need ran on this head. -->

## Risk

<!-- Required for every kind but documents. -->
Affects: <!-- models, ops, shapes; for code, which kernels or stages (often: every kernel) -->
If wrong: <!-- wrong output, a crash, slower than AITER, a CI failure -->
Undo: <!-- `AKLR_DISABLE=<contract id>` turns one op off; otherwise `git revert`, and what that
does not undo (a schema or measurement version, published measurements) -->
