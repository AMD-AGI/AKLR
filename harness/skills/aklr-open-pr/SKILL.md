---
name: aklr-open-pr
description: Open a AKLR pull request whose body carries the evidence each kind of change needs (contracts, kernels, code, withdrawn kernels, documents), following .github/pull_request_template.md and the review-tier check. Use when committing a AKLR change and opening or updating its PR.
---

# Open a AKLR Pull Request

The PR body is a one-minute briefing for a reviewer who has the diff, and the only evidence a
reviewer has when GPU CI did not run. Write it from runs on the PR's head, never from memory.

## Required Context

Read [.github/pull_request_template.md](../../../.github/pull_request_template.md) (sections and
what each kind needs), [.github/scripts/review_tier/policy.py](../../../.github/scripts/review_tier/policy.py)
(what the `review-tier` status asks of each kind) and [aklr-bad-cases](../aklr-bad-cases/SKILL.md);
check the change against it before opening the pull request.

## Procedure

1. **Find the kinds** in `git diff --name-status origin/main...HEAD`, as `review_tier/change.py` does:
   - `aklr/ops/<op>/impls/<language>/<name>/<16 hex>.json`: deleted is a **withdrawn kernel**;
     empty `kernel.sources` is a **contract**; otherwise a **kernel**.
   - `docs/`, `harness/`, `.claude/` or `*.md`: a **document** (docs and agent skills).
     Anything else: **code**.
2. **Split** when kernels or contracts are mixed with code: a PR of their own merges on its
   evidence alone; mixed with code it waits for CI validation of every schema. Ask the user
   before splitting their branch.
3. **Gather evidence on the head commit**; rerun when the head moves.
   - Contracts: the model, serving config and `AKLR_EXPORT` run the workloads came from;
     `python -m aklr export check`.
   - Kernels: `python -m aklr validate --base origin/main --stages audit,bench <schemas>` on the target GPU (a
     remote node if none is local). Without `--base` a new kernel is held to the retention
     threshold (0.99) instead of the entry one (1.05) and is not timed against a served kernel,
     so a local pass can still fail CI. One table row per contract: measured / all workloads,
     accuracy n/n and its bound, lowest – highest speedup. Note every `AKLR_*` or `AITER_*`
     variable set that loosens a check.
   - Code: the tests run and their result. CI validation of every schema is required: add the
     `ci:gpu` label so CI runs it, and link the run once its `validation` job passes.
     A PR that changes CI (`.github/workflows/`, `scripts/`, `actions/`) runs its own CI, so that
     run is not evidence:
     code owners decide; keep such changes in a PR of their own.
   - `aklr/core`: also whether a schema or measurement version changes, and which kernels changed
     result against main (below the baseline, failing the audit or accuracy).
4. **Write the body** from the template: keep the sections the kinds need, delete the rest and
   every unfilled comment. Paste only real output; put long logs in `<details>` or link them.
   `Not run` lists every check the kinds need that did not run; "Nothing" only when all ran.
   `Risk` names what is affected, how it fails if wrong, and how to undo it.
5. **Title**: Conventional Commits, `type(scope): subject`, e.g. `feat(flydsl): ...`.
6. **Open or update** with `gh pr create --base main --body-file <file>` or `gh pr edit`. Add
   `ci:gpu` for kernels and code.
7. **Check** `gh pr checks <number>` for the `review-tier` status and the kinds it found. When
   its kinds differ from the body's, the body is wrong: fix it.

## Boundary

Do not merge, force-push, approve, or remove another person's changes. Do not claim a check
passed that is pending or did not run; pending is not success.

## Completion

Report the PR URL, the kinds found, the `review-tier` state, and what did not run.
