# Validation and freshness

## Freshness

Evidence is bound to candidate digest, head, validation base, validator revision and image digest. It is
stale when the head, validator or image changes, or when

```bash
python3 -I harness/automations/shepherd/skills/optimize-and-deliver-kernels/scripts/candidate.py freshness --schema <path> --base <validation base> --main <main>
```

run at the repository root exits 1 (relevant changes, or rewritten history). Exit 2 is `unknown`. Never
judge relevance by hand. A summary never replaces the reports.

## Validating

On the campaign's own GPU. Start only after the worker's containers are observed gone. Make a fresh
validator clone at the pinned revision in `<campaign>/validate`, check its hash, and bring the frozen
commit in as a Git bundle, checked out by hash; the validation container mounts only
`<campaign>/validate`. Allow only the new implementation, the hash-matched schema and the reviewed
knowledge patch. Fetch main and record it as the validation base; merge the frozen commit into it there.
A conflict, or a schema file that differs byte for byte from the candidate after the merge, means no PR.

Run `aklr export check`, a targeted `aklr validate --base <validation base> --stages audit,bench`, and the
stress gate with a driver made in the validator: `python harness/scripts/forge_task prepare` from the
validator clone into `<campaign>/validate`, the frozen `kernel.py` placed in that workspace, then its
`driver.py --stress`. Never run the worker's driver. Read every target Point with
`harness/skills/read-validation/SKILL.md`. A re-optimization's Points are judged against the incumbent
(`graph/incumbent`), not the baseline: deliver it only when at least one Point's lower bound against the
incumbent reaches 1.05, and list those Points in the PR body; the new file goes beside the incumbent,
which keeps serving the other Points (dispatch serves the faster file at each Point). Pull the reports to this host at once and record their
SHA-256 in the ledger; delivery uses only that copy. Any failure means no PR and no automatic repair.

Before cleanup, while the GPU is still held, fetch main and run freshness from the validation base. If it
is stale, validate again on the same GPU against the new main, as a new operation within the campaign's
deadline; when those run out, hand the candidate to the owner.
