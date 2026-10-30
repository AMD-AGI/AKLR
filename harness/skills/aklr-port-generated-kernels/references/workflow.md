# Generated Kernel Porting Workflow

1. **Capture the source.** Record the upstream repository, immutable revision,
   original paths, stated license or provenance, generator or agent context,
   supported GPUs, and claimed result. Stop and report unresolved provenance.
2. **Find the contract and its Points.** The kernel serves a contract in
   `aklr/ops/<op>/contract.py` at a schema name (`aklr/ops/<op>/impls/<language>/<name>/`),
   or needs a new op ([new-op.md](new-op.md)). List the committed kernels under
   that name and the Points they serve (stored `speedup` at least
   `AKLR_MIN_SPEEDUP`): those are the incumbents to beat, besides AITER.
3. **Triage on AKLR's Points before porting.** Time the source as it is on the
   contract's Points and inputs, not on the shapes it was tuned for. Read its
   environment gates and per-shape tables: a fast path enabled only for the
   source's own sizes falls back silently on AKLR's. Drop the Points it cannot
   win, and the port of code only they need.
4. **Port into one schema JSON.** Self-contained sources that pass the audit
   (SA0-SA9, SH0-SH5): generated code frozen, environment knobs made constants,
   helpers vendored, scratch from `torch.empty`, nothing kept across calls
   ([porting-notes.md](porting-notes.md)). Check the contract's `determinism`: a
   `bitwise` contract rejects accumulation in arrival order (atomics, atomic
   split-K), so reduce in a fixed order; `bounded` allows it.
5. **Verify per Point.** Run `python -m aklr export check`, then
   `python -m aklr validate --base origin/main --stages audit,bench <new schema>` on
   the target GPU, naming only the new file. Keep a Point when its lower bound is
   at least 1.03 against the baseline and 1.0 against a served kernel; read
   failures with `aklr-read-validation`. Confirm that the ported fast path, not a
   fallback, produced the result.
6. **Prepare the handoff.** Summarize provenance, the schema and its Points,
   per-Point bounds with the run's environment, the Points left out and why, and
   the source's claims that were not reproduced. Open the pull request with
   `aklr-open-pr`.

Do not publish or open a pull request unless the invoking workflow authorizes
that external change.
