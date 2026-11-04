# Adding to this skill

For anyone, person or agent, who adds or edits a page. The rules exist so that an agent can find the one page that
answers its question, trust what the page says for the conditions it states, and be told when a FlyDSL upgrade may
have made it stale. `tools/kb_lint.py` enforces what can be checked mechanically; review does the rest.

## Page kinds

One page answers one kind of question. Start a page from its template in `templates/`.

| Kind | Answers | Template | Folder |
|---|---|---|---|
| `technique` | How to do one thing (one `lever`), when it pays, when it does not, how to see it in the binary | `templates/technique.md` | the op's package (`gemm/`, `moe/`) |
| `pattern` | What a symptom in a trace or counter means, and which techniques to try | `templates/pattern.md` | `patterns/` |
| `op` | What an op family computes, where its time goes per regime, which kernels to start from | `templates/op.md` | the op's package |
| `case` | How a delivered kernel got its speed, under which conditions, and how to reproduce it | `templates/case.md` | `cases/` |
| `pitfall` | A silent wrong result, a FlyDSL version trap or a measurement artifact | `templates/pitfall.md` | `pitfalls/` |
| `language-note` | What FlyDSL v0.3.2 does, read from its source: limits, compile errors, version differences | `templates/language.md` | `language/` |
| `topic` | One area of an op (a card of numbered sections, one lever or decision each), such as the GEMM main loop | `templates/topic.md` | the op's package |
| `method` | How to work: measure, check correctness, attribute time, tune, debug | `templates/method.md` | the package it serves |

Pages written before these rules — `kind: language`, `index` and `reference`, and the copies of FlyDSL's own
skills under `optimize/`, `profile/` and `bottleneck/` — are checked with warnings only. A card of numbered
sections becomes a `topic` or `method` page where it is: its path and section numbers stay, because other pages
cite them as `<card>.md §<n>`.

## Front matter

The fields each kind needs are in its template; lint lists the ones missing. Front matter is YAML: quote a value
that contains `: `. `flydsl` is the version the page was checked against (`0.3.2`), `gens` the architectures
(`[gfx950]`), `regimes` the M ranges or regimes it covers.

## Evidence

- **`basis`** says how the page knows what it says:
  - `source-semantics`: read from FlyDSL, LLVM or ISA source;
  - `measured N=<k>`: measured `k` times on a GPU, optionally followed by the ruler;
  - `single-observation`: seen once. Such a page does not say "always" or "never".
- **A measured number carries its conditions.** In the text, mark it *(field, E1)*, the only accepted form (not
  *(field)* alone, `*field:*` or *(field: …)*). Under `## Evidence` at the end of the page, give one line per mark:

  ```
  - E1 [2026-09, gfx950, FlyDSL ?, N=?, run_perftest cold-L2] 34.1 → 5.0 µs with split-K 12 (1×32×6144; local:/workspace/code/gpt_oss_docker/sync/flydsl_optimizer/flydsl_campaigns/20260904_151953/goal.md:197)
  - S1 kernels/preshuffle_gemm.py:210 the B tile is loaded once per K step
  ```

  `E` lines are measurements: month, architecture, toolchain, repeat count, ruler, then the claim, the workloads
  and a reference someone can open. `N=?` says the repeat count is unknown; such a line, like `N=1`, supports no
  "always" or "never" in the paragraph that cites it. The reference is durable (a commit, `#<PR>`, a CI run or a
  URL) or `local:<absolute path>` for a file on one machine, which may not outlive it. `S` lines cite source as
  `file:line`. The text stays readable and every number stays checkable.
- **A number measured on another kind of kernel says so in the sentence** ("on a grouped GEMM", "on an attention
  kernel"), not only in its evidence line: readers do not read the evidence.
- **A number worked out rather than measured** says "estimated" and gets no `E` line.
- **Speedups in AKLR are per Point** against the faster of the kernel AKLR serves at that Point and the op's
  baseline, the same yardstick dispatch uses; never a mean over Points.

## What not to write

- **No process log.** Headings such as Tried, What we tried, Track record, History, Timeline, Changelog or
  Attempts fail lint. Write the current best practice; an attempt that failed becomes a counter-signal or a risk,
  stated as the condition under which it fails.
- **No copy of an external rule.** AKLR's audit, contracts and serving threshold are linked to the section of
  `aklr/ops/README.md` that states them, not restated: a restated rule goes stale when the README changes.
- **No second page for the same thing.** A technique's `lever` is unique (`tools/kb_approve.py` refuses a
  duplicate); search the skill for the lever, the op and the symptom before writing, and extend the page you find.
- **No unchecked claim.** Every statement is checked against source, the ISA or a GPU run before it goes in.

## How a page gets in

1. Copy the template to `staging/<slug>.md` and fill it in. Write each link to another page as `../<folder>/<page>.md`,
   even to a page of the folder it will go to: that path resolves from `staging/` and from the destination.
2. `python tools/kb_lint.py staging/<slug>.md` reports no error.
3. Review: check each claim against its source or a run, look for an existing page that covers it, set `basis`.
   Review notes go to `staging/<slug>.review.md`.
4. Approval by a person: `python tools/kb_approve.py staging/<slug>.md --to <folder>` lints the page again, moves
   it, deletes its review notes and rebuilds the indexes. Add the files with the `git add` command it prints.
5. Link the page from where an agent looks for it: a row of `SKILL.md`'s task or symptom table, or the package's
   `INDEX.md`. `patterns/`, `cases/`, `language/` and `pitfalls/` need no link by hand: their `INDEX.md` is
   generated.

An edit to an existing page skips staging, but passes lint and review the same way, and a new number in it gets an
evidence line. A knowledge patch proposed by the optimization loop's distill step (Hyperloom `AKLR`
`forge-distill`) is reviewed like any other edit; a new page in it goes to `staging/` first. It is committed with
the kernel it came from, in that kernel's pull request (`forge-loop-campaign` skill, step 7).

## Upkeep

- **FlyDSL upgrade**: `python tools/kb_lint.py --stale <new version>` lists the pages checked against another
  version. Re-read each `language-note` against the new source; a `source-semantics` page is possibly stale, not
  wrong. Set `flydsl` once the page is checked.
- **Gaps**: the `KB: no-match <question>` lines that optimization rounds leave (`SKILL.md` § How to look things
  up; `tools/kb_gaps.py` collects them) are the next pages to write. Delivered kernels, fixes and known traps
  without a `cases/` or `pitfalls/` page are the others: each month and after a batch of kernels merges,
  `tools/kb_reconcile.py` lists the FlyDSL kernels on main that no case names.
- **Size**: `SKILL.md` stays at most 150 lines, because the optimization loop puts it into every round's prompt;
  a page over 600 lines is split.

## Tools

| Command | Does |
|---|---|
| `python tools/kb_lint.py [page ...]` | checks front matter, sections, marks and evidence lines, forbidden headings, links, `§<n>` references and sizes; exit 1 on an error; `--repo-root` for a copy of the skill |
| `python tools/kb_lint.py --stale <version>` | lists pages and evidence lines checked against another FlyDSL version |
| `python tools/kb_index.py [--check]` | rebuilds the generated `INDEX.md` of `patterns/`, `cases/`, `language/` and `pitfalls/` |
| `python tools/kb_approve.py staging/<slug>.md [more.md ...] --to <folder>` | moves reviewed pages in; pages that link each other go in one call |
| `python tools/kb_quote_check.py <sidecar.tsv>` | checks that each quote of an evidence sidecar is at the source line it names |
| `python tools/kb_numbers_guard.py <page> [--rev REV]` | lists the numbers an edit changed outside `## Evidence` and code |
| `python tools/kb_reanchor.py <page> [--dry-run]` | moves the `S<n> path:line` citations of a page whose lines moved |
| `python tools/kb_gaps.py <campaign workspace> ...` | what forge-loop rounds opened in this skill and the questions they found no page for |
| `python tools/kb_reconcile.py [--rev REV]` | the FlyDSL kernels at a revision that no case page names, and cases whose files are gone |
| `python tools/isa_diff.py <a.s> <b.s> [--stream]` | compares two `NN_final_isa.s` dumps: resources and opcode counts, registers and labels normalized |
| `python tools/budget.py --arch-vgpr N --agpr N --waves W [--lds B]` or `--isa <s>` | gfx950 register and LDS budget, occupancy and the register ceiling of each reachable occupancy |

They need Python 3.10 or later; PyYAML is used when installed.
