# Adding to this skill

For anyone, person or agent, who adds or edits a page of this skill, its language folders (`flydsl/`, `triton/`,
`hip/`) included. The rules exist so that an agent can find the one page
that answers its question, trust what the page says for the conditions it states, and be told when a FlyDSL upgrade
may have made it stale. `maintenance/kb_lint.py` enforces what can be checked mechanically; review does the rest.

## Folders

- Packages, one per op, each starting at its `INDEX.md`: `gemm/` dense GEMM · `moe/` fused MoE and its expert
  grouped GEMM · `attention/` attention
- `communication/` the collectives between GPUs, before AKLR has a communication op (EP against TP, collectives in
  traces); its pages are kept as an op package's are
- `patterns/` symptom pages · `cases/` delivered kernels · `pitfalls/` traps
- `profile.md` trace capture and analysis, and perf-regression bisection: links to FlyDSL's own skills, with what
  this repository adds
- `hardware/<arch>/` an architecture's ISA guide, and the guides to looking things up in it (`hardware/gfx950/`:
  CDNA4; `hardware/gfx942/`: CDNA3)
- `maintenance/` the scripts that keep this knowledge base consistent (lint, indexes, gaps, citations) · `analysis/`
  scripts an agent runs on its own kernel (trace hotspots, L2/HBM counters, ISA diffs, register and LDS budget);
  § Scripts lists both
- `flydsl/`, `triton/`, `hip/` a language's own pages, each starting at its `SKILL.md`. In `flydsl/`: `language/`
  what FlyDSL itself does (limits, compile errors, version differences), `guides.md` FlyDSL's own skills linked at
  `v0.3.2` with this repository's corrections, `debug-flydsl-kernel.md` debugging, `kernels/` aiter's FlyDSL
  kernels, their knobs and how to write a new one, `<op>.md` (`gemm.md`, `moe.md`, `attention.md`) the FlyDSL side
  of an op package's cards. A card states the mechanism and its numbers; how to write it in FlyDSL goes to that
  page, under a heading named after the card's section (`mainloop §8`), and the card's section ends with a pointer
  to it. In `triton/` (front matter `triton: 3.8.0` in place of `flydsl`): `authoring.md`, `debugging.md`,
  `tuning.md`, `numerics.md` for any Triton or Gluon kernel, `gluon/` Gluon's layouts, data movement, MFMA and inline
  assembly, `kernels/kernel_library.md` AKLR's Triton kernels by family, `<op>.md` (`gemm.md` with `gemm/`, `moe.md`,
  `attention.md`, `gdn.md`) the Triton side of an op

## Page kinds

One page answers one kind of question. A page of each kind has the front matter fields and the sections below;
`maintenance/kb_lint.py` lists the ones missing.

| Kind | Answers | Front matter; sections | Folder |
|---|---|---|---|
| `technique` | How to do one thing (one `lever`), when it pays, when it does not, how to see it in the binary | title, lever, ops, dtypes, regimes, gens, flydsl, basis, sources; What it does, How to write it, When it applies, Counter-signals, How to confirm it took effect, Cost and risk, Evidence | the op's package (`gemm/`, `moe/`, `attention/`); one that several ops use goes to the package it was measured in, and the others link it |
| `pattern` | What a symptom in a trace or counter means, and which techniques to try | title, signals, ops, gens, flydsl, basis; Symptom, Likely causes, Rule out first, Candidate techniques, Evidence | `patterns/` |
| `op` | What an op family computes, where its time goes per regime, which kernels to start from | title, ops, dtypes, gens, flydsl, basis; What it computes, Bottleneck by regime, Techniques that apply, Kernels to start from, In AKLR | the op's package; a new op gets its folder with its first design page (MLA and DSA: `attention/`) |
| `case` | How a delivered kernel got its speed, under which conditions, and how to reproduce it | title, ops, dtypes, regimes, contract, gens, flydsl, basis, measured, sources; Applies when, Counter-signals, Mechanism, Result, Reproduce, Rollback | `cases/`, named `<op>_<gist>.md` |
| `pitfall` | A silent wrong result, a FlyDSL version trap or a measurement artifact | title, scope, gens, flydsl, basis; Symptom, Root cause, How to detect, Fix or workaround, Scope | `pitfalls/` |
| `language-note` | What FlyDSL v0.3.2 does, read from its source: limits, compile errors, version differences | title, gens, flydsl, basis, sources | `flydsl/language/`; a correction of, or an addition to, one of FlyDSL's own skills is a bullet under that skill's section of `flydsl/guides.md` |
| `topic` | One area of an op (a card of numbered sections, one lever or decision each), such as the GEMM main loop | title, ops, dtypes, regimes, gens, flydsl; Route here when | the op's package |
| `method` | How to work: measure, check correctness, attribute time, tune, debug | title, ops, gens, flydsl; Route here when | the package it serves |

Pages written before these rules — `kind: language`, `index` and `reference` — are checked with warnings only. A
card of numbered sections becomes a `topic` or `method` page where it is: its path and section numbers stay,
because other pages cite them as `<card>.md §<n>`.

- **Names**: no type or language prefix; a language's own pages go in its folder (`flydsl/`, `triton/`, `hip/`). A
  page's path and its section numbers change only when the page is rewritten, and that pull request fixes every link
  to it.
- **Hardware**: `hardware/<arch>/` holds only an architecture's ISA guide and the guides to looking things up in
  it. What one architecture changes for an op goes to the op's package, with `gens: [<arch>]` (gfx942's GEMM
  notes: `gemm/gfx942.md`).
- **FlyDSL's own skills are linked, never copied**, pinned to `v0.3.2`: `https://raw.githubusercontent.com/ROCm/FlyDSL/v0.3.2/.claude/skills/<skill>/SKILL.md`. What this
  repository adds to or corrects in one goes to `flydsl/guides.md`, or to `profile.md` for the trace and
  bisection skills. A card links that page (`flydsl/guides.md` § <skill>, `profile.md` § <skill>), not the
  upstream file, so a reader meets the corrections first; a line of a skill is cited as
  `ROCm/FlyDSL@dd8373349:.claude/skills/<skill>/SKILL.md:<line>`. The exception is the two trace-analysis scripts
  in `analysis/`, this repository's edited copies (`profile.md` § Scripts).

## Front matter

Front matter is YAML: quote a value that contains `: `. `flydsl` is the version the page was checked against
(`0.3.2`), `gens` the architectures (`[gfx950]`; `[gfx942]` on a page about gfx942 only, such as
`gemm/gfx942.md`), `regimes` the M ranges, shapes or mechanisms it covers, never a phase (see below).

## Evidence

- **`basis`** says how the page knows what it says:
  - `source-semantics`: read from FlyDSL, LLVM or ISA source;
  - `measured N=<k>`: measured `k` times on a GPU, optionally followed by the ruler;
  - `single-observation`: seen once. Such a page does not say "always" or "never".
- **A measured number carries its conditions.** In the text, mark it *(field, E1)*, the only accepted form (not
  *(field)* alone, `*field:*` or *(field: …)*). Under `## Evidence` at the end of the page, give one line per mark:

  ```
  - E1 [2026-09, gfx950, FlyDSL ?, N=?, run_perftest cold-L2] 34.1 → 5.0 µs with split-K 12 (1×32×6144)
  - S1 kernels/preshuffle_gemm.py:210 the B tile is loaded once per K step
  ```

  `E` lines are measurements: month, architecture, toolchain, repeat count, ruler, then the claim and the
  workloads. `N=?` says the repeat count is unknown; such a line, like `N=1`, supports no "always" or "never" in
  the paragraph that cites it. A run with a public record cites it (a commit, `#<PR>`, a CI run or a URL); a run
  without one gives the result alone. A number with no run record says `unverified` in its line. This repository
  is public: no page names a file of one machine (`local:`
  references, workspace or shared-disk paths), and the linter rejects them. `S` lines cite public source as
  `file:line`; source with no public copy is stated without one. The text stays readable.
- **A number measured on another kind of kernel, or in another language, says so in the sentence** ("on a grouped
  GEMM", "on an attention kernel", "in Triton"), not only in its evidence line: readers do not read the evidence.
  The op packages' numbers were measured on FlyDSL kernels unless a sentence says otherwise, and each package's
  `INDEX.md` says so once.
- **A number worked out rather than measured** says "estimated" and gets no `E` line.
- **Speedups in AKLR are per Point** against the faster of the kernel AKLR serves at that Point and the op's
  baseline, the same yardstick dispatch uses; never a mean over Points.

## What not to write

- **No process log.** Headings such as Tried, What we tried, Track record, History, Timeline, Changelog or
  Attempts fail lint. Write the current best practice; an attempt that failed becomes a counter-signal or a risk,
  stated as the condition under which it fails.
- **No copy of an external rule.** AKLR's audit, contracts and serving threshold are linked to the section of
  `aklr/ops/README.md` that states them, not restated: a restated rule goes stale when the README changes.
- **No second page for the same thing.** A technique's `lever` is unique (`maintenance/kb_lint.py` reports a
  duplicate); search the skill for the lever, the op and the symptom before writing, and extend the page you find.
- **No training-only or inference-only scope.** The same GEMM, attention and MoE kernels run in training and
  inference, so a page says when it applies by mechanism and shape: the M range, rows per group, whether K varies
  per group, query rows against KV length, causal or sparse access. Titles, file names and `regimes` carry no phase
  word (training, inference, forward, backward, fwd, bwd, dgrad, wgrad, prefill, decode; `maintenance/kb_lint.py` reports
  them), and an index or "Route here when" line routes by the problem, not by the phase. A measurement may still say
  what it was measured on.
- **No unchecked claim.** Every statement is checked against source, the ISA or a GPU run before it goes in.

## How a page gets in

1. Write the page in its folder (the table above), with the front matter and sections of its kind.
2. `python maintenance/kb_lint.py <page>` reports no error. Link the page from where an agent looks for it: a row of
   `SKILL.md`'s task or symptom table, or the package's `INDEX.md`. `patterns/`, `cases/` and `pitfalls/` need no
   link by hand: `python maintenance/kb_index.py` rebuilds their generated `INDEX.md`.
3. Review, in the pull request that adds it: each claim against its source or a run, no existing page that covers
   it, `basis` set.

An edit to an existing page passes lint and review the same way, and a new number in it gets an evidence line. A
knowledge patch proposed by the optimization loop's distill step (Hyperloom `AKLR` `forge-distill`) is reviewed
like any other edit; one run covers this skill, its language folders included
(`harness/scripts/forge_task/README.md`). It is committed with the kernel it came from, in that kernel's pull request
(`forge-loop-campaign` skill, step 7).

## Upkeep

- **FlyDSL upgrade**: `python maintenance/kb_lint.py --stale <new version>` lists the pages checked against another
  version. Re-read each `language-note`
  against the new source; a `source-semantics` page is possibly stale, not wrong. Set `flydsl` once the page is
  checked. Move the links pinned to `v0.3.2` and re-check `flydsl/guides.md` against the new version.
- **Gaps**: the `KB: no-match <question>` lines that optimization rounds leave (`SKILL.md` § How to look things
  up; `maintenance/kb_gaps.py` collects them) are the next pages to write. Delivered kernels, fixes and known traps
  without a `cases/` or `pitfalls/` page are the others: each month and after a batch of kernels merges,
  `maintenance/kb_reconcile.py` lists the FlyDSL kernels on main that no case names.
- **Size**: each `SKILL.md`, this skill's and a language folder's, stays at most 150 lines and 12,000 bytes,
  because the optimization loop puts both into every round's prompt; a page over 600 lines is split.

## Scripts

| Command | Does |
|---|---|
| `python maintenance/kb_lint.py [page ...]` | checks front matter, sections, marks and evidence lines, forbidden headings, links, `§<n>` references, architecture names, duplicate levers and sizes; exit 1 on an error; `--repo-root` for a copy of the skill |
| `python maintenance/kb_lint.py --stale <version>` | lists pages and evidence lines checked against another FlyDSL version |
| `python maintenance/kb_index.py [--check]` | rebuilds the generated `INDEX.md` of `patterns/`, `cases/` and `pitfalls/` |
| `python maintenance/kb_quote_check.py <sidecar.tsv>` | checks that each quote of an evidence sidecar is at the source line it names |
| `python maintenance/kb_numbers_guard.py <page> [--rev REV]` | lists the numbers an edit changed outside `## Evidence` and code |
| `python maintenance/kb_reanchor.py <page> [--dry-run]` | moves the `S<n> path:line` citations of a page whose lines moved |
| `python maintenance/kb_gaps.py <campaign workspace> ...` | what forge-loop rounds opened in this skill and the questions they found no page for |
| `python maintenance/kb_reconcile.py [--rev REV]` | the FlyDSL kernels at a revision that no case page names, and cases whose files are gone |
| `python analysis/isa_diff.py <a.s> <b.s> [--stream]` | compares two `NN_final_isa.s` dumps: resources and opcode counts, registers and labels normalized |
| `python analysis/budget.py --arch-vgpr N --agpr N --waves W [--lds B]` or `--isa <s>` | gfx950 register and LDS budget, occupancy and the register ceiling of each reachable occupancy |
| `python analysis/hotspot_analyzer.py <dispatch_dir> --topk 15 --mode both` | the hot instructions of an ATT trace (`profile.md`) |
| `python analysis/pmc_l2_analyzer.py <counters.csv>` | L2 and HBM efficiency from rocprofv3 PMC counters (`profile.md`) |

They need Python 3.10 or later; PyYAML is used when installed.
