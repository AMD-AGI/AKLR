---
name: report-progress
description: Write a short progress report on long-running AKLR work (what changed, key numbers against their baseline, what is still running, what the user must decide), and start a `/loop` timer so an unattended run reports every N minutes (default 15). Use when another AKLR skill asks for a watch or 15-minute report, when the user asks for status, progress or a handoff, or before ending a session that leaves AKLR work, nodes or PRs in flight.
---

# AKLR progress report

Skip narrative. The calling skill supplies units, gates, done-predicate and extra
fields (extra fields go in `Key numbers`).

## When to report

- **Periodic:** on the `/loop` timer (default 15 minutes), and right away when the
  status changes.
- **Handoff:** before a session ends with anything still running or held.
- **Final:** when the work is done or stopped. `Running` is `none` or lists what
  could not be cleaned up; cancel the loop with `CronDelete`.

## Start the timer

You do not wake on your own, so before handing work to the background start a
timer (interval from the user or caller, else 15m):

```
/loop 15m /report-progress <task> state=<run state path> reports=<reports file>
```

Each firing receives only that line, so put every path the report needs in it.
List the job id (`CronList`) under `Running`. Jobs expire after 7 days; restart
longer work before then. If `/loop` is unavailable, say no timer is running.

## Gather with a subagent

Before each report, run the `something-important` agent and wait for it. Give it the
goal, the artifact paths and the reports file. If that agent is not available, give
a subagent [its prompt](../../agents/something-important.md) instead.

Write the report from what it returns, not from your own memory of the run, and
keep its verdicts.

## Format

```
**<task> — <status> — #<n>, <time>**

**Goal:** <done-predicate verbatim>   [CHANGED: <old> -> <new>, approved by <who>]

**Progress:** <one sentence: distance to the goal, in the goal's own terms>

**Changed since #<n-1>:**

- <change> — <before -> after> — <verdict>

**Key numbers** (@ <repo/branch sha | artifact hash>):

- <metric>: <now> vs <baseline> (<noise or tolerance>) — <verdict>

**Unchecked:** <checks not run, skipped, pending or stale, or "none">

**Running:** <what, where, alive at <time>, job id> or "none"

**Needs you:** <decision or unblock, or "none">

**Next:** <next action and when>
```

Keep the blank lines: the reply is rendered as Markdown, which joins lines with no
blank line between them into one paragraph, so without them every field runs
into one line.

The reader should get the point at a glance, so the whole report fits in about
12 lines of text, not counting blank lines:

- Each bullet is one short line: the fact and its verdict, no explanation. At most
  3 bullets under `Changed since` and 3 under `Key numbers`.
- Nothing outside the template: no extra sections for deviations, risks or
  housekeeping. A risk the user must act on is one line in `Needs you`; one they
  should only know about goes in `Unchecked`; the rest goes in a file you link.
- Link paths, URLs and shas instead of pasting logs or reasoning.

There are only two forms. Use the template above whenever anything changed. If
nothing changed, send the short form so a quiet run does not look dead:

```
**<task> — <status> — #<n>, <time>**: no change

**Running:** <what, where, alive at <time>>

**Next:** <next action and when>
```

The header is bold in every report, the first and the fortieth alike, and each
field sits on its own line with a blank line before it, in the short form too.

## Rules

- **Status:** `on track`, `at risk` (goal looks unreachable on the current path),
  `blocked` (needs what is in `Needs you`), `done` (goal evidenced), `stopped`
  (ended without the goal; say why in `Progress`).
- Loosening the goal without the user's approval is a gap to report, not a change.
- No percent complete unless the units can be counted.

## Where it goes

Send the report as your reply and append it to the reports file if there is one.
The report is the whole reply: no sentence before or after it about the subagent,
the tools or what you are about to do.
Write in the user's language; keep identifiers, status words and verdicts in
English.
