---
name: bad-cases
description: A list of mistakes agents have made in AKLR, with what to do instead and how to spot each one in a code change. Use when reviewing an AKLR change or opening its pull request. Add a case when a new mistake is found.
---

# AKLR Bad Cases

Mistakes agents have made in AKLR. Each case says what went wrong, why, what to do instead, and how to spot it.

## Use

When you review a change or open a pull request, go through the list below. If a case applies to the change, read
it and answer its Check questions for the change.

| Rule | Case |
|---|---|
| Do not change a contract to fit a kernel | [cases/contract_changed_to_fit_kernel.md](cases/contract_changed_to_fit_kernel.md) |
| Do not hide a slow fallback inside a kernel | [cases/slow_fallback_inside_kernel.md](cases/slow_fallback_inside_kernel.md) |

## Add a case

Add one file under `cases/` with these sections, a few lines each:

- **Rule**: what not to do, and what to do instead.
- **Bad case**: what happened, and where (file, PR).
- **Why it is wrong**: what it breaks.
- **Do instead**: the right fix.
- **Check**: questions that spot it in a code change.

Then add a row to the table above. Only add mistakes that really happened. Kernel performance tips and
measurement traps go to `kb` instead.
