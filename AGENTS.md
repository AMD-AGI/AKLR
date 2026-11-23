# Agent rules

## Protected code

Do not modify anything under `aklr/core/runtime/`.

If a task cannot be done without changing them, stop before editing and tell a code owner
(`.github/CODEOWNERS`): which files, what change, and why there is no way around it. Edit only after
a code owner approves. In a PR, say so in the description.

## Tests first

Change what code under `aklr/` does test-first: write the test, run it and see it fail (red), then
write the code. The `aklr:tdd` skill is the loop. Before writing a test, read `tests/AGENTS.md`.
