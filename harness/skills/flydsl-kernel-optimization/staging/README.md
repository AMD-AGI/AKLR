# staging/

Candidate pages waiting for review, one `<slug>.md` per page, written from a template in `../templates/`.
A reviewer's notes on a page go to `<slug>.review.md` next to it.

A page leaves through `python tools/kb_approve.py staging/<slug>.md --to <folder>` once it passes
`tools/kb_lint.py` and a person approves it; `../CONTRIBUTING.md` § How a page gets in has the steps.
Agents looking for knowledge do not read this folder: nothing here has been checked yet.
