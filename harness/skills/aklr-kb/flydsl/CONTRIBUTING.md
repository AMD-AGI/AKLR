# Adding to this skill

The rules are the kernel wiki's: `harness/skills/aklr-kb/CONTRIBUTING.md` (from this folder,
`../CONTRIBUTING.md`) gives the page kinds, front matter, evidence and review. What goes here:

- What FlyDSL itself does, read from its source (limits, compile errors, version differences): a `language-note` in
  `language/`.
- A correction of, or an addition to, one of FlyDSL's own skills: a bullet under that skill's section of
  `guides.md`, naming the skill's own section. FlyDSL's skills are linked at `v0.3.2`, never copied.
- Debugging a FlyDSL kernel: `debug-flydsl-kernel.md`; aiter's FlyDSL kernels, their knobs and how to write a new
  one: `kernels/`.

What holds whatever the language (how a technique pays, measurement, cases) goes to the kernel wiki instead.
Lint this skill with `python ../tools/kb_lint.py --skill-root .` from this folder.
