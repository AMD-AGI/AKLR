# shepherd

shepherd gives you one automation that optimizes AKLR kernels unattended. Each round observes GitHub
and the GPU jobs, advances each KernelForge campaign as far as confirmed results allow, admits new
kernel-less schemas into free GPUs, validates the kernel, opens a draft PR and follows it until a maintainer
merges or closes it.

The files in this directory are dormant setup and automation sources. They do not appear as slash skills.
The round reuses the optimization, GPU, validation, PR and reporting skills in `harness/skills/`.

## Set it up

1. Point an agent at [`setup-shepherd`](./skills/setup-shepherd/SKILL.md) in the AKLR checkout, on the
   base branch, that will run rounds. It already holds this pack; nothing is copied.
2. The Cursor automation runs each round itself, so its run shows every step. It reads the shared
   skills by path from that checkout, not from an installed plugin whose cached revision can lag.
3. Keep user-owned configuration outside the pack, in `.shepherd/configuration.yaml` (ignored by
   Git). Adapt [`configuration.example.yaml`](./templates/configuration.example.yaml).
4. Create the automation, or update the existing one, with the fields setup hands over and a prompt
   adapted from [`automation-prompt.example.md`](./templates/automation-prompt.example.md) (it begins
   the round itself, so a skipped trigger costs two commands). Enable it with
   `mode: observe`, run the setup tests, then switch to `run` on probation (one campaign's budget,
   `max_parallel_campaigns: 1`, one PR) until a real campaign reaches a draft PR, and raise the budget
   and parallelism after.

## Run it from GitHub issues

`.github/workflows/shepherd.yaml` is a second way to run the same rounds, beside the Cursor automation:
on the `aklr-gpu` runner (the `ci:gpu-direct` machine), with that host's own GPUs as a
[local pool](./skills/optimize-and-deliver-kernels/references/local.md) and no `remote-gpu`.

- Once, and to change the configuration later, an allowed user labels an issue `shepherd-setup`: an agent
  on the runner follows [`setup-shepherd`](./skills/setup-shepherd/SKILL.md#github-actions-host) with the
  issue's text as the owner's choices (budget and its authorization, GPUs, model), and comments the result.
- An allowed user opens an issue with the shepherd form, or labels one `shepherd`: its text is frozen
  as a [request](./skills/optimize-and-deliver-kernels/references/requests.md) and tells the rounds
  what to optimize. Later edits, by anyone, change nothing.
- Each PR says `Refs #<issue>`; one comment on the issue shows every campaign, and becomes the final
  report when shepherd closes the issue. An allowed user closing it first stops the request.
- Allowed users are the logins in the repository variable `SHEPHERD_ALLOWED_USERS` (a JSON list). Events
  from anyone else start no job.

Nobody logs in to the runner for any of this.
