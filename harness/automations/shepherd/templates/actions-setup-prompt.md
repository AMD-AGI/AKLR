You are setting up shepherd in GitHub Actions, on the GPU host itself, because an allowed user labeled
an issue `shepherd-setup`. The issue and its text as that user labeled it are at the end of this prompt:
the text holds the owner's choices and is the record of the budget authorization.

Read and follow `harness/automations/shepherd/skills/setup-shepherd/SKILL.md`, steps 1 to 5, for the
GitHub Actions host. `.shepherd/configuration.yaml` is a link to the host's configuration: write through
it, and keep each value of an existing file that the text does not change.

Nobody answers questions here. Take a choice from the text or from a found value the skill names; leave
every other paid-run choice `null` and the mode `observe`. Create no automation, start no campaign or
round, and never reinitialize an existing state directory. Install nothing except PyYAML for the user's
`python3`. Never use `remote-gpu`.

End by commenting on the issue (`gh issue comment`): each state the skill reports, each failed host
check, each choice still `null`, and the configuration. Never print a secret.

The issue:
