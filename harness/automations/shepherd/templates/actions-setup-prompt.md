You are setting up shepherd in GitHub Actions, on the GPU host itself, before the round that an allowed
user's `shepherd` issue starts. The prompt that sent you here names a JSON file: its `issue` is that issue
as the user labeled it, and `issue.body` is its text. What it says about the configuration (budget and its authorization, GPUs, model, caps) is the
owner's choice and the record of that authorization; what it asks to optimize is the round's, not yours.

Read and follow `harness/automations/shepherd/skills/setup-shepherd/SKILL.md`, steps 1 to 5, for the
GitHub Actions host. `.shepherd/configuration.yaml` is a link to the host's configuration: write through
it, and keep each value of an existing file that the text does not change.

Nobody answers questions here. Take a choice from the text or from a found value the skill names; leave
every other paid-run choice `null` and the mode `observe`. Create no automation, start no campaign or
round, and never reinitialize an existing state directory. Install nothing except PyYAML for the user's
`python3`. Never use `remote-gpu`.

When the configuration and state already stand and the text changes nothing, check the host and stop.
Otherwise end by commenting on the issue (`gh issue comment`): what you changed, each state the skill
reports, each failed host check and each choice still `null`. Never print a secret.
