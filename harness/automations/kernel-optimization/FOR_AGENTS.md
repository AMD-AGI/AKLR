# Set up or edit this automation pack

This file gives an agent directions for configuring or maintaining the pack.
Read it when asked to set up the automation or change its files.

The workflow entry is
[skills/kernel-automation/SKILL.md](skills/kernel-automation/SKILL.md).
For setup or configuration changes, follow its
[setup guide](skills/kernel-automation/references/setup.md).
For a runtime round, follow the entry skill and its four responsibility skills.

Use [templates/automation-prompt.md](templates/automation-prompt.md) as the prompt
in Cursor. That prompt reads the entry skill on each trigger. Keep the workflow
rules in the skills rather than copying them into the prompt.

Keep the pack's source in this directory. Read its skills directly by repository
path and reuse shared kernel skills from `harness/skills/`.
Preserve `.cursor/kernel-optimizer/configuration.yaml` and other user changes;
use the configuration template only when creating a new configuration. Keep the actual configuration
local and ignored by Git. Shared skills and templates must not contain fixed absolute paths, internal
hostnames, account identities or credentials; use repository-relative references and configuration values.
Use the owner's recorded model, environment, budgets, and permissions.

Configure one Automation. Update the existing instance if it is already present.
Use committed repository paths, so each invocation can read the same files.
