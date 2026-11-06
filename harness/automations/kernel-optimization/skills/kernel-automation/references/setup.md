# Set up the AKLR kernel automation

Read `harness/automations/kernel-optimization/FOR_AGENTS.md` and its configuration template from the
repository root. Keep user configuration at `.cursor/kernel-optimizer/configuration.yaml`, outside the
managed pack and excluded from Git. Preserve existing values and user edits. This setup does not start a paid campaign.
Keep deployment paths, internal hostnames, account identities and credentials out of shared skills and templates.

## Prepare configuration

Fill the example with the owner. Infer the repository and branch from the target checkout; get the model,
node and model budgets, execution environment, and publication authority from existing user instructions
or configuration. Do not guess missing paid-run choices.
Resolve configured model display names to the actual supported identifiers in Cursor and the campaign
provider. Verify `reasoning_effort` in both environments and record the effective setting; do not invent
a CLI flag, silently drop the requested effort, or substitute another model when it is unsupported. An approved configuration supplies the budget
and model that forge-loop-campaign would otherwise ask for on every run.

Use `observe` until readiness is verified. `run` admits new work; `drain` finishes existing work without
new admissions. Missing credentials or required runtime values block the actions that need them, not
unrelated read-only inspection. Secrets belong in the execution environment's credential mechanism.

Check schema version 1, strategy types `manual` / `kernel-less`, positive batch/PR/cooldown limits,
`max_active_campaigns: 1`, one supported backend, and positive campaign limits within cumulative limits.
Total node time must cover setup and final validation as well as loop time. Model budget covers all inner
agent calls; Cursor and GPU CI expenses have separate authorizations. Do not implement hard caps by
estimating token use from logs. Verify the configured model provider's hard limit and scheduler time limit.

## Verify the actual execution environment

Run checks in the environment each Automation invocation will use, not merely in this setup session:

- It reads the committed operational skill and locally provisioned config by repository-relative path. Confirm all four
  responsibility skills linked by the entrypoint exist, along with their operational dependencies
  (campaign, remote-gpu, validation, PR, reporting and the chosen backend knowledge skill).
- It can authenticate to the configured GitHub repository and read main, PRs, CI and reviews. With publication
  enabled it can push a new branch and create draft PRs. CI label authority must be configured separately.
- It reaches the declared control host and the **same persistent state directory** on every invocation.
  Use one dedicated identity and a local Linux filesystem with working `flock` and atomic rename.
  Do not use temporary storage, ephemeral agent checkouts, independent copies, or an unverified network filesystem.
- The existing remote-gpu CLI can read the configured gfx950 cluster. Follow remote-gpu for any setup changes.
  `forge_task` and KernelForge help commands resolve in the pinned runtime; no node allocation is needed just
  to prepare the automation pack. If runtime capability has not been tested, leave paid mode disabled.
- Establish a separate trusted validation checkout/environment: the optimization worker cannot edit its
  validator, overwrite final evidence or access GitHub/control-state credentials. Pin image/tool versions.
- Record the successful checks and the budget authorization at `execution.readiness_record`. Validate that
  its config, model channel and image still match before turning on paid mode.

Read `harness/automations/kernel-optimization/skills/kernel-automation-knowledge/references/state.md`. Initialize a new state directory once with the
state helper. An existing or missing-after-use state directory requires investigation, not reinitialization.
A new directory must not hide campaigns already running under this automation's identity.

## Prepare one Cursor Automation

Use `harness/automations/kernel-optimization/templates/automation-prompt.md` as the thin prompt. The operational
skill stays at `harness/automations/kernel-optimization/skills/kernel-automation/SKILL.md`; do not copy its body into the live prompt or depend
on slash-skill discovery. AKLR already exposes its shared skills through the repository's skill symlinks.
No pstack installation is required.

Verify the pack, skills, helper and generic configuration template are committed and available on the actual
branch the Automation will check out. Provision the ignored user configuration through the execution
environment so each invocation can read it at the expected repository-relative path. Keep actual machine
names, deployment paths, account details and authorization records out of commits. Prepare edits first;
do not commit unless requested.

For first creation in Cursor, use its built-in `/automate` when available and follow that interface's
review/editor flow. Configure one automation named by the config, repository, supported Cursor model,
15-minute schedule by default, execution access, and the thin prompt. Supported GitHub events may trigger
this same workflow; schedule polling remains the fallback. Do not depend on ordinary bot comments as triggers.
For an existing automation, update its editor fields rather than creating a duplicate. Do not invent a
Cursor API or replace it with a timer in a different product. If this environment cannot configure Cursor,
hand over the completed prompt, config path, readiness results and remaining editor fields explicitly.

## Verify before paid operation

Run one `observe` invocation: it queries GitHub, inspects any existing runs and lists candidates, produces a
report, and makes no GPU/model/GitHub mutations. Do not require unrelated missing model values for this check.
Run the state helper's CPU tests, then prove two concurrent invocations on the configured host cannot both
acquire the round lock. Verify a second fresh invocation reads the state saved by the first.

After configuration and environment readiness pass, enable the existing automation in `run` mode within the
owner's recorded authority. First real acceptance is one bounded campaign through verified cleanup and draft
PR, followed by a fresh invocation observing that PR. Do not claim this acceptance from CPU tests or a prompt
preview. Report whether the automation is prepared, saved, enabled, or runtime-verified; distinguish them.
