# Tests

This directory is reserved for tests of the AKLR skills, the scripts in
`harness/scripts/`, automation pack helpers, and the end-to-end kernel lifecycle.

- `forge_task/`: the forge_task tool against a AKLR stand-in, CPU only:
  `python -m pytest harness/tests/forge_task`.
- `remote_gpu/`: the remote-gpu CLI against a fake two-node cluster on this machine, no scheduler,
  ssh or GPU: `python -m pytest harness/tests/remote_gpu`.

- `kernel_automation/`: persistent round ownership, atomic state snapshots and crash recovery boundaries,
  CPU only: `python -m pytest harness/tests/kernel_automation`.

No CI workflow runs these tests yet.
