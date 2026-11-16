# Tests

This directory is reserved for tests of the AKLR skills, the scripts in
`harness/scripts/`, and the end-to-end kernel lifecycle.

- `forge_task/`: the forge_task tool against a AKLR stand-in, CPU only:
  `python -m pytest harness/tests/forge_task`.
- `remote_gpu/`: the remote-gpu CLI against a fake two-node cluster on this machine, no scheduler,
  ssh or GPU: `python -m pytest harness/tests/remote_gpu`.

No CI workflow runs these tests yet.
