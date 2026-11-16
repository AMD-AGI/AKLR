# Crystal correctness qualification

The integration submits a pinned main/release commit or exact PR image for
candidate model accuracy checks. It supplements
the native audit/unit/bench gates. It does not replace kernel-reference tests or
establish that a changed kernel was selected instead of a fallback.

## Operator setup

- Merge the reviewed workflow onto the default branch before running it.
- Provide separate self-hosted labels `aklr-crystal-build` (disposable isolated
  builder) and `aklr-crystal-submit` (trusted control runner with service access).
  Never assign both labels to the same machine. PR code runs only in the builder.
- Protect environments `crystal-build` and `crystal-submit` with maintainer
  approval and restrict them to the default branch. Only the submit environment
  has `CRYSTAL_URL` and `CRYSTAL_AKLR_SUBMISSION_SECRET` secrets. The App key is
  stored only by the qualification service, never on either runner.
- Leave repository variable `CRYSTAL_SCHEDULE_ENABLED` unset or `false` during
  setup. Only the exact value `true` enables scheduled qualification jobs.
  Do not enable it until the evaluation owners agree on the models, task/sample
  settings, workload coverage, cadence, GPU capacity and operational approvals.
  Existing environment approvals still apply to scheduled jobs; a schedule does
  not bypass them. Agree on any unattended-run policy with the environment owners.
- Set variable `CRYSTAL_SGLANG_BASE_IMAGE` to the approved
  `lmsysorg/sglang:v0.5.22-rocm10-mi35x@sha256:<digest>` image. The Dockerfile installs
  AKLR's dependencies and the same pinned AITER baseline as native CI, keeping
  system Triton. It verifies AKLR, SGLang, AITER and Triton imports/versions.
  V1 builds Linux x86-64 images for gfx950; other hardware/frameworks need their
  own reviewed profile and image. Registry access is required during the build.
- Pre-create `ghcr.io/amd-agi/aklr-sglang` as a **private** package and grant this
  repository's Actions write access. The workflow refuses to create a package
  implicitly or push when its visibility is not private. Provision private pull
  credentials on the service's GPU workers independently.

## Scheduled runs: merged changes only

The proposed schedule is Monday, Wednesday and Friday at **10:17 UTC**: gaps of
two, two and three days. GitHub captures the latest `main` commit when it starts
the scheduled run. That exact SHA is used throughout the build, submission and
Check publication, even if more PRs merge while it runs. Scheduled runs never
select an open PR or require a PR number. They cover the changes already merged
into that commit.

Scheduled jobs are **disabled by default** behind `CRYSTAL_SCHEDULE_ENABLED`.
The timer may appear as a skipped workflow while disabled; no qualification
runner, build or submission starts. The workflow must be merged onto the default
branch for GitHub's timer to take effect. No branch protections or required
checks are changed; periodic qualification is not a per-PR merge requirement.

For main/release commits, the latest native `ci.yaml` **push** run on `main` for
the exact SHA must have passed, including `unittest-aklr` and `validation`.
Missing, incomplete, failed or skipped GPU validation stops qualification. It
does not quietly substitute an older passing commit or fall back to PR CI.

## Manual runs

Select **Actions → Crystal Correctness Qualification → Run workflow** using
`main`, then choose one of these inputs:

- Leave both inputs empty to test the merged `main` commit captured at dispatch.
- Enter an open same-repository PR number for an optional pre-merge test. The PR
  still requires `ci:gpu` and successful native PR CI for its exact head. Forks
  are not supported. The workflow rechecks the PR head before submission.
- Enter a full commit SHA to recheck a release candidate already on `main`.
  The workflow verifies that it is an ancestor of the captured main head and
  applies the same native push-CI gate. Tags and arbitrary branch names are not
  accepted. Do not combine a commit SHA with a PR number.

Manual runs remain available while scheduled jobs are disabled. After validation,
the workflow builds the candidate using a trusted default-branch Dockerfile,
submits the signed request and exits. Crystal finishes the evaluation later.
Re-running submission with the same workflow run ID is idempotent; start a new
workflow to request a new evaluation or test a new PR commit.

Deploy the companion Crystal API before using this sender. It sends
`schema_version: 2`, with `pull_request: null` for main/release commits. Crystal
also accepts the original PR-only version 1 payloads. Every result is attached
to the exact commit; a scheduled result is not copied onto individual PR heads.

Only a generic GitHub Check is public. Detailed scores, models, paths, server
addresses and logs stay inside the service. The workflow never uploads build
logs or prints submission response bodies. The App has Checks write and Metadata
read only, and must never generate events that start other workflows.

## Verification and scope

`python3 -m unittest discover -s .github/scripts/tests -v` tests request signing,
validation and public output. `node --test .github/scripts/tests/*.test.cjs`
tests native-CI eligibility. Validate the YAML with actionlint before deployment.
The separate `Crystal integration contract tests` workflow runs the Python and
Node tests on relevant PRs without credentials, service access, or GPU runners.
No real image build, GPU execution or Check publication occurs in these tests.

The image enables `AKLR_SGLANG_PLUGIN=1`, retains AITER as a separate dependency,
and preserves normal dispatch admission rules. New unmeasured PR kernels can
fall back. V1 therefore checks candidate model accuracy only; baseline comparisons
and proof of changed-kernel execution must be added before claiming full downstream
release qualification. Do not bypass kernel validation to obtain a model pass.
