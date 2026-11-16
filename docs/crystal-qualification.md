# Crystal correctness qualification

V1 submits an exact PR image for candidate model accuracy checks. It supplements
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

## Run

After native GPU CI passes, select **Actions → Crystal Correctness Qualification
→ Run workflow**, use the default branch and enter an open same-repository PR
number. Fork PRs are not supported. The workflow verifies the latest native CI
and its GPU jobs, builds the PR head using a trusted Dockerfile, checks the PR
has not changed, submits a signed request and exits. Service runs finish later.
Re-running submission with the same workflow run ID is idempotent; start a new
workflow to request a new evaluation or test a new PR commit.

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
