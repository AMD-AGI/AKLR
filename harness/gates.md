# Kernel Lifecycle Gates

The gates describe the decisions a kernel must pass. This initial version is a
review contract, not an automated CI implementation.

## Gate 1: Integration Readiness

- The source and provenance are recorded.
- The kernel has a meaningful name and follows AKLR placement conventions.
- The public integration path remains AITER-compatible.
- Dispatch, fallback behavior, and supported GPU scope are clear.

## Gate 2: Kernel Qualification

- Unit and correctness tests cover representative and edge-case inputs.
- Results are compared with a trusted reference implementation.
- Performance is compared fairly with the current AKLR incumbent.
- The review checks for reward hacking, hidden fallbacks, selective inputs, and
  misleading benchmark methodology.
- Regressions and unsupported cases are explicitly reported.

## Gate 3: Release Qualification

- The runtime image is reproducible and identifies the exact AKLR revision.
- End-to-end correctness and performance are evaluated on supported hardware.
- Supported SGLang, vLLM, and ATOM integrations remain functional.
- The evidence is sufficient for a human release or merge decision.
