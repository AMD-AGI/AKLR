# Harness Rules

These high-level rules define the intended boundary of the AKLR kernel
lifecycle. Automated enforcement can be added later without changing the
meaning of the rules.

1. **AITER compatibility:** AKLR must preserve the public interfaces required
   by existing AITER consumers unless a migration is explicitly approved.
2. **Generated-kernel destination:** New agent-generated kernels should land in
   AKLR rather than extending AITER as a generated-kernel warehouse.
3. **Evidence before acceptance:** Correctness, coverage, and performance
   claims must be supported by reproducible evidence on declared hardware and
   software versions.
4. **Fair evaluation:** Benchmarks must include representative and edge-case
   inputs and must not hide warm-up, fallback, failure, or unfavorable cases.
5. **Clear integration:** Kernel names, placement, dispatch, fallbacks, and
   upstream provenance must be understandable to a reviewer.
6. **Reproducible environments:** Review and downstream qualification should
   use pinned, traceable runtime images.
7. **Downstream readiness:** Changes intended for release must be qualified
   against supported SGLang, vLLM, and ATOM integrations.
8. **Human-controlled mutation:** Skills may prepare changes and evidence, but
   publication, merging, and external updates require the authority of the
   invoking workflow or user.
