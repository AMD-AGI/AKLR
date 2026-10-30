# Harness Rules

These high-level rules define the intended boundary of the AKLR kernel
lifecycle. Automated enforcement can be added later without changing the
meaning of the rules.

1. **AITER compatibility:** AKLR must preserve the public interfaces required
   by existing AITER consumers unless a migration is explicitly approved.
2. **Generated-kernel destination:** New agent-generated kernels should land in
   AKLR rather than extending AITER as a generated-kernel warehouse.
3. **Selective AITER synchronization:** An AITER change is ported only when it
   is compatible, correct, and demonstrates a relevant performance or
   maintainability benefit. No measurable benefit means no sync by default.
4. **Evidence before acceptance:** Correctness, coverage, and performance
   claims must be supported by reproducible evidence on declared hardware and
   software versions.
5. **Fair evaluation:** Benchmarks must include representative and edge-case
   inputs and must not hide warm-up, fallback, failure, or unfavorable cases.
6. **Clear integration:** Kernel names, placement, dispatch, fallbacks, and
   upstream provenance must be understandable to a reviewer.
7. **Reproducible environments:** Review and downstream qualification should
   use pinned, traceable runtime images.
8. **Downstream readiness:** Changes intended for release must be qualified
   against supported SGLang, vLLM, and ATOM integrations.
9. **Human-controlled mutation:** Skills may prepare changes and evidence, but
   publication, merging, and external updates require the authority of the
   invoking workflow or user.
