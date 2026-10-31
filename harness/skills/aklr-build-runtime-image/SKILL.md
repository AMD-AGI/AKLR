---
name: aklr-build-runtime-image
description: Build and validate a reproducible runtime image for an exact AKLR revision and declared ROCm stack. Use when kernel review, CI, or downstream qualification needs a traceable test environment.
---

# Build a Runtime Image

Produce an image whose source, dependencies, build inputs, and validation can be
reproduced by another reviewer.

## Required Context

Read the harness [rules](../../rules.md), [gates](../../gates.md), current
[gate configuration](../../gate-config.md), and
[references/workflow.md](references/workflow.md).

## Responsibilities

- Resolve the requested AKLR version to an immutable commit.
- Pin the base image by version and, when available, digest.
- Record ROCm, Python, PyTorch, compiler, GPU targets, and relevant dependency
  revisions.
- Build without embedding credentials, workstation-only paths, or mutable
  unrecorded inputs.
- Validate installation, import, version reporting, and a representative smoke
  path before reporting success.
- Produce a traceable image tag and manifest information suitable for review
  and downstream qualification.

## Publication Boundary

Building locally does not authorize pushing to an image registry. Request or
use explicit publication authority before uploading, replacing, or deleting an
external image tag.

## Completion

Return the AKLR commit, base image, resulting image identifier or digest, build
command, validation results, target architectures, and known limitations. Use
[references/report-template.md](references/report-template.md).
