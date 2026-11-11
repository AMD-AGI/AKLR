# Runtime Image Workflow

1. **Resolve inputs.** Convert the requested AKLR ref to a commit and record the
   target GPU architectures and required downstream workload.
2. **Pin the environment.** Select a base image, ROCm, Python, PyTorch, compiler,
   and dependency revisions compatible with the target. Prefer immutable
   digests and record any input that remains mutable.
3. **Build reproducibly.** Keep credentials out of build layers and logs. Avoid
   dependencies on unrecorded local directories, caches, or environment state.
4. **Validate.** Confirm the image reports the intended AKLR revision, imports
   successfully, discovers the expected GPU architecture when run on hardware,
   and completes a representative smoke operation.
5. **Record the artifact.** Capture build command, tag, digest, source revision,
   base digest, dependency versions, build time, validation results, and known
   limitations.
6. **Publish only when authorized.** Treat registry push, tag replacement, and
   artifact deletion as separate external actions requiring authority.
