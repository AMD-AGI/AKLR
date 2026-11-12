# AKLR validation bench

The bench decides whether a kernel may serve its schema's workloads: it checks the kernel's outputs
against the schema's reference and times it against what users would run without it.

## Language

### What is compared

**Candidate**:
The kernel under validation: the kernel of a schema at the head commit.
_Avoid_: new kernel, PR kernel

**Workload**:
One point of a schema's axes at which a kernel is checked and timed.
_Avoid_: case, point, shape

**Standard**:
What a kernel of a schema is held to: the init that draws a workload's inputs, the Reference and the check that judge
its outputs, and the baseline that calls the Baseline. Defined with the schema in code; a schema json never holds it.
_Avoid_: yardstick, criteria, test spec

**Reference**:
The high-precision function of a schema's Standard that defines the correct outputs of a workload; it is never timed.
_Avoid_: golden, expected outputs

### What a candidate is timed against

**Baseline**:
The op's implementation that users get without AKLR: the original function a seam adapter wraps, which the
baseline of the schema's Standard calls.
_Avoid_: backend, framework op, original

**Incumbent**:
The kernel AKLR serves for a workload at the base commit.
_Avoid_: base kernel, old kernel, served kernel

### Verdicts

**Entry threshold**:
The least speedup a candidate needs over the Incumbent, or over the Baseline where no Incumbent serves, to start serving a workload.

**Retain threshold**:
The least speedup over the Baseline, a small slowdown allowed, a kernel needs to keep serving a workload it already serves.

**Admission**:
The bench's acceptance of a candidate schema: at least one workload is correct and clears the Entry threshold, or is correct where there is neither a Baseline nor an Incumbent to time it against.
_Avoid_: pass

**Demotion**:
A workload of an admitted candidate that is incorrect or too slow; it is recorded on that workload so dispatch skips it, and does not fail the bench.
_Avoid_: soft failure, warning
