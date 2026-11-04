# Review: aklr_dispatch_floor

- Root cause, S1: `GPU_FORCE_QUEUE_PROFILING="1"` read at `aklr/core/validation/stages/_bench/_execution/_runner.py:475`
  on AKLR main `a29581b2d`; the runner sets it before torch initializes HIP, for the whole runner process.
- E1: probe output read line by line in `tasks2/probe_floor2.out` (n_tokens 8 and 112 both show it; 8 quoted).
- E2: the same delivered bmm file, the same node and card, validate run twice; the only difference is the deleted
  `GPU_FORCE_QUEUE_PROFILING` line (`tasks2/step66.out` shows the diff stat of that one line).
- E3: Kineto dump of the back-to-back graph inside validate (`tasks2/step65.out`): a single kernel per call, so the
  4.1 µs is not extra work.
- Not claimed: that removing the variable is safe for AKLR's Kineto readings in general (one validate run did not
  fault, E2); the page only says where the floor comes from and what to do before a campaign.
- Duplicates: `pitfalls/speedup_from_another_ruler.md` is about a harness other than AKLR's disagreeing with CI;
  this page is about AKLR's own ruler. No overlap.
