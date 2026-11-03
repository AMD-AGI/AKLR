# Qwen3.5 GDN prefill and decode plugin

This plugin routes Qwen3.5-397B-A17B-FP8 attention-TP4 prefill on MI355X/gfx950
through `gdn_prefill_group_fp8_quant@1.0`, and decode through
`gdn_decode_group_fp8_quant@1.0` and `gdn_decode@1.0` ([Decode](#decode)). It
uses the shared SGLang plugin entry point, a route tested on SGLang 0.5.22,
contract dispatcher and kernel validation.

## Activation and selection

Install this checkout and launch SGLang 0.5.22 with `AKLR_SGLANG_PLUGIN=1`.
If `SGLANG_PLUGINS` filters entry points, include `aklr`. No GDN-specific flag
is added. `AKLR_DISABLE=1` disables AKLR; a contract ID disables one op.

The route targets `Qwen3_5GatedDeltaNet.forward` of SGLang 0.5.22, the release
it was tested on; on any other release SGLang runs its own code. It retains
SGLang's input/output projections and translates its live state pools into the
op's arguments. Implementation selection belongs to the shared
dispatcher: a workload needs passing measurements and sufficient speedup.
An unmeasured or unsupported call returns to native SGLang before state
mutation. A selected implementation's error propagates without a native retry.
Before SGLang's input projections run, the route asks the op whether it serves
the call (`fallback_reason`, given empty stand-ins of the projections' shape), so
a call no kernel serves skips them; an export records that call as dispatch
does.

On every call the route leaves to SGLang LoRA and GDN layers whose convolution
is not SiLU, whose gated norm is not swish, or whose out_proj is not block-FP8
on AITER's CK GEMM (SGLang hands AITER's Triton GEMM the FP8 op's row-major
scales transposed). It accepts BF16 hidden inputs or AMD's `(bf16, fp8, scale)`
input tuple and ordinary EXTEND, and asks the op at the call's own Point: its
tokens, sequences and longest sequence, and the state pools' shapes, dtypes and
layout. Prefill under verification, graph capture, state tracking or multi-item
scoring uses SGLang. DECODE takes the decode ops below.

## Contract and implementations

The op consumes contiguous Q/K/V/Z and B/A projections. `conv_state` and
`delta_state` are explicit `mutable=True` inputs; outputs are BF16 normalized
values, E4M3FN activations and one FP32 scale per 128-element value head.
The mathematical reference returns those outputs followed by both final state
pools. All other inputs and unindexed state slots remain unchanged.

There are nine implementations embedded in fifteen schema JSON files, with
twenty-one initial validation partitions (including interval boundaries):

- Packed short sequences: `(256,256)`, `(512)`, `(1024)`, `(512,512)`,
  `(1025)`, `(2048)`, `(2048,1)`, `(2048,2048)`,
  `(2048,2048,1)`, four 2048-token sequences plus one token, and seven
  2048-token sequences plus 1025 tokens. Boundary cases also cover four or
  eight 2048-token sequences and seven 2048-token sequences plus 1024 tokens.
- Contiguous: `(5000)`, `(5000,5000,5000)`, `(8192)`, `(10000)`, `(16384)`,
  `(10000,6384)`, `(8192,8192)`.

The actual maximum sequence length is a host contract attribute; the route
takes it from SGLang's extend lengths. The input distribution generates matching
positive partitions. Initially declared serving maxima are 256, 1025, 2048, 5000, 8192,
10000 and 16384, plus 512 and 1024; a small maximum of two is included for contract tests.
Other maxima and unmeasured token/batch combinations stay on SGLang.
State-pool capacity is an any-size axis within the signed 32-bit addressing
limit. Scheduler state slots must be distinct, valid indices.

The kernel ABI writes caller-provided outputs and returns nothing. Packing
for the six grouped-input implementations is inside the candidate and its
timing. The Gluon device computation is unchanged; host output buffers are
adapted to the ABI. A preceding device kernel clears fresh recurrent slots;
cached and unindexed slots are preserved. This initialization is included in
candidate timing and removes reliance on SGLang's allocator zeroing.

## Validation and performance boundary

`reference.py` independently computes the recurrence and rounding boundaries.
`baseline.py` replays SGLang 0.5.22's projection unpacking, causal convolution,
gates, chunk GDN, gated RMSNorm and block-128 activation quantization. Its host
maximum matches the convolution launch geometry without a timed GPU readback.
Fresh recurrent slots are initialized to implement this op's semantics;
normal serving already obtains fresh slots zeroed from its allocator. Projection
GEMMs and output projection are shared framework work outside this op.
Component timing therefore does not establish end-to-end serving speedup.

Run from the checkout:

```bash
python -m aklr export check
python -m aklr validate --base origin/main --stages audit,bench \
  aklr/plugins/sglang/models/qwen3_5/impls/triton/*/*.json
```

Correctness and performance fields are populated by validation/CI, not copied
from earlier results. Before merge, run native-versus-plugin forward checks
with fresh/cached/mixed states, consecutive calls and all supported partitions,
then Qwen3.5 TP4 accuracy and paired serving performance. No GPU qualification
or speedup is claimed by CPU tests, schema export or static audit alone.

## Decode

The same route serves DECODE: one token per row, at graph capture too. SGLang
warms every captured batch size up outside a capture first, as dispatch needs.
The route keeps SGLang's input projections, passes the layer's live
convolution and FP32 recurrent pools and the backend's `mamba_cache_indices`
unchanged, then finishes the layer as SGLang's decode does: its
`_track_mamba_state_decode` after both state updates, then `out_proj`.

`gdn_decode_group_fp8_quant@1.0` fuses the activation quantization of the
block-FP8 output projection: its E4M3FN values and row-major FP32 per-head
scales reach `out_proj` as the `(values, scales)` pair SGLang's linear takes
pre-quantized (`Fp8LinearMethod.apply` passes it to
`aiter_w8a8_block_fp8_linear` as `input` and `input_scale`; with ROCm 7.2 or
later that copies the scales to the bpreshuffle GEMM's column-major layout).
Where only `gdn_decode@1.0` has a kernel, its BF16 output reaches `out_proj`
as SGLang's own activation. ReplaySSM, speculative verification, idle batches,
LoRA, alternate quantizers and incompatible layouts use SGLang. Every fallback
precedes the projections, except an op declining the call after them.

SGLang pads a replayed graph batch with `-1` rows. A `-1` row reads and writes
no state, and its core output is zero, as in SGLang's kernels. Warm-up and
capture run every row on SGLang's reserved padding slot 0: the rows then race
on that slot alone, as SGLang's own decode kernels do, and no live state
changes. The index distribution draws distinct slots with `-1` at most once; each row
is independent in the kernels, so several padding rows take the same path as one.

Nine schemas hold one Point each: grouped FP8 at 1, 2, 4, 8, 16, 32, 128 and
256 rows, BF16 at 64 rows. Each workload's pool is the smallest above its rows
for which the timed inputs draw no `-1` row and a correctness input set draws
one. The kernels' entries write the caller's outputs; their Gluon device code
is unchanged. `reference.py` lists the decode rounding points: as SGLang's
`causal_conv1d_update` and every kernel here do, the convolution rounds each
tap's product to BF16 before an FP32 sum, oldest tap first. A Q/K/V channel
that a correct kernel rounds to the neighbouring BF16 value moves one step's
state by 2^-8 of `|update x key|`, so decode's `delta_state` tolerance is 1e-2
absolute plus 1e-2 relative. `baseline.py` replays SGLang 0.5.22's ROCm decode
kernels and, for the FP8 op, the per-1x128 AITER quantization that its output
projection applies to a BF16 input. Projection GEMMs stay outside both ops, so
component timing does not establish serving speedup.
