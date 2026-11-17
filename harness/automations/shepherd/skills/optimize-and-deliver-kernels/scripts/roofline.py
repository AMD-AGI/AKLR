"""The least time one call of a schema can take on the GPU, from the schema file alone.

The formulas and rates are the shepherd's own, to rank candidates; AKLR does not read them.
"""

import math
from dataclasses import dataclass, field
from typing import NamedTuple

BITS = {
    "bfloat16": 16,
    "float16": 16,
    "float32": 32,
    "float4_e2m1": 4,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e8m0": 8,
    "int8": 8,
    "uint8": 8,
    "int32": 32,
    "int64": 64,
}
GEMM = frozenset({"gemm", "gemm_a8w8", "gemm_a8w8_blockwise_scaled", "gemm_a4w4_mxfp4", "bmm"})
# The contracts whose bytes are all their work.
MEMORY_BOUND = frozenset(
    {"rms_norm", "add_rms_norm", "per_group_quant_fp8", "per_group_quant_mxfp8", "biased_grouped_topk"}
)


class Cost(NamedTuple):
    """The least work of one call: matrix-core operations, and bytes between memory and chip."""

    flops: int  # multiply-adds count two
    bytes_moved: int  # each tensor the call reads or writes, moved once
    flops_dtype: str | None  # the operand dtype whose matrix-core rate bounds flops; None: no matrix math

    @classmethod
    def of(cls, schema, axes):
        """One call of the schema file's json ``schema`` at a workload's ``axes``. ``ValueError`` for a
        contract with matrix math and no formula here, or a file this cannot read."""
        contract = str(schema.get("contract_id")).split("@")[0]
        try:
            symbols = {
                **{name: axis.get("value") for name, axis in schema["axes"].items()},
                **axes,
                **schema["attrs"],
            }
            # A scalar input has no shape and moves nothing.
            tensors = {
                name: arg
                for name, arg in {**schema["inputs"], **schema["outputs"]}.items()
                if arg["shape"] is not None
            }
            (variant,) = schema["layout_variants"]["by_name"].values()
            tensor_bytes = {}
            for name, tensor in tensors.items():
                shape = [d if isinstance(d, int) else symbols[d] for d in tensor["shape"]]
                # The one layout that stores more than its shape: its columns padded to a multiple of 8.
                if variant.get("layouts", {}).get(name, {}).get("kind") == "gate_up_interleaved_scales":
                    shape[-1] = -(-shape[-1] // 8) * 8
                # A 4-bit dtype packs two values a byte.
                tensor_bytes[name] = math.prod(shape) * BITS[tensor["dtype"]] // 8
            moved = sum(tensor_bytes.values())
            if contract in GEMM:
                # 2 * batch * m * n * k operations at the rate of a's dtype.
                flops = 2 * symbols.get("batch", 1) * symbols["m"] * symbols["n"] * symbols["k"]
                return cls(flops, moved, tensors["a"]["dtype"])
            if contract == "fused_moe":
                # Each route to one of this rank's experts runs that expert's gate-up and down projections,
                # and only the experts some route reaches move their weights (routes uniform, distinct
                # within a token). The math runs at the rate of the least precision the contract's bound
                # admits: fp8 for 8-bit activations and interleave, bf16 for swiglu below 256 rows, else fp4.
                n_tokens, topk, n_experts = (symbols[a] for a in ("n_tokens", "topk", "n_experts"))
                pool = symbols["global_experts"] if symbols["has_expert_mask"] else n_experts
                reached = 1 - (1 - topk / pool) ** n_tokens
                weights = sum(tensor_bytes[w] for w in ("w1", "w2", "w1_scale", "w2_scale"))
                routes = n_tokens * topk * n_experts / pool
                flops = round(
                    2 * routes * symbols["model_dim"] * (symbols["gate_up_rows"] + symbols["inter_dim"])
                )
                if symbols["gate_mode"] == "interleave" or symbols["act_quant"] == 8:
                    dtype = "float8_e4m3fn"
                else:
                    dtype = (
                        "bfloat16" if symbols["activation"] == "swiglu" and n_tokens < 256 else "float4_e2m1"
                    )
                return cls(flops, moved - weights + round(reached * weights), dtype)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"cannot read the cost of {contract}: {exc!r}") from None
        if contract not in MEMORY_BOUND:
            raise ValueError(f"no cost formula for {contract}")
        return cls(0, moved, None)


@dataclass(frozen=True)
class Roofline:
    """The ceilings of one GPU; the defaults are MI355X's, measured on it (harness/skills/kb)."""

    # Matrix-core FLOP/s by operand dtype. Measured: bf16 2.44-2.47e15 and fp8 4.90-4.95e15
    # (triton/attention.md); fp16 as bf16, int8 as fp8 and fp4 at twice fp8 follow the spec's ratios, unmeasured.
    flops_per_second: dict = field(
        default_factory=lambda: {
            "bfloat16": 2.45e15,
            "float16": 2.45e15,
            "float8_e4m3fn": 4.92e15,
            "float8_e5m2": 4.92e15,
            "int8": 4.92e15,
            "float4_e2m1": 9.84e15,
        }
    )
    bytes_per_second: float = 7.1e12  # A streaming HBM read, 7.13-7.17e12 measured (triton/attention.md).
    # AKLR's graph timer reads no call faster: the fastest of 3017 kernel times in main's full bench of
    # 2026-11-13 (ROCm 7.15) was 4.18 µs.
    floor_seconds: float = 4.1e-6

    def seconds(self, cost):
        """The least time ``cost`` takes: its math at its dtype's rate or its bytes at the bandwidth,
        whichever is longer, never under the timer's floor. ``ValueError`` for math in a dtype without a rate."""
        math_seconds = 0.0
        if cost.flops:
            if cost.flops_dtype not in self.flops_per_second:
                raise ValueError(f"no matrix-core rate for {cost.flops_dtype}")
            math_seconds = cost.flops / self.flops_per_second[cost.flops_dtype]
        return max(math_seconds, cost.bytes_moved / self.bytes_per_second, self.floor_seconds)
