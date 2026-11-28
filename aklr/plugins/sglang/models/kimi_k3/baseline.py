# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.23 runs on gfx950 for Kimi-K3.

kimi_k3_latent_moe: on aiter, KimiK3MoE._forward_fused up to its
all-reduce, on one stream: the merged front GEMM (_k3_bf16_gemm), the shared experts'
SituAndMul and down GEMM into their slice, then TopK (aiter's biased_grouped_topk on the bf16
logits and the bias cast to bf16, as select_experts hands them) and FusedMoE on aiter's runner
(fused_moe at per_1x32, Situv2 with the gate and up rows apart, the activations in MXFP8 under
AITER_SITUV2_A8W4=1, sglang's Kimi-K3 environment). With front_fp32, sglang 0.5.24's
_forward_fused_hip (Kimi-K3.1's image) up to its all-reduce: the front GEMM to fp32, the SiTU on
the fp32 gate and up (bf16 out), the top-k on the fp32 logits and bias, the latent input cast to
bf16.

kimi_k3_mla_vc_output_gate: rocm_absorb_v_bmm's bf16 torch.bmm into a (tokens, heads, dim)
buffer, flattened, then the o_proj wrapper's fused mla_output_gate (x * sigmoid(gate) where it
does not cover the shapes).

kimi_k3_mla_prefill_gate: forward_extend's extend_attention_fwd on a zero prefix, which casts
Q, K and V to FP8 for Kimi-K3's shapes on gfx950 (SGLANG_TRITON_FP8_PREFILL_ATTN, on by
default), then the fused output gate.

kimi_k3_routed_moe_mxfp8: from sglang 0.5.24.dev809 (Kimi-K3.1's image), its own gfx950 A8W4
experts, k3_a8w4_moe, on the input its row-sharded front gathered in MXFP8 (a_mx): aiter's
moe_sorting, then the copy of sglang's gateup, down and reduce kernels in _k3_a8w4_gfx950.hip, built
here as a torch extension, so a sglang without them (CI's release) runs the baseline too;
tests/plugins/sglang/test_k3_a8w4_baseline.py holds the copy to sglang's own kernels bit for bit.

kimi_k3_moe_front_gemm: sglang 0.5.24.dev809's _k3_bf16_gemm on the fp32 MoE front where aiter's FlyDSL
tile does not take it: torch.mm into the fp32 output it allocates (hipBLASLt on ROCm)."""

import functools
from pathlib import Path

import torch

from aklr.core.contract.baseline import OpBaseline

__all__ = [
    "KimiK3LatentMoeBaseline",
    "KimiK3MlaPrefillGateBaseline",
    "KimiK3MlaVcOutputGateBaseline",
    "KimiK3MoeFrontGemmBaseline",
    "KimiK3RoutedMoeMxfp8Baseline",
]


def _check_aiter_path():
    """Raise unless sglang takes the aiter path the baseline replays: it is chosen at import,
    from SGLANG_USE_AITER=1 on gfx950."""
    from sglang.srt.layers.quantization import unquant

    if not (unquant._use_aiter and unquant._is_gfx95):
        raise RuntimeError("sglang's aiter path is off: SGLANG_USE_AITER=1 on gfx950")


def _fp4x2(weight):
    """An MXFP4 weight as Mxfp4MoEMethod._apply_aiter hands it to the runner: a
    float4_e2m1fn_x2 view (where torch has the dtype), as shuffled as the weight."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return weight
    view = weight.view(torch.float4_e2m1fn_x2)
    view.is_shuffled = getattr(weight, "is_shuffled", False)
    return view


def _experts(z, weights, ids, w1, w1_scale, w2, w2_scale, out):
    """FusedMoE with Mxfp4MoEMethod on aiter's runner (AiterRunnerCore.run) for SiTU: aiter's
    fused_moe into ``out``, at sglang's sorting dispatch policy."""
    import os

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode
    from sglang.srt.layers.moe.moe_runner import aiter as runner

    saved = os.environ.get("AITER_SITUV2_A8W4")
    os.environ["AITER_SITUV2_A8W4"] = "1"
    try:
        extra = {"output": out} if runner._aiter_fused_moe_supports_output() else {}
        result = fused_moe(
            z,
            _fp4x2(w1),
            _fp4x2(w2),
            weights,
            ids,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale.view(torch.uint8),
            w2_scale=w2_scale.view(torch.uint8),
            gate_mode=GateMode.SEPARATED.value,
            beta=4.0,
            linear_beta=25.0,
            moe_sorting_dispatch_policy=runner._MOE_SORTING_DISPATCH_POLICY,
            **extra,
        )
    finally:
        if saved is None:
            os.environ.pop("AITER_SITUV2_A8W4", None)
        else:
            os.environ["AITER_SITUV2_A8W4"] = saved
    if result.data_ptr() != out.data_ptr():
        out.copy_(result)


@functools.lru_cache(maxsize=64)
def _bias_as(bias_ptr, version, bias, dtype):
    """The bias in the logits' dtype, cast once as TopKConfig.correction_bias_for_dtype caches it."""
    return bias.to(dtype)


class KimiK3LatentMoeBaseline(OpBaseline):
    def baseline(
        self,
        *,
        hidden_states,
        front_weight,
        correction_bias,
        w1,
        w1_scale,
        w2,
        w2_scale,
        shared_w2,
        topk=16,
        front_fp32=False,
        latent=None,
        shared=None,
    ):
        """_forward_fused's single-collective branch (no alt stream, k3_ar_fusion off) before its
        all-reduce: _k3_bf16_gemm, then _forward_shared, then _forward_routed. With front_fp32,
        sglang 0.5.24's _forward_fused_hip on aiter: the front GEMM to fp32, the SiTU on its fp32
        gate and up to bf16, the router on the fp32 logits with the fp32 bias, the latent input cast
        to bf16."""
        _check_aiter_path()
        from sglang.srt.layers.activation import SituAndMul
        from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu
        from sglang.srt.models import kimi_k3

        m, latent_dim = hidden_states.shape[0], w2.shape[1]
        shared_inter_dim = shared_w2.shape[1]
        n_routed = w1.shape[0]
        if latent is None:
            latent = hidden_states.new_empty((m, latent_dim))
        if shared is None:
            shared = hidden_states.new_empty((m, hidden_states.shape[1]))
        front = kimi_k3._k3_bf16_gemm(
            hidden_states, front_weight, out_dtype=torch.float32 if front_fp32 else None
        )
        gate_up, logits, z = torch.split(front, (2 * shared_inter_dim, n_routed, latent_dim), dim=-1)
        if m > 1 and not kimi_k3._aiter_k3_opt:
            logits = logits.contiguous()
        z = z.to(hidden_states.dtype).contiguous()
        # sglang's SiTU writes bf16 off an fp32 gate and up
        act = SituAndMul(beta=4.0, linear_beta=25.0)(gate_up).to(hidden_states.dtype)
        kimi_k3._k3_bf16_gemm(act, shared_w2, out=shared)
        weights, ids = biased_grouped_topk_gpu(
            hidden_states,
            logits,
            _bias_as(correction_bias.data_ptr(), correction_bias._version, correction_bias, logits.dtype),
            topk,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
        )
        _experts(z, weights, ids, w1, w1_scale, w2, w2_scale, latent)
        return {"latent": latent, "shared": shared}


class KimiK3MlaVcOutputGateBaseline(OpBaseline):
    def baseline(self, *, latent, vc_weight, gate, output=None):
        """rocm_absorb_v_bmm on bf16 weights, then the fused output gate."""
        from sglang.kernels.ops.elementwise import mla_output_gate

        tokens, heads = latent.shape[0], latent.shape[1]
        buffer = latent.new_empty((tokens, heads, vc_weight.shape[1]))
        torch.bmm(latent.transpose(0, 1), vc_weight.transpose(1, 2), out=buffer.transpose(0, 1))
        x = buffer.flatten(1, 2)
        if mla_output_gate.covered(x, gate):
            result = mla_output_gate.kimi_k3_mla_output_gate(x, gate)
        else:
            result = x * torch.sigmoid(gate)
        if output is None:
            return {"output": result}
        output.copy_(result)
        return {"output": output}


@functools.cache
def _fp8_pool(device):
    """A one-slot FP8 latent pool: zero-prefix extend reads none of it, its dtype selects the
    FP8 path."""
    pool = torch.zeros((1, 1, 576), dtype=torch.float8_e4m3fn, device=device)
    return pool, pool[..., :512]


class KimiK3MlaPrefillGateBaseline(OpBaseline):
    def baseline(self, *, q, k, v, gate, qo_indptr, sm_scale=192**-0.5, max_seq_len=8192, output=None):
        """extend_attention_fwd with forward_extend's arguments on a zero prefix, then the gate."""
        from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
        from sglang.kernels.ops.elementwise import mla_output_gate

        tokens, heads = q.shape[0], q.shape[1]
        k_pool, v_pool = _fp8_pool(q.device)
        attention = q.new_empty((tokens, heads, v.shape[-1]))
        no_prefix = torch.zeros_like(qo_indptr)
        extend_attention_fwd(
            q,
            k.contiguous(),
            v.contiguous(),
            attention,
            k_pool,
            v_pool,
            qo_indptr.to(torch.int64),
            no_prefix,
            no_prefix.to(torch.int64),
            None,
            True,
            None,
            max_seq_len,
            1.0,
            1.0,
            sm_scale,
        )
        x = attention.view(tokens, -1)
        if mla_output_gate.covered(x, gate):
            return {"output": mla_output_gate.kimi_k3_mla_output_gate(x, gate)}
        return {"output": x * torch.sigmoid(gate)}


_K3_A8W4_SOURCE = Path(__file__).with_name("_k3_a8w4_gfx950.hip")
_K3_A8W4_LIBRARY = "aklr_baseline_k3_a8w4_gfx950"
# sglang's k3_a8w4_gfx950.py: blocks of one expert per down work item, and the down's y column chunk
_DOWN_SEG_BLOCKS = 8
_Y_CHUNK_COLS = 512


@functools.cache
def _k3_a8w4_ops():
    """The copy's ops, its library built on the first call for this process's GPU."""
    from aklr.core.jit import hip_extension
    from aklr.core.platform.device import DeviceTarget
    from aklr.core.runtime.builder.builder import get_build_cache_path

    arch = DeviceTarget.current().arch
    if arch is None:
        raise RuntimeError("k3_a8w4_gfx950: no GPU arch to build for")
    root = get_build_cache_path() / "baselines" / _K3_A8W4_LIBRARY
    hip_extension.load_library(
        _K3_A8W4_LIBRARY, {"k3_a8w4_gfx950.hip": _K3_A8W4_SOURCE.read_text()}, root, arch
    )
    return getattr(torch.ops, _K3_A8W4_LIBRARY)


def _block_m(num_tokens: int, top_k: int, num_experts: int) -> int:
    """sglang's pick_block_m: rows of a sort block by the rows an expert gets."""
    rows_per_expert = num_tokens * top_k / num_experts
    if rows_per_expert <= 4:
        return 16
    if rows_per_expert <= 32:
        return 32
    return 64


class KimiK3RoutedMoeMxfp8Baseline(OpBaseline):
    def baseline(
        self,
        *,
        a,
        a_scale,
        topk_ids,
        topk_weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
        beta,
        linear_beta,
        out_dtype=None,
        out=None,
    ):
        """k3_a8w4_moe(..., a_mx=(a, a_scale)) as sglang's _k3_a8w4_moe runs it: aiter's moe_sorting into
        block_m-row blocks, gateup, down and the fp32 top-k reduce into out."""
        num_tokens, hidden = a.shape
        num_experts, top_k, inter_dim = w1.shape[0], topk_ids.shape[1], w1.shape[1] // 2
        # KimiK3MoEHipMixin._a8w4_moe_min_tokens: the experts are built for these; elsewhere aiter's fused_moe runs
        if hidden % 512 or inter_dim not in (384, 512) or top_k != 16 or num_experts > 1024:
            raise NotImplementedError(
                "sglang's k3_a8w4_moe: a hidden size a multiple of 512, inter 384 or 512, top-16, 1024 experts at most"
            )
        from aiter.fused_moe import moe_sorting

        if out is None:
            out = torch.empty(num_tokens, w2.shape[1], dtype=torch.bfloat16, device=a.device)
        if num_tokens == 0:
            return {"out": out}
        block_m = _block_m(num_tokens, top_k, num_experts)
        sorted_ids, _, expert_ids, num_valid, _ = moe_sorting(
            topk_ids, topk_weights, num_experts, hidden, torch.bfloat16, block_size=block_m, accumulate=False
        )
        ops, u8 = _k3_a8w4_ops(), torch.uint8
        rows = expert_ids.shape[0] * block_m
        inter = torch.empty(rows, inter_dim, dtype=u8, device=a.device)
        inter_scale = torch.empty(rows, inter_dim // 32, dtype=u8, device=a.device)
        ops.gateup(
            a.view(u8),
            a_scale.view(u8),
            w1.view(u8),
            w1_scale.view(u8),
            sorted_ids,
            expert_ids,
            num_valid,
            inter,
            inter_scale,
            block_m,
            float(beta),
            float(linear_beta),
        )
        y = torch.empty(sorted_ids.shape[0], w2.shape[1], dtype=torch.bfloat16, device=a.device)
        pos = torch.empty(num_tokens * top_k, dtype=torch.int32, device=a.device)
        # the down's work list: a header and at most max_blocks / seg_blocks + E items, int4 each
        max_items = -(-expert_ids.shape[0] // _DOWN_SEG_BLOCKS) + w2.shape[0]
        work = torch.empty(4 * (1 + max_items), dtype=torch.int32, device=a.device)
        ops.down(
            inter,
            inter_scale,
            w2.view(u8),
            w2_scale.view(u8),
            sorted_ids,
            expert_ids,
            num_valid,
            y,
            pos,
            topk_weights,
            work,
            block_m,
            _DOWN_SEG_BLOCKS,
        )
        ops.reduce(y, topk_weights, pos, out, _Y_CHUNK_COLS)
        return {"out": out}


class KimiK3MoeFrontGemmBaseline(OpBaseline):
    def baseline(self, *, x, weight, out_dtype=None, out=None):
        """_k3_bf16_gemm's torch call off an fp32 front: torch.mm(x, weight.t(), out=out, out_dtype=out.dtype) into
        the fp32 output it allocates."""
        if out is None:
            out = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device)
        torch.mm(x, weight.t(), out=out, out_dtype=out.dtype)
        return {"out": out}
