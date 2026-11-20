# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
import logging
import os

import torch

from aklr.core.runtime.dispatch import FallbackReason, api
from aklr.plugins.vllm.hook import vllm_route
from aklr.plugins.vllm.models.kimi_k3.kda_spec_stage_c.contract import KIMI_K3_KDA_SPEC_STAGE_C, MAX_WINDOW

__all__ = ["kimi_k3_kda_spec_stage_c"]

_LOGGER = logging.getLogger(__name__)
# AKLR_VLLM_KDA_STAGE_C=0 hands every call straight back to vLLM
_ENABLED = os.environ.get("AKLR_VLLM_KDA_STAGE_C", "1") != "0"
_SEEN: set[str] = set()
# the op's answer for a step's shape (None: it serves), by (device, rows, sequences, window): asked before the
# in-projection runs, with a stand-in for its rows, so that a declined step runs it once (vLLM's own path runs it)
_SERVES: dict[tuple, FallbackReason | None] = {}
_CONV_TAPS = 4  # Kimi-K3's short_conv_kernel_size: the contract's width


def _once(reason: str) -> None:
    """Log each distinct reason the route hands a call back to vLLM, once per process."""
    if reason not in _SEEN:
        _SEEN.add(reason)
        _LOGGER.warning("aklr kda_spec_stage_c route falls back: %s", reason)


@api(KIMI_K3_KDA_SPEC_STAGE_C)
def kimi_k3_kda_spec_stage_c(
    qkv,
    out_gate,
    f_a,
    raw_beta,
    f_b_weight,
    conv_weight,
    conv_state,
    a_log,
    g_bias,
    norm_weight,
    state_cache,
    state_indices,
    num_accepted_tokens,
    storage="row_major",
    scale=128**-0.5,
    lower_bound=-5.0,
    eps=1e-5,
    out_dtype=None,
    out=None,
): ...


@functools.cache
def _kda_class() -> type:
    from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention

    return KimiK3DeltaAttention


def _forward_metadata():
    """vLLM's attention metadata of the step being run: a dict by layer prefix."""
    from vllm.forward_context import get_forward_context

    return get_forward_context().attn_metadata


def _breakable_graphs() -> bool:
    """Whether vLLM breaks its graphs around the KDA layers' recurrence (VLLM_USE_BREAKABLE_CUDAGRAPH): it then runs
    _forward eagerly at every replay, with the step's own metadata, where a launch captured here would replay the
    capture's."""
    from vllm import envs

    return bool(getattr(envs, "VLLM_USE_BREAKABLE_CUDAGRAPH", False))


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _rows(hidden_states) -> int:
    """Rows of the attention input: a tensor, or vLLM's QuantizedActivation (fp8 rows, scales, original shape)."""
    shape = getattr(hidden_states, "orig_shape", None)
    return int(shape[0] if shape is not None else hidden_states.shape[0])


def _verify_step(attn, n_tokens):
    """(metadata, sequences, window) of a pure speculative verify step of uniform windows, every row a window's
    (vLLM's GDN metadata), else the reason it is not one."""
    metadata = _forward_metadata()
    m = metadata.get(attn.prefix) if isinstance(metadata, dict) else None
    if m is None:
        return "no KDA metadata"
    if m.num_prefills or m.num_decodes or not m.num_spec_decodes:
        return f"not a pure verify step ({m.num_prefills} prefills, {m.num_decodes} decodes)"
    slots, accepted, n_seqs = m.spec_state_indices_tensor, m.num_accepted_tokens, m.num_spec_decodes
    if slots is None or slots.dim() != 2 or accepted is None or accepted.dim() != 1:
        return "no window slots or accepted counts"
    window = slots.shape[1]
    checks = (
        ("slot dtypes", (slots.dtype, accepted.dtype) != (torch.int32,) * 2),
        (
            "fewer slot rows or accepted counts than sequences",
            slots.shape[0] < n_seqs or accepted.shape[0] < n_seqs,
        ),
        ("slot rows not dense", slots.stride(1) != 1),
        ("window", not 1 <= window <= MAX_WINDOW),
        ("windows not uniform", m.uniform_spec_sequence_length != window),
        ("rows not the windows'", n_tokens != n_seqs * window),
        ("breakable graphs", _breakable_graphs()),
    )
    failed = [name for name, bad in checks if bad]
    if failed:
        return (f"out of contract ({', '.join(failed)}): {n_tokens} rows, {n_seqs} sequences, slots "
                f"{tuple(slots.shape)}/{slots.stride()} {slots.dtype}, uniform {m.uniform_spec_sequence_length}")  # fmt: skip
    return m, n_seqs, window


def _layer_reason(attn, window) -> str | None:
    """Why the layer's weights or caches are outside the contract as vLLM holds them, else None."""
    heads, dim = attn.local_num_heads, attn.head_dim
    hd = heads * dim
    conv_state, state = attn.kv_cache
    conv_weight, f_b, norm = attn.conv1d.weight, attn.f_b_proj, attn.o_norm
    checks = (
        ("head_dim", dim != 128),
        ("conv width", attn.conv_size != _CONV_TAPS or conv_weight.shape != (3 * hd, 1, _CONV_TAPS)),
        ("conv weight", conv_weight.dtype != torch.float32 or not conv_weight.is_contiguous()),
        ("biases", attn.conv1d.bias is not None or getattr(f_b, "bias", None) is not None),
        ("f_b weight", f_b.weight.dtype != torch.bfloat16 or f_b.weight.shape != (hd, dim)
         or not f_b.weight.is_contiguous()),
        ("lower bound", attn.gate_lower_bound is None),
        ("A_log, dt_bias", attn.A_log.shape != (heads,) or attn.A_log.dtype != torch.float32
         or attn.dt_bias.numel() != hd or attn.dt_bias.dtype != torch.float32 or not attn.dt_bias.is_contiguous()),
        ("o_norm", getattr(norm, "activation", "sigmoid") != "sigmoid" or getattr(norm, "bias", None) is not None
         or norm.weight.dtype != torch.bfloat16 or norm.weight.shape != (dim,)),
        ("conv state not bf16 SD rows", conv_state.dtype != torch.bfloat16
         or conv_state.shape[1:] != (_CONV_TAPS - 2 + window, 3 * hd) or conv_state.stride()[1:] != (3 * hd, 1)),
        ("recurrent state", state.dtype != torch.float32 or state.shape[1:] != (heads, dim, dim)
         or state.stride()[1:] != (dim * dim, dim, 1)),
    )  # fmt: skip
    failed = [name for name, bad in checks if bad]
    return f"layer out of contract ({', '.join(failed)})" if failed else None


def _dense(indices):
    """indices as a dense tensor: itself, a fresh copy outside a graph capture, None inside one (vLLM's graphs pass
    dense ones; its eager steps view a column block of the block table)."""
    if indices.is_contiguous():
        return indices
    if _capturing(indices.device):
        return None
    return torch.empty_like(indices, memory_format=torch.contiguous_format).copy_(indices)


def _op_args(attn, proj, slots, accepted):
    """The op's operands: views of the in-projection's rows (q|k|v, the output gate, f_a, beta, then vLLM's padding
    columns), the layer's weights and its caches as kda.py holds them."""
    heads, dim = attn.local_num_heads, attn.head_dim
    hd = heads * dim
    conv_state, state = attn.kv_cache
    return (
        proj[:, : 3 * hd].unflatten(1, (3 * heads, dim)),
        proj[:, 3 * hd : 4 * hd].unflatten(1, (heads, dim)),
        proj[:, 4 * hd : 4 * hd + dim].unsqueeze(1),
        proj[:, 4 * hd + dim : 4 * hd + dim + heads].unsqueeze(-1),
        attn.f_b_proj.weight,
        attn.conv1d.weight.view(3 * hd, _CONV_TAPS),
        conv_state,
        attn.A_log,
        attn.dt_bias.view(heads, dim),
        attn.o_norm.weight,
        state.flatten(1, 2),  # (slots, heads * dim, dim), each slot a part of a page
        slots,
        accepted,
    )


def _op_kwargs(attn):
    return {
        "storage": "strided_rows",
        "scale": attn.head_dim**-0.5,  # fused_recurrent_kda's default
        "lower_bound": float(attn.gate_lower_bound),
        "eps": float(attn.o_norm.eps),
    }


def _proj_width(attn) -> int:
    """Columns of the in-projection's rows: q, k, v and g, f_a, beta, then vLLM's padding to 16."""
    return (
        4 * attn.local_num_heads * attn.head_dim + attn.head_dim + attn.local_num_heads + attn.in_proj_padding
    )


def _serve_reason(attn, n_tokens, n_seqs, window, slots, accepted) -> FallbackReason | None:
    """None if the op serves this step's shape, else its reason: asked once a shape, on a stand-in for the
    in-projection's rows."""
    device = attn.A_log.device
    key = (device, n_tokens, n_seqs, window)
    if key in _SERVES:
        return _SERVES[key]
    dtype = attn.f_b_proj.weight.dtype  # the model's: the in-projection's rows and the output are
    stand_in = torch.empty((n_tokens, _proj_width(attn)), dtype=dtype, device=device)
    out = torch.empty((n_tokens, attn.local_num_heads, attn.head_dim), dtype=dtype, device=device)
    reason = kimi_k3_kda_spec_stage_c.fallback_reason(
        *_op_args(attn, stand_in, slots, accepted), **_op_kwargs(attn), out=out
    )
    if reason is not FallbackReason.NOT_WARMED:  # a capture's answer before any eager call: asked again later
        _SERVES[key] = reason
    return reason


@vllm_route(
    "vllm.models.kimi_k3.amd.linear.KimiDecoderLayer.forward_attn_residual",
    "vllm.models.kimi_k3.amd.linear.KimiDecoderLayer._run_self_attn",
    versions=("0.34.0", "0.32.1rc1.dev276"),
)
def _run_self_attn(self, positions, hidden_states):
    """A KDA layer's attention in a pure speculative verify step: vLLM's in-projection, then the op in place of the
    copy of f_a, the f_b GEMM and aiter's fused convolution, recurrence and gated norm, then vLLM's o_proj (its GEMM
    and all-reduce). Every other call (MLA layers, prefills, mixed and plain decode steps, ragged windows, a layer or
    cache outside the contract, a shape the op does not serve) goes back to vLLM before anything runs; one the op
    declines after the in-projection (it said it would serve) goes back too, and vLLM runs the in-projection again."""
    if not _ENABLED or not isinstance(self.self_attn, _kda_class()):
        return FallbackReason.OUT_OF_CONTRACT
    attn = self.self_attn
    n_tokens = _rows(hidden_states)
    step = _verify_step(attn, n_tokens)
    if isinstance(step, str):
        _once(step)
        return FallbackReason.OUT_OF_CONTRACT
    m, n_seqs, window = step
    if (reason := _layer_reason(attn, window)) is not None:
        _once(reason)
        return FallbackReason.OUT_OF_CONTRACT
    slots = _dense(m.spec_state_indices_tensor[:n_seqs])
    accepted = _dense(m.num_accepted_tokens[:n_seqs])
    if slots is None or accepted is None:
        _once("strided window slots or accepted counts inside a graph capture")
        return FallbackReason.OUT_OF_CONTRACT
    slots = slots.view(-1)
    if (reason := _serve_reason(attn, n_tokens, n_seqs, window, slots, accepted)) is not None:
        _once(f"{reason.value}: {n_seqs} sequences of {window}")
        return reason
    if "entered" not in _SEEN:
        _SEEN.add("entered")
        _LOGGER.warning("aklr kda_spec_stage_c route reached")
    proj = attn.in_proj_qkvgfab(hidden_states)[0]
    core = torch.empty((n_tokens, attn.local_num_heads, attn.head_dim), dtype=proj.dtype, device=proj.device)
    result = kimi_k3_kda_spec_stage_c(*_op_args(attn, proj, slots, accepted), **_op_kwargs(attn), out=core)
    if isinstance(result, FallbackReason):
        _once(f"{result.value} after the in-projection: {n_seqs} sequences of {window}, rows {tuple(proj.shape)}/"
              f"{proj.stride()} {proj.dtype}")  # fmt: skip
        return result
    return attn.o_proj(result.flatten(1))[0]
