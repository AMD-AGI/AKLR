# Created by OpenAI
"""Sequential mathematical oracle; deliberately independent of the Gluon kernels.

Inputs use contiguous Q/K/V/Z and B/A projections. FP64 recurrence provides a
numerical oracle; BF16 interfaces and the FP8 quantization boundary are explicit.
This implementation is for correctness and must not be used as a speed baseline.
"""

import inspect
import itertools

import torch
import torch.nn.functional as F


def reference_prefill(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    cu_seqlens,
    has_initial_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    norm_weight,
    *,
    scale,
    eps=1e-6,
):
    slots, heads, value_dim, key_dim = delta_state.shape
    channels = conv_state.shape[1]
    key_heads = (channels - heads * value_dim) // (2 * key_dim)
    if heads % key_heads or norm_weight.numel() != value_dim:
        raise ValueError("unsupported head dimensions")
    rows = projected_qkvz.shape[0]
    starts = cu_seqlens.cpu().tolist()
    indices = cache_indices.cpu().tolist()
    initial = has_initial_state.cpu().tolist()
    if (
        len(starts) != len(indices) + 1
        or len(initial) != len(indices)
        or starts[0] != 0
        or starts[-1] != rows
        or any(a >= b for a, b in itertools.pairwise(starts))
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= slots for i in indices)
    ):
        raise ValueError("invalid sequence or state-slot metadata")
    conv_result, state_result = conv_state.clone(), delta_state.clone()
    output = torch.empty((rows, heads, value_dim), dtype=torch.bfloat16, device=projected_qkvz.device)
    ratio = heads // key_heads
    q_width, k_width = key_heads * key_dim, key_heads * key_dim
    taps = conv_weight.shape[1]
    if conv_state.shape[2] != taps - 1:
        raise ValueError("convolution history and taps disagree")
    for seq, slot in enumerate(indices):
        history = conv_state[slot].clone() if initial[seq] else torch.zeros_like(conv_state[slot])
        state = (
            delta_state[slot].double()
            if initial[seq]
            else torch.zeros_like(delta_state[slot], dtype=torch.float64)
        )
        for token in range(starts[seq], starts[seq + 1]):
            raw = projected_qkvz[token, :channels]
            window = torch.cat((history, raw[:, None]), dim=1)
            conv = (window.float() * conv_weight.float()).sum(dim=1) + conv_bias.float()
            activated = F.silu(conv).to(torch.bfloat16)
            history = window[:, 1:]
            q = activated[:q_width].reshape(key_heads, key_dim).float()
            k = activated[q_width : q_width + k_width].reshape(key_heads, key_dim).float()
            q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).double()
            k = (k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).double()
            q, k = q.repeat_interleave(ratio, 0), k.repeat_interleave(ratio, 0)
            v = activated[q_width + k_width :].reshape(heads, value_dim).double()
            b, a = projected_ba[token].float().split(heads)
            beta = b.sigmoid().to(torch.bfloat16).double()
            decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(a + dt_bias.float())).double()
            state = state * decay[:, None, None]
            prediction = (state * k[:, None, :]).sum(dim=-1)
            update = (v - prediction) * beta[:, None]
            state = state + update[:, :, None] * k[:, None, :]
            core = ((state * q[:, None, :]).sum(dim=-1) * scale).to(torch.bfloat16).float()
            gate = projected_qkvz[token, channels:].reshape(heads, value_dim).float()
            normalized = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
            output[token] = (normalized * norm_weight.float() * F.silu(gate)).to(torch.bfloat16)
        conv_result[slot] = history
        state_result[slot] = state.float()
    scales = output.float().abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    values = (output.float() / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return output, conv_result, state_result, values.reshape(rows, -1), scales


def reference_decode(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    *,
    scale,
    eps=1e-6,
):
    """One decode token per row, in float64 between the BF16 and FP8 rounding points.

    Rounding points: each convolution tap's product to BF16, summed in FP32 oldest
    tap first (as SGLang's decode convolution does); the SiLU convolution output, beta
    and the core output to BF16 (the tensors SGLang keeps between its decode
    kernels); the gated RMS-normalized output to BF16, then the per-head E4M3FN
    values. A row with cache index -1 is padding: its core output is zero and no
    state changes.
    """
    slots, heads, value_dim, key_dim = delta_state.shape
    channels = conv_state.shape[1]
    key_heads = (channels - heads * value_dim) // (2 * key_dim)
    if heads % key_heads or norm_weight.numel() != value_dim:
        raise ValueError("unsupported head dimensions")
    if conv_state.shape[2] != conv_weight.shape[1] - 1:
        raise ValueError("convolution history and taps disagree")
    rows = projected_qkvz.shape[0]
    indices = cache_indices.tolist()
    live = [i for i in indices if i != -1]
    if len(indices) != rows or len(set(live)) != len(live) or any(i < -1 or i >= slots for i in indices):
        raise ValueError("invalid state-slot metadata")
    conv_result, state_result = conv_state.clone(), delta_state.clone()
    core = torch.zeros((rows, heads, value_dim), dtype=torch.float64, device=projected_qkvz.device)
    ratio, width = heads // key_heads, key_heads * key_dim
    for row, slot in enumerate(indices):
        if slot == -1:
            continue
        window = torch.cat((conv_state[slot], projected_qkvz[row, :channels, None]), 1)
        # each tap's product rounds to BF16; FP32 sums them, the oldest tap first
        taps = (window.float() * conv_weight.float()).to(torch.bfloat16).float()
        conv = ((taps[:, 0] + taps[:, 1]) + taps[:, 2]) + taps[:, 3]
        activated = F.silu(conv.double()).to(torch.bfloat16).double()
        conv_result[slot] = window[:, 1:]
        q = activated[:width].reshape(key_heads, key_dim)
        k = activated[width : 2 * width].reshape(key_heads, key_dim)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q, k = q.repeat_interleave(ratio, 0), k.repeat_interleave(ratio, 0)
        v = activated[2 * width :].reshape(heads, value_dim)
        b, a = projected_ba[row].double().split(heads)
        beta = b.sigmoid().to(torch.bfloat16).double()
        decay = torch.exp(-torch.exp(a_log.double()) * F.softplus(a + dt_bias.double()))
        state = delta_state[slot].double() * decay[:, None, None]
        update = (v - (state * k[:, None, :]).sum(dim=-1)) * beta[:, None]
        state = state + update[:, :, None] * k[:, None, :]
        core[row] = (state * q[:, None, :]).sum(dim=-1).to(torch.bfloat16).double()
        state_result[slot] = state.float()
    gate = projected_qkvz[:, channels:].reshape(rows, heads, value_dim).double()
    normalized = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + eps)
    normalized = (normalized * norm_weight.double() * F.silu(gate)).to(torch.bfloat16)
    scales = normalized.double().abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    values = (normalized.double() / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return normalized, conv_result, state_result, values.reshape(rows, -1), scales


def _decode_reference(point, row_major_args, decode, outputs):
    names = inspect.signature(reference_decode).parameters
    normalized, conv, state, values, scales = reference_decode(
        **{k: v for k, v in row_major_args.items() if k in names}
    )
    results = {
        "normalized": normalized,
        "values": values,
        "scales": scales,
        "conv_state": conv,
        "delta_state": state,
    }
    return tuple(decode(results[n], point.dtypes[n]) for n in (*outputs, "conv_state", "delta_state"))


def gdn_decode_group_fp8_quant_reference(point, row_major_args, decode):
    """Independent decode; the FP8 outputs, then both final state pools."""
    return _decode_reference(point, row_major_args, decode, ("values", "scales"))


def gdn_decode_reference(point, row_major_args, decode):
    """Independent decode; the BF16 output, then both final state pools."""
    return _decode_reference(point, row_major_args, decode, ("normalized",))


def gdn_prefill_group_fp8_quant_reference(point, row_major_args, decode):
    """Independent recurrence; return outputs followed by final mutable inputs."""
    call = {k: v for k, v in row_major_args.items() if k != "max_seq_len"}
    output, conv, state, values, scales = reference_prefill(**call)
    return tuple(
        decode(t, point.dtypes[n])
        for n, t in zip(
            ("normalized", "values", "scales", "conv_state", "delta_state"),
            (output, values, scales, conv, state),
            strict=True,
        )
    )
