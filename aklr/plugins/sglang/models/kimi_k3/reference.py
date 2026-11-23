# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the Kimi-K3 contracts say, in torch: every input decoded and nothing rounded or quantized
in between but what the contract rounds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16, to_float32

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "kimi_k3_kda_verify_reference",
    "kimi_k3_latent_moe_reference",
    "kimi_k3_mla_prefill_gate_reference",
    "kimi_k3_mla_vc_output_gate_reference",
]

_BLOCK = 32


def _situ(gate: Tensor, up: Tensor) -> Tensor:
    import torch

    return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))


def _mxfp8(x: Tensor) -> Tensor:
    """``x`` in fp8 e4m3 at an e8m0 scale per 1 x 32, the scale 2**ceil(log2(amax / 448))."""
    import torch

    blocks = x.unflatten(-1, (-1, _BLOCK))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    mantissa, exponent = torch.frexp(amax / 448.0)
    exponent = (exponent - 1 + (mantissa != 0.5).int()).clamp(-126, 127)
    scale = torch.exp2(exponent.to(x.dtype))
    return ((blocks / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale).flatten(-2)


def kimi_k3_latent_moe_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """The two outputs, latent then shared. Rounding points: the front projection rounds to
    bf16; each routed linear's input goes to MXFP8 per 1 x 32 and the routed SiTU product to
    bf16, as aiter's a8w4 experts take them."""
    import torch

    args, axes = row_major_args, point.axes

    def f(name: str, index: Any = None) -> Tensor:
        tensor = args[name] if index is None else args[name][index]
        return to_float32(tensor, point.dtypes[name])

    front = rounded_to_bfloat16(f("hidden_states") @ f("front_weight").T)
    si, routed = axes["shared_inter_dim"], axes["n_routed"]
    gate, up, logits, z = front.split((si, si, routed, axes["latent_dim"]), dim=-1)

    shared = rounded_to_bfloat16(_situ(gate, up)) @ f("shared_w2").T

    scores = logits.sigmoid()
    ids = (scores + f("correction_bias")).topk(point.attrs["topk"], dim=-1, sorted=True).indices
    weights = scores.gather(-1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    def weight(name: str, e: int) -> Tensor:
        return f(name, e) * f(f"{name}_scale", e).repeat_interleave(_BLOCK, dim=-1)

    x = _mxfp8(z)
    latent = torch.zeros_like(z)
    for e in ids.unique().tolist():
        tokens, slots = (ids == e).nonzero(as_tuple=True)
        g, u = (x[tokens] @ weight("w1", e).T).chunk(2, dim=-1)
        h = rounded_to_bfloat16(_situ(g, u))
        latent.index_add_(0, tokens, weights[tokens, slots, None] * (_mxfp8(h) @ weight("w2", e).T))
    return latent, shared


def kimi_k3_mla_vc_output_gate_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """Rounding points: the value projection and the sigmoid each round to bf16 before their
    product."""
    import torch

    args = row_major_args
    latent = to_float32(args["latent"], point.dtypes["latent"])
    weight = to_float32(args["vc_weight"], point.dtypes["vc_weight"])
    gate = to_float32(args["gate"], point.dtypes["gate"])
    values = rounded_to_bfloat16(torch.einsum("thk,hvk->thv", latent, weight)).flatten(1)
    return values * rounded_to_bfloat16(torch.sigmoid(gate))


def kimi_k3_mla_prefill_gate_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> Tensor:
    """Rounding points: the attention output and the sigmoid each round to bf16 before their
    product."""
    import torch

    args = row_major_args
    q, k, v = (to_float32(args[n], point.dtypes[n]) for n in ("q", "k", "v"))
    gate = to_float32(args["gate"], point.dtypes["gate"])
    attention = torch.empty_like(v)
    bounds = args["qo_indptr"].long().tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        scores = torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]) * float(args["sm_scale"])
        causal = torch.ones(end - start, end - start, dtype=torch.bool, device=q.device).tril()
        weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        attention[start:end] = torch.einsum("hqk,khd->qhd", weights, v[start:end])
    return rounded_to_bfloat16(attention).flatten(1) * rounded_to_bfloat16(torch.sigmoid(gate))


def kimi_k3_kda_verify_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
) -> tuple[Tensor, ...]:
    """normalized, then the mutable inputs as the call leaves them: conv_state, conv_windows and
    the four rings. Rounding points: each convolution output (what the rings hold), forget and
    the core before its norm round to bf16."""
    import torch

    args, axes = row_major_args, point.axes

    def f(name: str) -> Tensor:
        return to_float32(args[name], point.dtypes[name])

    qkvg, bfa = f("projected_qkvg"), f("projected_bfa")
    requests, steps, _ = qkvg.shape
    heads, dim, rank = axes["n_heads"], axes["head_dim"], axes["gate_rank"]
    width = heads * dim
    slots, rows = args["cache_indices"].long(), args["window_indices"].long()
    # to_float32 may return its input itself: the mutable tensors are written in copies
    conv_state, windows = f("conv_state").clone(), f("conv_windows").clone()
    rings = {n: f(n).clone() for n in ("ring_raw_v", "ring_raw_k", "ring_gate", "ring_beta")}

    raw = torch.cat((conv_state[slots], qkvg[..., : 3 * width]), dim=1)
    taps = f("conv_weight")
    z = sum(raw[:, j : j + steps] * taps[:, j] for j in range(4))
    conv = rounded_to_bfloat16(torch.nn.functional.silu(z))
    for t in range(steps):
        windows[rows, t] = raw[:, t + 1 : t + 4]
    conv_state[slots] = raw[:, steps:]
    q, k, v = (x.unflatten(-1, (heads, dim)) for x in conv.split(width, dim=-1))

    forget = rounded_to_bfloat16(bfa[..., :rank] @ f("f_b_weight").T).unflatten(-1, (heads, dim))
    magnitude = torch.exp(f("a_log"))[:, None]
    gate = args["lower_bound"] * torch.sigmoid(magnitude * (forget + f("dt_bias").view(heads, dim)))
    beta = torch.sigmoid(bfa[..., rank : rank + heads])
    for name, value in (("ring_raw_v", v), ("ring_raw_k", k), ("ring_gate", gate), ("ring_beta", beta)):
        rings[name][slots, :, :steps] = value.transpose(1, 2)

    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * args["scale"]
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state = f("kda_state")[slots]
    core = torch.empty_like(v)
    for t in range(steps):
        state = state * torch.exp(gate[:, t, :, None, :])
        update = (v[:, t] - (state * k[:, t, :, None, :]).sum(-1)) * beta[:, t, :, None]
        state = state + update[..., None] * k[:, t, :, None, :]
        core[:, t] = (state * q[:, t, :, None, :]).sum(-1)
    core = rounded_to_bfloat16(core)
    g = qkvg[..., 3 * width :].unflatten(-1, (heads, dim))
    normalized = core * torch.rsqrt(core.square().mean(-1, keepdim=True) + args["eps"])
    normalized = normalized * f("norm_weight") * torch.sigmoid(g)
    return (normalized, conv_state, windows, *rings.values())
