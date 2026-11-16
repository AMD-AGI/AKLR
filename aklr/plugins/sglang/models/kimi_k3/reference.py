# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What kimi_k3_latent_moe's contract says, in torch, in fp32: every input decoded to fp32 and
nothing rounded or quantized in between but what the contract rounds."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from aklr.core.contract.semantics import rounded_to_bfloat16

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "kimi_k3_latent_moe_reference",
    "kimi_k3_mla_decode_reference",
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
    decode: Callable[[Tensor, str], Tensor],
) -> tuple[Tensor, Tensor]:
    """The two outputs, latent then shared. Rounding points: the front projection, exact
    products summed in float64, rounds once to bf16; each routed linear's input goes to MXFP8
    per 1 x 32 and the routed SiTU product to bf16, as aiter's a8w4 experts take them."""
    import torch

    args, axes = row_major_args, point.axes

    def f(name: str, index: Any = None) -> Tensor:
        tensor = args[name] if index is None else args[name][index]
        return decode(tensor, point.dtypes[name]).float()

    front = rounded_to_bfloat16(
        decode(args["hidden_states"], point.dtypes["hidden_states"]).double()
        @ decode(args["front_weight"], point.dtypes["front_weight"]).double().T
    ).float()
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


def kimi_k3_mla_decode_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Each row's softmax over its slots of q . row * sm_scale, applied to the rows' first 512
    values; no rounding point."""
    import torch

    args = row_major_args
    q = decode(args["q"], point.dtypes["q"]).float()
    cache = decode(args["kv_cache"], point.dtypes["kv_cache"]).float()[:, 0]
    indptr, indices = args["kv_indptr"].long().tolist(), args["kv_indices"].long()
    output = q.new_zeros((q.shape[0], q.shape[1], 512))
    for r in range(q.shape[0]):
        rows = cache[indices[indptr[r] : indptr[r + 1]]]
        scores = (q[r] @ rows.T) * float(args["sm_scale"])
        output[r] = torch.softmax(scores, dim=-1) @ rows[:, :512]
    return output


def kimi_k3_mla_vc_output_gate_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Rounding points: the value projection and the sigmoid each round to bf16 before their
    product."""
    import torch

    args = row_major_args
    latent = decode(args["latent"], point.dtypes["latent"]).double()
    weight = decode(args["vc_weight"], point.dtypes["vc_weight"]).double()
    gate = decode(args["gate"], point.dtypes["gate"]).double()
    values = rounded_to_bfloat16(torch.einsum("thk,hvk->thv", latent, weight)).flatten(1)
    return (values * rounded_to_bfloat16(torch.sigmoid(gate))).float()


def kimi_k3_mla_prefill_gate_reference(
    point: Any,
    row_major_args: Mapping[str, Any],
    decode: Callable[[Tensor, str], Tensor],
) -> Tensor:
    """Rounding points: the attention output and the sigmoid each round to bf16 before their
    product."""
    import torch

    args = row_major_args
    q, k, v = (decode(args[n], point.dtypes[n]).float() for n in ("q", "k", "v"))
    gate = decode(args["gate"], point.dtypes["gate"]).float()
    attention = torch.empty_like(v)
    bounds = args["qo_indptr"].long().tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        scores = torch.einsum("qhd,khd->hqk", q[start:end], k[start:end]) * float(args["sm_scale"])
        causal = torch.ones(end - start, end - start, dtype=torch.bool, device=q.device).tril()
        weights = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        attention[start:end] = torch.einsum("hqk,khd->qhd", weights, v[start:end])
    return rounded_to_bfloat16(attention).flatten(1) * rounded_to_bfloat16(torch.sigmoid(gate))
