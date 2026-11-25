###############################################################################
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

from __future__ import annotations

from dataclasses import replace

import torch

from aklr.core.schema import Const, Schema, Standard, Tensor, Var, default_check
from aklr.plugins.aiter.kda_conv.baseline import causal_conv1d_split_qkv_baseline
from aklr.plugins.aiter.kda_conv.reference import kimi_k3_kda_conv_reference

__all__ = ["KIMI_K3_KDA_CONV", "KIMI_K3_KDA_CONV_STRIDED_ROWS", "KIMI_K3_KDA_CONV_VARIANTS"]


def _kda_conv_check(reference_outputs, actual_outputs, **_unused):
    # a bf16 rounding of a float32 sum of four products and the silu: about 2**-8 relative, and small outputs
    # absolute
    return default_check(reference_outputs, actual_outputs, rtol=2**-7, atol=2**-9)


def _kda_conv_inputs(n_tokens, n_seqs, n_slots, proj, width, row_stride, device, seed):
    torch.manual_seed(seed)
    dim, state_len = 3 * proj, width - 1
    # x and the state are the leading values of rows row_stride times as wide: row_stride 1 is row-major
    x = torch.zeros(n_tokens, row_stride * dim, dtype=torch.bfloat16, device=device)[:, :dim]
    x.copy_(torch.randn(n_tokens, dim, device=device))
    conv_state = torch.zeros(n_slots, row_stride * state_len * dim, dtype=torch.bfloat16, device=device)
    conv_state = conv_state[:, : state_len * dim]
    conv_state.copy_(torch.randn(n_slots, state_len * dim, device=device))
    # each sequence has at least a token; the spare ones are split by weights exp(N(0, 1) clipped to +-2)
    spare = n_tokens - n_seqs
    shares = torch.randn(n_seqs, dtype=torch.float64).clamp(-2, 2).exp().cumsum(0)
    ends = (shares / shares[-1:] * spare).floor().long()
    ends[-1:] = spare
    ends += torch.arange(1, n_seqs + 1)
    return {
        "x": x.unflatten(1, (3, proj)),
        # lecun normal over the taps
        "weight": torch.randn(dim, width, device=device) * width**-0.5,
        "conv_state": conv_state.unflatten(1, (state_len, dim)),
        "state_indices": torch.randperm(n_slots)[:n_seqs].to(torch.int32).to(device),
        "has_initial_state": (torch.rand(n_seqs) < 0.5).to(device),
        "cu_seqlens": torch.cat((ends.new_zeros(1), ends)).to(torch.int32).to(device),
        "storage": "row_major" if row_stride == 1 else "strided_rows",
    }


def _kda_conv_init(*, n_tokens, n_seqs, n_slots, proj, width, device="cuda", seed=0):
    return _kda_conv_inputs(n_tokens, n_seqs, n_slots, proj, width, 1, device, seed)


def _kda_conv_strided_rows_init(*, n_tokens, n_seqs, n_slots, proj, width, device="cuda", seed=0):
    return _kda_conv_inputs(n_tokens, n_seqs, n_slots, proj, width, 2, device, seed)


def _out(description: str) -> Tensor:
    return Tensor(["n_tokens", "proj"], description=description, dtype_from="x")


# shortcut: conv_state is written in place and is no output, so no check reads the state a kernel leaves;
# declare it checked once a Standard's check is given the inputs after the call
KIMI_K3_KDA_CONV = Schema(
    op_type="kimi_k3_kda_conv",
    variant={"default": "storage == 'row_major'"},
    description=(
        "Kimi-K3's KDA short convolution as vLLM runs it on ROCm (kda.py's fused q/k/v causal conv1d, aiter's "
        "causal_conv1d_split_qkv_triton_tile_fn), every sequence's tokens back to back: sequence s owns rows "
        "cu_seqlens[s]:cu_seqlens[s + 1] of x and slot state_indices[s] of conv_state. Per channel c of the "
        "packed q|k|v (c = part * proj + j for x[:, part, j]), the window is conv_state[slot, :, c] (when "
        "has_initial_state[s], else width - 1 zeros) then the sequence's inputs; token t's y[t, c] = "
        "silu(sum_j weight[c, j] * window[t + j]) in float32, rounded to x's dtype; the window's last width - 1 "
        "entries are written back to conv_state[slot, :, c]. q = y[:, :proj], k = y[:, proj:2 * proj], "
        "v = y[:, 2 * proj:]. Slots no sequence names are left unchanged."
    ),
    axes={
        "n_tokens": Var("Tokens of every sequence, back to back."),
        "n_seqs": Var("Sequences."),
        "n_offsets": Var("Sequence offsets: sequences + 1."),
        "n_slots": Var("Slots of the convolution state."),
        "proj": Const("Channels of each of q, k and v on this rank."),
        "width": Const("The convolution's taps."),
        "dim": Const("Channels of the packed q|k|v: 3 * proj.", abbrev=""),
        "state_len": Const("Inputs a slot keeps: width - 1.", abbrev=""),
        "parts": Const("q, k and v.", abbrev="", value=3),
    },
    constraints=[
        "dim == 3 * proj",
        "state_len == width - 1",
        "n_offsets == n_seqs + 1",
        "n_seqs <= n_tokens",
        "n_seqs <= n_slots",
        "x.dtype == 'bfloat16'",
        "conv_state.dtype == x.dtype",
    ],
    inputs={
        "x": Tensor(
            ["n_tokens", "parts", "proj"],
            description=(
                "The packed q|k|v projection before the convolution: x[t, 0] is q's, x[t, 1] k's, x[t, 2] v's."
            ),
        ),
        "weight": Tensor(
            ["dim", "width"],
            description="Each packed channel's taps, the oldest input's first; float32, as vLLM keeps Kimi-K3's.",
        ),
        "conv_state": Tensor(
            ["n_slots", "state_len", "dim"],
            description=(
                "The paged convolution state, vLLM's SD layout: per slot, its last width - 1 inputs, oldest "
                "first, each a row of every packed channel (vLLM's slot may hold more rows, for speculative "
                "decoding: the first width - 1 are these). Written in place."
            ),
        ),
        "state_indices": Tensor(["n_seqs"], description="Each sequence's slot, distinct."),
        "has_initial_state": Tensor(
            ["n_seqs"], description="Whether each sequence resumes from its slot's state, else from zeros."
        ),
        "cu_seqlens": Tensor(
            ["n_offsets"], description="Where each sequence's tokens start, then where the last ends."
        ),
    },
    outputs={
        "q": _out("Queries after the convolution."),
        "k": _out("Keys after the convolution."),
        "v": _out("Values after the convolution."),
    },
    standard=Standard(
        reference=kimi_k3_kda_conv_reference,
        check=_kda_conv_check,
        init=_kda_conv_init,
        baseline=causal_conv1d_split_qkv_baseline,
    ),
)
# vLLM's: x the q|k|v columns of a wider projection row, conv_state slots apart by a page
KIMI_K3_KDA_CONV_STRIDED_ROWS = replace(
    KIMI_K3_KDA_CONV,
    variant={"strided_rows": "storage == 'strided_rows'"},
    description=(
        f"{KIMI_K3_KDA_CONV.description} x's rows and conv_state's slots are dense and more than a row or a "
        "slot apart."
    ),
    standard=replace(KIMI_K3_KDA_CONV.standard, init=_kda_conv_strided_rows_init),
)

# the op's variants, in resolution order: its api takes the first whose condition matches the call
KIMI_K3_KDA_CONV_VARIANTS = (KIMI_K3_KDA_CONV, KIMI_K3_KDA_CONV_STRIDED_ROWS)
