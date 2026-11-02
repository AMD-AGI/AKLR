# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""An Engram layer's lookup sum (custom all-reduce in BF16: one rank owns each row, the
others hold zeros, so the sum is the owner's bits) and its gated write in one launch.
``AKLR_DSV41_ENGRAM=0`` keeps sglang's for both."""

from __future__ import annotations

import os

import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

ENV_ENGRAM = "AKLR_DSV41_ENGRAM"
ENV_ENGRAM_POST = "AKLR_DSV41_ENGRAM_POST"

SGLANG_READS = {
    "srt/layers/engram.py": (
        "EngramEmbedding",
        "_reduce_owned_rows(values)",
        "get_parallel",
        "_cuda_kernels",
        "engram_gate",
        "embed",
        "wkv",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
    ),
    "srt/distributed/parallel_state.py": (
        "outplace_all_reduce(outplace_all_reduce_method)",
        "ca_comm",
        "unique_name",
    ),
    "kernels/ops/embeddings/engram_gate.py": (
        "_engram_gate_kernel",
        "fused_engram_gate",
    ),
}

# the gate's verdict: whether its kernel wrote sglang's bits at its first eager call
_AGREED: dict[str, bool] = {}


def _enabled() -> bool:
    value = os.environ.get(ENV_ENGRAM) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_ENGRAM}={value!r}: expected '1' or '0'")
    return value == "1"


def _custom_group(values: torch.Tensor):
    """The TP group whose custom all-reduce takes ``values``, else None (with why)."""
    if not (_enabled() and torch.version.hip and values.is_cuda):
        return None
    if values.dtype != torch.bfloat16:
        stats.set_info("deepseek_v4.engram:all_reduce", f"sglang's ({values.dtype} rows)")
        return None
    from sglang.srt.layers.engram import get_parallel

    group = get_parallel().tp_group
    ca = getattr(group, "ca_comm", None)
    if ca is None or ca.disabled or not ca.should_custom_ar(values):
        if torch.cuda.is_current_stream_capturing():
            stats.set_info(
                "deepseek_v4.engram:all_reduce",
                f"sglang's (no custom all-reduce for {tuple(values.shape)})",
            )
        return None
    if torch.cuda.is_current_stream_capturing():
        stats.set_info("deepseek_v4.engram:all_reduce", "custom all-reduce (BF16)")
    return group


@sglang_route(
    "sglang.srt.layers.engram.EngramEmbedding._reduce_owned_rows",
    versions=VERSIONS,
    around=True,
)
def _reduce_owned_rows(orig, self, values):
    group = _custom_group(values) if serving() else None
    if group is None:
        return orig(self, values)
    from sglang.srt.distributed.parallel_state import outplace_all_reduce

    return outplace_all_reduce(values, group_name=group.unique_name, outplace_all_reduce_method="ca")


def _sglangs(engram, x, kv, input_ids, image_token_id):
    from sglang.srt.layers.engram import engram_gate

    out = engram_gate(x, kv, engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value)
    return torch.where((input_ids == image_token_id)[:, None, None], x, out)


def _gate_agreed(engram, x, kv, input_ids, image_token_id) -> bool | None:
    if "gate" not in _AGREED:
        if torch.cuda.is_current_stream_capturing():
            return None
        from aklr.plugins.sglang.models.deepseek_v4 import engram as kernels

        want = _sglangs(engram, x, kv, input_ids, image_token_id)
        image = input_ids.clone()
        image[::2] = image_token_id  # both kinds of rows, whatever the batch holds
        want_image = _sglangs(engram, x, kv, image, image_token_id)
        got = kernels.gate_keeping(
            x,
            kv,
            engram.q_weight,
            engram.k_weight,
            engram.eps,
            engram.clamp_value,
            input_ids,
            image_token_id,
        )
        got_image = kernels.gate_keeping(
            x,
            kv,
            engram.q_weight,
            engram.k_weight,
            engram.eps,
            engram.clamp_value,
            image,
            image_token_id,
        )
        same = torch.equal(want.view(torch.int16), got.view(torch.int16)) and torch.equal(
            want_image.view(torch.int16), got_image.view(torch.int16)
        )
        _AGREED["gate"] = same
        stats.set_info(
            "deepseek_v4.engram:gate",
            "one launch" if same else "sglang's (the kernel's bits differ)",
        )
    return _AGREED["gate"]


def forward_keeping(engram, x, hash_ids, forward_batch, input_ids, image_token_id):
    """``engram(x, hash_ids, forward_batch)`` with each image-token row kept at ``x``,
    as sglang's ``DeepseekV4Model.forward`` keeps it."""
    from sglang.srt.layers.engram import _cuda_kernels

    if not (_enabled() and x.shape[0] and _cuda_kernels(x) and x.dtype == torch.bfloat16):
        out = engram(x, hash_ids, forward_batch)
        return torch.where((input_ids == image_token_id)[:, None, None], x, out)
    emb = engram.embed(hash_ids, forward_batch)
    kv, _ = engram.wkv(emb.flatten(-2))
    return _gate(engram, x, kv, input_ids, image_token_id)


def _gate(engram, x, kv, input_ids, image_token_id):
    from aklr.plugins.sglang.models.deepseek_v4 import engram as kernels

    w = (engram.q_weight, engram.k_weight)
    if not kernels.applies(x, kv, *w, input_ids) or not _gate_agreed(
        engram, x, kv, input_ids, image_token_id
    ):
        return _sglangs(engram, x, kv, input_ids, image_token_id)
    return kernels.gate_keeping(x, kv, *w, engram.eps, engram.clamp_value, input_ids, image_token_id)


def post_enabled() -> bool:
    """Whether the layer before an Engram layer leaves its post owed to the Engram
    write (``forward_keeping_owed``); ``AKLR_DSV41_ENGRAM_POST=0``: it settles it."""
    return _enabled() and os.environ.get(ENV_ENGRAM_POST, "1") != "0"


def forward_keeping_owed(engram, layer, pending, hash_ids, forward_batch, input_ids, image_token_id):
    """``forward_keeping`` on what ``pending`` (x, residual, post, comb) settles to; an owed
    post runs in the gate's launch where its first eager call writes the bits of the post
    then the gate, else the post first."""
    from sglang.srt.layers.engram import _cuda_kernels

    from aklr.plugins.sglang.models.deepseek_v4 import engram as kernels

    x, residual, post, comb = pending
    args = (forward_batch, input_ids, image_token_id)
    if x is None:
        return forward_keeping(engram, residual, hash_ids, *args)
    if not (
        _AGREED.get("post", True)
        and residual.shape[0]
        and _cuda_kernels(residual)
        and kernels.post_applies(x, residual, post, comb)
    ):
        return forward_keeping(engram, layer.hc_post(x, residual, post, comb), hash_ids, *args)
    emb = engram.embed(hash_ids, forward_batch)
    kv, _ = engram.wkv(emb.flatten(-2))
    w = (engram.q_weight, engram.k_weight, engram.eps, engram.clamp_value)
    if "post" in _AGREED:
        return kernels.gate_keeping_post(x, residual, post, comb, kv, *w, input_ids, image_token_id)
    settled = layer.hc_post(x, residual, post, comb)
    if torch.cuda.is_current_stream_capturing():
        return _gate(engram, settled, kv, input_ids, image_token_id)
    gate_ok = kernels.applies(settled, kv, *w[:2], input_ids) and _gate_agreed(
        engram, settled, kv, input_ids, image_token_id
    )
    same = False
    if gate_ok:
        image = input_ids.clone()
        image[::2] = image_token_id  # both kinds of rows, whatever the batch holds
        same = all(
            torch.equal(
                kernels.gate_keeping(settled, kv, *w, ids, image_token_id).view(torch.int16),
                kernels.gate_keeping_post(x, residual, post, comb, kv, *w, ids, image_token_id).view(
                    torch.int16
                ),
            )
            for ids in (input_ids, image)
        )
    _AGREED["post"] = same
    why = "the kernel's bits differ" if gate_ok else "sglang's gate"
    stats.set_info(
        "deepseek_v4.engram:post",
        "in the gate's launch" if same else f"the post, then the gate ({why})",
    )
    return _gate(engram, settled, kv, input_ids, image_token_id)
