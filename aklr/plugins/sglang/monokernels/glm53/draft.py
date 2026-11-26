# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3's NextN (MTP) draft. Monokernel API version 1 routes no draft step, so at build the spec wraps the NextN
model's methods:

- the decoder layer: a draft extend (DRAFT_EXTEND_V2) runs ``glm53_moe_index_layer``, whose top-k SGLang publishes as
  the seed; a draft decode step reusing it (SGLang's MTP index share) runs ``glm53_moe_layer``;
- the NextN forward's input (embedding, enorm, hnorm, eh_proj) runs ``glm53_nextn_entry``;
- the LM head is the target's ``glm53_head`` (``spec.install_head``).

Any other call, or one at a width AKLR does not serve, runs SGLang's code. The choice reads only the step's mode,
rows and reused top-k, so a captured graph replays it."""

from __future__ import annotations

import functools

import torch
from sglang.kernels.plugin_kernels.amd.spec import MonoSpec

from aklr.plugins.sglang.monokernels.glm53 import HF_CONFIG
from aklr.plugins.sglang.monokernels.glm53 import spec as target
from aklr.plugins.sglang.monokernels.glm53.spec import INDEX, MOE

ENTRY = "glm53_nextn_entry"


def _tokens_per_request(model) -> int:
    """A draft extend's rows a request: the target's verify rows."""
    from sglang.srt.runtime_context import get_spec

    return int(get_spec().speculative_num_draft_tokens)


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """After load: the NextN layer's layout, and the MTP index share of its draft decode steps."""
    from sglang.srt.configs.model_config import get_dsa_mtp_topk_width

    decoder = model.model.decoder
    layer_id = decoder.self_attn.attn_mqa.layer_id
    if decoder.self_attn.indexer is None or not hasattr(decoder.mlp, "experts"):
        return ["needs the NextN layer an MoE one with an indexer"]
    reasons = target.layer_reasons(layer_id, decoder, INDEX) + target.index_reasons(layer_id, decoder)
    if not getattr(hf_config, "index_share_for_mtp_iteration", False):
        reasons.append("the draft decode steps need SGLang's MTP index share")
    elif get_dsa_mtp_topk_width(hf_config) != target.INDEX_TOPK:
        reasons.append(f"a reused top-k of {get_dsa_mtp_topk_width(hf_config)} keys")
    return reasons


class _Draft:
    """The NextN layer's tensors and the widths each kind's launch serves, the draft extend's top-k buffers,
    and the decoder's wrapper."""

    def __init__(self, lm, layer_id: int, args: dict, served: dict, shared: dict):
        self.lm, self.layer_id, self.args, self.served, self.shared = lm, layer_id, args, served, shared
        device = shared["cos_sin_cache"].device
        self.topk = {rows: target.topk_buffer(rows, device) for rows in served[INDEX]}

    def forward(self, plan, *args, **kwargs):
        """A step the runtime routes by the target's key, which the draft never runs: SGLang's forward."""
        return type(self.lm).forward(self.lm, *args, **kwargs)

    def _kind(self, forward_batch, rows: int, prev_topk_indices):
        mode = forward_batch.forward_mode
        if mode.is_draft_extend_v2() and rows in self.served[INDEX]:
            return INDEX
        if mode.is_decode() and prev_topk_indices is not None and rows in self.served[MOE]:
            return MOE
        return None

    def wrap(self, standard):
        """The decoder's forward: a draft step's launch, else ``standard``."""

        @functools.wraps(standard)
        def forward(positions, hidden_states, forward_batch, *args, **kwargs):
            rows, prev = hidden_states.shape[0], kwargs.get("prev_topk_indices")
            kind = self._kind(forward_batch, rows, prev)
            if kind is None:
                return standard(positions, hidden_states, forward_batch, *args, **kwargs)
            from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

            topk = self.topk[rows] if kind == INDEX else prev[:rows]
            hidden = residual_batch.fold(hidden_states, forward_batch)
            slots = forward_batch.out_cache_loc.long()
            out, topk = target.launch(
                kind,
                self.args[kind],
                self.shared,
                self.layer_id,
                positions,
                slots,
                hidden,
                forward_batch,
                topk,
            )
            return residual_batch.set_written(out, forward_batch), topk

        return forward

    def wrap_entry(self, standard, served: frozenset):
        """The NextN model's forward: a step at a width in ``served`` from the entry's launch, else ``standard``."""
        nextn = self.lm.model

        @functools.wraps(standard)
        def forward(input_ids, positions, forward_batch, input_embeds=None):
            from aklr.core.runtime.dispatch import FallbackReason

            h, rows = getattr(forward_batch.spec_info, "hidden_states", None), input_ids.shape[0]
            plain = (
                input_embeds is None
                and rows in served
                and input_ids.is_contiguous()
                and isinstance(h, torch.Tensor)
                and tuple(h.shape) == (rows, HF_CONFIG["hidden_size"])
                and h.is_contiguous()
                and not forward_batch.contains_mm_inputs()
            )
            hidden = target.api_of(ENTRY)(**_entry_call(nextn, input_ids, h)) if plain else None
            if hidden is None or isinstance(hidden, FallbackReason):
                return standard(input_ids, positions, forward_batch, input_embeds)
            from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
            from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
            from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
            from sglang.srt.utils import BumpAllocator

            # DeepseekModelNextN.forward from eh_proj's output on
            residual_batch.start(forward_batch)
            index_topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
            # the layer's launch takes no zero allocator (a zero fill in the step); SGLang's layer, if it runs, does
            zero_allocator = None
            if self._kind(forward_batch, rows, index_topk_share.topk_indices) is None:
                zero_allocator = BumpAllocator(buffer_size=2, dtype=torch.float32, device=input_ids.device)
            with get_global_expert_distribution_recorder().disable_this_region():
                hidden, topk_indices = nextn.decoder(
                    positions,
                    hidden,
                    forward_batch,
                    zero_allocator,
                    prev_topk_indices=index_topk_share.topk_indices,
                )
            hidden = residual_batch.complete_output(hidden, forward_batch)
            if not forward_batch.forward_mode.is_idle():
                hidden = residual_batch.final_norm(hidden, forward_batch, nextn.shared_head.norm)
            index_topk_share.update(topk_indices)
            index_topk_share.publish()
            return hidden

        return forward


def _entry_call(nextn, input_ids, hidden_states) -> dict:
    """api.glm53_nextn_entry's call on the step's rows and the NextN model's weights (read at the call: EAGLE may
    hand the draft the target's embedding after the build)."""
    from sglang.srt.runtime_context import get_parallel

    return {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "embed_weight": nextn.embed_tokens.weight,
        "enorm_weight": nextn.enorm.weight,
        "hnorm_weight": nextn.hnorm.weight,
        "eh_proj_weight": nextn.eh_proj.weight,
        "eps": nextn.enorm.variance_epsilon,
        "tp_size": get_parallel().tp_size,
    }


def _entry_widths(nextn, widths) -> frozenset:
    """The widths at which AKLR serves the entry on ``nextn``'s weights; none where the NextN forward is not sglang's
    plain one (an embedding other than the plain lookup, a rotated or quantized eh_proj, unlike norms)."""
    if (
        not target.plain_vocab_embedding(nextn.embed_tokens)
        or getattr(nextn, "rot_weight", None) is not None
        or not isinstance(nextn.eh_proj, torch.nn.Linear)
        or nextn.enorm.variance_epsilon != nextn.hnorm.variance_epsilon
    ):
        return frozenset()
    device = nextn.enorm.weight.device
    calls = {
        rows: _entry_call(
            nextn,
            torch.zeros(rows, dtype=torch.int64, device=device),
            torch.zeros(rows, HF_CONFIG["hidden_size"], dtype=torch.bfloat16, device=device),
        )
        for rows in widths
    }
    return target.served_widths(target.api_of(ENTRY), calls)


def _build(model, hf_config, ctx) -> _Draft:
    """The NextN layer's launches: the draft extend's (INDEX) at the routed widths, the draft decode steps'
    (MOE) at their batch sizes, each where AKLR serves it, compiled and run once on every rank outside the
    capture; then the decoder's wrapper, the logits processor's head and the NextN forward's entry."""
    decoder = model.model.decoder
    shared = target.shared_args(decoder)
    device = shared["cos_sin_cache"].device
    per = _tokens_per_request(model)
    widths = {INDEX: sorted(ctx.widths), MOE: sorted({w // per for w in ctx.widths})}
    args, served = {}, {}
    for kind, rows_list in widths.items():
        args[kind] = target.layer_call_args(decoder, kind, hf_config)
        calls = {rows: target.stand_in_call(kind, rows, device) | args[kind] | shared for rows in rows_list}
        served[kind] = target.served_widths(target.api_of(kind), calls)
    # a draft extend's selected rows and a draft decode step's: a row a request
    target.install_head(model, target.ROWS)
    entry_widths = _entry_widths(model.model, target.ROWS)
    torch.cuda.synchronize()
    draft = _Draft(model, decoder.self_attn.attn_mqa.layer_id, args, served, shared)
    decoder.forward = draft.wrap(decoder.forward)
    model.model.forward = draft.wrap_entry(model.model.forward, entry_widths)
    return draft


SPEC = MonoSpec(
    name="glm53_nextn_aklr",
    devices=target.SPEC.devices,
    features=target.SPEC.features,
    requires=target.SPEC.requires,
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=target.config_disable_reasons,
    tokens_per_request=_tokens_per_request,
)
