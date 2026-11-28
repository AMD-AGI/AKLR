# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contracts say, in torch, in float32: nothing rounded in between but what a contract
rounds. Each rank's call runs on its own device; the group's sums run in rank order on the calling
rank's, as the TP group's all-reduce and all-gather leave them on every rank.

The indexer rounds where sglang's gfx950 indexer does, as dsa_indexer_prepare's reference: wk's,
wq_b's and weights_proj's outputs, k_norm's LayerNorm, RoPE's turned pairs and the Hadamard transform
to bf16, a gate times n_index_heads^-0.5 to bf16, and every key and query to E4M3 at its power-of-two
scale."""

import contextlib

import torch

from aklr.core.contract.reference import DistOpReference
from aklr.core.contract.semantics import rounded_to_bfloat16
from aklr.plugins.sglang.layers.dsa_indexer.reference import (
    fp8_scales,
    k_norm,
    key_columns,
    restated_scale_steps,
    rope_hadamard,
    stored_index_cache,
)
from aklr.plugins.sglang.models.deepseek_v2.reference import DeepseekV2Fp8MoeReference, fp8_linear

__all__ = [
    "INDEX_DIM",
    "INDEX_PAGE",
    "ROPE_DIM",
    "Glm53DenseIndexLayerReference",
    "Glm53EmbedReference",
    "Glm53HeadReference",
    "Glm53MoeIndexLayerReference",
    "Glm53MoeLayerReference",
    "Glm53NextnEntryReference",
]

# a query head's and a KV row's RoPE part
ROPE_DIM = 64
# an indexer head's and key's width
INDEX_DIM = 128
# keys an index-K page
INDEX_PAGE = 64
# a query head's fp8 scale floors its amax here, a key's at dsa_indexer's SCALE_FLOOR
_QUERY_SCALE_FLOOR = 1e-10
# the index scores two correct kernels may order either way, relative to the row's largest |score|: a sum
# in another order moves a value's E4M3 one step (6 to 12% of it), and a head's dot product by percents.
# Measured on MI355X, 8 ranks, 48 workloads: sglang's own top-k reaches 4.9% below the reference's
# index_topk-th score, the kernel's 1.5%
_SWAP_WIDTH = 2**-4


def _rms_norm(x, weight, eps):
    return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float()


def _rope(x, cos, sin):
    """GPT-J style: each pair 2i, 2i + 1 rotated by cos[i], sin[i]."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)


def _qkv_a(call):
    """(x, q_lora, latent, k_pe): the attention's normalized input, RMSNorm(q_a), and qkv_a's other
    two parts."""
    eps = call["eps"]
    x = _rms_norm(call["hidden_states"].float(), call["input_norm_weight"], eps)
    q_lora_rank, kv_lora_rank = call["q_a_norm_weight"].shape[0], call["kv_a_norm_weight"].shape[0]
    q_a, latent, k_pe = fp8_linear(x, call["qkv_a_weight"].float(), call["qkv_a_scale"].float()).split(
        [q_lora_rank, kv_lora_rank, ROPE_DIM], dim=-1
    )
    return x, _rms_norm(q_a, call["q_a_norm_weight"], eps), latent, k_pe


def _attention(call, qkv_a, slots):
    """This rank's attention output on its ``slots`` (-1: none), before the group's sum, and its KV
    pool after the writes. Rounding point: each written row, cast to the pool's dtype."""
    _, q_lora, latent, k_pe = qkv_a
    n_heads, nope_dim, kv_lora_rank = call["w_kc"].shape
    q = fp8_linear(q_lora, call["q_b_weight"].float(), call["q_b_scale"].float())
    q_nope, q_pe = q.unflatten(-1, (n_heads, -1)).split([nope_dim, ROPE_DIM], dim=-1)
    cos, sin = call["cos_sin_cache"].float()[call["positions"]].chunk(2, dim=-1)
    q_pe, k_pe = _rope(q_pe, cos[:, None], sin[:, None]), _rope(k_pe, cos, sin)

    pool = call["kv_cache"].clone()
    rows = torch.cat([_rms_norm(latent, call["kv_a_norm_weight"], call["eps"]), k_pe], dim=-1)
    pool[call["cache_locations"]] = rows.to(pool.dtype)

    w_scale = call["w_scale"].float()
    slots = slots.long()
    read = pool.float()[slots.clamp(min=0)]  # [tokens, topk, kv_dim]
    query = torch.cat([torch.einsum("thn,hnc->thc", q_nope, call["w_kc"].float() * w_scale), q_pe], dim=-1)
    scores = torch.einsum("thc,tkc->thk", query, read) * call["sm_scale"]
    # a row that reads no slot gets 0
    weights = scores.masked_fill((slots < 0)[:, None], -torch.inf).softmax(-1).nan_to_num(0.0)
    latent_out = torch.einsum("thk,tkr->thr", weights, read[..., :kv_lora_rank])
    heads = torch.einsum("thr,hrv->thv", latent_out, call["w_vc"].float() * w_scale).flatten(1)
    return fp8_linear(heads, call["o_weight"].float(), call["o_scale"].float()), pool


def _moe(call, x):
    """This rank's MoE output on x, before the group's sum: deepseek_v2_fp8_moe's, ungrouped."""
    return DeepseekV2Fp8MoeReference().reference(
        hidden_states=x,
        router_weight=call["router_weight"],
        correction_bias=call["correction_bias"],
        w1=call["w1"],
        w1_scale=call["w1_scale"],
        w2=call["w2"],
        w2_scale=call["w2_scale"],
        topk=call["topk"],
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
        routed_scaling_factor=call["routed_scaling_factor"],
    )["out"]


def _dense_mlp(call, x):
    """This rank's MLP output on x, before the group's sum."""
    gate, up = fp8_linear(x, call["gate_up_weight"].float(), call["gate_up_scale"].float()).chunk(2, dim=-1)
    return fp8_linear(
        torch.nn.functional.silu(gate) * up, call["down_weight"].float(), call["down_scale"].float()
    )


def _layer_out(calls, rank, attentions, mlp):
    """``rank``'s stream plus the group's ``attentions`` summed, then plus the group's ``mlp``
    outputs on each rank's normalized stream summed."""
    device = attentions[rank].device
    attention = sum(a.to(device) for a in attentions)
    streams = [c["hidden_states"].float() + attention.to(a.device) for c, a in zip(calls, attentions)]
    out = sum(
        mlp(c, _rms_norm(s, c["post_norm_weight"], c["eps"])).to(device) for c, s in zip(calls, streams)
    )
    return streams[rank] + out


class Glm53MoeLayerReference(DistOpReference):
    def reference(self, calls, rank):
        """out: ``rank``'s stream plus the group's attention outputs summed, then plus the group's MoE
        outputs on each rank's stream summed; kv_cache: ``rank``'s pool after its writes."""
        attentions = [_attention(c, _qkv_a(c), c["topk_indices"]) for c in calls]
        out = _layer_out(calls, rank, [a for a, _ in attentions], _moe)
        return {"out": out.to(calls[rank]["hidden_states"].dtype), "kv_cache": attentions[rank][1]}


def _cached_keys(cache):
    """(keys [slots, INDEX_DIM], scales [slots]) the index-K cache holds (stored_index_cache's layout):
    each slot's E4M3 values and the f32 scale its four bytes make."""
    pages = cache.view(torch.uint8).reshape(cache.shape[0], -1)
    page, values, scales = key_columns(torch.arange(cache.shape[0] * INDEX_PAGE, device=cache.device))
    keys = pages[page, values].view(torch.float8_e4m3fn).float()
    return keys, pages[page, scales].contiguous().view(torch.float32)[:, 0]


def _indexed(call, x, q_lora):
    """(index_k_cache, scores [n_tokens, max_pages * INDEX_PAGE], slots alike): the indexer's cache after
    every row's key is stored, each row's index score of the key at each position (-inf past its
    seq_lens), and that key's slot."""
    # torch's ldexp (fp8_scales) launches on the current device, not on its tensors'
    with torch.cuda.device(x.device) if x.is_cuda else contextlib.nullcontext():
        positions = call["positions"].long()
        cos, sin = call["index_cos"].float()[positions], call["index_sin"].float()[positions]
        k = rounded_to_bfloat16(fp8_linear(x, call["wk_weight"].float(), call["wk_scale"].float()))
        k = rope_hadamard(k_norm(k, call["k_norm_weight"], call["k_norm_bias"], call["index_eps"]), cos, sin)
        cache = stored_index_cache(call["index_k_cache"], call["cache_locations"], k, fp8_scales(k))
        keys, key_scales = _cached_keys(cache)

        n_heads = call["weights_proj_weight"].shape[0]
        q = rounded_to_bfloat16(fp8_linear(q_lora, call["wq_b_weight"].float(), call["wq_b_scale"].float()))
        q = rope_hadamard(q.unflatten(-1, (n_heads, INDEX_DIM)), cos[:, None], sin[:, None])
        q_scales = fp8_scales(q, _QUERY_SCALE_FLOOR)
        q = (q / q_scales[..., None]).to(torch.float8_e4m3fn).float()
        gates = rounded_to_bfloat16(
            rounded_to_bfloat16(rounded_to_bfloat16(x) @ call["weights_proj_weight"].float().T)
            * n_heads**-0.5
        )
        gates = gates * q_scales * INDEX_DIM**-0.5

        pages = call["page_table"].long()
        positions = torch.arange(pages.shape[-1] * INDEX_PAGE, device=pages.device)
        slots = pages[:, positions // INDEX_PAGE] * INDEX_PAGE + positions % INDEX_PAGE
        logits = torch.einsum("thd,tpd->thp", q, keys[slots]).relu()
        scores = torch.einsum("th,thp->tp", gates, logits) * key_scales[slots]
        scores = scores.masked_fill(positions >= call["seq_lens"][:, None], -torch.inf)
        return cache, scores, slots


class Glm53MoeIndexLayerReference(DistOpReference):
    """glm53_moe_layer's attention on the layer's own top-k, then ``mlp``: the MoE."""

    mlp = staticmethod(_moe)

    def reference(self, calls, rank):
        """layer_out, topk_indices, kv_cache and index_k_cache of ``rank``: each rank's attention on its
        own indexer's top-k; and for check, the slots ``rank`` writes and its rows' index scores by slot
        (topk_id_scores, -inf where a row has no key)."""
        attentions, results = [], []
        for c in calls:
            qkv_a = _qkv_a(c)
            cache, scores, slots = _indexed(c, *qkv_a[:2])
            k = c["topk_indices"].shape[1]
            top = torch.full((len(slots), k), -1, dtype=c["topk_indices"].dtype, device=slots.device)
            for r, n in enumerate(c["seq_lens"].tolist()):
                taken = scores[r, :n].topk(k).indices if n > k else torch.arange(n, device=slots.device)
                top[r, : len(taken)] = slots[r, taken].to(top.dtype)
            attentions.append(_attention(c, qkv_a, top))
            results.append((top, cache, scores, slots))
        call = calls[rank]
        out = _layer_out(calls, rank, [a for a, _ in attentions], self.mlp)
        top, cache, scores, slots = results[rank]
        by_slot = torch.full((len(slots), cache.shape[0] * INDEX_PAGE), -torch.inf, device=scores.device)
        by_slot.scatter_(1, slots, scores)
        return {
            "layer_out": out.to(call["hidden_states"].dtype),
            "topk_indices": top,
            "kv_cache": attentions[rank][1],
            "index_k_cache": cache,
            "written_slots": call["cache_locations"],
            "topk_id_scores": by_slot,
        }

    def check(self, outputs, expected, schema):
        """``OpReference.check``, but: a key of index_k_cache stored at twice or half the reference's
        scale stands for the reference's (restated_scale_steps); and a row of topk_indices the
        reference fills (no -1) may list its slots in any order, and may take, in place of a taken
        slot whose index score is within _SWAP_WIDTH of the row's index_topk-th, a slot it leaves
        that is as near: two kernels' projections summed in another order may order those either
        way. A row the reference does not fill is its slots in order, then -1, exactly."""
        cache = restated_scale_steps(
            outputs["index_k_cache"], expected["index_k_cache"], expected["written_slots"]
        )
        actual, reference = outputs["topk_indices"].clone(), expected["topk_indices"].clone()
        for r in (reference >= 0).all(dim=-1).nonzero().flatten().tolist():
            scores, taken, got = expected["topk_id_scores"][r], reference[r].long(), actual[r].long()
            kth = scores[taken].min()
            width = _SWAP_WIDTH * scores[scores.isfinite()].abs().max()
            sure = taken[scores[taken] > kth + width]
            stands_for = (
                (got >= 0).all()
                and (got < scores.numel()).all()
                and got.unique().numel() == got.numel()
                and (scores[got.clamp(0, scores.numel() - 1)] >= kth - width).all()
                and torch.isin(sure, got).all()
            )
            reference[r] = reference[r].sort().values
            actual[r] = reference[r] if stands_for else actual[r].sort().values
        return super().check(
            {**outputs, "index_k_cache": cache, "topk_indices": actual},
            {**expected, "topk_indices": reference},
            schema,
        )


class Glm53DenseIndexLayerReference(Glm53MoeIndexLayerReference):
    """glm53_moe_index_layer's, the dense MLP in the MoE's place."""

    mlp = staticmethod(_dense_mlp)


class Glm53HeadReference(DistOpReference):
    def reference(self, calls, rank):
        """logits: each rank's hidden_states times its lm_head_weight^T rounded to bf16, gathered in
        rank order."""
        device = calls[rank]["hidden_states"].device
        shards = [
            rounded_to_bfloat16(c["hidden_states"].float() @ c["lm_head_weight"].float().T).to(device)
            for c in calls
        ]
        return {"logits": torch.cat(shards, dim=-1)}


class Glm53NextnEntryReference(DistOpReference):
    def reference(self, calls, rank):
        """out: eh_proj(enorm(e) | hnorm(hidden_states)), rank q's columns from rank q's call, in rank
        order; e a token's row of the rank holding it; the norms' outputs and the sums rounded to bf16."""
        device = calls[rank]["hidden_states"].device
        vocab_rows = calls[0]["embed_weight"].shape[0]
        cols = calls[0]["eh_proj_weight"].shape[0] // len(calls)
        shares = []
        for q, call in enumerate(calls):
            ids = call["input_ids"].tolist()
            e = torch.stack([calls[t // vocab_rows]["embed_weight"][t % vocab_rows].to(device) for t in ids])
            hidden = call["hidden_states"].float().to(device)
            x = torch.cat(
                [
                    rounded_to_bfloat16(_rms_norm(e.float(), call["enorm_weight"].to(device), call["eps"])),
                    rounded_to_bfloat16(_rms_norm(hidden, call["hnorm_weight"].to(device), call["eps"])),
                ],
                dim=-1,
            )
            shares.append(x @ call["eh_proj_weight"][q * cols : (q + 1) * cols].float().to(device).T)
        return {"out": torch.cat(shares, dim=-1).to(calls[rank]["hidden_states"].dtype)}


class Glm53EmbedReference(DistOpReference):
    def reference(self, calls, rank):
        """embeddings: as the vocab-parallel embedding's all-reduce leaves them on every rank, the group's
        sum, in rank order, of each rank's lookup of its own token ids in its rows of the vocabulary (rank
        q's from q * vocab_rows; zeros for an id another rank holds), in the table's dtype."""
        device, total = calls[rank]["embed_weight"].device, 0
        for q, c in enumerate(calls):
            table = c["embed_weight"]
            local = c["input_ids"].long() - q * table.shape[0]
            held = (local >= 0) & (local < table.shape[0])
            total = total + (table.float()[local.clamp(0, table.shape[0] - 1)] * held[:, None]).to(device)
        return {"embeddings": total.to(calls[rank]["embed_weight"].dtype)}
