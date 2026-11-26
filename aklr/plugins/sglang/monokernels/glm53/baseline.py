# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What sglang 0.5.23 runs on gfx950 (SGLANG_USE_AITER=1, --attention-backend dsa with tilelang decode,
--kv-cache-dtype fp8_e4m3 or bfloat16, aiter's MoE, the block-FP8 linears in aiter's (16, 16) layout) for a GLM-5.3
decoder layer in a decode or MTP verify step, call for call as its decode CUDA graph holds them:
DeepseekV2DecoderLayer with forward_absorb_rocm_prepare and forward_absorb_rocm_core (MLA_ROCM, the
q absorb, RoPE and KV write fused, no alt stream), on an indexer layer the gfx950 fused DSA indexer
(SGLANG_DSA_FUSE_TOPK), DeepseekV2MoE.forward_normal or DeepseekV2MLP, and the layer boundary's sums,
aiter's fused all-reduce + RMSNorm (enable_aiter_allreduce_fusion, which sglang turns on for GLM-5.3's
decode). SGLang folds a layer's input RMSNorm into the previous layer's last sum; the layer here
starts at a complete stream, so it runs that RMSNorm alone (as sglang's first layer does) and ends
with the fused sum + RMSNorm whose normalized output is the next layer's input, by this layer's
input_norm_weight. The group's sums take CustomAllreduce's copy-in path: sglang registers the
graph's buffers after its capture, which the bench's capture does not."""

from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from aklr.core.contract.baseline import DistOpBaseline
from aklr.plugins.sglang.layers.dsa_indexer.baseline import replicated_linear
from aklr.plugins.sglang.models.deepseek_v2.baseline import DeepseekV2Fp8MoeBaseline
from aklr.plugins.sglang.monokernels.glm53.reference import ROPE_DIM

__all__ = [
    "Glm53DenseIndexLayerBaseline",
    "Glm53EmbedBaseline",
    "Glm53HeadBaseline",
    "Glm53MoeIndexLayerBaseline",
    "Glm53MoeLayerBaseline",
    "Glm53NextnEntryBaseline",
]

_BLOCK = [128, 128]  # the linears' weight_block_size


def _held(tensor, name, make):
    """``make(tensor)``, made once per tensor as sglang holds it from the weights' loading on."""
    held = getattr(tensor, name, None)
    if held is None:
        held = make(tensor)
        setattr(tensor, name, held)
    return held


def _linear(x, weight, scale):
    """A block-FP8 linear's Fp8LinearMethod.apply on aiter: aiter_w8a8_block_fp8_linear."""
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    return aiter_w8a8_block_fp8_linear(x, weight, _BLOCK, scale)


def _all_reduce_rms_norm(comm, partial, residual, weight, eps):
    """GroupCoordinator.fused_allreduce_rmsnorm on the rank's CustomAllreduce: (RMSNorm of the
    group's sum plus residual, that sum plus residual)."""
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    fused = None
    if comm is not None:
        fused = GroupCoordinator.fused_allreduce_rmsnorm(
            SimpleNamespace(ca_comm=comm), partial, residual, weight, eps
        )
    if fused is None:
        raise NotImplementedError("sglang sums the group apart from the RMSNorm here")
    return fused


def _q_and_latent(call):
    """forward_absorb_rocm_prepare up to q_b_proj: (x, q_lora, q, k_nope, k_pe), x the input RMSNorm's
    output (RMSNorm.forward_aiter) and q_lora q_a's (aiter's fused q/kv norm)."""
    from sglang.srt.layers import layernorm
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
        fused_qk_rmsnorm_bf16,
    )

    eps, kv_lora_rank = call["eps"], call["kv_a_norm_weight"].shape[0]
    x = layernorm.rms_norm(call["hidden_states"], call["input_norm_weight"], eps)
    q, latent = _linear(x, call["qkv_a_weight"], call["qkv_a_scale"]).split(
        [call["q_a_norm_weight"].shape[0], kv_lora_rank + ROPE_DIM], dim=-1
    )
    q_lora, k_nope = fused_qk_rmsnorm_bf16(
        q, call["q_a_norm_weight"], eps, latent[..., :kv_lora_rank], call["kv_a_norm_weight"], eps
    )
    n_heads = call["w_kc"].shape[0]
    q = _linear(q_lora, call["q_b_weight"], call["q_b_scale"]).view(
        -1, n_heads, call["q_b_weight"].shape[0] // n_heads
    )
    return x, q_lora, q, k_nope.unsqueeze(1), latent[..., kv_lora_rank:].unsqueeze(1)


def _attention(call, q, k_nope, k_pe, topk_indices):
    """forward_absorb_rocm_core from the split q on: aiter's fused q absorb, RoPE and KV write
    (_fused_bmm_rope_cat_and_cache), the tilelang sparse decode (DeepseekSparseAttnBackend's
    _forward_tilelang on the fused top-k), rocm_absorb_v_bmm and o_proj; the rank's output."""
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.rocm_linear_utils import fused_fp8_bmm_rope_cat_and_cache_mla
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
        rocm_absorb_v_bmm,
    )

    w_kc, w_vc, kv_cache = call["w_kc"], call["w_vc"], call["kv_cache"]
    n_heads, nope_dim, kv_lora_rank = w_kc.shape
    # sglang's w_kc and w_vc: each head's [kv_lora_rank, nope_dim] and [v_head_dim, kv_lora_rank] row-major
    w_kc_t = _held(w_kc, "_aklr_sglang_t", lambda t: t.transpose(1, 2).contiguous())
    w_vc_sglang = _held(w_vc, "_aklr_sglang", lambda t: t.transpose(1, 2).contiguous().transpose(1, 2))
    # aiter's RotaryEmbedding holds its tables in the model's dtype
    cos, sin = _held(
        call["cos_sin_cache"],
        "_aklr_sglang_bf16",
        lambda t: tuple(c.contiguous() for c in t.to(torch.bfloat16).chunk(2, dim=-1)),
    )
    pool = kv_cache.view(kv_cache.shape[0], 1, -1)
    q_nope, q_pe = q.split([nope_dim, ROPE_DIM], dim=-1)
    q_all, _, _, _ = fused_fp8_bmm_rope_cat_and_cache_mla(
        q_nope.transpose(0, 1),
        w_kc_t,
        call["w_scale"],
        q_pe,
        k_nope,
        k_pe,
        pool,
        call["cache_locations"].contiguous(),
        call["positions"],
        cos,
        sin,
        group_size=128,
        k_scale=None,
        is_neox=False,
        q_out_dtype=kv_cache.dtype if kv_cache.dtype == torch.float8_e4m3fn else q_nope.dtype,
    )
    attended = DeepseekSparseAttnBackend._forward_tilelang(
        None,
        q_all=q_all,
        kv_cache=pool,
        v_head_dim=kv_lora_rank,
        page_table_1=topk_indices,
        sm_scale=call["sm_scale"],
    )
    attn = SimpleNamespace(
        w_kc=w_kc,
        w_vc=w_vc_sglang,
        w_scale=call["w_scale"],
        num_local_heads=n_heads,
        o_proj=SimpleNamespace(weight=call["o_weight"]),
    )
    heads = rocm_absorb_v_bmm(attn, attended.view(-1, n_heads, kv_lora_rank))
    return _linear(heads, call["o_weight"], call["o_scale"])


class _Linear:
    """A ReplicatedLinear as Gfx950FusedIndexer calls it: its weight, and (its output, None)."""

    def __init__(self, weight, forward):
        self.weight, self._forward = weight, forward

    def __call__(self, x):
        return self._forward(x), None


def _fused_indexer(call):
    """The layer's Gfx950FusedIndexer, made once as Indexer.__init__ makes it, over a stand-in
    Indexer of its tensors: wq_b and wk block-FP8, weights_proj's bf16 output on HIP."""
    from sglang.srt.layers.attention.dsa.gfx950_fused_indexer import HEAD_DIM, Gfx950FusedIndexer

    def make(wk_weight):
        wq_b_weight, weights_proj_weight = call["wq_b_weight"], call["weights_proj_weight"]
        indexer = SimpleNamespace(
            n_heads=weights_proj_weight.shape[0],
            softmax_scale=HEAD_DIM**-0.5,
            wq_b=_Linear(wq_b_weight, lambda x: _linear(x, wq_b_weight, call["wq_b_scale"])),
            wk=_Linear(wk_weight, lambda x: _linear(x, wk_weight, call["wk_scale"])),
            _weights_proj_bf16_in_fp32_out=lambda x: replicated_linear(x, weights_proj_weight),
            k_norm=SimpleNamespace(
                weight=call["k_norm_weight"], bias=call["k_norm_bias"], variance_epsilon=call["index_eps"]
            ),
            rotary_emb=SimpleNamespace(cos_cache=call["index_cos"], sin_cache=call["index_sin"]),
        )
        return Gfx950FusedIndexer(indexer)

    return _held(call["wk_weight"], "_aklr_gfx950_fused_indexer", make)


def _index(call, x, q_lora, widest):
    """Indexer.forward_cuda's gfx950 fused decode (_gfx950_fused_decode): the four launches, which
    store the rows' index keys and return their top-k slots; the shared workspace allocated at the
    first call, ``widest`` keys wide, as sglang allocates it before its first step for the widest its
    decode graphs take: one allocation, never outgrown."""
    from sglang.srt.layers.attention.dsa.gfx950_fused_indexer import (
        CACHE_TOK_STRIDE,
        MAX_ROWS,
        PAGE_SIZE,
        prealloc_workspace,
    )

    page_table, index_k_cache = call["page_table"], call["index_k_cache"]
    rows, max_cols = x.shape[0], page_table.shape[1] * PAGE_SIZE
    state = _fused_indexer(call)
    prealloc_workspace(device=x.device, max_cols=widest, fp8_dtype=torch.float8_e4m3fn)
    if rows > MAX_ROWS or not state.ensure_workspace(
        device=x.device, max_cols=max_cols, fp8_dtype=torch.float8_e4m3fn
    ):
        raise NotImplementedError("sglang's standard indexer serves this call")
    return state.run(
        x=x,
        q_lora=q_lora,
        positions=call["positions"],
        out_cache_loc=call["cache_locations"],
        kv_cache=index_k_cache.view(-1, PAGE_SIZE, CACHE_TOK_STRIDE),
        kv_cache_read=index_k_cache.view(index_k_cache.shape[0], -1),
        seqlens_int32=call["seq_lens"],
        row_ends_int32=call["seq_lens"],
        page_table_64=page_table,
        rows=rows,
    )


def _moe(call, x):
    """DeepseekV2MoE.forward_normal on the rank: deepseek_v2_fp8_moe's baseline, ungrouped."""
    return DeepseekV2Fp8MoeBaseline().baseline(
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
        scoring_func="sigmoid",
        routed_scaling_factor=call["routed_scaling_factor"],
    )["out"]


def _dense_mlp(call, x):
    """DeepseekV2MLP.forward without a swiglu limit: gate_up_proj, SiluAndMul (sgl_kernel's
    silu_and_mul on HIP), down_proj."""
    from sgl_kernel import silu_and_mul

    gate_up = _linear(x, call["gate_up_weight"], call["gate_up_scale"])
    act = gate_up.new_empty((gate_up.shape[0], gate_up.shape[1] // 2))
    silu_and_mul(gate_up, act)
    return _linear(act, call["down_weight"], call["down_scale"])


def _embedding(comm, input_ids, embed_weight, tp_size):
    """VocabParallelEmbedding.forward on gfx950: its fused Triton lookup of this rank's rows (the
    others' zeros; no padding or added vocabulary), then the group's sum
    (tensor_model_parallel_all_reduce: aiter's custom all-reduce)."""
    from sglang.kernels.ops.embeddings.vocab_parallel_embedding import vocab_parallel_embedding

    if tp_size == 1:
        return torch.nn.functional.embedding(input_ids.long(), embed_weight)
    vocab_rows = embed_weight.shape[0]
    start, vocab = dist.get_rank() * vocab_rows, tp_size * vocab_rows
    e = vocab_parallel_embedding(input_ids, embed_weight, start, start + vocab_rows, 0, vocab, vocab)
    if comm is None or not comm.should_custom_ar(e):
        raise NotImplementedError("sglang sums the embedding on RCCL here")
    return comm.custom_all_reduce(e)


class _Glm53Baseline(DistOpBaseline):
    """The TP group's custom all-reduce, which every glm53 op's sums run on."""

    _comm = None

    def setup(self):
        """What sglang 0.5.23 sets up on gfx950 before the first step: the small-batch MoE sort
        AiterRunnerCore patches into aiter (process-wide, as sglang's), and with more than one rank its
        TP group's custom all-reduce, aiter's CustomAllreduce over a gloo group of the ranks
        (GroupCoordinator's ca_comm). CustomAllreduce trades its IPC handles through the default store,
        which it takes to be a TCPStore (sglang's is): it gets one of the group's, rank 0 hosting it."""
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
        from sglang.kernels.ops.moe.moe_sorting_small import apply_aiter_small_moe_sort_patch

        apply_aiter_small_moe_sort_patch()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size == 1:
            return
        store = (
            dist.TCPStore("127.0.0.1", 0, world_size, is_master=True, wait_for_workers=False)
            if rank == 0
            else None
        )
        port = [store.port if store else None]
        dist.broadcast_object_list(port)
        store = store or dist.TCPStore("127.0.0.1", port[0], world_size, is_master=False)
        device = torch.device("cuda", torch.cuda.current_device())
        group = dist.new_group(backend="gloo")
        with mock.patch.object(dist.distributed_c10d, "_get_default_store", return_value=store):
            self._comm = CustomAllreduce(group=group, device=device)

    def teardown(self):
        """Frees the custom all-reduce's buffers; aiter keeps the MoE sort patch, as sglang does."""
        if self._comm is not None:
            self._comm.close()
            self._comm = None

    def _layer(self, call, mlp, topk_of):
        """The layer on the complete stream: (the stream leaving it, the top-k its attention read,
        ``topk_of(x, q_lora)``)."""
        x, q_lora, q, k_nope, k_pe = _q_and_latent(call)
        topk_indices = topk_of(x, q_lora)
        attention = _attention(call, q, k_nope, k_pe, topk_indices)
        eps = call["eps"]
        post_norm, stream = _all_reduce_rms_norm(
            self._comm, attention, call["hidden_states"], call["post_norm_weight"], eps
        )
        _, out = _all_reduce_rms_norm(
            self._comm, mlp(call, post_norm), stream, call["input_norm_weight"], eps
        )
        return out, topk_indices


class Glm53MoeLayerBaseline(_Glm53Baseline):
    def baseline(self, **call):
        """out: the MoE layer reading the previous indexer layer's top-k."""
        return {"out": self._layer(call, _moe, lambda x, q_lora: call["topk_indices"])[0]}


class Glm53MoeIndexLayerBaseline(_Glm53Baseline):
    """The MoE layer selecting its own top-k; ``widest``: the widest page table its calls take, in
    keys, which the indexer's workspace is allocated for."""

    mlp = staticmethod(_moe)

    def __init__(self, widest):
        self.widest = widest

    def baseline(self, **call):
        """layer_out and topk_indices, both new, as sglang's indexer allocates its top-k."""
        out, topk_indices = self._layer(
            call, self.mlp, lambda x, q_lora: _index(call, x, q_lora, self.widest)
        )
        return {"layer_out": out, "topk_indices": topk_indices}


class Glm53DenseIndexLayerBaseline(Glm53MoeIndexLayerBaseline):
    """glm53_moe_index_layer's, the dense MLP in the MoE's place."""

    mlp = staticmethod(_dense_mlp)


class Glm53HeadBaseline(_Glm53Baseline):
    def baseline(self, *, hidden_states, lm_head_weight, tp_size, logits=None):
        """logits: LogitsProcessor._get_logits without softcapping or a logit scale: _compute_lm_head's
        bf16 torch.matmul, GroupCoordinator.all_gather on aiter's custom all-gather (MultimemAllGatherer
        has no multicast on ROCm), its copy of the gathered rows into columns, then
        cast_to_float32_logits into ``logits`` (the logits buffer)."""
        from sglang.kernels.ops.activation.softcap import cast_to_float32_logits

        local = torch.matmul(hidden_states, lm_head_weight.T)
        rows, width = local.shape
        if tp_size > 1:
            if self._comm is None or not self._comm.should_custom_ag(local):
                raise NotImplementedError("sglang gathers the logits on RCCL here")
            gathered = torch.empty(tp_size * rows, width, dtype=local.dtype, device=local.device)
            self._comm.all_gather_unreg(local, out=gathered, dim=0)
            local = gathered.view(tp_size, rows, width).movedim(0, 1).reshape(rows, tp_size * width)
        if logits is None:
            logits = torch.empty(local.shape, dtype=torch.float32, device=local.device)
        return {"logits": cast_to_float32_logits(local, logits)}


class Glm53NextnEntryBaseline(_Glm53Baseline):
    def baseline(
        self,
        *,
        input_ids,
        hidden_states,
        embed_weight,
        enorm_weight,
        hnorm_weight,
        eh_proj_weight,
        eps,
        tp_size,
        **_,
    ):
        """out: DeepseekModelNextN.forward up to its decoder layer on gfx950: embed_tokens
        (VocabParallelEmbedding), enorm and hnorm (RMSNorm.forward_aiter), torch.cat, then eh_proj
        (nn.Linear: a bf16 torch.matmul)."""
        from sglang.srt.layers import layernorm

        e = _embedding(self._comm, input_ids, embed_weight, tp_size)
        x = torch.cat(
            [layernorm.rms_norm(e, enorm_weight, eps), layernorm.rms_norm(hidden_states, hnorm_weight, eps)],
            dim=-1,
        )
        return {"out": torch.matmul(x, eh_proj_weight.T)}


class Glm53EmbedBaseline(_Glm53Baseline):
    def baseline(self, *, input_ids, embed_weight, tp_size, embeddings=None):
        """embeddings: embed_tokens (VocabParallelEmbedding.forward)."""
        out = _embedding(self._comm, input_ids, embed_weight, tp_size)
        return {"embeddings": out if embeddings is None else embeddings.copy_(out)}
