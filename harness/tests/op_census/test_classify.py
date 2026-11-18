"""op_class on kernel names seen in real MI355X traces (K3 and Qwen3.8 on SGLang)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from op_census import classify  # noqa: E402
from op_census.classify import op_class  # noqa: E402

CASES = [
    ("_sparse_gqa_chunk_prefill", "attention"),  # "prefill" contains "fill"
    (
        "void sglang::grouped_gemma_rmsnorm_kernel<2560l, false, __hip_bfloat16>(sglang::GroupedGemmaRMSNormParams)",
        "norm",
    ),
    ("triton_tem_fused_div_mm_silu_t_0", "gemm"),
    ("mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v33", "moe"),
    ("mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale_fix3_fp4opt_v1_persist_cu256_abuffer", "moe"),
    (
        "void vllm::moe::topkGatingSoftmax<std::bfloat16_t, 32, 512, 2, 64, true, 0, (vllm::moe::SharedExpertScoringFunc)0>",
        "router_topk",
    ),
    (
        "void sglang::fast_topk_detail::fast_topk_kernel<512, false>(sglang::fast_topk_detail::FastTopKParams)",
        "sparse_attention_indexer",
    ),
    ("_pack_qsa_prefill_kv", "sparse_attention_indexer"),
    ("chunk_fwd_kernel_o", "linear_attention"),
    ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "linear_attention"),
    ("_causal_conv1d_fwd_kernel", "linear_attention"),
    ("_fused_mamba_state_scatter_with_mask_kernel", "linear_attention"),
    (
        "void sglang::hc_combine_kernel<4l, 2560l, false, __hip_bfloat16>(sglang::HcCombineParams)",
        "hyper_connection",
    ),
    ("aiter::qwen38_flash_next_layers_s4_r1_pgggq_f0_b1", "fused_decoder_layers"),
    ("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x16x128_MI16x16x1_SN_LDSB0", "gemm"),
    (
        "void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle_2lds<ck::GridwiseGemmMultiD_xdl_cshuffle_v3_b",
        "gemm",
    ),
    ("_per_token_group_quant_8bit", "quantization"),
    ("_draft_topk1_finalize_kernel", "sampling"),
    (
        "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>",
        "sampling",
    ),
    ("__amd_rocclr_copyBuffer", "elementwise_copy"),
    (
        "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >",
        "elementwise_copy",
    ),
    (
        "_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii",
        "communication",
    ),
    ("aiter::mla_pfl_qh192_vh128_m32x8_n128x1_causal1", "attention"),
    ("void opus_moe_stage2_a8w4_decode_kernel_gfx950<OpusMoeStage2A8W4DecodeShape<32, 128, true>", "moe"),
    ("concat_and_cast_mha_k_pad_kernel", "attention"),
    ("_hc_mix_persistent_kernel", "hyper_connection"),
    ("_compact_kv", "sparse_attention_indexer"),
    ("_fa2_valid_counts", "attention"),
    ("_fused_sigmoid_mul_kernel", "activation"),
    ("_qwen4_verify_conv_prepare_kernel", "linear_attention"),
    (
        "void at::native::(anonymous namespace)::conv_depthwise2d_forward_kernel_generic<c10::BFloat16, int>",
        "linear_attention",
    ),
    ("_build_chain_tree_kernel", "spec_decode_bookkeeping"),
    ("void sglang::speculative::VerifyTreeGreedy<int, long>(int*, int*)", "spec_decode_bookkeeping"),
    ("assign_draft_cache_locs_contiguous", "spec_decode_bookkeeping"),
    (
        "void rocprim::ROCPRIM_400600_NS::detail::trampoline_kernel<rocprim::default_config",
        "elementwise_copy",
    ),
    ("void sglang::k3_a8w4::gateup_kernel<64, 4, 2>(sglang::k3_a8w4::GateUpParams)", "moe"),
    ("void sglang::k3_a8w4::down_ws_kernel<64, 8, 3>(sglang::k3_a8w4::DownParams)", "moe"),
    (
        "void sglang::k3_a8w4::reduce_kernel<16>(unsigned short const*, unsigned long, float const*, int const*)",
        "moe",
    ),
    ("void sglang::route_radix4_kernel<float, 896, 16, 256>(sglang::RouteRadix4Params)", "router_topk"),
    (
        "void aiter::lamportAllReduceTwoShotWaveSplit<std::bfloat16_t, 8>(std::bfloat16_t const*)",
        "communication",
    ),
    ("ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>)", "communication"),
    (
        "void sglang::sparse_mla_prefill_gfx950::sparse_mla_prefill_kernel<2, false>(sglang::sparse_mla_prefill_gfx950:",
        "attention",
    ),
    ("_gluon_fp8_mqa_logits_kernel_NUM_HEADS_32_HEAD_SIZE_128_BLOCK_KV_64", "sparse_attention_indexer"),
    (
        "void sglang::dsa_gfx950::logits::logits_hist_m<8, 12>(unsigned char const*)",
        "sparse_attention_indexer",
    ),
    (
        "void sglang::dsa_gfx950::topk::k_scatter<0, true, 16>(float const*, int const*)",
        "sparse_attention_indexer",
    ),
    (
        "void sgl::dsa_topk::coop_topk_kernel<2048u, 12u, 4096u, 1024u>(sgl::dsa_topk::CoopParams<2048u>)",
        "sparse_attention_indexer",
    ),
    ("void aiter::kn_entry_2c_sbhd_cached_indirect_inplace<aiter::OpCachedFwd, 1, true, false, t", "rope"),
    ("_dflash_grouped_conv_kernel", "draft_model"),
    ("k3_mono_k2_s8_nb1_x0_r1_t0_f1_o891", "fused_decoder_layers"),
    ("k3_mono_kda_pre_s8_l8_nb2_d0_g0_t0_r1", "fused_decoder_layers"),
    (
        "void sglang::dcp_topk_hip_detail::dcp_local_candidates_kernel<16>(float const*, long, int const*)",
        "sparse_attention_indexer",
    ),
    ("_localize_compact_kernel", "sparse_attention_indexer"),
    ("_latent_tail_partial_kernel", "gemm"),
    ("_dcp_pack_a2a_send_kernel", "communication"),
    ("_dcp_lse_combine_kernel", "attention"),
    ("_merge_kernel", "attention"),
    ("void sglang::k3_attn_res_read::k3_ag_attn_res_kernel<5, false>", "attention_residual"),
    (
        "kernel_kernel",
        "other",
    ),  # FlyDSL's generic name: several kernels share it; see the census's context column
]


@pytest.mark.parametrize("name,cls", CASES)
def test_op_class(name, cls):
    assert op_class("kernel", name) == cls


def test_copies_are_their_kind():
    assert op_class("memcpy", "Memcpy DtoD (Device -> Device)") == "memcpy"
    assert op_class("memset", "Memset (Device)") == "memset"


def test_attributed_by_deployment():
    # TileLang's default name: Qwen3.8's QSA indexer scoring there, unknown anywhere else.
    q = "qwen3.8next-fp4-sglang-agentic-mtp.qwen3.8next_tp1_conc48_kvdram-hicache_spec-mtp"
    assert classify.op_class_at(q, "kernel", "kernel_kernel") == "sparse_attention_indexer"
    assert (
        classify.op_class_at("dsv41flash-fp4-sglang-agentic-dspark.x", "kernel", "kernel_kernel") == "other"
    )
    assert classify.op_class_at(q, "kernel", "fast_topk_kernel") == classify.op_class(
        "kernel", "fast_topk_kernel"
    )


def test_kv_index_helpers():
    for name in (
        "create_flashinfer_kv_indices_triton",
        "create_chunked_prefix_cache_kv_indices",
        "create_dcp_kv_indices",
        "_Z20transfer_kernel_implITnDaXadL_Z24get_global_offset_lf_tblIKcEPT_S3_PKmllllEETnDaXadL_Z",
    ):
        assert classify.op_class("kernel", name) == "kv_cache", name


def test_structure_check_fixes():
    # Found by matching per-step call counts against the models' layer counts.
    k = "k31_kda_layer_m16_16x128x1x8x8_16x128x1x8x4_L8_n2096_a10_m10of8_l68_p16_c0_t0_r0_fr1536x2432x0_srt16_"
    assert op_class("kernel", k) == "fused_decoder_layers"
    assert (
        op_class("kernel", "fused_gdn_gating_kernel") == "linear_attention"
    )  # not the MoE router ("gating")
    assert (
        op_class("kernel", "recompute_w_u_fwd_kernel") == "linear_attention"
    )  # not attention ("_fwd_kernel")
    assert op_class("kernel", "sglang::transfer_mamba_load_kernel(sglang::MambaTransferParams)") == "kv_cache"
    assert (
        op_class("kernel", "sglang::transfer_mamba_backup_kernel(sglang::MambaTransferParams)") == "kv_cache"
    )
    assert op_class("kernel", "_index_k_norm_store_kernel") == "sparse_attention_indexer"
    assert (
        op_class("kernel", "void compute_cuda_kernel<int>(int const*, long const*, int*, long, long)")
        == "elementwise_copy"
    )
    q = "qwen3.8next-fp4-sglang-agentic-mtp.qwen3.8next_tp1_conc12_kvnone_spec-mtp"
    for n in (
        "triton_tem_fused_div_mm_silu_t_0",
        "triton_tem_fused_div_mm_silu_t_1",
        "triton_poi_fused_mean_mul_sigmoid_view_2",
    ):
        assert classify.op_class_at(q, "kernel", n) == "hyper_connection"
    assert classify.op_class_at("glm5.2-fp4-x.y", "kernel", "triton_tem_fused_div_mm_silu_t_0") == "gemm"
