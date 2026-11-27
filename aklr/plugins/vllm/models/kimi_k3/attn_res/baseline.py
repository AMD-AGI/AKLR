# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What vLLM 0.34.0 runs: its Triton kernel, one launch per call, with the launch
configuration of its ``attn_res`` (copied in ``_triton``, so the baseline runs without
vLLM)."""

from aklr.core.contract.baseline import OpBaseline

__all__ = ["KimiK3AttnResBaseline", "KimiK3AttnResFp8Baseline"]


class KimiK3AttnResBaseline(OpBaseline):
    def baseline(
        self,
        *,
        prefix,
        bank,
        score_norm_weight,
        score_proj_weight,
        n_valid_rows=8,
        write_bank=False,
        addend=None,
        norm_weight=None,
        score_eps=1e-5,
        norm_eps=1e-5,
        out_dtype=None,
        out=None,
    ):
        """vLLM's kernel; it writes p back to ``prefix`` with an addend, and bank row
        n_valid_rows with ``write_bank``."""
        from aklr.plugins.vllm.models.kimi_k3.attn_res._triton import attn_res

        # vLLM allocates the output itself; so does its launcher here, whatever ``out`` is
        return {
            "out": attn_res(
                prefix,
                addend,
                bank,
                score_norm_weight,
                score_proj_weight,
                norm_weight,
                n_valid_rows,
                n_valid_rows if write_bank else -1,
                score_eps,
                norm_eps if norm_weight is not None else 0.0,
            )
        }


class KimiK3AttnResFp8Baseline(OpBaseline):
    def baseline(
        self,
        *,
        prefix,
        bank,
        score_norm_weight,
        score_proj_weight,
        n_valid_rows=8,
        write_bank=False,
        addend=None,
        norm_weight=None,
        score_eps=1e-5,
        norm_eps=1e-5,
        quantized=None,
        scale=None,
    ):
        """vLLM's kernel with its per-token fp8 output: quantized and scale, which its launcher
        allocates whatever ``quantized`` and ``scale`` are."""
        import torch

        from aklr.plugins.vllm.models.kimi_k3.attn_res._triton import attn_res
        from aklr.plugins.vllm.models.kimi_k3.attn_res.contract import FP8_MAX

        quantized, scale = attn_res(
            prefix,
            addend,
            bank,
            score_norm_weight,
            score_proj_weight,
            norm_weight,
            n_valid_rows,
            n_valid_rows if write_bank else -1,
            score_eps,
            norm_eps if norm_weight is not None else 0.0,
            quant_dtype=torch.float8_e4m3fn,
            quant_max=FP8_MAX,
        )
        return {"quantized": quantized, "scale": scale}
