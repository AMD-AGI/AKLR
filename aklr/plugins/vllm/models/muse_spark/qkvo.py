# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Measured MI355X BF16 projection kernels for Muse Spark target attention on vLLM 0.34."""

from __future__ import annotations

import copy
import logging

import torch

from aklr.plugins.vllm.models.muse_spark.configuration import supports_muse_tp8

_LOG = logging.getLogger("vllm.aklr.muse_qkvo")
_ROWS = {(2304, 8192): (8, 128, 256, 512), (8192, 1024): (8, 256, 384)}


def supports_qkvo(x, weight, bias):
    """Match the measured BF16, bias-free, row-major operand contract."""
    return (
        x.ndim == 2
        and weight.ndim == 2
        and x.shape[1] == weight.shape[1]
        and x.shape[0] in _ROWS.get(tuple(weight.shape), ())
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and bias is None
        and not getattr(weight, "is_shuffled", False)
    )


def qualified_rows(plugin, weight):
    """Ask AITER once at load time, outside compilation and graph capture."""
    n, k = weight.shape
    rows = set()
    for m in _ROWS[n, k]:
        x = weight.new_empty((m, k))
        decision = plugin.explain("gemm_a16w16", x, weight).decision
        if decision is not None and decision.route == "provider" and decision.provider == "aklr":
            rows.add(m)
    return frozenset(rows)


def _wrap(native, gemm, rows):
    def project(layer, x, weight, bias):
        if (
            x.device.type == "cuda"
            and x.device == weight.device
            and supports_qkvo(x, weight, bias)
            and x.shape[0] in rows
        ):
            return gemm(x, weight)
        return native(layer, x, weight, bias)

    return project


def configure_qkvo(worker):
    """Select target-model attention layers after weight loading and before graph capture."""
    if not supports_muse_tp8(worker.vllm_config):
        return
    from aiter import plugin
    from aiter.tuned_gemm import gemm_a16w16
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm
    from vllm.models.muse_spark.amd.model import MuseSparkAttention

    if "gemm_a16w16" not in plugin.status()["routed_ops"]:
        return
    qualified = {}
    count = 0
    for _, attention in worker.model_runner.get_model().named_modules():
        if not isinstance(attention, MuseSparkAttention):
            continue
        for role, expected in (("qkv_proj", (2304, 8192)), ("o_proj", (8192, 1024))):
            layer = getattr(attention, role)
            method = layer.quant_method
            if (
                type(method) is not UnquantizedLinearMethod
                or method._gemm_impl is not rocm_unquantized_gemm
                or tuple(layer.weight.shape) != expected
                or layer.weight.dtype != torch.bfloat16
                or layer.weight.device.type != "cuda"
                or not layer.weight.is_contiguous()
                or getattr(layer.weight, "is_shuffled", False)
            ):
                continue
            if expected not in qualified:
                qualified[expected] = qualified_rows(plugin, layer.weight)
            if not qualified[expected]:
                continue
            # Some quantization configs reuse a method. Own a copy for this target layer only.
            local = copy.copy(method)
            local._gemm_impl = _wrap(method._gemm_impl, gemm_a16w16, qualified[expected])
            layer.quant_method = local
            count += 1
    _LOG.info("Muse QKV/O: enabled %d layers on rank %d, qualified rows %s", count, worker.rank, qualified)
