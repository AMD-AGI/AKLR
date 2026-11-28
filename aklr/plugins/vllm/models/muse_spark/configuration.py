# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Configuration shared by the opt-in MI355X routes measured on Muse Spark."""

from importlib.metadata import version

from packaging.version import Version

from aklr.core.platform.device import DeviceTarget


def supports_muse_tp8_configuration(config, release=None):
    if Version(release or version("vllm")).public != "0.34.0":
        return False
    pc = config.parallel_config
    return (
        getattr(config.model_config.hf_text_config, "model_type", None) == "muse_spark_text"
        and pc.tensor_parallel_size == 8
        and pc.data_parallel_size == pc.pipeline_parallel_size == pc.nnodes == 1
        and pc.prefill_context_parallel_size == pc.decode_context_parallel_size == 1
        and pc.enable_expert_parallel
        and pc.expert_placement_strategy == "linear"
        and not pc.enable_eplb
        and not pc.use_sequence_parallel_moe
    )


def supports_muse_tp8(config, release=None, target=None):
    return supports_muse_tp8_configuration(config, release) and (
        (target or DeviceTarget.current()) == DeviceTarget(arch="gfx950", hardware="MI355X")
    )
