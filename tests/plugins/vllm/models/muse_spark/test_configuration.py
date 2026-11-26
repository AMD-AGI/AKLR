# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest

from aklr.core.platform.device import DeviceTarget
from aklr.plugins.vllm.models.muse_spark.configuration import supports_muse_tp8


@pytest.mark.parametrize(
    "overrides,release,hardware,want",
    [
        ({}, "0.34.0+rocm100", "MI355X", True),
        ({}, "0.33.0", "MI355X", False),
        ({}, "0.34.0", "MI350X", False),
        ({"expert_placement_strategy": "round_robin"}, "0.34.0", "MI355X", False),
        ({"use_sequence_parallel_moe": True}, "0.34.0", "MI355X", False),
        ({"decode_context_parallel_size": 2}, "0.34.0", "MI355X", False),
        ({"data_parallel_size": 2}, "0.34.0", "MI355X", False),
        ({"nnodes": 2}, "0.34.0", "MI355X", False),
        ({"enable_eplb": True}, "0.34.0", "MI355X", False),
    ],
)
def test_only_measured_configuration_enables_custom_routes(overrides, release, hardware, want):
    parallel = dict(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        nnodes=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        enable_eplb=False,
        expert_placement_strategy="linear",
        use_sequence_parallel_moe=False,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(**(parallel | overrides)),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="muse_spark_text")),
    )
    assert supports_muse_tp8(config, release, DeviceTarget(arch="gfx950", hardware=hardware)) is want
