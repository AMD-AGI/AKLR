# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# From sglang 0.5.22's python/sglang/srt/monokernels/spec.py, without its spec registry: the
# stand-in sglang of these tests (conftest.py) serves them.
"""Monokernel spec contract types (RFC #43949)."""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

import msgspec

if TYPE_CHECKING:
    import torch

MONOKERNEL_API_VERSION = 1


class Feature(enum.Enum):
    """SGLang features refused unless a spec lists them."""

    PIPELINE_PARALLEL = "pipeline parallelism"
    MULTI_NODE_TP = "multi-node tensor parallelism"
    DP_ATTENTION = "DP attention"
    EXPERT_PARALLEL = "expert parallelism"
    EPLB = "EPLB"
    CONTEXT_PARALLEL = "context parallelism"
    SPECULATIVE_DECODING = "speculative decoding"
    RAGGED_VERIFY = "uneven tokens per request in verify (SGLANG_RAGGED_VERIFY_MODE)"
    TWO_BATCH_OVERLAP = "two-batch / single-batch overlap"
    LORA = "LoRA"
    PD_DISAGGREGATION = "PD disaggregation"
    RETURN_ROUTED_EXPERTS = "--enable-return-routed-experts"
    RETURN_INDEXER_TOPK = "--enable-return-indexer-topk"
    EAGLE3_HIDDEN_STATES = "EAGLE3 hidden states at covered layers"
    KV_OFFLOAD = "KV offload"
    MEMORY_RELEASE = "memory release"
    CPU_OFFLOAD = "CPU offload"
    TORCH_COMPILE = "torch-compiled decode"
    PARTITIONED_GPU = "partitioned GPUs"


class StepMetadata(enum.Enum):
    """Per-step attention metadata a backend can skip building."""

    LOW_RATIO_DECODE_SCHEDULE = "the aiter decode schedule of each low-ratio indexer"


class DeviceLimits(msgspec.Struct, frozen=True):
    """What a spec's kernels support on one ISA, e.g. gfx950."""

    widths: tuple[int, ...]  # step rows, ascending
    tp_sizes: tuple[int, ...]
    resident_blocks: int  # the launch grid: one workgroup a CU


class Requirements(msgspec.Struct, frozen=True):
    """Configuration a spec needs, checked alike for every spec at startup; an empty
    field accepts any value."""

    dtypes: frozenset[str] = frozenset()
    # checked against the decode-phase backend, which runs the served steps
    attention_backends: frozenset[str] = frozenset()
    # a server argument -> its accepted values, as resolved for this server
    server_args: Mapping[str, frozenset] = {}
    # an HF config field -> the value the kernels are built for
    hf_config: Mapping[str, Any] = {}
    # an SGLang environment variable in envs -> its required value
    env: Mapping[str, Any] = {}


class NumericsKind(enum.Enum):
    BITWISE = "bitwise"
    TOLERANCE = "tolerance"


class Numerics(msgspec.Struct, frozen=True):
    """Numerical bounds against the standard path on the same GPU, as
    <launch>.<output>.<metric> (max_abs, rel or cosine); a max_abs of zero is bitwise.
    """

    kind: NumericsKind
    bounds: Mapping[str, float] = {}

    def __post_init__(self):
        metrics = ("max_abs", "rel", "cosine")
        bad = [n for n in self.bounds if n.count(".") < 2 or n.rsplit(".", 1)[1] not in metrics]
        if bad or (self.kind is NumericsKind.TOLERANCE and not self.bounds):
            raise ValueError(f"bounds must be <launch>.<output>.<metric>: {bad or 'none'}")

    def bound(self, name: str) -> float:
        return self.bounds[name]


class ShapeKey(msgspec.Struct, frozen=True):
    """The graph key a step is routed by; every rank holds it alike."""

    forward_mode: str  # "decode" or "target_verify"
    batch_size: int  # requests, padded to the graph's batch size
    tokens_per_request: int

    @property
    def rows(self) -> int:
        return self.batch_size * self.tokens_per_request


class BuildContext(msgspec.Struct, frozen=True):
    isa: str
    widths: tuple[int, ...]  # the rows the capture plan routes
    deadline_s: float  # every wait's deadline, below --watchdog-timeout
    copies: dict  # (kind, ...) -> tensor: the load hook's copies; build adds its own


class Plan(msgspec.Struct, frozen=True):
    """A routed forward's graph key and state shared by its layers within that step."""

    key: ShapeKey
    state: dict = {}


class BuiltMonokernels(Protocol):
    """The object a spec's build returns; it runs every built width's steps."""

    def forward(self, plan: Plan, *args: Any, **kwargs: Any) -> Any:
        """Run a routed step in place of the model's forward."""


class MonoSpec(msgspec.Struct, frozen=True):
    """A model-specific plugin contract; build returns its BuiltMonokernels."""

    name: str
    devices: Mapping[str, DeviceLimits]  # per ISA
    features: frozenset[Feature]  # the refused-by-default features it accepts
    requires: Requirements
    numerics: Numerics
    # (model, hf_config, copies): after load, with the load hook's copies
    model_disable_reasons: Callable[[torch.nn.Module, Any, dict], list[str]]
    build: Callable[[torch.nn.Module, Any, BuildContext], BuiltMonokernels]
    api_version: int  # the MONOKERNEL_API_VERSION this spec is written against
    # (cfg, hf_config), before load: the checks requires cannot state
    config_disable_reasons: Callable[[Any, Any], list[str]] | None = None
    # (copies, model, name, tensor): retain weights before model-specific conversion
    load_tensor: Callable[[dict, torch.nn.Module, str, torch.Tensor], None] | None = None
    # (model) -> tokens per request when different from the target, e.g. draft blocks
    tokens_per_request: Callable[[torch.nn.Module], int] | None = None
    # per-step metadata the forward never reads; backends skip it on routed steps
    unread_metadata: frozenset[StepMetadata] = frozenset()

    def widths(self, isa: str) -> tuple[int, ...]:
        return self.devices[isa].widths
