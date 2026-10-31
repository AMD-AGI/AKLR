# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stand-ins for the few sglang modules the plugin imports, so its tests need neither sglang nor a GPU."""

import contextlib
import sys
from collections import namedtuple
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from types import ModuleType, SimpleNamespace

import pytest


@dataclass(frozen=True, eq=False)
class UnreducedOutput:
    """sglang's partial of a sum its group still owes."""

    partial: object
    group: object

    def complete(self):
        return self.group.all_reduce(self.partial)


class DeferredFinalize:
    """sglang's MoE output that only its producer finishes."""

    def complete(self):
        raise NotImplementedError


@dataclass(eq=False)
class Contribution:
    value: object
    update: object
    owed: object = None

    def complete(self):
        if self.owed is not None:
            self.value, self.owed = self.owed.complete(), None
        return self.value

    def release(self):
        self.value = self.owed = None


@dataclass(frozen=True, eq=False)
class OwedOutput:
    contribution: Contribution


class ResidualStream:
    """sglang's residual stream of one forward: the residual a layer wrote and the
    contribution a producer left pending on it, with sglang's checks."""

    def __init__(self, residual=None):
        self.residual, self.pending = residual, None

    def record(self, output, update):
        if self.pending is not None:
            raise RuntimeError("cannot replace an unconsumed producer contribution")
        if isinstance(output, UnreducedOutput):
            self.pending = Contribution(output.partial, update, output)
        elif isinstance(output, DeferredFinalize):
            self.pending = Contribution(None, update, output)
        else:
            self.pending = Contribution(output, update)
        return OwedOutput(self.pending) if self.pending.owed is not None else output

    def check(self, hidden):
        pending = self.pending
        if pending is None:
            valid = not isinstance(hidden, OwedOutput) and (self.residual is None or hidden is self.residual)
        elif pending.owed is not None:
            valid = isinstance(hidden, OwedOutput) and hidden.contribution is pending
        else:
            valid = hidden is pending.value
        if not valid:
            raise RuntimeError("output does not belong to this residual stream")

    def complete(self, hidden):
        self.check(hidden)
        return hidden if self.pending is None else self.pending.complete()

    def snapshot(self, hidden):
        self.check(hidden)
        pending = self.pending
        if pending is None:
            return hidden.clone()
        if not pending.update.is_plain_add:
            raise NotImplementedError("snapshot requires a plain residual update")
        if pending.owed is None:
            value = pending.value
        elif isinstance(pending.owed, UnreducedOutput):
            value = replace(pending.owed, partial=pending.value.clone()).complete()
        else:
            raise NotImplementedError("a finalize handoff requires main-output capture")
        return value.clone() if self.residual is None else value + self.residual

    def write(self, residual):
        if self.pending is not None:
            self.pending.release()
        self.residual, self.pending = residual, None
        return residual

    def export(self, hidden):
        self.check(hidden)
        if self.pending is None:
            return hidden, None
        return self.pending.complete(), self.residual


class Fp8MoEMethod(SimpleNamespace):
    """sglang's fp8 FusedMoE method."""


def stream_of(forward_batch):
    if forward_batch.residual_stream is None:
        raise RuntimeError("start the layer stack before entering a stage")
    return forward_batch.residual_stream


def final_norm(hidden_states, forward_batch, layernorm):
    """sglang's residual_batch.final_norm: the norm of the stream's last output and
    residual, the stream closed."""
    hidden_states, residual = stream_of(forward_batch).export(hidden_states)
    forward_batch.residual_stream = None
    if residual is None:
        return layernorm(hidden_states)
    return layernorm(hidden_states, residual)[0]


@pytest.fixture
def sglang(monkeypatch):
    """A recording HookRegistry, the runtime config and the flags the routes read, all off (one
    draft token a step); the attention backend and KV pool are the layer tests' to set.
    """
    state = SimpleNamespace(
        registered=[],
        batch_invariant=False,
        backend=None,
        pool=None,
        lora=False,
        dp_attention=False,
        piecewise_cuda_graph=False,
        breakable_cuda_graph=False,
    )
    state.config = SimpleNamespace(
        moe=SimpleNamespace(
            enable_eplb=False,
            expert_distribution_recorder_mode=None,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
        ),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        features=SimpleNamespace(enable_return_routed_experts=False),
    )
    state.parallel = SimpleNamespace(dcp_enabled=False, pp_size=1, tp_size=8, attn_tp_size=8, moe_ep_size=1)
    state.spec = SimpleNamespace(speculative_eagle_topk=1, speculative_algorithm=None)
    state.topk_shares = []
    # what sglang's post-experts all-reduce summed, over the TP group's 8 ranks
    state.all_reduced = []

    class IndexTopKShareState:
        """The MTP carry's top-k (the batch's carried_topk), what updates it and whether
        it was published."""

        def __init__(self, topk_indices):
            self.topk_indices, self.updated, self.published = topk_indices, None, False

        @classmethod
        def from_mtp_carry(cls, forward_batch):
            state.topk_shares.append(cls(forward_batch.carried_topk))
            return state.topk_shares[-1]

        def update(self, topk_indices):
            self.updated = topk_indices

        def publish(self):
            self.published = True

    on, off = SimpleNamespace(get=lambda: True), SimpleNamespace(get=lambda: False)
    modules = {
        "sglang.srt.plugins.hook_registry": {
            "HookType": SimpleNamespace(AROUND="around"),
            "HookRegistry": SimpleNamespace(register=lambda *args: state.registered.append(args)),
        },
        "sglang.srt.environ": {
            "envs": SimpleNamespace(
                SGLANG_SIMULATE_UNIFORM_EXPERTS=off,
                SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=off,
                SGLANG_DSA_FUSE_TOPK=on,
                SGLANG_DSA_TOPK_BROADCAST=off,
            )
        },
        "sglang.srt.runtime_context": {
            "get_exec": lambda: state.config,
            "get_parallel": lambda: state.parallel,
            "get_spec": lambda: state.spec,
        },
        "sglang.srt.batch_invariant_ops": {"is_batch_invariant_mode_enabled": lambda: state.batch_invariant},
        "sglang.srt.model_executor.forward_context": {
            "get_attn_backend": lambda: state.backend,
            "get_token_to_kv_pool": lambda: state.pool,
        },
        "sglang.srt.lora.deepseek_mla_correction": {"is_kv_b_lora_active": lambda attn: state.lora},
        "sglang.srt.models.deepseek_common.attention_forward_methods": {
            "AttnForwardMethod": SimpleNamespace(MLA="mla", MLA_ROCM="mla_rocm")
        },
        "sglang.srt.state_capturer.indexer_topk": {"maybe_capture_indexer_topk": lambda layer_id, topk: topk},
        # not HIP: the indexer layers' attention op leaves them to sglang
        "sglang.srt.layers.attention.dsa.dsa_indexer": {
            "Indexer": type("Indexer", (SimpleNamespace,), {}),
            "_is_hip": False,
            "_use_aiter": False,
            "_use_aiter_preshuffle": False,
            "_is_in_piecewise_or_breakable_cuda_graph": lambda: False,
        },
        "sglang.srt.layers.attention.dsa.dsa_indexer_metadata": {
            "DSAIndexerMetadata": type("DSAIndexerMetadata", (SimpleNamespace,), {})
        },
        "sglang.srt.layers.attention.dsa.dsa_topk_backend": {
            "TopkTransformMethod": IntEnum("TopkTransformMethod", "PAGED RAGGED")
        },
        "sglang.srt.layers.layernorm": {"LayerNorm": type("LayerNorm", (), {})},
        "sglang.srt.layers.dp_attention": {"is_dp_attention_enabled": lambda: state.dp_attention},
        "sglang.srt.layers.moe": {
            "post_experts_all_reduce": lambda hidden_states: (
                state.all_reduced.append(hidden_states) or hidden_states * 8
            )
        },
        "sglang.srt.layers.layer_boundary.contracts": {
            "BatchVariant": Enum(
                "BatchVariant",
                "ORDINARY CONTEXT_PARALLEL INPUT_SCATTERED SEQUENCE_PARALLEL UNPADDED",
            )
        },
        "sglang.srt.layers.layer_boundary.output": {
            "UnreducedOutput": UnreducedOutput,
            "DeferredFinalize": DeferredFinalize,
        },
        "sglang.srt.layers.layer_boundary.residual": {
            "batch": SimpleNamespace(
                start=lambda forward_batch: setattr(forward_batch, "residual_stream", ResidualStream()),
                complete_output=lambda hidden_states, forward_batch: stream_of(forward_batch).complete(
                    hidden_states
                ),
                final_norm=final_norm,
            )
        },
        "sglang.srt.layers.layer_boundary.residual.stream": {"ResidualStream": ResidualStream},
        "sglang.srt.eplb.expert_distribution": {
            "get_global_expert_distribution_recorder": lambda: SimpleNamespace(
                disable_this_region=contextlib.nullcontext
            )
        },
        "sglang.srt.layers.attention.index_topk_share": {"IndexTopKShareState": IndexTopKShareState},
        "sglang.srt.utils": {"BumpAllocator": lambda **kwargs: SimpleNamespace(**kwargs)},
        "sglang.srt.layers.quantization.fp8": {"Fp8MoEMethod": Fp8MoEMethod},
        "sglang.srt.layers.moe.topk": {
            "StandardTopKOutput": namedtuple("StandardTopKOutput", "topk_weights topk_ids router_logits")
        },
        "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager": {
            "is_in_tc_piecewise_cuda_graph": lambda: state.piecewise_cuda_graph
        },
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context": {
            "is_in_breakable_cuda_graph": lambda: state.breakable_cuda_graph
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return state
