# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang-AMD-Mono project
# From SGLang-AMD-Mono's sgl_amd_mono/deepseek_v41/draft.py; AKLR: the import paths,
# the spec name, the step counter (aklr.plugins.sglang.stats).
"""DeepSeek-V4.1-Flash's DSpark draft on gfx950: its spec and forward, each stage the
target's K1 and K2 on the draft's route and window. The stages' seams use the fused mHC
boundary's arithmetic (sglang's draft runs the unfused one): acceptance is the gate."""

from __future__ import annotations

import re

import torch

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.models.deepseek_v4 import DRAFT, SPEC_NAMES, embed
from aklr.plugins.sglang.models.deepseek_v4 import spec as target
from aklr.plugins.sglang.models.deepseek_v4.compat import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Requirements,
)
from aklr.plugins.sglang.models.deepseek_v4.kernels.common.plan import BLOCKS

EXPERTS, TOPK = 128, 3  # the draft's route
WINDOW, SWA_WIDTH = 128, 192  # a row's window index slots: the window + the block
SERVED = "deepseek_v4.draft:served"
# the checkpoint's draft stages are mtp.{stage}
_CKPT_SHARED = re.compile(r"mtp\.(?P<i>\d+)\.ffn\.shared_experts\.(?P<w>w[123])\.(?P<kind>weight|scale)")

# The SGLang names this module reads, by file under the sglang package (``f(p)``: a
# parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/environ.py": ("envs", "SGLANG_DSPARK_EMBED_IN_GRAPH"),
    "srt/layers/logits_processor.py": (
        "LogitsProcessorOutput",
        "next_token_logits",
        "hidden_states",
    ),
    # DeepseekV4ForCausalLMDSpark.forward, which the routed forward stands in for
    "srt/models/deepseek_v4_dspark.py": (
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "stages",
        "forward_embed",
        "embed_tokens",
        "hc_mult",
        "hc_pre_from_prev_sublayer",
        "sample_from_anchor",
        '"dspark_n_routed_experts"',
        '"dspark_num_experts_per_tok"',
    ),
    "srt/models/deepseek_v4.py": (
        "wq_a",
        "wkv",
        "fuse_wqa_wkv",
        "q_head_norm",
        "n_local_heads",
        "compress_ratio",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_spec", '"tp_group"'),
    "srt/speculative/dspark_components/dspark_config.py": ("dspark_gamma_from_num_draft_tokens",),
}


def _tokens_per_request(model) -> int:
    """A draft block's rows a request: gamma, or gamma + 1 without the anchor."""
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.dspark_components.dspark_config import (
        dspark_gamma_from_num_draft_tokens,
    )

    gamma = dspark_gamma_from_num_draft_tokens(int(get_spec().speculative_num_draft_tokens))
    return gamma if model.sample_from_anchor else gamma + 1


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )

    reasons, unseen = [], []
    route = (
        getattr(hf_config, "dspark_n_routed_experts", None),
        getattr(hf_config, "dspark_num_experts_per_tok", None),
    )
    if route != (EXPERTS, TOPK):
        reasons.append(f"draft route {route}: the kernels take {EXPERTS} experts, top-{TOPK}")
    rows = _tokens_per_request(model)
    if WINDOW + rows > SWA_WIDTH:
        reasons.append(f"a draft block of {rows} rows past {SWA_WIDTH} window slots")
    if not getattr(model, "hc_pre_from_prev_sublayer", False):
        reasons.append("needs V4.1's mHC hand-off between stages")
    if not envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get():
        reasons.append("needs SGLANG_DSPARK_EMBED_IN_GRAPH: the forward embeds the block")
    heads = Dims(get_parallel().tp_size).heads
    for i, stage in enumerate(model.stages):
        a, moe = stage.self_attn, stage.mlp
        if a.compress_ratio != 0 or a.q_head_norm or a.n_local_heads != heads:
            reasons.append(
                f"stage {i}: needs a window-only attention with {heads} local heads and no q head norm"
            )
        if moe.gate.weight.shape[0] != EXPERTS or not getattr(moe.experts.w13_weight, "is_shuffled", False):
            reasons.append(f"stage {i}: needs {EXPERTS} experts in aiter's shuffle")
        linears = [a.wqkv_a] if a.fuse_wqa_wkv else [a.wq_a, a.wkv]
        if any(
            getattr(lin, "aiter_bpreshuffled", False) or getattr(lin, "mxfp8_native_ready", False)
            for lin in linears + [a.wq_b, a.wo_a, a.wo_b]
        ):
            reasons.append(
                f"stage {i}: an FP8 linear method reordered an attention weight in "
                "place; K1 and K2 read them row-major"
            )
        if any(
            ("shared expert fp8", i, p, k) not in copies for p in target._SHARED for k in ("weight", "scale")
        ):
            unseen.append(i)
    if unseen:
        reasons.append(f"the shared expert's FP8 checkpoint tensors were not seen at load (stages {unseen})")
    return reasons


def _load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each stage's FP8 shared expert, by its checkpoint
    name (``mtp.{stage}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}``)."""
    from sglang.srt.runtime_context import get_parallel

    m = _CKPT_SHARED.fullmatch(name)
    if m is None or int(m["i"]) >= len(model.stages):
        return
    i, proj = int(m["i"]), target._CKPT_PROJ[m["w"]]
    p = get_parallel()
    dim = 1 if proj == "down_proj" else 0  # the TP-sharded intermediate dimension
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device())
    copies["shared expert fp8", i, proj, m["kind"]] = shard.contiguous()


class _Draft:
    """One rank's draft monokernels: its runner, each stage's weights, the forward."""

    def __init__(self, lm, runner, stages: list):
        self.lm, self.runner, self.stages = lm, runner, stages
        self._served = stats.device_counter(SERVED, runner.device)
        stats.set_info(SERVED, {"stages": len(stages)})

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **__):
        """``DeepseekV4ForCausalLMDSpark.forward`` for a routed draft step."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if not target.counted():
            self._served.add_(1)
        if not torch.cuda.is_current_stream_capturing():
            stats.record(SERVED, "served", str(plan.key))
        lm = self.lm
        x = input_embeds
        if x is None:  # the target's module where SGLang shares it: its whole table
            x = embed.lookup(lm.embed_tokens, input_ids, lm.hc_mult)
        if x is None:
            x = lm.forward_embed(input_ids)
        pre = None
        pending = (None, x, None, None)
        for j, L in enumerate(self.stages):  # each stage settles its post: what SGLang's returns
            out = target._launch(
                self.runner,
                L,
                plan,
                positions,
                forward_batch,
                (None, pending[1], None, None),
                pre,
            )
            # the last stage's post owed to the combine (target.head_post_combine)
            defer = j + 1 == len(self.stages) and target._post_combine_enabled()
            x, pre, post = target._boundary_out(L.module, defer, *out)
            pending = post if defer else (None, x, None, None)
        _, x = target.head_post_combine(self.stages[-1].module, pending, pre, lm.hc_mult)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)


def _ug_groups(tp: int) -> int:
    """FP4 groups an ug task: one (twice the tasks, ending closer together) from TP4 up;
    at TP2 two are faster, for the target and the draft alike."""
    return 1 if tp >= 4 else 2


def _build(model, hf_config, ctx) -> _Draft:
    from sglang.srt.runtime_context import get_parallel

    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
    )

    first = len(target.ROPE_TABLES)
    stages = [
        target._weights(ctx.copies, target.ROPE_TABLES, stage, i) for i, stage in enumerate(model.stages)
    ]
    for n, (_, t) in enumerate(target.ROPE_TABLES[first:], first):
        ctx.copies["rope table", n] = t
    p = get_parallel()
    runner = DSV41MonoRunner(
        p.tp_size,
        p.tp_rank,
        p.tp_group.cpu_group,
        model.stages[0].input_layernorm.weight.device,
        experts=EXPERTS,
        topk=TOPK,
        ug_groups=_ug_groups(p.tp_size),
        swa_width=SWA_WIDTH,
    )
    runner.set_deadline(ctx.deadline_s)
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // p.tp_size
    for m in widths[start:] + widths[:start]:
        # K1 on a settled residual, K2 on the window: no index source, no top-k
        runner.precompile(m, k1={(False, False)}, k2={(False, False)})
    return _Draft(model, runner, stages)


SPEC = MonoSpec(
    name=SPEC_NAMES[DRAFT],
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, target.MAX_ROWS + 1)),
            tp_sizes=(2, 4),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    # the launcher checks the target's requirements; the capture check, this
    requires=Requirements(attention_backends=target.SPEC.requires.attention_backends),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    load_tensor=_load_tensor,
    tokens_per_request=_tokens_per_request,
)
