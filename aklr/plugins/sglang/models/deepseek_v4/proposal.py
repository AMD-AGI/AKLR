# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSpark draft's proposal: an FP8 vocab projection, the Markov steps on fused kernels and
the step's draft sampling params in one launch. The target verifies with its own bf16 head, so
only the acceptance rate may move."""

from __future__ import annotations

import functools
import logging
import os

import torch
import torch.distributed as dist

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

_LOGGER = logging.getLogger(__name__)

ENV_FP8 = "AKLR_DSV41_DRAFT_HEAD_FP8"
ENV_MARKOV = "AKLR_DSV41_MARKOV"
# "0": under simulated acceptance a sampled draft still gathers and keeps its step logits
ENV_SIM_DRAFT = "AKLR_DSV41_SIM_DRAFT"
# "0": that sampled draft's steps in sglang's kernels, not the Markov kernels
ENV_SIM_MARKOV = "AKLR_DSV41_SIM_MARKOV"
# the sampled Markov kernels' batches: a block scores its sequences in turn
SIM_MARKOV_MAX_BS = 8
DRAFT = "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark"
DSPARK_DRAFT = "sglang.srt.speculative.dspark_components.dspark_draft"
MARKOV = f"{DSPARK_DRAFT}.draft_markov_block"
PARAMS = f"{DSPARK_DRAFT}.resolve_draft_params"
# "0": a step's draft sampling params as sglang's three launches
ENV_PARAMS = "AKLR_DSV41_DRAFT_PARAMS"
# "0": the sampled block's noise counter advanced by its own launch, not the last step's finish
ENV_BUMP = "AKLR_DSV41_MARKOV_BUMP"
PARAMS_MAX_BS = 256
# whether the one-launch params agreed with sglang's at their first call
_PARAMS_AGREED: dict = {}
E4M3_MAX = 448.0
MAX_ROWS = 64
CHUNK_ROWS = 4096

SGLANG_READS = {
    "srt/models/deepseek_v4_dspark.py": (
        "DeepseekV4ForCausalLMDSpark",
        "attach_shared_modules(lm_head)",
        "_logits_from_x_post_hc(x_post_hc)",
        "_use_fp32_lm_head",
        "_opt_markov_w2_tp_shard",
        "stages",
        "norm",
        "lm_head",
        "markov_head",
        "DSparkV4MarkovHead",
        "markov_w1",
        "markov_w2",
        "_tp_shard",
        "_shard_group",
        "org_vocab_start",
        "org_vocab_end",
        "num_embeddings_per_partition",
    ),
    "srt/models/dspark.py": ("gather_and_crop_vocab",),
    "srt/layers/logits_processor.py": ("should_apply_lm_head_quant_method",),
    "srt/speculative/dspark_components/dspark_draft.py": (
        "draft_markov_block",
        "draft_markov_block(markov_head)",
        "draft_markov_block(base_logits)",
        "draft_markov_block(anchor_tokens)",
        "draft_markov_block(logits_out)",
        "draft_markov_block(greedy_draft)",
        "resolve_draft_params",
        "resolve_draft_params(sampling_info)",
        "resolve_draft_params(bs)",
        "resolve_draft_params(out)",
    ),
}


def _flag(name: str) -> bool:
    value = os.environ.get(name) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{name}={value!r}: expected '1' or '0'")
    return value == "1"


def _fp8_head(lm_head):
    """(e4m3 weight [V, K], f32 row scales [1, V], f32 ones [MAX_ROWS, 1]), or None
    when the head is not a plain bf16 matrix."""
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    weight = getattr(lm_head, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None))
    ):
        return None
    codes = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(weight.shape[0], 1, dtype=torch.float32, device=weight.device)
    # the KV cache is sized by now: a whole-matrix f32 temporary may not fit
    with torch.no_grad():
        for start in range(0, weight.shape[0], CHUNK_ROWS):
            w = weight[start : start + CHUNK_ROWS].float()
            s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / E4M3_MAX
            scale[start : start + CHUNK_ROWS] = s
            codes[start : start + CHUNK_ROWS] = (w / s).to(torch.float8_e4m3fn)
    ones = torch.ones(MAX_ROWS, 1, dtype=torch.float32, device=weight.device)
    return codes, scale.t().contiguous(), ones


class _Markov:
    """A greedy draft's Markov steps on the kernels: the rank's head rows, the whole
    embedding, the group's exchange."""

    def __init__(self, embed, head, exchange, ops):
        from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
            hipops,
        )

        self.embed, self.head, self.exchange, self.ops = embed, head, exchange, ops
        self.part, self.max_sequences = head.shape[0], hipops.MAX_SEQUENCES

    def block(self, base_logits, anchor_tokens, sampled=None):
        bs, gamma, _ = base_logits.shape
        x = self.exchange
        anchor = anchor_tokens.long()
        if bs > 1:
            anchor = anchor.contiguous()
            calls, tok = x.seq_calls, x.seq_tok
        else:
            calls, tok = x.calls, x.tok
        out = torch.empty(bs * gamma, dtype=torch.int64, device=base_logits.device)
        bump = bool(sampled) and sampled.get("bump_offset", False)
        for step in range(gamma):
            kw = dict(sampled or {}, bump_offset=bump and step == gamma - 1) if sampled else {}
            self.ops.markov_step(
                base_logits[0, step] if bs == 1 else base_logits[:, step],
                self.embed,
                self.head,
                anchor,
                None,
                out,
                step,
                x.size,
                x.rank,
                x.peers,
                calls,
                tok,
                gamma,
                **kw,
            )
        return out.view(bs, gamma)

    def sample_block(self, base_logits, anchor_tokens, temperatures, noise, round_bf16):
        """sglang's sampled Markov steps, step logits unwritten: each row at its temperature
        with the same Philox noise by global token id, a greedy row its argmax."""
        bump = _flag(ENV_BUMP)
        sampled = {
            "temps": temperatures,
            "key": int(noise.key),
            "offset": noise.offset,
            "vocab_start": self.exchange.rank * self.part,
            "round_bf16": round_bf16,
            "bump_offset": bump,
        }
        tokens = self.block(base_logits, anchor_tokens, sampled)
        if not bump:  # else the last step's finish advanced it
            noise.offset.add_(1)
        return tokens


def _markov_why_not(head) -> str | None:
    """Why the kernels cannot run this Markov head's steps, or None."""
    shard = getattr(head, "_tp_shard", None)
    group = getattr(head, "_shard_group", None)
    if type(head).__name__ != "DSparkV4MarkovHead":
        return f"a {type(head).__name__}"
    if shard is None or group is None:
        return "markov_w2 not TP-sharded (SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD)"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return f"markov weights {w1.dtype} / {w2.dtype}, not bf16"
    if w1.shape[1] != hipops.DIM or w2.shape[1] != hipops.DIM:
        return f"markov rank {w1.shape[1]}, the kernels' is {hipops.DIM}"
    part = shard.num_embeddings_per_partition
    if shard.org_vocab_end - shard.org_vocab_start != part or (
        shard.org_vocab_start != group.rank_in_group * part
    ):
        return "a vocab shard of unequal rows"
    if not 1 < group.world_size <= 8:
        return f"TP {group.world_size}"
    return None


def _build_markov(head):
    """The kernels' Markov steps for ``head``, or None (every rank alike)."""
    why = _markov_why_not(head)
    group = getattr(head, "_shard_group", None)
    ops, failure = None, why
    if why is None:
        from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
            hipops,
        )

        try:
            ops = hipops.ops()
        # every rank must still reach the agreement below
        except Exception as e:  # noqa: BLE001
            failure = f"the kernels do not build: {e}"
    if group is not None and group.world_size > 1:
        ok = torch.tensor([int(failure is None)], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group.cpu_group)
        if failure is None and not ok.item():
            failure = "another rank cannot run them"
    if failure is not None:
        return None, failure
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    w2 = head.markov_w2.weight
    exchange = hipops.Exchange(group.cpu_group, group.rank_in_group, group.world_size, w2.device)
    shard = head._tp_shard
    rows = w2[shard.org_vocab_start : shard.org_vocab_end].contiguous()
    return _Markov(head.markov_w1.weight.contiguous(), rows, exchange, ops), None


@sglang_route(f"{DRAFT}.attach_shared_modules", versions=VERSIONS, around=True)
def _attach(orig, self, *args, **kwargs):
    result = orig(self, *args, **kwargs)
    if not serving():
        return result
    head = None
    if _flag(ENV_FP8) and not self._use_fp32_lm_head and self.lm_head is not None:
        head = _fp8_head(self.lm_head)
    self.__dict__["_aklr_head_fp8"] = head
    stats.set_info("draft_head", "fp8 (row scales)" if head is not None else "sglang's projection")
    markov_head = getattr(self, "markov_head", None)
    if markov_head is not None and _flag(ENV_MARKOV):
        steps, why = _build_markov(markov_head)
        markov_head.__dict__["_aklr_markov"] = steps
        stats.set_info("draft_markov", "kernels" if steps is not None else why)
        if why is not None:
            _LOGGER.warning("DeepSeek-V4.1 draft: sglang's Markov steps run: %s", why)
    return result


@sglang_route(f"{DRAFT}._logits_from_x_post_hc", versions=VERSIONS, around=True)
def _draft_logits(orig, self, x_post_hc):
    head = self.__dict__.get("_aklr_head_fp8")
    rows = x_post_hc.shape[0] if x_post_hc.dim() == 2 else 0
    if head is None or not 0 < rows <= MAX_ROWS:
        return orig(self, x_post_hc)
    from sglang.srt.models.dspark import gather_and_crop_vocab

    codes, scale, ones = head
    x = self.stages[-1].norm(x_post_hc).to(torch.float8_e4m3fn)
    local_logits = torch._scaled_mm(
        x, codes.t(), scale_a=ones[:rows], scale_b=scale, out_dtype=torch.bfloat16
    )
    if self._opt_markov_w2_tp_shard:
        return local_logits
    return gather_and_crop_vocab(local_logits, self.lm_head)


def _sim_skips_step_logits() -> bool:
    """Whether a sampled draft's step logits go unread: under SGLANG_SIMULATE_ACC_LEN the
    accept runs greedy with the simulated length (``accept_sampling_info`` None) and the
    block-accept estimator, their only other reader, refuses to start."""
    if not _flag(ENV_SIM_DRAFT):
        return False
    from sglang.srt.environ import envs

    return envs.SGLANG_SIMULATE_ACC_LEN.get() > 0


def _sampled_block(
    *,
    markov_head,
    base_logits,
    anchor_tokens,
    draft_hidden,
    step_temperatures,
    noise,
    tp_sync,
    logits_out,
    greedy_draft=False,
):
    """sglang's sampled ``draft_markov_block`` with the step logits unwritten: each rank
    samples its vocab shard and the ranks swap per-block candidates. The tokens equal sglang's.
    """
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
    from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

    shard = markov_head.vocab_shard
    sharded = {} if shard is None else {"vocab_start": shard.start, "gather": shard.gather.gather_stacked}
    step_dtype = markov_head.step_logits_dtype(base_logits.dtype)
    state = markov_head.init_state(draft_hidden)
    prev = anchor_tokens.long()
    tokens = []
    for step in range(base_logits.shape[1]):
        bias, state = markov_head.step_bias(
            prev,
            hidden_states=None if draft_hidden is None else draft_hidden[:, step],
            state=state,
        )
        prev = MarkovSampleStep.execute(
            base_logits=base_logits[:, step],
            bias=bias,
            temperatures=step_temperatures,
            key=noise.key,
            offset=noise.offset,
            step=step,
            step_dtype=step_dtype,
            **sharded,
        )
        prev = tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, prev)
        tokens.append(prev)
    noise.offset.add_(1)
    return torch.stack(tokens, dim=1), logits_out


def _sample_why_not(steps, kwargs) -> str | None:
    """Why the kernels cannot run this sampled block's steps, or None."""
    base, temps = kwargs.get("base_logits"), kwargs.get("step_temperatures")
    head = kwargs.get("markov_head")
    if not _flag(ENV_SIM_MARKOV):
        return f"{ENV_SIM_MARKOV}=0"
    if kwargs.get("draft_hidden") is not None and type(head).__name__ != "DSparkV4MarkovHead":
        return "a Markov head reading hidden states"
    if (
        base is None
        or base.dim() != 3
        or base.shape[-1] != steps.part
        or base.stride(-1) != 1
        or base.dtype not in (torch.bfloat16, torch.float32)
    ):
        return "base logits not [bs, gamma, vocab shard]"
    if not 1 <= base.shape[0] <= SIM_MARKOV_MAX_BS:
        # past it the sequences' rounds of four outlast sglang's per-step kernels
        return f"batch {base.shape[0]}"
    if (
        temps is None
        or temps.dtype != torch.float32
        or temps.numel() < base.shape[0]
        or not temps.is_contiguous()
    ):
        return "step temperatures not fp32 [bs]"
    if head.step_logits_dtype(base.dtype) not in (torch.bfloat16, torch.float32):
        return f"step logits in {head.step_logits_dtype(base.dtype)}"
    return None


@sglang_route(MARKOV, versions=VERSIONS, around=True)
def _markov_block(orig, *args, **kwargs):
    head = kwargs.get("markov_head")
    steps = None if head is None else head.__dict__.get("_aklr_markov")
    if not serving():
        return orig(*args, **kwargs)
    if (
        not args
        and not kwargs.get("greedy_draft")
        and kwargs.get("logits_out") is not None
        and _sim_skips_step_logits()
    ):
        base = kwargs.get("base_logits")
        why = None if steps is None else _sample_why_not(steps, kwargs)
        if steps is not None and why is None:
            stats.record(MARKOV, "served", "sampled, kernels")
            head = kwargs["markov_head"]
            tokens = steps.sample_block(
                base,
                kwargs["anchor_tokens"],
                kwargs["step_temperatures"],
                kwargs["noise"],
                head.step_logits_dtype(base.dtype) == torch.bfloat16,
            )
            return tokens, kwargs["logits_out"]
        stats.record(MARKOV, "served", f"sampled, sglang's kernels ({why or 'no kernels'})")
        return _sampled_block(**kwargs)
    if steps is None:
        return orig(*args, **kwargs)
    base = kwargs.get("base_logits")
    why = None
    if args:
        why = "positional arguments"
    elif not kwargs.get("greedy_draft") or kwargs.get("logits_out") is not None:
        why = "a sampled draft (SGLANG_DSPARK_USE_GREEDY_DRAFT=0)"
    elif (
        base is None
        or base.dim() != 3
        or base.shape[-1] != steps.part
        or base.stride(-1) != 1
        or base.dtype not in (torch.bfloat16, torch.float32)
    ):
        why = "base logits not [bs, gamma, vocab shard]"
    elif not 1 <= base.shape[0] <= steps.max_sequences:
        why = f"batch {base.shape[0]}"
    stats.record(MARKOV, "served" if why is None else "fallback", why or "")
    if why is not None:
        return orig(*args, **kwargs)
    return steps.block(base, kwargs["anchor_tokens"]), None


@functools.cache
def _params_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(temp_ptr, topk_ptr, t_out, g_out, s_out, bs, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        live = i < bs
        t = tl.maximum(tl.load(temp_ptr + i, mask=live, other=1.0).to(tl.float32), 1e-5)
        g = tl.load(topk_ptr + i, mask=live, other=0) <= 1
        tl.store(t_out + i, t, mask=live)
        tl.store(g_out + i, g, mask=live)
        tl.store(s_out + i, tl.where(g, 0.0, t), mask=live)

    return kernel


def _params_inputs(sampling_info, bs, out):
    """(temperatures, top_ks) [bs] for the one-launch draft params, or None."""
    if sampling_info is None or out is None or not 0 < bs <= PARAMS_MAX_BS:
        return None
    temps, top_ks = sampling_info.temperatures.view(-1), sampling_info.top_ks.view(-1)
    if (
        temps.dtype != torch.float32
        or top_ks.dtype not in (torch.int32, torch.int64)
        or temps.numel() < bs
        or top_ks.numel() < bs
        or not (temps.is_cuda and top_ks.is_cuda)
        or not all(b.is_contiguous() and b.numel() >= bs for b in out)
        or [b.dtype for b in out] != [torch.float32, torch.bool, torch.float32]
    ):
        return None
    return temps.contiguous(), top_ks.contiguous()


@sglang_route(PARAMS, versions=VERSIONS, around=True)
def _draft_params(orig, *args, **kwargs):
    """sglang's draft sampling params of a step (a clamp, a compare and a select into the
    draft graph's buffers) in one launch."""
    bs, out = kwargs.get("bs"), kwargs.get("out")
    inputs = (
        None
        if args or not serving() or not _flag(ENV_PARAMS)
        else _params_inputs(kwargs.get("sampling_info"), bs, out)
    )
    if inputs is None or ("ok" not in _PARAMS_AGREED and torch.cuda.is_current_stream_capturing()):
        return orig(*args, **kwargs)
    if _PARAMS_AGREED.get("ok") is None:
        want = [b[:bs].clone() for b in orig(*args, **kwargs)]
        _launch_params(inputs, bs, out)
        ok = all(torch.equal(a, b[:bs]) for a, b in zip(want, out))
        _PARAMS_AGREED["ok"] = ok
        stats.set_info("draft_params", "one launch" if ok else "sglang's (disagreed)")
        if not ok:
            for a, b in zip(want, out):
                b[:bs].copy_(a)
            return type(out)(*(b[:bs] for b in out))
    if not _PARAMS_AGREED["ok"]:
        return orig(*args, **kwargs)
    _launch_params(inputs, bs, out)
    return type(out)(*(b[:bs] for b in out))


def _launch_params(inputs, bs, out):
    temps, top_ks = inputs
    _params_kernel()[(1,)](temps, top_ks, out[0], out[1], out[2], bs, BLOCK=PARAMS_MAX_BS)
