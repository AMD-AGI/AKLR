# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DSpark verify step's greedy accept off each rank's vocab slice (Infera-Flash's argmax
exchange, ``hipops.dspark_accept_xchg``): the full logits are gathered after the replay
only when read; a simulated-acceptance sampling step takes the graph's accept."""

from __future__ import annotations

import logging
import os
import sys

import torch
import torch.distributed as dist

from aklr.plugins.sglang import stats
from aklr.plugins.sglang.hook import sglang_route
from aklr.plugins.sglang.models.deepseek_v4 import serving
from aklr.plugins.sglang.models.deepseek_v4.routes import VERSIONS

_LOGGER = logging.getLogger(__name__)

ENV = "AKLR_DSV41_ACCEPT_XCHG"
SIM_ENV = "AKLR_DSV41_SIM_FOLD"
# "0": a simulated accept in the verify graph keeps sglang's gather and accept
SIM_XCHG_ENV = "AKLR_DSV41_SIM_XCHG"
# "0": the simulated length's broadcast and sglang's finishing kernels after the exchange
SIM_ONE_ENV = "AKLR_DSV41_SIM_ONE"
# "0": sglang gates the commit off before every step a simulated accept arms again
SIM_KEEP_ENV = "AKLR_DSV41_SIM_KEEP"
LOGITS = "sglang.srt.layers.logits_processor.LogitsProcessor"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
EPILOGUE = f"{VERIFY}.DsparkVerifyEpilogue"
EXECUTOR = f"{VERIFY}.TargetVerifyExecutor"
MAX_ROWS = 64  # argmax_xchg's rows
# sglang's debug observers that read the verify logits
DEBUG_ENVS = (
    "SGLANG_DSPARK_STS_COLLECT_PATH",
    "SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS",
)

SGLANG_READS = {
    "srt/layers/logits_processor.py": (
        "LogitsProcessor",
        "_get_logits(hidden_states)",
        "_get_logits(lm_head)",
        "_get_logits(logits_metadata)",
        "_get_logits(embedding_bias)",
        "_get_logits(use_logits_buffer)",
        "_compute_lm_head",
        "_logits_gatherer",
        "do_tensor_parallel_all_gather",
        "do_tensor_parallel_all_gather_dp_attn",
        "use_attn_tp_group",
        "use_tp_lm_head_all_to_all",
        "logit_scale",
        "final_logit_softcapping",
        "vocab_size",
        "next_token_logits_buffer",
        "forward_mode",
        "get_forward",
        "LoRABatchLayout",
        "LogitsProcessorOutput",
        "next_token_logits",
    ),
    "srt/speculative/dspark_components/dspark_verify.py": (
        "DsparkVerifyEpilogue",
        "__init__(tp_sync)",
        "capture_hook(runner)",
        "capture_hook(out)",
        "capture_hook(forward_batch)",
        "capture_hook(num_tokens)",
        "_accept(candidates)",
        "_accept(logits)",
        "_accept(draft_tokens)",
        "_accept(seq_lens)",
        "_accept(cutoff_verify_lens)",
        "read_accept(bs)",
        "simulated_correct_len",
        "correct_len",
        "bonus",
        "cap_trim_lens",
        "commit_lens",
        "new_seq_lens",
        "out_tokens",
        "TargetVerifyExecutor",
        "_forward_prepared_verify(batch)",
        "_forward_prepared_verify(verify_input)",
        "_forward_prepared_verify(seq_lens_cpu_backup)",
        "_forward_prepared_verify(seq_lens_sum_backup)",
        "accept_and_finalize(folded_accept)",
        "accept_and_finalize(target_logits)",
        "accept_and_finalize(bs)",
        "run_non_compact(batch)",
        "run_non_compact(verify_ids_2d)",
        "run_non_compact(sampling_info)",
        "commit_hidden(bs)",
        "commit_hidden(run_compact)",
        "verify_epilogue",
        "_simulated_correct_len",
        "simulate_fold",
        "begin_static_step",
        "begin_static_step(bs)",
        "begin_static_step(armed)",
        "_static_step_state",
        "verify_logits_adjustments_are_noop",
        "can_run_cuda_graph",
        "logits_output",
        "_static_epilogue(out)",
        "_static_epilogue(forward_batch)",
        "folds_commit",
        "verify_lens_buf",
        "inject_gate_buf",
        "stride",
        "commit_ctx",
        "kv_injector",
    ),
    "srt/speculative/dspark_components/dspark_kv_inject.py": (
        "inject_target_hidden(target_hidden)",
        "inject_target_hidden(cache_loc)",
        "inject_target_hidden(cache_loc_2d)",
        "inject_target_hidden(positions)",
        "inject_target_hidden(commit_lens)",
        "inject_target_hidden(state_slot)",
    ),
    "srt/model_executor/runner/decode_cuda_graph_runner.py": ("ragged_verify_mode",),
    "srt/distributed/parallel_state.py": ("get_tp_group",),
    "srt/managers/schedule_batch.py": (
        "has_grammar",
        "return_logprob",
        "sampling_info",
    ),
}


def _flag(name: str) -> bool:
    value = os.environ.get(name) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{name}={value!r}: expected '1' or '0'")
    return value == "1"


def _enabled() -> bool:
    return _flag(ENV)


class _Xchg:
    """The TP group's exchange for the verify argmax (its own), and the kernels."""

    def __init__(self, ops, exchange):
        self.ops, self.exchange = ops, exchange


def _build_xchg():
    """The exchange on every rank of the TP group, or (None, why); collective."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    failure = None
    if not 1 < group.world_size <= 8:
        failure = f"TP {group.world_size}"
    ops = None
    if failure is None:
        from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
            hipops,
        )

        try:
            ops = hipops.ops()
        # every rank must still reach the agreement below
        except Exception as e:  # noqa: BLE001
            failure = f"the kernels do not build: {e}"
    if group.world_size > 1:
        ok = torch.tensor([int(failure is None)], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group.cpu_group)
        if failure is None and not ok.item():
            failure = "another rank cannot run them"
    if failure is not None:
        return None, failure
    from aklr.plugins.sglang.models.deepseek_v4.kernels.deepseek_v41.gfx950 import (
        hipops,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    exchange = hipops.Exchange(group.cpu_group, group.rank_in_group, group.world_size, device)
    return _Xchg(ops, exchange), None


# the process's exchange (one target per scheduler process)
_XCHG: _Xchg | None = None


class _Deferred:
    """A captured verify graph's logits: the rank's slice (written by the replay), the
    logits buffer the step reads (written only by ``fill``)."""

    def __init__(self, processor, local, buf, valid: int):
        self.processor, self.local, self.buf, self.valid = processor, local, buf, valid
        self.fused = False  # the graph accepts off the slice (buf unwritten by the replay)
        self.owed = False  # this step's buf is still to be filled

    def fill(self) -> None:
        gathered = self.processor._logits_gatherer(self.local)
        self.buf.copy_(gathered[:, : self.buf.shape[1]])
        self.owed = False


# (buffer address, rows) -> the graph's deferred logits
_DEFERRED: dict[tuple[int, int], _Deferred] = {}


def _deferred(logits) -> _Deferred | None:
    """The graph whose logits these are: a padded replay returns the first rows of the
    smallest captured batch's buffer that holds them (the graphs' buffers share one base).
    """
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        return None
    ptr, rows = logits.data_ptr(), logits.shape[0]
    best = None
    for (p, r), d in _DEFERRED.items():
        if (
            p == ptr
            and r >= rows
            and d.buf.shape[1] == logits.shape[1]
            and (best is None or r < best.buf.shape[0])
        ):
            best = d
    return best


@sglang_route(f"{EPILOGUE}.__init__", versions=VERSIONS, around=True)
def _epilogue_init(orig, self, *args, **kwargs):
    global _XCHG
    result = orig(self, *args, **kwargs)
    if _enabled() and _XCHG is None and serving():
        _XCHG, why = _build_xchg()
        stats.set_info("verify_accept", "argmax exchange" if _XCHG else why)
        if why is not None:
            _LOGGER.warning("DeepSeek-V4.1 verify: sglang's logits gather and accept run: %s", why)
    return result


def _vocab_slice(processor, lm_head, local) -> int:
    """The slice's valid columns when it is rank r's columns r * part .. of the vocab, else 0."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    part = local.shape[1]
    shard = getattr(getattr(lm_head, "shard_indices", None), "org_vocab_start_index", None)
    if getattr(lm_head, "tp_size", None) != group.world_size or shard != group.rank_in_group * part:
        return 0
    return max(0, min(part, processor.vocab_size - group.rank_in_group * part))


def _deferrable(self, hidden_states, logits_metadata, embedding_bias) -> bool:
    buf = getattr(logits_metadata, "next_token_logits_buffer", None)
    return (
        _XCHG is not None
        and torch.cuda.is_current_stream_capturing()
        and logits_metadata.forward_mode.is_target_verify()
        and self.do_tensor_parallel_all_gather
        and not self.use_attn_tp_group
        and not self.do_tensor_parallel_all_gather_dp_attn
        and not getattr(self, "use_tp_lm_head_all_to_all", False)
        and self.logit_scale is None
        and not self.final_logit_softcapping
        and embedding_bias is None
        and hidden_states.dim() == 2
        and 0 < hidden_states.shape[0] <= MAX_ROWS
        and buf is not None
        and buf.dtype == torch.float32
        and tuple(buf.shape) == (hidden_states.shape[0], self.vocab_size)
    )


@sglang_route(f"{LOGITS}._get_logits", versions=VERSIONS, around=True)
def _get_logits(
    orig,
    self,
    hidden_states,
    lm_head,
    logits_metadata,
    embedding_bias=None,
    use_logits_buffer=True,
):
    if not _deferrable(self, hidden_states, logits_metadata, embedding_bias):
        return orig(
            self,
            hidden_states,
            lm_head,
            logits_metadata,
            embedding_bias,
            use_logits_buffer,
        )
    lp = sys.modules[type(self)._get_logits.__module__]
    with lp.get_forward().scoped(lora_batch_layout=lp.LoRABatchLayout.DP_LOCAL):
        local = self._compute_lm_head(hidden_states, lm_head, embedding_bias)
    buf = logits_metadata.next_token_logits_buffer
    valid = _vocab_slice(self, lm_head, local) if local.dim() == 2 and local.is_contiguous() else 0
    d = _Deferred(self, local, buf, valid)
    if not valid or local.dtype not in (torch.bfloat16, torch.float32):
        d.fill()  # in the graph, as sglang's
        return buf
    key = (buf.data_ptr(), buf.shape[0])
    first = _DEFERRED.get(key)
    if first is not None and first.local.shape == local.shape and first.local.dtype == local.dtype:
        # another graph of these rows (its fill reads one slice): this one writes it too
        first.local.copy_(local)
        d.local = first.local
    _DEFERRED[key] = d
    return buf


@sglang_route(f"{EPILOGUE}.capture_hook", versions=VERSIONS, around=True)
def _capture_hook(orig, self, runner, out, forward_batch, num_tokens):
    capturing = torch.cuda.is_current_stream_capturing()
    d = _deferred(getattr(out, "next_token_logits", None)) if capturing else None
    if d is not None and runner.ragged_verify_mode:
        _fill_in_graph(d)  # the ragged epilogue scatters the full logits first
        d = None
    result = orig(self, runner, out, forward_batch, num_tokens)
    if d is not None and not d.fused and any(v is d for v in _DEFERRED.values()):
        _fill_in_graph(d)  # no static accept took the slice
    return result


def _fill_in_graph(d: _Deferred) -> None:
    """The full logits in this graph, as sglang's, and no fill after its replays."""
    d.fill()
    for key, value in list(_DEFERRED.items()):
        if value is d:
            del _DEFERRED[key]


@sglang_route(f"{EPILOGUE}._accept", versions=VERSIONS, around=True)
def _accept(orig, self, *, candidates, logits, draft_tokens, seq_lens, cutoff_verify_lens=None):
    def sglang():
        return orig(
            self,
            candidates=candidates,
            logits=logits,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            cutoff_verify_lens=cutoff_verify_lens,
        )

    d = _deferred(logits) if torch.cuda.is_current_stream_capturing() else None
    if d is None:
        return sglang()
    if not _fusable(self, d, candidates, seq_lens) or cutoff_verify_lens is not None:
        _fill_in_graph(d)
        return sglang()
    return _fused_accept(self, d, candidates, seq_lens)


def _fusable(self, d: _Deferred, candidates, seq_lens) -> bool:
    return (
        bool(d.valid)
        and (self.simulated_correct_len is None or _flag(SIM_XCHG_ENV))
        and not getattr(self, "simulate_fold", False)
        and candidates.dtype == torch.int64
        and candidates.is_contiguous()
        and candidates.numel() == d.local.shape[0]
        and seq_lens.dtype in (torch.int64, torch.int32)
    )


def _fused_accept(self, d: _Deferred, candidates, seq_lens, gated=None):
    bs = candidates.shape[0]
    x = _XCHG
    out = self.read_accept(bs)
    sim = self.simulated_correct_len
    # the simulated length in the exchange's kernel (rank 0's in its packets)
    in_kernel = sim is not None and _flag(SIM_ONE_ENV) and self.gamma == self.stride - 1
    gating = (
        {
            "verify_lens": self.verify_lens_buf[:bs],
            "gate": self.inject_gate_buf,
            "gated": gated,
        }
        if gated is not None and (sim is None or in_kernel)
        else {}
    )
    if in_kernel:
        verify = sys.modules[VERIFY]
        site = verify.SpecTpSyncSite.DSPARK_ACCEPT_GRAPH
        gating.update(sim=sim, sim_bcast=self._tp_sync.enabled(site))
    x.ops.dspark_accept_xchg(
        d.local,
        d.valid,
        x.exchange.rank,
        x.exchange.peers,
        x.exchange.argmax_calls,
        candidates,
        seq_lens[:bs],
        out.correct_len,
        out.bonus,
        out.cap_trim_lens,
        out.commit_lens,
        out.new_seq_lens,
        out.out_tokens,
        **gating,
    )
    if in_kernel:
        stats.set_info("simulated_accept", "the exchange's kernel, rank 0's length")
    elif sim is not None:
        _simulated_outcome(self, out, candidates, seq_lens[:bs], sim, gated)
        stats.set_info("simulated_accept", "the exchange's, the simulated length")
    d.fused = True
    return out.commit_lens


def _simulated_outcome(self, out, candidates, prefix_lens, sim, gated) -> None:
    """sglang's ``build_accept_outs`` under SGLANG_SIMULATE_ACC_LEN on the exchange's greedy
    accept: rank 0's simulated length replaces the greedy one, the bonus stays the greedy one,
    and sglang's own kernels finish the commit lengths and out tokens."""
    verify = sys.modules[VERIFY]
    bs = candidates.shape[0]
    out.correct_len.copy_(sim.repeat(bs))
    out.correct_len.copy_(self._tp_sync.sync(verify.SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, out.correct_len))
    fin = verify.FinalizeAcceptLens.execute(
        correct_len=out.correct_len,
        cap_trim_lens=out.cap_trim_lens,
        prefix_lens=prefix_lens,
    )
    out.cap_trim_lens.copy_(fin.cap_trim_lens)
    out.commit_lens.copy_(fin.commit_lens)
    out.new_seq_lens.copy_(fin.new_seq_lens)
    out.out_tokens.copy_(
        verify.BuildOutTokens.execute(
            draft_tokens=candidates[:, 1:],
            correct_len=out.correct_len,
            bonus=out.bonus,
            verify_num_draft_tokens=self.stride,
            gamma=self.gamma,
        )
    )
    if gated is not None:
        verify_lens = self.verify_lens_buf[:bs].to(torch.int32)
        gated.copy_(torch.minimum(out.commit_lens, verify_lens) * self.inject_gate_buf)


@sglang_route(f"{EPILOGUE}._static_epilogue", versions=VERSIONS, around=True)
def _static_epilogue(orig, self, out, forward_batch):
    """sglang's static epilogue with the accept's draft tokens not copied out (the fused
    accept reads the candidates) and the folded commit lengths written by the accept."""
    d = _deferred(out.next_token_logits) if torch.cuda.is_current_stream_capturing() else None
    bs = forward_batch.batch_size
    candidates = forward_batch.input_ids.view(bs, self.stride)
    seq_lens = forward_batch.seq_lens
    if d is None or not _fusable(self, d, candidates, seq_lens):
        return orig(self, out, forward_batch)
    if not self.folds_commit:
        _fused_accept(self, d, candidates, seq_lens)
        return None
    if _ringed(self.commit_ctx.kv_injector):
        return orig(self, out, forward_batch)
    gated = torch.empty(bs, dtype=torch.int32, device=candidates.device)
    _fused_accept(self, d, candidates, seq_lens, gated)
    # sglang's inject of the committed rows, as _static_epilogue's without a ring
    cache_loc = forward_batch.out_cache_loc
    self.commit_ctx.kv_injector.inject_target_hidden(
        target_hidden=out.hidden_states,
        cache_loc=cache_loc,
        cache_loc_2d=cache_loc.view(bs, self.stride),
        positions=forward_batch.positions,
        commit_lens=gated,
        state_slot=None,
    )
    stats.set_info("verify_commit", "folded into the accept")
    return None


def _ringed(injector) -> bool:
    """Whether sglang's static epilogue passes the requests' ring slots to the inject: a
    per-request SWA ring (later builds) or unified KV (release/v0.5.22's test)."""
    unified = getattr(sys.modules[VERIFY], "is_unified_kv_triton", None)
    return getattr(injector, "swa_ring_size", None) is not None or bool(unified is not None and unified())


def _reads_logits(batch, sim_folded: bool = False) -> bool:
    """Whether this step reads the full verify logits after the replay, as sglang's DSpark
    worker reads them; the same on every rank. A ``sim_folded`` step reads them only for a
    requested sampling mask."""
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        verify_logits_adjustments_are_noop,
    )

    info = batch.sampling_info
    if sim_folded:
        sampled = getattr(info, "sampling_mask_batch_indices", None) is not None
    else:
        sampled = not (info is None or (info.is_all_greedy and verify_logits_adjustments_are_noop(info)))
    return bool(
        batch.has_grammar
        or batch.return_logprob
        or sampled
        or any(os.environ.get(name) for name in DEBUG_ENVS)
    )


@sglang_route(f"{EXECUTOR}._forward_prepared_verify", versions=VERSIONS, around=True)
def _forward_prepared_verify(orig, self, *args, **kwargs):
    result = orig(self, *args, **kwargs)
    d = _deferred(result.logits_output.next_token_logits)
    if d is not None and d.fused:
        d.owed = bool(result.can_run_cuda_graph)
        if d.owed:
            batch = kwargs.get("batch", args[0] if args else None)
            if _reads_logits(batch, _SIM_STEP is self):
                d.fill()
            else:
                stats.set_info("verify_logits", "gathered only when read")
    return result


# the step whose simulated accept the verify graph made: (executor, bs); then its commit
_SIM_ACCEPT: tuple | None = None
_SIM_COMMIT: tuple | None = None
# the executor whose verify forward runs a sim-foldable step (during run_non_compact)
_SIM_STEP = None


def _sim_foldable(self, batch, sampling_info, bs: int) -> bool:
    """Whether this step's eager accept and commit are the static graph's: simulated
    acceptance with a greedy bonus of a sampling batch whose logits nothing adjusts."""
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        verify_logits_adjustments_are_noop,
    )

    epilogue = self.verify_epilogue
    return bool(
        epilogue is not None
        and batch is not None
        and self._simulated_correct_len is not None
        and not epilogue.simulate_fold
        and sampling_info is not None
        and not sampling_info.is_all_greedy
        and verify_logits_adjustments_are_noop(sampling_info)
        and not batch.has_grammar
        and not batch.return_logprob
        # sglang's begin_static_step of this step: a static graph, its commit gated off
        # (or that call deferred, the buffers still armed by the last such step)
        and (epilogue._static_step_state == (bs, False) or epilogue.__dict__.get("_aklr_pending") == bs)
        and epilogue.folds_commit
    )


@sglang_route(f"{EPILOGUE}.begin_static_step", versions=VERSIONS, around=True)
def _begin_static_step(orig, self, bs, armed):
    """With the buffers still armed for this ``bs`` by the last simulated accept, gating
    the commit off waits for ``run_non_compact``, which keeps them or makes the call."""
    self.__dict__["_aklr_begin"] = orig
    if (
        not armed
        and serving()
        and _flag(SIM_ENV)
        and _flag(SIM_KEEP_ENV)
        and self.__dict__.get("_aklr_armed") == bs
        and self._static_step_state == (bs, True)
    ):
        self.__dict__["_aklr_pending"] = bs
        return None
    self.__dict__["_aklr_pending"] = self.__dict__["_aklr_armed"] = None
    return orig(self, bs, armed)


@sglang_route(f"{EXECUTOR}.run_non_compact", versions=VERSIONS, around=True)
def _run_non_compact(orig, self, *args, **kwargs):
    global _SIM_ACCEPT, _SIM_COMMIT
    _SIM_ACCEPT = _SIM_COMMIT = None
    ids = kwargs.get("verify_ids_2d")
    epilogue = self.verify_epilogue
    pending = None
    if epilogue is not None:
        pending = epilogue.__dict__.get("_aklr_pending")
    if (
        args
        or ids is None
        or not serving()
        or not _flag(SIM_ENV)
        or not _sim_foldable(self, kwargs.get("batch"), kwargs.get("sampling_info"), ids.shape[0])
    ):
        if pending is not None:  # the deferred begin_static_step(bs, False)
            epilogue.__dict__["_aklr_pending"] = epilogue.__dict__["_aklr_armed"] = None
            epilogue.__dict__["_aklr_begin"](epilogue, pending, False)
        return orig(self, *args, **kwargs)
    bs = ids.shape[0]
    epilogue.__dict__["_aklr_pending"] = None
    if pending is None:
        # begin_static_step(bs, True) but its verify lengths, unchanged; sglang's next
        # begin_static_step(bs, False) waits for this route (_begin_static_step)
        epilogue.inject_gate_buf.fill_(1)
    epilogue._static_step_state = (bs, True)
    epilogue.__dict__["_aklr_armed"] = bs
    global _SIM_STEP
    _SIM_STEP = self
    try:
        result = orig(self, *args, **kwargs)
    finally:
        _SIM_STEP = None
    if result.can_run_cuda_graph:
        _SIM_ACCEPT = (self, bs)
    else:
        stats.record(f"{EXECUTOR}.run_non_compact", "fallback", "an eager verify forward")
    return result


@sglang_route(f"{EXECUTOR}.accept_and_finalize", versions=VERSIONS, around=True)
def _accept_and_finalize(orig, self, *args, **kwargs):
    global _SIM_ACCEPT, _SIM_COMMIT
    sim, _SIM_ACCEPT = _SIM_ACCEPT, None
    if not kwargs.get("folded_accept", True):
        mine = sim is not None and sim[0] is self and kwargs.get("bs") == sim[1]
        d = _deferred(kwargs.get("target_logits"))
        if d is not None and d.owed and not mine:
            d.fill()
        if mine:
            _SIM_COMMIT = sim
            stats.record(f"{EXECUTOR}.run_non_compact", "served")
            stats.set_info("simulated_accept", "the verify graph's")
            return self.verify_epilogue.read_accept(sim[1])
    return orig(self, *args, **kwargs)


@sglang_route(f"{EXECUTOR}.commit_hidden", versions=VERSIONS, around=True)
def _commit_hidden(orig, self, *args, **kwargs):
    global _SIM_COMMIT
    sim, _SIM_COMMIT = _SIM_COMMIT, None
    if (
        sim is not None
        and sim[0] is self
        and kwargs.get("bs") == sim[1]
        and kwargs.get("run_compact") is False
    ):
        return None  # the verify graph committed these rows
    return orig(self, *args, **kwargs)
