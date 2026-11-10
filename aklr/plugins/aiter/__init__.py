# SPDX-License-Identifier: MIT
"""Serve AITER's routed ops from AKLR's kernels.

Unlike the sglang and vllm plugins this one replaces nothing. AITER keeps the
call, owns the contract, and asks per call whether we serve it; we answer, and
may decline. Declining is ordinary operation, not a failure.

Declared through packaging:

    [project.entry-points."aiter.plugins"]
    aklr = "aklr.plugins.aiter:provider"

Installing it does not enable it. AITER is default-deny, and enablement has a
single owner: a framework turns us on with
``aiter.plugin.configure(enable=["aklr"], source=...)``, which records who
asked. Nothing in the environment can enable a provider; ``AITER_PLUGINS=0``
only turns routing off.
"""

from __future__ import annotations

import dataclasses
import logging

import torch
from aiter import plugin

from aklr.ops.gemm.api import gemm_a8w8_blockwise_scaled
from aklr.ops.rms_norm.api import add_rms_norm, rms_norm

_LOGGER = logging.getLogger("aklr.plugins.aiter")
_started: bool | None = None


def _up() -> bool:
    """Bring AKLR's dispatcher up once, and survive not being able to.

    AKLR builds its dispatch table against the device it finds and raises where
    it does not recognise one. Without a table every lookup answers NO_SCHEMA,
    so we would decline every call for a reason that says nothing. Start once,
    remember the answer, and decline honestly if it did not work.

    This runs on the first select() rather than at import: importing a provider
    must not do work AITER did not ask for. It is the one call here that is not
    cheap, and it happens once.
    """

    global _started
    if _started is None:
        try:
            from aklr.core.runtime.dispatch._registry import DispatcherRegistry

            DispatcherRegistry.default().init_once()
            _started = True
        except Exception as error:  # noqa: BLE001 - our failure, not AITER's
            _LOGGER.info("AKLR cannot serve in this process: %s", error)
            _started = False
    return _started


# Set once: AITER gained the field in ROCm/aiter#7654, and this adapter has to
# work against the releases on either side of it.
_REPORTS_LATENCY = "latency_us" in {
    field.name for field in dataclasses.fields(plugin.KernelCandidate)
}


def _reported_latency_us(contract_id: str) -> float | None:
    """What AKLR can tell AITER this kernel costs, for the call in hand.

    Nothing, today. That is the honest answer, not a placeholder.

    AITER asks one question: for *this call*, how many microseconds does this
    kernel take. AKLR's dispatcher cannot answer it -- `lookup()` returns a
    pair of fast keys and a runnable, and there is no timing anywhere on that
    path.

    AKLR does hold timings, in `aklr.core.validation`: `TimingRecord` carries
    `candidate_milliseconds` against a baseline. Reporting those would be
    worse than reporting nothing. They are measured offline, on the shapes
    validation picked, as a ratio against a versus kernel -- while AITER labels
    whatever it records with the conditions of the call in front of it. Handing
    it a validation number attaches a measurement to a shape it never ran on,
    which is precisely what the protocol's "omit rather than extrapolate" rule
    exists to prevent. A number that looks sourced and is not is harder to
    catch than a missing one.

    To answer properly AKLR needs a read-only accessor on the dispatcher that,
    given the resolved kernel and the call's shapes, returns the cost it
    expects from its own tuning table. That does not exist yet. Until it does,
    omission is correct and costs nothing -- which is why the field is optional
    on AITER's side.
    """

    return None


def _candidate(contract_id: str, run):
    extra = {}
    if _REPORTS_LATENCY:
        extra["latency_us"] = _reported_latency_us(contract_id)
    return plugin.KernelCandidate(
        provider="aklr",
        # TODO: AKLR's dispatcher knows which kernel file serves a Point -- the
        # uuid hashes the kernel -- but does not expose it. AITER's provenance
        # wants that identity rather than the contract's, so this needs a small
        # read-only accessor on the AKLR side.
        kernel_id=contract_id,
        artifact_digest=f"aklr:{contract_id}",
        qualification="https://github.com/AMD-AGI/AKLR",
        run=run,
        **extra,
    )


# --------------------------------------------------------------------------
# rms_norm


def _run_rmsnorm(input, weight, epsilon, use_model_sensitive_rmsnorm=0):
    return rms_norm(input, weight, epsilon)


def _run_rmsnorm_with_add(
    out,
    input,
    residual_in,
    residual_out,
    weight,
    epsilon,
    gemma_norm=False,
    use_model_sensitive_rmsnorm=0,
):
    add_rms_norm(input, residual_in, weight, epsilon, out, residual_out)


# --------------------------------------------------------------------------
# gemm
#
# AITER exposes the shuffled-b weight layout as a separate public function;
# AKLR models it as a layout variant of one contract, selected by
# operand_format. So both AITER ops map to the same AKLR contract with a
# different operand_format, and neither side has to change for that.


def _run_gemm_blockscale(
    XQ, WQ, x_scale, w_scale, dtype=None, isBpreshuffled=False, split_k=None
):
    return gemm_a8w8_blockwise_scaled(
        XQ,
        WQ,
        x_scale,
        w_scale,
        out_dtype=dtype,
        operand_format="none:preshuffled" if isBpreshuffled else "none:none",
    )


def _run_gemm_bpreshuffle(XQ, WQ, x_scale, w_scale, dtype=None, out=None):
    return gemm_a8w8_blockwise_scaled(
        XQ,
        WQ,
        x_scale,
        w_scale,
        out_dtype=dtype,
        out=out,
        operand_format="none:preshuffled",
    )


def _gemm_is_out_of_contract(x_scale, w_scale, split_k) -> str | None:
    """Why this GEMM call is outside AKLR's contract, or None.

    Two limits AITER's signature allows and AKLR's contract does not:
    AITER also accepts E8M0 group32 scales, where AKLR declares fp32 per
    128x128 of b and 1x128 of a; and AITER's split_k has no counterpart here.
    """

    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        return "scales are not fp32"
    if split_k is not None:
        return "split_k has no counterpart in this contract"
    return None


# --------------------------------------------------------------------------
# One selector per contract. AITER applies the contract's defaults before
# calling, so each unpacks the full positional list by name: a contract that
# grew a parameter fails loudly here -- recorded as a select error, routed
# native -- instead of reading the wrong index quietly.


def _select_rmsnorm(args):
    value, weight, epsilon, _sensitive = args
    if rms_norm.fallback_reason(value, weight, epsilon):
        return None
    return _candidate("rms_norm@1.0", _run_rmsnorm)


def _select_rmsnorm_with_add(args):
    out, value, residual_in, residual_out, weight, epsilon, gemma_norm, _s = args
    if gemma_norm:
        return None  # gemma_norm is not in AKLR's contract
    if add_rms_norm.fallback_reason(
        value, residual_in, weight, epsilon, out, residual_out
    ):
        return None
    return _candidate("add_rms_norm@1.0", _run_rmsnorm_with_add)


def _select_gemm(args):
    XQ, WQ, x_scale, w_scale, dtype, is_bpreshuffled, split_k = args
    return _gemm_candidate(
        _run_gemm_blockscale,
        XQ,
        WQ,
        x_scale,
        w_scale,
        dtype,
        bool(is_bpreshuffled),
        split_k,
    )


def _select_gemm_bpreshuffle(args):
    XQ, WQ, x_scale, w_scale, dtype, _out = args
    return _gemm_candidate(
        _run_gemm_bpreshuffle, XQ, WQ, x_scale, w_scale, dtype, True, None
    )


def _gemm_candidate(run, XQ, WQ, x_scale, w_scale, dtype, bpreshuffled, split_k):
    # Both AITER entries reach one AKLR contract, but the run function follows
    # the contract, not the layout: each must accept its own contract's call,
    # which AITER checks and refuses. The shuffled layout is a flag inside the
    # blockscale contract and a separate entry point of its own; serving the
    # first with the second's run function offers a six-argument kernel for a
    # seven-argument call.
    operand_format = "none:preshuffled" if bpreshuffled else "none:none"
    if _gemm_is_out_of_contract(x_scale, w_scale, split_k):
        return None
    if gemm_a8w8_blockwise_scaled.fallback_reason(
        XQ, WQ, x_scale, w_scale, out_dtype=dtype, operand_format=operand_format
    ):
        return None
    return _candidate("gemm_a8w8_blockwise_scaled@1.0", run)


_SELECTORS = {
    "rmsnorm": _select_rmsnorm,
    "rmsnorm_with_add": _select_rmsnorm_with_add,
    "gemm_a8w8_blockscale": _select_gemm,
    "gemm_a8w8_blockscale_bpreshuffle": _select_gemm_bpreshuffle,
}


class Provider:
    name = "aklr"
    api_versions = {plugin.API_VERSION}
    op_versions = set(_SELECTORS)

    def select(self, op, args, kwargs):
        """Answer for this call, cheaply, without running anything.

        AKLR's dispatcher exposes exactly this question as fallback_reason: it
        looks the call up in the dispatch table and reports why it would not
        serve, running nothing. A Point with no kernel for this device, none
        measured, or a measured speedup under AKLR_MIN_SPEEDUP answers here
        rather than by failing later.
        """

        selector = _SELECTORS.get(op.name)
        if selector is None or not _up():
            return None
        return selector(args)


provider = Provider()
