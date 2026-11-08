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

import logging

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


def _candidate(contract_id: str, run):
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


def _gemm_is_out_of_contract(x_scale, w_scale, split_k=None) -> str | None:
    """Why this GEMM call is outside AKLR's contract, or None.

    Two limits AITER's signature allows and AKLR's contract does not:
    AITER also accepts E8M0 group32 scales, where AKLR declares fp32 per
    128x128 of b and 1x128 of a; and AITER's split_k has no counterpart here.
    """

    import torch

    if x_scale.dtype != torch.float32 or w_scale.dtype != torch.float32:
        return "scales are not fp32"
    if split_k is not None:
        return "split_k has no counterpart in this contract"
    return None


class Provider:
    name = "aklr"
    api_versions = {plugin.API_VERSION}
    op_versions = {
        "rmsnorm",
        "rmsnorm_with_add",
        "gemm_a8w8_blockscale",
        "gemm_a8w8_blockscale_bpreshuffle",
    }

    def select(self, op, args, kwargs):
        """Answer for this call, cheaply, without running anything.

        AKLR's dispatcher exposes exactly this question as fallback_reason: it
        looks the call up in the dispatch table and reports why it would not
        serve, running nothing. A Point with no kernel for this device, none
        measured, or a measured speedup under AKLR_MIN_SPEEDUP answers here
        rather than by failing later.
        """

        if not _up():
            return None

        if op.name == "rmsnorm":
            value, weight, epsilon = args[0], args[1], args[2]
            if rms_norm.fallback_reason(value, weight, epsilon) is not None:
                return None
            return _candidate("rms_norm@1.0", _run_rmsnorm)

        if op.name == "rmsnorm_with_add":
            out, value, residual_in, residual_out, weight, epsilon = args[:6]
            if len(args) > 6 and args[6]:
                return None  # gemma_norm is not in AKLR's contract
            if add_rms_norm.fallback_reason(
                value, residual_in, weight, epsilon, out, residual_out
            ) is not None:
                return None
            return _candidate("add_rms_norm@1.0", _run_rmsnorm_with_add)

        if op.name in ("gemm_a8w8_blockscale", "gemm_a8w8_blockscale_bpreshuffle"):
            XQ, WQ, x_scale, w_scale = args[:4]
            bpreshuffled = op.name.endswith("_bpreshuffle")
            split_k = args[6] if len(args) > 6 and not bpreshuffled else None
            if not bpreshuffled and len(args) > 5 and args[5]:
                bpreshuffled = True  # isBpreshuffled=True
            outside = _gemm_is_out_of_contract(x_scale, w_scale, split_k)
            if outside is not None:
                return None
            if gemm_a8w8_blockwise_scaled.fallback_reason(
                XQ,
                WQ,
                x_scale,
                w_scale,
                out_dtype=args[4] if len(args) > 4 else None,
                operand_format="none:preshuffled" if bpreshuffled else "none:none",
            ) is not None:
                return None
            return _candidate(
                "gemm_a8w8_blockwise_scaled@1.0",
                _run_gemm_bpreshuffle if bpreshuffled else _run_gemm_blockscale,
            )

        return None


provider = Provider()
