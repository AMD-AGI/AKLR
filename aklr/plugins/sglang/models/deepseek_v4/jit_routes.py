# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4.1 prefill kernels compiled once, not once per context length: aiter's FP4 MQA
logits kernel with the block-table stride a runtime argument, and the prefill schedule's
Triton kernels jitted without argument specializations."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import torch

from aklr.plugins.sglang import stats

_LOGGER = logging.getLogger(__name__)

ENV = "AKLR_DSV41_PREFILL_JIT"
AITER_MODULE = "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
SCHEDULE_MODULE = "sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip"
# aiter's row plan binary search: covers s_max up to 2**13 (2M tokens of context)
ROW_PLAN_STEPS = 14
# the integer arguments of sglang's _prefill_schedule_prep_kernel
PREP_INTS = (
    "pt_src_stride",
    "pt_dst_stride",
    "T",
    "P",
    "s_max",
    "w_src",
    "w_dst",
    "pt_tiles_per_row",
)

# (old, new) edits of aiter's module source: the stride an argument of the kernel and its
# launcher, the compile cache keyed without it
_EDITS = (
    (
        "from .pa_mqa_logits_fp4_common import",
        "from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_common import",
    ),
    (
        "        stride_out_row: Int32,\n        weight_scale: fx.Float32,\n    ):",
        (
            "        stride_out_row: Int32,\n        weight_scale: fx.Float32,\n"
            "        stride_bt_row: Int32,\n    ):"
        ),
    ),
    (
        "phys_vec = bt_bt[batch_id * _stride_bt + bi_base]",
        "phys_vec = bt_bt[batch_id * stride_bt_row + bi_base]",
    ),
    (
        (
            "        stride_out: fx.Int32,\n        weight_scale: fx.Float32,\n"
            "        gx: fx.Int32,\n        stream: fx.Stream,\n    ):"
        ),
        (
            "        stride_out: fx.Int32,\n        weight_scale: fx.Float32,\n"
            "        gx: fx.Int32,\n        stride_bt: fx.Int32,\n        stream: fx.Stream,\n    ):"
        ),
    ),
    (
        "kfn(out, q, qs, kv, kvs, bt, w, cta_info_, stride_out, weight_scale).launch(",
        "kfn(out, q, qs, kv, kvs, bt, w, cta_info_, stride_out, weight_scale, stride_bt).launch(",
    ),
    (
        "    max_blocks_per_seq = block_tables.shape[1]\n\n    if (cta_info is None) != (n_ctas is None):",
        (
            "    max_blocks_per_seq = 0  # aklr: the stride is the launcher's argument\n\n"
            "    if (cta_info is None) != (n_ctas is None):"
        ),
    ),
    (
        "        float(weight_scale),\n        n_ctas,\n        stream,\n    )\n    return out",
        (
            "        float(weight_scale),\n        n_ctas,\n        block_tables.stride(0),\n"
            "        stream,\n    )\n    return out"
        ),
    ),
)


def _enabled() -> bool:
    value = os.environ.get(ENV) or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV}={value!r}: expected '1' or '0'")
    return value == "1"


def _strided_module():
    """aiter's FP4 prefill MQA logits module with the block-table stride a kernel argument,
    loaded from a file beside the kernel cache (FlyDSL reads kernel sources), or None.
    """
    from aklr.core.runtime.builder.builder import get_build_cache_path

    spec = importlib.util.find_spec(AITER_MODULE)
    if spec is None or spec.origin is None:
        return None, "aiter has no FP4 prefill MQA logits module"
    src = Path(spec.origin).read_text()
    for old, new in _EDITS:
        if src.count(old) != 1:
            return None, f"aiter's source differs at {old.strip()[:60]!r}"
        src = src.replace(old, new)
    import hashlib

    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = get_build_cache_path() / "dsv41" / f"pa_mqa_logits_fp4_prefill_strided_{digest}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)
    name = f"aklr_pa_mqa_logits_fp4_prefill_{digest}"
    mspec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    return module, None


class _Checked:
    """The strided launch, checked once against aiter's own on the first call."""

    def __init__(self, strided, original):
        self._strided, self._original = strided, original
        self._checked = False
        self.off = None

    def __call__(self, *args, **kwargs):
        if self.off is not None:
            return self._original(*args, **kwargs)
        if self._checked or torch.cuda.is_current_stream_capturing():
            return self._strided(*args, **kwargs)
        self._checked = True
        out = kwargs.pop("out", None)
        want = self._original(*args, **kwargs, out=None if out is None else out.clone())
        why = "the strided kernel's logits differ from aiter's"
        try:
            got = self._strided(*args, **kwargs, out=out)
        except Exception as e:  # noqa: BLE001 -- aiter's kernel serves instead
            got, why = None, f"the strided kernel failed: {type(e).__name__}: {e}"
        # bitwise: the pooled logits rows no CTA writes hold whatever they held (NaN too)
        same = got is not None and torch.equal(got.view(torch.int32), want.view(torch.int32))
        stats.set_info("prefill_mqa_logits", "strided (checked)" if same else "aiter's")
        if not same:
            self.off = why
            _LOGGER.warning("DeepSeek-V4.1 prefill MQA logits: %s; aiter's runs", self.off)
            if out is not None:
                out.copy_(want)
                return out
            return want
        return got


def _patch_mqa_logits() -> str:
    import aiter.ops.flydsl as flydsl_ops

    original = flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill
    module, why = _strided_module()
    if module is None:
        return why
    flydsl_ops.flydsl_pa_mqa_logits_fp4_prefill = _Checked(module.flydsl_pa_mqa_logits_fp4_prefill, original)
    return "strided"


def _patch_schedule() -> str:
    import importlib

    mod = importlib.import_module(SCHEDULE_MODULE)
    kernel = getattr(mod, "_prefill_schedule_prep_kernel", None)
    fn = getattr(kernel, "fn", None)
    if fn is None:
        return "no _prefill_schedule_prep_kernel"
    params = set(kernel.arg_names) if hasattr(kernel, "arg_names") else set()
    if params and not set(PREP_INTS) <= params:
        return "its integer arguments differ"
    mod._prefill_schedule_prep_kernel = _unspecialized(kernel)
    import importlib

    aiter_mod = importlib.import_module(AITER_MODULE)
    cta = getattr(aiter_mod, "_prefill_cta_info_kernel", None)
    if getattr(cta, "fn", None) is not None:
        aiter_mod._prefill_cta_info_kernel = _unspecialized(cta)
    return "arguments unspecialized"


def _unspecialized(kernel):
    """``kernel`` jitted again without Triton's specializations of its non-constexpr
    arguments (integers by divisibility by 16 and equality to 1, pointers by 16-byte
    alignment): its variants then follow its constexprs and dtypes only."""
    import triton

    names = [p.name for p in kernel.params if not p.is_constexpr]
    return triton.jit(kernel.fn, do_not_specialize=names)


class _FixedSteps:
    """aiter's row plan kernel launched with ``ROW_PLAN_STEPS`` binary-search steps, not
    ``s_max``'s bit length: the surplus steps change nothing (the search has converged),
    and one variant serves every context length."""

    def __init__(self, kernel):
        self.kernel, self.fn = kernel, kernel.fn

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            if kwargs.get("SEARCH_STEPS", ROW_PLAN_STEPS) <= ROW_PLAN_STEPS:
                kwargs["SEARCH_STEPS"] = ROW_PLAN_STEPS
            return launch(*args, **kwargs)

        return run


def _patch_row_plan() -> str:
    import importlib

    mod = importlib.import_module(AITER_MODULE)
    kernel = getattr(mod, "_prefill_row_plan_kernel", None)
    fn = getattr(kernel, "fn", None)
    if fn is None or not {"T", "P", "block_k", "s_max", "SEARCH_STEPS"} <= set(
        getattr(kernel, "arg_names", ())
    ):
        return "aiter's row plan kernel differs"
    mod._prefill_row_plan_kernel = _FixedSteps(_unspecialized(kernel))
    return f"{ROW_PLAN_STEPS} search steps, arguments unspecialized"


def prewarm(device) -> None:
    """Compile the prefill schedule's kernel variants now, not at the first prefill of each
    row count: sglang's prep kernel per power of two of the rows up to its fused limit,
    aiter's row plan and CTA kernels past it."""
    if not _enabled():
        return
    import time

    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import _guarded_pages
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
        MAX_FUSED_ROWS,
        build_prefill_schedule,
    )

    t0 = time.perf_counter()
    rows = [16 << i for i in range(16) if 16 << i <= MAX_FUSED_ROWS] + [2 * MAX_FUSED_ROWS]
    try:
        for t in rows:
            pt = torch.zeros((t, 4), dtype=torch.int32, device=device)
            le = torch.full((t,), 64, dtype=torch.int32, device=device)
            ctas = max(1024, t)
            build_prefill_schedule(
                page_table=pt,
                local_ends=le,
                cta_info_out=torch.empty((ctas, 6), dtype=torch.int32, device=device),
                parallel_unit_num=ctas,
                max_seq_len=_guarded_pages(4, 4) * 64,
                block_k=256,
            )
        torch.cuda.synchronize(device)
    except Exception as e:  # noqa: BLE001
        stats.set_info("prefill_prewarm", f"{type(e).__name__}: {e}")
        _LOGGER.warning("DeepSeek-V4.1 prefill schedule prewarm: %s", e)
        return
    how = f"{len(rows)} row counts in {time.perf_counter() - t0:.1f} s"
    stats.set_info("prefill_prewarm", how)
    _LOGGER.info("DeepSeek-V4.1 prefill schedule kernels compiled: %s", how)


_ACTIVE = False


def activate() -> None:
    global _ACTIVE
    if _ACTIVE:
        return
    _ACTIVE = True
    if not _enabled():
        stats.set_info("prefill_jit", f"{ENV}=0")
        return
    for name, patch in (
        ("prefill_mqa_logits", _patch_mqa_logits),
        ("prefill_schedule", _patch_schedule),
        ("prefill_row_plan", _patch_row_plan),
    ):
        try:
            how = patch()
        except Exception as e:  # noqa: BLE001
            how = f"{type(e).__name__}: {e}"
        stats.set_info(name, how)
        _LOGGER.info("DeepSeek-V4.1 %s: %s", name, how)
