# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, TypeAdapter, model_validator

from ._dtypes import _Dims, _operand
from ._frozen_model import FrozenModel

if TYPE_CHECKING:
    from torch import Tensor


def _shuffle_tiles(
    tensor: Tensor,
    split: tuple[int, ...],
    order: tuple[int, ...],
    *,
    inverse: bool = False,
) -> Tensor:
    import torch

    x = tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor
    lead = tuple(x.shape[:-2])
    if inverse:
        split = tuple(split[i] for i in order)
        order = tuple(sorted(range(len(order)), key=order.__getitem__))
    n = len(lead)
    x = x.reshape(*lead, *split).permute(*range(n), *(n + i for i in order))
    return x.contiguous().view(tensor.shape).view(tensor.dtype)


def _ceil_to(dim: int | str, multiple: int) -> int | str:
    if multiple == 1:
        return dim
    if isinstance(dim, int):
        return -(-dim // multiple) * multiple
    return f"ceil_div({dim}, {multiple}) * {multiple}"


_PRESHUFFLE_ORDER = (0, 2, 3, 1, 4)
_SCALE_ORDER = (0, 3, 5, 2, 4, 1)


def _e8m0_scales_split(rows: int, cols: int) -> tuple[int, ...]:
    return rows // 32, 2, 16, cols // 8, 2, 4


class Layout(FrozenModel):
    """How a row-major tensor is stored, and the shape it is stored in."""

    kind: str

    @abstractmethod
    def encode(self, tensor: Tensor) -> Tensor:
        """``tensor`` (row-major layout) as this layout stores it."""

    def storage_shape(self, shape: Sequence[int | str]) -> _Dims:
        """The stored shape of dtype storage shape ``shape``; a dim computed from ints
        only is an int, else an expression text."""
        return tuple(shape)

    def view_rewrite(self) -> str | None:
        """Statements that make a non-contiguous ``T`` (``call[K]``) the stored tensor,
        returning None when it is not this layout's bytes; None: only the stored
        tensor is accepted."""
        return None

    def storage_test(self) -> str:
        """Expression text: whether ``T``'s strides are the ones this layout stores."""
        return "T.is_contiguous()"


class RowMajor(Layout):
    """Row-major: the tensor as it is."""

    kind: Literal["row_major"] = "row_major"

    def encode(self, tensor):
        return tensor


class ColumnMajor(Layout):
    """The last two dims stored column by column: the transpose's bytes, read in the
    dtype storage shape."""

    kind: Literal["column_major"] = "column_major"

    def encode(self, tensor):
        return tensor.mT.contiguous().view(tensor.shape)

    def view_rewrite(self):
        # sglang passes these bytes as the zero-copy view with strides (1, rows)
        return "if not T.mT.is_contiguous():\n    return None\ncall[K] = T = T.mT.reshape(T.shape)"


class Preshuffle(Layout):
    """aiter ``shuffle_weight(x, layout=tile)`` on the last two dims: tiles of
    ``tile[0]`` rows by ``2 * tile[1]`` columns, each in 16-byte vectors."""

    kind: Literal["preshuffle"] = "preshuffle"
    tile: tuple[int, int]

    def encode(self, tensor):
        rows, cols = tensor.shape[-2:]
        tile_rows, tile_cols = self.tile[0], self.tile[1] * 2
        vector = 16 // tensor.element_size()  # elements per 16-byte load
        split = (
            rows // tile_rows,
            tile_rows,
            cols // tile_cols,
            tile_cols // vector,
            vector,
        )
        return _shuffle_tiles(tensor, split, _PRESHUFFLE_ORDER)


class E8M0Scales(Layout):
    """Block scales (one per ``block_size`` values) padded with ``0x00`` (e8m0 2**-127)
    to multiples of ``pad_to``, as aiter's ``dynamic_mxfp4_quant(shuffle=True)`` writes
    them; ``shuffled``: aiter ``shuffle_scale`` on the last 2 dims. aiter's asm
    ``gemm_a4w4`` reads the padded groups when K is not a multiple of 256: padded with
    1.0 (``0x7F``) instead, its output is wrong."""

    kind: Literal["e8m0_scales"] = "e8m0_scales"
    block_size: int
    pad_to: tuple[int, int] = (1, 1)
    shuffled: bool = False

    @model_validator(mode="after")
    def _check(self) -> E8M0Scales:
        if self.shuffled and (self.pad_to[0] % 32 or self.pad_to[1] % 8):
            raise ValueError("shuffled scales need pad_to multiples of (32, 8)")
        return self

    def encode(self, tensor):
        import torch

        row_major_bytes = tensor.view(torch.uint8)
        shape = self.storage_shape(row_major_bytes.shape)
        stored = row_major_bytes.new_zeros(shape)
        rows, cols = row_major_bytes.shape[-2:]
        stored[..., :rows, :cols] = row_major_bytes
        if self.shuffled:
            stored = _shuffle_tiles(stored, _e8m0_scales_split(*shape[-2:]), _SCALE_ORDER)
        return stored.view(tensor.dtype)

    def storage_shape(self, shape):
        *lead, rows, cols = shape
        return (*lead, _ceil_to(rows, self.pad_to[0]), _ceil_to(cols, self.pad_to[1]))


# The gate-up interleaved layouts are what sglang writes for a SiTUv2 call.
class GateUpInterleavedWeight(Layout):
    """aiter ``shuffle_weight`` with ``is_guinterleave=True`` on expert weights: 16-row
    by 64-byte tiles; with ``gate_up`` the gate and up halves alternate tile by tile."""

    kind: Literal["gate_up_interleaved_weight"] = "gate_up_interleaved_weight"
    gate_up: bool

    def encode(self, tensor):
        rows, cols = tensor.shape[-2:]
        if self.gate_up:
            split, order = (2, rows // 32, 16, cols // 64, 4, 16), (1, 0, 3, 4, 2, 5)
        else:
            split, order = (rows // 16, 16, cols // 64, 4, 16), _PRESHUFFLE_ORDER
        return _shuffle_tiles(tensor, split, order)


class GateUpInterleavedScales(Layout):
    """aiter ``shuffle_scale`` with ``is_guinterleave=True`` on per-expert e8m0 scales:
    padded with 1.0 (``0x7F``) to 8 columns, shuffled, stored 2-D ``(experts * rows,
    cols)``; with ``gate_up`` the gate and up halves alternate tile by tile."""

    kind: Literal["gate_up_interleaved_scales"] = "gate_up_interleaved_scales"
    gate_up: bool

    def encode(self, tensor):
        import torch

        row_major_bytes = tensor.view(torch.uint8)
        experts, rows, cols = row_major_bytes.shape
        stored = row_major_bytes.new_full((experts, rows, _ceil_to(cols, 8)), 0x7F)
        stored[..., :cols] = row_major_bytes
        split, order = _e8m0_scales_split(*stored.shape[-2:]), _SCALE_ORDER
        if self.gate_up:
            split = (2, rows // 32, 16, stored.shape[-1] // 8, 2, 4)
            order = (1, 3, 5, 2, 4, 0)
        stored = _shuffle_tiles(stored, split, order)
        shape = self.storage_shape(row_major_bytes.shape)
        return stored.view(shape).view(tensor.dtype)

    def storage_shape(self, shape):
        experts, rows, cols = shape
        if isinstance(experts, int) and isinstance(rows, int):
            product = experts * rows
        else:
            product = f"{_operand(experts)} * {_operand(rows)}"
        return product, _ceil_to(cols, 8)


_KV_PAGE_ALIGN = 576  # sglang DeepSeekV4TokenToKVPool.create_buffer


class Deepseekv4KvPages(Layout):
    """sglang's DeepSeek-V4 kv pool: page ``p``'s row-major ``(page_size, 1, width)``
    elements start at ``p * ceil_div(page_size * width, 576) * 576``; the elements
    between pages are padding."""

    kind: Literal["deepseekv4_kv_pages"] = "deepseekv4_kv_pages"

    def encode(self, tensor):
        pages, page_size, _, width = tensor.shape
        used = page_size * width
        stored = tensor.new_zeros(pages, _ceil_to(used, _KV_PAGE_ALIGN))
        stored[:, :used] = tensor.reshape(pages, used)
        return stored[:, :used].view(tensor.shape)

    def storage_test(self):
        a = _KV_PAGE_ALIGN
        return (
            "T.stride()[1:] == (T.shape[3], T.shape[3], 1)"
            f" and T.stride(0) == -(-T.shape[1] * T.shape[3] // {a}) * {a}"
        )


class KvSplitPages(Layout):
    """vLLM's paged KV cache: a ``(blocks, block_size, heads, head_dim)`` side is the
    first or second half of each row of a row-major ``(blocks, heads, block_size, 2 *
    head_dim)`` buffer, the key and the value sharing each row."""

    kind: Literal["kv_split_pages"] = "kv_split_pages"

    def encode(self, tensor):
        blocks, block_size, heads, head_dim = tensor.shape
        stored = tensor.new_zeros(blocks, heads, block_size, 2 * head_dim)
        stored[..., :head_dim] = tensor.transpose(1, 2)
        return stored.transpose(1, 2)[..., :head_dim]

    def storage_test(self):
        return (
            "T.stride() == (T.shape[1] * T.shape[2] * 2 * T.shape[3],"
            " 2 * T.shape[3], T.shape[1] * 2 * T.shape[3], 1)"
        )


class StridedRows(Layout):
    """A ``(rows, heads, head_dim)`` tensor whose rows are dense and any distance apart,
    at least a row: rows of a wider row-major buffer, as a query sliced from a fused
    projection's output. A row-major tensor is one too."""

    kind: Literal["strided_rows"] = "strided_rows"

    def encode(self, tensor):
        rows = tensor.flatten(1)
        stored = rows.new_zeros(rows.shape[0], 2 * rows.shape[1])
        stored[:, : rows.shape[1]] = rows
        return stored[:, : rows.shape[1]].unflatten(1, tensor.shape[1:])

    def storage_test(self):
        return "T.stride()[1:] == (T.shape[2], 1) and T.stride(0) >= T.shape[1] * T.shape[2]"


LayoutUnion = Annotated[
    RowMajor
    | ColumnMajor
    | Preshuffle
    | E8M0Scales
    | GateUpInterleavedWeight
    | GateUpInterleavedScales
    | Deepseekv4KvPages
    | KvSplitPages
    | StridedRows,
    Field(discriminator="kind"),
]
_LAYOUT_ADAPTER = TypeAdapter(LayoutUnion)


def stored_inputs(
    schema: Mapping[str, Any], row_major_args: Mapping[str, Any], *, device: str
) -> dict[str, Any]:
    """``row_major_args`` copied to ``device`` and encoded into the schema's one layout
    variant there (a move would drop padded strides), with the variant's tensor facts
    set; scalars as they are."""
    (variant,) = schema["layout_variants"]["by_name"].values()
    stored = {}
    for name, value in row_major_args.items():
        if schema["inputs"][name]["kind"] != "scalar":
            layout = _LAYOUT_ADAPTER.validate_python(variant["layouts"].get(name, {"kind": "row_major"}))
            value = layout.encode(value.to(device, copy=True))
        stored[name] = value
    for fact in variant["detect_facts"]:
        FactLocation(fact["location"]).set_on_tensor(stored, fact["name"], fact["expected_value"])
    return stored


class FactLocation(str, Enum):
    """Where a call shows a layout fact ``name``: ``ARG``, the api parameter ``name``;
    ``TENSOR_ATTRIBUTE``, ``name`` is ``"<tensor>.<attribute>"``, a Python attribute
    (absent: False)."""

    ARG = "arg"
    TENSOR_ATTRIBUTE = "tensor_attr"

    def read_source(self, name: str) -> str:
        """Python expression text for the fact's value over the mapping ``call`` (api
        parameter -> value), with ``_getattr`` bound to ``getattr`` (generated code)."""
        match self:
            case FactLocation.ARG:
                return f"call[{name!r}]"
            case FactLocation.TENSOR_ATTRIBUTE:
                tensor, _, attribute_name = name.partition(".")
                return f"_getattr(call[{tensor!r}], {attribute_name!r}, False)"
            case _:
                raise ValueError(f"no rule for {self!r}")

    def param_name(self, name: str) -> str | None:
        """The api parameter the fact ``name`` is, else None."""
        match self:
            case FactLocation.ARG:
                return name
            case FactLocation.TENSOR_ATTRIBUTE:
                return None
            case _:
                raise ValueError(f"no rule for {self!r}")

    def tensor_name(self, name: str) -> str | None:
        """The tensor the fact ``name`` is on, else None."""
        match self:
            case FactLocation.ARG:
                return None
            case FactLocation.TENSOR_ATTRIBUTE:
                return name.partition(".")[0]
            case _:
                raise ValueError(f"no rule for {self!r}")

    def set_on_tensor(self, stored: Mapping[str, Any], name: str, expected_value: Any) -> None:
        """Make the fact hold on its tensor in ``stored``, if present; one already
        holding is left alone."""
        match self:
            case FactLocation.ARG:
                pass
            case FactLocation.TENSOR_ATTRIBUTE:
                tensor_name, _, attribute_name = name.partition(".")
                tensor = stored.get(tensor_name)
                if tensor is not None and getattr(tensor, attribute_name, False) != (expected_value):
                    setattr(tensor, attribute_name, expected_value)
            case _:
                raise ValueError(f"no rule for {self!r}")
