# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from aklr.core.contract.op import OpContract
from aklr.core.contract.semantics import DTYPE_TRAITS, DType, torch_dtypes
from aklr.core.contract.spec import PointSpec

if TYPE_CHECKING:
    import torch

FastKey = tuple
CallArgs = dict[str, Any]  # an api call's arguments by parameter name


@dataclass(frozen=True, eq=False)
class FastKeyCodec:
    """``contract``'s FastKey layout, computed once; converts a Point to its key and
    back, offline. The generated closures read the other fields."""

    contract: OpContract
    # the key's entries by closure name: axes, ``_dt_<t>`` per tensor, attrs, layout index
    key_names: tuple[str, ...]
    # tensor -> (primary storage dtype -> DType value)
    dtype_names: Mapping[str, Mapping[torch.dtype, str]]
    # tensor -> (every storage dtype it may arrive in -> its primary storage dtype)
    primary_storage_dtypes: Mapping[str, Mapping[torch.dtype, torch.dtype]]
    # the 4-bit tensors, which store their last dim halved
    packed_tensors: frozenset[str]
    # the key's indices of any_size axes, which match_key leaves out
    any_size_indices: tuple[int, ...]

    @classmethod
    def from_contract(cls, contract: OpContract) -> FastKeyCodec:
        """``contract``'s layout. Imports torch."""
        tensors = contract.tensors()
        key_names = (
            *contract.domain.axes,
            *(f"_dt_{t}" for t in tensors),
            *contract.attrs,
            "_layout_variant_idx",
        )
        any_size_indices = tuple(i for i, n in enumerate(contract.domain.axes) if contract.axes[n].any_size)
        dtype_names, primary_storage_dtypes, packed_tensors = {}, {}, set()
        for t in tensors:
            dtypes = contract.dtypes_of(t)
            storages = [torch_dtypes(d) for d in dtypes]
            dtype_names[t] = MappingProxyType(
                {ds[0]: d.value for d, ds in zip(dtypes, storages, strict=True)}
            )
            primary_storage_dtypes[t] = MappingProxyType({s: ds[0] for ds in storages for s in ds})
            # One answer per tensor: rejects an any_of mixing 4-bit and wider.
            (is_packed,) = {DTYPE_TRAITS[d].bits == 4 for d in dtypes}
            if is_packed:
                packed_tensors.add(t)
        return cls(
            contract,
            key_names,
            MappingProxyType(dtype_names),
            MappingProxyType(primary_storage_dtypes),
            frozenset(packed_tensors),
            any_size_indices,
        )

    def match_key(self, key: FastKey) -> FastKey:
        """``key`` as kernels are matched: each any_size axis ``None``, so a schema
        serving one value of it serves every value."""
        values = list(key)
        for i in self.any_size_indices:
            values[i] = None
        return tuple(values)

    def to_fast_key(self, point: PointSpec) -> FastKey:
        """``point``'s FastKey; ``ValueError`` unless the domain contains it."""
        contract = self.contract
        if not contract.domain.contains(point):
            raise ValueError(f"{point} is outside {contract.contract_id}")
        dtypes = (point.dtypes[t] for t in contract.tensors())
        return (
            *(point.axes[n] for n in contract.domain.axes),
            *(None if d is None else torch_dtypes(DType(d))[0] for d in dtypes),
            *(point.attrs[n] for n in contract.attrs),
            contract.layout_variants.names().index(point.layout_variant_name),
        )

    def to_point(self, key: FastKey) -> PointSpec:
        """The exact inverse of ``to_fast_key``."""
        contract = self.contract
        axes, tensors = tuple(contract.domain.axes), tuple(contract.tensors())
        n_axes, n_axes_tensors = len(axes), len(axes) + len(tensors)
        *values, layout_variant_idx = key
        dtypes = {
            t: None if d is None else self.dtype_names[t][d]
            for t, d in zip(tensors, values[n_axes:n_axes_tensors], strict=True)
        }
        return PointSpec(
            contract_id=contract.contract_id,
            axes=dict(zip(axes, values[:n_axes], strict=True)),
            dtypes=dtypes,
            attrs=dict(zip(contract.attrs, values[n_axes_tensors:], strict=True)),
            layout_variant_name=contract.layout_variants.names()[layout_variant_idx],
        )
