###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Source: flashinfer_bench/compile/{builder,utils}.py and
#         flashinfer_bench/compile/builders/{python_builder,triton_builder}.py
# Modified by Advanced Micro Devices, Inc. for the AKLR project.
# Modified for aklr/core/runtime/builder
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of aiter (see LICENSE).
###############################################################################


from __future__ import annotations

import inspect
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

import torch

from aklr.core.contract.locate import import_entry
from aklr.core.contract.spec import write_sources
from aklr.core.jit import hip_extension
from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import (
    ENV_DISABLE,
    ENV_KERNEL_CACHE_DIR,
    RuntimeEnv,
)
from aklr.core.runtime import LoadError
from aklr.core.runtime.builder.runnable import Runnable, RunnableMetadata

if TYPE_CHECKING:
    from collections.abc import Collection

    from aklr.core.contract.op import OpContract
    from aklr.core.contract.spec import SourceFile

__all__ = [
    "KERNEL_PACKAGE_PREFIX",
    "BuildError",
    "Builder",
    "FlyDSLBuilder",
    "HipBuilder",
    "PythonBuilder",
    "TritonBuilder",
    "get_build_cache_path",
]

# every kernel package starts with it: bench detects a stray kernel import by it
KERNEL_PACKAGE_PREFIX: Final = "_aklr_"
_CACHE_HINT: Final = (
    f"set {ENV_KERNEL_CACHE_DIR} to a writable directory, or {ENV_DISABLE}=1 to start without AKLR"
)


class BuildError(LoadError):
    """A schema's kernel could not be made callable; ``__cause__`` an ``OSError``: the platform's fault."""


def get_build_cache_path() -> Path:
    """``RuntimeEnv().build_cache_path``, the kernel cache's root."""
    try:
        root = RuntimeEnv().build_cache_path
    except ValueError as e:
        raise BuildError(str(e)) from None
    if root is None:
        raise BuildError(f"no kernel cache: HOME is not set; {_CACHE_HINT}")
    return root


class Builder(ABC):
    """One language's route from a schema's sources to a :class:`Runnable`."""

    _package_prefix: str
    _build_dir_name: str

    def __init__(self, package_prefix: str, build_dir_name: str) -> None:
        self._package_prefix = package_prefix
        self._build_dir_name = build_dir_name

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Whether this builder's toolchain is installed here."""

    @abstractmethod
    def can_build(self, schema: OpContract) -> bool:
        """Whether this builder claims ``schema.kernel.language``."""

    @abstractmethod
    def build(self, schema: OpContract) -> Runnable:
        """Import ``schema``'s kernel, running none of it; raises :class:`BuildError`."""

    def _get_package_name_and_build_path(self, schema: OpContract) -> tuple[str, Path]:
        assert schema.kernel is not None
        package = f"{self._package_prefix}{schema.kernel.content_hash[:16]}"
        root = get_build_cache_path()
        return package, root / self._build_dir_name / package

    def _update_sources(self, sources: Sequence[SourceFile], build_path: Path) -> None:
        try:
            changed = [
                source
                for source in sources
                if not (build_path / source.path).is_file()
                or (build_path / source.path).read_bytes() != source.content.encode()
            ]
            if changed:
                write_sources(changed, build_path)
        except OSError as e:
            raise BuildError(f"cannot write {build_path}: {e}; {_CACHE_HINT}") from e


class PythonBuilder(Builder):
    """Writes the sources into the kernel cache and imports the entry point as a package made in memory."""

    _LANGUAGE: ClassVar[str] = "python"

    def __init__(self) -> None:
        super().__init__(f"{KERNEL_PACKAGE_PREFIX}{self._LANGUAGE}_", self._LANGUAGE)

    @staticmethod
    def is_available() -> bool:
        """Python builds always work."""
        return True

    def can_build(self, schema: OpContract) -> bool:
        """Whether ``schema``'s kernel is in this builder's language."""
        return schema.kernel is not None and schema.kernel.language == self._LANGUAGE

    def build(self, schema: OpContract) -> Runnable:
        """``_update_sources``, then import ``entry_point`` from an empty package made in memory."""
        kernel = schema.kernel
        assert kernel is not None
        entry_file = kernel.entry_file
        package, build_path = self._get_package_name_and_build_path(schema)
        # the registry builds a content hash once: another hash with these 16 digits
        if package in sys.modules:
            raise BuildError(f"{package} is already imported")
        # before the package exists: nothing to clean up, and not an import failure
        self._update_sources(kernel.sources, build_path)

        def clean_up_build() -> None:
            for name in [name for name in sys.modules if f"{name}.".startswith(f"{package}.")]:
                sys.modules.pop(name, None)

        module_name = f"{package}.{entry_file.removesuffix('.py')}"
        try:
            module = types.ModuleType(package)
            module.__path__ = [str(build_path)]
            sys.modules[package] = module
            entry = import_entry(f"{module_name}.{kernel.entry_symbol}")
            if not callable(entry):
                raise TypeError(f"{type(entry).__name__} is not callable")
            call_keys = schema.kernel_call_keys
            try:
                signature = inspect.signature(entry)
            except (ValueError, TypeError):  # uninspectable entry: bind nothing
                signature = None
            if signature is not None:
                try:
                    signature.bind(**dict.fromkeys(call_keys))
                except TypeError as e:
                    raise BuildError(
                        f"{kernel.entry_point} does not take the call keys {list(call_keys)}: {e}"
                    ) from e
            if len(call_keys) < len(schema.api_params):  # an optional attr hides some
                kernel_fn = entry
                entry = lambda **call: kernel_fn(**{k: call[k] for k in call_keys})
        except BaseException as e:  # KeyboardInterrupt too: leave no package behind
            clean_up_build()
            if isinstance(e, BuildError) or not isinstance(e, Exception):
                raise
            raise BuildError(f"cannot import {kernel.entry_point}: {type(e).__name__}: {e}") from e
        metadata = RunnableMetadata(
            build_type=self._LANGUAGE,
            contract_id=schema.contract_id,
            schema_name=schema.name,
            content_hash=kernel.content_hash,
            misc={"module_name": module_name, "entry_symbol": kernel.entry_symbol},
        )
        return Runnable(
            entry,
            metadata,
            cleaner=clean_up_build,
            state_bytes=kernel.state_bytes,
            peer_bytes=kernel.peer_bytes,
        )


class TritonBuilder(PythonBuilder):
    """Triton is python: only the language differs."""

    _LANGUAGE: ClassVar[str] = "triton"

    @staticmethod
    def is_available() -> bool:
        """Whether triton is importable here."""
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        return True


class FlyDSLBuilder(PythonBuilder):
    """FlyDSL is python too: it JIT-compiles from inside the imported module."""

    _LANGUAGE: ClassVar[str] = "flydsl"

    @staticmethod
    def is_available() -> bool:
        """Whether flydsl is importable here."""
        try:
            import flydsl  # noqa: F401
        except ImportError:
            return False
        return True


class HipBuilder(Builder):
    """Builds the sources into a torch extension for this GPU's arch
    (``aklr.core.jit.hip_extension``); the kernel is its op
    ``torch.ops.<package>.<entry symbol>``."""

    _LANGUAGE: ClassVar[str] = "hip"

    def __init__(self) -> None:
        # the package names the op namespace too, which TORCH_LIBRARY pastes after an
        # underscore: a leading one would make a reserved C++ identifier
        super().__init__("aklr_hip_", self._LANGUAGE)

    @staticmethod
    def is_available() -> bool:
        """Whether torch is a ROCm build and hipcc and ninja are here."""
        return hip_extension.is_available()

    def can_build(self, schema: OpContract) -> bool:
        """Whether ``schema``'s kernel is in this builder's language."""
        return schema.kernel is not None and schema.kernel.language == self._LANGUAGE

    def build(self, schema: OpContract) -> Runnable:
        """The library's entry op, the library built first on a new toolchain; raises :class:`BuildError` unless the op takes exactly the call keys and writes exactly the outputs, the mutable inputs and any state."""
        kernel = schema.kernel
        assert kernel is not None
        package, build_path = self._get_package_name_and_build_path(schema)
        arch = DeviceTarget.current().arch
        if arch is None:
            raise BuildError(f"{kernel.entry_point}: no GPU arch to build for")
        sources = {source.path: source.content for source in kernel.sources}
        try:
            library = hip_extension.load_library(package, sources, build_path, arch)
        except OSError as e:
            raise BuildError(f"cannot build in {build_path}: {e}; {_CACHE_HINT}") from e
        except RuntimeError as e:
            raise BuildError(f"{kernel.entry_point}: {e}") from e
        try:
            op = getattr(getattr(torch.ops, package), kernel.entry_symbol).default
        except (AttributeError, RuntimeError):
            raise BuildError(
                f"{kernel.entry_point}: the library registers no op {kernel.entry_symbol}"
                " in TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m)"
            ) from None
        state = ("state",) if kernel.state_bytes else ()
        _check_op_schema(
            kernel.entry_point,
            op._schema,
            schema.kernel_call_keys,
            (*schema.mutable_tensors, *state),
        )
        metadata = RunnableMetadata(
            build_type=self._LANGUAGE,
            contract_id=schema.contract_id,
            schema_name=schema.name,
            content_hash=kernel.content_hash,
            misc={"library": str(library), "op": f"{package}::{kernel.entry_symbol}"},
        )
        # a loaded library stays: no cleaner
        return Runnable(
            op,
            metadata,
            state_bytes=kernel.state_bytes,
            peer_bytes=kernel.peer_bytes,
        )


def _check_op_schema(
    entry_point: str,
    op_schema: torch._C.FunctionSchema,
    call_keys: Collection[str],
    mutable_tensors: Collection[str],
) -> None:
    names = [a.name for a in op_schema.arguments]
    if sorted(names) != sorted(call_keys):
        raise BuildError(f"{entry_point} takes {names}, not the call keys {list(call_keys)}")
    writes = {a.name for a in op_schema.arguments if a.alias_info is not None and a.alias_info.is_write}
    if writes != set(mutable_tensors) or op_schema.returns:
        raise BuildError(
            f"{entry_point}: {op_schema} must write exactly"
            f" {sorted(mutable_tensors)}, each a Tensor(a!), and return ()"
        )
