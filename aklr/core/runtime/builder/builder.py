###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, List

from aklr.core.platform.device import DeviceTarget
from aklr.core.platform.env import RuntimeEnv
from aklr.core.runtime import LoadError
from aklr.core.runtime.builder.runnable import Runnable, RunnableMetadata
from aklr.core.utils import create_package_name, write_sources_to_path

__all__ = [
    "BuildError",
    "Builder",
    "FlyDSLBuilder",
    "PythonBuilder",
    "TorchBuilder",
    "TritonBuilder",
]

if TYPE_CHECKING:
    from aklr.core.schema import Schema
    from aklr.core.schema._kernel import Kernel


class BuildError(LoadError):
    """A schema's kernel could not be made callable; an ``OSError`` cause marks the platform's fault."""


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
        pass

    @abstractmethod
    def can_build(self, schema: Schema) -> bool:
        pass

    @abstractmethod
    def build(self, schema: Schema) -> Runnable:
        """Make ``schema``'s kernel callable without calling it; failures raise :class:`BuildError`."""

    def _get_package_name_and_build_path(self, kernel: Kernel) -> tuple[str, Path]:
        package_name = create_package_name(kernel, self._package_prefix)
        build_path = RuntimeEnv().build_cache_path / self._build_dir_name / package_name
        return package_name, build_path

    def _signature(self, callable: Callable) -> inspect.Signature:
        return inspect.signature(callable)

    def _try_validate_signature(self, callable: Callable, schema: Schema) -> None:

        try:
            signature = self._signature(callable)
        except (ValueError, TypeError):
            return
        assert schema.kernel is not None
        dps = schema.kernel.build.destination_passing_style
        expected_nparam = len(schema.inputs) + len(schema.outputs) if dps else len(schema.inputs)

        # the runtime passes exactly these args positionally: extra params need defaults
        sample_args = [object() for _ in range(expected_nparam)]
        try:
            signature.bind(*sample_args)
        except TypeError as e:
            style = "Destination-passing" if dps else "Value-returning"
            raise BuildError(
                f"{style} style callable: expected {expected_nparam} positional parameters, "
                f"but signature '{signature}' is incompatible: {e}"
            ) from e

        if not dps:
            num_outputs = len(schema.outputs)
            ret_ann = signature.return_annotation
            if ret_ann is not inspect.Signature.empty:
                origin = getattr(ret_ann, "__origin__", None)
                if origin is tuple:
                    args = getattr(ret_ann, "__args__", None)
                    if args is not None and len(args) != num_outputs:
                        raise BuildError(
                            f"Value-returning style callable with {num_outputs} outputs must "
                            f"return a {num_outputs}-element tuple, got {ret_ann}"
                        )
                elif ret_ann is None:
                    if num_outputs != 0:
                        raise BuildError(
                            f"Value-returning style callable with {num_outputs} outputs must not return None"
                        )
                elif num_outputs != 1:
                    raise BuildError(
                        f"Value-returning style callable returning non-tuple must "
                        f"only have one output, got {num_outputs}"
                    )


class PythonBuilder(Builder):
    """Imports the entry point from the sources written to the build cache, as a per-kernel package."""

    _LANGUAGE: ClassVar[str] = "python"

    def __init__(self) -> None:
        super().__init__(f"{self._LANGUAGE}_", self._LANGUAGE)

    @staticmethod
    def is_available() -> bool:
        return True

    def can_build(self, schema: Schema) -> bool:
        return schema.kernel is not None and schema.kernel.build.language == self._LANGUAGE

    def _get_cleaner(self, package: str, build_path: Path) -> Callable[[], None]:
        def cleaner() -> None:
            try:
                to_delete = [m for m in sys.modules if m == package or m.startswith(package + ".")]
                for m in to_delete:
                    sys.modules.pop(m, None)
            except Exception:
                pass

            try:
                build_path_str = str(build_path)
                if build_path_str in sys.path:
                    sys.path.remove(build_path_str)
            finally:
                shutil.rmtree(build_path, ignore_errors=True)

        return cleaner

    def build(self, schema: Schema) -> Runnable:
        kernel = schema.kernel
        assert kernel is not None

        entry_file = kernel.get_entry_path()
        entry_symbol = kernel.get_entry_symbol()

        if entry_file.suffix != ".py":
            raise BuildError(f"Entry file '{entry_file}' is not a Python file")

        package_name, build_path = self._get_package_name_and_build_path(kernel)
        module_name = package_name + "." + ".".join(Path(entry_file).with_suffix("").parts)

        package_path = build_path / package_name
        write_sources_to_path(package_path, kernel.sources)

        cleaner = self._get_cleaner(package_name, build_path)

        sys.path.insert(0, str(build_path))

        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            cleaner()
            raise BuildError(f"Failed importing module '{module_name}' from sources: {e}") from e

        try:
            fn: Any = getattr(mod, entry_symbol)
        except AttributeError as e:
            cleaner()
            raise BuildError(f"Entry symbol '{entry_symbol}' not found in module '{module_name}'") from e

        if not callable(fn):
            cleaner()
            raise BuildError(f"Entry symbol '{entry_symbol}' is not callable")

        metadata = RunnableMetadata(
            build_type=self._LANGUAGE,
            op_type=schema.op_type,
            schema_name=schema.name,
            kernel_name=kernel.name,
            destination_passing_style=kernel.build.destination_passing_style,
            schema=schema,
            misc={"module_name": module_name, "entry_symbol": entry_symbol},
        )

        self._try_validate_signature(fn, schema)
        return Runnable(callable=fn, metadata=metadata, cleaner=cleaner)

    def check_call_keys(self, schema: Schema, entry: Callable[..., Any]) -> None:
        call_keys = schema.kernel_call_keys
        try:
            signature = inspect.signature(entry)
        except (ValueError, TypeError):
            return
        try:
            signature.bind(**dict.fromkeys(call_keys))
        except TypeError as e:
            assert schema.kernel is not None
            raise BuildError(
                f"{schema.kernel.build.entry_point} does not take the call keys {list(call_keys)}: {e}"
            ) from e


class TritonBuilder(PythonBuilder):
    _LANGUAGE: ClassVar[str] = "triton"

    @staticmethod
    def is_available() -> bool:
        try:
            import triton  # noqa: F401
        except ImportError:
            return False
        return True


class FlyDSLBuilder(PythonBuilder):
    """FlyDSL is python too: its kernels JIT-compile themselves inside the imported module."""

    _LANGUAGE: ClassVar[str] = "flydsl"

    @staticmethod
    def is_available() -> bool:
        try:
            import flydsl  # noqa: F401
        except ImportError:
            return False
        return True


class TorchBuilder(Builder):
    """Builds hip sources for this GPU's arch into the TORCH_LIBRARY op ``torch.ops.<package>.<symbol>``."""

    _LANGUAGE: ClassVar[str] = "torch"

    _CPP_HIP_CUDA_EXTENSIONS: ClassVar[List[str]] = [".hip", ".cu", ".cpp", ".cc", ".cxx", ".c"]

    def __init__(self) -> None:
        # the package is TORCH_LIBRARY's namespace, pasted after "_": a leading "_" would make a reserved "__"
        super().__init__(f"aklr_{self._LANGUAGE}_", self._LANGUAGE)

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available()

    def can_build(self, schema: Schema) -> bool:
        return schema.kernel is not None and schema.kernel.build.language in ("hip", "cuda", "cpp")

    def _filter_sources(self, source_paths: List[Path]) -> List[str]:
        return [str(path) for path in source_paths if path.suffix in self._CPP_HIP_CUDA_EXTENSIONS]

    def _get_cleaner(self, build_dir: Path) -> Callable[[], None]:
        def cleaner() -> None:
            shutil.rmtree(build_dir, ignore_errors=True)

        return cleaner

    def _signature(self, callable: Callable) -> inspect.Signature:
        """The op's signature from its schema: ``inspect.signature`` sees only ``(*args, **kwargs)``."""
        import torch

        op_schema = callable._schema
        return inspect.Signature(
            [
                inspect.Parameter(
                    argument.name,
                    (
                        inspect.Parameter.KEYWORD_ONLY
                        if argument.kwarg_only
                        else inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                    default=(
                        argument.default_value if argument.has_default_value() else inspect.Parameter.empty
                    ),
                )
                for argument in op_schema.arguments
            ],
            return_annotation=(
                inspect.Signature.empty  # a list return has no fixed length to check
                if any(isinstance(r.type, torch.ListType) for r in op_schema.returns)
                else tuple[(Any,) * len(op_schema.returns)]
            ),
        )

    def build(self, schema: Schema) -> Runnable:
        import torch
        from torch.utils.cpp_extension import load

        assert schema.kernel is not None
        kernel = schema.kernel

        entry_file_extension = kernel.get_entry_path().suffix
        if entry_file_extension not in self._CPP_HIP_CUDA_EXTENSIONS:
            raise BuildError(
                f"Entry file type not recognized. Must be one of {self._CPP_HIP_CUDA_EXTENSIONS}, "
                f"got {entry_file_extension}."
            )

        symbol = kernel.get_entry_symbol()
        package_name, build_dir = self._get_package_name_and_build_path(kernel)
        source_paths = write_sources_to_path(build_dir, kernel.sources)

        cpp_cuda_paths = self._filter_sources(source_paths)

        extra_include_paths = [str(build_dir)]
        with_cuda = kernel.build.language in ("cuda", "hip")

        # a killed build leaves torch's lock and load() would wait forever; BuilderRegistry serializes builds
        (build_dir / "lock").unlink(missing_ok=True)
        arch = DeviceTarget.current().arch
        if arch is None:
            raise BuildError(f"{kernel.name}: no GPU arch to build for")
        try:
            # torch adds its archs (PYTORCH_ROCM_ARCH) unless extra_cflags, not extra_cuda_cflags, name one
            load(
                name=package_name,
                sources=cpp_cuda_paths,
                extra_include_paths=extra_include_paths,
                extra_cflags=[f"--offload-arch={arch}"],  # host c++ also gets it and fails: no .cpp sources
                extra_cuda_cflags=["-O3"],
                with_cuda=with_cuda,
                build_directory=build_dir,
                is_python_module=False,
                verbose=True,
            )
        except Exception as e:
            raise BuildError(f"Torch build failed for kernel '{kernel.name}': {e}") from e

        try:
            callable = getattr(getattr(torch.ops, package_name), symbol).default
        except AttributeError as e:
            raise BuildError(f"torch.ops.{package_name}.{symbol} is not registered") from e

        metadata = RunnableMetadata(
            build_type=self._LANGUAGE,
            op_type=schema.op_type,
            schema_name=schema.name,
            kernel_name=kernel.name,
            destination_passing_style=kernel.build.destination_passing_style,
            schema=schema,
            misc={"entry_symbol": symbol},
        )

        cleaner = self._get_cleaner(build_dir)

        self._try_validate_signature(callable, schema)

        return Runnable(callable=callable, metadata=metadata, cleaner=cleaner)
