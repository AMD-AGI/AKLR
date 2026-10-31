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
    """A schema's kernel could not be made callable; ``__cause__`` an ``OSError``: the platform's fault."""


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
    def can_build(self, schema: Schema) -> bool:
        """Whether this builder claims ``schema.kernel.language``."""

    @abstractmethod
    def build(self, schema: Schema) -> Runnable:
        """Import ``schema``'s kernel, running none of it; raises :class:`BuildError`."""

    def _get_package_name_and_build_path(self, kernel: Kernel) -> tuple[str, Path]:
        package_name = create_package_name(kernel, self._package_prefix)
        build_path = RuntimeEnv().build_cache_path / self._build_dir_name / package_name
        return package_name, build_path

    def _signature(self, callable: Callable) -> inspect.Signature:
        return inspect.signature(callable)

    def _try_validate_signature(self, callable: Callable, schema: Schema) -> None:

        # Analyze the signature of the callable (skip all checks if signature unavailable)
        try:
            signature = self._signature(callable)
        except (ValueError, TypeError):
            return
        assert schema.kernel is not None
        dps = schema.kernel.build.destination_passing_style
        expected_nparam = len(schema.inputs) + len(schema.outputs) if dps else len(schema.inputs)

        # Validate against the actual call shape used by benchmark/apply runtime: a fixed number
        # of positional arguments derived from the definition. This accepts extra optional
        # parameters with defaults while still rejecting missing required or keyword-only
        # parameters that cannot be satisfied by positional calling.
        sample_args = [object() for _ in range(expected_nparam)]
        try:
            signature.bind(*sample_args)
        except TypeError as e:
            style = "Destination-passing" if dps else "Value-returning"
            raise BuildError(
                f"{style} style callable: expected {expected_nparam} positional parameters, "
                f"but signature '{signature}' is incompatible: {e}"
            ) from e

        # Check return annotation (only for value-returning style)
        if not dps:
            num_outputs = len(schema.outputs)
            ret_ann = signature.return_annotation
            if ret_ann is not inspect.Signature.empty:
                origin = getattr(ret_ann, "__origin__", None)
                if origin is tuple:
                    # If returning tuple, length must match num_outputs
                    args = getattr(ret_ann, "__args__", None)
                    if args is not None and len(args) != num_outputs:
                        raise BuildError(
                            f"Value-returning style callable with {num_outputs} outputs must "
                            f"return a {num_outputs}-element tuple, got {ret_ann}"
                        )
                elif ret_ann is None:
                    # If return None, num_outputs must be 0
                    if num_outputs != 0:
                        raise BuildError(
                            f"Value-returning style callable with {num_outputs} outputs must not return None"
                        )
                elif num_outputs != 1:
                    # If returning non-tuple, num_outputs must be 1
                    raise BuildError(
                        f"Value-returning style callable returning non-tuple must "
                        f"only have one output, got {num_outputs}"
                    )


class PythonBuilder(Builder):
    """Writes the sources into the kernel cache and imports the entry point as a package made in memory."""

    _LANGUAGE: ClassVar[str] = "python"

    def __init__(self) -> None:
        super().__init__(f"{self._LANGUAGE}_", self._LANGUAGE)

    @staticmethod
    def is_available() -> bool:
        """Python builds always work."""
        return True

    def can_build(self, schema: Schema) -> bool:
        """Whether ``schema``'s kernel is in this builder's language."""
        return schema.kernel is not None and schema.kernel.build.language == self._LANGUAGE

    def _get_cleaner(self, package: str, build_path: Path) -> Callable[[], None]:
        """Create a cleaner function that removes build artifacts.

        The cleaner unloads the imported module, removes it from sys.path, and
        deletes the build directory.

        Parameters
        ----------
        package : str
            The package name to unload from sys.modules.
        build_path : Path
            The directory to delete.

        Returns
        -------
        Callable[[], None]
            A function that performs the cleanup.
        """

        def cleaner() -> None:
            try:
                # Unload module and submodules
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
        """Write the changed sources, then import ``entry_point`` from an empty package made in memory."""
        kernel = schema.kernel
        assert kernel is not None

        entry_file = kernel.get_entry_path()
        entry_symbol = kernel.get_entry_symbol()

        if entry_file.suffix != ".py":
            raise BuildError(f"Entry file '{entry_file}' is not a Python file")

        package_name, build_path = self._get_package_name_and_build_path(kernel)
        module_name = package_name + "." + ".".join(Path(entry_file).with_suffix("").parts)

        # Create package directory structure: build_path/<package_name>/...
        package_path = build_path / package_name
        write_sources_to_path(package_path, kernel.sources)

        cleaner = self._get_cleaner(package_name, build_path)

        # Insert build_path into sys.path so we can import <package_name>.<module>
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
        """Raise :class:`BuildError` unless the call keys bind to ``entry``'s
        signature; an uninspectable entry binds anything."""
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


class TorchBuilder(Builder):
    """Builds hip sources into a torch extension for this GPU's arch with
    ``torch.utils.cpp_extension.load``; the kernel is its op ``torch.ops.<package>.<entry symbol>``."""

    _LANGUAGE: ClassVar[str] = "torch"

    _CPP_HIP_CUDA_EXTENSIONS: ClassVar[List[str]] = [".hip", ".cu", ".cpp", ".cc", ".cxx", ".c"]

    def __init__(self) -> None:
        # the package names the op namespace too, which TORCH_LIBRARY pastes after an
        # underscore: a leading one would make a reserved C++ identifier
        super().__init__(f"aklr_{self._LANGUAGE}_", self._LANGUAGE)

    @staticmethod
    def is_available() -> bool:
        """Whether torch is a ROCm build and ROCm's hipcc and ninja are here."""
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available()

    def can_build(self, schema: Schema) -> bool:
        """Whether ``schema``'s kernel is in this builder's language."""
        return schema.kernel is not None and schema.kernel.build.language in ("hip", "cuda", "cpp")

    def _filter_sources(self, source_paths: List[Path]) -> List[str]:
        """Filter source files to include only C/C++/CUDA files.

        Parameters
        ----------
        source_paths : List[Path]
            List of all source file paths.

        Returns
        -------
        List[str]
            List of source file paths with C/C++/CUDA extensions.
        """
        return [str(path) for path in source_paths if path.suffix in self._CPP_HIP_CUDA_EXTENSIONS]

    def _get_cleaner(self, build_dir: Path) -> Callable[[], None]:
        """Create a cleaner function that removes the build directory.

        Parameters
        ----------
        build_dir : Path
            The directory to delete.

        Returns
        -------
        Callable[[], None]
            A function that performs the cleanup.
        """

        def cleaner() -> None:
            shutil.rmtree(build_dir, ignore_errors=True)

        return cleaner

    def _signature(self, callable: Callable) -> inspect.Signature:
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
                inspect.Signature.empty
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

        (build_dir / "lock").unlink(missing_ok=True)
        arch = DeviceTarget.current().arch
        if arch is None:
            raise BuildError(f"{kernel.name}: no GPU arch to build for")
        try:
            load(
                name=package_name,
                sources=cpp_cuda_paths,
                extra_include_paths=extra_include_paths,
                extra_cflags=[f"--offload-arch={arch}"],
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
