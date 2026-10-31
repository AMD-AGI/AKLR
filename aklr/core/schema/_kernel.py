import hashlib
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from pydantic import Field, model_validator

from aklr.core._frozen_model import NonEmptyStr, frozen_dataclass
from aklr.core.platform.device import DeviceTarget

# kernel.language and impls/<language>/: a new language is a code PR first, then a schema PR.
Language = Literal["flydsl", "hip", "triton"]


@frozen_dataclass
class SourceFile:
    """A single source code file in a kernel implementation.

    Represents a source code file with its relative path and complete content.
    The file content is validated for syntax correctness based on the file extension.
    """

    path: NonEmptyStr
    """The relative path of the file, including its name and extension (e.g., 'src/kernel.cu',
    'main.py'). When compiling the kernel, a temporary kernel source directory will be
    created, and the file will be placed according to this path. The path should not contain
    parent directory traversal ("..")."""
    content: NonEmptyStr
    """The complete text content of the source file."""

    @model_validator(mode="after")
    def _validate_source_path(self) -> "SourceFile":
        """Validate source path for security.

        Raises
        ------
        ValueError
            If the path contains security issues (absolute paths or path traversal).
        """
        src_path = Path(self.path)
        if src_path.is_absolute():
            raise ValueError(f"Invalid source path (absolute path not allowed): {self.path}")
        if ".." in src_path.parts:
            raise ValueError(f"Invalid source path (parent directory traversal not allowed): {self.path}")
        return self


@frozen_dataclass
class Build:
    """Build specification for a kernel implementation.

    Contains all technical specifications required to build and execute a kernel, including
    language, hardware targets, dependencies, entry point, and build commands.
    """

    language: NonEmptyStr
    """The primary programming language (e.g., 'triton', 'cuda', 'python')."""
    target_hardware: Annotated[List[str], Field(min_length=1)]
    """Devices this kernel runs on, each '<arch>' or '<arch>/<hardware>', e.g. 'gfx950' (any
    gfx950 device) or 'gfx950/MI355X' (only MI355X); dispatch serves it only on a device matching
    one of them."""
    entry_point: NonEmptyStr
    """The exact path to the function to be called. Format: '{file_path}::{function_name}'
    (e.g., 'main.py::run')."""
    dependencies: List[NonEmptyStr] = Field(default_factory=list)
    """Optional list of required libraries or packages. E.g. for CUDA, we support 'cublas',
    'cudnn', 'cutlass'"""
    destination_passing_style: bool = True
    """Whether to use destination passing style for the kernel. If True, the kernel should
    accept the output tensors as the last arguments. If False, the kernel should return the
    output tensors."""
    binding: Optional[NonEmptyStr] = None
    """The binding type to use for C++/CUDA kernels. If None, defaults to 'tvm-ffi' for
    C++/CUDA languages. Ignored for Python and Triton languages."""

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "Build":
        """Validate entry_point format.

        Raises
        ------
        ValueError
            If entry_point doesn't follow the required format.
        """
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f'Invalid entry point format: {self.entry_point}. Expected "<file_path>::<function_name>".'
            )
        return self

    @model_validator(mode="after")
    def _validate_target_hardware(self) -> "Build":
        for name in self.target_hardware:
            try:
                DeviceTarget(*name.split("/"))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid target_hardware entry '{name}'. Expected '<arch>' or "
                    "'<arch>/<hardware>', e.g. 'gfx950' or 'gfx950/MI355X'."
                ) from e
        return self

    def runs_on(self, target: DeviceTarget) -> bool:
        """Whether ``target`` matches an entry of ``target_hardware``, both its arch and, if
        given, its hardware."""
        return DeviceTarget.matches_target(
            target, [DeviceTarget(*name.split("/")) for name in self.target_hardware]
        )


@frozen_dataclass
class Kernel:
    """A concrete implementation of the schema that carries it.

    Contains all source code,
    build specifications, and metadata required for building, interfacing, and
    benchmarking the implementation.
    """

    name: NonEmptyStr
    """A unique, human-readable name for this specific kernel (e.g., 'rmsnorm_triton_v1_h100')."""
    author: NonEmptyStr
    """The name of the author or agent system that created this kernel."""
    build: Build
    """Technical specifications for building and executing this kernel."""
    sources: Annotated[List[SourceFile], Field(min_length=1)]
    """Array of source code files representing the complete implementation."""
    description: Optional[str] = Field(default=None)
    """Optional human-readable description of the kernel's technique or approach."""

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "Kernel":
        """Validate source file paths for uniqueness and entry file existence.

        Raises
        ------
        ValueError
            If duplicate source file paths are found or the entry file is not found in the sources.
        """
        seen_paths = set()
        for source in self.sources:
            # Check for duplicates
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.build.entry_point.split("::")[0]

        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")

        return self

    def get_entry_path(self) -> Path:
        """Extract the file path from the entry point specification.

        The entry point format is '{file_path}::{function_name}', and this method
        returns the file path component as a Path object.

        Returns
        -------
        Path
            The relative path to the entry source file (e.g., 'main.py', 'src/kernel.cu').
        """
        return Path(self.build.entry_point.split("::")[0])

    def get_entry_symbol(self) -> str:
        """Extract the function/symbol name from the entry point specification.

        The entry point format is '{file_path}::{function_name}', and this method
        returns the function name component. This is the symbol that builders will
        look up in the compiled module or imported Python module.

        Returns
        -------
        str
            The function or symbol name to be loaded (e.g., 'run', 'forward', 'kernel').
        """
        return self.build.entry_point.split("::")[-1]

    def get_entry_source(self) -> Optional[SourceFile]:
        """Get the entry source file specified in the build spec.

        Returns
        -------
        Optional[SourceFile]
            The SourceFile object containing the entry point, or None if not found.
        """
        entry_path = self.build.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == entry_path:
                return source
        return None

    @cached_property
    def _hash_cache(self) -> str:
        # Memoized content hash (excluding name); cached_property bypasses the frozen __setattr__.
        h = hashlib.sha1()
        for s in (
            self.build.language,
            self.build.entry_point,
            self.build.binding if self.build.binding else "",
            str(self.build.destination_passing_style),
            *self.build.dependencies,
            *(part for src in self.sources for part in (src.path, src.content)),
        ):
            h.update(s.encode())
        return h.hexdigest()

    def with_unique_name(self) -> "Kernel":
        """Return a copy with name suffixed by content hash for uniqueness.

        The generated name is deterministic: same code always produces the
        same unique name regardless of submission order or environment.

        Returns
        -------
        Kernel
            A new Kernel instance with name ``{name}_{hash[:8]}``.
        """
        return replace(self, name=f"{self.name}_{self.hash()[:8]}")

    def hash(self) -> str:
        """Return the memoized deterministic hash of the kernel content (excluding name).

        This hash is computed from all fields that affect the kernel's behavior:
        language, entry point, binding, destination passing style, dependencies,
        and all source file paths and contents. This ensures that any meaningful change to the kernel
        results in a different hash.

        The hash is used for caching build artifacts, allowing kernels with the same
        hash to reuse the same cached build result.

        Returns
        -------
        str
            A SHA1 hash (40 hex characters) uniquely identifying this kernel's content.
        """
        return self._hash_cache

    def __hash__(self) -> int:  # pragma: no cover - trivial wrapper
        # Use the memoized content hash for fast hashing in dict/set keys.
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial wrapper
        if not isinstance(other, Kernel):
            return NotImplemented
        return self._hash_cache == other._hash_cache
