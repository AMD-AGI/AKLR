from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from aklr.core.schema._kernel import Kernel, SourceFile


def dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = getattr(torch, dtype_str, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"Unsupported dtype '{dtype_str}'")


def torch_dtype_to_dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@cache
def _get_integer_dtypes() -> frozenset:
    import torch

    return frozenset(
        (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.bool,
        )
    )


def is_dtype_integer(dtype: torch.dtype) -> bool:
    return dtype in _get_integer_dtypes()


def write_sources_to_path(path: Path, sources: List[SourceFile]) -> List[Path]:
    """Write source files to a directory and return their paths. Create path if not exists.

    This function writes all source files from a kernel to a specified directory,
    creating subdirectories as needed. It performs security checks to prevent path
    traversal attacks and absolute path injection.

    Parameters
    ----------
    path : Path
        The root directory where source files will be written.
    sources : List[SourceFile]
        The list of source files to write. Each file's path must be relative and
        not contain parent directory references ("..").

    Returns
    -------
    List[Path]
        List of absolute paths to the written files.

    Raises
    ------
    AssertionError
        If any source file has an absolute path or contains path traversal.
    """
    path.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for src in sources:
        # Defensive assertion: path should be validated at Kernel creation time
        src_path_obj = Path(src.path)

        assert not src_path_obj.is_absolute(), f"Absolute path detected: {src.path}"
        assert ".." not in src_path_obj.parts, f"Path traversal detected: {src.path}"

        src_path = path / src.path

        # Ensure parent directories exist
        src_path.parent.mkdir(parents=True, exist_ok=True)

        # Write source file
        src_path.write_text(src.content)
        paths.append(src_path)

    return paths


def create_package_name(kernel: Kernel, package_prefix: str = "") -> str:
    """Generate a unique package name for a kernel.

    The package name is constructed from three parts:
    1. A prefix (typically identifying the builder)
    2. The normalized kernel name (alphanumeric and underscores only)
    3. A 6-character hash of the kernel content

    This ensures the package name is both human-readable and uniquely identifies
    the kernel's content.

    Parameters
    ----------
    kernel : Kernel
        The kernel to create a package name for.
    prefix : str, optional
        The prefix to prepend to the package name. Default is empty string.

    Returns
    -------
    str
        A unique package name in the format: {prefix}{normalized_name}_{hash}.

    Examples
    --------
    >>> create_package_name(kernel, "fib_python_")
    'fib_python_rmsnorm_v1_a3f2b1'
    """
    # Normalize the kernel name
    s = re.sub(r"[^0-9a-zA-Z_]", "_", kernel.name)
    if not s or s[0].isdigit():
        s = "_" + s

    return package_prefix + s + "_" + kernel.hash()[:6]
