# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build hooks of the amd-aiter distribution; its metadata is in pyproject.toml.

The hooks reproduce the default build of 3rdparty/aiter/setup.py; its kernel prebuild
(PREBUILD_KERNELS, PRETUNE_MODULES) is not supported, aiter compiles kernels at first use.
"""

import glob
import os
import shutil

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from setuptools_scm import get_version

AITER_DIR = "3rdparty/aiter"


class AiterBuildPy(build_py):
    """Install aiter's sources as the aiter_meta package and mark the install, as aiter's setup.py does."""

    def run(self):
        super().run()
        # An editable aiter has no install_mode and reads its sources in 3rdparty/aiter.
        if self.editable_mode:
            return
        meta_dir = os.path.join(self.build_lib, "aiter_meta")
        for name in ("3rdparty", "hsa", "gradlib", "csrc"):
            self.copy_tree(os.path.join(AITER_DIR, name), os.path.join(meta_dir, name))
        # Analysis dumps next to the runtime .co files.
        for asm_dir in glob.glob(os.path.join(meta_dir, "csrc/opus_gemm/gen_co/*/asm")):
            shutil.rmtree(asm_dir)
        open(os.path.join(meta_dir, "__init__.py"), "w").close()
        with open(os.path.join(self.build_lib, "aiter", "install_mode"), "w") as f:
            f.write("install")
        # aiter's top-level py_module: its JIT imports it to size every build. An editable
        # aiter finds it in 3rdparty/aiter, which aiter.jit.core puts on sys.path.
        self.copy_file(os.path.join(AITER_DIR, "aiter_worker_limits.py"), self.build_lib)


class PlatlibDistribution(Distribution):
    """aiter's wheel is platform specific: it carries .co kernels and JIT sources."""

    def has_ext_modules(self):
        return True


# AKLR's own commit; the version below is aiter's (root = 3rdparty/aiter in pyproject.toml).
# shortcut: read from git, so builds need a checkout, not an sdist.
aklr_version = get_version(
    relative_to=__file__,
    version_scheme=lambda scm: f"aklr.{scm.short_node}" + (".dirty" if scm.dirty else ""),
    local_scheme="no-local-version",
)
setup(
    # aiter's version, then AKLR's commit: 0.1.26+aklr.ge64734fac.
    use_scm_version={"local_scheme": lambda _: f"+{aklr_version}"},
    cmdclass={"build_py": AiterBuildPy},
    distclass=PlatlibDistribution,
)
