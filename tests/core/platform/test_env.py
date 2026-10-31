# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aklr.core.platform.env import ENV_DISABLE, RuntimeEnv


class TestRuntimeEnv:
    """The ``AKLR_*`` variables parsed; a bad value is a ``ValueError``."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("AKLR_ENABLE_TRACING", "2"),
            ("AKLR_MIN_SPEEDUP", "abc"),
            ("AKLR_MIN_SPEEDUP", "nan"),
        ],
    )
    def test_bad_value_raises(self, monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
        """``RuntimeEnv()`` raises ``ValueError`` naming the variable."""
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            RuntimeEnv()

    def test_disable_names_op_types(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AKLR_DISABLE=gemm,bmm`` disables those two op types, not all."""
        monkeypatch.setenv(ENV_DISABLE, "gemm,bmm")
        env = RuntimeEnv()
        assert env.disabled_op_types == {"gemm", "bmm"}
        assert not env.is_all_disabled
