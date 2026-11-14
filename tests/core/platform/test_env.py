# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aklr.core.platform.env import ENV_DISABLE, ENV_FLYDSL_FASTCALL, RuntimeEnv


class TestRuntimeEnv:
    """The ``AKLR_*`` variables parsed; a bad value is a ``ValueError``."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("AKLR_EXPORT", "2"),
            ("AKLR_MIN_SPEEDUP", "abc"),
            ("AKLR_MIN_SPEEDUP", "nan"),
            ("AKLR_FLYDSL_FASTCALL", "yes"),
        ],
    )
    def test_bad_value_raises(self, monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
        """``RuntimeEnv()`` raises ``ValueError`` naming the variable."""
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            RuntimeEnv()

    def test_disable_names_contracts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AKLR_DISABLE=gemm@1.0,bmm@1.0`` disables those two contracts, not all."""
        monkeypatch.setenv(ENV_DISABLE, "gemm@1.0,bmm@1.0")
        env = RuntimeEnv()
        assert env.disabled_contracts == {"gemm@1.0", "bmm@1.0"}
        assert not env.is_all_disabled

    @pytest.mark.parametrize(("value", "enabled"), [(None, True), ("1", True), ("0", False)])
    def test_flydsl_fastcall(self, monkeypatch: pytest.MonkeyPatch, value: str | None, enabled: bool) -> None:
        """``AKLR_FLYDSL_FASTCALL`` unset or ``1`` enables the fast path, ``0`` disables it."""
        if value is None:
            monkeypatch.delenv(ENV_FLYDSL_FASTCALL, raising=False)
        else:
            monkeypatch.setenv(ENV_FLYDSL_FASTCALL, value)
        assert RuntimeEnv().is_flydsl_fastcall_enabled is enabled
