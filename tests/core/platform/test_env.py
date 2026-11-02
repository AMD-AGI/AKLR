# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aklr.core.platform.env import (
    ENV_DISABLE,
    ENV_PLUGIN_MIN_SPEEDUP,
    ENV_SGLANG_PLUGIN,
    RuntimeEnv,
)


class TestRuntimeEnv:
    """The ``AKLR_*`` variables parsed; a bad value is a ``ValueError``."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("AKLR_EXPORT", "2"),
            ("AKLR_MIN_SPEEDUP", "abc"),
            ("AKLR_MIN_SPEEDUP", "nan"),
            ("AKLR_PLUGIN_MIN_SPEEDUP", "inf"),
            ("AKLR_SGLANG_PLUGIN", "yes"),
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

    def test_disable_takes_a_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AKLR_DISABLE=aklr.plugins,gemm@1.0``: every plugin op package, one contract."""
        monkeypatch.setenv(ENV_DISABLE, "aklr.plugins,gemm@1.0")
        monkeypatch.setenv(ENV_SGLANG_PLUGIN, "1")
        env = RuntimeEnv()
        assert env.disabled_namespaces == {"aklr.plugins"}
        assert env.disabled_contracts == {"gemm@1.0"}
        assert env.is_package_disabled("aklr.plugins.sglang.models.x")
        assert not env.is_package_disabled("aklr.ops.gemm")

    def test_disable_takes_a_backend_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An item without ``@`` that is no namespace names a plugin's model backend."""
        monkeypatch.setenv(ENV_DISABLE, "aklr_deepseek_v41_dspark_draft,gemm@1.0")
        env = RuntimeEnv()
        assert env.disabled_names == {"aklr_deepseek_v41_dspark_draft"}
        assert env.disabled_contracts == {"gemm@1.0"}

    def test_the_sglang_plugin_is_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_SGLANG_PLUGIN, raising=False)
        assert not RuntimeEnv().is_sglang_plugin_enabled

    @pytest.mark.parametrize("plugin", ["0", "1"])
    def test_plugin_contracts_build_either_way(self, monkeypatch: pytest.MonkeyPatch, plugin: str) -> None:
        """The switch hooks sglang only: bench and validation serve plugin ops without it."""
        monkeypatch.delenv(ENV_DISABLE, raising=False)
        monkeypatch.setenv(ENV_SGLANG_PLUGIN, plugin)
        assert not RuntimeEnv().is_package_disabled("aklr.plugins.sglang.x")

    def test_plugin_ops_have_their_own_min_speedup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``AKLR_PLUGIN_MIN_SPEEDUP`` gates plugin ops alone; unset, they share
        ``AKLR_MIN_SPEEDUP``."""
        monkeypatch.setenv("AKLR_MIN_SPEEDUP", "1.1")
        monkeypatch.delenv(ENV_PLUGIN_MIN_SPEEDUP, raising=False)
        assert RuntimeEnv().min_speedup_of("aklr.plugins.sglang.x") == 1.1
        monkeypatch.setenv(ENV_PLUGIN_MIN_SPEEDUP, "0")
        env = RuntimeEnv()
        assert env.min_speedup_of("aklr.ops.gemm") == 1.1
        assert env.min_speedup_of("aklr.plugins.sglang.x") == 0
