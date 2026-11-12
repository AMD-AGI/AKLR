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

from typing import Any

from pydantic import BaseModel, Field, model_validator

from aklr.core.runtime._config_registry import ConfigRegistry
from aklr.core.runtime.tracing._policy import (
    FilterPolicy,
    InputDumpPolicy,
    PolicyRegistry,
)


class TracingConfig(BaseModel):
    input_dump_policy: str | list[str] = "dump_none"

    input_dump_policy_kwargs: dict[str, Any] = Field(default_factory=dict)

    # flush keeps one workload per axes anyway, and keep_all would hold every call in memory
    filter_policy: str = "keep_first_k_by_axes"

    filter_policy_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_policies(self) -> TracingConfig:
        if isinstance(self.input_dump_policy, str):
            if PolicyRegistry.get_input_dump_policy(self.input_dump_policy) is None:
                raise ValueError(
                    f"Unknown input_dump_policy: {self.input_dump_policy}. "
                    f"Available: {PolicyRegistry.list_input_dump_policies()}"
                )

        if PolicyRegistry.get_filter_policy(self.filter_policy) is None:
            raise ValueError(
                f"Unknown filter_policy: {self.filter_policy}. "
                f"Available: {PolicyRegistry.list_filter_policies()}"
            )
        return self

    def create_input_dump_policy(self) -> InputDumpPolicy:
        if isinstance(self.input_dump_policy, list):
            raise ValueError("input_dump_policy is a static list, use get_inputs_to_dump() instead")
        policy_cls = PolicyRegistry.get_input_dump_policy(self.input_dump_policy)
        if policy_cls is None:
            raise ValueError(f"Unknown input_dump_policy: {self.input_dump_policy}")
        return policy_cls(**self.input_dump_policy_kwargs)

    def create_filter_policy(self) -> FilterPolicy:
        policy_cls = PolicyRegistry.get_filter_policy(self.filter_policy)
        if policy_cls is None:
            raise ValueError(f"Unknown filter_policy: {self.filter_policy}")
        return policy_cls(**self.filter_policy_kwargs)

    def get_inputs_to_dump(self, names: list[str], values: list[Any]) -> dict[str, Any]:
        name_to_value = dict(zip(names, values))

        if isinstance(self.input_dump_policy, list):
            names_to_dump = self.input_dump_policy
        else:
            policy = self.create_input_dump_policy()
            names_to_dump = policy.dump(name_to_value)

        if not isinstance(names_to_dump, list):
            raise ValueError("input_dump_policy.dump() must return a list of strings")

        result: dict[str, Any] = {}
        for name in names_to_dump:
            if not isinstance(name, str) or name not in name_to_value:
                raise ValueError(f"input_dump_policy returned invalid input name: {name}")
            result[name] = name_to_value[name]
        return result


class TracingConfigRegistry(ConfigRegistry[TracingConfig]):
    """Tracing configs by schema name; a name with no config and no default is not traced."""
