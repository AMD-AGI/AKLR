# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from aklr.core.contract.locate import op_contracts, op_packages
from aklr.core.plugin import Plugin

__all__ = ["Plugin"]


def _declare_families(plugin: Plugin) -> None:
    """Contribute this host's published operator contracts during its atomic build."""
    prefix = f"aklr.plugins.{plugin.name}."
    for package in op_packages():
        if package.startswith(prefix):
            for contract in op_contracts(package):
                plugin.define(contract)
