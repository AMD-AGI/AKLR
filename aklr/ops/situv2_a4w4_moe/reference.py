# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the contract says, in torch: fused_moe's reference with MXFP4 activations (act_quant 4), SiTUv2, gate and
up separated, no expert parallelism."""

from aklr.core.contract.reference import OpReference
from aklr.ops.fused_moe.reference import FusedMoeReference

__all__ = ["Situv2A4w4MoeReference"]


class Situv2A4w4MoeReference(OpReference):
    def reference(self, **call):
        """Each token through its top-k experts, weighted by its routing weights, summed."""
        return FusedMoeReference().reference(
            **call, activation="situv2", act_quant=4, gate_mode="separated", doweight_stage1=False
        )
