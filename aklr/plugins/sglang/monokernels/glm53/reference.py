# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The launches in torch, in fp32, one ``<contract>_reference(point, group_args, rank, decode)`` per
contract: each rank's layer on its own inputs, the attention and MLP outputs summed over the group in
rank order (TODO). The FP8 linears and the MoE are deepseek_v2's (``aklr.plugins.sglang.models``), the
sparse MLA dsa_backend's (``aklr.plugins.sglang.layers``)."""
