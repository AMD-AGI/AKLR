# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3's launches on one rank of the TP group, one contract per kind (TODO):

- ``glm53_moe_layer``: an MoE layer reading the previous indexer layer's top-k (57 of the 78 layers).
- ``glm53_moe_index_layer``: an MoE layer selecting its own top-2048 (18 layers).
- ``glm53_dense_index_layer``: a dense layer selecting its own top-2048 (layers 0 to 2).
- ``glm53_head``: the final norm, the LM head and the logits' all-gather.

Each has ``ranks="tp_size"``, as ``mega_moe``. A layer takes the hidden stream, the rows' positions
and KV slots, the top-k slots, its KV cache (mutable) and SGLang's weights as it holds them (FP8 at a
scale per 128 x 128 block, in AITER's 16 x 16 shuffle), and returns the stream plus its attention and
MLP outputs, each summed over the group.
"""

from __future__ import annotations

__all__ = ["CONTRACTS"]

CONTRACTS = ()
