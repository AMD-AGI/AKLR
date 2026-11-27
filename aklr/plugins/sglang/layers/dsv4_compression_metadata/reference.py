# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aklr.core.contract.reference import OpReference

__all__ = ["Dsv4CompressionMetadataReference"]


class Dsv4CompressionMetadataReference(OpReference):
    def reference(self, *, seq_lens, positions, raw_out_loc, page_table, c4_out_loc, c128_page_indices, **_):
        seq = seq_lens.long()
        pos = positions.long()
        n_write = raw_out_loc.shape[0]
        loc = raw_out_loc.long()
        max_pages = page_table.shape[1]
        c128_page_size = c128_page_indices.shape[1] // max_pages
        out = {}
        for ratio, name in ((4, "c4"), (128, "c128")):
            compress = (seq[:n_write] % ratio) == 0
            out[f"{name}_out_loc"] = torch.where(compress, torch.div(loc, ratio, rounding_mode="trunc"), 0)
            out[f"{name}_positions"] = (pos & ~(ratio - 1)).int()
            raw = torch.div(seq, ratio, rounding_mode="trunc")
            out[f"{name}_seq_lens_raw"] = raw.int()
            out[f"{name}_seq_lens_clamp1"] = raw.clamp_min(1).int()
        cols = torch.arange(c128_page_size * max_pages, device=page_table.device)
        pages = page_table.long()[:, cols // c128_page_size]
        slots = pages * c128_page_size + cols % c128_page_size
        valid = cols[None, :] < out["c128_seq_lens_raw"].long()[:, None]
        out["c128_page_indices"] = torch.where(valid, slots, -1).int()
        return out
