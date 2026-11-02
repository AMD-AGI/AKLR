// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// index_q (Infera-Flash's IndexQParams, csrc/dsv41/indexer.h) with its outputs in K2i's FP4 layout.
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "common/device.h"

namespace infera_flash::dsv41 {

struct IndexQPackParams {
    static constexpr int DIM = 128, MAX_ROWS = 8, QRANK = 1280;

    const bfloat16 *qr;      // [rows, QRANK] the attention's normed q latent
    const uint8_t  *w;       // [heads * DIM, QRANK] e4m3
    const uint8_t  *ws;      // [heads * DIM / 32, QRANK / 32] ue8m0
    const bfloat16 *x;       // [rows, dim] the attention's normed input
    const bfloat16 *wp;      // [heads, dim] weights_proj
    const int64_t  *pos;     // [rows]
    const float    *table;   // [positions, pairs, 2]
    uint8_t        *q_fp4;   // [rows, heads, DIM / 2]
    uint8_t        *q_scale; // [rows, 1, 4, 16, 4]
    bfloat16       *weights; // [rows, heads]
    // the compressor's GEMV of x in the same launch (Infera-Flash's gate_gemv, csrc/dsv41/moe_route.h): gate_out
    // [rows, gate_n] fp32 = x . gate_w [gate_n, dim] bf16, fp32 sums; gate_w null: none
    const bfloat16 *gate_w;
    float          *gate_out;
    int             gate_n;
    float           weight_scale;
    int64_t         positions;
    int             rows, heads, dim, pairs;
    int             total; // > rows: groups of rows rows (grid y), every pointer moved on by the group's first row
};

void index_q_pack(const IndexQPackParams &p, hipStream_t stream);

} // namespace infera_flash::dsv41
