// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// AKLR: a DSpark verify's committed target-hidden KV into every draft stage's SWA cache in one launch, as
// sglang's fused_k_norm_rope_flashmla (kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh) per stage.
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "common/device.h"

namespace infera_flash::dsv41 {

struct SwaStoreParams {
    static constexpr int DIM = 512, ROPE = 64, MAX_STAGES = 4;

    const bfloat16 *kv;           // [rows, kv_stride]: stage s's row at kv + s * stage_stride
    int64_t         kv_stride;    // elements
    int64_t         stage_stride; // elements
    const bfloat16 *weight[MAX_STAGES]; // [DIM] the stage's kv_norm
    const float    *freqs[MAX_STAGES];  // [positions, ROPE] cos / sin pairs
    // [pages, page_bytes]: a page's rows, then their scale rows. V4: 576 B rows (448 e4m3 dims, a UE8M0 scale a 64,
    // then the 64 rotated dims bf16), 8 B scale rows; V4.1: 512 B rows (e4m3, a UE8M0 scale a 32), 16 B scale rows
    uint8_t        *cache[MAX_STAGES];
    const void     *pos; // [rows] int64 / int32
    bool            pos_i32;
    bool            v41;
    const int32_t  *loc;                // [rows] < 0: no slot
    // AKLR: loc null: row r's slot from the verify's committed rows, as sglang's committed_swa_locations: map[cache_loc[r]]
    // (a negative location wrapped) where r % width < commit[r / width], else none
    const int64_t *cache_loc;
    const int64_t *map;
    const int32_t *commit;
    int64_t        map_n;
    int            width;
    int             rows, stages, page_bits;
    int64_t         page_bytes;
    float           eps;
};

void swa_store(const SwaStoreParams &p, hipStream_t stream);

} // namespace infera_flash::dsv41
