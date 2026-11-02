// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Activation and KV quantization the way DeepSeek-V4.1's reference (inference/kernel.py) rounds: fp8 1x32 ue8m0 e4m3fn
// RNE (2^ceil(log2(max(amax, 1e-4) / 448))), fp4 1x32 ue8m0 e2m1 (floor 6 * 2^-126), fp4s 1x16 e4m3fn-scaled e2m1
// (floor 6 * 2^-9). act_quant writes fp8 bytes and scales; qdq writes the dequantized values back as bf16.
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "common/device.h"

namespace infera_flash::dsv41 {

enum class QFormat : int { FP8 = 0, FP4 = 1, FP4S = 2, FP8DS = 3 };

struct ActQuantParams {
    const bfloat16 *x;     // [rows, cols]
    uint8_t        *q;     // [rows, cols] e4m3fn
    uint8_t        *scale; // [rows, cols / 32] ue8m0
    int64_t         groups; // rows * cols / 32
};

struct QdqParams {
    bfloat16 *x; // [rows, cols], rewritten in place
    int64_t   groups;
    QFormat   format;
    // pos non-null: first RoPE on the last 2 * pairs of every width values (rounded as rope_), at pos[i / per] for
    // the i-th width values (the indexer's q [m, heads, head_dim]: width head_dim, per heads)
    const int64_t *pos;
    const float   *table; // [positions, pairs, 2]
    int            pairs, width, per;
};

// A compressed entry's write: RMSNorm (gamma non-null), RoPE on the last 2 * pairs dims at the group's first
// position max(pos + 1 - ratio, 0) (rounded as rope_), qdq to format, into out row pos / ratio when pos completes a
// group, else the scratch row (out's last). A block a row, dim / 8 lanes.
struct NormRopeQdqStoreParams {
    const bfloat16 *x;     // [rows, dim], row stride x_stride
    const bfloat16 *gamma; // [dim] or null
    const float    *table; // [positions, pairs, 2]
    const int64_t  *pos;   // [rows]
    bfloat16       *out;   // [entries + 1, dim]
    int64_t         rows, x_stride, scratch;
    int             dim, pairs, ratio;
    float           eps;
    QFormat         format;
    uint8_t        *records; // FP8DS: also the completed entries' records (blocks of rec_blk), or null
    int             rec_blk;
    // FP4: also the entries packed, or null: e2m1 codes two a byte (the even dim low) [entries + 1, dim / 2] and their
    // ue8m0 scales [entries + 1, dim / 32]
    uint8_t        *fp4, *fp4_scale;
};

// bf16 rows on the FP8DS grid <-> fp8_ds_mla records: row r is record r
struct RecordsParams {
    bfloat16 *rows;    // [n, 512]
    uint8_t  *records; // [blocks, blk, DS_REC]
    int64_t   n;
    int       blk;
};

void pack_records(const RecordsParams &p, hipStream_t stream);
void unpack_records(const RecordsParams &p, hipStream_t stream);

void act_quant(const ActQuantParams &p, hipStream_t stream);
void qdq(const QdqParams &p, hipStream_t stream);
void norm_rope_qdq_store(const NormRopeQdqStoreParams &p, hipStream_t stream);

// A kv source's compressor after its GEMV, a block a row: the latent (ratio > 1: compress_pool's pooling of kv_score
// [rows, 2 DIM] and the rings, updated; ratio 1: kv_score rounded to bf16), RMSNorm, the entry through comp; with wk
// the index key wk . latent through index (FP4 with k_norm). comp.x / index.x unused.
struct CompressStoreParams {
    static constexpr int DIM = 512, IDIM = 128, MAX_ROWS = 16, MAX_RATIO = 4;

    const float           *kv_score;
    float                 *kv_ring, *sc_ring; // [ring, DIM] (ratio > 1)
    const bfloat16        *gamma;             // [DIM]
    const bfloat16        *wk;                // [IDIM, DIM] or null
    NormRopeQdqStoreParams comp, index;
    int64_t                rows;
    int                    ring, ratio;
    float                  eps;
    // a batch: sequences of seq_rows rows (0: one), sequence s's rings, entries, records and index keys at s times
    // ring * DIM floats, comp_stride / index_stride elements, rec_stride bytes
    int64_t                seq_rows, comp_stride, index_stride, rec_stride;
};

void compress_store(const CompressStoreParams &p, hipStream_t stream);

// 2^ceil(log2(x)) for x > 0, from the bits as the reference does (no log2 rounding)
__device__ inline float pow2_ceil(float x) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, x);
    const int      e    = static_cast<int>((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffff) != 0);
    return __builtin_bit_cast(float, static_cast<uint32_t>(e + 127) << 23);
}

// e2m1 round to nearest even of |v| <= 6: 0 0.5 1 1.5 2 3 4 6
__device__ inline float fp4_round(float v) { // selects top down, no branches (NaN: 6, as the if chain gave)
    const float a = fabsf(v);
    float       r = 6.0f;
    r             = a <= 5.0f ? 4.0f : r;
    r             = a < 3.5f ? 3.0f : r;
    r             = a <= 2.5f ? 2.0f : r;
    r             = a < 1.75f ? 1.5f : r;
    r             = a <= 1.25f ? 1.0f : r;
    r             = a < 0.75f ? 0.5f : r;
    r             = a <= 0.25f ? 0.0f : r;
    return copysignf(r, v);
}

// the e2m1 code of r, a value of the fp4 grid (fp4_round's)
__device__ inline uint32_t e2m1_code(float r) {
    const uint32_t b = __builtin_bit_cast(uint32_t, r), a = b & 0x7fffffffu;
    const uint32_t mag = a >= 0x3f800000u ? ((a >> 23) - 126u) << 1 | (a >> 22 & 1u) : (a != 0u ? 1u : 0u);
    return mag | (b >> 28 & 8u);
}

// e4m3fn of v (|v| <= 448), round to nearest even, as its byte and its value
__device__ inline uint8_t fp8_byte(float v) {
    return static_cast<uint8_t>(__builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false) & 0xff);
}
__device__ inline float fp8_value(uint8_t b) { return __builtin_amdgcn_cvt_f32_fp8(b, 0); }

// fp8_ds_mla, vLLM's V4 KV record (the mono layer's caches): dims < 448 e4m3 with one UE8M0 a 64,
// 2^ceil(log2(max(amax, 1e-4) / 448)); the RoPE dims 448.. bf16. A record is 576 data bytes (448 e4m3, 64 bf16);
// a block of `blk` records keeps their 8 scale bytes each (7 codes, a zero) after its data rows.
constexpr int DS_NOPE = 448, DS_DATA = 576, DS_SCALES = 8, DS_REC = DS_DATA + DS_SCALES;

// a lane's dims c .. c + 7 of a 512-row, a 64-group's 8 lanes adjacent: the group's scale (every lane computes one)
__device__ inline float ds_scale(const float (&v)[8]) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(amax, __shfl_xor(amax, 1));
    amax = fmaxf(amax, __shfl_xor(amax, 2));
    amax = fmaxf(amax, __shfl_xor(amax, 4));
    return pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
}

// v onto the record's grid: the NoPE dims through e4m3 at scale s, the RoPE dims as they are (bf16 already)
__device__ inline void ds_qdq(float (&v)[8], int c, float s) {
    if (c < DS_NOPE) {
        const float inv = __builtin_bit_cast(float, (254u << 23) - __builtin_bit_cast(uint32_t, s)); // 1 / s exactly
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = fp8_value(fp8_byte(fminf(fmaxf(v[i] * inv, -448.0f), 448.0f))) * s;
        }
    }
}

// a lane's 8 values (on the grid, scale s) into record `entry` of a cache of blk-record blocks
__device__ inline void ds_store_record(uint8_t *cache, int blk, int64_t entry, int c, const float (&v)[8], float s) {
    uint8_t *page = cache + entry / blk * (static_cast<int64_t>(blk) * DS_REC);
    uint8_t *data = page + entry % blk * DS_DATA;
    if (c < DS_NOPE) {
        const float inv = __builtin_bit_cast(float, (254u << 23) - __builtin_bit_cast(uint32_t, s));
        uint8_t     q[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            q[i] = fp8_byte(v[i] * inv);
        }
        *reinterpret_cast<uint2 *>(data + c) = *reinterpret_cast<const uint2 *>(q);
        uint8_t *scales = page + static_cast<int64_t>(blk) * DS_DATA + entry % blk * DS_SCALES;
        if (c % 64 == 0) {
            scales[c / 64] = static_cast<uint8_t>(__builtin_bit_cast(uint32_t, s) >> 23);
        }
        if (c == 0) {
            scales[7] = 0;
        }
    } else {
        bfloat16 b[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            b[i] = static_cast<bfloat16>(v[i]);
        }
        *reinterpret_cast<uint4 *>(data + DS_NOPE + (c - DS_NOPE) * 2) = *reinterpret_cast<const uint4 *>(b);
    }
}

// a lane's 8 values of record `entry`
__device__ inline void ds_load_record(const uint8_t *cache, int blk, int64_t entry, int c, float (&v)[8]) {
    const uint8_t *page = cache + entry / blk * (static_cast<int64_t>(blk) * DS_REC);
    const uint8_t *data = page + entry % blk * DS_DATA;
    if (c < DS_NOPE) {
        const uint8_t code = page[static_cast<int64_t>(blk) * DS_DATA + entry % blk * DS_SCALES + c / 64];
        const float   s    = __builtin_bit_cast(float, static_cast<uint32_t>(code) << 23);
        const uint2   q    = *reinterpret_cast<const uint2 *>(data + c);
        const uint8_t *b   = reinterpret_cast<const uint8_t *>(&q);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = fp8_value(b[i]) * s;
        }
    } else {
        const uint4     w = *reinterpret_cast<const uint4 *>(data + DS_NOPE + (c - DS_NOPE) * 2);
        const bfloat16 *b = reinterpret_cast<const bfloat16 *>(&w);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = static_cast<float>(b[i]);
        }
    }
}

} // namespace infera_flash::dsv41
