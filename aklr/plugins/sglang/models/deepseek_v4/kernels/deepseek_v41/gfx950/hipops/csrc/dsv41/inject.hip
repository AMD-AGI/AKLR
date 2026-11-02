// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Every DSpark draft stage's SWA KV of the committed target hidden in one launch (grid rows x stages): sglang's
// fused_k_norm_rope_flashmla (V4.1 rows by kv_layout.cuh's store_row_fp8) with the same sums in the same order, so
// the bytes are its bytes. V4: 448 dims e4m3 (UE8M0 a 64), 64 rotated bf16; V4.1: every dim e4m3 (UE8M0 a 32).
#include <c10/hip/HIPException.h>

#include "dsv41/inject.h"

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK // the rocm10 image's torch names it C10_CUDA_
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace infera_flash::dsv41 {

namespace {

using P = SwaStoreParams;

constexpr int SS_THREADS   = P::DIM / 2; // a thread 2 dims
constexpr int SS_LANES     = 32;         // sglang's warp
constexpr int SS_WARPS     = SS_THREADS / SS_LANES;
constexpr int SS_ROPE_WARP = SS_WARPS - 1; // its 64 dims are the rotated ones
static_assert(P::ROPE == SS_LANES * 2);

__device__ inline float bf16_round(float v) { return static_cast<float>(static_cast<bfloat16>(v)); }

template <int WIDTH> __device__ inline float lanes_sum(float v) {
#pragma unroll
    for (int m = WIDTH / 2; m >= 1; m >>= 1) {
        v = v + __shfl_xor(v, m, WIDTH);
    }
    return v;
}

template <int WIDTH> __device__ inline float lanes_max(float v) {
#pragma unroll
    for (int m = WIDTH / 2; m >= 1; m >>= 1) {
        v = fmaxf(v, __shfl_xor(v, m, WIDTH));
    }
    return v;
}

__device__ inline uint16_t pack_e4m3(float x, float y) {
    x = fmaxf(fminf(x, 448.0f), -448.0f);
    y = fmaxf(fminf(y, 448.0f), -448.0f);
    return static_cast<uint16_t>(__builtin_amdgcn_cvt_pk_fp8_f32(x, y, 0, false) & 0xffff);
}

__device__ inline float inv_scale_ue8m0(int32_t exponent) {
    return __uint_as_float(static_cast<uint32_t>(127 + 127 - exponent) << 23);
}

template <bool V41> __global__ void __launch_bounds__(SS_THREADS) swa_store_kernel(const P p) {
    constexpr int64_t ROW = V41 ? 512 : 576, SCALES = V41 ? 16 : 8;
    const int         row = blockIdx.x, stage = blockIdx.y, tx = threadIdx.x;
    const int         warp = tx / SS_LANES, lane = tx % SS_LANES;
    const bfloat16   *x = p.kv + row * p.kv_stride + stage * p.stage_stride;
    const int64_t     position =
        p.pos_i32 ? static_cast<const int32_t *>(p.pos)[row] : static_cast<const int64_t *>(p.pos)[row];
    const int pos32 = static_cast<int32_t>(position);
    int loc;
    if (p.loc != nullptr) {
        loc = p.loc[row];
    } else {
        loc = -1;
        if (row % p.width < p.commit[row / p.width]) {
            int64_t c = p.cache_loc[row];
            c         = c < 0 ? c + p.map_n : c;
            loc       = static_cast<int32_t>(p.map[c]);
        }
    }

    __shared__ float partial[SS_WARPS];
    float            data[2], freq[2] = {0.0f, 0.0f};
    {
        const float x0 = static_cast<float>(x[2 * tx]), x1 = static_cast<float>(x[2 * tx + 1]);
        const float w0 = static_cast<float>(p.weight[stage][2 * tx]);
        const float w1 = static_cast<float>(p.weight[stage][2 * tx + 1]);
        if (warp == SS_ROPE_WARP) {
            const float *f = p.freqs[stage] + pos32 * P::ROPE;
            freq[0]        = f[2 * lane];
            freq[1]        = f[2 * lane + 1];
        }
        float sum_of_squares = 0.0f;
        sum_of_squares += x0 * x0;
        sum_of_squares += x1 * x1;
        const float warp_sum = lanes_sum<SS_LANES>(sum_of_squares);
        if (lane == 0) partial[warp] = warp_sum;
        __syncthreads();
        sum_of_squares          = lanes_sum<SS_WARPS>(partial[lane % SS_WARPS]);
        const float norm_factor = rsqrtf(sum_of_squares / P::DIM + p.eps);
        data[0]                 = x0 * norm_factor * w0;
        data[1]                 = x1 * norm_factor * w1;
    }
    if (loc < 0) return;

    const int64_t page = loc >> p.page_bits, slot = loc & ((1 << p.page_bits) - 1);
    uint8_t      *base      = p.cache[stage] + page * p.page_bytes;
    uint8_t      *data_row  = base + slot * ROW;
    uint8_t      *scale_row = base + (ROW << p.page_bits) + slot * SCALES;

    if constexpr (V41) {
        float v0 = bf16_round(data[0]), v1 = bf16_round(data[1]);
        if (warp == SS_ROPE_WARP) {
            const float re = v0, im = v1;
            v0             = bf16_round(re * freq[0] - im * freq[1]);
            v1             = bf16_round(re * freq[1] + im * freq[0]);
        }
        constexpr int TILE = 16; // a 32-value tile's lanes
        float         amax = fabsf(v0);
        amax               = fmaxf(amax, fabsf(v1));
        amax               = lanes_max<TILE>(amax);
        const uint32_t bits = __float_as_uint(amax);
        const int32_t  exponent =
            max(static_cast<int32_t>(bits >> 23) - 8 + static_cast<int32_t>((bits & 0x7FFFFFu) > 0x600000u), 114);
        const float inv_scale = inv_scale_ue8m0(exponent);
        reinterpret_cast<uint16_t *>(data_row)[tx] = pack_e4m3(v0 * inv_scale, v1 * inv_scale);
        if (tx % TILE == 0) scale_row[tx / TILE] = static_cast<uint8_t>(exponent);
    } else if (warp == SS_ROPE_WARP) {
        const float re = data[0], im = data[1];
        const float v0 = re * freq[0] - im * freq[1];
        const float v1 = re * freq[1] + im * freq[0];
        bfloat16   *rope = reinterpret_cast<bfloat16 *>(data_row + 448);
        rope[2 * lane]     = static_cast<bfloat16>(v0);
        rope[2 * lane + 1] = static_cast<bfloat16>(v1);
    } else {
        const float    abs_max   = lanes_max<SS_LANES>(fmaxf(fabsf(data[0]), fabsf(data[1])));
        const float    scale_raw = fmaxf(1e-4f, abs_max) / 448.0f;
        const uint32_t u         = __float_as_uint(scale_raw);
        const int32_t  exponent  = static_cast<int32_t>((u >> 23) & 0xFF) + ((u & 0x7FFFFF) != 0);
        const float    inv_scale = inv_scale_ue8m0(exponent);
        reinterpret_cast<uint16_t *>(data_row)[tx] = pack_e4m3(data[0] * inv_scale, data[1] * inv_scale);
        if (lane == 0) scale_row[warp] = static_cast<uint8_t>(exponent);
    }
}

} // namespace

void swa_store(const SwaStoreParams &p, hipStream_t stream) {
    if (p.rows == 0) return;
    const auto kernel = p.v41 ? swa_store_kernel<true> : swa_store_kernel<false>;
    hipLaunchKernelGGL(kernel, dim3(p.rows, p.stages), dim3(SS_THREADS), 0, stream, p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace infera_flash::dsv41
