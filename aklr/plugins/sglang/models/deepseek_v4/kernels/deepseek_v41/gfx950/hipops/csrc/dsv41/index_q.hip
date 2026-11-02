// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// The indexer's query for a decode's rows in one launch (Infera-Flash's index_q, optionally with the compressor's
// gate GEMV), writing what sglang's index_q_rope_pack_weights_flydsl writes: q_fp4 [rows, heads, 64] e2m1 pairs,
// q_scale [rows, 1, 4, 16, 4] ue8m0 (chunk c of head h at [row, 0, c, h % 16, h / 16]) and weights [rows, heads].
#include <algorithm>

#include <c10/hip/HIPException.h>

#include "dsv41/index_q.h"
#include "dsv41/quant.h"

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK // the rocm10 image's torch names it C10_CUDA_
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace infera_flash::dsv41 {

__device__ inline float bf16_round(float v) { return static_cast<float>(static_cast<bfloat16>(v)); }

constexpr int IQ_GROUPS      = IndexQPackParams::QRANK / 32; // 40
constexpr int IQ_THREADS     = 8 * IQ_GROUPS;                // a thread 4 columns of one group
constexpr int IQ_LOGIT_HEADS = 2;                            // a logits block's heads
constexpr int IQ_GATE_CHUNKS = 10;                           // the gate GEMV's dim: 10 chunks of 512
constexpr int IQ_GATE_WAVES  = IQ_THREADS / THREADS_PER_WARP;

// gate output e: gate_w row e against the ROWS rows of x, a lane 8 values of each 512 chunk (gate_gemv_wave)
template <int ROWS> __device__ inline void gate_gemv_wave(const IndexQPackParams &p, int e, int lane) {
    bfloat16 w[IQ_GATE_CHUNKS][8];
#pragma unroll
    for (int j = 0; j < IQ_GATE_CHUNKS; ++j) {
        load_data<bfloat16, 8>(p.gate_w + static_cast<int64_t>(e) * p.dim + (j * THREADS_PER_WARP + lane) * 8, w[j]);
    }
#if __clang_major__ >= 23
    // ROCm 10's clang hoists every row's x of the unrolled rows below and spills; a row at a time, each row's sum
    // as below
#pragma unroll 1
    for (int r = 0; r < ROWS; ++r) {
        bfloat16 x[IQ_GATE_CHUNKS][8];
#pragma unroll
        for (int j = 0; j < IQ_GATE_CHUNKS; ++j) {
            load_data<bfloat16, 8>(p.x + static_cast<int64_t>(r) * p.dim + (j * THREADS_PER_WARP + lane) * 8, x[j]);
        }
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < IQ_GATE_CHUNKS; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a += static_cast<float>(x[j][i]) * static_cast<float>(w[j][i]);
            }
        }
        a = WarpReduce<SumOp>(a);
        if (lane == 0) {
            p.gate_out[r * p.gate_n + e] = a;
        }
    }
#else
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        bfloat16 x[IQ_GATE_CHUNKS][8];
#pragma unroll
        for (int j = 0; j < IQ_GATE_CHUNKS; ++j) {
            load_data<bfloat16, 8>(p.x + static_cast<int64_t>(r) * p.dim + (j * THREADS_PER_WARP + lane) * 8, x[j]);
        }
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < IQ_GATE_CHUNKS; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a += static_cast<float>(x[j][i]) * static_cast<float>(w[j][i]);
            }
        }
        acc[r] = a;
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = WarpReduce<SumOp>(acc[r]);
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            p.gate_out[r * p.gate_n + e] = acc[r];
        }
    }
#endif
}

__device__ inline void gate_gemv_wave_rows(const IndexQPackParams &p, int e, int lane) {
    switch (p.rows) {
#define CASE(N)                                                                                                        \
    case N:                                                                                                            \
        gate_gemv_wave<N>(p, e, lane);                                                                                 \
        break;
        CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    default:
        break;
    }
}

__global__ __launch_bounds__(IQ_THREADS) void index_q_pack_kernel(const IndexQPackParams group) {
    using Q = IndexQPackParams;
    IndexQPackParams p = group; // rows group.rows * blockIdx.y ..
    if (blockIdx.y > 0) {
        const int r0 = group.rows * blockIdx.y;
        p.rows = min(group.rows, group.total - r0);
        p.qr += static_cast<int64_t>(r0) * Q::QRANK, p.x += static_cast<int64_t>(r0) * p.dim, p.pos += r0;
        p.q_fp4 += static_cast<int64_t>(r0) * p.heads * (Q::DIM / 2), p.q_scale += static_cast<int64_t>(r0) * 256;
        p.weights += static_cast<int64_t>(r0) * p.heads;
        p.gate_out += p.gate_w != nullptr ? static_cast<int64_t>(r0) * p.gate_n : 0;
    }
    constexpr int XS = 36, PS = IQ_GROUPS + 1, XQ = Q::MAX_ROWS * IQ_GROUPS * XS, PART = Q::MAX_ROWS * 32 * PS;
    __shared__ __align__(16) float buf[XQ > PART ? XQ : PART];
    __shared__ float y[Q::MAX_ROWS][32];
    __shared__ float red[(IQ_THREADS + THREADS_PER_WARP - 1) / THREADS_PER_WARP][IQ_LOGIT_HEADS * Q::MAX_ROWS];
    const int        tid = threadIdx.x, m = p.rows, bid = blockIdx.x;
    const int        gate0 = p.heads * 4 + p.heads / IQ_LOGIT_HEADS;
    if (bid >= gate0) { // the compressor's GEMV: a wave an output
        const int e = (bid - gate0) * IQ_GATE_WAVES + tid / THREADS_PER_WARP;
        if (e < p.gate_n) {
            gate_gemv_wave_rows(p, e, tid % THREADS_PER_WARP);
        }
        return;
    }
    if (bid >= p.heads * 4) { // logits: x . weights_proj[h] of IQ_LOGIT_HEADS heads
        const int h0 = (bid - p.heads * 4) * IQ_LOGIT_HEADS;
        float     acc[IQ_LOGIT_HEADS][Q::MAX_ROWS] = {};
        for (int c = tid * 8; c < p.dim; c += IQ_THREADS * 8) {
            float w[IQ_LOGIT_HEADS][8];
#pragma unroll
            for (int hh = 0; hh < IQ_LOGIT_HEADS; ++hh) {
                bfloat16 b[8];
                *reinterpret_cast<uint4 *>(b) =
                    *reinterpret_cast<const uint4 *>(p.wp + static_cast<int64_t>(h0 + hh) * p.dim + c);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    w[hh][i] = static_cast<float>(b[i]);
                }
            }
#pragma unroll
            for (int r = 0; r < Q::MAX_ROWS; ++r) {
                if (r < m) {
                    bfloat16 b[8];
                    *reinterpret_cast<uint4 *>(b) =
                        *reinterpret_cast<const uint4 *>(p.x + static_cast<int64_t>(r) * p.dim + c);
#pragma unroll
                    for (int hh = 0; hh < IQ_LOGIT_HEADS; ++hh) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[hh][r] += w[hh][i] * static_cast<float>(b[i]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int hh = 0; hh < IQ_LOGIT_HEADS; ++hh) {
#pragma unroll
            for (int r = 0; r < Q::MAX_ROWS; ++r) {
                const float v = WarpReduce<SumOp>(acc[hh][r]);
                if (tid % THREADS_PER_WARP == 0) {
                    red[tid / THREADS_PER_WARP][hh * Q::MAX_ROWS + r] = v;
                }
            }
        }
        __syncthreads();
        if (tid < IQ_LOGIT_HEADS * m) {
            const int hh = tid / m, r = tid % m;
            float     v  = 0.0f;
            for (int wv = 0; wv < IQ_THREADS / THREADS_PER_WARP; ++wv) {
                v += red[wv][hh * Q::MAX_ROWS + r];
            }
            p.weights[r * p.heads + h0 + hh] = static_cast<bfloat16>(bf16_round(v) * p.weight_scale);
        }
        return;
    }
    const int h = bid / 4, j = bid % 4; // columns j * 32 .. + 31 of head h
    // the RoPE's (cos, sin) of thread (row, column) < m * 32, in flight from the start
    const int rope0 = Q::DIM - 2 * p.pairs;
    float     cs0 = 0.0f, cs1 = 0.0f;
    if (tid < m * 32 && j * 32 + tid % 32 >= rope0) {
        int64_t pos = p.pos[tid / 32];
        pos         = pos >= 0 && pos < p.positions ? pos : 0; // as sglang's: past the table, entry 0
        const float *cs = p.table + (pos * p.pairs + (j * 32 + tid % 32 - rope0) / 2) * 2;
        cs0 = cs[0], cs1 = cs[1];
    }
    // act_quant(qr): a thread a (row, group): 1e-4 floor, power-of-two scale, e4m3
    if (tid < m * IQ_GROUPS) {
        const int r = tid / IQ_GROUPS, g = tid % IQ_GROUPS;
        bfloat16  b[32];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<uint4 *>(b + 8 * i) =
                *reinterpret_cast<const uint4 *>(p.qr + static_cast<int64_t>(r) * Q::QRANK + g * 32 + 8 * i);
        }
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, fabsf(static_cast<float>(b[i])));
        }
        const float s = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            buf[(r * IQ_GROUPS + g) * XS + i] =
                fp8_value(fp8_byte(fminf(fmaxf(static_cast<float>(b[i]) / s, -448.0f), 448.0f))) * s;
        }
    }
    // a thread columns cq * 4 .. + 3 over group g: the weights' loads before the barrier
    const int     cq = tid / IQ_GROUPS, g = tid % IQ_GROUPS;
    const int64_t col0 = static_cast<int64_t>(h) * Q::DIM + j * 32 + cq * 4;
    const float   sw   = __builtin_bit_cast(float, static_cast<uint32_t>(p.ws[(col0 / 32) * IQ_GROUPS + g]) << 23);
    uint8_t       wb[4][32];
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
        const uint8_t *src = p.w + (col0 + cc) * Q::QRANK + g * 32;
        *reinterpret_cast<uint4 *>(wb[cc])      = *reinterpret_cast<const uint4 *>(src);
        *reinterpret_cast<uint4 *>(wb[cc] + 16) = *reinterpret_cast<const uint4 *>(src + 16);
    }
    __syncthreads();
    float wv[4][32]; // the 4 columns' group, dequantized (exact)
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            wv[cc][i] = fp8_value(wb[cc][i]) * sw;
        }
    }
    float acc[4][Q::MAX_ROWS] = {};
#pragma unroll
    for (int r = 0; r < Q::MAX_ROWS; ++r) {
        if (r < m) { // the row's whole group in one batch of loads, then its sums in k order
            float4 xv[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                xv[u] = *reinterpret_cast<const float4 *>(&buf[(r * IQ_GROUPS + g) * XS + 4 * u]);
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    acc[cc][r] += wv[cc][4 * u] * xv[u].x;
                    acc[cc][r] += wv[cc][4 * u + 1] * xv[u].y;
                    acc[cc][r] += wv[cc][4 * u + 2] * xv[u].z;
                    acc[cc][r] += wv[cc][4 * u + 3] * xv[u].w;
                }
            }
        }
    }
    __syncthreads(); // every read of the quantized qr done: the partial sums take its words
#pragma unroll
    for (int r = 0; r < Q::MAX_ROWS; ++r) {
        if (r < m) {
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                buf[(r * 32 + cq * 4 + cc) * PS + g] = acc[cc][r];
            }
        }
    }
    __syncthreads();
    if (tid < m * 32) { // the groups' sums in group order -> the GEMM's bf16 output
        const int r = tid / 32, c = tid % 32;
        float     part[IQ_GROUPS];
#pragma unroll
        for (int gg = 0; gg < IQ_GROUPS; ++gg) { // every load issued, then the sum in group order
            part[gg] = buf[(r * 32 + c) * PS + gg];
        }
        float v = 0.0f;
#pragma unroll
        for (int gg = 0; gg < IQ_GROUPS; ++gg) {
            v += part[gg];
        }
        y[r][c] = bf16_round(v);
    }
    __syncthreads();
    // RoPE and the fp4 grid, then sglang's quantization of those values: half a wave a row (whole rows a wave, so the
    // 32-lane shuffles stay in the row)
    if (tid < Q::MAX_ROWS * 32 && tid / 32 < m + (m & 1)) {
        const int  r = tid / 32, c = tid % 32, d = j * 32 + c;
        const bool live = r < m;
        float      v    = live ? y[r][c] : 0.0f;
        if (live && d >= rope0) {
            const float x0 = y[r][c & ~1], x1 = y[r][c | 1];
            v = (c & 1) == 0 ? bf16_round(x0 * cs0 - x1 * cs1) : bf16_round(x0 * cs1 + x1 * cs0);
        }
        float amax = fabsf(v);
#pragma unroll
        for (int off = 1; off < 32; off *= 2) {
            amax = fmaxf(amax, __shfl_xor(amax, off));
        }
        const float s  = pow2_ceil(fmaxf(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
        const float fq = bf16_round(fp4_round(fminf(fmaxf(v / s, -6.0f), 6.0f)) * s);
        float       a2 = fabsf(fq);
#pragma unroll
        for (int off = 1; off < 32; off *= 2) {
            a2 = fmaxf(a2, __shfl_xor(a2, off));
        }
        // quantize_fp4_indexer_row: exponent ceil(log2(max(amax / 6, 1e-4))) in [1, 254] biased
        const uint32_t bits = __builtin_bit_cast(uint32_t, fmaxf(a2 / 6.0f, 1.0e-4f));
        const uint32_t e    = min(max((bits >> 23 & 0xffu) + ((bits & 0x7fffffu) != 0u), 1u), 254u);
        const float    s2   = __builtin_bit_cast(float, e << 23);
        uint32_t       code = e2m1_code(fp4_round(fminf(fmaxf(fq / s2, -6.0f), 6.0f)));
        code                = (code & 7u) == 0u ? 0u : code; // no negative zero
        const uint32_t hi   = __shfl_xor(code, 1);
        if (live) {
            if ((c & 1) == 0) {
                p.q_fp4[(static_cast<int64_t>(r) * p.heads + h) * (Q::DIM / 2) + d / 2] =
                    static_cast<uint8_t>(code | hi << 4);
            }
            uint8_t *sc = p.q_scale + static_cast<int64_t>(r) * 256 + j * 64 + (h % 16) * 4;
            if (c == 0) {
                sc[h / 16] = static_cast<uint8_t>(e);
            } else if (h < 16 && c < 4 && c >= p.heads / 16) { // the head groups this head count lacks: zero
                sc[c] = 0;
            }
        }
    }
}

void index_q_pack(const IndexQPackParams &p, hipStream_t stream) {
    if (p.total == 0) {
        return;
    }
    TORCH_CHECK(p.rows <= IndexQPackParams::MAX_ROWS && p.heads % 16 == 0 && p.heads <= 64 && p.dim % 8 == 0,
                "index_q_pack: rows <= 8, heads % 16, heads <= 64, dim % 8");
    TORCH_CHECK(p.gate_w == nullptr || p.dim == IQ_GATE_CHUNKS * THREADS_PER_WARP * 8, "index_q_pack's gate: dim 5120");
    const int      gate_blocks = p.gate_w == nullptr ? 0 : (p.gate_n + IQ_GATE_WAVES - 1) / IQ_GATE_WAVES;
    const unsigned blocks      = static_cast<unsigned>(p.heads * 4 + p.heads / IQ_LOGIT_HEADS + gate_blocks);
    const unsigned groups = p.total > p.rows ? static_cast<unsigned>((p.total + p.rows - 1) / p.rows) : 1;
    index_q_pack_kernel<<<dim3(blocks, groups), IQ_THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace infera_flash::dsv41
