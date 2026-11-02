// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <climits>

#include <c10/hip/HIPException.h>
#ifndef C10_HIP_KERNEL_LAUNCH_CHECK // aklr: the rocm10 image's torch names it C10_CUDA_
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

#include "dsv41/markov.h"

namespace infera_flash::dsv41 {

using P                     = MarkovParams;
constexpr int THREADS       = 256;
constexpr int ROW_LANES     = 16; // a vocab row's DIM by 16 lanes, 16 values each
constexpr int ROWS_PER_PASS = THREADS / ROW_LANES;
constexpr int PASSES        = 4;
constexpr int BLOCK_ROWS    = ROWS_PER_PASS * PASSES;
constexpr int FINISH        = 1024;
constexpr int SAMPLE_SEQS   = THREADS / BLOCK_ROWS; // AKLR: a wave a sampled sequence

// (v, i) <- the better of it and (v2, i2): the larger value, the lower index on ties (as a full argmax)
__device__ inline void keep_best(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2, i = i2;
    }
}

// the token of an all-reduced [tp, 2]: the best rank's index (the lowest rank on equal bests)
__device__ inline int64_t resolve(const float *packed, int tp) {
    int best = 0;
    for (int r = 1; r < tp; ++r) {
        if (packed[2 * r] > packed[2 * best]) {
            best = r;
        }
    }
    return static_cast<int64_t>(packed[2 * best + 1]);
}

__device__ inline void load16f(const bfloat16 *src, float (&v)[16]) {
    bfloat16 b[16];
    *reinterpret_cast<uint4 *>(b)     = *reinterpret_cast<const uint4 *>(src);
    *reinterpret_cast<uint4 *>(b + 8) = *reinterpret_cast<const uint4 *>(src + 8);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = static_cast<float>(b[i]);
    }
}

__device__ inline float logit(const P &p, int64_t off, int v) {
    return p.logits_bf16 != nullptr ? static_cast<float>(p.logits_bf16[off + v]) : p.logits[off + v];
}

// AKLR: Philox4x32-10 (tl.philox's), the word of the counter's four that index k picks
__device__ inline uint32_t philox_word(uint32_t k0, uint32_t k1, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                       int k) {
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint32_t a0 = c0, a2 = c2;
        c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * a2;
        c3 = 0xD2511F53u * a0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return k == 0 ? c0 : k == 1 ? c1 : k == 2 ? c2 : c3;
}

__device__ inline float round_bf16(float x) { return static_cast<float>(static_cast<bfloat16>(x)); }

// AKLR: a sampled step's block best: lane t of the first wave scores row t of the block (its dot in sdot), the wave
// keeps the best (the lower index on ties); written to part by lane 0
__device__ inline void sampled_part(const P &p, int seq, int64_t off, const float *sdot, float *part);

// a vocab row's score: the greedy logit + bias, or a sampled step's (MarkovParams)
__device__ inline float score(const P &p, int seq, int v, float lg, float dot) {
    if (!p.sample) {
        return lg + dot;
    }
    const float val = p.round_bf16 ? round_bf16(lg + round_bf16(dot)) : lg + dot;
    const float t   = p.temps[seq];
    if (t == 0.0f) {
        return val;
    }
    const uint32_t g = static_cast<uint32_t>(p.vocab_start + v);
    const uint32_t w = philox_word(p.key_lo, p.key_hi, g >> 2, static_cast<uint32_t>(seq), static_cast<uint32_t>(p.step),
                                   static_cast<uint32_t>(p.offset[0]), static_cast<int>(g & 3u));
    const float u = (static_cast<float>(w) + 0.5f) * 2.3283064365386963e-10f;
    return val / t - logf(-log1pf(-u));
}

__device__ inline void sampled_part(const P &p, int seq, int64_t off, const float *sdot, float *part) {
    if (threadIdx.x >= BLOCK_ROWS) {
        return;
    }
    const int v    = blockIdx.x * BLOCK_ROWS + threadIdx.x;
    float     best = -INFINITY;
    int       bi   = INT_MAX;
    if (v < p.vocab) {
        best = score(p, seq, v, logit(p, off, v), sdot[threadIdx.x]);
        bi   = v;
    }
#pragma unroll
    for (int m = BLOCK_ROWS / 2; m > 0; m /= 2) {
        keep_best(best, bi, __shfl_xor(best, m, BLOCK_ROWS), __shfl_xor(bi, m, BLOCK_ROWS));
    }
    if (threadIdx.x == 0) {
        part[0] = best;
        part[1] = static_cast<float>(bi); // exact: vocab < 2^24
    }
}

// one sequence: the draft's hot path
__global__ __launch_bounds__(THREADS) void markov_step_one_kernel(const P p) {
    __shared__ float sv[ROWS_PER_PASS];
    __shared__ int   si[ROWS_PER_PASS];
    __shared__ float sdot[BLOCK_ROWS];
    const int64_t    prev = p.step == 0 ? p.anchor[0] : p.tok != nullptr ? p.tok[0] : resolve(p.prev, p.tp);
    if (p.step > 0 && p.tok == nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
        p.out[p.step - 1] = prev;
    }
    const int lane = threadIdx.x % ROW_LANES, grp = threadIdx.x / ROW_LANES;
    float     e[16], h[PASSES][16];
    load16f(p.embed + prev * P::DIM + lane * 16, e);
#pragma unroll
    for (int ps = 0; ps < PASSES; ++ps) {
        const int v = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
        if (v < p.vocab) {
            load16f(p.head + static_cast<int64_t>(v) * P::DIM + lane * 16, h[ps]);
        }
    }
    float best = -INFINITY;
    int   bi   = INT_MAX;
#pragma unroll
    for (int ps = 0; ps < PASSES; ++ps) {
        const int v   = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
        float     dot = 0.0f;
        if (v < p.vocab) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                dot += e[i] * h[ps][i];
            }
        }
        dot = WarpReduce<SumOp, float, ROW_LANES>(dot); // a whole 16-lane row: v is the row's
        if (p.sample) {
            if (lane == 0) {
                sdot[ps * ROWS_PER_PASS + grp] = dot;
            }
        } else if (v < p.vocab) {
            keep_best(best, bi, logit(p, 0, v) + dot, v);
        }
    }
    if (p.sample) {
        __syncthreads();
        sampled_part(p, 0, 0, sdot, p.part + 2 * blockIdx.x);
        return;
    }
    if (lane == 0) {
        sv[grp] = best, si[grp] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int g = 1; g < ROWS_PER_PASS; ++g) {
            keep_best(best, bi, sv[g], si[g]);
        }
        p.part[2 * blockIdx.x]     = best;
        p.part[2 * blockIdx.x + 1] = static_cast<float>(bi); // exact: vocab < 2^24
    }
}

__global__ __launch_bounds__(FINISH) void markov_finish_one_kernel(const P p) {
    __shared__ float sv[FINISH];
    __shared__ int   si[FINISH];
    float            best = -INFINITY;
    int              bi   = INT_MAX;
    for (int b = threadIdx.x; b < p.blocks; b += FINISH) {
        keep_best(best, bi, p.part[2 * b], static_cast<int>(p.part[2 * b + 1]));
    }
    sv[threadIdx.x] = best, si[threadIdx.x] = bi;
    __syncthreads();
    for (int half = FINISH / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            keep_best(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + half], si[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (p.tok != nullptr) {
        if (threadIdx.x == 0) {
            uint32_t call = static_cast<uint32_t>(p.calls[0]) + 1;
            call          = call ? call : 1;
            const auto slot = [&](int src) { return static_cast<int>(call % 2) * 8 + src; };
            const uint32x4 mine{__builtin_bit_cast(uint32_t, sv[0]), static_cast<uint32_t>(si[0] + p.rank * p.vocab), 0,
                                call};
            for (int r = 0; r < p.tp; ++r) {
                store_nt(static_cast<uint32x4 *>(p.recv[r]) + slot(p.rank), mine);
            }
            float   best = 0.0f;
            int64_t tok  = 0;
            for (int r = 0; r < p.tp; ++r) { // the best rank's, the lowest on equal bests (as resolve)
                const uint32x4 *src = static_cast<const uint32x4 *>(p.recv[p.rank]) + slot(r);
                uint32x4        packet;
                packet = poll_tag(src, call);
                const float v = __builtin_bit_cast(float, packet.x);
                if (r == 0 || v > best) {
                    best = v, tok = static_cast<int64_t>(packet.y);
                }
            }
            p.out[p.step] = tok;
            p.tok[0]      = tok;
            p.calls[0]    = static_cast<int>(call);
            if (p.bump_offset) {
                const_cast<int64_t *>(p.offset)[0] += 1;
            }
        }
        return;
    }
    if (threadIdx.x < 2 * p.tp) {
        const int r = threadIdx.x / 2;
        p.packed[threadIdx.x] =
            r != p.rank ? 0.0f : threadIdx.x % 2 == 0 ? sv[0] : static_cast<float>(si[0] + p.rank * p.vocab);
    }
}

// a batch: a block BLOCK_ROWS vocab rows of the head, held in registers, against every sequence's step in turn
__global__ __launch_bounds__(THREADS) void markov_step_kernel(const P p) {
    __shared__ float sv[ROWS_PER_PASS];
    __shared__ int   si[ROWS_PER_PASS];
    __shared__ float sdot[SAMPLE_SEQS][BLOCK_ROWS];
    const int        lane = threadIdx.x % ROW_LANES, grp = threadIdx.x / ROW_LANES;
    const auto       prev_of = [&](int seq) {
        return p.step == 0 ? p.anchor[seq] : p.tok != nullptr ? p.tok[seq] : resolve(p.prev, p.tp);
    };
    float e[16], h[PASSES][16];
    load16f(p.embed + prev_of(0) * P::DIM + lane * 16, e); // the first sequence's embedding row ahead of the head's
#pragma unroll
    for (int ps = 0; ps < PASSES; ++ps) {
        const int v = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
        if (v < p.vocab) {
            load16f(p.head + static_cast<int64_t>(v) * P::DIM + lane * 16, h[ps]);
        }
    }
    if (p.sample && p.tok != nullptr) { // AKLR: the sequences' scores in parallel, a wave a sequence
        for (int s0 = 0; s0 < p.nseq; s0 += SAMPLE_SEQS) {
            const int n = min(SAMPLE_SEQS, p.nseq - s0);
            for (int k = 0; k < n; ++k) {
                if (s0 + k > 0) {
                    load16f(p.embed + prev_of(s0 + k) * P::DIM + lane * 16, e);
                }
#pragma unroll
                for (int ps = 0; ps < PASSES; ++ps) {
                    const int v   = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
                    float     dot = 0.0f;
                    if (v < p.vocab) {
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            dot += e[i] * h[ps][i];
                        }
                    }
                    dot = WarpReduce<SumOp, float, ROW_LANES>(dot);
                    if (lane == 0) {
                        sdot[k][ps * ROWS_PER_PASS + grp] = dot;
                    }
                }
            }
            __syncthreads();
            const int k = threadIdx.x / BLOCK_ROWS, row = threadIdx.x % BLOCK_ROWS;
            if (k < n) {
                const int seq  = s0 + k;
                const int v    = blockIdx.x * BLOCK_ROWS + row;
                float     best = -INFINITY;
                int       bi   = INT_MAX;
                if (v < p.vocab) {
                    best = score(p, seq, v, logit(p, seq * p.logits_stride, v), sdot[k][row]);
                    bi   = v;
                }
#pragma unroll
                for (int m = BLOCK_ROWS / 2; m > 0; m /= 2) {
                    keep_best(best, bi, __shfl_xor(best, m, BLOCK_ROWS), __shfl_xor(bi, m, BLOCK_ROWS));
                }
                if (row == 0) {
                    float *part = p.part + 2 * (static_cast<int64_t>(seq) * p.blocks + blockIdx.x);
                    part[0]     = best;
                    part[1]     = static_cast<float>(bi); // exact: vocab < 2^24
                }
            }
            __syncthreads(); // sdot free for the next sequences
        }
        return;
    }
    for (int seq = 0; seq < p.nseq; ++seq) {
        const int64_t off  = seq * p.logits_stride;
        const int64_t prev = prev_of(seq);
        if (p.step > 0 && p.tok == nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
            p.out[p.step - 1] = prev;
        }
        if (seq > 0) {
            load16f(p.embed + prev * P::DIM + lane * 16, e);
        }
        float best = -INFINITY;
        int   bi   = INT_MAX;
#pragma unroll
        for (int ps = 0; ps < PASSES; ++ps) {
            const int v   = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
            float     dot = 0.0f;
            if (v < p.vocab) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    dot += e[i] * h[ps][i];
                }
            }
            dot = WarpReduce<SumOp, float, ROW_LANES>(dot); // a whole 16-lane row: v is the row's
            if (p.sample) {
                if (lane == 0) {
                    sdot[0][ps * ROWS_PER_PASS + grp] = dot;
                }
            } else if (v < p.vocab) {
                keep_best(best, bi, logit(p, off, v) + dot, v);
            }
        }
        if (p.sample) {
            __syncthreads();
            sampled_part(p, seq, off, sdot[0], p.part + 2 * (static_cast<int64_t>(seq) * p.blocks + blockIdx.x));
        } else if (lane == 0) {
            sv[grp] = best, si[grp] = bi;
        }
        if (!p.sample) {
            __syncthreads();
        }
        if (!p.sample && threadIdx.x == 0) {
            for (int g = 1; g < ROWS_PER_PASS; ++g) {
                keep_best(best, bi, sv[g], si[g]);
            }
            float *part = p.part + 2 * (static_cast<int64_t>(seq) * p.blocks + blockIdx.x);
            part[0]     = best;
            part[1]     = static_cast<float>(bi); // exact: vocab < 2^24
        }
        if (seq + 1 < p.nseq) {
            __syncthreads(); // sv / si free for the next sequence
        }
    }
}

// a block a sequence
__global__ __launch_bounds__(FINISH) void markov_finish_kernel(const P p) {
    __shared__ float sv[FINISH];
    __shared__ int   si[FINISH];
    const int        seq  = blockIdx.x;
    const float     *part = p.part + 2 * static_cast<int64_t>(seq) * p.blocks;
    float            best = -INFINITY;
    int              bi   = INT_MAX;
    for (int b = threadIdx.x; b < p.blocks; b += FINISH) {
        keep_best(best, bi, part[2 * b], static_cast<int>(part[2 * b + 1]));
    }
    sv[threadIdx.x] = best, si[threadIdx.x] = bi;
    __syncthreads();
    for (int half = FINISH / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            keep_best(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + half], si[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (p.tok != nullptr) {
        if (threadIdx.x == 0) {
            uint32_t call = static_cast<uint32_t>(p.calls[seq]) + 1;
            call          = call ? call : 1;
            const auto slot = [&](int src) { return p.slot0 + seq * 16 + static_cast<int>(call % 2) * 8 + src; };
            const uint32x4 mine{__builtin_bit_cast(uint32_t, sv[0]), static_cast<uint32_t>(si[0] + p.rank * p.vocab), 0,
                                call};
            for (int r = 0; r < p.tp; ++r) {
                store_nt(static_cast<uint32x4 *>(p.recv[r]) + slot(p.rank), mine);
            }
            float   best = 0.0f;
            int64_t tok  = 0;
            for (int r = 0; r < p.tp; ++r) { // the best rank's, the lowest on equal bests (as resolve)
                const uint32x4 *src = static_cast<const uint32x4 *>(p.recv[p.rank]) + slot(r);
                uint32x4        packet;
                packet = poll_tag(src, call);
                const float v = __builtin_bit_cast(float, packet.x);
                if (r == 0 || v > best) {
                    best = v, tok = static_cast<int64_t>(packet.y);
                }
            }
            p.out[seq * p.out_stride + p.step] = tok;
            p.tok[seq]                         = tok;
            p.calls[seq]                       = static_cast<int>(call);
            if (p.bump_offset && seq == 0) {
                const_cast<int64_t *>(p.offset)[0] += 1;
            }
        }
        return;
    }
    if (threadIdx.x < 2 * p.tp) {
        const int r = threadIdx.x / 2;
        p.packed[threadIdx.x] =
            r != p.rank ? 0.0f : threadIdx.x % 2 == 0 ? sv[0] : static_cast<float>(si[0] + p.rank * p.vocab);
    }
}

__global__ void markov_resolve_kernel(const P p) { p.out[p.step] = resolve(p.prev, p.tp); }

void markov_step(const P &p, hipStream_t stream) {
    TORCH_CHECK(p.nseq >= 1 && p.nseq <= MARKOV_BATCH_MAX && (p.nseq == 1 || p.tok != nullptr),
                "markov_step: 1 .. 32 sequences, a batch exchanging in the kernel");
    if (p.nseq > 1) {
        markov_step_kernel<<<p.blocks, THREADS, 0, stream>>>(p);
        C10_HIP_KERNEL_LAUNCH_CHECK();
        markov_finish_kernel<<<p.nseq, FINISH, 0, stream>>>(p);
    } else {
        markov_step_one_kernel<<<p.blocks, THREADS, 0, stream>>>(p);
        C10_HIP_KERNEL_LAUNCH_CHECK();
        markov_finish_one_kernel<<<1, FINISH, 0, stream>>>(p);
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

constexpr int PLAN_THREADS = 256;

// one block: wave w the rows w, w + 4, ..; then thread 0 picks k and every thread writes the rows
__global__ __launch_bounds__(PLAN_THREADS) void verify_plan_kernel(const VerifyPlanParams p) {
    __shared__ float conf[VerifyPlanParams::MAX_N];
    __shared__ int   s_k;
    const int        wave = threadIdx.x / THREADS_PER_WARP, lane = threadIdx.x % THREADS_PER_WARP;
    for (int i = wave; i < p.n; i += PLAN_THREADS / THREADS_PER_WARP) {
        float acc = 0.0f;
        for (int c = lane * 8; c < p.dim; c += THREADS_PER_WARP * 8) {
            bfloat16 x[8], w[8];
            *reinterpret_cast<uint4 *>(x) = *reinterpret_cast<const uint4 *>(p.col + static_cast<int64_t>(i) * p.dim + c);
            *reinterpret_cast<uint4 *>(w) = *reinterpret_cast<const uint4 *>(p.w + c);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc += static_cast<float>(x[u]) * static_cast<float>(w[u]);
            }
        }
        const bfloat16 *e = p.embed + p.tok[i] * p.mr;
        for (int c = lane; c < p.mr; c += THREADS_PER_WARP) {
            acc += static_cast<float>(e[c]) * static_cast<float>(p.w[p.dim + c]);
        }
        acc = WarpReduce<SumOp>(acc);
        if (lane == 0) {
            const float z = fminf(fmaxf(acc * p.inv_temp, -60.0f), 60.0f);
            conf[i]       = 1.0f / (1.0f + expf(-z));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int k = p.fixed_k >= 0 ? min(p.fixed_k, p.n) : p.n;
        if (p.fixed_k < 0) { // the best expected tokens a unit of time
            float run = 1.0f, expect = 1.0f, best = expect / p.cost[0];
            k         = 0;
            for (int i = 0; i < p.n; ++i) {
                run *= conf[i];
                expect += run;
                const float v = expect / p.cost[i + 1];
                if (v > best) {
                    best = v, k = i + 1;
                }
            }
        }
        s_k = k;
        p.k_out[0] = k;
    }
    __syncthreads();
    const int k = s_k;
    for (int j = threadIdx.x; j <= p.n; j += PLAN_THREADS) {
        const int src = min(j, k);
        p.vtok[j]     = p.tok[src];
        p.vpos[j]     = p.pos[src];
        if (j < p.n) {
            p.vdrafts[j] = j < k ? p.tok[j + 1] : -1;
            p.conf[j]    = conf[j];
        }
    }
}

void verify_plan(const VerifyPlanParams &p, hipStream_t stream) {
    TORCH_CHECK(p.n >= 1 && p.n <= VerifyPlanParams::MAX_N && p.dim % (THREADS_PER_WARP * 8) == 0,
                "verify_plan: 1 .. 8 drafts, dim % 512");
    verify_plan_kernel<<<1, PLAN_THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void markov_resolve(const P &p, hipStream_t stream) {
    markov_resolve_kernel<<<1, 1, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

int markov_blocks(int vocab) { return (vocab + BLOCK_ROWS - 1) / BLOCK_ROWS; }

// markov_chain (markov.h): a thread its row's head in registers, its step's dot product summed as markov_step's
__global__ __launch_bounds__(MarkovChainParams::THREADS) void markov_chain_kernel(const MarkovChainParams p) {
    using C = MarkovChainParams;
    __shared__ float e[C::DIM];
    __shared__ float sv[C::THREADS / THREADS_PER_WARP];
    __shared__ int   si[C::THREADS / THREADS_PER_WARP];
    const int        tid = threadIdx.x, b = blockIdx.x, lane = tid % THREADS_PER_WARP, wave = tid / THREADS_PER_WARP;
    const int        v   = b * p.per + tid; // this thread's vocab row
    const bool       own = tid < p.per && v < p.vocab;
    float            h[C::DIM];
    if (own) {
#pragma unroll
        for (int c = 0; c < C::DIM; c += 16) {
            float t[16];
            load16f(p.head + static_cast<int64_t>(v) * C::DIM + c, t);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                h[c + i] = t[i];
            }
        }
    }
    uint32x4 *mine  = static_cast<uint32x4 *>(p.recv[p.rank]) + C::SLOT0;
    uint32x4 *cross = mine + 2 * C::CHAIN_BLOCKS;
    const int base  = p.calls[0] * p.steps; // the sequence before this launch
    int64_t   prev  = p.anchor[0];
    for (int s = 0; s < p.steps; ++s) {
        const uint32_t tag = static_cast<uint32_t>(base + s + 1), par = tag % 2;
        for (int k = tid; k < C::DIM; k += C::THREADS) {
            e[k] = static_cast<float>(p.embed[prev * C::DIM + k]);
        }
        __syncthreads();
        float best = -INFINITY;
        int   bi   = INT_MAX;
        if (own) { // markov_step's sums: 16 runs of 16 products, then the 16-lane butterfly's tree
            float r[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    dot += e[c * 16 + i] * h[c * 16 + i];
                }
                r[c] = dot;
            }
#pragma unroll
            for (int w = 1; w < 16; w *= 2) {
#pragma unroll
                for (int c = 0; c < 16; c += 2 * w) {
                    r[c] = r[c] + r[c + w];
                }
            }
            best = p.logits[s * p.stride + v] + r[0], bi = v;
        }
        for (int off = 1; off < THREADS_PER_WARP; off *= 2) {
            keep_best(best, bi, __shfl_xor(best, off), __shfl_xor(bi, off));
        }
        if (lane == 0) {
            sv[wave] = best, si[wave] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < C::THREADS / THREADS_PER_WARP; ++w) {
                keep_best(best, bi, sv[w], si[w]);
            }
            store_nt(mine + par * C::CHAIN_BLOCKS + b, uint32x4{__builtin_bit_cast(uint32_t, best),
                                                                static_cast<uint32_t>(bi), 0, tag});
        }
        // every CTA's best -> this rank's (CTA order immaterial: keep_best's tie rule is the index's)
        float rb = -INFINITY;
        int   ri = INT_MAX;
        for (int k = tid; k < static_cast<int>(gridDim.x); k += C::THREADS) {
            uint32x4 pk;
            pk = poll_tag(mine + par * C::CHAIN_BLOCKS + k, tag);
            keep_best(rb, ri, __builtin_bit_cast(float, pk.x), static_cast<int>(pk.y));
        }
        for (int off = 1; off < THREADS_PER_WARP; off *= 2) {
            keep_best(rb, ri, __shfl_xor(rb, off), __shfl_xor(ri, off));
        }
        __syncthreads(); // sv / si reused
        if (lane == 0) {
            sv[wave] = rb, si[wave] = ri;
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 1; w < C::THREADS / THREADS_PER_WARP; ++w) {
                keep_best(rb, ri, sv[w], si[w]);
            }
            if (b == 0) { // this rank's (best, global index) to every rank
                const uint32x4 pk{__builtin_bit_cast(uint32_t, rb), static_cast<uint32_t>(ri + p.rank * p.vocab), 0,
                                  tag};
                for (int r = 0; r < p.tp; ++r) {
                    store_nt(static_cast<uint32x4 *>(p.recv[r]) + C::SLOT0 + 2 * C::CHAIN_BLOCKS + par * 8 + p.rank,
                             pk);
                }
            }
            float   bv  = 0.0f;
            int64_t tok = 0;
            for (int r = 0; r < p.tp; ++r) { // the best rank's, the lowest on equal bests (as resolve)
                uint32x4 pk;
                pk = poll_tag(cross + par * 8 + r, tag);
                const float val = __builtin_bit_cast(float, pk.x);
                if (r == 0 || val > bv) {
                    bv = val, tok = static_cast<int64_t>(pk.y);
                }
            }
            si[0] = static_cast<int>(tok);
            if (b == 0) {
                p.out[s] = tok;
            }
        }
        __syncthreads();
        prev = si[0];
        __syncthreads(); // si[0] read before the next step's writes
    }
    if (b == 0 && tid == 0) {
        p.calls[0] += 1;
    }
}

void markov_chain(const MarkovChainParams &p, hipStream_t stream) {
    using C = MarkovChainParams;
    TORCH_CHECK(p.per <= C::THREADS && p.tp <= 8 && p.steps >= 1, "markov_chain: vocab / tp <= 65536, tp <= 8");
    markov_chain_kernel<<<C::CHAIN_BLOCKS, C::THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

constexpr int ARGMAX_THREADS = 256, ARGMAX_PER = 8; // a thread 8 consecutive logits a pass

// grid (chunks, rows): a chunk's (best, index) into part [rows, chunks, 2]
__global__ __launch_bounds__(ARGMAX_THREADS) void argmax_part_kernel(const ArgmaxParams p) {
    __shared__ float sv[ARGMAX_THREADS];
    __shared__ int   si[ARGMAX_THREADS];
    const int        row = blockIdx.y, chunk = blockIdx.x, chunks = gridDim.x;
    const int        n    = p.valid > 0 ? p.valid : p.part;
    const float     *x    = p.logits + static_cast<int64_t>(row) * p.part;
    const bfloat16  *xb   = p.logits_bf16 + static_cast<int64_t>(row) * p.part;
    float            best = -INFINITY;
    int              bi   = INT_MAX;
    for (int v0 = (chunk * ARGMAX_THREADS + threadIdx.x) * ARGMAX_PER; v0 < n;
         v0 += chunks * ARGMAX_THREADS * ARGMAX_PER) {
        float v[ARGMAX_PER];
#pragma unroll
        for (int i = 0; i < ARGMAX_PER; ++i) {
            v[i] = v0 + i >= n ? -INFINITY : p.logits != nullptr ? x[v0 + i] : static_cast<float>(xb[v0 + i]);
        }
#pragma unroll
        for (int i = 0; i < ARGMAX_PER; ++i) {
            if (v0 + i < n) {
                keep_best(best, bi, v[i], v0 + i);
            }
        }
    }
    sv[threadIdx.x] = best, si[threadIdx.x] = bi;
    __syncthreads();
    for (int half = ARGMAX_THREADS / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            keep_best(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + half], si[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p.scratch[(row * chunks + chunk) * 2]     = sv[0];
        p.scratch[(row * chunks + chunk) * 2 + 1] = static_cast<float>(si[0]); // exact: vocab < 2^24
    }
}

// a thread a row: its chunks' best -> this rank's slot of packed (the others 0)
__global__ __launch_bounds__(64) void argmax_pack_kernel(const ArgmaxParams p) {
    const int row = threadIdx.x;
    if (row >= p.rows) {
        return;
    }
    float best = -INFINITY;
    int   bi   = INT_MAX;
    for (int c = 0; c < p.chunks; ++c) {
        keep_best(best, bi, p.scratch[(row * p.chunks + c) * 2], static_cast<int>(p.scratch[(row * p.chunks + c) * 2 + 1]));
    }
    for (int r = 0; r < p.tp; ++r) {
        p.packed[(r * p.rows + row) * 2]     = r != p.rank ? 0.0f : best;
        p.packed[(r * p.rows + row) * 2 + 1] = r != p.rank ? 0.0f : static_cast<float>(bi + p.rank * p.part);
    }
}

constexpr int XCHG_WAVES = 16;

// a wave a row (rows w, w + XCHG_WAVES, ..): its chunks' best (a lane a chunk) to every rank, every rank's back ->
// the token; then the acceptance
__global__ __launch_bounds__(XCHG_WAVES *THREADS_PER_WARP) void argmax_xchg_kernel(const ArgmaxParams p) {
    __shared__ int64_t tok[64];
    __shared__ uint32_t s_call;
    __shared__ int      s_sim, s_sim0; // this rank's simulated length, rank 0's
    const int           wave = threadIdx.x / THREADS_PER_WARP, lane = threadIdx.x % THREADS_PER_WARP;
    if (threadIdx.x == 0) {
        const uint32_t call = static_cast<uint32_t>(p.calls[0]) + 1;
        s_call              = call ? call : 1;
        s_sim               = p.sim != nullptr ? static_cast<int>(p.sim[0]) : 0;
    }
    __syncthreads();
    const uint32_t call    = s_call;
    const uint32_t own_sim = static_cast<uint32_t>(s_sim);
    for (int row = wave; row < p.rows; row += XCHG_WAVES) {
        float best = -INFINITY;
        int   bi   = INT_MAX;
        if (lane < p.chunks) {
            best = p.scratch[(row * p.chunks + lane) * 2];
            bi   = static_cast<int>(p.scratch[(row * p.chunks + lane) * 2 + 1]);
        }
#pragma unroll
        for (int off = THREADS_PER_WARP / 2; off > 0; off /= 2) { // the order-free best of the wave's
            keep_best(best, bi, __shfl_xor(best, off), __shfl_xor(bi, off));
        }
        if (lane == 0) {
            const auto slot = [&](int src) { return ARGMAX_SLOT0 + (static_cast<int>(call % 2) * 8 + src) * 64 + row; };
            const uint32x4 mine{__builtin_bit_cast(uint32_t, best), static_cast<uint32_t>(bi + p.rank * p.part),
                                own_sim, call};
            for (int r = 0; r < p.tp; ++r) {
                store_nt(static_cast<uint32x4 *>(p.recv[r]) + slot(p.rank), mine);
            }
            float   bv = 0.0f;
            int64_t t  = 0;
            for (int r = 0; r < p.tp; ++r) { // the best rank's, the lowest on equal bests
                const uint32x4 *src = static_cast<const uint32x4 *>(p.recv[p.rank]) + slot(r);
                uint32x4        packet;
                packet = poll_tag(src, call);
                const float v = __builtin_bit_cast(float, packet.x);
                if (r == 0 || v > bv) {
                    bv = v, t = static_cast<int64_t>(packet.y);
                }
                if (r == 0 && row == 0) {
                    s_sim0 = static_cast<int>(packet.z);
                }
            }
            tok[row]   = t;
            p.out[row] = t;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        p.calls[0] = static_cast<int>(call);
    }
    if (p.drafts != nullptr && static_cast<int>(threadIdx.x) < p.nseq) { // a thread a sequence's acceptance
        const int      s = threadIdx.x, n = p.rows / p.nseq;
        const int64_t *drafts = p.drafts + s * p.drafts_stride, *t = tok + s * n;
        int64_t       *res    = p.res + s * (n + 1);
        int            acc    = 0;
        while (acc < p.block && drafts[acc] == t[acc]) {
            ++acc;
        }
        res[0] = acc;
        for (int i = 0; i < n; ++i) {
            res[1 + i] = t[i];
        }
        p.anchor[s] = t[acc];
        p.last[s] += acc + 1;
    }
    if (p.cand != nullptr && static_cast<int>(threadIdx.x) < p.nseq) { // a thread a request's sglang accept
        const int      s = threadIdx.x, n = p.rows / p.nseq;
        const int64_t *cand = p.cand + s * n, *t = tok + s * n;
        int            acc  = 0;
        while (acc < n - 1 && cand[acc + 1] == t[acc]) {
            ++acc;
        }
        const int64_t bonus  = t[acc];
        const int64_t prefix = p.prefix_i32 ? static_cast<const int32_t *>(p.prefix)[s]
                                            : static_cast<const int64_t *>(p.prefix)[s];
        if (p.sim != nullptr) { // the simulated length in the greedy one's place, the bonus kept
            acc = p.sim_bcast ? s_sim0 : s_sim;
        }
        p.correct[s]  = acc;
        p.bonus[s]    = bonus;
        p.cap_trim[s] = 0;
        p.commit[s]   = acc + 1;
        p.new_lens[s] = prefix + acc + 1;
        if (p.gated != nullptr) {
            p.gated[s] = min(acc + 1, static_cast<int32_t>(p.verify_lens[s])) * p.gate[0];
        }
        for (int k = 0; k < n; ++k) {
            p.out_tokens[s * n + k] = k == acc ? bonus : k < n - 1 ? cand[k + 1] : 0;
        }
    }
}

void argmax_xchg(const ArgmaxParams &p, hipStream_t stream) {
    TORCH_CHECK(p.rows <= 64 && p.tp <= 8 && p.nseq >= 1 && p.rows % p.nseq == 0,
                "argmax_xchg: rows <= 64, tp <= 8, rows a multiple of the sequences");
    argmax_part_kernel<<<dim3(static_cast<unsigned>(p.chunks), static_cast<unsigned>(p.rows)), ARGMAX_THREADS, 0,
                         stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    argmax_xchg_kernel<<<1, std::min(XCHG_WAVES, p.rows) * THREADS_PER_WARP, 0, stream>>>(p); // >= 64 >= nseq threads
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

__global__ __launch_bounds__(64) void argmax_resolve_kernel(const ArgmaxParams p) {
    __shared__ int64_t tok[64];
    const int          row = threadIdx.x;
    if (row < p.rows) {
        int best = 0;
        for (int r = 1; r < p.tp; ++r) {
            if (p.packed[(r * p.rows + row) * 2] > p.packed[(best * p.rows + row) * 2]) {
                best = r;
            }
        }
        tok[row] = static_cast<int64_t>(p.packed[(best * p.rows + row) * 2 + 1]);
        p.out[row] = tok[row];
    }
    __syncthreads();
    if (p.drafts != nullptr && threadIdx.x == 0) {
        int acc = 0;
        while (acc < p.block && p.drafts[acc] == tok[acc]) {
            ++acc;
        }
        p.res[0] = acc;
        for (int i = 0; i < p.rows; ++i) {
            p.res[1 + i] = tok[i];
        }
        p.anchor[0] = tok[acc];
        p.last[0] += acc + 1;
    }
}

int argmax_chunks(int part) { // one pass a chunk up to 64 chunks
    constexpr int per = ARGMAX_THREADS * ARGMAX_PER;
    return std::max(1, std::min(64, (part + per - 1) / per));
}

void argmax_pack(const ArgmaxParams &p, hipStream_t stream) {
    TORCH_CHECK(p.rows <= 64, "argmax_pack: rows <= 64");
    argmax_part_kernel<<<dim3(static_cast<unsigned>(p.chunks), static_cast<unsigned>(p.rows)), ARGMAX_THREADS, 0,
                         stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    argmax_pack_kernel<<<1, 64, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

void argmax_resolve(const ArgmaxParams &p, hipStream_t stream) {
    TORCH_CHECK(p.rows <= 64, "argmax_resolve: rows <= 64");
    argmax_resolve_kernel<<<1, 64, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace infera_flash::dsv41
