// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Device primitives: vector types, loads and stores with cache-policy bits, fp8 MFMA fragments,
// wave and block reductions.
#pragma once

#include <hip/hip_runtime.h>

namespace infera_flash {

constexpr int THREADS_PER_WARP = 64;

using bfloat16 = __bf16; // not hip_bfloat16: it rounds in software, with a branch per value

using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));
using int32x4  = int __attribute__((ext_vector_type(4)));
using int32x8  = int __attribute__((ext_vector_type(8)));
using floatx2  = float __attribute__((ext_vector_type(2)));
using floatx4  = float __attribute__((ext_vector_type(4)));

template <typename T, int N> __device__ inline void load_data(const T *src, T *dst) {
    static_assert(N * sizeof(T) == 16, "Only 16 bytes are supported.");
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

template <typename T, int N> __device__ inline void store_data(T *dst, const T *src) {
    static_assert(N * sizeof(T) == 16, "Only 16 bytes are supported.");
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

// sc1 sends a global access through the device-wide coherence point, so other XCDs see the value.
// s_nop 1: a VALU must not overwrite the store's data VGPRs for 2 wait states (ISA 4.5), and the
// compiler does not pad stores inside asm.
__device__ inline void store_sc1(uint32x4 *ptr, uint32x4 val) {
    asm volatile("global_store_dwordx4 %0, %1, off sc1\n\ts_nop 1" ::"v"(ptr), "v"(val) : "memory");
}
__device__ inline void store_sc1(uint2 *ptr, uint2 val) {
    asm volatile("global_store_dwordx2 %0, %1, off sc1\n\ts_nop 1" ::"v"(ptr), "v"(val) : "memory");
}
__device__ inline void store_sc1(uint32_t *ptr, uint32_t val) {
    asm volatile("global_store_dword %0, %1, off sc1\n\ts_nop 1" ::"v"(ptr), "v"(val) : "memory");
}
__device__ inline void store_sc1(bfloat16 *ptr, bfloat16 val) {
    const uint32_t bits = __builtin_bit_cast(uint16_t, val);
    asm volatile("global_store_short %0, %1, off sc1\n\ts_nop 1" ::"v"(ptr), "v"(bits) : "memory");
}

// 16 B packets between blocks of one XCD: the store reaches their L2, an nt load reads it there
// past the CU cache (CDNA4 ISA, tables 49 and 50). asm store, so it is not merged or moved.
__device__ inline void store_l2(uint32x4 *ptr, uint32x4 val) {
    asm volatile("global_store_dwordx4 %0, %1, off\n\ts_nop 1" ::"v"(ptr), "v"(val) : "memory");
}
__device__ inline uint32x4 buffer_load_l2(__amdgpu_buffer_rsrc_t buffer, int offset) {
    constexpr int NT = 2; // the cache-policy operand's nt bit
    return __builtin_bit_cast(uint32x4,
                              __builtin_amdgcn_raw_buffer_load_b128(buffer, offset, 0, NT));
}

// Packets between GPUs, on uncached memory: system scope (sc0 sc1) and nt keep them out of every
// cache even where a mapping is cached. asm, so a polling loop really reloads.
__device__ inline void store_nt(uint32x4 *ptr, uint32x4 val) {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_nop 1" ::"v"(ptr), "v"(val)
                 : "memory");
}

__device__ inline uint32x4 load_nt(const uint32x4 *ptr) {
    uint32x4 val;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)"
                 : "=v"(val)
                 : "v"(ptr)
                 : "memory");
    return val;
}

// A tagged packet's poll (a packet showing tag is complete) with a deadline: past DEADLINE_TICKS of the 100 MHz clock
// (counted from the first check, every POLL_CHECK polls) a lost peer stops the launch (a trap: the process fails)
// instead of hanging it
constexpr uint64_t POLL_DEADLINE_TICKS = 20ull * 100000000ull; // 20 s
constexpr int      POLL_CHECK          = 4096;

__device__ inline uint32x4 poll_tag(const uint32x4 *src, uint32_t tag) {
    uint64_t t0 = 0;
    for (int n = 1;; ++n) {
        const uint32x4 pk = load_nt(src);
        if (pk.w == tag) {
            return pk;
        }
        if (n % POLL_CHECK == 0) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > POLL_DEADLINE_TICKS) {
                __builtin_trap();
            }
        }
    }
}

// Code that uses val stays after this point and memory accesses stay on their side of it: placed
// after later loads, it keeps the wait for val's load from holding them back.
template <typename T> __device__ inline void pin(T &val) {
    asm volatile("" : "+v"(val)::"memory");
}

// __syncthreads also waits for every global load in flight; this waits for LDS accesses only.
__device__ inline void block_sync_lds() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
}

// A raw buffer resource over bytes [base, base + size), for loads with cache-policy bits.
__device__ inline __amdgpu_buffer_rsrc_t raw_buffer(const void *base, int size) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), 0, size, 0x00020000);
}

// A 16 B sc1 load at a byte offset, counted by the compiler like any load (an asm load is not: the
// compiler may move its register before the data lands).
__device__ inline uint32x4 buffer_load_sc1(__amdgpu_buffer_rsrc_t buffer, int offset) {
    constexpr int SC1 = 16; // the cache-policy operand's sc1 bit
    return __builtin_bit_cast(uint32x4,
                              __builtin_amdgcn_raw_buffer_load_b128(buffer, offset, 0, SC1));
}

// Loads N consecutive 16-byte packets with sc1, waiting once for all of them.
template <int N> __device__ inline void load_sc1(const uint32x4 *ptr, uint32x4 (&vals)[N]) {
    static_assert(N == 1 || N == 2);
    if constexpr (N == 1) {
        asm volatile("global_load_dwordx4 %0, %1, off sc1\n\ts_waitcnt vmcnt(0)"
                     : "=v"(vals[0])
                     : "v"(ptr)
                     : "memory");
    } else {
        asm volatile("global_load_dwordx4 %0, %2, off sc1\n\t"
                     "global_load_dwordx4 %1, %2, off offset:16 sc1\n\t"
                     "s_waitcnt vmcnt(0)"
                     : "=&v"(vals[0]), "=&v"(vals[1])
                     : "v"(ptr)
                     : "memory");
    }
}

// One lane's 32 B of an fp8 MFMA 16x16x128 operand: K bytes 16g..16g+15 and 64+16g..64+16g+15 of
// row (A) or column (B) l % 16, for g = l / 16 (ISA layout).
__device__ inline int32x8 load_fragment(const uint8_t *first, const uint8_t *second) {
    const int32x4 lo = *reinterpret_cast<const int32x4 *>(first);
    const int32x4 hi = *reinterpret_cast<const int32x4 *>(second);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

// load_fragment with nt, for weights read once a layer: they stream through L2 and the MALL instead
// of pushing out what is read again (data read ahead for the next kernel, the lm_head between steps)
__device__ inline int32x8 load_fragment_nt(const uint8_t *first, const uint8_t *second) {
    const int32x4 lo = __builtin_nontemporal_load(reinterpret_cast<const int32x4 *>(first));
    const int32x4 hi = __builtin_nontemporal_load(reinterpret_cast<const int32x4 *>(second));
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

// D = A B for 16x16 fp8 e4m3 tiles over K = 128; lane l gets D[4 (l / 16) + i][l % 16] in element
// i. Zero scales make the compiler emit the instruction without block scaling.
__device__ inline floatx4 mfma_16x16x128_fp8(int32x8 a, int32x8 b) {
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, floatx4{}, 0, 0, 0, 0, 0, 0);
}

template <typename T> struct SumOp {
    __device__ static T op(const T &x, const T &y) { return x + y; }
};

template <typename T> struct MaxOp {
    __device__ static T op(const T &x, const T &y) { return x > y ? x : y; }
};

// Each lane gets the value of the lane that DPP_CTRL picks within its 16-lane row. bound_ctrl lets
// the compiler fold the move into the instruction that uses it.
template <int DPP_CTRL, typename T> __device__ inline T dpp_mov(T val) {
    static_assert(sizeof(T) == 4);
    return __builtin_bit_cast(
        T, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, val), DPP_CTRL, 0xF, 0xF, true));
}

template <typename T> __device__ inline T readlane(T val, int lane) {
    static_assert(sizeof(T) == 4);
    return __builtin_bit_cast(T, __builtin_amdgcn_readlane(__builtin_bit_cast(int, val), lane));
}

// Reduces over each aligned group of WIDTH lanes; every lane gets its group's result. DPP within
// a 16-lane row; across rows, row 3 adds row 2 and then rows 0 + 1, and lane 63 is read out. No
// LDS round trip.
template <template <class> class Func, typename T, int WIDTH = THREADS_PER_WARP>
__device__ inline T WarpReduce(T val) {
    static_assert(WIDTH == 16 || WIDTH == THREADS_PER_WARP);
    val = Func<T>::op(dpp_mov<0xB1>(val), val);  // quad_perm [1,0,3,2]
    val = Func<T>::op(dpp_mov<0x4E>(val), val);  // quad_perm [2,3,0,1]
    val = Func<T>::op(dpp_mov<0x141>(val), val); // row_half_mirror
    val = Func<T>::op(dpp_mov<0x140>(val), val); // row_mirror
    if constexpr (WIDTH == THREADS_PER_WARP) {
        val = Func<T>::op(val, dpp_mov<0x142>(val)); // row_bcast:15, row r gets row r - 1
        val = Func<T>::op(val, dpp_mov<0x143>(val)); // row_bcast:31, rows 2, 3 get row 1
        val = readlane(val, 63);
    }
    return val;
}

// Every thread gets the result. Warps are combined in a fixed order so all blocks get the same
// bits. smem holds NUM_WARPS * N elements and can be reused as soon as this returns. LDS_ONLY
// syncs on LDS alone, leaving global loads in flight.
template <template <class> class Func, int BLOCK_SIZE, bool LDS_ONLY = false, typename T, int N>
__device__ inline void BlockReduce(T (&vals)[N], T *smem) {
    const auto sync = [] {
        if constexpr (LDS_ONLY) {
            block_sync_lds();
        } else {
            __syncthreads();
        }
    };
    constexpr int NUM_WARPS = BLOCK_SIZE / THREADS_PER_WARP;
    const int     warp_id   = threadIdx.x / THREADS_PER_WARP;
    const int     lane_id   = threadIdx.x % THREADS_PER_WARP;

#pragma unroll
    for (int n = 0; n < N; ++n) {
        vals[n] = WarpReduce<Func>(vals[n]);
    }
    // not inside the loop above: a branch there would serialize the N independent reductions
    if (lane_id == 0) {
#pragma unroll
        for (int n = 0; n < N; ++n) {
            smem[warp_id * N + n] = vals[n];
        }
    }
    sync();
#pragma unroll
    for (int n = 0; n < N; ++n) {
        vals[n] = smem[n];
#pragma unroll
        for (int w = 1; w < NUM_WARPS; ++w) {
            vals[n] = Func<T>::op(vals[n], smem[w * N + n]);
        }
    }
    sync();
}

} // namespace infera_flash
