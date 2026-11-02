// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// The DSpark draft's greedy Markov chain, a step = argmax over the vocab (split over the TP ranks) of logits[step] +
// markov_embed[prev] . markov_head^T in fp32 sums of the bf16 products. markov_step: this rank's (best, global index)
// into its slot of a zeroed [tp, 2] for the all-reduce; markov_resolve: the last step's token to out[step].
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "common/device.h"

namespace infera_flash::dsv41 {

struct MarkovParams {
    static constexpr int DIM = 256;

    const float    *logits;      // [vocab / tp] this step's row
    const bfloat16 *logits_bf16; // or as bf16 (logits null)
    const bfloat16 *embed;       // [vocab, DIM]
    const bfloat16 *head;   // [vocab / tp, DIM]
    const int64_t  *anchor; // [1] (step 0)
    const float    *prev;   // [tp, 2] the previous step's all-reduced (best, index) (step > 0)
    float          *part;   // [blocks, 2]
    float          *packed; // [tp, 2]
    int64_t        *out;    // [steps]
    int             vocab, tp, rank, step, blocks;
    // tok non-null: the ranks' bests swapped in the finish kernel instead (a 16 B packet with the call number to
    // every rank's exchange buffer recv, two slots by the call's parity); the token to out[step] and tok, which the
    // next step reads for prev (no all-reduce, no markov_resolve)
    void    *recv[8];
    int     *calls; // [1]
    int64_t *tok;   // [1]
    // a batch of nseq sequences' chains (with tok only): sequence s's logits row at logits + s * logits_stride, its
    // anchor[s], tok[s], call counter calls[s] and packets (16 slots from slot0 + s * 16: a batch's from
    // MARKOV_BATCH_SLOT0, its own counters), its tokens at out + s * out_stride
    int      nseq, slot0;
    int64_t  logits_stride, out_stride;
    // AKLR: a sampled step (sample non-zero), as sglang's MarkovSampleStep: round(logits + round(bias)) / temps[s] (round:
    // bf16 with round_bf16) plus Gumbel noise -log(-log1p(-u)), u from Philox4x32-10 keyed (key_lo, key_hi) at counter
    // ((vocab_start + v) / 4, s, step, offset[0]), word (vocab_start + v) % 4; at temperature 0 the plain score
    const float   *temps;
    const int64_t *offset;
    uint32_t       key_lo, key_hi;
    int            vocab_start, sample, round_bf16;
    // AKLR: the finish adds one to offset[0] (bump_offset non-zero: the block's last step, the noise's
    // counter for the next block, after every step kernel read it)
    int bump_offset;
};

// The whole chain in one persistent launch of CHAIN_BLOCKS resident CTAs, a thread a vocab row held in registers, bit
// for bit markov_step + markov_finish + resolve. Tagged 16 B packets carry a running sequence number
// (calls[0] * steps + step + 1), two slots by its parity.
struct MarkovChainParams {
    static constexpr int DIM = 256, CHAIN_BLOCKS = 256, THREADS = 256, SLOT0 = 4096;

    const float    *logits; // [steps, vocab / tp], row stride
    const bfloat16 *embed;  // [vocab, DIM]
    const bfloat16 *head;   // [vocab / tp, DIM]
    const int64_t  *anchor; // [1]
    int64_t        *out;    // [steps]
    void           *recv[8];
    int            *calls; // [1]
    int64_t         stride;
    int             vocab, tp, rank, steps, per;
};

void markov_chain(const MarkovChainParams &p, hipStream_t stream);

int  markov_blocks(int vocab); // markov_step's blocks, the length of part
void markov_step(const MarkovParams &p, hipStream_t stream);
void markov_resolve(const MarkovParams &p, hipStream_t stream);

// The greedy argmax of logits split over the vocab by rank (the lowest rank on equal bests), and with drafts a DSpark
// round's acceptance: accepted = the drafts' leading matches of tokens[:block], res = [accepted, tokens],
// anchor = tokens[accepted], last += accepted + 1.
struct ArgmaxParams {
    const float   *logits;  // [rows, part]
    float         *scratch; // [rows, chunks, 2]: argmax_pack's per-chunk bests
    float         *packed;  // [tp, rows, 2]
    int64_t       *out;     // [rows]
    const int64_t *drafts;  // [block] or null
    int64_t       *res, *anchor, *last;
    int            rows, part, tp, rank, block, chunks;
    // argmax_xchg: the rows' bests swapped in the kernel (16 B packets with the call number to every rank's
    // exchange buffer, from packet ARGMAX_SLOT0, two halves by the call's parity) -> out (and the acceptance)
    void *recv[8];
    int  *calls; // [1]
    // argmax_xchg's acceptance for a batch of nseq sequences of rows / nseq rows each (block + 1): sequence s's drafts
    // at drafts + s * drafts_stride, res [nseq, rows / nseq + 1], anchor [nseq], last [nseq]
    int     nseq;
    int64_t drafts_stride;
    // AKLR: logits as bf16 (logits null); the slice's first valid columns only (0: part)
    const bfloat16 *logits_bf16;
    int             valid;
    // AKLR: sglang's greedy DSpark accept (cand non-null) of nseq requests of n = rows / nseq rows: cand [nseq, n] (anchor,
    // drafts), prefix [nseq] -> correct (leading matches), bonus, cap_trim = 0, commit = correct + 1, new_lens = prefix +
    // commit, out_tokens [nseq, n] (the drafts, the bonus at correct, 0 past them)
    const int64_t *cand;
    const void    *prefix;
    int            prefix_i32;
    int64_t       *correct, *bonus, *new_lens, *out_tokens;
    int32_t       *cap_trim, *commit;
    // AKLR: gated [nseq] = min(commit, verify_lens) * gate[0] (sglang's folded commit lengths; gated null: none),
    // verify_lens [nseq] int64, gate [1] int32
    const int64_t *verify_lens;
    const int32_t *gate;
    int32_t       *gated;
    // AKLR: simulated acceptance (sim [1] int64 non-null), as sglang's build_accept_outs under SGLANG_SIMULATE_ACC_LEN:
    // every request's correct is sim[0] (rank 0's with sim_bcast, carried in the exchange's packets, as sglang's TP
    // broadcast), the bonus the greedy one placed there, the commit lengths and out tokens following from it
    const int64_t *sim;
    int            sim_bcast;
};

// DSpark's variable-length verification of one sequence: c_i = sigmoid([col_i, markov_embed[tok_i]] . w / temperature),
// k = fixed_k >= 0 or the k maximizing (1 + sum_{i < k} prod_{j <= i} c_j) / cost[k]. Verify rows past [anchor,
// drafts[:k]] repeat the last one (no extra experts, the same cache writes); the drafts past k are -1.
struct VerifyPlanParams {
    static constexpr int MAX_N = 8;
    const bfloat16 *col;     // [n, dim]
    const bfloat16 *embed;   // [vocab, mr] the Markov head's embedding
    const bfloat16 *w;       // [dim + mr]
    const int64_t  *tok;     // [n + 1] anchor, drafts
    const int64_t  *pos;     // [n + 1]
    const float    *cost;    // [n + 1] a round's time verifying k = 0 .. n drafts (null: fixed_k)
    int64_t        *vtok;    // [n + 1]
    int64_t        *vpos;    // [n + 1]
    int64_t        *vdrafts; // [n]
    float          *conf;    // [n] (out)
    int32_t        *k_out;   // [1] (out)
    int             n, dim, mr, fixed_k;
    float           inv_temp;
};
void verify_plan(const VerifyPlanParams &p, hipStream_t stream);

constexpr int ARGMAX_SLOT0 = 1024; // the Markov steps' packets below
constexpr int MARKOV_BATCH_SLOT0 = 512, MARKOV_BATCH_MAX = 32; // a batch's Markov steps' packets: up to 32 sequences

int  argmax_chunks(int part);
void argmax_xchg(const ArgmaxParams &p, hipStream_t stream);
void argmax_pack(const ArgmaxParams &p, hipStream_t stream);
void argmax_resolve(const ArgmaxParams &p, hipStream_t stream);

} // namespace infera_flash::dsv41
