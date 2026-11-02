// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// The torch ops of dsv41/markov.hip, index_q.hip and inject.hip: Infera-Flash's markov_step_op and argmax_xchg_op
// (csrc/dsv41/torch_ops.hip) and AKLR's index_q_pack_op and swa_store_op, registered in this library's own namespace
// (TORCH_EXTENSION_NAME).
#include <algorithm>
#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/library.h>

#include "common/torch_utils.h"
#include "dsv41/index_q.h"
#include "dsv41/inject.h"
#include "dsv41/markov.h"

using namespace infera_flash;
using namespace infera_flash::dsv41;

static hipStream_t stream_of(const at::Tensor &t) {
    return at::hip::getCurrentHIPStreamMasqueradingAsCUDA(t.device().index()).stream();
}

// logits [(sequences,) vocab / tp] this step's rows, embed [vocab, 256], head [vocab / tp, 256], anchor; peers (every
// rank's exchange buffer, int64 host), calls [sequences] int32 and tok [sequences] int64: the bests swapped in the
// kernel, sequence s's token to out[s * out_stride + step] and tok[s]
static at::Tensor markov_step_op(const at::Tensor &logits, const at::Tensor &embed, const at::Tensor &head,
                                 const at::Tensor &anchor, const std::optional<at::Tensor> &prev, at::Tensor &out,
                                 int64_t step, int64_t tp, int64_t rank, const std::optional<at::Tensor> &peers,
                                 const std::optional<at::Tensor> &calls, const std::optional<at::Tensor> &tok,
                                 int64_t out_stride, const std::optional<at::Tensor> &temps, int64_t key,
                                 const std::optional<at::Tensor> &offset, int64_t vocab_start, bool round_bf16,
                                 bool bump_offset) {
    using P = MarkovParams;
    const int64_t nseq = logits.dim() == 2 ? logits.size(0) : 1;
    const int64_t part = logits.size(logits.dim() - 1);
    TORCH_CHECK((logits.dim() == 1 || logits.dim() == 2) && logits.stride(-1) == 1 && head.dim() == 2 &&
                    head.size(0) == part && head.size(1) == P::DIM && embed.dim() == 2 && embed.size(1) == P::DIM,
                "logits [(sequences,) vocab / tp], head [vocab / tp, ", P::DIM, "], embed [vocab, ", P::DIM, "]");
    TORCH_CHECK(step == 0 || peers || (prev && prev->numel() == 2 * tp), "a step after the first takes prev [tp, 2]");
    TORCH_CHECK(out.dim() == 1 && (nseq - 1) * out_stride + step < out.size(0), "out [steps] (a sequence)");
    TORCH_CHECK(nseq == 1 || (peers && anchor.numel() >= nseq && tok->numel() >= nseq && calls->numel() >= nseq),
                "markov_step batch: peers, anchor / tok / calls [sequences]");
    const bool bf16 = logits.scalar_type() == at::kBFloat16;
    check(logits[0], logits, bf16 ? at::kBFloat16 : at::kFloat, "logits");
    check(embed, logits, at::kBFloat16, "embed");
    check(head, logits, at::kBFloat16, "head");
    TORCH_CHECK(anchor.numel() >= 1 && anchor.scalar_type() == at::kLong && out.scalar_type() == at::kLong,
                "anchor, out int64");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    P p{};
    p.vocab         = static_cast<int>(part);
    p.blocks        = markov_blocks(p.vocab);
    p.nseq          = static_cast<int>(nseq);
    p.slot0         = nseq > 1 ? MARKOV_BATCH_SLOT0 : 0;
    p.logits_stride = nseq > 1 ? logits.stride(0) : 0;
    p.out_stride    = out_stride;
    auto packed = at::empty({tp, 2}, logits.options().dtype(at::kFloat));
    auto parts  = at::empty({nseq, p.blocks, 2}, packed.options());
    if (bf16) {
        p.logits_bf16 = static_cast<const bfloat16 *>(logits.data_ptr());
    } else {
        p.logits = logits.data_ptr<float>();
    }
    p.embed  = static_cast<const bfloat16 *>(embed.data_ptr());
    p.head   = static_cast<const bfloat16 *>(head.data_ptr());
    p.anchor = anchor.data_ptr<int64_t>();
    p.prev   = prev ? prev->data_ptr<float>() : nullptr;
    p.part   = parts.data_ptr<float>();
    p.packed = packed.data_ptr<float>();
    p.out    = out.data_ptr<int64_t>();
    p.tp     = static_cast<int>(tp);
    p.rank   = static_cast<int>(rank);
    p.step   = static_cast<int>(step);
    if (peers) {
        TORCH_CHECK(calls && tok && peers->device().is_cpu() && peers->scalar_type() == at::kLong &&
                        peers->numel() == tp && tp <= 8,
                    "peers: every rank's exchange buffer (int64, host), with calls and tok");
        check(*calls, logits, at::kInt, "calls");
        check(*tok, logits, at::kLong, "tok");
        p.calls = calls->data_ptr<int>();
        p.tok   = tok->data_ptr<int64_t>();
        for (int r = 0; r < tp; ++r) {
            p.recv[r] = reinterpret_cast<void *>(peers->data_ptr<int64_t>()[r]);
        }
    }
    if (temps) { // AKLR: a sampled step (MarkovParams)
        TORCH_CHECK(offset && temps->numel() >= nseq, "a sampled step takes temps [sequences] and offset [1]");
        check(*temps, logits, at::kFloat, "temps");
        check(*offset, logits, at::kLong, "offset");
        p.temps       = temps->data_ptr<float>();
        p.offset      = offset->data_ptr<int64_t>();
        p.key_lo      = static_cast<uint32_t>(static_cast<uint64_t>(key) & 0xFFFFFFFFull);
        p.key_hi      = static_cast<uint32_t>(static_cast<uint64_t>(key) >> 32);
        p.vocab_start = static_cast<int>(vocab_start);
        p.sample      = 1;
        p.round_bf16  = round_bf16 ? 1 : 0;
        TORCH_CHECK(!bump_offset || p.tok != nullptr, "bump_offset: the finish swapping in the kernel (peers)");
        p.bump_offset = bump_offset ? 1 : 0;
    }
    markov_step(p, stream_of(logits));
    return packed;
}

// logits [rows, part] fp32 (this rank's vocab slice), peers, calls [1] int32 -> tokens [rows] int64, the ranks' bests
// swapped in the kernel
static at::Tensor argmax_xchg_op(const at::Tensor &logits, int64_t rank, const at::Tensor &peers, at::Tensor &calls) {
    TORCH_CHECK(logits.dim() == 2 && logits.scalar_type() == at::kFloat && logits.stride(1) == 1 &&
                    (logits.size(0) == 1 || logits.stride(0) == logits.size(1)),
                "logits [rows, part] fp32 contiguous");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && peers.numel() <= 8,
                "peers: every rank's exchange buffer (int64, host)");
    check(calls, logits, at::kInt, "calls");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    const int    chunks  = argmax_chunks(static_cast<int>(logits.size(1)));
    auto         scratch = at::empty({logits.size(0), chunks, 2}, logits.options());
    auto         out     = at::empty({logits.size(0)}, logits.options().dtype(at::kLong));
    ArgmaxParams p{};
    p.logits  = logits.data_ptr<float>();
    p.scratch = scratch.data_ptr<float>();
    p.out     = out.data_ptr<int64_t>();
    p.rows    = static_cast<int>(logits.size(0));
    p.part    = static_cast<int>(logits.size(1));
    p.tp      = static_cast<int>(peers.numel());
    p.rank    = static_cast<int>(rank);
    p.chunks  = chunks;
    p.calls   = calls.data_ptr<int>();
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    p.nseq = 1;
    argmax_xchg(p, stream_of(logits));
    return out;
}

// AKLR: a DSpark verify's greedy accept off this rank's vocab slice, as sglang's AcceptGreedy, FinalizeAcceptLens and
// BuildOutTokens after the full-vocab argmax, the ranks' bests swapped in the kernel: logits [bs * n, part] bf16 / fp32,
// cand [bs, n], prefix [bs] -> ArgmaxParams' accept outputs (gated given verify_lens and gate); returns the tokens
static at::Tensor dspark_accept_xchg_op(const at::Tensor &logits, int64_t valid, int64_t rank, const at::Tensor &peers,
                                        at::Tensor &calls, const at::Tensor &cand, const at::Tensor &prefix,
                                        at::Tensor &correct, at::Tensor &bonus, at::Tensor &cap_trim, at::Tensor &commit,
                                        at::Tensor &new_lens, at::Tensor &out_tokens,
                                        const std::optional<at::Tensor> &verify_lens,
                                        const std::optional<at::Tensor> &gate, const std::optional<at::Tensor> &gated,
                                        const std::optional<at::Tensor> &sim, bool sim_bcast) {
    const bool bf16 = logits.scalar_type() == at::kBFloat16;
    TORCH_CHECK(logits.dim() == 2 && (bf16 || logits.scalar_type() == at::kFloat) && logits.is_contiguous(),
                "logits [rows, part] bf16 / fp32 contiguous");
    TORCH_CHECK(0 < valid && valid <= logits.size(1), "valid in (0, part]");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && peers.numel() <= 8,
                "peers: every rank's exchange buffer (int64, host)");
    const int64_t bs = cand.size(0), n = cand.dim() == 2 ? cand.size(1) : 0;
    TORCH_CHECK(cand.dim() == 2 && cand.is_contiguous() && bs * n == logits.size(0) && n >= 1,
                "cand [bs, n] contiguous, bs * n logits rows");
    TORCH_CHECK(prefix.dim() == 1 && prefix.size(0) >= bs &&
                    (prefix.scalar_type() == at::kLong || prefix.scalar_type() == at::kInt),
                "prefix [bs] int64 / int32");
    check(calls, logits, at::kInt, "calls");
    check(cand, logits, at::kLong, "cand");
    TORCH_CHECK(prefix.device() == logits.device(), "prefix on the logits' device");
    for (at::Tensor *t : {&correct, &bonus, &new_lens}) {
        check(*t, logits, at::kLong, "correct / bonus / new_lens");
        TORCH_CHECK(t->numel() >= bs, "correct / bonus / new_lens [bs]");
    }
    for (at::Tensor *t : {&cap_trim, &commit}) {
        check(*t, logits, at::kInt, "cap_trim / commit");
        TORCH_CHECK(t->numel() >= bs, "cap_trim / commit [bs]");
    }
    check(out_tokens, logits, at::kLong, "out_tokens");
    TORCH_CHECK(out_tokens.numel() >= bs * n, "out_tokens [bs, n]");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    const int    chunks  = argmax_chunks(static_cast<int>(valid));
    auto         scratch = at::empty({logits.size(0), chunks, 2}, logits.options().dtype(at::kFloat));
    auto         out     = at::empty({logits.size(0)}, logits.options().dtype(at::kLong));
    ArgmaxParams p{};
    if (bf16) {
        p.logits_bf16 = static_cast<const bfloat16 *>(logits.data_ptr());
    } else {
        p.logits = logits.data_ptr<float>();
    }
    p.valid      = static_cast<int>(valid);
    p.scratch    = scratch.data_ptr<float>();
    p.out        = out.data_ptr<int64_t>();
    p.rows       = static_cast<int>(logits.size(0));
    p.part       = static_cast<int>(logits.size(1));
    p.tp         = static_cast<int>(peers.numel());
    p.rank       = static_cast<int>(rank);
    p.chunks     = chunks;
    p.calls      = calls.data_ptr<int>();
    p.nseq       = static_cast<int>(bs);
    p.cand       = cand.data_ptr<int64_t>();
    p.prefix     = prefix.data_ptr();
    p.prefix_i32 = prefix.scalar_type() == at::kInt;
    p.correct    = correct.data_ptr<int64_t>();
    p.bonus      = bonus.data_ptr<int64_t>();
    p.new_lens   = new_lens.data_ptr<int64_t>();
    p.out_tokens = out_tokens.data_ptr<int64_t>();
    p.cap_trim   = cap_trim.data_ptr<int32_t>();
    p.commit     = commit.data_ptr<int32_t>();
    TORCH_CHECK(verify_lens.has_value() == gated.has_value() && gate.has_value() == gated.has_value(),
                "verify_lens, gate and gated together");
    if (gated.has_value()) {
        TORCH_CHECK(verify_lens->dim() == 1 && verify_lens->size(0) >= bs && verify_lens->is_contiguous() &&
                        verify_lens->scalar_type() == at::kLong && verify_lens->device() == logits.device(),
                    "verify_lens [bs] int64");
        TORCH_CHECK(gate->numel() == 1 && gate->scalar_type() == at::kInt && gate->device() == logits.device(),
                    "gate [1] int32");
        TORCH_CHECK(gated->numel() >= bs && gated->is_contiguous() && gated->scalar_type() == at::kInt &&
                        gated->device() == logits.device(),
                    "gated [bs] int32");
        p.verify_lens = verify_lens->data_ptr<int64_t>();
        p.gate        = gate->data_ptr<int32_t>();
        p.gated       = gated->data_ptr<int32_t>();
    }
    if (sim.has_value()) {
        TORCH_CHECK(sim->numel() == 1 && sim->scalar_type() == at::kLong && sim->device() == logits.device(),
                    "sim [1] int64");
        p.sim       = sim->data_ptr<int64_t>();
        p.sim_bcast = sim_bcast;
    }
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    argmax_xchg(p, stream_of(logits));
    return out;
}

// qr [rows, 1280] bf16, wq_b (w [heads * 128, 1280] e4m3, ws [heads * 4, 40] ue8m0), x [rows, dim] bf16, weights_proj
// [heads, dim] bf16, pos [rows] int64, table [positions, pairs, 2] fp32 -> index_q_rope_pack_weights_flydsl's (q_fp4,
// q_scale, weights); with gate_w [n, 5120] bf16 a fourth, x . gate_w^T [rows, n] fp32
static std::vector<at::Tensor> index_q_pack_op(const at::Tensor &qr, const at::Tensor &w, const at::Tensor &ws,
                                               const at::Tensor &x, const at::Tensor &wp, const at::Tensor &pos,
                                               const at::Tensor &table, double weight_scale,
                                               const std::optional<at::Tensor> &gate_w) {
    using Q = IndexQPackParams;
    TORCH_CHECK(qr.dim() == 2 && qr.size(1) == Q::QRANK && qr.is_contiguous() && qr.scalar_type() == at::kBFloat16,
                "qr [rows, 1280] bf16 contiguous");
    const int64_t rows = qr.size(0), heads = w.size(0) / Q::DIM;
    TORCH_CHECK(w.dim() == 2 && w.size(0) % Q::DIM == 0 && w.size(1) == Q::QRANK && w.element_size() == 1 &&
                    w.is_contiguous(),
                "w [heads * 128, 1280] e4m3");
    TORCH_CHECK(ws.dim() == 2 && ws.size(0) == w.size(0) / 32 && ws.size(1) == Q::QRANK / 32 && ws.element_size() == 1 &&
                    ws.is_contiguous(),
                "ws [heads * 4, 40] ue8m0");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == rows && x.is_contiguous() && x.scalar_type() == at::kBFloat16,
                "x [rows, dim] bf16 contiguous");
    TORCH_CHECK(wp.dim() == 2 && wp.size(0) == heads && wp.size(1) == x.size(1) && wp.is_contiguous() &&
                    wp.scalar_type() == at::kBFloat16,
                "weights_proj [heads, dim] bf16");
    TORCH_CHECK(pos.dim() == 1 && pos.size(0) == rows && pos.scalar_type() == at::kLong, "pos [rows] int64");
    TORCH_CHECK(table.dim() == 3 && table.size(2) == 2 && table.scalar_type() == at::kFloat && table.is_contiguous() &&
                    table.size(1) * 2 <= Q::DIM,
                "table [n, pairs, 2] fp32");
    for (const at::Tensor *t : {&w, &ws, &x, &wp, &pos, &table}) {
        TORCH_CHECK(t->device() == qr.device(), "every tensor on qr's device");
    }
    if (gate_w.has_value()) {
        TORCH_CHECK(gate_w->dim() == 2 && gate_w->size(1) == x.size(1) && gate_w->is_contiguous() &&
                        gate_w->scalar_type() == at::kBFloat16 && gate_w->device() == qr.device(),
                    "gate_w [n, dim] bf16 contiguous");
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(qr.device());
    std::vector<at::Tensor> outs{at::empty({rows, heads, Q::DIM / 2}, qr.options().dtype(at::kChar)),
                                 at::empty({rows, 1, 4, 16, 4}, qr.options().dtype(at::kByte)),
                                 at::empty({rows, heads}, qr.options())};
    if (gate_w.has_value()) {
        outs.push_back(at::empty({rows, gate_w->size(0)}, qr.options().dtype(at::kFloat)));
    }
    IndexQPackParams p{};
    p.qr           = static_cast<const bfloat16 *>(qr.data_ptr());
    p.w            = static_cast<const uint8_t *>(w.data_ptr());
    p.ws           = static_cast<const uint8_t *>(ws.data_ptr());
    p.x            = static_cast<const bfloat16 *>(x.data_ptr());
    p.wp           = static_cast<const bfloat16 *>(wp.data_ptr());
    p.pos          = pos.data_ptr<int64_t>();
    p.table        = table.data_ptr<float>();
    p.q_fp4        = static_cast<uint8_t *>(outs[0].data_ptr());
    p.q_scale      = static_cast<uint8_t *>(outs[1].data_ptr());
    p.weights      = static_cast<bfloat16 *>(outs[2].data_ptr());
    p.gate_w       = gate_w.has_value() ? static_cast<const bfloat16 *>(gate_w->data_ptr()) : nullptr;
    p.gate_out     = gate_w.has_value() ? outs[3].data_ptr<float>() : nullptr;
    p.gate_n       = gate_w.has_value() ? static_cast<int>(gate_w->size(0)) : 0;
    p.weight_scale = static_cast<float>(weight_scale);
    p.positions    = table.size(0);
    p.rows         = static_cast<int>(std::min<int64_t>(rows, Q::MAX_ROWS));
    p.total        = static_cast<int>(rows);
    p.heads        = static_cast<int>(heads);
    p.dim          = static_cast<int>(x.size(1));
    p.pairs        = static_cast<int>(table.size(1));
    index_q_pack(p, stream_of(qr));
    return outs;
}

// AKLR: kv [rows, >= (stages - 1) * stage_stride + 512] bf16 (stage s's rows from column s * stage_stride), weights
// [stages] of [512] bf16, freqs [stages] of [positions, 64] fp32, pos [rows] int64 / int32, loc [rows] int32 ->
// every stage's V4 / V4.1 (v41) SWA cache caches[s] [pages, page_bytes] uint8 at loc (rows with loc < 0 skipped)
static void swa_store_op(const at::Tensor &kv, int64_t stage_stride, at::TensorList weights, at::TensorList freqs,
                         const at::Tensor &pos, const at::Tensor &loc, at::TensorList caches, double eps,
                         int64_t page_size, bool v41, const std::optional<at::Tensor> &cache_loc,
                         const std::optional<at::Tensor> &mapping, const std::optional<at::Tensor> &commit,
                         int64_t width) {
    using P             = SwaStoreParams;
    const int64_t stages = caches.size();
    TORCH_CHECK(0 < stages && stages <= P::MAX_STAGES && weights.size() == caches.size() &&
                    freqs.size() == caches.size(),
                "1 to ", P::MAX_STAGES, " stages: a weight, freqs and cache each");
    TORCH_CHECK(kv.dim() == 2 && kv.scalar_type() == at::kBFloat16 && kv.stride(1) == 1 &&
                    kv.size(1) >= P::DIM && stage_stride >= 0,
                "kv [rows, 512] bf16 rows");
    const int64_t rows = kv.size(0);
    TORCH_CHECK(pos.dim() == 1 && pos.size(0) == rows && pos.is_contiguous() &&
                    (pos.scalar_type() == at::kLong || pos.scalar_type() == at::kInt) && pos.device() == kv.device(),
                "pos [rows] int64 / int32");
    const bool committed = cache_loc.has_value();
    TORCH_CHECK(committed == mapping.has_value() && committed == commit.has_value(),
                "cache_loc, mapping and commit together");
    if (committed) { // AKLR: the slots from the committed rows (SwaStoreParams::cache_loc)
        TORCH_CHECK(cache_loc->dim() == 1 && cache_loc->size(0) == rows && cache_loc->is_contiguous() &&
                        cache_loc->scalar_type() == at::kLong && cache_loc->device() == kv.device(),
                    "cache_loc [rows] int64");
        TORCH_CHECK(mapping->dim() == 1 && mapping->is_contiguous() && mapping->scalar_type() == at::kLong &&
                        mapping->device() == kv.device(),
                    "mapping int64");
        TORCH_CHECK(width > 0 && rows % width == 0 && commit->numel() >= rows / width && commit->is_contiguous() &&
                        commit->scalar_type() == at::kInt && commit->device() == kv.device(),
                    "commit [rows / width] int32");
    } else {
        TORCH_CHECK(loc.dim() == 1 && loc.size(0) == rows && loc.is_contiguous() && loc.scalar_type() == at::kInt &&
                        loc.device() == kv.device(),
                    "loc [rows] int32");
    }
    TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size a power of 2");
    const int     page_bits  = __builtin_ctzll(page_size);
    const int64_t token_bytes = v41 ? P::DIM + P::DIM / 32 : 576 + 8;
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(kv.device());
    P p{};
    for (int64_t s = 0; s < stages; ++s) {
        check(weights[s], kv, at::kBFloat16, "weight");
        TORCH_CHECK(weights[s].numel() == P::DIM, "weight [512]");
        check(freqs[s], kv, at::kFloat, "freqs");
        TORCH_CHECK(freqs[s].dim() == 2 && freqs[s].size(1) == P::ROPE, "freqs [positions, 64]");
        TORCH_CHECK(caches[s].dim() == 2 && caches[s].scalar_type() == at::kByte && caches[s].stride(1) == 1 &&
                        caches[s].device() == kv.device() && caches[s].size(1) >= token_bytes * page_size,
                    "cache [pages, page_bytes] uint8, a V4 / V4.1 page");
        TORCH_CHECK(s == 0 || caches[s].stride(0) == caches[0].stride(0), "the caches' page bytes differ");
        p.weight[s] = static_cast<const bfloat16 *>(weights[s].data_ptr());
        p.freqs[s]  = freqs[s].data_ptr<float>();
        p.cache[s]  = static_cast<uint8_t *>(caches[s].data_ptr());
    }
    p.kv           = static_cast<const bfloat16 *>(kv.data_ptr());
    p.kv_stride    = kv.stride(0);
    p.stage_stride = stage_stride;
    p.pos          = pos.data_ptr();
    p.pos_i32      = pos.scalar_type() == at::kInt;
    p.v41          = v41;
    if (committed) {
        p.loc       = nullptr;
        p.cache_loc = cache_loc->data_ptr<int64_t>();
        p.map       = mapping->data_ptr<int64_t>();
        p.map_n     = mapping->numel();
        p.commit    = commit->data_ptr<int32_t>();
        p.width     = static_cast<int>(width);
    } else {
        p.loc = loc.data_ptr<int32_t>();
    }
    p.rows         = static_cast<int>(rows);
    p.stages       = static_cast<int>(stages);
    p.page_bits    = page_bits;
    p.page_bytes   = caches[0].stride(0);
    p.eps          = static_cast<float>(eps);
    swa_store(p, stream_of(kv));
}

#ifndef TORCH_LIBRARY_EXPAND // torch < 2.6
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("markov_step(Tensor logits, Tensor embed, Tensor head, Tensor anchor, Tensor? prev, Tensor(a!) out, int step, "
          "int tp, int rank, Tensor? peers=None, Tensor(b!)? calls=None, Tensor(c!)? tok=None, int out_stride=0, "
          "Tensor? temps=None, int key=0, Tensor? offset=None, int vocab_start=0, bool round_bf16=False, "
          "bool bump_offset=False) -> Tensor",
          markov_step_op);
    m.def("argmax_xchg(Tensor logits, int rank, Tensor peers, Tensor(a!) calls) -> Tensor", argmax_xchg_op);
    m.def("dspark_accept_xchg(Tensor logits, int valid, int rank, Tensor peers, Tensor(a!) calls, Tensor cand, "
          "Tensor prefix, Tensor(b!) correct, Tensor(c!) bonus, Tensor(d!) cap_trim, Tensor(e!) commit, "
          "Tensor(f!) new_lens, Tensor(g!) out_tokens, Tensor? verify_lens=None, Tensor? gate=None, "
          "Tensor(h!)? gated=None, Tensor? sim=None, bool sim_bcast=False) -> Tensor",
          dspark_accept_xchg_op);
    m.def("index_q_pack(Tensor qr, Tensor w, Tensor ws, Tensor x, Tensor wp, Tensor pos, Tensor table, "
          "float weight_scale, Tensor? gate_w=None) -> Tensor[]",
          index_q_pack_op);
    m.def("swa_store(Tensor kv, int stage_stride, Tensor[] weights, Tensor[] freqs, Tensor pos, Tensor loc, "
          "Tensor(a!)[] caches, float eps, int page_size, bool v41, Tensor? cache_loc=None, Tensor? mapping=None, "
          "Tensor? commit=None, int width=0) -> ()",
          swa_store_op);
}
