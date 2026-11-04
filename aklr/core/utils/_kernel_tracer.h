/***************************************************************************************************
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 **************************************************************************************************/

// The kernel tracer's window as torch.profiler's GPU tracer reads it
#pragma once

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <string>
#include <vector>

namespace aklr::kernel_tracer {

// a HIP call timed for torch.profiler, and the op it ran in; 0 for none
struct Call {
  rocprofiler_tracing_operation_t op;
  uint64_t correlation;
  uint64_t cpu;
  uint64_t start;
  uint64_t end;
  int32_t thread;
  int device;
};

// a kernel dispatch or memory copy of the window
struct Dispatch {
  uint64_t correlation; // the HIP call's that enqueued it
  uint64_t external;    // its graph launch and ordinal; 0 for none
  uint64_t dispatch_id;
  uint64_t kernel_id;
  uint64_t start;
  uint64_t end;
  uint64_t queue;
  rocprofiler_memory_copy_operation_t copy; // NONE for a kernel
  uint64_t bytes;
};

// a dispatch's graph launch in the window (-1 for none), its ordinal there,
// the HIP call and scope that captured it (0 when unknown), and the launched
// graph's kernel nodes (-1 for a graph not captured)
struct Attribution {
  int64_t launch = -1;
  uint64_t ordinal = 0;
  uint64_t call = 0;
  uint64_t scope = 0;
  int64_t kernels = -1;
};

// what torch.profiler's window recorded, the dispatches' kernels demangled
struct Profile {
  std::vector<Call> calls;
  std::vector<Dispatch> dispatches;
  std::vector<Attribution> attribution;
  std::vector<std::string> kernels;
};

// a torch.profiler window: every kernel dispatch and copy, and every HIP call
// with the op it ran in
void start_profile();
Profile stop_profile();

// the op a thread runs, for the HIP calls it makes
void push_op(uint64_t correlation);
void pop_op();

// the record_function a thread runs: a graph it captures replays its kernels
// in it
void push_scope(uint64_t correlation);
void pop_scope();

std::string hip_api_name(rocprofiler_tracing_operation_t op);

// makes the tracer torch.profiler's GPU tracer
void register_kineto_profiler();

} // namespace aklr::kernel_tracer
