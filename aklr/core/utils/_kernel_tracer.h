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
  uint64_t tag;         // the graph exec plan and node it replays; 0 for none
  uint64_t dispatch_id;
  uint64_t kernel_id;
  uint64_t start;
  uint64_t end;
  uint64_t queue;
  rocprofiler_memory_copy_operation_t copy; // NONE for a kernel
  uint64_t bytes;
  uint32_t grid[3];      // in work-items
  uint32_t workgroup[3]; // in work-items
};

// the graph exec plan and node a dispatch replays (-1 for none), and the HIP
// call and scope that captured the node (0 for a node added by hand)
struct Attribution {
  int64_t exec = -1;
  int64_t node = -1;
  uint64_t call = 0;
  uint64_t scope = 0;
};

// what torch.profiler's window recorded, the dispatches' kernels demangled
struct Profile {
  std::vector<Call> calls;
  std::vector<Dispatch> dispatches;
  std::vector<Attribution> attribution;
  std::vector<std::string> kernels;
};

// attaches the tool to the process, once
void attach();

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

// offers the tracer to torch.profiler runs, as a Kineto child profiler
void register_kineto_profiler();

} // namespace aklr::kernel_tracer
