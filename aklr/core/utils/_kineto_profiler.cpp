/***************************************************************************************************
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * The chrome trace follows Kineto's ChromeTraceLogger (pytorch/kineto,
 * BSD-3-Clause).
 **************************************************************************************************/

// torch.profiler's GPU tracer in place of Kineto's: Kineto's profiler, its GPU
// events the kernel tracer's
#include "_kernel_tracer.h"

#include <ActivityProfilerInterface.h>
#include <ActivityTraceInterface.h>
#include <GenericTraceActivity.h>
#include <fmt/format.h>
#include <libkineto.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace aklr::kernel_tracer {
namespace {

// a string as a JSON string's contents
std::string json_escape(std::string_view text) {
  std::string out;
  for (char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += fmt::format("\\u{:04x}", c);
    } else {
      out += c;
    }
  }
  return out;
}

// torch.profiler's trace: its CPU events and the tracer's GPU ones
class Trace : public libkineto::ActivityTraceInterface {
public:
  const std::vector<const libkineto::ITraceActivity *> *activities() override {
    return &all;
  }

  // the chrome trace Kineto's ChromeTraceLogger writes, timestamps relative
  // to the earliest event's
  void save(const std::string &path) override {
    int64_t base = INT64_MAX;
    for (const auto *activity : all)
      base = std::min(base, activity->timestamp());
    if (all.empty())
      base = 0;
    auto us = [](int64_t ns) {
      return fmt::format("{}.{:03}", ns / 1000, ns % 1000);
    };
    std::string out =
        "{\n  \"schemaVersion\": 1,\n  \"deviceProperties\": [],\n";
    for (const auto &[key, value] : metadata)
      out += fmt::format("  \"{}\": {},\n", json_escape(key), value);
    out += fmt::format("  \"displayTimeUnit\": \"ms\",\n  "
                       "\"baseTimeNanoseconds\": {},\n  \"traceEvents\": [",
                       base);
    for (const auto *activity : all) {
      const auto type = activity->type();
      // Kineto's metadata is JSON already; it drops what JSON cannot hold
      std::string metadata = activity->metadataJson();
      std::replace(metadata.begin(), metadata.end(), '\\', '/');
      metadata.erase(std::remove(metadata.begin(), metadata.end(), '\n'),
                     metadata.end());
      const auto pid = activity->deviceId();
      const auto tid = std::abs(static_cast<int64_t>(activity->resourceId()));
      if (type == libkineto::ActivityType::CPU_INSTANT_EVENT) {
        out += fmt::format(
            "\n  {{\"ph\": \"i\", \"cat\": \"{}\", \"s\": \"t\", \"name\": "
            "\"{}\", \"pid\": {}, \"tid\": {}, \"ts\": {}, \"args\": {{{}}}}},",
            libkineto::toString(type), json_escape(activity->name()), pid, tid,
            us(activity->timestamp() - base), metadata);
        continue;
      }
      int64_t external = 0;
      if (activity->linkedActivity()) {
        external = activity->linkedActivity()->correlationId();
      } else if (type != libkineto::ActivityType::GPU_MEMCPY &&
                 type != libkineto::ActivityType::GPU_MEMSET &&
                 type != libkineto::ActivityType::CONCURRENT_KERNEL &&
                 type != libkineto::ActivityType::CUDA_RUNTIME &&
                 type != libkineto::ActivityType::CUDA_DRIVER) {
        external = activity->correlationId();
      }
      std::string args = external ? fmt::format("\"External id\": {}", external)
                                  : std::string();
      if (metadata.find_first_not_of(" \t") != std::string::npos)
        args += (args.empty() ? "" : ", ") + metadata;
      out += fmt::format(
          "\n  {{\"ph\": \"X\", \"cat\": \"{}\", \"name\": \"{}\", \"pid\": "
          "{}, \"tid\": {}, \"ts\": {}, \"dur\": {}{}}},",
          libkineto::toString(type), json_escape(activity->name()), pid, tid,
          us(activity->timestamp() - base),
          us(std::max<int64_t>(activity->duration(), 0)),
          args.empty() ? "" : ", \"args\": {" + args + "}");
      const char *flow =
          activity->flowType() == libkineto::kLinkFwdBwd        ? "fwdbwd"
          : activity->flowType() == libkineto::kLinkAsyncCpuGpu ? "ac2g"
                                                                : nullptr;
      if (activity->flowId() > 0 && flow) {
        out += fmt::format(
            "\n  {{\"ph\": \"{}\", \"id\": {}, \"pid\": {}, \"tid\": {}, "
            "\"ts\": {}, \"cat\": \"{}\", \"name\": \"{}\"{}}},",
            activity->flowStart() ? 's' : 'f', activity->flowId(), pid, tid,
            us(activity->timestamp() - base), flow, flow,
            activity->flowStart() ? "" : ", \"bp\": \"e\"");
      }
    }
    out += fmt::format("\n  {{\"name\": \"Record Window End\", \"ph\": \"i\", "
                       "\"s\": \"g\", \"pid\": \"\", \"tid\": \"\", \"ts\": "
                       "{}}}\n  ],\n  \"traceName\": \"{}\"\n}}\n",
                       us(end - base), json_escape(path));
    // written whole, then moved into place, as Kineto does
    const std::string partial = path + ".tmp";
    {
      std::ofstream file(partial, std::ios::binary | std::ios::trunc);
      file << out;
      if (!file.flush())
        throw std::runtime_error("aklr kernel tracer: cannot write " + partial);
    }
    if (std::rename(partial.c_str(), path.c_str()) != 0)
      throw std::runtime_error("aklr kernel tracer: cannot write " + path);
  }

  std::unique_ptr<libkineto::CpuTraceBuffer> cpu;
  libkineto::TraceSpan span{0, 0, "aklr kernel tracer"};
  std::deque<libkineto::GenericTraceActivity> gpu;
  std::vector<const libkineto::ITraceActivity *> all;
  std::map<std::string, std::string> metadata; // key -> JSON value
  int64_t end = 0;
};

// Kineto's profiler, its GPU half the tracer's: torch.profiler drives it
// through libkineto::api(), and so never starts Kineto's own GPU tracer
class Profiler : public libkineto::ActivityProfilerInterface {
public:
  bool isInitialized() override { return true; }
  bool isActive() override { return active_; }

  void prepareTrace(const std::set<libkineto::ActivityType> &types,
                    const std::string &) override {
    gpu_ = types.count(libkineto::ActivityType::CONCURRENT_KERNEL) > 0;
  }

  void startTrace() override {
    if (gpu_)
      start_profile();
    active_ = true;
  }

  std::unique_ptr<libkineto::ActivityTraceInterface> stopTrace() override {
    active_ = false;
    auto trace = std::make_unique<Trace>();
    trace->metadata = std::move(metadata_);
    trace->cpu = std::move(cpu_);
    std::unordered_map<int64_t, const libkineto::ITraceActivity *> ops;
    if (trace->cpu) {
      trace->end = trace->cpu->span.endTime;
      for (const auto &op : trace->cpu->activities) {
        ops.emplace(op->correlationId(), op.get());
        trace->all.push_back(op.get());
      }
    }
    if (!gpu_)
      return trace;
    auto profile = stop_profile();
    // rocprofiler-sdk's clock to the system clock torch stamps ops with
    rocprofiler_timestamp_t before = 0, after = 0;
    rocprofiler_get_timestamp(&before);
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    rocprofiler_get_timestamp(&after);
    const int64_t offset = now - static_cast<int64_t>(before / 2 + after / 2);

    auto op_of = [&ops](uint64_t cpu) -> const libkineto::ITraceActivity * {
      auto op = ops.find(static_cast<int64_t>(cpu));
      return cpu && op != ops.end() ? op->second : nullptr;
    };
    const int32_t pid = static_cast<int32_t>(getpid());
    std::unordered_map<uint64_t, const Call *> calls;
    for (const auto &call : profile.calls) {
      calls.emplace(call.correlation, &call);
      auto &activity = trace->gpu.emplace_back(
          trace->span, libkineto::ActivityType::CUDA_RUNTIME,
          hip_api_name(call.op));
      activity.startTime = static_cast<int64_t>(call.start) + offset;
      activity.endTime = static_cast<int64_t>(call.end) + offset;
      activity.id = static_cast<int32_t>(call.correlation);
      activity.device = pid;
      activity.resource = call.thread;
      activity.threadId = call.thread;
      activity.linked = op_of(call.cpu);
      activity.flow.id = activity.id;
      activity.flow.type = libkineto::kLinkAsyncCpuGpu;
      activity.flow.start = true;
      activity.addMetadata("cid", call.op);
      activity.addMetadata("correlation", call.correlation);
    }
    for (size_t i = 0; i < profile.dispatches.size(); ++i) {
      const auto &dispatch = profile.dispatches[i];
      const auto &a = profile.attribution[i];
      auto call = calls.find(dispatch.correlation);
      const bool copy = dispatch.copy != ROCPROFILER_MEMORY_COPY_NONE;
      static constexpr const char *kKinds[] = {"<unknown>", "HtoH", "HtoD",
                                               "DtoH", "DtoD"};
      static constexpr const char *kEnds[][2] = {{"?", "?"},
                                                 {"Host", "Host"},
                                                 {"Host", "Device"},
                                                 {"Device", "Host"},
                                                 {"Device", "Device"}};
      const auto kind = copy && dispatch.copy < ROCPROFILER_MEMORY_COPY_LAST
                            ? dispatch.copy
                            : ROCPROFILER_MEMORY_COPY_NONE;
      auto &activity = trace->gpu.emplace_back(
          trace->span,
          copy ? libkineto::ActivityType::GPU_MEMCPY
               : libkineto::ActivityType::CONCURRENT_KERNEL,
          copy ? fmt::format("Memcpy {} ({} -> {})", kKinds[kind],
                             kEnds[kind][0], kEnds[kind][1])
               : profile.kernels[i]);
      activity.startTime = static_cast<int64_t>(dispatch.start) + offset;
      activity.endTime = static_cast<int64_t>(dispatch.end) + offset;
      activity.id = static_cast<int32_t>(dispatch.correlation);
      activity.device = call == calls.end() ? 0 : call->second->device;
      activity.resource = static_cast<int32_t>(dispatch.queue);
      // a replayed kernel belongs to the record_function that captured it, as
      // proton.scope's does; any other to the op that launched it
      activity.linked = a.scope && op_of(a.scope) ? op_of(a.scope)
                        : call == calls.end()     ? nullptr
                                                  : op_of(call->second->cpu);
      activity.flow.id = activity.id;
      activity.flow.type = libkineto::kLinkAsyncCpuGpu;
      activity.addMetadata("device", activity.device);
      activity.addMetadata("stream", dispatch.queue);
      activity.addMetadata("correlation", dispatch.correlation);
      activity.addMetadataQuoted("kind",
                                 copy ? kKinds[kind] : "Dispatch Kernel");
      if (copy)
        activity.addMetadata("bytes", dispatch.bytes);
      if (a.launch >= 0) {
        activity.addMetadata("graph launch", a.launch);
        activity.addMetadata("graph ordinal", a.ordinal);
        activity.addMetadata("graph kernels", a.kernels);
        if (a.call)
          activity.addMetadata("capture correlation", a.call);
        if (a.scope)
          activity.addMetadata("capture scope", a.scope);
      }
    }
    for (const auto &activity : trace->gpu)
      trace->all.push_back(&activity);
    return trace;
  }

  void pushCorrelationId(uint64_t id) override { push_op(id); }
  void popCorrelationId() override { pop_op(); }
  void pushUserCorrelationId(uint64_t id) override { push_scope(id); }
  void popUserCorrelationId() override { pop_scope(); }

  void
  transferCpuTrace(std::unique_ptr<libkineto::CpuTraceBuffer> cpu) override {
    cpu_ = std::move(cpu);
  }

  void addMetadata(const std::string &key, const std::string &value) override {
    metadata_[key] = value;
  }

private:
  bool gpu_ = false;
  bool active_ = false;
  std::unique_ptr<libkineto::CpuTraceBuffer> cpu_;
  std::map<std::string, std::string> metadata_;
};

} // namespace

void register_kineto_profiler() {
  // torch registers Kineto's profiler at its first profile, unless one is
  if (libkineto::api().isProfilerRegistered()) {
    throw std::runtime_error("load the aklr kernel tracer before the first "
                             "torch.profiler run: Kineto's is registered");
  }
  libkineto::api().registerProfiler(std::make_unique<Profiler>());
}

} // namespace aklr::kernel_tracer
