/***************************************************************************************************
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 **************************************************************************************************/

// The kernel tracer as a Kineto child profiler: a torch.profiler run that opts
// in, with custom_profiler_config "aklr_kernel_tracer" and the CPU activity
// only, gets its kernels, copies and HIP calls from the tracer
#include "_kernel_tracer.h"

#include <Config.h>
#include <GenericTraceActivity.h>
#include <IActivityProfiler.h>
#include <fmt/format.h>
#include <libkineto.h>
#include <output_base.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aklr::kernel_tracer {
namespace {

constexpr const char *kOptIn = "aklr_kernel_tracer";

// one torch.profiler run
class Session : public libkineto::IActivityProfilerSession {
public:
  void start() override { start_profile(); }
  void stop() override { profile_ = stop_profile(); }

  void pushCorrelationId(uint64_t id) override { push_op(id); }
  void popCorrelationId() override { pop_op(); }
  void pushUserCorrelationId(uint64_t id) override { push_scope(id); }
  void popUserCorrelationId() override { pop_scope(); }

  std::vector<std::string> errors() override { return {}; }
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return nullptr;
  }
  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return {};
  }
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    return nullptr;
  }

  void processTrace(libkineto::ActivityLogger &) override {}

  // linked gives the op or record_function of a correlation id
  void processTrace(libkineto::ActivityLogger &logger,
                    libkineto::getLinkedActivityCallback linked, int64_t,
                    int64_t) override {
    // rocprofiler-sdk's clock to the system clock torch stamps ops with
    rocprofiler_timestamp_t before = 0, after = 0;
    rocprofiler_get_timestamp(&before);
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    rocprofiler_get_timestamp(&after);
    const int64_t offset = now - static_cast<int64_t>(before / 2 + after / 2);

    auto op_of = [&linked](uint64_t cpu) -> const libkineto::ITraceActivity * {
      return cpu ? linked(static_cast<int32_t>(cpu)) : nullptr;
    };
    const int32_t pid = static_cast<int32_t>(getpid());
    std::deque<libkineto::GenericTraceActivity> activities;
    std::unordered_map<uint64_t, const Call *> calls;
    for (const auto &call : profile_.calls) {
      calls.emplace(call.correlation, &call);
      auto &activity = activities.emplace_back(
          span_, libkineto::ActivityType::CUDA_RUNTIME, hip_api_name(call.op));
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
    for (size_t i = 0; i < profile_.dispatches.size(); ++i) {
      const auto &dispatch = profile_.dispatches[i];
      const auto &a = profile_.attribution[i];
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
      auto &activity = activities.emplace_back(
          span_,
          copy ? libkineto::ActivityType::GPU_MEMCPY
               : libkineto::ActivityType::CONCURRENT_KERNEL,
          copy ? fmt::format("Memcpy {} ({} -> {})", kKinds[kind],
                             kEnds[kind][0], kEnds[kind][1])
               : profile_.kernels[i]);
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
      if (copy) {
        activity.addMetadata("bytes", dispatch.bytes);
      } else {
        activity.addMetadata("grid",
                             fmt::format("[{}, {}, {}]", dispatch.grid[0],
                                         dispatch.grid[1], dispatch.grid[2]));
        activity.addMetadata(
            "block", fmt::format("[{}, {}, {}]", dispatch.workgroup[0],
                                 dispatch.workgroup[1], dispatch.workgroup[2]));
      }
      if (a.exec >= 0) {
        activity.addMetadata("graph exec", a.exec);
        activity.addMetadata("graph node", a.node);
        if (a.call)
          activity.addMetadata("capture correlation", a.call);
        if (a.scope)
          activity.addMetadata("capture scope", a.scope);
      }
    }
    for (const auto &activity : activities)
      logger.handleGenericActivity(activity);
  }

private:
  Profile profile_;
  libkineto::TraceSpan span_{0, 0, kOptIn};
};

class Child : public libkineto::IActivityProfiler {
public:
  const std::string &name() const override { return name_; }
  const std::set<libkineto::ActivityType> &
  availableActivities() const override {
    return types_;
  }

  std::unique_ptr<libkineto::IActivityProfilerSession>
  configure(const std::set<libkineto::ActivityType> &types,
            const libkineto::Config &config) override {
    if (config.getCustomConfig().find(kOptIn) == std::string::npos)
      return nullptr;
    // Kineto's own GPU tracer would record each kernel a second time
    if (types.count(libkineto::ActivityType::CONCURRENT_KERNEL)) {
      throw std::runtime_error(
          "aklr_kernel_tracer records the GPU itself: profile "
          "ProfilerActivity.CPU only");
    }
    attach(); // here, as Kineto drops a session's own errors
    return std::make_unique<Session>();
  }

  std::unique_ptr<libkineto::IActivityProfilerSession>
  configure(int64_t, int64_t, const std::set<libkineto::ActivityType> &types,
            const libkineto::Config &config) override {
    return configure(types, config);
  }

private:
  std::string name_ = kOptIn;
  std::set<libkineto::ActivityType> types_{
      libkineto::ActivityType::CONCURRENT_KERNEL,
      libkineto::ActivityType::GPU_MEMCPY,
      libkineto::ActivityType::CUDA_RUNTIME};
};

} // namespace

void register_kineto_profiler() {
  libkineto::api().registerProfilerFactory(
      [] { return std::make_unique<Child>(); });
}

} // namespace aklr::kernel_tracer
