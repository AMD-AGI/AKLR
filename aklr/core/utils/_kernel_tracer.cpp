/***************************************************************************************************
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * The graph capture attribution follows rocprofiler-sdk's hip/graph.cpp in
 *ROCm/rocm-systems (feature/rocprofiler-graph-capture-attribution, MIT),
 *rebuilt on rocprofiler-sdk's public API.
 **************************************************************************************************/

// Kernel dispatch timestamps from rocprofiler-sdk, each replayed graph dispatch
// matched to the HIP call and the scope that captured it, for
// testing.bench_cudagraph; and, for torch.profiler (_kineto_profiler.cpp), HIP
// calls and copies, collected as Kineto's rocprofiler-sdk backend
// (pytorch/kineto RocprofLogger, BSD-3-Clause) collects them
#include "_kernel_tracer.h"

#include <cxxabi.h>
#include <hip/hip_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// shortcut: rebuilds in a tool what rocprofiler-sdk 1.3.5 lacks; retire it when
// rocprofiler-sdk records each dispatch's capture_correlation_id
// (ROCm/rocm-systems #14204, #14211)
namespace aklr::kernel_tracer {
namespace {

// a dispatch's external correlation id: its graph launch above, its ordinal in
// the launch below
constexpr int kOrdinalBits = 24;

// a captured graph node a replay can dispatch: a kernel, or (kernel empty) a
// copy or memset
struct Node {
  std::string kernel;
  uint64_t call = 0;
  uint64_t scope = 0;
};

// a captured graph's nodes in the order its launch enqueues them
struct Plan {
  std::vector<Node> nodes;
  bool copies = false; // every node of the graph was captured, so the runtime's
                       // blit kernels are its copies
  int64_t kernels = 0; // the graph's kernel nodes, captured or not: the
                       // non-blit dispatches of each launch
};

struct Captured {
  hipGraphNode_t node;
  uint64_t call;
  uint64_t scope;
};

// a HIP call that changed its stream's capture dependencies: the newest of them
// is the last node it added
struct Mark {
  std::vector<hipGraphNode_t> leaves;
  uint64_t call;
  uint64_t scope;
};

// a stream capture begun on this thread
struct Capture {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  size_t known = 0; // the graph's nodes before the capture
  std::unordered_map<hipStream_t, std::vector<hipGraphNode_t>>
      leaves; // each capturing stream's, last seen
  std::vector<Mark> marks;
};

struct Launch {
  uint64_t id;
  uint64_t next_ordinal;
};

thread_local bool t_in_tool =
    false; // the tool's own HIP queries reach its HIP API callback too
thread_local std::vector<Capture> t_captures;
thread_local std::vector<Launch> t_launches;
thread_local std::vector<uint64_t> t_ops; // torch.profiler's op correlation ids

class KernelTracer {
public:
  // reaches the process's one tool: rocprofiler_force_configure's callback
  // carries no pointer to it
  static KernelTracer &instance() {
    static KernelTracer tracer;
    return tracer;
  }

  void configure() {
    // WORKAROUND: ROCM-29087, fixed in rocprofiler-sdk 1.4.2 (ROCm 10.2);
    // delete when the images ship it
#if ROCPROFILER_SDK_VERSION < 10402
    // below 1.4.2 the default queue interposition deadlocks a HIP graph whose
    // streams fork and join
    setenv("ROCPROFILER_QUEUE_INTERPOSITION", "0",
           1); // the legacy queue intercept does not
#endif
    std::vector<std::pair<const char *, std::optional<std::string>>> inherited;
    for (const char *name :
         {"ROCPROFILER_REGISTER_FORCE_LOAD", "ROCPROFILER_REGISTER_LIBRARY"}) {
      const char *value = std::getenv(name);
      inherited.emplace_back(name, value ? std::optional<std::string>(value)
                                         : std::nullopt);
    }
    check(rocprofiler_force_configure(&KernelTracer::tool), "configure");
    // force_configure exports these; a child inheriting them loads
    // rocprofiler-sdk before it can configure
    for (const auto &[name, value] : inherited) {
      value ? setenv(name, value->c_str(), 1) : unsetenv(name);
    }
    if (dispatches_.handle == 0) {
      throw std::runtime_error(
          "rocprofiler-sdk: kernel dispatch tracing did not initialize");
    }
  }

  // a window records every dispatch; torch.profiler's times HIP calls too
  void start(bool calls) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      window_.clear();
      calls_.clear();
      launched_.clear();
      dropped_ = 0;
      first_launch_ = next_launch_;
    }
    check(rocprofiler_start_context(dispatches_), "start");
    if (!probed_) {
      probe();
      probed_ = true;
    }
    timing_ = calls;
  }

  // the HIP calls a thread makes between push and pop capture their graph nodes
  // into this scope
  void push(uint64_t scope) {
    check(rocprofiler_push_external_correlation_id(captures_, thread_id(),
                                                   {.value = scope}),
          "push a scope");
  }

  void pop() {
    rocprofiler_user_data_t scope{};
    check(
        rocprofiler_pop_external_correlation_id(captures_, thread_id(), &scope),
        "pop a scope");
  }

  void stop() {
    check(rocprofiler_stop_context(dispatches_), "stop");
    timing_ = false;
  }

  // torch.profiler's window, after stop()
  Profile profile() {
    await_replays();
    std::lock_guard<std::mutex> lock(mutex_);
    Profile profile;
    profile.attribution = attribute();
    const auto unmatched = std::count_if(
        profile.attribution.begin(), profile.attribution.end(),
        [](const Attribution &a) { return a.kernels >= 0 && !a.call; });
    if (unmatched) {
      throw std::runtime_error(std::to_string(unmatched) +
                               " dispatches of the graph replays matched no "
                               "captured call");
    }
    for (const auto &dispatch : window_) {
      auto name = names_.find(dispatch.kernel_id);
      profile.kernels.push_back(name == names_.end() ? "<unknown kernel>"
                                                     : demangle(name->second));
    }
    profile.calls.swap(calls_);
    profile.dispatches.swap(window_);
    return profile;
  }

private:
  KernelTracer() = default;

  static void check(rocprofiler_status_t status, const char *what) {
    if (status != ROCPROFILER_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("rocprofiler-sdk: ") + what + ": " +
                               rocprofiler_get_status_string(status));
    }
  }

  static rocprofiler_thread_id_t thread_id() {
    rocprofiler_thread_id_t id = 0;
    check(rocprofiler_get_thread_id(&id), "thread id");
    return id;
  }

  // rocprofiler-sdk takes a tool only before the ROCm runtime initializes: a
  // tool configured later sees no dispatch to a queue made before it
  void probe() {
    t_in_tool = true; // the probe's calls are no one's
    void *word = nullptr;
    const bool ran = hipMalloc(&word, sizeof(uint32_t)) == hipSuccess &&
                     hipMemsetD32(word, 0, 1) == hipSuccess &&
                     hipDeviceSynchronize() == hipSuccess;
    t_in_tool = false;
    bool seen = false;
    // a completed dispatch's record may reach the buffer a little later
    for (int wait = 0; ran && !seen && wait < 1000; ++wait) {
      if (wait)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      check(rocprofiler_flush_buffer(buffer_), "flush");
      std::lock_guard<std::mutex> lock(mutex_);
      seen = !window_.empty();
      window_.clear();
    }
    if (word)
      (void)hipFree(word);
    if (!seen) {
      check(rocprofiler_stop_context(dispatches_), "stop");
      throw std::runtime_error("aklr kernel tracer: load it before the "
                               "process's first GPU call");
    }
  }

  // a dispatch's record can trail the synchronize that saw its kernel end (0.1
  // s seen under load): wait until each launch of a captured graph has its
  // graph's kernel nodes, or 10 s pass
  void await_replays() {
    for (int wait = 0;; ++wait) {
      check(rocprofiler_flush_buffer(buffer_), "flush");
      std::lock_guard<std::mutex> lock(mutex_);
      if (dropped_) {
        throw std::runtime_error("rocprofiler-sdk dropped " +
                                 std::to_string(dropped_) + " records");
      }
      std::map<uint64_t, int64_t> kernels;
      for (const auto &dispatch : window_) {
        auto name = names_.find(dispatch.kernel_id);
        if (dispatch.external >> kOrdinalBits && name != names_.end() &&
            !is_blit(name->second)) {
          ++kernels[dispatch.external >> kOrdinalBits];
        }
      }
      int64_t recorded = 0, expected = 0;
      for (const auto &[launch, plan] : launched_) {
        if (plan && plan->kernels >= 0) {
          recorded += std::min(kernels[launch], plan->kernels);
          expected += plan->kernels;
        }
      }
      if (recorded == expected)
        return;
      if (wait == 10000) {
        throw std::runtime_error(
            "rocprofiler-sdk recorded " + std::to_string(recorded) + " of " +
            std::to_string(expected) + " kernels of the graph replays");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // the runtime's blit kernels replay copy and memset nodes, not kernel nodes
  static bool is_blit(const std::string &kernel) {
    return kernel.rfind("__amd_rocclr_", 0) == 0;
  }

  // each window dispatch's graph launch, ordinal and capturing call; the
  // caller holds mutex_
  std::vector<Attribution> attribute() const {
    std::vector<Attribution> attribution(window_.size());
    std::map<uint64_t, std::vector<std::pair<uint64_t, size_t>>> by_launch;
    for (size_t i = 0; i < window_.size(); ++i) {
      const auto &dispatch = window_[i];
      if (dispatch.copy == ROCPROFILER_MEMORY_COPY_NONE &&
          dispatch.external >> kOrdinalBits) {
        by_launch[dispatch.external >> kOrdinalBits].emplace_back(
            dispatch.external & ((1ull << kOrdinalBits) - 1), i);
      }
    }
    for (auto &[launch, dispatches] : by_launch) {
      std::sort(dispatches.begin(), dispatches.end());
      std::vector<std::string> kernels;
      for (const auto &[ordinal, i] : dispatches) {
        auto name = names_.find(window_[i].kernel_id);
        kernels.push_back(name == names_.end() ? std::string() : name->second);
      }
      auto plan = launched_.find(launch);
      auto nodes = plan == launched_.end() || !plan->second
                       ? std::vector<const Node *>(dispatches.size())
                       : match(*plan->second, kernels);
      for (size_t j = 0; j < dispatches.size(); ++j) {
        const auto &[ordinal, i] = dispatches[j];
        attribution[i] = {
            static_cast<int64_t>(launch - first_launch_), ordinal,
            nodes[j] ? nodes[j]->call : 0, nodes[j] ? nodes[j]->scope : 0,
            plan == launched_.end() || !plan->second ? -1
                                                     : plan->second->kernels};
      }
    }
    return attribution;
  }

  static std::string demangle(const std::string &name) {
    int status = 0;
    std::unique_ptr<char, decltype(&std::free)> demangled(
        abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
        &std::free);
    return status == 0 ? demangled.get() : name;
  }

  // the node each dispatch of a launch replays, or null: a dispatch takes the
  // plan's next kernel node when its kernel is that node's, else, if a blit
  // kernel, the next copy or memset node before it
  static std::vector<const Node *>
  match(const Plan &plan, const std::vector<std::string> &kernels) {
    std::vector<const Node *> nodes(kernels.size());
    size_t next = 0, gap = 0;  // the plan's next node; the first node after the
                               // last kernel matched
    std::vector<size_t> blits; // the dispatches the gap's copies took
    auto next_kernel = [&] {
      size_t kernel = next;
      while (kernel < plan.nodes.size() && plan.nodes[kernel].kernel.empty())
        ++kernel;
      return kernel;
    };
    // a copy that took no blit kernel, or two, leaves all of its gap's blits
    // unknown
    auto close_gap = [&](size_t kernel) {
      if (blits.size() != kernel - gap) {
        for (size_t blit : blits)
          nodes[blit] = nullptr;
      }
      blits.clear();
    };
    for (size_t i = 0; i < kernels.size(); ++i) {
      size_t kernel = next_kernel();
      if (kernel < plan.nodes.size() &&
          kernels[i] == plan.nodes[kernel].kernel) {
        close_gap(kernel);
        nodes[i] = &plan.nodes[kernel];
        next = gap = kernel + 1;
      } else if (plan.copies && next < kernel && is_blit(kernels[i])) {
        nodes[i] = &plan.nodes[next++];
        blits.push_back(i);
      }
    }
    close_gap(next_kernel());
    return nodes;
  }

  static rocprofiler_tool_configure_result_t *
  tool(uint32_t, const char *, uint32_t, rocprofiler_client_id_t *id) {
    id->name = "aklr-kernel-tracer";
    static rocprofiler_tool_configure_result_t result{
        sizeof(rocprofiler_tool_configure_result_t), &KernelTracer::initialize,
        nullptr, &instance()};
    return &result;
  }

  static int initialize(rocprofiler_client_finalize_t, void *tool_data) {
    auto &tracer = *static_cast<KernelTracer *>(tool_data);
    rocprofiler_tracing_operation_t symbol =
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    rocprofiler_external_correlation_id_request_kind_t dispatch =
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH;
    // captures_ runs for the process: a graph is captured and instantiated
    // before its replays are traced
    if (rocprofiler_create_context(&tracer.captures_) !=
            ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_callback_tracing_service(
            tracer.captures_, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, &symbol,
            1, &KernelTracer::on_symbol,
            &tracer) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_callback_tracing_service(
            tracer.captures_, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
            nullptr, 0, &KernelTracer::on_hip,
            &tracer) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_create_context(&tracer.dispatches_) !=
            ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_create_buffer(
            tracer.dispatches_, 1 << 22, 1 << 21,
            ROCPROFILER_BUFFER_POLICY_LOSSLESS, &KernelTracer::collect, &tracer,
            &tracer.buffer_) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_buffer_tracing_service(
            tracer.dispatches_, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
            nullptr, 0, tracer.buffer_) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_buffer_tracing_service(
            tracer.dispatches_, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, nullptr,
            0, tracer.buffer_) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_external_correlation_id_request_service(
            tracer.dispatches_, &dispatch, 1, &KernelTracer::on_dispatch,
            &tracer) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_start_context(tracer.captures_) !=
            ROCPROFILER_STATUS_SUCCESS) {
      return -1;
    }
    return 0;
  }

  static void on_symbol(rocprofiler_callback_tracing_record_t record,
                        rocprofiler_user_data_t *, void *data) {
    // the code object's own load comes here too, with another payload
    if (record.operation !=
            ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
        record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) {
      return;
    }
    const auto *symbol = static_cast<
        const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t
            *>(record.payload);
    if (!symbol->kernel_name)
      return;
    auto &tracer = *static_cast<KernelTracer *>(data);
    std::string_view name = symbol->kernel_name;
    if (name.size() > 3 && name.substr(name.size() - 3) == ".kd") {
      name.remove_suffix(3); // the kernel descriptor's, not the kernel's
    }
    std::lock_guard<std::mutex> lock(tracer.mutex_);
    tracer.names_.emplace(symbol->kernel_id, name);
  }

  static void on_hip(rocprofiler_callback_tracing_record_t record,
                     rocprofiler_user_data_t *user_data, void *data) {
    if (t_in_tool) {
      return;
    }
    t_in_tool = true;
    auto &tracer = *static_cast<KernelTracer *>(data);
    tracer.time_call(record, *user_data);
    const auto *api =
        static_cast<const rocprofiler_callback_tracing_hip_api_data_t *>(
            record.payload);
    const auto &args = api->args;
    const bool ok = api->retval.hipError_t_retval == hipSuccess;
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
      if (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch ||
          record.operation ==
              ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt) {
        tracer.launch(args.hipGraphLaunch.graphExec);
      }
      t_in_tool = false;
      return;
    }
    switch (record.operation) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt:
      t_launches.pop_back();
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCaptureToGraph:
      // the stream is the first argument of all three
      if (ok)
        begin_capture(args.hipStreamBeginCapture.stream);
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamEndCapture_spt:
      tracer.end_capture(args.hipStreamEndCapture.stream,
                         ok && args.hipStreamEndCapture.pGraph
                             ? *args.hipStreamEndCapture.pGraph
                             : nullptr);
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiate:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithFlags:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphInstantiateWithParams:
      // the exec and the graph are the first two arguments of all three
      if (ok && args.hipGraphInstantiate.pGraphExec &&
          *args.hipGraphInstantiate.pGraphExec) {
        tracer.plan(*args.hipGraphInstantiate.pGraphExec,
                    args.hipGraphInstantiate.graph);
      }
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroy:
      if (ok) {
        std::lock_guard<std::mutex> lock(tracer.mutex_);
        tracer.captured_.erase(args.hipGraphDestroy.graph);
      }
      break;
    // a changed or destroyed exec no longer replays its plan: its launches go
    // unattributed
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecUpdate:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecKernelNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecChildGraphNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParams1D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParamsFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParamsToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemsetNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeSetEnabled:
      if (ok) { // a failed change leaves the exec as it was
        // the exec is the first argument of each
        std::lock_guard<std::mutex> lock(tracer.mutex_);
        tracer.plans_.erase(args.hipGraphExecDestroy.graphExec);
      }
      break;
    default:
      break;
    }
    // the nodes a call adds to a capture are the call's, in its caller's scope;
    // hipGraphGetNodes costs a graph's size, so a call is marked by its
    // stream's new dependencies and the nodes are listed at the end
    if (!t_captures.empty()) {
      if (auto stream = stream_of(record))
        mark(*stream, record.correlation_id);
    }
    t_in_tool = false;
  }

  // Kineto leaves these out of its traces: frequent, and no GPU work
  static bool untimed(rocprofiler_tracing_operation_t op) {
    switch (op) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipSetDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGetLastError:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipCtxSetCurrent:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipEventQuery:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDevicePropertiesR0600:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipPeekAtLastError:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleGetFunction:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipEventCreateWithFlags:
      return true;
    default:
      return false;
    }
  }

  // a call's enter keeps its start in the call's user data
  void time_call(const rocprofiler_callback_tracing_record_t &record,
                 rocprofiler_user_data_t &start) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
      start.value = 0;
      if (timing_ && !untimed(record.operation))
        rocprofiler_get_timestamp(&start.value);
      return;
    }
    if (start.value == 0 || !timing_)
      return;
    // the system thread id torch stamps the thread's ops with
    Call call{record.operation,
              record.correlation_id.internal,
              t_ops.empty() ? 0 : t_ops.back(),
              start.value,
              0,
              static_cast<int32_t>(syscall(SYS_gettid)),
              0};
    rocprofiler_get_timestamp(&call.end);
    (void)hipGetDevice(&call.device); // on failure, device 0
    std::lock_guard<std::mutex> lock(mutex_);
    calls_.push_back(call);
  }

  // the hipStream_t argument of a HIP call, if it has one
  static std::optional<hipStream_t>
  stream_of(const rocprofiler_callback_tracing_record_t &record) {
    // its offset in the call's arguments, found once per operation: iterating
    // stringifies them
    thread_local std::unordered_map<rocprofiler_tracing_operation_t,
                                    std::ptrdiff_t>
        offsets;
    const auto *api =
        static_cast<const rocprofiler_callback_tracing_hip_api_data_t *>(
            record.payload);
    auto offset = offsets.find(record.operation);
    if (offset == offsets.end()) {
      // the iteration hands out copies of the arguments, laid out as the call's
      // arguments, the first at 0
      std::pair<const char *, std::ptrdiff_t> found{nullptr, -1};
      rocprofiler_iterate_callback_tracing_kind_operation_args(
          record,
          [](rocprofiler_callback_tracing_kind_t,
             rocprofiler_tracing_operation_t, uint32_t number,
             const void *const address, int32_t indirection, const char *type,
             const char *, const char *, int32_t, void *data) {
            auto &[first, at] =
                *static_cast<std::pair<const char *, std::ptrdiff_t> *>(data);
            if (number == 0)
              first = static_cast<const char *>(address);
            auto distance =
                first ? static_cast<const char *>(address) - first : -1;
            if (at < 0 && indirection == 1 &&
                std::string_view(type).find("ihipStream_t") !=
                    std::string_view::npos &&
                distance >= 0 &&
                distance + sizeof(hipStream_t) <=
                    sizeof(rocprofiler_hip_api_args_t)) {
              at = distance;
            }
            return 0;
          },
          0, &found);
      offset = offsets.emplace(record.operation, found.second).first;
    }
    if (offset->second < 0)
      return std::nullopt;
    hipStream_t stream;
    std::memcpy(&stream,
                reinterpret_cast<const char *>(&api->args) + offset->second,
                sizeof(stream));
    return stream;
  }

  static void mark(hipStream_t stream,
                   const rocprofiler_correlation_id_t &call) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    const hipGraphNode_t *deps = nullptr;
    size_t n_deps = 0;
    if (hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, &deps,
                                   &n_deps) != hipSuccess ||
        status != hipStreamCaptureStatusActive) {
      return;
    }
    auto capture =
        std::find_if(t_captures.begin(), t_captures.end(),
                     [graph](const Capture &c) { return c.graph == graph; });
    if (capture == t_captures.end())
      return; // begun on another thread
    std::vector<hipGraphNode_t> leaves(deps, deps + n_deps);
    auto &last = capture->leaves[stream];
    if (leaves != last) {
      capture->marks.push_back({leaves, call.internal, call.external.value});
      last = std::move(leaves);
    }
  }

  static int on_dispatch(rocprofiler_thread_id_t, rocprofiler_context_id_t,
                         rocprofiler_external_correlation_id_request_kind_t,
                         rocprofiler_tracing_operation_t, uint64_t,
                         rocprofiler_user_data_t *external, void *) {
    if (t_launches.empty() || t_launches.back().id == 0) {
      return 1; // not a traced graph launch's
    }
    auto &launch = t_launches.back();
    external->value = (launch.id << kOrdinalBits) | launch.next_ordinal++;
    return 0;
  }

  void launch(hipGraphExec_t exec) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (first_launch_ == 0) {
      t_launches.push_back(
          {0, 0}); // before the first start(): nothing to attribute
      return;
    }
    auto plan = plans_.find(exec);
    launched_[next_launch_] = plan == plans_.end() ? nullptr : plan->second;
    t_launches.push_back({next_launch_++, 0});
  }

  // a graph's nodes in the order they were added
  static std::vector<hipGraphNode_t> graph_nodes(hipGraph_t graph) {
    size_t count = 0;
    if (!graph || hipGraphGetNodes(graph, nullptr, &count) != hipSuccess)
      return {};
    std::vector<hipGraphNode_t> nodes(count);
    if (count && hipGraphGetNodes(graph, nodes.data(), &count) != hipSuccess)
      return {};
    nodes.resize(count);
    return nodes;
  }

  static void begin_capture(hipStream_t stream) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    if (hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, nullptr,
                                   nullptr) != hipSuccess ||
        status != hipStreamCaptureStatusActive) {
      return;
    }
    // a stream captured again, or a capture that ended elsewhere, leaves a
    // stale one
    t_captures.erase(std::remove_if(t_captures.begin(), t_captures.end(),
                                    [stream, graph](const Capture &c) {
                                      return c.stream == stream ||
                                             c.graph == graph;
                                    }),
                     t_captures.end());
    // a graph captured into keeps the nodes it had before uncaptured
    t_captures.push_back({stream, graph, graph_nodes(graph).size()});
  }

  // the graph's nodes in creation order, each the call's whose mark first
  // reaches it: a call creates its nodes one after another, and its stream's
  // newest dependency is the last of them
  void end_capture(hipStream_t stream, hipGraph_t graph) {
    auto capture =
        std::find_if(t_captures.begin(), t_captures.end(),
                     [stream](const Capture &c) { return c.stream == stream; });
    if (capture == t_captures.end())
      return;
    if (graph) {
      auto nodes = graph_nodes(graph);
      std::unordered_map<hipGraphNode_t, size_t> created;
      for (size_t i = 0; i < nodes.size(); ++i)
        created.emplace(nodes[i], i);
      std::vector<Captured> captured;
      size_t next = capture->known;
      for (const auto &mark : capture->marks) {
        size_t end = 0; // past the mark's newest leaf
        for (auto leaf : mark.leaves) {
          if (auto found = created.find(leaf); found != created.end())
            end = std::max(end, found->second + 1);
        }
        for (; next < end; ++next)
          captured.push_back({nodes[next], mark.call, mark.scope});
      }
      std::lock_guard<std::mutex> lock(mutex_);
      captured_[graph] = std::move(captured);
    }
    t_captures.erase(capture);
  }

  // A graph's nodes in the order the HIP runtime enqueues them: depth-first
  // from each root in creation order, children in creation order, a node once
  // all its parents are. A graph captured from one stream is a chain; across
  // streams the runtime enqueues a whole branch before the next. Empty if
  // unreadable.
  static std::vector<hipGraphNode_t> enqueue_order(hipGraph_t graph) {
    auto nodes = graph_nodes(graph);
    size_t n_edges = 0;
    if (nodes.empty() ||
        hipGraphGetEdges(graph, nullptr, nullptr, &n_edges) != hipSuccess)
      return {};
    std::vector<hipGraphNode_t> from(n_edges), to(n_edges);
    if (n_edges &&
        hipGraphGetEdges(graph, from.data(), to.data(), &n_edges) != hipSuccess)
      return {};

    std::unordered_map<hipGraphNode_t, size_t> index;
    for (size_t i = 0; i < nodes.size(); ++i)
      index.emplace(nodes[i], i);
    std::vector<std::vector<size_t>> parents(nodes.size()),
        children(nodes.size());
    for (size_t e = 0; e < std::min(n_edges, from.size()); ++e) {
      auto f = index.find(from[e]), t = index.find(to[e]);
      if (f == index.end() || t == index.end())
        continue;
      parents[t->second].push_back(f->second);
      children[f->second].push_back(t->second);
    }
    for (auto &c : children)
      std::sort(c.begin(), c.end());

    std::vector<bool> done(nodes.size());
    auto ready = [&](size_t j) {
      return !done[j] && std::all_of(parents[j].begin(), parents[j].end(),
                                     [&](size_t p) { return done[p]; });
    };
    std::vector<hipGraphNode_t> order;
    std::vector<std::pair<size_t, size_t>> stack; // node, its next child
    for (size_t root = 0; root < nodes.size(); ++root) {
      if (!ready(root))
        continue;
      done[root] = true;
      order.push_back(nodes[root]);
      stack.emplace_back(root, 0);
      while (!stack.empty()) {
        auto [node, next] = stack.back();
        if (next >= children[node].size()) {
          stack.pop_back();
          continue;
        }
        ++stack.back().second;
        size_t child = children[node][next];
        if (!ready(child))
          continue;
        done[child] = true;
        order.push_back(nodes[child]);
        stack.emplace_back(child, 0);
      }
    }
    return order.size() == nodes.size() ? order : std::vector<hipGraphNode_t>{};
  }

  // at instantiation: the captured nodes of the source graph in enqueue order,
  // kernels named as dispatches are
  void plan(hipGraphExec_t exec, hipGraph_t graph) {
    std::vector<Captured> captured;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto found = captured_.find(graph);
      if (found == captured_.end())
        return;
      captured = found->second;
    }
    auto order = enqueue_order(graph);
    std::unordered_map<hipGraphNode_t, size_t> rank;
    for (size_t i = 0; i < order.size(); ++i)
      rank.emplace(order[i], i);
    std::stable_sort(captured.begin(), captured.end(),
                     [&rank](const Captured &l, const Captured &r) {
                       auto a = rank.find(l.node), b = rank.find(r.node);
                       return (a == rank.end() ? rank.size() : a->second) <
                              (b == rank.end() ? rank.size() : b->second);
                     });

    auto plan = std::make_shared<Plan>();
    // a node not captured (added by hand, or a child graph's) may dispatch a
    // blit no captured copy owns
    std::unordered_set<hipGraphNode_t> owned;
    for (const auto &c : captured)
      owned.insert(c.node);
    plan->copies = !order.empty() && std::all_of(order.begin(), order.end(),
                                                 [&owned](hipGraphNode_t node) {
                                                   return owned.count(node);
                                                 });
    plan->kernels = order.empty() ? -1 : 0;
    for (auto node : order) {
      hipGraphNodeType type = hipGraphNodeTypeEmpty;
      plan->kernels += hipGraphNodeGetType(node, &type) == hipSuccess &&
                       type == hipGraphNodeTypeKernel;
    }
    for (const auto &c : captured) {
      hipGraphNodeType type = hipGraphNodeTypeEmpty;
      if (hipGraphNodeGetType(c.node, &type) != hipSuccess)
        continue;
      if (type == hipGraphNodeTypeMemcpy || type == hipGraphNodeTypeMemset) {
        if (plan->copies)
          plan->nodes.push_back({"", c.call, c.scope});
        continue;
      }
      hipKernelNodeParams params{};
      if (type != hipGraphNodeTypeKernel ||
          hipGraphKernelNodeGetParams(c.node, &params) != hipSuccess)
        continue;
      // a kernel node holds a registered host function, or a module launch's
      // hipFunction_t
      const char *name = hipKernelNameRefByPtr(params.func, nullptr);
      if ((!name || !*name) && params.func)
        name = hipKernelNameRef(static_cast<hipFunction_t>(params.func));
      if (!name || !*name)
        continue; // a node with no name cannot be matched
      plan->nodes.push_back({name, c.call, c.scope});
    }
    std::lock_guard<std::mutex> lock(mutex_);
    plans_[exec] = std::move(plan);
  }

  static void collect(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                      rocprofiler_record_header_t **headers, size_t n_headers,
                      void *data, uint64_t drop_count) {
    auto &tracer = *static_cast<KernelTracer *>(data);
    std::lock_guard<std::mutex> lock(tracer.mutex_);
    tracer.dropped_ += drop_count;
    if (tracer.first_launch_ == 0)
      return; // before the first start()
    for (size_t i = 0; i < n_headers; ++i) {
      const auto *header = headers[i];
      if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING)
        continue;
      if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
        const auto *record = static_cast<
            const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
            header->payload);
        // a record names its launch however its kernels interleave or however
        // late it arrives
        const uint64_t external = record->correlation_id.external.value;
        const uint64_t launch = external >> kOrdinalBits;
        if (launch && launch < tracer.first_launch_)
          continue; // an earlier window's
        tracer.window_.push_back(
            {record->correlation_id.internal, external,
             record->dispatch_info.dispatch_id, record->dispatch_info.kernel_id,
             record->start_timestamp, record->end_timestamp,
             record->dispatch_info.queue_id.handle,
             ROCPROFILER_MEMORY_COPY_NONE, 0});
      } else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
        const auto *record = static_cast<
            const rocprofiler_buffer_tracing_memory_copy_record_t *>(
            header->payload);
        tracer.window_.push_back(
            {record->correlation_id.internal, 0, 0, 0, record->start_timestamp,
             record->end_timestamp, 0, record->operation, record->bytes});
      }
    }
  }

  rocprofiler_context_id_t captures_{0};
  rocprofiler_context_id_t dispatches_{0};
  rocprofiler_buffer_id_t buffer_{0};
  std::mutex mutex_;
  std::unordered_map<uint64_t, std::string> names_; // kernel id -> kernel name
  std::unordered_map<hipGraph_t, std::vector<Captured>> captured_;
  std::unordered_map<hipGraphExec_t, std::shared_ptr<const Plan>> plans_;
  std::unordered_map<uint64_t, std::shared_ptr<const Plan>>
      launched_; // the window's launches
  std::vector<Dispatch> window_;
  std::vector<Call> calls_;
  std::atomic<bool> timing_{false}; // HIP calls, for torch.profiler
  bool probed_ = false;
  uint64_t dropped_ = 0;
  uint64_t next_launch_ = 1;
  uint64_t first_launch_ =
      0; // the window's first launch; 0 before the first start()
};

} // namespace

void start_profile() { KernelTracer::instance().start(true); }

Profile stop_profile() {
  auto &tracer = KernelTracer::instance();
  tracer.stop();
  return tracer.profile();
}

void push_op(uint64_t correlation) { t_ops.push_back(correlation); }

void push_scope(uint64_t correlation) {
  KernelTracer::instance().push(correlation);
}

void pop_scope() { KernelTracer::instance().pop(); }

void pop_op() {
  if (!t_ops.empty())
    t_ops.pop_back();
}

std::string hip_api_name(rocprofiler_tracing_operation_t op) {
  const char *name = nullptr;
  uint64_t length = 0;
  rocprofiler_query_callback_tracing_kind_operation_name(
      ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, op, &name, &length);
  return name ? std::string(name, length) : "<unknown HIP call>";
}

} // namespace aklr::kernel_tracer

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("rocprofiler_sdk_version") = pybind11::make_tuple(
      ROCPROFILER_SDK_VERSION_MAJOR, ROCPROFILER_SDK_VERSION_MINOR,
      ROCPROFILER_SDK_VERSION_PATCH);
  // rocprofiler-sdk takes a tool only before the ROCm runtime initializes
  aklr::kernel_tracer::KernelTracer::instance().configure();
  aklr::kernel_tracer::register_kineto_profiler();
}
