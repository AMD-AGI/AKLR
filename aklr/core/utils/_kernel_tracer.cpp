/***************************************************************************************************
 * SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 **************************************************************************************************/

// Kernel dispatch timestamps from rocprofiler-sdk, each replayed graph dispatch
// bound to the graph node, HIP call and scope that captured it, with the HIP
// calls and copies, collected as Kineto's rocprofiler-sdk backend
// (pytorch/kineto RocprofLogger, BSD-3-Clause) collects them, for
// torch.profiler runs that opt in (_kineto_profiler.cpp). The tool attaches to
// the process at the first run, through rocprofiler-register, which the process
// loads rocprofiler-sdk's attach library for at startup (ROCP_TOOL_ATTACH=1).
//
// A graph exec dispatches each kernel, copy and memset node from a kernel
// packet the HIP runtime forms at instantiation, its kernel arguments at an
// address fixed for the exec's life. The tool reads every packet from an HSA
// queue interceptor, and binds each address to its node by launching the exec
// with chosen nodes enabled, every kernel packet of those launches swallowed.
#include "_kernel_tracer.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>
#include <sys/syscall.h>
#include <unistd.h>
// ROCr's tool table, the only holder of hsa_amd_queue_intercept_register; its
// includes resolve beside it only in this mode
#define AMD_INTERNAL_BUILD
#include <hsa/hsa_api_trace.h>
#undef AMD_INTERNAL_BUILD

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
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// shortcut: rebuilds in a tool what the HIP runtime and rocprofiler-sdk 1.3.5
// lack; retire it when they record each dispatch's graph node, as CUPTI does
// (ROCm/rocm-systems #14204 and #14211 infer it from dispatch order)
namespace aklr::kernel_tracer {
namespace {

// a dispatch's external correlation id: its window above, the plan and node
// that dispatched it below
constexpr int kNodeBits = 24;
constexpr int kTagBits = 48;
constexpr uint64_t kNodeMask = (1ull << kNodeBits) - 1;
constexpr uint64_t kPlanMask = (1ull << (kTagBits - kNodeBits)) - 1;

// a node of a graph exec, and the HIP call and scope a capture made it in
struct Node {
  uint64_t call = 0;
  uint64_t scope = 0;
  bool captured = false; // a capture made it; one added by hand has no call
};

// what a graph exec dispatches: each kernel packet's kernel argument address,
// bound to its node by calibration
struct Plan {
  uint64_t serial = 0;
  std::vector<hipGraphNode_t> nodes; // the source graph's, top level
  std::vector<Node> info;
  std::unordered_map<hipGraphNode_t, size_t> index;
  std::vector<bool> enabled;     // as hipGraphNodeSetEnabled last left them
  std::vector<uint32_t> packets; // kernel packets per launch, per node
  std::unordered_map<uint64_t, uint32_t> slots; // kernel argument address ->
                                                // node
  std::string unattributable; // why its dispatches cannot be bound
};

struct Captured {
  hipGraphNode_t node;
  uint64_t call;
  uint64_t scope;
};

// a stream capture: the nodes each HIP call added to its graph
struct Capture {
  hipStream_t stream = nullptr;             // the stream that began it
  std::thread::id thread;                   // the thread that began it
  std::unordered_set<hipGraphNode_t> known; // the graph's nodes so far owned
  std::unordered_map<hipStream_t, std::vector<hipGraphNode_t>>
      leaves; // each capturing stream's dependencies, last seen
  std::vector<Captured> captured;
};

// runs work as its scope ends, however it ends
template <typename Work> struct AtExit {
  Work work;
  ~AtExit() { work(); }
};
template <typename Work> AtExit(Work) -> AtExit<Work>;

// marks a thread inside the tool for a callback's span
struct Entered {
  explicit Entered(bool &flag) : flag(flag) { flag = true; }
  ~Entered() { flag = false; }
  bool &flag;
};

// the kernel packets of a batch the interceptor hands rocprofiler-sdk, each
// with its plan and node, which on_dispatch takes in order
struct Batch {
  const std::vector<uint64_t> *tags = nullptr;
  size_t next = 0;
};

thread_local bool t_in_tool =
    false; // the tool's own HIP queries reach its HIP API callback too
thread_local bool t_in_hsa = false; // and its HSA calls its HSA API callback
thread_local bool t_calibrating = false; // the slots it reserves are swallowed
thread_local Batch t_batch;
// the interceptor's per-thread buffers, never destroyed, as below
thread_local auto *t_tags = new std::vector<uint64_t>();
thread_local auto *t_rewritten =
    new std::vector<hsa_kernel_dispatch_packet_t>();

// torch.profiler's op correlation ids, of this thread. Whatever a callback
// reaches is never destroyed, here and below: the runtime still submits work,
// and so calls the tool, as the process exits
std::vector<uint64_t> &ops() {
  thread_local auto *ids = new std::vector<uint64_t>();
  return *ids;
}

// a kernel dispatch packet's kernel argument address; nullopt for any other
// packet. AMD's extended dispatch packet lays it out as the standard one does
std::optional<uint64_t> kernarg_of(const hsa_kernel_dispatch_packet_t &packet) {
  const auto type = (packet.header >> HSA_PACKET_HEADER_TYPE) &
                    ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
  if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH ||
      (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
       reinterpret_cast<const hsa_amd_ext_kernel_dispatch_packet_t &>(packet)
               .amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)) {
    return reinterpret_cast<uint64_t>(packet.kernarg_address);
  }
  return std::nullopt;
}

class KernelTracer {
public:
  // reaches the process's one tool: rocprofiler_configure's callback carries
  // no pointer to it
  static KernelTracer &instance() {
    static auto *tracer = new KernelTracer();
    return *tracer;
  }

  static rocprofiler_tool_configure_result_t *
  tool(uint32_t, const char *, uint32_t, rocprofiler_client_id_t *id) {
    id->name = "aklr-kernel-tracer";
    rocprofiler_at_intercept_table_registration(
        &KernelTracer::on_hsa_table, ROCPROFILER_HSA_TABLE, &instance());
    static rocprofiler_tool_configure_result_t result{
        sizeof(rocprofiler_tool_configure_result_t), &KernelTracer::initialize,
        nullptr, &instance()};
    return &result;
  }

  // loads this library as rocprofiler-sdk's tool, which configures it; the
  // runtime configured it already if it found the library loaded as it started
  void attach() {
    if (dispatches_.handle)
      return;
    void *reg = dlopen("librocprofiler-register.so.0", RTLD_NOW | RTLD_NOLOAD);
    auto *attach = reg ? reinterpret_cast<int (*)(const char *, const char *)>(
                             dlsym(reg, "rocprofiler_register_attach"))
                       : nullptr;
    if (!attach)
      throw std::runtime_error("aklr kernel tracer: the ROCm runtime does not "
                               "load rocprofiler-register");
    Dl_info self{};
    dladdr(reinterpret_cast<void *>(&KernelTracer::tool), &self);
    // the attach exports these; a child inheriting them loads this tool at
    // startup
    std::vector<std::pair<const char *, std::optional<std::string>>> inherited;
    for (const char *name :
         {"ROCP_TOOL_LIBRARIES", "ROCPROFILER_REGISTER_TOOL_ATTACHED"}) {
      const char *value = std::getenv(name);
      inherited.emplace_back(name, value ? std::optional<std::string>(value)
                                         : std::nullopt);
    }
    const uint32_t no_variables = 0;
    const int status =
        attach(reinterpret_cast<const char *>(&no_variables), self.dli_fname);
    for (const auto &[name, value] : inherited) {
      value ? setenv(name, value->c_str(), 1) : unsetenv(name);
    }
    if (status == 9) { // ROCP_REG_ATTACHMENT_NOT_AVAILABLE
      throw std::runtime_error(
          "aklr kernel tracer: start the process with ROCP_TOOL_ATTACH=1 "
          "(import aklr.core.utils.testing before its first GPU call)");
    }
    if (status || dispatches_.handle == 0) {
      throw std::runtime_error(
          "aklr kernel tracer: rocprofiler-register attach failed: " +
          std::to_string(status));
    }
    if (!intercept_register_) {
      throw std::runtime_error("aklr kernel tracer: rocprofiler-sdk gave no "
                               "HSA table to intercept queues with");
    }
  }

  // a window records every dispatch and HIP call; between windows the tool
  // sees only code objects load and queues come into use
  void start() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      window_.clear();
      calls_.clear();
      launched_.clear();
      unattributable_.clear();
      dropped_ = 0;
      // an exec or graph may have changed while the HIP calls went unseen
      plans_.clear();
      planned_.clear();
      captured_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(fault_mutex_);
      fault_.clear();
    }
    {
      std::unique_lock<std::shared_mutex> lock(slots_mutex_);
      slots_.clear();
      planned_slots_ = 0;
    }
    enqueued_ = 0;
    misaligned_ = false;
    window_serial_ = window_serial_ % 0xffff + 1;
    check(rocprofiler_start_context(hip_), "start");
    check(rocprofiler_start_context(dispatches_), "start");
    timing_ = true;
  }

  // the HIP calls a thread makes between push and pop capture their graph nodes
  // into this scope
  void push(uint64_t scope) {
    check(rocprofiler_push_external_correlation_id(hip_, thread_id(),
                                                   {.value = scope}),
          "push a scope");
  }

  void pop() {
    rocprofiler_user_data_t scope{};
    check(rocprofiler_pop_external_correlation_id(hip_, thread_id(), &scope),
          "pop a scope");
  }

  void stop() {
    check(rocprofiler_stop_context(dispatches_), "stop");
    check(rocprofiler_stop_context(hip_), "stop");
    timing_ = false;
  }

  // torch.profiler's window, after stop()
  Profile profile() {
    await_dispatches();
    std::lock_guard<std::mutex> lock(mutex_);
    Profile profile;
    profile.attribution = attribute();
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

  // a dispatch's record arrives after the GPU ran it, from rocprofiler-sdk's
  // own thread
  void await_dispatches() {
    for (int wait = 0;; ++wait) {
      check(rocprofiler_flush_buffer(buffer_), "flush");
      std::lock_guard<std::mutex> lock(mutex_);
      if (dropped_) {
        throw std::runtime_error("rocprofiler-sdk dropped " +
                                 std::to_string(dropped_) + " records");
      }
      const auto recorded = static_cast<uint64_t>(
          std::count_if(window_.begin(), window_.end(), [](const Dispatch &d) {
            return d.copy == ROCPROFILER_MEMORY_COPY_NONE;
          }));
      if (recorded == enqueued_)
        return;
      if (wait == 10000) {
        throw std::runtime_error("rocprofiler-sdk recorded " +
                                 std::to_string(recorded) + " of " +
                                 std::to_string(enqueued_.load()) +
                                 " kernel dispatches of the graph replays");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // each window dispatch's plan, node and capturing call, every replayed
  // dispatch of a planned exec bound; the caller holds mutex_
  std::vector<Attribution> attribute() const {
    if (!unattributable_.empty()) {
      throw std::runtime_error("cannot attribute the replays of a graph with " +
                               unattributable_);
    }
    {
      std::lock_guard<std::mutex> lock(fault_mutex_);
      if (!fault_.empty())
        throw std::runtime_error("the kernel tracer failed: " + fault_);
    }
    if (unrestored_) {
      throw std::runtime_error("calibration could not restore a graph node's "
                               "enabled state");
    }
    if (misaligned_) {
      throw std::runtime_error("rocprofiler-sdk took the kernel packets of the "
                               "graph replays out of order");
    }
    std::vector<Attribution> attribution(window_.size());
    std::map<uint64_t, uint64_t> bound; // plan -> its window dispatches
    for (size_t i = 0; i < window_.size(); ++i) {
      const uint64_t tag = window_[i].tag;
      if (!tag)
        continue;
      const uint64_t serial = tag >> kNodeBits, node = tag & kNodeMask;
      auto plan = planned_.find(serial);
      if (plan == planned_.end() || node >= plan->second->info.size()) {
        throw std::runtime_error(
            "a dispatch of the graph replays matched no graph node");
      }
      const auto &info = plan->second->info[node];
      if (info.captured && !info.call) {
        throw std::runtime_error("a dispatch of the graph replays matched a "
                                 "node no traced HIP call captured");
      }
      attribution[i] = {static_cast<int64_t>(serial),
                        static_cast<int64_t>(node), info.call, info.scope};
      ++bound[serial];
    }
    for (const auto &[serial, expected] : launched_) {
      if (bound[serial] != expected) {
        throw std::runtime_error(std::to_string(bound[serial]) + " of " +
                                 std::to_string(expected) +
                                 " dispatches of the graph replays bound to "
                                 "their graph nodes");
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

  static void on_hsa_table(rocprofiler_intercept_table_t type, uint64_t,
                           uint64_t, void **tables, uint64_t count,
                           void *data) {
    if (type != ROCPROFILER_HSA_TABLE || count == 0)
      return;
    const auto *table = static_cast<const HsaApiTable *>(tables[0]);
    static_cast<KernelTracer *>(data)->intercept_register_ =
        table->amd_ext_->hsa_amd_queue_intercept_register_fn;
  }

  static int initialize(rocprofiler_client_finalize_t, void *tool_data) {
    auto &tracer = *static_cast<KernelTracer *>(tool_data);
    rocprofiler_tracing_operation_t symbol =
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    rocprofiler_external_correlation_id_request_kind_t requests[] = {
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY};
    // the calls that create a queue, reserve its slots or destroy it
    std::vector<rocprofiler_tracing_operation_t> core = {
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_create,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_destroy,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_scacq_screl,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_scacquire,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_relaxed,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_screlease,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_scacq_screl,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_scacquire,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_relaxed,
        ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_screlease};
    rocprofiler_tracing_operation_t amd_queue_create =
        ROCPROFILER_HSA_AMD_EXT_API_ID_hsa_amd_queue_create;
    // symbols_ and queues_ run from the attach on: rocprofiler-sdk reports a
    // code object loaded while it was stopped never again, and a queue must be
    // intercepted before a window's packets reach it
    if (rocprofiler_create_context(&tracer.symbols_) !=
            ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_create_context(&tracer.queues_) !=
            ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_create_context(&tracer.hip_) !=
            ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_callback_tracing_service(
            tracer.symbols_, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, &symbol,
            1, &KernelTracer::on_symbol,
            &tracer) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_callback_tracing_service(
            tracer.queues_, ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
            core.data(), core.size(), &KernelTracer::on_queue,
            &tracer) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_callback_tracing_service(
            tracer.queues_, ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
            &amd_queue_create, 1, &KernelTracer::on_queue,
            &tracer) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_configure_callback_tracing_service(
            tracer.hip_, ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, nullptr,
            0, &KernelTracer::on_hip, &tracer) != ROCPROFILER_STATUS_SUCCESS ||
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
            tracer.dispatches_, requests, 2, &KernelTracer::on_dispatch,
            &tracer) != ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_start_context(tracer.symbols_) !=
            ROCPROFILER_STATUS_SUCCESS ||
        rocprofiler_start_context(tracer.queues_) !=
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

  // shortcut: a queue in use before the attach is intercepted when a thread
  // next reserves its slots, racing another thread's doorbell on it (ROCr
  // registers interceptors unlocked); intercept every queue at its creation
  // once the tool loads before the runtime starts
  static void on_queue(rocprofiler_callback_tracing_record_t record,
                       rocprofiler_user_data_t *, void *data) {
    if (t_in_hsa)
      return;
    Entered entered(t_in_hsa);
    auto &tracer = *static_cast<KernelTracer *>(data);
    tracer.guarded([&] { tracer.queue_call(record); });
  }

  void queue_call(const rocprofiler_callback_tracing_record_t &record) {
    auto &tracer = *this;
    const auto *api =
        static_cast<const rocprofiler_callback_tracing_hsa_api_data_t *>(
            record.payload);
    const auto &args = api->args;
    const bool enter = record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER;
    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API) {
      if (!enter && api->retval.hsa_status_t_retval == HSA_STATUS_SUCCESS) {
        const auto &create = args.hsa_amd_queue_create;
        for (uint32_t i = 0; i < create.num_descs; ++i) {
          if (create.descs[i].queue &&
              create.descs[i].engine_type == HSA_AMD_QUEUE_ENGINE_COMPUTE)
            tracer.intercept(create.descs[i].queue);
        }
      }
      return;
    }
    switch (record.operation) {
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_create:
      if (!enter && api->retval.hsa_status_t_retval == HSA_STATUS_SUCCESS)
        tracer.intercept(*args.hsa_queue_create.queue);
      break;
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_destroy:
      if (enter)
        tracer.forget(args.hsa_queue_destroy.queue);
      break;
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_scacq_screl:
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_scacquire:
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_relaxed:
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_add_write_index_screlease: {
      // the four take the same arguments
      const auto &add = args.hsa_queue_add_write_index_relaxed;
      if (enter) {
        tracer.intercept(add.queue);
      } else if (t_calibrating) {
        tracer.reserve(add.queue, api->retval.uint64_t_retval, add.value);
      }
      break;
    }
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_scacq_screl:
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_scacquire:
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_relaxed:
    case ROCPROFILER_HSA_CORE_API_ID_hsa_queue_cas_write_index_screlease: {
      const auto &cas = args.hsa_queue_cas_write_index_relaxed;
      if (enter) {
        tracer.intercept(cas.queue);
      } else if (t_calibrating && api->retval.uint64_t_retval == cas.expected &&
                 cas.value > cas.expected) {
        tracer.reserve(cas.queue, cas.expected, cas.value - cas.expected);
      }
      break;
    }
    default:
      break;
    }
  }

  void intercept(const hsa_queue_t *queue) {
    if (!queue || !intercept_register_)
      return;
    std::lock_guard<std::mutex> lock(queues_mutex_);
    if (!intercepted_.insert(queue).second)
      return;
    auto *q = const_cast<hsa_queue_t *>(queue);
    if (intercept_register_(q, &KernelTracer::on_packets, q) !=
        HSA_STATUS_SUCCESS) {
      unintercepted_ = true; // not a ROCr intercept queue
    }
  }

  void forget(const hsa_queue_t *queue) {
    std::lock_guard<std::mutex> lock(queues_mutex_);
    intercepted_.erase(queue);
  }

  void reserve(const hsa_queue_t *queue, uint64_t first, uint64_t count) {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    reserved_.emplace_back(queue, first, first + count);
  }

  // ROCr hands each batch of packets written to an intercepted queue here
  // first, on the thread that rang its doorbell, before rocprofiler-sdk's own
  // interceptor; the batch's slots begin at first
  static void on_packets(const void *data, uint64_t count, uint64_t first,
                         void *queue,
                         hsa_amd_queue_intercept_packet_writer writer) {
    auto &tracer = instance();
    const bool calibrating = tracer.calibrating_.load();
    if (!calibrating && tracer.planned_slots_.load() == 0) {
      writer(data, count);
      return;
    }
    const auto *packets =
        static_cast<const hsa_kernel_dispatch_packet_t *>(data);
    bool tagged = false;
    tracer.guarded([&] {
      tracer.read_batch(packets, count, first,
                        static_cast<const hsa_queue_t *>(queue), calibrating);
      tagged = true;
    });
    auto &tags = *t_tags;
    auto &rewritten = *t_rewritten;
    const Batch outer = t_batch;
    t_batch = {tagged ? &tags : nullptr, 0};
    writer(tagged && !rewritten.empty() ? rewritten.data() : data, count);
    // rocprofiler-sdk takes every kernel packet of a batch it traces, or none
    if (tagged && t_batch.next != 0 && t_batch.next != tags.size())
      tracer.misaligned_ = true;
    t_batch = outer;
  }

  // a batch's kernel packets: the calibration's own swallowed into
  // t_rewritten, every other's plan and node into t_tags
  void read_batch(const hsa_kernel_dispatch_packet_t *packets, uint64_t count,
                  uint64_t first, const hsa_queue_t *queue, bool calibrating) {
    auto &tracer = *this;
    auto &tags = *t_tags;
    auto &rewritten = *t_rewritten;
    tags.clear();
    rewritten.clear();
    for (uint64_t i = 0; i < count; ++i) {
      const auto kernarg = kernarg_of(packets[i]);
      if (!kernarg)
        continue;
      if (calibrating && tracer.swallow(queue, first + i, *kernarg)) {
        if (rewritten.empty())
          rewritten.assign(packets, packets + count);
        // a barrier keeps the packet's fences, barrier bit and completion
        // signal, and runs nothing
        const uint16_t header = packets[i].header;
        const hsa_signal_t completion = packets[i].completion_signal;
        auto &barrier =
            reinterpret_cast<hsa_barrier_and_packet_t &>(rewritten[i]);
        barrier = hsa_barrier_and_packet_t{};
        barrier.completion_signal = completion;
        barrier.header = static_cast<uint16_t>(
            (header & ~((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1)) |
            (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE));
        continue;
      }
      tags.push_back(tracer.tag(*kernarg));
    }
  }

  // a kernel packet in a slot the calibrating thread reserved: one of the
  // calibration's own launch
  bool swallow(const hsa_queue_t *queue, uint64_t slot, uint64_t kernarg) {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    for (const auto &[reserved, begin, end] : reserved_) {
      if (reserved == queue && slot >= begin && slot < end) {
        swallowed_.insert(kernarg);
        return true;
      }
    }
    return false;
  }

  uint64_t tag(uint64_t kernarg) const {
    std::shared_lock<std::shared_mutex> lock(slots_mutex_);
    auto slot = slots_.find(kernarg);
    return slot == slots_.end() ? 0 : slot->second;
  }

  static void on_hip(rocprofiler_callback_tracing_record_t record,
                     rocprofiler_user_data_t *user_data, void *data) {
    if (t_in_tool)
      return;
    Entered entered(t_in_tool);
    auto &tracer = *static_cast<KernelTracer *>(data);
    tracer.guarded([&] { tracer.hip_call(record, *user_data); });
  }

  void hip_call(const rocprofiler_callback_tracing_record_t &record,
                rocprofiler_user_data_t &user_data) {
    auto &tracer = *this;
    tracer.time_call(record, user_data);
    const auto *api =
        static_cast<const rocprofiler_callback_tracing_hip_api_data_t *>(
            record.payload);
    const auto &args = api->args;
    const bool ok = api->retval.hipError_t_retval == hipSuccess;
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
      // the runtime frees an exec's kernel arguments with it
      if (record.operation ==
          ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecDestroy)
        tracer.unplan(args.hipGraphExecDestroy.graphExec);
      return;
    }
    switch (record.operation) {
    // a replay owes its nodes' packets once it is launched; one that failed
    // dispatched none
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch_spt:
      if (ok)
        tracer.launch(args.hipGraphLaunch.graphExec);
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCapture_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamBeginCaptureToGraph:
      // the stream is the first argument of all three
      if (ok)
        tracer.begin_capture(args.hipStreamBeginCapture.stream);
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
    // a node added by hand to a graph under capture is its call's; a node
    // destroyed leaves its address to another
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphAddMemFreeNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphAddMemcpyNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipDrvGraphAddMemsetNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddBatchMemOpNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddChildGraphNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddEmptyNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddEventRecordNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddEventWaitNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddExternalSemaphoresSignalNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddExternalSemaphoresWaitNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddHostNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddKernelNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemAllocNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemFreeNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNode1D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNodeFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemcpyNodeToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddMemsetNode:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphAddNode:
      // the new node and its graph are the first two arguments of each
      if (ok && tracer.capturing_.load() &&
          args.hipGraphAddKernelNode.pGraphNode)
        tracer.own(*args.hipGraphAddKernelNode.pGraphNode,
                   args.hipGraphAddKernelNode.graph, record.correlation_id);
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphDestroyNode:
      if (ok)
        tracer.disown(args.hipGraphDestroyNode.node);
      break;
    // an exec updated from another graph runs that graph's nodes, which no
    // capture of the exec's own graph owns
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecUpdate:
      if (ok)
        tracer.replan(args.hipGraphExecUpdate.hGraphExec,
                      "an exec updated from another graph");
      break;
    // a changed exec may dispatch from other kernel arguments: bound again
    // before its caller can launch it
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecKernelNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecChildGraphNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParams:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParams1D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParamsFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemcpyNodeSetParamsToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphExecMemsetNodeSetParams:
      if (ok) // a failed change leaves the exec as it was
        // the exec is the first argument of each
        tracer.replan(args.hipGraphExecDestroy.graphExec);
      break;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphNodeSetEnabled:
      if (ok) {
        tracer.enable(args.hipGraphNodeSetEnabled.hGraphExec,
                      args.hipGraphNodeSetEnabled.hNode,
                      args.hipGraphNodeSetEnabled.isEnabled != 0);
      }
      break;
    default:
      break;
    }
    // the nodes a call adds to a capture are the call's, in its caller's scope
    if (tracer.capturing_.load())
      tracer.mark(stream_of(record), record.correlation_id);
  }

  // runs a callback's work: an exception escaping into the runtime would end
  // the process, so it fails the window instead
  template <typename Work> void guarded(Work &&work) noexcept {
    try {
      work();
    } catch (const std::exception &error) {
      fault(error.what());
    } catch (...) {
      fault("an unknown error");
    }
  }

  void fault(const char *what) noexcept {
    try {
      std::lock_guard<std::mutex> lock(fault_mutex_);
      if (fault_.empty())
        fault_ = what;
    } catch (...) {
      unrestored_ = true; // nothing else left to tell it with
    }
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
              ops().empty() ? 0 : ops().back(),
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
    thread_local auto *kept_offsets =
        new std::unordered_map<rocprofiler_tracing_operation_t,
                               std::ptrdiff_t>();
    auto &offsets = *kept_offsets;
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

  // shortcut: in a capture several threads feed at once, a node whose call has
  // not returned yet goes to the first call after it that reaches it; mark at
  // node creation once the runtime reports it
  // a call owns the nodes its capturing stream's dependencies gained, and
  // every node behind them not yet owned: a call's nodes depend on its
  // stream's earlier ones, or on nodes other calls own. A call whose arguments
  // show no stream, as a launch that carries it in a struct, is matched
  // against every stream the captures its thread began have reached
  void mark(std::optional<hipStream_t> stream,
            const rocprofiler_correlation_id_t &call) {
    std::lock_guard<std::mutex> lock(captures_mutex_);
    if (stream) {
      mark_stream(*stream, call);
      return;
    }
    std::vector<hipStream_t> streams;
    for (const auto &[graph, capture] : captures_) {
      if (capture.thread != std::this_thread::get_id())
        continue;
      for (const auto &[reached, leaves] : capture.leaves) {
        if (std::find(streams.begin(), streams.end(), reached) == streams.end())
          streams.push_back(reached);
      }
    }
    for (auto reached : streams)
      mark_stream(reached, call);
  }

  // the caller holds captures_mutex_
  void mark_stream(hipStream_t stream,
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
    auto capture = captures_.find(graph);
    if (capture == captures_.end())
      return; // begun before the window
    std::vector<hipGraphNode_t> leaves(deps, deps + n_deps);
    auto &last = capture->second.leaves[stream];
    if (leaves == last)
      return;
    auto &known = capture->second.known;
    std::vector<hipGraphNode_t> behind;
    for (auto leaf : leaves) {
      if (!known.count(leaf))
        behind.push_back(leaf);
    }
    while (!behind.empty()) {
      auto node = behind.back();
      behind.pop_back();
      if (!known.insert(node).second)
        continue;
      capture->second.captured.push_back(
          {node, call.internal, call.external.value});
      size_t count = 0;
      if (hipGraphNodeGetDependencies(node, nullptr, &count) != hipSuccess)
        continue;
      std::vector<hipGraphNode_t> parents(count);
      if (count && hipGraphNodeGetDependencies(node, parents.data(), &count) ==
                       hipSuccess) {
        for (size_t p = 0; p < count; ++p) {
          if (!known.count(parents[p]))
            behind.push_back(parents[p]);
        }
      }
    }
    last = std::move(leaves);
  }

  // stamps each kernel dispatch the window enqueues with the window, and the
  // plan and node the interceptor bound its packet to, and counts it; a copy
  // with the window alone
  static int
  on_dispatch(rocprofiler_thread_id_t, rocprofiler_context_id_t,
              rocprofiler_external_correlation_id_request_kind_t kind,
              rocprofiler_tracing_operation_t, uint64_t,
              rocprofiler_user_data_t *external, void *data) {
    auto &tracer = *static_cast<KernelTracer *>(data);
    uint64_t tag = 0;
    if (kind == ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH) {
      if (t_batch.tags && t_batch.next < t_batch.tags->size())
        tag = (*t_batch.tags)[t_batch.next++];
      ++tracer.enqueued_;
    }
    external->value =
        (static_cast<uint64_t>(tracer.window_serial_.load()) << kTagBits) | tag;
    return 0;
  }

  // a replay of a planned exec owes each enabled node's packets once; an exec
  // instantiated unseen has no nodes to bind its packets to
  void launch(hipGraphExec_t exec) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = plans_.find(exec);
    if (found == plans_.end()) {
      unattributable_ = "an exec instantiated outside the profiling window";
      return;
    }
    const Plan &plan = *found->second;
    if (!plan.unattributable.empty()) {
      unattributable_ = plan.unattributable;
      return;
    }
    uint64_t expected = 0;
    for (size_t i = 0; i < plan.packets.size(); ++i)
      expected += plan.enabled[i] ? plan.packets[i] : 0;
    launched_[plan.serial] += expected;
  }

  void enable(hipGraphExec_t exec, hipGraphNode_t node, bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = plans_.find(exec);
    if (found == plans_.end())
      return;
    auto index = found->second->index.find(node);
    if (index != found->second->index.end())
      found->second->enabled[index->second] = enabled;
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

  void begin_capture(hipStream_t stream) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    hipGraph_t graph = nullptr;
    if (hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, nullptr,
                                   nullptr) != hipSuccess ||
        status != hipStreamCaptureStatusActive) {
      return;
    }
    // a graph captured into keeps the nodes it had before unowned
    auto before = graph_nodes(graph);
    std::lock_guard<std::mutex> lock(captures_mutex_);
    // a stream captured again, or a capture that ended unseen, leaves a stale
    // one
    for (auto c = captures_.begin(); c != captures_.end();) {
      c = c->second.stream == stream ? captures_.erase(c) : std::next(c);
    }
    auto &capture = captures_[graph];
    capture = Capture{};
    capture.stream = stream;
    capture.thread = std::this_thread::get_id();
    capture.leaves[stream]; // its dependencies, none yet seen
    capture.known.insert(before.begin(), before.end());
    capturing_ = true;
  }

  // a node a call added by hand to a graph under capture is the call's
  void own(hipGraphNode_t node, hipGraph_t graph,
           const rocprofiler_correlation_id_t &call) {
    std::lock_guard<std::mutex> lock(captures_mutex_);
    auto capture = captures_.find(graph);
    if (capture != captures_.end() && capture->second.known.insert(node).second)
      capture->second.captured.push_back(
          {node, call.internal, call.external.value});
  }

  // a destroyed node's address may come back as another node
  void disown(hipGraphNode_t node) {
    {
      std::lock_guard<std::mutex> lock(captures_mutex_);
      for (auto &[graph, capture] : captures_) {
        capture.known.erase(node);
        auto &c = capture.captured;
        c.erase(std::remove_if(
                    c.begin(), c.end(),
                    [node](const Captured &x) { return x.node == node; }),
                c.end());
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &[graph, captured] : captured_) {
      captured.erase(
          std::remove_if(captured.begin(), captured.end(),
                         [node](const Captured &x) { return x.node == node; }),
          captured.end());
    }
  }

  // the nodes of the graph each call owns; any the capture made that no
  // traced call owns has no call
  void end_capture(hipStream_t stream, hipGraph_t graph) {
    Capture capture;
    {
      std::lock_guard<std::mutex> lock(captures_mutex_);
      auto found = std::find_if(
          captures_.begin(), captures_.end(),
          [stream](const auto &c) { return c.second.stream == stream; });
      if (found == captures_.end())
        return;
      capture = std::move(found->second);
      captures_.erase(found);
      capturing_ = !captures_.empty();
    }
    if (!graph)
      return;
    for (auto node : graph_nodes(graph)) {
      if (!capture.known.count(node))
        capture.captured.push_back({node, 0, 0});
    }
    std::lock_guard<std::mutex> lock(mutex_);
    captured_[graph] = std::move(capture.captured);
  }

  // memory of the current device, which the runtime copies and fills with
  // kernels, never by DMA
  static bool on_this_device(const void *ptr) {
    hipPointerAttribute_t attributes{};
    int device = -1;
    if (hipPointerGetAttributes(&attributes, ptr) != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    return hipGetDevice(&device) == hipSuccess &&
           attributes.type == hipMemoryTypeDevice &&
           attributes.device == device;
  }

  // why the graph cannot be calibrated: a node the runtime dispatches anew
  // each launch, may run without a kernel, or that calibration cannot
  // disable; empty if none
  static std::string unattributable(hipGraph_t graph) {
    for (auto node : graph_nodes(graph)) {
      hipGraphNodeType type = hipGraphNodeTypeEmpty;
      if (hipGraphNodeGetType(node, &type) != hipSuccess)
        return "a node of unknown type";
      switch (type) {
      case hipGraphNodeTypeEmpty:
        break;
      case hipGraphNodeTypeKernel: {
        hipKernelNodeAttrValue value{};
        if (hipGraphKernelNodeGetAttribute(
                node, hipKernelNodeAttributeCooperative, &value) != hipSuccess)
          return "a kernel node of unknown launch";
        if (value.cooperative)
          return "a cooperative kernel node, which the runtime dispatches "
                 "anew each launch";
        break;
      }
      case hipGraphNodeTypeMemset: {
        hipMemsetParams params{};
        if (hipGraphMemsetNodeGetParams(node, &params) != hipSuccess ||
            !on_this_device(params.dst))
          return "a memset node of memory off this device";
        break;
      }
      case hipGraphNodeTypeMemcpy: {
        hipMemcpy3DParms params{};
        if (hipGraphMemcpyNodeGetParams(node, &params) != hipSuccess ||
            params.srcArray || params.dstArray ||
            !on_this_device(params.srcPtr.ptr) ||
            !on_this_device(params.dstPtr.ptr))
          return "a memcpy node with an end off this device's memory, which "
                 "the runtime may copy by DMA";
        break;
      }
      case hipGraphNodeTypeGraph:
        return "a child graph node, whose nodes cannot be disabled";
      default:
        return "a node of type " + std::to_string(type) +
               ", whose launch acts beyond kernels";
      }
    }
    return {};
  }

  std::shared_ptr<Plan> new_plan() {
    auto plan = std::make_shared<Plan>();
    std::lock_guard<std::mutex> lock(mutex_);
    plan->serial = next_plan_;
    next_plan_ = next_plan_ % kPlanMask + 1;
    return plan;
  }

  // at instantiation: the exec's nodes, each with the call that captured it,
  // and its kernel packets bound to them
  void plan(hipGraphExec_t exec, hipGraph_t graph) {
    auto plan = new_plan();
    plan->nodes = graph_nodes(graph);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::unordered_map<hipGraphNode_t, const Captured *> captured;
      if (auto found = captured_.find(graph); found != captured_.end()) {
        for (const auto &c : found->second)
          captured.emplace(c.node, &c);
      }
      for (size_t i = 0; i < plan->nodes.size(); ++i) {
        auto c = captured.find(plan->nodes[i]);
        plan->info.push_back(
            c == captured.end()
                ? Node{}
                : Node{c->second->call, c->second->scope, true});
        plan->index.emplace(plan->nodes[i], i);
      }
    }
    plan->unattributable = plan->nodes.size() > kNodeMask
                               ? "more than 2^24 nodes"
                               : unattributable(graph);
    if (plan->unattributable.empty())
      calibrate(*plan, exec);
    install(exec, std::move(plan));
  }

  // after an exec changes: the same nodes, bound to its packets again, unless
  // the change leaves them unattributable
  void replan(hipGraphExec_t exec, const char *unattributable = "") {
    std::shared_ptr<const Plan> old;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto found = plans_.find(exec);
      if (found == plans_.end())
        return;
      old = found->second;
    }
    auto plan = new_plan();
    plan->nodes = old->nodes;
    plan->info = old->info;
    plan->index = old->index;
    plan->unattributable =
        old->unattributable.empty() ? unattributable : old->unattributable;
    if (plan->unattributable.empty())
      calibrate(*plan, exec);
    install(exec, std::move(plan));
  }

  void install(hipGraphExec_t exec, std::shared_ptr<Plan> plan) {
    unplan(exec);
    {
      std::unique_lock<std::shared_mutex> lock(slots_mutex_);
      for (const auto &[kernarg, node] : plan->slots)
        slots_[kernarg] = (plan->serial << kNodeBits) | node;
      planned_slots_ = slots_.size();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    planned_[plan->serial] = plan;
    plans_[exec] = std::move(plan);
  }

  void unplan(hipGraphExec_t exec) {
    std::shared_ptr<Plan> plan;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto found = plans_.find(exec);
      if (found == plans_.end())
        return;
      plan = std::move(found->second);
      plans_.erase(found);
    }
    std::unique_lock<std::shared_mutex> lock(slots_mutex_);
    for (const auto &[kernarg, node] : plan->slots) {
      auto slot = slots_.find(kernarg);
      if (slot != slots_.end() && slot->second >> kNodeBits == plan->serial)
        slots_.erase(slot);
    }
    planned_slots_ = slots_.size();
  }

  // launches the exec once, every kernel packet of the launch swallowed; the
  // packets' kernel arguments
  std::optional<std::unordered_set<uint64_t>>
  calibration_launch(hipGraphExec_t exec, hipStream_t stream) {
    calibrating_ = true;
    t_calibrating = true;
    const bool launched = hipGraphLaunch(exec, stream) == hipSuccess &&
                          hipStreamSynchronize(stream) == hipSuccess;
    t_calibrating = false;
    calibrating_ = false;
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    reserved_.clear();
    auto swallowed = std::exchange(swallowed_, {});
    if (!launched)
      return std::nullopt;
    return swallowed;
  }

  // binds each kernel packet's kernel arguments to its node. The first
  // launch disables every node and swallows nothing: it runs none of the
  // program's work, only what the runtime does at an exec's first launch, as
  // initializing the device heap. Then each kernel, memcpy and memset node is
  // disabled through its own run [l, r) of the launches that follow, every
  // kernel packet of which is swallowed, so the launches that miss a packet
  // name its node. A node toggles twice: the runtime refilters every packet
  // of its segment at each
  void calibrate(Plan &plan, hipGraphExec_t exec) {
    std::lock_guard<std::mutex> serial(calibrate_mutex_);
    {
      std::lock_guard<std::mutex> lock(queues_mutex_);
      if (unintercepted_) {
        plan.unattributable = "queues rocprofiler-sdk does not intercept";
        return;
      }
    }
    int device = 0;
    hipStream_t stream = nullptr;
    if (hipGetDevice(&device) != hipSuccess ||
        !(stream = calibration_stream(device))) {
      plan.unattributable = "no stream to calibrate it on";
      return;
    }
    const size_t n = plan.nodes.size();
    plan.enabled.assign(n, true);
    plan.packets.assign(n, 0);
    std::vector<unsigned> state(n, 1);
    std::vector<size_t> toggled;
    for (size_t i = 0; i < n; ++i) {
      hipGraphNodeType type = hipGraphNodeTypeEmpty;
      (void)hipGraphNodeGetType(plan.nodes[i], &type);
      if (type != hipGraphNodeTypeKernel && type != hipGraphNodeTypeMemcpy &&
          type != hipGraphNodeTypeMemset)
        continue;
      if (hipGraphNodeGetEnabled(exec, plan.nodes[i], &state[i]) !=
          hipSuccess) {
        plan.unattributable = "a node whose enabled state cannot be read";
        return;
      }
      plan.enabled[i] = state[i] != 0;
      toggled.push_back(i);
    }
    auto set = [&](size_t i, unsigned enabled) {
      if (state[i] == enabled)
        return true;
      state[i] = enabled;
      return hipGraphNodeSetEnabled(exec, plan.nodes[i], enabled) == hipSuccess;
    };
    // the program's nodes come back as they were, whatever ends calibration
    AtExit restore{[&] {
      for (size_t i : toggled) {
        if (!set(i, plan.enabled[i] ? 1 : 0))
          unrestored_ = true;
      }
    }};

    // the launches: enough runs [l, r) short of all of them for every node
    size_t launches = 1;
    while (launches * (launches + 1) / 2 - 1 < toggled.size())
      ++launches;
    std::map<std::pair<size_t, size_t>, size_t> owner; // run -> node
    std::vector<std::pair<size_t, size_t>> runs;
    for (size_t l = 0; l < launches && runs.size() < toggled.size(); ++l) {
      for (size_t r = l + 1; r <= launches && runs.size() < toggled.size();
           ++r) {
        if (l == 0 && r == launches)
          continue; // a node every launch misses names no kernel arguments
        owner[{l, r}] = toggled[runs.size()];
        runs.emplace_back(l, r);
      }
    }
    bool ran = true;
    for (size_t i : toggled)
      ran = ran && set(i, 0);
    ran = ran && hipGraphLaunch(exec, stream) == hipSuccess &&
          hipStreamSynchronize(stream) == hipSuccess;
    std::vector<std::unordered_set<uint64_t>> seen;
    for (size_t t = 0; t < launches && ran; ++t) {
      for (size_t k = 0; k < runs.size() && ran; ++k)
        ran = set(toggled[k], t < runs[k].first || t >= runs[k].second);
      auto launched = ran ? calibration_launch(exec, stream) : std::nullopt;
      ran = launched.has_value();
      if (ran)
        seen.push_back(std::move(*launched));
    }
    if (!ran) {
      plan.unattributable = "a launch calibration could not run";
      return;
    }
    std::unordered_set<uint64_t> all;
    for (const auto &launch : seen)
      all.insert(launch.begin(), launch.end());
    for (uint64_t kernarg : all) {
      // the launches that miss it, which must be one run of them
      std::vector<bool> missed(launches);
      for (size_t t = 0; t < launches; ++t)
        missed[t] = !seen[t].count(kernarg);
      const size_t l =
          std::find(missed.begin(), missed.end(), true) - missed.begin();
      const size_t r =
          std::find(missed.begin() + l, missed.end(), false) - missed.begin();
      const bool run =
          std::find(missed.begin() + r, missed.end(), true) == missed.end();
      auto found = owner.find({l, r});
      if (kernarg == 0) {
        plan.unattributable = "a kernel packet with no kernel arguments";
      } else if (l == launches) {
        plan.unattributable = "kernel packets no node accounts for";
      } else if (!run || found == owner.end()) {
        plan.unattributable =
            "kernel packets whose arguments move between launches";
      }
      if (!plan.unattributable.empty()) {
        plan.slots.clear();
        return;
      }
      plan.slots[kernarg] = static_cast<uint32_t>(found->second);
      ++plan.packets[found->second];
    }
  }

  // a stream of the tool's own on the device, so calibration waits on no
  // stream of the program's
  hipStream_t calibration_stream(int device) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (auto found = streams_.find(device); found != streams_.end())
        return found->second;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess)
      return nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    streams_[device] = stream;
    return stream;
  }

  static void collect(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
                      rocprofiler_record_header_t **headers, size_t n_headers,
                      void *data, uint64_t drop_count) {
    auto &tracer = *static_cast<KernelTracer *>(data);
    tracer.guarded([&] { tracer.record(headers, n_headers, drop_count); });
  }

  void record(rocprofiler_record_header_t **headers, size_t n_headers,
              uint64_t drop_count) {
    auto &tracer = *this;
    std::lock_guard<std::mutex> lock(tracer.mutex_);
    tracer.dropped_ += drop_count;
    for (size_t i = 0; i < n_headers; ++i) {
      const auto *header = headers[i];
      if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING)
        continue;
      if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
        const auto *record = static_cast<
            const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(
            header->payload);
        const uint64_t external = record->correlation_id.external.value;
        if (external >> kTagBits != tracer.window_serial_.load())
          continue; // an earlier window's, arriving late
        const auto &info = record->dispatch_info;
        tracer.window_.push_back(
            {record->correlation_id.internal,
             external & ((1ull << kTagBits) - 1),
             info.dispatch_id,
             info.kernel_id,
             record->start_timestamp,
             record->end_timestamp,
             info.queue_id.handle,
             ROCPROFILER_MEMORY_COPY_NONE,
             0,
             {info.grid_size.x, info.grid_size.y, info.grid_size.z},
             {info.workgroup_size.x, info.workgroup_size.y,
              info.workgroup_size.z}});
      } else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
        const auto *record = static_cast<
            const rocprofiler_buffer_tracing_memory_copy_record_t *>(
            header->payload);
        if (record->correlation_id.external.value >> kTagBits !=
            tracer.window_serial_.load())
          continue; // an earlier window's, arriving late
        tracer.window_.push_back({record->correlation_id.internal,
                                  0,
                                  0,
                                  0,
                                  record->start_timestamp,
                                  record->end_timestamp,
                                  0,
                                  record->operation,
                                  record->bytes,
                                  {},
                                  {}});
      }
    }
  }

  rocprofiler_context_id_t symbols_{0};
  rocprofiler_context_id_t queues_{0};
  rocprofiler_context_id_t hip_{0};
  rocprofiler_context_id_t dispatches_{0};
  rocprofiler_buffer_id_t buffer_{0};
  decltype(AmdExtTable::hsa_amd_queue_intercept_register_fn)
      intercept_register_ = nullptr;
  std::mutex mutex_;
  std::unordered_map<uint64_t, std::string> names_; // kernel id -> kernel name
  std::unordered_map<hipGraph_t, std::vector<Captured>> captured_;
  std::mutex captures_mutex_;
  std::unordered_map<hipGraph_t, Capture> captures_; // the captures under way
  std::atomic<bool> capturing_{false};
  std::unordered_map<hipGraphExec_t, std::shared_ptr<Plan>> plans_;
  std::unordered_map<uint64_t, std::shared_ptr<const Plan>>
      planned_; // the window's plans, by serial
  std::map<uint64_t, uint64_t>
      launched_;               // plan -> the dispatches its replays owe
  std::string unattributable_; // why a replayed exec cannot be bound
  std::unordered_map<int, hipStream_t> streams_;
  std::vector<Dispatch> window_;
  std::vector<Call> calls_;
  uint64_t dropped_ = 0;
  uint64_t next_plan_ = 1;
  std::atomic<bool> timing_{false};   // HIP calls, for torch.profiler
  std::atomic<uint64_t> enqueued_{0}; // the window's kernel dispatches
  std::atomic<uint16_t> window_serial_{0};
  std::atomic<bool> misaligned_{false};
  std::atomic<bool> unrestored_{false}; // sticky: the program's graph changed
  mutable std::mutex fault_mutex_;
  std::string fault_; // the window's first error inside a callback
  // kernel argument address -> plan and node, read by every intercepted batch
  mutable std::shared_mutex slots_mutex_;
  std::unordered_map<uint64_t, uint64_t> slots_;
  std::atomic<size_t> planned_slots_{0};
  std::mutex queues_mutex_;
  std::unordered_set<const hsa_queue_t *> intercepted_;
  bool unintercepted_ = false;
  // one calibration at a time; the slots its thread reserves, and the kernel
  // arguments of the packets swallowed from them
  std::mutex calibrate_mutex_;
  std::mutex calibration_mutex_;
  std::atomic<bool> calibrating_{false};
  std::vector<std::tuple<const hsa_queue_t *, uint64_t, uint64_t>> reserved_;
  std::unordered_set<uint64_t> swallowed_;
};

} // namespace

void attach() { KernelTracer::instance().attach(); }

void start_profile() { KernelTracer::instance().start(); }

// a window's error, for check(): one thrown into Kineto's stop would leave
// Kineto unable to start the next window
std::string &failure = *new std::string(), &next_failure = *new std::string();

Profile stop_profile() {
  auto &tracer = KernelTracer::instance();
  try {
    tracer.stop();
    if (!next_failure.empty())
      throw std::runtime_error(std::exchange(next_failure, {}));
    return tracer.profile();
  } catch (const std::exception &error) {
    failure = error.what();
    return {};
  }
}

void push_op(uint64_t correlation) { ops().push_back(correlation); }

void push_scope(uint64_t correlation) {
  KernelTracer::instance().push(correlation);
}

void pop_scope() { KernelTracer::instance().pop(); }

void pop_op() {
  if (!ops().empty())
    ops().pop_back();
}

std::string hip_api_name(rocprofiler_tracing_operation_t op) {
  const char *name = nullptr;
  uint64_t length = 0;
  rocprofiler_query_callback_tracing_kind_operation_name(
      ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, op, &name, &length);
  return name ? std::string(name, length) : "<unknown HIP call>";
}

} // namespace aklr::kernel_tracer

// found by rocprofiler-sdk in the library rocprofiler-register attaches
extern "C" ROCPROFILER_PUBLIC_API rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtime_version,
                      uint32_t priority, rocprofiler_client_id_t *id) {
  return aklr::kernel_tracer::KernelTracer::tool(version, runtime_version,
                                                 priority, id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // WORKAROUND: ROCM-29087, fixed in rocprofiler-sdk 1.4.2 (ROCm 10.2);
  // delete when the images ship it
#if ROCPROFILER_SDK_VERSION < 10402
  // below 1.4.2 the default queue interposition deadlocks a HIP graph whose
  // streams fork and join, and traces no queue made after an attach; set here,
  // as a library loaded before the ROCm runtime initializes is configured then
  setenv("ROCPROFILER_QUEUE_INTERPOSITION", "0",
         1); // the legacy queue intercept does neither
#endif
  m.attr("rocprofiler_sdk_version") = pybind11::make_tuple(
      ROCPROFILER_SDK_VERSION_MAJOR, ROCPROFILER_SDK_VERSION_MINOR,
      ROCPROFILER_SDK_VERSION_PATCH);
  m.attr("hip_version") = pybind11::make_tuple(
      HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH);
  m.def("check", [] {
    if (!aklr::kernel_tracer::failure.empty())
      throw std::runtime_error(std::exchange(aklr::kernel_tracer::failure, {}));
  });
  // for tests: the next window fails with this error
  m.def("fail_next_window", [](std::string error) {
    aklr::kernel_tracer::next_failure = std::move(error);
  });
  aklr::kernel_tracer::register_kineto_profiler();
}
