// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Kernel dispatch timestamps from rocprofiler-sdk, for testing.bench_cudagraph: a context traces
// every kernel dispatch into a lossless buffer while it is started. Each launch between push and
// pop carries its own external correlation id, so a record names the launch it came from, however
// its kernels interleave with the next launch's and however late it arrives; a window (start) keeps
// only its launches' records.
//
// shortcut: stands in for torch.profiler while torch's Kineto traces ROCm with roctracer (torch
// 2.11); torch >= 2.12's Kineto traces with rocprofiler-sdk, and testing.kernel_tracer warns there.

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Record = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>;  // launch, kernel id, start ns, end ns

// The process's one rocprofiler-sdk tool: rocprofiler_force_configure's callback carries no pointer
// to it, so it is reached through instance().
class KernelTracer {
   public:
    static KernelTracer& instance() {
        static KernelTracer tracer;
        return tracer;
    }

    void configure() {
        // WORKAROUND: ROCM-29087, fixed in rocprofiler-sdk 1.4.2 (ROCm 10.2); delete this when the images
        // ship it. Below 1.4.2, the queue interposition rocprofiler-sdk traces through by default deadlocks a
        // HIP graph whose streams fork and join: a barrier waits on a kernel's completion signal that only a
        // free one of 4 busy-waiting handler threads releases. Its legacy queue intercept does not.
#if ROCPROFILER_SDK_VERSION < 10402
        setenv("ROCPROFILER_QUEUE_INTERPOSITION", "0", 1);
#else
        if (PyErr_WarnEx(PyExc_UserWarning,
                         "rocprofiler-sdk " ROCPROFILER_SDK_VERSION_STRING
                         " fixed ROCM-29087: delete the WORKAROUND in aklr/core/utils/_kernel_tracer.cpp",
                         1) < 0) {
            throw pybind11::error_already_set();
        }
#endif
        // rocprofiler-sdk takes a tool only before the ROCm runtime initializes
        check(rocprofiler_force_configure(&KernelTracer::tool), "configure before the first GPU call");
        if (context_.handle == 0) {
            throw std::runtime_error("rocprofiler-sdk: kernel dispatch tracing did not initialize");
        }
    }

    void start() {
        if (context_.handle == 0) {
            throw std::runtime_error("rocprofiler-sdk: configure() the tracer before the first GPU call");
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            records_.clear();
            dropped_ = 0;
            first_launch_ = next_launch_;
        }
        check(rocprofiler_start_context(context_), "start");
    }

    void push() {
        rocprofiler_user_data_t launch{};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            launch.value = next_launch_++;
        }
        check(rocprofiler_push_external_correlation_id(context_, thread_id(), launch), "push a launch");
    }

    void pop() {
        rocprofiler_user_data_t launch{};
        check(rocprofiler_pop_external_correlation_id(context_, thread_id(), &launch), "pop a launch");
    }

    void stop() { check(rocprofiler_stop_context(context_), "stop"); }

    std::vector<Record> records() {
        check(rocprofiler_flush_buffer(buffer_), "flush");
        std::lock_guard<std::mutex> lock(mutex_);
        if (dropped_) {
            throw std::runtime_error("rocprofiler-sdk dropped " + std::to_string(dropped_) + " records");
        }
        return records_;
    }

   private:
    KernelTracer() = default;

    static void check(rocprofiler_status_t status, const char* what) {
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

    static rocprofiler_tool_configure_result_t* tool(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
        id->name = "aklr-kernel-tracer";
        static rocprofiler_tool_configure_result_t result{sizeof(rocprofiler_tool_configure_result_t),
                                                          &KernelTracer::initialize, nullptr, &instance()};
        return &result;
    }

    static int initialize(rocprofiler_client_finalize_t, void* tool_data) {
        auto& tracer = *static_cast<KernelTracer*>(tool_data);
        if (rocprofiler_create_context(&tracer.context_) != ROCPROFILER_STATUS_SUCCESS ||
            rocprofiler_create_buffer(tracer.context_, 1 << 22, 1 << 21, ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                      &KernelTracer::collect, &tracer, &tracer.buffer_) != ROCPROFILER_STATUS_SUCCESS ||
            rocprofiler_configure_buffer_tracing_service(tracer.context_, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                         nullptr, 0, tracer.buffer_) != ROCPROFILER_STATUS_SUCCESS) {
            return -1;
        }
        return 0;
    }

    static void collect(rocprofiler_context_id_t, rocprofiler_buffer_id_t, rocprofiler_record_header_t** headers,
                        size_t n_headers, void* data, uint64_t drop_count) {
        auto& tracer = *static_cast<KernelTracer*>(data);
        std::lock_guard<std::mutex> lock(tracer.mutex_);
        tracer.dropped_ += drop_count;
        for (size_t i = 0; i < n_headers; ++i) {
            const auto* header = headers[i];
            if (header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
                header->kind != ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
                continue;
            }
            const auto* record =
                static_cast<const rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
            const uint64_t launch = record->correlation_id.external.value;
            if (tracer.first_launch_ != 0 && launch >= tracer.first_launch_) {
                tracer.records_.emplace_back(launch - tracer.first_launch_, record->dispatch_info.kernel_id,
                                             record->start_timestamp, record->end_timestamp);
            }
        }
    }

    rocprofiler_context_id_t context_{0};
    rocprofiler_buffer_id_t buffer_{0};
    std::mutex mutex_;
    std::vector<Record> records_;
    uint64_t dropped_ = 0;
    uint64_t next_launch_ = 1;
    uint64_t first_launch_ = 0;  // of the window; 0 before the first
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("configure", [] { KernelTracer::instance().configure(); });
    m.def("start", [] { KernelTracer::instance().start(); });
    m.def("push", [] { KernelTracer::instance().push(); });
    m.def("pop", [] { KernelTracer::instance().pop(); });
    m.def("stop", [] { KernelTracer::instance().stop(); });
    m.def("records", [] { return KernelTracer::instance().records(); });
}
