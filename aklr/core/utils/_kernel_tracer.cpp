/***************************************************************************************************
 * SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
 **************************************************************************************************/

// Kernel dispatch timestamps from rocprofiler-sdk, for testing.bench_cudagraph
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

// shortcut: stands in for torch.profiler while torch's Kineto traces ROCm with roctracer (torch 2.11)
namespace {

using Record = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>;  // launch, kernel id, start ns, end ns

// retire it on torch >= 2.12 with rocprofiler-sdk >= 1.4.2, where testing.kernel_tracer warns
class KernelTracer {
   public:
    // reaches the process's one tool: rocprofiler_force_configure's callback carries no pointer to it
    static KernelTracer& instance() {
        static KernelTracer tracer;
        return tracer;
    }

    void configure() {
        // WORKAROUND: ROCM-29087, fixed in rocprofiler-sdk 1.4.2 (ROCm 10.2); delete when the images ship it
#if ROCPROFILER_SDK_VERSION < 10402
        // below 1.4.2 the default queue interposition deadlocks a HIP graph whose streams fork and join
        setenv("ROCPROFILER_QUEUE_INTERPOSITION", "0", 1);  // the legacy queue intercept does not
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

    // each launch between push and pop carries its own external correlation id
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
            // a record names its launch however its kernels interleave or however late it arrives
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
    uint64_t first_launch_ = 0;  // the window's first launch; 0 before the first start()
};

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("rocprofiler_sdk_version") = pybind11::make_tuple(
        ROCPROFILER_SDK_VERSION_MAJOR, ROCPROFILER_SDK_VERSION_MINOR, ROCPROFILER_SDK_VERSION_PATCH);
    m.def("configure", [] { KernelTracer::instance().configure(); });
    m.def("start", [] { KernelTracer::instance().start(); });
    m.def("push", [] { KernelTracer::instance().push(); });
    m.def("pop", [] { KernelTracer::instance().pop(); });
    m.def("stop", [] { KernelTracer::instance().stop(); });
    m.def("records", [] { return KernelTracer::instance().records(); });
}
