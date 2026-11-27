// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>

struct Dim3 { unsigned x, y, z; };
using Callback = void (*)(const char *, unsigned, unsigned, unsigned,
                         unsigned, unsigned, unsigned, size_t, void **);
static std::mutex lock;
static std::unordered_map<const void *, std::string> symbols;
static std::unordered_map<void *, bool> modules;
static std::unordered_map<const void *, void *> functions;
static std::unordered_map<const void *, bool> symbol_functions;
static std::string target;
static Callback callback = nullptr;

extern "C" void aklr_verify_capture(const char *library, Callback capture) {
    std::lock_guard<std::mutex> guard(lock);
    target = library;
    callback = capture;
}

static bool belongs(const void *address) {
    Dl_info info{};
    if (!address || !dladdr(address, &info) || !info.dli_fname || target.empty()) return false;
    if (target[0] == '/') {
        char *resolved = realpath(info.dli_fname, nullptr);
        bool match = resolved && target == resolved;
        free(resolved);
        return match;
    }
    const char *base = strrchr(info.dli_fname, '/');
    return target == (base ? base + 1 : info.dli_fname);
}

template <typename Function, typename... Args>
static int forward(const char *name, Args... args) {
    auto next = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
    return next ? next(args...) : 801;
}

static bool capture_host(const void *function, Dim3 grid, Dim3 block,
                         size_t shared, void **args, bool unsupported = false) {
    Callback cb;
    std::string name;
    {
        std::lock_guard<std::mutex> guard(lock);
        if (!callback || !belongs(function)) return false;
        cb = callback;
        auto found = symbols.find(function);
        if (!unsupported && found != symbols.end()) name = found->second;
    }
    cb(name.c_str(), grid.x, grid.y, grid.z, block.x, block.y, block.z, shared, args);
    return true;
}

static bool capture_unknown(const void *caller) {
    Callback cb;
    {
        std::lock_guard<std::mutex> guard(lock);
        if (!callback || !belongs(caller)) return false;
        cb = callback;
    }
    cb("", 1, 1, 1, 1, 1, 1, 0, nullptr);
    return true;
}

static bool capture_module(const void *function, const void *caller) {
    Callback cb;
    {
        std::lock_guard<std::mutex> guard(lock);
        if (!callback) return false;
        auto direct = symbol_functions.find(function);
        if (direct != symbol_functions.end() && !direct->second) return false;
        auto found = functions.find(function);
        auto module = found == functions.end() ? modules.end() : modules.find(found->second);
        if (direct == symbol_functions.end()) {
            if (module != modules.end()) {
                if (!module->second) return false;
            } else if (!belongs(caller)) {
                return false;
            }
        }
        cb = callback;
    }
    cb("", 1, 1, 1, 1, 1, 1, 0, nullptr);
    return true;
}

static void remember_module(int status, void **module, const void *caller, const void *image) {
    if (status || !module) return;
    std::lock_guard<std::mutex> guard(lock);
    modules[*module] = belongs(caller) || belongs(image);
}

extern "C" void __hipRegisterFunction(void **modules, const void *host,
                                      char *device, const char *name, unsigned limit,
                                      void *tid, void *bid, Dim3 *block, Dim3 *grid,
                                      int *wave) {
    {
        std::lock_guard<std::mutex> guard(lock);
        symbols[host] = name ? name : (device ? device : "");
    }
    using Function = void (*)(void **, const void *, char *, const char *, unsigned,
                             void *, void *, Dim3 *, Dim3 *, int *);
    auto next = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "__hipRegisterFunction"));
    if (!next) std::abort();
    next(modules, host, device, name, limit, tid, bid, block, grid, wave);
}

#define HOST_LAUNCH(api) \
extern "C" int api(const void *function, Dim3 grid, Dim3 block, \
                   void **args, size_t shared, void *stream) { \
    if (capture_host(function, grid, block, shared, args)) return 0; \
    using Function = int (*)(const void *, Dim3, Dim3, void **, size_t, void *); \
    return forward<Function>(#api, function, grid, block, args, shared, stream); \
}
HOST_LAUNCH(hipLaunchKernel)
HOST_LAUNCH(hipLaunchKernel_spt)

extern "C" int hipExtLaunchKernel(const void *function, Dim3 grid, Dim3 block,
                                   void **args, size_t shared, void *stream,
                                   void *start, void *stop, int flags) {
    if (capture_host(function, grid, block, shared, args)) return 0;
    using Function = int (*)(const void *, Dim3, Dim3, void **, size_t, void *, void *, void *, int);
    return forward<Function>("hipExtLaunchKernel", function, grid, block, args, shared, stream, start, stop, flags);
}

#define COOPERATIVE_LAUNCH(api) \
extern "C" int api(const void *function, Dim3 grid, Dim3 block, \
                   void **args, unsigned shared, void *stream) { \
    if (capture_host(function, grid, block, shared, args, true)) return 0; \
    using Function = int (*)(const void *, Dim3, Dim3, void **, unsigned, void *); \
    return forward<Function>(#api, function, grid, block, args, shared, stream); \
}
COOPERATIVE_LAUNCH(hipLaunchCooperativeKernel)
COOPERATIVE_LAUNCH(hipLaunchCooperativeKernel_spt)

extern "C" int hipLaunchKernelExC(const void *config, const void *function, void **args) {
    if (capture_host(function, {1, 1, 1}, {1, 1, 1}, 0, nullptr, true)) return 0;
    using Function = int (*)(const void *, const void *, void **);
    return forward<Function>("hipLaunchKernelExC", config, function, args);
}

extern "C" int hipLaunchByPtr(const void *function) {
    if (capture_host(function, {1, 1, 1}, {1, 1, 1}, 0, nullptr, true)) return 0;
    using Function = int (*)(const void *);
    return forward<Function>("hipLaunchByPtr", function);
}

extern "C" int hipModuleLoad(void **module, const char *path) {
    using Function = int (*)(void **, const char *);
    int status = forward<Function>("hipModuleLoad", module, path);
    remember_module(status, module, __builtin_return_address(0), path);
    return status;
}

extern "C" int hipModuleLoadData(void **module, const void *image) {
    using Function = int (*)(void **, const void *);
    int status = forward<Function>("hipModuleLoadData", module, image);
    remember_module(status, module, __builtin_return_address(0), image);
    return status;
}

extern "C" int hipModuleLoadDataEx(void **module, const void *image, unsigned count,
                                    void *options, void **values) {
    using Function = int (*)(void **, const void *, unsigned, void *, void **);
    int status = forward<Function>("hipModuleLoadDataEx", module, image, count, options, values);
    remember_module(status, module, __builtin_return_address(0), image);
    return status;
}

extern "C" int hipModuleGetFunction(void **function, void *module, const char *name) {
    using Function = int (*)(void **, void *, const char *);
    int status = forward<Function>("hipModuleGetFunction", function, module, name);
    if (!status && function) {
        std::lock_guard<std::mutex> guard(lock);
        functions[*function] = module;
        symbol_functions.erase(*function);
    }
    return status;
}

extern "C" int hipGetFuncBySymbol(void **function, const void *symbol) {
    using Function = int (*)(void **, const void *);
    int status = forward<Function>("hipGetFuncBySymbol", function, symbol);
    if (!status && function) {
        std::lock_guard<std::mutex> guard(lock);
        symbol_functions[*function] = belongs(symbol);
        functions.erase(*function);
    }
    return status;
}

extern "C" int hipModuleUnload(void *module) {
    using Function = int (*)(void *);
    int status = forward<Function>("hipModuleUnload", module);
    if (!status) {
        std::lock_guard<std::mutex> guard(lock);
        modules.erase(module);
        for (auto it = functions.begin(); it != functions.end();) {
            if (it->second == module) it = functions.erase(it); else ++it;
        }
    }
    return status;
}

extern "C" int hipModuleLaunchKernel(void *function, unsigned gx, unsigned gy,
                                      unsigned gz, unsigned bx, unsigned by,
                                      unsigned bz, unsigned shared, void *stream,
                                      void **args, void **extra) {
    if (capture_module(function, __builtin_return_address(0))) return 0;
    using Function = int (*)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, unsigned, void *, void **, void **);
    return forward<Function>("hipModuleLaunchKernel", function, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
}

extern "C" int hipModuleLaunchCooperativeKernel(void *function, unsigned gx, unsigned gy,
                                                 unsigned gz, unsigned bx, unsigned by,
                                                 unsigned bz, unsigned shared, void *stream,
                                                 void **args) {
    if (capture_module(function, __builtin_return_address(0))) return 0;
    using Function = int (*)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, unsigned, void *, void **);
    return forward<Function>("hipModuleLaunchCooperativeKernel", function, gx, gy, gz, bx, by, bz, shared, stream, args);
}

extern "C" int hipExtModuleLaunchKernel(void *function, unsigned gx, unsigned gy,
                                         unsigned gz, unsigned bx, unsigned by,
                                         unsigned bz, size_t shared, void *stream,
                                         void **args, void **extra, void *start,
                                         void *stop, unsigned flags) {
    if (capture_module(function, __builtin_return_address(0))) return 0;
    using Function = int (*)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, size_t, void *, void **, void **, void *, void *, unsigned);
    return forward<Function>("hipExtModuleLaunchKernel", function, gx, gy, gz, bx, by, bz, shared, stream, args, extra, start, stop, flags);
}

extern "C" int hipDrvLaunchKernelEx(const void *config, void *function, void **args, void **extra) {
    if (capture_module(function, __builtin_return_address(0))) return 0;
    using Function = int (*)(const void *, void *, void **, void **);
    return forward<Function>("hipDrvLaunchKernelEx", config, function, args, extra);
}

#define GRAPH_LAUNCH(api) \
extern "C" int api(void *graph, void *stream) { \
    if (capture_unknown(__builtin_return_address(0))) return 0; \
    using Function = int (*)(void *, void *); \
    return forward<Function>(#api, graph, stream); \
}
GRAPH_LAUNCH(hipGraphLaunch)
GRAPH_LAUNCH(hipGraphLaunch_spt)

#define MULTI_LAUNCH(api, count_type) \
extern "C" int api(void *params, count_type count, unsigned flags) { \
    if (capture_unknown(__builtin_return_address(0))) return 0; \
    using Function = int (*)(void *, count_type, unsigned); \
    return forward<Function>(#api, params, count, flags); \
}
MULTI_LAUNCH(hipLaunchCooperativeKernelMultiDevice, int)
MULTI_LAUNCH(hipExtLaunchMultiKernelMultiDevice, int)
MULTI_LAUNCH(hipModuleLaunchCooperativeKernelMultiDevice, unsigned)

extern "C" int hipHccModuleLaunchKernel(void *function, unsigned gx, unsigned gy,
                                         unsigned gz, unsigned bx, unsigned by,
                                         unsigned bz, size_t shared, void *stream,
                                         void **args, void **extra, void *start,
                                         void *stop) {
    if (capture_module(function, __builtin_return_address(0))) return 0;
    using Function = int (*)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, size_t, void *, void **, void **, void *, void *);
    return forward<Function>("hipHccModuleLaunchKernel", function, gx, gy, gz, bx, by, bz, shared, stream, args, extra, start, stop);
}
