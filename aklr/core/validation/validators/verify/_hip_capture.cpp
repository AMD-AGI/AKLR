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
static std::string target;
static Callback callback = nullptr;

extern "C" void aklr_verify_capture(const char *library, Callback capture) {
    std::lock_guard<std::mutex> guard(lock);
    target = library;
    callback = capture;
}

static bool belongs(const void *address) {
    Dl_info info{};
    if (!dladdr(address, &info) || !info.dli_fname) return false;
    if (target.empty()) return false;
    if (target[0] == '/') {
        char *resolved = realpath(info.dli_fname, nullptr);
        bool match = resolved && target == resolved;
        free(resolved);
        return match;
    }
    const char *base = strrchr(info.dli_fname, '/');
    return target == (base ? base + 1 : info.dli_fname);
}

static bool capture(const void *function, const void *caller, Dim3 grid, Dim3 block,
                    size_t shared, void **args) {
    Callback cb;
    std::string name;
    {
        std::lock_guard<std::mutex> guard(lock);
        if (!callback || (!belongs(function) && !belongs(caller))) return false;
        cb = callback;
        auto found = symbols.find(function);
        if (found != symbols.end()) name = found->second;
    }
    cb(name.c_str(), grid.x, grid.y, grid.z, block.x, block.y, block.z, shared, args);
    return true;
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

extern "C" int hipLaunchKernel(const void *function, Dim3 grid, Dim3 block,
                                void **args, size_t shared, void *stream) {
    if (capture(function, __builtin_return_address(0), grid, block, shared, args)) return 0;
    using Function = int (*)(const void *, Dim3, Dim3, void **, size_t, void *);
    auto next = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hipLaunchKernel"));
    return next ? next(function, grid, block, args, shared, stream) : 801;
}

extern "C" int hipExtLaunchKernel(const void *function, Dim3 grid, Dim3 block,
                                   void **args, size_t shared, void *stream,
                                   void *start, void *stop, int flags) {
    if (capture(function, __builtin_return_address(0), grid, block, shared, args)) return 0;
    using Function = int (*)(const void *, Dim3, Dim3, void **, size_t, void *, void *, void *, int);
    auto next = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hipExtLaunchKernel"));
    return next ? next(function, grid, block, args, shared, stream, start, stop, flags) : 801;
}

extern "C" int hipModuleLaunchKernel(void *function, unsigned gx, unsigned gy,
                                      unsigned gz, unsigned bx, unsigned by,
                                      unsigned bz, unsigned shared, void *stream,
                                      void **args, void **extra) {
    if (capture(function, __builtin_return_address(0), {gx, gy, gz}, {bx, by, bz}, shared, nullptr)) return 0;
    using Function = int (*)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, unsigned, void *, void **, void **);
    auto next = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "hipModuleLaunchKernel"));
    return next ? next(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra) : 801;
}
