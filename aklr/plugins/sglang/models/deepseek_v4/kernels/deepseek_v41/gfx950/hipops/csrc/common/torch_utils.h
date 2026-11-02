// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side checks of the tensors an op gets.
#pragma once

#include <cstdint>

#include <ATen/ATen.h>

namespace infera_flash {

inline void check(const at::Tensor &t, const at::Tensor &ref, at::ScalarType dtype, const char *name) {
    TORCH_CHECK(t.device() == ref.device() && t.scalar_type() == dtype, name,
                " must be on the input's GPU with dtype ", dtype);
    TORCH_CHECK(t.is_contiguous() && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
                " must be contiguous and 16 B aligned");
}

} // namespace infera_flash
