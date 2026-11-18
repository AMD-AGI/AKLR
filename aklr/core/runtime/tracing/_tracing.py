###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 by FlashInfer-Bench Contributors
#
# Adapted from FlashInfer-Bench (https://github.com/flashinfer-ai/flashinfer-bench)
# Modified by the AKLR team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of AKLR (see LICENSE).
###############################################################################

import logging

from aklr.core.runtime.tracing._config import TracingConfig, TracingConfigRegistry
from aklr.core.runtime.tracing._runtime import TracingRuntime

logger = logging.getLogger(__name__)


def enable_tracing(
    tracing_config: (TracingConfigRegistry | TracingConfig | dict[str, TracingConfig] | None) = None,
) -> TracingRuntime:
    runtime = TracingRuntime(tracing_config)
    runtime.start()
    return runtime


def disable_tracing():
    """Stop the active tracing runtime, which flushes its workloads; a failure is only logged."""
    tracing_runtime = TracingRuntime.get_instance()

    if tracing_runtime is not None:
        try:
            tracing_runtime.stop()
        except Exception as e:
            logger.error(f"Cannot stop existing tracing runtime: {e}, ignoring")
