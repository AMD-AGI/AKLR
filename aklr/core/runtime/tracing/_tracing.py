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
    tracing_runtime = TracingRuntime.get_instance()

    if tracing_runtime is not None:
        try:
            tracing_runtime.stop()
        except Exception as e:
            logger.error(f"Cannot stop existing tracing runtime: {e}, ignoring")
