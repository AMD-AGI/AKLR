###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


class ExecutionError(RuntimeError):
    pass


class ExecutorClosed(ExecutionError):
    pass


class SubmissionRejected(ExecutionError):
    pass


class NoResourceAvailable(ExecutionError):
    pass


class RunnerStartupFailed(ExecutionError):
    pass


class RunnerCrashed(ExecutionError):
    pass


class TaskExecutionFailed(ExecutionError):
    pass


class TaskExecutionTimeout(ExecutionError):
    pass


class TaskCancelled(ExecutionError):
    pass


class RunnerExitedEarly(ExecutionError):
    pass


class ChannelFailure(ExecutionError):
    pass


class InvalidRunnerState(ExecutionError):
    pass


_FAILURES_THE_TASK_CAUSED = (RunnerCrashed, TaskExecutionFailed, TaskExecutionTimeout)


def caused_by_the_task(error: BaseException) -> bool:
    """Whether what the task carries is what failed, rather than what carried it."""
    return isinstance(error, _FAILURES_THE_TASK_CAUSED)
