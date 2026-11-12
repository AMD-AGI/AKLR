###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


class ExecutionError(RuntimeError):
    """Base error for submission, scheduling, runner lifecycle, and task execution."""


class ExecutorClosed(ExecutionError):
    """Work was submitted after the executor or backend stopped accepting tasks."""


class SubmissionRejected(ExecutionError):
    """The submission was rejected before entering the scheduling queue."""


class InsufficientResources(ExecutionError):
    """Too few matching healthy resources exist; merely busy resources can be waited for."""


class RunnerStartupFailed(ExecutionError):
    """The runner could not become ready to execute the task."""


class RunnerCrashed(ExecutionError):
    """The task's child process exited before completing execution."""


class TaskFailed(ExecutionError):
    """The task raised an exception or failed to return a valid execution response."""


class TaskTimeout(ExecutionError):
    """Task execution exceeded its deadline, excluding runner startup."""


class TaskCancelled(ExecutionError):
    """Runner shutdown interrupted execution; queued Future cancellation uses CancelledError."""


class ChannelError(ExecutionError):
    """Internal error sending, receiving, or decoding a runner message."""


class InvalidRunnerState(ExecutionError):
    """Internal runner lifecycle or lease ownership invariant was violated."""
