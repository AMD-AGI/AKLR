###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################


class ExecutionError(RuntimeError): ...


class ExecutorClosed(ExecutionError): ...


class SubmissionRejected(ExecutionError): ...


class InsufficientResources(ExecutionError): ...


class RunnerStartupFailed(ExecutionError): ...


class RunnerCrashed(ExecutionError): ...


class TaskFailed(ExecutionError): ...


class TaskTimeout(ExecutionError): ...


class TaskCancelled(ExecutionError): ...


class ChannelError(ExecutionError): ...


class InvalidRunnerState(ExecutionError): ...
