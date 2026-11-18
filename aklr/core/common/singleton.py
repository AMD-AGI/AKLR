# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A base for classes with one process-wide instance."""

from __future__ import annotations

import threading
from typing import Any, ClassVar, Self, cast


class Singleton:
    """A class with one process-wide instance, built without arguments on first use.

    Each subclass has its own instance, a subclass of a singleton included.
    ``set_instance`` swaps it (tests); ``set_instance(None)`` makes the next call build a new one.
    """

    _instance: ClassVar[Any] = None
    # one reentrant lock for all classes: a constructor may ask for another singleton
    _instance_lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def get_instance(cls) -> Self:
        with Singleton._instance_lock:
            # cls.__dict__, not cls._instance: a parent's instance is not the subclass's
            if cls.__dict__.get("_instance") is None:
                cls._instance = cls()
            return cast(Self, cls._instance)

    @classmethod
    def set_instance(cls, instance: Self | None) -> None:
        with Singleton._instance_lock:
            cls._instance = instance
