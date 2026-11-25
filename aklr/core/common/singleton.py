# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import threading
from typing import Any, ClassVar, Self, cast


class Singleton:
    _instance: ClassVar[Any] = None
    _instance_lock: ClassVar[threading.RLock] = threading.RLock()

    @classmethod
    def get_instance(cls) -> Self:
        with Singleton._instance_lock:
            if cls.__dict__.get("_instance") is None:
                cls._instance = cls()
            return cast(Self, cls._instance)

    @classmethod
    def set_instance(cls, instance: Self | None) -> None:
        with Singleton._instance_lock:
            cls._instance = instance
