# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import threading
import time

from aklr.core.common import Singleton


def test_instance_is_created_once_per_class() -> None:
    class Base(Singleton):
        pass

    class Child(Base):
        pass

    assert Base.get_instance() is Base.get_instance()
    assert type(Child.get_instance()) is Child
    assert Child.get_instance() is not Base.get_instance()


def test_set_instance_replaces_and_none_resets() -> None:
    class Thing(Singleton):
        pass

    first = Thing.get_instance()
    replacement = Thing()
    Thing.set_instance(replacement)
    assert Thing.get_instance() is replacement

    Thing.set_instance(None)
    assert Thing.get_instance() not in (first, replacement)


def test_constructor_may_use_another_singleton() -> None:
    class Inner(Singleton):
        pass

    class Outer(Singleton):
        def __init__(self) -> None:
            self.inner = Inner.get_instance()

    assert Outer.get_instance().inner is Inner.get_instance()


def test_concurrent_first_calls_construct_once() -> None:
    constructed = []

    class Slow(Singleton):
        def __init__(self) -> None:
            time.sleep(0.01)
            constructed.append(self)

    threads = [threading.Thread(target=Slow.get_instance) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(constructed) == 1
