# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Registry behavior against ordinary Python modules, without a vLLM installation."""

import functools
import inspect
import math
import sys
from types import ModuleType

import pytest

from aklr.core.runtime._hook_registry import KEEP_RESULT, HookRegistry, HookSpec, HookType

PHASE = "device_ready"
ERRORS = (ValueError, RuntimeError)


@pytest.fixture
def hooks(monkeypatch):
    module = ModuleType("aklr_test_hook_targets")
    module.fn = lambda value: value * 2
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = HookRegistry()
    yield registry, module
    registry.restore()


def spec(module, hook_id, hook, **kwargs):
    return HookSpec(hook_id=hook_id, target=f"{module.__name__}.fn", hook=hook, **kwargs)


def test_repeated_registration_and_apply_keep_one_wrapper(hooks):
    registry, module = hooks
    original = module.fn
    declaration = spec(module, "double", lambda next_fn, value: next_fn(value) + 1)
    registry.register(declaration)
    registry.register(declaration)
    assert module.fn is original

    assert registry.apply(phase=PHASE) == [f"{module.__name__}.fn"]
    installed = module.fn
    registry.register(declaration)
    registry.apply(phase=PHASE)

    assert module.fn is installed
    assert module.fn(3) == 7
    assert module.fn.__wrapped__ is original
    assert registry.original("double") is original
    assert any(row["hook_id"] == "double" and row["phase"] == PHASE for row in registry.report())

    registry.restore()
    registry.restore()
    assert module.fn is original
    registry.apply(phase=PHASE)
    assert module.fn(3) == 7
    assert registry.original("double") is original


def test_hook_identity_and_applied_phase_cannot_be_redefined(hooks):
    registry, module = hooks
    registry.register(spec(module, "identity", lambda next_fn, value: next_fn(value)))
    with pytest.raises(ERRORS):
        registry.register(spec(module, "identity", lambda next_fn, value: value))

    registry.apply(phase=PHASE)
    installed = module.fn
    with pytest.raises(ERRORS):
        registry.register(spec(module, "late", lambda next_fn, value: next_fn(value)))
    assert module.fn is installed


def test_before_after_and_around_have_defined_nesting(hooks):
    registry, module = hooks
    events = []

    def original(value, *, scale):
        events.append("original")
        return value * scale

    def before(value, *, scale):
        events.append("before")
        return (value + 1,), {"scale": scale + 1}

    def after(result, *args, **kwargs):
        events.append(("after", result))
        return KEEP_RESULT

    def around(next_fn, *args, **kwargs):
        events.append("enter")
        result = next_fn(*args, **kwargs)
        events.append("leave")
        return result + 1

    module.fn = original
    # Registration order is deliberately different from nesting order.
    registry.register(spec(module, "around", around, order=30))
    registry.register(spec(module, "before", before, hook_type=HookType.BEFORE, order=10))
    registry.register(spec(module, "after", after, hook_type=HookType.AFTER, order=20))
    registry.apply(phase=PHASE)

    assert module.fn(2, scale=3) == 13
    assert events == ["enter", "before", "original", ("after", 12), "leave"]


def test_after_can_explicitly_return_none(hooks):
    registry, module = hooks
    registry.register(spec(module, "none", lambda result, *args: None, hook_type=HookType.AFTER))
    registry.apply(phase=PHASE)
    assert module.fn(3) is None


def test_around_can_skip_the_inner_chain(hooks):
    registry, module = hooks
    calls = []
    registry.register(spec(module, "inner", lambda next_fn, value: calls.append(value) or next_fn(value)))
    registry.register(spec(module, "outer", lambda next_fn, value: "handled", order=1))
    registry.apply(phase=PHASE)
    assert module.fn(3) == "handled"
    assert calls == []


def test_after_is_not_a_finally_callback(hooks):
    registry, module = hooks
    calls = []

    def original(value):
        raise RuntimeError("original failed")

    module.fn = original
    registry.register(
        spec(
            module,
            "after",
            lambda result, *args: calls.append(result) or KEEP_RESULT,
            hook_type=HookType.AFTER,
        )
    )
    registry.apply(phase=PHASE)
    with pytest.raises(RuntimeError, match="original failed"):
        module.fn(3)
    assert calls == []


def test_replace_is_inside_around_but_does_not_change_original_handle(hooks):
    registry, module = hooks
    original = module.fn
    registry.register(spec(module, "around", lambda next_fn, value: next_fn(value) + 1, order=-10))
    registry.register(spec(module, "replace", lambda value: 100, hook_type=HookType.REPLACE, order=100))
    registry.apply(phase=PHASE)
    assert module.fn(3) == 101
    assert registry.original("around") is original
    assert registry.original("replace")(3) == 6


@pytest.mark.parametrize("kind", ["replace", "exclusive"])
def test_conflicting_replacements_leave_the_target_unchanged(hooks, kind):
    registry, module = hooks
    original = module.fn
    settings = {"hook_type": HookType.REPLACE} if kind == "replace" else {"exclusive": True}
    callback = (lambda value: value) if kind == "replace" else (lambda next_fn, value: next_fn(value))
    with pytest.raises(ERRORS):
        registry.register(spec(module, "one", callback, **settings))
        registry.register(spec(module, "two", callback, **settings))
        registry.apply(phase=PHASE)
    assert module.fn is original


@pytest.mark.parametrize("hook_type", list(HookType))
@pytest.mark.parametrize("order", [-1, 1])
def test_exclusive_hook_rejects_every_other_hook_before_writes(hooks, hook_type, order):
    registry, module = hooks
    original = module.fn
    registry.register(spec(module, "route", lambda next_fn, value: next_fn(value), exclusive=True))
    registry.register(
        spec(module, "other", lambda *args, **kwargs: "intercepted", hook_type=hook_type, order=order)
    )
    with pytest.raises(RuntimeError, match="Exclusive hook cannot be combined"):
        registry.apply(PHASE)
    assert module.fn is original
    assert all(row["state"] == "registered" for row in registry.report())
    assert next(row for row in registry.report() if row["hook_id"] == "route")["exclusive"] is True


def test_validator_sees_saved_original_once_at_preflight(hooks):
    registry, module = hooks
    original = module.fn
    validated = []
    registry.register(
        spec(module, "route", lambda next_fn, value: next_fn(value) + 1, validate=validated.append)
    )
    registry.apply(PHASE)
    registry.apply(PHASE)
    assert validated == [original]
    assert module.fn(3) == 7
    assert validated == [original]
    registry.restore()
    registry.apply(PHASE)
    assert validated == [original, original]


def test_validator_failure_leaves_the_whole_batch_untouched(hooks):
    registry, module = hooks
    original = module.fn
    module.other = lambda value: value + 1
    other = module.other

    def validate(fn):
        assert fn is other
        assert module.fn is original
        raise ValueError("incompatible parameters")

    registry.register(spec(module, "first", lambda next_fn, value: next_fn(value)))
    registry.register(
        HookSpec(
            "second", f"{module.__name__}.other", lambda next_fn, value: next_fn(value), validate=validate
        )
    )
    with pytest.raises(RuntimeError, match="incompatible parameters"):
        registry.apply(PHASE)
    assert module.fn is original
    assert module.other is other
    with pytest.raises(RuntimeError, match="has not been applied"):
        registry.original("first")


def test_preflight_failure_does_not_install_half_a_group(hooks):
    registry, module = hooks
    original = module.fn
    registry.register(spec(module, "norm", lambda next_fn, value: next_fn(value), group="muse"))
    registry.register(
        HookSpec(
            hook_id="experts",
            target=f"{module.__name__}.missing",
            hook=lambda next_fn, value: next_fn(value),
            group="muse",
        )
    )
    with pytest.raises((AttributeError, *ERRORS)):
        registry.apply(phase=PHASE)
    assert module.fn is original
    assert "missing" not in vars(module)


def test_failed_assignment_rolls_back_earlier_writes(hooks, monkeypatch):
    registry, _ = hooks
    writes = []

    class FailingModule(ModuleType):
        def __setattr__(self, name, value):
            if getattr(self, "reject", False):
                if name == "z_experts":
                    raise RuntimeError("assignment failed")
                if name == "a_norm":
                    writes.append(value)
            super().__setattr__(name, value)

    module = FailingModule("aklr_test_hook_assignment")
    module.a_norm = lambda value: value
    module.z_experts = lambda value: value
    original_norm, original_experts = module.a_norm, module.z_experts
    module.reject = True
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for target in ("a_norm", "z_experts"):
        registry.register(
            HookSpec(
                hook_id=target,
                target=f"{module.__name__}.{target}",
                hook=lambda next_fn, value: next_fn(value),
                group="muse",
            )
        )

    with pytest.raises(RuntimeError):
        registry.apply(phase=PHASE)
    assert len(writes) >= 2  # A wrapper was written and subsequently restored.
    assert writes[0] is not original_norm and writes[-1] is original_norm
    assert module.a_norm is original_norm
    assert module.z_experts is original_experts


@pytest.mark.parametrize("target_name", ["Base", "Child"])
def test_method_descriptors_and_inheritance_survive_restore(hooks, target_name):
    registry, module = hooks

    class Base:
        def instance(self, value):
            return self, value

        @classmethod
        def classify(cls, value):
            return cls, value

        @staticmethod
        def compute(value):
            return value * 2

    class Child(Base):
        pass

    module.Base, module.Child = Base, Child
    target = getattr(module, target_name)
    names = ("instance", "classify", "compute")
    original_dict = {name: vars(target).get(name) for name in names}
    original_static = {name: inspect.getattr_static(target, name) for name in names}

    def around(next_fn, *args, **kwargs):
        return "wrapped", next_fn(*args, **kwargs)

    for name in names:
        registry.register(
            HookSpec(hook_id=name, target=f"{module.__name__}.{target_name}.{name}", hook=around)
        )
    registry.apply(phase=PHASE)

    instance = target()
    assert instance.instance(3) == ("wrapped", (instance, 3))
    assert target.instance(instance, 3) == ("wrapped", (instance, 3))
    assert target.classify(3) == instance.classify(3) == ("wrapped", (target, 3))
    assert target.compute(3) == instance.compute(3) == ("wrapped", 6)
    assert isinstance(inspect.getattr_static(target, "classify"), classmethod)
    assert isinstance(inspect.getattr_static(target, "compute"), staticmethod)
    assert target.instance.__name__ == "instance"
    assert inspect.signature(target.instance) == inspect.signature(Base.instance)

    registry.restore()
    for name in names:
        assert inspect.getattr_static(target, name) is original_static[name]
        if original_dict[name] is None:
            assert name not in vars(target)
        else:
            assert vars(target)[name] is original_dict[name]
    assert instance.instance(3) == (instance, 3)
    assert instance.classify(3) == (target, 3)
    assert instance.compute(3) == 6


@pytest.mark.parametrize("kind", ["builtin", "partial", "bound", "callable"])
@pytest.mark.parametrize("target_name", ["Owner", "Child"])
def test_callable_class_attributes_keep_their_original_binding(hooks, kind, target_name):
    registry, module = hooks

    class CallableObject:
        def __call__(self, value):
            return self, value

    callback = CallableObject()
    original = {
        "builtin": math.sqrt,
        "partial": functools.partial(pow, 2),
        "bound": callback.__call__,
        "callable": callback,
    }[kind]

    class Owner:
        fn = original

    class Child(Owner):
        pass

    module.Owner, module.Child = Owner, Child
    target = getattr(module, target_name)
    instance = target()
    expected = original(9)
    assert target.fn(9) == instance.fn(9) == expected
    registry.register(
        HookSpec(
            "route",
            f"{module.__name__}.{target_name}.fn",
            lambda next_fn, *args: ("hooked", next_fn(*args)),
        )
    )
    registry.apply(PHASE)
    assert target.fn(9) == instance.fn(9) == ("hooked", expected)
    assert registry.original("route") is original
    registry.restore()
    assert inspect.getattr_static(target, "fn") is original
    assert ("fn" in vars(target)) == (target is Owner)
    assert target.fn(9) == instance.fn(9) == expected


def test_nonbinding_class_callable_can_share_an_explicit_module_alias(hooks):
    registry, module = hooks

    class Owner:
        fn = math.sqrt

    module.Owner = Owner
    module.fn = Owner.fn
    registry.register(
        HookSpec(
            "sqrt",
            f"{module.__name__}.Owner.fn",
            lambda next_fn, value: next_fn(value) + 1,
            aliases=(f"{module.__name__}.fn",),
        )
    )
    registry.apply(PHASE)
    assert module.fn(9) == Owner.fn(9) == Owner().fn(9) == 4
    registry.restore()
    assert module.fn is Owner.fn is math.sqrt


def test_metaclass_only_target_is_rejected_before_any_writes(hooks):
    registry, module = hooks

    class Meta(type):
        def fn(cls, value):
            return cls, value

    class Owner(metaclass=Meta):
        pass

    module.Owner = Owner
    original = module.fn
    assert Owner.fn(9) == (Owner, 9)
    registry.register(spec(module, "ordinary", lambda next_fn, value: next_fn(value)))
    registry.register(HookSpec("meta", f"{module.__name__}.Owner.fn", lambda next_fn, *args: next_fn(*args)))
    with pytest.raises(RuntimeError, match="Metaclass-only hook targets are not supported"):
        registry.apply(PHASE)
    assert module.fn is original
    assert "fn" not in vars(Owner)
    assert Owner.fn(9) == (Owner, 9)
    assert not hasattr(Owner(), "fn")


def test_module_subclass_method_and_class_target_preserve_their_receivers(hooks, monkeypatch):
    registry, module = hooks

    class MethodModule(ModuleType):
        def fn(self, value):
            return self.__name__, value

    module.MethodModule = MethodModule
    consumer = MethodModule("aklr_test_method_module")
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    other = MethodModule("other")
    native = MethodModule.fn
    registry.register(
        HookSpec(
            "module",
            f"{consumer.__name__}.fn",
            lambda next_fn, *args: ("module", next_fn(*args)),
        )
    )
    registry.register(
        HookSpec(
            "class",
            f"{module.__name__}.MethodModule.fn",
            lambda next_fn, *args: ("class", next_fn(*args)),
        )
    )
    registry.apply(PHASE)
    assert consumer.fn(9) == ("module", (consumer.__name__, 9))
    assert other.fn(9) == ("class", ("other", 9))
    original = registry.original("module")
    assert original.__self__ is consumer and original.__func__ is native
    registry.restore()
    assert "fn" not in vars(consumer)
    assert MethodModule.fn is native
    assert consumer.fn(9) == (consumer.__name__, 9)


@pytest.mark.parametrize("names", [("Base", "Child"), ("ZBase", "AChild")])
@pytest.mark.parametrize("method", ["instance", "classify", "compute"])
def test_parent_and_child_use_batch_originals_regardless_of_path_order(hooks, names, method):
    registry, module = hooks

    class Base:
        def instance(self, value):
            return type(self), value

        @classmethod
        def classify(cls, value):
            return cls, value

        @staticmethod
        def compute(value):
            return value

    class Child(Base):
        pass

    def wrap(label):
        def around(next_fn, *args, **kwargs):
            return label, next_fn(*args, **kwargs)

        return around

    before = inspect.getattr_static(Base, method)
    for name, target, label in zip(names, (Base, Child), ("base", "child")):
        setattr(module, name, target)
        registry.register(HookSpec(label, f"{module.__name__}.{name}.{method}", wrap(label)))
    registry.apply(PHASE)
    installed = [inspect.getattr_static(cls, method) for cls in (Base, Child)]
    registry.apply(PHASE)
    assert [inspect.getattr_static(cls, method) for cls in (Base, Child)] == installed
    for cls, label in ((Base, "base"), (Child, "child")):
        value = 9 if method == "compute" else (cls, 9)
        assert getattr(cls(), method)(9) == (label, value)
    assert registry.original("base") is registry.original("child")
    registry.restore()
    assert inspect.getattr_static(Base, method) is before
    assert method not in vars(Child)


def test_multilevel_inheritance_and_singleton_targets_restore_lookup(hooks):
    registry, module = hooks

    class Base:
        def fn(self, value):
            return value

    class Middle(Base):
        pass

    class Child(Middle):
        pass

    module.Base, module.Middle, module.Child, module.singleton = Base, Middle, Child, Child()
    native = Base.fn
    for name in ("Base", "Middle", "Child", "singleton"):

        def around(next_fn, *args, label=name):
            return label, next_fn(*args)

        registry.register(HookSpec(name, f"{module.__name__}.{name}.fn", around))
    registry.apply(PHASE)
    for name in ("Base", "Middle", "Child"):
        assert getattr(module, name)().fn(9) == (name, 9)
    assert module.singleton.fn(9) == ("singleton", 9)
    original = registry.original("singleton")
    assert original.__self__ is module.singleton and original.__func__ is native
    registry.restore()
    assert Base.fn is native
    assert all("fn" not in vars(owner) for owner in (Middle, Child, module.singleton))
    assert module.singleton.fn(9) == 9


def test_inherited_batch_rolls_back_when_parent_assignment_mutates_then_raises(hooks):
    registry, module = hooks
    writes = []
    reject = [True]

    class Meta(type):
        def __setattr__(cls, name, value):
            super().__setattr__(name, value)
            if name == "fn":
                writes.append(cls.__name__)
                if cls.__name__ == "Base" and reject:
                    reject.pop()
                    raise RuntimeError("assignment failed after mutation")

    class Base(metaclass=Meta):
        def fn(self, value):
            return value

    class Child(Base):
        pass

    module.Base, module.Child = Base, Child
    original = Base.fn
    for name in ("Base", "Child"):
        registry.register(
            HookSpec(name, f"{module.__name__}.{name}.fn", lambda next_fn, *args: next_fn(*args))
        )
    with pytest.raises(RuntimeError, match="failed and was rolled back"):
        registry.apply(PHASE)
    assert writes == ["Child", "Base", "Base"]
    assert Base.fn is original
    assert "fn" not in vars(Child)
    assert Child().fn(9) == 9
    assert all(row["state"] == "registered" for row in registry.report())


@pytest.mark.parametrize("propagate", [False, True])
def test_only_explicit_module_aliases_are_changed(hooks, monkeypatch, propagate):
    registry, module = hooks
    original = module.fn
    consumer = ModuleType("aklr_test_hook_consumer")
    consumer.fn = original  # Represents a from-import performed before installation.
    outside = ModuleType("aklr_test_hook_outside")
    outside.fn = original
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    monkeypatch.setitem(sys.modules, outside.__name__, outside)
    aliases = (f"{consumer.__name__}.fn",) if propagate else ()
    registry.register(spec(module, "route", lambda next_fn, value: next_fn(value) + 1, aliases=aliases))
    registry.apply(phase=PHASE)

    assert module.fn(3) == 7
    assert consumer.fn(3) == (7 if propagate else 6)
    assert outside.fn is original
    assert registry.original("route") is original
    registry.restore()
    assert module.fn is consumer.fn is outside.fn is original


def test_explicit_alias_with_a_different_identity_is_rejected(hooks, monkeypatch):
    registry, module = hooks
    original = module.fn
    consumer = ModuleType("aklr_test_hook_wrong_alias")
    consumer.fn = lambda value: -value
    consumer_original = consumer.fn
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    registry.register(
        spec(module, "route", lambda next_fn, value: next_fn(value), aliases=(f"{consumer.__name__}.fn",))
    )
    with pytest.raises(ERRORS):
        registry.apply(phase=PHASE)
    assert module.fn is original
    assert consumer.fn is consumer_original


@pytest.mark.parametrize("collision", ["target", "alias", "group"])
def test_targets_aliases_and_groups_cannot_span_phases(hooks, collision):
    registry, module = hooks
    group = "together" if collision == "group" else None
    registry.register(
        spec(module, "bootstrap", lambda next_fn, value: next_fn(value), phase="bootstrap", group=group)
    )
    registry.apply(phase="bootstrap")
    installed = module.fn
    module.other = installed if collision == "alias" else (lambda value: -value)
    other = module.other
    target = "fn" if collision == "target" else "other"
    aliases = (f"{module.__name__}.fn",) if collision == "alias" else ()

    with pytest.raises(ERRORS):
        registry.register(
            HookSpec(
                hook_id="device",
                target=f"{module.__name__}.{target}",
                hook=lambda next_fn, value: next_fn(value),
                group=group,
                aliases=aliases,
            )
        )
        registry.apply(phase=PHASE)
    assert module.fn is installed
    assert module.other is other


def test_restore_never_overwrites_a_later_external_patch(hooks):
    registry, module = hooks
    registry.register(spec(module, "route", lambda next_fn, value: next_fn(value) + 1))
    registry.apply(phase=PHASE)
    installed = module.fn
    external = lambda value: "external"
    module.fn = external
    try:
        with pytest.raises(ERRORS):
            registry.restore()
        assert module.fn is external
        assert module.fn(3) == "external"
    finally:
        # Restore ownership so the fixture can clean up the Registry's own installation.
        module.fn = installed
