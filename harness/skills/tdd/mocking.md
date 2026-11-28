# When to Mock

Mock at **system boundaries** only:

- Environment variables (`AKLR_*`, `HIP_VISIBLE_DEVICES`)
- The GPU and its driver (sometimes - prefer a real GPU)
- git (sometimes - prefer a real repository in `tmp_path`)
- Time/randomness
- File system (sometimes - prefer `tmp_path`)

Replace a boundary with pytest's `monkeypatch` fixture (`setattr`, `setenv`), which undoes itself after the test:

```python
def test_physical_device_follows_hip_visible_devices(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    assert physical_device("cuda:1") == "5"
```

Don't mock:

- Your own classes/modules
- Internal collaborators
- Anything you control

## Designing for Mockability

At system boundaries, design interfaces that are easy to mock:

**1. Use dependency injection**

Pass external dependencies in rather than creating them internally:

```python
# Easy to mock: plan() is handed its target, so the test names one on a machine with no GPU
class BenchValidator:
    def plan(self, scope, target): ...


def test_plan_has_one_entry_per_bench_path(scope):
    target = DeviceTarget.from_names("gfx950", "MI355X")
    assert len(BenchValidator(["cuda:0"]).plan(scope, target)) == len(scope.bench_paths)


# Hard to mock: plan() asks the driver itself, so the test has to patch the driver query
class BenchValidator:
    def plan(self, scope):
        target = DeviceTarget.current()
        ...


def test_plan_has_one_entry_per_bench_path(monkeypatch, scope):
    target = DeviceTarget.from_names("gfx950", "MI355X")
    monkeypatch.setattr(DeviceTarget, "current", classmethod(lambda cls, device=None: target))
    assert len(BenchValidator(["cuda:0"]).plan(scope)) == len(scope.bench_paths)
```

**2. Prefer SDK-style interfaces over generic fetchers**

Create specific methods for each external operation instead of one generic function with conditional logic:

```python
# GOOD: Each git question is its own method, independently mockable
class ValidationScope:
    def changed_paths(self): ...
    def paths_at_base(self, directory): ...
    def text_at_base(self, path): ...


def test_base_without_schema_data_is_a_first_release(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ValidationScope, "paths_at_base", lambda self, directory: ("aklr/plugins/aiter/provider/gemm/api.py",)
    )
    assert ValidationScope(tmp_path, "0" * 40, "1" * 40, (), ()).is_first_release


# BAD: One generic git(), so the mock needs conditional logic
class ValidationScope:
    def git(self, *args): ...


def test_base_without_schema_data_is_a_first_release(monkeypatch, tmp_path):
    def fake_git(self, *args):
        if args[0] == "ls-tree":
            return b"aklr/plugins/aiter/provider/gemm/api.py\0"
        if args[0] == "cat-file":
            return b"{}"
        raise AssertionError(args)

    monkeypatch.setattr(ValidationScope, "git", fake_git)
    assert ValidationScope(tmp_path, "0" * 40, "1" * 40, (), ()).is_first_release
```

The SDK approach means:
- Each mock returns one specific shape
- No conditional logic in test setup
- Easier to see which operations a test exercises
- Type safety per operation
