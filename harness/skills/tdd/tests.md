# Good and Bad Tests

## Good Tests

**Integration-style**: Test through real interfaces, not mocks of internal parts.

```python
# GOOD: Tests observable behavior
def test_disable_names_op_types(monkeypatch):
    monkeypatch.setenv(ENV_DISABLE, "gemm,bmm")
    env = RuntimeEnv()
    assert env.disabled_op_types == {"gemm", "bmm"}
    assert not env.is_all_disabled
```

Characteristics:

- Tests behavior users/callers care about
- Uses public API only
- Survives internal refactors
- Describes WHAT, not HOW
- One logical assertion per test

## Bad Tests

**Implementation-detail tests**: Coupled to internal structure.

```python
# BAD: Tests implementation details
def test_runtime_env_reads_aklr_disable(monkeypatch):
    names = []
    monkeypatch.setattr(os.environ, "get", lambda name, default=None: names.append(name))
    RuntimeEnv()
    assert ENV_DISABLE in names
```

Red flags:

- Mocking internal collaborators
- Testing private methods
- Asserting on call counts/order
- Test breaks when refactoring without behavior change
- Test name describes HOW not WHAT
- Verifying through external means instead of interface

```python
# BAD: Reaches behind the interface
def test_git_raises_outside_a_repository(tmp_path):
    with pytest.raises(ValueError, match="git cat-file"):
        ValidationScope._git(tmp_path, "cat-file", "blob", f"{'1' * 40}:aklr/plugins/aiter/gemm/plugin.py")


# GOOD: Verifies through interface
def test_text_at_base_raises_outside_a_repository(tmp_path):
    scope = ValidationScope(tmp_path, "0" * 40, "1" * 40, (), ())
    with pytest.raises(ValueError, match="git cat-file"):
        scope.text_at_base("aklr/plugins/aiter/gemm/plugin.py")
```

**Tautological tests**: Expected value restates the implementation, so the test passes by construction.

```python
# BAD: Expected value is recomputed the way the code computes it
def test_physical_device_follows_hip_visible_devices(monkeypatch):
    visible = "4, 5"
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", visible)
    assert physical_device("cuda:1") == visible.split(",")[1].strip()


# GOOD: Expected value is an independent, known literal
def test_physical_device_follows_hip_visible_devices(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4, 5")
    assert physical_device("cuda:1") == "5"
```
