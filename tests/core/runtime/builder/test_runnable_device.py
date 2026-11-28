import pytest
import torch

from aklr.core.runtime.builder import Runnable, RunnableMetadata


@pytest.fixture
def metadata():
    return RunnableMetadata(
        build_type="python", contract_id="test@1.0", schema_name="test_schema", content_hash="0" * 64
    )


def test_cpu_runnable_keeps_its_behavior(metadata):
    runnable = Runnable(lambda x: x + 1, metadata)
    torch.testing.assert_close(runnable(x=torch.tensor([1])), torch.tensor([2]))


@pytest.mark.gpu
@pytest.mark.parametrize("raises", [False, True], ids=["return", "raise"])
def test_runnable_binds_tensor_device_and_restores_caller(metadata, raises):
    assert torch.cuda.device_count() >= 2
    previous = torch.cuda.current_device()
    tensor = torch.empty(1, device="cuda:0")
    observed = []

    def called(x):
        observed.append(torch.cuda.current_device())
        if raises:
            raise RuntimeError("test failure")
        return x

    runnable = Runnable(called, metadata)
    try:
        torch.cuda.set_device(1)
        if raises:
            with pytest.raises(RuntimeError, match="test failure"):
                runnable(x=tensor)
        else:
            assert runnable(x=tensor) is tensor
        assert observed == [0]
        assert torch.cuda.current_device() == 1
    finally:
        torch.cuda.set_device(previous)


@pytest.mark.parametrize("raises", [False, True], ids=["return", "raise"])
def test_runnable_device_context_boundary(metadata, monkeypatch, raises):
    from contextlib import contextmanager

    class DeviceTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:0")

    tensor = torch.empty(1).as_subclass(DeviceTensor)
    current = [1]

    @contextmanager
    def device_context(device):
        saved = current[0]
        current[0] = device.index
        try:
            yield
        finally:
            current[0] = saved

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    observed = []

    def called(x):
        observed.append(torch.cuda.current_device())
        if raises:
            raise RuntimeError("test failure")
        return x

    runnable = Runnable(called, metadata)
    if raises:
        with pytest.raises(RuntimeError, match="test failure"):
            runnable(x=tensor)
    else:
        assert runnable(x=tensor) is tensor
    assert observed == [0]
    assert current == [1]


def test_scalar_runnable_does_not_import_torch(metadata, monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "torch")
    runnable = Runnable(lambda value: value + 1, metadata)
    assert runnable(value=4) == 5
    assert "torch" not in sys.modules
