"""GLM-5.2 benchmark runtime-context publication."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import aklr


@pytest.mark.parametrize("rank_aware", [False, True])
@pytest.mark.parametrize("published", [False, True])
def test_publish_once(monkeypatch, rank_aware, published):
    monkeypatch.syspath_prepend(str(Path(aklr.__file__).parent / "benchmarks/models"))
    model_bench = importlib.import_module("aklr.benchmarks.models.glm52.model_bench")
    runtime = ModuleType("sglang.srt.runtime_context")
    received = Mock()

    def legacy_publish(server_args, *, role):
        received(server_args, role=role)

    def ranked_publish(server_args, *, role, ranks):
        received(server_args, role=role, ranks=ranks)

    runtime.publish = ranked_publish if rank_aware else legacy_publish
    runtime.get_context = lambda: SimpleNamespace(is_config_namespace_published=lambda name: published)
    runtime.SpawnRanks = SimpleNamespace
    runtime.spawn_world_rank = Mock(return_value=11)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    args = SimpleNamespace()

    model_bench.publish_once(args, gpu_id=3, tp_rank=3)

    if published:
        received.assert_not_called()
        runtime.spawn_world_rank.assert_not_called()
    elif rank_aware:
        received.assert_called_once_with(
            args, role="scheduler", ranks=SimpleNamespace(world_rank=11, gpu_id=3)
        )
        runtime.spawn_world_rank.assert_called_once_with(args, tp_rank=3, pp_rank=0)
    else:
        received.assert_called_once_with(args, role="scheduler")
        runtime.spawn_world_rank.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_parallel_state(monkeypatch, legacy):
    monkeypatch.syspath_prepend(str(Path(aklr.__file__).parent / "benchmarks/models"))
    bench = importlib.import_module("sglang_bench")
    state = SimpleNamespace(attn_dp_size=2)
    runner = SimpleNamespace(ps=state) if legacy else SimpleNamespace()
    runtime = ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = Mock(return_value=state)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    assert bench.parallel_state(runner) is state
    assert bench.local_conc(SimpleNamespace(CONC=8), runner) == 4
    assert runtime.get_parallel.call_count == (0 if legacy else 2)
