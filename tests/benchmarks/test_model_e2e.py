"""The model e2e benchmark's chunked request lifecycle."""

import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import aklr


@pytest.mark.parametrize("budget", [4, 10])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_work_retains_prefill_requests(monkeypatch, tmp_path, budget, phase):
    monkeypatch.syspath_prepend(str(Path(aklr.__file__).parent / "benchmarks/models"))
    entry = importlib.import_module("aklr.benchmarks.models.main")
    tokens = list(range(32))
    reqs = [
        SimpleNamespace(
            origin_input_ids=tokens[offset : offset + 10],
            prefix_len=0,
            extend_end=10,
            kv=SimpleNamespace(req_pool_idx=None),
        )
        for offset in (0, 16)
    ]
    rows = {}
    batches = []
    forced = []

    def make_extend_batch(requests, runner, reset):
        assert requests is reqs
        prefix, end = reqs[0].prefix_len, reqs[0].extend_end
        assert reset == (prefix == 0)
        for index, req in enumerate(requests, start=1):
            assert (req.prefix_len, req.extend_end) == (prefix, end)
            if reset:
                assert req.kv.req_pool_idx is None
                req.kv.req_pool_idx = index
                rows[index] = []
            assert req.kv.req_pool_idx == index
            assert rows[index] == req.origin_input_ids[:prefix]
        batch = SimpleNamespace(reqs=requests, prefix=prefix, end=end)
        batches.append(batch)
        return batch

    def forward(batch):
        if hasattr(batch, "input_ids"):
            for req, token in zip(batch.reqs, batch.input_ids.tolist()):
                rows[req.kv.req_pool_idx].append(token)
        else:
            for req in batch.reqs:
                rows[req.kv.req_pool_idx].extend(req.origin_input_ids[batch.prefix : batch.end])

    def prepare_decode(batch, runner, input_ids):
        assert batch is batches[-1]
        assert batch.end == 10
        assert all(len(row) == 10 + len(forced) for row in rows.values())
        forced.append(input_ids.tolist())
        batch.input_ids = input_ids

    def bench_eager(fn, reps):
        assert reps == 1
        fn()
        return 2.0

    runner = SimpleNamespace(device="cpu", max_total_num_tokens=26, forward=Mock(side_effect=forward))
    model = ModuleType("model_bench")
    model.setup = Mock(return_value=(runner, Mock()))
    model.chunk_tokens = Mock(return_value=budget)
    model.make_extend_batch = Mock(side_effect=make_extend_batch)
    model.free_req_rows = Mock(side_effect=AssertionError("freed a live KV row"))
    one_batch = ModuleType("sglang.benchmark.one_batch")
    one_batch.prepare_synthetic_inputs_for_latency_test = Mock(return_value=reqs)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=Mock(return_value=SimpleNamespace(encode=Mock(return_value=tokens)))
    )
    for module in (model, one_batch, transformers):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(entry, "DECODE_STEPS", 3)
    monkeypatch.setattr(entry, "TEXTS", {"test": SimpleNamespace(read=lambda: "text")})
    monkeypatch.setattr(entry, "local_conc", lambda args, mr: 2)
    monkeypatch.setattr(entry, "forward_ctx", lambda mr: nullcontext())
    monkeypatch.setattr(entry, "forward_batch_of", lambda batch, mr: batch)
    monkeypatch.setattr(entry, "prepare_decode", prepare_decode)
    monkeypatch.setattr(entry, "bench_eager", bench_eager)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(entry, "time", SimpleNamespace(perf_counter=Mock(side_effect=range(6))))
    result = tmp_path / "result.json"
    args = SimpleNamespace(ISL=10, CONC=2, text="test", phase=phase, result_json=result)
    server_args = SimpleNamespace(tokenizer_path=None, model_path="test", trust_remote_code=False)

    entry.work(server_args, None, args, gpu_id=0, tp_rank=0)

    one_batch.prepare_synthetic_inputs_for_latency_test.assert_called_once_with(
        2, 10, custom_inputs=[tokens[:10], tokens[16:26]]
    )
    step = budget // len(reqs)  # both requests share the scheduler's chunk budget
    assert [(batch.prefix, batch.end) for batch in batches] == [
        (prefix, min(prefix + step, 10)) for prefix in range(0, 10, step)
    ]
    assert forced == [[10, 26], [11, 27], [12, 28]]
    assert rows == {1: tokens[:13], 2: tokens[16:29]}
    model.free_req_rows.assert_not_called()
    assert runner.forward.call_count == len(batches) + 3
    latency = 1000.0 if phase == "decode" else 2.0 * len(batches)
    assert json.loads(result.read_text()) == {
        "tokens_per_s": (2 if phase == "decode" else 10) * 1000 / latency,
        "latency_ms": latency,
    }
    assert [path.name for path in tmp_path.iterdir()] == ["result.json"]
