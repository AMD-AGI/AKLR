"""trace.analyse and census.build on hand-made Kineto traces: step labels, phases, correlation, tails, no-step."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from op_census import census, trace  # noqa: E402


def span(name, ts, dur, tid=1):
    return {"ph": "X", "cat": "user_annotation", "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur}


def launch(corr, ts, tid=1, name="hipLaunchKernel"):
    return {
        "ph": "X",
        "cat": "cuda_runtime",
        "name": name,
        "pid": 1,
        "tid": tid,
        "ts": ts,
        "dur": 1,
        "args": {"correlation": corr},
    }


def kernel(corr, name, ts, dur):
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "pid": 0,
        "tid": 7,
        "ts": ts,
        "dur": dur,
        "args": {"correlation": corr},
    }


def sglang_trace():
    ev = [
        # EXTEND (prefill) then its draft extend, then a decode round: DRAFT, VERIFY.
        span("step[EXTEND bs=2 toks=4096]", 100, 50),
        launch(1, 110),
        launch(2, 120),
        span("step[DRAFT bs=2]", 200, 20),
        launch(3, 205),
        span("step[DRAFT bs=2]", 300, 20),
        launch(4, 305),
        # A graph replay: one launch, three kernels.
        span("step[VERIFY bs=2]", 400, 20),
        launch(5, 405, name="hipGraphLaunch"),
        launch(6, 430),  # after VERIFY's span, before the next step: its tail (sampling)
        kernel(1, "Cijk_gemm_prefill", 1000, 100),
        kernel(2, "fmha_fwd_prefill", 1100, 50),
        kernel(3, "Cijk_gemm_draft", 1200, 10),
        kernel(4, "Cijk_gemm_draft", 1300, 10),
        kernel(5, "Cijk_gemm_verify", 1400, 5),
        kernel(5, "mla_decode", 1405, 5),
        kernel(5, "allreduce_x", 1410, 5),
        kernel(6, "sampling_kernel", 1420, 2),
        kernel(99, "orphan", 1500, 1),  # no runtime record
    ]
    ev.insert(0, launch(7, 50))  # before the first step on its thread
    ev.append(kernel(7, "warm_copy", 900, 3))
    return ev


def test_sglang_steps_phases_and_attribution():
    rt = trace.analyse(sglang_trace())
    assert [s.stage for s in rt.steps] == ["EXTEND", "DRAFT", "DRAFT", "VERIFY"]
    assert [s.phase for s in rt.steps] == ["prefill", "prefill", "decode", "decode"]
    assert rt.steps[0].tokens == 4096 and rt.steps[0].bs == 2
    names = {s.stage + str(s.index): sorted(a[1] for a in s.activities) for s in rt.steps}
    assert names["VERIFY3"] == ["Cijk_gemm_verify", "allreduce_x", "mla_decode", "sampling_kernel"]
    tail = [a for a in rt.steps[3].activities if a[4] == "tail"]
    assert [a[1] for a in tail] == ["sampling_kernel"]
    assert [a[1] for a in rt.no_step] == ["warm_copy"]
    assert [a[1] for a in rt.unmatched] == ["orphan"]


def test_census_shares_add_up():
    rt = trace.analyse(sglang_trace())
    c = census.build({"id": "t"}, [("w0", [rt])])
    total = sum(v["share_of_all"] for v in c["phases"].values())
    assert abs(total - 1) < 1e-9
    for ph in c["phases"]:
        s = sum(o["share_of_phase"] for o in c["ops"] if o["phase"] == ph)
        assert abs(s - 1) < 1e-9
    pre = {o["op_class"]: o for o in c["ops"] if o["phase"] == "prefill"}
    assert pre["gemm"]["us_per_rank"] == 110 and pre["attention"]["us_per_rank"] == 50
    dec = {o["op_class"]: o for o in c["ops"] if o["phase"] == "decode"}
    assert dec["communication"]["us_per_rank"] == 5
    assert c["phases"]["decode"]["steps"] == 2  # DRAFT + VERIFY
    assert c["checks"]["unmatched_share"] > 0 and c["checks"]["no_step_share"] > 0


def test_two_ranks_average():
    a, b = trace.analyse(sglang_trace()), trace.analyse(sglang_trace())
    c = census.build({"id": "t"}, [("w0", [a, b])])
    pre = {o["op_class"]: o for o in c["ops"] if o["phase"] == "prefill"}
    assert pre["gemm"]["us_per_rank"] == 110
    assert c["checks"]["rank_activity_spread"] == 0


def test_vllm_labels():
    ev = [
        span("execute_context_1(8192)_generation_0(0)", 100, 50),
        launch(1, 110),
        span("execute_context_0(0)_generation_4(32)", 200, 50),
        launch(2, 210),
        launch(3, 260),
        span("execute_context_1(100)_generation_3(24)", 300, 50),
        launch(4, 310),
        kernel(1, "Cijk_a", 1000, 10),
        kernel(2, "Cijk_b", 1100, 10),
        kernel(3, "rejection_sample", 1120, 1),
        kernel(4, "Cijk_c", 1200, 10),
    ]
    rt = trace.analyse(ev)
    assert [s.phase for s in rt.steps] == ["prefill", "decode", "mixed"]
    assert rt.steps[1].generation_requests == 4 and rt.steps[1].generation_tokens == 32
    assert [a[1] for a in rt.steps[1].activities] == ["Cijk_b", "rejection_sample"]


def test_overlap_scheduler_correlation_wins_over_gpu_time():
    # Step 2's CPU span runs while step 1's kernels are still on the GPU; correlation keeps them apart.
    ev = [
        span("step[DECODE bs=1]", 100, 10),
        launch(1, 105),
        span("step[EXTEND bs=1 toks=10]", 120, 10),
        launch(2, 125),
        kernel(1, "decode_k", 500, 100),
        kernel(2, "extend_k", 600, 10),
    ]
    rt = trace.analyse(ev)
    assert [a[1] for a in rt.steps[0].activities] == ["decode_k"]
    assert [a[1] for a in rt.steps[1].activities] == ["extend_k"]


def test_gzip_roundtrip(tmp_path):
    import gzip

    p = tmp_path / "t.trace.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump({"traceEvents": sglang_trace()}, f)
    assert len(trace.load_events(str(p))) == len(sglang_trace())


def test_draft_phase_from_detailed_annotations():
    ev = [
        span("step[EXTEND bs=1 toks=900 c_sq=900 c_sk=900]", 100, 10),
        launch(1, 105),
        span("step[DRAFT bs=1 c_sq=900 c_sk=900]", 120, 10),
        launch(2, 125),  # draft extend of the prefill
        span("step[DRAFT bs=1 g_sq=1 g_sk=901]", 140, 10),
        launch(3, 145),  # draft decode
        span("step[DRAFT bs=1 g_sq=1 g_sk=902]", 160, 10),
        launch(4, 165),
        span("step[VERIFY bs=1 g_sq=4 g_sk=905]", 180, 10),
        launch(5, 185),
        span("step[DRAFT bs=1 c_sq=3 c_sk=905]", 200, 10),
        launch(6, 205),  # draft extend after verify
        *[kernel(i, f"k{i}", 1000 + 10 * i, 1) for i in range(1, 7)],
    ]
    rt = trace.analyse(ev)
    assert [s.phase for s in rt.steps] == ["prefill", "prefill", "decode", "decode", "decode", "decode"]


def test_stall_is_left_out():
    ev = [
        span("step[DECODE bs=1]", 100, 10),
        launch(1, 105),
        span("step[DECODE bs=1]", 200, 10),
        launch(2, 205),
        span("step[DECODE bs=1]", 300, 10),
        launch(3, 305),
        kernel(1, "k", 1000, 50),
        kernel(2, "k", 2000, 50),
        kernel(3, "k", 3000, 27_000_000),
    ]
    c = census.build({"id": "t"}, [("w0", [trace.analyse(ev)])])
    assert c["checks"]["stalls"] == 1 and c["stalls"][0]["us"] == 27_000_000
    assert c["phases"]["decode"]["us_per_rank"] == 100


def test_load_buckets():
    assert census.load_bucket("decode", 3, 0) == ("bs 3", 3)
    assert census.load_bucket("prefill", 2, 4096) == ("tokens <= 4096", 4096)
    assert census.load_bucket("prefill", 1, 4097) == ("tokens <= 8192", 8192)
    assert census.load_bucket("prefill", 1, 100) == ("tokens <= 512", 512)
    assert census.load_bucket("mixed", 9, 20000) == ("tokens > 16384", 1 << 30)
    assert census.load_bucket("idle", 0, 0) is None


def test_ops_by_load():
    c = census.build({"id": "t"}, [("w0", [trace.analyse(sglang_trace())])])
    rows = {(o["phase"], o["bucket"], o["op_class"]): o for o in c["ops_by_load"]}
    # The draft extend after the 4096-token EXTEND (no token count of its own) is in that prefill's bucket.
    pre = rows[("prefill", "tokens <= 4096", "gemm")]
    assert pre["steps"] == 2 and pre["us_per_rank"] == 110 and pre["bucket_share_of_phase"] == 1.0
    assert not any(k[0] == "prefill" and k[1] != "tokens <= 4096" for k in rows)
    dec = [o for o in c["ops_by_load"] if o["phase"] == "decode"]
    assert {o["bucket"] for o in dec} == {"bs 2"}
    assert abs(sum(o["share_of_bucket"] for o in dec) - 1) < 1e-9


def test_cache_traffic_before_a_prefill_moves_to_it():
    ev = [
        span("step[VERIFY bs=8]", 100, 20),
        launch(1, 105),
        launch(2, 130),
        launch(3, 131),
        launch(4, 132),
        span("step[EXTEND bs=1 toks=4096]", 200, 50),
        launch(5, 210),
        kernel(1, "Cijk_gemm_verify", 1000, 10),
        kernel(2, "sampling_kernel", 1100, 2),  # an ordinary tail: stays with the decode step
        kernel(3, "sglang::transfer_mamba_load_kernel(sglang::MambaTransferParams)", 1110, 30),
        {
            "ph": "X",
            "cat": "gpu_memcpy",
            "name": "Memcpy DtoH (Device -> Host)",
            "pid": 0,
            "tid": 8,
            "ts": 1150,
            "dur": 40,
            "args": {"correlation": 4},
        },
        kernel(5, "Cijk_gemm_prefill", 1200, 100),
    ]
    c = census.build({"id": "t"}, [("w0", [trace.analyse(ev)])])
    dec = {o["op_class"]: o["us_per_rank"] for o in c["ops"] if o["phase"] == "decode"}
    pre = {o["op_class"]: o["us_per_rank"] for o in c["ops"] if o["phase"] == "prefill"}
    assert dec == {"gemm": 10, "sampling": 2}
    assert pre == {"gemm": 100, "kv_cache": 30, "memcpy": 40}
    assert abs(c["checks"]["cache_traffic_moved_share"] - 70 / 182) < 1e-9
    stages = {(o["stage"], o["op_class"]) for o in c["ops_by_stage"] if o["phase"] == "prefill"}
    assert ("EXTEND/before", "kv_cache") in stages and ("EXTEND", "gemm") in stages
