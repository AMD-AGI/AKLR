"""compare.compare on hand-made censuses: class shares, kernels above the threshold, phases seen in one run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from op_census import compare  # noqa: E402


def census(pid, decode_ops, decode_kernels, mixed=None):
    phases = {
        "prefill": {"share_of_all": 0.1, "steps": 10, "us_per_step": 900.0},
        "decode": {"share_of_all": 0.9, "steps": 1000, "us_per_step": 50.0},
    }
    ops = [{"phase": "prefill", "op_class": "gemm", "share_of_phase": 1.0}]
    ops += [{"phase": "decode", "op_class": c, "share_of_phase": s} for c, s in decode_ops.items()]
    kernels = [{"phase": "prefill", "name": "gemm_p", "share_of_phase": 1.0}]
    kernels += [{"phase": "decode", "name": n, "share_of_phase": s} for n, s in decode_kernels.items()]
    if mixed is not None:
        phases["mixed"] = {"share_of_all": mixed, "steps": 3, "us_per_step": 100.0}
        ops.append({"phase": "mixed", "op_class": "gemm", "share_of_phase": 1.0})
        kernels.append({"phase": "mixed", "name": "gemm_m", "share_of_phase": 1.0})
    return {
        "point": {"id": pid},
        "windows": [{"name": "w0"}],
        "phases": phases,
        "ops": ops,
        "kernels": kernels,
    }


A_OPS = {"gemm": 0.50, "moe": 0.30, "attention": 0.20}
A_KERN = {"gemm_d": 0.50, "moe_d": 0.30, "attn_d": 0.196, "tiny": 0.004}


def test_same_run_agrees():
    r = compare.compare(census("a", A_OPS, A_KERN), census("b", A_OPS, A_KERN))
    assert r["agree"] and [p["phase"] for p in r["phases"]] == ["prefill", "decode"]


def test_class_share_beyond_tolerance_disagrees():
    b_ops = {"gemm": 0.45, "moe": 0.35, "attention": 0.20}
    r = compare.compare(census("a", A_OPS, A_KERN), census("b", b_ops, A_KERN))
    dec = r["phases"][1]
    assert not r["agree"] and not dec["agree"]
    assert {c["op_class"] for c in dec["classes"] if not c["ok"]} == {"gemm", "moe"}
    # Within the tolerance it agrees.
    assert compare.compare(census("a", A_OPS, A_KERN), census("b", b_ops, A_KERN), class_tol=0.06)["agree"]


def test_kernel_above_threshold_in_one_run_only_disagrees():
    b_kern = dict(A_KERN, attn_d=0.186, new_kernel=0.01)
    r = compare.compare(census("a", A_OPS, A_KERN), census("b", A_OPS, b_kern))
    assert not r["agree"]
    assert [k["name"] for k in r["phases"][1]["kernels_missing"]] == ["new_kernel"]


def test_kernel_near_threshold_is_listed_not_failed():
    a_kern = dict(A_KERN, edge=0.006)
    b_kern = dict(A_KERN, edge=0.004)
    r = compare.compare(census("a", A_OPS, a_kern), census("b", A_OPS, b_kern))
    assert r["agree"]
    assert [k["name"] for k in r["phases"][1]["kernels_near"]] == ["edge"]
    # Present in both but far apart (0.4% against 1.5%): not near, a disagreement.
    r = compare.compare(
        census("a", A_OPS, dict(A_KERN, gqa=0.004)), census("b", A_OPS, dict(A_KERN, gqa=0.015))
    )
    assert not r["agree"] and [k["name"] for k in r["phases"][1]["kernels_missing"]] == ["gqa"]


def test_phase_in_one_run_only():
    # A sliver of mixed steps in one run is fine; a real mixed phase in one run only is not.
    assert compare.compare(census("a", A_OPS, A_KERN, mixed=0.001), census("b", A_OPS, A_KERN))["agree"]
    r = compare.compare(census("a", A_OPS, A_KERN, mixed=0.05), census("b", A_OPS, A_KERN))
    assert not r["agree"] and r["phases"][-1]["only_in"] == "a"
    assert "DISAGREE" in compare.markdown(r)


def test_a_run_without_capture_never_agrees():
    empty = {"point": {"id": "b"}, "windows": [], "phases": {}, "ops": [], "kernels": []}
    r = compare.compare(census("a", A_OPS, A_KERN), empty)
    assert not r["agree"] and r["reason"] == "no capture in run b"
    assert "no capture in run b" in compare.markdown(r)


def with_load(c, decode_buckets):
    """``decode_buckets``: bucket -> (steps, share of phase, {class: share of bucket})."""
    c["ops_by_load"] = [
        {
            "phase": "decode",
            "bucket": f"bs {bs}",
            "bucket_key": bs,
            "op_class": cls,
            "steps": steps,
            "share_of_bucket": s,
            "bucket_share_of_phase": share,
        }
        for bs, (steps, share, classes) in decode_buckets.items()
        for cls, s in classes.items()
    ]
    return c


BS1 = {"fused_decoder_layers": 0.7, "gemm": 0.3}
BS2 = {"moe": 0.6, "gemm": 0.4}


def mixed_run(pid, f1):
    """Decode at bs 1 (fused layers) for a fraction f1 of the phase time, bs 2 for the rest."""
    ops = {c: f1 * BS1.get(c, 0) + (1 - f1) * BS2.get(c, 0) for c in set(BS1) | set(BS2)}
    kern = {c + "_k": v for c, v in ops.items()}
    return with_load(
        census(pid, ops, kern), {1: (int(4000 * f1), f1, BS1), 2: (int(4000 * (1 - f1)), 1 - f1, BS2)}
    )


def test_load_mix_only():
    # 44% against 57% of the decode time at bs 1: the classes move by more than 3 points as a whole, not per bucket.
    r = compare.compare(mixed_run("a", 0.44), mixed_run("b", 0.57))
    dec = r["phases"][1]
    assert not r["agree"] and r["verdict"] == "load_mix" and dec["by_load"]["status"] == "load_mix"
    assert all(b["ok"] for b in dec["by_load"]["buckets"])
    assert "load mix differs" in compare.markdown(r)


def test_a_bucket_that_disagrees():
    a, b = mixed_run("a", 0.44), mixed_run("b", 0.57)
    for o in b["ops_by_load"]:
        if o["bucket"] == "bs 2":
            o["share_of_bucket"] = {"moe": 0.5, "gemm": 0.5}[o["op_class"]]
    r = compare.compare(a, b)
    assert r["verdict"] == "disagree" and r["phases"][1]["by_load"]["status"] == "disagree"


def test_too_few_steps_to_judge():
    a, b = mixed_run("a", 0.44), mixed_run("b", 0.57)
    for c in (a, b):
        for o in c["ops_by_load"]:
            o["steps"] = 50  # under MIN_STEPS["decode"]
    r = compare.compare(a, b)
    assert r["verdict"] == "few_steps" and r["phases"][1]["by_load"]["judged_cover"] == [0, 0]
