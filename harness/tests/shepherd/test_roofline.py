"""The shepherd's own cost of one call and MI355X's roofline, on a schema stand-in."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "automations/shepherd/skills/optimize-and-deliver-kernels/scripts"
    ),
)
from roofline import Cost, Roofline  # noqa: E402


def gemm(contract="gemm@1.0", dtype="bfloat16"):
    return {
        "contract_id": contract,
        "axes": {
            "m": {"type": "var"},
            "n": {"type": "const", "value": 4096},
            "k": {"type": "const", "value": 8192},
        },
        "attrs": {},
        "inputs": {
            "a": {"dtype": dtype, "shape": ["m", "k"]},
            "b": {"dtype": dtype, "shape": ["n", "k"]},
            "scale": {"dtype": "float32", "shape": None},
        },
        "outputs": {"out": {"dtype": "bfloat16", "shape": ["m", "n"]}},
        "layout_variants": {"by_name": {"row_major": {}}},
    }


def test_cost_and_roofline():
    """A GEMM's math at a's dtype's rate and every tensor moved once (a 4-bit one two values a byte, a scalar
    nothing); the roofline is the longer of the two, never under the timer's floor."""
    cost = Cost.of(gemm(), {"m": 1024})
    moved = (1024 * 8192 + 4096 * 8192 + 1024 * 4096) * 2
    assert cost == (2 * 1024 * 4096 * 8192, moved, "bfloat16")
    assert Roofline().seconds(cost) == pytest.approx(2 * 1024 * 4096 * 8192 / 2.45e15)
    assert Roofline().seconds(Cost.of(gemm(), {"m": 1})) == pytest.approx(
        (8192 + 4096 * 8192 + 4096) * 2 / 7.1e12
    )
    assert Roofline().seconds(Cost(0, 64, None)) == 4.1e-6
    assert (
        Cost.of(gemm(dtype="float4_e2m1"), {"m": 2}).bytes_moved
        == (2 * 8192 + 4096 * 8192) // 2 + 2 * 4096 * 2
    )
    assert Cost.of(gemm("rms_norm@1.0"), {"m": 2}) == (0, Cost.of(gemm(), {"m": 2}).bytes_moved, None)
    with pytest.raises(ValueError, match="no cost formula for unified_attention"):
        Cost.of(gemm("unified_attention@1.0"), {"m": 2})
    with pytest.raises(ValueError, match="cannot read the cost of gemm"):
        Cost.of(gemm(), {})
    with pytest.raises(ValueError, match="no matrix-core rate for float32"):
        Roofline().seconds(Cost(1, 1, "float32"))
