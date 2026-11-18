"""inventory on hand-made censuses: columns by concurrency, absent classes, every kernel in the CSV."""

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from op_census import inventory  # noqa: E402


def census(conc, decode_ops, kernels, exp="m_tp8_conc{}_kvnone"):
    return {
        "point": {
            "id": f"cfg.{exp.format(conc)}",
            "config_key": "cfg",
            "conc": conc,
            "exp_name": exp.format(conc),
            "model": "m",
            "framework": "sglang",
            "image": "img",
        },
        "phases": {"decode": {"share_of_all": 1.0}},
        "ops": [{"phase": "decode", "op_class": c, "share_of_phase": s} for c, s in decode_ops.items()],
        "kernels": [
            {
                "phase": "decode",
                "op_class": c,
                "name": n,
                "share_of_phase": s,
                "share_of_all": s,
                "calls_per_step": 1.0,
                "us_per_step": 10.0,
                "p50_us": 10.0,
            }
            for n, (c, s) in kernels.items()
        ],
    }


def test_classes_by_concurrency_and_absent_classes():
    a = census(
        1,
        {"fused_decoder_layers": 0.7, "gemm": 0.3},
        {"mono": ("fused_decoder_layers", 0.7), "g": ("gemm", 0.3)},
    )
    b = census(16, {"moe": 0.6, "gemm": 0.4}, {"moe_k": ("moe", 0.6), "g": ("gemm", 0.4)})
    cols, rows = inventory.class_table([b, a][::-1], "decode")
    assert cols == ["c1 tp8", "c16 tp8"]
    rows = dict(rows)
    assert (
        rows["moe"] == [0.0, 0.6]
        and rows["fused_decoder_layers"] == [0.7, 0.0]
        and rows["gemm"] == [0.3, 0.4]
    )


def test_every_kernel_listed():
    a = census(1, {"gemm": 1.0}, {"g": ("gemm", 1.0)})
    b = census(16, {"gemm": 0.9, "moe": 0.1}, {"g": ("gemm", 0.9), "rare": ("moe", 0.1)})
    ks = inventory.kernel_rows([a, b])
    assert [(k["name"], k["points_seen"]) for k in ks] == [("g", 2), ("rare", 1)]
    rows = list(csv.DictReader(io.StringIO(inventory.to_csv([a, b]))))
    assert sorted((r["conc"], r["kernel"]) for r in rows) == [("1", "g"), ("16", "g"), ("16", "rare")]
    md = inventory.markdown([a, b])
    assert "# cfg" in md and "`rare`" in md
