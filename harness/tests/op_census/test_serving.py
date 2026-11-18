"""The rendered server command and client environment against the AgentX recipes AKLR reviewed.

``benchmarks/agentx/recipes`` holds reviewed ``serving.json`` files for some InferenceX configs. Binding the same
configs here must give the same server flags, server environment (synthetic acceptance included) and client
environment. Needs an InferenceX checkout at the commit those files were reviewed against:
``OP_CENSUS_INFERENCEX=<checkout>``; skipped otherwise.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "harness" / "scripts"))

from op_census import inferencex, serving  # noqa: E402

RECIPES = ROOT / "benchmarks" / "agentx" / "recipes"


def test_apply_overrides_set_unset():
    recipe = {"roles": {"agg": {"args": {"a": 1, "parser": "x"}, "env": {"K": "v"}}}}
    out = inferencex.apply_overrides(
        recipe,
        [
            "--set",
            'roles.agg.env.SGLANG_SIMULATE_ACC_LEN="3.61"',
            "--set",
            'roles.agg.args.speculative-config={"method": "dspark", "synthetic_acceptance_length": 4.04}',
            "--unset",
            "roles.agg.args.parser",
            "--unset",
            "roles.agg.env.MISSING",
        ],
    )
    assert out["roles"]["agg"]["env"] == {"K": "v", "SGLANG_SIMULATE_ACC_LEN": "3.61"}
    assert out["roles"]["agg"]["args"] == {
        "a": 1,
        "speculative-config": {"method": "dspark", "synthetic_acceptance_length": 4.04},
    }
    assert recipe["roles"]["agg"]["args"]["parser"] == "x"  # the input is not changed


def test_flags():
    recipe = {
        "engine": "vllm",
        "model": {"path": "hf:org/m"},
        "roles": {
            "agg": {
                "args": {
                    "served-model-name": "org/m",
                    "trust-remote-code": True,
                    "enable-x": False,
                    "unset": None,
                    "speculative-config": {"method": "dspark", "model": "org/draft"},
                    "attention-config": '{"a":1}',
                }
            }
        },
    }
    paths = {"org/m": "/w/m", "org/draft": "/w/d"}
    assert serving.repositories(recipe) == ["org/m", "org/draft"]
    argv = serving.server_command(recipe, paths, "127.0.0.1", 8000)
    assert argv[:7] == ["vllm", "serve", "/w/m", "--host", "127.0.0.1", "--port", "8000"]
    assert "--trust-remote-code" in argv and "--enable-x" not in argv and "--unset" not in argv
    assert argv[argv.index("--speculative-config") + 1] == '{"method": "dspark", "model": "/w/d"}'
    assert argv[argv.index("--attention-config") + 1] == '{"a":1}'
    assert argv[argv.index("--served-model-name") + 1] == "org/m"


def test_list_flag_is_one_token_per_item():
    # Kimi K3.1 SGLang: cuda-graph-bs-prefill is argparse nargs; JSON there makes the server exit at startup.
    recipe = {
        "engine": "sglang",
        "model": {"path": "hf:org/m"},
        "roles": {"agg": {"args": {"cuda-graph-bs-prefill": [256, 512, 768]}}},
    }
    argv = serving.server_command(recipe, {"org/m": "/w/m"}, "127.0.0.1", 1)
    i = argv.index("--cuda-graph-bs-prefill")
    assert argv[i + 1 : i + 4] == ["256", "512", "768"]


def _pairs(argv):
    out, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[argv[i]] = argv[i + 1]
            i += 2
        else:
            out[argv[i]] = True
            i += 1
    return out


@pytest.mark.skipif(
    not os.environ.get("OP_CENSUS_INFERENCEX"), reason="needs OP_CENSUS_INFERENCEX=<checkout>"
)
def test_matches_reviewed_recipes():
    files = sorted(RECIPES.glob("*/*/*/serving.*.json"))
    reviewed = [json.loads(f.read_text()) for f in files]
    keys = [r["source"]["config_key"] for r in reviewed]
    points = inferencex.bind(Path(os.environ["OP_CENSUS_INFERENCEX"]), keys)
    compared = 0
    for r in reviewed:
        for q in r["points"]:
            kv = "dram" if "dram" in q["arm"] else "none"
            m = [
                p
                for p in points
                if p["config_key"] == r["source"]["config_key"]
                and p["row"]["conc"] == q["conc"]
                and p["recipe"]["roles"]["agg"]["gpus"] == q["gpus"]
                and p["row"]["kv-offloading"] == kv
            ]
            assert len(m) == 1, (q["arm"], q["conc"])
            p = m[0]
            argv = serving.server_command(
                p["recipe"],
                {x: "${AGENTX_MODEL_PATH}" for x in serving.repositories(p["recipe"])},
                "127.0.0.1",
                0,
            )
            mine, theirs = _pairs(argv), _pairs(q["launch"]["command"])
            mine.pop("--port"), theirs.pop("--port")
            assert mine == theirs, (q["arm"], q["conc"])
            assert serving.server_env(p["recipe"]) == q["launch"]["env"], (q["arm"], q["conc"])
            client = serving.client_env(p, "u", "r", int(q["client_env"]["DURATION"]), "t")
            for k, v in q["client_env"].items():
                assert str(client.get(k)) == str(v), (q["arm"], q["conc"], k)
            compared += 1
    assert compared >= 20
