# SPDX-License-Identifier: MIT
"""Run InferenceX's bench_serving for one concurrency, with the arguments of its fixed-seq client_argv."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from perf_utils.server import execution_env, run_client


def client_argv(
    base_url: str, served_model_name: str, tokenizer: str, workload: dict, conc: int, result: Path
) -> list[str]:
    """--backend vllm requests OpenAI /v1/completions, which SGLang and vLLM both serve."""
    return [
        sys.executable, "-m", "infx.bench_serving.benchmark_serving",
        "--backend", "vllm",
        "--base-url", base_url,
        "--model", served_model_name,
        "--tokenizer", tokenizer,
        "--trust-remote-code",
        "--dataset-name", "random",
        "--random-input-len", str(workload["isl"]),
        "--random-output-len", str(workload["osl"]),
        "--random-range-ratio", str(workload["random_range_ratio"]),
        "--seed", str(workload["seed"]),
        "--num-prompts", str(workload["num_prompts_per_conc"] * conc),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--ignore-eos",
        "--num-warmups", str(workload["warmups_per_conc"] * conc),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-dir", str(result.parent),
        "--result-filename", result.name,
    ]  # fmt: skip


def run(argv: list[str], inferencex: Path, log: Path, timeout: float) -> int:
    with log.open("w") as output:
        return run_client(
            argv,
            cwd=inferencex,
            env={**execution_env(), "PYTHONPATH": str(inferencex)},
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
