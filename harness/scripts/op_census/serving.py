"""A bound recipe as the server command, its environment and the AgentX client's environment.

The command is the one ``srtctl`` starts for a single-node aggregated worker: ``python3 -m sglang.launch_server``
or ``vllm serve`` with every ``roles.agg.args`` entry as a flag (``true`` is a bare flag, ``false`` and ``null`` are
left out, a list is one token per item, a mapping is JSON). The only changes are local: the checkpoint and draft model ids become the paths they
were found at, and the server listens on ``host:port``. ``served-model-name`` stays the Hugging Face id the client
asks for.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

# Recipe arguments whose value names a Hugging Face repository to load.
PATH_ARGS = ("tokenizer-path", "speculative-draft-model-path", "model-path", "tokenizer")
# Keys of a JSON-valued argument (vLLM's speculative-config) that name a repository.
PATH_KEYS = ("model", "draft_model", "tokenizer")


def checkpoint_id(recipe: Mapping) -> str:
    path = recipe["model"]["path"]
    return path[3:] if path.startswith("hf:") else path


def repositories(recipe: Mapping) -> list[str]:
    """Every repository the server loads: the checkpoint, then tokenizers and draft models in its arguments."""
    found = [checkpoint_id(recipe)]
    for key, value in recipe["roles"]["agg"]["args"].items():
        if key in PATH_ARGS and isinstance(value, str):
            found.append(value)
        spec = _json_arg(value)
        if isinstance(spec, dict):
            found += [spec[k] for k in PATH_KEYS if isinstance(spec.get(k), str)]
    return list(dict.fromkeys(found))


def _json_arg(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _flag_values(value) -> list[str]:
    """A flag's value tokens: a list is one token per item (argparse ``nargs``), a mapping is JSON."""
    if isinstance(value, list):
        return [json.dumps(v) if isinstance(v, (dict, list)) else str(v) for v in value]
    if isinstance(value, dict):
        return [json.dumps(value)]
    return [str(value)]


def server_command(recipe: Mapping, paths: Mapping[str, str], host: str, port: int) -> list[str]:
    """The server argv; ``paths`` maps each of ``repositories(recipe)`` to a local directory."""
    engine = recipe["engine"]["type"] if isinstance(recipe["engine"], dict) else recipe["engine"]
    model = paths[checkpoint_id(recipe)]
    if engine == "sglang":
        argv = ["python3", "-m", "sglang.launch_server", "--model-path", model]
    elif engine == "vllm":
        argv = ["vllm", "serve", model]
    else:
        raise ValueError(f"unsupported engine {engine!r}")
    argv += ["--host", host, "--port", str(port)]
    for key, value in recipe["roles"]["agg"]["args"].items():
        if value is None or value is False:
            continue
        if key in PATH_ARGS and isinstance(value, str):
            value = paths.get(value, value)
        spec = _json_arg(value)
        if isinstance(spec, dict) and any(
            isinstance(spec.get(k), str) and spec[k] in paths for k in PATH_KEYS
        ):
            spec = {
                k: (paths.get(v, v) if k in PATH_KEYS and isinstance(v, str) else v) for k, v in spec.items()
            }
            value = spec if isinstance(value, dict) else json.dumps(spec)
        argv += [f"--{key}"] if value is True else [f"--{key}", *_flag_values(value)]
    return argv


def server_env(recipe: Mapping) -> dict[str, str]:
    """``roles.agg.env``, with ``HF_HUB_OFFLINE`` so that nothing is downloaded at load."""
    env = {k: str(v) for k, v in (recipe["roles"]["agg"].get("env") or {}).items() if v is not None}
    return {"HF_HUB_OFFLINE": "1", **env}


def client_env(point: Mapping, url: str, result_dir: str, duration: int, tokenizer: str) -> dict[str, str]:
    """What ``python3 -m infx.bench agentic`` reads: the matrix environment, the recipe's ``benchmark.env`` and the
    runtime fields ``single_node.runtime_arguments`` sets for an AgentX point. ``duration`` replaces DURATION."""
    env = {**point["environment"], **{k: str(v) for k, v in point["recipe"]["benchmark"]["env"].items()}}
    env.update(
        RESULT_DIR=result_dir,
        RESULT_FILENAME=point["row"]["exp-name"],
        EVAL_ONLY="false",
        RUN_EVAL="false",
        IS_MULTINODE="false",
        DURATION=str(duration),
        AIPERF_SERVER_URL=url,
        TOKENIZER=tokenizer,
        AGENTIC_OUTPUT_DIR=result_dir,
        ENABLE_AGENTX_POWER="false",
        REQUIRE_POWER="0",
    )
    # The workflow sets the offload backend's name from the matrix row; the client checks it against the metadata.
    backend = point["row"].get("kv-offload-backend")
    if env.get("KV_OFFLOADING", "none") != "none" and isinstance(backend, dict):
        env["KV_OFFLOAD_BACKEND"] = backend["name"]
    env.setdefault("AIPERF_EXPERIMENTAL_FAST", "0")
    env.setdefault("AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS", "3600")
    env.setdefault("AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID", "false")
    env.setdefault("AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING", "1")
    return env
