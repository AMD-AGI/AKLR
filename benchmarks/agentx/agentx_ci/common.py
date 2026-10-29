# SPDX-License-Identifier: MIT
"""Shared helpers: the run definition, canonical hashes and JSON files."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parents[1]
DEFAULT_CONFIG = APP_DIR / "agentx.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def sha256_json(value) -> str:
    """Hash of ``value``'s canonical JSON: sorted keys, no whitespace."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    return path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
