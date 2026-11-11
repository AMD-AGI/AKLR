"""Exercise persistent ownership and state integrity through the public CLI."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "automations/shepherd/skills/shepherd/scripts/state.py"
)


def call(root, *args):
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def success(root, *args):
    result = call(root, *args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "state"
    success(path, "init")
    return path


def test_root_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_TEST_DIR", str(tmp_path))
    success("$SHEPHERD_TEST_DIR/state", "init")
    assert (tmp_path / "state" / "state.json").is_file()


def test_root_under_a_shared_directory_is_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    result = call(shared / "state", "init")
    assert result.returncode != 0
    assert "world-writable" in result.stderr
    assert not (shared / "state").exists()


def test_concurrent_rounds_have_one_durable_owner(root):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: call(root, "acquire", "--owner", f"run-{i}"), range(8)))
    winners = [result for result in results if result.returncode == 0]
    assert len(winners) == 1
    token = json.loads(winners[0].stdout)["token"]
    # All acquire processes have exited, but ownership must not expire.
    assert call(root, "acquire", "--owner", "later-run").returncode != 0
    assert call(root, "release", "--token", "wrong").returncode != 0
    success(root, "release", "--token", token)
    assert success(root, "acquire", "--owner", "next")["token"] != token
    assert call(root, "release", "--token", token).returncode != 0


def test_save_survives_restart_and_rejects_stale_writers(root, tmp_path):
    first = success(root, "acquire", "--owner", "first")
    state = {"campaigns": [{"id": "c1", "operation": "allocation-intent"}]}
    source = tmp_path / "next.json"
    source.write_text(json.dumps(state))
    saved = success(root, "save", "--token", first["token"], "--revision", "0", "--input", str(source))
    assert saved == {"revision": 1, "state": state}
    assert (
        call(root, "save", "--token", first["token"], "--revision", "0", "--input", str(source)).returncode
        != 0
    )
    assert call(root, "save", "--token", "wrong", "--revision", "1", "--input", str(source)).returncode != 0
    success(root, "release", "--token", first["token"])
    resumed = success(root, "acquire", "--owner", "second")
    assert resumed["state"] == state
    assert resumed["revision"] == 1


@pytest.mark.parametrize("content", ["{broken", "[]", '{"bad": NaN}'])
def test_invalid_save_preserves_snapshot(root, tmp_path, content):
    owner = success(root, "acquire", "--owner", "run")
    before = (root / "state.json").read_bytes()
    source = tmp_path / "invalid.json"
    source.write_text(content)
    assert (
        call(root, "save", "--token", owner["token"], "--revision", "0", "--input", str(source)).returncode
        != 0
    )
    assert (root / "state.json").read_bytes() == before


def test_missing_or_corrupt_state_never_becomes_an_empty_ledger(root, tmp_path):
    missing = tmp_path / "missing"
    assert call(missing, "acquire", "--owner", "run").returncode != 0
    assert not missing.exists()
    assert call(root, "init").returncode != 0
    (root / "state.json").write_text("{broken")
    assert call(root, "acquire", "--owner", "run").returncode != 0
    assert call(root, "init").returncode != 0
    assert not (root / "owner.json").exists()
