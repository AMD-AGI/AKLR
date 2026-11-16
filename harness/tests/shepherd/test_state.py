"""Exercise round leases, takeover, due checks and the saved-state rules through the public CLI."""

import json
import os
import pwd
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "automations/shepherd/skills/optimize-and-deliver-kernels/scripts/state.py"
)
CONFIG = """
mode: run
execution:
  round_hours: {round_hours}
policy:
  max_parallel_campaigns: 2
campaign:
  max_model_usd: 250
budget:
  total_model_usd: 500
  total_node_hours: 48
"""


def call(root, *args, env=None):
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def success(root, *args, env=None):
    result = call(root, *args, env=env)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def owner(root):
    path = root / "owner.json"
    return json.loads(path.read_text()) if path.exists() else None


def expire(root):
    record = owner(root)
    record["lease_until"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    (root / "owner.json").write_text(json.dumps(record))


def wake(root, delta):
    value = json.loads((root / "state.json").read_text())
    value["state"]["wake_at"] = (datetime.now(timezone.utc) + delta).isoformat()
    (root / "state.json").write_text(json.dumps(value))


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "state"
    success(path, "init")
    return path


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "configuration.yaml"
    path.write_text(CONFIG.format(round_hours=2))
    return path


class Round:
    """One round as the automation runs it: begin, saves with the returned id, end."""

    def __init__(self, root, config, tmp_path, now=True):
        self.root, self.tmp_path = root, tmp_path
        self.started = success(root, "begin", "--config", str(config), *(["--now"] if now else []))
        self.id = self.started.get("round")

    def save(self, state, revision=None):
        source = self.tmp_path / "next.json"
        source.write_text(json.dumps(state))
        revision = success(self.root, "show")["revision"] if revision is None else revision
        return call(
            self.root, "save", "--round", self.id, "--revision", str(revision), "--input", str(source)
        )

    def end(self, code=0):
        return success(self.root, "end", "--round", self.id, "--exit-code", str(code))


def campaign(id_="c1", phase="admitted", model_usd=200, gpu=0, **fields):
    return {"id": id_, "phase": phase, "pool": "p1", "gpu": gpu, "bounds": {"model_usd": model_usd}, **fields}


def pool(id_="p1", node_hours=24, **fields):
    return {"id": id_, "bounds": {"node_hours": node_hours}, **fields}


def test_root_expands_environment_variables_and_needs_no_user(tmp_path, monkeypatch, config):
    monkeypatch.setenv("SHEPHERD_TEST_DIR", str(tmp_path))
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "SHEPHERD_TEST_DIR": str(tmp_path)}
    success("$SHEPHERD_TEST_DIR/$USER/state", "init", env=env)
    name = pwd.getpwuid(os.getuid()).pw_name
    assert (tmp_path / name / "state" / "state.json").is_file()
    assert "round" in success("$SHEPHERD_TEST_DIR/$USER/state", "begin", "--config", str(config), env=env)


def test_init_creates_missing_private_parents(tmp_path):
    success(tmp_path / "user" / "shepherd" / "state", "init")
    assert (tmp_path / "user" / "shepherd").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("mode", [0o777, 0o770])
def test_root_under_a_shared_directory_is_refused(tmp_path, mode):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(mode)
    result = call(shared / "state", "init")
    assert result.returncode != 0
    assert "writable" in result.stderr
    assert not (shared / "state").exists()


def test_init_makes_every_new_parent_private_whatever_the_umask(tmp_path):
    root = tmp_path / "a" / "b" / "state"
    result = subprocess.run(
        ["bash", "-c", f'umask 000; exec "{sys.executable}" -I "{SCRIPT}" --root "{root}" init'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for path in (tmp_path / "a", tmp_path / "a" / "b", root):
        assert path.stat().st_mode & 0o777 == 0o700, path


def test_concurrent_begins_start_one_round(root, config):
    with ThreadPoolExecutor(4) as pool_:
        results = list(
            pool_.map(lambda _: success(root, "begin", "--config", str(config), "--now"), range(4))
        )
    assert sum("round" in result for result in results) == 1
    assert sum("busy" in result for result in results) == 3
    assert "still running" in (root / "reports.md").read_text()


def test_end_releases_the_ledger_and_only_the_owner_saves(root, config, tmp_path):
    first = Round(root, config, tmp_path)
    assert first.save({"campaigns": []}).returncode == 0
    first.end()
    assert owner(root) is None
    assert json.loads((root / "last_round.json").read_text())["exit_code"] == 0
    # The round is over and its id is no longer the owner's.
    assert first.save({"campaigns": []}).returncode != 0
    second = Round(root, config, tmp_path)
    assert second.started["recovered_from"] is None
    assert call(root, "end", "--round", first.id, "--exit-code", "0").returncode != 0


def test_expired_lease_is_fenced_and_recovered(root, config, tmp_path):
    stale = Round(root, config, tmp_path)
    expire(root)
    rejected = stale.save({"campaigns": []})
    assert rejected.returncode != 0 and "lease expired" in rejected.stderr
    wake(root, timedelta(minutes=30))
    taken = Round(root, config, tmp_path, now=False)
    assert taken.started["recovered_from"] == stale.id
    assert stale.save({"campaigns": []}).returncode != 0
    assert taken.save({"campaigns": []}).returncode == 0


@pytest.mark.parametrize("code", [1, 3])
def test_a_round_that_did_not_end_cleanly_is_recovered(root, config, tmp_path, code):
    Round(root, config, tmp_path).end()
    failed = Round(root, config, tmp_path)
    failed.end(code)
    # A failed round is recovered at once, whatever wake_at says.
    wake(root, timedelta(minutes=30))
    recovering = Round(root, config, tmp_path, now=False)
    assert recovering.started["recovered_from"] == failed.id
    recovering.end()
    # That recovery ended cleanly, so the next trigger waits again.
    assert "waiting_until" in success(root, "begin", "--config", str(config))


def test_round_waits_for_wake_at_unless_something_changed(root, config, tmp_path):
    Round(root, config, tmp_path).end()
    wake(root, timedelta(minutes=30))
    assert "waiting_until" in success(root, "begin", "--config", str(config))
    Round(root, config, tmp_path, now=True).end()
    config.write_text(CONFIG.format(round_hours=1))
    Round(root, config, tmp_path, now=False).end()
    for delta in (timedelta(minutes=-1), timedelta(hours=3)):
        wake(root, delta)
        assert Round(root, config, tmp_path, now=False).id
        success(root, "end", "--round", owner(root)["id"], "--exit-code", "0")


def test_round_hours_must_be_sane(root, config, tmp_path):
    config.write_text(CONFIG.format(round_hours=0))
    assert call(root, "begin", "--config", str(config)).returncode != 0
    assert owner(root) is None


def test_save_rejects_stale_revisions_and_invalid_input(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.save({"campaigns": []}, revision=0).returncode == 0
    assert current.save({"campaigns": []}, revision=0).returncode != 0
    before = (root / "state.json").read_bytes()
    for content in ("{broken", "[]", '{"bad": NaN}'):
        (tmp_path / "next.json").write_text(content)
        result = call(
            root, "save", "--round", current.id, "--revision", "1", "--input", str(tmp_path / "next.json")
        )
        assert result.returncode != 0, content
    assert (root / "state.json").read_bytes() == before


def test_missing_or_corrupt_state_never_becomes_an_empty_ledger(root, config, tmp_path):
    missing = tmp_path / "missing"
    assert call(missing, "begin", "--config", str(config)).returncode != 0
    assert not missing.exists()
    assert call(root, "init").returncode != 0
    (root / "state.json").write_text("{broken")
    assert call(root, "begin", "--config", str(config)).returncode != 0
    assert call(root, "init").returncode != 0
    assert owner(root) is None


@pytest.mark.parametrize("kind", ["campaigns", "pools"])
def test_save_never_drops_a_budgeted_entry_or_changes_its_bounds(root, config, tmp_path, kind):
    current = Round(root, config, tmp_path)
    entry = campaign() if kind == "campaigns" else pool()
    assert current.save({kind: [entry]}).returncode == 0
    changed = {**entry, "bounds": {key: 1 for key in entry["bounds"]}}
    for bad in (
        {kind: []},
        {},
        {kind: [changed]},
        {kind: [entry, {"phase": "admitted"}]},
        {kind: [entry, entry]},
        {kind: [entry, {"id": "x"}]},
    ):
        assert current.save(bad).returncode != 0, bad


def test_phases_only_move_forward_and_end_after_cleanup(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.save({"campaigns": [campaign(phase="optimizing")]}).returncode != 0
    assert current.save({"campaigns": [campaign()]}).returncode == 0
    assert current.save({"campaigns": [campaign(phase="optimizing")]}).returncode == 0
    for bad in ("preparing", "ended", "done"):
        assert current.save({"campaigns": [campaign(phase=bad)]}).returncode != 0, bad
    assert current.save({"campaigns": [campaign(phase="cleaning")]}).returncode == 0
    assert current.save({"campaigns": [campaign(phase="ended", settled_model_usd=200)]}).returncode == 0
    assert current.save({"campaigns": [campaign(phase="cleaning")]}).returncode != 0


def test_gpus_and_budget_are_capped(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.save({"campaigns": [campaign(model_usd=251)]}).returncode != 0
    two = [campaign("c1"), campaign("c2", gpu=1)]
    assert current.save({"campaigns": [campaign("c1"), campaign("c2", gpu=0)]}).returncode != 0
    assert current.save({"campaigns": [campaign("c1"), campaign("c2", gpu=None)]}).returncode != 0
    assert current.save({"campaigns": two, "pools": [pool(node_hours=49)]}).returncode != 0
    assert current.save({"campaigns": two, "pools": [pool()]}).returncode == 0
    # A third GPU is over max_parallel_campaigns, and a third campaign's bound over total_model_usd.
    assert current.save({"campaigns": [*two, campaign("c3", gpu=2, model_usd=50)]}).returncode != 0
    ended = [campaign("c1", phase="ended", settled_model_usd=0), campaign("c2", gpu=1)]
    assert current.save({"campaigns": ended, "pools": [pool()]}).returncode == 0
    assert (
        current.save(
            {"campaigns": [*ended, campaign("c3", gpu=0, model_usd=250)], "pools": [pool()]}
        ).returncode
        == 0
    )
    assert (
        current.save(
            {
                "campaigns": [
                    *ended,
                    campaign("c3", gpu=0, model_usd=250),
                    campaign("c4", gpu=2, model_usd=1),
                ],
                "pools": [pool()],
            }
        ).returncode
        != 0
    )
    # Settling lies at 0 or the admitted bound, never between.
    half = [
        campaign("c1", phase="ended", settled_model_usd=0),
        campaign("c2", gpu=1, phase="cleaning", settled_model_usd=100),
    ]
    assert current.save({"campaigns": half, "pools": [pool()]}).returncode != 0


def test_lowering_a_cap_still_lets_rounds_settle(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.save({"campaigns": [campaign()], "pools": [pool()]}).returncode == 0
    config.write_text(CONFIG.format(round_hours=2).replace("total_model_usd: 500", "total_model_usd: 100"))
    assert current.save({"campaigns": [campaign(phase="preparing")], "pools": [pool()]}).returncode == 0
    assert (
        current.save(
            {
                "campaigns": [campaign(phase="preparing"), campaign("c2", gpu=1, model_usd=1)],
                "pools": [pool()],
            }
        ).returncode
        != 0
    )
