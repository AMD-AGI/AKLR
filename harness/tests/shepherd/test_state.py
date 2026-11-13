"""Exercise round ownership, takeover, due checks and state integrity through the public CLI."""

import json
import os
import pwd
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "automations/shepherd/skills/optimize-and-deliver-kernels/scripts/state.py"
# The agent records its round environment, then optionally sleeps, saves, or leaves a detached child behind.
AGENT = """
import json, os, subprocess, sys, time
out, mode = sys.argv[1], sys.argv[2]
with open(out, "a") as stream:
    stream.write(json.dumps({k: v for k, v in os.environ.items() if k.startswith("AKLR_")}) + "\\n")
if mode == "detach":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=True)
if mode.startswith("sleep"):
    time.sleep(float(mode[5:]))
if mode == "fail":
    sys.exit(3)
if mode == "stubborn":
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(600)
if mode == "save":
    sys.exit(subprocess.run(sys.argv[3:]).returncode)
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


def round_command(root, config, out, mode, *extra, now=False):
    return [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--root",
        str(root),
        "round",
        "--config",
        str(config),
        *(["--now"] if now else []),
        "--",
        sys.executable,
        "-I",
        "-c",
        AGENT,
        str(out),
        mode,
        *extra,
    ]


def agent_runs(out):
    return [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []


def write_owner(root, **fields):
    here = {"id": "x", "pid": 1, "start": "0", "boot_id": "", "pid_namespace": "", "config_sha256": ""}
    (root / "owner.json").write_text(json.dumps({**here, **fields}))


def host():
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
    }


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "state"
    success(path, "init")
    return path


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "configuration.yaml"
    path.write_text("mode: run\n")
    return path


def test_root_expands_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEPHERD_TEST_DIR", str(tmp_path))
    success("$SHEPHERD_TEST_DIR/state", "init")
    assert (tmp_path / "state" / "state.json").is_file()


def test_round_runs_under_a_scheduler_without_user_or_local_path(tmp_path, config):
    home = tmp_path / "home"
    (home / ".local/bin").mkdir(parents=True)
    agent = home / ".local/bin/agent-cli"
    agent.write_text(f'#!/bin/sh\necho "$USER" > {tmp_path}/user\n')
    agent.chmod(0o755)
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    success(f"{tmp_path}/$USER/state", "init", env=env)
    name = pwd.getpwuid(os.getuid()).pw_name
    assert (tmp_path / name / "state" / "state.json").is_file()
    success(f"{tmp_path}/$USER/state", "round", "--config", str(config), "--", "agent-cli", env=env)
    assert (tmp_path / "user").read_text().strip() == name


def test_init_creates_missing_private_parents(tmp_path):
    success(tmp_path / "user" / "shepherd" / "state", "init")
    assert (tmp_path / "user" / "shepherd" / "state" / "state.json").is_file()
    assert (tmp_path / "user" / "shepherd").stat().st_mode & 0o777 == 0o700


def test_root_under_a_shared_directory_is_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    result = call(shared / "state", "init")
    assert result.returncode != 0
    assert "writable by other users" in result.stderr
    assert not (shared / "state").exists()


def test_concurrent_rounds_run_one_agent(root, config, tmp_path):
    out = tmp_path / "runs"
    rounds = [
        subprocess.Popen(round_command(root, config, out, "sleep2"), stdout=subprocess.PIPE) for _ in range(4)
    ]
    results = [json.loads(process.communicate()[0]) for process in rounds]
    assert sum("round" in result for result in results) == 1
    assert sum("busy" in result for result in results) == 3
    assert len(agent_runs(out)) == 1
    assert not (root / "owner.json").exists()
    assert json.loads((root / "last_round.json").read_text())["id"] == agent_runs(out)[0]["AKLR_ROUND_ID"]


def test_agent_exit_code_is_returned(root, config, tmp_path):
    result = subprocess.run(
        round_command(root, config, tmp_path / "out", "save"), capture_output=True, check=False
    )
    assert result.returncode != 0


def test_only_the_running_round_saves(root, config, tmp_path):
    state = {"campaigns": [{"id": "c1", "operation": "allocation-intent"}]}
    source = tmp_path / "next.json"
    source.write_text(json.dumps(state))
    save = [
        sys.executable,
        "-I",
        str(SCRIPT),
        "--root",
        str(root),
        "save",
        "--revision",
        "0",
        "--input",
        str(source),
    ]
    out = tmp_path / "out"
    subprocess.run(round_command(root, config, out, "save", *save), check=True)
    assert success(root, "show")["state"] == state
    # The round is over and its id is no longer the owner's.
    stale = {**os.environ, "AKLR_ROUND_ID": agent_runs(out)[0]["AKLR_ROUND_ID"]}
    assert call(root, "save", "--revision", "1", "--input", str(source), env=stale).returncode != 0
    assert call(root, "save", "--revision", "1", "--input", str(source)).returncode != 0


def test_save_rejects_stale_revisions_and_survives_restart(root, tmp_path):
    write_owner(root, id="live", **host())
    env = {**os.environ, "AKLR_ROUND_ID": "live"}
    source = tmp_path / "next.json"
    source.write_text(json.dumps({"campaigns": []}))
    assert success(root, "save", "--revision", "0", "--input", str(source), env=env)["revision"] == 1
    assert call(root, "save", "--revision", "0", "--input", str(source), env=env).returncode != 0
    assert success(root, "show")["revision"] == 1


@pytest.mark.parametrize("content", ["{broken", "[]", '{"bad": NaN}'])
def test_invalid_save_preserves_snapshot(root, tmp_path, content):
    write_owner(root, id="live", **host())
    before = (root / "state.json").read_bytes()
    source = tmp_path / "invalid.json"
    source.write_text(content)
    env = {**os.environ, "AKLR_ROUND_ID": "live"}
    assert call(root, "save", "--revision", "0", "--input", str(source), env=env).returncode != 0
    assert (root / "state.json").read_bytes() == before


def test_killed_wrapper_stays_busy_until_its_agent_exits(root, config, tmp_path):
    out = tmp_path / "out"
    wrapper = subprocess.Popen(round_command(root, config, out, "sleep3"))
    while not agent_runs(out):
        time.sleep(0.05)
    wrapper.send_signal(signal.SIGKILL)
    wrapper.wait()
    dead_id = agent_runs(out)[0]["AKLR_ROUND_ID"]
    busy = subprocess.run(
        round_command(root, config, out, "fast"), capture_output=True, text=True, check=True
    )
    assert json.loads(busy.stdout) == {"busy": dead_id}
    assert "still running" in (root / "reports.md").read_text()
    while Path(f"/proc/{wrapper.pid}").exists() or any(
        f"AKLR_ROUND_ID={dead_id}".encode() in environ.read_bytes()
        for environ in Path("/proc").glob("[0-9]*/environ")
        if os.access(environ, os.R_OK)
    ):
        time.sleep(0.1)
    taken = subprocess.run(
        round_command(root, config, out, "fast"), capture_output=True, text=True, check=True
    )
    assert json.loads(taken.stdout)["recovered_from"] == dead_id
    assert agent_runs(out)[-1]["AKLR_RECOVERED_FROM"] == dead_id


def test_detached_leftovers_are_killed(root, config, tmp_path):
    out = tmp_path / "out"
    subprocess.run(round_command(root, config, out, "detach"), check=True)
    marker = f"AKLR_ROUND_ID={agent_runs(out)[0]['AKLR_ROUND_ID']}".encode()
    survivors = []
    for environ in Path("/proc").glob("[0-9]*/environ"):
        try:
            if marker in environ.read_bytes():
                survivors.append(environ)
        except OSError:
            continue
    assert survivors == []


def test_owner_from_another_boot_is_taken_over_and_another_namespace_is_busy(root, config, tmp_path):
    out = tmp_path / "out"
    write_owner(root, id="old-boot", boot_id="another-boot", pid_namespace=host()["pid_namespace"])
    taken = subprocess.run(
        round_command(root, config, out, "fast"), capture_output=True, text=True, check=True
    )
    assert json.loads(taken.stdout)["recovered_from"] == "old-boot"
    write_owner(root, id="elsewhere", boot_id=host()["boot_id"], pid_namespace="pid:[1]")
    busy = subprocess.run(
        round_command(root, config, out, "fast"), capture_output=True, text=True, check=True
    )
    assert json.loads(busy.stdout) == {"busy": "elsewhere"}


def wake(root, delta):
    value = json.loads((root / "state.json").read_text())
    value["state"]["wake_at"] = (datetime.now(timezone.utc) + delta).isoformat()
    (root / "state.json").write_text(json.dumps(value))


def test_round_waits_for_wake_at_unless_something_changed(root, config, tmp_path):
    out = tmp_path / "out"
    subprocess.run(round_command(root, config, out, "fast"), check=True)
    wake(root, timedelta(minutes=30))
    waiting = subprocess.run(
        round_command(root, config, out, "fast"), capture_output=True, text=True, check=True
    )
    assert "waiting_until" in json.loads(waiting.stdout)
    assert len(agent_runs(out)) == 1
    subprocess.run(round_command(root, config, out, "fast", now=True), check=True)
    assert len(agent_runs(out)) == 2
    config.write_text("mode: drain\n")
    subprocess.run(round_command(root, config, out, "fast"), check=True)
    assert len(agent_runs(out)) == 3
    for delta in (timedelta(minutes=-1), timedelta(hours=3)):
        wake(root, delta)
        subprocess.run(round_command(root, config, out, "fast"), check=True)
    assert len(agent_runs(out)) == 5


def test_missing_or_corrupt_state_never_becomes_an_empty_ledger(root, config, tmp_path):
    missing = tmp_path / "missing"
    assert (
        subprocess.run(round_command(missing, config, tmp_path / "out", "fast"), check=False).returncode != 0
    )
    assert not missing.exists()
    assert call(root, "init").returncode != 0
    (root / "state.json").write_text("{broken")
    assert subprocess.run(round_command(root, config, tmp_path / "out", "fast"), check=False).returncode != 0
    assert call(root, "init").returncode != 0
    assert not (root / "owner.json").exists()
    assert not (tmp_path / "out").exists()


def save_as_owner(root, tmp_path, revision, state):
    write_owner(root, id="live", **host())
    source = tmp_path / f"next{revision}.json"
    source.write_text(json.dumps(state))
    return call(
        root,
        "save",
        "--revision",
        str(revision),
        "--input",
        str(source),
        env={**os.environ, "AKLR_ROUND_ID": "live"},
    )


@pytest.mark.parametrize("kind", ["campaigns", "pools"])
def test_save_never_drops_a_budgeted_entry_or_changes_its_bounds(root, tmp_path, kind):
    # Campaigns bound model spend and pools (shared node allocations) node hours: the committed budget is derived
    # from both.
    admitted = {"id": "c1", "phase": "admitted", "bounds": {"model_usd": 400, "node_hours": 20}}
    assert save_as_owner(root, tmp_path, 0, {kind: [admitted]}).returncode == 0
    assert save_as_owner(root, tmp_path, 1, {kind: [{**admitted, "phase": "preparing"}]}).returncode == 0
    for bad in (
        {kind: []},
        {},
        {kind: [{**admitted, "bounds": {"model_usd": 1, "node_hours": 20}}]},
        {kind: [admitted, {"phase": "admitted"}]},
        {kind: [admitted, admitted]},
    ):
        result = save_as_owner(root, tmp_path, 2, bad)
        assert result.returncode != 0, bad
    assert success(root, "show")["state"][kind][0]["phase"] == "preparing"


@pytest.mark.parametrize("ending", ["fail", "missing-agent", "terminated"])
def test_a_round_that_did_not_end_cleanly_is_recovered(root, config, tmp_path, ending):
    out = tmp_path / "out"
    subprocess.run(round_command(root, config, out, "fast"), check=True)
    if ending == "missing-agent":
        command = [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--root",
            str(root),
            "round",
            "--config",
            str(config),
            "--",
            str(tmp_path / "no-such-agent"),
        ]
        assert subprocess.run(command, capture_output=True, check=False).returncode != 0
        dead_id = json.loads((root / "last_round.json").read_text())["id"]
    else:
        wrapper = subprocess.Popen(
            round_command(root, config, out, "sleep30" if ending == "terminated" else "fail")
        )
        while len(agent_runs(out)) < 2:
            time.sleep(0.05)
        if ending == "terminated":
            wrapper.send_signal(signal.SIGTERM)
        assert wrapper.wait(timeout=30) != 0
        dead_id = agent_runs(out)[1]["AKLR_ROUND_ID"]
    assert not (root / "owner.json").exists()
    # A dead round is recovered at once, whatever wake_at says.
    wake(root, timedelta(minutes=30))
    taken = subprocess.run(
        round_command(root, config, out, "fast"), capture_output=True, text=True, check=True
    )
    assert json.loads(taken.stdout)["recovered_from"] == dead_id
    assert agent_runs(out)[-1]["AKLR_RECOVERED_FROM"] == dead_id
    # That recovery ended cleanly, so the next trigger waits again.
    assert "waiting_until" in json.loads(
        subprocess.run(
            round_command(root, config, out, "fast"), capture_output=True, text=True, check=True
        ).stdout
    )


def test_terminated_round_is_killed_when_its_agent_ignores_sigterm(root, config, tmp_path):
    out = tmp_path / "out"
    wrapper = subprocess.Popen(round_command(root, config, out, "stubborn"))
    while not agent_runs(out):
        time.sleep(0.05)
    time.sleep(0.5)
    wrapper.send_signal(signal.SIGTERM)
    assert wrapper.wait(timeout=30) != 0
    assert not (root / "owner.json").exists()


def test_group_writable_parent_is_refused(tmp_path):
    shared = tmp_path / "group"
    shared.mkdir()
    shared.chmod(0o770)
    result = call(shared / "state", "init")
    assert result.returncode != 0
    assert "writable" in result.stderr


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
    assert success(root, "show")["revision"] == 0
