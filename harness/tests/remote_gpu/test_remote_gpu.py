"""remote-gpu against a fake two-node cluster on this machine: no scheduler, no ssh, no GPU.

The fake `ssh` runs its command here, so the paths "on the cluster" are local paths. Node n0
has a holder that starts bg tasks; n1 has none, so a task never starts there.
"""

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

RG = Path(__file__).resolve().parents[2] / "skills/remote-gpu/scripts/remote-gpu"

CLUSTER_ENV = """\
DESCRIPTION='fake'
HOST=fakehost
LOGIN_HOSTS=''
LOGIN_PROFILE=0
PRIORITY=1
WORK_ROOT=$HOME/work
HOME_FULL=0
DOCKER_FLAGS=''
ACCOUNTS='first:qfirst roomy:qroomy'
BURST_QOS=burst
ALLOC_AGGRESSIVE=0
PARTITION=compute
EXTEND=1
MAX_TIME=24:00:00
GPUS_PER_NODE=8
GPU_ARCH=gfx950
"""

CLUSTER_SH = r"""
read -r -d '' CLUSTER_RLIB <<'EOF'
LATE_HINT=x DOCKER_X=() NODE_ENV=()
scheduler_submit() {
  local root=$HOME/scheduler/$C job
  if [[ ${FAKE_REJECT:-} == "$5" ]]; then return 201; fi
  mkdir -p "$root"
  job=$(($(cat "$root/count" 2>/dev/null || echo 0) + 1))
  echo "$job" >"$root/count"
  printf '%s\n' "$@" >"$root/$job.args"
  echo "$job"
}
scheduler_query() {
  local state=RUNNING args=() account=first qos=qfirst root=$HOME/scheduler/$C
  if [[ -e $root/$1.args ]]; then
    mapfile -t args <"$root/$1.args"
    account=${args[4]} qos=${args[5]}
  fi
  if [[ -e $HOME/jobstate ]]; then state=$(cat "$HOME/jobstate"); fi
  if [[ -e $HOME/state_$C ]]; then state=$(cat "$HOME/state_$C"); fi
  if [[ -e $HOME/state_$account:$qos ]]; then state=$(cat "$HOME/state_$account:$qos"); fi
  if [[ -n ${FAKE_START_TEAM_AFTER:-} && $qos != burst && $SECONDS -ge $FAKE_START_TEAM_AFTER ]]; then state=RUNNING; fi
  if [[ -e $root/$1.cancelled ]]; then state=CANCELLED; fi
  printf '%s|%s|01:00:00|%s|tomorrow|compute|%s|%s|4:00:00|%s|%s-test|00:01:00|%s|2\n' \
    "$state" "${FAKE_NODES-n0,n1}" "${FAKE_REASON:-Resources}" "$account" "$qos" "${FAKE_OWNER:-$USER}" "$C" "${FAKE_END:-}"
}
scheduler_jobs() {
  local file job
  for file in "$HOME/scheduler/$C/"*.args; do
    [[ -e $file ]] || continue
    job=${file##*/}; job=${job%.args}
    printf '%s|' "$job"; scheduler_query "$job"
  done
}
scheduler_cancel() {
  if [[ -n ${FAKE_CANCEL_FAILS:-} ]]; then return 1; fi
  mkdir -p "$HOME/scheduler/$C"
  echo "$C:$1 ${2:-all}" >>"$HOME/cancelled"
  touch "$HOME/scheduler/$C/$1.cancelled"
}
scheduler_run() {
  shift 3
  if [[ $* == *gfx_target_version* ]]; then
    printf '%s\n0\n0\n1\n' "${FAKE_GPUS:-8}"
  else
    ${ONTO:+timeout "$ONTO"} "$@" </dev/null
  fi
}
scheduler_capacity() {
  if [[ -e $HOME/capacity_$C ]]; then cat "$HOME/capacity_$C"
  else echo "OK cluster=$C free=2 account=roomy qos=qroomy burst=4"; fi
}
scheduler_holder() {
  J=1 H=n0
  if [[ ${FAKE_MAINTENANCE:-} == 1 ]]; then SECONDS=60; fi
}
scheduler_extend() { :; }
EOF
"""

FAKE_SSH = """\
#!/bin/bash
while [[ $1 == -* ]]; do case $1 in -o) shift 2 ;; *) shift ;; esac; done
shift; exec setsid bash -c "$*"
"""


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    c = home / ".remote-gpu/clusters/fake"
    c.mkdir(parents=True)
    (c / "cluster.env").write_text(CLUSTER_ENV)
    (c / "cluster.sh").write_text(CLUSTER_SH)
    (home / "work").mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ssh").write_text(FAKE_SSH)
    (bindir / "ssh").chmod(0o755)
    e = dict(os.environ, HOME=str(home), PATH=f"{bindir}:{os.environ['PATH']}")
    e.pop("REMOTE_GPU_JOB", None)
    # Map a synthetic HTTPS origin to a local bare repository only inside this test transport.
    real_git = shutil.which("git")
    (bindir / "git").write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "if 'clone' in args:\n"
        "    with open(os.environ['HOME'] + '/clone_calls', 'a') as log: log.write(repr(args) + '\\n')\n"
        "    for i, arg in enumerate(args):\n"
        "        if arg.startswith('https://source.invalid/'):\n"
        "            args[i] = arg.removeprefix('https://source.invalid')\n"
        "            os.environ['GIT_ALLOW_PROTOCOL'] = 'file'\n"
        "            args = ['-c', 'protocol.file.allow=always'] + args\n"
        "            break\n"
        f"os.execv({real_git!r}, [{real_git!r}] + args)\n"
    )
    (bindir / "git").chmod(0o755)
    (bindir / "docker").write_text('#!/bin/bash\nprintf "%s\\n" "$*" >>"$HOME/docker_calls"\n')
    (bindir / "docker").chmod(0o755)
    return e


def rg(env, *args, **kw):
    return subprocess.run(
        [str(RG), "--cluster", "fake", *args], env=env, capture_output=True, text=True, **kw
    )


def git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def repo(tmp_path):
    """A clone of a bare origin, with a tracked file deleted and an untracked one added."""
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    src = tmp_path / "src"
    git(tmp_path, "clone", "-q", str(origin), str(src))
    (src / "kept").write_text("kept\n")
    (src / "gone").write_text("gone\n")
    git(src, "add", ".")
    git(src, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c1")
    git(src, "push", "-q", "origin", "HEAD")
    git(src, "remote", "set-url", "origin", f"https://source.invalid{origin}")
    (src / "gone").unlink()
    (src / "new").write_text("new\n")
    return src


def start_holder(env, *, idle=0):
    result = rg(env, "alloc", "--no-wait")
    assert result.returncode == 0, result.stderr
    holder = submitted(env)[0][6]
    return subprocess.Popen(
        ["bash", holder, str(idle), holder, str(Path(env["HOME"]) / "work/fake"), "fake"],
        env=env,
        start_new_session=True,
    )


def stop_holder(holder):
    if holder.poll() is None:
        holder.terminate()
        try:
            holder.wait(timeout=25)
        finally:
            if holder.poll() is None:
                os.killpg(holder.pid, signal.SIGKILL)
                holder.wait(timeout=5)


def test_wait_fails_when_a_rank_never_started(env, tmp_path):
    jd = tmp_path / "home/work/fake/jobs/1"
    (jd / "spawn").mkdir(parents=True)
    holder = start_holder(env)
    try:
        r = rg(env, "bg", "-j", "fake:1", "--all", "--name", "t", "--", "true", timeout=120)
        assert r.returncode == 207, r.stderr
        r = rg(env, "wait", "-j", "fake:1", "t", "--timeout", "30", timeout=120)
        assert r.returncode == 207, r.stdout + r.stderr
        assert "rank=1 node=n1 state=NOT_STARTED" in r.stdout
    finally:
        stop_holder(holder)


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "unpushed-commit"])
def test_push_new_dir_is_the_source_tree_with_its_git(env, tmp_path, seeded):
    src = repo(tmp_path)
    if not seeded:  # the origin lacks HEAD, so the clone cannot reach it
        git(src, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "local", "--allow-empty")
    dst = tmp_path / "home/work/dst"
    r = rg(env, "push", str(src), str(dst))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in dst.iterdir()) == [".git", "kept", "new"]
    assert git(dst, "rev-parse", "HEAD") == git(src, "rev-parse", "HEAD")


def test_push_on_the_login_node_keeps_git(env, tmp_path):
    src = repo(tmp_path)
    dst = tmp_path / "home/work/dst"
    r = rg(dict(env, REMOTE_GPU_ON_LOGIN="1"), "push", str(src), str(dst))
    assert r.returncode == 0, r.stderr
    assert git(dst, "rev-parse", "HEAD") == git(src, "rev-parse", "HEAD")


def test_pull_new_dir_is_the_remote_tree(env, tmp_path):
    src = repo(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    r = rg(env, "pull", str(src), str(out))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (out / "src").iterdir()) == [".git", "kept", "new"]


@pytest.mark.parametrize("cred", ["url", "extraheader"])
def test_push_never_copies_a_git_that_holds_credentials(env, tmp_path, cred):
    src = repo(tmp_path)
    if cred == "url":
        git(src, "remote", "set-url", "origin", "https://u:tok@example.invalid/x.git")
    else:
        git(src, "config", "http.https://example.invalid/.extraheader", "AUTHORIZATION: bearer tok")
    dst = tmp_path / "home/work/dst"
    r = rg(env, "push", str(src), str(dst))
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in dst.iterdir()) == ["kept", "new"]


def submitted(env, cluster="fake"):
    root = Path(env["HOME"]) / "scheduler" / cluster
    return [p.read_text().splitlines() for p in sorted(root.glob("*.args"))]


def configure(env, cluster="fake", *, aggressive=False, state=None):
    home = Path(env["HOME"])
    directory = home / ".remote-gpu/clusters" / cluster
    directory.mkdir(exist_ok=True)
    (directory / "cluster.env").write_text(CLUSTER_ENV + ("ALLOC_AGGRESSIVE=1\n" if aggressive else ""))
    (directory / "cluster.sh").write_text(CLUSTER_SH)
    if state:
        (home / f"state_{cluster}").write_text(state)
    return directory


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--no-wait"], [("roomy", "qroomy")]),
        (["-A", "first"], [("first", "qfirst")]),
        (["-q", "qroomy"], [("roomy", "qroomy")]),
        (["-q", "burst"], [("first", "burst")]),
        (["--no-wait", "--preemptible"], [("roomy", "burst")]),
        ([], [("first", "qfirst"), ("roomy", "qroomy")]),
        (["--aggressive"], [("first", "qfirst"), ("roomy", "qroomy")]),
        (["--preemptible"], [("first", "qfirst"), ("roomy", "qroomy"), ("first", "burst")]),
        (["--aggressive", "-A", "roomy"], [("roomy", "qroomy")]),
        (["--aggressive", "--no-wait"], [("roomy", "qroomy")]),
    ],
)
def test_alloc_selects_pairs_and_races_in_shared_policy(env, args, expected):
    configure(env, aggressive=True)
    result = rg(env, "alloc", *args, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [tuple(a[4:6]) for a in submitted(env)] == expected
    assert result.stderr.count("SUBMITTED job=") == len(expected)
    if len(expected) > 1:
        assert "RACE kept=fake:1" in result.stderr
        assert "fake:1" not in (Path(env["HOME"]) / "cancelled").read_text()


def test_aggressive_is_supported_when_not_the_cluster_default(env):
    result = rg(env, "alloc", "--aggressive")
    assert result.returncode == 0, result.stderr
    assert len(submitted(env)) == 2


def test_alloc_without_a_cluster_races_every_cluster(env):
    configure(env, "slow", state="PENDING")
    result = subprocess.run(
        [str(RG), "alloc", "--aggressive"], env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "job=fake:1" in result.stdout
    assert "RACE kept=fake" in result.stderr
    cancelled = (Path(env["HOME"]) / "cancelled").read_text()
    for job in range(1, len(submitted(env, "slow")) + 1):
        assert f"slow:{job} " in cancelled


def test_wait_ends_a_rank_that_never_started_once_the_job_is_gone(env, tmp_path):
    td = tmp_path / "home/work/fake/jobs/1/tasks"
    (td / "t.claim").mkdir(parents=True)
    (td / "t.claim/ranks").write_text("0 n0\n1 n1\n")
    (td / "t.0.pid").write_text("n0 1\n")
    (td / "t.0.rc").write_text("0\n")
    (tmp_path / "home/jobstate").write_text("COMPLETED\n")
    r = rg(env, "wait", "-j", "fake:1", "t", "--timeout", "1")
    assert r.returncode == 207, r.stdout + r.stderr
    assert "rank=1 node=n1 state=NOT_STARTED" in r.stdout


def test_pull_never_overwrites_or_brings_back_a_git_it_should_not(env, tmp_path):
    src = repo(tmp_path)
    out = tmp_path / "out"
    (out / "src/.git").mkdir(parents=True)
    (out / "src/.git/mine").write_text("local\n")
    assert rg(env, "pull", str(src), str(out)).returncode == 0
    assert sorted(p.name for p in (out / "src/.git").iterdir()) == ["mine"]
    git(src, "remote", "set-url", "origin", "https://u:tok@example.invalid/x.git")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert rg(env, "pull", str(src), str(fresh)).returncode == 0
    assert sorted(p.name for p in (fresh / "src").iterdir()) == ["kept", "new"]


def test_an_interrupted_cluster_race_releases_every_submitted_job(env):
    configure(env, state="PENDING", aggressive=True)
    configure(env, "slow", state="PENDING", aggressive=True)
    process = subprocess.Popen([str(RG), "alloc"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len(submitted(env)) == 2 and len(submitted(env, "slow")) == 2:
                break
            time.sleep(0.1)
        else:
            pytest.fail("race did not submit candidates")
        process.terminate()
        process.wait(timeout=30)
        cancelled = (Path(env["HOME"]) / "cancelled").read_text()
        for cluster in ("fake", "slow"):
            for job in (1, 2):
                assert f"{cluster}:{job} " in cancelled
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.parametrize("signal_number", [signal.SIGINT, signal.SIGTERM])
def test_interrupted_account_race_cancels_its_candidates(env, signal_number):
    configure(env, aggressive=True, state="PENDING")
    process = subprocess.Popen(
        [str(RG), "--cluster", "fake", "alloc"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while len(submitted(env)) < 2 and time.monotonic() < deadline:
            time.sleep(0.1)
        assert len(submitted(env)) == 2
        os.killpg(process.pid, signal_number)
        process.communicate(timeout=15)
        cancelled = Path(env["HOME"]) / "cancelled"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if cancelled.exists() and all(f"fake:{j} " in cancelled.read_text() for j in (1, 2)):
                break
            time.sleep(0.1)
        else:
            pytest.fail("account race left a submitted candidate")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


@pytest.mark.parametrize("team_starts", [True, False])
def test_burst_winner_waits_ten_seconds_and_yields_to_team(env, team_starts):
    configure(env, aggressive=True)
    for account, qos in (("first", "qfirst"), ("roomy", "qroomy")):
        (Path(env["HOME"]) / f"state_{account}:{qos}").write_text("PENDING")
    if team_starts:
        env["FAKE_START_TEAM_AFTER"] = "2"
    start = time.monotonic()
    result = rg(env, "alloc", "--preemptible", timeout=20)
    assert result.returncode == 0, result.stderr
    assert f"RACE kept=fake:{1 if team_starts else 3}" in result.stderr
    if not team_starts:
        assert time.monotonic() - start >= 10


@pytest.mark.parametrize("preemptible", [False, True])
def test_no_wait_picks_capacity_and_its_pair_once(env, preemptible):
    configure(env, "roomier")
    home = Path(env["HOME"])
    (home / "capacity_fake").write_text("OK cluster=fake free=1 account=first qos=qfirst burst=5\n")
    (home / "capacity_roomier").write_text("OK cluster=roomier free=4 account=roomy qos=qroomy burst=0\n")
    result = subprocess.run(
        [str(RG), "alloc", "--no-wait", "-N", "2", *(["--preemptible"] if preemptible else [])],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    cluster = "fake" if preemptible else "roomier"
    assert f"PICK cluster={cluster}" in result.stderr
    assert submitted(env, cluster)[0][4:6] == (["first", "burst"] if preemptible else ["roomy", "qroomy"])
    assert not submitted(env, "roomier" if preemptible else "fake")


def test_no_free_nodes_submits_nothing(env):
    result = subprocess.run(
        [str(RG), "alloc", "--no-wait", "-N", "9"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 210
    assert "ERR code=NO_FREE_NODES" in result.stderr
    assert not submitted(env)


@pytest.mark.parametrize("args", [["-A", "unknown"], ["-A", "first", "-q", "qroomy"]])
def test_alloc_rejects_unlisted_pairs_before_submit(env, args):
    result = rg(env, "alloc", *args)
    assert result.returncode == 200
    assert "not a pair in ACCOUNTS" in result.stderr
    assert not submitted(env)


def test_alloc_passthrough_values_are_not_parsed_as_policy(env):
    result = rg(env, "alloc", "--no-wait", "--constraint", "--aggressive", "--reservation", "two words")
    assert result.returncode == 0, result.stderr
    assert len(submitted(env)) == 1
    assert submitted(env)[0][8:] == ["--constraint", "--aggressive", "--reservation", "two words"]


@pytest.mark.parametrize(
    ("command", "reason", "expected_rc", "cancelled"),
    [
        ("alloc", "Resources", 202, True),
        ("alloc", "QOSMaxJobsPerUserLimit", 201, True),
        ("ready", "Resources", 205, False),
        ("ready", "PartitionTimeLimit", 206, True),
    ],
)
def test_wait_policy_uses_query_reason_and_preserves_ready_queue(
    env, command, reason, expected_rc, cancelled
):
    configure(env, state="PENDING")
    env["FAKE_REASON"] = reason
    args = [command, "--wait", "0"]
    if command == "ready":
        args += ["-j", "fake:1"]
    result = rg(env, *args, timeout=10)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    assert (Path(env["HOME"]) / "cancelled").exists() == cancelled
    if command == "ready" and not cancelled:
        assert "start_estimate=tomorrow" in result.stdout


def test_bad_health_cancels_the_account_race_winner_too(env):
    configure(env, aggressive=True)
    result = rg(dict(env, FAKE_GPUS="1"), "alloc", timeout=10)
    assert result.returncode == 204, result.stderr
    cancelled = (Path(env["HOME"]) / "cancelled").read_text()
    assert "fake:1 " in cancelled and "fake:2 " in cancelled


def test_alloc_keeps_a_running_winner_whose_nodes_cannot_be_resolved(env):
    result = rg(dict(env, FAKE_NODES=""), "alloc", timeout=10)
    assert result.returncode == 206, result.stderr
    assert "ERR code=NODES_UNKNOWN" in result.stderr
    assert not (Path(env["HOME"]) / "cancelled").exists()


def test_release_waits_for_holder_markers_and_checks_ownership(env):
    directory = Path(env["HOME"]) / "work/fake/jobs/1"
    directory.mkdir(parents=True)
    for node in ("n0", "n1"):
        (directory / f"cleaned.{node}").touch()
    result = rg(dict(env, FAKE_OWNER="another-user"), "release", "-j", "fake:1")
    assert result.returncode == 0
    assert not (Path(env["HOME"]) / "cancelled").exists()
    result = rg(env, "release", "-j", "fake:1", timeout=10)
    assert result.returncode == 0, result.stderr
    assert (directory / "release").exists()
    assert "OK released job=fake:1" in result.stdout


@pytest.mark.parametrize("hook", ["scheduler_submit", "scheduler_holder", "scheduler_extend"])
def test_missing_adapter_hook_fails_before_connecting(env, hook):
    adapter = Path(env["HOME"]) / ".remote-gpu/clusters/fake/cluster.sh"
    adapter.write_text(CLUSTER_SH.replace(f"{hook}()", "omitted_hook()"))
    result = rg(env, "free")
    assert result.returncode == 200
    assert hook in result.stderr


@pytest.mark.parametrize("command", ["exec", "bg"])
def test_exec_bg_envelope_preserves_env_and_command_argv(env, command):
    import json

    holder = start_holder(env) if command == "bg" else None
    directory = Path(env["HOME"]) / "work/dir with spaces"
    directory.mkdir(parents=True)
    value = "quotes '\" dollars $() bang ! and spaces"
    code = "import json,os,sys; print(json.dumps([os.getcwd(),os.environ['VALUE'],sys.argv[1:]]))"
    try:
        result = rg(
            env,
            command,
            "-j",
            "fake:1",
            "-C",
            "dir with spaces",
            "-e",
            f"VALUE={value}",
            "--",
            "python3",
            "-c",
            code,
            value,
            "",
            "--no-wait",
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        if command == "bg":
            task = re.search(r"task=([^ ]+)", result.stdout)[1]
            result = rg(env, "wait", "-j", "fake:1", task, "--timeout", "5", timeout=15)
            assert result.returncode == 0, result.stderr
        assert json.dumps([str(directory), value, [value, "", "--no-wait"]]) in result.stdout
    finally:
        if holder:
            stop_holder(holder)


@pytest.mark.parametrize("cause", ["release", "time-limit", "idle"])
def test_shared_holder_cleans_up_and_exits(env, cause):
    home = Path(env["HOME"])
    directory = home / "work/fake/jobs/1"
    directory.mkdir(parents=True)
    if cause == "release":
        (directory / "release").touch()
    else:
        env["FAKE_MAINTENANCE"] = "1"
    if cause == "time-limit":
        env["FAKE_END"] = str(int(time.time()) + 30)
    if cause == "idle":
        # Only stat of the heartbeat is old; the holder's start still refreshes the file.
        bindir = Path(env["PATH"].split(":")[0])
        real_stat = shutil.which("stat")
        (bindir / "stat").write_text(
            f'#!/bin/bash\nif [[ $* == *heartbeat* ]]; then echo $(($(date +%s) - 120)); else exec {real_stat} "$@"; fi\n'
        )
        (bindir / "stat").chmod(0o755)
    holder = start_holder(env, idle=1)
    try:
        assert holder.wait(timeout=10) == 0
        assert (directory / "cleaned.n0").exists()
        assert f"cleanup ({cause})" in (home / "work/fake/logs/1.n0.log").read_text()
        assert not (home / "cancelled").exists()
    finally:
        stop_holder(holder)


def test_content_addressed_holder_is_reused_and_foreign_file_replaced(env):
    assert rg(env, "alloc", "--no-wait").returncode == 0
    holder = Path(submitted(env)[0][6])
    original = holder.read_bytes()
    inode = holder.stat().st_ino
    assert rg(env, "alloc", "--no-wait").returncode == 0
    assert holder.stat().st_ino == inode
    holder.unlink()
    holder.symlink_to("/bin/true")
    assert rg(env, "alloc", "--no-wait").returncode == 0
    assert not holder.is_symlink()
    assert holder.read_bytes() == original


def test_work_root_skips_a_writable_directory_not_owned_by_user(env, tmp_path):
    foreign = tmp_path / "foreign"
    owned = tmp_path / "owned"
    foreign.mkdir()
    (foreign / env["USER"]).symlink_to("/tmp", target_is_directory=True)
    (owned / env["USER"]).mkdir(parents=True)
    bindir = Path(env["PATH"].split(":")[0])
    (bindir / "findmnt").write_text(f"#!/bin/bash\nprintf '%s\\n' '100 {foreign}' '50 {owned}'\n")
    (bindir / "findmnt").chmod(0o755)
    manifest = Path(env["HOME"]) / ".remote-gpu/clusters/fake/cluster.env"
    manifest.write_text(CLUSTER_ENV.replace("WORK_ROOT=$HOME/work", "WORK_ROOT=auto"))
    result = rg(env, "free")
    assert result.returncode == 0, result.stderr
    assert (Path(env["HOME"]) / ".remote-gpu/fake.work_root").read_text().strip() == str(owned / env["USER"])


@pytest.mark.parametrize("direction", ["push", "pull"])
def test_seeded_sync_deletes_source_deletions_and_keeps_source_commit(env, tmp_path, direction):
    source = repo(tmp_path)
    target = tmp_path / "destination"
    source_arg = str(source) + ("/" if direction == "pull" else "")
    result = rg(env, direction, source_arg, str(target))
    assert result.returncode == 0, result.stderr
    assert (Path(env["HOME"]) / "clone_calls").exists()
    assert sorted(p.name for p in target.iterdir()) == [".git", "kept", "new"]
    assert git(target, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")


@pytest.mark.parametrize("direction", ["push", "pull"])
@pytest.mark.parametrize("credential", ["url", "extraheader", "pushurl"])
def test_sync_blocks_credentials_but_login_transport_can_copy_git(env, tmp_path, direction, credential):
    source = repo(tmp_path)
    if credential == "url":
        git(source, "remote", "set-url", "origin", "https://u:tok@example.invalid/repo")
    elif credential == "pushurl":
        git(source, "config", "remote.origin.pushurl", "https://u:tok@example.invalid/repo")
    else:
        git(source, "config", "http.extraheader", "Authorization: bearer tok")
    args = [direction, str(source) + ("/" if direction == "pull" else "")]
    if direction == "push":
        args += ["--no-exclude"]
    result = rg(env, *args, str(tmp_path / "remote"))
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "remote/.git").exists()
    result = rg(dict(env, REMOTE_GPU_ON_LOGIN="1"), *args, str(tmp_path / "login"))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "login/.git").exists()


def test_push_no_exclude_keeps_existing_git(env, tmp_path):
    source = repo(tmp_path)
    target = tmp_path / "destination"
    (target / ".git").mkdir(parents=True)
    (target / ".git/mine").touch()
    result = rg(env, "push", "--no-exclude", "--delete", str(source), str(target))
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (target / ".git").iterdir()) == ["mine"]


@pytest.mark.parametrize("origin", ["-bad", "file:///tmp/no-repo", "/tmp/no-repo", "ext::sh -c evil"])
def test_git_seed_rejects_options_and_non_network_transports(env, tmp_path, origin):
    source = repo(tmp_path)
    git(source, "config", "remote.origin.url", origin)
    result = rg(env, "push", str(source), str(tmp_path / "target"))
    assert result.returncode == 0, result.stderr
    calls = Path(env["HOME"]) / "clone_calls"
    assert not calls.exists(), calls.read_text() if calls.exists() else ""


def test_all_embedded_remote_scripts_parse():
    for path in (RG, RG.parents[1] / "templates/slurm.sh"):
        text = path.read_text()
        for match in re.finditer(r"read -r -d '' (\w+) <<'([^']+)'\n(.*?)\n\2\n", text, re.S):
            result = subprocess.run(["bash", "-n"], input=match[3], capture_output=True, text=True)
            assert result.returncode == 0, f"{path.name}:{match[1]}: {result.stderr}"


def embedded_script(path, name):
    match = re.search(r"read -r -d '' " + name + r" <<'([^']+)'\n(.*?)\n\1\n", path.read_text(), re.S)
    return match[2]


def test_slurm_query_normalizes_nodes_start_estimate_and_pending_count(env):
    library = embedded_script(RG.parents[1] / "templates/slurm.sh", "CLUSTER_RLIB")
    script = r"""
C=fake REL=$HOME B=$HOME/work HOME_FULL=0 DOCKER_FLAGS=''
squeue() {
  if [[ $1 == --start ]]; then echo 2030-01-02T03:04:05; return; fi
  echo 'PENDING|n[0-1]|01:00:00|Resources|N/A|compute|roomy|qroomy|4:00:00|user|name|00:00:00|N/A|2'
}
scontrol() { printf '%s\n' n0 n1; }
"""
    result = subprocess.run(
        ["bash", "-c", script + library + "\nscheduler_query 1"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    fields = result.stdout.strip().split("|")
    assert fields[:5] == ["PENDING", "n0,n1", "01:00:00", "Resources", "2030-01-02T03:04:05"]
    assert fields[-2:] == ["", "2"]


def test_slurm_submit_passes_the_selected_pair_holder_and_scheduler_options(env):
    library = embedded_script(RG.parents[1] / "templates/slurm.sh", "CLUSTER_RLIB")
    script = r"""
C=fake REL=$HOME B=$HOME/work HOME_FULL=0 DOCKER_FLAGS='' GPUS_PER_NODE=8
sbatch() { printf '%s\n' "$@" >"$HOME/submit_args"; echo '42;site'; }
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            script
            + library
            + '\nscheduler_submit 2 1:00:00 label compute roomy qroomy holder 7 --constraint "two words"',
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "42"
    arguments = (Path(env["HOME"]) / "submit_args").read_text().splitlines()
    for flag, value in (
        ("-A", "roomy"),
        ("-q", "qroomy"),
        ("-N", "2"),
        ("--gpus-per-node", "8"),
        ("--constraint", "two words"),
    ):
        assert arguments[arguments.index(flag) + 1] == value
    assert "--no-requeue" in arguments
    assert arguments[-5:] == ["holder", "7", "holder", str(Path(env["HOME"]) / "work"), "fake"]


def test_git_seed_rejects_a_non_sha_without_invoking_git(env):
    library = embedded_script(RG, "SYNCLIB")
    result = subprocess.run(
        [
            "bash",
            "-c",
            "warn() { :; }\n" + library + '\ngit_seed "$HOME/new" https://example.invalid/repo not-a-sha',
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert not (Path(env["HOME"]) / "clone_calls").exists()


def test_mount_replaces_the_default_mounts(env):
    home = env["HOME"]
    keys = Path(home) / "keys"
    (Path(home) / "work/run/worker").mkdir(parents=True)
    keys.mkdir()
    result = rg(
        env,
        "exec",
        "-j",
        "fake:1",
        "--image",
        "img",
        "--mount",
        "run/worker",
        "--mount",
        f"{keys}:ro",
        "--",
        "true",
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    call = (Path(home) / "docker_calls").read_text().split()
    mounts = [call[i + 1] for i, arg in enumerate(call) if arg == "-v"]
    assert mounts == [f"{home}/work/run/worker:{home}/work/run/worker", f"{keys}:{keys}:ro"]
    assert f"HOME={home}/work/run/worker" in call
    assert rg(env, "exec", "-j", "fake:1", "--mount", "x", "--", "true").returncode != 0


def test_mount_refuses_a_directory_that_does_not_exist(env):
    # docker -v would create it owned by root, and the user could no longer write it.
    result = rg(
        env, "exec", "-j", "fake:1", "--image", "img", "--mount", "run/missing", "--", "true", timeout=15
    )
    assert result.returncode != 0
    assert "run/missing" in result.stderr
    assert not (Path(env["HOME"]) / "docker_calls").exists()
    result = rg(env, "exec", "-j", "fake:1", "--image", "img", "-C", "run/missing", "--", "true", timeout=15)
    assert result.returncode != 0
    assert not (Path(env["HOME"]) / "docker_calls").exists()


def test_mount_refuses_extra_mounts_from_the_manifest(env):
    (Path(env["HOME"]) / "work/run/worker").mkdir(parents=True)
    directory = configure(env)
    env_file = directory / "cluster.env"
    env_file.write_text(
        env_file.read_text().replace("DOCKER_FLAGS=''", "DOCKER_FLAGS='--privileged -v /validate:/validate'")
    )
    result = rg(
        env, "exec", "-j", "fake:1", "--image", "img", "--mount", "run/worker", "--", "true", timeout=15
    )
    assert result.returncode != 0
    assert "DOCKER_FLAGS" in result.stderr
    assert not (Path(env["HOME"]) / "docker_calls").exists()


def test_account_race_reports_a_loser_it_could_not_cancel(env):
    configure(env, aggressive=True)
    env["FAKE_CANCEL_FAILS"] = "1"
    result = rg(env, "alloc", timeout=30)
    assert 'cancelled=""' in result.stderr
    assert re.search(r"CANCEL_FAILED job=fake:2\b", result.stderr), result.stderr


def test_holder_leaves_no_clean_marker_while_a_container_survives(env):
    home = Path(env["HOME"])
    bindir = Path(env["PATH"].split(":")[0])
    # Removal fails, so the job's container stays listed.
    (bindir / "docker").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >>"$HOME/docker_calls"\n'
        'case $1 in ps) [[ $* == *"label=fake.job=1"* ]] && echo c1 ;; stop | rm) exit 1 ;; esac\nexit 0\n'
    )
    directory = home / "work/fake/jobs/1"
    directory.mkdir(parents=True)
    (directory / "release").touch()
    holder = start_holder(env)
    try:
        holder.wait(timeout=60)
        assert not (directory / "cleaned.n0").exists()
        assert "cleanup incomplete" in (home / "work/fake/logs/1.n0.log").read_text()
    finally:
        stop_holder(holder)
