"""remote-gpu against a fake two-node cluster on this machine: no scheduler, no ssh, no GPU.

The fake `ssh` runs its command here, so the paths "on the cluster" are local paths. Node n0
has a holder that starts bg tasks; n1 has none, so a task never starts there.
"""

import os
import subprocess
import textwrap
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
"""

CLUSTER_SH = r"""
read -r -d '' CLUSTER_RLIB <<'EOF'
SQUEUE=fake_squeue SCANCEL=true LATE_HINT=x DOCKER_X=() NODE_ENV=()
fake_squeue() { local f; while (($#)); do [[ $1 == -o ]] && f=$2; shift; done
  [[ $f == *%L* ]] && echo 'RUNNING|01:00:00' || echo RUNNING; }
jfinal() { echo COMPLETED; }
nodes_of() { printf '%s\n' n0 n1; }
on_node() { local n=$2; shift 2; ${ONTO:+timeout $ONTO} "$@" </dev/null; }
EOF
"""

FAKE_SSH = """\
#!/bin/bash
while [[ $1 == -* ]]; do case $1 in -o) shift 2 ;; *) shift ;; esac; done
shift; exec bash -c "$*"
"""

HOLDER = r"""
D=$1
while :; do
  for r in "$D"/spawn/n0.*; do
    [ -e "$r" ] || continue
    mapfile -d '' a <"$r"; rm -f "$r"; F=${a[0]}
    setsid nohup bash "${a[1]}" "$D" "$F" "${a[@]:2}" >"$F.log" 2>&1 </dev/null &
    echo "n0 $!" >"$F.pid"
  done
  sleep 0.5
done
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
    # The test's origins are local paths; remote-gpu allows only network transports.
    e.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="protocol.file.allow", GIT_CONFIG_VALUE_0="always")
    return e


def rg(env, *args, **kw):
    return subprocess.run([str(RG), "--cluster", "fake", *args], env=env, capture_output=True, text=True, **kw)


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


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
    (src / "gone").unlink()
    (src / "new").write_text("new\n")
    return src


def test_wait_fails_when_a_rank_never_started(env, tmp_path):
    jd = tmp_path / "home/work/fake/jobs/1"
    (jd / "spawn").mkdir(parents=True)
    holder = subprocess.Popen(["bash", "-c", HOLDER, "_", str(jd)])
    try:
        r = rg(env, "bg", "-j", "fake:1", "--all", "--name", "t", "--", "true", timeout=120)
        assert r.returncode == 207, r.stderr
        r = rg(env, "wait", "-j", "fake:1", "t", "--timeout", "30", timeout=120)
        assert r.returncode == 207, r.stdout + r.stderr
        assert "rank=1 node=n1 state=NOT_STARTED" in r.stdout
    finally:
        holder.kill()


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
