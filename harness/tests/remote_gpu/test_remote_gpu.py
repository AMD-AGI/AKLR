"""remote-gpu against a fake two-node cluster on this machine: no scheduler, no ssh, no GPU.

The fake `ssh` runs its command here, so the paths "on the cluster" are local paths. Node n0
has a holder that starts bg tasks; n1 has none, so a task never starts there.
"""

import os
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
"""

CLUSTER_SH = r"""
read -r -d '' CLUSTER_RLIB <<'EOF'
SQUEUE=fake_squeue SCANCEL=true LATE_HINT=x DOCKER_X=() NODE_ENV=()
fake_squeue() { local f s; while (($#)); do [[ $1 == -o ]] && f=$2; shift; done
  s=$(cat "$HOME/jobstate" 2>/dev/null || echo RUNNING)
  [[ $f == *%L* ]] && echo "$s|01:00:00" || echo "$s"; }
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


def test_alloc_on_a_named_cluster_uses_the_account_free_reports(env, tmp_path):
    sh = tmp_path / "home/.remote-gpu/clusters/fake/cluster.sh"
    sh.write_text(
        sh.read_text()
        + """
read -r -d '' R_FREE <<'EOF'
echo "OK cluster=$C free=0 account=roomy idle=0"
EOF
cmd_alloc() { echo "ALLOC $*"; }
"""
    )
    assert rg(env, "alloc", "--no-wait").stdout.strip() == "ALLOC --no-wait -A roomy"
    assert rg(env, "alloc", "-A", "mine").stdout.strip() == "ALLOC -A mine"


def test_alloc_races_the_team_accounts_where_the_cluster_can(env, tmp_path):
    c = tmp_path / "home/.remote-gpu/clusters/fake"
    (c / "cluster.env").write_text(CLUSTER_ENV + "ALLOC_AGGRESSIVE=1\n")
    (c / "cluster.sh").write_text(
        CLUSTER_SH
        + """
read -r -d '' R_FREE <<'EOF'
echo "OK cluster=$C free=1 account=roomy idle=1"
EOF
cmd_alloc() { echo "ALLOC $*"; }
"""
    )
    auto = subprocess.run([str(RG), "alloc"], env=env, capture_output=True, text=True)
    assert auto.stdout.strip() == "ALLOC --aggressive"
    assert rg(env, "alloc").stdout.strip() == "ALLOC --aggressive"
    assert rg(env, "alloc", "--preemptible").stdout.strip() == "ALLOC --preemptible --aggressive"
    assert rg(env, "alloc", "--aggressive").stdout.strip() == "ALLOC --aggressive"
    assert rg(env, "alloc", "--no-wait").stdout.strip() == "ALLOC --no-wait -A roomy"
    assert rg(env, "alloc", "-q", "burst").stdout.strip() == "ALLOC -q burst"
    sh = c / "cluster.sh"
    sh.write_text(sh.read_text().replace("free=1", "free=0"))
    auto = subprocess.run([str(RG), "alloc"], env=env, capture_output=True, text=True)
    assert auto.stdout.strip() == "ALLOC --aggressive", auto.stderr


def test_alloc_without_a_cluster_races_every_cluster(env, tmp_path):
    clusters = tmp_path / "home/.remote-gpu/clusters"
    released = tmp_path / "released"
    fake_alloc = {
        "fake": 'cmd_alloc() { sleep 1; echo "OK cluster=fake job=fake:1 args=$*"; }',
        "slow": 'cmd_alloc() { echo "SUBMITTED job=slow:7" >&2; sleep 60; }',
    }
    for name, body in fake_alloc.items():
        (clusters / name).mkdir(exist_ok=True)
        (clusters / name / "cluster.env").write_text(CLUSTER_ENV)
        (clusters / name / "cluster.sh").write_text(
            CLUSTER_SH
            + f"""
read -r -d '' R_RELEASE <<'EOF'
echo "$C $*" >>{released}
EOF
{body}
"""
        )
    r = subprocess.run(
        [str(RG), "alloc", "--aggressive"], env=env, capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK cluster=fake job=fake:1")
    assert "RACE kept=fake" in r.stderr
    assert released.read_text().split() == ["slow", "0", "7"]


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


def test_template_alloc_takes_the_qos_paired_with_the_account(env, tmp_path):
    c = tmp_path / "home/.remote-gpu/clusters/fake"
    (c / "cluster.env").write_text(
        CLUSTER_ENV + "ACCOUNTS='first:qfirst roomy:qroomy'\nPARTITION=''\nBURST_QOS=''\n"
    )
    template = (RG.parents[1] / "templates/slurm.sh").read_text()
    (c / "cluster.sh").write_text(template + '\nremote() { shift; echo "SUBMIT $*"; }\n')
    r = rg(env, "alloc", "-A", "roomy")
    assert "-A roomy -q qroomy" in r.stdout, r.stdout + r.stderr


def test_an_interrupted_cluster_race_releases_every_submitted_job(env, tmp_path):
    clusters = tmp_path / "home/.remote-gpu/clusters"
    released = tmp_path / "released"
    for name, job in (("fake", 3), ("slow", 7)):
        (clusters / name).mkdir(exist_ok=True)
        (clusters / name / "cluster.env").write_text(CLUSTER_ENV)
        (clusters / name / "cluster.sh").write_text(
            CLUSTER_SH
            + f"""
read -r -d '' R_RELEASE <<'EOF'
echo "$C $*" >>{released}
EOF
cmd_alloc() {{ echo "SUBMITTED job={name}:{job}" >&2; sleep 60; }}
"""
        )
    p = subprocess.Popen([str(RG), "alloc"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)
    p.terminate()
    p.wait(timeout=30)
    assert sorted(released.read_text().splitlines()) == ["fake 0 3", "slow 0 7"]
