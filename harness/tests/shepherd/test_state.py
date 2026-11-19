"""Exercise round leases, takeover, due checks, patches, archiving and migration through the public CLI."""

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

    def __init__(self, root, config, tmp_path, now=True, observation=None):
        self.root, self.tmp_path = root, tmp_path
        options = [*(["--now"] if now else []), *(["--observation", observation] if observation else [])]
        self.started = success(root, "begin", "--config", str(config), *options)
        self.id = self.started.get("round")

    def save(self, patch, revision=None):
        source = self.tmp_path / "next.json"
        source.write_text(json.dumps(patch))
        revision = success(self.root, "show")["revision"] if revision is None else revision
        return call(
            self.root, "save", "--round", self.id, "--revision", str(revision), "--patch", str(source)
        )

    def saves(self, patch):
        return self.save(patch).returncode == 0

    def refuses(self, patch, why):
        """The save fails, for the reason named."""
        result = self.save(patch)
        return result.returncode != 0 and why in result.stderr

    def end(self, code=0):
        return success(self.root, "end", "--round", self.id, "--exit-code", str(code))


BOUNDS = {"model_usd": 200, "deadline": "2100-01-01T00:00:00+00:00"}


def campaign(id_="c1", phase="admitted", model_usd=200, gpu=0, **fields):
    bounds = {**BOUNDS, "model_usd": model_usd}
    return {"id": id_, "phase": phase, "pool": "p1", "gpu": gpu, "bounds": bounds, **fields}


def pool(id_="p1", node_hours=24, **fields):
    """A started pool: it has its end."""
    return {"id": id_, "bounds": {"node_hours": node_hours}, "end": "2100-01-02T00:00:00+00:00", **fields}


def admit(*campaigns):
    """A patch that admits the campaigns onto the started pool."""
    return {"campaigns": by_id(*campaigns), "pools": by_id(pool())}


def by_id(*entries):
    """Entries as a patch addresses them: each under its id."""
    return {entry["id"]: entry for entry in entries}


def state(root):
    return success(root, "show")["state"]


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


def test_end_releases_the_ledger_and_only_the_owner_saves(root, config, tmp_path):
    first = Round(root, config, tmp_path)
    assert first.save({}).returncode == 0
    first.end()
    assert owner(root) is None
    assert json.loads((root / "last_round.json").read_text())["exit_code"] == 0
    # The round is over and its id is no longer the owner's.
    assert first.save({}).returncode != 0
    second = Round(root, config, tmp_path)
    assert second.started["recovered_from"] is None
    assert call(root, "end", "--round", first.id, "--exit-code", "0").returncode != 0


def test_expired_lease_is_fenced_and_recovered(root, config, tmp_path):
    stale = Round(root, config, tmp_path)
    expire(root)
    rejected = stale.save({})
    assert rejected.returncode != 0 and "lease expired" in rejected.stderr
    wake(root, timedelta(minutes=30))
    taken = Round(root, config, tmp_path, now=False)
    assert taken.started["recovered_from"] == stale.id
    assert stale.save({}).returncode != 0
    assert taken.save({}).returncode == 0


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
    assert current.save({}, revision=0).returncode == 0
    assert current.save({}, revision=0).returncode != 0
    before = (root / "state.json").read_bytes()
    for content in ("{broken", "[]", '{"bad": NaN}'):
        (tmp_path / "next.json").write_text(content)
        result = call(
            root, "save", "--round", current.id, "--revision", "1", "--patch", str(tmp_path / "next.json")
        )
        assert result.returncode != 0, content
    assert (root / "state.json").read_bytes() == before


def test_an_id_is_one_remote_gpu_takes_as_a_task_name(root, config, tmp_path):
    # worker.md starts each step as `remote-gpu bg --name <campaign>-<step>`, which takes [A-Za-z0-9_-].
    current = Round(root, config, tmp_path)
    assert current.refuses(admit(campaign("bmm.20261116")), "must be letters")
    assert current.saves(admit(campaign("bmm_n128-20261116")))


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
    assert current.save({"pools": by_id(pool()), kind: by_id(entry)}).returncode == 0
    for bad in (
        {kind: {entry["id"]: None}},
        {kind: None},
        {kind: [entry]},
        {kind: {entry["id"]: {"bounds": {key: 1 for key in entry["bounds"]}}}},
        {kind: {entry["id"]: {"bounds": None}}},
        {kind: {"x": {"phase": "admitted"}}},
        {kind: {"../x": entry}},
    ):
        assert current.save(bad).returncode != 0, bad
    assert state(root)[kind] == [entry]


def test_a_patch_merges_into_the_state_and_null_removes_a_key(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.save({**admit(campaign(evidence={"head": "a", "base": "b"})), "note": 1}).returncode == 0
    patch = {"campaigns": {"c1": {"evidence": {"head": None, "report": "r"}}}, "note": None, "wake_at": "t"}
    assert current.save(patch).returncode == 0
    assert state(root) == {
        "campaigns": [campaign(evidence={"base": "b", "report": "r"})],
        "pools": [pool()],
        "last_main_sha": None,
        "wake_at": "t",
    }


def test_an_operation_saves_its_intent_before_its_result_and_stays(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    done = {"operations": {"loop": {"intent_at": "t", "result": "ok"}}}
    assert current.save({"campaigns": by_id(campaign(**done))}).returncode != 0
    assert current.save(admit(campaign(operations={"loop": {"intent_at": "t"}}))).returncode == 0
    assert current.save({"campaigns": {"c1": {"operations": {"loop": None}}}}).returncode != 0
    assert current.save({"campaigns": {"c1": {"operations": [{"id": "loop"}]}}}).returncode != 0
    assert current.save({"campaigns": {"c1": done}}).returncode == 0
    assert current.save({"pools": by_id(pool(**done))}).returncode != 0


def test_a_finished_entry_moves_to_its_archive(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    full = campaign(schema_path="s.json", evidence={"head": "a"}, operations={"loop": {"intent_at": "t"}})
    assert current.save({"campaigns": by_id(full), "pools": by_id(pool(requests=["r"]))}).returncode == 0
    ending = {"phase": "ended", "outcome": "stopped", "settled_model_usd": 0}
    patch = {"campaigns": {"c1": ending}, "pools": {"p1": {"settled_node_hours": 3}}}
    assert current.save(patch).returncode == 0
    kept = state(root)
    assert kept["campaigns"] == [
        {
            "id": "c1",
            "bounds": BOUNDS,
            "schema_path": "s.json",
            **ending,
            "archive": "runs/c1/campaign.json",
        }
    ]
    assert json.loads((root / "runs/c1/campaign.json").read_text()) == {**full, **ending}
    settled = {"id": "p1", "bounds": {"node_hours": 24}, "settled_node_hours": 3}
    assert kept["pools"] == [{**settled, "archive": "runs/p1/pool.json"}]
    # A settled pool has left the ledger, so nothing more is placed on it.
    assert current.refuses({"campaigns": by_id(campaign("c2", gpu=1))}, "while a campaign holds")
    assert json.loads((root / "runs/p1/pool.json").read_text())["requests"] == ["r"]
    # A finished entry still counts in the budget and takes later notes, such as a lifted cooldown.
    assert current.save({"campaigns": {"c1": {"cooldown_until": "t"}}}).returncode == 0
    assert current.refuses({"campaigns": {"c1": {"bounds": {"model_usd": 1}}}}, "changes its admitted bounds")
    # Only the helper names the archive: a patch that dropped the path would write the short form over it.
    for archive in (None, "runs/c1/other.json"):
        assert current.refuses({"campaigns": {"c1": {"archive": archive}}}, "archive is set by the helper")
    assert current.refuses(
        {"campaigns": by_id(campaign("c2", gpu=1, archive="x"))}, "archive is set by the helper"
    )
    assert json.loads((root / "runs/c1/campaign.json").read_text()) == {**full, **ending}


def test_an_idle_round_is_skipped_for_a_day_while_its_observation_holds(root, config, tmp_path):
    current = Round(root, config, tmp_path, observation="seen")
    assert current.saves(admit(campaign()))
    # Something is still running, so no observation can make the round idle.
    assert current.refuses({"idle_observation": "seen"}, "every campaign ended")
    assert current.saves({"campaigns": {"c1": {"phase": "ended", "settled_model_usd": 0}}})
    assert current.saves({"pools": {"p1": {"settled_node_hours": 1}}})
    # GitHub may have moved since the round began: only what it began with is known to be covered.
    assert current.refuses({"idle_observation": "later"}, "the observation this round began with")
    assert current.saves({"idle_observation": "seen"})
    current.end()
    wake(root, timedelta(hours=20))
    begin = ("begin", "--config", str(config))
    assert "waiting_until" in success(root, *begin, "--observation", "seen")
    wake(root, timedelta(hours=25))
    assert Round(root, config, tmp_path, now=False, observation="seen").id
    success(root, "end", "--round", owner(root)["id"], "--exit-code", "0")
    wake(root, timedelta(hours=20))
    for changed in ("moved", None):
        assert Round(root, config, tmp_path, now=False, observation=changed).id, changed
        success(root, "end", "--round", owner(root)["id"], "--exit-code", "0")
    # A round that began without an observation cannot become idle, and any save ends the idle skip.
    current = Round(root, config, tmp_path)
    assert current.refuses({"idle_observation": "seen"}, "the observation this round began with")
    assert current.saves({"note": 1})
    assert "idle_observation" not in state(root)


def test_a_wake_at_beyond_the_limit_waits_the_limit(root, config, tmp_path):
    # Ignored, it would run a round at every trigger of the longest waits.
    current = Round(root, config, tmp_path)
    assert current.saves({"wake_at": (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()})
    current.end()
    waiting = success(root, "begin", "--config", str(config))["waiting_until"]
    assert datetime.fromisoformat(waiting) <= datetime.now(timezone.utc) + timedelta(hours=2)


def test_a_save_that_sets_no_wake_at_leaves_none(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.saves({"wake_at": (datetime.now(timezone.utc) + timedelta(minutes=100)).isoformat()})
    current.end(1)
    recovering = Round(root, config, tmp_path, now=False)
    # It leaves wake_at out because an action could run now; the earlier round's must not survive it.
    assert recovering.saves({"note": 1})
    recovering.end()
    assert "wake_at" not in state(root)
    assert Round(root, config, tmp_path, now=False).id


OLD_ROUND = {"id": "r0", "pid": 1, "start": "2", "boot_id": "b", "pid_namespace": "n", "config_sha256": "c"}


def old_ledger(root, campaigns, pools=()):
    """A schema 1 ledger as the earlier helper left it, and its text."""
    text = json.dumps({"revision": 7, "state": {"campaigns": campaigns, "pools": list(pools)}})
    (root / "state.json").write_text(text)
    (root / "last_round.json").write_text(json.dumps({**OLD_ROUND, "exit_code": 0}))
    return text


def test_begin_migrates_a_schema_1_ledger_and_keeps_it(root, config, tmp_path):
    loop = {"id": "loop", "intent_at": "t", "result": "ok"}
    before = old_ledger(
        root,
        [
            campaign("c1", phase="delivered", settled={"model_usd": 200}, operations=[loop, loop]),
            campaign("c2", phase="no_gain", gpu=1, settled={"model_usd": 0}, outcome={"reason": "r"}),
            campaign("c3", phase="pr_open", gpu=2, settled={"model_usd": 200}, operations=[loop]),
            campaign("c4", phase="stopped", gpu=3, settled={"model_usd": 0}, outcome="owner stop"),
        ],
        [pool(settled_node_hours=5, operations=[{"id": "alloc", "result": "ok"}])],
    )
    current = Round(root, config, tmp_path)
    assert current.started["recovered_from"] is None
    shown = success(root, "show")
    assert (shown["schema"], shown["revision"]) == (2, 8)
    first, second, third, fourth = shown["state"]["campaigns"]
    assert (first["phase"], first["outcome"], first["settled_model_usd"]) == ("ended", "delivered", 200)
    assert (second["phase"], second["outcome"], second["settled_model_usd"]) == ("ended", "no_gain", 0)
    assert (fourth["phase"], fourth["outcome"]) == ("ended", "owner stop")
    # Nothing an old record held is lost: a repeated operation id, a detailed outcome, a pool's operations.
    assert json.loads((root / "runs/c1/campaign.json").read_text())["operations"] == {
        "loop": loop,
        "loop#1": loop,
    }
    assert json.loads((root / "runs/c2/campaign.json").read_text())["outcome_detail"] == {"reason": "r"}
    assert json.loads((root / "runs/p1/pool.json").read_text())["operations"] == {
        "alloc": {"id": "alloc", "result": "ok"}
    }
    assert third["phase"] == "pr_open" and third["operations"] == {"loop": loop}
    assert shown["state"]["pools"][0]["archive"] == "runs/p1/pool.json"
    assert json.loads((root / "schema-1/state.json").read_text()) == json.loads(before)
    # The migrated ledger takes saves, and the next begin does not migrate again.
    assert current.saves({"campaigns": {"c3": {"phase": "ended", "outcome": "closed"}}})
    current.end()
    Round(root, config, tmp_path).end()
    assert success(root, "show")["revision"] == 9
    (root / "state.json").write_text(json.dumps({**shown, "schema": 3}))
    assert "newer" in call(root, "begin", "--config", str(config)).stderr


@pytest.mark.parametrize(
    "bad, why",
    [
        # Still on its GPU: ending it would free the slot and drop its operations without cleanup.
        (campaign(phase="stopping"), "unknown phase"),
        (campaign(phase="pr_open"), "settled exactly when it leaves its GPU"),
    ],
)
def test_a_schema_1_ledger_that_breaks_a_rule_is_left_alone(root, config, bad, why):
    before = old_ledger(root, [bad])
    result = call(root, "begin", "--config", str(config))
    assert result.returncode != 0 and why in result.stderr
    assert (root / "state.json").read_text() == before
    assert not (root / "schema-1").exists() and owner(root) is None


def test_a_schema_1_round_is_taken_over_and_a_running_round_is_not_migrated_under(root, config, tmp_path):
    before = old_ledger(root, [campaign(phase="delivered", settled={"model_usd": 0})])
    (root / "owner.json").write_text(json.dumps({**OLD_ROUND, "id": "r1", "exit_code": None}))
    assert success(root, "show")["owner"]["id"] == "r1"
    taken = Round(root, config, tmp_path)
    assert taken.started["recovered_from"] == "r1"
    # A trigger that finds the ledger busy leaves it as it is, whatever its schema, and so does the
    # round that holds it: only a begin migrates.
    (root / "state.json").write_text(before)
    assert success(root, "begin", "--config", str(config)) == {"busy": taken.id}
    assert taken.refuses({"note": 1}, "only the next begin migrates")
    assert (root / "state.json").read_text() == before


def test_a_round_whose_end_was_cut_off_after_its_record_has_ended(root, config, tmp_path):
    first = Round(root, config, tmp_path)
    record = owner(root)
    first.end()
    # `end` wrote last_round.json and crashed before releasing the ledger.
    (root / "owner.json").write_text(json.dumps(record))
    second = Round(root, config, tmp_path)
    assert second.started["recovered_from"] is None
    assert json.loads((root / "last_round.json").read_text())["exit_code"] == 0


def test_phases_only_move_forward_and_leave_the_gpu_after_cleanup(root, config, tmp_path):
    current = Round(root, config, tmp_path)

    def move(phase, **fields):
        return current.save(admit(campaign(phase=phase, **fields)))

    assert "first be saved as admitted" in move("optimizing").stderr
    assert move("admitted").returncode == 0
    assert move("optimizing").returncode == 0
    assert "cannot move back" in move("preparing").stderr
    assert "unknown phase" in move("done").stderr
    # Skipping ahead may not skip the cleanup: every way off the GPU goes through `cleaning`.
    for bad in ("ended", "pr_open", "ready_to_deliver"):
        assert "clean it up first" in move(bad, settled_model_usd=200).stderr, bad
    assert move("cleaning").returncode == 0
    assert move("ready_to_deliver", settled_model_usd=200).returncode == 0
    assert move("ended", settled_model_usd=200).returncode == 0
    assert "has ended" in move("pr_open", settled_model_usd=200).stderr


def test_a_campaign_is_settled_once_and_only_as_it_leaves_its_gpu(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    why = "settled exactly when it leaves its GPU"
    # Settled at 0 it would count as nothing against the budget while it can still spend.
    assert current.refuses({"campaigns": by_id(campaign(settled_model_usd=0))}, why)
    assert current.saves(admit(campaign()))
    assert current.saves({"campaigns": {"c1": {"phase": "optimizing"}}})
    assert current.refuses({"campaigns": {"c1": {"settled_model_usd": 0}}}, why)
    assert current.saves({"campaigns": {"c1": {"phase": "cleaning"}}})
    assert current.refuses({"campaigns": {"c1": {"phase": "ready_to_deliver"}}}, why)
    for between in (100, True):
        patch = {"campaigns": {"c1": {"phase": "ready_to_deliver", "settled_model_usd": between}}}
        assert current.refuses(patch, "never in between"), between
    assert current.saves({"campaigns": {"c1": {"phase": "ready_to_deliver", "settled_model_usd": 200}}})
    assert current.refuses({"campaigns": {"c1": {"settled_model_usd": 0}}}, "settled once")
    assert current.refuses({"campaigns": {"c1": {"settled_model_usd": None}}}, why)


def test_a_pool_is_settled_once_after_its_campaigns_left_it(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    # Settled as it is created, it would leave the ledger for its archive with its allocation still live.
    assert current.refuses(
        {"pools": by_id(pool(settled_node_hours=0))}, "later save than the one that creates it"
    )
    assert current.saves({"pools": by_id(pool()), "campaigns": by_id(campaign())})
    assert current.refuses({"pools": {"p1": {"settled_node_hours": 3}}}, "while a campaign holds")
    assert current.saves({"campaigns": {"c1": {"phase": "ended", "settled_model_usd": 0}}})
    for bad in (-1000, "3", True):
        assert current.refuses({"pools": {"p1": {"settled_node_hours": bad}}}, "non-negative number"), bad
    assert current.saves({"pools": {"p1": {"settled_node_hours": 3}}})
    # Neither lowered nor removed: a removed amount could be settled again, lower.
    for again in (0, None):
        assert current.refuses({"pools": {"p1": {"settled_node_hours": again}}}, "settled once"), again
    assert state(root)["pools"][0]["settled_node_hours"] == 3


def test_gpus_and_budget_are_capped(root, config, tmp_path):
    current = Round(root, config, tmp_path)

    def patch(*campaigns, pools=None):
        return {"campaigns": by_id(*campaigns), "pools": by_id(*(pools or [pool()]))}

    assert current.refuses(patch(campaign(model_usd=251)), "exceeds campaign.max_model_usd")
    for same in (0, "0"):
        assert current.refuses(patch(campaign("c1"), campaign("c2", gpu=same)), "hold the same GPU"), same
    assert current.refuses(patch(campaign("c1"), campaign("c2", gpu=None)), "records its pool and gpu")
    two = (campaign("c1"), campaign("c2", gpu=1))
    assert current.refuses(patch(*two, pools=[pool(node_hours=49)]), "exceed budget.total_node_hours")
    assert current.saves(patch(*two, pools=[pool()]))
    assert current.refuses(patch(campaign("c3", gpu=2, model_usd=50)), "max_parallel_campaigns")
    # c1 ends having spent its 200, so its GPU is free but c2's 200 and a third 250 are over the 500.
    assert current.saves({"campaigns": {"c1": {"phase": "ended", "settled_model_usd": 200}}})
    assert current.refuses(patch(campaign("c3", gpu=0, model_usd=250)), "exceed budget.total_model_usd")
    assert current.saves(patch(campaign("c3", gpu=0, model_usd=100)))


def test_lowering_a_cap_blocks_admissions_and_nothing_else(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.saves({"campaigns": by_id(campaign("c1"), campaign("c2", gpu=1)), "pools": by_id(pool())})
    lowered = CONFIG.format(round_hours=2)
    for cap, low in (
        ("total_model_usd: 500", 100),
        ("max_model_usd: 250", 120),
        ("max_parallel_campaigns: 2", 1),
    ):
        lowered = lowered.replace(cap, f"{cap.split(':')[0]}: {low}")
    config.write_text(lowered)
    assert current.saves({"wake_at": "t"})
    assert current.saves({"campaigns": {"c1": {"phase": "cleaning"}}})
    # Ending one campaign does not make room for another above the lowered cap.
    swap = {"c1": {"phase": "ended", "settled_model_usd": 200}, **by_id(campaign("c3", model_usd=1))}
    assert current.refuses({"campaigns": swap}, "max_parallel_campaigns")
    assert current.saves({"campaigns": {"c1": {"phase": "ended", "settled_model_usd": 200}}})
    assert current.refuses(
        {"campaigns": by_id(campaign("c3", model_usd=121))}, "exceeds campaign.max_model_usd"
    )
    assert current.refuses({"campaigns": by_id(campaign("c3", model_usd=1))}, "max_parallel_campaigns")
    assert current.saves({"campaigns": {"c2": {"phase": "ended", "settled_model_usd": 0}}})
    assert current.refuses({"campaigns": by_id(campaign("c3", model_usd=1))}, "exceed budget.total_model_usd")
    # A pool settles at what it used, even above its bound and the cap.
    assert current.saves({"pools": {"p1": {"settled_node_hours": 60}}})
    # A migration is held to the lowered caps no more than a save is.
    current.end()
    shown = success(root, "show")
    (root / "state.json").write_text(json.dumps({"revision": shown["revision"], "state": shown["state"]}))
    assert Round(root, config, tmp_path).id


def test_a_campaign_is_placed_only_on_a_started_pool_that_outlives_it(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    why = "started, unsettled pool"
    assert current.refuses({"campaigns": by_id(campaign())}, why)
    late = {"model_usd": 200, "deadline": "2100-01-03T00:00:00+00:00"}
    for bad in (campaign(bounds={"model_usd": 200}), campaign(bounds=late), campaign(pool="p2")):
        assert current.refuses({"campaigns": by_id(bad), "pools": by_id(pool())}, why), bad
    # Requested, not started: its end is not known yet.
    requested = {"id": "p1", "bounds": {"node_hours": 24}}
    assert current.refuses({"campaigns": by_id(campaign()), "pools": by_id(requested)}, why)
    assert current.saves(admit(campaign()))
    # Only the save that places it is held to this.
    assert current.saves({"campaigns": {"c1": {"phase": "optimizing"}}, "pools": {"p1": {"end": None}}})


def test_only_run_and_drain_save_and_drain_admits_nothing(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.saves(admit(campaign()))
    text = CONFIG.format(round_hours=2)
    config.write_text(text.replace("mode: run", "mode: drain"))
    assert current.refuses(admit(campaign("c2", gpu=1)), "drain admits no campaign")
    assert current.saves({"campaigns": {"c1": {"phase": "cleaning"}}})
    for mode in ("mode: observe", ""):
        config.write_text(text.replace("mode: run", mode))
        assert current.refuses({"note": 1}, "mode run or drain"), mode


def test_an_unset_cap_blocks_admissions_by_name_and_nothing_else(root, config, tmp_path):
    current = Round(root, config, tmp_path)
    assert current.saves(admit(campaign()))
    config.write_text(CONFIG.format(round_hours=2).replace("total_model_usd: 500", "total_model_usd: null"))
    assert current.refuses(admit(campaign("c2", gpu=1)), "budget.total_model_usd is not set")
    assert current.saves({"campaigns": {"c1": {"phase": "cleaning"}}})


def test_begin_takes_the_state_directory_from_the_configuration(root, config):
    config.write_text(CONFIG.format(round_hours=2).replace("execution:", f"execution:\n  state_dir: {root}"))
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "begin", "--config", str(config)], capture_output=True, text=True
    )
    assert "round" in json.loads(result.stdout), result.stderr
    assert (
        "--root is required"
        in subprocess.run([sys.executable, "-I", str(SCRIPT), "show"], capture_output=True, text=True).stderr
    )
