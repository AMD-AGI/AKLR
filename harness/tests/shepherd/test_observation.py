"""The fingerprint an idle round is skipped on."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "automations/shepherd/skills/optimize-and-deliver-kernels/scripts/observation.py"
)
spec = importlib.util.spec_from_file_location("shepherd_observation", SCRIPT)
observation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observation)


def test_fingerprint_follows_what_was_seen_and_not_its_order():
    seen = observation.Observation("main1", ((1, "a"), (2, "b")), "head1", "ready1")
    same = observation.Observation("main1", ((2, "b"), (1, "a")), "head1", "ready1")
    assert seen.fingerprint == same.fingerprint
    for other in (
        observation.Observation("main2", seen.pulls, "head1", "ready1"),
        observation.Observation("main1", ((1, "a"), (2, "c")), "head1", "ready1"),
        observation.Observation("main1", ((1, "a"),), "head1", "ready1"),
        observation.Observation("main1", seen.pulls, "head2", "ready1"),
        observation.Observation("main1", seen.pulls, "head1", "ready2"),
    ):
        assert other.fingerprint != seen.fingerprint
