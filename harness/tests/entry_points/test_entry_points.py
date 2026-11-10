"""Every client reaches every harness skill and agent through the repository's entry points."""

from pathlib import Path

import pytest


def repo_root():
    return Path(__file__).resolve().parents[3]


def harness_agents():
    return sorted(path.stem for path in (repo_root() / "harness/agents").glob("*.md"))


def test_claude_code_loads_the_harness_as_the_aklr_plugin():
    repo = repo_root()
    assert (repo / ".claude/skills/aklr").resolve() == repo / "harness"


def test_codex_and_cursor_read_the_harness_skills():
    repo = repo_root()
    assert (repo / ".agents/skills").resolve() == repo / "harness/skills"


@pytest.mark.parametrize("name", harness_agents())
def test_cursor_reads_each_agent(name):
    repo = repo_root()
    assert (repo / f".cursor/agents/{name}.md").resolve() == repo / f"harness/agents/{name}.md"


@pytest.mark.parametrize("name", harness_agents())
def test_codex_reads_each_agent(name):
    assert (repo_root() / f".codex/agents/{name}.toml").is_file()
