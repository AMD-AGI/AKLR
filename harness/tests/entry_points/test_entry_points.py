"""Claude Code, Codex and Cursor reach the harness skills, and Claude Code and Codex its agents, through the repository's entry points."""

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
def test_codex_reads_each_agent(name):
    assert (repo_root() / f".codex/agents/{name}.toml").is_file()
