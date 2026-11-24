# SPDX-License-Identifier: MIT
"""The kernel optimization book cites nothing of the repository it is kept in, and lints clean. CPU only."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

BOOK = Path(__file__).resolve().parents[2] / "skills/kernel-optimization-book"
sys.path.insert(0, str(BOOK / "maintenance"))
import kb_index  # noqa: E402
import kb_lint  # noqa: E402

# Built from parts, so that this file does not hold what it tests for.
NAME = "ak" + "lr"
HASH = "c735728b" + "d507cf29"


@pytest.mark.parametrize(
    "line",
    [
        f"measured by {NAME.upper()}'s validate",
        f"`python -m {NAME} validate x.json`",
        f"{NAME.upper()}_MIN_SPEEDUP is 1.05",
        f"the file {HASH} lists m 1 to 128",
        f"`{HASH[:8]}…` runs the strip builder",
        "the kernel of `gemm_n6144_k4096`",
        "contract `gemm@1.0`, layout variant row_major",
        "- S2 x.json#kernel.py:26 the table per m",
        "ops/gemm/impls/flydsl/x/y.json",
        "`harness/scripts/forge_task/assets/driver.py`",
        "the image pinned in `stack.json`",
        "`tests/core/test_dispatch.py`",
        "all 16 sizes reach it (#125)",
        "fixed in PR 134",
        "on one MI355X at main 0e02a81",
        "CI run 45334576993",
        "under measurement profile 19",
    ],
)
def test_a_citation_of_the_host_repository_is_an_error(line):
    issues = list(kb_lint.host_repository_issues("page.md", f"first line\n{line}\n"))
    assert [(issue.level, issue.line) for issue in issues] == [("error", 2)], line


@pytest.mark.parametrize(
    "line",
    [
        "- S1 ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py:214 LDS holds only the A tile",
        "ROCm/FlyDSL#1008 changed the loop; ROCm/aiter#487 the table",
        "aiter fork commit 4e80e6619, `aiter/ops/flydsl/kernels/x.py`",
        "the multiplier `9E3779B97F4A7C15` and the mask `ffffffff00000000`",
        "see [§12](#12-all-zeros) and `gemm/small_m.md` §4",
        "N 6144 × K 4096 at m 1–128 on gfx950, 1.130–1.290 over aiter's tuned bf16 GEMM",
        "FlyDSL's `tests/kernels/test_gemm.py` and the main loop",
    ],
)
def test_other_repositories_and_plain_text_pass(line):
    assert not list(kb_lint.host_repository_issues("page.md", line + "\n"))


def test_an_isa_snapshot_keeps_its_hexadecimal_words_and_numbers():
    snapshot = f"opcode {HASH} (#125)\n"
    assert not list(kb_lint.host_repository_issues("x.md", snapshot, kb_lint.HOST_REPOSITORY_ALWAYS))
    named = f"see {NAME}/ops/README.md\n"
    assert list(kb_lint.host_repository_issues("x.md", named, kb_lint.HOST_REPOSITORY_ALWAYS))


def test_the_book_lints_clean_and_its_indexes_are_current(capsys):
    assert kb_lint.main(["--errors-only"]) == 0, capsys.readouterr().out
    assert kb_index.main(["--check"]) == 0, capsys.readouterr().out


def test_a_copy_of_the_book_lints_like_the_book(tmp_path, monkeypatch, capsys):
    copy = tmp_path / "copy"
    shutil.copytree(BOOK, copy, ignore=shutil.ignore_patterns("*.png", "__pycache__"))
    for name in ("SKILL_ROOT", "REPO_ROOT", "HOME_ROOT", "CHECK_HOST_REPOSITORY"):
        monkeypatch.setattr(kb_lint, name, getattr(kb_lint, name))  # main() rebinds them
    assert (
        kb_lint.main(["--errors-only", "--skill-root", str(copy), "--repo-root", str(BOOK.parents[2])]) == 0
    ), capsys.readouterr().out
    page = copy / "gemm" / "small_m.md"
    text = page.read_text()
    page.write_text(text + f"\nSee [the rules](../../{NAME}-kernel-rules/SKILL.md).\n")
    assert kb_lint.main(["--errors-only", "--skill-root", str(copy), str(page)]) == 1
    assert "names the repository" in capsys.readouterr().out
    # a link to a neighbouring skill resolves where the book sits in its repository, and is refused
    page.write_text(text + "\nSee [reading a result](../../read-validation/SKILL.md).\n")
    root = ["--skill-root", str(copy), "--repo-root", str(BOOK.parents[2])]
    assert kb_lint.main(["--errors-only", *root, str(page)]) == 1
    assert "link leaves the book" in capsys.readouterr().out
