# SPDX-License-Identifier: MIT
"""The kernel optimization book cites nothing of the repository it is kept in, keeps kernel code in its case pages
and public source in `sources`, and lints clean. CPU only."""

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


CASE = """---
title: "bf16 GEMM, N 64 x K 512: K split over the waves"
kind: case
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-64"]
accuracy: bitwise
gens: [gfx950]
{pin}
basis: measured N=1
measured:
  - {{metric: speedup, value: "1.2", gpu: MI355X, workload: "m 1-64", date: "2026-11"}}
---

# A case

## Applies when
## Counter-signals
## Mechanism
{code}
## Result
"""
ENTRY = "### {n}. The fold\n\n```python\ntotal = a + b\n```\n"
TECHNIQUE = """---
title: "Fold in split order"
kind: technique
lever: fold_in_split_order
ops: [gemm]
dtypes: [bf16]
regimes: ["M 1-64"]
gens: [gfx950]
flydsl: 0.3.2
basis: source-semantics
{backing}
---

# A technique

## What it does
## How to write it
{text}
## When it applies
## Counter-signals
## How to confirm it took effect
## Cost and risk
## Evidence
"""


def _book(tmp_path, monkeypatch, pages):
    """A book of these pages, linted in place of the real one: the errors of one of its pages."""
    for rel, text in pages.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    for name in ("SKILL_ROOT", "REPO_ROOT", "HOME_ROOT"):
        monkeypatch.setattr(kb_lint, name, tmp_path)

    def errors(rel):
        issues = kb_lint.lint_page(kb_lint.read_page(tmp_path / rel))
        return [issue.message for issue in issues if issue.level == "error"]

    return errors


ONE_ENTRY = "## Code\n" + ENTRY.format(n=1)


def _case(code=ONE_ENTRY, pin="flydsl: 0.3.2"):
    return CASE.format(code=code, pin=pin)


@pytest.mark.parametrize(
    ("entry", "passes"),
    [
        ("ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py", True),
        ("ROCm/FlyDSL@dd8373349:python/flydsl/compiler/jit_function.py:1633", True),
        ("ROCm/aiter@dbd8bf5:aiter/ops/flydsl/", True),
        ("ROCm/FlyDSL@v0.3.2", True),
        ("someone/aiter@40ae5ad7f (branch split-k)", True),
        ("https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html", True),
        ("https://github.com/ROCm/aiter/issues/5322 (the report)", True),
        ("aiter/ops/flydsl/kernels/splitk_hgemm.py", False),  # a path of no repository
        ("ROCm/FlyDSL:kernels/gemm/preshuffle_gemm.py", False),  # a repository at no revision
        ("ROCm/FlyDSL@main:kernels/gemm/preshuffle_gemm.py", False),  # a branch moves
        ("ROCm/FlyDSL@dd8373349:kernels/x.py and the kernel's own source", False),  # text after the source
        ("https://example.com/a page", False),
        ("../cases/a.md § Mechanism", False),  # a case page is listed by its path from the book's root
        ("the kernel's own source", False),
    ],
)
def test_a_source_is_public_source_at_a_revision_or_a_url(tmp_path, monkeypatch, entry, passes):
    page = TECHNIQUE.format(backing=f'sources:\n  - "{entry}"', text="")
    errors = _book(tmp_path, monkeypatch, {"gemm/fold.md": page})("gemm/fold.md")
    assert ["`sources` entry" in message for message in errors] == ([] if passes else [True]), errors


def test_a_skill_that_may_cite_its_repository_keeps_its_sources(tmp_path, monkeypatch):
    page = TECHNIQUE.format(backing="sources:\n  - scripts/driver.py", text="")
    errors = _book(tmp_path, monkeypatch, {"gemm/fold.md": page})
    monkeypatch.setattr(kb_lint, "CHECK_HOST_REPOSITORY", False)
    assert not errors("gemm/fold.md")


def test_a_technique_is_backed_by_public_source_or_by_a_case(tmp_path, monkeypatch):
    linked = "The fold: `../cases/a.md` §1."
    pages = {
        "cases/a.md": _case(),
        "gemm/bare.md": TECHNIQUE.format(backing="", text=""),
        "gemm/case.md": TECHNIQUE.format(backing="sources:\n  - cases/a.md", text=linked),
    }
    errors = _book(tmp_path, monkeypatch, pages)
    assert ["lacks `sources`" in message for message in errors("gemm/bare.md")] == [True]
    assert not errors("gemm/case.md")
    assert not errors("cases/a.md")


@pytest.mark.parametrize(
    ("listed", "text", "expected"),
    [
        ("", "The fold: `../cases/a.md` §1.", "`sources` lacks cases/a.md"),
        ("  - cases/a.md\n", "", "which the text does not link"),
        (
            "  - cases/a.md\n  - gemm/other.md\n",
            "See [the case](../cases/a.md).",
            "`sources` entry `gemm/other.md`",
        ),
        ("  - cases/a.md\n  - cases/INDEX.md\n", "See [the case](../cases/a.md).", "names no page of kind"),
        ("  - cases/a.md\n  - cases/b.md\n", "See [the case](../cases/a.md).", "names no page of kind"),
        ("  - cases/a.md\n", "The fold: `../cases/a.md` §2.", "has no section 2"),
        (
            "  - cases/a.md\n",
            "[The case](../cases/a.md): `../cases/a.md §1`.",
            "in one code span is not checked",
        ),
        ("  - cases/a.md\n", "The fold: [the case](../cases/a.md) §1, and `../cases/INDEX.md`.", None),
        ("", "```\nsee `../cases/a.md` §1\n```", None),  # a path in fenced code is no link
        (
            "",
            "A figure: ![the grid](../cases/grid.png).",
            None,
        ),  # nor is a file of cases/ that is no case page
        ("cases:\n  - cases/a.md\n", "See [the case](../cases/a.md).", "`cases` is not a field"),
    ],
)
def test_sources_lists_the_case_pages_the_text_links(tmp_path, monkeypatch, listed, text, expected):
    source = "sources:\n  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py\n"
    pages = {
        "cases/a.md": _case(),
        "cases/INDEX.md": "---\nkind: index\n---\n",
        "cases/grid.png": "",
        "gemm/other.md": TECHNIQUE.format(backing=source, text=""),
        "gemm/fold.md": TECHNIQUE.format(backing=source + listed, text=text),
    }
    errors = _book(tmp_path, monkeypatch, pages)("gemm/fold.md")
    if expected == "`cases` is not a field":  # and the case page it links is missing from `sources`
        assert sorted(expected in message for message in errors) == [False, True], errors
    else:
        assert [expected in message for message in errors] == ([True] if expected else []), errors


def test_sources_also_lists_the_case_pages_that_link_the_page(tmp_path, monkeypatch):
    source = "sources:\n  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py\n"
    citing = _case().replace(
        "## Mechanism\n", "## Mechanism\nFolded as [gemm/fold](../gemm/fold.md) has it.\n"
    )
    pages = {
        "cases/a.md": citing,
        "cases/b.md": _case(),
        "gemm/fold.md": TECHNIQUE.format(backing=source, text=""),
        "gemm/other.md": TECHNIQUE.format(backing=source + "  - cases/a.md\n", text=""),
    }
    errors = _book(tmp_path, monkeypatch, pages)
    fold = tmp_path / "gemm/fold.md"
    assert ["`sources` lacks cases/a.md" in message for message in errors("gemm/fold.md")] == [True]
    assert ["does not link this page" in message for message in errors("gemm/other.md")] == [True]
    # the index tool writes them: what the text links and what links the page, after the public sources
    fold.write_text(kb_index.sources_updated(fold))
    assert "kernels/gemm/preshuffle_gemm.py\n  - cases/a.md\n" in fold.read_text()
    assert not errors("gemm/fold.md")
    assert kb_index.sources_updated(fold) == fold.read_text()
    assert "cases/a.md" not in kb_index.sources_updated(tmp_path / "gemm/other.md")
    assert kb_index.sources_updated(tmp_path / "cases/a.md") is None  # a case page lists no case page


def test_the_index_tool_writes_the_case_pages_after_the_public_sources():
    page = '---\ntitle: A\nsources:\n  - a/b@c:d.py\n  - "a/b@c:e.py:3"\nbasis: measured N=1\n---\n\n# A\n'
    two = kb_index.with_case_sources(page, ["cases/a.md", "cases/b.md"])
    assert two == page.replace("basis:", "  - cases/a.md\n  - cases/b.md\nbasis:")
    assert kb_index.with_case_sources(two, ["cases/b.md"]) == page.replace("basis:", "  - cases/b.md\nbasis:")
    assert kb_index.with_case_sources(two, []) == page  # the public sources stay as written
    bare = "---\ntitle: A\n---\n# A\n"
    only = kb_index.with_case_sources(bare, ["cases/a.md"])
    assert only == "---\ntitle: A\nsources:\n  - cases/a.md\n---\n# A\n"  # last, where the page had none
    assert kb_index.with_case_sources(only, []) == bare  # and no empty field is left
    flow = "---\nsources: [a/b@c:d.py, a/b@c:e.py]\n---\n# A\n"
    assert kb_index.with_case_sources(flow, ["cases/a.md"]) == (
        "---\nsources:\n  - a/b@c:d.py\n  - a/b@c:e.py\n  - cases/a.md\n---\n# A\n"
    )


def test_the_length_of_a_page_is_the_length_of_its_text(tmp_path, monkeypatch):
    source = "sources:\n" + "  - ROCm/FlyDSL@dd8373349:kernels/gemm/preshuffle_gemm.py\n" * 40
    pages = {
        "gemm/fold.md": TECHNIQUE.format(backing=source, text="A line.\n" * (kb_lint.CARD_MAX_LINES - 60))
    }
    _book(tmp_path, monkeypatch, pages)

    def warnings():
        issues = kb_lint.lint_page(kb_lint.read_page(tmp_path / "gemm/fold.md"))
        return [issue.message for issue in issues if issue.level == "warn"]

    assert not warnings()  # over the limit with its front matter, under it without
    (tmp_path / "gemm/fold.md").write_text(pages["gemm/fold.md"] + "A line.\n" * 60)
    assert ["split it" in message for message in warnings()] == [True]


@pytest.mark.parametrize(
    ("code", "pin", "expected"),
    [
        ("## Code\n" + ENTRY.format(n=1) + ENTRY.format(n=2), "flydsl: 0.3.2", None),
        ("", "triton: 3.8.0", None),  # a case of a Triton kernel pins Triton
        ("", "", "front matter lacks `flydsl`"),
        ("### 1. The fold\n", "flydsl: 0.3.2", "which this page lacks"),
        ("## Code\n" + ENTRY.format(n=1) + ENTRY.format(n=3), "flydsl: 0.3.2", "should be 2"),
        ("## Code\n" + ENTRY.format(n=2), "flydsl: 0.3.2", "should be 1"),
        (
            "## Code\n" + ENTRY.format(n=1) + "### 2. The store\n\nIn words only.\n",
            "flydsl: 0.3.2",
            "entry 2 of",
        ),
        ("## Code\n### 1. The store\n\nIn words only.\n" + ENTRY.format(n=2), "flydsl: 0.3.2", "entry 1 of"),
        (
            "## Code\n" + ENTRY.format(n=1) + "## Notes\n### 2. Late\n```\nx\n```\n",
            "flydsl: 0.3.2",
            "outside `## Code`",
        ),
        # a numbered heading of the text does not shift the entries' numbers
        (
            "### 1. A step\n## Code\n" + ENTRY.format(n=1) + ENTRY.format(n=2),
            "flydsl: 0.3.2",
            "outside `## Code`",
        ),
        # neither a heading that starts with a decimal nor a line of an excerpt is an entry
        ("## Code\n" + ENTRY.format(n=1) + "## Notes\n### 1.5× over the baseline\n", "flydsl: 0.3.2", None),
        (
            "## Code\n" + ENTRY.format(n=1).replace("total", "### 2. a line of it\ntotal"),
            "flydsl: 0.3.2",
            None,
        ),
        ("", "flydsl: 0.3.2\nsources:\n  - cases/a.md", "a case page lists no case page"),
    ],
)
def test_a_case_page_numbers_its_code_entries(tmp_path, monkeypatch, code, pin, expected):
    errors = _book(tmp_path, monkeypatch, {"cases/a.md": _case(code, pin)})("cases/a.md")
    assert [expected in message for message in errors] == ([True] if expected else []), errors


def test_a_section_cited_in_a_case_is_a_heading_not_a_line_of_its_code(tmp_path, monkeypatch):
    case = _case("## Code\n" + ENTRY.format(n=1).replace("total", "# 2. accumulate the partials\ntotal"))
    page = TECHNIQUE.format(backing="sources:\n  - cases/a.md", text="The fold: `../cases/a.md` §2.")
    errors = _book(tmp_path, monkeypatch, {"cases/a.md": case, "gemm/fold.md": page})("gemm/fold.md")
    assert ["has no section 2" in message for message in errors] == [True], errors


def test_a_case_page_is_a_page_of_kind_case_in_the_cases_folder(tmp_path, monkeypatch):
    technique = TECHNIQUE.format(backing="sources:\n  - ROCm/FlyDSL@v0.3.2", text="")
    errors = _book(tmp_path, monkeypatch, {"gemm/a.md": _case(), "cases/fold.md": technique})
    for rel in ("gemm/a.md", "cases/fold.md"):
        assert ["goes in `cases/`" in message for message in errors(rel)] == [True], rel


def test_only_a_case_pins_triton_outside_the_triton_folder(tmp_path, monkeypatch):
    technique = TECHNIQUE.format(backing="sources:\n  - ROCm/FlyDSL@v0.3.2", text="")
    page = technique.replace("flydsl: 0.3.2", "triton: 3.8.0")
    errors = _book(tmp_path, monkeypatch, {"gemm/fold.md": page})("gemm/fold.md")
    assert ["lacks `flydsl`" in message for message in errors] == [True], errors


def test_stale_skips_a_case_that_pins_triton_alone(tmp_path, monkeypatch):
    pages = {
        "cases/fly.md": _case(pin="flydsl: 0.3.1"),
        "cases/tri.md": _case(pin="triton: 3.8.0"),
        "cases/both.md": _case(pin="flydsl: 0.3.1\ntriton: 3.8.0"),
    }
    _book(tmp_path, monkeypatch, pages)
    assert kb_lint.stale([tmp_path / rel for rel in sorted(pages)], "0.3.2") == [
        "cases/both.md: flydsl 0.3.1",
        "cases/fly.md: flydsl 0.3.1",
    ]


def test_the_cases_index_names_the_toolchain_a_case_pins():
    assert kb_index._toolchain({"flydsl": "0.3.2"}) == "FlyDSL 0.3.2"
    assert kb_index._toolchain({"triton": "3.8.0"}) == "Triton 3.8.0"
    assert kb_index._toolchain({"flydsl": "0.3.2", "triton": "3.8.0"}) == "FlyDSL 0.3.2, Triton 3.8.0"


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
