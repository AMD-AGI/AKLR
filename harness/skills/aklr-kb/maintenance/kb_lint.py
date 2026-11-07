#!/usr/bin/env python3
"""Lint aklr-kb, its language folders included, against CONTRIBUTING.md.

Pages of the new kinds (technique, pattern, op, case, pitfall, language-note, topic, method) and SKILL.md are held
to the rules: an error fails the run. Pages written before the rules (kinds language, index, reference,
and pages without front matter) only get warnings until they are converted.

    python maintenance/kb_lint.py                     # the whole skill
    python maintenance/kb_lint.py moe/foo.md          # some pages
    python maintenance/kb_lint.py --stale 0.3.2       # pages and evidence lines checked against another FlyDSL version
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _repository_of(skill_root: Path) -> Path:
    """The repository a skill in place sits in: <repo>/harness/skills/<skill>."""
    return skill_root.parents[2] if len(skill_root.parents) > 2 else skill_root


# --skill-root lints another skill by the same rules; --repo-root names the repository of a copy of the skill.
SKILL_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = _repository_of(SKILL_ROOT)
# Where the skill sits in its repository: SKILL_ROOT, unless SKILL_ROOT is a copy outside REPO_ROOT.
HOME_ROOT = SKILL_ROOT
# The architectures a page may cover, and every name each goes by (ISA family, product). A page names only the
# architectures of its own `gens` (gfx950 when it has none); SKILL.md, CONTRIBUTING.md and index pages may name any.
ARCH_NAMES = {
    "gfx950": frozenset({"gfx950", "CDNA4", "MI350", "MI355"}),
    "gfx942": frozenset({"gfx942", "CDNA3", "MI300", "MI308", "MI325"}),
}
DEFAULT_GENS = ("gfx950",)
# Whole names and family forms (gfx9, gfx95x, gfx94*); the `*` of a bold `**gfx950**` is not a wildcard.
ARCH_TOKEN_RE = re.compile(r"\bgfx\d{1,4}(?:[a-z]\b|\*(?!\*)|\b)|\bCDNA ?\d\b|\bMI\d{3}X?\b")
PINNED_FLYDSL = "0.3.2"
# The Triton that AKLR's CI pins (`.github/workflows/ci.yaml` checks triton==3.8.0); pages under triton/ give it in
# `triton` where other pages give `flydsl`.
PINNED_TRITON = "3.8.0"
SKILL_MD_MAX_LINES = 150  # forge-loop inlines SKILL.md into every round's prompt
SKILL_MD_MAX_BYTES = 12_000  # of the body it inlines; the front matter is stripped
CARD_MAX_LINES = 600

# Not knowledge pages: the ISA chapters, code snapshots, review notes and the scripts.
SKIP_PARTS = ("gfx950/cdna4/", "gfx942/cdna3/", "reference/", "maintenance/", "analysis/")

NEW_KINDS = (
    "technique",
    "pattern",
    "op",
    "case",
    "pitfall",
    "language-note",
    "topic",
    "method",
)
LEGACY_KINDS = ("language", "index", "reference")

REQUIRED_FIELDS = {
    "technique": (
        "title",
        "lever",
        "ops",
        "dtypes",
        "regimes",
        "gens",
        "flydsl",
        "basis",
        "sources",
    ),
    "pattern": ("title", "signals", "ops", "gens", "flydsl", "basis"),
    "op": ("title", "ops", "dtypes", "gens", "flydsl", "basis"),
    "case": (
        "title",
        "ops",
        "dtypes",
        "regimes",
        "contract",
        "gens",
        "flydsl",
        "basis",
        "measured",
        "sources",
    ),
    "pitfall": ("title", "scope", "gens", "flydsl", "basis"),
    "language-note": ("title", "gens", "flydsl", "basis", "sources"),
    "topic": ("title", "ops", "dtypes", "regimes", "gens", "flydsl"),
    "method": ("title", "ops", "gens", "flydsl"),
}
REQUIRED_SECTIONS = {
    "technique": (
        "What it does",
        "How to write it",
        "When it applies",
        "Counter-signals",
        "How to confirm it took effect",
        "Cost and risk",
        "Evidence",
    ),
    "pattern": (
        "Symptom",
        "Likely causes",
        "Rule out first",
        "Candidate techniques",
        "Evidence",
    ),
    "op": (
        "What it computes",
        "Bottleneck by regime",
        "Techniques that apply",
        "Kernels to start from",
        "In AKLR",
    ),
    "case": (
        "Applies when",
        "Counter-signals",
        "Mechanism",
        "Result",
        "Reproduce",
        "Rollback",
    ),
    "pitfall": ("Symptom", "Root cause", "How to detect", "Fix or workaround", "Scope"),
    "language-note": (),
    "topic": ("Route here when",),
    "method": ("Route here when",),
}
MEASURED_KEYS = ("metric", "value", "gpu", "workload", "run_ref", "date")

BASIS_RE = re.compile(r"^(source-semantics|measured N=\d+\b.*|single-observation)$")
FORBIDDEN_HEADING_RE = re.compile(
    r"^#{2,4}\s+(tried|what we tried|track record|history|timeline|changelog|attempts?)\b",
    re.IGNORECASE | re.MULTILINE,
)
# Every way a measured number has been marked; only the first form below is accepted on a new page.
FIELD_ANY_RE = re.compile(r"\*\(field\b[^)]*\)\*|\*field:\*|\(field:[^)]*\)")
FIELD_CANON_RE = re.compile(r"\*\(field,\s*(E\d+(?:,\s*E\d+)*)\)\*")
EVIDENCE_E_RE = re.compile(r"^-\s+(E\d+)\s+\[(\d{4}-\d{2}),([^\]]*)\](.*)$")
EVIDENCE_S_RE = re.compile(r"^-\s+(S\d+)\s+\S+:\d+")
N_RE = re.compile(r"\bN=(\d+|\?)")
TOOLCHAIN_RE = re.compile(r"\bFlyDSL\s+(\d[\w.+-]*|\?)")
DURABLE_REF_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b|#\d+|https?://|\bCI run\b")
LOCAL_REF_RE = re.compile(r"\blocal:/\S+")
LOCAL_RELATIVE_RE = re.compile(r"\blocal:(?!/)[^\s;)]+")
ABSOLUTE_RE = re.compile(r"\b(always|never)\b", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
CODE_PATH_RE = re.compile(r"`([\w./-]+\.md)`")
SECTION_REF_RE = re.compile(
    r"(?:`([\w./-]+\.md)`|\]\(([\w./-]+\.md)\))\s*§\s?(\d+(?:\s*[–-]\s*\d+)?(?:\s*,\s*§\s?\d+(?:\s*[–-]\s*\d+)?)*)"
)
NUMBERED_HEADING_RE = re.compile(r"^#{1,4}\s+(?:Chapter\s+)?(\d+)\.", re.MULTILINE)
FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
BLOCK_START_RE = re.compile(r"^\s*(?:[-*+]\s|\d+\.\s|\|)")


@dataclass(frozen=True)
class Issue:
    level: str  # "error" or "warn"
    path: str
    line: int
    message: str

    def __str__(self) -> str:
        tag = "ERROR" if self.level == "error" else "warn "
        return f"{tag} {self.path}:{self.line}: {self.message}"


@dataclass
class Page:
    path: Path
    text: str
    front: dict | None
    body: str
    body_line: int  # 1-based line of the body's first line
    front_error: str | None = None  # set when the front matter is not valid YAML

    @property
    def rel(self) -> str:
        if self.path.is_relative_to(SKILL_ROOT):
            return self.path.relative_to(SKILL_ROOT).as_posix()
        return self.path.as_posix()

    @property
    def kind(self) -> str:
        if self.path.name == "SKILL.md":  # aklr-kb's and each language folder's
            return "entry"
        if self.path.name == "CONTRIBUTING.md":
            return "guide"
        if self.front is None:
            return "untyped"
        if "kind" in self.front:
            return str(self.front["kind"])
        if "name" in self.front and "description" in self.front:
            return "upstream"  # a copy of one of FlyDSL's own skills
        return "untyped"


@dataclass(frozen=True)
class Evidence:
    ident: str
    line: int
    n: str | None  # "3", "?" or None for an S line
    flydsl: str | None  # the FlyDSL version the line names


def _flat_front(text: str) -> dict:
    """Top-level `key: value` pairs, for front matter that is not valid YAML."""
    front: dict = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if match:
            front[match.group(1)] = match.group(2).strip() or None
    return front


def _parse_front(text: str) -> tuple[dict, str | None]:
    """The front matter and, when it is not valid YAML, why."""
    try:
        import yaml
    except ImportError:
        return _flat_front(text), None
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return _flat_front(text), str(exc).splitlines()[0]
    if not isinstance(loaded, dict):
        return _flat_front(text), "front matter is not a mapping"
    return loaded, None


def read_page(path: Path) -> Page:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            front, error = _parse_front(text[4:end])
            body_start = text.find("\n", end + 1) + 1
            front_lines = text[:body_start].count("\n")
            return Page(path, text, front, text[body_start:], front_lines + 1, error)
    return Page(path, text, None, text, 1)


def _prose(body: str) -> str:
    """The body with fenced code and HTML comments blanked out; offsets into it are offsets into the body."""

    def blank(match: re.Match) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return COMMENT_RE.sub(blank, FENCE_RE.sub(blank, body))


def _line_of(page: Page, offset: int) -> int:
    return page.body_line + page.body[:offset].count("\n")


def _sections(prose: str) -> dict[str, tuple[int, str]]:
    """Each H2 section's lowercased title -> (offset of its text, its text)."""
    heads = list(H2_RE.finditer(prose))
    out = {}
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(prose)
        out[head.group(1).strip().lower()] = (head.end(), prose[head.end() : end])
    return out


def _blocks(prose: str) -> Iterable[tuple[int, str]]:
    """Paragraphs, list items and table rows, with their offsets."""
    start, lines, offset = 0, [], 0
    for line in prose.splitlines(keepends=True):
        if not line.strip() or BLOCK_START_RE.match(line):
            if lines:
                yield start, "".join(lines)
            start, lines = offset, ([line] if line.strip() else [])
        else:
            if not lines:
                start = offset
            lines.append(line)
        offset += len(line)
    if lines:
        yield start, "".join(lines)


def _resolve(page: Page, target: str) -> Path | None:
    """Relative to the page, else to the skill; a path whose first part is a top-level folder of the repository
    and not of the skill (`aklr/`, `aiter/`) is relative to the repository root; a bare file name may also sit one
    folder below the page. A path that leaves the skill (`flydsl/`) is also relative to where
    the page sits in the repository, which a copy of the skill is not in; one that stays in it is the copy's.
    """
    bases = [page.path.parent, SKILL_ROOT]
    if HOME_ROOT != SKILL_ROOT and page.path.is_relative_to(SKILL_ROOT):
        home = HOME_ROOT / page.path.parent.relative_to(SKILL_ROOT)
        if not (home / target).resolve().is_relative_to(HOME_ROOT.resolve()):
            bases.append(home)
    first = target.split("/", 1)[0]
    if "/" in target and (REPO_ROOT / first).is_dir() and not (SKILL_ROOT / first).exists():
        bases.append(REPO_ROOT)
    if "/" not in target:
        bases += [p for p in sorted(page.path.parent.iterdir()) if p.is_dir()]
    for base in bases:
        if (base / target).exists():
            return base / target
    return None


def _skill_local(target: str) -> bool:
    """A `code` path meant as a page of this skill; other `.md` paths in code spans name files in other repositories."""
    first = target.split("/", 1)[0]
    if "/" not in target:
        # A bare name is a page of this skill; name another repository's file by its path (`docs/x.md`).
        return True
    return (SKILL_ROOT / first).is_dir()


def check_links(page: Page, level: str) -> Iterable[Issue]:
    prose = _prose(page.body)
    seen = set()
    for regex in (MD_LINK_RE, CODE_PATH_RE):
        for match in regex.finditer(prose):
            target = match.group(1)
            if target in seen or "<" in target or "*" in target:
                continue
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if regex is CODE_PATH_RE and not _skill_local(target):
                continue
            seen.add(target)
            if _resolve(page, target) is None:
                yield Issue(
                    level,
                    page.rel,
                    _line_of(page, match.start()),
                    f"link does not resolve: {target}",
                )
    for match in SECTION_REF_RE.finditer(prose):
        target = match.group(1) or match.group(2)
        path = _resolve(page, target)
        if path is None:
            continue
        numbers = set(NUMBERED_HEADING_RE.findall(path.read_text(encoding="utf-8")))
        for cited in re.findall(r"\d+", match.group(3)):
            if cited not in numbers:
                yield Issue(
                    level,
                    page.rel,
                    _line_of(page, match.start()),
                    f"{target} has no section {cited}",
                )


def check_front(page: Page) -> Iterable[Issue]:
    kind, front = page.kind, page.front or {}
    triton_page = page.rel.startswith("triton/")
    for field in REQUIRED_FIELDS[kind]:
        if triton_page and field == "flydsl":
            field = "triton"
        if front.get(field) in (None, "", []):
            yield Issue("error", page.rel, 1, f"front matter lacks `{field}` (kind {kind})")
    triton = str(front.get("triton") or "")
    if triton and triton != PINNED_TRITON:
        yield Issue(
            "warn",
            page.rel,
            1,
            f"pinned to Triton {triton}, AKLR's CI is on {PINNED_TRITON}: re-check",
        )
    basis = str(front.get("basis") or "")
    if basis and not BASIS_RE.match(basis):
        yield Issue(
            "error",
            page.rel,
            1,
            f"basis `{basis}` is not source-semantics, measured N=<k> or single-observation",
        )
    flydsl = str(front.get("flydsl") or "")
    if flydsl and flydsl != PINNED_FLYDSL:
        yield Issue(
            "warn",
            page.rel,
            1,
            f"pinned to FlyDSL {flydsl}, the skill is on {PINNED_FLYDSL}: re-check",
        )
    if kind == "case":
        if front.get("contract") not in ("bitwise", "bounded"):
            yield Issue("error", page.rel, 1, "contract must be bitwise or bounded")
        for i, entry in enumerate(front.get("measured") or []):
            missing = [k for k in MEASURED_KEYS if not isinstance(entry, dict) or entry.get(k) in (None, "")]
            if missing:
                yield Issue("error", page.rel, 1, f"measured[{i}] lacks {', '.join(missing)}")


def read_evidence(page: Page) -> tuple[dict[str, Evidence], list[Issue]]:
    """The page's evidence lines by id, and the lines that are not evidence lines."""
    prose = _prose(page.body)
    start, text = _sections(prose).get("evidence", (0, ""))
    found: dict[str, Evidence] = {}
    issues = []
    offset = start
    for raw in text.splitlines(keepends=True):
        line, line_no = raw.strip(), _line_of(page, offset)
        offset += len(raw)
        if not line.startswith("- "):
            continue
        source = EVIDENCE_S_RE.match(line)
        if source:
            found[source.group(1)] = Evidence(source.group(1), line_no, None, None)
            continue
        measured = EVIDENCE_E_RE.match(line)
        if not measured:
            issues.append(
                Issue(
                    "error",
                    page.rel,
                    line_no,
                    f"not `- E<n> [YYYY-MM, arch, toolchain, N=k, ruler] …` or `- S<n> file:line …`: {line[:60]}",
                )
            )
            continue
        ident, conditions, rest = (
            measured.group(1),
            measured.group(3),
            measured.group(4),
        )
        n = N_RE.search(conditions)
        toolchain = TOOLCHAIN_RE.search(conditions)
        if not n:
            issues.append(Issue("error", page.rel, line_no, f"{ident} gives no N=<k> or N=?"))
        if not (DURABLE_REF_RE.search(rest) or LOCAL_REF_RE.search(rest)):
            issues.append(
                Issue(
                    "error",
                    page.rel,
                    line_no,
                    f"{ident} has no run reference (commit, PR, CI run, URL or local:<absolute path>)",
                )
            )
        for rel in LOCAL_RELATIVE_RE.findall(rest):
            issues.append(
                Issue(
                    "error",
                    page.rel,
                    line_no,
                    f"{ident}: {rel} is not an absolute path",
                )
            )
        found[ident] = Evidence(
            ident,
            line_no,
            n.group(1) if n else None,
            toolchain.group(1) if toolchain else None,
        )
    return found, issues


def check_body(page: Page, level: str) -> Iterable[Issue]:
    prose = _prose(page.body)
    for match in FORBIDDEN_HEADING_RE.finditer(prose):
        yield Issue(
            level,
            page.rel,
            _line_of(page, match.start()),
            "a process log section; write the current best practice and move failures to Counter-signals",
        )
    if page.kind not in NEW_KINDS:
        return
    sections = _sections(prose)
    for name in REQUIRED_SECTIONS[page.kind]:
        if name.lower() not in sections:
            yield Issue("error", page.rel, page.body_line, f"missing section `## {name}`")
    evidence, issues = read_evidence(page)
    yield from issues
    unquoted = INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), prose)
    marks = list(FIELD_ANY_RE.finditer(unquoted))
    if marks and "evidence" not in sections:
        yield Issue(
            "error",
            page.rel,
            _line_of(page, marks[0].start()),
            "measured numbers without a `## Evidence` section",
        )
    for mark in marks:
        canon = FIELD_CANON_RE.fullmatch(mark.group(0))
        line = _line_of(page, mark.start())
        if not canon:
            yield Issue(
                "error",
                page.rel,
                line,
                f"`{mark.group(0)[:30]}`: mark a measured number *(field, E<n>)*",
            )
            continue
        for ident in re.findall(r"E\d+", canon.group(1)):
            if ident not in evidence:
                yield Issue("error", page.rel, line, f"{ident} is not in ## Evidence")
    weak = {e.ident for e in evidence.values() if e.n in ("1", "?")}
    single = str((page.front or {}).get("basis")) == "single-observation"
    for start, block in _blocks(unquoted):
        absolute = ABSOLUTE_RE.search(block)
        if not absolute:
            continue
        cited = {i for m in FIELD_CANON_RE.finditer(block) for i in re.findall(r"E\d+", m.group(1))}
        if single or cited & weak:
            reason = (
                "on a single-observation page"
                if single
                else f"beside {', '.join(sorted(cited & weak))}, seen once or an unknown number of times"
            )
            yield Issue(
                "error",
                page.rel,
                _line_of(page, start + absolute.start()),
                f"`{absolute.group(1)}` {reason}",
            )


def skill_home() -> Path:
    """Where the skill sits in its repository: its own folder, unless it is a copy outside the repository, whose
    original is `harness/skills/<the name in its SKILL.md>` there."""
    if SKILL_ROOT.is_relative_to(REPO_ROOT):
        return SKILL_ROOT
    entry = SKILL_ROOT / "SKILL.md"
    name = (read_page(entry).front or {}).get("name") if entry.is_file() else None
    return REPO_ROOT / "harness" / "skills" / str(name or SKILL_ROOT.name)


def check_arch(page: Page) -> Iterable[Issue]:
    """`gens` lists known architectures, and every architecture the page names is one of them."""
    gens = (page.front or {}).get("gens")
    if gens is not None and (not isinstance(gens, list) or any(str(g) not in ARCH_NAMES for g in gens)):
        yield Issue("error", page.rel, 1, f"gens {gens}: a list of {', '.join(ARCH_NAMES)}")
        return
    own = [str(g) for g in gens or DEFAULT_GENS]
    covered = ARCH_NAMES if page.kind in ("entry", "guide", "index") else own
    names = frozenset().union(*(ARCH_NAMES[g] for g in covered))
    for match in ARCH_TOKEN_RE.finditer(page.text):
        token = match.group(0).replace(" ", "").rstrip("X")
        if token not in names:
            line = page.text[: match.start()].count("\n") + 1
            yield Issue(
                "error",
                page.rel,
                line,
                f"names {match.group(0)}; the page covers {', '.join(own)}",
            )


def duplicate_levers(paths: Iterable[Path]) -> list[Issue]:
    """A `lever` belongs to one technique page: a second page with it extends the first instead."""
    owners: dict[str, str] = {}
    issues = []
    for path in paths:
        page = read_page(path)
        lever = (page.front or {}).get("lever") if page.kind == "technique" else None
        if not lever:
            continue
        if str(lever) in owners:
            issues.append(
                Issue("error", page.rel, 1, f"lever `{lever}` is already {owners[str(lever)]}; merge into it")
            )
        else:
            owners[str(lever)] = page.rel
    return issues


def lint_page(page: Page) -> list[Issue]:
    kind = page.kind
    issues: list[Issue] = list(check_arch(page))
    if page.front_error:
        level = "error" if kind in NEW_KINDS else "warn"
        issues.append(
            Issue(
                level,
                page.rel,
                1,
                f"front matter is not valid YAML ({page.front_error}); quote the value",
            )
        )
    if kind in ("entry", "guide"):
        lines = page.text.count("\n")
        if kind == "entry" and lines > SKILL_MD_MAX_LINES:
            issues.append(
                Issue(
                    "error",
                    page.rel,
                    1,
                    f"{lines} lines; SKILL.md is inlined every round, keep ≤ {SKILL_MD_MAX_LINES}",
                )
            )
        size = len(page.body.encode("utf-8"))
        if kind == "entry" and size > SKILL_MD_MAX_BYTES:
            issues.append(
                Issue(
                    "error",
                    page.rel,
                    1,
                    f"{size} bytes after the front matter; keep ≤ {SKILL_MD_MAX_BYTES}",
                )
            )
        issues += check_links(page, "error")
        issues += check_body(page, "error")
        return issues
    if kind in NEW_KINDS:
        issues += check_front(page)
        issues += check_body(page, "error")
        issues += check_links(page, "error")
    elif kind in LEGACY_KINDS or kind in ("upstream", "untyped"):
        if kind == "untyped":
            issues.append(
                Issue(
                    "warn",
                    page.rel,
                    1,
                    "no front matter; give it a kind when next edited",
                )
            )
        elif kind not in ("upstream", "index") and not (page.front or {}).get("basis"):
            issues.append(
                Issue(
                    "warn",
                    page.rel,
                    1,
                    f"legacy kind `{kind}`; convert it to a kind of CONTRIBUTING.md",
                )
            )
        prose = _prose(page.body)
        marks = sum(
            1
            for m in FIELD_ANY_RE.finditer(INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), prose))
            if not FIELD_CANON_RE.fullmatch(m.group(0))
        )
        if marks and kind != "upstream":
            issues.append(
                Issue(
                    "warn",
                    page.rel,
                    1,
                    f"{marks} measured numbers without an evidence id",
                )
            )
        issues += check_body(page, "warn")
        issues += check_links(page, "warn")
    else:
        issues.append(Issue("error", page.rel, 1, f"unknown kind `{kind}`; see CONTRIBUTING.md"))
    lines = page.text.count("\n")
    if lines > CARD_MAX_LINES and kind != "upstream":
        issues.append(Issue("warn", page.rel, 1, f"{lines} lines; split it (> {CARD_MAX_LINES})"))
    return issues


def is_page(path: Path) -> bool:
    """A knowledge page of this skill: a Markdown file outside the scripts, snapshots and review notes."""
    path = path.resolve()
    if path.suffix != ".md":
        return False
    rel = path.relative_to(SKILL_ROOT).as_posix() if path.is_relative_to(SKILL_ROOT) else path.as_posix()
    skipped = any(part in rel for part in SKIP_PARTS) or rel.endswith(".review.md")
    return not skipped


def knowledge_pages() -> list[Path]:
    return [path for path in sorted(SKILL_ROOT.rglob("*.md")) if is_page(path)]


def lint_paths(paths: Iterable[Path]) -> list[Issue]:
    issues = []
    for path in paths:
        issues += lint_page(read_page(path.resolve()))
    return issues


def stale(paths: Iterable[Path], version: str) -> list[str]:
    """Pages, and evidence lines, checked against another FlyDSL version."""
    out = []
    for path in paths:
        page = read_page(path.resolve())
        if page.kind in ("upstream", "entry", "guide", "index") or page.rel.startswith("triton/"):
            continue  # triton/ pages pin `triton`, not FlyDSL
        pinned = str((page.front or {}).get("flydsl") or "none")
        if pinned != version:
            out.append(f"{page.rel}: flydsl {pinned}")
        for item in read_evidence(page)[0].values():
            if item.flydsl and item.flydsl != version:
                out.append(f"{page.rel}:{item.line}: {item.ident} measured on FlyDSL {item.flydsl}")
    return out


def main(argv: list[str] | None = None) -> int:
    global HOME_ROOT, REPO_ROOT, SKILL_ROOT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="*", type=Path, help="pages to lint (default: the whole skill)")
    parser.add_argument(
        "--stale",
        metavar="VERSION",
        help="list what is not checked against VERSION, then exit",
    )
    parser.add_argument("--errors-only", action="store_true", help="do not print warnings")
    parser.add_argument(
        "--skill-root",
        type=Path,
        help="lint the skill in this folder instead of the one the tool is in",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="the repository the skill belongs to, for a copy of the skill",
    )
    args = parser.parse_args(argv)
    if args.skill_root:
        SKILL_ROOT = args.skill_root.resolve()
        REPO_ROOT = _repository_of(SKILL_ROOT)
    if args.repo_root:
        REPO_ROOT = args.repo_root.resolve()
    HOME_ROOT = skill_home()
    paths = [path for path in args.paths if is_page(path)] if args.paths else knowledge_pages()
    if args.stale:
        print("\n".join(stale(paths, args.stale)))
        return 0
    issues = lint_paths(paths) + duplicate_levers(knowledge_pages())
    errors = [i for i in issues if i.level == "error"]
    for issue in issues:
        if issue.level == "error" or not args.errors_only:
            print(issue)
    print(f"{len(paths)} pages: {len(errors)} errors, {len(issues) - len(errors)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
