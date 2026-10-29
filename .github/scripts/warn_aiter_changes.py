"""Warn about AITER path changes without linting, executing, or editing AITER."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "<!-- aklr:aiter-change-warning:v1 -->"
LABEL = "touches-aiter"
ZERO_SHA = "0" * 40
MAX_COMMITS = 500
POLICY = (
    "Avoid modifying aiter/ unless necessary. Explain the necessity, why the "
    "change cannot live in AKLR, and the validation performed in the PR. "
    "This reminder does not block commits or merges."
)


class InspectionError(RuntimeError):
    """Git metadata could not be inspected completely."""


@dataclass
class Report:
    scope: str
    base: str = ""
    head: str = ""
    current: list[tuple[str, str]] = field(default_factory=list)
    commits: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    scanned: int = 0
    total: int = 0

    @property
    def needs_attention(self) -> bool:
        return bool(self.current or self.commits or self.incomplete)

    @property
    def status(self) -> str:
        if self.current:
            return "AITER changes are present in the current diff"
        if self.incomplete:
            return "AITER inspection is incomplete; absence of changes is unverified"
        if self.commits:
            return "AITER changed in commit history; the current diff is clean"
        return "No AITER changes in the current diff or scanned commits"


class Git:
    def __init__(self, root: Path):
        self.root = root

    def run(self, *args: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                check=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise InspectionError(f"git {args[0]} could not complete") from error
        return result.stdout

    def ensure_commit(self, sha: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise InspectionError("Invalid commit SHA in event")
        try:
            self.run("cat-file", "-e", f"{sha}^{{commit}}")
        except InspectionError:
            self.run("fetch", "--no-tags", "origin", sha)
            self.run("cat-file", "-e", f"{sha}^{{commit}}")

    def changes(self, *revisions: str, staged: bool = False) -> list[tuple[str, str]]:
        # Disabling rename detection avoids reading blob contents. Renames across
        # the boundary appear as an A or D under aiter/, so both directions warn.
        args = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--name-status",
            "-z",
        ]
        if staged:
            args.append("--cached")
        raw = self.run(*args, *revisions, "--", "aiter")
        parts = raw.split(b"\0")
        if parts[-1] == b"":
            parts.pop()
        if len(parts) % 2:
            raise InspectionError("Unexpected Git path-list format")
        return [
            (parts[i].decode("ascii"), parts[i + 1].decode("utf-8", "surrogateescape"))
            for i in range(0, len(parts), 2)
        ]

    def empty_tree(self) -> str:
        # Store an empty tree object without altering the index or worktree.
        try:
            result = subprocess.run(
                ["git", "hash-object", "-w", "-t", "tree", "--stdin"],
                cwd=self.root,
                input=b"",
                capture_output=True,
                check=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise InspectionError("Could not determine the empty tree") from error
        return result.stdout.decode().strip()


def inspect_range(git: Git, report: Report, start: str) -> None:
    report.current = git.changes(start or git.empty_tree(), report.head)
    revisions = [report.head] if not start else [f"{start}..{report.head}"]
    commits = git.run("rev-list", "--reverse", *revisions, "--").decode().splitlines()
    report.total = len(commits)
    if len(commits) > MAX_COMMITS:
        report.incomplete.append(
            f"History has {len(commits)} commits; only the latest {MAX_COMMITS} were scanned."
        )
    for sha in commits[-MAX_COMMITS:]:
        parents = git.run("rev-list", "--parents", "-n", "1", sha).decode().split()
        parent = parents[1] if len(parents) > 1 else git.empty_tree()
        changes = git.changes(parent, sha)
        if changes:
            report.commits.append((sha, changes))
        report.scanned += 1


def inspect_event(git: Git, event_name: str, event: dict) -> Report:
    report = Report("PR" if event_name.startswith("pull_request") else "push")
    try:
        if event_name.startswith("pull_request"):
            pr = event["pull_request"]
            report.base, report.head = pr["base"]["sha"], pr["head"]["sha"]
            # Fetch objects only. In pull_request_target, the worktree and the
            # script remain at the trusted base revision, never at the PR head.
            number = int(pr["number"])
            git.run("fetch", "--no-tags", "origin", f"refs/pull/{number}/head")
            git.ensure_commit(report.base)
            git.ensure_commit(report.head)
            start = git.run("merge-base", report.base, report.head).decode().strip()
        elif event_name == "push":
            if event.get("deleted") or event.get("after") == ZERO_SHA:
                return report
            report.head = event["after"]
            git.ensure_commit(report.head)
            before = event["before"]
            if before != ZERO_SHA:
                git.ensure_commit(before)
                report.base = start = before
            else:
                default = event["repository"]["default_branch"]
                if event["ref"] == f"refs/heads/{default}":
                    start = ""  # The initial push of the default branch.
                else:
                    ref = f"refs/remotes/origin/{default}"
                    git.run(
                        "fetch", "--no-tags", "origin", f"refs/heads/{default}:{ref}"
                    )
                    start = git.run("merge-base", ref, report.head).decode().strip()
                report.base = start
        else:
            raise InspectionError(f"Unsupported event: {event_name}")
        inspect_range(git, report, start)
    except (InspectionError, KeyError, ValueError) as error:
        report.incomplete.append(str(error))
    return report


def quoted_path(path: str) -> str:
    # JSON quoting keeps newlines, terminal controls and non-UTF-8 names inert.
    display = json.dumps(path, ensure_ascii=True)
    return html.escape(display[:240] + ("..." if len(display) > 240 else ""))


def render_markdown(report: Report) -> str:
    icon = "⚠️" if report.needs_attention else "✅"
    lines = [MARKER, f"### {icon} {report.status}", "", POLICY, ""]
    if report.head:
        lines += [
            f"Head: `{report.head}`; comparison base: `{report.base or 'empty tree'}`.",
            "",
        ]
    lines += [f"**Current {report.scope} diff:** {len(report.current)} AITER path(s)."]
    for status, path in report.current[:30]:
        lines.append(f"- `{status}` <code>{quoted_path(path)}</code>")
    if len(report.current) > 30:
        lines.append(f"- ... {len(report.current) - 30} more paths; see the Git diff.")
    if report.scope != "staged commit":
        lines += [
            "",
            (
                f"**Commit history:** {len(report.commits)} AITER-touching commit(s); "
                f"{report.scanned}/{report.total} commits inspected."
            ),
        ]
        for sha, changes in report.commits[:20]:
            paths = ", ".join(f"<code>{quoted_path(p)}</code>" for _, p in changes[:3])
            suffix = f" (+{len(changes) - 3} more paths)" if len(changes) > 3 else ""
            lines.append(f"- `{sha}`: {paths}{suffix}")
        if len(report.commits) > 20:
            lines.append(f"- ... {len(report.commits) - 20} more affected commits.")
    if report.incomplete:
        lines += ["", "**Inspection warnings (do not treat this result as clean):**"]
        lines += [f"- {html.escape(reason)}" for reason in report.incomplete]
    lines += [
        "",
        "Ruff/Black still exclude `aiter/`. Build, GPU and downstream tests are unchanged.",
    ]
    return "\n".join(lines) + "\n"


def emit_report(report: Report, *, github: bool = False) -> None:
    if not report.needs_attention:
        return
    if github:
        message = (
            report.status.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        )
        print(f"::warning title=AITER directory policy::{message}")
    else:
        print(f"WARNING: {report.status}")
    print(POLICY)
    for status, path in report.current[:30]:
        print(f"  {status} {json.dumps(path, ensure_ascii=True)}")
    for sha, changes in report.commits[:20]:
        print(f"  commit {sha}: {len(changes)} AITER path(s)")
        for status, path in changes[:3]:
            print(f"    {status} {json.dumps(path, ensure_ascii=True)}")
    for reason in report.incomplete:
        print(f"  Inspection incomplete: {json.dumps(reason, ensure_ascii=True)}")


class GitHub:
    def __init__(self, token: str, repository: str, api_url: str):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
            raise ValueError("Invalid repository name")
        if not api_url.startswith("https://"):
            raise ValueError("GitHub API URL must use HTTPS")
        self.root = f"{api_url.rstrip('/')}/repos/{repository}"
        self.token = token

    def request(self, method: str, path: str, data: dict | None = None):
        request = urllib.request.Request(
            self.root + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
        return json.loads(body) if body else None

    def pages(self, path: str) -> list[dict]:
        items = []
        page = 1
        while True:
            batch = self.request("GET", f"{path}?per_page=100&page={page}")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1


def publish_report(api: GitHub, number: int, report: Report) -> None:
    pr = api.request("GET", f"/pulls/{number}")
    if pr["head"]["sha"] != report.head or pr["base"]["sha"] != report.base:
        print("Skipping a stale AITER report; the PR revision or target has changed.")
        return
    comments = api.pages(f"/issues/{number}/comments")
    owned = next(
        (
            comment
            for comment in comments
            if comment.get("user", {}).get("login") == "github-actions[bot]"
            and comment.get("body", "").startswith(MARKER)
        ),
        None,
    )
    body = render_markdown(report)
    if owned:
        if owned["body"] != body:
            api.request("PATCH", f"/issues/comments/{owned['id']}", {"body": body})
    elif report.needs_attention:
        api.request("POST", f"/issues/{number}/comments", {"body": body})

    attached = any(label["name"].casefold() == LABEL for label in pr["labels"])
    if report.needs_attention and not attached:
        labels = api.pages("/labels")
        if not any(label["name"].casefold() == LABEL for label in labels):
            try:
                api.request(
                    "POST",
                    "/labels",
                    {
                        "name": LABEL,
                        "color": "FBCA04",
                        "description": "AITER paths touched or inspection incomplete",
                    },
                )
            except urllib.error.HTTPError as error:
                if error.code != 422:  # Another PR may have created it concurrently.
                    raise
        api.request("POST", f"/issues/{number}/labels", {"labels": [LABEL]})
    elif attached and not report.needs_attention:
        api.request("DELETE", f"/issues/{number}/labels/{urllib.parse.quote(LABEL)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["staged", "github"])
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    git = Git(Path.cwd())
    if args.mode == "staged":
        report = Report("staged commit")
        try:
            report.current = git.changes(staged=True)
        except InspectionError as error:
            report.incomplete.append(str(error))
        emit_report(report)
        return 0

    event_name = os.environ["GITHUB_EVENT_NAME"]
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    report = inspect_event(git, event_name, event)
    emit_report(report, github=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(render_markdown(report))
    if args.publish:
        if event_name != "pull_request_target":
            parser.error("Publishing is only allowed from pull_request_target")
        try:
            api = GitHub(
                os.environ["GITHUB_TOKEN"],
                os.environ["GITHUB_REPOSITORY"],
                os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            )
            publish_report(api, int(event["pull_request"]["number"]), report)
        except (OSError, ValueError, KeyError) as error:
            # Publishing is advisory too. Never convert an API failure to a clean report.
            detail = (
                f"HTTP {error.code}"
                if isinstance(error, urllib.error.HTTPError)
                else type(error).__name__
            )
            print(
                f"::warning title=AITER reminder delivery::Could not update PR comment/label ({detail}); see the job summary."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
