"""Shepherd's issue requests: which issues ask for which schemas, who approved them, and one progress comment each.

The flow follows Sweep's issue handling (a title prefix adds the label, the label starts work, one bot comment
is edited in place), written here from its behaviour; no Sweep code is used. Only members of the configured
team can request; GitHub replies are data, and every failure exits non-zero so the round treats it as unknown.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from pathlib import Path

import yaml


class RequestError(Exception):
    pass


@dataclass(frozen=True)
class Request:
    """One issue's request. `observe` gives the GitHub side; the ledger entry adds the round's status and version."""

    issue: int
    # GitHub side: active, paused, unapproved, unlabeled or closed. The ledger adds draining, stopping and done.
    status: str
    reason: str | None = None
    body_sha256: str | None = None
    schemas: tuple[str, ...] = ()
    max_prs: int | None = None
    max_campaigns: int | None = None
    prompt: str = ""
    version: int | None = None

    @classmethod
    def from_entry(cls, entry):
        names = {field.name for field in fields(cls)}
        known = {key: value for key, value in entry.items() if key in names}
        return cls(**{**known, "schemas": tuple(known.get("schemas", ()))})

    @classmethod
    def from_body(cls, issue, body):
        """The request in the body's first ```yaml block; the text around it is the requester's prompt."""
        digest = hashlib.sha256(body.encode()).hexdigest()
        block = re.search(r"^```ya?ml[ \t]*\n(.*?)^```[ \t]*$", body, re.S | re.M)
        if block is None:
            return cls(issue, "paused", "no_yaml_block", digest)
        try:
            spec = yaml.safe_load(block[1])
        except yaml.YAMLError:
            return cls(issue, "paused", "invalid_yaml", digest)
        if not isinstance(spec, dict):
            return cls(issue, "paused", "invalid_yaml", digest)
        if unknown := sorted(set(spec) - {"schemas", "max_prs", "max_campaigns"}):
            return cls(issue, "paused", f"unknown_keys:{','.join(map(str, unknown))}", digest)
        schemas = spec.get("schemas")
        if (
            not isinstance(schemas, list)
            or not schemas
            or not all(isinstance(path, str) for path in schemas)
            or len(set(schemas)) != len(schemas)
        ):
            return cls(issue, "paused", "schemas_not_a_list_of_unique_paths", digest)
        limits = {"max_prs": spec.get("max_prs", 1), "max_campaigns": spec.get("max_campaigns", len(schemas))}
        for key, value in limits.items():
            if type(value) is not int or value < 1:
                return cls(issue, "paused", f"{key}_not_a_positive_integer", digest)
        prompt = (body[: block.start()] + body[block.end() :]).strip()
        return cls(issue, "active", None, digest, tuple(schemas), **limits, prompt=prompt)


@dataclass(frozen=True)
class RequestBoard:
    """The repository's request label, the team whose members may request, and the App that writes for shepherd."""

    repository: str
    label: str
    team: str
    bot: str | None
    allowed_ops: tuple[str, ...]

    @classmethod
    def from_config(cls, path):
        config = yaml.safe_load(Path(path).read_text())
        requests = config.get("requests") or {}
        if not requests.get("team"):
            raise RequestError("configuration has no requests.team; requests are off")
        return cls(
            repository=config["repository"],
            label=requests.get("label") or "shepherd",
            team=requests["team"],
            # REST names the App `<slug>[bot]`, a login no person can hold.
            bot=requests.get("bot") or None,
            allowed_ops=tuple(config["policy"]["allowed_ops"]),
        )

    def gh(self, *args):
        result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RequestError(f"gh {args[0]} {args[1] if len(args) > 1 else ''} failed: {result.stderr.strip()}")
        return result.stdout

    def graphql(self, query, **variables):
        owner, name = self.repository.split("/")
        args = ["api", "graphql", "-f", f"query={query}", "-f", f"owner={owner}", "-f", f"name={name}"]
        for key, value in variables.items():
            args += ["-F" if isinstance(value, int) else "-f", f"{key}={value}"]
        return json.loads(self.gh(*args))["data"]

    @cached_property
    def members(self):
        org = self.repository.split("/")[0]
        return set(self.gh("api", "--paginate", f"orgs/{org}/teams/{self.team}/members", "--jq", ".[].login").split())

    def last_outsider_edit(self, issue):
        """When someone outside the team (or a deleted account) last edited the body, or None."""
        # shortcut: the last 100 edits only, enough unless a body is edited more often than that.
        return max(
            (edit["editedAt"] for edit in issue["userContentEdits"]["nodes"]
             if (edit.get("editor") or {}).get("login") not in self.members),
            default=None,
        )

    def label_new(self, numbers=()):
        """Label open issues titled `<label>:` that a team member opened, as Sweep does on `issues opened`.

        With issue numbers, only those (the runner passes the event's issue: search lags new issues); without, a search.
        """
        fields = "number state title author { login } labels(first: 100) { nodes { name } } " \
                 "userContentEdits(last: 100) { nodes { editedAt editor { login } } }"
        if numbers:
            issues = [
                self.graphql(
                    f"""query($owner: String!, $name: String!, $number: Int!) {{
                        repository(owner: $owner, name: $name) {{ issue(number: $number) {{ {fields} }} }} }}""",
                    number=number,
                )["repository"]["issue"]
                for number in numbers
            ]
        else:
            # shortcut: first 100 search hits only, page when requests outgrow that.
            issues = self.graphql(
                f"""query($q: String!) {{ search(query: $q, type: ISSUE, first: 100) {{
                    nodes {{ ... on Issue {{ {fields} }} }} }} }}""",
                q=f"repo:{self.repository} is:issue is:open in:title {self.label}",
            )["search"]["nodes"]
        labeled = []
        for issue in issues:
            if (
                issue["state"] == "OPEN"
                and issue["title"].lower().startswith(f"{self.label}:")
                and self.label not in {label["name"] for label in issue["labels"]["nodes"]}
                and (issue.get("author") or {}).get("login") in self.members
                # The App's label approves the author's text only; an outsider's edit waits for a member's label.
                and self.last_outsider_edit(issue) is None
            ):
                self.gh("api", "-X", "POST", f"repos/{self.repository}/issues/{issue['number']}/labels",
                        "-f", f"labels[]={self.label}")
                labeled.append(issue["number"])
        return labeled

    def trusts(self, login):
        """Whether an event's sender (a REST login) may start a paid round: a team member or the App itself."""
        return login == self.bot or login in self.members

    def requests(self, tracked):
        """Every open labeled issue and every tracked one, as Requests; untracked issues nobody approved are left out."""
        listed = self.graphql(
            """query($owner: String!, $name: String!, $label: String!) { repository(owner: $owner, name: $name) {
                issues(labels: [$label], states: OPEN, first: 100) { nodes { number } } } }""",
            label=self.label,
        )
        # shortcut: first 100 open labeled issues only, page when requests outgrow that.
        numbers = sorted(set(tracked) | {issue["number"] for issue in listed["repository"]["issues"]["nodes"]})
        found = []
        for number in numbers:
            issue = self.graphql(
                """query($owner: String!, $name: String!, $number: Int!) { repository(owner: $owner, name: $name) {
                    issue(number: $number) { state body author { login }
                    userContentEdits(last: 100) { nodes { editedAt editor { login } } }
                    labels(first: 100) { nodes { name } }
                    timelineItems(last: 100, itemTypes: [LABELED_EVENT]) { nodes { ... on LabeledEvent {
                        createdAt actor { __typename login } label { name } } } } } } }""",
                number=number,
            )["repository"]["issue"]
            if issue["state"] == "CLOSED":
                request = Request(number, "closed")
            elif self.label not in {label["name"] for label in issue["labels"]["nodes"]}:
                request = Request(number, "unlabeled")
            else:
                events = [e for e in issue["timelineItems"]["nodes"] if e["label"]["name"] == self.label]
                actor = (events[-1].get("actor") if events else None) or {}
                # GraphQL drops the `[bot]` suffix, so only the actor type tells the App from a person of that name.
                by_bot = actor.get("__typename") == "Bot" and f"{actor.get('login')}[bot]" == self.bot
                by_member = actor.get("__typename") == "User" and actor.get("login") in self.members
                author = (issue.get("author") or {}).get("login")
                outsider_edit = self.last_outsider_edit(issue)
                if not (by_member or (by_bot and author in self.members)):
                    request = Request(number, "unapproved", "labeler_not_in_team")
                elif outsider_edit is not None and not (
                    # Only a team member's label after every outsider edit approves the text; the App's label never does.
                    by_member and events[-1]["createdAt"] > outsider_edit
                ):
                    request = Request(number, "paused", "editor_not_in_team")
                else:
                    request = Request.from_body(number, issue["body"] or "")
                    for path in request.schemas:
                        op = re.fullmatch(r"aklr/ops/([a-z0-9_]+)/impls/[^/]+/[^/]+/[^/]+\.json", path)
                        if op is None or op[1] not in self.allowed_ops:
                            request = Request(number, "paused", f"schema_not_allowed:{path}", request.body_sha256)
                            break
            if number in tracked or request.status in ("active", "paused"):
                found.append(request)
        return found

    def write_progress(self, request, campaigns):
        """Create or edit in place the App's one progress comment on the issue; unchanged text writes nothing."""
        if self.bot is None:
            raise RequestError("configuration has no requests.bot; cannot tell shepherd's comment from others")
        marker = "<!-- shepherd-progress -->"
        rows = []
        for schema in request.schemas:
            mine = [c for c in campaigns if c.get("schema") == schema]
            rows += [
                f"| `{schema}` | {c['id']} | {c.get('phase', '')} | {'#' + str(c['pr']) if c.get('pr') else ''} |"
                for c in mine
            ] or [f"| `{schema}` | | not started | |"]
        state = {"done": "finished; close this issue when you are satisfied", "closed": "stopped: the issue is closed"}
        lines = [
            marker,
            f"**shepherd**, request version {request.version}: "
            f"{state.get(request.status, request.status)}{f' ({request.reason})' if request.reason else ''}.",
            "",
            "| Schema | Campaign | Phase | PR |",
            "|---|---|---|---|",
            *rows,
            "",
            f"PRs {sum(1 for c in campaigns if c.get('pr'))} of {request.max_prs}, "
            f"campaigns {len(campaigns)} of {request.max_campaigns}.",
        ]
        body = "\n".join(lines)
        comments = [
            json.loads(line)
            for line in self.gh(
                "api", "--paginate", f"repos/{self.repository}/issues/{request.issue}/comments",
                "--jq", ".[] | {id, body, login: .user.login, type: .user.type}",
            ).splitlines()
        ]
        mine = [
            c for c in comments if c["type"] == "Bot" and c["login"] == self.bot and c["body"].startswith(marker)
        ]
        if mine and mine[0]["body"] == body:
            return "unchanged"
        if mine:
            self.gh("api", "-X", "PATCH", f"repos/{self.repository}/issues/comments/{mine[0]['id']}", "-f", f"body={body}")
            return "updated"
        self.gh("api", "-X", "POST", f"repos/{self.repository}/issues/{request.issue}/comments", "-f", f"body={body}")
        return "created"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label", help="label open issues titled `<label>:` that a team member opened")
    label.add_argument("--issue", type=int, action="append", default=[], help="only this issue (repeatable)")
    trusts = sub.add_parser("trusts", help="exit 0 if the login may start a paid round, 3 if not")
    trusts.add_argument("--login", required=True)
    observe = sub.add_parser("observe", help="print every request with its GitHub-side status")
    observe.add_argument("--root", required=True, help="state directory, read for tracked issues")
    progress = sub.add_parser("progress", help="write the issue's progress comment from the ledger")
    progress.add_argument("--root", required=True)
    progress.add_argument("--issue", required=True, type=int)
    args = parser.parse_args()
    try:
        board = RequestBoard.from_config(args.config)
        if args.command == "label":
            result = {"labeled": board.label_new(args.issue)}
        elif args.command == "trusts":
            trusted = board.trusts(args.login)
            print(json.dumps({"trusted": trusted}))
            return 0 if trusted else 3
        else:
            state = json.loads((Path(args.root) / "state.json").read_text())["state"]
            entries = {entry["issue"]: entry for entry in state.get("requests", [])}
            if args.command == "observe":
                result = {"requests": [asdict(request) for request in board.requests(set(entries))]}
            else:
                if args.issue not in entries:
                    raise RequestError(f"issue {args.issue} is not a request in the ledger")
                campaigns = [c for c in state.get("campaigns", []) if c.get("request") == args.issue]
                result = {"comment": board.write_progress(Request.from_entry(entries[args.issue]), campaigns)}
    except (RequestError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
