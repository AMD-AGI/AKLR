"""A kernel optimization request frozen from a GitHub issue, at `<execution.state_dir>/requests/<issue>.json`.

Only schema paths leave the issue's text, read once from the event that an allowed user caused; the
issue is never read again, so editing it changes nothing. `candidate.py list` admits an open request's
schemas through the `issue` strategy. One comment on the issue shows the request's campaigns and
becomes its final report. `gh` needs a token that may comment on and close issues.

`request.py freeze --config <file> --event <issues event JSON>` freezes the labeled issue's request and
prints it (`null` when the text names no schema); an issue that has an open request keeps it.
`request.py stop --config <file> --event <file>` stops the closed issue's request: nothing more is admitted.
`request.py list --config <file>` prints the open requests.
`request.py progress --config <file> --ledger <state.py show's output>` rewrites each open request's comment.
`request.py close --config <file> --issue <n> --ledger <file> [--listed <candidate.py list's output>]`
writes the final report, closes the issue and the request.
Exit 2: error.
"""

import argparse
import hashlib
import json
import os
import pwd
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# <op_dir>/impls/<language>/<name>/<file>.json, as `candidate.py` reads it; no `..` can match.
SCHEMA_PATH = re.compile(r"\baklr(?:/[A-Za-z0-9_-]+)+/impls(?:/[A-Za-z0-9_-]+){2}/[A-Za-z0-9_-]+\.json\b")


@dataclass(frozen=True)
class Request:
    issue: int
    schemas: tuple
    by: str
    at: str
    body_sha256: str
    dropped: int = 0
    comment: int | None = None
    stopped_at: str | None = None
    closed_at: str | None = None

    @classmethod
    def frozen(cls, event, limit):
        """The request in an `issues` event's payload, None when its text names no schema."""
        body = event["issue"].get("body") or ""
        schemas = tuple(dict.fromkeys(SCHEMA_PATH.findall(body)))
        if not schemas:
            return None
        return cls(
            issue=int(event["issue"]["number"]),
            schemas=schemas[:limit],
            by=event["sender"]["login"],
            at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            body_sha256=hashlib.sha256(body.encode()).hexdigest(),
            dropped=len(schemas[limit:]),
        )

    def report(self, campaigns, skipped):
        """The comment: the frozen request and each schema's campaigns, read from the ledger."""
        rows = []
        for schema in self.schemas:
            mine = [c for c in campaigns if c.get("issue") == self.issue and c.get("schema_path") == schema]
            for campaign in mine:
                regime = campaign.get("regime")
                regime = regime.get("name", "") if isinstance(regime, dict) else regime or ""
                rows.append(
                    f"| `{schema}` {regime} | {campaign['id']} "
                    f"| {campaign.get('outcome') or campaign.get('phase')} "
                    f"| {campaign.get('pr_url', '')} | {campaign.get('summary', '')} |"
                )
            if not mine:
                rows.append(f"| `{schema}` | | {skipped.get(schema, 'no campaign yet')} | | |")
        title = "Final report" if self.closed_at else "Progress"
        notes = [
            f"**shepherd request: {title}.** Frozen from this issue as `{self.by}` labeled it at {self.at}"
        ]
        notes.append(
            f"(text `{self.body_sha256[:12]}`). Later edits have no effect; for another request, open a new issue."
        )
        if self.dropped:
            notes.append(f"{self.dropped} more schema paths were over the batch size and dropped.")
        if self.stopped_at:
            notes.append(f"Closed at {self.stopped_at}: stopped, nothing more is admitted.")
        table = ["| Schema | Campaign | State | PR | Result |", "|---|---|---|---|---|", *rows]
        return "\n".join([" ".join(notes), "", *table])


@dataclass(frozen=True)
class RequestBoard:
    """The requests of one repository in a state directory, and their comments on GitHub."""

    repository: str
    state_dir: Path

    def gh(self, *command):
        return subprocess.run(["gh", *command], capture_output=True, text=True, check=True).stdout

    def opened(self):
        """The requests not closed yet, by issue."""
        requests = {}
        for path in sorted(self.state_dir.glob("requests/*.json")):
            value = json.loads(path.read_text())
            if not value.get("closed_at"):
                requests[value["issue"]] = Request(**{**value, "schemas": tuple(value["schemas"])})
        return requests

    def save(self, request, **changes):
        request = Request(**{**asdict(request), **changes})
        directory = self.state_dir / "requests"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = directory / f".{request.issue}.tmp"
        temporary.write_text(json.dumps(asdict(request)))
        os.replace(temporary, directory / f"{request.issue}.json")
        return request

    def freeze(self, event, limit):
        """The issue's open request, or the one frozen now from the event with its comment; None without a schema."""
        issue = int(event["issue"]["number"])
        comments = f"repos/{self.repository}/issues/{issue}/comments"
        request = self.opened().get(issue)
        if request:
            return request
        request = Request.frozen(event, limit)
        if request is None:
            nothing = "shepherd: no schema path (`aklr/.../impls/<language>/<name>/<file>.json`) in this issue; nothing was requested."
            self.gh("api", comments, "-f", f"body={nothing}")
            return None
        # The file comes first: a lost comment only loses the progress display.
        self.save(request)
        created = self.gh("api", comments, "-f", f"body={request.report([], {})}", "--jq", ".id")
        return self.save(request, comment=int(created))

    def publish(self, request, ledger, listed=None):
        """Rewrite the request's comment from the ledger (a file holding `state.py show`'s output)."""
        if not request.comment:
            return
        # A finished campaign's ledger entry is cut down; its archive holds the whole of it.
        campaigns = [
            json.loads((self.state_dir / c["archive"]).read_text()) if "archive" in c else c
            for c in json.loads(Path(ledger).read_text())["state"]["campaigns"]
        ]
        skipped = json.loads(Path(listed).read_text())["skipped"] if listed else ()
        reasons = {entry["path"]: entry["reason"] for entry in skipped}
        body = f"body={request.report(campaigns, reasons)}"
        self.gh(
            "api", "-X", "PATCH", f"repos/{self.repository}/issues/comments/{request.comment}", "-f", body
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "stop", "list", "progress", "close"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        if name in ("freeze", "stop"):
            command.add_argument("--event", required=True, help="the `issues` event payload")
        if name in ("progress", "close"):
            command.add_argument("--ledger", required=True, help="file holding `state.py show`'s output")
    sub.choices["close"].add_argument("--issue", required=True, type=int)
    sub.choices["close"].add_argument("--listed", help="file holding `candidate.py list`'s output")
    args = parser.parse_args()
    try:
        import yaml

        with open(args.config) as stream:
            config = yaml.safe_load(stream)
        if "USER" not in os.environ:
            os.environ["USER"] = pwd.getpwuid(os.getuid()).pw_name
        board = RequestBoard(
            config["repository"], Path(os.path.expandvars(config["execution"]["state_dir"])).expanduser()
        )
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if args.command in ("freeze", "stop"):
            event = json.loads(Path(args.event).read_text())
            if event["repository"]["full_name"] != board.repository:
                raise ValueError(
                    f"the event is of {event['repository']['full_name']}, not {board.repository}"
                )
        if args.command == "freeze":
            request = board.freeze(event, int(config["policy"]["batch_size"]))
            print(json.dumps(request and asdict(request)))
        elif args.command == "stop":
            request = board.opened().get(int(event["issue"]["number"]))
            if request and not request.stopped_at:
                board.save(request, stopped_at=now)
        elif args.command == "list":
            print(json.dumps([asdict(request) for request in board.opened().values()]))
        elif args.command == "progress":
            for request in board.opened().values():
                board.publish(request, args.ledger)
        else:
            # shortcut: a crash between the report and the file repeats both next time; they are idempotent.
            request = Request(**{**asdict(board.opened()[args.issue]), "closed_at": now})
            board.publish(request, args.ledger, args.listed)
            board.gh("issue", "close", str(request.issue), "-R", board.repository)
            board.save(request)
    except Exception as exc:  # Whatever fails, the caller reads one line and treats the request as unknown.
        detail = getattr(exc, "stderr", "") or ""
        print(json.dumps({"error": f"{type(exc).__name__}: {exc} {detail}".strip()}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
