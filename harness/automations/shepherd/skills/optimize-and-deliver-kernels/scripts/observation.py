"""What an idle shepherd round waits on: the base branch's head, the heads of the open pull requests, this
checkout's commit (the pack and forge_task) and the readiness record.

`observation.py --config .shepherd/configuration.yaml` prints their fingerprint, which `state.py begin
--observation` compares with the one the last idle round saved. Exit 2: GitHub could not be read; begin
without an observation.
"""

import argparse
import hashlib
import json
import os
import pwd
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    base: str
    # (number, head SHA) of each open pull request.
    pulls: tuple
    checkout: str = ""
    # SHA-256 of `readiness.md` in the state directory, empty when it is missing.
    readiness: str = ""

    @classmethod
    def collect(cls, repository, branch, state_dir):
        def run(*args):
            return subprocess.run(args, capture_output=True, text=True, check=True).stdout

        def gh(*args):
            return run("gh", *args)

        if "USER" not in os.environ:  # getpwuid fails for a uid without a passwd entry: ask only when needed
            os.environ["USER"] = pwd.getpwuid(os.getuid()).pw_name
        readiness = Path(os.path.expandvars(state_dir)).expanduser() / "readiness.md"

        pulls = json.loads(
            gh(
                "pr",
                "list",
                "-R",
                repository,
                "--state",
                "open",
                "--limit",
                "1000",
                "--json",
                "number,headRefOid",
            )
        )
        return cls(
            base=gh("api", f"repos/{repository}/branches/{branch}", "--jq", ".commit.sha").strip(),
            pulls=tuple((pull["number"], pull["headRefOid"]) for pull in pulls),
            checkout=run("git", "rev-parse", "HEAD").strip(),
            readiness=hashlib.sha256(readiness.read_bytes()).hexdigest() if readiness.exists() else "",
        )

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps([self.base, sorted(self.pulls), self.checkout, self.readiness]).encode()
        ).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        import yaml

        with open(args.config) as stream:
            config = yaml.safe_load(stream)
        observation = Observation.collect(
            config["repository"], config["base_branch"], config["execution"]["state_dir"]
        )
    except Exception as exc:  # Whatever fails, the round then begins without an observation.
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    if not observation.base:
        return 2
    print(observation.fingerprint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
