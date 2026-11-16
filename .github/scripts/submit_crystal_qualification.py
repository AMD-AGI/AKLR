"""Submit one signed, asynchronous AKLR qualification request to Crystal."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
IMAGE_DIGEST = re.compile(r"ghcr\.io/amd-agi/aklr-sglang@sha256:[0-9a-f]{64}")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Submission redirect rejected")


def crystal_endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Crystal URL must be an HTTP(S) origin")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/api/v1/qualification-runs", "", ""))


def request_body(args: argparse.Namespace) -> bytes:
    if args.repository != "AMD-AGI/AKLR":
        raise ValueError("repository must be AMD-AGI/AKLR")
    if args.pull_request < 1:
        raise ValueError("pull request must be positive")
    if COMMIT_SHA.fullmatch(args.commit_sha) is None:
        raise ValueError("commit SHA must contain 40 lowercase hexadecimal characters")
    if IMAGE_DIGEST.fullmatch(args.image) is None:
        raise ValueError("image must use an immutable sha256 digest")
    if args.profile_id != "aklr-sglang-v1":
        raise ValueError("unsupported qualification profile")
    if re.fullmatch(r"aklr-[1-9][0-9]*", args.request_id) is None:
        raise ValueError("invalid request ID")
    if (
        re.fullmatch(r"https://github\.com/AMD-AGI/AKLR/actions/runs/[1-9][0-9]*", args.workflow_run_url)
        is None
    ):
        raise ValueError("invalid workflow URL")
    payload = {
        "schema_version": 1,
        "repository": args.repository,
        "pull_request": args.pull_request,
        "commit_sha": args.commit_sha,
        "image": args.image,
        "profile_id": args.profile_id,
        "workflow_run_url": args.workflow_run_url,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def submit(args: argparse.Namespace, *, timestamp: int | None = None) -> dict:
    secret = os.environ.get("CRYSTAL_AKLR_SUBMISSION_SECRET", "").strip()
    if len(secret) < 32:
        raise ValueError("CRYSTAL_AKLR_SUBMISSION_SECRET must contain at least 32 characters")
    body = request_body(args)
    timestamp_text = str(int(time.time()) if timestamp is None else timestamp)
    signed = timestamp_text.encode() + b"\n" + args.request_id.encode() + b"\n" + body
    signature = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    request = urllib.request.Request(
        crystal_endpoint(args.crystal_url),
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Crystal-Timestamp": timestamp_text,
            "X-Crystal-Request-Id": args.request_id,
            "X-Crystal-Signature": f"sha256={signature}",
        },
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            if response.status != 202:
                raise RuntimeError(f"Crystal returned unexpected HTTP {response.status}")
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Qualification submission returned HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError):
        raise RuntimeError("Qualification service is unavailable") from None
    if not isinstance(result, dict) or not result.get("run_id"):
        raise RuntimeError("Crystal returned an invalid response")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--crystal-url", required=True)
    result.add_argument("--request-id", required=True)
    result.add_argument("--repository", default="AMD-AGI/AKLR")
    result.add_argument("--pull-request", required=True, type=int)
    result.add_argument("--commit-sha", required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--profile-id", required=True)
    result.add_argument("--workflow-run-url", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        submit(args)
    except Exception:
        # Public Actions logs must not include server responses, internal URLs or paths.
        print("Qualification submission failed; contact the service operator.")
        return 1
    print("Qualification accepted. A GitHub Check will report the outcome.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
