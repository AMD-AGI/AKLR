import argparse
import hashlib
import hmac
import importlib.util
import io
import json
import os
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

PATH = Path(__file__).parents[1] / "submit_crystal_qualification.py"
SPEC = importlib.util.spec_from_file_location("submit_crystal_qualification", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def arguments(**changes):
    values = {
        "crystal_url": "http://crystal.example.test:8080",
        "request_id": "aklr-12345",
        "repository": "AMD-AGI/AKLR",
        "pull_request": 42,
        "commit_sha": "b" * 40,
        "image": "ghcr.io/amd-agi/aklr-sglang@sha256:" + "a" * 64,
        "profile_id": "aklr-sglang-v1",
        "workflow_run_url": "https://github.com/AMD-AGI/AKLR/actions/runs/12345",
    }
    values.update(changes)
    return argparse.Namespace(**values)


class Response:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"run_id":"run-1","ui_url":"/runs/run-1"}'


class SubmitCrystalQualificationTests(unittest.TestCase):
    def test_builds_deterministic_body(self):
        body = json.loads(MODULE.request_body(arguments()))
        self.assertEqual(body["schema_version"], 2)
        self.assertEqual(body["pull_request"], 42)
        self.assertEqual(body["commit_sha"], "b" * 40)
        self.assertTrue(body["image"].endswith("@sha256:" + "a" * 64))

    def test_main_commit_has_no_invented_pull_request(self):
        body = json.loads(MODULE.request_body(arguments(pull_request=None)))
        self.assertEqual(body["schema_version"], 2)
        self.assertIsNone(body["pull_request"])
        self.assertEqual(body["commit_sha"], "b" * 40)

    def test_selected_profile_is_sent_without_embedding_its_recipe(self):
        body = json.loads(MODULE.request_body(arguments(profile_id="aklr-reviewed-pilot-v1")))
        self.assertEqual(body["profile_id"], "aklr-reviewed-pilot-v1")
        self.assertNotIn("tasks", body)
        self.assertNotIn("model", body)

    def test_invalid_pull_requests_are_not_treated_as_main(self):
        for value in [True, 0, -1, 1.5, "42", 2**31]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "pull request"):
                MODULE.request_body(arguments(pull_request=value))

    def test_signs_exact_body_and_posts_once(self):
        secret = "submission-secret-that-is-long-enough"
        args = arguments()
        with (
            mock.patch.dict(os.environ, {"CRYSTAL_AKLR_SUBMISSION_SECRET": secret}),
            mock.patch("urllib.request.OpenerDirector.open", return_value=Response()) as urlopen,
        ):
            result = MODULE.submit(args, timestamp=1234567890)

        request = urlopen.call_args.args[0]
        body = MODULE.request_body(args)
        signed = b"1234567890\n" + args.request_id.encode() + b"\n" + body
        expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        self.assertEqual(request.full_url, "http://crystal.example.test:8080/api/v1/qualification-runs")
        self.assertEqual(request.data, body)
        self.assertEqual(request.headers["X-crystal-signature"], f"sha256={expected}")
        self.assertEqual(result["run_id"], "run-1")
        urlopen.assert_called_once()

    def test_rejects_mutable_image_and_invalid_origin(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            MODULE.request_body(arguments(image="ghcr.io/amd-agi/aklr-sglang:latest"))
        with self.assertRaisesRegex(ValueError, "origin"):
            MODULE.crystal_endpoint("http://user:password@crystal.example.test:8080")

    def test_rejects_other_repositories_profiles_and_workflow_urls(self):
        for change in [
            {"repository": "AMD-AGI/SIKL"},
            {"profile_id": "../../arbitrary-profile"},
            {"profile_id": ""},
            {"profile_id": "a" * 65},
            {"workflow_run_url": "https://example.test/runs/42"},
            {"request_id": "aklr-123\nheader-injection"},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                MODULE.request_body(arguments(**change))

    def test_http_failure_does_not_include_response_or_url(self):
        error = urllib.error.HTTPError(
            "http://private-host:8080", 403, "private-model", {}, io.BytesIO(b"private-log")
        )
        with (
            mock.patch.dict(os.environ, {"CRYSTAL_AKLR_SUBMISSION_SECRET": "x" * 32}),
            mock.patch("urllib.request.OpenerDirector.open", side_effect=error),
            self.assertRaisesRegex(RuntimeError, "^Qualification submission returned HTTP 403$"),
        ):
            MODULE.submit(arguments())

    def test_rejects_redirect_without_reposting_signature(self):
        with self.assertRaisesRegex(RuntimeError, "redirect rejected"):
            MODULE.NoRedirect().redirect_request(None, None, 307, "redirect", {}, "https://other")

    def test_public_command_output_is_generic_on_success_and_failure(self):
        for error in [None, RuntimeError("http://private-host:8080 model-name score=0.95 /log")]:
            with (
                self.subTest(error=error),
                mock.patch.object(MODULE, "parser") as parser,
                mock.patch.object(
                    MODULE,
                    "submit",
                    side_effect=error,
                    return_value={"run_id": "private-run", "ui_url": "http://private-host"},
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                parser.return_value.parse_args.return_value = arguments()
                status = MODULE.main()
                self.assertEqual(status, int(error is not None))
                self.assertEqual(
                    output.getvalue(),
                    "Qualification submission failed; contact the service operator.\n"
                    if error
                    else "Qualification accepted. A GitHub Check will report the outcome.\n",
                )


if __name__ == "__main__":
    unittest.main()
