"""Exercise path inspection with real Git history and mock GitHub writes."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "warn_aiter_changes.py"
SPEC = importlib.util.spec_from_file_location("warn_aiter_changes", SCRIPT)
warning = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = warning
SPEC.loader.exec_module(warning)


class GitInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git = warning.Git(self.root)
        self.git.run("init", "-q", "--initial-branch=main")
        self.git.run("config", "user.name", "Warning test")
        self.git.run("config", "user.email", "warning@example.invalid")
        self.git.run("config", "commit.gpgsign", "false")
        self.git.run("config", "core.hooksPath", str(self.root / "no-hooks"))
        self.git.run("remote", "add", "origin", str(self.root))

    def write(self, path, content="baseline\n"):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def commit(self, message="fixture"):
        self.git.run("add", "-A")
        self.git.run("commit", "-qm", message)
        return self.git.run("rev-parse", "HEAD").decode().strip()

    def seed(self):
        self.write("aiter/kernel.py")
        self.write("aklr/host.py")
        return self.commit()

    def pr(self, base, head):
        self.git.run("update-ref", "refs/pull/22/head", head)
        return warning.inspect_event(
            self.git,
            "pull_request_target",
            {
                "pull_request": {
                    "number": 22,
                    "base": {"sha": base},
                    "head": {"sha": head},
                }
            },
        )

    def push(self, before, after, ref="refs/heads/main", **extra):
        return warning.inspect_event(
            self.git,
            "push",
            {
                "before": before,
                "after": after,
                "ref": ref,
                "repository": {"default_branch": "main"},
                **extra,
            },
        )

    def test_staged_initial_commit_and_unstaged_files(self):
        self.write("aiter/added.cpp")
        self.git.run("add", "aiter/added.cpp")
        self.write("aiter/not staged.py")
        self.write("aklr/host.py")
        self.git.run("add", "aklr/host.py")
        self.assertEqual(self.git.changes(staged=True), [("A", "aiter/added.cpp")])
        self.assertEqual((self.root / "aiter/added.cpp").read_text(), "baseline\n")

    def test_unstaged_aiter_change_does_not_warn(self):
        self.seed()
        self.write("aiter/kernel.py", "unstaged\n")
        self.write("aklr/host.py", "staged\n")
        self.git.run("add", "aklr/host.py")
        self.assertEqual(self.git.changes(staged=True), [])

    def test_every_file_type_nested_names_and_terminal_controls(self):
        self.seed()
        names = [
            "aiter/dist/data.bin",
            "aiter/nested path/header.h",
            "aiter/new\n\x1b[31m.py",
        ]
        for name in names:
            self.write(name, "invalid python is intentionally never parsed!\n")
        self.git.run("add", "aiter")
        self.assertEqual(
            {path for _, path in self.git.changes(staged=True)}, set(names)
        )
        report = warning.Report("staged commit", current=self.git.changes(staged=True))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            warning.emit_report(report)
        self.assertNotIn("\x1b", stream.getvalue())
        self.assertIn(r"\n\u001b", stream.getvalue())

    def test_delete_and_moves_across_boundary(self):
        self.seed()
        self.git.run("mv", "aiter/kernel.py", "aklr/from_aiter.py")
        self.git.run("mv", "aklr/host.py", "aiter/from_aklr.py")
        self.assertEqual(
            set(self.git.changes(staged=True)),
            {("D", "aiter/kernel.py"), ("A", "aiter/from_aklr.py")},
        )

    def test_directory_replaced_by_symlink_and_similar_prefix_ignored(self):
        self.seed()
        self.git.run("rm", "aiter/kernel.py")
        (self.root / "aiter").symlink_to("aklr", target_is_directory=True)
        self.write("aiter-extra/unrelated.py")
        self.git.run("add", "-A")
        self.assertEqual(
            set(self.git.changes(staged=True)),
            {("D", "aiter/kernel.py"), ("A", "aiter")},
        )

    def test_change_then_restore_warns_on_push_and_draft_pr_history(self):
        base = self.seed()
        self.write("aiter/kernel.py", "changed\n")
        changed = self.commit()
        self.git.run("restore", f"--source={base}", "--", "aiter")
        restored = self.commit()
        for report in [self.push(base, restored), self.pr(base, restored)]:
            self.assertEqual(report.current, [])
            self.assertEqual([sha for sha, _ in report.commits], [changed, restored])
            self.assertTrue(report.needs_attention)
            self.assertIn("current diff is clean", report.status)
            self.assertEqual(report.incomplete, [])

    def test_merge_commit_and_introduced_side_branch_are_both_inspected(self):
        self.seed()
        self.git.run("checkout", "-qb", "feature")
        self.write("aiter/kernel.py", "feature\n")
        feature = self.commit()
        self.git.run("checkout", "-q", "main")
        self.write("aklr/host.py", "main advanced\n")
        base = self.commit()
        self.git.run("merge", "--no-ff", "-qm", "merge", "feature")
        head = self.git.run("rev-parse", "HEAD").decode().strip()
        report = self.push(base, head)
        self.assertEqual({sha for sha, _ in report.commits}, {feature, head})
        self.assertEqual(report.current, [("M", "aiter/kernel.py")])

    def test_force_push_backwards_reports_net_change_without_new_commits(self):
        base = self.seed()
        self.write("aiter/kernel.py", "removed by reset\n")
        old = self.commit()
        report = self.push(old, base, forced=True)
        self.assertEqual(report.current, [("M", "aiter/kernel.py")])
        self.assertEqual(report.commits, [])
        self.assertTrue(report.needs_attention)

    def test_new_branch_does_not_warn_for_inherited_aiter(self):
        self.seed()
        self.git.run("checkout", "-qb", "feature")
        self.write("aklr/host.py", "feature\n")
        head = self.commit()
        report = self.push(warning.ZERO_SHA, head, ref="refs/heads/feature")
        self.assertFalse(report.needs_attention)
        self.assertEqual(report.scanned, 1)

    def test_first_default_branch_push_inspects_root_commit(self):
        head = self.seed()
        report = self.push(warning.ZERO_SHA, head)
        self.assertEqual(report.current, [("A", "aiter/kernel.py")])
        self.assertEqual([sha for sha, _ in report.commits], [head])
        self.assertEqual(report.incomplete, [])

    def test_pr_uses_merge_base_when_main_advanced(self):
        self.seed()
        self.git.run("checkout", "-qb", "feature")
        self.write("aklr/host.py", "feature\n")
        head = self.commit()
        self.git.run("checkout", "-q", "main")
        self.write("aiter/kernel.py", "unrelated main change\n")
        base = self.commit()
        report = self.pr(base, head)
        self.assertFalse(report.needs_attention)
        self.assertEqual(report.base, base)
        self.assertEqual(report.scanned, 1)

    def test_truncated_history_is_not_reported_clean(self):
        base = self.seed()
        self.write("aiter/kernel.py", "old change\n")
        self.commit()
        self.git.run("restore", f"--source={base}", "--", "aiter")
        self.commit()
        self.write("aklr/host.py", "latest\n")
        head = self.commit()
        with patch.object(warning, "MAX_COMMITS", 1):
            report = self.push(base, head)
        self.assertEqual(report.current, [])
        self.assertEqual(report.commits, [])
        self.assertTrue(report.needs_attention)
        self.assertIn("incomplete", report.status)
        self.assertEqual((report.scanned, report.total), (1, 3))

    def test_missing_revision_is_incomplete(self):
        head = self.seed()
        report = self.push("f" * 40, head)
        self.assertTrue(report.incomplete)
        self.assertTrue(report.needs_attention)

    def test_deleted_branch_does_not_try_to_read_missing_head(self):
        report = self.push("f" * 40, warning.ZERO_SHA, deleted=True)
        self.assertFalse(report.needs_attention)

    def test_bad_sha_cannot_be_used_as_git_option(self):
        with self.assertRaises(warning.InspectionError):
            self.git.ensure_commit("--upload-pack=anything")

    def test_staged_warning_and_inspection_failure_both_exit_successfully(self):
        self.write("aiter/kernel.py")
        self.git.run("add", "aiter")
        with (
            patch.object(warning, "Git", return_value=self.git),
            patch.object(sys, "argv", [str(SCRIPT), "staged"]),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(warning.main(), 0)
            self.assertIn("WARNING", output.getvalue())
            with patch.object(
                self.git, "changes", side_effect=warning.InspectionError("unavailable")
            ):
                self.assertEqual(warning.main(), 0)
                self.assertIn("incomplete", output.getvalue())


class FakeGitHub(warning.GitHub):
    """In-memory GitHub API, including the real client's pagination loop."""

    def __init__(self, report):
        self.pr = {
            "head": {"sha": report.head},
            "base": {"sha": report.base},
            "labels": [],
        }
        self.comments = []
        self.labels = []
        self.writes = []

    def request(self, method, path, data=None):
        parsed = urllib.parse.urlsplit(path)
        route = parsed.path
        if method == "GET":
            if route == "/pulls/22":
                return self.pr
            items = self.comments if route == "/issues/22/comments" else self.labels
            page = int(urllib.parse.parse_qs(parsed.query)["page"][0])
            return items[(page - 1) * 100 : page * 100]
        self.writes.append((method, route, data))
        if method == "POST" and route == "/issues/22/comments":
            self.comments.append(
                {
                    "id": len(self.comments) + 1,
                    "user": {"login": "github-actions[bot]"},
                    **data,
                }
            )
        elif method == "PATCH":
            comment_id = int(route.rsplit("/", 1)[1])
            next(item for item in self.comments if item["id"] == comment_id).update(
                data
            )
        elif method == "POST" and route == "/labels":
            self.labels.append(data)
        elif method == "POST" and route == "/issues/22/labels":
            self.pr["labels"] = [{"name": name} for name in data["labels"]]
        elif method == "DELETE":
            self.pr["labels"] = []
        else:
            raise AssertionError((method, path, data))


class PublishingTests(unittest.TestCase):
    def setUp(self):
        self.report = warning.Report(
            "PR", base="a" * 40, head="b" * 40, current=[("M", "aiter/kernel.py")]
        )
        self.api = FakeGitHub(self.report)

    def publish(self):
        warning.publish_report(self.api, 22, self.report)

    def test_create_once_and_unchanged_update_is_noop(self):
        self.publish()
        self.assertEqual(len(self.api.comments), 1)
        self.assertEqual(self.api.pr["labels"], [{"name": warning.LABEL}])
        self.api.writes.clear()
        self.publish()
        self.assertEqual(self.api.writes, [])

    def test_restored_history_updates_same_comment_and_keeps_label(self):
        self.publish()
        self.report.current = []
        self.report.commits = [("c" * 40, [("M", "aiter/kernel.py")])]
        self.publish()
        self.assertEqual(len(self.api.comments), 1)
        self.assertIn("current diff is clean", self.api.comments[0]["body"])
        self.assertEqual(self.api.pr["labels"], [{"name": warning.LABEL}])

    def test_clean_rebase_updates_old_comment_and_removes_label(self):
        self.publish()
        self.report.current = []
        self.publish()
        self.assertEqual(len(self.api.comments), 1)
        self.assertIn("✅", self.api.comments[0]["body"])
        self.assertEqual(self.api.pr["labels"], [])

    def test_untouched_pr_has_no_comment_or_label(self):
        self.report.current = []
        self.publish()
        self.assertEqual(self.api.writes, [])

    def test_incomplete_scan_creates_warning_and_label(self):
        self.report.current = []
        self.report.incomplete = ["Missing revision"]
        self.publish()
        self.assertIn("incomplete", self.api.comments[0]["body"])
        self.assertEqual(self.api.pr["labels"], [{"name": warning.LABEL}])

    def test_stale_head_or_target_does_not_overwrite_current_report(self):
        for field in ["head", "base"]:
            with self.subTest(field=field), contextlib.redirect_stdout(io.StringIO()):
                self.api = FakeGitHub(self.report)
                self.api.pr[field]["sha"] = "c" * 40
                self.publish()
                self.assertEqual(self.api.writes, [])

    def test_pagination_and_human_comment_marker_do_not_create_duplicates(self):
        self.api.comments = [
            {"id": index, "user": {"login": "human"}, "body": warning.MARKER}
            for index in range(100)
        ]
        self.api.comments.append(
            {
                "id": 100,
                "user": {"login": "github-actions[bot]"},
                "body": warning.MARKER,
            }
        )
        self.api.labels = [{"name": f"unrelated-{index}"} for index in range(100)]
        self.api.labels.append({"name": warning.LABEL})
        self.publish()
        self.assertEqual(len(self.api.comments), 101)
        self.assertEqual(self.api.comments[0]["body"], warning.MARKER)
        self.assertIn("AITER changes", self.api.comments[-1]["body"])
        self.assertFalse(
            any(
                method == "POST" and path == "/labels"
                for method, path, _ in self.api.writes
            )
        )

    def test_markup_escaping_and_comment_size_remain_bounded(self):
        path = "aiter/</code><script>alert(1)</script>\n" + "x" * 1000
        self.report.current = [("M", path)] * 1000
        self.report.commits = [("c" * 40, self.report.current)] * 500
        body = warning.render_markdown(self.report)
        self.assertNotIn("<script>", body)
        self.assertIn("&lt;script&gt;", body)
        self.assertIn("970 more paths", body)
        self.assertLess(len(body), 60000)

    def test_api_delivery_failure_remains_advisory_and_keeps_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            summary_path = Path(directory) / "summary.md"
            event_path.write_text(json.dumps({"pull_request": {"number": 22}}))
            with (
                patch.dict(
                    warning.os.environ,
                    {
                        "GITHUB_EVENT_NAME": "pull_request_target",
                        "GITHUB_EVENT_PATH": str(event_path),
                        "GITHUB_STEP_SUMMARY": str(summary_path),
                        "GITHUB_TOKEN": "test",
                        "GITHUB_REPOSITORY": "owner/repo",
                    },
                ),
                patch.object(sys, "argv", [str(SCRIPT), "github", "--publish"]),
                patch.object(warning, "inspect_event", return_value=self.report),
                patch.object(
                    warning,
                    "publish_report",
                    side_effect=urllib.error.HTTPError(
                        "https://api.github.com", 403, "Forbidden", {}, None
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(warning.main(), 0)
            self.assertIn("HTTP 403", output.getvalue())
            self.assertIn(warning.MARKER, summary_path.read_text())

    def test_read_only_event_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text("{}")
            with (
                patch.dict(
                    warning.os.environ,
                    {
                        "GITHUB_EVENT_NAME": "pull_request",
                        "GITHUB_EVENT_PATH": str(event_path),
                        "GITHUB_STEP_SUMMARY": "",
                    },
                ),
                patch.object(sys, "argv", [str(SCRIPT), "github", "--publish"]),
                patch.object(warning, "inspect_event", return_value=self.report),
                patch.object(warning, "publish_report") as publish,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                warning.main()
            publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
