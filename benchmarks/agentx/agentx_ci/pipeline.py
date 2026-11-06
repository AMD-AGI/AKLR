# SPDX-License-Identifier: MIT
"""Plan, run and report one model target locally."""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

from agentx_ci import plan, report_all, sweep
from agentx_ci.model_config import model_config, validate_model_id, validate_target_id
from agentx_ci.recipe_resolve import add_arguments, validate_options
from agentx_ci.utils import DEFAULT_CONFIG, REPO_ROOT, load_config, read_json, target_path, utc_now


def cleanup(root: Path, args: argparse.Namespace) -> bool:
    ok = True
    persistent = [Path(value).resolve() for value in (args.work, args.cache) if value is not None]
    for name in ("work", "cache"):
        if getattr(args, name) is not None:
            continue
        path = root / name
        if any(value == path or path in value.parents for value in persistent):
            continue
        try:
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
        except OSError as error:
            ok = False
            print(f"ERROR: unable to remove temporary directory {path}: {error}", file=sys.stderr)
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentx_bench pipeline", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model", required=True, help="One enabled model id; all and model lists are not accepted"
    )
    parser.add_argument("--target", help="One enabled target id; optional only when the model has one target")
    parser.add_argument("--hub", type=Path, required=True, help="Host HF hub cache with the pinned weights")
    parser.add_argument(
        "--out", type=Path, required=True, help="Fresh run directory containing plan/, results/ and site/"
    )
    parser.add_argument(
        "--work", type=Path, help="Persistent resolver/runtime workspace; default is deleted after the run"
    )
    parser.add_argument(
        "--cache", type=Path, help="Persistent kernel cache root; default cache is deleted after the run"
    )
    parser.add_argument("--run-id", help="Shared run identifier for all stages; generated if omitted")
    parser.add_argument("--inferencex-ref")
    parser.add_argument("--run-image", help="Image used by both the plan and local Docker execution")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--points")
    parser.add_argument("--duration", type=int, help="Optional per-point smoke duration override")
    parser.add_argument("--model-path", help="Optional model path visible inside the container")
    parser.add_argument("--git-dir", type=Path, default=REPO_ROOT, help="Checkout used for report history")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--baseline-dir", type=Path)
    source.add_argument("--baseline-url")
    add_arguments(parser)
    args = parser.parse_args(argv)
    root = args.out.resolve()
    created = False
    try:
        validate_options(args)
        validate_model_id(args.model)
        if args.target is not None:
            validate_target_id(args.target)
        definition = model_config(load_config(args.config), args.model, args.target)
        target_id = definition["target"]["id"]
        relative = target_path(args.model, target_id)
        if args.duration is not None and args.duration <= 0:
            raise ValueError("--duration must be positive")
        if root.exists():
            raise ValueError(f"pipeline output already exists: {root}; choose a fresh directory")
        if not args.hub.is_dir():
            raise ValueError(f"HF hub directory does not exist: {args.hub}")
        if args.run_id is not None and not args.run_id.strip():
            raise ValueError("--run-id cannot be empty")
        run_id = args.run_id or f"local-{uuid.uuid4().hex[:12]}"
        work = (args.work or root / "work").resolve()
        cache = (args.cache or root / "cache").resolve()
        if any(path.is_relative_to(REPO_ROOT.resolve()) for path in (root, work, cache)):
            raise ValueError("run, work and cache directories must be outside the source checkout")
        plan_args = argparse.Namespace(**vars(args), models=args.model, targets=target_id)
        plan_args.out, plan_args.work, plan_args.run_id = root / "plan", work, run_id
        root.mkdir(parents=True, exist_ok=False)
        created = True
        print(f"[{utc_now()}] pipeline {run_id}: creating plan in {root / 'plan'}", flush=True)
        manifest = plan.build(plan_args)
        targets = manifest.get("targets", [])
        if (
            len(targets) != 1
            or targets[0]["model"]["id"] != args.model
            or targets[0]["target"]["id"] != target_id
        ):
            raise ValueError("pipeline plan must contain exactly the requested model and target")
    except (ValueError, OSError, plan.rr.ResolveError) as error:
        print(f"ERROR: pipeline planning failed: {error}", file=sys.stderr)
        if created:
            cleanup(root, args)
        return 1

    failed = False
    interrupt_code = 0
    cleanup_safe = True
    target = targets[0]
    label = f"{args.model}/{target_id}"
    if target["status"] != "resolved":
        failed = True
        print(f"ERROR [{label}]: {target.get('error') or 'target plan was not resolved'}", file=sys.stderr)
    else:
        print(f"[{utc_now()}] pipeline {run_id}: testing {label}", flush=True)
        command = [
            "--serving",
            str(root / "plan" / relative / "serving.json"),
            "--hub",
            str(args.hub.resolve()),
            "--out",
            str(root / "results" / relative),
            "--cache",
            str(cache / relative),
            "--work",
            str(work / "sweeps" / relative),
            "--run-id",
            run_id,
        ]
        if args.model_path is not None:
            command.extend(["--model-path", args.model_path])
        try:
            code = sweep.main(command)
            if code:
                failed = True
                print(f"ERROR [{label}]: local benchmark exited with {code}", file=sys.stderr)
                if code in (130, 143, -2, -15):
                    interrupt_code = 128 - code if code < 0 else code
        except KeyboardInterrupt:
            interrupt_code = 130
            cleanup_safe = False
        except Exception as error:
            failed = True
            cleanup_safe = False
            print(f"ERROR [{label}]: {error}", file=sys.stderr)

    print(f"[{utc_now()}] pipeline {run_id}: generating report", flush=True)
    try:
        site = report_all.build(
            root / "plan/manifest.json",
            root / "results",
            root / "site",
            config=args.config,
            git_dir=args.git_dir,
            baseline_dir=args.baseline_dir,
            baseline_url=args.baseline_url,
        )
        failed |= any(item["status"] not in ("ok", "not-run") for item in site["targets"])
        print(f"Report: {root / 'site/index.html'}", flush=True)
    except Exception as error:
        failed = True
        print(f"ERROR: pipeline report failed: {error}", file=sys.stderr)
    results_path = root / "results" / relative / "results.json"
    try:
        if results_path.is_file():
            run = read_json(results_path)["run"]
            cleanup_safe &= isinstance(run, dict) and not run.get("container_cleanup_failed")
    except (OSError, ValueError, KeyError, TypeError):
        cleanup_safe = False
    if cleanup_safe:
        failed |= not cleanup(root, args)
    else:
        failed = True
        print(
            f"ERROR: temporary work and cache retained because container cleanup could not be confirmed: {root}",
            file=sys.stderr,
        )
    return interrupt_code or (1 if failed else 0)
