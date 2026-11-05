# SPDX-License-Identifier: MIT
"""Collect serial model runs into a bilingual index and independent model reports/baselines."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from html import escape
from pathlib import Path

from agentx_ci.model_config import validate_model_id
from agentx_ci.recipe_resolve import POLICY, SelectionError, point_id, validate_manifest
from agentx_ci.report import STYLE, write_report
from agentx_ci.utils import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    complete_measurement,
    load_config,
    read_json,
    utc_now,
    validated_results,
    write_json,
)


def baseline_kind(results: dict) -> str:
    policy = results["run"].get("point_policy", "all")
    if policy not in ("all", POLICY):
        raise ValueError(f"unknown point policy: {policy}")
    return "pareto" if policy == POLICY else "live"


def eligible(results: dict | None, *, historical: bool = False) -> bool:
    """Complete formal runs update only the baseline for their selection policy."""
    if not results or not results.get("model"):
        return False
    run = results["run"]
    try:
        kind = baseline_kind(results)
        if kind == "pareto":
            selection = run["selection"]
            validate_manifest(selection)
            actual = [point_id(p) for p in results["points"]]
            if len(actual) != len(set(actual)) or set(actual) != set(selection["points"]):
                return False
    except (SelectionError, ValueError, KeyError, TypeError):
        return False
    return bool(
        run.get("mode") == "live"
        and complete_measurement(results)
        and (run.get("source_dirty") is False or (historical and "source_dirty" not in run))
        and run.get("arms_requested") == "all"
        and not run.get("points_requested")
        and run.get("duration_override_s") is None
        and not run.get("model_path_override")
    )


class PreviousSite:
    def __init__(self, directory: Path | None, url: str | None):
        self.directory, self.url = directory, url.rstrip("/") if url else None

    def read(self, relative: str) -> dict | None:
        if self.directory:
            path = self.directory / relative
            return read_json(path) if path.is_file() else None
        if self.url:
            try:
                with urllib.request.urlopen(f"{self.url}/{relative}", timeout=30) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                raise
        return None


def render_index(manifest: dict) -> str:
    labels = {
        "ok": "成功 / OK",
        "partial": "部分失败 / Partial",
        "failed": "失败 / Failed",
        "resolve-failed": "解析失败 / Resolve failed",
        "not-run": "本轮未选择 / Not selected",
    }
    rows = []
    for item in manifest["models"]:
        label = escape(item["display_name"])
        if item.get("report"):
            label = f'<a href="models/{item["id"]}/index.html">{label}</a>'
        baselines = (
            "<br>".join(f"{mode}: {escape(date)}" for mode, date in item.get("baselines", {}).items()) or "—"
        )
        rows.append(
            f"<tr><td>{label}<br><code>{item['id']}</code></td>"
            f"<td>{labels[item['status']]}</td><td>{escape(item.get('mode', '—'))}</td>"
            f"<td>{item.get('points_ok', 0)}/{item.get('points_total', 0)}</td>"
            f"<td>{escape(item.get('measured_at', '—'))}</td><td>{baselines}</td>"
            f"<td>{escape(item.get('error') or '')}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentX · 模型总览 / Models</title><style>{STYLE}</style></head><body>
<header><h1>AgentX · 模型总览 / Models</h1><div class="meta">Run {escape(manifest["run_id"])} · {escape(manifest["generated_at"])}</div></header>
<main><p>模型按串行运行，性能只与同一模型的历史基线比较。失败或未选择的模型如有历史报告，会明确标注测量时间。</p>
<p>Models run serially and are compared only with their own baselines. Historical reports retain their measurement date.</p>
<p>只有完整成功的正式测量更新基线；smoke 保留原基线。<br>
Only complete successful runs update baselines; smoke runs preserve them.
全量与前沿选点分别保存基线。 / Full-grid and Pareto-selected runs keep separate baselines.</p>
<div class="card" style="overflow-x:auto"><table><thead><tr><th>模型 / Model</th><th>本轮状态 / Status</th>
<th>模式 / Mode</th><th>成功点 / Points OK</th><th>测量时间 / Measured</th><th>有效基线 / Baselines</th><th>说明 / Details</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div></main></body></html>"""


def collect(
    plan: dict, results_dir: Path, out: Path, previous: PreviousSite, repo_url: str, git_dir: Path
) -> dict:
    prior_manifest = previous.read("manifest.json") or {"models": []}
    prior = {validate_model_id(item["id"]): item for item in prior_manifest["models"]}
    selected = {validate_model_id(item["id"]): item for item in plan["models"]}
    manifest = {"schema": 2, "run_id": plan["run_id"], "generated_at": utc_now(), "models": []}
    for model_id in dict.fromkeys([*selected, *prior]):
        planned = selected.get(model_id)
        item = dict(planned or prior[model_id])
        item.update(
            report=False, baselines={}, points_ok=0, points_total=0, error=None, mode="—", measured_at="—"
        )
        current = None
        if planned:
            path = results_dir / "models" / model_id / "results.json"
            if planned["status"] == "resolved" and path.is_file():
                try:
                    current = validated_results(read_json(path))
                    if (
                        current.get("model", {}).get("id") != model_id
                        or current["run"]["id"] != plan["run_id"]
                    ):
                        raise ValueError("result model/run identity does not match the plan")
                    if current["run"].get("mode") != "live":
                        raise ValueError("only live measurement results are supported")
                    if current["run"].get("point_policy", "all") != planned.get("point_policy", "all"):
                        raise ValueError("result selection policy does not match the plan")
                    if (
                        not planned.get("serving_sha256")
                        or current["run"].get("serving_sha256") != planned["serving_sha256"]
                        or not current["run"].get("attempt_id")
                    ):
                        raise ValueError(
                            "result serving identity does not match the plan; regenerate the plan/run"
                        )
                    identities = [point_id(p) for p in current["points"]]
                    expected = planned["point_ids"]
                    if (
                        len(identities) != len(set(identities))
                        or not set(identities) <= set(expected)
                        or current["run"]["points_expected"] != len(expected)
                    ):
                        raise ValueError("result points do not match the planned points")
                    if planned.get("point_policy") == POLICY:
                        selection = current["run"]["selection"]
                        validate_manifest(selection)
                        if selection["manifest_sha256"] != planned.get("selection_manifest_sha256"):
                            raise ValueError("result selection manifest does not match the plan")
                        identities = [point_id(p) for p in current["points"]]
                        if (
                            len(identities) != len(set(identities))
                            or not set(identities) <= set(selection["points"])
                            or current["run"]["points_expected"] != len(selection["points"])
                        ):
                            raise ValueError("result points do not match the selected plan")
                    ok = sum(p["status"] == "ok" for p in current["points"])
                    total = current["run"]["points_expected"]
                    complete = complete_measurement(current)
                    item["status"] = (
                        "ok" if complete and ok == total and total else "partial" if ok else "failed"
                    )
                    if item["status"] != "ok":
                        item["error"] = "测量未全部完成或有失败点 / Measurement incomplete or points failed"
                except (ValueError, KeyError, TypeError) as error:
                    current = None
                    item.update(status="failed", error=str(error))
            else:
                item.update(
                    status="resolve-failed" if planned["status"] != "resolved" else "failed",
                    error=planned.get("error") or "没有测量结果 / No measurement results",
                )
        else:
            item["status"] = "not-run"

        baselines = {}
        for kind in ("live", "pareto"):
            candidate = previous.read(f"models/{model_id}/baseline-{kind}.json")
            if candidate is not None:
                if (
                    candidate.get("model", {}).get("id") != model_id
                    or not eligible(candidate, historical=True)
                    or baseline_kind(candidate) != kind
                ):
                    raise ValueError(f"invalid {kind} baseline for {model_id}")
                baselines[kind] = candidate
        baseline = baselines.get(baseline_kind(current)) if current is not None else baselines.get("live")

        shown = current
        if shown is None:
            shown = previous.read(f"models/{model_id}/results.json")
            if shown is not None and shown.get("model", {}).get("id") != model_id:
                raise ValueError(f"historical results belong to another model: {model_id}")
        if shown is not None and shown["run"].get("mode") != "live":
            shown = baseline
        if shown is not None:
            mode = shown["run"]["mode"]
            notice = None
            if current is None:
                notice = f"Historical measurement / 历史测量：本轮状态 {item['status']}；测量时间 {shown['run']['finished_at']}"
            shown = validated_results(shown)
            item.update(
                report=True,
                mode=mode + (" / inferencex-pareto" if baseline_kind(shown) == "pareto" else ""),
                measured_at=shown["run"]["finished_at"],
                points_ok=sum(p["status"] == "ok" for p in shown["points"]),
                points_total=shown["run"]["points_expected"],
            )
            write_report(
                shown,
                baseline if current is not None else None,
                out / "models" / model_id,
                repo_url,
                git_dir,
                notice=notice,
            )
        if eligible(current):
            baselines[baseline_kind(current)] = current
        for kind, baseline in baselines.items():
            write_json(out / "models" / model_id / f"baseline-{kind}.json", baseline)
            item["baselines"][kind] = baseline["run"]["finished_at"]
        manifest["models"].append(item)
    write_json(out / "manifest.json", manifest)
    (out / "index.html").write_text(render_index(manifest))
    return manifest


def build(
    plan_path: Path,
    results: Path,
    out: Path,
    *,
    config: Path | str = DEFAULT_CONFIG,
    git_dir: Path = REPO_ROOT,
    baseline_dir: Path | None = None,
    baseline_url: str | None = None,
) -> dict:
    """Generate a planned report for the CLI, pipeline and CI."""
    if baseline_dir is not None and baseline_url is not None:
        raise ValueError("baseline_dir and baseline_url are mutually exclusive")
    if (out / "manifest.json").exists():
        raise ValueError("report output already exists; choose a fresh site directory")
    return collect(
        read_json(plan_path),
        results,
        out,
        PreviousSite(baseline_dir, baseline_url),
        load_config(config)["report"]["repo_url"].rstrip("/"),
        git_dir,
    )
