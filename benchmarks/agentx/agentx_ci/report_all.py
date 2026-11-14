# SPDX-License-Identifier: MIT
"""Build a model/target/framework index with independent reports and baselines."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from html import escape
from pathlib import Path

from agentx_ci.model_config import validate_framework_id, validate_model_id, validate_target_id
from agentx_ci.performance_analysis import aklr_pairing_error
from agentx_ci.report import LANGS, language_switch, render_document, status_label, tr, write_report
from agentx_ci.serving_config import POLICY, SelectionError, point_id, validate_manifest
from agentx_ci.utils import (
    DEFAULT_CONFIG,
    LEGACY_METRIC_AXES,
    METRIC_AXES,
    REPO_ROOT,
    SCHEMA,
    complete_measurement,
    has_successful_measurement,
    load_config,
    read_json,
    target_path,
    utc_now,
    validated_results,
    write_json,
)


def target_key(item: dict) -> tuple[str, str, str]:
    try:
        key = (
            validate_model_id(item["model"]["id"]),
            validate_target_id(item["target"]["id"]),
            validate_framework_id(item["framework"]["id"]),
        )
        if "control" in item and target_key(item["control"]) != key:
            raise ValueError("AKLR modes belong to different model/target/framework runs")
        return key
    except (KeyError, TypeError) as error:
        raise ValueError(
            "model, target and framework identities are required; regenerate the plan/run"
        ) from error


def indexed_targets(manifest: dict) -> dict[tuple[str, str, str], dict]:
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("targets"), list):
        raise ValueError(f"run manifest schema {SCHEMA} is required; regenerate the plan/run")
    entries = {}
    for item in manifest["targets"]:
        key = target_key(item)
        if key in entries:
            raise ValueError(f"duplicate model/target/framework in manifest: {'/'.join(key)}")
        entries[key] = item
    return entries


def baseline_kind(results: dict) -> str:
    policy = results["run"].get("point_policy")
    if policy != POLICY:
        raise ValueError(f"unknown point policy: {policy}")
    return POLICY


def eligible(
    results: dict | None, *, historical: bool = False, metric_axes: tuple[str, str] = METRIC_AXES
) -> bool:
    """A formal run with any valid successful measurement can update its baseline."""
    if not results:
        return False
    try:
        target_key(results)
        run = results["run"]
        kind = baseline_kind(results)
        modes = [results, *([results["control"]] if "control" in results else [])]
        for mode in modes:
            if baseline_kind(mode) != kind or mode["run"].get("points_expected") != run.get(
                "points_expected"
            ):
                return False
            selection = mode["run"]["selection"]
            validate_manifest(selection)
            actual = [point_id(p) for p in mode["points"]]
            if (
                len(actual) != len(set(actual))
                or not set(actual) <= set(selection["points"])
                or mode["run"].get("points_expected") != len(selection["points"])
                or selection["manifest_sha256"] != run["selection"]["manifest_sha256"]
            ):
                return False
        return bool(
            run.get("mode") == "live"
            and (run.get("source_dirty") is False or (historical and "source_dirty" not in run))
            and run.get("arms_requested") == "all"
            and not run.get("points_requested")
            and run.get("duration_override_s") is None
            and not run.get("model_path_override")
            and has_successful_measurement(results, axes=metric_axes)
            and (historical or aklr_pairing_error(results) is None)
        )
    except (SelectionError, ValueError, KeyError, TypeError, AttributeError):
        return False


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
    titles = {
        lang: "AgentX · " + tr(lang, "Models, targets and frameworks", "模型、测试目标与框架")
        for lang in LANGS
    }
    pages = "\n".join(index_page(manifest, lang, titles[lang]) for lang in LANGS)
    return render_document(titles, pages)


def index_page(manifest: dict, lang: str, title: str) -> str:
    modes = {"live / pareto": tr(lang, "Live · Pareto-selected", "实测 · Pareto 选点")}
    baseline_labels = {"pareto": tr(lang, "Pareto-selected", "前沿选点")}
    rows = []
    for item in manifest["targets"]:
        model_id, target_id, framework_id = target_key(item)
        label = escape(item["model"]["display_name"])
        if item.get("report"):
            relative = target_path(model_id, target_id, framework_id).as_posix()
            label = f'<a href="{relative}/index.html?lang={lang}">{label}</a>'
        baseline_lines = [
            f"{escape(baseline_labels.get(mode, mode))}: {escape(date)}"
            for mode, date in item.get("baselines", {}).items()
        ]
        baseline_lines += [
            f"{escape(baseline_labels.get(mode, mode))}: "
            + tr(lang, "full-response P90 unavailable", "缺少完整响应 P90，暂不可比")
            for mode in item.get("baseline_unavailable", {})
        ]
        baselines = "<br>".join(baseline_lines) or "—"
        mode = item.get("mode", "—")
        error = str(item.get("error") or "")
        detail = (
            f'<br><span class="muted">{escape(error[:300])}{"…" if len(error) > 300 else ""}</span>'
            if error
            else ""
        )
        rows.append(
            f"<tr><td>{label}<br><code>{model_id}</code></td><td><code>{target_id}</code></td>"
            f"<td><code>{framework_id}</code></td>"
            f"<td>{escape(status_label(lang, item['status']))}{detail}</td><td>{escape(modes.get(mode, mode))}</td>"
            f"<td>{item.get('points_ok', 0)}/{item.get('points_total', 0)}</td>"
            f"<td>{escape(item.get('measured_at', '—'))}</td><td>{baselines}</td></tr>"
        )
    headers = (
        tr(lang, "Model", "模型"),
        tr(lang, "Target", "测试目标"),
        tr(lang, "Framework", "框架"),
        tr(lang, "Status", "本轮状态"),
        tr(lang, "Mode", "模式"),
        tr(lang, "Points OK", "成功点"),
        tr(lang, "Measured", "测量时间"),
        tr(lang, "Baselines", "有效基线"),
    )
    return f"""<div class="page" data-page="{lang}" lang="{tr(lang, "en", "zh-CN")}">
<header><div class="top"><h1>{escape(title)}</h1>{language_switch(lang)}</div>
<div class="meta">{tr(lang, "Run", "运行")} {escape(manifest["run_id"])} · {
        escape(manifest["generated_at"])
    }</div></header>
<main><p>{
        tr(
            lang,
            "Runs execute serially and are compared only with baselines for the same model, target and "
            "framework. Historical reports retain their measurement date.",
            "各运行串行执行，性能只与同一模型、同一目标、同一框架的历史基线比较。"
            "失败或未选择的运行如有历史报告，会明确标注测量时间。",
        )
    }</p><p>{
        tr(
            lang,
            "Formal runs from a clean source tree with verified execution environments update baselines "
            "when at least one measurement in either AKLR mode succeeds. Failed points remain recorded; "
            "comparisons use successful matched points, or Pareto frontiers when the selection, image or "
            "dependencies changed. Changed hardware, AgentX client or trace dataset leaves no baseline. "
            "Smoke runs preserve the previous baseline.",
            "源码干净、执行环境经校验的正式运行，只要 AKLR 开启或关闭任一侧至少一个测量点成功，就更新基线。"
            "失败点保留记录，比较使用成功且匹配的点，选点、镜像或依赖变化时比较 Pareto 前沿；硬件、AgentX client 或 trace 数据集变化时不作比较；冒烟验证保留原基线。",
        )
    }</p>
<div class="card" style="overflow-x:auto"><table><thead><tr>{
        "".join(f"<th>{label}</th>" for label in headers)
    }</tr>
</thead><tbody>{"".join(rows)}</tbody></table></div></main></div>"""


def collect(
    plan: dict, results_dir: Path, out: Path, previous: PreviousSite, repo_url: str, git_dir: Path
) -> dict:
    selected = indexed_targets(plan)
    prior_manifest = previous.read("manifest.json")
    if prior_manifest and prior_manifest.get("schema") != SCHEMA:
        print("Previous site predates serving frameworks; starting fresh baselines.")
        prior_manifest = None
    if prior_manifest is None:
        prior = {}
        previous = PreviousSite(None, None)
    else:
        prior = indexed_targets(prior_manifest)
    manifest = {"schema": SCHEMA, "run_id": plan["run_id"], "generated_at": utc_now(), "targets": []}
    for key in dict.fromkeys([*selected, *prior]):
        label = "/".join(key)
        relative = target_path(*key)
        planned = selected.get(key)
        item = dict(planned or prior[key])
        # Only plan/report failures fail the report job; sweep owns measurement failures.
        item.update(
            report=False,
            baselines={},
            baseline_unavailable={},
            points_ok=0,
            points_total=0,
            error=None,
            failure_stage=None,
            mode="—",
            measured_at="—",
        )
        current, current_archive = None, None
        if planned:
            path = results_dir / relative / "results.json"
            if planned["status"] == "resolved" and path.is_file():
                try:
                    current_archive = read_json(path)
                    if not isinstance(current_archive.get("control"), dict):
                        raise ValueError("AKLR enabled and disabled execution records are required")
                    current = validated_results(current_archive)
                    if target_key(current) != key or current["run"]["id"] != plan["run_id"]:
                        raise ValueError("result model/target/framework/run identity does not match the plan")
                    if current["run"].get("mode") != "live":
                        raise ValueError("only live measurement results are supported")
                    if current["run"].get("point_policy") != planned.get("point_policy"):
                        raise ValueError("result selection policy does not match the plan")
                    if (
                        not planned.get("serving_sha256")
                        or current["run"].get("serving_sha256") != planned["serving_sha256"]
                        or not current["run"].get("attempt_id")
                    ):
                        raise ValueError(
                            "result serving identity does not match the plan; regenerate the plan/run"
                        )
                    if any(
                        current["control"]["run"].get(field) != current["run"].get(field)
                        for field in ("id", "attempt_id", "serving_sha256")
                    ):
                        raise ValueError("AKLR modes do not belong to the same planned execution attempt")
                    expected = planned["point_ids"]
                    modes = ((current, "enabled"), (current["control"], "disabled"))
                    for mode, aklr_mode in modes:
                        if mode["run"].get("mode") != "live" or mode["run"].get("aklr_mode") != aklr_mode:
                            raise ValueError("AKLR mode labels are missing or invalid")
                        identities = [point_id(p) for p in mode["points"]]
                        if (
                            len(identities) != len(set(identities))
                            or not set(identities) <= set(expected)
                            or type(mode["run"].get("points_expected")) is not int
                            or mode["run"]["points_expected"] != len(expected)
                        ):
                            raise ValueError("result points do not match the planned points")
                        if mode["run"].get("point_policy") != planned.get("point_policy"):
                            raise ValueError("result selection policy does not match the plan")
                        selection = mode["run"]["selection"]
                        validate_manifest(selection)
                        if selection["manifest_sha256"] != planned.get("selection_manifest_sha256"):
                            raise ValueError("result selection manifest does not match the plan")
                        if not set(identities) <= set(selection["points"]) or mode["run"][
                            "points_expected"
                        ] != len(selection["points"]):
                            raise ValueError("result points do not match the selected plan")
                    ok = sum(
                        p["status"] == "ok"
                        for p in current["points"] + current.get("control", {}).get("points", [])
                    )
                    total = current["run"]["points_expected"] + current.get("control", {}).get("run", {}).get(
                        "points_expected", 0
                    )
                    complete = complete_measurement(current)
                    item["status"] = (
                        "ok" if complete and ok == total and total else "partial" if ok else "failed"
                    )
                    if item["status"] != "ok":
                        item["error"] = (
                            current["run"].get("error")
                            or "测量未全部完成或有失败点 / Measurement incomplete or points failed"
                        )
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    current, current_archive = None, None
                    item.update(status="failed", error=str(error), failure_stage="report")
            else:
                item.update(
                    status="resolve-failed" if planned["status"] != "resolved" else "failed",
                    error=planned.get("error") or "没有测量结果 / No measurement results",
                    failure_stage="plan" if planned["status"] != "resolved" else "report",
                )
        else:
            item["status"] = "not-run"

        baselines, report_baselines = {}, {}
        for kind in (POLICY,):
            candidate = previous.read((relative / f"baseline-{kind}.json").as_posix())
            if candidate is not None:
                if target_key(candidate) != key or baseline_kind(candidate) != kind:
                    raise ValueError(f"invalid {kind} baseline for {label}")
                normalized = validated_results(candidate)
                if eligible(normalized, historical=True):
                    baselines[kind] = candidate
                elif eligible(candidate, historical=True) or eligible(
                    candidate, historical=True, metric_axes=LEGACY_METRIC_AXES
                ):
                    # Preserve historical evidence even when its current or legacy
                    # cached axes cannot be verified. Only normalized views compare.
                    baselines[kind] = candidate
                    item["baseline_unavailable"][kind] = "historical baseline lacks full-response P90 ITL"
                else:
                    raise ValueError(f"invalid {kind} baseline for {label}")
                report_baselines[kind] = normalized
        baseline = (
            report_baselines.get(baseline_kind(current))
            if current is not None
            else report_baselines.get(POLICY)
        )

        shown = current_archive
        if shown is None:
            shown = previous.read((relative / "results.json").as_posix())
            if shown is not None and target_key(shown) != key:
                raise ValueError(f"historical results belong to another run: {label}")
        if shown is not None and shown["run"].get("mode") != "live":
            shown = baselines.get(POLICY)
        if shown is not None:
            mode = shown["run"]["mode"]
            notice = None
            if current is None:
                notice = {
                    lang: tr(
                        lang,
                        f"Historical measurement: this run is {status_label(lang, item['status'])}; "
                        f"measured at {shown['run']['finished_at']}.",
                        f"历史测量：本轮状态为{status_label(lang, item['status'])}；"
                        f"测量时间 {shown['run']['finished_at']}。",
                    )
                    + (
                        tr(lang, " Current failure: ", " 本轮失败原因：") + str(item["error"])
                        if item["error"]
                        else ""
                    )
                    for lang in LANGS
                }
            elif baseline_kind(current) in item["baseline_unavailable"]:
                notice = {
                    lang: tr(
                        lang,
                        "The historical baseline is preserved, but lacks full-response P90 ITL; "
                        "its E2E interactivity cannot be compared on this axis.",
                        "历史基线已保留，但缺少完整响应 P90 ITL；不能用其 E2E 交互速度比较当前横轴。",
                    )
                    for lang in LANGS
                }
            view = validated_results(shown)
            item.update(
                report=True,
                mode=f"{mode} / {baseline_kind(view)}",
                measured_at=view["run"]["finished_at"],
                points_ok=sum(
                    p["status"] == "ok" for p in view["points"] + view.get("control", {}).get("points", [])
                ),
                points_total=view["run"]["points_expected"]
                + view.get("control", {}).get("run", {}).get("points_expected", 0),
            )
            write_report(
                shown,
                baseline if current is not None else None,
                out / relative,
                repo_url,
                git_dir,
                notice=notice,
            )
        if eligible(current):
            baselines[baseline_kind(current)] = current_archive
            item["baseline_unavailable"].pop(baseline_kind(current), None)
        for kind, baseline in baselines.items():
            write_json(out / relative / f"baseline-{kind}.json", baseline)
            if kind not in item["baseline_unavailable"]:
                item["baselines"][kind] = baseline["run"]["finished_at"]
        manifest["targets"].append(item)
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
    """Build a site from a plan and its results."""
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
