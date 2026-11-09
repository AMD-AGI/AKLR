# SPDX-License-Identifier: MIT
"""Render target reports and planned sites."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from html import escape
from pathlib import Path

from agentx_ci.git_history import landed
from agentx_ci.performance_analysis import X, Y, compare, compare_aklr, frontier, frontiers, measured
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

ARM_COLORS = ("#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2")
COMBINED_COLOR = "#111827"
BASELINE_COLOR = "#9ca3af"
# Highlight threshold, not a significance test.
NOTABLE = 0.03
LANGS = ("en", "zh")
ZH_MESSAGES = [
    (r"measurement incomplete or points failed", "测量未完成或有失败点，不计算周环比"),
    (r"source is dirty or its state is unknown", "源码包含未提交改动或状态未知，不计算周环比"),
    (r"point selection policy or selected recipes changed", "选点策略或选中的 recipe 已变化"),
    (r"model identity differs or is missing", "模型标识不同或缺失"),
    (r"target identity differs or is missing", "测试目标标识不同或缺失"),
    (r"AKLR modes have different or missing model identities\.", "两种 AKLR 模式的模型标识不同或缺失。"),
    (r"AKLR modes have different or missing target identities\.", "两种 AKLR 模式的测试目标标识不同或缺失。"),
    (
        r"measurement settings changed \(weights, launch, workload or selected points\)",
        "测量设置已变化（权重、启动配置、工作负载或所选数据点）",
    ),
    (r"arm absent from the baseline", "上次运行没有这个实验组"),
    (r"serving config changed \(arm (\S+) → (\S+)\)", r"serving 配置已变化（arm \1 → \2）"),
    (r"no previous run", "没有上次运行"),
    (r"previous run is (\S+), this one is (\S+)", r"上次是 \1 模式，本次是 \2 模式"),
    (r"run image changed", "运行镜像已变化"),
    (r"concurrency grid changed", "并发网格已变化"),
    (r"execution identity is missing or unverified", "执行身份缺失或未经验证"),
    (
        r"execution image, InferenceX source, dependencies, hardware or traces changed",
        "执行镜像、InferenceX 源码、依赖、硬件或 traces 已变化",
    ),
]


def tr(lang: str, en: str, zh: str) -> str:
    return zh if lang == "zh" else en


def localize(lang: str, message: str) -> str:
    if lang == "zh":
        for pattern, zh in ZH_MESSAGES:
            match = re.fullmatch(pattern, message)
            if match:
                return match.expand(zh)
    return message


def text_width(text: str) -> float:
    return sum(12.0 if ord(ch) >= 0x2E80 else 6.8 for ch in text)


def arm_color(arm: str, arms: list[str]) -> str:
    return ARM_COLORS[arms.index(arm) % len(ARM_COLORS)]


def nice_ticks(high: float, count: int = 5) -> list[float]:
    raw = high / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return [i * step for i in range(int(high / step) + 2)]


def svg_chart(title: str, series: list[dict], lang: str, width: int = 620, height: int = 380) -> str:
    """Render scatter points and Pareto frontiers."""
    left, right, bottom = 64, 16, 52
    legend_rows: list[list[tuple[dict, float]]] = [[]]
    row_x = 0.0
    for s in series:
        item_w = 26 + text_width(s["label"])
        if legend_rows[-1] and row_x + item_w > width - left - right:
            legend_rows.append([])
            row_x = 0.0
        legend_rows[-1].append((s, row_x))
        row_x += item_w + 16
    top = 40 + 18 * len(legend_rows)
    height += 18 * (len(legend_rows) - 1)
    extent = [
        point
        for chart_series in series
        for point in [*chart_series["points"], *(chart_series.get("frontier") or [])]
    ]
    xs = [point[X] for point in extent] or [1.0]
    ys = [point[Y] for point in extent] or [1.0]
    x_ticks, y_ticks = nice_ticks(max(xs) * 1.08), nice_ticks(max(ys) * 1.08)
    x_max, y_max = x_ticks[-1], y_ticks[-1]
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(v: float) -> float:
        return left + v / x_max * plot_w

    def sy(v: float) -> float:
        return top + plot_h - v / y_max * plot_h

    out = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{escape(title)}">',
        f'<text x="{left}" y="22" class="chart-title">{escape(title)}</text>',
    ]
    for t in x_ticks:
        out.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" y2="{top + plot_h}" class="grid"/>')
        out.append(
            f'<text x="{sx(t):.1f}" y="{top + plot_h + 16}" class="tick" text-anchor="middle">{t:g}</text>'
        )
    for t in y_ticks:
        out.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>')
        out.append(f'<text x="{left - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    out.append(
        f'<text x="{left + plot_w / 2}" y="{height - 12}" class="axis" text-anchor="middle">'
        + tr(lang, "P90 E2E interactivity", "P90 端到端交互速度")
        + " (tok/s/user) →</text>"
    )
    out.append(
        f'<text transform="translate(16 {top + plot_h / 2}) rotate(-90)" class="axis" '
        'text-anchor="middle">' + tr(lang, "throughput per GPU", "单 GPU 吞吐") + " (tok/s/GPU) →</text>"
    )
    for s in series:
        line = s.get("frontier") or []
        if len(line) > 1:
            path = " ".join(f"{sx(p[X]):.1f},{sy(p[Y]):.1f}" for p in line)
            dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{s["color"]}" stroke-width="2"{dash}/>'
            )
        for p in s["points"]:
            pareto_coordinates = s.get("pareto_coordinates")
            off_frontier = pareto_coordinates is not None and (p[X], p[Y]) not in pareto_coordinates
            out.append('<g class="non-pareto-point">' if off_frontier else "<g>")
            fill = "white" if s.get("hollow") else s["color"]
            tip = f"{s['label']} {p.get('arm', '')} c{p['conc']}: {p[Y]:.1f} tok/s/GPU, {p[X]:.1f} tok/s/user"
            out.append(
                f'<circle cx="{sx(p[X]):.1f}" cy="{sy(p[Y]):.1f}" r="4.5" fill="{fill}" '
                f'stroke="{s["color"]}" stroke-width="1.6"><title>{escape(tip)}</title></circle>'
            )
            if s.get("labels"):
                out.append(
                    f'<text x="{sx(p[X]) + 7:.1f}" y="{sy(p[Y]) - 6:.1f}" class="point-label">c{p["conc"]}</text>'
                )
            out.append("</g>")
    for i, row in enumerate(legend_rows):
        y = 44 + i * 18
        for s, offset in row:
            x = left + offset
            dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
            out.append(
                f'<line x1="{x:.1f}" x2="{x + 18:.1f}" y1="{y - 4}" y2="{y - 4}" stroke="{s["color"]}" stroke-width="2"{dash}/>'
            )
            out.append(f'<text x="{x + 24:.1f}" y="{y}" class="legend">{escape(s["label"])}</text>')
    out.append("</svg>")
    return "\n".join(out)


def pct(r: float | None) -> str:
    return "—" if r is None else f"{(r - 1) * 100:+.1f}%"


def delta_class(r: float | None) -> str:
    if r is None or abs(r - 1) < NOTABLE:
        return ""
    return "good" if r > 1 else "bad"


def num(value) -> str:
    return "—" if value is None else f"{value:.1f}"


def short(sha: str) -> str:
    return sha[:10]


def charts(results: dict, baseline: dict | None, comparison: dict, lang: str) -> str:
    control = results.get("control")
    if control is not None:
        title = tr(lang, "AKLR enabled vs disabled · Pareto frontiers", "开启与关闭 AKLR · Pareto 前沿")
        series = []
        for data, enabled in ((control, False), (results, True)):
            points = measured(data)
            line = frontier(points)
            series.append(
                {
                    "label": tr(lang, "AKLR enabled", "开启 AKLR")
                    if enabled
                    else tr(lang, "AKLR disabled", "关闭 AKLR"),
                    "color": "#16a34a" if enabled else "#64748b",
                    "points": points,
                    "frontier": line,
                    "dashed": not enabled,
                    "hollow": not enabled,
                    "labels": True,
                    "pareto_coordinates": {(p[X], p[Y]) for p in line},
                }
            )
    else:
        title = tr(lang, "All arms · combined Pareto frontier", "全部实验组 · 合并 Pareto 前沿")
        current = frontiers(results)
        pareto_coordinates = {(p[X], p[Y]) for p in current["combined"]}
        arms = sorted(current["arms"])
        series = [
            {
                "label": tr(lang, f"{arm} (this run)", f"{arm}（本次）"),
                "color": arm_color(arm, arms),
                "points": measured(results, arm),
                "labels": True,
                "pareto_coordinates": pareto_coordinates,
            }
            for arm in arms
        ]
        series.append(
            {
                "label": tr(lang, "combined frontier", "合并前沿"),
                "color": COMBINED_COLOR,
                "points": [],
                "frontier": current["combined"],
            }
        )
    if comparison["status"] == "ok" and all(
        view["status"] == "comparable" for view in comparison["arms"].values()
    ):
        series.append(
            {
                "label": tr(lang, "AKLR enabled · previous run", "开启 AKLR · 上次运行")
                if control is not None
                else tr(lang, "previous run frontier", "上次前沿"),
                "color": BASELINE_COLOR,
                "points": [],
                "frontier": frontier(measured(baseline)),
                "dashed": True,
            }
        )
    chart = svg_chart(title, series, lang, width=1000, height=380)
    toggle = (
        '<label class="chart-toggle"><input type="checkbox" data-show-non-pareto checked> '
        + tr(lang, "Show non-Pareto points", "显示非 Pareto 点")
        + "</label>"
    )
    return f'<div class="card">{toggle}{chart}</div>'


def aklr_section(results: dict, lang: str) -> str:
    control = results.get("control")
    if control is None:
        return ""
    comparison = compare_aklr(results)
    out = [
        f"<h2>{tr(lang, 'AKLR performance impact', 'AKLR 性能收益')}</h2>",
        '<p class="muted">'
        + tr(
            lang,
            "Same run, image, hardware, weights and traces. Disabled mode runs first. Higher and farther right is better. "
            "Deltas compare identical arm/concurrency pairs, not interpolated points at equal interactivity; no statistical significance is implied.",
            "同一次运行，使用相同镜像、硬件、权重和 traces，先关闭再开启 AKLR。曲线越靠右上越好。"
            "增幅按相同实验组和并发点计算，不代表相同交互速度下的插值对比，也不代表统计显著性。",
        )
        + "</p>",
    ]
    if comparison["status"] != "ok":
        out.append(
            '<p class="notes">'
            + tr(
                lang,
                "Paired comparison unavailable: both modes must complete with matching settings. Curves show available measurements only.",
                "暂不计算成对增幅：两种模式需完整成功且配置匹配。曲线仅展示已获得的测量点。",
            )
            + f' <span class="muted">{escape(localize(lang, comparison["reason"]))}</span></p>'
        )
        return "\n".join(out)
    gain = comparison["tput_geomean_ratio"]
    out.append(
        '<p class="card">'
        + tr(lang, "Throughput geomean, enabled / disabled: ", "吞吐几何平均，开启 / 关闭：")
        + f'<strong class="{delta_class(gain)}" style="font-size:24px">{gain:.3f}× ({pct(gain)})</strong>'
        + tr(
            lang,
            f" · {len(comparison['points'])} matched pairs",
            f" · {len(comparison['points'])} 对匹配测量点",
        )
        + "</p>"
    )
    rows = []
    for row in comparison["points"]:
        on, off = row["enabled"], row["disabled"]
        rows.append(
            f"<tr><td>{escape(row['arm'])}</td><td>{row['conc']}</td>"
            f"<td>{num(off[Y])}</td><td>{num(on[Y])}</td>"
            f'<td class="{delta_class(row["tput_ratio"])}">{pct(row["tput_ratio"])}</td>'
            f"<td>{num(off[X])}</td><td>{num(on[X])}</td>"
            f'<td class="{delta_class(row["intvty_ratio"])}">{pct(row["intvty_ratio"])}</td></tr>'
        )
    out.append(
        '<div class="card" style="overflow-x:auto"><table><thead><tr>'
        + "".join(
            f"<th>{label}</th>"
            for label in (
                tr(lang, "Arm", "实验组"),
                tr(lang, "Concurrency", "并发"),
                tr(lang, "Off tok/s/GPU", "关闭 tok/s/GPU"),
                tr(lang, "On tok/s/GPU", "开启 tok/s/GPU"),
                tr(lang, "Throughput Δ", "吞吐变化"),
                tr(lang, "Off P90 tok/s/user", "关闭 P90 tok/s/user"),
                tr(lang, "On P90 tok/s/user", "开启 P90 tok/s/user"),
                tr(lang, "Interactivity Δ", "交互速度变化"),
            )
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return "\n".join(out)


def comparison_tables(comparison: dict, lang: str) -> str:
    if comparison["status"] != "ok":
        reason = escape(localize(lang, comparison["reason"]))
        return (
            '<p class="muted">'
            + tr(lang, f"No week-over-week comparison: {reason}.", f"没有周环比：{reason}。")
            + "</p>"
        )
    out = []
    if comparison["notes"]:
        items = "".join(f"<li>{escape(localize(lang, n))}</li>" for n in comparison["notes"])
        out.append(f'<ul class="notes">{items}</ul>')
    for arm, view in comparison["arms"].items():
        badge_class = view["status"]
        reason = localize(lang, view.get("reason", ""))
        badge = {
            "comparable": (
                tr(lang, "comparable", "可比"),
                tr(lang, "same serving config as the previous run", "serving 配置与上次相同"),
            ),
            "new-segment": (tr(lang, "new segment", "新分段"), reason),
            "new-arm": (tr(lang, "new arm", "新实验组"), reason),
        }[view["status"]]
        if view["status"] == "comparable" and not any(
            (row["baseline"] or {}).get("status") == "ok" for row in view["points"]
        ):
            badge_class = "no-baseline"
            badge = (
                tr(lang, "no baseline", "无基线"),
                tr(
                    lang,
                    "same serving config, but the previous run has no successful point to compare",
                    "serving 配置相同，但上次没有成功的数据点可比",
                ),
            )
        head = f'<h3>{escape(arm)} <span class="badge {badge_class}">{badge[0]}</span> <span class="muted">{escape(badge[1])}</span></h3>'
        if view["status"] == "new-segment":
            changed = ", ".join(f"c{c}" for c in view["changed_concs"]) or tr(lang, "none", "无")
            out.append(
                head
                + '<p class="muted">'
                + tr(
                    lang,
                    f"Points whose serving config changed: {changed}. Values are not compared.",
                    f"serving 配置变化的点：{changed}。不做数值比较。",
                )
                + "</p>"
            )
            continue
        if view["status"] != "comparable":
            out.append(head)
            continue
        rows = []
        for row in view["points"]:
            cur, old = row["current"], row["baseline"] or {}
            t, i = row.get("tput_ratio"), row.get("intvty_ratio")
            status = (
                cur["status"]
                if cur["status"] != "ok"
                else (tr(lang, "no baseline", "无基线") if not old or old.get("status") != "ok" else "")
            )
            rows.append(
                f"<tr><td>c{row['conc']}</td>"
                f"<td>{num(cur.get(Y))}</td><td>{num(old.get(Y))}</td><td class='{delta_class(t)}'>{pct(t)}</td>"
                f"<td>{num(cur.get(X))}</td><td>{num(old.get(X))}</td><td class='{delta_class(i)}'>{pct(i)}</td>"
                f"<td class='muted'>{escape(status)}</td></tr>"
            )
        geo = view.get("tput_geomean_ratio")
        previous = tr(lang, "previous", "上次")
        out.append(
            head + f"<table class='compact'><thead><tr><th>{tr(lang, 'conc', '并发')}</th>"
            f"<th>tok/s/GPU</th><th>{previous}</th><th>Δ</th>"
            f"<th>P90 tok/s/user</th><th>{previous}</th><th>Δ</th><th></th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table><p class='muted'>"
            + tr(lang, "Throughput geomean vs previous: ", "吞吐几何平均 vs 上次：")
            + f"<span class='{delta_class(geo)}'>{pct(geo)}</span></p>"
        )
    return "\n".join(out)


def commit_link(sha: str, repo_url: str) -> str:
    return f'<a href="{escape(repo_url)}/commit/{escape(sha)}"><code>{short(sha)}</code></a>'


def history_section(history: dict | None, this: str, repo_url: str, lang: str) -> str:
    if history is None:
        link = commit_link(this, repo_url)
        return (
            "<p>"
            + tr(
                lang,
                f"This run {link} · first run, no previous run to compare with.",
                f"本次 {link} · 首次运行，没有可比较的上次运行。",
            )
            + "</p>"
        )
    base, head = history["base"], history["head"]
    compare_url = f"{repo_url}/compare/{base}...{head}"
    line = (
        f"<p>{tr(lang, 'Previous run', '上次')} {commit_link(base, repo_url)} → "
        f"{tr(lang, 'this run', '本次')} "
        f"{commit_link(head, repo_url)} · "
        f'<a href="{escape(compare_url)}">{tr(lang, "compare on GitHub", "在 GitHub 上对比")}</a> · '
        f'<span class="muted">git log --first-parent {short(base)}..{short(head)}</span></p>'
    )
    if history["error"]:
        return (
            line
            + f'<p class="bad">{tr(lang, "History unavailable: ", "无法获取历史：")}'
            + f"{escape(history['error'])}</p>"
        )
    if not history["entries"]:
        return (
            line
            + '<p class="muted">'
            + tr(lang, "Nothing landed on main in this range.", "这段区间内 main 上没有新合入。")
            + "</p>"
        )
    items = []
    for e in history["entries"]:
        if e["kind"] == "pr":
            ref = f'<a href="{escape(e["url"])}">#{e["number"]}</a>'
        else:
            ref = (
                f'<a href="{escape(e["commit_url"])}"><code>{short(e["sha"])}</code></a> '
                f'<span class="badge new-arm">{tr(lang, "direct commit", "直接提交")}</span>'
            )
        items.append(
            f'<li>{ref} {escape(e["title"])} <span class="muted">{escape(e["author"])} · {escape(e["date"])}</span></li>'
        )
    prs = sum(e["kind"] == "pr" for e in history["entries"])
    direct = len(history["entries"]) - prs
    count = (
        '<p class="muted">'
        + tr(lang, f"{prs} PRs, {direct} direct commits.", f"{prs} 个 PR，{direct} 个直接提交。")
        + "</p>"
    )
    return line + f'<ul class="prs">{"".join(items)}</ul>{count}'


def points_table(results: dict, lang: str) -> str:
    rows = "".join(
        f"<tr><td>{escape(p['arm'])}</td><td>c{p['conc']}</td><td><code>{escape(p['exp_name'])}</code></td>"
        f"<td class='{'' if p['status'] == 'ok' else 'bad'}'>{escape(p['status'])}</td>"
        f"<td>{num(p.get(Y))}</td><td>{num(p.get(X))}</td>"
        f"<td><code>{p['serve_spec_sha256'][:12]}</code></td>"
        f"<td class='muted'>{escape(p.get('error') or '')}</td></tr>"
        for p in sorted(results["points"], key=lambda p: (p["arm"], p["conc"]))
    )
    return (
        f"<table><thead><tr><th>{tr(lang, 'arm', '实验组')}</th><th>{tr(lang, 'conc', '并发')}</th>"
        f"<th>{tr(lang, 'point', '数据点')}</th><th>{tr(lang, 'status', '状态')}</th><th>tok/s/GPU</th>"
        "<th>P90 tok/s/user</th><th>serve spec</th><th></th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


STYLE = """
:root { color-scheme: light; --fg:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#f8fafc; }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif; color:var(--fg); background:var(--bg); }
html[data-lang="en"] .page:not([data-page="en"]), html[data-lang="zh"] .page:not([data-page="zh"]) { display:none; }
header { background:#0f172a; color:#f8fafc; padding:24px 32px; }
header .top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
header h1 { margin:0 0 4px; font-size:22px; font-weight:650; }
header .meta { color:#cbd5e1; font-size:13px; }
header a { color:#93c5fd; }
.lang-switch { display:flex; flex:none; border:1px solid #475569; border-radius:6px; overflow:hidden; }
.lang-switch button { background:transparent; color:#cbd5e1; border:0; padding:4px 12px; font:inherit; font-size:13px; cursor:pointer; }
.lang-switch button[aria-pressed="true"] { background:#f8fafc; color:#0f172a; font-weight:600; }
main { max-width:1320px; margin:0 auto; padding:24px 32px 48px; }
h2 { font-size:17px; margin:28px 0 12px; }
h3 { font-size:15px; margin:18px 0 8px; }
.card { background:white; border:1px solid var(--line); border-radius:10px; padding:12px 16px; }
.chart { width:100%; height:auto; }
.chart-toggle { display:flex; align-items:center; gap:6px; width:fit-content; cursor:pointer; margin:2px 0 10px; }
.chart-toggle input { accent-color:#2563eb; }
html[data-show-non-pareto="false"] .non-pareto-point { display:none; }
.chart-title { font-size:14px; font-weight:600; fill:var(--fg); }
.grid { stroke:#eef0f3; } .tick { font-size:11px; fill:var(--muted); }
.axis { font-size:12px; fill:#374151; } .legend { font-size:12px; fill:#374151; }
.point-label { font-size:11px; fill:#374151; }
table { border-collapse:collapse; width:100%; background:white; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
table.compact { width:auto; min-width:640px; } table.compact th, table.compact td { padding:6px 16px; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; font-variant-numeric:tabular-nums; }
th { background:#f1f5f9; font-weight:600; font-size:13px; }
.muted { color:var(--muted); } .good { color:#15803d; font-weight:600; } .bad { color:#b91c1c; font-weight:600; }
.badge { display:inline-block; font-size:12px; padding:1px 8px; border-radius:999px; vertical-align:middle; }
.badge.comparable { background:#dcfce7; color:#166534; } .badge.new-segment { background:#fef3c7; color:#92400e; }
.badge.new-arm { background:#e0e7ff; color:#3730a3; }
.badge.no-baseline { background:#f1f5f9; color:#475569; }
.warn { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:8px 12px; }
.warn li, .notes li { margin:2px 0; }
code { font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }
dl { display:grid; grid-template-columns:fit-content(32%) 1fr; gap:4px 16px; margin:0; }
dt { color:var(--muted); overflow-wrap:anywhere; } dd { margin:0; min-width:0; overflow-wrap:anywhere; }
"""

SCRIPT = """
(function () {
  var root = document.documentElement, titles = %s;
  function set(lang, save) {
    root.dataset.lang = lang;
    root.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = titles[lang];
    if (save) { try { localStorage.setItem("agentx-lang", lang); } catch (e) {} }
  }
  var asked = new URLSearchParams(location.search).get("lang"), saved = null;
  try { saved = localStorage.getItem("agentx-lang"); } catch (e) {}
  set(titles[asked] ? asked : titles[saved] ? saved : "en", false);
  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-set-lang]");
    if (button) { set(button.dataset.setLang, true); }
  });
  document.addEventListener("change", function (event) {
    if (!event.target.matches("[data-show-non-pareto]")) { return; }
    var show = event.target.checked;
    root.dataset.showNonPareto = String(show);
    document.querySelectorAll("[data-show-non-pareto]").forEach(function (input) {
      input.checked = show;
    });
  });
})();
"""


def render(
    results: dict,
    baseline: dict | None,
    comparison: dict,
    history: dict | None,
    repo_url: str,
    notice: str | None = None,
) -> str:
    generated = utc_now()
    titles = {
        lang: tr(lang, "AgentX weekly", "AgentX 周报")
        + " · "
        + results.get("model", {}).get("display_name", "AgentX")
        + " · "
        + results.get("target", {}).get("id", "unknown")
        for lang in LANGS
    }
    pages = "\n".join(
        page(results, baseline, comparison, history, repo_url, lang, generated, notice) for lang in LANGS
    )
    titles_json = json.dumps(titles, ensure_ascii=False).replace("<", r"\u003c")
    return f"""<!doctype html>
<html lang="en" data-lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(titles["en"])}</title><style>{STYLE}</style>
<script>{SCRIPT % titles_json}</script></head>
<body>
{pages}
</body></html>
"""


def page(
    results: dict,
    baseline: dict | None,
    comparison: dict,
    history: dict | None,
    repo_url: str,
    lang: str,
    generated: str,
    notice: str | None = None,
) -> str:
    run, inferencex = results["run"], results["run"]["inferencex"]
    aklr_link = f'<a href="{repo_url}/commit/{run["aklr_sha"]}">{short(run["aklr_sha"])}</a>'
    inferencex_link = f'<a href="https://github.com/{inferencex["repo"]}/commit/{inferencex["sha"]}">{short(inferencex["sha"])}</a>'
    warn = f'<div class="warn">{escape(notice)}</div>' if notice else ""
    arms = "".join(
        f"<dt>{escape(name)}</dt><dd>c{', c'.join(map(str, arm['concs']))} · "
        f"<code>{arm['arm_serve_sha256'][:16]}</code></dd>"
        for name, arm in results["serving"]["arms"].items()
    )
    sources = "".join(
        f"<dt>{escape(path.removeprefix('inferencex-e2e/')).replace('/', '/<wbr>')}</dt><dd><code>{blob[:12]}</code></dd>"
        for path, blob in inferencex.get("sources", {}).items()
    )
    baseline_line = (
        tr(
            lang,
            f"previous run {escape(baseline['run']['id'])} ({escape(baseline['run']['finished_at'][:10])})",
            f"上次运行 {escape(baseline['run']['id'])}（{escape(baseline['run']['finished_at'][:10])}）",
        )
        if baseline
        else tr(lang, "no previous run", "没有上次运行")
    )
    ok = sum(p["status"] == "ok" for p in results["points"])
    total = run["points_expected"]
    progress = tr(lang, f"{ok}/{total} points ok", f"{ok}/{total} 个点成功")
    control_table = ""
    if "control" in results:
        control = results["control"]
        off_ok = sum(p["status"] == "ok" for p in control["points"])
        off_total = control["run"]["points_expected"]
        progress = tr(
            lang,
            f"AKLR on {ok}/{total} · off {off_ok}/{off_total}",
            f"开启 AKLR {ok}/{total} · 关闭 AKLR {off_ok}/{off_total}",
        )
        control_table = f"<h3>{tr(lang, 'AKLR disabled points', '关闭 AKLR 数据点')}</h3>" + points_table(
            control, lang
        )
    switch = "".join(
        f'<button type="button" data-set-lang="{code}" aria-pressed="{str(code == lang).lower()}">{label}</button>'
        for code, label in (("en", "EN"), ("zh", "中文"))
    )
    return f"""<div class="page" data-page="{lang}" lang="{tr(lang, "en", "zh-CN")}">
<header>
<div class="top"><h1>{tr(lang, "AgentX weekly", "AgentX 周报")} · {
        escape(results.get("model", {}).get("display_name", "AgentX"))
    } · {escape(results.get("target", {}).get("id", "unknown"))}</h1>
<div class="lang-switch" role="group" aria-label="Language / 语言">{switch}</div></div>
<div class="meta">{tr(lang, "run", "运行")} {escape(run["id"])} · {escape(run["finished_at"])} · {
        tr(lang, "host", "主机")
    } {escape(run["host"])} ·
AKLR {aklr_link} · InferenceX {inferencex_link} · {progress} · {baseline_line}</div>
</header>
<main>
{warn}
{selection_section(results, lang)}
<h2>{tr(lang, "Pareto frontiers", "Pareto 前沿")}</h2>
{charts(results, baseline, comparison, lang)}
{aklr_section(results, lang)}
<h2>{tr(lang, "Week over week", "周环比")}</h2>
<p class="muted">{
        tr(
            lang,
            "An arm is compared only when its serving and effective measurement settings equal the previous run's; "
            f"otherwise it starts a new segment. Deltas are colored beyond ±{NOTABLE * 100:.0f}% (no noise model yet).",
            "只有当实验组的 serving 配置及实际测量设置与上次相同时才做比较，否则开启新分段。"
            f"变化超过 ±{NOTABLE * 100:.0f}% 才着色（暂无噪声模型）。",
        )
    }</p>
{comparison_tables(comparison, lang)}
<h2>{tr(lang, "AKLR changes since the previous run", "上次运行以来的 AKLR 改动")}</h2>
{history_section(history, run["aklr_sha"], repo_url, lang)}
<h2>{tr(lang, "Points", "数据点")}</h2>
{points_table(results, lang)}
{control_table}
<h2>{tr(lang, "Provenance", "来源信息")}</h2>
<div class="card"><dl>
<dt>{tr(lang, "model id", "模型标识")}</dt><dd>{escape(results.get("model", {}).get("id", "unknown"))}</dd>
<dt>{tr(lang, "target id", "测试目标标识")}</dt><dd>{
        escape(results.get("target", {}).get("id", "unknown"))
    }</dd>
<dt>{tr(lang, "weights", "权重")}</dt><dd>{escape(run.get("weights", {}).get("repo", "unknown"))} @ {
        escape(run.get("weights", {}).get("revision", "unknown"))
    }</dd>
<dt>{tr(lang, "planned image", "计划镜像")}</dt><dd><code>{escape(run["image"]["run"])}</code></dd>
<dt>{tr(lang, "verified image ID", "已验证镜像 ID")}</dt><dd><code>{
        escape(run.get("execution", {}).get("image_id", "unverified"))
    }</code></dd>
<dt>{tr(lang, "observed GPUs", "实际 GPU")}</dt><dd>{
        escape(json.dumps(run.get("hardware", {}), ensure_ascii=False))
    }</dd>
<dt>{tr(lang, "InferenceX image", "InferenceX 镜像")}</dt><dd><code>{
        escape(run["image"]["inferencex"])
    }</code></dd>
<dt>InferenceX</dt><dd>{escape(inferencex["repo"])} {inferencex_link} · {
        escape(inferencex.get("commit_subject", ""))
    }</dd>
<dt>{tr(lang, "config key", "配置键")}</dt><dd><code>{escape(inferencex["config_key"])}</code></dd>
{sources}{arms}
<dt>{tr(lang, "generated", "生成时间")}</dt><dd>{generated}</dd>
</dl></div>
</main>
</div>
"""


def selection_section(results: dict, lang: str) -> str:
    selection = results["run"].get("selection")
    if results["run"].get("point_policy", "all") != "inferencex-pareto" or not selection:
        return ""
    selected = len(selection["points"])
    successful = sum(p["status"] == "ok" for p in results["points"])
    scope = selection["scope"]
    text = tr(
        lang,
        f"InferenceX Pareto selection: {selected} of {selection['candidate_count']} local candidates selected; "
        f"{successful}/{selected} succeeded. This validates the selected plan, not the full concurrency grid.",
        f"InferenceX 前沿选点：从 {selection['candidate_count']} 个本地候选点中选中 {selected} 个，"
        f"成功 {successful}/{selected}。本次验证范围为选点计划，不代表完整并发矩阵已验证。",
    )
    source = (
        tr(lang, "Upstream curve", "上游曲线")
        + ": "
        + " / ".join(str(scope[key]) for key in ("model", "hardware", "framework", "precision"))
        + f" · {selection['curve_date']} · run {selection['curve_workflow_run_id']}"
    )
    reasons = {
        "off-upstream-frontier": tr(lang, "Off upstream frontier", "未进入上游前沿"),
        "no-matching-upstream-measurement": tr(lang, "No matching upstream measurement", "无匹配的上游测量"),
    }
    rows = "".join(
        f"<tr><td><code>{escape(p['point_id'])}</code></td><td>"
        + tr(lang, "Not run", "未运行")
        + f"</td><td>{escape(reasons.get(p['reason'], p['reason']))}</td></tr>"
        for p in selection["unselected"]
    )
    return (
        f'<div class="card"><p>{escape(text)}</p><p>{escape(source)}</p>'
        "<p>P90 E2E normalized interactivity × throughput/GPU · "
        f"<code>{escape(selection['selection_sha256'][:12])}</code></p>"
        + "<p>"
        + tr(
            lang,
            "The upstream frontier selects candidates; the chart below uses only this run's measurements. "
            "Different hardware or runtime may have a different full frontier.",
            "上游前沿仅用于选点；下图根据本次实测重新计算。硬件或运行环境不同，完整前沿可能不同。",
        )
        + "</p><details><summary>"
        + tr(lang, "Unselected points", "未选中的点")
        + f"</summary><table><tbody>{rows}</tbody></table></details></div>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m agentx_bench report", description=__doc__)
    parser.add_argument(
        "--plan", type=Path, help="Plan manifest; enables plan validation and baseline management"
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="results.json, or results root with --plan"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--baseline", type=Path, help="Previous results file, without --plan")
    source.add_argument("--baseline-dir", type=Path, help="Previous site directory, with --plan")
    source.add_argument("--baseline-url", help="Previous site URL, with --plan")
    parser.add_argument("--out", type=Path, required=True, help="site directory")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--git-dir",
        type=Path,
        default=REPO_ROOT,
        help="AKLR checkout whose first-parent history lists the changes",
    )
    args = parser.parse_args(argv)
    if args.plan is not None and args.baseline is not None:
        parser.error(
            "--baseline requires single-result mode; use --baseline-dir or --baseline-url with --plan"
        )
    if args.plan is None and (args.baseline_dir is not None or args.baseline_url is not None):
        parser.error("--baseline-dir and --baseline-url require --plan")
    try:
        if args.plan is not None:
            from agentx_ci.report_all import build

            build(
                args.plan,
                args.results,
                args.out,
                config=args.config,
                git_dir=args.git_dir,
                baseline_dir=args.baseline_dir,
                baseline_url=args.baseline_url,
            )
        else:
            repo_url = load_config(args.config)["report"]["repo_url"].rstrip("/")
            results = read_json(args.results)
            baseline = read_json(args.baseline) if args.baseline and args.baseline.is_file() else None
            write_report(results, baseline, args.out, repo_url, args.git_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    # CI checks target failures separately.
    return 0


def write_report(
    results: dict,
    baseline: dict | None,
    out: Path,
    repo_url: str,
    git_dir: Path,
    notice: str | None = None,
) -> dict:
    results = validated_results(results)
    if baseline is not None:
        baseline = validated_results(baseline)
    notices = [notice] if notice else []
    if not complete_measurement(results):
        notices.append(
            "测量未完成或有失败点；不计算周环比。 / Incomplete measurement or failed points; week-over-week comparison disabled."
        )
    if results["run"].get("source_dirty") is not False:
        notices.append(
            "源码包含未提交改动或状态未知；不更新基线。 / Source dirty or unknown; baseline promotion disabled."
        )
    notice = " ".join(notices) or None
    results = {
        **results,
        "serving": {key: value for key, value in results["serving"].items() if key != "warnings"},
    }
    if results["run"].get("mode") != "live":
        raise ValueError("only live measurement results are supported")
    if baseline is not None and baseline["run"].get("mode") != "live":
        baseline = None
    comparison = compare(results, baseline)
    history = None
    if baseline is not None:
        base, head = baseline["run"]["aklr_sha"], results["run"]["aklr_sha"]
        history = landed(base, head, git_dir, repo_url)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(results, baseline, comparison, history, repo_url, notice))
    write_json(out / "results.json", results)
    write_json(
        out / "report.json",
        {
            "model": results.get("model", {}),
            "target": results.get("target", {}),
            "frontiers": frontiers(results),
            "comparison": comparison,
            "history": history,
            "aklr_comparison": compare_aklr(results),
            "disabled_frontiers": frontiers(results["control"]) if "control" in results else None,
        },
    )
    print(f"wrote {out / 'index.html'} ({comparison['status']})")
    return comparison
