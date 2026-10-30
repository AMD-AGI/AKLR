# SPDX-License-Identifier: MIT
"""Render the weekly page: Pareto charts, the week-over-week table and the AKLR commits compared.

Writes ``index.html`` (self-contained, inline SVG), ``results.json`` (this run, next week's
baseline) and ``report.json`` (frontiers, comparison) into ``--out``.
"""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path

from agentx_ci.common import (
    DEFAULT_CONFIG,
    load_config,
    read_json,
    utc_now,
    write_json,
)
from agentx_ci.pareto import X, Y, compare, frontier, frontiers, measured

ARM_COLORS = {"tp8": "#2563eb", "tp4": "#ea580c"}
FALLBACK_COLORS = ("#16a34a", "#9333ea", "#0891b2")
COMBINED_COLOR = "#111827"
BASELINE_COLOR = "#9ca3af"
# No noise model yet: deltas are colored only beyond this band.
NOTABLE = 0.03


def arm_color(arm: str, arms: list[str]) -> str:
    return (
        ARM_COLORS.get(arm) or FALLBACK_COLORS[arms.index(arm) % len(FALLBACK_COLORS)]
    )


def nice_ticks(high: float, count: int = 5) -> list[float]:
    raw = high / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return [i * step for i in range(int(high / step) + 2)]


def svg_chart(
    title: str, series: list[dict], width: int = 620, height: int = 380
) -> str:
    """Scatter + frontier lines. A series: label, color, points, frontier, dashed, hollow, labels."""
    left, right, top, bottom = 64, 16, 40, 52
    xs = [p[X] for s in series for p in s["points"]] or [1.0]
    ys = [p[Y] for s in series for p in s["points"]] or [1.0]
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
        out.append(
            f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" y2="{top + plot_h}" class="grid"/>'
        )
        out.append(
            f'<text x="{sx(t):.1f}" y="{top + plot_h + 16}" class="tick" text-anchor="middle">{t:g}</text>'
        )
    for t in y_ticks:
        out.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{left - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>'
        )
    out.append(
        f'<text x="{left + plot_w / 2}" y="{height - 12}" class="axis" text-anchor="middle">'
        "P90 E2E interactivity (tok/s/user) →</text>"
    )
    out.append(
        f'<text transform="translate(16 {top + plot_h / 2}) rotate(-90)" class="axis" '
        'text-anchor="middle">throughput per GPU (tok/s/GPU) →</text>'
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
            fill = "white" if s.get("hollow") else s["color"]
            tip = f"{s['label']} c{p['conc']}: {p[Y]:.1f} tok/s/GPU, {p[X]:.1f} tok/s/user"
            out.append(
                f'<circle cx="{sx(p[X]):.1f}" cy="{sy(p[Y]):.1f}" r="4.5" fill="{fill}" '
                f'stroke="{s["color"]}" stroke-width="1.6"><title>{escape(tip)}</title></circle>'
            )
            if s.get("labels"):
                out.append(
                    f'<text x="{sx(p[X]) + 7:.1f}" y="{sy(p[Y]) - 6:.1f}" class="point-label">c{p["conc"]}</text>'
                )
    legend_x = left + plot_w - 8
    for i, s in enumerate(series):
        y = top + 14 + i * 18
        dash = ' stroke-dasharray="6 4"' if s.get("dashed") else ""
        out.append(
            f'<line x1="{legend_x - 22}" x2="{legend_x - 4}" y1="{y - 4}" y2="{y - 4}" stroke="{s["color"]}" stroke-width="2"{dash}/>'
        )
        out.append(
            f'<text x="{legend_x - 28}" y="{y}" class="legend" text-anchor="end">{escape(s["label"])}</text>'
        )
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


def charts(results: dict, baseline: dict | None, comparison: dict) -> str:
    arms = sorted(results["serving"]["arms"])
    current = frontiers(results)
    series = [
        {
            "label": f"{arm} (this run)",
            "color": arm_color(arm, arms),
            "points": measured(results, arm),
            "labels": True,
        }
        for arm in arms
    ]
    series.append(
        {
            "label": "combined frontier",
            "color": COMBINED_COLOR,
            "points": [],
            "frontier": current["combined"],
        }
    )
    if comparison["status"] == "ok":
        series.append(
            {
                "label": "previous run frontier",
                "color": BASELINE_COLOR,
                "points": [],
                "frontier": frontier(measured(baseline)),
                "dashed": True,
            }
        )
    blocks = [
        svg_chart("All arms · combined Pareto frontier", series, width=1000, height=380)
    ]
    for arm in arms:
        color = arm_color(arm, arms)
        arm_series = [
            {
                "label": "this run",
                "color": color,
                "points": measured(results, arm),
                "frontier": current["arms"][arm],
                "labels": True,
            }
        ]
        verdict = comparison["arms"].get(arm, {}).get("status")
        if baseline is not None and verdict in ("comparable", "new-segment"):
            old = measured(baseline, arm)
            label = (
                "previous run"
                if verdict == "comparable"
                else "previous run (other config, reference)"
            )
            arm_series.append(
                {
                    "label": label,
                    "color": color if verdict == "comparable" else BASELINE_COLOR,
                    "points": old,
                    "frontier": frontier(old),
                    "dashed": True,
                    "hollow": True,
                }
            )
        blocks.append(svg_chart(f"Arm {arm}", arm_series))
    return "\n".join(f'<div class="card chart-card">{b}</div>' for b in blocks)


def comparison_tables(comparison: dict) -> str:
    if comparison["status"] != "ok":
        return f'<p class="muted">No week-over-week comparison: {escape(comparison["reason"])}.</p>'
    out = []
    if comparison["notes"]:
        items = "".join(f"<li>{escape(n)}</li>" for n in comparison["notes"])
        out.append(f'<ul class="notes">{items}</ul>')
    for arm, view in comparison["arms"].items():
        badge = {
            "comparable": ("comparable", "same serving config as the previous run"),
            "new-segment": ("new segment", view.get("reason", "")),
            "new-arm": ("new arm", view.get("reason", "")),
        }[view["status"]]
        head = f'<h3>{escape(arm)} <span class="badge {view["status"]}">{badge[0]}</span> <span class="muted">{escape(badge[1])}</span></h3>'
        if view["status"] == "new-segment":
            changed = ", ".join(f"c{c}" for c in view["changed_concs"]) or "none"
            out.append(
                head
                + f'<p class="muted">Points whose serving config changed: {changed}. Values are not compared.</p>'
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
                else ("no baseline" if not old or old.get("status") != "ok" else "")
            )
            rows.append(
                f"<tr><td>c{row['conc']}</td>"
                f"<td>{num(cur.get(Y))}</td><td>{num(old.get(Y))}</td><td class='{delta_class(t)}'>{pct(t)}</td>"
                f"<td>{num(cur.get(X))}</td><td>{num(old.get(X))}</td><td class='{delta_class(i)}'>{pct(i)}</td>"
                f"<td class='muted'>{escape(status)}</td></tr>"
            )
        geo = view.get("tput_geomean_ratio")
        out.append(
            head
            + "<table><thead><tr><th>conc</th><th>tok/s/GPU</th><th>previous</th><th>Δ</th>"
            "<th>P90 tok/s/user</th><th>previous</th><th>Δ</th><th></th></tr></thead><tbody>"
            + "".join(rows)
            + f"</tbody></table><p class='muted'>Throughput geomean vs previous: "
            f"<span class='{delta_class(geo)}'>{pct(geo)}</span></p>"
        )
    return "\n".join(out)


def aklr_section(results: dict, baseline: dict | None, repo_url: str) -> str:
    def link(sha: str) -> str:
        return f'<a href="{escape(repo_url)}/commit/{escape(sha)}"><code>{short(sha)}</code></a>'

    this = results["run"]["aklr_sha"]
    if baseline is None:
        return f"<p>This run {link(this)} · first run, no previous run to compare with.</p>"
    base = baseline["run"]["aklr_sha"]
    compare_url = f"{repo_url}/compare/{base}...{this}"
    return (
        f"<p>Previous run {link(base)} → this run {link(this)} · "
        f'<a href="{escape(compare_url)}">compare on GitHub</a></p>'
    )


def points_table(results: dict) -> str:
    rows = "".join(
        f"<tr><td>{escape(p['arm'])}</td><td>c{p['conc']}</td><td><code>{escape(p['exp_name'])}</code></td>"
        f"<td class='{'' if p['status'] == 'ok' else 'bad'}'>{escape(p['status'])}</td>"
        f"<td>{num(p.get(Y))}</td><td>{num(p.get(X))}</td>"
        f"<td><code>{p['serve_spec_sha256'][:12]}</code></td>"
        f"<td class='muted'>{escape(p.get('error') or '')}</td></tr>"
        for p in sorted(results["points"], key=lambda p: (p["arm"], p["conc"]))
    )
    return (
        "<table><thead><tr><th>arm</th><th>conc</th><th>point</th><th>status</th><th>tok/s/GPU</th>"
        "<th>P90 tok/s/user</th><th>serve spec</th><th></th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


STYLE = """
:root { color-scheme: light; --fg:#111827; --muted:#6b7280; --line:#e5e7eb; --bg:#f8fafc; }
* { box-sizing: border-box; }
body { margin:0; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--fg); background:var(--bg); }
header { background:#0f172a; color:#f8fafc; padding:24px 32px; }
header h1 { margin:0 0 4px; font-size:22px; font-weight:650; }
header .meta { color:#cbd5e1; font-size:13px; }
header a { color:#93c5fd; }
main { max-width:1320px; margin:0 auto; padding:24px 32px 48px; }
h2 { font-size:17px; margin:28px 0 12px; }
h3 { font-size:15px; margin:18px 0 8px; }
.grid-charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:16px; }
.chart-card:first-child { grid-column:1 / -1; }
.card { background:white; border:1px solid var(--line); border-radius:10px; padding:12px 16px; }
.chart { width:100%; height:auto; }
.chart-title { font-size:14px; font-weight:600; fill:var(--fg); }
.grid { stroke:#eef0f3; } .tick { font-size:11px; fill:var(--muted); }
.axis { font-size:12px; fill:#374151; } .legend { font-size:12px; fill:#374151; }
.point-label { font-size:11px; fill:#374151; }
table { border-collapse:collapse; width:100%; background:white; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; font-variant-numeric:tabular-nums; }
th { background:#f1f5f9; font-weight:600; font-size:13px; }
.muted { color:var(--muted); } .good { color:#15803d; font-weight:600; } .bad { color:#b91c1c; font-weight:600; }
.badge { display:inline-block; font-size:12px; padding:1px 8px; border-radius:999px; vertical-align:middle; }
.badge.comparable { background:#dcfce7; color:#166534; } .badge.new-segment { background:#fef3c7; color:#92400e; }
.badge.new-arm { background:#e0e7ff; color:#3730a3; } .badge.mock { background:#fde68a; color:#78350f; margin-left:8px; }
.warn { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:8px 12px; }
.warn li, .notes li { margin:2px 0; }
code { font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }
dl { display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; margin:0; }
dt { color:var(--muted); } dd { margin:0; word-break:break-all; }
"""


def render(
    results: dict,
    baseline: dict | None,
    comparison: dict,
    repo_url: str,
) -> str:
    run, ix = results["run"], results["run"]["inferencex"]
    mock = '<span class="badge mock">MOCK DATA</span>' if run["mode"] == "mock" else ""
    aklr_link = (
        f'<a href="{repo_url}/commit/{run["aklr_sha"]}">{short(run["aklr_sha"])}</a>'
    )
    ix_link = f'<a href="https://github.com/{ix["repo"]}/commit/{ix["sha"]}">{short(ix["sha"])}</a>'
    warnings = results["serving"]["warnings"]
    warn = (
        '<div class="warn"><ul>'
        + "".join(f"<li>{escape(w)}</li>" for w in warnings)
        + "</ul></div>"
        if warnings
        else ""
    )
    arms = "".join(
        f"<dt>{escape(name)}</dt><dd>c{', c'.join(map(str, arm['concs']))} · "
        f"<code>{arm['arm_serve_sha256'][:16]}</code></dd>"
        for name, arm in results["serving"]["arms"].items()
    )
    sources = "".join(
        f"<dt>{escape(path.removeprefix('inferencex-e2e/'))}</dt><dd><code>{blob[:12]}</code></dd>"
        for path, blob in ix.get("sources", {}).items()
    )
    baseline_line = (
        f"previous run {escape(baseline['run']['id'])} ({escape(baseline['run']['finished_at'][:10])})"
        if baseline
        else "no previous run"
    )
    ok = sum(p["status"] == "ok" for p in results["points"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentX weekly · GLM-5.2 MXFP4 · MI355X</title><style>{STYLE}</style></head>
<body>
<header>
<h1>AgentX weekly · GLM-5.2 MXFP4 · MI355X · sglang + AKLR{mock}</h1>
<div class="meta">run {escape(run['id'])} · {escape(run['finished_at'])} · host {escape(run['host'])} ·
AKLR {aklr_link} · InferenceX {ix_link} · {ok}/{len(results['points'])} points ok · {baseline_line}</div>
</header>
<main>
{warn}
<h2>Pareto frontiers</h2>
<div class="grid-charts">{charts(results, baseline, comparison)}</div>
<h2>Week over week</h2>
<p class="muted">An arm is compared only when its whole serving config (arm_serve_sha256) equals the previous run's;
otherwise it starts a new segment. Deltas are colored beyond ±{NOTABLE * 100:.0f}% (no noise model yet).</p>
{comparison_tables(comparison)}
<h2>AKLR commits</h2>
{aklr_section(results, baseline, repo_url)}
<h2>Points</h2>
{points_table(results)}
<h2>Provenance</h2>
<div class="card"><dl>
<dt>run image</dt><dd><code>{escape(run['image']['run'])}</code></dd>
<dt>InferenceX image</dt><dd><code>{escape(run['image']['inferencex'])}</code></dd>
<dt>InferenceX</dt><dd>{escape(ix['repo'])} {ix_link} · {escape(ix.get('commit_subject', ''))}</dd>
<dt>config key</dt><dd><code>{escape(ix['config_key'])}</code></dd>
{sources}{arms}
<dt>generated</dt><dd>{utc_now()}</dd>
</dl></div>
</main></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m agentx_ci report", description=__doc__
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, help="previous run's results.json, if any"
    )
    parser.add_argument("--out", type=Path, required=True, help="site directory")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    repo_url = load_config(args.config)["report"]["repo_url"].rstrip("/")
    results = read_json(args.results)
    baseline = (
        read_json(args.baseline) if args.baseline and args.baseline.is_file() else None
    )
    comparison = compare(results, baseline)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.html").write_text(
        render(results, baseline, comparison, repo_url)
    )
    write_json(args.out / "results.json", results)
    write_json(
        args.out / "report.json",
        {"frontiers": frontiers(results), "comparison": comparison},
    )
    print(f"wrote {args.out / 'index.html'} ({comparison['status']})")
    return 0
