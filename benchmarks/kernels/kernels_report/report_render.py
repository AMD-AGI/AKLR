# SPDX-License-Identifier: MIT
"""The kernels section's pages, section.json and data.json."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict
from html import escape
from pathlib import Path

from kernels_report.data_collect import SERVED, STATUSES, Context, ContractReport, PointReport

LANGS = ("en", "zh")
TITLE = {"en": "Kernels", "zh": "Kernel"}
STATUS_LABELS = {
    "ok": ("All served", "全部生效"),
    "partial": ("Partly served", "部分生效"),
    "not-served": ("Not served", "未生效"),
    "not-measured": ("Not measured", "未测量"),
}
REASON_LABELS = {
    SERVED: ("Served", "生效"),
    "not_qualified": ("Below threshold", "未达门槛"),
    "inaccurate": ("Accuracy failed", "精度失败"),
    "not_measured": ("Not measured", "未测量"),
}
REASON_CLASS = {
    SERVED: "ok",
    "not_qualified": "partial",
    "inaccurate": "not-served",
    "not_measured": "not-measured",
}

# The dark console theme of the AgentX pages: palette, header and language switch copied from
# benchmarks/agentx/agentx_ci/report.py (STYLE, SCRIPT), which needs yaml this CPU job does not
# install; keep the two in sync.
STYLE = """
:root { color-scheme:dark; --fg:#e6edf7; --muted:#8a97ad; --line:#1e2a40; --edge:#2b3d5c; --bg:#070b14; --panel:#0e1525; --panel2:#121b2e; --cyan:#22d3ee; --ok:#34d399; --warn:#fbbf24; --bad:#f87171; }
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; font:14px/1.55 "Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif; color:var(--fg); background:radial-gradient(1200px 500px at 15% -10%,#123a5a55,transparent 60%) no-repeat,radial-gradient(900px 400px at 100% 0,#3b1d6a44,transparent 60%) no-repeat,var(--bg); }
html[data-lang="en"] .page:not([data-page="en"]), html[data-lang="zh"] .page:not([data-page="zh"]) { display:none; }
a { color:var(--cyan); text-decoration:none; } a:hover { text-decoration:underline; }
header { max-width:1320px; margin:0 auto; padding:28px 32px 18px; border-bottom:1px solid var(--line); }
header .top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
header h1 { margin:0 0 6px; font-size:26px; font-weight:650; letter-spacing:-.02em; background:linear-gradient(90deg,#fff,#9fb4d6); -webkit-background-clip:text; background-clip:text; color:transparent; overflow-wrap:anywhere; }
header .meta { color:var(--muted); font-size:13px; }
header h1 code { font-size:inherit; color:inherit; }
.lang-switch { display:flex; flex:none; border:1px solid var(--edge); border-radius:999px; padding:3px; }
.lang-switch button { background:none; color:var(--muted); border:0; padding:3px 12px; border-radius:999px; font:inherit; font-size:12px; cursor:pointer; }
.lang-switch button[aria-pressed="true"] { background:var(--cyan); color:#04111a; font-weight:700; }
main { max-width:1320px; margin:0 auto; padding:24px 32px 48px; }
main > p { color:#c4d0e3; }
h2 { font-size:17px; margin:28px 0 12px; }
.card { background:linear-gradient(180deg,var(--panel2),var(--panel)); border:1px solid var(--line); border-radius:14px; padding:12px 16px; box-shadow:0 8px 24px #0006; }
.chart { width:100%; height:auto; }
.grid { stroke:var(--line); } .tick { font-size:11px; fill:var(--muted); } .axis { font-size:12px; fill:#c4d0e3; }
.ref { stroke:#7c8aa5; stroke-dasharray:4 4; } .ref.threshold { stroke:var(--ok); stroke-dasharray:2 4; opacity:.7; }
.chart [data-point-tooltip] { cursor:help; }
.chart [data-point-tooltip]:focus { outline:2px solid var(--cyan); outline-offset:3px; }
.point-tooltip { position:fixed; z-index:10; max-width:min(420px,calc(100vw - 24px)); padding:10px 12px; border-radius:8px; background:var(--panel2); color:var(--fg); border:1px solid var(--edge); box-shadow:0 8px 24px #000a; font-size:12px; white-space:pre-line; overflow-wrap:anywhere; pointer-events:none; }
table { border-collapse:collapse; width:100%; }
th, td { padding:8px 12px; border-bottom:1px solid var(--line); text-align:left; font-variant-numeric:tabular-nums; vertical-align:top; }
tr:last-child td { border-bottom:0; } tbody tr:hover td { background:#ffffff05; }
th { background:#0a101c; color:var(--muted); font-weight:600; font-size:12px; letter-spacing:.02em; }
td.num, th.num { text-align:right; white-space:nowrap; }
.muted { color:var(--muted); } .good { color:var(--ok); font-weight:600; }
.st { display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:600; white-space:nowrap; color:var(--muted); }
.st i { width:8px; height:8px; border-radius:50%; background:#3a4760; }
.st.ok { color:var(--ok); } .st.ok i { background:var(--ok); box-shadow:0 0 8px var(--ok); }
.st.partial { color:var(--warn); } .st.partial i { background:var(--warn); box-shadow:0 0 8px var(--warn); }
.st.not-served { color:var(--bad); } .st.not-served i { background:var(--bad); box-shadow:0 0 8px var(--bad); }
td.baseline { font-size:12px; color:var(--muted); min-width:200px; max-width:300px; }
td.baseline div { display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
td.source code { white-space:nowrap; }
td.contract { min-width:200px; max-width:280px; } td.when { white-space:nowrap; }
code { font:12px "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace; color:#c4d0e3; overflow-wrap:anywhere; }
dl { display:grid; grid-template-columns:fit-content(32%) 1fr; gap:4px 16px; margin:0; }
dt { color:var(--muted); overflow-wrap:anywhere; } dd { margin:0; min-width:0; overflow-wrap:anywhere; }
@media (max-width:760px) { header, main { padding-left:16px; padding-right:16px; } header h1 { font-size:21px; } }
"""

SCRIPT = """
(function () {
  var root = document.documentElement, titles = %s;
  function valid(lang) { return lang === "en" || lang === "zh"; }
  function set(lang, save) {
    hidePoint();
    root.dataset.lang = lang;
    root.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = titles[lang];
    document.querySelectorAll("[data-set-lang]").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.setLang === lang));
    });
    if (save) {
      try { localStorage.setItem("agentx-lang", lang); } catch (e) {}
      try {
        var url = new URL(location.href);
        url.searchParams.set("lang", lang);
        history.replaceState(null, "", url);
      } catch (e) {}
    }
  }
  var asked = new URLSearchParams(location.search).get("lang"), saved = null;
  try { saved = localStorage.getItem("agentx-lang"); } catch (e) {}
  set(valid(asked) ? asked : valid(saved) ? saved : "en", false);
  document.addEventListener("DOMContentLoaded", function () { set(root.dataset.lang, false); });
  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-set-lang]");
    if (button && valid(button.dataset.setLang)) { set(button.dataset.setLang, true); }
  });
  var activePoint = null;
  function hidePoint() {
    var tooltip = document.getElementById("point-tooltip");
    if (tooltip) { tooltip.hidden = true; }
    activePoint = null;
  }
  function showPoint(point, x, y) {
    var tooltip = document.getElementById("point-tooltip");
    activePoint = point;
    tooltip.textContent = point.dataset.pointTooltip;
    tooltip.hidden = false;
    var box = tooltip.getBoundingClientRect();
    tooltip.style.left = Math.max(12, Math.min(x + 14, innerWidth - box.width - 12)) + "px";
    tooltip.style.top = Math.max(12, y + box.height + 26 > innerHeight ? y - box.height - 14 : y + 14) + "px";
  }
  document.addEventListener("pointerover", function (event) {
    var point = event.target.closest("[data-point-tooltip]");
    if (point) { showPoint(point, event.clientX, event.clientY); }
  });
  document.addEventListener("pointerout", function (event) {
    if (event.target === activePoint) { hidePoint(); }
  });
  document.addEventListener("focusin", function (event) {
    var point = event.target.closest("[data-point-tooltip]");
    if (point) { var box = point.getBoundingClientRect(); showPoint(point, box.right, box.bottom); }
  });
  document.addEventListener("focusout", function (event) {
    if (event.target === activePoint) { hidePoint(); }
  });
  window.addEventListener("scroll", hidePoint, true);
  window.addEventListener("resize", hidePoint);
})();
"""


def tr(lang: str, en: str, zh: str) -> str:
    return zh if lang == "zh" else en


def when(value: str) -> str:
    match = re.match(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d)", value)
    return f"{match[1]} {match[2]} UTC" if match else value


def speedup_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}×"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def page_name(contract_id: str) -> str:
    return contract_id.replace("@", "-") + ".html"


def status_badge(lang: str, css: str, label: tuple[str, str]) -> str:
    return f'<span class="st {css}"><i></i>{escape(tr(lang, *label))}</span>'


def language_switch(lang: str) -> str:
    buttons = "".join(
        f'<button type="button" data-set-lang="{code}" aria-pressed="{str(code == lang).lower()}">{label}</button>'
        for code, label in (("en", "EN"), ("zh", "中文"))
    )
    return (
        f'<div class="lang-switch" role="group" aria-label="{tr(lang, "Language", "语言")}">{buttons}</div>'
    )


def document(titles: dict[str, str], pages: str) -> str:
    titles_json = json.dumps(titles, ensure_ascii=False).replace("<", r"\u003c")
    return f"""<!doctype html>
<html lang="en" data-lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(titles["en"])}</title><style>{STYLE}</style>
<script>{SCRIPT % titles_json}</script></head>
<body>
{pages}
<div id="point-tooltip" class="point-tooltip" role="tooltip" hidden></div>
</body></html>
"""


def page(lang: str, title: str, meta: str, body: str) -> str:
    return f"""<div class="page" data-page="{lang}" lang="{tr(lang, "en", "zh-CN")}">
<header><div class="top"><h1>{title}</h1>{language_switch(lang)}</div>
<div class="meta">{meta}</div></header>
<main>{body}</main></div>"""


def commit_link(context: Context, sha: str) -> str:
    return f'<a href="{escape(context.repo_url)}/commit/{escape(sha)}"><code>{escape(sha[:8])}</code></a>'


def run_link(context: Context, run_id: str) -> str:
    return f'<a href="{escape(context.repo_url)}/actions/runs/{escape(run_id)}">run {escape(run_id)}</a>'


def measured_cell(lang: str, report: ContractReport, context: Context) -> str:
    m = report.measured
    if m is None:
        return "—"
    return f'<span title="{escape(when(m.measured_at))}">{escape(m.measured_at[:10])}</span><br>{run_link(context, m.run_id)}'


def threshold_text(lang: str, report: ContractReport) -> str:
    if report.baseline is None:
        return tr(lang, "none (no baseline)", "无（没有 baseline）")
    return f"≥ {report.min_speedup:g}×"


def dims_text(dims: dict[str, str]) -> str:
    return ", ".join(f"{name}={value}" for name, value in dims.items()) or "—"


def index_page(reports: list[ContractReport], context: Context, lang: str) -> str:
    rows = []
    for report in sorted(reports, key=lambda r: (r.group, r.contract_id)):
        href = f"contracts/{page_name(report.contract_id)}?lang={lang}"
        speedups = report.speedups
        spread = f"{min(speedups):.2f}–{max(speedups):.2f}×" if speedups else "—"
        source = commit_link(context, report.measured.commit) if report.measured else "—"
        baseline = (
            escape(report.baseline) if report.baseline else tr(lang, "none: no threshold", "无：不设门槛")
        )
        rows.append(
            f'<tr><td class="contract"><a href="{escape(href)}"><code>{escape(report.contract_id)}</code></a><br>'
            f'<code class="muted">{escape(report.group)}</code></td>'
            f"<td>{status_badge(lang, report.status, STATUS_LABELS[report.status])}</td>"
            f'<td class="num">{report.served}/{len(report.points)}</td>'
            f'<td class="num{" good" if report.geomean and report.geomean >= 1 else ""}">'
            f"{speedup_text(report.geomean)}</td>"
            f'<td class="num muted">{spread}</td><td class="num">{report.kernels}</td>'
            f'<td class="when">{measured_cell(lang, report, context)}</td><td class="source">{source}</td>'
            f'<td class="baseline"><div title="{baseline}">{baseline}</div></td></tr>'
        )
    headers = (
        (tr(lang, "Contract", "Contract"), ""),
        (tr(lang, "Status", "状态"), ""),
        (tr(lang, "Served", "生效点"), "num"),
        (tr(lang, "Geomean", "几何平均"), "num"),
        (tr(lang, "Range", "范围"), "num"),
        (tr(lang, "Kernels", "Kernel 数"), "num"),
        (tr(lang, "Measured", "测量时间"), ""),
        (tr(lang, "Source (main)", "来源（main）"), ""),
        (tr(lang, "Baseline", "Baseline"), ""),
    )
    head = "".join(f'<th class="{css}">{label}</th>' for label, css in headers)
    newest = max((r.measured for r in reports if r.measured), key=lambda m: m.measured_at, default=None)
    latest = (
        f" · {tr(lang, 'latest measurement', '最近测量')} {run_link(context, newest.run_id)} "
        f"({escape(when(newest.measured_at))}, {escape(newest.hardware)})"
        if newest
        else ""
    )
    meta = (
        f'<a href="../index.html?lang={lang}">← {tr(lang, "AKLR Weekly Report", "AKLR 每周报告")}</a> · '
        f"main {commit_link(context, context.commit)}{latest} · "
        f"{tr(lang, 'generated', '生成于')} {escape(when(context.generated_at))}<br>"
        + tr(
            lang,
            "Served assumes the installed stack meets each kernel's <code>kernel.requires</code>.",
            "“生效”以所装软件栈满足各 kernel 的 <code>kernel.requires</code> 为前提。",
        )
    )
    body = (
        f'<div class="card" style="overflow-x:auto"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    title = TITLE[lang]
    return page(lang, escape(title), meta, body)


def chart(lang: str, report: ContractReport) -> str:
    served = sorted(
        (p for p in report.points if p.reason == SERVED and p.speedup is not None), key=lambda p: p.speedup
    )
    if not served:
        if report.baseline is None:
            return f'<p class="muted">{tr(lang, "No baseline, so no speedup to chart.", "没有 baseline，没有可画的加速比。")}</p>'
        return (
            f'<p class="muted">{tr(lang, "No served point has a speedup.", "没有带 speedup 的生效点。")}</p>'
        )
    width, height, left, right, top, bottom = 1200, 300, 56, 16, 16, 34
    low = min(0.9, served[0].speedup * 0.9)
    high = max(1.5, served[-1].speedup * 1.1)

    def y(value: float) -> float:
        return top + (math.log(high) - math.log(value)) / (math.log(high) - math.log(low)) * (
            height - top - bottom
        )

    def x(index: int) -> float:
        return left + (index + 0.5) / len(served) * (width - left - right)

    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 10, 15, 25) if low <= t <= high]
    grid = "".join(
        f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
        f'<text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:g}×</text>'
        for t in ticks
    )
    refs = f'<line class="ref" x1="{left}" x2="{width - right}" y1="{y(1):.1f}" y2="{y(1):.1f}"/>'
    if report.baseline is not None and low <= report.min_speedup <= high:
        level = y(report.min_speedup)
        refs += f'<line class="ref threshold" x1="{left}" x2="{width - right}" y1="{level:.1f}" y2="{level:.1f}"/>'
    radius = 3.2 if len(served) < 200 else 2.2
    dots = "".join(
        f'<circle cx="{x(i):.1f}" cy="{y(p.speedup):.1f}" r="{radius}" '
        f'fill="{"#34d399" if p.speedup >= 1 else "#fbbf24"}" tabindex="0" '
        f'data-point-tooltip="{escape(dims_text(p.dims))}&#10;{p.speedup:.3f}× · {escape(p.kernel or "")}"/>'
        for i, p in enumerate(served)
    )
    caption = tr(
        lang,
        f"{len(served)} served points by speedup, log scale; dashed lines at 1× and the threshold",
        f"{len(served)} 个生效点，按 speedup 排序，对数刻度；虚线为 1× 与生效门槛",
    )
    axis = f'<text class="axis" x="{(left + width - right) / 2}" y="{height - 8}" text-anchor="middle">{caption}</text>'
    return f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">{grid}{refs}{dots}{axis}</svg>'


def requires_text(requires: dict[str, str] | None) -> str:
    return ", ".join(f"{name}{specifier}" for name, specifier in (requires or {}).items())


def point_order(p: PointReport) -> tuple:
    return (p.reason != SERVED, -(p.speedup or 0), list(REASON_LABELS).index(p.reason), dims_text(p.dims))


def contract_page(report: ContractReport, context: Context, lang: str) -> str:
    rows = []
    for p in sorted(report.points, key=point_order):
        served = p.reason == SERVED
        kernel = "—"
        if served:
            link = f"{context.repo_url}/blob/{context.commit}/{p.path}"
            requires = requires_text(p.requires)
            kernel = (
                f'<a href="{escape(link)}"><code>{escape(p.kernel or "")}</code></a><br>'
                f'<span class="muted">{escape(p.language or "")} · {escape(Path(p.path or "").stem)}</span>'
                + (
                    f'<br><span class="muted">{tr(lang, "requires", "依赖")} <code>{escape(requires)}</code></span>'
                    if requires
                    else ""
                )
            )
        rows.append(
            f"<tr><td><code>{escape(dims_text(p.dims))}</code></td>"
            f"<td>{status_badge(lang, REASON_CLASS[p.reason], REASON_LABELS[p.reason])}</td>"
            f"<td>{kernel}</td>"
            f'<td class="num{" good" if served and p.speedup and p.speedup >= 1 else ""}">'
            f"{speedup_text(p.speedup if served else None)}</td>"
            f'<td class="num">{number(p.microseconds if served else None)}</td>'
            f'<td class="num">{p.candidates}</td></tr>'
        )
    headers = (
        (tr(lang, "Point", "点"), ""),
        (tr(lang, "Status", "状态"), ""),
        (tr(lang, "Serving kernel", "胜出 kernel"), ""),
        (tr(lang, "Speedup", "加速比"), "num"),
        (tr(lang, "Kernel µs", "Kernel µs"), "num"),
        (tr(lang, "Candidates", "候选数"), "num"),
    )
    head = "".join(f'<th class="{css}">{label}</th>' for label, css in headers)
    speedups = report.speedups
    spread = f" ({min(speedups):.2f}–{max(speedups):.2f}×)" if speedups else ""
    m = report.measured
    measured = (
        f"{escape(when(m.measured_at))} · {run_link(context, m.run_id)} · "
        f"{tr(lang, 'measured main', '测量的 main')} {commit_link(context, m.commit)}"
        if m
        else "—"
    )
    details = (
        ("Baseline", "Baseline", escape(report.baseline) if report.baseline else tr(lang, "none", "无")),
        ("Serving threshold", "生效门槛", threshold_text(lang, report)),
        (
            "Assumes",
            "前提",
            tr(
                lang,
                "the installed stack meets each serving kernel's <code>kernel.requires</code>; dispatch skips a kernel whose requirements are unmet",
                "所装软件栈满足各胜出 kernel 的 <code>kernel.requires</code>；不满足时 dispatch 会跳过该 kernel",
            ),
        ),
        ("Package", "所在包", f"<code>{escape(report.op_package)}</code>"),
        ("Fixed for all points", "所有点相同", f"<code>{escape(dims_text(report.fixed))}</code>"),
        ("Kernels", "Kernel 数", str(report.kernels)),
        (
            "Served points",
            "生效点",
            f"{report.served}/{len(report.points)} · {status_badge(lang, report.status, STATUS_LABELS[report.status])}",
        ),
        ("Geomean speedup", "几何平均加速比", f"{speedup_text(report.geomean)}{spread}"),
        ("Last measured", "最近测量", measured),
    )
    dl = "".join(f"<dt>{tr(lang, en, zh)}</dt><dd>{value}</dd>" for en, zh, value in details)
    body = (
        f'<div class="card"><dl>{dl}</dl></div>'
        f"<h2>{tr(lang, 'Speedup of each served point', '各生效点的加速比')}</h2>"
        f'<div class="card">{chart(lang, report)}</div>'
        f"<h2>{tr(lang, 'Points', '逐点明细')}</h2>"
        f'<div class="card" style="overflow-x:auto"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    meta = (
        f'<a href="../index.html?lang={lang}">← {TITLE[lang]}</a> · '
        f"{escape(report.op_type)} · main {commit_link(context, context.commit)} · "
        f"{tr(lang, 'generated', '生成于')} {escape(when(context.generated_at))}"
    )
    return page(lang, f"<code>{escape(report.contract_id)}</code>", meta, body)


def section_summary(reports: list[ContractReport], context: Context) -> dict:
    counts = {s: sum(r.status == s for r in reports) for s in STATUSES}
    return {
        "schema": 1,
        "id": "kernels",
        "title": TITLE,
        "path": "index.html",
        "updated_at": context.generated_at,
        "last_run": {
            "run_id": context.run_id,
            "run_url": context.run_url,
            "commit": context.commit,
            "ref": context.ref,
            "event": context.event,
        },
        "status_counts": {s: n for s, n in counts.items() if n},
    }


def data(reports: list[ContractReport], context: Context) -> dict:
    return {
        "schema": 1,
        "commit": context.commit,
        "generated_at": context.generated_at,
        "contracts": [
            {
                **{k: v for k, v in asdict(r).items() if k != "points"},
                "group": r.group,
                "status": r.status,
                "served": r.served,
                "geomean": r.geomean,
                "points": [asdict(p) for p in r.points],
            }
            for r in reports
        ],
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def write_section(out: Path, reports: list[ContractReport], context: Context) -> dict:
    """Replace ``out`` (site-data's kernels/) with the pages of ``reports``; returns section.json."""
    if out.exists():
        for child in out.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    (out / "contracts").mkdir(parents=True, exist_ok=True)
    pages = "\n".join(index_page(reports, context, lang) for lang in LANGS)
    (out / "index.html").write_text(document(TITLE, pages))
    for report in reports:
        titles = {lang: f"{report.contract_id} · {TITLE[lang]}" for lang in LANGS}
        pages = "\n".join(contract_page(report, context, lang) for lang in LANGS)
        (out / "contracts" / page_name(report.contract_id)).write_text(document(titles, pages))
    write_json(out / "data.json", data(reports, context))
    summary = section_summary(reports, context)
    write_json(out / "section.json", summary)
    return summary
