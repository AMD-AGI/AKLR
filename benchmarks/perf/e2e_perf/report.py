# SPDX-License-Identifier: MIT
"""Render the index of groups and one page per measured group, AKLR on against off at each workload point."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from html import escape
from pathlib import Path

from perf_utils.gh import repo_url

from e2e_perf.model_config import DEFAULT_CONFIG, group_key, group_path, load_config
from e2e_perf.utils import INTERACTIVITY, SCHEMA, THROUGHPUT, pair, read_json, utc_now, write_json

NOTABLE = 0.03
LANGS = ("en", "zh")
TITLE = {"en": "Daily end-to-end performance", "zh": "端到端性能"}
HOME = {"en": "AKLR Dashboard", "zh": "AKLR 看板"}


def tr(lang: str, en: str, zh: str) -> str:
    return zh if lang == "zh" else en


def status_label(lang: str, status: str) -> str:
    labels = {
        "ok": ("OK", "全部成功"),
        "partial": ("Partial success", "部分成功"),
        "failed": ("Failed", "失败"),
        "not-completed": ("Not completed", "未完成"),
    }
    return tr(lang, *labels[status]) if status in labels else status


def pct(r: float | None) -> str:
    return "—" if r is None else f"{(r - 1) * 100:+.1f}%"


def delta_class(r: float | None) -> str:
    """Color a change once its shown percentage reaches the threshold, so +3.0% is always colored."""
    if r is None or round(abs(r - 1) * 100, 1) < NOTABLE * 100:
        return ""
    return "delta good" if r > 1 else "delta bad"


def num(value) -> str:
    return "—" if value is None else f"{value:.1f}"


def times(r: float | None) -> str:
    return "—" if r is None else f"{r:.3f}×"


# The stylesheet and language script are copied from benchmarks/perf/agentx/report.py, which
# .github/scripts/build_site.py copies too; keep the three in sync.
STYLE = """
:root { color-scheme:dark; --fg:#e6edf7; --muted:#8a97ad; --line:#1e2a40; --edge:#2b3d5c; --bg:#070b14; --panel:#0e1525; --panel2:#121b2e; --cyan:#22d3ee; --ok:#34d399; --warn:#fbbf24; --bad:#f87171; }
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; font:14px/1.55 "Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif; color:var(--fg); background:radial-gradient(1200px 500px at 15% -10%,#123a5a55,transparent 60%) no-repeat,radial-gradient(900px 400px at 100% 0,#3b1d6a44,transparent 60%) no-repeat,var(--bg); }
html[data-lang="en"] .page:not([data-page="en"]), html[data-lang="zh"] .page:not([data-page="zh"]) { display:none; }
a { color:var(--cyan); text-decoration:none; } a:hover { text-decoration:underline; }
header { max-width:1320px; margin:0 auto; padding:28px 32px 18px; border-bottom:1px solid var(--line); }
header .top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
header h1 { margin:0 0 6px; font-size:26px; font-weight:650; letter-spacing:-.02em; background:linear-gradient(90deg,#fff,#9fb4d6); -webkit-background-clip:text; background-clip:text; color:transparent; }
header .meta { color:var(--muted); font-size:13px; }
header a { color:var(--cyan); }
.lang-switch { display:flex; flex:none; border:1px solid var(--edge); border-radius:999px; padding:3px; }
.lang-switch button { background:none; color:var(--muted); border:0; padding:3px 12px; border-radius:999px; font:inherit; font-size:12px; cursor:pointer; }
.lang-switch button[aria-pressed="true"] { background:var(--cyan); color:#04111a; font-weight:700; }
main { max-width:1320px; margin:0 auto; padding:24px 32px 48px; }
main > p { color:#c4d0e3; }
h2 { font-size:17px; margin:28px 0 12px; }
h3 { font-size:15px; margin:18px 0 8px; }
.card { background:linear-gradient(180deg,var(--panel2),var(--panel)); border:1px solid var(--line); border-radius:14px; padding:12px 16px; box-shadow:0 8px 24px #0006; }
.chart { width:100%; height:auto; }
.chart-title { font-size:14px; font-weight:600; fill:var(--fg); }
.grid { stroke:var(--line); } .tick { font-size:11px; fill:var(--muted); }
.axis { font-size:12px; fill:#c4d0e3; } .legend { font-size:12px; fill:#c4d0e3; }
.chart [data-point-tooltip] { cursor:help; }
.chart [data-point-tooltip]:focus { outline:2px solid var(--cyan); outline-offset:3px; }
.chart [fill="white"] { fill:var(--panel); }
.chart [stroke="#2563eb"] { stroke:#60a5fa; } .chart [fill="#2563eb"] { fill:#60a5fa; }
.chart [stroke="#ea580c"] { stroke:#fb923c; } .chart [fill="#ea580c"] { fill:#fb923c; }
.chart [stroke="#16a34a"] { stroke:#34d399; } .chart [fill="#16a34a"] { fill:#34d399; }
.chart [stroke="#9333ea"] { stroke:#c084fc; } .chart [fill="#9333ea"] { fill:#c084fc; }
.chart [stroke="#0891b2"] { stroke:#22d3ee; } .chart [fill="#0891b2"] { fill:#22d3ee; }
.chart [stroke="#64748b"] { stroke:#cbd5e1; } .chart [fill="#64748b"] { fill:#cbd5e1; }
.chart [stroke="#9ca3af"] { stroke:#7c8aa5; } .chart [fill="#9ca3af"] { fill:#7c8aa5; }
.point-tooltip { position:fixed; z-index:10; max-width:min(420px,calc(100vw - 24px)); padding:10px 12px; border-radius:8px; background:var(--panel2); color:var(--fg); border:1px solid var(--edge); box-shadow:0 8px 24px #000a; font-size:12px; white-space:pre-line; overflow-wrap:anywhere; pointer-events:none; }
table { border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.card table { background:transparent; border:0; }
table.compact { width:auto; min-width:640px; } table.compact th, table.compact td { padding:6px 16px; }
th, td { padding:8px 12px; border-bottom:1px solid var(--line); text-align:left; font-variant-numeric:tabular-nums; }
tr:last-child td { border-bottom:0; } tbody tr:hover td { background:#ffffff05; }
th { background:#0a101c; color:var(--muted); font-weight:600; font-size:12px; letter-spacing:.02em; }
.muted { color:var(--muted); } .good { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
tbody tr td.delta.good { background:#34d3991f; } tbody tr td.delta.bad { background:#f871711f; }
.badge { display:inline-block; font-size:12px; padding:1px 8px; border-radius:999px; vertical-align:middle; border:1px solid transparent; }
.badge.comparable { background:#34d3991a; border-color:#34d39955; color:#6ee7b7; } .badge.new-segment { background:#fbbf241a; border-color:#fbbf2455; color:#fcd34d; }
.badge.new-arm { background:#818cf81f; border-color:#818cf866; color:#c7d2fe; }
.badge.no-baseline { background:#94a3b81a; border-color:#94a3b855; color:#cbd5e1; }
.badge.curve, .badge.curve-no-overlap { background:#818cf81f; border-color:#818cf866; color:#c7d2fe; }
.warn { background:#fbbf2412; border:1px solid #fbbf2466; color:#fde68a; border-radius:10px; padding:10px 14px; }
.warn li, .notes li { margin:2px 0; }
code { font:12px "JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace; color:#c4d0e3; }
dl { display:grid; grid-template-columns:fit-content(32%) 1fr; gap:4px 16px; margin:0; }
dt { color:var(--muted); overflow-wrap:anywhere; } dd { margin:0; min-width:0; overflow-wrap:anywhere; }
@media (max-width:760px) { header, main { padding-left:16px; padding-right:16px; } header h1 { font-size:21px; } }
"""

SCRIPT = """
(function () {
  var root = document.documentElement, titles = %s;
  function syncButtons() {
    document.querySelectorAll("[data-set-lang]").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.setLang === root.dataset.lang));
    });
  }
  function set(lang, save) {
    hidePoint();
    root.dataset.lang = lang;
    root.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = titles[lang];
    syncButtons();
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
  function valid(lang) { return lang === "en" || lang === "zh"; }
  set(valid(asked) ? asked : valid(saved) ? saved : "en", false);
  document.addEventListener("DOMContentLoaded", syncButtons);
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
  document.addEventListener("pointermove", function (event) {
    if (activePoint && event.target === activePoint) {
      showPoint(activePoint, event.clientX, event.clientY);
    }
  });
  document.addEventListener("pointerout", function (event) {
    if (event.target === activePoint) { hidePoint(); }
  });
  document.addEventListener("focusin", function (event) {
    var point = event.target.closest("[data-point-tooltip]");
    if (point) {
      var box = point.getBoundingClientRect();
      showPoint(point, box.right, box.bottom);
    }
  });
  document.addEventListener("focusout", function (event) {
    if (event.target === activePoint) { hidePoint(); }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { hidePoint(); }
  });
  window.addEventListener("scroll", hidePoint, true);
  window.addEventListener("resize", hidePoint);
})();
"""


def language_switch(lang: str) -> str:
    buttons = "".join(
        f'<button type="button" data-set-lang="{code}" aria-pressed="{str(code == lang).lower()}">{label}</button>'
        for code, label in (("en", "EN"), ("zh", "中文"))
    )
    return (
        f'<div class="lang-switch" role="group" aria-label="{tr(lang, "Language", "语言")}">{buttons}</div>'
    )


def render_document(titles: dict[str, str], pages: str) -> str:
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


def run_link(lang: str, run_id: str | None, run_url: str | None) -> str:
    if not run_id:
        return ""
    label = f"{tr(lang, 'run', '运行')} {escape(str(run_id))}"
    return f'<a href="{escape(run_url)}">{label}</a>' if run_url else label


def commit_link(sha: str | None, repo_url: str) -> str:
    if not sha or sha == "unknown":
        return "—"
    return f'<a href="{escape(repo_url)}/commit/{escape(sha)}"><code>{escape(sha[:10])}</code></a>'


def entry(item: dict, results: dict | None, run_url: str | None) -> dict:
    """One group's row for the site manifest and latest.json."""
    ids = (item["model"]["id"], item["target"]["id"], item["framework"]["id"])
    row = {
        "key": group_key(*ids),
        "model": item["model"],
        "target": item["target"],
        "framework": item["framework"],
        "run_key": item.get("run_key"),
        "complete": False,
        "status": "failed",
        "matched": 0,
        "expected": None,
        "tput_geomean_ratio": None,
        "min_ratio": None,
        "max_ratio": None,
        "measured_at": None,
        "aklr_sha": None,
        "run_id": None,
        "run_url": run_url,
        "error": item.get("error"),
        "gpus": item.get("gpus"),
        "image": None,
        "host": None,
        "gpu": None,
        "report": False,
    }
    if results is None:
        row["error"] = row["error"] or "the sweep saved no results"
        return row
    run = results["run"]
    comparison = pair(results)
    devices = (run.get("hardware") or {}).get("devices") or []
    row.update(
        {key: comparison[key] for key in ("complete", "status", "matched", "expected")},
        tput_geomean_ratio=comparison["tput_geomean_ratio"],
        min_ratio=comparison["min_ratio"],
        max_ratio=comparison["max_ratio"],
        measured_at=run.get("finished_at"),
        aklr_sha=run.get("aklr_sha"),
        run_id=run.get("id"),
        error=run.get("error"),
        gpus=run.get("gpus"),
        image=(run.get("image") or {}).get("run"),
        host=run.get("host"),
        gpu=devices[0]["name"] if devices else None,
        report=True,
    )
    return row


def header(lang: str, title: str, back: str, back_label: str, generated: str) -> str:
    return f"""<header><div class="top"><h1>{escape(title)}</h1>{language_switch(lang)}</div>
<div class="meta"><a href="{back}?lang={lang}">← {escape(back_label)}</a> · {
        tr(lang, "generated", "生成于")
    } {escape(generated)}</div></header>"""


def index_page(
    entries: list[dict], lang: str, repo_url: str, workloads: list[dict], generated: str, home: str
) -> str:
    rows = []
    for row in entries:
        ids = (row["model"]["id"], row["target"]["id"], row["framework"]["id"])
        label = escape(row["model"]["display_name"])
        if row.get("report"):
            label = f'<a href="{group_path(*ids).as_posix()}/index.html?lang={lang}">{label}</a>'
        parallel = f" · TP{row['gpus']}" if row.get("gpus") else ""
        detail = f'<br><span class="muted">{escape(row["target"]["id"])} · {escape(row["framework"]["id"])}{parallel}</span>'
        error = str(row.get("error") or "")
        error_line = f'<br><span class="muted">{escape(error[:300])}{"…" if len(error) > 300 else ""}</span>'
        span = (
            f"{times(row['min_ratio'])} – {times(row['max_ratio'])}"
            if row.get("min_ratio") is not None
            else "—"
        )
        gain = row.get("tput_geomean_ratio")
        measured = escape((row.get("measured_at") or "—")[:10])
        link = run_link(lang, row.get("run_id"), row.get("run_url"))
        rows.append(
            f"<tr><td>{label}{detail}</td>"
            f"<td>{escape(status_label(lang, row['status']))}{error_line if error else ''}</td>"
            f"<td>{row.get('matched', 0)}/{row.get('expected') or '—'}</td>"
            f'<td class="{delta_class(gain)}">{times(gain)}</td><td>{span}</td>'
            f"<td>{measured}{'<br>' + link if link else ''}</td>"
            f"<td>{commit_link(row.get('aklr_sha'), repo_url)}</td></tr>"
        )
    headers = (
        tr(lang, "Model", "模型"),
        tr(lang, "Status", "状态"),
        tr(lang, "Matched", "匹配点"),
        tr(lang, "Throughput geomean", "吞吐几何平均"),
        tr(lang, "Range", "范围"),
        tr(lang, "Measured", "测量时间"),
        "main",
    )
    shapes = ", ".join(
        f"{escape(workload['id'])} (ISL {workload['isl']} / OSL {workload['osl']})" for workload in workloads
    )
    lead = tr(
        lang,
        f"Random input, workloads {shapes}. Total tok/s/GPU counts input + output throughput; "
        "P90 tok/s/user is 1 / P90 TPOT. Throughput geomean is AKLR on / off over the matched points "
        "of every workload and concurrency.",
        f"随机输入，负载 {shapes}。总 tok/s/GPU 为输入+输出吞吐，P90 tok/s/user 为 1 / P90 TPOT。"
        "吞吐几何平均为所有负载与并发下匹配测量点的 AKLR 开启 / 关闭之比。",
    )
    empty = f'<tr><td colspan="7" class="muted">{tr(lang, "No results yet.", "暂无结果。")}</td></tr>'
    return f"""<div class="page" data-page="{lang}" lang="{tr(lang, "en", "zh-CN")}">
{header(lang, TITLE[lang], home, HOME[lang], generated)}
<main><p>{lead}</p>
<div class="card" style="overflow-x:auto"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr></thead>
<tbody>{"".join(rows) or empty}</tbody></table></div></main></div>"""


def render_index(
    entries: list[dict], repo_url: str, workloads: list[dict], home: str = "../index.html"
) -> str:
    generated = utc_now()
    pages = "\n".join(index_page(entries, lang, repo_url, workloads, generated, home) for lang in LANGS)
    return render_document(dict(TITLE), pages)


def cell(point: dict | None, metric: str, lang: str) -> str:
    if point and point.get("status") == "ok":
        return num(point.get(metric))
    status = point["status"] if point else "not-completed"
    error = (point or {}).get("error") or ""
    return f'<span class="muted" title="{escape(error)}">{escape(status_label(lang, status))}</span>'


def group_page(results: dict, row: dict, lang: str, repo_url: str, generated: str) -> str:
    comparison = pair(results)
    run = results["run"]
    gain = comparison["tput_geomean_ratio"]
    summary = (
        tr(
            lang,
            "Throughput geomean over matched pairs, enabled / disabled: ",
            "匹配点对的吞吐几何平均，开启 / 关闭：",
        )
        + (
            f'<strong class="{delta_class(gain)}" style="font-size:24px">{gain:.3f}× ({pct(gain)})</strong>'
            if gain is not None
            else "<strong>—</strong>"
        )
        + tr(
            lang,
            f" · {comparison['matched']}/{comparison['expected']} matched pairs",
            f" · {comparison['matched']}/{comparison['expected']} 对匹配测量点",
        )
    )
    rows = []
    for item in comparison["rows"]:
        off, on = item["disabled"], item["enabled"]
        rows.append(
            f"<tr><td>{escape(str(item['workload']))}</td><td>{item['conc']}</td>"
            f"<td>{cell(off, THROUGHPUT, lang)}</td><td>{cell(on, THROUGHPUT, lang)}</td>"
            f'<td class="{delta_class(item["tput_ratio"])}">{pct(item["tput_ratio"])}</td>'
            f"<td>{cell(off, INTERACTIVITY, lang)}</td><td>{cell(on, INTERACTIVITY, lang)}</td>"
            f'<td class="{delta_class(item["intvty_ratio"])}">{pct(item["intvty_ratio"])}</td></tr>'
        )
    headers = (
        tr(lang, "Workload", "负载"),
        tr(lang, "Concurrency", "并发"),
        tr(lang, "Off total tok/s/GPU", "关闭 总 tok/s/GPU"),
        tr(lang, "On total tok/s/GPU", "开启 总 tok/s/GPU"),
        tr(lang, "Throughput Δ", "吞吐变化"),
        tr(lang, "Off P90 tok/s/user", "关闭 P90 tok/s/user"),
        tr(lang, "On P90 tok/s/user", "开启 P90 tok/s/user"),
        tr(lang, "Interactivity Δ", "交互速度变化"),
    )
    error = run.get("error")
    warn = f'<div class="warn">{escape(str(error))}</div>' if error else ""
    machine = " · ".join(escape(str(value)) for value in (row.get("host"), row.get("gpu")) if value)
    footer = " · ".join(
        part
        for part in (
            escape((row.get("measured_at") or "—")[:10]),
            f"AKLR {commit_link(run.get('aklr_sha'), repo_url)}",
            f"<code>{escape(row.get('image') or '—')}</code>",
            machine,
            run_link(lang, run.get("id"), row.get("run_url")),
        )
        if part
    )
    title = f"{results['model']['display_name']} · {results['framework']['id']} · {TITLE[lang]}"
    return f"""<div class="page" data-page="{lang}" lang="{tr(lang, "en", "zh-CN")}">
{header(lang, title, "../" * 6 + "index.html", TITLE[lang], generated)}
<main>{warn}
<p class="card">{summary}</p>
<div class="card" style="overflow-x:auto"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="muted">{footer}</p></main></div>"""


def render_group(results: dict, row: dict, repo_url: str) -> str:
    generated = utc_now()
    titles = {
        lang: f"{results['model']['display_name']} · {results['framework']['id']} · {TITLE[lang]}"
        for lang in LANGS
    }
    return render_document(
        titles, "\n".join(group_page(results, row, lang, repo_url, generated) for lang in LANGS)
    )


def mismatch(results: dict, item: dict, run_id: str) -> str | None:
    """Why results do not belong to this run's plan of this group, or None if they do."""
    run = results["run"]
    for part in ("model", "target", "framework"):
        if results[part]["id"] != item[part]["id"]:
            return f"results are for {part} {results[part]['id']!r}, not {item[part]['id']!r}"
    if str(run.get("id")) != str(run_id):
        return f"results are from run {run.get('id')!r}, not {run_id!r}"
    if run.get("run_key") != item.get("run_key"):
        return "results were measured for another run key"
    return None


def build(plan: Path, results_dir: Path, out: Path, run_url: str | None, config: Path) -> dict:
    """Write site/<group>/index.html and results.json for each measured group, the index and manifest.json."""
    loaded = load_config(config)
    repo = repo_url()
    manifest = read_json(plan)
    entries = []
    for item in manifest["groups"]:
        if item["status"] == "skipped":
            continue
        relative = group_path(item["model"]["id"], item["target"]["id"], item["framework"]["id"])
        source = results_dir / relative / "results.json"
        results = None
        if item["status"] == "planned" and source.is_file():
            try:
                results = read_json(source)
                pair(results)
                if reason := mismatch(results, item, manifest["run_id"]):
                    raise ValueError(reason)
            except (OSError, ValueError, KeyError, TypeError) as error:
                item = {**item, "error": f"unreadable results: {type(error).__name__}: {error}"}
                results = None
        row = entry(item, results, run_url)
        if results is not None:
            (out / relative).mkdir(parents=True, exist_ok=True)
            (out / relative / "index.html").write_text(render_group(results, row, repo))
            shutil.copyfile(source, out / relative / "results.json")
        entries.append(row)
    site = {"schema": SCHEMA, "run_id": manifest["run_id"], "generated_at": utc_now(), "groups": entries}
    write_json(out / "manifest.json", site)
    (out / "index.html").write_text(render_index(entries, repo, loaded["workloads"]))
    return site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m e2e_perf report", description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="the plan's manifest.json")
    parser.add_argument("--results", type=Path, required=True, help="the sweeps' results, by group path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-url")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        site = build(args.plan, args.results, args.out, args.run_url, args.config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for row in site["groups"]:
        print(f"{row['key']}: {row['status']}, {row['matched']}/{row['expected'] or '—'} matched")
    return 0
