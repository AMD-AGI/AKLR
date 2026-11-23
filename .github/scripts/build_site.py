#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the Pages site from a site-data checkout: each section as is, plus a home page."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from html import escape
from pathlib import Path

SECTION_SCHEMA = 1
SECTION_ID = re.compile(r"[a-z][a-z0-9-]*")
TITLE = {"en": "AKLR Weekly Report", "zh": "AKLR 每周报告"}
# Status keys a section's status_counts may hold, in display order; any other key is shown as is.
STATUS_LABELS = {
    "ok": ("OK", "成功"),
    "partial": ("Partial", "部分成功"),
    "failed": ("Failed", "失败"),
    "resolve-failed": ("Resolve failed", "解析失败"),
    "not-run": ("Not selected", "本轮未选择"),
    "not-served": ("Not served", "未生效"),
    "not-measured": ("Not measured", "未测量"),
}

# Dark console theme. The palette, header and language switch are copied in
# benchmarks/agentx/agentx_ci/report.py (STYLE); keep the two in sync.
STYLE = """
:root{color-scheme:dark;--bg:#070b14;--panel:#0e1525;--panel2:#121b2e;--fg:#e6edf7;--muted:#8a97ad;--line:#1e2a40;--edge:#2b3d5c;--cyan:#22d3ee;--ok:#34d399;--warn:#fbbf24;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--fg);font:14px/1.55 "Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;background:radial-gradient(1200px 500px at 15% -10%,#123a5a55,transparent 60%) no-repeat,radial-gradient(900px 400px at 100% 0,#3b1d6a44,transparent 60%) no-repeat,var(--bg)}
html[data-lang="en"] [lang="zh"],html[data-lang="zh"] [lang="en"]{display:none}
a{color:var(--cyan);text-decoration:none}a:hover{text-decoration:underline}
.mono,code{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
code{font-size:12px;color:#c4d0e3}
.wrap{max-width:1120px;margin:0 auto;padding:0 28px}
header{padding:22px 0;display:flex;justify-content:space-between;align-items:center;gap:16px}
.brand{display:flex;align-items:center;gap:12px}
.brand .mark{width:32px;height:32px;border:1px solid var(--edge);border-radius:8px;display:grid;place-items:center;color:var(--cyan);font-weight:700;box-shadow:0 0 18px #22d3ee33 inset}
.brand b{font-size:16px;letter-spacing:.06em}.brand small{display:block;color:var(--muted);font-size:12px}
.right{display:flex;align-items:center}
.live{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);margin-right:14px}
.live i{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 8px var(--ok)}
.lang{display:flex;flex:none;border:1px solid var(--edge);border-radius:999px;padding:3px}
.lang button{border:0;background:none;color:var(--muted);font:inherit;font-size:12px;padding:3px 12px;border-radius:999px;cursor:pointer}
.lang button[aria-pressed="true"]{background:var(--cyan);color:#04111a;font-weight:700}
h1{font-size:40px;letter-spacing:-.03em;margin:28px 0 6px;background:linear-gradient(90deg,#fff,#9fb4d6);-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{color:var(--muted);margin:0 0 28px}
.panel{margin:0 0 28px;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:16px;overflow:hidden}
table{width:100%;border-collapse:collapse}
th{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600;text-align:left;padding:10px 22px;border-bottom:1px solid var(--line)}
td{padding:14px 22px;border-bottom:1px solid var(--line);vertical-align:middle}tr:last-child td{border-bottom:0}
tbody tr:hover td{background:#ffffff05}
.st{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:600;color:var(--muted)}
.st i{width:8px;height:8px;border-radius:50%;background:var(--muted)}
.st.ok{color:var(--ok)}.st.ok i{background:var(--ok);box-shadow:0 0 8px var(--ok)}
.st.partial{color:var(--warn)}.st.partial i{background:var(--warn);box-shadow:0 0 8px var(--warn)}
.st.failed,.st.resolve-failed{color:var(--bad)}.st.failed i,.st.resolve-failed i{background:var(--bad);box-shadow:0 0 8px var(--bad)}
.empty{padding:18px 22px;color:var(--muted)}
.section{font-weight:600}.counts{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px}.counts .sep{color:var(--edge)}
.st b{font-family:"JetBrains Mono",ui-monospace,monospace;color:var(--fg);font-weight:600}
footer{color:var(--muted);font-size:12px;padding:0 0 40px}
@media (max-width:760px){h1{font-size:30px}.live{display:none}th,td{padding-left:14px;padding-right:14px}}
"""


# Language choice shared with the AgentX pages (benchmarks/agentx/agentx_ci/report.py SCRIPT): ?lang=
# first, then localStorage "agentx-lang"; on a first visit the browser language, where the AgentX pages
# take English. Links marked data-href carry the choice on as ?lang=.
SCRIPT = """
(function () {
  var root = document.documentElement, titles = %s;
  function valid(lang) { return lang === "en" || lang === "zh"; }
  function set(lang, save) {
    root.dataset.lang = lang;
    root.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = titles[lang];
    document.querySelectorAll("[data-set-lang]").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.setLang === lang));
    });
    document.querySelectorAll("a[data-href]").forEach(function (link) {
      link.href = link.dataset.href + "?lang=" + lang;
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
  var asked = null, saved = null;
  try { asked = new URLSearchParams(location.search).get("lang"); } catch (e) {}
  try { saved = localStorage.getItem("agentx-lang"); } catch (e) {}
  var browser = (navigator.language || "").toLowerCase().indexOf("zh") === 0 ? "zh" : "en";
  set(valid(asked) ? asked : valid(saved) ? saved : browser, false);
  document.addEventListener("DOMContentLoaded", function () { set(root.dataset.lang, false); });
  document.addEventListener("click", function (event) {
    var button = event.target.closest("[data-set-lang]");
    if (button && valid(button.dataset.setLang)) { set(button.dataset.setLang, true); }
  });
})();
"""


def both(en: str, zh: str) -> str:
    """Both translations of an already escaped text; CSS shows the selected one."""
    return f'<span lang="en">{en}</span><span lang="zh">{zh}</span>'


def when(value: object) -> str:
    match = re.match(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d)", str(value))
    return f"{match[1]} {match[2]} UTC" if match else str(value)


def count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def read_sections(data: Path) -> list[dict]:
    sections = []
    for path in sorted(data.glob("*/section.json")):
        section = json.loads(path.read_text())
        directory = path.parent.name
        if section.get("schema") != SECTION_SCHEMA or section.get("id") != directory:
            raise ValueError(f"{path}: needs schema {SECTION_SCHEMA} and id {directory!r}")
        entry = Path(section.get("path", ""))
        if not SECTION_ID.fullmatch(directory) or entry.is_absolute() or ".." in entry.parts:
            raise ValueError(f"{path}: invalid section id or path")
        if not (path.parent / entry).is_file():
            raise ValueError(f"{path}: {entry} does not exist")
        sections.append(section)
    return sections


def status_counts(counts: object) -> str:
    if not isinstance(counts, dict) or not counts:
        return "—"
    order = [*(key for key in STATUS_LABELS if key in counts), *sorted(set(counts) - set(STATUS_LABELS))]
    items = []
    for key in order:
        en, zh = STATUS_LABELS.get(key, (str(key), str(key)))
        css = f"st {key}" if key in STATUS_LABELS else "st"
        items.append(
            f'<span class="{css}"><i></i>{both(escape(en), escape(zh))} <b>{count(counts[key])}</b></span>'
        )
    return '<span class="counts">' + '<span class="sep">·</span>'.join(items) + "</span>"


def section_row(section: dict) -> str:
    title = section.get("title") or {}
    en = escape(str(title.get("en") or section["id"]))
    zh = escape(str(title.get("zh") or title.get("en") or section["id"]))
    href = escape(f"{section['id']}/{section['path']}")
    run = section.get("last_run") or {}
    url = str(run.get("run_url") or "")
    latest = (
        f'<a href="{escape(url)}">{both("run", "运行")} {escape(str(run.get("run_id", "")))}</a> · '
        f"<code>{escape(str(run.get('commit', ''))[:12])}</code>"
        if url.startswith("https://")
        else "—"
    )
    updated = escape(when(section["updated_at"])) if section.get("updated_at") else "—"
    return (
        f'<tr><td class="section"><a data-href="{href}" href="{href}">{both(en, zh)}</a></td>'
        f'<td class="mono">{updated}</td><td>{latest}</td><td>{status_counts(section.get("status_counts"))}</td></tr>'
    )


def render_home(sections: list[dict], commit: str) -> str:
    head = "".join(
        f"<th>{both(en, zh)}</th>"
        for en, zh in (
            ("Section", "栏目"),
            ("Updated", "更新时间"),
            ("Latest run", "最近运行"),
            ("Status", "状态"),
        )
    )
    table = (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(section_row(s) for s in sections)}</tbody></table>"
        if sections
        else f'<div class="empty">{both("No results published yet.", "暂无已发布的结果。")}</div>'
    )
    latest = max((str(section.get("updated_at") or "") for section in sections), default="")
    live = (
        f'<span class="live"><i></i>{both(f"Updated {escape(when(latest))}", f"更新于 {escape(when(latest))}")}</span>'
        if latest
        else ""
    )
    titles = json.dumps(TITLE, ensure_ascii=False).replace("<", r"\u003c")
    return f"""<!doctype html>
<html lang="en" data-lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE["en"]}</title>
<style>{STYLE}</style>
<script>{SCRIPT % titles}</script>
</head><body><div class="wrap">
<header><div class="brand"><div class="mark mono">A</div><div><b>AKLR</b><small>{both("CI results on main", "main 分支 CI 结果")}</small></div></div>
<div class="right">{live}<div class="lang" role="group" aria-label="Language / 语言"><button type="button" data-set-lang="en" aria-pressed="true">EN</button><button type="button" data-set-lang="zh" aria-pressed="false">中文</button></div></div></header>
<h1>{both(TITLE["en"], TITLE["zh"])}</h1>
<p class="lead">{both("Every result is measured on <code>main</code> by a scheduled or manual CI run.", "所有结果均来自 <code>main</code> 分支上的定时或手动 CI 运行。")}</p>
<section class="panel">{table}</section>
<footer>site-data <code>{escape(commit[:12])}</code> · <a href="https://github.com/AMD-AGI/AKLR">github.com/AMD-AGI/AKLR</a></footer>
</div></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="site-data checkout")
    parser.add_argument("--out", type=Path, required=True, help="new site directory")
    parser.add_argument("--commit", required=True, help="the site-data commit being deployed")
    args = parser.parse_args(argv)
    try:
        sections = read_sections(args.data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True)
    for section in sections:
        shutil.copytree(args.data / section["id"], args.out / section["id"])
    (args.out / "index.html").write_text(render_home(sections, args.commit))
    print(f"Built {len(sections)} section(s) from site-data {args.commit[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
