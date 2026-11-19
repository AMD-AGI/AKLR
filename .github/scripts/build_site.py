#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the Pages site from a site-data checkout: each section as is, plus a home page."""

import argparse
import json
import re
import shutil
import sys
from html import escape
from pathlib import Path

SECTION_SCHEMA = 1
SECTION_ID = re.compile(r"[a-z][a-z0-9-]*")


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


def section_card(section: dict) -> str:
    title = section.get("title", {})
    run = section.get("last_run") or {}
    counts = ", ".join(f"{escape(k)} {int(v)}" for k, v in sorted(section.get("status_counts", {}).items()))
    source = (
        f'<a href="{escape(run["run_url"])}">run {escape(str(run.get("run_id", "")))}</a> · '
        f"<code>{escape(run.get('commit', '')[:12])}</code>"
        if run.get("run_url")
        else "—"
    )
    return (
        f'<tr><td><a href="{escape(section["id"])}/{escape(section["path"])}">'
        f"{escape(title.get('en', section['id']))}</a><br>{escape(title.get('zh', ''))}</td>"
        f"<td>{escape(section.get('updated_at', '—'))}</td><td>{source}</td>"
        f"<td>{int(section.get('targets', 0))}</td><td>{counts or '—'}</td></tr>"
    )


def render_home(sections: list[dict], commit: str) -> str:
    rows = "".join(section_card(section) for section in sections) or (
        '<tr><td colspan="5">No results published yet · 暂无已发布的结果</td></tr>'
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AKLR results</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:2rem auto;max-width:960px;padding:0 1rem;color:#1f2328}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d0d7de;padding:.5rem;text-align:left;vertical-align:top}}
th{{background:#f6f8fa}}.muted{{color:#656d76;font-size:.9em}}a{{color:#0969da}}
</style></head><body>
<h1>AKLR results · AKLR 测试结果</h1>
<p class="muted">Every result is measured on <code>main</code> by a scheduled or manual CI run.
所有结果均来自 <code>main</code> 分支上的定时或手动 CI 运行。</p>
<table><thead><tr><th>Section · 栏目</th><th>Updated · 更新时间</th><th>Latest run · 最近运行</th>
<th>Targets · 目标数</th><th>Status · 状态</th></tr></thead><tbody>{rows}</tbody></table>
<p class="muted">site-data <code>{escape(commit[:12])}</code></p>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="site-data checkout")
    parser.add_argument("--out", type=Path, required=True, help="new site directory")
    parser.add_argument("--commit", required=True, help="the site-data commit being deployed")
    args = parser.parse_args()
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
