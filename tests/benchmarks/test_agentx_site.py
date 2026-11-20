# SPDX-License-Identifier: MIT
"""The Pages home page that .github/scripts/build_site.py builds from the site-data sections."""

from __future__ import annotations

import json
import subprocess
import sys
from html import escape
from pathlib import Path

from tests.benchmarks.agentx_results import REPO

BUILD_SITE = REPO / ".github/scripts/build_site.py"


def write_section(data: Path, section_id: str = "agentx", **fields) -> Path:
    """A section as a CI publishes it: its entry page and section.json."""
    section = data / section_id
    section.mkdir(parents=True)
    (section / "index.html").write_text(section_id)
    summary = {
        "schema": 1,
        "id": section_id,
        "title": {"en": "AgentX end-to-end performance", "zh": "AgentX 端到端性能"},
        "path": "index.html",
        "updated_at": "2026-11-19T15:16:41Z",
        "last_run": {
            "run_id": "44491219482",
            "run_url": "https://github.com/AMD-AGI/AKLR/actions/runs/44491219482",
            "commit": "8806c22a61f61b0fccb1322b4e44edf36d7895e1",
        },
        "targets": 3,
        "status_counts": {"failed": 1, "partial": 2},
        **fields,
    }
    (section / "section.json").write_text(json.dumps(summary))
    return data


def build(data: Path, out: Path) -> str:
    data.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(BUILD_SITE),
            "--data",
            str(data),
            "--out",
            str(out),
            "--commit",
            "f6cb46c665c2aa",
        ],
        check=True,
        capture_output=True,
    )
    return (out / "index.html").read_text()


def test_home_lists_each_section_in_four_columns(tmp_path):
    data = write_section(tmp_path / "data")
    write_section(data, "accuracy", title={"en": "Accuracy", "zh": "精度"}, last_run=None, status_counts={})
    html = build(data, tmp_path / "site")
    head = html[html.index("<thead>") : html.index("</thead>")]
    assert head.count("<th>") == 4
    for en, zh in (
        ("Section", "栏目"),
        ("Updated", "更新时间"),
        ("Latest run", "最近运行"),
        ("Status", "状态"),
    ):
        assert f'<th><span lang="en">{en}</span><span lang="zh">{zh}</span></th>' in head
    assert "Targets" not in html and "manifest" not in html
    assert html.count("<tr><td") == 2
    agentx = html[html.index('href="agentx/index.html"') :].split("</tr>")[0]
    assert (
        '<span lang="en">AgentX end-to-end performance</span><span lang="zh">AgentX 端到端性能</span>'
        in agentx
    )
    assert '<td class="mono">2026-11-19 15:16 UTC</td>' in agentx
    assert (
        '<a href="https://github.com/AMD-AGI/AKLR/actions/runs/44491219482"><span lang="en">run</span>'
        '<span lang="zh">运行</span> 44491219482</a> · <code>8806c22a61f6</code>'
    ) in agentx
    accuracy = html[html.index('href="accuracy/index.html"') :].split("</tr>")[0]
    assert '<span lang="en">Accuracy</span><span lang="zh">精度</span>' in accuracy
    assert accuracy.endswith("<td>—</td><td>—</td>")
    assert 'data-href="accuracy/index.html"' in html
    assert (tmp_path / "site/accuracy/index.html").is_file()


def test_home_renders_status_counts_with_their_names_and_colors(tmp_path):
    counts = {"weird-state": 4, "failed": 1, "partial": 2, "ok": 3}
    html = build(write_section(tmp_path / "data", status_counts=counts), tmp_path / "site")
    status = html[html.index('<span class="counts">') :].split("</td>")[0]
    assert (
        status.index("st ok") < status.index("st partial") < status.index("st failed") < status.index("weird")
    )
    assert (
        '<span class="st ok"><i></i><span lang="en">OK</span><span lang="zh">成功</span> <b>3</b></span>'
        in status
    )
    assert (
        '<span class="st partial"><i></i><span lang="en">Partial</span><span lang="zh">部分成功</span> <b>2</b>'
        in status
    )
    assert (
        '<span class="st failed"><i></i><span lang="en">Failed</span><span lang="zh">失败</span> <b>1</b>'
        in status
    )
    assert (
        '<span class="st"><i></i><span lang="en">weird-state</span><span lang="zh">weird-state</span> <b>4</b>'
        in status
    )


def test_home_without_sections_says_nothing_is_published(tmp_path):
    html = build(tmp_path / "data", tmp_path / "site")
    assert "<table>" not in html
    assert '<span lang="en">No results published yet.</span><span lang="zh">暂无已发布的结果。</span>' in html
    assert "site-data <code>f6cb46c665c2</code>" in html


def test_home_carries_both_languages_and_the_shared_choice(tmp_path):
    html = build(write_section(tmp_path / "data"), tmp_path / "site")
    assert "<title>AKLR Weekly Report</title>" in html
    assert '<h1><span lang="en">AKLR Weekly Report</span><span lang="zh">AKLR 每周报告</span></h1>' in html
    assert '"zh": "AKLR 每周报告"' in html
    assert 'data-set-lang="en"' in html and 'data-set-lang="zh"' in html
    assert 'localStorage.getItem("agentx-lang")' in html
    assert 'link.href = link.dataset.href + "?lang=" + lang' in html
    assert "kpi" not in html
    assert "AKLR results" not in html and "AKLR 测试结果" not in html


def test_home_escapes_published_text(tmp_path):
    title = '<img src=x onerror="alert(1)">'
    data = write_section(
        tmp_path / "data",
        title={"en": title, "zh": title},
        status_counts={"<b>odd</b>": 1},
        last_run={"run_id": "<i>1</i>", "run_url": 'https://x/"><script>', "commit": "<c>"},
    )
    html = build(data, tmp_path / "site")
    assert escape(title) in html and title not in html
    assert "&lt;b&gt;odd&lt;/b&gt;" in html and "<b>odd</b>" not in html
    assert 'href="https://x/&quot;&gt;&lt;script&gt;"' in html
    assert "&lt;i&gt;1&lt;/i&gt;" in html and "<code>&lt;c&gt;</code>" in html


def test_home_does_not_link_a_run_url_that_is_not_https(tmp_path):
    data = write_section(tmp_path / "data", last_run={"run_id": "1", "run_url": "javascript:alert(1)"})
    html = build(data, tmp_path / "site")
    assert "javascript:" not in html
