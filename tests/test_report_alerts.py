import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from regresscheck.alerts import build_slack_payload, send_slack, should_alert
from regresscheck.config import Thresholds
from regresscheck.diff import Status, compare
from regresscheck.drift import detect_drift
from regresscheck.report import render_html, render_markdown, trend_svg
from regresscheck.store import HistoryPoint

from .conftest import make_case, make_run

BASE = make_run([make_case(f"tec-{i:03d}", "technical", "technical") for i in range(10)], "base1", "v2")
CAND = make_run(
    [make_case(f"tec-{i:03d}", "technical", "general" if i < 3 else "technical") for i in range(10)], "cand1", "v3<b>"
)
DIFF = compare(BASE, CAND, Thresholds())
HIST = [
    HistoryPoint(run_id=f"r{i}", created_at="t", prompt_version=f"v{i}", accuracy=a, mean_summary_score=4, status=None)
    for i, a in enumerate([0.9, 0.9, 0.7])
]
DRIFT = detect_drift([h.accuracy for h in HIST], Thresholds(drift_window=2))


def test_trend_svg() -> None:
    assert "No run history" in trend_svg([], [])
    svg = trend_svg(HIST, DRIFT.series)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<circle") == 3
    assert "stroke-dasharray" in svg  # moving average drawn
    single = trend_svg(HIST[:1], [None])
    assert single.count("<circle") == 1 and "stroke-dasharray" not in single


def test_render_html_contains_every_section_and_escapes() -> None:
    html = render_html(DIFF, BASE, CAND, HIST, DRIFT, Status.CRITICAL)
    assert "CRITICAL" in html
    assert "v3&lt;b&gt;" in html and "v3<b>" not in html
    for section in (
        "Run metadata",
        "Scorecard",
        "Per-category accuracy",
        "Regressions (3)",
        "Improvements (0)",
        "Trend (last 3 runs)",
        "Slow drift",
    ):
        assert section in html, section
    assert "tec-000" in html
    assert "#cf222e" in html
    assert "SLOW DRIFT" in html


def test_render_html_without_drift_history() -> None:
    html = render_html(DIFF, BASE, CAND, [], None, Status.PASS)
    assert "Not enough history" in html and "No run history" in html and "#1a7f37" in html


def test_render_markdown() -> None:
    md = render_markdown(DIFF, Status.CRITICAL, DRIFT, "https://ci/run/1", max_rows=2)
    assert md.startswith("<!-- regresscheck -->")
    assert "**CRITICAL**" in md
    assert "| Category accuracy | 100.0% | 70.0% | -30.0% |" in md
    assert "Regressions / improvements | | | 3 / 0 |" in md
    assert "| `tec-000` | technical | technical | general |" in md
    assert "| ... | 1 more | | |" in md
    assert "Full HTML diff report: https://ci/run/1" in md
    assert "Slow drift:" in md
    assert "Per-category: technical -30.0%" in md
    clean = render_markdown(compare(BASE, BASE, Thresholds()), Status.PASS, None)
    assert "No threshold exceeded." in clean and "Regressed cases" not in clean and "Full HTML" not in clean


def test_slack_payload() -> None:
    payload = build_slack_payload(DIFF, Status.CRITICAL, DRIFT, "https://ci/run/1")
    assert payload["text"].startswith(":rotating_light: regresscheck CRITICAL: v3<b> vs v2")
    assert "3 regressions detected, accuracy dropped from 100% to 70%" in payload["text"]
    blocks = payload["blocks"]
    assert blocks[0]["type"] == "header"
    assert "Mean summary score 4.00 -> 4.00" in blocks[1]["text"]["text"]
    assert "Slow drift" in blocks[2]["text"]["text"]
    assert blocks[-1]["text"]["text"] == "<https://ci/run/1|Full diff report>"
    plain = build_slack_payload(compare(BASE, BASE, Thresholds()), Status.PASS, None, None)
    assert len(plain["blocks"]) == 2
    assert "moved from 100% to 100%" in plain["text"]


def test_should_alert() -> None:
    assert not should_alert(Status.PASS, None)
    assert should_alert(Status.PASS, None, notify_on_pass=True)
    assert should_alert(Status.WARN, None)
    assert should_alert(Status.PASS, DRIFT)
    assert not should_alert(Status.PASS, detect_drift([0.9], Thresholds()))


@pytest.fixture(autouse=True)
def _no_proxy_for_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")


def _server(status: int) -> tuple[HTTPServer, list[bytes]]:
    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_send_slack_posts_json_to_a_real_http_server() -> None:
    server, received = _server(200)
    try:
        send_slack(f"http://127.0.0.1:{server.server_port}/hook", {"text": "hi"})
    finally:
        server.shutdown()
    assert json.loads(received[0]) == {"text": "hi"}


def test_send_slack_raises_on_http_error() -> None:
    server, _ = _server(500)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            send_slack(f"http://127.0.0.1:{server.server_port}/hook", {"text": "hi"})
    finally:
        server.shutdown()


def test_slack_header_is_truncated_to_slack_limit() -> None:
    long = make_run(CAND.cases, "c2", "v" * 300)
    payload = build_slack_payload(compare(BASE, long, Thresholds()), Status.CRITICAL, None, None)
    assert len(payload["blocks"][0]["text"]["text"]) == 150


def test_send_slack_uses_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class Resp:
        def raise_for_status(self) -> None:
            return None

    def fake_post(url: str, json: object, timeout: float) -> Resp:
        seen.update(url=url, json=json, timeout=timeout)
        return Resp()

    monkeypatch.setattr("regresscheck.alerts.httpx.post", fake_post)
    send_slack("https://hooks.example/x", {"text": "t"})
    assert seen == {"url": "https://hooks.example/x", "json": {"text": "t"}, "timeout": 10.0}


def test_trend_axis_starts_below_the_lowest_point() -> None:
    svg = trend_svg(HIST, DRIFT.series)
    assert ">60%</text>" in svg and ">100%</text>" in svg and ">0%</text>" not in svg
