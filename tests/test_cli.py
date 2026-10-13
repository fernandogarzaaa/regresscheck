"""End-to-end: the CLI against the real golden dataset with the mock provider."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from regresscheck.cli import baseline_path, main
from regresscheck.config import Settings
from regresscheck.store import RunStore

from .conftest import ROOT


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for var in ("SLACK_WEBHOOK_URL", "REGRESSCHECK_REPORT_URL", "REGRESSCHECK_TRACK"):
        monkeypatch.delenv(var, raising=False)
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    (tmp_path / "regresscheck.toml").write_text(
        f'[regresscheck]\ndataset = "{ROOT / "golden" / "dataset.json"}"\nprompts_dir = "{ROOT / "prompts"}"\n'
        f'active_prompt = "{ROOT / "prompts" / "v2.yaml"}"\n'
    )
    return tmp_path


V1 = str(ROOT / "prompts" / "v1.yaml")
BAD = str(ROOT / "demo" / "prompts" / "v3-bad.yaml")


def test_baseline_path_is_filesystem_safe() -> None:
    assert baseline_path(Settings(), "openai", "meta/llama 3:8b") == Path("baselines/openai__meta_llama_3_8b.json")


def test_validate(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate"]) == 0
    out = capsys.readouterr().out
    assert "80 cases" in out and "prompt" in out and "version v2" in out


def test_run_saves_and_writes_json(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--json", "out.json", "--concurrency", "4"]) == 0
    record = json.loads((workdir / "out.json").read_text())
    assert record["prompt_version"] == "v2" and len(record["cases"]) == 80
    assert "accuracy" in capsys.readouterr().out
    with RunStore(workdir / ".regresscheck" / "history.db") as store:
        assert store.get(record["run_id"]) is not None
    assert main(["run", "--no-save", "--prompt", V1]) == 0
    with RunStore(workdir / ".regresscheck" / "history.db") as store:
        assert len(store.history("default", "mock", "mock-classifier-v1")) == 1


def test_check_without_baseline_records_first_point(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check"]) == 0
    assert "no baseline" in capsys.readouterr().out


def test_bad_prompt_fails_the_gate_and_good_prompt_passes(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["baseline"]) == 0
    assert (workdir / "baselines" / "mock__mock-classifier-v1.json").is_file()
    assert main(["check", "--markdown", "pr.md", "--report", "r.html", "--report-url", "https://x/1"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(["check", "--prompt", BAD, "--markdown", "pr.md", "--report", "bad.html"]) == 2
    out = capsys.readouterr().out
    assert "CRITICAL" in out and "technical accuracy dropped" in out
    md = (workdir / "pr.md").read_text()
    assert "**CRITICAL**" in md and "v3-bad" in md
    assert "Regressions (" in (workdir / "bad.html").read_text()
    assert main(["check", "--prompt", BAD, "--fail-on", "never"]) == 0
    assert main(["check", "--prompt", BAD, "--fail-on", "warn"]) == 2


def test_warn_does_not_fail_by_default_but_fails_with_fail_on_warn(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["baseline"]) == 0
    # Raise the critical bar so the bad prompt only warns.
    monkeypatch.setenv("REGRESSCHECK_CRITICAL_THRESHOLD", "0.9")
    monkeypatch.setenv("REGRESSCHECK_SUMMARY_CRITICAL_THRESHOLD", "0.9")
    assert main(["check", "--prompt", BAD]) == 0
    assert main(["check", "--prompt", BAD, "--fail-on", "warn"]) == 2


def test_check_against_previous_and_run_id_then_diff_and_history(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", "--against", "previous"]) == 0
    assert main(["check", "--against", "previous", "--prompt", V1]) == 2
    with RunStore(workdir / ".regresscheck" / "history.db") as store:
        runs = store.history("default", "mock", "mock-classifier-v1")
    first, second = runs[0].run_id, runs[1].run_id
    assert [r.status for r in runs] == ["pass", "critical"]
    assert main(["check", "--against", first]) == 0
    assert main(["diff", first, second, "--report", "d.html"]) == 2
    assert (workdir / "d.html").is_file()
    assert main(["diff", second, first]) == 0
    capsys.readouterr()
    assert main(["history"]) == 0
    out = capsys.readouterr().out
    assert first in out and "critical" in out
    assert main(["history", "--model", "nothing"]) == 0
    assert "no runs recorded" in capsys.readouterr().out


def test_errors_exit_1(workdir: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert main(["check", "--against", "nope"]) == 1
    assert main(["diff", "a", "b"]) == 1
    assert main(["run", "--prompt", "missing.yaml"]) == 1
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["run", "--provider", "openai"]) == 1
    (workdir / "bad.toml").write_text("[regresscheck.thresholds]\nwarn = 0.5\ncritical = 0.1\n")
    assert main(["--config", "bad.toml", "validate"]) == 1
    assert "error:" in capsys.readouterr().err


def test_slack_alert_is_sent_on_critical(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    received: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert main(["baseline"]) == 0
        assert main(["check", "--slack"]) == 0
        assert "SLACK_WEBHOOK_URL not set" in capsys.readouterr().out
        monkeypatch.setenv("SLACK_WEBHOOK_URL", f"http://127.0.0.1:{server.server_port}/hook")
        monkeypatch.setenv("REGRESSCHECK_REPORT_URL", "https://ci.example/run/7")
        assert main(["check", "--slack"]) == 0  # pass: no alert
        assert received == []
        assert main(["check", "--slack", "--notify-on-pass"]) == 0
        assert main(["check", "--slack", "--prompt", BAD]) == 2
    finally:
        server.shutdown()
    assert len(received) == 2
    assert "CRITICAL" in str(received[1]["text"])
    assert "https://ci.example/run/7" in json.dumps(received[1])


def test_explicit_missing_config_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(tmp_path / "nope.toml"), "validate"]) == 1
    assert "config file not found" in capsys.readouterr().err


def test_malformed_config_toml_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[x\n")
    assert main(["--config", str(bad), "validate"]) == 1
    assert "invalid TOML" in capsys.readouterr().err
