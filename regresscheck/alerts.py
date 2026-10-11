"""Slack incoming-webhook alerts."""

from __future__ import annotations

from typing import Any

import httpx

from .diff import DiffResult, Status
from .drift import DriftResult

SLACK_ICONS = {Status.PASS: ":white_check_mark:", Status.WARN: ":warning:", Status.CRITICAL: ":rotating_light:"}


def build_slack_payload(
    diff: DiffResult, status: Status, drift: DriftResult | None, report_url: str | None
) -> dict[str, Any]:
    headline = (
        f"{len(diff.regressions)} regressions detected, accuracy "
        f"{'dropped' if diff.accuracy_delta < 0 else 'moved'} "
        f"from {diff.accuracy_before:.0%} to {diff.accuracy_after:.0%}"
    )
    title = (
        f"{SLACK_ICONS[status]} regresscheck {status.value.upper()}: {diff.candidate_prompt} vs {diff.baseline_prompt}"
    )
    details = list(diff.reasons)
    if drift is not None and drift.triggered and drift.reason:
        details.append(f"Slow drift: {drift.reason}")
    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{headline}*\nMean summary score {diff.summary_before:.2f} -> {diff.summary_after:.2f}",
            },
        },
    ]
    if details:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(f"- {d}" for d in details)}})
    if report_url:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"<{report_url}|Full diff report>"}})
    return {"text": f"{title}: {headline}", "blocks": blocks}


def should_alert(status: Status, drift: DriftResult | None, notify_on_pass: bool = False) -> bool:
    return notify_on_pass or status is not Status.PASS or bool(drift and drift.triggered)


def send_slack(webhook_url: str, payload: dict[str, Any], timeout: float = 10.0) -> None:
    resp = httpx.post(webhook_url, json=payload, timeout=timeout)
    resp.raise_for_status()
