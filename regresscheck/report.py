"""HTML diff report (self-contained, inline SVG chart) and Markdown summary for PR comments."""

from __future__ import annotations

import math
from html import escape

from .diff import CaseFlip, DiffResult, Status
from .drift import DriftResult
from .records import RunRecord
from .store import HistoryPoint

STATUS_COLOURS = {Status.PASS: "#1a7f37", Status.WARN: "#9a6700", Status.CRITICAL: "#cf222e"}
STATUS_ICONS = {Status.PASS: "\u2705", Status.WARN: "\u26a0\ufe0f", Status.CRITICAL: "\u274c"}


def trend_svg(points: list[HistoryPoint], moving: list[float | None], width: int = 720, height: int = 220) -> str:
    """Accuracy per run (solid) and its moving average (dashed). Y axis from just below the lowest point to 100%."""
    if not points:
        return "<p class='muted'>No run history yet.</p>"
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 28
    inner_w, inner_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(points)
    lowest = min([p.accuracy for p in points] + [v for v in moving if v is not None])
    floor = max(0.0, math.floor((lowest - 0.05) * 10) / 10)
    span = 1.0 - floor or 1.0

    def x(i: int) -> float:
        return pad_l + (inner_w * i / (n - 1) if n > 1 else inner_w / 2)

    def y(v: float) -> float:
        return pad_t + inner_h * (1 - (v - floor) / span)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="accuracy trend">']
    for tick in (floor + span * k / 4 for k in range(5)):
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#eee"/>')
        parts.append(
            f'<text x="{pad_l - 6}" y="{y(tick) + 4:.1f}" font-size="11" text-anchor="end" fill="#666">'
            f"{round(tick * 100, 1):g}%</text>"
        )
    acc = " ".join(f"{x(i):.1f},{y(p.accuracy):.1f}" for i, p in enumerate(points))
    parts.append(f'<polyline fill="none" stroke="#0969da" stroke-width="2" points="{acc}"/>')
    ma = [(i, v) for i, v in enumerate(moving[-n:]) if v is not None]
    if len(ma) > 1:
        line = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in ma)
        parts.append(
            f'<polyline fill="none" stroke="#8250df" stroke-width="2" stroke-dasharray="5 4" points="{line}"/>'
        )
    for i, p in enumerate(points):
        parts.append(
            f'<circle cx="{x(i):.1f}" cy="{y(p.accuracy):.1f}" r="3" fill="#0969da">'
            f"<title>{escape(p.prompt_version)} {p.accuracy:.1%} ({escape(p.run_id)})</title></circle>"
        )
        if n <= 16 or i % max(1, n // 8) == 0:
            parts.append(
                f'<text x="{x(i):.1f}" y="{height - 8}" font-size="10" text-anchor="middle" fill="#666">'
                f"{escape(p.prompt_version)}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _flip_rows(flips: list[CaseFlip]) -> str:
    if not flips:
        return "<p class='muted'>None.</p>"
    rows = []
    for f in flips:
        rows.append(
            "<tr>"
            f"<td><code>{escape(f.case_id)}</code><br><span class='muted'>{escape(f.difficulty)}"
            f"{' · ' + escape(', '.join(f.tags)) if f.tags else ''}</span></td>"
            f"<td>{escape(f.expected)}</td>"
            f"<td><b>{escape(str(f.old_predicted))}</b><br>{escape(f.old_summary or '')} "
            f"<span class='muted'>[{f.old_summary_score}/5]</span></td>"
            f"<td><b>{escape(str(f.new_predicted))}</b><br>{escape(f.new_summary or '')} "
            f"<span class='muted'>[{f.new_summary_score}/5]</span></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Case</th><th>Expected</th><th>Baseline output</th><th>Candidate output</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def render_html(
    diff: DiffResult,
    baseline: RunRecord,
    candidate: RunRecord,
    history: list[HistoryPoint],
    drift: DriftResult | None,
    status: Status,
) -> str:
    b, c = baseline.summary, candidate.summary

    def row(label: str, before: str, after: str, delta: str) -> str:
        return f"<tr><td>{label}</td><td>{before}</td><td>{after}</td><td>{delta}</td></tr>"

    b_tok = b.total_prompt_tokens + b.total_completion_tokens
    c_tok = c.total_prompt_tokens + c.total_completion_tokens
    scorecard = "".join(
        [
            row(
                "Category accuracy",
                f"{diff.accuracy_before:.1%}",
                f"{diff.accuracy_after:.1%}",
                f"{diff.accuracy_delta:+.1%}",
            ),
            row(
                "Mean summary score (1-5)",
                f"{diff.summary_before:.2f}",
                f"{diff.summary_after:.2f}",
                f"{diff.summary_delta:+.2f}",
            ),
            row(
                "Latency p50 (ms)",
                f"{b.latency_p50_ms:.0f}",
                f"{c.latency_p50_ms:.0f}",
                f"{c.latency_p50_ms - b.latency_p50_ms:+.0f}",
            ),
            row(
                "Latency p95 (ms)",
                f"{b.latency_p95_ms:.0f}",
                f"{c.latency_p95_ms:.0f}",
                f"{c.latency_p95_ms - b.latency_p95_ms:+.0f}",
            ),
            row("Feature tokens (prompt+completion)", f"{b_tok}", f"{c_tok}", f"{c_tok - b_tok:+d}"),
            row("Errors / contract violations", f"{b.errors}", f"{c.errors}", f"{c.errors - b.errors:+d}"),
        ]
    )
    cats = "".join(
        f"<tr><td>{escape(cd.category)}</td><td>{cd.before:.1%}</td><td>{cd.after:.1%}</td>"
        f"<td class='{'neg' if cd.delta < 0 else 'pos' if cd.delta > 0 else ''}'>{cd.delta:+.1%}</td><td>{cd.total}</td></tr>"
        for cd in diff.per_category
    )
    reasons = "".join(f"<li>{escape(r)}</li>" for r in diff.reasons) or "<li>No threshold exceeded.</li>"
    drift_html = "<p class='muted'>Not enough history for drift detection.</p>"
    if drift is not None and drift.moving_average is not None:
        verdict = escape(drift.reason or "") if drift.triggered else "No slow drift."
        drift_html = (
            f"<p><b>{'SLOW DRIFT' if drift.triggered else 'OK'}</b>: {verdict} "
            f"<span class='muted'>({drift.window}-run MA {drift.moving_average:.1%}"
            f"{f', peak {drift.peak:.1%}' if drift.peak is not None else ''})</span></p>"
        )
    colour = STATUS_COLOURS[status]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>regresscheck: {escape(candidate.prompt_version)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}} td,th{{border:1px solid #d0d7de;padding:6px 8px;vertical-align:top;text-align:left}}
th{{background:#f6f8fa}} .muted{{color:#656d76;font-size:12px}} .neg{{color:#cf222e}} .pos{{color:#1a7f37}}
.banner{{padding:10px 14px;border-radius:6px;color:#fff;background:{colour};font-weight:600}} code{{font-size:12px}}
</style></head><body>
<h1>regresscheck diff report</h1>
<div class="banner">{status.value.upper()} &mdash; {escape(diff.headline())}</div>
<ul>{reasons}</ul>
<h2>Run metadata</h2>
<table><thead><tr><th></th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>
<tr><td>Run ID</td><td><code>{escape(baseline.run_id)}</code></td><td><code>{escape(candidate.run_id)}</code></td></tr>
<tr><td>Prompt version</td><td>{escape(baseline.prompt_version)} <span class="muted">{escape(baseline.prompt_sha)}</span></td>
<td>{escape(candidate.prompt_version)} <span class="muted">{escape(candidate.prompt_sha)}</span></td></tr>
<tr><td>Provider / model</td><td>{escape(baseline.provider)} / {escape(baseline.model)}</td><td>{escape(candidate.provider)} / {escape(candidate.model)}</td></tr>
<tr><td>Judge</td><td>{escape(baseline.judge)}</td><td>{escape(candidate.judge)}</td></tr>
<tr><td>Dataset</td><td>{escape(baseline.dataset_version)}</td><td>{escape(candidate.dataset_version)}</td></tr>
<tr><td>Timestamp (UTC)</td><td>{escape(baseline.created_at)}</td><td>{escape(candidate.created_at)}</td></tr>
</tbody></table>
<p class="muted">Compared {diff.compared_cases} shared cases. Added: {len(diff.added_cases)}, removed: {len(diff.removed_cases)}.
Exact McNemar p = {diff.mcnemar_p:.3f} on {len(diff.regressions) + len(diff.improvements)} flipped cases.</p>
<h2>Scorecard</h2>
<table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead><tbody>{scorecard}</tbody></table>
<h2>Per-category accuracy</h2>
<table><thead><tr><th>Category</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Cases</th></tr></thead><tbody>{cats}</tbody></table>
<h2>Regressions ({len(diff.regressions)}): passed before, fail now</h2>{_flip_rows(diff.regressions)}
<h2>Improvements ({len(diff.improvements)})</h2>{_flip_rows(diff.improvements)}
<h2>Summary-score drops of 2+ points ({len(diff.summary_drops)})</h2>{_flip_rows(diff.summary_drops)}
<h2>Trend (last {len(history)} runs)</h2>
{trend_svg(history, drift.series if drift else [])}
<p class="muted">Solid: accuracy per run. Dashed: moving average used for drift detection.</p>
<h2>Slow drift</h2>{drift_html}
</body></html>
"""


def render_markdown(
    diff: DiffResult, status: Status, drift: DriftResult | None, report_url: str | None = None, max_rows: int = 10
) -> str:
    lines = [
        "<!-- regresscheck -->",
        f"### {STATUS_ICONS[status]} regresscheck: **{status.value.upper()}** for `{diff.candidate_prompt}`",
        "",
        f"Baseline `{diff.baseline_prompt}` -> candidate `{diff.candidate_prompt}` on {diff.compared_cases} golden cases.",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---|---|---|",
        f"| Category accuracy | {diff.accuracy_before:.1%} | {diff.accuracy_after:.1%} | {diff.accuracy_delta:+.1%} |",
        f"| Mean summary score (1-5) | {diff.summary_before:.2f} | {diff.summary_after:.2f} | {diff.summary_delta:+.2f} |",
        f"| Regressions / improvements | | | {len(diff.regressions)} / {len(diff.improvements)} |",
        "",
    ]
    lines += [f"- {r}" for r in diff.reasons] or ["- No threshold exceeded."]
    if drift is not None and drift.triggered:
        lines.append(f"- Slow drift: {drift.reason}")
    cat_moves = [cd for cd in diff.per_category if cd.delta != 0]
    if cat_moves:
        lines += ["", "Per-category: " + ", ".join(f"{cd.category} {cd.delta:+.1%}" for cd in cat_moves)]
    if diff.regressions:
        lines += [
            "",
            "<details><summary>Regressed cases</summary>",
            "",
            "| Case | Expected | Was | Now |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| `{f.case_id}` | {f.expected} | {f.old_predicted} | {f.new_predicted} |"
            for f in diff.regressions[:max_rows]
        ]
        if len(diff.regressions) > max_rows:
            lines.append(f"| ... | {len(diff.regressions) - max_rows} more | | |")
        lines += ["", "</details>"]
    if report_url:
        lines += ["", f"Full HTML diff report: {report_url}"]
    return "\n".join(lines) + "\n"
