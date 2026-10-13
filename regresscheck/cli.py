"""Command-line entry point: `regresscheck <command>`."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .alerts import build_slack_payload, send_slack, should_alert
from .config import ConfigError, Settings, load_settings
from .contract import GoldenDataset, PromptConfig
from .diff import DiffResult, Status, compare, worst
from .drift import DriftResult, detect_drift
from .judge import Judge, LexicalJudge, LLMJudge
from .loader import LoadError, load_dataset, load_prompt
from .providers import Provider, ProviderError, build_provider
from .records import RunRecord
from .report import render_html, render_markdown
from .runner import run_eval
from .store import RunStore

EXIT_OK, EXIT_ERROR, EXIT_CRITICAL = 0, 1, 2


def baseline_path(settings: Settings, provider: str, model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{provider}__{model}")
    return Path(settings.baseline_dir) / f"{safe}.json"


def pick_judge(kind: str, provider: Provider) -> Judge:
    if kind == "lexical" or (kind == "auto" and provider.name == "mock"):
        return LexicalJudge()
    return LLMJudge(provider)


def _execute(prompt: PromptConfig, dataset: GoldenDataset, args: argparse.Namespace, settings: Settings) -> RunRecord:
    provider = build_provider(args.provider)
    judge = pick_judge(args.judge, provider)

    async def go() -> RunRecord:
        try:
            return await run_eval(
                prompt,
                dataset,
                provider,
                judge,
                concurrency=args.concurrency or settings.concurrency,
                track=settings.track,
            )
        finally:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()

    return asyncio.run(go())


def _print_summary(record: RunRecord) -> None:
    s = record.summary
    print(
        f"run {record.run_id}  prompt {record.prompt_version} ({record.prompt_sha})  {record.provider}/{record.model}"
    )
    print(
        f"  accuracy {s.accuracy:.1%} ({sum(c.passed for c in record.cases)}/{s.n})  "
        f"summary {s.mean_summary_score:.2f}/5  p50 {s.latency_p50_ms:.0f}ms  p95 {s.latency_p95_ms:.0f}ms  "
        f"tokens {s.total_prompt_tokens + s.total_completion_tokens}  errors {s.errors}"
    )
    print("  per category: " + "  ".join(f"{k} {v.accuracy:.0%}" for k, v in s.per_category.items() if v.total))
    print("  per difficulty: " + "  ".join(f"{k} {v.accuracy:.0%}" for k, v in s.per_difficulty.items()))


def cmd_validate(args: argparse.Namespace, settings: Settings) -> int:
    dataset = load_dataset(args.dataset or settings.dataset)
    cats = Counter(c.expected_category.value for c in dataset.cases)
    diffs = Counter(c.expected_difficulty.value for c in dataset.cases)
    tags = Counter(t for c in dataset.cases for t in c.tags)
    print(f"dataset {dataset.version}: {len(dataset.cases)} cases")
    print("  categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    print("  difficulty: " + ", ".join(f"{k}={v}" for k, v in sorted(diffs.items())))
    print("  tags: " + ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))
    for path in sorted(Path(settings.prompts_dir).glob("*.yaml")):
        p = load_prompt(path)
        print(f"prompt {path}: version {p.version}, {len(p.few_shot)} few-shot examples")
    return EXIT_OK


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    record = _execute(
        load_prompt(args.prompt or settings.active_prompt),
        load_dataset(args.dataset or settings.dataset),
        args,
        settings,
    )
    _print_summary(record)
    if not args.no_save:
        with RunStore(settings.db_path) as store:
            store.save(record)
    if args.json:
        Path(args.json).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return EXIT_OK


def cmd_baseline(args: argparse.Namespace, settings: Settings) -> int:
    record = _execute(
        load_prompt(args.prompt or settings.active_prompt),
        load_dataset(args.dataset or settings.dataset),
        args,
        settings,
    )
    _print_summary(record)
    path = baseline_path(settings, record.provider, record.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"baseline written to {path}")
    return EXIT_OK


def _resolve_baseline(
    args: argparse.Namespace, settings: Settings, candidate: RunRecord, store: RunStore
) -> RunRecord | None:
    if args.against == "baseline":
        path = baseline_path(settings, candidate.provider, candidate.model)
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None
    if args.against == "previous":
        return store.latest(candidate.track, candidate.provider, candidate.model, exclude=candidate.run_id)
    found = store.get(args.against)
    if found is None:
        raise LoadError(f"run {args.against!r} not found in {settings.db_path}")
    return found


def _finalise(
    diff: DiffResult,
    baseline: RunRecord,
    candidate: RunRecord,
    store: RunStore,
    settings: Settings,
    args: argparse.Namespace,
) -> tuple[Status, DriftResult]:
    history = store.history(candidate.track, candidate.provider, candidate.model, limit=args.history)
    drift = detect_drift([h.accuracy for h in history], settings.thresholds)
    status = worst(diff.status, Status.WARN if drift.triggered else Status.PASS)
    store.set_status(candidate.run_id, status.value)
    history = store.history(candidate.track, candidate.provider, candidate.model, limit=args.history)

    report_url = args.report_url or os.environ.get("REGRESSCHECK_REPORT_URL") or None
    report = Path(args.report or Path(settings.report_dir) / f"{candidate.run_id}.html")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_html(diff, baseline, candidate, history, drift, status), encoding="utf-8")
    markdown = render_markdown(diff, status, drift, report_url)
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")

    print(f"{status.value.upper()}: {diff.headline()}")
    for reason in diff.reasons:
        print(f"  - {reason}")
    if drift.triggered:
        print(f"  - slow drift: {drift.reason}")
    print(f"  report: {report}")

    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if args.slack and webhook and should_alert(status, drift, args.notify_on_pass):
        send_slack(webhook, build_slack_payload(diff, status, drift, report_url))
        print("  slack alert sent")
    elif args.slack and not webhook:
        print("  slack: SLACK_WEBHOOK_URL not set, skipped")
    return status, drift


def _exit_for(status: Status, fail_on: str) -> int:
    if fail_on == "never":
        return EXIT_OK
    threshold = Status.WARN if fail_on == "warn" else Status.CRITICAL
    return EXIT_CRITICAL if status.rank >= threshold.rank else EXIT_OK


def cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    candidate = _execute(
        load_prompt(args.prompt or settings.active_prompt),
        load_dataset(args.dataset or settings.dataset),
        args,
        settings,
    )
    _print_summary(candidate)
    with RunStore(settings.db_path) as store:
        store.save(candidate)
        baseline = _resolve_baseline(args, settings, candidate, store)
        if baseline is None:
            store.set_status(candidate.run_id, Status.PASS.value)
            print(
                f"no baseline to compare against ({args.against}); recorded run {candidate.run_id} as the first point"
            )
            return EXIT_OK
        diff = compare(baseline, candidate, settings.thresholds)
        status, _ = _finalise(diff, baseline, candidate, store, settings, args)
    return _exit_for(status, args.fail_on)


def cmd_diff(args: argparse.Namespace, settings: Settings) -> int:
    with RunStore(settings.db_path) as store:
        baseline, candidate = store.get(args.baseline), store.get(args.candidate)
        if baseline is None or candidate is None:
            missing = args.baseline if baseline is None else args.candidate
            raise LoadError(f"run {missing!r} not found in {settings.db_path}")
        diff = compare(baseline, candidate, settings.thresholds)
        status, _ = _finalise(diff, baseline, candidate, store, settings, args)
    return _exit_for(status, args.fail_on)


def cmd_history(args: argparse.Namespace, settings: Settings) -> int:
    with RunStore(settings.db_path) as store:
        points = store.history(settings.track, args.provider_name, args.model, limit=args.limit)
    if not points:
        print("no runs recorded")
        return EXIT_OK
    for p in points:
        print(
            f"{p.created_at}  {p.run_id}  {p.prompt_version:<16} acc {p.accuracy:6.1%}  "
            f"summary {p.mean_summary_score:.2f}  {p.status or '-'}"
        )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regresscheck", description="CI for LLM prompt and model changes.")
    parser.add_argument("--version", action="version", version=f"regresscheck {__version__}")
    parser.add_argument("--config", default=None, help="settings file (default: regresscheck.toml, if present)")
    sub = parser.add_subparsers(dest="command", required=True)

    def eval_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--prompt", help="prompt YAML (default: active_prompt from config)")
        p.add_argument("--dataset", help="golden dataset JSON (default: from config)")
        p.add_argument(
            "--provider", choices=["mock", "openai"], default=os.environ.get("REGRESSCHECK_PROVIDER", "mock")
        )
        p.add_argument(
            "--judge",
            choices=["auto", "lexical", "llm"],
            default="auto",
            help="auto: lexical for the mock provider, LLM-as-judge otherwise",
        )
        p.add_argument("--concurrency", type=int, default=0)

    def report_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--report", help="HTML report path (default: reports/<run_id>.html)")
        p.add_argument("--markdown", help="also write a Markdown summary (for PR comments)")
        p.add_argument("--report-url", help="link to the hosted report, used in Slack and Markdown")
        p.add_argument("--slack", action="store_true", help="post to SLACK_WEBHOOK_URL on warn/critical/drift")
        p.add_argument("--notify-on-pass", action="store_true")
        p.add_argument("--history", type=int, default=30, help="runs shown in the trend chart and used for drift")
        p.add_argument("--fail-on", choices=["critical", "warn", "never"], default="critical")

    p = sub.add_parser("validate", help="validate the golden dataset and every prompt file")
    p.add_argument("--dataset")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run one eval and record it")
    eval_opts(p)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--json", help="write the full run record as JSON")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("check", help="run, diff against a baseline, report, alert; exit 2 on critical")
    eval_opts(p)
    report_opts(p)
    p.add_argument(
        "--against", default="baseline", help="'baseline' (committed file), 'previous' (last run), or a run ID"
    )
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("baseline", help="run the eval and write the committed baseline file")
    eval_opts(p)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("diff", help="diff two recorded runs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    report_opts(p)
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("history", help="list recorded runs")
    p.add_argument("--provider", dest="provider_name", default="mock")
    p.add_argument("--model", default="mock-classifier-v1")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_history)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.config is not None and not Path(args.config).is_file():
            raise ConfigError(f"config file not found: {args.config}")
        settings = load_settings(args.config if args.config is not None else "regresscheck.toml")
        result: int = args.func(args, settings)
        return result
    except (ConfigError, LoadError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
