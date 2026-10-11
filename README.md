# regresscheck

[![ci](https://github.com/fernandogarzaaa/regresscheck/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/regresscheck/actions/workflows/ci.yml)

regresscheck is CI for model behaviour. Every pull request that touches a prompt runs our
support-email classifier (category + one-sentence summary) against a hand-labelled golden
dataset of 80 emails, scores each case on four dimensions (category exact match, LLM-as-judge
summary quality 1-5, latency, tokens), diffs the run against the last accepted baseline, and
posts the result on the PR. A drop of more than 8 points in accuracy (or in summary quality)
fails the check; more than 3 points is a warning. A 7-run moving average catches slow drift
that no single PR trips. Runs are kept in SQLite, every check produces a self-contained HTML
diff report, and warnings go to Slack. It runs for $0: CI and tests use a deterministic
built-in mock provider, and any OpenAI-compatible endpoint can be plugged in through env vars.

---

## Current numbers

All numbers below were produced by the code in this repo; nothing is estimated.

### Mock provider (CI default, reproducible with `python scripts/run_demo.py`)

The mock is a deterministic rule-based simulator, not a language model (see
[Architecture decisions](#architecture-decisions)). These numbers prove the pipeline, not a model.
Full tables: [`demo/RESULTS.md`](demo/RESULTS.md).

| Prompt | Accuracy | Summary (1-5) | Tokens / run | Gate vs previous run |
|---|---|---|---|---|
| `v1` (short definitions, no examples) | 66.2% (53/80) | 3.62 | 12,886 | first run |
| `v2` (current, richer definitions + 3 few-shot) | 87.5% (70/80) | 3.89 | 43,494 | **WARN**: +21.2 pts overall, but `general` lost 20 pts (4 regressions hidden under 21 improvements) |
| `demo/prompts/v3-bad` (a "trim the prompt" edit) | 77.5% (62/80) | 3.26 | 23,986 | **CRITICAL**, exit 2: 12 regressions, `technical` 80% -> 25%, summary -15.6% of scale |

Slow-drift demo: seven stable `v2` runs, then seven small prompt trims that each lose exactly
one golden case (-1.25 pts per run, under the 3-point warn line). Every per-run check passes.
The 7-run moving average falls to 83.8%, 3.7 pts under its 87.5% peak, at the sixth trim,
and the drift detector fires. The per-run gate alone would have let accuracy slide from
87.5% to 78.8% without a single alert.

### Real model (Groq free tier, `openai/gpt-oss-20b`, `reasoning_effort=low`)

Baseline run of the active prompt `v2` (80 cases, LLM-as-judge on the same endpoint):
**96.2% accuracy (77/80)**, mean summary score **4.78/5**, p50 latency 250 ms, p95 343 ms
(measured client-side, includes network), 47,038 tokens including the judge, 1 contract
violation. Per category: billing 100%, technical 85%, account 100%, general 100%.
Raw records: `results/groq/baselines/`. The repeat-run, `v3-bad` and `v1` comparisons are being
recorded in `results/groq/`.

### Test suite

- 216 tests, ruff + mypy `--strict` clean, Python 3.10 / 3.11 / 3.12 in CI.
- Mutation score (Fernando's [testteeth](https://pypi.org/project/testteeth/)): **95.6%** (482 of 504 mutants killed) across the ten core modules, up from 87.8% (426/485) before a hardening pass driven by testteeth's surviving-mutant report. The 22 survivors are equivalent or near-equivalent mutants (rounding precision of reported deltas, the `sys.version_info` import guard, `>` vs `>=` in `worst()` where equal ranks are the same status, an `asyncio.sleep(0)` yield). `report.py` and `cli.py` are covered by the end-to-end CLI tests but are not in the mutation scope.

---

## Setup

```bash
git clone https://github.com/fernandogarzaaa/regresscheck && cd regresscheck
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

regresscheck validate                                   # dataset + prompt files parse, coverage stats
regresscheck check                                      # active prompt vs committed baseline, mock provider
regresscheck check --prompt demo/prompts/v3-bad.yaml    # exits 2: critical regression
open reports/*.html
```

Against a real model (any OpenAI-compatible API: OpenAI, Groq, OpenRouter, Ollama, vLLM, llama.cpp):

```bash
export REGRESSCHECK_PROVIDER=openai
export OPENAI_BASE_URL=https://api.groq.com/openai/v1   # default: https://api.openai.com/v1
export OPENAI_API_KEY=...                               # never commit; .env is git-ignored
export MODEL=openai/gpt-oss-20b                         # default: gpt-4o-mini
export OPENAI_EXTRA_BODY='{"reasoning_effort": "low"}'  # optional, merged into every request
export OPENAI_MAX_RETRIES=10                            # optional; 429s honour Retry-After
regresscheck baseline                                   # writes baselines/openai__<model>.json
regresscheck check --concurrency 2
```

With `--provider openai` the judge defaults to the same endpoint (`--judge auto`); pass
`--judge lexical` to keep the deterministic judge.

### Commands

| Command | What it does |
|---|---|
| `regresscheck validate` | Parse the golden dataset and every prompt; print category / difficulty / tag counts. |
| `regresscheck run [--prompt P] [--json out.json]` | One eval run, recorded in SQLite. |
| `regresscheck baseline [--prompt P]` | Run and (over)write the committed baseline for this provider + model. |
| `regresscheck check [--prompt P] [--against baseline\|previous\|RUN_ID]` | Run, diff, drift check, HTML report, optional Markdown + Slack. Exit 0 pass/warn, 2 critical, 1 error. `--fail-on warn\|never` changes the bar. |
| `regresscheck diff RUN_A RUN_B` | Diff two recorded runs without calling a model. |
| `regresscheck history` | Recorded runs for a provider/model with accuracy, summary score and status. |

---

## Changing a prompt

1. Add `prompts/vN.yaml` (or edit an existing file). Required keys: `version`, `created`,
   `system_prompt`; optional `description`, `few_shot` (email/category/summary), `temperature`,
   `max_tokens`. Unknown keys are rejected.
2. Open a PR. The `prompt-gate` workflow evaluates every changed prompt file against the committed
   baseline, comments the scorecard on the PR (one comment, updated in place), uploads the HTML
   reports as a build artifact, and fails on critical.
3. If the change is accepted, point `active_prompt` in `regresscheck.toml` at it and run
   `regresscheck baseline` in the same PR so the bar moves with the merge. A baseline change is a
   reviewed diff in `baselines/`, which is the point.

To make the gate actually block merges, mark the `prompt-gate / eval` check as required in the
branch protection rules for `main` (a repo-admin setting; it is not configured by this repo).

## Adding test cases to the golden dataset

The dataset is `golden/dataset.json`. Evaluation quality is bounded by this file, so the rules are strict:

- **Hand-written and human-labelled only.** Do not generate cases or reference summaries with an
  LLM; the whole value is that the labels are ground truth a person signed off on. The best source
  of new cases is real failures: when a misrouted ticket shows up in production, anonymise it and
  add it here with a note.
- **Fields:** `id` (`bil-`, `tec-`, `acc-`, `gen-` + three digits, never reused or renumbered,
  because diffs are keyed on it), `input`, `expected_category`, `ideal_summary` (one sentence,
  third person, at most 30 words), `expected_difficulty` (`easy` / `medium` / `hard`), `tags`,
  and `notes` explaining why the case exists and, for ambiguous ones, why the label is what it is.
- **Tags in use:** `ambiguous`, `short`, `typos`, `mixed_language`, `sarcastic`, `multi_issue`,
  `long`, `angry`, `polite`. `tests/test_dataset.py` enforces 60-100 cases, at least 15 per
  category, at least 3 of each core edge-case tag, and id/category consistency.
- **Routing policy the labels follow:** money at risk goes to billing even when something also
  broke (`bil-008`); a broken entitlement after a successful payment goes to technical (`tec-019`);
  closing an account is account, cancelling a subscription to stop billing is billing
  (`acc-018` vs `bil-015`); pre-sales and feature requests are general.
- **Bump `version`** (format `YYYY.MM.N`) and `updated` on every change, then re-run
  `regresscheck baseline`. Diffs only compare cases present in both runs and list added/removed
  ids, so a dataset change never shows up as a fake regression.

Current composition: 80 cases, 20 per category; 29 easy / 33 medium / 18 hard; tags: ambiguous 14,
mixed_language 7 (Spanish, Tagalog/Taglish, French, German), short 5, long 5, typos 4, sarcastic 4,
angry 3, multi_issue 2, polite 2.

## Thresholds

Configured in `regresscheck.toml` under `[regresscheck.thresholds]`; every key can be overridden
per run with an environment variable (useful in the Docker image and in CI).

| Key | Default | Env override | Meaning |
|---|---|---|---|
| `warn` | 0.03 | `REGRESSCHECK_WARN_THRESHOLD` | Accuracy drop (absolute) that warns |
| `critical` | 0.08 | `REGRESSCHECK_CRITICAL_THRESHOLD` | Accuracy drop that fails CI |
| `summary_warn` | 0.03 | `REGRESSCHECK_SUMMARY_WARN_THRESHOLD` | Mean judge-score drop, as a fraction of the 1-5 scale (0.03 = 0.12 points) |
| `summary_critical` | 0.08 | `REGRESSCHECK_SUMMARY_CRITICAL_THRESHOLD` | Same, failing |
| `category_warn` | 0.10 | `REGRESSCHECK_CATEGORY_WARN_THRESHOLD` | Any single category losing more than this warns |
| `drift_window` | 7 | `REGRESSCHECK_DRIFT_WINDOW` | Runs in the moving average |
| `drift_max_drop` | 0.03 | `REGRESSCHECK_DRIFT_MAX_DROP` | Moving average this far under its own peak = slow drift (warn) |
| `drift_min_accuracy` | unset | `REGRESSCHECK_DRIFT_MIN_ACCURACY` | Optional absolute floor for the moving average |

"Exceeds" means strictly greater, after rounding deltas to 6 decimals (so `0.92 - 0.89` is not
treated as 3.0000000000000027%). With 80 cases one case is 1.25 points: warn means a net loss of
3+ cases, critical a net loss of 7+.

The report also prints an exact McNemar p-value over the flipped cases. It is informational, not
part of the gate: the v3-bad demo is critical at p = 0.077 (12 regressions vs 4 improvements),
which is exactly the situation where a fixed business threshold is the better decision rule than
"wait until it's statistically significant".

## CI

- `.github/workflows/ci.yml` (every push and PR): ruff, ruff format, mypy `--strict`; pytest on
  3.10 / 3.11 / 3.12 with coverage; a CLI smoke test that asserts the bad demo prompt exits 2;
  a testteeth job that mutation-tests only the functions changed by the push/PR (fails under 85%);
  Docker build plus container smoke test.
- `.github/workflows/prompt-gate.yml` (PRs touching `prompts/`, `golden/` or `regresscheck.toml`,
  plus manual dispatch): `scripts/prompt_gate.sh` evaluates every changed prompt file (or the
  active prompt when only the dataset changed), posts/updates one PR comment, uploads reports,
  restores and saves the SQLite history through `actions/cache` (so drift has a history to look at),
  and fails on critical. It uses the mock provider unless an `OPENAI_API_KEY` repository secret is
  set; `OPENAI_BASE_URL` and `MODEL` come from repository variables, `SLACK_WEBHOOK_URL` from a secret.

## Slack

Create an incoming webhook, export `SLACK_WEBHOOK_URL`, and pass `--slack`. Alerts go out on warn,
critical, or slow drift (`--notify-on-pass` to always send). The message carries the status, the
headline ("12 regressions detected, accuracy dropped from 88% to 78%"), the threshold reasons, and a
link to the report (`--report-url` or `REGRESSCHECK_REPORT_URL`; CI links the Actions run page that
holds the report artifact).

## Docker

```bash
docker build -t regresscheck .
docker run --rm regresscheck validate
docker run --rm -e REGRESSCHECK_PROVIDER=openai -e OPENAI_API_KEY -e OPENAI_BASE_URL -e MODEL \
  -e SLACK_WEBHOOK_URL -e REGRESSCHECK_CRITICAL_THRESHOLD=0.05 \
  -v "$PWD/.regresscheck:/app/.regresscheck" -v "$PWD/reports:/app/reports" regresscheck
```

The image contains the runner, golden dataset, prompts, baselines and reporting layer, runs as a
non-root user, and defaults to `check --report reports/latest.html --slack`.

---

## Architecture

```
prompts/*.yaml ──► loader ──► PromptConfig (pydantic) ─┐
golden/dataset.json ──► GoldenDataset (pydantic) ──────┤
                                                       ▼
                     runner (asyncio, semaphore) ── classifier ── Provider (mock | OpenAI-compatible)
                                │                        └─ parse_output: JSON contract, violations = failures
                                ▼
                     judge (lexical F1 | LLM-as-judge 1-5) ──► CaseResult × N ──► RunRecord + RunSummary
                                                                                      │
                 baselines/*.json (committed) or SQLite "previous" ──► diff.compare ──┤
                                                                       drift.detect ──┤
                                                                                      ▼
                                    store (SQLite) · report (HTML + SVG trend, Markdown) · alerts (Slack) · exit code
```

| Module | Responsibility |
|---|---|
| `contract.py` | Pydantic models: `PromptConfig`, `ClassifierOutput`, `GoldenCase`, `GoldenDataset`. |
| `classifier.py` | The feature under test: builds messages (system + few-shot turns + email), parses and validates output. |
| `providers.py` | `OpenAICompatibleProvider` (httpx, retries, Retry-After) and the deterministic `MockProvider`. |
| `judge.py` | `LLMJudge` (rubric prompt, JSON score) and `LexicalJudge` (content-word F1 bucketed to 1-5). |
| `runner.py` | Async eval over the dataset with bounded concurrency; provider/parse errors become failed cases, never crashes. |
| `records.py` | `CaseResult`, `RunRecord`, aggregate `RunSummary` (accuracy per category/difficulty, p50/p95, tokens). |
| `diff.py` | Run-to-run comparison, thresholds, per-category deltas, flips, McNemar. |
| `drift.py` | Moving-average slow-drift detector. |
| `store.py` | SQLite history (`runs` + `case_results` tables). |
| `report.py`, `alerts.py`, `cli.py` | HTML/Markdown reports, Slack, command line. |

## Architecture decisions

**Slow drift is tracked separately from per-run regressions.** Per-run checks compare a PR with
its predecessor, so a sequence of individually acceptable changes is invisible to them: the drift
demo loses 8.7 points over seven PRs and every one passes. The detector compares the 7-run moving
average with its own historical peak rather than with the previous average, so a slow slide keeps
accumulating instead of resetting each run, and a single noisy run is averaged away instead of
firing an alert. It warns rather than fails, because no individual PR is to blame and blocking the
next unrelated PR would be the wrong response; the right response is a look at the trend chart.

**Compare against a committed baseline by default, not "the previous run".** In CI, "previous"
depends on which runs happened to land in the cache, so two PRs can be judged against different
bars. A JSON baseline in `baselines/` is reviewed in the PR that moves it and is the same for
everyone. `--against previous` is still there (the demo and drift tracking use it), and every run
is in SQLite regardless.

**The gate uses net accuracy and summary deltas with fixed thresholds; flips are listed, not
gated.** Product owners can reason about "8 points of routing accuracy"; they cannot reason about
"p < 0.05 on 80 cases". The flip lists (old vs new output side by side) are what a reviewer reads
to decide whether a warning is acceptable, and the per-category check catches the v1 -> v2 case
where a 21-point overall gain hid a 20-point loss in one queue.

**A deterministic mock provider instead of recorded responses.** Cassettes only replay prompts
that were recorded, so they cannot evaluate a new prompt, which is the one thing this tool exists
to do. The mock reads the system prompt the way an instruction-following model does: category
definitions and few-shot examples become weighted keywords (stated rules weigh 2, words seen only
in examples weigh 1, words claimed by two categories are dropped), and it honours the tie-break
order, the default category, the summary word limit, third-person phrasing, and typo guidance
when the prompt asks for them. Prompt edits therefore change behaviour in the expected direction,
deterministically and offline. Its limits are real: it has no world knowledge, so it misreads
sarcasm and anything not covered by a keyword, and its latency is simulated from token counts.
Mock numbers validate the pipeline; only real-model numbers say anything about a prompt.

**Two judges.** With a real endpoint the summary is graded by an LLM against the reference with
a fixed 1-5 rubric (JSON reply, tolerant parser, failures score 1 and are reported). With the mock,
an LLM judge would be meaningless, so a content-word F1 bucketed to 1-5 stands in. Neither is
calibrated against human ratings yet (see below). No RAGAS / DeepEval: for one rubric on short
summaries they add a large dependency tree for about 40 lines of code.

**Contract violations are failures, not crashes.** Malformed JSON, an unknown category, or a
provider error becomes a failed case with the error recorded, so a prompt that breaks the output
format shows up as a large regression with a reason instead of a red stack trace.

**SQLite + JSON, no service.** One file of history, inspectable with `sqlite3`; baselines and the
dataset are plain JSON in git. Zero infrastructure, which also makes the CI cost $0.

**Few dependencies:** pydantic, PyYAML, httpx (plus tomli on 3.10). The HTML report is a single
file with an inline SVG chart and no JavaScript.

---

## Testing

```bash
ruff check . && ruff format --check . && mypy && pytest -q
testteeth run --workers 2      # mutation test; paths and pytest args in [tool.testteeth]
python scripts/run_demo.py     # regenerates demo/out/ and demo/RESULTS.md
```

The tests are hermetic: the OpenAI-compatible provider is exercised with `httpx.MockTransport`
(success, 408/429/5xx retries, Retry-After, client errors, transport errors, malformed bodies),
Slack with a real local HTTP server, and the CLI end-to-end against the real golden dataset.

## Known limitations and what is untested

- **Mock numbers are not model numbers.** The mock validates the pipeline (diffing, thresholds,
  drift, reports, CI gate). Only the Groq section above measures a real model.
- **One real model, small sample.** Real-model results cover a single model (`openai/gpt-oss-20b` on
  Groq's free tier). Run-to-run noise was measured with one repeat run, which is not enough to
  set thresholds per model with confidence.
- **The LLM judge is uncalibrated and self-grading.** In the Groq runs the same model writes and
  grades summaries, which invites self-preference bias, and no human ratings exist yet to calibrate
  the 1-5 scale. The lexical judge used with the mock is a crude proxy.
- **No held-out split.** `v2` was written from a failure review of `v1` on this same dataset, so its
  score is an in-sample number. A held-out slice is the next dataset change worth making.
- **Docker** is built and smoke-tested only in GitHub Actions (no Docker daemon in the development
  sandbox).
- **Slack** is tested against a local HTTP server that records the webhook payload, not against
  a real Slack workspace.
- **Merge blocking** depends on marking the `prompt-gate / eval` check as required in branch
  protection; this repo does not configure that.
- **Drift in CI** depends on `actions/cache` keeping the SQLite history (caches are branch-scoped
  and evicted after 7 days without use). For durable history, mount `.regresscheck/` somewhere
  persistent (the Docker image supports this) or store the DB as a release asset.
- **Not done from the original brief:** the Loom walkthrough (Phase 6), and RAGAS/DeepEval
  (deliberately skipped, see above). Python 3.13 is not in the CI matrix.

## Repository layout

```
regresscheck/          package (see Architecture)
prompts/               versioned prompts (v1, v2 = active)
golden/dataset.json    80 hand-labelled cases
baselines/             committed baseline per provider + model
demo/                  v3-bad prompt, drift prompt chain, generated demo output and RESULTS.md
results/groq/          real-model runs (history.db, baselines, reports, PR-comment Markdown)
scripts/               run_demo.py, prompt_gate.sh (used by the prompt-gate workflow)
tests/                 pytest suite
```

License: MIT.
