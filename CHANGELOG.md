# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-13

First tagged release.

### Added
- CI for LLM prompt and model changes: runs a prompt against an 80-email golden set, diffs against the baseline, renders an HTML/Markdown scorecard, comments on the PR and fails on regressions.
- `prompt-gate` GitHub workflow that evaluates every changed prompt file (or the active prompt) and keeps run history even when the gate fails.
- Provider retries honour `Retry-After` without undercutting exponential backoff.
- Mutation testing in CI, Dockerfile (non-root), committed Groq baseline.
