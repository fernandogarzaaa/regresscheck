"""SQLite run history. One row per run plus one row per case for ad-hoc SQL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel

from .records import RunRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    track TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_sha TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    judge TEXT NOT NULL,
    accuracy REAL NOT NULL,
    mean_summary_score REAL NOT NULL,
    latency_p50_ms REAL NOT NULL,
    latency_p95_ms REAL NOT NULL,
    total_tokens INTEGER NOT NULL,
    errors INTEGER NOT NULL,
    status TEXT,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    expected TEXT NOT NULL,
    predicted TEXT,
    passed INTEGER NOT NULL,
    summary_score INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    tokens INTEGER NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE INDEX IF NOT EXISTS runs_by_track ON runs(track, provider, model, seq);
"""


class HistoryPoint(BaseModel):
    run_id: str
    created_at: str
    prompt_version: str
    accuracy: float
    mean_summary_score: float
    status: str | None


class RunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save(self, record: RunRecord, status: str | None = None) -> None:
        s = record.summary
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (run_id, created_at, track, prompt_version, prompt_sha, dataset_version, provider,"
                " model, judge, accuracy, mean_summary_score, latency_p50_ms, latency_p95_ms, total_tokens, errors,"
                " status, record_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.run_id,
                    record.created_at,
                    record.track,
                    record.prompt_version,
                    record.prompt_sha,
                    record.dataset_version,
                    record.provider,
                    record.model,
                    record.judge,
                    s.accuracy,
                    s.mean_summary_score,
                    s.latency_p50_ms,
                    s.latency_p95_ms,
                    s.total_prompt_tokens + s.total_completion_tokens + s.total_judge_tokens,
                    s.errors,
                    status,
                    record.model_dump_json(),
                ),
            )
            self.conn.executemany(
                "INSERT INTO case_results VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        record.run_id,
                        c.case_id,
                        c.expected,
                        c.predicted,
                        int(c.passed),
                        c.summary_score,
                        c.latency_ms,
                        c.prompt_tokens + c.completion_tokens,
                    )
                    for c in record.cases
                ],
            )

    def set_status(self, run_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def get(self, run_id: str) -> RunRecord | None:
        row = self.conn.execute("SELECT record_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def latest(self, track: str, provider: str, model: str, *, exclude: str | None = None) -> RunRecord | None:
        row = self.conn.execute(
            "SELECT record_json FROM runs WHERE track = ? AND provider = ? AND model = ? AND run_id != ?"
            " ORDER BY seq DESC LIMIT 1",
            (track, provider, model, exclude or ""),
        ).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def history(self, track: str, provider: str, model: str, limit: int = 50) -> list[HistoryPoint]:
        rows = self.conn.execute(
            "SELECT run_id, created_at, prompt_version, accuracy, mean_summary_score, status FROM runs"
            " WHERE track = ? AND provider = ? AND model = ? ORDER BY seq DESC LIMIT ?",
            (track, provider, model, limit),
        ).fetchall()
        points = [
            HistoryPoint(
                run_id=r[0], created_at=r[1], prompt_version=r[2], accuracy=r[3], mean_summary_score=r[4], status=r[5]
            )
            for r in rows
        ]
        return list(reversed(points))
