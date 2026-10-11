from pathlib import Path

from regresscheck.store import RunStore

from .conftest import make_case, make_run


def test_save_get_latest_history(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "h.db"
    with RunStore(db) as store:
        a = make_run([make_case("bil-001")], "a", "v1")
        b = make_run([make_case("bil-001", "billing", "general")], "b", "v2")
        other = make_run([make_case("bil-001")], "c", "v3").model_copy(update={"model": "other"})
        store.save(a, "pass")
        store.save(b)
        store.save(other)
        assert store.get("a") == a
        assert store.get("zzz") is None
        latest = store.latest("default", "mock", "m")
        assert latest is not None and latest.run_id == "b"
        excluded = store.latest("default", "mock", "m", exclude="b")
        assert excluded is not None and excluded.run_id == "a"
        assert store.latest("nope", "mock", "m") is None
        store.set_status("b", "critical")
        hist = store.history("default", "mock", "m")
        assert [h.run_id for h in hist] == ["a", "b"]
        assert [h.status for h in hist] == ["pass", "critical"]
        assert [h.accuracy for h in hist] == [1.0, 0.0]
        assert hist[0].prompt_version == "v1" and hist[0].mean_summary_score == 4.0
        assert hist[0].created_at == a.created_at
        assert [h.run_id for h in store.history("default", "mock", "m", limit=1)] == ["b"]
        rows = store.conn.execute("SELECT case_id, passed, tokens FROM case_results WHERE run_id='b'").fetchall()
        assert rows == [("bil-001", 0, 15)]
        total = store.conn.execute("SELECT total_tokens, errors FROM runs WHERE run_id='a'").fetchone()
        assert total == (15, 0)
    assert db.is_file()
    # persisted across connections
    with RunStore(db) as again:
        assert again.get("b") is not None


def test_in_memory_store() -> None:
    store = RunStore(":memory:")
    store.save(make_run([make_case("bil-001")], "x"))
    assert store.get("x") is not None
    store.close()


def test_deep_paths_close_and_token_totals(tmp_path: Path) -> None:
    import sqlite3

    import pytest

    db = tmp_path / "a" / "b" / "h.db"
    with RunStore(db) as store:
        case = make_case("bil-001").model_copy(update={"judge_tokens": 100})
        store.save(make_run([case], "j"))
        assert store.conn.execute("SELECT total_tokens FROM runs").fetchone() == (115,)
        conn = store.conn
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    store = RunStore(":memory:")
    for i in range(51):
        store.save(make_run([make_case("bil-001")], f"r{i}"))
    assert len(store.history("default", "mock", "m")) == 50
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.conn.execute("SELECT 1")
