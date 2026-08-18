"""Unit tests for session persistence and resume manager."""

from datetime import datetime
from pathlib import Path

from mini_agent.session import (
    SessionData,
    SessionMeta,
    delete_session,
    generate_session_id,
    get_latest_session,
    list_sessions,
    load_session,
    save_session,
)


def test_generate_session_id() -> None:
    sid1 = generate_session_id()
    sid2 = generate_session_id()
    assert len(sid1) > 10
    assert sid1 != sid2


def test_save_and_load_session(tmp_path: Path) -> None:
    sid = "20260818_120000_abc123"
    meta = SessionMeta(
        session_id=sid,
        workspace_root=tmp_path.as_posix(),
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
        model="deepseek-chat",
        title="测试会话",
        turn_count=2,
    )
    history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    session = SessionData(meta=meta, history=history)

    saved_path = save_session(session, sessions_dir=tmp_path)
    assert saved_path.exists()

    loaded = load_session(sid, sessions_dir=tmp_path)
    assert loaded is not None
    assert loaded.meta.session_id == sid
    assert loaded.meta.title == "测试会话"
    assert loaded.meta.turn_count == 2
    assert len(loaded.history) == 3


def test_load_non_existent_session(tmp_path: Path) -> None:
    loaded = load_session("non_existent_id", sessions_dir=tmp_path)
    assert loaded is None


def test_list_sessions_and_filter(tmp_path: Path) -> None:
    ws1 = tmp_path / "ws1"
    ws2 = tmp_path / "ws2"
    ws1.mkdir()
    ws2.mkdir()

    sessions_dir = tmp_path / "sessions"

    # Create session 1 for ws1
    s1 = SessionData(
        meta=SessionMeta(
            session_id="s1",
            workspace_root=ws1.resolve().as_posix(),
            created_at="2026-08-18T10:00:00",
            updated_at="2026-08-18T10:00:00",
            model="gpt-4o",
            title="会话 1",
            turn_count=1,
        )
    )
    save_session(s1, sessions_dir=sessions_dir)

    # Create session 2 for ws2
    s2 = SessionData(
        meta=SessionMeta(
            session_id="s2",
            workspace_root=ws2.resolve().as_posix(),
            created_at="2026-08-18T11:00:00",
            updated_at="2026-08-18T11:00:00",
            model="deepseek-chat",
            title="会话 2",
            turn_count=3,
        )
    )
    save_session(s2, sessions_dir=sessions_dir)

    # List all
    all_sessions = list_sessions(sessions_dir=sessions_dir)
    assert len(all_sessions) == 2
    assert all_sessions[0].session_id == "s2"  # updated later

    # Filter ws1
    ws1_sessions = list_sessions(workspace_root=ws1, sessions_dir=sessions_dir)
    assert len(ws1_sessions) == 1
    assert ws1_sessions[0].session_id == "s1"

    # Filter ws2
    ws2_sessions = list_sessions(workspace_root=ws2, sessions_dir=sessions_dir)
    assert len(ws2_sessions) == 1
    assert ws2_sessions[0].session_id == "s2"


def test_get_latest_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    ws = tmp_path / "my_project"
    ws.mkdir()

    assert get_latest_session(workspace_root=ws, sessions_dir=sessions_dir) is None

    s1 = SessionData(
        meta=SessionMeta(
            session_id="first",
            workspace_root=ws.resolve().as_posix(),
            created_at="2026-08-18T09:00:00",
            updated_at="2026-08-18T09:00:00",
            model="gpt-4o",
            title="旧对话",
            turn_count=1,
        )
    )
    save_session(s1, sessions_dir=sessions_dir)

    s2 = SessionData(
        meta=SessionMeta(
            session_id="second",
            workspace_root=ws.resolve().as_posix(),
            created_at="2026-08-18T12:00:00",
            updated_at="2026-08-18T12:00:00",
            model="gpt-4o",
            title="新对话",
            turn_count=2,
        )
    )
    save_session(s2, sessions_dir=sessions_dir)

    latest = get_latest_session(workspace_root=ws, sessions_dir=sessions_dir)
    assert latest is not None
    assert latest.meta.session_id == "second"
    assert latest.meta.title == "新对话"


def test_delete_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    s = SessionData(
        meta=SessionMeta(
            session_id="to_delete",
            workspace_root=tmp_path.as_posix(),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            model="gpt-4o",
            title="将被删除",
            turn_count=1,
        )
    )
    save_session(s, sessions_dir=sessions_dir)

    assert delete_session("to_delete", sessions_dir=sessions_dir) is True
    assert load_session("to_delete", sessions_dir=sessions_dir) is None
    assert delete_session("to_delete", sessions_dir=sessions_dir) is False


def test_corrupt_session_file_ignored(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = sessions_dir / "bad.json"
    corrupt_file.write_text("invalid json content", encoding="utf-8")

    assert load_session("bad", sessions_dir=sessions_dir) is None
    assert list_sessions(sessions_dir=sessions_dir) == []
