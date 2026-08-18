"""Unit tests for context window management and compaction."""

import json

from mini_agent.context import compact_history


def test_compact_short_history_untouched() -> None:
    history = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = compact_history(history, max_recent_messages=8)
    assert result == history


def test_compact_long_history_collapses_old_tool_output() -> None:
    system_msg = {"role": "system", "content": "system"}
    old_tool_output = {
        "type": "function_call_output",
        "call_id": "c1",
        "output": json.dumps({"ok": True, "content": "A" * 1000}),
    }
    history = [system_msg, old_tool_output]

    # Add 10 recent messages
    for i in range(10):
        history.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"})

    compacted = compact_history(history, max_recent_messages=8, max_tool_output_chars=100)
    assert len(compacted) == len(history)
    assert compacted[0] == system_msg

    # Old tool output should be compacted
    old_compacted = compacted[1]
    assert old_compacted["type"] == "function_call_output"
    assert "已折叠" in old_compacted["output"]
    assert len(old_compacted["output"]) < 200

    # Recent messages should be untouched
    assert compacted[-1]["content"] == "msg 9"
