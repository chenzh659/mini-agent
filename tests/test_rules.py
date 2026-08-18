"""Unit tests for project rules discovery and loading."""

from pathlib import Path

from mini_agent.rules import find_project_rule_file, load_project_rules


def test_find_and_load_agentrules(tmp_path: Path) -> None:
    rule_file = tmp_path / ".agentrules"
    rule_file.write_text("项目规则：必须写类型注解与单元测试。", encoding="utf-8")

    found = find_project_rule_file(tmp_path)
    assert found == rule_file

    loaded = load_project_rules(tmp_path)
    assert loaded is not None
    assert "必须写类型注解与单元测试" in loaded
    assert ".agentrules" in loaded


def test_find_priority_rules(tmp_path: Path) -> None:
    # .agentrules has higher priority than CLAUDE.md
    (tmp_path / "CLAUDE.md").write_text("CLAUDE rules", encoding="utf-8")
    (tmp_path / ".agentrules").write_text("Agent rules", encoding="utf-8")

    found = find_project_rule_file(tmp_path)
    assert found == tmp_path / ".agentrules"


def test_no_rules_returns_none(tmp_path: Path) -> None:
    found = find_project_rule_file(tmp_path)
    assert found is None

    loaded = load_project_rules(tmp_path)
    assert loaded is None
