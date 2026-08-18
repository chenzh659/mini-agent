"""Project rules discovery and loader (.agentrules, MINI_AGENT.md, CLAUDE.md)."""

from pathlib import Path

RULE_FILE_CANDIDATES = [
    ".agentrules",
    "MINI_AGENT.md",
    ".mini_agent.md",
    "CLAUDE.md",
    ".cursorrules",
]


def find_project_rule_file(workspace_root: Path) -> Path | None:
    """Search for the highest priority project rules file in workspace root."""
    resolved_root = workspace_root.resolve()
    for filename in RULE_FILE_CANDIDATES:
        rule_path = resolved_root / filename
        if rule_path.is_file():
            return rule_path
    return None


def load_project_rules(workspace_root: Path, max_bytes: int = 16 * 1024) -> str | None:
    """Load project rules content from workspace root if available."""
    rule_file = find_project_rule_file(workspace_root)
    if not rule_file:
        return None

    try:
        size = rule_file.stat().st_size
        if size > max_bytes:
            # Read first max_bytes to prevent prompt bloat
            with open(rule_file, encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
            content += f"\n... [规则文件较大，已截断前 {max_bytes} 字节] ..."
        else:
            with open(rule_file, encoding="utf-8", errors="ignore") as f:
                content = f.read()

        stripped = content.strip()
        if not stripped:
            return None

        return f"\n\n## 当前项目自定义规则与开发规范 (来自 {rule_file.name})\n{stripped}"
    except Exception:
        return None
