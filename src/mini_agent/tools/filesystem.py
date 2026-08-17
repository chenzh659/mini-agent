"""Safe filesystem tools for mini-agent."""

import os
from pathlib import Path
from typing import Any

from mini_agent.models import ListFilesInput, ReadFileInput, ToolResult

IGNORED_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
}


def truncate_text(text: str, max_chars: int = 12_000) -> tuple[str, bool]:
    """Truncate text if it exceeds max_chars, keeping head and tail with a marker."""
    if len(text) <= max_chars:
        return text, False

    if max_chars < 50:
        return text[:max_chars] + "...", True

    half = (max_chars - 50) // 2
    omitted = len(text) - (half * 2)
    truncated = f"{text[:half]}\n... [已省略 {omitted} 字符] ...\n{text[-half:]}"
    return truncated, True


def resolve_relative_path(
    workspace_root: Path, user_path: str | Path
) -> tuple[Path | None, str | None]:
    """Safely resolve user-provided relative path within workspace_root.

    Returns:
        tuple[Path | None, str | None]: (resolved_path, error_message)
    """
    raw_str = str(user_path).strip()
    if not raw_str:
        return None, "路径不能为空"

    # Reject absolute paths explicitly
    path_obj = Path(raw_str)
    if path_obj.is_absolute() or raw_str.startswith("/") or raw_str.startswith("\\"):
        return None, f"非法绝对路径: '{raw_str}'。文件工具只允许访问工作区内的相对路径。"

    resolved_root = workspace_root.resolve()
    candidate = resolved_root / path_obj

    # Check for path traversal escaping workspace
    try:
        resolved_path = candidate.resolve()
    except OSError as exc:
        return None, f"路径解析失败: '{raw_str}' ({exc})"

    # Must be workspace root or inside workspace root
    if resolved_path != resolved_root and not resolved_path.is_relative_to(resolved_root):
        return None, f"路径越界被拒绝: '{raw_str}' 指向工作区外部。"

    # Check symlinks for symlink escape
    if candidate.is_symlink():
        try:
            target = candidate.resolve()
            if target != resolved_root and not target.is_relative_to(resolved_root):
                return None, f"符号链接越界被拒绝: '{raw_str}' 指向工作区外部。"
        except OSError as exc:
            return None, f"无法解析符号链接: '{raw_str}' ({exc})"

    return resolved_path, None


def read_file(
    input_data: ReadFileInput,
    workspace_root: Path,
    max_file_bytes: int = 100 * 1024,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """Read UTF-8 content of a file within workspace."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )

    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"文件不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    if not resolved_path.is_file():
        return ToolResult(
            ok=False,
            content="",
            error=f"路径不是普通文件: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    try:
        size = resolved_path.stat().st_size
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"无法获取文件状态: '{input_data.path}' ({exc})",
            metadata={"path": input_data.path},
        )

    if size > max_file_bytes:
        return ToolResult(
            ok=False,
            content="",
            error=(
                f"文件体积过大 ({size} 字节，上限 {max_file_bytes} 字节): '{input_data.path}'，"
                "请指定更小文件。"
            ),
            metadata={"path": input_data.path, "size_bytes": size, "truncated": False},
        )

    try:
        with open(resolved_path, encoding="utf-8", errors="strict") as f:
            raw_content = f.read()
    except UnicodeDecodeError:
        return ToolResult(
            ok=False,
            content="",
            error=f"文件不是 UTF-8 编码文本文件: '{input_data.path}'",
            metadata={"path": input_data.path, "size_bytes": size},
        )
    except OSError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=f"无法读取文件 '{input_data.path}': {exc}",
            metadata={"path": input_data.path, "size_bytes": size},
        )

    content, truncated = truncate_text(raw_content, max_chars=max_output_chars)
    return ToolResult(
        ok=True,
        content=content,
        error=None,
        metadata={"path": input_data.path, "size_bytes": size, "truncated": truncated},
    )


def list_files(
    input_data: ListFilesInput,
    workspace_root: Path,
    max_entries: int = 500,
    max_output_chars: int = 12_000,
) -> ToolResult:
    """List directory contents within workspace up to max_depth."""
    resolved_path, error = resolve_relative_path(workspace_root, input_data.path)
    if error or resolved_path is None:
        return ToolResult(
            ok=False,
            content="",
            error=error,
            metadata={"path": input_data.path},
        )

    if not resolved_path.exists():
        return ToolResult(
            ok=False,
            content="",
            error=f"目录不存在: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    if not resolved_path.is_dir():
        return ToolResult(
            ok=False,
            content="",
            error=f"路径不是目录: '{input_data.path}'",
            metadata={"path": input_data.path},
        )

    resolved_root = workspace_root.resolve()
    max_depth = max(1, min(input_data.max_depth, 5))
    entries: list[str] = []
    skipped_count = 0
    reached_limit = False

    def _traverse(current_dir: Path, current_depth: int) -> None:
        nonlocal skipped_count, reached_limit
        if current_depth > max_depth or reached_limit:
            return

        try:
            items = sorted(os.scandir(current_dir), key=lambda e: (not e.is_dir(), e.name.lower()))
        except (PermissionError, OSError):
            skipped_count += 1
            return

        for item in items:
            if len(entries) >= max_entries:
                reached_limit = True
                return

            name = item.name
            if name in IGNORED_NAMES:
                continue

            try:
                is_dir = item.is_dir(follow_symlinks=False)
                is_symlink = item.is_symlink()
            except OSError:
                skipped_count += 1
                continue

            # Check for symlinks pointing outside
            if is_symlink:
                try:
                    target = Path(item.path).resolve()
                    if target != resolved_root and not target.is_relative_to(resolved_root):
                        skipped_count += 1
                        continue
                except OSError:
                    skipped_count += 1
                    continue

            # Compute relative path to workspace root
            rel_to_root = Path(item.path).relative_to(resolved_root).as_posix()
            display_path = f"{rel_to_root}/" if is_dir else rel_to_root
            entries.append(display_path)

            if is_dir and not is_symlink:
                _traverse(Path(item.path), current_depth + 1)

    _traverse(resolved_path, current_depth=1)

    metadata: dict[str, Any] = {
        "path": input_data.path,
        "max_depth": max_depth,
        "total_entries": len(entries),
        "skipped_inaccessible": skipped_count,
        "reached_limit": reached_limit,
    }

    if not entries:
        content = "(目录为空)"
    else:
        lines = [f"- {e}" for e in entries]
        if reached_limit:
            lines.append(f"\n[已达到最大条目上限 {max_entries} 项]")
        if skipped_count > 0:
            lines.append(f"[跳过 {skipped_count} 个无法访问或越界条目]")
        content = "\n".join(lines)

    content, truncated = truncate_text(content, max_chars=max_output_chars)
    metadata["truncated"] = truncated

    return ToolResult(
        ok=True,
        content=content,
        error=None,
        metadata=metadata,
    )
