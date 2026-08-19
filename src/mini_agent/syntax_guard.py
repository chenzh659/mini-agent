"""Pre-commit syntax validation guard to prevent code corruption."""

import ast
import json
from pathlib import Path


def validate_syntax(code: str, file_path: str | Path) -> tuple[bool, str | None]:
    """Validate syntax of code before saving to disk.

    Returns:
        tuple[bool, str | None]: (is_valid, error_message_if_any)
    """
    path_obj = Path(file_path)
    ext = path_obj.suffix.lower()

    if ext == ".py":
        try:
            ast.parse(code, filename=path_obj.name)
            return True, None
        except SyntaxError as exc:
            line_no = exc.lineno or 1
            col_offset = exc.offset or 1
            error_line = (exc.text or "").strip()
            msg = (
                f"Python 代码语法错误 (第 {line_no} 行, 第 {col_offset} 列): {exc.msg}\n"
                f"错误行: '{error_line}'\n"
                "修改已被安全拦截，未写入磁盘，请检查缩进、括号或语法并重新生成修改。"
            )
            return False, msg

    if ext == ".json":
        try:
            json.loads(code)
            return True, None
        except json.JSONDecodeError as exc:
            msg = (
                f"JSON 格式错误 (第 {exc.lineno} 行, 第 {exc.colno} 列): {exc.msg}\n"
                "修改已被安全拦截，未写入磁盘。"
            )
            return False, msg

    # Other extensions: default to valid
    return True, None
