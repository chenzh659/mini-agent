"""Unit tests for pre-commit syntax validation guard."""

from mini_agent.syntax_guard import validate_syntax


def test_valid_python_syntax() -> None:
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    is_valid, err = validate_syntax(code, "test.py")
    assert is_valid is True
    assert err is None


def test_invalid_python_syntax_missing_colon() -> None:
    code = "def add(a, b)\n    return a + b\n"
    is_valid, err = validate_syntax(code, "test.py")
    assert is_valid is False
    assert err is not None
    assert "语法错误" in err


def test_invalid_python_indentation_error() -> None:
    code = "def add(a, b):\nreturn a + b\n"
    is_valid, err = validate_syntax(code, "test.py")
    assert is_valid is False
    assert err is not None


def test_valid_json_syntax() -> None:
    code = '{"name": "mini-agent", "version": "0.2.0"}'
    is_valid, err = validate_syntax(code, "config.json")
    assert is_valid is True
    assert err is None


def test_invalid_json_syntax() -> None:
    code = '{"name": "mini-agent", "version": }'
    is_valid, err = validate_syntax(code, "config.json")
    assert is_valid is False
    assert err is not None
    assert "JSON 格式错误" in err
