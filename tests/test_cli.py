"""Unit tests for Typer CLI and REPL interface."""

from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from mini_agent.cli import (
    RichAgentEventListener,
    app,
    render_banner,
    render_help,
    run_cli,
)
from mini_agent.llm import LLMClient, LLMResponse
from mini_agent.models import ToolResult

runner = CliRunner()


class DummyLLM(LLMClient):
    """Dummy LLM for CLI testing."""

    def __init__(self, answer: str = "这是回答") -> None:
        self.answer = answer

    def create_response(
        self,
        history: list[dict[str, object]],
        tools: list[dict[str, object]],
        model: str = "gpt-4o-mini",
    ) -> LLMResponse:
        return LLMResponse(text=self.answer)


class TestCliCommands:
    """Test CLI options, arguments, and validation."""

    def test_cli_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--workspace" in result.stdout
        assert "--model" in result.stdout
        assert "--base-url" in result.stdout
        assert "--verbose" in result.stdout

    def test_cli_invalid_workspace(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "not_found_dir"
        result = runner.invoke(app, ["--workspace", str(non_existent)])
        assert result.exit_code != 0
        assert "不存在" in result.stdout

    def test_cli_missing_api_key_exits_non_zero(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["--workspace", str(tmp_path)])
            assert result.exit_code != 0
            assert "OPENAI_API_KEY" in result.stdout

    def test_cli_render_help_content(self) -> None:
        test_console = Console(record=True)
        render_help(test_console)
        out = test_console.export_text()
        assert "/help" in out
        assert "/exit" in out
        assert "/clear" in out
        assert "/model" in out
        assert "list_files" in out
        assert "read_file" in out
        assert "run_shell" in out

    def test_cli_render_banner(self, tmp_path: Path) -> None:
        test_console = Console(record=True)
        render_banner(test_console, tmp_path, "deepseek-chat")
        out = test_console.export_text()
        assert "MINI-AGENT" in out
        assert "deepseek-chat" in out


class TestCliReplExecution:
    """Test interactive REPL input handling and commands."""

    def test_repl_exit_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/exit\n",
                )
                assert result.exit_code == 0
                assert "再见" in result.stdout

    def test_repl_help_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/help\n/exit\n",
                )
                assert result.exit_code == 0
                assert "快捷指令" in result.stdout

    def test_repl_model_command(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("test answer")
        with patch("mini_agent.cli.OpenAIChatCompletionsClient", return_value=dummy_llm):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
                result = runner.invoke(
                    app,
                    ["--workspace", str(tmp_path)],
                    input="/model deepseek-reasoner\n/exit\n",
                )
                assert result.exit_code == 0
                assert "deepseek-reasoner" in result.stdout

    def test_rich_listener_events(self) -> None:
        test_console = Console(record=True)
        listener = RichAgentEventListener(console=test_console, verbose=True)

        listener.on_tool_start("read_file", {"path": "main.py"})
        listener.on_tool_finished(
            "read_file", ToolResult(ok=True, content="code", metadata={"size_bytes": 100})
        )
        listener.on_tool_finished("run_shell", ToolResult(ok=False, error="command failed"))

        out = test_console.export_text()
        assert "read_file" in out
        assert "成功" in out
        assert "失败" in out

    def test_run_cli_with_injected_client(self, tmp_path: Path) -> None:
        dummy_llm = DummyLLM("你好，这是测试！")
        with patch("rich.prompt.Prompt.ask", side_effect=["你好", "/exit"]):
            run_cli(workspace=tmp_path, llm_client=dummy_llm)
