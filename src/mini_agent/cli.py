"""Typer CLI interface and Rich REPL implementation (Antigravity Style)."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from mini_agent.agent import Agent, AgentEventListener
from mini_agent.llm import LLMClient, LLMError, OpenAIChatCompletionsClient
from mini_agent.models import AgentConfig, ToolResult

app = typer.Typer(
    name="mini-agent",
    help="mini-agent: 本地终端 AI 编程助手",
    add_completion=False,
)
console = Console()


def render_banner(console: Console, workspace: Path, model: str) -> None:
    """Render sleek Antigravity-style header banner."""
    content = Text()
    content.append("✦ ", style="bold cyan")
    content.append("MINI-AGENT", style="bold white")
    content.append("  v0.1.0\n", style="dim")
    content.append("📁 工作区: ", style="bold bright_black")
    content.append(f"{workspace.as_posix()}\n", style="white")
    content.append("⚡ 模型:   ", style="bold bright_black")
    content.append(f"{model}\n", style="bright_cyan")
    content.append("💡 提示:   ", style="bold bright_black")
    content.append("输入问题开始协作，使用 ", style="dim")
    content.append("/help", style="bold cyan")
    content.append(" 查看指令，", style="dim")
    content.append("/clear", style="bold cyan")
    content.append(" 清屏，", style="dim")
    content.append("/exit", style="bold cyan")
    content.append(" 退出", style="dim")

    console.print(
        Panel(
            content,
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )


class RichAgentEventListener(AgentEventListener):
    """Rich terminal event listener with structured step cards."""

    def __init__(self, console: Console, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose

    def on_turn_start(self, user_input: str) -> None:
        pass

    def on_model_start(self) -> None:
        if self.verbose:
            self.console.print("  [dim cyan]⏺ 正在请求模型思考...[/dim cyan]")

    def on_tool_start(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "read_file":
            path = arguments.get("path", "")
            self.console.print(
                f"  [bold cyan]⚡ Tool: read_file[/bold cyan] [dim](路径: {path})[/dim]"
            )
        elif tool_name == "list_files":
            path = arguments.get("path", ".")
            depth = arguments.get("max_depth", 2)
            self.console.print(
                f"  [bold cyan]⚡ Tool: list_files[/bold cyan] "
                f"[dim](路径: {path}, 深度: {depth})[/dim]"
            )
        elif tool_name == "run_shell":
            cmd = arguments.get("command", "")
            self.console.print(
                f"  [bold cyan]⚡ Tool: run_shell[/bold cyan] [dim](命令: {cmd})[/dim]"
            )
        else:
            self.console.print(f"  [bold cyan]⚡ Tool: {tool_name}[/bold cyan]")

    def on_tool_confirm(self, command: str) -> bool:
        self.console.print(
            Panel(
                f"[yellow]Agent 请求执行以下非只读 Shell 命令：[/yellow]\n\n"
                f"  [bold cyan]{command}[/bold cyan]\n\n"
                f"[dim]请确认该命令在当前工作区内执行是否安全。[/dim]",
                title="[bold yellow]⚠️  安全确认 (Security Confirmation)[/bold yellow]",
                box=box.ROUNDED,
                border_style="yellow",
                padding=(0, 2),
            )
        )
        return Confirm.ask("是否允许执行该命令？", default=False, console=self.console)

    def on_tool_finished(self, tool_name: str, result: ToolResult) -> None:
        if result.ok:
            metadata_parts: list[str] = []
            if "size_bytes" in result.metadata:
                metadata_parts.append(f"{result.metadata['size_bytes']} 字节")
            if "total_entries" in result.metadata:
                metadata_parts.append(f"{result.metadata['total_entries']} 项")
            if "exit_code" in result.metadata and self.verbose:
                metadata_parts.append(f"返回码: {result.metadata['exit_code']}")

            extra_info = f" [dim]({', '.join(metadata_parts)})[/dim]" if metadata_parts else ""
            self.console.print(f"  [bold green]✔ 执行成功[/bold green]{extra_info}")
        else:
            reason = result.error or "未知错误"
            self.console.print(f"  [bold red]✗ 执行失败[/bold red]: [red]{reason}[/red]")

    def on_turn_finished(self, response: str) -> None:
        pass


def render_help(console: Console) -> None:
    """Print beautifully formatted help table."""
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=True, header_style="bold cyan")
    table.add_column("指令", style="bold white", width=16)
    table.add_column("说明与用途", style="white")

    table.add_row("/help", "显示快捷指令与 Agent 工具能力说明")
    table.add_row("/clear", "清屏并重新展示顶部状态 Banner")
    table.add_row("/model [name]", "查看或临时切换当前模型 (如 /model deepseek-chat)")
    table.add_row("/exit, /quit", "退出当前 mini-agent 会话")

    console.print(table)

    tools_table = Table(
        box=box.ROUNDED, border_style="dim", show_header=True, header_style="bold green"
    )
    tools_table.add_column("内置工具", style="bold green", width=16)
    tools_table.add_column("功能", style="white", width=26)
    tools_table.add_column("安全策略与约束", style="dim")

    tools_table.add_row(
        "list_files",
        "列出工作区目录结构",
        "仅允许工作区内相对路径，深度 1~5，跳过 .git/.venv",
    )
    tools_table.add_row(
        "read_file",
        "读取 UTF-8 文件内容",
        "工作区相对路径沙箱，单文件上限 100 KiB",
    )
    tools_table.add_row(
        "run_shell",
        "受控 Shell 命令执行",
        "阻断破坏性高危指令，环境脱敏，非白名单命令弹窗确认",
    )

    console.print(tools_table)
    console.print("[dim]提示：按 Ctrl-C 取消当前行输入，按 Ctrl-D 正常退出。[/dim]\n")


def repl_loop(agent: Agent, console: Console) -> None:
    """Main interactive REPL loop with Antigravity styling."""
    render_banner(console, agent.config.workspace_root, agent.config.model)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]>[/bold cyan]", console=console).strip()
        except KeyboardInterrupt:
            console.print("\n[yellow]已取消当前输入[/yellow]")
            continue
        except EOFError:
            console.print("\n[dim]👋 再见！[/dim]")
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            console.print("[dim]👋 再见！[/dim]")
            break

        if user_input == "/clear":
            console.clear()
            render_banner(console, agent.config.workspace_root, agent.config.model)
            continue

        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                new_model = parts[1].strip()
                agent.config.model = new_model
                console.print(
                    f"[green]✔ 已切换当前模型为:[/green] [bold cyan]{new_model}[/bold cyan]\n"
                )
            else:
                console.print(
                    f"[dim]当前会话模型:[/dim] [bold cyan]{agent.config.model}[/bold cyan]\n"
                )
            continue

        if user_input == "/help":
            render_help(console)
            continue

        try:
            with console.status("[bold cyan]⏺ Agent 正在思考并执行...[/bold cyan]", spinner="dots"):
                answer = agent.step(user_input)

            console.print("\n[bold cyan]🤖 Mini-Agent[/bold cyan]")
            try:
                console.print(Markdown(answer))
            except Exception:
                console.print(answer)
            console.print()
        except LLMError as exc:
            console.print(f"\n[bold red]LLM 错误[/bold red]: {exc}\n")
        except Exception as exc:
            console.print(f"\n[bold red]执行错误[/bold red]: {exc}\n")


def load_dotenv(workspace_root: Path | None = None) -> None:
    """Lightweight loader for .env file within workspace."""
    path = (workspace_root / ".env") if workspace_root else Path(".env")
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, val = stripped.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass


def run_cli(
    workspace: Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    verbose: bool = False,
    agent_factory: Callable[[AgentConfig, LLMClient, AgentEventListener], Agent] | None = None,
    llm_client: LLMClient | None = None,
) -> None:
    """Core logic to run the CLI."""
    target_workspace = (workspace or Path.cwd()).resolve()
    load_dotenv(target_workspace)
    if not target_workspace.exists():
        console.print(f"[bold red]错误[/bold red]: 指定的工作区路径不存在: '{target_workspace}'")
        raise typer.Exit(code=1)
    if not target_workspace.is_dir():
        console.print(f"[bold red]错误[/bold red]: 指定的工作区路径不是目录: '{target_workspace}'")
        raise typer.Exit(code=1)

    effective_base_url = (
        base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    )
    # Default model: If DeepSeek base_url is detected, default to deepseek-chat
    if model:
        resolved_model = model
    elif os.environ.get("MINI_AGENT_MODEL"):
        resolved_model = os.environ["MINI_AGENT_MODEL"]
    elif effective_base_url and "deepseek" in effective_base_url.lower():
        resolved_model = "deepseek-chat"
    else:
        resolved_model = "gpt-4o-mini"

    # Verify API key if default client is used
    if llm_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            console.print(
                "[bold red]错误[/bold red]: 未检测到 OPENAI_API_KEY 环境变量。\n"
                "请先设置您的 API Key，例如在终端执行：\n"
                "  [cyan]export OPENAI_API_KEY='sk-...'[/cyan]\n"
                "若使用 DeepSeek，可同时配置：\n"
                "  [cyan]export OPENAI_BASE_URL='https://api.deepseek.com'[/cyan]\n"
                "  [cyan]export MINI_AGENT_MODEL='deepseek-chat'[/cyan]"
            )
            raise typer.Exit(code=1)
        client: LLMClient = OpenAIChatCompletionsClient(
            api_key=api_key,
            base_url=effective_base_url,
        )
    else:
        client = llm_client

    config = AgentConfig(
        workspace_root=target_workspace,
        model=resolved_model,
    )
    listener = RichAgentEventListener(console=console, verbose=verbose)

    if agent_factory is not None:
        agent = agent_factory(config, client, listener)
    else:
        agent = Agent(config=config, llm_client=client, listener=listener)

    repl_loop(agent, console=console)


@app.command()
def main(
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help="目标工作区根目录路径（缺省为当前工作目录）",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="覆盖本次会话的模型名称（如 deepseek-chat 或 gpt-4o）",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            "-b",
            help="自定义 API Base URL（如 https://api.deepseek.com）",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="显示诊断与执行详细信息",
        ),
    ] = False,
) -> None:
    """启动 mini-agent 交互式 REPL。"""
    run_cli(workspace=workspace, model=model, base_url=base_url, verbose=verbose)
