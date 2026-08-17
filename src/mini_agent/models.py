"""Data models and configuration contracts for mini-agent."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentConfig(BaseModel):
    """Configuration for the agent and execution environment."""

    workspace_root: Path = Field(
        default_factory=Path.cwd,
        description="Root directory of the workspace. Must be an existing directory.",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model identifier to use.",
    )
    max_tool_rounds: int = Field(
        default=8,
        gt=0,
        description="Maximum consecutive tool calling rounds allowed in a single turn.",
    )
    shell_timeout_seconds: int = Field(
        default=30,
        gt=0,
        description="Timeout in seconds for shell command execution.",
    )
    max_output_chars: int = Field(
        default=12_000,
        gt=0,
        description="Maximum characters allowed in tool output before truncation.",
    )

    @field_validator("workspace_root", mode="before")
    @classmethod
    def validate_and_resolve_workspace_root(cls, v: Any) -> Path:
        path = Path(v).resolve()
        if not path.exists():
            raise ValueError(f"Workspace path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Workspace path is not a directory: {path}")
        return path


class ToolResult(BaseModel):
    """Unified result structure returned by all tools."""

    ok: bool = Field(description="Whether the tool execution succeeded.")
    content: str = Field(
        default="",
        description="Standard output or formatted result text intended for LLM context.",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable and LLM-readable error message when ok=False.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured metadata such as truncated flags, status codes, paths, etc.",
    )


class ReadFileInput(BaseModel):
    """Input parameters for the read_file tool."""

    path: str = Field(
        ...,
        min_length=1,
        description="Relative path of the UTF-8 text file to read within workspace.",
    )


class ListFilesInput(BaseModel):
    """Input parameters for the list_files tool."""

    path: str = Field(
        default=".",
        description="Relative path of the directory to list within workspace.",
    )
    max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum directory recursion depth (1-5, default 2).",
    )


class RunShellInput(BaseModel):
    """Input parameters for the run_shell tool."""

    command: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Shell command line string to execute in workspace.",
    )
