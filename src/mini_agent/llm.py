"""LLM protocol, Chat Completions adapter (DeepSeek/OpenAI), and system prompts."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class LLMError(Exception):
    """Base exception for LLM operations."""


class LLMAuthError(LLMError):
    """Authentication or missing API key error."""


class LLMConnectionError(LLMError):
    """Network connection or timeout error."""


@dataclass
class FunctionCall:
    """Represents a function call requested by the model."""

    name: str
    call_id: str
    arguments: str


@dataclass
class LLMResponse:
    """Unified response extracted from LLM output."""

    text: str | None = None
    function_calls: list[FunctionCall] = field(default_factory=list)
    raw_output: list[dict[str, Any]] = field(default_factory=list)


class LLMClient(Protocol):
    """Protocol defining the interface between Agent and LLM backend."""

    def create_response(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
    ) -> LLMResponse:
        """Send conversation history and tool schemas to LLM and return structured response."""
        ...


def get_system_prompt(workspace_root: Path) -> str:
    """Generate system instructions for the Agent."""
    return (
        f"你是一个运行在工作区 '{workspace_root.as_posix()}' 的本地开发助手 (mini-agent)。\n"
        "请严格遵守以下规则：\n"
        "1. 不了解项目结构时，优先调用 `list_files` 了解目录布局；"
        "需要查看具体文件时才调用 `read_file`，不要盲目扫描所有文件。\n"
        "2. 仅在确有必要时使用 `run_shell` 工具，优先选择安全只读命令（如 pytest、git status）。\n"
        "3. 所有文件路径必须相对于工作区根目录；如果工具执行失败，请解释原因或换用安全方式，"
        "严禁编造未读取的文件内容。\n"
        "4. 工具执行完毕后，必须基于真实结果使用中文回答用户，简洁说明做了什么。\n"
        "5. 当前 MVP 阶段没有文件写入工具，严禁声称已修改或创建了任何文件。"
    )


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON schemas for available agent tools."""
    return [
        {
            "type": "function",
            "name": "read_file",
            "description": "读取工作区内指定 UTF-8 文本文件的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于工作区根目录的文件相对路径。",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "list_files",
            "description": "列出工作区内指定相对目录的文件和子目录结构。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于工作区的相对目录路径，默认 '.'。",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "递归目录深度（1 到 5 之间，默认 2）。",
                    },
                },
                "additionalProperties": False,
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "run_shell",
            "description": "在工作区根目录下执行受控的 Shell 命令行指令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要在工作区中执行的 Shell 命令行字符串。",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


class OpenAIChatCompletionsClient:
    """OpenAI & DeepSeek compatible client using the standard Chat Completions API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        import openai

        effective_base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        )
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=effective_base_url,
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tools schema to standard Chat Completions format."""
        converted = []
        for t in tools:
            if "function" in t:
                converted.append(t)
            else:
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description"),
                            "parameters": t.get("parameters"),
                        },
                    }
                )
        return converted

    def _convert_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize history items to standard Chat Completions messages."""
        messages: list[dict[str, Any]] = []
        for item in history:
            role = item.get("role")
            item_type = item.get("type")

            if item_type == "function_call_output" or role == "tool":
                call_id = item.get("call_id") or item.get("tool_call_id", "")
                output = item.get("output") or item.get("content", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output,
                    }
                )
            elif role in ("system", "user"):
                messages.append({"role": role, "content": item.get("content", "")})
            elif role == "assistant":
                msg: dict[str, Any] = {"role": "assistant"}
                if item.get("content"):
                    msg["content"] = item["content"]
                if item.get("tool_calls"):
                    msg["tool_calls"] = item["tool_calls"]
                messages.append(msg)
            elif item_type == "message":
                messages.append(
                    {
                        "role": item.get("role", "assistant"),
                        "content": item.get("content", ""),
                    }
                )
        return messages

    def create_response(
        self,
        history: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = "gpt-4o-mini",
    ) -> LLMResponse:
        """Call Chat Completions API and convert output to LLMResponse."""
        import openai

        messages = self._convert_messages(history)
        chat_tools = self._convert_tools(tools)

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=chat_tools if chat_tools else None,
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthError(f"API 身份验证失败，请检查 API Key 有效性: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LLMConnectionError(
                f"无法连接到 API 服务，请检查网络或 base_url 配置: {exc}"
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"API 速率限制 (Rate Limit): {exc}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"API 调用失败: {exc}") from exc

        if not response.choices:
            return LLMResponse(text=None)

        choice = response.choices[0]
        msg = choice.message
        text = msg.content
        function_calls: list[FunctionCall] = []
        tool_calls_raw: list[dict[str, Any]] = []

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                function_calls.append(
                    FunctionCall(
                        name=tc.function.name,
                        call_id=tc.id,
                        arguments=tc.function.arguments,
                    )
                )
                tool_calls_raw.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        raw_output_item: dict[str, Any] = {"role": "assistant"}
        if text is not None:
            raw_output_item["content"] = text
        if tool_calls_raw:
            raw_output_item["tool_calls"] = tool_calls_raw

        return LLMResponse(
            text=text,
            function_calls=function_calls,
            raw_output=[raw_output_item],
        )


# Backward compatibility alias
OpenAIResponsesClient = OpenAIChatCompletionsClient
