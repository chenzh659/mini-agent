# mini-agent

> 一个轻量级、安全受控的本地终端 AI 编程助手 CLI（MVP）。

`mini-agent` 运行在指定的本地工作区目录，能够理解项目结构、安全读取文件并执行受控的 Shell 命令，帮助开发者快速探索项目、定位入口代码并运行测试。

**原生支持 OpenAI（GPT-4o / GPT-4o-mini）以及 DeepSeek（DeepSeek-V3 / DeepSeek-R1）和各类兼容接口。**

---

## 目录

- [功能特性与非目标](#功能特性与非目标)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用 DeepSeek 模型](#使用-deepseek-模型)
- [命令行参数与使用示例](#命令行参数与使用示例)
- [Agent 内置工具与限制](#agent-内置工具与限制)
- [安全声明与隐私策略](#安全声明与隐私策略)
- [自动化测试与代码质量](#自动化测试与代码质量)
- [项目代码结构](#项目代码结构)
- [已知限制与未来演进](#已知限制与未来演进)

---

## 功能特性与非目标

### ✅ MVP 支持的功能
- **多模型服务商兼容**：全面兼容 **DeepSeek**（`deepseek-chat` / `deepseek-reasoner`）、**OpenAI**、**Ollama 本地模型**及任何兼容 OpenAI 标准的服务商。
- **项目结构感知**：通过 `list_files` 快速遍历并展示工作区目录树。
- **安全文件读取**：通过 `read_file` 安全读取 UTF-8 文本文件内容。
- **受控 Shell 执行**：通过 `run_shell` 执行只读/低风险命令；高危操作自动拦截，其余操作需用户在终端敲击 `[y/N]` 显式确认。
- **环境敏感变量剥离**：向子进程透传时彻底清洗 `OPENAI_API_KEY`、云厂商凭据与各类敏感 Token。
- **现代化交互终端**：基于 Typer 与 Rich 构建，支持 Markdown 最终渲染、状态加载指示、`/help` 帮助与 `/exit` 退出。
- **100% 离线单元测试**：基于 `LLMClient` 协议和 FakeLLM 驱动，零 API 费用、离线秒级跑通全部测试。

### ❌ MVP 非目标（当前不做）
- 不支持写文件、代码编辑、补丁应用或 Git 提交。
- 不支持多 Agent 协作、MCP 协议、网页浏览器工具。
- 不包含容器级 OS 沙箱；`run_shell` 直接运行在宿主机的当前工作区。
- 不展示大模型内部思维链，仅展示工具调用与最终回答。

---

## 环境要求

- **操作系统**：macOS / Linux / Windows
- **Python**：`>= 3.12`
- **包管理器**：`uv` (推荐)
- **API 凭据**：OpenAI API Key 或 DeepSeek API Key

---

## 快速开始

### 方式一：一行命令免安装直接运行 (uvx / pipx)
无需手动克隆代码或配置环境：
```bash
# 使用 uvx 运行
uvx chiv-mini-agent

# 或者全局安装到系统
uv tool install chiv-mini-agent
```

### 方式二：从源码克隆运行
```bash
git clone https://github.com/chenzh659/mini-agent.git
cd mini-agent
uv sync --all-groups
```

### 2. 配置 API Key

#### 方案 A：使用 OpenAI 官方 API
```bash
export OPENAI_API_KEY="sk-your-openai-key"
uv run mini-agent
```

#### 方案 B：使用 DeepSeek 官方 API（推荐，高性价比）
```bash
export OPENAI_API_KEY="sk-your-deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export MINI_AGENT_MODEL="deepseek-chat"

uv run mini-agent
```

---

## 使用 DeepSeek 模型

`mini-agent` 针对 DeepSeek 进行了专门优化适配：

```bash
# 启动时直接指定 DeepSeek base-url 和模型
uv run mini-agent --base-url https://api.deepseek.com --model deepseek-chat
```

也可以在 `.env` 文件中配置：
```bash
OPENAI_API_KEY=sk-你的DeepSeekKey
OPENAI_BASE_URL=https://api.deepseek.com
MINI_AGENT_MODEL=deepseek-chat
```

---

## 命令行参数与使用示例

### 参数说明

```text
用法: mini-agent [选项]

选项:
  -w, --workspace PATH   指定目标工作区根目录（缺省为当前终端工作目录）
  -m, --model MODEL      覆盖本次会话的模型名称（如 deepseek-chat 或 gpt-4o）
  -b, --base-url URL     自定义 API Base URL（如 https://api.deepseek.com）
  -v, --verbose          显示工具执行耗时、返回码等诊断信息
  --help                 显示命令行帮助说明
```

### 运行示例

```bash
# 在当前工作区启动
uv run mini-agent

# 指定分析其他项目目录并使用 DeepSeek
uv run mini-agent --workspace /path/to/my-project --model deepseek-chat

# 开启详细诊断模式
uv run mini-agent --verbose
```

---

## Agent 内置工具与限制

| 工具名称 | 输入参数 | 核心功能 | 限制与约束 |
| :--- | :--- | :--- | :--- |
| **`list_files`** | `path` (默认 `.`), `max_depth` (1~5) | 列出工作区目录与文件 | 自动忽略 `.git`、`.venv`、`__pycache__`；跳过越界符号链接；上限 500 项。 |
| **`read_file`** | `path` (相对路径) | 读取文本文件内容 | 仅允许工作区内相对路径（拦截绝对路径与 `..` 越界）；仅读取 UTF-8 文本；单文件大小上限 100 KiB。 |
| **`edit_file`** | `path`, `target_content`, `replacement_content` | 精准局部修改代码 | 必须在文件中唯一匹配 `target_content`，避免歧义替换；工作区相对路径沙箱。 |
| **`write_file`** | `path`, `content` | 创建新文件或全量写入 | 工作区相对路径沙箱，自动创建父级目录。 |
| **`run_shell`** | `command` (Shell 字符串) | 执行工作区受控 Shell 指令 | 拦截 `rm -rf`、`mkfs`、`sudo`、`cat /etc/passwd` 等高危命令；非白名单命令强制提示用户确认；超时（默认 30s）强杀进程组。 |

---

## 安全声明与隐私策略

> ⚠️ **重要安全提示**：
> 1. `run_shell` 工具直接在本机工作区运行，**不是 OS 级的强安全沙箱**。
> 2. 请勿在不可信的目录、存有明文核心私钥的目录或生产服务器环境直接运行本工具。
> 3. 虽然工具层在执行子进程时已严格剥离 `OPENAI_API_KEY` 等敏感环境变量，但仍建议用户在执行非只读命令提示确认时仔细审阅命令内容。

---

## 自动化测试与代码质量

本项目采用 100% 离线 Mock 机制，运行测试不需要配置真实 API Key 或访问外网：

```bash
# 1. 运行所有单元测试 (pytest)
uv run pytest

# 2. 静态代码检查 (Ruff Lint)
uv run ruff check .

# 3. 代码格式化检查 (Ruff Format)
uv run ruff format --check .
```

---

## 项目代码结构

```text
mini-agent/
├── .env.example              # 环境变量模板（含 DeepSeek 配置示例）
├── .gitignore                # Git 忽略规则
├── README.md                 # 项目完整使用说明
├── pyproject.toml            # 构建配置、依赖项与脚本入口
├── uv.lock                   # 依赖精确锁定文件
├── src/
│   └── mini_agent/
│       ├── __init__.py       # 包版本定义
│       ├── main.py           # 控制台脚本入口 (mini-agent)
│       ├── cli.py            # Typer CLI、REPL 循环与 Rich 界面
│       ├── agent.py          # 核心 Agent Loop、History 管理与工具分发
│       ├── llm.py            # LLMClient 协议、OpenAI & DeepSeek 兼容适配器
│       ├── models.py         # Pydantic v2 数据契约与配置模型
│       └── tools/
│           ├── __init__.py   # 工具模块导出
│           ├── filesystem.py # 安全相对路径解析、read_file 与 list_files
│           └── shell.py      # 受控 run_shell、阻断/白名单与脱敏环境
└── tests/
    ├── __init__.py
    ├── conftest.py           # pytest 配置与通用 Fixture
    ├── test_smoke.py         # 导入冒烟测试
    ├── test_filesystem.py    # 路径安全与文件工具测试 (21 项)
    ├── test_shell.py         # 受控 Shell、环境脱敏与确认机制测试 (11 项)
    ├── test_agent.py         # 基于 FakeLLM 的 Agent Loop 多步闭环测试 (11 项)
    └── test_cli.py           # CLI 选项、异常退出与 REPL 交互测试 (8 项)
```
