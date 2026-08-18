"""Provider presets for mainstream LLM services (DeepSeek, OpenAI, Ollama, Qwen, etc.)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    """Configuration preset for an LLM provider."""

    name: str
    display_name: str
    default_model: str
    base_url: str
    description: str
    env_key: str = "OPENAI_API_KEY"


PREDEFINED_PROVIDERS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        name="deepseek",
        display_name="DeepSeek (官方 V3)",
        default_model="deepseek-chat",
        base_url="https://api.deepseek.com",
        description="深度求索官方接口，高性价比，编程与通用能力极强",
    ),
    "deepseek-r1": ProviderPreset(
        name="deepseek-r1",
        display_name="DeepSeek-R1 (深度推理)",
        default_model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        description="深度求索推理模型，擅长复杂算法设计与深度代码推理",
    ),
    "openai": ProviderPreset(
        name="openai",
        display_name="OpenAI (官方)",
        default_model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        description="OpenAI 官方接口，支持 GPT-4o 及 GPT-4o-mini",
    ),
    "ollama": ProviderPreset(
        name="ollama",
        display_name="Ollama (本地离线)",
        default_model="qwen2.5-coder:latest",
        base_url="http://localhost:11434/v1",
        description="本地离线开源大模型，完全免费、零网络开销、数据不出本地",
    ),
    "qwen": ProviderPreset(
        name="qwen",
        display_name="通义千问 (阿里云百炼)",
        default_model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="阿里云 DashScope 兼容接口，支持 Qwen-Plus / Qwen-Turbo",
    ),
    "siliconflow": ProviderPreset(
        name="siliconflow",
        display_name="硅基流动 (SiliconFlow)",
        default_model="deepseek-ai/DeepSeek-V3",
        base_url="https://api.siliconflow.cn/v1",
        description="国内高并发模型托管平台，支持 DeepSeek-V3 / R1 等",
    ),
    "moonshot": ProviderPreset(
        name="moonshot",
        display_name="Moonshot (Kimi)",
        default_model="moonshot-v1-8k",
        base_url="https://api.moonshot.cn/v1",
        description="月之暗面 Kimi 兼容接口",
    ),
    "zhipu": ProviderPreset(
        name="zhipu",
        display_name="智谱 AI (GLM)",
        default_model="glm-4-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        description="智谱清言大模型开放平台",
    ),
}


def get_provider_preset(name: str) -> ProviderPreset | None:
    """Lookup a provider preset by name (case-insensitive)."""
    clean_name = name.strip().lower()
    return PREDEFINED_PROVIDERS.get(clean_name)


def list_provider_presets() -> list[ProviderPreset]:
    """Return all predefined provider presets."""
    return list(PREDEFINED_PROVIDERS.values())
