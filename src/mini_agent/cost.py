"""Token usage tracking and cost estimation for mini-agent."""

from pydantic import BaseModel, Field


class UsageStats(BaseModel):
    """Token usage counters."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def add(self, other: "UsageStats") -> "UsageStats":
        """Add another usage stats to this one."""
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


# Pricing per 1M tokens in CNY (RMB)
# (Input price per 1M, Output price per 1M)
MODEL_PRICING_CNY: dict[str, tuple[float, float]] = {
    # DeepSeek
    "deepseek-chat": (1.0, 2.0),
    "deepseek-v3": (1.0, 2.0),
    "deepseek-reasoner": (4.0, 16.0),
    "deepseek-r1": (4.0, 16.0),
    # OpenAI (USD converted to approx CNY @ 7.2)
    "gpt-4o-mini": (1.08, 4.32),
    "gpt-4o": (18.0, 72.0),
    "o1-mini": (21.6, 86.4),
    "o1": (108.0, 432.0),
    # Qwen
    "qwen-turbo": (0.3, 0.6),
    "qwen-plus": (0.8, 2.0),
    "qwen-max": (20.0, 60.0),
    # Moonshot
    "moonshot-v1-8k": (12.0, 12.0),
    # GLM
    "glm-4-flash": (0.0, 0.0),  # Free tier
    "glm-4": (10.0, 10.0),
    # Ollama / Local
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}


def estimate_tokens_from_text(text: str) -> int:
    """Fallback token estimation: ~1 token per 3.5 characters for mixed text."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


def calculate_cost_cny(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate estimated cost in CNY (RMB) for a given token usage and model."""
    clean_model = model.strip().lower()

    pricing: tuple[float, float] | None = None
    if "ollama" in clean_model or "local" in clean_model:
        return 0.0

    for m_key, m_price in MODEL_PRICING_CNY.items():
        if m_key in clean_model:
            pricing = m_price
            break

    if pricing is None:
        # Default fallback pricing: 2.0 CNY / 1M input, 4.0 CNY / 1M output
        pricing = (2.0, 4.0)

    input_price_per_m, output_price_per_m = pricing
    cost = (prompt_tokens / 1_000_000 * input_price_per_m) + (
        completion_tokens / 1_000_000 * output_price_per_m
    )
    return round(cost, 6)


def format_cost_cny(cost: float) -> str:
    """Format cost nicely for terminal display."""
    if cost == 0.0:
        return "免费 (¥0.00)"
    if cost < 0.0001:
        return f"¥{cost:.6f}"
    if cost < 0.01:
        return f"¥{cost:.4f}"
    return f"¥{cost:.2f}"
