"""Unit tests for token usage and cost calculations."""

from mini_agent.cost import (
    UsageStats,
    calculate_cost_cny,
    estimate_tokens_from_text,
    format_cost_cny,
)


def test_usage_stats_add() -> None:
    u1 = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    u2 = UsageStats(prompt_tokens=200, completion_tokens=80, total_tokens=280)
    u3 = u1.add(u2)

    assert u3.prompt_tokens == 300
    assert u3.completion_tokens == 130
    assert u3.total_tokens == 430


def test_estimate_tokens_from_text() -> None:
    assert estimate_tokens_from_text("") == 0
    assert estimate_tokens_from_text("hello world") >= 1
    long_text = "def calculate_sum(a: int, b: int) -> int:\n    return a + b\n"
    tokens = estimate_tokens_from_text(long_text)
    assert tokens > 10


def test_calculate_cost_deepseek() -> None:
    # 1M prompt @ 1.0 CNY, 1M completion @ 2.0 CNY
    cost = calculate_cost_cny(1_000_000, 1_000_000, "deepseek-chat")
    assert cost == 3.0

    # 1K prompt @ 0.001, 1K completion @ 0.002 -> 0.003
    cost_1k = calculate_cost_cny(1000, 1000, "deepseek-chat")
    assert abs(cost_1k - 0.003) < 1e-5


def test_calculate_cost_deepseek_r1() -> None:
    # 1M prompt @ 4.0 CNY, 1M completion @ 16.0 CNY
    cost = calculate_cost_cny(1_000_000, 1_000_000, "deepseek-reasoner")
    assert cost == 20.0


def test_calculate_cost_ollama_free() -> None:
    cost = calculate_cost_cny(500_000, 500_000, "ollama/qwen2.5-coder")
    assert cost == 0.0


def test_format_cost_cny() -> None:
    assert format_cost_cny(0.0) == "免费 (¥0.00)"
    assert "¥" in format_cost_cny(0.0024)
    assert "¥" in format_cost_cny(1.50)
