"""Tests for cost tracking functionality."""

import pytest
from datetime import datetime
from yamllm.core.cost_tracker import (
    CostTracker,
    CostOptimizer,
    BudgetExceededError,
    PROVIDER_PRICING
)


def test_cost_tracker_initialization():
    """Test cost tracker initialization."""
    tracker = CostTracker()

    assert tracker.current_session is not None
    assert tracker.current_session.total_cost == 0.0
    assert tracker.current_session.total_calls == 0
    assert tracker.current_session.total_tokens == 0


def test_calculate_cost():
    """Test cost calculation for different providers."""
    tracker = CostTracker()

    # OpenAI GPT-4o-mini
    cost = tracker.calculate_cost("openai", "gpt-4o-mini", 1000, 500)
    expected = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
    assert cost == pytest.approx(expected)

    # Anthropic Claude 3.5 Sonnet
    cost = tracker.calculate_cost("anthropic", "claude-3.5-sonnet", 1000, 500)
    expected = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
    assert cost == pytest.approx(expected)


def test_record_usage():
    """Test recording usage."""
    tracker = CostTracker()

    tracker.record_usage(
        provider="openai",
        model="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500
    )

    summary = tracker.get_summary()
    assert summary.total_calls == 1
    assert summary.total_tokens == 1500
    assert summary.total_cost > 0


def test_budget_exceeded():
    """Test budget limit enforcement."""
    tracker = CostTracker()
    tracker.set_budget(0.001)  # Very low budget

    # First call should succeed
    tracker.record_usage("openai", "gpt-4", 10000, 5000)

    # Second call should exceed budget
    with pytest.raises(BudgetExceededError):
        tracker.record_usage("openai", "gpt-4", 10000, 5000)


def test_budget_warning():
    """Test budget warning threshold."""
    tracker = CostTracker()
    tracker.set_budget(1.0, warning_threshold=0.5)

    warnings = []

    def warning_handler(msg):
        warnings.append(msg)

    # Record usage approaching threshold
    tracker.record_usage("openai", "gpt-4", 100000, 50000)

    summary = tracker.get_summary()
    assert summary.budget_limit == 1.0


def test_session_summary():
    """Test session summary generation."""
    tracker = CostTracker()

    # Record multiple calls
    tracker.record_usage("openai", "gpt-4o-mini", 1000, 500)
    tracker.record_usage("anthropic", "claude-3.5-sonnet", 2000, 1000)
    tracker.record_usage("openai", "gpt-4o-mini", 1500, 750)

    summary = tracker.get_summary()
    assert summary.total_calls == 3
    assert summary.total_tokens == 6750
    assert summary.total_cost > 0
    assert "openai" in summary.provider_breakdown
    assert "anthropic" in summary.provider_breakdown


def test_reset_session():
    """Test session reset."""
    tracker = CostTracker()

    tracker.record_usage("openai", "gpt-4o-mini", 1000, 500)
    assert tracker.get_summary().total_calls == 1

    tracker.reset_session()
    assert tracker.get_summary().total_calls == 0
    assert tracker.get_summary().total_cost == 0.0


def test_cost_optimizer():
    """Test cost optimizer."""
    tracker = CostTracker()

    # Record usage with expensive model
    tracker.record_usage("openai", "gpt-4", 10000, 5000)

    optimizer = CostOptimizer(tracker)
    analysis = optimizer.analyze()

    assert "current_model" in analysis
    assert "recommendations" in analysis
    assert len(analysis["recommendations"]) > 0

    # Check that cheaper alternatives are suggested
    for rec in analysis["recommendations"]:
        assert rec["savings"] > 0
        assert rec["savings_percent"] > 0


def test_unknown_model_fallback():
    """Test fallback pricing for unknown models."""
    tracker = CostTracker()

    # Unknown model should use fallback pricing
    cost = tracker.calculate_cost("unknown_provider", "unknown_model", 1000, 500)
    assert cost > 0  # Should use default fallback pricing


def test_provider_breakdown():
    """Test provider-level cost breakdown."""
    tracker = CostTracker()

    tracker.record_usage("openai", "gpt-4o-mini", 1000, 500)
    tracker.record_usage("openai", "gpt-4", 1000, 500)
    tracker.record_usage("anthropic", "claude-3.5-sonnet", 1000, 500)

    summary = tracker.get_summary()

    # Check OpenAI total
    openai_cost = summary.provider_breakdown.get("openai", 0)
    assert openai_cost > 0

    # Check Anthropic total
    anthropic_cost = summary.provider_breakdown.get("anthropic", 0)
    assert anthropic_cost > 0


def test_model_breakdown():
    """Test model-level cost breakdown."""
    tracker = CostTracker()

    tracker.record_usage("openai", "gpt-4o-mini", 1000, 500)
    tracker.record_usage("openai", "gpt-4o-mini", 2000, 1000)
    tracker.record_usage("openai", "gpt-4", 1000, 500)

    summary = tracker.get_summary()

    # Check model breakdown
    gpt4o_mini_cost = summary.model_breakdown.get("gpt-4o-mini", 0)
    gpt4_cost = summary.model_breakdown.get("gpt-4", 0)

    assert gpt4o_mini_cost > 0
    assert gpt4_cost > 0
    assert gpt4_cost > gpt4o_mini_cost  # GPT-4 should be more expensive


def test_average_cost_per_call():
    """Test average cost per call calculation."""
    tracker = CostTracker()

    tracker.record_usage("openai", "gpt-4o-mini", 1000, 500)
    tracker.record_usage("openai", "gpt-4o-mini", 2000, 1000)

    summary = tracker.get_summary()
    assert summary.avg_cost_per_call == summary.total_cost / 2


def test_pricing_data_completeness():
    """Test that pricing data is complete for all providers."""
    required_providers = ["openai", "anthropic", "google", "mistral", "azure_openai"]

    for provider in required_providers:
        assert provider in PROVIDER_PRICING, f"Missing pricing for {provider}"
        assert len(PROVIDER_PRICING[provider]) > 0, f"No models for {provider}"

        # Check that each model has input and output pricing
        for model, pricing in PROVIDER_PRICING[provider].items():
            assert "input" in pricing, f"Missing input pricing for {provider}/{model}"
            assert "output" in pricing, f"Missing output pricing for {provider}/{model}"
            assert pricing["input"] > 0
            assert pricing["output"] > 0
