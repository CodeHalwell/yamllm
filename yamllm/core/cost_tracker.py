"""Cost tracking for LLM usage across all providers."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json


# Pricing per 1M tokens (as of Jan 2025)
PROVIDER_PRICING = {
    "openai": {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    },
    "anthropic": {
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    },
    "google": {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-pro": {"input": 0.50, "output": 1.50},
    },
    "mistral": {
        "mistral-large": {"input": 4.0, "output": 12.0},
        "mistral-medium": {"input": 2.7, "output": 8.1},
        "mistral-small": {"input": 1.0, "output": 3.0},
        "mistral-tiny": {"input": 0.25, "output": 0.25},
    },
    "deepseek": {
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-coder": {"input": 0.14, "output": 0.28},
    }
}


@dataclass
class UsageRecord:
    """Record of a single LLM API call."""

    timestamp: datetime
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    request_type: str = "completion"  # completion, embedding, tool_call
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "request_type": self.request_type,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UsageRecord":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class CostSummary:
    """Summary of costs for a session or time period."""

    total_cost: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    request_count: int = 0
    by_provider: Dict[str, float] = field(default_factory=dict)
    by_model: Dict[str, float] = field(default_factory=dict)
    records: List[UsageRecord] = field(default_factory=list)

    def add_record(self, record: UsageRecord) -> None:
        """Add a usage record to the summary."""
        self.total_cost += record.cost
        self.total_tokens += record.total_tokens
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.request_count += 1

        # Track by provider
        if record.provider not in self.by_provider:
            self.by_provider[record.provider] = 0.0
        self.by_provider[record.provider] += record.cost

        # Track by model
        model_key = f"{record.provider}/{record.model}"
        if model_key not in self.by_model:
            self.by_model[model_key] = 0.0
        self.by_model[model_key] += record.cost

        self.records.append(record)

    def get_top_costs(self, n: int = 5) -> List[tuple]:
        """Get top N models by cost."""
        sorted_models = sorted(
            self.by_model.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_models[:n]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "request_count": self.request_count,
            "by_provider": self.by_provider,
            "by_model": self.by_model,
            "records": [r.to_dict() for r in self.records]
        }


class CostTracker:
    """Track costs across all LLM providers."""

    def __init__(self):
        """Initialize cost tracker."""
        self.current_session = CostSummary()
        self.budget_limit: Optional[float] = None
        self.budget_warning_threshold: float = 0.8

    def calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate cost for a request.

        Args:
            provider: Provider name (openai, anthropic, etc.)
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        # Normalize provider and model names
        provider = provider.lower()
        model = model.lower()

        # Get pricing for provider
        if provider not in PROVIDER_PRICING:
            # Unknown provider, return 0 (can't calculate)
            return 0.0

        provider_pricing = PROVIDER_PRICING[provider]

        # Find matching model (handle partial matches)
        model_pricing = None
        for price_model, pricing in provider_pricing.items():
            if price_model in model or model in price_model:
                model_pricing = pricing
                break

        if model_pricing is None:
            # Unknown model, return 0
            return 0.0

        # Calculate cost (pricing is per 1M tokens)
        input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost

    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_type: str = "completion",
        metadata: Optional[Dict] = None
    ) -> UsageRecord:
        """
        Record a usage event.

        Args:
            provider: Provider name
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            request_type: Type of request
            metadata: Optional metadata

        Returns:
            UsageRecord
        """
        cost = self.calculate_cost(provider, model, prompt_tokens, completion_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            request_type=request_type,
            metadata=metadata or {}
        )

        self.current_session.add_record(record)

        # Check budget
        if self.budget_limit and self.current_session.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget limit ${self.budget_limit:.2f} exceeded. "
                f"Current cost: ${self.current_session.total_cost:.2f}"
            )

        # Warn if approaching budget
        if (self.budget_limit and
            self.current_session.total_cost > self.budget_limit * self.budget_warning_threshold and
            self.current_session.total_cost < self.budget_limit):
            import warnings
            warnings.warn(
                f"Approaching budget limit: ${self.current_session.total_cost:.2f} "
                f"/ ${self.budget_limit:.2f}"
            )

        return record

    def set_budget(self, limit: float, warning_threshold: float = 0.8) -> None:
        """
        Set budget limit.

        Args:
            limit: Budget limit in USD
            warning_threshold: Warn when this fraction of budget is reached
        """
        self.budget_limit = limit
        self.budget_warning_threshold = warning_threshold

    def get_summary(self) -> CostSummary:
        """Get current session summary."""
        return self.current_session

    def reset_session(self) -> CostSummary:
        """Reset session and return old summary."""
        old_summary = self.current_session
        self.current_session = CostSummary()
        return old_summary

    def save_session(self, filepath: str) -> None:
        """Save session to file."""
        with open(filepath, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2)

    def estimate_cost(
        self,
        provider: str,
        model: str,
        text: str,
        is_completion: bool = False
    ) -> float:
        """
        Estimate cost for a text.

        Args:
            provider: Provider name
            model: Model name
            text: Text to estimate
            is_completion: If True, treat as completion tokens (higher cost)

        Returns:
            Estimated cost in USD
        """
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4

        if is_completion:
            return self.calculate_cost(provider, model, 0, estimated_tokens)
        else:
            return self.calculate_cost(provider, model, estimated_tokens, 0)


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""
    pass


class CostOptimizer:
    """Analyze costs and suggest optimizations."""

    def __init__(self, summary: CostSummary):
        """Initialize with a cost summary."""
        self.summary = summary

    def analyze(self) -> Dict:
        """Analyze costs and provide recommendations."""
        recommendations = []
        potential_savings = 0.0

        # Find expensive models
        top_costs = self.summary.get_top_costs(3)

        for model_key, cost in top_costs:
            provider, model = model_key.split("/")

            # Suggest cheaper alternatives
            if "gpt-4" in model and "gpt-4o-mini" not in model:
                savings_per_call = cost * 0.95  # Could save ~95%
                recommendations.append({
                    "current": model_key,
                    "suggestion": f"{provider}/gpt-4o-mini",
                    "reason": "Much cheaper for simple tasks",
                    "estimated_savings": savings_per_call
                })
                potential_savings += savings_per_call

            elif "claude-3-opus" in model:
                savings_per_call = cost * 0.80  # Could save ~80%
                recommendations.append({
                    "current": model_key,
                    "suggestion": f"{provider}/claude-3.5-sonnet",
                    "reason": "Better performance at lower cost",
                    "estimated_savings": savings_per_call
                })
                potential_savings += savings_per_call

            elif "gemini-1.5-pro" in model:
                savings_per_call = cost * 0.90  # Could save ~90%
                recommendations.append({
                    "current": model_key,
                    "suggestion": f"{provider}/gemini-1.5-flash",
                    "reason": "Much faster and cheaper for most tasks",
                    "estimated_savings": savings_per_call
                })
                potential_savings += savings_per_call

        return {
            "total_cost": self.summary.total_cost,
            "potential_savings": potential_savings,
            "savings_percentage": (potential_savings / self.summary.total_cost * 100) if self.summary.total_cost > 0 else 0,
            "recommendations": recommendations
        }
