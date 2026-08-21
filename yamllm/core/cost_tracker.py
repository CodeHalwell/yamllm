"""Cost tracking for LLM usage across all providers."""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json


# Pricing per 1M tokens (as of Jan 2025).
# Embedding models are kept in EMBEDDING_PRICING below — they have $0 output
# pricing which would fail the completeness invariant for completion models.
PROVIDER_PRICING = {
    "openai": {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "o1": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
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
    },
    "azure_openai": {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
}

EMBEDDING_PRICING = {
    "openai": {
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    },
}

# Fallback used when the provider/model is unknown — small but non-zero so that
# usage of unknown models is still surfaced in the cost summary.
FALLBACK_PRICING = {"input": 1.0, "output": 2.0}


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
    request_type: str = "completion"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "request_type": self.request_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UsageRecord":
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
    total_calls: int = 0
    provider_breakdown: Dict[str, float] = field(default_factory=dict)
    # Bare-model-name view, kept for human-readable summaries and
    # backwards-compatible callers that look up by ``model_breakdown["gpt-4o"]``.
    model_breakdown: Dict[str, float] = field(default_factory=dict)
    # Disambiguated view keyed by ``"<provider>/<model>"``. Use this when the
    # answer matters per-provider (e.g. distinguishing OpenAI's gpt-4o from
    # Azure's gpt-4o for cost optimisation recommendations).
    by_provider_model: Dict[str, float] = field(default_factory=dict)
    records: List[UsageRecord] = field(default_factory=list)
    budget_limit: Optional[float] = None

    # Backwards-compatible alias used by older call sites; reads/writes the
    # canonical total_calls field.
    @property
    def request_count(self) -> int:
        return self.total_calls

    @request_count.setter
    def request_count(self, value: int) -> None:
        self.total_calls = value

    @property
    def avg_cost_per_call(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_cost / self.total_calls

    # Older callers used by_provider / by_model. Keep these as aliases so we
    # don't break existing integrations on this rename.
    @property
    def by_provider(self) -> Dict[str, float]:
        return self.provider_breakdown

    @property
    def by_model(self) -> Dict[str, float]:
        return self.model_breakdown

    def add_record(self, record: UsageRecord) -> None:
        self.total_cost += record.cost
        self.total_tokens += record.total_tokens
        self.prompt_tokens += record.prompt_tokens
        self.completion_tokens += record.completion_tokens
        self.total_calls += 1

        self.provider_breakdown[record.provider] = (
            self.provider_breakdown.get(record.provider, 0.0) + record.cost
        )
        self.model_breakdown[record.model] = (
            self.model_breakdown.get(record.model, 0.0) + record.cost
        )
        composite = f"{record.provider}/{record.model}"
        self.by_provider_model[composite] = (
            self.by_provider_model.get(composite, 0.0) + record.cost
        )

        self.records.append(record)

    def get_top_costs(self, n: int = 5) -> List[tuple]:
        # Use the disambiguated view so OpenAI gpt-4o and Azure gpt-4o aren't
        # collapsed; optimisation suggestions need to know which provider is
        # responsible for the spend.
        return sorted(self.by_provider_model.items(), key=lambda x: x[1], reverse=True)[:n]

    def to_dict(self) -> Dict:
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_calls": self.total_calls,
            "provider_breakdown": self.provider_breakdown,
            "model_breakdown": self.model_breakdown,
            "by_provider_model": self.by_provider_model,
            "budget_limit": self.budget_limit,
            "records": [r.to_dict() for r in self.records],
        }


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""

    def __init__(self, message: str, current_cost: float = 0.0, budget_limit: float = 0.0):
        super().__init__(message)
        self.current_cost = current_cost
        self.budget_limit = budget_limit


class CostTracker:
    """Track costs across all LLM providers."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.current_session = CostSummary()
        self.budget_limit: Optional[float] = None
        self.budget_warning_threshold: float = 0.8

    def _resolve_pricing(self, provider: str, model: str) -> Dict[str, float]:
        provider = provider.lower()
        model = model.lower()

        for table in (PROVIDER_PRICING, EMBEDDING_PRICING):
            if provider in table:
                provider_prices = table[provider]
                if model in provider_prices:
                    return provider_prices[model]
                for price_model, pricing in provider_prices.items():
                    if price_model in model or model in price_model:
                        return pricing

        return FALLBACK_PRICING

    def calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        pricing = self._resolve_pricing(provider, model)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def _enforce_budget(self) -> None:
        """Raise BudgetExceededError if the running total is over budget.

        Centralised here so any future code path that records usage stays
        consistent — call this before recording, never inline the check.
        """
        if self.budget_limit is None:
            return
        if self.current_session.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget limit ${self.budget_limit:.2f} exceeded. "
                f"Current cost: ${self.current_session.total_cost:.2f}",
                current_cost=self.current_session.total_cost,
                budget_limit=self.budget_limit,
            )

    def _maybe_warn_near_budget(self) -> None:
        """Emit a warning if we're past the warning threshold but still under.

        Matches the bracket of (warning_threshold * limit, limit). Outside
        that range the warning is irrelevant: under the threshold means we
        haven't gotten close yet; at-or-over the limit means _enforce_budget
        already raised.
        """
        if self.budget_limit is None:
            return
        running = self.current_session.total_cost
        if (
            running > self.budget_limit * self.budget_warning_threshold
            and running < self.budget_limit
        ):
            warnings.warn(
                f"Approaching budget limit: ${running:.2f} / ${self.budget_limit:.2f}"
            )

    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_type: str = "completion",
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        # Budget check happens *before* recording so the first call after
        # set_budget always succeeds, and the next call that finds the running
        # total already over the limit raises.
        self._enforce_budget()

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
            metadata=metadata or {},
        )
        self.current_session.add_record(record)

        self._maybe_warn_near_budget()

        return record

    def set_budget(self, limit: float, warning_threshold: float = 0.8) -> None:
        self.budget_limit = limit
        self.budget_warning_threshold = warning_threshold
        self.current_session.budget_limit = limit

    def get_summary(self) -> CostSummary:
        return self.current_session

    def reset_session(self) -> CostSummary:
        old_summary = self.current_session
        self.current_session = CostSummary(budget_limit=self.budget_limit)
        return old_summary

    def save_session(self, filepath: str) -> None:
        with open(filepath, "w") as f:
            json.dump(self.current_session.to_dict(), f, indent=2)

    def estimate_cost(
        self,
        provider: str,
        model: str,
        text: str,
        is_completion: bool = False,
    ) -> float:
        estimated_tokens = max(1, len(text) // 4)
        if is_completion:
            return self.calculate_cost(provider, model, 0, estimated_tokens)
        return self.calculate_cost(provider, model, estimated_tokens, 0)


# Cheaper alternatives suggested by the optimizer. Each entry is keyed by a
# substring of the current model and produces the recommended replacement.
_OPTIMIZER_SUGGESTIONS = [
    ("gpt-4", "gpt-4o-mini", 0.95, "Much cheaper for simple tasks"),
    ("claude-3-opus", "claude-3.5-sonnet", 0.80, "Comparable quality at lower cost"),
    ("gemini-1.5-pro", "gemini-1.5-flash", 0.90, "Faster and cheaper for most tasks"),
    ("mistral-large", "mistral-small", 0.70, "Cheaper for general tasks"),
]


class CostOptimizer:
    """Analyze costs and suggest optimizations."""

    def __init__(self, tracker: "CostTracker"):
        self.tracker = tracker
        self.summary = tracker.current_session

    def analyze(self) -> Dict:
        recommendations: List[Dict] = []
        potential_savings = 0.0

        top_costs = self.summary.get_top_costs(3)
        current_model = top_costs[0][0] if top_costs else None

        for model, cost in top_costs:
            for needle, replacement, factor, reason in _OPTIMIZER_SUGGESTIONS:
                if needle in model and replacement not in model:
                    savings = cost * factor
                    savings_percent = factor * 100
                    recommendations.append(
                        {
                            "current": model,
                            "suggestion": replacement,
                            "reason": reason,
                            "savings": savings,
                            "savings_percent": savings_percent,
                        }
                    )
                    potential_savings += savings
                    break

        return {
            "current_model": current_model,
            "total_cost": self.summary.total_cost,
            "potential_savings": potential_savings,
            "savings_percentage": (
                (potential_savings / self.summary.total_cost * 100)
                if self.summary.total_cost > 0
                else 0.0
            ),
            "recommendations": recommendations,
        }
