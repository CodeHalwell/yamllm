"""Intelligent model routing system."""

import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TaskComplexity(Enum):
    """Task complexity levels."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class TaskType(Enum):
    """Types of tasks."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    CREATIVE = "creative"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENTATION = "documentation"
    Q_AND_A = "q_and_a"
    EXPERT = "expert"
    SIMPLE = "simple"
    GENERAL = "general"


@dataclass
class ModelCapability:
    """Model capability profile."""

    provider: str
    model: str
    strengths: List[TaskType]
    cost_tier: int  # 1=cheapest, 5=most expensive
    speed_tier: int  # 1=fastest, 5=slowest
    context_length: int
    weaknesses: List[TaskType] = field(default_factory=list)
    supports_tools: bool = True


MODEL_PROFILES: Dict[str, ModelCapability] = {
    "openai/gpt-4": ModelCapability(
        provider="openai",
        model="gpt-4",
        strengths=[TaskType.REASONING, TaskType.EXPERT, TaskType.EXPLANATION],
        weaknesses=[TaskType.SIMPLE, TaskType.SUMMARIZATION],
        cost_tier=5,
        speed_tier=4,
        context_length=8192,
    ),
    "openai/gpt-4o": ModelCapability(
        provider="openai",
        model="gpt-4o",
        strengths=[
            TaskType.REASONING,
            TaskType.CODE_GENERATION,
            TaskType.CODE_REVIEW,
            TaskType.DATA_ANALYSIS,
            TaskType.GENERAL,
        ],
        weaknesses=[],
        cost_tier=3,
        speed_tier=2,
        context_length=128000,
    ),
    "openai/o1": ModelCapability(
        provider="openai",
        model="o1",
        strengths=[TaskType.REASONING, TaskType.EXPERT, TaskType.DEBUGGING],
        weaknesses=[TaskType.SIMPLE, TaskType.TRANSLATION],
        cost_tier=4,
        speed_tier=4,
        context_length=200000,
    ),
    "openai/gpt-4o-mini": ModelCapability(
        provider="openai",
        model="gpt-4o-mini",
        strengths=[
            TaskType.GENERAL,
            TaskType.SIMPLE,
            TaskType.SUMMARIZATION,
            TaskType.TRANSLATION,
            TaskType.Q_AND_A,
        ],
        weaknesses=[TaskType.EXPERT],
        cost_tier=1,
        speed_tier=1,
        context_length=128000,
    ),
    "openai/gpt-3.5-turbo": ModelCapability(
        provider="openai",
        model="gpt-3.5-turbo",
        strengths=[TaskType.GENERAL, TaskType.SIMPLE, TaskType.Q_AND_A],
        weaknesses=[TaskType.EXPERT, TaskType.REASONING],
        cost_tier=1,
        speed_tier=1,
        context_length=16385,
    ),
    "anthropic/claude-3.5-sonnet": ModelCapability(
        provider="anthropic",
        model="claude-3.5-sonnet",
        strengths=[
            TaskType.CODE_GENERATION,
            TaskType.CODE_REVIEW,
            TaskType.REASONING,
            TaskType.DOCUMENTATION,
            TaskType.DEBUGGING,
        ],
        weaknesses=[],
        cost_tier=3,
        speed_tier=2,
        context_length=200000,
    ),
    "anthropic/claude-3-opus": ModelCapability(
        provider="anthropic",
        model="claude-3-opus",
        strengths=[TaskType.EXPERT, TaskType.CREATIVE, TaskType.REASONING],
        weaknesses=[TaskType.SIMPLE],
        cost_tier=5,
        speed_tier=4,
        context_length=200000,
    ),
    "anthropic/claude-3-haiku": ModelCapability(
        provider="anthropic",
        model="claude-3-haiku",
        strengths=[
            TaskType.GENERAL,
            TaskType.SIMPLE,
            TaskType.SUMMARIZATION,
            TaskType.Q_AND_A,
        ],
        weaknesses=[TaskType.EXPERT],
        cost_tier=1,
        speed_tier=1,
        context_length=200000,
    ),
    "google/gemini-1.5-pro": ModelCapability(
        provider="google",
        model="gemini-1.5-pro",
        strengths=[TaskType.REASONING, TaskType.EXPLANATION, TaskType.DATA_ANALYSIS],
        weaknesses=[],
        cost_tier=2,
        speed_tier=2,
        context_length=1000000,
    ),
    "google/gemini-1.5-flash": ModelCapability(
        provider="google",
        model="gemini-1.5-flash",
        strengths=[
            TaskType.GENERAL,
            TaskType.SIMPLE,
            TaskType.SUMMARIZATION,
            TaskType.Q_AND_A,
        ],
        weaknesses=[TaskType.EXPERT],
        cost_tier=1,
        speed_tier=1,
        context_length=1000000,
    ),
    "mistral/mistral-large": ModelCapability(
        provider="mistral",
        model="mistral-large",
        strengths=[TaskType.REASONING, TaskType.CODE_GENERATION],
        weaknesses=[],
        cost_tier=3,
        speed_tier=2,
        context_length=128000,
    ),
}


# Ordered list of (TaskType, keywords) — first match wins so put the more
# specific patterns ahead of the general ones.
_TASK_TYPE_PATTERNS: List[Tuple[TaskType, Tuple[str, ...]]] = [
    (TaskType.CODE_REVIEW, ("review", "audit", "lint")),
    (
        TaskType.DEBUGGING,
        ("debug", "fix bug", "fix error", "stack trace", "traceback", "error:"),
    ),
    (
        TaskType.DATA_ANALYSIS,
        ("analyze data", "dataset", "trends", "statistics", "data analysis"),
    ),
    (
        TaskType.DOCUMENTATION,
        ("docs", "documentation", "readme", "doc string", "docstring"),
    ),
    (TaskType.Q_AND_A, ("answer question", "question:", "answer:")),
    (
        TaskType.CODE_GENERATION,
        (
            "write code",
            "code",
            "function",
            "class ",
            "implement",
            "refactor",
            "script",
            "program",
        ),
    ),
    (TaskType.TRANSLATION, ("translate", "translation")),
    (TaskType.SUMMARIZATION, ("summarize", "summary", "tldr", "tl;dr")),
    (
        TaskType.REASONING,
        ("explain", "why", "reason", "philosophical", "implication", "prove"),
    ),
    (TaskType.CREATIVE, ("story", "poem", "creative", "imagine")),
]

# Keywords that promote complexity. Order: most-specific first.
_COMPLEXITY_KEYWORDS: List[Tuple[TaskComplexity, Tuple[str, ...]]] = [
    (
        TaskComplexity.EXPERT,
        (
            "prove",
            "theorem",
            "hypothesis",
            "quantum",
            "comprehensive",
            "framework",
            "research",
            "scientific",
            "expert",
            "novel",
            "phd",
            "philosophical",
        ),
    ),
    (
        TaskComplexity.COMPLEX,
        (
            "complex",
            "distributed",
            "architecture",
            "fault tolerance",
            "high availability",
            "advanced",
            "comprehensive",
        ),
    ),
    (
        TaskComplexity.MODERATE,
        ("multi-step", "several", "multiple", "moderate"),
    ),
    (
        TaskComplexity.SIMPLE,
        ("simple", "quick", "basic"),
    ),
]

# Trivial-task patterns: only match very short, low-information prompts.
_TRIVIAL_PATTERNS = (
    "hello",
    "hi ",
    "hey",
    "your name",
    "say hello",
    "what is 2+2",
    "what is 1+1",
)


class ModelRouter:
    """Intelligent model routing system."""

    def __init__(
        self,
        optimize_for: str = "balanced",
        logger: Optional[logging.Logger] = None,
    ):
        self.optimize_for = optimize_for
        self.logger = logger or logging.getLogger(__name__)
        self.usage_history: List[Dict] = []
        self.learning_enabled = False

    def analyze_task(self, prompt: str) -> Tuple[TaskType, TaskComplexity]:
        prompt_lower = prompt.lower().strip()

        task_type = TaskType.GENERAL
        for candidate_type, keywords in _TASK_TYPE_PATTERNS:
            if any(kw in prompt_lower for kw in keywords):
                task_type = candidate_type
                break

        complexity = self._detect_complexity(prompt_lower, prompt)
        return task_type, complexity

    def _detect_complexity(self, prompt_lower: str, prompt_raw: str) -> TaskComplexity:
        # Trivial: short, formulaic
        if len(prompt_lower) <= 30 and any(
            p in prompt_lower for p in _TRIVIAL_PATTERNS
        ):
            return TaskComplexity.TRIVIAL

        # Highest-priority keyword wins
        for complexity, keywords in _COMPLEXITY_KEYWORDS:
            if any(kw in prompt_lower for kw in keywords):
                return complexity

        # Length-based fallback
        if len(prompt_raw) > 500:
            return TaskComplexity.COMPLEX
        if len(prompt_raw) > 200:
            return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE

    def select_model(
        self,
        prompt: str,
        available_providers: Optional[List[str]] = None,
    ) -> Tuple[str, str, str]:
        task_type, complexity = self.analyze_task(prompt)

        candidates = [
            (model_key, profile)
            for model_key, profile in MODEL_PROFILES.items()
            if not available_providers or profile.provider in available_providers
        ]

        if not candidates:
            provider, model, reasoning = "openai", "gpt-4o-mini", "Default model"
        else:
            scored = sorted(
                (
                    (self._score_model(profile, task_type, complexity), key, profile)
                    for key, profile in candidates
                ),
                reverse=True,
                key=lambda triple: triple[0],
            )
            _, _, best = scored[0]
            provider, model = best.provider, best.model
            reasoning = self._generate_reasoning(best, task_type, complexity)

        self.usage_history.append(
            {
                "prompt": prompt,
                "selected_model": f"{provider}/{model}",
                "task_type": task_type.value,
                "complexity": complexity.value,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return provider, model, reasoning

    def _score_model(
        self,
        profile: ModelCapability,
        task_type: TaskType,
        complexity: TaskComplexity,
    ) -> float:
        score = 0.0

        if task_type in profile.strengths:
            score += 50.0
        if task_type in profile.weaknesses:
            score -= 30.0

        # Complexity → cost_tier preference
        if complexity == TaskComplexity.EXPERT:
            if profile.cost_tier >= 4:
                score += 40.0
            elif profile.cost_tier == 3:
                score += 15.0
        elif complexity == TaskComplexity.COMPLEX:
            # Prefer mid-tier (3-4) for COMPLEX, penalize cost_tier=5
            # so we don't always max out on the most expensive model.
            if profile.cost_tier == 3:
                score += 30.0
            elif profile.cost_tier == 4:
                score += 25.0
            elif profile.cost_tier == 2:
                score += 15.0
            elif profile.cost_tier == 5:
                score -= 5.0
        elif complexity == TaskComplexity.MODERATE:
            if profile.cost_tier in (2, 3):
                score += 20.0
        elif complexity == TaskComplexity.SIMPLE:
            if profile.cost_tier <= 2:
                score += 30.0
            elif profile.cost_tier >= 4:
                score -= 15.0
        elif complexity == TaskComplexity.TRIVIAL:
            if profile.cost_tier == 1:
                score += 40.0
            else:
                score -= 10.0

        # Optimization strategy
        if self.optimize_for == "cost":
            score += (5 - profile.cost_tier) * 12
        elif self.optimize_for == "speed":
            score += (5 - profile.speed_tier) * 12
        elif self.optimize_for == "quality":
            score += profile.cost_tier * 8
        else:  # balanced
            if profile.cost_tier <= 3:
                score += 5.0

        return score

    def _generate_reasoning(
        self,
        profile: ModelCapability,
        task_type: TaskType,
        complexity: TaskComplexity,
    ) -> str:
        parts: List[str] = []

        if complexity == TaskComplexity.SIMPLE:
            parts.append("Simple task — using a cost-effective model")
        elif complexity == TaskComplexity.TRIVIAL:
            parts.append("Basic task — using the smallest fast model")
        elif complexity == TaskComplexity.EXPERT:
            parts.append("Expert-level task — using a premium model")
        elif complexity == TaskComplexity.COMPLEX:
            parts.append("Complex task — using a capable mid-tier model")

        if task_type in profile.strengths:
            parts.append(f"Excels at {task_type.value}")
        if profile.cost_tier == 1:
            parts.append("Most cost-effective")
        if profile.speed_tier == 1:
            parts.append("Fastest response time")

        return "; ".join(parts) if parts else "Good general-purpose model"

    def enable_learning(self) -> None:
        self.learning_enabled = True

    def record_usage(
        self,
        provider: str,
        model: str,
        task_type: TaskType,
        success: bool,
        execution_time: float,
    ) -> None:
        if not self.learning_enabled:
            return

        self.usage_history.append(
            {
                "provider": provider,
                "model": model,
                "task_type": task_type.value,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_recommendations(self) -> Dict:
        if not self.usage_history:
            return {"message": "Not enough usage data yet"}

        successful_models: Dict[str, Dict[str, int]] = {}
        for record in self.usage_history:
            if not record.get("success"):
                continue
            task = record.get("task_type", "unknown")
            model_key = f"{record.get('provider')}/{record.get('model')}"
            successful_models.setdefault(task, {}).setdefault(model_key, 0)
            successful_models[task][model_key] += 1

        recommendations = {}
        for task, models in successful_models.items():
            best_model, count = max(models.items(), key=lambda x: x[1])
            recommendations[task] = {"model": best_model, "success_count": count}
        return recommendations
