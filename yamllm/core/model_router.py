"""Intelligent model routing system."""

import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"  # Simple queries, basic math
    SIMPLE = "simple"  # Straightforward tasks
    MODERATE = "moderate"  # Requires some reasoning
    COMPLEX = "complex"  # Multi-step reasoning
    EXPERT = "expert"  # Requires specialized knowledge


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
    supports_tools: bool = True


# Model capability profiles
MODEL_PROFILES = {
    "openai/gpt-4": ModelCapability(
        provider="openai",
        model="gpt-4",
        strengths=[TaskType.REASONING, TaskType.EXPERT, TaskType.EXPLANATION],
        cost_tier=5,
        speed_tier=4,
        context_length=8192
    ),
    "openai/gpt-4o": ModelCapability(
        provider="openai",
        model="gpt-4o",
        strengths=[TaskType.REASONING, TaskType.CODE_GENERATION, TaskType.GENERAL],
        cost_tier=3,
        speed_tier=2,
        context_length=128000
    ),
    "openai/gpt-4o-mini": ModelCapability(
        provider="openai",
        model="gpt-4o-mini",
        strengths=[TaskType.GENERAL, TaskType.SUMMARIZATION, TaskType.TRANSLATION],
        cost_tier=1,
        speed_tier=1,
        context_length=128000
    ),
    "openai/gpt-3.5-turbo": ModelCapability(
        provider="openai",
        model="gpt-3.5-turbo",
        strengths=[TaskType.GENERAL, TaskType.SIMPLE],
        cost_tier=1,
        speed_tier=1,
        context_length=16385
    ),
    "anthropic/claude-3.5-sonnet": ModelCapability(
        provider="anthropic",
        model="claude-3.5-sonnet",
        strengths=[TaskType.CODE_GENERATION, TaskType.CODE_REVIEW, TaskType.REASONING],
        cost_tier=3,
        speed_tier=2,
        context_length=200000
    ),
    "anthropic/claude-3-opus": ModelCapability(
        provider="anthropic",
        model="claude-3-opus",
        strengths=[TaskType.EXPERT, TaskType.CREATIVE, TaskType.REASONING],
        cost_tier=5,
        speed_tier=4,
        context_length=200000
    ),
    "anthropic/claude-3-haiku": ModelCapability(
        provider="anthropic",
        model="claude-3-haiku",
        strengths=[TaskType.GENERAL, TaskType.SIMPLE],
        cost_tier=1,
        speed_tier=1,
        context_length=200000
    ),
    "google/gemini-1.5-pro": ModelCapability(
        provider="google",
        model="gemini-1.5-pro",
        strengths=[TaskType.REASONING, TaskType.EXPLANATION],
        cost_tier=2,
        speed_tier=2,
        context_length=1000000
    ),
    "google/gemini-1.5-flash": ModelCapability(
        provider="google",
        model="gemini-1.5-flash",
        strengths=[TaskType.GENERAL, TaskType.SIMPLE, TaskType.SUMMARIZATION],
        cost_tier=1,
        speed_tier=1,
        context_length=1000000
    ),
    "mistral/mistral-large": ModelCapability(
        provider="mistral",
        model="mistral-large",
        strengths=[TaskType.REASONING, TaskType.CODE_GENERATION],
        cost_tier=3,
        speed_tier=2,
        context_length=128000
    ),
}


class ModelRouter:
    """Intelligent model routing system."""

    def __init__(self, optimize_for: str = "balanced"):
        """
        Initialize router.

        Args:
            optimize_for: Optimization strategy (cost, speed, quality, balanced)
        """
        self.optimize_for = optimize_for
        self.usage_history: List[Dict] = []
        self.learning_enabled = False

    def analyze_task(self, prompt: str) -> Tuple[TaskType, TaskComplexity]:
        """
        Analyze task to determine type and complexity.

        Args:
            prompt: Task prompt

        Returns:
            (TaskType, TaskComplexity)
        """
        prompt_lower = prompt.lower()

        # Detect task type
        task_type = TaskType.GENERAL

        # Code-related keywords
        if any(word in prompt_lower for word in ["code", "function", "class", "implement", "refactor", "bug", "debug"]):
            if any(word in prompt_lower for word in ["review", "analyze", "check"]):
                task_type = TaskType.CODE_REVIEW
            elif any(word in prompt_lower for word in ["bug", "debug", "fix", "error"]):
                task_type = TaskType.DEBUGGING
            else:
                task_type = TaskType.CODE_GENERATION

        # Reasoning keywords
        elif any(word in prompt_lower for word in ["explain", "why", "how", "reason", "analyze"]):
            if any(word in prompt_lower for word in ["explain", "describe"]):
                task_type = TaskType.EXPLANATION
            else:
                task_type = TaskType.REASONING

        # Creative keywords
        elif any(word in prompt_lower for word in ["write", "create", "generate", "story", "poem"]):
            task_type = TaskType.CREATIVE

        # Translation keywords
        elif any(word in prompt_lower for word in ["translate", "translation"]):
            task_type = TaskType.TRANSLATION

        # Summarization keywords
        elif any(word in prompt_lower for word in ["summarize", "summary", "tldr", "brief"]):
            task_type = TaskType.SUMMARIZATION

        # Determine complexity
        complexity = TaskComplexity.SIMPLE

        # Length-based heuristics
        if len(prompt) > 500:
            complexity = TaskComplexity.COMPLEX
        elif len(prompt) > 200:
            complexity = TaskComplexity.MODERATE

        # Keyword-based complexity
        if any(word in prompt_lower for word in ["complex", "advanced", "detailed", "comprehensive"]):
            complexity = TaskComplexity.COMPLEX
        elif any(word in prompt_lower for word in ["multi-step", "several", "multiple"]):
            complexity = TaskComplexity.MODERATE
        elif any(word in prompt_lower for word in ["simple", "quick", "basic", "what is"]):
            complexity = TaskComplexity.SIMPLE

        # Expert-level indicators
        if any(word in prompt_lower for word in ["research", "scientific", "technical", "expert"]):
            complexity = TaskComplexity.EXPERT

        return task_type, complexity

    def select_model(
        self,
        prompt: str,
        available_providers: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        """
        Select best model for the task.

        Args:
            prompt: Task prompt
            available_providers: List of available providers (None = all)

        Returns:
            (provider, model, reasoning)
        """
        task_type, complexity = self.analyze_task(prompt)

        # Filter available models
        available_models = []
        for model_key, profile in MODEL_PROFILES.items():
            if available_providers and profile.provider not in available_providers:
                continue
            available_models.append((model_key, profile))

        if not available_models:
            # Default fallback
            return "openai", "gpt-4o-mini", "Default model"

        # Score models
        scored_models = []
        for model_key, profile in available_models:
            score = self._score_model(profile, task_type, complexity)
            scored_models.append((score, model_key, profile))

        # Sort by score (descending)
        scored_models.sort(reverse=True, key=lambda x: x[0])

        # Select best model
        best_score, best_model_key, best_profile = scored_models[0]

        # Generate reasoning
        reasoning = self._generate_reasoning(best_profile, task_type, complexity)

        return best_profile.provider, best_profile.model, reasoning

    def _score_model(
        self,
        profile: ModelCapability,
        task_type: TaskType,
        complexity: TaskComplexity
    ) -> float:
        """
        Score a model for the task.

        Args:
            profile: Model capability profile
            task_type: Type of task
            complexity: Task complexity

        Returns:
            Score (higher is better)
        """
        score = 0.0

        # Base score for strengths
        if task_type in profile.strengths:
            score += 50.0

        # Complexity matching
        if complexity == TaskComplexity.EXPERT and profile.cost_tier >= 4:
            score += 30.0
        elif complexity == TaskComplexity.COMPLEX and profile.cost_tier >= 3:
            score += 20.0
        elif complexity == TaskComplexity.MODERATE and profile.cost_tier >= 2:
            score += 15.0
        elif complexity == TaskComplexity.SIMPLE and profile.cost_tier <= 2:
            score += 25.0

        # Optimization strategy
        if self.optimize_for == "cost":
            # Prefer cheaper models
            score += (5 - profile.cost_tier) * 10
        elif self.optimize_for == "speed":
            # Prefer faster models
            score += (5 - profile.speed_tier) * 10
        elif self.optimize_for == "quality":
            # Prefer higher-tier models
            score += profile.cost_tier * 10
        else:  # balanced
            # Moderate preference for cost-effective models
            if profile.cost_tier <= 3:
                score += 5

        return score

    def _generate_reasoning(
        self,
        profile: ModelCapability,
        task_type: TaskType,
        complexity: TaskComplexity
    ) -> str:
        """Generate human-readable reasoning for model selection."""
        reasons = []

        # Strength match
        if task_type in profile.strengths:
            reasons.append(f"Excels at {task_type.value}")

        # Cost consideration
        if profile.cost_tier == 1:
            reasons.append("Most cost-effective")
        elif profile.cost_tier == 5:
            reasons.append("Premium model for best quality")

        # Speed consideration
        if profile.speed_tier == 1:
            reasons.append("Fastest response time")

        # Complexity match
        if complexity in [TaskComplexity.EXPERT, TaskComplexity.COMPLEX] and profile.cost_tier >= 4:
            reasons.append("Handles complex reasoning well")

        return "; ".join(reasons) if reasons else "Good general-purpose model"

    def enable_learning(self) -> None:
        """Enable learning from usage patterns."""
        self.learning_enabled = True

    def record_usage(
        self,
        provider: str,
        model: str,
        task_type: TaskType,
        success: bool,
        execution_time: float
    ) -> None:
        """
        Record usage for learning.

        Args:
            provider: Provider used
            model: Model used
            task_type: Type of task
            success: Whether task succeeded
            execution_time: How long it took
        """
        if not self.learning_enabled:
            return

        self.usage_history.append({
            "provider": provider,
            "model": model,
            "task_type": task_type.value,
            "success": success,
            "execution_time": execution_time
        })

    def get_recommendations(self) -> Dict:
        """Get recommendations based on usage history."""
        if not self.usage_history:
            return {"message": "Not enough usage data yet"}

        # Analyze patterns
        successful_models = {}
        for record in self.usage_history:
            if record["success"]:
                model_key = f"{record['provider']}/{record['model']}"
                task = record["task_type"]

                if task not in successful_models:
                    successful_models[task] = {}

                if model_key not in successful_models[task]:
                    successful_models[task][model_key] = 0

                successful_models[task][model_key] += 1

        # Generate recommendations
        recommendations = {}
        for task, models in successful_models.items():
            best_model = max(models.items(), key=lambda x: x[1])
            recommendations[task] = {
                "model": best_model[0],
                "success_count": best_model[1]
            }

        return recommendations
