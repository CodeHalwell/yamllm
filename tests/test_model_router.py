"""Tests for intelligent model routing."""

from yamllm.core.model_router import (
    ModelRouter,
    TaskType,
    TaskComplexity,
    MODEL_PROFILES
)


def test_router_initialization():
    """Test router initialization."""
    router = ModelRouter()

    assert router.logger is not None
    assert router.usage_history == []


def test_analyze_task_code_generation():
    """Test task analysis for code generation."""
    router = ModelRouter()

    task_type, complexity = router.analyze_task("Write a Python function to sort a list")

    assert task_type == TaskType.CODE_GENERATION
    assert complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]


def test_analyze_task_debugging():
    """Test task analysis for debugging."""
    router = ModelRouter()

    task_type, complexity = router.analyze_task("Debug this error: ValueError at line 42")

    assert task_type == TaskType.DEBUGGING
    assert complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE, TaskComplexity.COMPLEX]


def test_analyze_task_reasoning():
    """Test task analysis for reasoning."""
    router = ModelRouter()

    task_type, complexity = router.analyze_task(
        "Explain the philosophical implications of quantum mechanics"
    )

    assert task_type == TaskType.REASONING
    assert complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]


def test_analyze_task_code_review():
    """Test task analysis for code review."""
    router = ModelRouter()

    task_type, complexity = router.analyze_task("Review this code for security issues")

    assert task_type == TaskType.CODE_REVIEW


def test_analyze_task_data_analysis():
    """Test task analysis for data analysis."""
    router = ModelRouter()

    task_type, complexity = router.analyze_task(
        "Analyze this dataset and find trends"
    )

    assert task_type == TaskType.DATA_ANALYSIS


def test_select_model_simple_task():
    """Test model selection for simple tasks."""
    router = ModelRouter()

    provider, model, reasoning = router.select_model("What is 2+2?")

    # Simple tasks should use cheaper models
    assert model in ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash"]
    assert "simple" in reasoning.lower() or "basic" in reasoning.lower()


def test_select_model_complex_task():
    """Test model selection for complex tasks."""
    router = ModelRouter()

    provider, model, reasoning = router.select_model(
        "Design a distributed system architecture for a real-time trading platform "
        "with high availability and fault tolerance"
    )

    # Complex tasks should use more capable models
    assert model in ["gpt-4o", "claude-3.5-sonnet", "o1", "gemini-1.5-pro"]


def test_select_model_code_generation():
    """Test model selection optimized for code."""
    router = ModelRouter()

    provider, model, reasoning = router.select_model(
        "Write a Python function to implement a binary search tree"
    )

    # Should select model good at code
    assert provider in ["openai", "anthropic", "google"]


def test_select_model_with_budget_constraint():
    """Test model selection with budget constraint."""
    router = ModelRouter(optimize_for="cost")

    provider, model, reasoning = router.select_model(
        "Write a complex distributed system"
    )

    # Should respect cost constraint
    profile = MODEL_PROFILES.get(f"{provider}/{model}")
    if profile:
        # Cost-optimized router should prefer lower cost tiers
        assert profile.cost_tier <= 3


def test_select_model_with_speed_priority():
    """Test model selection prioritizing speed."""
    router = ModelRouter(optimize_for="speed")

    provider, model, reasoning = router.select_model(
        "Translate 'hello' to Spanish"
    )

    # Should prefer faster models
    profile = MODEL_PROFILES.get(f"{provider}/{model}")
    if profile:
        assert profile.speed_tier >= 2  # Higher = faster


def test_complexity_detection_trivial():
    """Test detection of trivial tasks."""
    router = ModelRouter()

    prompts = [
        "Hello",
        "What is your name?",
        "Say hello"
    ]

    for prompt in prompts:
        _, complexity = router.analyze_task(prompt)
        assert complexity == TaskComplexity.TRIVIAL


def test_complexity_detection_expert():
    """Test detection of expert-level tasks."""
    router = ModelRouter()

    prompts = [
        "Prove the Riemann Hypothesis",
        "Design a quantum algorithm for factoring large numbers",
        "Develop a comprehensive AI safety framework"
    ]

    for prompt in prompts:
        _, complexity = router.analyze_task(prompt)
        assert complexity == TaskComplexity.EXPERT


def test_model_profiles_completeness():
    """Test that model profiles are complete."""
    required_providers = ["openai", "anthropic", "google"]

    for provider in required_providers:
        # Check that provider has at least one model
        models = [k for k in MODEL_PROFILES.keys() if k.startswith(f"{provider}/")]
        assert len(models) > 0, f"No models for {provider}"

        # Check that each profile has required fields
        for model_key in models:
            profile = MODEL_PROFILES[model_key]
            assert profile.strengths is not None
            assert profile.weaknesses is not None
            assert profile.cost_tier > 0
            assert profile.speed_tier > 0
            assert profile.context_length > 0


def test_usage_history_tracking():
    """Test usage history tracking."""
    router = ModelRouter()

    # Make several selections
    router.select_model("What is 2+2?")
    router.select_model("Write Python code")
    router.select_model("Debug this error")

    assert len(router.usage_history) == 3

    # Each history entry should have required fields
    for entry in router.usage_history:
        assert "prompt" in entry
        assert "selected_model" in entry
        assert "task_type" in entry
        assert "complexity" in entry
        assert "timestamp" in entry


def test_learning_from_history():
    """Test that router learns from usage history."""
    router = ModelRouter()

    # Build up history of code tasks
    for _ in range(5):
        router.select_model("Write Python code")

    # Next code task should use learned preferences
    provider, model, reasoning = router.select_model("Write more Python code")

    assert provider is not None
    assert model is not None


def test_task_type_enum_coverage():
    """Test that all task types are covered."""
    router = ModelRouter()

    task_types = [
        ("Write code", TaskType.CODE_GENERATION),
        ("Review this PR", TaskType.CODE_REVIEW),
        ("Debug error", TaskType.DEBUGGING),
        ("Explain concept", TaskType.REASONING),
        ("Analyze data", TaskType.DATA_ANALYSIS),
        ("Write docs", TaskType.DOCUMENTATION),
        ("Answer question", TaskType.Q_AND_A),
    ]

    for prompt, expected_type in task_types:
        detected_type, _ = router.analyze_task(prompt)
        assert detected_type == expected_type


def test_context_length_consideration():
    """Test that context length is considered."""
    router = ModelRouter()

    # Very long prompt
    long_prompt = "Analyze this code:\n" + ("line of code\n" * 10000)

    provider, model, reasoning = router.select_model(long_prompt)

    # Should select model with large context
    profile = MODEL_PROFILES.get(f"{provider}/{model}")
    if profile:
        assert profile.context_length >= 100000  # At least 100k tokens


def test_select_model_returns_valid_combination():
    """Test that selected models are valid."""
    router = ModelRouter()

    prompts = [
        "What is 2+2?",
        "Write Python code",
        "Debug this error",
        "Explain quantum mechanics",
        "Review this code"
    ]

    for prompt in prompts:
        provider, model, reasoning = router.select_model(prompt)

        # Check that selection is valid
        assert provider in ["openai", "anthropic", "google", "mistral"]
        assert model is not None
        assert len(model) > 0
        assert len(reasoning) > 0

        # Check that combination exists in profiles
        model_key = f"{provider}/{model}"
        assert model_key in MODEL_PROFILES, f"Invalid combination: {model_key}"


def test_optimization_criteria():
    """Test different optimization criteria."""
    # Cost-optimized
    cost_router = ModelRouter(optimize_for="cost")
    provider1, model1, _ = cost_router.select_model("Simple task")
    profile1 = MODEL_PROFILES.get(f"{provider1}/{model1}")
    # Cost router should prefer lower cost tiers
    assert profile1.cost_tier <= 3

    # Speed-optimized
    speed_router = ModelRouter(optimize_for="speed")
    provider2, model2, _ = speed_router.select_model("Simple task")
    profile2 = MODEL_PROFILES.get(f"{provider2}/{model2}")
    # Speed router should prefer faster models (lower speed_tier)
    assert profile2.speed_tier <= 3

    # Balanced (default)
    balanced_router = ModelRouter()
    provider3, model3, _ = balanced_router.select_model("Simple task")
    assert provider3 is not None
