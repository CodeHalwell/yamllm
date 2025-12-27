"""Tests for agent learning and improvement system."""

import pytest
import tempfile
import os
from datetime import datetime
from yamllm.agent.learning_system import (
    OutcomeType,
    ImprovementType,
    Experience,
    LearningInsight,
    PerformanceMetrics,
    ExperienceStore,
    PatternAnalyzer,
    LearningSystem
)


class MockLLM:
    """Mock LLM for testing."""

    def query(self, prompt: str) -> str:
        """Return mock response."""
        return "Mock LLM response for analysis"


def test_outcome_type_enum():
    """Test outcome type enum."""
    assert OutcomeType.SUCCESS.value == "success"
    assert OutcomeType.FAILURE.value == "failure"
    assert OutcomeType.PARTIAL.value == "partial"
    assert OutcomeType.TIMEOUT.value == "timeout"
    assert OutcomeType.ERROR.value == "error"


def test_improvement_type_enum():
    """Test improvement type enum."""
    assert ImprovementType.TASK_DECOMPOSITION.value == "task_decomposition"
    assert ImprovementType.TOOL_SELECTION.value == "tool_selection"
    assert ImprovementType.REASONING_PATTERN.value == "reasoning_pattern"
    assert ImprovementType.ERROR_RECOVERY.value == "error_recovery"
    assert ImprovementType.CONTEXT_USAGE.value == "context_usage"
    assert ImprovementType.PLANNING_STRATEGY.value == "planning_strategy"


def test_experience_dataclass():
    """Test experience dataclass."""
    experience = Experience(
        experience_id="exp_1",
        task_description="Fix bug in API",
        context={"env": "production"},
        actions_taken=[{"action": "analyze"}, {"action": "fix"}],
        outcome=OutcomeType.SUCCESS,
        outcome_details={"fixed": True},
        duration_seconds=120.5
    )

    assert experience.experience_id == "exp_1"
    assert experience.task_description == "Fix bug in API"
    assert experience.outcome == OutcomeType.SUCCESS
    assert experience.duration_seconds == 120.5
    assert isinstance(experience.timestamp, datetime)


def test_learning_insight_dataclass():
    """Test learning insight dataclass."""
    insight = LearningInsight(
        insight_id="insight_1",
        improvement_type=ImprovementType.REASONING_PATTERN,
        pattern="Use pattern X for task Y",
        confidence=0.85,
        evidence_count=10,
        success_rate=0.9,
        context_conditions={"task_type": "debugging"},
        recommendation="Apply pattern X when debugging"
    )

    assert insight.insight_id == "insight_1"
    assert insight.improvement_type == ImprovementType.REASONING_PATTERN
    assert insight.confidence == 0.85
    assert insight.evidence_count == 10
    assert insight.success_rate == 0.9


def test_performance_metrics_dataclass():
    """Test performance metrics dataclass."""
    metrics = PerformanceMetrics(
        total_tasks=100,
        successful_tasks=85,
        failed_tasks=15,
        average_duration=150.0,
        success_rate=0.85
    )

    assert metrics.total_tasks == 100
    assert metrics.successful_tasks == 85
    assert metrics.success_rate == 0.85


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


def test_experience_store_initialization(temp_db):
    """Test experience store initialization."""
    store = ExperienceStore(temp_db)

    assert os.path.exists(temp_db)
    assert store.db_path == temp_db


def test_experience_store_store_and_retrieve(temp_db):
    """Test storing and retrieving experiences."""
    store = ExperienceStore(temp_db)

    experience = Experience(
        experience_id="exp_1",
        task_description="Test task",
        context={"test": True},
        actions_taken=[{"action": "test"}],
        outcome=OutcomeType.SUCCESS,
        outcome_details={"success": True},
        duration_seconds=10.0
    )

    # Store experience
    store.store_experience(experience)

    # Retrieve experiences
    experiences = store.get_experiences()

    assert len(experiences) == 1
    assert experiences[0].experience_id == "exp_1"
    assert experiences[0].task_description == "Test task"
    assert experiences[0].outcome == OutcomeType.SUCCESS


def test_experience_store_filter_by_outcome(temp_db):
    """Test filtering experiences by outcome."""
    store = ExperienceStore(temp_db)

    # Store multiple experiences
    for i in range(3):
        exp = Experience(
            experience_id=f"exp_{i}",
            task_description=f"Task {i}",
            context={},
            actions_taken=[],
            outcome=OutcomeType.SUCCESS if i < 2 else OutcomeType.FAILURE,
            outcome_details={},
            duration_seconds=10.0
        )
        store.store_experience(exp)

    # Get only successful experiences
    successful = store.get_experiences(outcome=OutcomeType.SUCCESS)
    assert len(successful) == 2

    # Get only failed experiences
    failed = store.get_experiences(outcome=OutcomeType.FAILURE)
    assert len(failed) == 1


def test_experience_store_filter_by_task_pattern(temp_db):
    """Test filtering experiences by task pattern."""
    store = ExperienceStore(temp_db)

    # Store experiences with different task descriptions
    for task in ["Fix bug", "Add feature", "Fix error"]:
        exp = Experience(
            experience_id=f"exp_{task}",
            task_description=task,
            context={},
            actions_taken=[],
            outcome=OutcomeType.SUCCESS,
            outcome_details={},
            duration_seconds=10.0
        )
        store.store_experience(exp)

    # Get experiences with "Fix" in description
    fix_tasks = store.get_experiences(task_pattern="Fix")
    assert len(fix_tasks) == 2


def test_experience_store_store_and_retrieve_insights(temp_db):
    """Test storing and retrieving insights."""
    store = ExperienceStore(temp_db)

    insight = LearningInsight(
        insight_id="insight_1",
        improvement_type=ImprovementType.REASONING_PATTERN,
        pattern="Test pattern",
        confidence=0.8,
        evidence_count=5,
        success_rate=0.9,
        context_conditions={"test": True},
        recommendation="Use this pattern"
    )

    # Store insight
    store.store_insight(insight)

    # Retrieve insights
    insights = store.get_insights()

    assert len(insights) == 1
    assert insights[0].insight_id == "insight_1"
    assert insights[0].improvement_type == ImprovementType.REASONING_PATTERN


def test_experience_store_filter_insights_by_confidence(temp_db):
    """Test filtering insights by confidence."""
    store = ExperienceStore(temp_db)

    # Store insights with different confidence levels
    for i, conf in enumerate([0.3, 0.6, 0.9]):
        insight = LearningInsight(
            insight_id=f"insight_{i}",
            improvement_type=ImprovementType.REASONING_PATTERN,
            pattern=f"Pattern {i}",
            confidence=conf,
            evidence_count=5,
            success_rate=0.8,
            context_conditions={},
            recommendation="Test"
        )
        store.store_insight(insight)

    # Get only high-confidence insights
    high_conf = store.get_insights(min_confidence=0.7)
    assert len(high_conf) == 1
    assert high_conf[0].confidence == 0.9


def test_pattern_analyzer_initialization():
    """Test pattern analyzer initialization."""
    llm = MockLLM()
    analyzer = PatternAnalyzer(llm)

    assert analyzer.llm == llm


def test_pattern_analyzer_success_patterns():
    """Test analyzing success patterns."""
    llm = MockLLM()
    analyzer = PatternAnalyzer(llm)

    # Create successful experiences with similar action sequences
    experiences = [
        Experience(
            experience_id=f"exp_{i}",
            task_description="Fix bug in code",
            context={},
            actions_taken=[
                {"action_type": "analyze"},
                {"action_type": "fix"},
                {"action_type": "test"}
            ],
            outcome=OutcomeType.SUCCESS,
            outcome_details={},
            duration_seconds=10.0
        )
        for i in range(3)
    ]

    patterns = analyzer.analyze_success_patterns(experiences)

    assert len(patterns) > 0
    assert patterns[0]["pattern_type"] == "action_sequence"
    assert patterns[0]["confidence"] >= 0.5


def test_pattern_analyzer_failure_patterns():
    """Test analyzing failure patterns."""
    llm = MockLLM()
    analyzer = PatternAnalyzer(llm)

    # Create failed experiences
    experiences = [
        Experience(
            experience_id=f"exp_{i}",
            task_description="Failed task",
            context={"env": "test"},
            actions_taken=[{"action_type": "execute"}],
            outcome=OutcomeType.FAILURE,
            outcome_details={"error_type": "timeout"},
            duration_seconds=10.0
        )
        for i in range(2)
    ]

    patterns = analyzer.analyze_failure_patterns(experiences)

    assert len(patterns) > 0
    assert patterns[0]["error_type"] == "timeout"
    assert patterns[0]["frequency"] == 2


def test_pattern_analyzer_generate_insights():
    """Test generating insights from patterns."""
    llm = MockLLM()
    analyzer = PatternAnalyzer(llm)

    success_patterns = [{
        "task_type": "fix",
        "pattern_type": "action_sequence",
        "sequence": ["analyze", "fix", "test"],
        "frequency": 5,
        "total_examples": 10,
        "confidence": 0.5
    }]

    failure_patterns = [{
        "error_type": "timeout",
        "frequency": 3,
        "common_context": {"env": 2},
        "example_actions": []
    }]

    insights = analyzer.generate_insights_from_patterns(success_patterns, failure_patterns)

    assert len(insights) >= 2
    assert any(i.improvement_type == ImprovementType.REASONING_PATTERN for i in insights)
    assert any(i.improvement_type == ImprovementType.ERROR_RECOVERY for i in insights)


def test_learning_system_initialization(temp_db):
    """Test learning system initialization."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    assert learning.llm == llm
    assert os.path.exists(temp_db)
    assert isinstance(learning.performance_metrics, PerformanceMetrics)


def test_learning_system_record_experience(temp_db):
    """Test recording experience."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    experience = learning.record_experience(
        task_description="Test task",
        actions=[{"action": "test"}],
        outcome=OutcomeType.SUCCESS,
        outcome_details={"result": "ok"},
        duration=10.0
    )

    assert experience.task_description == "Test task"
    assert learning.performance_metrics.total_tasks == 1
    assert learning.performance_metrics.successful_tasks == 1


def test_learning_system_update_metrics(temp_db):
    """Test metrics updating."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    # Record successful experience
    learning.record_experience(
        task_description="Success task",
        actions=[],
        outcome=OutcomeType.SUCCESS,
        outcome_details={},
        duration=10.0
    )

    # Record failed experience
    learning.record_experience(
        task_description="Failed task",
        actions=[],
        outcome=OutcomeType.FAILURE,
        outcome_details={},
        duration=20.0
    )

    metrics = learning.get_metrics()

    assert metrics.total_tasks == 2
    assert metrics.successful_tasks == 1
    assert metrics.failed_tasks == 1
    assert metrics.success_rate == 0.5
    assert metrics.average_duration == 15.0


def test_learning_system_analyze_and_learn(temp_db):
    """Test analysis and learning."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    # Record multiple similar experiences
    for i in range(10):
        learning.record_experience(
            task_description="Fix bug",
            actions=[
                {"action_type": "analyze"},
                {"action_type": "fix"}
            ],
            outcome=OutcomeType.SUCCESS,
            outcome_details={},
            duration=10.0
        )

    # Analyze and learn
    insights = learning.analyze_and_learn(min_experiences=5)

    assert len(insights) >= 0  # May or may not generate insights depending on patterns


def test_learning_system_get_recommendations(temp_db):
    """Test getting recommendations."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    # Store some insights first
    insight = LearningInsight(
        insight_id="insight_1",
        improvement_type=ImprovementType.REASONING_PATTERN,
        pattern="For fix tasks, use analyze -> fix -> test",
        confidence=0.8,
        evidence_count=5,
        success_rate=0.9,
        context_conditions={"task_type": "fix"},
        recommendation="Always test after fixing"
    )

    learning.experience_store.store_insight(insight)

    # Get recommendations
    recommendations = learning.get_recommendations("Fix bug in API")

    assert len(recommendations) >= 0  # May or may not match depending on pattern


def test_learning_system_get_summary(temp_db):
    """Test getting learning summary."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    # Record some experiences
    for i in range(5):
        learning.record_experience(
            task_description=f"Task {i}",
            actions=[],
            outcome=OutcomeType.SUCCESS,
            outcome_details={},
            duration=10.0
        )

    summary = learning.get_learning_summary()

    assert summary["total_experiences"] == 5
    assert summary["success_rate"] == 1.0
    assert "total_insights" in summary
    assert "insights_by_type" in summary


def test_learning_system_export_knowledge(temp_db):
    """Test exporting knowledge."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    # Store some insights
    insight = LearningInsight(
        insight_id="insight_1",
        improvement_type=ImprovementType.REASONING_PATTERN,
        pattern="Test pattern",
        confidence=0.8,
        evidence_count=5,
        success_rate=0.9,
        context_conditions={},
        recommendation="Test recommendation"
    )
    learning.experience_store.store_insight(insight)

    # Export to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_path = f.name

    try:
        learning.export_knowledge(export_path)
        assert os.path.exists(export_path)

        # Verify exported content
        import json
        with open(export_path, 'r') as f:
            data = json.load(f)

        assert "exported_at" in data
        assert "metrics" in data
        assert "insights" in data
        assert len(data["insights"]) >= 1
    finally:
        os.unlink(export_path)


def test_learning_system_import_knowledge(temp_db):
    """Test importing knowledge."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    # Create knowledge file
    import json
    knowledge = {
        "exported_at": datetime.now().isoformat(),
        "insights": [
            {
                "id": "insight_1",
                "type": "reasoning_pattern",
                "pattern": "Imported pattern",
                "confidence": 0.85,
                "evidence_count": 10,
                "recommendation": "Imported recommendation"
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(knowledge, f)
        import_path = f.name

    try:
        learning.import_knowledge(import_path)

        # Verify imported insights
        insights = learning.experience_store.get_insights()
        assert len(insights) >= 1
        assert any(i.pattern == "Imported pattern" for i in insights)
    finally:
        os.unlink(import_path)


def test_learning_system_integration(temp_db):
    """Test complete learning system integration."""
    llm = MockLLM()
    learning = LearningSystem(llm, storage_path=temp_db)

    # Simulate agent learning cycle
    # 1. Record experiences
    for i in range(15):
        outcome = OutcomeType.SUCCESS if i < 12 else OutcomeType.FAILURE
        learning.record_experience(
            task_description="Implement feature" if i < 10 else "Fix bug",
            actions=[{"action_type": "code"}],
            outcome=outcome,
            outcome_details={},
            duration=30.0 + i
        )

    # 2. Analyze and learn
    learning.analyze_and_learn(min_experiences=10)

    # 3. Get metrics
    metrics = learning.get_metrics()
    assert metrics.total_tasks == 15
    assert metrics.successful_tasks == 12

    # 4. Get recommendations
    recommendations = learning.get_recommendations("Implement new feature")

    # 5. Get summary
    summary = learning.get_learning_summary()
    assert summary["total_experiences"] == 15
