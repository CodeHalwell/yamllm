"""Agent learning and improvement system for continuous optimization."""

import logging
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from enum import Enum
import sqlite3


class OutcomeType(Enum):
    """Types of task outcomes."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


class ImprovementType(Enum):
    """Types of improvements that can be learned."""
    TASK_DECOMPOSITION = "task_decomposition"
    TOOL_SELECTION = "tool_selection"
    REASONING_PATTERN = "reasoning_pattern"
    ERROR_RECOVERY = "error_recovery"
    CONTEXT_USAGE = "context_usage"
    PLANNING_STRATEGY = "planning_strategy"


@dataclass
class Experience:
    """Represents a single agent experience."""
    experience_id: str
    task_description: str
    context: Dict[str, Any]
    actions_taken: List[Dict[str, Any]]
    outcome: OutcomeType
    outcome_details: Dict[str, Any]
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    agent_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningInsight:
    """Insight learned from experiences."""
    insight_id: str
    improvement_type: ImprovementType
    pattern: str
    confidence: float  # 0-1
    evidence_count: int
    success_rate: float
    context_conditions: Dict[str, Any]
    recommendation: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking improvement."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_duration: float = 0.0
    success_rate: float = 0.0
    most_common_errors: List[Tuple[str, int]] = field(default_factory=list)
    improvement_over_time: List[Tuple[datetime, float]] = field(default_factory=list)


class ExperienceStore:
    """Persistent storage for agent experiences."""

    def __init__(self, db_path: str = "agent_experiences.db"):
        """
        Initialize experience store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Experiences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                experience_id TEXT PRIMARY KEY,
                task_description TEXT NOT NULL,
                context TEXT,
                actions_taken TEXT,
                outcome TEXT NOT NULL,
                outcome_details TEXT,
                duration_seconds REAL,
                timestamp TEXT,
                agent_state TEXT,
                metadata TEXT
            )
        """)

        # Insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                insight_id TEXT PRIMARY KEY,
                improvement_type TEXT NOT NULL,
                pattern TEXT NOT NULL,
                confidence REAL,
                evidence_count INTEGER,
                success_rate REAL,
                context_conditions TEXT,
                recommendation TEXT,
                created_at TEXT,
                last_updated TEXT
            )
        """)

        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                metric_date TEXT PRIMARY KEY,
                total_tasks INTEGER,
                successful_tasks INTEGER,
                failed_tasks INTEGER,
                average_duration REAL,
                success_rate REAL
            )
        """)

        conn.commit()
        conn.close()

    def store_experience(self, experience: Experience):
        """Store an experience."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO experiences
            (experience_id, task_description, context, actions_taken, outcome,
             outcome_details, duration_seconds, timestamp, agent_state, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experience.experience_id,
            experience.task_description,
            json.dumps(experience.context),
            json.dumps(experience.actions_taken),
            experience.outcome.value,
            json.dumps(experience.outcome_details),
            experience.duration_seconds,
            experience.timestamp.isoformat(),
            json.dumps(experience.agent_state) if experience.agent_state else None,
            json.dumps(experience.metadata)
        ))

        conn.commit()
        conn.close()

    def get_experiences(
        self,
        outcome: Optional[OutcomeType] = None,
        limit: int = 100,
        task_pattern: Optional[str] = None
    ) -> List[Experience]:
        """
        Retrieve experiences with optional filters.

        Args:
            outcome: Filter by outcome type
            limit: Maximum number of experiences to return
            task_pattern: Filter tasks containing this string

        Returns:
            List of experiences
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM experiences WHERE 1=1"
        params = []

        if outcome:
            query += " AND outcome = ?"
            params.append(outcome.value)

        if task_pattern:
            query += " AND task_description LIKE ?"
            params.append(f"%{task_pattern}%")

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        experiences = []
        for row in rows:
            experiences.append(Experience(
                experience_id=row[0],
                task_description=row[1],
                context=json.loads(row[2]) if row[2] else {},
                actions_taken=json.loads(row[3]) if row[3] else [],
                outcome=OutcomeType(row[4]),
                outcome_details=json.loads(row[5]) if row[5] else {},
                duration_seconds=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                agent_state=json.loads(row[8]) if row[8] else None,
                metadata=json.loads(row[9]) if row[9] else {}
            ))

        return experiences

    def store_insight(self, insight: LearningInsight):
        """Store a learning insight."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO insights
            (insight_id, improvement_type, pattern, confidence, evidence_count,
             success_rate, context_conditions, recommendation, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight.insight_id,
            insight.improvement_type.value,
            insight.pattern,
            insight.confidence,
            insight.evidence_count,
            insight.success_rate,
            json.dumps(insight.context_conditions),
            insight.recommendation,
            insight.created_at.isoformat(),
            insight.last_updated.isoformat()
        ))

        conn.commit()
        conn.close()

    def get_insights(
        self,
        improvement_type: Optional[ImprovementType] = None,
        min_confidence: float = 0.5
    ) -> List[LearningInsight]:
        """
        Retrieve learning insights.

        Args:
            improvement_type: Filter by improvement type
            min_confidence: Minimum confidence threshold

        Returns:
            List of insights
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM insights WHERE confidence >= ?"
        params = [min_confidence]

        if improvement_type:
            query += " AND improvement_type = ?"
            params.append(improvement_type.value)

        query += " ORDER BY confidence DESC, evidence_count DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        insights = []
        for row in rows:
            insights.append(LearningInsight(
                insight_id=row[0],
                improvement_type=ImprovementType(row[1]),
                pattern=row[2],
                confidence=row[3],
                evidence_count=row[4],
                success_rate=row[5],
                context_conditions=json.loads(row[6]) if row[6] else {},
                recommendation=row[7],
                created_at=datetime.fromisoformat(row[8]),
                last_updated=datetime.fromisoformat(row[9])
            ))

        return insights


class PatternAnalyzer:
    """Analyzes experiences to identify patterns."""

    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        Initialize pattern analyzer.

        Args:
            llm: LLM instance for pattern analysis
            logger: Optional logger
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)

    def analyze_success_patterns(
        self,
        experiences: List[Experience]
    ) -> List[Dict[str, Any]]:
        """
        Analyze successful experiences to identify patterns.

        Args:
            experiences: List of experiences to analyze

        Returns:
            List of identified patterns
        """
        successful = [e for e in experiences if e.outcome == OutcomeType.SUCCESS]

        if not successful:
            return []

        # Group by task type
        task_groups = defaultdict(list)
        for exp in successful:
            # Simple grouping by first word in task description
            task_type = exp.task_description.split()[0].lower() if exp.task_description else "general"
            task_groups[task_type].append(exp)

        patterns = []

        for task_type, group_experiences in task_groups.items():
            if len(group_experiences) < 2:
                continue

            # Analyze common action sequences
            action_sequences = []
            for exp in group_experiences:
                action_seq = [a.get("action_type", "unknown") for a in exp.actions_taken]
                action_sequences.append(action_seq)

            # Find most common sequence
            if action_sequences:
                # Simple frequency count
                seq_counts = defaultdict(int)
                for seq in action_sequences:
                    seq_counts[tuple(seq)] += 1

                most_common_seq = max(seq_counts.items(), key=lambda x: x[1])

                if most_common_seq[1] >= 2:  # Appears at least twice
                    patterns.append({
                        "task_type": task_type,
                        "pattern_type": "action_sequence",
                        "sequence": list(most_common_seq[0]),
                        "frequency": most_common_seq[1],
                        "total_examples": len(group_experiences),
                        "confidence": most_common_seq[1] / len(group_experiences)
                    })

        return patterns

    def analyze_failure_patterns(
        self,
        experiences: List[Experience]
    ) -> List[Dict[str, Any]]:
        """
        Analyze failed experiences to identify common failure modes.

        Args:
            experiences: List of experiences to analyze

        Returns:
            List of identified failure patterns
        """
        failed = [e for e in experiences if e.outcome in [OutcomeType.FAILURE, OutcomeType.ERROR]]

        if not failed:
            return []

        # Group by error type
        error_groups = defaultdict(list)
        for exp in failed:
            error_type = exp.outcome_details.get("error_type", "unknown")
            error_groups[error_type].append(exp)

        patterns = []

        for error_type, group_experiences in error_groups.items():
            # Common context conditions
            context_keys = defaultdict(int)
            for exp in group_experiences:
                for key in exp.context.keys():
                    context_keys[key] += 1

            patterns.append({
                "error_type": error_type,
                "frequency": len(group_experiences),
                "common_context": dict(context_keys),
                "example_actions": group_experiences[0].actions_taken[:3] if group_experiences else []
            })

        return patterns

    def generate_insights_from_patterns(
        self,
        success_patterns: List[Dict[str, Any]],
        failure_patterns: List[Dict[str, Any]]
    ) -> List[LearningInsight]:
        """
        Generate actionable insights from identified patterns.

        Args:
            success_patterns: Patterns from successful experiences
            failure_patterns: Patterns from failed experiences

        Returns:
            List of learning insights
        """
        insights = []

        # Generate insights from success patterns
        for pattern in success_patterns:
            if pattern["confidence"] >= 0.5:
                insight = LearningInsight(
                    insight_id=f"insight_{pattern['task_type']}_{len(insights)}",
                    improvement_type=ImprovementType.REASONING_PATTERN,
                    pattern=f"For {pattern['task_type']} tasks, use sequence: {' -> '.join(pattern['sequence'])}",
                    confidence=pattern["confidence"],
                    evidence_count=pattern["frequency"],
                    success_rate=pattern["confidence"],
                    context_conditions={"task_type": pattern["task_type"]},
                    recommendation=f"When handling {pattern['task_type']} tasks, follow the action sequence: {' -> '.join(pattern['sequence'])}"
                )
                insights.append(insight)

        # Generate insights from failure patterns
        for pattern in failure_patterns:
            if pattern["frequency"] >= 2:
                insight = LearningInsight(
                    insight_id=f"insight_error_{pattern['error_type']}_{len(insights)}",
                    improvement_type=ImprovementType.ERROR_RECOVERY,
                    pattern=f"Common failure: {pattern['error_type']}",
                    confidence=min(pattern["frequency"] / 10, 0.9),  # Cap at 0.9
                    evidence_count=pattern["frequency"],
                    success_rate=0.0,  # Failure pattern
                    context_conditions=pattern["common_context"],
                    recommendation=f"Implement error recovery for {pattern['error_type']}. Common context: {list(pattern['common_context'].keys())[:3]}"
                )
                insights.append(insight)

        return insights


class LearningSystem:
    """
    Complete learning and improvement system for agents.

    Tracks experiences, identifies patterns, and provides recommendations.
    """

    def __init__(
        self,
        llm,
        storage_path: str = "agent_learning.db",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize learning system.

        Args:
            llm: LLM instance for analysis
            storage_path: Path to experience database
            logger: Optional logger
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.experience_store = ExperienceStore(storage_path)
        self.pattern_analyzer = PatternAnalyzer(llm, logger)

        # In-memory cache
        self.recent_insights: List[LearningInsight] = []
        self.performance_metrics = PerformanceMetrics()

    def record_experience(
        self,
        task_description: str,
        actions: List[Dict[str, Any]],
        outcome: OutcomeType,
        outcome_details: Dict[str, Any],
        duration: float,
        context: Optional[Dict[str, Any]] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> Experience:
        """
        Record a new experience.

        Args:
            task_description: Description of the task
            actions: List of actions taken
            outcome: Outcome of the task
            outcome_details: Details about the outcome
            duration: Duration in seconds
            context: Optional context
            agent_state: Optional agent state snapshot

        Returns:
            Created experience
        """
        experience = Experience(
            experience_id=f"exp_{datetime.now().timestamp()}",
            task_description=task_description,
            context=context or {},
            actions_taken=actions,
            outcome=outcome,
            outcome_details=outcome_details,
            duration_seconds=duration,
            agent_state=agent_state
        )

        self.experience_store.store_experience(experience)
        self.logger.info(f"Recorded experience: {experience.experience_id} ({outcome.value})")

        # Update metrics
        self._update_metrics(experience)

        return experience

    def _update_metrics(self, experience: Experience):
        """Update performance metrics."""
        self.performance_metrics.total_tasks += 1

        if experience.outcome == OutcomeType.SUCCESS:
            self.performance_metrics.successful_tasks += 1
        elif experience.outcome in [OutcomeType.FAILURE, OutcomeType.ERROR]:
            self.performance_metrics.failed_tasks += 1

        # Update success rate
        if self.performance_metrics.total_tasks > 0:
            self.performance_metrics.success_rate = (
                self.performance_metrics.successful_tasks / self.performance_metrics.total_tasks
            )

        # Update average duration
        total_duration = (
            self.performance_metrics.average_duration * (self.performance_metrics.total_tasks - 1)
            + experience.duration_seconds
        )
        self.performance_metrics.average_duration = total_duration / self.performance_metrics.total_tasks

        # Track improvement over time
        self.performance_metrics.improvement_over_time.append(
            (experience.timestamp, self.performance_metrics.success_rate)
        )

    def analyze_and_learn(self, min_experiences: int = 10) -> List[LearningInsight]:
        """
        Analyze recent experiences and generate insights.

        Args:
            min_experiences: Minimum number of experiences needed for analysis

        Returns:
            List of new insights
        """
        # Get recent experiences
        experiences = self.experience_store.get_experiences(limit=100)

        if len(experiences) < min_experiences:
            self.logger.info(f"Not enough experiences for analysis: {len(experiences)}/{min_experiences}")
            return []

        self.logger.info(f"Analyzing {len(experiences)} experiences...")

        # Analyze patterns
        success_patterns = self.pattern_analyzer.analyze_success_patterns(experiences)
        failure_patterns = self.pattern_analyzer.analyze_failure_patterns(experiences)

        # Generate insights
        insights = self.pattern_analyzer.generate_insights_from_patterns(
            success_patterns,
            failure_patterns
        )

        # Store insights
        for insight in insights:
            self.experience_store.store_insight(insight)

        self.recent_insights = insights
        self.logger.info(f"Generated {len(insights)} new insights")

        return insights

    def get_recommendations(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get recommendations for a task based on learned insights.

        Args:
            task_description: Description of the task
            context: Optional context

        Returns:
            List of recommendations
        """
        # Get relevant insights
        insights = self.experience_store.get_insights(min_confidence=0.5)

        recommendations = []

        # Match insights to current task
        task_words = set(task_description.lower().split())

        for insight in insights:
            # Check if insight is relevant
            pattern_words = set(insight.pattern.lower().split())
            overlap = task_words & pattern_words

            if overlap or not task_words:  # If no overlap, still consider general insights
                recommendations.append(insight.recommendation)

        # Limit to top recommendations
        return recommendations[:5]

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics

    def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get summary of learning progress.

        Returns:
            Dictionary with learning statistics
        """
        insights = self.experience_store.get_insights(min_confidence=0.5)

        return {
            "total_experiences": self.performance_metrics.total_tasks,
            "success_rate": self.performance_metrics.success_rate,
            "total_insights": len(insights),
            "insights_by_type": self._count_insights_by_type(insights),
            "average_task_duration": self.performance_metrics.average_duration,
            "recent_insights": [
                {
                    "type": i.improvement_type.value,
                    "pattern": i.pattern,
                    "confidence": i.confidence
                }
                for i in insights[:5]
            ]
        }

    def _count_insights_by_type(self, insights: List[LearningInsight]) -> Dict[str, int]:
        """Count insights by improvement type."""
        counts = defaultdict(int)
        for insight in insights:
            counts[insight.improvement_type.value] += 1
        return dict(counts)

    def export_knowledge(self, output_path: str):
        """
        Export learned knowledge to a file.

        Args:
            output_path: Path to export file
        """
        insights = self.experience_store.get_insights(min_confidence=0.3)

        knowledge = {
            "exported_at": datetime.now().isoformat(),
            "metrics": asdict(self.performance_metrics),
            "insights": [
                {
                    "id": i.insight_id,
                    "type": i.improvement_type.value,
                    "pattern": i.pattern,
                    "confidence": i.confidence,
                    "evidence_count": i.evidence_count,
                    "recommendation": i.recommendation
                }
                for i in insights
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(knowledge, f, indent=2)

        self.logger.info(f"Exported knowledge to {output_path}")

    def import_knowledge(self, input_path: str):
        """
        Import learned knowledge from a file.

        Args:
            input_path: Path to import file
        """
        with open(input_path, 'r') as f:
            knowledge = json.load(f)

        for insight_data in knowledge.get("insights", []):
            insight = LearningInsight(
                insight_id=insight_data["id"],
                improvement_type=ImprovementType(insight_data["type"]),
                pattern=insight_data["pattern"],
                confidence=insight_data["confidence"],
                evidence_count=insight_data["evidence_count"],
                success_rate=0.0,  # Will be recalculated
                context_conditions={},
                recommendation=insight_data["recommendation"]
            )
            self.experience_store.store_insight(insight)

        self.logger.info(f"Imported knowledge from {input_path}")
