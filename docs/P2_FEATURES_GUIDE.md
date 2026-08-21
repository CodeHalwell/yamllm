# P2 Features: Multi-Agent Collaboration & Learning System

## Overview

YAMLLM P2 features bring advanced capabilities for building collaborative multi-agent systems and continuous learning from experience. These features enable:

1. **Multi-Agent Collaboration**: Coordinate multiple specialized agents working together on complex tasks
2. **Learning & Improvement**: Agents learn from experiences and improve over time

## Table of Contents

- [Multi-Agent Collaboration](#multi-agent-collaboration)
  - [Architecture](#architecture)
  - [Agent Roles](#agent-roles)
  - [Quick Start](#multi-agent-quick-start)
  - [API Reference](#multi-agent-api-reference)
  - [CLI Commands](#multi-agent-cli-commands)
- [Learning & Improvement](#learning--improvement)
  - [Architecture](#learning-architecture)
  - [Quick Start](#learning-quick-start)
  - [API Reference](#learning-api-reference)
  - [CLI Commands](#learning-cli-commands)
- [Examples](#examples)
- [Best Practices](#best-practices)

---

## Multi-Agent Collaboration

### Architecture

The multi-agent system consists of:

- **CollaborativeAgent**: Individual agents with specific roles and capabilities
- **AgentCoordinator**: Orchestrates agent collaboration, task assignment, and message routing
- **AgentMessage**: Communication protocol between agents
- **CollaborativeTask**: Tasks that require multiple agent roles

### Agent Roles

YAMLLM provides 8 specialized agent roles:

| Role | Description | Typical Skills |
|------|-------------|----------------|
| **Coordinator** | Orchestrates other agents | planning, coordination |
| **Researcher** | Gathers and analyzes information | research, analysis |
| **Coder** | Writes and implements code | coding, implementation |
| **Reviewer** | Reviews work from other agents | code review, validation |
| **Tester** | Tests functionality and finds bugs | testing, QA |
| **Debugger** | Fixes bugs and issues | debugging, troubleshooting |
| **Documenter** | Creates documentation | writing, documentation |
| **Analyst** | Analyzes data and problems | data analysis, insights |

### Multi-Agent Quick Start

#### Python API

```python
from yamllm import LLM
from yamllm.agent.multi_agent import (
    AgentCoordinator, CollaborativeAgent,
    AgentCapability, AgentRole
)

# Create LLM
llm = LLM(provider="openai", model="gpt-4")

# Create coordinator
coordinator = AgentCoordinator(coordinator_llm=llm)

# Create and register specialized agents
researcher = CollaborativeAgent(
    agent_id="researcher_1",
    llm=llm,
    capability=AgentCapability(
        role=AgentRole.RESEARCHER,
        skills=["research", "analysis"],
        max_concurrent_tasks=2
    )
)

coder = CollaborativeAgent(
    agent_id="coder_1",
    llm=llm,
    capability=AgentCapability(
        role=AgentRole.CODER,
        skills=["python", "javascript"],
        max_concurrent_tasks=1
    )
)

reviewer = CollaborativeAgent(
    agent_id="reviewer_1",
    llm=llm,
    capability=AgentCapability(
        role=AgentRole.REVIEWER,
        skills=["code_review", "best_practices"],
        max_concurrent_tasks=2
    )
)

# Register agents
coordinator.register_agent(researcher)
coordinator.register_agent(coder)
coordinator.register_agent(reviewer)

# Execute collaborative task
result = coordinator.execute_collaborative_task(
    goal="Build a REST API with proper testing",
    max_iterations=10
)

print(f"Completed {result['tasks_completed']} tasks")
print(f"Results: {result['results']}")
```

#### CLI

```bash
# Execute collaborative task with specific roles
yamllm multi-agent execute "Build a REST API" \
  --roles researcher \
  --roles coder \
  --roles reviewer \
  --roles tester \
  --max-iterations 10 \
  --verbose

# Use default agent set
yamllm multi-agent execute "Implement user authentication" \
  --config my_config.yaml
```

### Multi-Agent API Reference

#### AgentCoordinator

```python
class AgentCoordinator:
    """Coordinates multiple agents for collaborative tasks."""

    def __init__(self, coordinator_llm, logger: Optional[logging.Logger] = None):
        """Initialize coordinator with LLM for planning."""

    def register_agent(self, agent: CollaborativeAgent):
        """Register an agent with the coordinator."""

    def create_task(
        self,
        task_id: str,
        description: str,
        required_roles: List[AgentRole],
        dependencies: Optional[List[str]] = None
    ) -> CollaborativeTask:
        """Create a collaborative task."""

    def execute_collaborative_task(
        self,
        goal: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Execute a goal using collaborative agents."""

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
```

#### CollaborativeAgent

```python
class CollaborativeAgent:
    """An agent that can collaborate with others."""

    def __init__(
        self,
        agent_id: str,
        llm,
        capability: AgentCapability,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize collaborative agent."""

    def can_handle(self, task: CollaborativeTask) -> bool:
        """Check if agent can handle task based on role."""

    def send_message(
        self,
        to_agent: str,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 1
    ):
        """Send message to another agent."""

    def receive_message(self, message: AgentMessage):
        """Receive message from another agent."""
```

### Multi-Agent CLI Commands

```bash
# Execute collaborative task
yamllm multi-agent execute <goal> [OPTIONS]

Options:
  -c, --config PATH          LLM config file
  -r, --roles TEXT          Agent roles to use (can be specified multiple times)
  -m, --max-iterations INT  Maximum coordination iterations (default: 10)
  -v, --verbose            Verbose output

# Show system status
yamllm multi-agent status [--config PATH]
```

---

## Learning & Improvement

### Learning Architecture

The learning system consists of:

- **LearningSystem**: Main interface for recording and analyzing experiences
- **ExperienceStore**: Persistent storage for experiences and insights
- **PatternAnalyzer**: Identifies patterns in successful and failed experiences
- **LearningInsight**: Actionable insights derived from patterns

### Learning Quick Start

#### Python API

```python
from yamllm import LLM
from yamllm.agent.learning_system import LearningSystem, OutcomeType

# Create learning system
llm = LLM(provider="openai", model="gpt-4")
learning = LearningSystem(llm, storage_path="agent_learning.db")

# Record experience
experience = learning.record_experience(
    task_description="Implement user authentication",
    actions=[
        {"action_type": "research", "details": "Studied OAuth2"},
        {"action_type": "code", "details": "Implemented auth endpoints"},
        {"action_type": "test", "details": "Tested with Postman"}
    ],
    outcome=OutcomeType.SUCCESS,
    outcome_details={"tests_passed": True, "coverage": 0.95},
    duration=3600.0,  # seconds
    context={"language": "python", "framework": "fastapi"}
)

# Record multiple experiences...

# Analyze and generate insights (requires minimum 10 experiences)
insights = learning.analyze_and_learn(min_experiences=10)

for insight in insights:
    print(f"Type: {insight.improvement_type.value}")
    print(f"Pattern: {insight.pattern}")
    print(f"Confidence: {insight.confidence:.1%}")
    print(f"Recommendation: {insight.recommendation}")
    print()

# Get recommendations for a new task
recommendations = learning.get_recommendations(
    task_description="Build payment integration",
    context={"language": "python"}
)

for rec in recommendations:
    print(f"- {rec}")

# View performance metrics
metrics = learning.get_metrics()
print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Average duration: {metrics.average_duration:.1f}s")

# Export knowledge for sharing or backup
learning.export_knowledge("knowledge.json")

# Import knowledge from another agent
learning.import_knowledge("shared_knowledge.json")
```

#### CLI

```bash
# Record an experience
yamllm learn record "Fix authentication bug" \
  --outcome success \
  --duration 120.5 \
  --actions '[{"action": "debug"}, {"action": "fix"}]' \
  --details '{"bug_type": "null_pointer"}'

# Analyze experiences and generate insights
yamllm learn analyze --min-experiences 10 --verbose

# Get recommendations for a task
yamllm learn recommend "Implement payment processing"

# View performance metrics
yamllm learn metrics --export metrics.json

# Export learned knowledge
yamllm learn export knowledge.json

# Import knowledge from another system
yamllm learn import shared_knowledge.json
```

### Learning API Reference

#### LearningSystem

```python
class LearningSystem:
    """Complete learning and improvement system for agents."""

    def __init__(
        self,
        llm,
        storage_path: str = "agent_learning.db",
        logger: Optional[logging.Logger] = None
    ):
        """Initialize learning system."""

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
        """Record a new experience."""

    def analyze_and_learn(
        self,
        min_experiences: int = 10
    ) -> List[LearningInsight]:
        """Analyze recent experiences and generate insights."""

    def get_recommendations(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get recommendations for a task based on learned insights."""

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress."""

    def export_knowledge(self, output_path: str):
        """Export learned knowledge to a file."""

    def import_knowledge(self, input_path: str):
        """Import learned knowledge from a file."""
```

#### OutcomeType

```python
class OutcomeType(Enum):
    """Types of task outcomes."""
    SUCCESS = "success"       # Task completed successfully
    FAILURE = "failure"       # Task failed
    PARTIAL = "partial"       # Task partially completed
    TIMEOUT = "timeout"       # Task timed out
    ERROR = "error"          # Task encountered an error
```

#### ImprovementType

```python
class ImprovementType(Enum):
    """Types of improvements that can be learned."""
    TASK_DECOMPOSITION = "task_decomposition"
    TOOL_SELECTION = "tool_selection"
    REASONING_PATTERN = "reasoning_pattern"
    ERROR_RECOVERY = "error_recovery"
    CONTEXT_USAGE = "context_usage"
    PLANNING_STRATEGY = "planning_strategy"
```

### Learning CLI Commands

```bash
# Record experience
yamllm learn record <task_description> [OPTIONS]

Options:
  -o, --outcome [success|failure|partial|timeout|error]  Required
  -d, --duration FLOAT                                     Required (seconds)
  -a, --actions TEXT                                       JSON string of actions
  --details TEXT                                           JSON string of outcome details
  --db PATH                                                Database path (default: agent_learning.db)

# Analyze experiences
yamllm learn analyze [OPTIONS]

Options:
  -c, --config PATH           LLM config file
  --db PATH                   Database path
  -m, --min-experiences INT   Minimum experiences needed (default: 10)
  -v, --verbose              Show detailed analysis

# Get recommendations
yamllm learn recommend <task_description> [OPTIONS]

Options:
  -c, --config PATH  LLM config file
  --db PATH          Database path

# View metrics
yamllm learn metrics [OPTIONS]

Options:
  --db PATH        Database path
  -e, --export PATH  Export metrics to JSON file

# Export knowledge
yamllm learn export <output_path> [OPTIONS]

Options:
  -c, --config PATH  LLM config file
  --db PATH          Database path

# Import knowledge
yamllm learn import <input_path> [OPTIONS]

Options:
  -c, --config PATH  LLM config file
  --db PATH          Database path
```

---

## Examples

### Example 1: Building a Web Application with Multi-Agent Team

```python
from yamllm import LLM
from yamllm.agent.multi_agent import (
    AgentCoordinator, CollaborativeAgent,
    AgentCapability, AgentRole
)

# Setup
llm = LLM(provider="openai", model="gpt-4")
coordinator = AgentCoordinator(coordinator_llm=llm)

# Create specialized team
team = {
    "architect": AgentRole.ANALYST,
    "backend_dev": AgentRole.CODER,
    "frontend_dev": AgentRole.CODER,
    "qa_engineer": AgentRole.TESTER,
    "tech_writer": AgentRole.DOCUMENTER
}

for agent_id, role in team.items():
    agent = CollaborativeAgent(
        agent_id=agent_id,
        llm=llm,
        capability=AgentCapability(role=role, skills=[role.value])
    )
    coordinator.register_agent(agent)

# Execute project
result = coordinator.execute_collaborative_task(
    goal="Build a todo list web application with REST API, React frontend, and comprehensive tests",
    max_iterations=20
)

print(f"Project completed in {result['iterations']} iterations")
```

### Example 2: Agent Learning from Debugging Sessions

```python
from yamllm import LLM
from yamllm.agent.learning_system import LearningSystem, OutcomeType
import time

llm = LLM(provider="openai", model="gpt-4")
learning = LearningSystem(llm, storage_path="debug_learning.db")

# Simulate debugging sessions
debug_sessions = [
    {
        "task": "Fix null pointer exception",
        "actions": [
            {"action_type": "analyze_logs"},
            {"action_type": "add_null_check"},
            {"action_type": "test"}
        ],
        "outcome": OutcomeType.SUCCESS,
        "duration": 300
    },
    {
        "task": "Fix performance issue",
        "actions": [
            {"action_type": "profile_code"},
            {"action_type": "optimize_query"},
            {"action_type": "test"}
        ],
        "outcome": OutcomeType.SUCCESS,
        "duration": 1800
    },
    # ... more sessions
]

# Record all sessions
for session in debug_sessions:
    learning.record_experience(
        task_description=session["task"],
        actions=session["actions"],
        outcome=session["outcome"],
        outcome_details={},
        duration=session["duration"]
    )

# Analyze after collecting enough data
if len(debug_sessions) >= 10:
    insights = learning.analyze_and_learn()

    print("Learned debugging patterns:")
    for insight in insights:
        print(f"- {insight.pattern} (confidence: {insight.confidence:.1%})")

# Get recommendations for new bug
recommendations = learning.get_recommendations("Fix memory leak")
print("\nRecommendations for fixing memory leak:")
for rec in recommendations:
    print(f"- {rec}")
```

### Example 3: Combining Multi-Agent and Learning

```python
from yamllm import LLM
from yamllm.agent.multi_agent import AgentCoordinator, CollaborativeAgent, AgentCapability, AgentRole
from yamllm.agent.learning_system import LearningSystem, OutcomeType
import time

llm = LLM(provider="openai", model="gpt-4")

# Setup multi-agent coordinator
coordinator = AgentCoordinator(coordinator_llm=llm)

# Setup learning system
learning = LearningSystem(llm, storage_path="collaborative_learning.db")

# Create agents
for role in [AgentRole.RESEARCHER, AgentRole.CODER, AgentRole.REVIEWER]:
    agent = CollaborativeAgent(
        agent_id=f"{role.value}_agent",
        llm=llm,
        capability=AgentCapability(role=role, skills=[role.value])
    )
    coordinator.register_agent(agent)

# Execute task while learning
goal = "Implement caching layer for API"
start_time = time.time()

result = coordinator.execute_collaborative_task(goal, max_iterations=10)

duration = time.time() - start_time

# Record the collaborative experience
learning.record_experience(
    task_description=goal,
    actions=[{"agent_count": len(coordinator.agents), "iterations": result["iterations"]}],
    outcome=OutcomeType.SUCCESS if result["tasks_completed"] > 0 else OutcomeType.FAILURE,
    outcome_details={
        "tasks_completed": result["tasks_completed"],
        "iterations_used": result["iterations"]
    },
    duration=duration,
    context={"agent_roles": [a.capability.role.value for a in coordinator.agents.values()]}
)

# Get recommendations for similar future tasks
recommendations = learning.get_recommendations(
    "Implement rate limiting for API",
    context={"task_similarity": "high"}
)

print("Based on past collaborative tasks, recommendations:")
for rec in recommendations:
    print(f"- {rec}")
```

---

## Best Practices

### Multi-Agent Collaboration

1. **Role Assignment**
   - Match agent roles to task requirements
   - Don't over-assign: 3-5 agents is typically optimal
   - Use Coordinator role for complex multi-stage tasks

2. **Task Decomposition**
   - Let the coordinator decompose complex goals automatically
   - Provide clear, specific goals
   - Set appropriate max_iterations (10-20 for most tasks)

3. **Agent Capabilities**
   - Specify relevant skills for each agent
   - Set realistic max_concurrent_tasks (1-2 for complex reasoning)
   - Use higher confidence values for specialized agents

4. **Communication**
   - Agents communicate via message passing
   - Coordinator handles message routing automatically
   - Use priority levels for urgent messages

### Learning & Improvement

1. **Experience Recording**
   - Record experiences consistently
   - Include detailed context
   - Track both successes and failures
   - Be specific in action descriptions

2. **Analysis**
   - Collect at least 10 experiences before analyzing
   - Run analysis periodically (e.g., after every 10-20 tasks)
   - Review generated insights manually
   - Adjust min_confidence threshold based on your needs

3. **Recommendations**
   - Use recommendations as suggestions, not rules
   - Provide task context for better matching
   - Review recommendation relevance
   - Export/import knowledge across agents

4. **Performance Metrics**
   - Monitor success rate trends
   - Track average duration improvements
   - Identify common error patterns
   - Use metrics to guide agent improvements

5. **Knowledge Sharing**
   - Export knowledge from experienced agents
   - Import shared knowledge for new agents
   - Maintain separate databases for different domains
   - Version your exported knowledge files

### Integration Tips

1. **Combining Features**
   - Use multi-agent for complex tasks
   - Record all multi-agent executions
   - Analyze collaborative patterns
   - Apply learned insights to future task assignment

2. **Monitoring**
   - Use verbose mode during development
   - Monitor agent coordination iterations
   - Track learning database size
   - Review insights periodically

3. **Optimization**
   - Start with fewer agents, add as needed
   - Use faster models (GPT-3.5) for routine tasks
   - Cache learning insights for quick recommendations
   - Batch analyze experiences for efficiency

4. **Error Handling**
   - Set reasonable timeouts
   - Handle agent failures gracefully
   - Record failures for learning
   - Implement retry logic for transient errors

---

## Troubleshooting

### Multi-Agent Issues

**Problem**: Agents not completing tasks
- Check role assignments match task requirements
- Increase max_iterations
- Review agent capacity settings
- Check LLM response quality

**Problem**: Tasks taking too long
- Reduce number of agents
- Simplify task descriptions
- Use faster LLM models
- Set stricter iteration limits

### Learning Issues

**Problem**: Not generating insights
- Ensure you have minimum required experiences
- Check experience diversity
- Review pattern detection thresholds
- Verify outcome types are varied

**Problem**: Recommendations not relevant
- Provide more context when requesting recommendations
- Record more detailed actions in experiences
- Increase min_confidence threshold
- Review and curate insights manually

---

## Next Steps

- Explore [P1+ Interactive Steering](INTERACTIVE_STEERING.md) for human-in-the-loop control
- Check out [Code Intelligence](CODE_INTELLIGENCE.md) for smart code analysis
- Review [Advanced Git Workflows](ADVANCED_GIT.md) for intelligent git operations
- See [API Reference](API_REFERENCE.md) for complete API documentation

## Support

For issues, questions, or contributions, visit the [GitHub repository](https://github.com/yourusername/yamllm).
