"""
YAMLLM P2 Features Examples
============================

Examples demonstrating Multi-Agent Collaboration and Learning & Improvement features.
"""

import time
from pathlib import Path


# =============================================================================
# Example 1: Basic Multi-Agent Collaboration
# =============================================================================

def example_basic_multi_agent():
    """
    Basic example of multi-agent collaboration.

    This example shows how to:
    - Create a coordinator
    - Register specialized agents
    - Execute a collaborative task
    """
    from yamllm import LLM
    from yamllm.agent.multi_agent import (
        AgentCoordinator, CollaborativeAgent,
        AgentCapability, AgentRole
    )

    print("=" * 80)
    print("Example 1: Basic Multi-Agent Collaboration")
    print("=" * 80)

    # Create LLM
    llm = LLM(provider="openai", model="gpt-4")

    # Create coordinator
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Create specialized agents
    researcher = CollaborativeAgent(
        agent_id="researcher_1",
        llm=llm,
        capability=AgentCapability(
            role=AgentRole.RESEARCHER,
            skills=["research", "analysis", "data_gathering"],
            max_concurrent_tasks=2
        )
    )

    coder = CollaborativeAgent(
        agent_id="coder_1",
        llm=llm,
        capability=AgentCapability(
            role=AgentRole.CODER,
            skills=["python", "javascript", "api_development"],
            max_concurrent_tasks=1
        )
    )

    reviewer = CollaborativeAgent(
        agent_id="reviewer_1",
        llm=llm,
        capability=AgentCapability(
            role=AgentRole.REVIEWER,
            skills=["code_review", "best_practices", "security"],
            max_concurrent_tasks=2
        )
    )

    # Register agents
    coordinator.register_agent(researcher)
    coordinator.register_agent(coder)
    coordinator.register_agent(reviewer)

    # Check status
    status = coordinator.get_status()
    print(f"\nRegistered {status['registered_agents']} agents:")
    for agent_id, agent_info in status["agents"].items():
        print(f"  - {agent_id} ({agent_info['role']})")

    # Execute collaborative task
    print("\n Executing collaborative task...")
    result = coordinator.execute_collaborative_task(
        goal="Design and implement a simple REST API for a todo list",
        max_iterations=10
    )

    print(f"\n✓ Completed {result['tasks_completed']} tasks in {result['iterations']} iterations")
    print(f"  Results: {result['results']}")


# =============================================================================
# Example 2: Software Development Team
# =============================================================================

def example_software_dev_team():
    """
    Simulate a complete software development team.

    This example shows:
    - Multiple specialized agents working together
    - Different agent roles and responsibilities
    - Complex task coordination
    """
    from yamllm import LLM
    from yamllm.agent.multi_agent import (
        AgentCoordinator, CollaborativeAgent,
        AgentCapability, AgentRole
    )

    print("\n" + "=" * 80)
    print("Example 2: Software Development Team")
    print("=" * 80)

    llm = LLM(provider="openai", model="gpt-4")
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Create full development team
    team = {
        "analyst": (AgentRole.ANALYST, ["requirements", "analysis", "planning"]),
        "backend_dev": (AgentRole.CODER, ["python", "fastapi", "databases"]),
        "frontend_dev": (AgentRole.CODER, ["react", "typescript", "ui"]),
        "qa_engineer": (AgentRole.TESTER, ["testing", "automation", "quality"]),
        "tech_writer": (AgentRole.DOCUMENTER, ["documentation", "technical_writing"]),
        "reviewer": (AgentRole.REVIEWER, ["code_review", "architecture_review"])
    }

    print("\nBuilding development team:")
    for agent_id, (role, skills) in team.items():
        agent = CollaborativeAgent(
            agent_id=agent_id,
            llm=llm,
            capability=AgentCapability(
                role=role,
                skills=skills,
                max_concurrent_tasks=1
            )
        )
        coordinator.register_agent(agent)
        print(f"  ✓ {agent_id} ({role.value})")

    # Execute software project
    print("\n🚀 Starting project...")
    start_time = time.time()

    result = coordinator.execute_collaborative_task(
        goal="Build a complete authentication system with JWT tokens, "
             "password hashing, user registration, login, and logout endpoints",
        max_iterations=15
    )

    duration = time.time() - start_time

    print(f"\n✓ Project completed in {duration:.1f}s")
    print(f"  Tasks completed: {result['tasks_completed']}")
    print(f"  Iterations used: {result['iterations']}")


# =============================================================================
# Example 3: Basic Learning System
# =============================================================================

def example_basic_learning():
    """
    Basic example of the learning system.

    This example shows:
    - Recording experiences
    - Analyzing patterns
    - Getting recommendations
    - Viewing metrics
    """
    from yamllm import LLM
    from yamllm.agent.learning_system import LearningSystem, OutcomeType

    print("\n" + "=" * 80)
    print("Example 3: Basic Learning System")
    print("=" * 80)

    llm = LLM(provider="openai", model="gpt-4")
    learning = LearningSystem(llm, storage_path="example_learning.db")

    print("\n📝 Recording experiences...")

    # Record successful experiences
    experiences = [
        {
            "task": "Implement user authentication",
            "actions": [
                {"action_type": "research", "details": "Studied OAuth2"},
                {"action_type": "design", "details": "Designed auth flow"},
                {"action_type": "code", "details": "Implemented endpoints"},
                {"action_type": "test", "details": "Wrote unit tests"}
            ],
            "outcome": OutcomeType.SUCCESS,
            "duration": 3600,
            "context": {"language": "python", "framework": "fastapi"}
        },
        {
            "task": "Add password reset feature",
            "actions": [
                {"action_type": "design", "details": "Designed reset flow"},
                {"action_type": "code", "details": "Implemented reset logic"},
                {"action_type": "test", "details": "Tested reset process"}
            ],
            "outcome": OutcomeType.SUCCESS,
            "duration": 1800,
            "context": {"language": "python", "framework": "fastapi"}
        },
        {
            "task": "Fix authentication bug",
            "actions": [
                {"action_type": "debug", "details": "Analyzed logs"},
                {"action_type": "fix", "details": "Fixed token validation"},
                {"action_type": "test", "details": "Verified fix"}
            ],
            "outcome": OutcomeType.SUCCESS,
            "duration": 600,
            "context": {"language": "python"}
        }
    ]

    # Record failed experiences
    failures = [
        {
            "task": "Implement OAuth integration",
            "actions": [
                {"action_type": "code", "details": "Started implementation"},
                {"action_type": "test", "details": "Tests failed"}
            ],
            "outcome": OutcomeType.FAILURE,
            "duration": 2400,
            "context": {"language": "python"}
        }
    ]

    # Record all experiences
    for exp in experiences + failures:
        learning.record_experience(
            task_description=exp["task"],
            actions=exp["actions"],
            outcome=exp["outcome"],
            outcome_details={},
            duration=exp["duration"],
            context=exp.get("context", {})
        )
        print(f"  ✓ Recorded: {exp['task']} ({exp['outcome'].value})")

    # View metrics
    print("\n📊 Performance Metrics:")
    metrics = learning.get_metrics()
    print(f"  Total tasks: {metrics.total_tasks}")
    print(f"  Success rate: {metrics.success_rate:.1%}")
    print(f"  Average duration: {metrics.average_duration:.1f}s")

    # Get recommendations
    print("\n💡 Getting recommendations...")
    recommendations = learning.get_recommendations(
        task_description="Implement two-factor authentication",
        context={"language": "python"}
    )

    if recommendations:
        print("  Recommendations for 'Implement two-factor authentication':")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  (Not enough data for recommendations yet)")


# =============================================================================
# Example 4: Learning from Debugging Sessions
# =============================================================================

def example_debugging_learning():
    """
    Example of learning from debugging sessions.

    This example shows:
    - Recording detailed debugging experiences
    - Identifying common patterns in successful bug fixes
    - Learning from failed attempts
    - Getting recommendations for new bugs
    """
    from yamllm import LLM
    from yamllm.agent.learning_system import LearningSystem, OutcomeType
    import random

    print("\n" + "=" * 80)
    print("Example 4: Learning from Debugging Sessions")
    print("=" * 80)

    llm = LLM(provider="openai", model="gpt-4")
    learning = LearningSystem(llm, storage_path="debugging_learning.db")

    # Simulate debugging sessions
    debug_patterns = [
        {
            "bug_type": "null_pointer",
            "actions": ["analyze_logs", "add_null_check", "test"],
            "success_rate": 0.9
        },
        {
            "bug_type": "performance",
            "actions": ["profile_code", "optimize_algorithm", "benchmark"],
            "success_rate": 0.8
        },
        {
            "bug_type": "memory_leak",
            "actions": ["profile_memory", "fix_references", "test"],
            "success_rate": 0.7
        }
    ]

    print("\n🐛 Recording debugging sessions...")

    for i in range(15):
        pattern = random.choice(debug_patterns)
        is_success = random.random() < pattern["success_rate"]

        learning.record_experience(
            task_description=f"Fix {pattern['bug_type']} bug",
            actions=[{"action_type": action} for action in pattern["actions"]],
            outcome=OutcomeType.SUCCESS if is_success else OutcomeType.FAILURE,
            outcome_details={"bug_type": pattern["bug_type"]},
            duration=random.uniform(300, 1800),
            context={"bug_category": pattern["bug_type"]}
        )

    print(f"  ✓ Recorded {i+1} debugging sessions")

    # Analyze patterns
    print("\n🔍 Analyzing debugging patterns...")
    insights = learning.analyze_and_learn(min_experiences=10)

    if insights:
        print(f"  ✓ Generated {len(insights)} insights")
        for insight in insights[:3]:
            print(f"\n  Pattern: {insight.pattern}")
            print(f"  Confidence: {insight.confidence:.1%}")
            print(f"  Recommendation: {insight.recommendation}")
    else:
        print("  (Need more diverse data for pattern analysis)")

    # Get recommendations for new bug
    print("\n💡 Getting recommendations for new bug...")
    recommendations = learning.get_recommendations(
        "Fix memory leak in application",
        context={"bug_category": "memory_leak"}
    )

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


# =============================================================================
# Example 5: Combining Multi-Agent and Learning
# =============================================================================

def example_combined_multi_agent_learning():
    """
    Example combining multi-agent collaboration with learning.

    This example shows:
    - Running multi-agent tasks
    - Recording collaborative experiences
    - Learning from multi-agent patterns
    - Applying insights to future tasks
    """
    from yamllm import LLM
    from yamllm.agent.multi_agent import (
        AgentCoordinator, CollaborativeAgent,
        AgentCapability, AgentRole
    )
    from yamllm.agent.learning_system import LearningSystem, OutcomeType

    print("\n" + "=" * 80)
    print("Example 5: Multi-Agent Collaboration with Learning")
    print("=" * 80)

    llm = LLM(provider="openai", model="gpt-4")

    # Setup multi-agent coordinator
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Setup learning system
    learning = LearningSystem(llm, storage_path="collaborative_learning.db")

    # Create agents
    print("\n👥 Setting up agent team...")
    for role in [AgentRole.RESEARCHER, AgentRole.CODER, AgentRole.REVIEWER]:
        agent = CollaborativeAgent(
            agent_id=f"{role.value}_agent",
            llm=llm,
            capability=AgentCapability(role=role, skills=[role.value])
        )
        coordinator.register_agent(agent)
        print(f"  ✓ {role.value} agent")

    # Execute multiple collaborative tasks and learn from them
    tasks = [
        "Implement API caching layer",
        "Add rate limiting to endpoints",
        "Implement request logging"
    ]

    print("\n🚀 Executing and learning from tasks...")

    for task_desc in tasks:
        print(f"\n  Task: {task_desc}")

        # Execute task
        start_time = time.time()
        result = coordinator.execute_collaborative_task(task_desc, max_iterations=8)
        duration = time.time() - start_time

        # Record experience
        outcome = OutcomeType.SUCCESS if result["tasks_completed"] > 0 else OutcomeType.FAILURE

        learning.record_experience(
            task_description=task_desc,
            actions=[{
                "agent_count": len(coordinator.agents),
                "iterations": result["iterations"],
                "tasks_completed": result["tasks_completed"]
            }],
            outcome=outcome,
            outcome_details={
                "tasks_completed": result["tasks_completed"],
                "iterations_used": result["iterations"]
            },
            duration=duration,
            context={
                "agent_roles": [a.capability.role.value for a in coordinator.agents.values()],
                "team_size": len(coordinator.agents)
            }
        )

        print(f"    ✓ Completed ({result['tasks_completed']} tasks, {duration:.1f}s)")

    # Analyze collaborative patterns
    print("\n📊 Analyzing collaborative patterns...")
    metrics = learning.get_metrics()
    print(f"  Success rate: {metrics.success_rate:.1%}")
    print(f"  Average duration: {metrics.average_duration:.1f}s")

    # Get recommendations for similar task
    print("\n💡 Recommendations for future collaborative tasks:")
    recommendations = learning.get_recommendations(
        "Implement API authentication middleware",
        context={"team_size": len(coordinator.agents)}
    )

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


# =============================================================================
# Example 6: Knowledge Export and Import
# =============================================================================

def example_knowledge_sharing():
    """
    Example of exporting and importing learned knowledge.

    This example shows:
    - Exporting knowledge from one agent
    - Importing knowledge to another agent
    - Sharing insights across systems
    """
    from yamllm import LLM
    from yamllm.agent.learning_system import LearningSystem, OutcomeType
    import json

    print("\n" + "=" * 80)
    print("Example 6: Knowledge Export and Import")
    print("=" * 80)

    llm = LLM(provider="openai", model="gpt-4")

    # Create first agent and record experiences
    print("\n🤖 Agent A: Recording experiences...")
    agent_a = LearningSystem(llm, storage_path="agent_a.db")

    for i in range(5):
        agent_a.record_experience(
            task_description=f"Implement feature {i}",
            actions=[{"action_type": "code"}],
            outcome=OutcomeType.SUCCESS,
            outcome_details={},
            duration=1000.0
        )

    print(f"  ✓ Recorded {agent_a.get_metrics().total_tasks} experiences")

    # Export knowledge
    print("\n📤 Exporting knowledge from Agent A...")
    agent_a.export_knowledge("agent_a_knowledge.json")
    print("  ✓ Exported to agent_a_knowledge.json")

    # View exported content
    with open("agent_a_knowledge.json", 'r') as f:
        knowledge = json.load(f)
    print(f"  Exported {len(knowledge.get('insights', []))} insights")

    # Create second agent and import knowledge
    print("\n🤖 Agent B: Importing knowledge...")
    agent_b = LearningSystem(llm, storage_path="agent_b.db")

    agent_b.import_knowledge("agent_a_knowledge.json")
    print("  ✓ Knowledge imported successfully")

    # Agent B can now use Agent A's insights
    summary = agent_b.get_learning_summary()
    print(f"  Agent B now has {summary['total_insights']} insights")

    # Cleanup
    Path("agent_a_knowledge.json").unlink()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("YAMLLM P2 Features Examples")
    print("=" * 80)
    print("\nThese examples demonstrate Multi-Agent Collaboration and Learning features.")
    print("\nNote: These examples require valid LLM API credentials.")
    print("=" * 80)

    try:
        # Run examples
        example_basic_multi_agent()
        example_software_dev_team()
        example_basic_learning()
        example_debugging_learning()
        example_combined_multi_agent_learning()
        example_knowledge_sharing()

        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\nMake sure you have:")
        print("  1. Valid LLM API credentials configured")
        print("  2. Required dependencies installed")
        print("  3. YAMLLM installed in development mode (pip install -e .)")

    finally:
        # Cleanup example databases
        for db_file in ["example_learning.db", "debugging_learning.db",
                        "collaborative_learning.db", "agent_a.db", "agent_b.db"]:
            db_path = Path(db_file)
            if db_path.exists():
                db_path.unlink()
