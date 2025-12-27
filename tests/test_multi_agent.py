"""Tests for multi-agent collaboration system."""

from datetime import datetime
from yamllm.agent.multi_agent import (
    AgentRole,
    AgentMessage,
    AgentCapability,
    CollaborativeTask,
    CollaborativeAgent,
    AgentCoordinator
)


class MockLLM:
    """Mock LLM for testing."""

    def query(self, prompt: str) -> str:
        """Return mock response."""
        if "decompose" in prompt.lower():
            return """```json
[
    {"task_id": "task_1", "description": "Research requirements", "required_roles": ["researcher"], "dependencies": []},
    {"task_id": "task_2", "description": "Write code", "required_roles": ["coder"], "dependencies": ["task_1"]}
]
```"""
        return "Mock response for task execution"


def test_agent_role_enum():
    """Test agent role enum."""
    assert AgentRole.COORDINATOR.value == "coordinator"
    assert AgentRole.RESEARCHER.value == "researcher"
    assert AgentRole.CODER.value == "coder"
    assert AgentRole.REVIEWER.value == "reviewer"
    assert AgentRole.TESTER.value == "tester"
    assert AgentRole.DEBUGGER.value == "debugger"
    assert AgentRole.DOCUMENTER.value == "documenter"
    assert AgentRole.ANALYST.value == "analyst"


def test_agent_message():
    """Test agent message dataclass."""
    message = AgentMessage(
        from_agent="agent1",
        to_agent="agent2",
        message_type="request",
        content={"data": "test"},
        priority=5
    )

    assert message.from_agent == "agent1"
    assert message.to_agent == "agent2"
    assert message.message_type == "request"
    assert message.content["data"] == "test"
    assert message.priority == 5
    assert isinstance(message.timestamp, datetime)


def test_agent_capability():
    """Test agent capability dataclass."""
    capability = AgentCapability(
        role=AgentRole.CODER,
        skills=["python", "javascript"],
        max_concurrent_tasks=3,
        confidence=0.9
    )

    assert capability.role == AgentRole.CODER
    assert "python" in capability.skills
    assert capability.max_concurrent_tasks == 3
    assert capability.confidence == 0.9


def test_collaborative_task():
    """Test collaborative task dataclass."""
    task = CollaborativeTask(
        task_id="task_1",
        description="Implement feature X",
        required_roles=[AgentRole.CODER, AgentRole.REVIEWER]
    )

    assert task.task_id == "task_1"
    assert task.description == "Implement feature X"
    assert AgentRole.CODER in task.required_roles
    assert task.status == "pending"
    assert task.result is None


def test_collaborative_agent_initialization():
    """Test collaborative agent initialization."""
    llm = MockLLM()
    capability = AgentCapability(
        role=AgentRole.CODER,
        skills=["coding"]
    )

    agent = CollaborativeAgent(
        agent_id="coder_1",
        llm=llm,
        capability=capability
    )

    assert agent.agent_id == "coder_1"
    assert agent.capability.role == AgentRole.CODER
    assert len(agent.inbox) == 0
    assert len(agent.outbox) == 0
    assert len(agent.current_tasks) == 0


def test_collaborative_agent_can_handle():
    """Test agent can handle task."""
    llm = MockLLM()
    capability = AgentCapability(role=AgentRole.CODER, skills=["coding"])
    agent = CollaborativeAgent("coder_1", llm, capability)

    task = CollaborativeTask(
        task_id="task_1",
        description="Write code",
        required_roles=[AgentRole.CODER]
    )

    assert agent.can_handle(task)

    task2 = CollaborativeTask(
        task_id="task_2",
        description="Test code",
        required_roles=[AgentRole.TESTER]
    )

    assert not agent.can_handle(task2)


def test_collaborative_agent_send_message():
    """Test agent sending message."""
    llm = MockLLM()
    capability = AgentCapability(role=AgentRole.CODER, skills=["coding"])
    agent = CollaborativeAgent("coder_1", llm, capability)

    message = agent.send_message(
        to_agent="reviewer_1",
        message_type="request",
        content={"task": "review my code"}
    )

    assert message.from_agent == "coder_1"
    assert message.to_agent == "reviewer_1"
    assert message.message_type == "request"
    assert len(agent.outbox) == 1


def test_collaborative_agent_receive_message():
    """Test agent receiving message."""
    llm = MockLLM()
    capability = AgentCapability(role=AgentRole.CODER, skills=["coding"])
    agent = CollaborativeAgent("coder_1", llm, capability)

    message = AgentMessage(
        from_agent="coordinator",
        to_agent="coder_1",
        message_type="request",
        content={"task": "write code"}
    )

    agent.receive_message(message)

    assert len(agent.inbox) == 1
    assert agent.inbox[0].from_agent == "coordinator"


def test_collaborative_agent_process_messages():
    """Test agent processing messages."""
    llm = MockLLM()
    capability = AgentCapability(role=AgentRole.CODER, skills=["coding"])
    agent = CollaborativeAgent("coder_1", llm, capability)

    # Send execute_task request
    message = AgentMessage(
        from_agent="coordinator",
        to_agent="coder_1",
        message_type="request",
        content={
            "request_type": "execute_task",
            "task_description": "Write a function"
        }
    )

    agent.receive_message(message)
    responses = agent.process_messages()

    assert len(responses) == 1
    assert responses[0].message_type == "response"
    assert "result" in responses[0].content


def test_agent_coordinator_initialization():
    """Test coordinator initialization."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    assert len(coordinator.agents) == 0
    assert len(coordinator.tasks) == 0
    assert len(coordinator.task_queue) == 0


def test_agent_coordinator_register_agent():
    """Test registering agent with coordinator."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    capability = AgentCapability(role=AgentRole.CODER, skills=["coding"])
    agent = CollaborativeAgent("coder_1", llm, capability)

    coordinator.register_agent(agent)

    assert "coder_1" in coordinator.agents
    assert coordinator.agents["coder_1"] == agent


def test_agent_coordinator_create_task():
    """Test creating collaborative task."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    task = coordinator.create_task(
        task_id="task_1",
        description="Implement feature",
        required_roles=[AgentRole.CODER]
    )

    assert task.task_id == "task_1"
    assert "task_1" in coordinator.tasks
    assert "task_1" in coordinator.task_queue


def test_agent_coordinator_assign_agents():
    """Test assigning agents to task."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Register agents
    coder = CollaborativeAgent(
        "coder_1",
        llm,
        AgentCapability(role=AgentRole.CODER, skills=["coding"])
    )
    reviewer = CollaborativeAgent(
        "reviewer_1",
        llm,
        AgentCapability(role=AgentRole.REVIEWER, skills=["reviewing"])
    )

    coordinator.register_agent(coder)
    coordinator.register_agent(reviewer)

    # Create task
    task = coordinator.create_task(
        task_id="task_1",
        description="Write and review code",
        required_roles=[AgentRole.CODER, AgentRole.REVIEWER]
    )

    # Assign agents
    success = coordinator.assign_agents("task_1")

    assert success
    assert AgentRole.CODER in task.assigned_agents
    assert AgentRole.REVIEWER in task.assigned_agents
    assert "task_1" in coder.current_tasks
    assert "task_1" in reviewer.current_tasks


def test_agent_coordinator_decompose_goal():
    """Test goal decomposition."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    tasks = coordinator._decompose_goal("Build a web app")

    assert len(tasks) >= 1
    assert all(isinstance(task, CollaborativeTask) for task in tasks)


def test_agent_coordinator_check_dependencies():
    """Test dependency checking."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Create tasks with dependencies
    task1 = coordinator.create_task(
        task_id="task_1",
        description="First task",
        required_roles=[AgentRole.RESEARCHER]
    )

    task2 = coordinator.create_task(
        task_id="task_2",
        description="Second task",
        required_roles=[AgentRole.CODER],
        dependencies=["task_1"]
    )

    # task2 depends on task1, which is not completed
    assert not coordinator._check_dependencies(task2)

    # Complete task1
    task1.status = "completed"
    assert coordinator._check_dependencies(task2)


def test_agent_coordinator_execute_task():
    """Test task execution."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Register agent
    coder = CollaborativeAgent(
        "coder_1",
        llm,
        AgentCapability(role=AgentRole.CODER, skills=["coding"])
    )
    coordinator.register_agent(coder)

    # Create and assign task
    task = coordinator.create_task(
        task_id="task_1",
        description="Write code",
        required_roles=[AgentRole.CODER]
    )
    coordinator.assign_agents("task_1")

    # Execute task
    result = coordinator._execute_task(task)

    assert "coder_1" in result
    assert task.status == "completed"
    assert "task_1" not in coder.current_tasks
    assert "task_1" in coder.completed_tasks


def test_agent_coordinator_route_messages():
    """Test message routing."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Register agents
    agent1 = CollaborativeAgent(
        "agent_1",
        llm,
        AgentCapability(role=AgentRole.CODER, skills=["coding"])
    )
    agent2 = CollaborativeAgent(
        "agent_2",
        llm,
        AgentCapability(role=AgentRole.REVIEWER, skills=["reviewing"])
    )

    coordinator.register_agent(agent1)
    coordinator.register_agent(agent2)

    # Agent1 sends message to agent2
    agent1.send_message(
        to_agent="agent_2",
        message_type="request",
        content={"data": "test"}
    )

    # Route messages
    coordinator._route_messages()

    # Agent2 should have received the message
    assert len(agent2.inbox) == 1
    assert agent2.inbox[0].from_agent == "agent_1"


def test_agent_coordinator_get_status():
    """Test getting coordinator status."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Register agents
    coder = CollaborativeAgent(
        "coder_1",
        llm,
        AgentCapability(role=AgentRole.CODER, skills=["coding"])
    )
    coordinator.register_agent(coder)

    # Create task
    coordinator.create_task(
        task_id="task_1",
        description="Write code",
        required_roles=[AgentRole.CODER]
    )

    status = coordinator.get_status()

    assert status["registered_agents"] == 1
    assert status["active_tasks"] == 1
    assert status["completed_tasks"] == 0
    assert "coder_1" in status["agents"]


def test_collaborative_execution_integration():
    """Test end-to-end collaborative execution."""
    llm = MockLLM()
    coordinator = AgentCoordinator(coordinator_llm=llm)

    # Register multiple agents
    researcher = CollaborativeAgent(
        "researcher_1",
        llm,
        AgentCapability(role=AgentRole.RESEARCHER, skills=["research"])
    )
    coder = CollaborativeAgent(
        "coder_1",
        llm,
        AgentCapability(role=AgentRole.CODER, skills=["coding"])
    )

    coordinator.register_agent(researcher)
    coordinator.register_agent(coder)

    # Execute collaborative task
    result = coordinator.execute_collaborative_task(
        goal="Build a simple calculator",
        max_iterations=5
    )

    assert "goal" in result
    assert result["tasks_completed"] >= 0
    assert result["iterations"] <= 5
    assert "results" in result
