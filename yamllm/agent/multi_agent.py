"""Collaborative multi-agent system for coordinated task execution."""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class AgentRole(Enum):
    """Specialized agent roles."""
    COORDINATOR = "coordinator"  # Coordinates other agents
    RESEARCHER = "researcher"    # Gathers information
    CODER = "coder"             # Writes code
    REVIEWER = "reviewer"        # Reviews work
    TESTER = "tester"           # Tests functionality
    DEBUGGER = "debugger"        # Fixes bugs
    DOCUMENTER = "documenter"    # Writes documentation
    ANALYST = "analyst"          # Analyzes data/problems


@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    message_type: str  # "request", "response", "broadcast", "notification"
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 5=high


@dataclass
class AgentCapability:
    """Defines what an agent can do."""
    role: AgentRole
    skills: List[str]
    max_concurrent_tasks: int = 1
    confidence: float = 0.8  # Self-reported confidence (0-1)


@dataclass
class CollaborativeTask:
    """Task for multi-agent collaboration."""
    task_id: str
    description: str
    required_roles: List[AgentRole]
    assigned_agents: Dict[AgentRole, str] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)


class CollaborativeAgent:
    """An agent that can collaborate with others."""

    def __init__(
        self,
        agent_id: str,
        llm,
        capability: AgentCapability,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize collaborative agent.

        Args:
            agent_id: Unique identifier for this agent
            llm: LLM instance for this agent
            capability: Agent's capabilities
            logger: Optional logger
        """
        self.agent_id = agent_id
        self.llm = llm
        self.capability = capability
        self.logger = logger or logging.getLogger(__name__)

        # Communication
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []

        # State
        self.current_tasks: List[str] = []
        self.completed_tasks: List[str] = []

    def can_handle(self, task: CollaborativeTask) -> bool:
        """Check if agent can handle task based on role."""
        return self.capability.role in task.required_roles

    def send_message(
        self,
        to_agent: str,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 1
    ):
        """Send message to another agent."""
        message = AgentMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            priority=priority
        )
        self.outbox.append(message)
        return message

    def receive_message(self, message: AgentMessage):
        """Receive message from another agent."""
        self.inbox.append(message)

    def process_messages(self) -> List[AgentMessage]:
        """Process pending messages and generate responses."""
        responses = []

        for message in self.inbox:
            if message.message_type == "request":
                response = self._handle_request(message)
                if response:
                    responses.append(response)

        # Clear processed messages
        self.inbox.clear()

        return responses

    def _handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle request message."""
        request_type = message.content.get("request_type")

        if request_type == "execute_task":
            # Execute task and return result
            task_desc = message.content.get("task_description")
            result = self._execute_task(task_desc)

            return self.send_message(
                to_agent=message.from_agent,
                message_type="response",
                content={"result": result, "status": "completed"}
            )

        elif request_type == "review":
            # Review work
            work = message.content.get("work")
            review = self._review_work(work)

            return self.send_message(
                to_agent=message.from_agent,
                message_type="response",
                content={"review": review}
            )

        return None

    def _execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a task using LLM."""
        prompt = f"""As a {self.capability.role.value}, execute this task:

Task: {task_description}

Your skills: {', '.join(self.capability.skills)}

Provide your response:"""

        try:
            response = self.llm.query(prompt)
            return {
                "success": True,
                "result": response,
                "agent": self.agent_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": self.agent_id
            }

    def _review_work(self, work: str) -> Dict[str, Any]:
        """Review work from another agent."""
        prompt = f"""As a {self.capability.role.value}, review this work:

{work}

Provide feedback on:
1. Quality
2. Completeness
3. Issues or improvements needed

Your review:"""

        try:
            review = self.llm.query(prompt)
            return {
                "reviewer": self.agent_id,
                "feedback": review,
                "approved": "approve" in review.lower()
            }
        except Exception as e:
            return {
                "reviewer": self.agent_id,
                "error": str(e),
                "approved": False
            }


class AgentCoordinator:
    """Coordinates multiple agents for collaborative tasks."""

    def __init__(
        self,
        coordinator_llm,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize agent coordinator.

        Args:
            coordinator_llm: LLM for coordination decisions
            logger: Optional logger
        """
        self.coordinator_llm = coordinator_llm
        self.logger = logger or logging.getLogger(__name__)

        # Agent registry
        self.agents: Dict[str, CollaborativeAgent] = {}

        # Task management
        self.tasks: Dict[str, CollaborativeTask] = {}
        self.task_queue: List[str] = []

        # Communication
        self.message_bus: List[AgentMessage] = []

    def register_agent(self, agent: CollaborativeAgent):
        """Register an agent with the coordinator."""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.capability.role.value})")

    def create_task(
        self,
        task_id: str,
        description: str,
        required_roles: List[AgentRole],
        dependencies: Optional[List[str]] = None
    ) -> CollaborativeTask:
        """Create a collaborative task."""
        task = CollaborativeTask(
            task_id=task_id,
            description=description,
            required_roles=required_roles,
            dependencies=dependencies or []
        )

        self.tasks[task_id] = task
        self.task_queue.append(task_id)

        return task

    def assign_agents(self, task_id: str) -> bool:
        """Assign agents to a task based on roles."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        # Find agents for each required role
        for role in task.required_roles:
            # Find available agent with this role
            for agent_id, agent in self.agents.items():
                if agent.capability.role == role and agent_id not in task.assigned_agents.values():
                    # Check capacity
                    if len(agent.current_tasks) < agent.capability.max_concurrent_tasks:
                        task.assigned_agents[role] = agent_id
                        agent.current_tasks.append(task_id)
                        break

        # Check if all roles assigned
        return len(task.assigned_agents) == len(task.required_roles)

    def execute_collaborative_task(
        self,
        goal: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Execute a goal using collaborative agents.

        Args:
            goal: High-level goal
            max_iterations: Maximum coordination iterations

        Returns:
            Results from collaborative execution
        """
        self.logger.info(f"Starting collaborative execution for: {goal}")

        # Decompose goal into tasks
        self._decompose_goal(goal)

        # Execute tasks collaboratively
        results = {}

        for iteration in range(max_iterations):
            # Process message bus
            self._route_messages()

            # Execute ready tasks
            for task_id in list(self.task_queue):
                task = self.tasks[task_id]

                # Check dependencies
                if not self._check_dependencies(task):
                    continue

                # Assign agents if not already assigned
                if not task.assigned_agents:
                    if not self.assign_agents(task_id):
                        self.logger.warning(f"Could not assign agents to task: {task_id}")
                        continue

                # Execute task
                task_result = self._execute_task(task)
                results[task_id] = task_result

                # Remove from queue
                self.task_queue.remove(task_id)

            # Check if done
            if not self.task_queue:
                break

        return {
            "goal": goal,
            "tasks_completed": len(results),
            "results": results,
            "iterations": iteration + 1
        }

    def _decompose_goal(self, goal: str) -> List[CollaborativeTask]:
        """Decompose goal into collaborative tasks."""
        prompt = f"""Decompose this goal into collaborative tasks:

Goal: {goal}

Available agent roles:
- researcher: Gathers information
- coder: Writes code
- reviewer: Reviews work
- tester: Tests functionality
- debugger: Fixes bugs
- documenter: Writes documentation

Create a list of tasks in JSON format:
[
  {{"task_id": "task_1", "description": "...", "required_roles": ["researcher"], "dependencies": []}},
  {{"task_id": "task_2", "description": "...", "required_roles": ["coder"], "dependencies": ["task_1"]}}
]

Tasks:"""

        try:
            response = self.coordinator_llm.query(prompt)

            # Parse JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response

            task_defs = json.loads(json_str)

            tasks = []
            for task_def in task_defs:
                task = self.create_task(
                    task_id=task_def["task_id"],
                    description=task_def["description"],
                    required_roles=[AgentRole(r) for r in task_def["required_roles"]],
                    dependencies=task_def.get("dependencies", [])
                )
                tasks.append(task)

            return tasks

        except Exception as e:
            self.logger.error(f"Failed to decompose goal: {e}")
            # Fallback: create single task
            task = self.create_task(
                task_id="task_1",
                description=goal,
                required_roles=[AgentRole.RESEARCHER, AgentRole.CODER]
            )
            return [task]

    def _check_dependencies(self, task: CollaborativeTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != "completed":
                return False
        return True

    def _execute_task(self, task: CollaborativeTask) -> Dict[str, Any]:
        """Execute a collaborative task."""
        task.status = "in_progress"

        results = {}

        # Execute with each assigned agent
        for role, agent_id in task.assigned_agents.items():
            agent = self.agents[agent_id]

            # Send task to agent
            message = AgentMessage(
                from_agent="coordinator",
                to_agent=agent_id,
                message_type="request",
                content={
                    "request_type": "execute_task",
                    "task_description": task.description,
                    "task_id": task.task_id
                }
            )

            agent.receive_message(message)

            # Process agent messages
            responses = agent.process_messages()

            # Collect results
            for response in responses:
                results[agent_id] = response.content

        # If there's a reviewer, have them review
        if AgentRole.REVIEWER in task.assigned_agents:
            reviewer_id = task.assigned_agents[AgentRole.REVIEWER]
            reviewer = self.agents[reviewer_id]

            # Get work to review
            work_results = [r.get("result") for r in results.values() if r.get("result")]

            if work_results:
                review_msg = AgentMessage(
                    from_agent="coordinator",
                    to_agent=reviewer_id,
                    message_type="request",
                    content={
                        "request_type": "review",
                        "work": "\n".join(str(w) for w in work_results)
                    }
                )

                reviewer.receive_message(review_msg)
                review_responses = reviewer.process_messages()

                for response in review_responses:
                    results["review"] = response.content

        task.status = "completed"
        task.result = results

        # Clean up agent tasks
        for agent_id in task.assigned_agents.values():
            agent = self.agents[agent_id]
            if task.task_id in agent.current_tasks:
                agent.current_tasks.remove(task.task_id)
                agent.completed_tasks.append(task.task_id)

        return results

    def _route_messages(self):
        """Route messages between agents via message bus."""
        # Collect outgoing messages from all agents
        for agent in self.agents.values():
            self.message_bus.extend(agent.outbox)
            agent.outbox.clear()

        # Deliver messages to recipients
        for message in self.message_bus:
            if message.to_agent == "coordinator":
                # Message for coordinator
                self.logger.info(f"Coordinator received: {message.content}")
            elif message.to_agent == "broadcast":
                # Broadcast to all agents
                for agent in self.agents.values():
                    if agent.agent_id != message.from_agent:
                        agent.receive_message(message)
            else:
                # Direct message
                recipient = self.agents.get(message.to_agent)
                if recipient:
                    recipient.receive_message(message)

        # Clear message bus
        self.message_bus.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "registered_agents": len(self.agents),
            "active_tasks": len(self.task_queue),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "agents": {
                agent_id: {
                    "role": agent.capability.role.value,
                    "current_tasks": len(agent.current_tasks),
                    "completed_tasks": len(agent.completed_tasks)
                }
                for agent_id, agent in self.agents.items()
            }
        }
