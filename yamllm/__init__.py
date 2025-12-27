"""YAMLLM - YAML-based LLM configuration and execution."""

from .core.llm import LLM
from .llm import OpenAIGPT, MistralAI, DeepSeek, GoogleGemini, AnthropicAI
from .core.config import Config
from .memory.conversation_store import ConversationStore, VectorStore
from .tools import Tool, WebSearch, Calculator, TimezoneTool, UnitConverter, WeatherTool, WebScraper
from .agent import Agent, SimpleAgent, WorkflowManager, AgentState, Task

# Advanced features (P0)
from .core.cost_tracker import CostTracker, CostSummary, CostOptimizer, BudgetExceededError
from .core.model_router import ModelRouter, TaskType, TaskComplexity
from .core.ensemble import EnsembleManager, ParallelEnsembleManager, EnsembleStrategy, EnsembleResult
from .agent.recording import SessionRecorder, SessionPlayer, RecordingManager

# P1 features
from .tools.dynamic_tool_creator import ToolCreator, DynamicTool, ToolValidator
from .code.context_intelligence import CodeContextIntelligence, CodeSymbol, SymbolType
from .tools.advanced_git import AdvancedGitWorkflow, BranchStrategy, ConflictResolutionStrategy

# P1+: Interactive Steering
from .agent.interactive_steering import InteractiveAgent, InteractiveSteering, SteeringAction, SteeringDecision
from .ui.steering_ui import SteeringUI

# P2: Multi-Agent Collaboration
from .agent.multi_agent import (
    AgentRole, AgentMessage, AgentCapability, CollaborativeTask,
    CollaborativeAgent, AgentCoordinator
)

# P2: Learning & Improvement
from .agent.learning_system import (
    LearningSystem, Experience, LearningInsight, PerformanceMetrics,
    OutcomeType, ImprovementType, ExperienceStore, PatternAnalyzer
)

__version__ = "0.1.13"

__all__ = [
    # Core
    "LLM",
    "OpenAIGPT",
    "MistralAI",
    "DeepSeek",
    "GoogleGemini",
    "AnthropicAI",
    "Config",
    # Memory
    "ConversationStore",
    "VectorStore",
    # Tools
    "Tool",
    "WebSearch",
    "Calculator",
    "TimezoneTool",
    "UnitConverter",
    "WeatherTool",
    "WebScraper",
    # Agent
    "Agent",
    "SimpleAgent",
    "WorkflowManager",
    "AgentState",
    "Task",
    # Cost Tracking
    "CostTracker",
    "CostSummary",
    "CostOptimizer",
    "BudgetExceededError",
    # Model Routing
    "ModelRouter",
    "TaskType",
    "TaskComplexity",
    # Ensemble
    "EnsembleManager",
    "ParallelEnsembleManager",
    "EnsembleStrategy",
    "EnsembleResult",
    # Recording
    "SessionRecorder",
    "SessionPlayer",
    "RecordingManager",
    # P1: Dynamic Tools
    "ToolCreator",
    "DynamicTool",
    "ToolValidator",
    # P1: Code Intelligence
    "CodeContextIntelligence",
    "CodeSymbol",
    "SymbolType",
    # P1: Advanced Git
    "AdvancedGitWorkflow",
    "BranchStrategy",
    "ConflictResolutionStrategy",
    # P1+: Interactive Steering
    "InteractiveAgent",
    "InteractiveSteering",
    "SteeringAction",
    "SteeringDecision",
    "SteeringUI",
    # P2: Multi-Agent Collaboration
    "AgentRole",
    "AgentMessage",
    "AgentCapability",
    "CollaborativeTask",
    "CollaborativeAgent",
    "AgentCoordinator",
    # P2: Learning & Improvement
    "LearningSystem",
    "Experience",
    "LearningInsight",
    "PerformanceMetrics",
    "OutcomeType",
    "ImprovementType",
    "ExperienceStore",
    "PatternAnalyzer",
]