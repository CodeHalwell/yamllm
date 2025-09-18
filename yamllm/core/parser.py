from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import yaml

class ProviderSettings(BaseModel):
    name: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MCPConnectorSettings(BaseModel):
    """
    Settings for an MCP (Model Context Protocol) connector.
    
    MCP allows integration with external tools and services in a standardized way.
    Supports WebSocket, HTTP, and stdio transports.
    """
    name: str
    url: str  # MCP server endpoint URL or command for stdio
    transport: str = "http"  # "websocket", "http", or "stdio"
    authentication: Optional[str] = None  # Can be an environment variable reference like ${MCP_API_KEY}
    description: Optional[str] = None
    tool_prefix: Optional[str] = None  # Prefix for tool names from this connector
    enabled: bool = True

class ModelSettings(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: List[str] = Field(default_factory=list)

class RetrySettings(BaseModel):
    max_attempts: int = 3
    initial_delay: int = 1
    backoff_factor: int = 2

class RequestSettings(BaseModel):
    timeout: int = 30
    retry: RetrySettings

class VectorStoreSettings(BaseModel):
    index_path: Optional[str] = None
    metadata_path: Optional[str] = None
    top_k: Optional[int] = None

class MemorySettings(BaseModel):
    enabled: bool = False
    max_messages: int = 10
    session_id: Optional[str] = None
    conversation_db: Optional[str] = None
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

class ContextSettings(BaseModel):
    system_prompt: str = "You are a helpful assistant."
    max_context_length: int = 4096
    memory: MemorySettings

class OutputSettings(BaseModel):
    format: str = "text"
    stream: bool = False

class LoggingSettings(BaseModel):
    level: str = "INFO"
    file: str = "yamllm.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console: bool = False
    rotate: bool = False
    rotate_max_bytes: int = 1048576  # 1MB
    rotate_backup_count: int = 3
    json_format: bool = False
    modules: Dict[str, str] = Field(default_factory=dict)  # per-module levels

class SafetySettings(BaseModel):
    content_filtering: bool = True
    max_requests_per_minute: int = 60
    sensitive_keywords: List[str] = Field(default_factory=list)

class Tools(BaseModel):
    enabled: bool = True
    tool_timeout: int = 30
    # Accept legacy key `tool_list` by alias while normalizing to `tools` internally
    tools: List[str] = Field(default_factory=list)
    tool_list: Optional[List[str]] = Field(default=None, alias="tool_list")
    packs: List[str] = Field(default_factory=list)
    mcp_connectors: List[MCPConnectorSettings] = Field(default_factory=list)
    include_help_tool: bool = True
    # Security settings
    safe_mode: bool = False
    allow_network: bool = True
    allow_filesystem: bool = True
    allowed_paths: List[str] = Field(default_factory=list)
    blocked_domains: List[str] = Field(default_factory=list)
    # Gating
    gate_web_search: bool = True

class EmbeddingsSettings(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None

class ThinkingSettings(BaseModel):
    enabled: bool = False
    show_tool_reasoning: bool = True
    model: Optional[str] = None
    max_tokens: int = 2000
    stream_thinking: bool = True
    save_thinking: bool = False
    thinking_temperature: float = 0.7
    # If true, when the prompt clearly doesn't need tools, emit a compact, local
    # thinking note (no model call) and proceed directly to the answer.
    compact_for_non_tool: bool = True
    # New: thinking display mode and hygiene
    # off: never show thinking; on: always show; auto: show only for complex tasks
    mode: str = "auto"
    # Redact thinking content from logs (UI still displays)
    redact_logs: bool = True

class YamlLMConfig(BaseModel):
    provider: ProviderSettings
    model_settings: ModelSettings
    request: RequestSettings
    context: ContextSettings
    output: OutputSettings
    logging: LoggingSettings
    safety: SafetySettings
    tools: Tools
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    thinking: ThinkingSettings = Field(default_factory=ThinkingSettings)


def parse_yaml_config(yaml_file_path: str) -> YamlLMConfig:
    """
    Parses a YAML file into a YamlLMConfig Pydantic model.

            Args:
        yaml_file_path (str): The path to the YAML file to be parsed.

    Returns:
        YamlLMConfig: An instance of YamlLMConfig populated with the data from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file is not found at the specified path.
        yaml.YAMLError: If there is an error parsing the YAML file.
        ValueError: If the YAML file is empty or could not be parsed into a dictionary.
        Exception: For any other unexpected errors that occur during parsing.
    """
    try: # Added try-except block for file opening
        with open(yaml_file_path, 'r') as file:
            yaml_content = file.read() # Read the file content into a string
            yaml_dict = yaml.safe_load(yaml_content)

            if yaml_dict is None: # Check if yaml_dict is None
                raise ValueError("YAML file was empty or could not be parsed into a dictionary.")

            config = YamlLMConfig(**yaml_dict)
            return config

    except FileNotFoundError:
        # Library code should not print; re-raise for caller to handle
        raise
    except yaml.YAMLError:
        raise
    except ValueError:
        raise
    except Exception:
        raise
