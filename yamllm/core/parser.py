from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl
import yaml

class ProviderSettings(BaseModel):
    name: str
    model: str
    api_key: None
    base_url: Optional[str] = None
    extra_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MCPConnectorSettings(BaseModel):
    """
    Settings for an MCP (Model Context Protocol) connector.
    
    MCP allows integration with external tools and services in a standardized way.
    """
    name: str
    url: str  # MCP server endpoint URL
    authentication: Optional[str] = None  # Can be an environment variable reference like ${MCP_API_KEY}
    description: Optional[str] = None
    tool_prefix: Optional[str] = None  # Prefix for tool names from this connector
    enabled: bool = True

class ModelSettings(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: Optional[float]
    presence_penalty: Optional[float]
    stop_sequences: List[str] = []

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
    vector_store: VectorStoreSettings = VectorStoreSettings()

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

class SafetySettings(BaseModel):
    content_filtering: bool = True
    max_requests_per_minute: int = 60
    sensitive_keywords: List[str] = []

class Tools(BaseModel):
    enabled: bool = True
    tool_timeout: int = 30
    tool_list: List[str] = []
    mcp_connectors: List[MCPConnectorSettings] = Field(default_factory=list)

class YamlLMConfig(BaseModel):
    provider: ProviderSettings
    model_settings: ModelSettings
    request: RequestSettings
    context: ContextSettings
    output: OutputSettings
    logging: LoggingSettings
    safety: SafetySettings
    tools: Tools


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
        print(f"Error: YAML file not found at path: {yaml_file_path}")
        raise # Re-raise the exception so the main block catches it
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise # Re-raise the exception
    except ValueError as ve: # Catch the new ValueError for empty/unparsable YAML
        print(f"ValueError: {ve}")
        raise # Re-raise the exception
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise # Re-raise the exception
