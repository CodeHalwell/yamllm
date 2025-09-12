from yamllm.core.parser import parse_yaml_config, YamlLMConfig
from yamllm.core.config_validator import ConfigValidator
from yamllm.core.exceptions import ConfigurationError
from yamllm.memory import ConversationStore, VectorStore
from openai import OpenAI, OpenAIError
from typing import Optional, Dict, Any, Union, List, Callable, TypeVar, cast
import time
import os
import logging
from logging.handlers import RotatingFileHandler
import dotenv
import json
import re
from yamllm.tools.utility_tools import (
    WebSearch,
    Calculator,
    TimezoneTool,
    UnitConverter,
    WeatherTool,
    WebScraper,
    DateTimeTool,
    UUIDTool,
    RandomStringTool,
    RandomNumberTool,
    Base64EncodeTool,
    Base64DecodeTool,
    HashTool,
    JSONTool,
    RegexExtractTool,
    LoremIpsumTool,
    FileReadTool,
    FileSearchTool,
    CSVPreviewTool,
    URLMetadataTool,
    WebHeadlinesTool,
    ToolsHelpTool,
)
from yamllm.tools.manager import ToolManager
from yamllm.tools.security import SecurityManager

# Import provider interfaces
from yamllm.providers.base import BaseProvider
from yamllm.providers.factory import ProviderFactory
from yamllm.providers.capabilities import get_provider_capabilities
from typing import Optional as _OptionalCallback
from yamllm.core.thinking import ThinkingManager


dotenv.load_dotenv()


def setup_logging(config):
    """
    Set up logging configuration for the yamllm application.
    This function configures the logging settings based on the provided configuration.
    It sets the logging level for the 'httpx' and 'urllib3' libraries to WARNING to suppress
    INFO messages, disables propagation to the root logger, and configures the 'yamllm' logger
    with the specified logging level, file handler, and formatter.
    Args:
        config (object): A configuration object that contains logging settings. It should have
                         the following attributes:
                         - logging.level (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
                         - logging.file (str): The file path where log messages should be written.
                         - logging.format (str): The format string for log messages.
    Returns:
        logging.Logger: The configured logger for the 'yamllm' application.
    """
    # Set logging level for httpx and urllib3 to WARNING to suppress INFO messages
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Disable propagation to root logger
    logging.getLogger('yamllm').propagate = False

    # Get or create yamllm logger
    logger = logging.getLogger('yamllm')
    logger.setLevel(getattr(logging, config.logging.level))

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Ensure log directory exists if a path is provided
    log_path = config.logging.file
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # Best-effort; if directory creation fails, fallback to default behavior
        pass

    # Choose handler: rotating or plain file
    class _JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            import json as _json
            base = {
                "ts": self.formatTime(record, datefmt=None),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                base["exc_info"] = self.formatException(record.exc_info)
            return _json.dumps(base, ensure_ascii=False)

    formatter = _JSONFormatter() if getattr(config.logging, "json", False) else logging.Formatter(config.logging.format)
    if getattr(config.logging, "rotate", False):
        max_bytes = getattr(config.logging, "rotate_max_bytes", 1048576)
        backup_count = getattr(config.logging, "rotate_backup_count", 3)
        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    else:
        file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional console handler
    if getattr(config.logging, "console", False):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Per-module levels (optional)
    try:
        modules = getattr(config.logging, "modules", {}) or {}
        for name, lvl in modules.items():
            try:
                logging.getLogger(name).setLevel(getattr(logging, str(lvl).upper(), logging.INFO))
            except Exception:
                pass
    except Exception:
        pass

    return logger


class LLM(object):
    """
    Main LLM interface class for YAMLLM.

    This class handles configuration loading and API interactions
    with language models. It serves as a base class for provider-specific
    implementations and supports a provider-agnostic interface.

    Args:
        config_path (str): Path to YAML configuration file
        api_key (str): API key for the LLM service

    Examples:
        >>> llm = LLM(config_path = "config.yaml", api_key = "your-api-key")
        >>> response = llm.query("Hello, world!")
    """
    # Provider selection is handled exclusively by ProviderFactory
    
    def __init__(self, config_path: str, api_key: str) -> None:
        """
        Initialize the LLM instance with the given configuration path.

        Args:
            config_path (str): Path to the YAML configuration file
            api_key (str): API key for the LLM service
        """
        # Basic configuration
        self.config_path = config_path
        self.api_key = api_key
        self.config: YamlLMConfig = self.load_config()
        self.logger = setup_logging(self.config)

        # Provider settings
        self.provider_name = self.config.provider.name if not hasattr(self, 'provider') else self.provider
        # Normalize provider attribute for internal checks
        self.provider = self.provider_name
        self.model = self.config.provider.model
        self.base_url = self.config.provider.base_url
        self.extra_settings = getattr(self.config.provider, 'extra_settings', {})

        # Model settings
        self.temperature = self.config.model_settings.temperature
        self.max_tokens = self.config.model_settings.max_tokens
        self.top_p = self.config.model_settings.top_p
        self.stop_sequences = self.config.model_settings.stop_sequences

        # Request settings
        self.request_timeout = self.config.request.timeout
        self.retry_max_attempts = self.config.request.retry.max_attempts
        self.retry_initial_delay = self.config.request.retry.initial_delay
        self.retry_backoff_factor = self.config.request.retry.backoff_factor

        # Context settings
        self.system_prompt = self.config.context.system_prompt
        self.max_context_length = self.config.context.max_context_length

        # Memory settings (robust to partial/mocked config)
        try:
            self.memory_enabled = bool(self.config.context.memory.enabled)
            self.memory_max_messages = int(self.config.context.memory.max_messages)
            self.session_id = getattr(self.config.context.memory, 'session_id', None)
            self.conversation_db_path = getattr(self.config.context.memory, 'conversation_db', None)
            self.vector_index_path = getattr(self.config.context.memory.vector_store, 'index_path', None)
            self.vector_metadata_path = getattr(self.config.context.memory.vector_store, 'metadata_path', None)
            self.vector_store_top_k = int(getattr(self.config.context.memory.vector_store, 'top_k', 5) or 5)
        except Exception:
            # Defaults when config is mocked or fields missing
            self.memory_enabled = False
            self.memory_max_messages = 10
            self.session_id = None
            self.conversation_db_path = None
            self.vector_index_path = os.path.join("memory", "vector_store", "faiss_index.idx")
            self.vector_metadata_path = os.path.join("memory", "vector_store", "metadata.pkl")
            self.vector_store_top_k = 5

        # Output settings
        self.output_format = self.config.output.format
        self.output_stream = self.config.output.stream
        # Streaming callback (optional) for UI rendering
        self.stream_callback: _OptionalCallback[[str], None] = None
        # Event callback (optional) for tool/model events
        self.event_callback: _OptionalCallback[[Dict[str, Any]], None] = None

        # Tool settings
        self.tools_enabled = self.config.tools.enabled
        self.tools = self.config.tools.tools
        self.tools_timeout = self.config.tools.tool_timeout
        # Tool manager: central registration/execution
        self.tool_manager = ToolManager(timeout=self.tools_timeout, logger=self.logger)

        # Usage and tool stats
        self._last_usage: Dict[str, int] | None = None
        self._total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._tool_call_count: int = 0

        # Safety settings
        self.content_filtering = self.config.safety.content_filtering
        self.max_requests_per_minute = self.config.safety.max_requests_per_minute
        self.sensitive_keywords = self.config.safety.sensitive_keywords

        # Embeddings settings (optional)
        self.embeddings_provider_name = None
        self.embedding_model = "text-embedding-3-small"
        try:
            if hasattr(self.config, "embeddings") and self.config.embeddings:
                self.embeddings_provider_name = getattr(self.config.embeddings, "provider", None)
                if getattr(self.config.embeddings, "model", None):
                    self.embedding_model = self.config.embeddings.model
        except Exception:
            # Backwards compatibility if config is a mock or missing fields
            pass

        # Initialize provider client
        self.provider_client = self._initialize_provider()
        
        # Initialize MCP client if MCP connectors are configured
        self.mcp_client = None
        if self.config.tools.mcp_connectors:
            try:
                from yamllm.mcp.client import MCPClient
                from yamllm.mcp.connector import MCPConnector
                
                self.mcp_client = MCPClient()
                
                # Register all configured MCP connectors
                for connector_config in self.config.tools.mcp_connectors:
                    if not connector_config.enabled:
                        continue
                        
                    connector = MCPConnector(
                        name=connector_config.name,
                        url=connector_config.url,
                        authentication=connector_config.authentication,
                        description=connector_config.description,
                        tool_prefix=connector_config.tool_prefix
                    )
                    
                    self.mcp_client.register_connector(connector)
                    self.logger.info(f"Registered MCP connector: {connector_config.name}")
            except Exception as e:
                self.logger.error(f"Error initializing MCP client: {str(e)}")
                self.mcp_client = None

        # Do not unconditionally initialize embeddings client; lazily create when needed
        self.embedding_client = None

        # Initialize memory and vector store if enabled
        self.memory = None
        self.vector_store = None
        # Initialize a security manager for tools
        try:
            self.security_manager = SecurityManager(
                allowed_paths=getattr(self.config.tools, 'allowed_paths', []) or [],
                safe_mode=bool(getattr(self.config.tools, 'safe_mode', False)),
                allow_network=bool(getattr(self.config.tools, 'allow_network', True)),
                allow_filesystem=bool(getattr(self.config.tools, 'allow_filesystem', True)),
                blocked_domains=getattr(self.config.tools, 'blocked_domains', []) or [],
            )
        except Exception:
            # Fallback to defaults if config is mocked/missing
            self.security_manager = SecurityManager()
        
        # Register tools after security manager is initialized
        self._register_configured_tools()
        
        if self.memory_enabled:
            self._initialize_memory()
            
        # No longer using real-time keywords - models will determine when to use tools via their native SDKs

        # Thinking mode manager
        try:
            tcfg = getattr(self.config, "thinking", None)
            self.thinking_manager = ThinkingManager(
                enabled=bool(getattr(tcfg, "enabled", False)),
                show_tool_reasoning=bool(getattr(tcfg, "show_tool_reasoning", True)),
                thinking_model=getattr(tcfg, "model", None),
                max_thinking_tokens=int(getattr(tcfg, "max_tokens", 2000) or 2000),
                stream_thinking=bool(getattr(tcfg, "stream_thinking", True)),
                save_thinking=bool(getattr(tcfg, "save_thinking", False)),
                temperature=float(getattr(tcfg, "thinking_temperature", 0.7) or 0.7),
            )
        except Exception:
            self.thinking_manager = ThinkingManager(enabled=False)

    def _initialize_provider(self) -> BaseProvider:
        """
        Initialize the provider based on configuration.
        
        Returns:
            BaseProvider: The initialized provider client
        """
        # Determine provider name safely: subclass override or config-derived
        provider_name = (
            getattr(self, 'provider', None) or getattr(self, 'provider_name', None) or self.config.provider.name
        ).lower()
        # Use the unified ProviderFactory for initialization
        try:
            return ProviderFactory.create_provider(
                provider_name,
                api_key=self.api_key,
                base_url=self.base_url,
                **self.extra_settings,
            )
        except ValueError:
            # Fallback: default to OpenAI for unknown provider names
            return ProviderFactory.create_provider(
                "openai",
                api_key=self.api_key,
                base_url=self.base_url,
                **self.extra_settings,
            )

    def _initialize_memory(self):
        """Initialize memory and vector store"""
        self.memory = ConversationStore(db_path=self.conversation_db_path)
        # Infer embedding dimension based on configured embedding model
        vector_dim = self._infer_embedding_dimension(self.embedding_model)
        self.vector_store = VectorStore(
            store_path=os.path.dirname(self.vector_index_path),
            vector_dim=vector_dim,
        )
        if not self.memory.db_exists():
            self.memory.create_db()

    def _infer_embedding_dimension(self, model_name: str | None) -> int:
        """Best-effort mapping from embedding model name to vector dimension.

        Defaults to 1536 when unknown.
        """
        m = (model_name or "").lower()
        if "text-embedding-3-large" in m:
            return 3072
        if "text-embedding-3-small" in m:
            return 1536
        if "ada-002" in m:
            return 1536
        # Add other known providers/models as needed
        return 1536

    def _register_configured_tools(self) -> None:
        """Register configured built-in tools with the ToolManager."""
        try:
            # Tool packs map
            tool_packs = {
                "common": [
                    "calculator",
                    "datetime",
                    "uuid",
                    "random_string",
                    "json_tool",
                    "regex_extract",
                    "lorem_ipsum",
                ],
                "web": [
                "web_search",
                "web_scraper",
                "url_metadata",
                "web_headlines",
                "weather",
            ],
                "files": [
                    "file_read",
                    "file_search",
                    "csv_preview",
                ],
                "crypto": [
                    "hash_text",
                    "base64_encode",
                    "base64_decode",
                ],
                "numbers": [
                    "random_number",
                    "unit_converter",
                ],
                "time": [
                    "datetime",
                    "timezone",
                ],
                "dev": [
                    "json_tool",
                    "regex_extract",
                    "hash_text",
                    "base64_encode",
                    "base64_decode",
                    "file_read",
                    "file_search",
                    "csv_preview",
                    "uuid",
                    "random_string",
                ],
                "all": [],  # special handled below
            }

            tool_classes = {
                "web_search": WebSearch,
                "calculator": Calculator,
                "timezone": TimezoneTool,
                "unit_converter": UnitConverter,
                "weather": WeatherTool,
                "web_scraper": WebScraper,
                "datetime": DateTimeTool,
                "uuid": UUIDTool,
                "random_string": RandomStringTool,
                "random_number": RandomNumberTool,
                "base64_encode": Base64EncodeTool,
                "base64_decode": Base64DecodeTool,
                "hash_text": HashTool,
                "json_tool": JSONTool,
                "regex_extract": RegexExtractTool,
                "lorem_ipsum": LoremIpsumTool,
                "file_read": FileReadTool,
                "file_search": FileSearchTool,
                "csv_preview": CSVPreviewTool,
                "url_metadata": URLMetadataTool,
                "web_headlines": WebHeadlinesTool,
            }
            # Expand packs
            selected: List[str] = []
            packs = getattr(self.config.tools, "packs", []) or []
            for p in packs:
                if p == "all":
                    selected.extend(tool_classes.keys())
                else:
                    selected.extend(tool_packs.get(p, []))
            # Add explicit tool_list
            if self.tools:
                selected.extend(self.tools)
            # Deduplicate while preserving order
            seen = set()
            expanded = [t for t in selected if not (t in seen or seen.add(t))]
            
            self.logger.debug(f"Selected tools to register: {expanded}")

            for name in expanded:
                cls = tool_classes.get(name)
                if not cls:
                    self.logger.warning(f"Tool '{name}' not found in available tools")
                    continue
                try:
                    if name == "weather":
                        tool = cls(api_key=os.environ.get("WEATHER_API_KEY"), security_manager=self.security_manager)
                    elif name in {"web_search", "web_scraper"}:
                        tool = cls(security_manager=self.security_manager)
                    elif name in {"url_metadata", "web_headlines"}:
                        tool = cls()  # These inherit NetworkTool security through their base class
                    elif name in {"file_read", "file_search", "csv_preview"}:
                        tool = cls(security_manager=self.security_manager)
                    else:
                        tool = cls()
                    self.tool_manager.register(tool)
                    self.logger.debug(f"Successfully registered tool: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to register tool '{name}': {e}")

            # Always register a help/introspection tool last
            self.tool_manager.register(ToolsHelpTool(self.tool_manager))
        except Exception as e:
            self.logger.error(f"Tool registration error: {e}")
            import traceback
            traceback.print_exc()

    def create_embedding(self, text: str) -> Union[List[float], bytes]:
        """
        Create an embedding for the given text.

        Args:
            text (str): The text to create an embedding for.

        Returns:
            Union[List[float], bytes]: The embedding as a list of floats or bytes.

        Raises:
            Exception: If there is an error creating the embedding.
        """
        try:
            # For providers that implement embeddings, try provider method first
            if hasattr(self.provider_client, 'create_embedding') and (
                not self.embeddings_provider_name or self.embeddings_provider_name == self.provider
            ):
                try:
                    return self.provider_client.create_embedding(text, self.embedding_model)
                except Exception as e:
                    # Fallback if provider embeddings are not available/supported
                    self.logger.debug(f"Provider embeddings failed; falling back to OpenAI: {e}")
            
            # Fallback to OpenAI embeddings for other providers
            if self.embedding_client is None:
                self.embedding_client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY") or self.api_key,
                )
            response = self.embedding_client.embeddings.create(input=text, model=self.embedding_model)
            return response.data[0].embedding

        except Exception as e:
            self.logger.error(f"Error creating embedding: {str(e)}")
            raise Exception(f"Error creating embedding: {str(e)}")

    def find_similar_messages(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Find messages similar to the query.

        Args:
            query (str): The text to find similar messages for.
            k (int): Number of similar messages to return.

        Returns:
            List[Dict[str, Any]]: List of similar messages with their metadata and similarity scores.
        """
        query_embedding = self.create_embedding(query)
        similar_messages = self.vector_store.search(query_embedding, self.vector_store_top_k)
        return similar_messages

    def load_config(self) -> YamlLMConfig:
        """
        Load configuration from YAML file.

        Returns:
            YamlLMConfig: Parsed configuration.

        Raises:
            FileNotFoundError: If config file is not found.
            ValueError: If config file is empty or could not be parsed.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at path: {self.config_path}")
        cfg = parse_yaml_config(self.config_path)
        # Run additional validation beyond pydantic schema
        try:
            cfg_dict = cfg.model_dump()
        except Exception:
            # Fallback for older pydantic versions
            cfg_dict = cfg.dict()  # type: ignore[attr-defined]
        errors = ConfigValidator.validate_config(cfg_dict)
        if errors:
            # Single-line actionable error (multiple joined with '; ')
            message = "; ".join(errors)
            raise ConfigurationError(f"Invalid configuration: {message}")
        return cfg

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a query to the language model.

        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (Optional[str]): An optional system prompt to provide context.

        Returns:
            str: The response from the language model.

        Raises:
            ValueError: If API key is not initialized or invalid.
            Exception: If there is an error during the query.
        """
        if not self.api_key:
            raise ValueError("API key is not initialized or invalid.")
        try:
            return self.get_response(prompt, system_prompt)
        except Exception as e:
            self.logger.error(f"API error: {str(e)}")
            raise Exception(f"API error: {str(e)}")

    def set_stream_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked with each streamed text delta."""
        self.stream_callback = callback

    def clear_stream_callback(self) -> None:
        """Clear any registered stream callback."""
        self.stream_callback = None

    def set_event_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback invoked for tool/model events (e.g., tool calls, results, usage)."""
        self.event_callback = callback

    def clear_event_callback(self) -> None:
        """Clear any registered event callback."""
        self.event_callback = None

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """
        Generates a response from the language model based on the provided prompt and optional system prompt.
        Supports tool calling if tools_enabled is True.

        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (Optional[str]): An optional system prompt for context.

        Returns:
            Optional[str]: The response from the language model, or None if streaming is enabled.
        """
        messages = self._prepare_messages(prompt, system_prompt)

        # Thinking mode preface (non-blocking on errors)
        if getattr(self, "thinking_manager", None) and self.thinking_manager.enabled:
            try:
                available_tools = list(self.tool_manager._tools.keys()) if self.tool_manager else []
                blocks = self.thinking_manager.generate_thinking(
                    prompt, available_tools, self.provider_client, self.model
                )
                formatted = self.thinking_manager.format_thinking_for_display(blocks)
                if formatted and self.thinking_manager.stream_thinking and self.stream_callback:
                    try:
                        self.stream_callback(formatted)
                    except Exception:
                        pass
                if formatted and self.thinking_manager.save_thinking and self.memory_enabled:
                    try:
                        self.memory.add_message(session_id=self.session_id, role="system", content=formatted)
                    except Exception:
                        pass
                # Emit events for each block if event_callback is set
                if self.event_callback:
                    for b in blocks:
                        try:
                            self.event_callback({
                                "type": "thinking",
                                "step": b.step.value,
                                "content": b.content,
                                "tools": b.tools_considered or [],
                                "confidence": getattr(b, 'confidence', 0.0),
                            })
                        except Exception:
                            pass
            except Exception as e:
                self.logger.debug(f"Thinking mode error: {e}")

        # Prepare tools if enabled (consider packs as well)
        tools_param = self._prepare_tools() if self.tools_enabled else None
        if tools_param:
            tools_param = self._filter_tools_for_prompt(tools_param, messages)
        if tools_param == []:
            tools_param = None

        # Generate response based on tools/streaming preference
        used_tools_flow = False
        if tools_param:
            # When tools are available, prefer non-streaming to ensure proper tool execution
            response_text = self._handle_non_streaming_response(messages, tools_param)
            used_tools_flow = True
        else:
            # No tools; follow streaming preference
            if self.output_stream:
                response_text = self._handle_streaming_with_tool_detection(messages, tools_param)
            else:
                response_text = self._handle_non_streaming_response(messages, tools_param)

        # Store conversation in memory
        if response_text:
            self._store_memory(prompt, response_text, self.session_id)

        # If we executed a non-streaming tools flow, return the text even when output_stream is True
        if used_tools_flow:
            return response_text
        return None if self.output_stream else response_text

    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Prepare messages for the API request.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str]): An optional system prompt.

        Returns:
            List[Dict[str, str]]: Formatted messages for the API request.
        """
        messages = []

        # Add system prompt if provided
        if system_prompt or self.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.system_prompt
            })

        # Add memory context if enabled
        if self.memory_enabled:           
            # Get conversation history
            history = self.memory.get_messages(
                session_id=self.session_id, 
                limit=self.memory_max_messages
            )

            # Add history messages
            messages.extend(history)

            # Optionally include similar past content as a separate system note
            try:
                similar_results = self.find_similar_messages(prompt, k=2)
                # Only include highly similar content and for non-trivial prompts
                filtered = [m for m in (similar_results or []) if float(m.get('similarity', 0)) >= 0.85]
                if filtered and len(str(prompt)) >= 20:
                    context_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in filtered])
                    messages.append({
                        "role": "system",
                        "content": "Relevant prior context (for reference only):\n" + context_text
                    })
            except Exception as e:
                self.logger.debug(f"Error finding similar messages: {str(e)}")

            # Add current prompt
            messages.append({
                "role": "user", 
                "content": str(prompt)
            })
        else:
            # Just add the current prompt if memory is disabled
            messages.append({
                "role": "user",
                "content": str(prompt)
            })

        return messages

    def _filter_tools_for_prompt(self, tools_param: Optional[List[Dict[str, Any]]], messages: List[Dict[str, Any]]):
        """
        Gate certain tools (currently 'web_search') based on the latest user prompt.
        Avoid unnecessary search for plain math or unit conversions.
        """
        try:
            if not tools_param:
                return tools_param
            # Respect config toggle
            gate = True
            try:
                gate = bool(getattr(self.config.tools, 'gate_web_search', True))
            except Exception:
                gate = True
            if not gate:
                return tools_param
            # Latest user message
            prompt_text = ""
            for m in reversed(messages or []):
                if (m or {}).get("role") == "user":
                    prompt_text = str((m or {}).get("content", ""))
                    break
            if not prompt_text:
                return tools_param
            p = prompt_text.strip()
            pl = p.lower()
            # If indicates current events, keep search
            include_kw = [
                "latest", "today", "current", "news", "breaking", "recent",
                "weather", "stock", "price", "live", "now", "update", "happening"
            ]
            if any(k in pl for k in include_kw):
                return tools_param
            # Math / unit conversion heuristics
            is_math_like = bool(re.search(r"\d", p) and re.search(r"[\+\-\*/xX]", p))
            numbers_only = bool(re.fullmatch(r"[0-9\s\.+\-\*/()%xX]+", p))
            unit_convert = ("convert" in pl and " to " in pl and re.search(r"\d", pl) is not None)
            if is_math_like or numbers_only or unit_convert:
                filtered: List[Dict[str, Any]] = []
                for t in tools_param:
                    fn = (t or {}).get("function", {})
                    if fn.get("name") != "web_search":
                        filtered.append(t)
                return filtered
        except Exception:
            return tools_param
        return tools_param

    def _prepare_tools(self):
        """
        Prepare tool definitions for the API call.

        Returns:
            list: List of tool definitions in the format expected by the provider
        """        
        # Gate on provider tool support to avoid unsupported flows
        try:
            capabilities = get_provider_capabilities(self.provider)
            if not capabilities.supports_tools:
                self.logger.warning(
                    f"Provider '{self.provider}' does not support tools; disabling tool usage"
                )
                return []
        except Exception:
            # On any error, fall back to allowing tools (legacy behavior)
            pass

        tool_definitions: List[Dict[str, Any]] = []

        # Add local tools via ToolManager (already registered)
        try:
            tool_definitions.extend(self.tool_manager.get_tool_definitions())
        except Exception as e:
            self.logger.debug(f"Unable to get local tool definitions: {e}")
            
        # Add MCP tools if any are configured
        if hasattr(self, 'mcp_client') and self.mcp_client:
            try:
                mcp_tools = self.mcp_client.convert_mcp_tools_to_definitions()
                # Convert ToolDefinition instances to provider schema dicts
                tool_definitions.extend([tool.to_dict() for tool in mcp_tools])
                self.logger.debug(f"Added {len(mcp_tools)} tools from MCP connectors")
            except Exception as e:
                self.logger.error(f"Error adding MCP tools: {str(e)}")

        return tool_definitions

    def get_response_with_thinking(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Return both the formatted thinking and the model response in a single call."""
        result: Dict[str, Any] = {"thinking": None, "response": None}
        # Generate thinking (will stream/save according to config)
        if getattr(self, "thinking_manager", None) and self.thinking_manager.enabled:
            try:
                available_tools = list(self.tool_manager._tools.keys()) if self.tool_manager else []
                blocks = self.thinking_manager.generate_thinking(
                    prompt, available_tools, self.provider_client, self.model
                )
                formatted = self.thinking_manager.format_thinking_for_display(blocks)
                result["thinking"] = formatted
            except Exception as e:
                self.logger.debug(f"Thinking mode error: {e}")
        # Then get the response via normal path
        result["response"] = self.get_response(prompt, system_prompt)
        return result

    def _get_tool_parameters(self, tool_name):
        """Use ToolManager signatures to get parameters schema."""
        try:
            sig = self.tool_manager.get_signature(tool_name)
            return sig.get("function", {}).get("parameters", {"type": "object", "properties": {}})
        except Exception:
            return {"type": "object", "properties": {}}

    def _handle_streaming_with_tool_detection(self, messages, tools_param=None):
        """
        Handle streaming with tool detection.

        First checks if tools are needed with a small preview request, then
        either processes tools or streams the response.

        Args:
            messages (list): List of message objects.
            tools_param (list, optional): Tool definitions.

        Returns:
            str: Response text.
        """
        try:
            # We'll now rely on the model's ability to decide when to use tools
            # through the provider's SDK, rather than keyword matching for real-time queries
            
            # Make a preview request to see if the model will use tools
            try:
                # Make a low-token request to see if the model will use tools
                params = {
                    "messages": messages,
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": 10,  # Just enough to detect tool usage
                    "top_p": self.top_p,
                    "stop_sequences": self.stop_sequences if self.stop_sequences else None
                }
                
                if tools_param:
                    params["tools"] = tools_param
                    
                # Use the provider client to make the request
                preview_response = self.provider_client.get_completion(**params)
                
                # Check if the model wants to use tools - this is provider-specific
                tool_calls = []
                if self.provider.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek") and tools_param:
                    if hasattr(preview_response.choices[0].message, "tool_calls"):
                        tool_calls = preview_response.choices[0].message.tool_calls
                elif self.provider.lower() == "anthropic" and tools_param:
                    tool_calls = preview_response.get("tool_calls", [])
                
                if tool_calls:
                    # Model wants to use tools, use non-streaming
                    return self._handle_non_streaming_response(messages, tools_param)
                else:
                    # Model doesn't need tools, use streaming
                    return self._handle_streaming_response(messages)
            except Exception as e:
                self.logger.warning(f"Preview request failed: {str(e)}")
                # Fall back to streaming without tool detection
                return self._handle_streaming_response(messages)
                
        except Exception as e:
            self.logger.warning(f"Tool detection request failed: {str(e)}")
            return self._handle_streaming_response(messages)

    def _handle_streaming_response(self, messages):
        """
        Handle streaming response from the model.

        Args:
            messages (list): List of message objects

        Returns:
            str: Concatenated response text
        """
        try:
            # Use the provider client to get a streaming response
            response = self.provider_client.get_streaming_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop_sequences=self.stop_sequences if self.stop_sequences else None
            )
            
            response_text = ""
            
            # Handle the streaming response based on the provider
            if self.provider.lower() in ("openai", "azure_openai", "mistral"):
                for chunk in response:
                    if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        response_text += delta
                        if self.stream_callback:
                            try:
                                self.stream_callback(delta)
                            except Exception:
                                pass
            elif self.provider.lower() == "anthropic":
                for line in response:
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8').strip())
                            if chunk.get("type") == "content_block_delta" and chunk.get("delta", {}).get("text"):
                                delta = chunk["delta"]["text"]
                                response_text += delta
                                if self.stream_callback:
                                    try:
                                        self.stream_callback(delta)
                                    except Exception:
                                        pass
                        except Exception as e:
                            self.logger.debug(f"Error parsing streaming chunk: {str(e)}")
            elif self.provider.lower() == "google":
                # Best-effort parsing of google-generativeai streaming events
                for ev in response:
                    delta = getattr(ev, "text", None)
                    if delta:
                        response_text += delta
                        if self.stream_callback:
                            try:
                                self.stream_callback(delta)
                            except Exception:
                                pass
            else:
                # Generic fallback for other providers
                for chunk in response:
                    chunk_text = self._extract_text_from_chunk(chunk)
                    if chunk_text:
                        response_text += chunk_text
                        if self.stream_callback:
                            try:
                                self.stream_callback(chunk_text)
                            except Exception:
                                pass
            
            return response_text
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            raise Exception(f"Error getting streaming response: {str(e)}")
            
    def _extract_text_from_chunk(self, chunk):
        """Extract text from a streaming chunk based on provider-specific format."""
        if self.provider.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek"):
            if hasattr(chunk.choices[0], "delta") and chunk.choices[0].delta.content:
                return chunk.choices[0].delta.content
        elif self.provider.lower() == "anthropic":
            try:
                if isinstance(chunk, bytes):
                    chunk = json.loads(chunk.decode('utf-8').strip())
                if chunk.get("type") == "content_block_delta" and chunk.get("delta", {}).get("text"):
                    return chunk["delta"]["text"]
            except Exception:
                pass
        
        # If we can't extract text in any known way, return empty string
        return ""

    def _handle_non_streaming_response(self, messages, tools_param=None):
        """
        Handle non-streaming response from the model, with optional tool calling.

        Args:
            messages (list): List of message objects.
            tools_param (list, optional): Tool definitions.

        Returns:
            str: Response text.
        """
        try:
            # We'll now rely on the model's ability to decide when to use tools
            # through the provider's SDK, rather than keyword matching for real-time queries
            
            # Use the provider client to get a completion
            response = self.provider_client.get_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop_sequences=self.stop_sequences if self.stop_sequences else None,
                tools=tools_param,
                tool_choice="auto"
            )
            # Track usage if present
            self._record_usage(response)
            if self.event_callback and self._last_usage:
                try:
                    self.event_callback({"type": "model_usage", "usage": self._last_usage})
                except Exception:
                    pass
            
            # Check if the model wants to use tools - this is provider-specific
            if self.provider.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek") and tools_param:
                if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    if self.event_callback:
                        try:
                            calls = self.provider_client.format_tool_calls(response.choices[0].message.tool_calls)
                            self.event_callback({"type": "tool_request", "tool_calls": calls})
                        except Exception:
                            pass
                    return self._process_tool_calls(messages, response.choices[0].message)
            elif self.provider.lower() == "anthropic" and tools_param:
                if "tool_calls" in response:
                    # Format the Anthropic tool calls to our standard format
                    tool_calls = self.provider_client.format_tool_calls(response.get("tool_calls", []))
                    if self.event_callback:
                        try:
                            self.event_callback({"type": "tool_request", "tool_calls": tool_calls})
                        except Exception:
                            pass
                    # Process the tool calls with our standard method
                    return self._process_tool_calls(messages, {"content": response.get("content", ""), "tool_calls": tool_calls})
            
            # Get the response text based on the provider
            if self.provider.lower() in ("openai", "azure_openai"):
                response_text = response.choices[0].message.content
            elif self.provider.lower() == "anthropic":
                response_text = response.get("content", [{"text": "No response generated."}])[0]["text"]
            else:
                response_text = str(response)
            
            return response_text
                
        except Exception as e:
            self.logger.error(f"Non-streaming error: {str(e)}")
            raise Exception(f"Error getting non-streaming response: {str(e)}")

    def _process_tool_calls(self, messages, model_message, max_iterations=5):
        """
        Process tool calls from the model and get the final response.
        Args:
            messages (list): List of message objects.
            model_message: The model's message containing tool calls.
            max_iterations (int): Maximum number of tool call iterations.

        Returns:
            str: Final response text.
        """
        if max_iterations <= 0:
            return "Maximum tool call iterations reached. Unable to complete the request."
            
        # Log tool call request
        self.logger.debug("Tool call requested by model")
        
        # Ensure content is never null
        message_content = model_message.content if hasattr(model_message, "content") else model_message.get("content", "")
        
        # Format the tool calls to a standardized format
        if hasattr(model_message, "tool_calls"):
            tool_calls = self.provider_client.format_tool_calls(model_message.tool_calls)
        else:
            tool_calls = model_message.get("tool_calls", [])
        
        # Add the assistant's message with tool calls to conversation
        messages.append({
            "role": "assistant",
            "content": message_content,
            "tool_calls": tool_calls
        })
        
        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            function_args_str = tool_call.get("function", {}).get("arguments")
            tool_call_id = tool_call.get("id")
            
            try:
                function_args = json.loads(function_args_str)
            except (json.JSONDecodeError, TypeError):
                function_args = {}
                
            self.logger.debug(f"Tool function: {function_name}; args: {function_args_str}")
            if self.event_callback:
                try:
                    self.event_callback({"type": "tool_call", "name": function_name, "args": function_args})
                except Exception:
                    pass
            
            # Execute the tool
            self._tool_call_count += 1
            tool_result = self._execute_tool(function_name, function_args)
            if self.event_callback:
                try:
                    preview = tool_result
                    try:
                        s = json.dumps(tool_result)
                        preview = s if len(s) <= 400 else s[:400] + "…"
                    except Exception:
                        pass
                    self.event_callback({"type": "tool_result", "name": function_name, "result": preview, "tool_call_id": tool_call_id})
                except Exception:
                    pass
            
            # Format the result for display (truncated if too long)
            result_str = str(tool_result)
            if len(result_str) > 200:
                display_result = result_str[:200] + "..."
            else:
                display_result = result_str
                
            self.logger.debug(f"Tool result: {display_result}")
            
            # Add the tool result to conversation with formatting instructions
            tool_content = json.dumps(tool_result) if isinstance(tool_result, (list, dict)) else str(tool_result)
            
            # Add to messages
            tool_result_msg = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_content,
                "name": function_name  # For Anthropic compatibility
            }
            
            messages.append(tool_result_msg)
            
            # For web search, add instructions for formatting
            if function_name == "web_search":
                messages.append({
                    "role": "user",
                    "content": "Please summarize these search results in a natural, conversational way. Highlight the most important points and present them as if you're having a conversation with me."
                })
        
        # Get the next response from the model
        try:
            # Use the provider client to get a completion
            # Apply gating for subsequent tool rounds
            next_tools = self._prepare_tools() if self.tools_enabled else None
            if next_tools:
                next_tools = self._filter_tools_for_prompt(next_tools, messages)
            next_response = self.provider_client.get_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop_sequences=self.stop_sequences if self.stop_sequences else None,
                tools=next_tools,
                tool_choice="auto"
            )
            # Track usage if present
            self._record_usage(next_response)
            if self.event_callback and self._last_usage:
                try:
                    self.event_callback({"type": "model_usage", "usage": self._last_usage})
                except Exception:
                    pass
            
            # Check if we need more tool calls - provider-specific handling
            if self.provider.lower() in ("openai", "azure_openai"):
                if hasattr(next_response.choices[0].message, "tool_calls") and next_response.choices[0].message.tool_calls:
                    # Recursive call for next round of tool calls
                    return self._process_tool_calls(messages, next_response.choices[0].message, max_iterations - 1)
            elif self.provider.lower() == "anthropic":
                if "tool_calls" in next_response:
                    # Format the Anthropic tool calls to our standard format
                    tool_calls = self.provider_client.format_tool_calls(next_response.get("tool_calls", []))
                    if tool_calls:
                        # Process the tool calls with our standard method
                        return self._process_tool_calls(messages, {"content": next_response.get("content", ""), "tool_calls": tool_calls}, max_iterations - 1)
            
            # We have our final text response - extract based on provider
            if self.provider.lower() in ("openai", "azure_openai"):
                response_text = next_response.choices[0].message.content
            elif self.provider.lower() == "anthropic":
                response_text = next_response.get("content", [{"text": "No response generated."}])[0]["text"]
            else:
                response_text = str(next_response)
            
            return response_text
                
        except Exception as e:
            self.logger.error(f"Error processing tool calls: {str(e)}")           
            # Attempt to create a direct response based on the tool results
            try:
                # Collect the tool results we already have
                tool_results_summary = ""
                calculator_answer = None
                for msg in messages:
                    if msg.get("role") == "tool" and msg.get("content"):
                        tool_name = msg.get("name", "unknown tool")
                        
                        # Format the result information
                        if tool_name == "web_search":
                            # Parse the search results
                            try:
                                search_data = json.loads(msg.get("content"))
                                tool_results_summary += "\nSearch results:\n"
                                
                                if isinstance(search_data, dict) and "results" in search_data:
                                    for i, result in enumerate(search_data.get("results", [])[:3]):
                                        tool_results_summary += f"- {result.get('title', 'No title')}: {result.get('snippet', 'No information')}\n"
                            except Exception:
                                tool_results_summary += f"\n{tool_name} results: {msg.get('content')[:200]}...\n"
                        elif tool_name == "calculator":
                            try:
                                calc_data = json.loads(msg.get("content"))
                                if isinstance(calc_data, dict) and "result" in calc_data:
                                    calculator_answer = calc_data.get("result")
                                tool_results_summary += f"\ncalculator results: {msg.get('content')[:200]}...\n"
                            except Exception:
                                tool_results_summary += f"\ncalculator results: {msg.get('content')[:200]}...\n"
                        else:
                            tool_results_summary += f"\n{tool_name} results: {msg.get('content')[:200]}...\n"
                
                # If we have a direct calculator answer, prefer a concise helpful response
                if calculator_answer is not None:
                    response_text = f"The result is {calculator_answer}."
                else:
                    # Create a helpful summary response even if search failed
                    response_text = (
                        f"I ran into a technical issue after using tools, but here is what I gathered:{tool_results_summary}\n\n"
                        f"If you'd like, I can try again or adjust the approach."
                    )
                
                # No direct printing in core; return response_text
                return response_text
                
            except Exception:
                # If even our fallback fails, return a generic message
                response_text = "I encountered a technical issue while processing the tool results. Please try your question again or phrase it differently."
                # No direct printing in core; return response_text
                return response_text

    def _execute_tool(self, tool_name, tool_args):
        """
        Execute a tool with the provided arguments.

        Args:
            tool_name (str): Name of the tool to execute
            tool_args (dict): Arguments for the tool

        Returns:
            Any: The result of the tool execution
        """
        # Check if this is an MCP tool
        if hasattr(self, 'mcp_client') and self.mcp_client:
            # Try to find the tool in the MCP tool definitions
            try:
                mcp_tools = self.mcp_client.convert_mcp_tools_to_definitions()
                for tool in mcp_tools:
                    if tool.name == tool_name:
                        # This is an MCP tool, execute it via the MCP client
                        return self.mcp_client.execute_tool(
                            connector_name=tool.mcp_connector_name,
                            tool_id=tool.mcp_tool_id,
                            parameters=tool_args
                        )
            except Exception as e:
                self.logger.error(f"Error checking MCP tools: {str(e)}")
                # Continue with local tool execution
        
        # Execute via ToolManager
        try:
            return self.tool_manager.execute(tool_name, tool_args)
        except Exception as e:
            self.logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return {"error": str(e)}

    def _store_memory(self, prompt: str, response_text: str, session_id: str, tool_interactions=None) -> None:
        """
        Store the conversation in memory.

        Args:
            prompt (str): User's prompt
            response_text (str): Model's response
            session_id (str): Session ID for the conversation
            tool_interactions (list, optional): List of tool interactions
        """
        if not self.memory_enabled:
            return

        try:
            # Store user message
            message_id = self.memory.add_message(
                session_id=session_id, 
                role="user", 
                content=prompt
            )
            prompt_embedding = self.create_embedding(prompt)
            self.vector_store.add_vector(
                vector=prompt_embedding,
                message_id=message_id,
                content=prompt,
                role="user"
            )

            # Store assistant response 
            response_id = self.memory.add_message(
                session_id=session_id, 
                role="assistant", 
                content=response_text
            )
            response_embedding = self.create_embedding(response_text)
            self.vector_store.add_vector(
                vector=response_embedding,
                message_id=response_id,
                content=response_text,
                role="assistant"
            )

            # Store tool interactions if provided
            if tool_interactions:
                for interaction in tool_interactions:
                    self.memory.add_message(
                        session_id=session_id,
                        role="tool",
                        content=json.dumps(interaction)
                    )

        except Exception as e:
            self.logger.error(f"Error storing memory: {str(e)}")
    
    # Define a type variable for the return type
    T = TypeVar('T')
    
    def _make_api_call(self, api_func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Make an API call with retry and exponential backoff for transient failures.
        
        Args:
            api_func: The API function to call
            *args: Positional arguments to pass to the API function
            **kwargs: Keyword arguments to pass to the API function
            
        Returns:
            The result of the API function call
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempts = 0
        max_attempts = self.retry_max_attempts
        initial_delay = self.retry_initial_delay
        backoff_factor = self.retry_backoff_factor
        last_exception = None
        
        while attempts < max_attempts:
            try:
                return api_func(*args, **kwargs)
            except (ConnectionError, TimeoutError, OpenAIError) as e:
                attempts += 1
                last_exception = e
                
                # If this was the last attempt, re-raise the exception
                if attempts >= max_attempts:
                    self.logger.error(f"API call failed after {max_attempts} attempts: {str(e)}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = initial_delay * (backoff_factor ** (attempts - 1))
                self.logger.warning(f"API call attempt {attempts} failed: {str(e)}. Retrying in {delay} seconds...")
                
                # Sleep before the next attempt
                time.sleep(delay)
        
        # This should never happen, but just in case
        if last_exception:
            raise last_exception
        
        # This line should never be reached but is needed for type checking
        return cast(T, None)

    def _record_usage(self, response: Any) -> None:
        """Record token usage stats from a provider response when available."""
        try:
            def _coerce(v):
                try:
                    return int(v or 0)
                except Exception:
                    return 0

            def _to_usage(obj: Any) -> Dict[str, int]:
                # Accept dict-like or object with attributes
                def get_val(names):
                    for n in names:
                        if isinstance(obj, dict) and n in obj:
                            return obj.get(n)
                        if hasattr(obj, n):
                            return getattr(obj, n)
                    return 0

                prompt = _coerce(get_val(["prompt_tokens", "input_tokens", "prompt_token_count"]))
                completion = _coerce(get_val(["completion_tokens", "output_tokens", "completion_token_count", "candidates_token_count"]))
                total = _coerce(get_val(["total_tokens", "total_token_count"]))
                if not total:
                    total = prompt + completion
                return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

            usage: Optional[Dict[str, int]] = None
            if hasattr(response, "usage"):
                usage = _to_usage(getattr(response, "usage"))
            elif isinstance(response, dict) and "usage" in response:
                usage = _to_usage(response.get("usage", {}))
            # Google Generative AI style: usage_metadata
            elif hasattr(response, "usage_metadata"):
                usage = _to_usage(getattr(response, "usage_metadata"))
            # Some SDKs may expose fields at top-level
            else:
                # Best-effort direct mapping
                usage = _to_usage(response)

            if usage:
                self._last_usage = usage
                for k, v in usage.items():
                    self._total_usage[k] = self._total_usage.get(k, 0) + v
        except Exception:
            pass

    def update_settings(self, **kwargs: Dict[str, Any]) -> None:
        """
        Update the settings of the instance with the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments where the key is the attribute name
                     and the value is the new value for that attribute.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def print_settings(self) -> None:
        """
        Print the current settings of the LLM in an organized format.
        """
        settings = {
            "Provider Settings": {
                "Provider": self.provider,
                "Model": self.model,
                "Base URL": self.base_url
            },
            "Model Settings": {
                "Temperature": self.temperature,
                "Max Tokens": self.max_tokens,
                "Top P": self.top_p,
                "Stop Sequences": self.stop_sequences
            },
            "Request Settings": {
                "Timeout": self.request_timeout,
                "Max Retry Attempts": self.retry_max_attempts,
                "Initial Retry Delay": self.retry_initial_delay,
                "Retry Backoff Factor": self.retry_backoff_factor
            },
            "Context Settings": {
                "System Prompt": self.system_prompt,
                "Max Context Length": self.max_context_length
            },
            "Memory Settings": {
                "Enabled": self.memory_enabled,
                "Max Messages": self.memory_max_messages,
                "Session ID": self.session_id,
                "Conversation DB Path": self.conversation_db_path,
                "Vector Index Path": self.vector_index_path,
                "Vector Metadata Path": self.vector_metadata_path,
                "Vector Store Top K": self.vector_store_top_k,
            },
            "Output Settings": {
                "Format": self.output_format,
                "Stream": self.output_stream
            },
            "Tool Settings": {
                "Enabled": self.tools_enabled,
                "Tools": self.tools,
                "Timeout": self.tools_timeout
            },
            "Safety Settings": {
                "Content Filtering": self.content_filtering,
                "Max Requests/Minute": self.max_requests_per_minute,
                "Sensitive Keywords": self.sensitive_keywords
            }
        }

        print("\nLLM Configuration Settings:")
        print("=" * 50)
        for category, values in settings.items():
            print(f"\n{category}:")
            print("-" * len(category))
            for key, value in values.items():
                print(f"{key:20}: {value}")

    def __repr__(self) -> str:
        """Return a detailed string representation of the LLM instance."""
        return f"{self.__class__.__name__}(provider='{self.provider}', model='{self.model}')"

    def __str__(self) -> str:
        """Return a human-readable string representation of the LLM instance."""
        return f"{self.__class__.__name__} using {self.provider} {self.model}"

    def cleanup(self):
        """
        Explicitly clean up resources.
        
        This method is called automatically when using the context manager,
        but can also be called manually when needed.
        """
        self.__exit__(None, None, None)
    
    def __enter__(self):
        """Support context manager interface."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up resources when exiting context manager."""
        # Clean up provider client
        try:
            if hasattr(self, 'provider_client') and self.provider_client:
                self.provider_client.close()
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error closing provider client: {e}")
        
        # Clean up embedding client
        try:
            if hasattr(self, 'embedding_client') and self.embedding_client:
                self.embedding_client.close()
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error closing embedding client: {e}")
        
        # Clean up MCP client
        try:
            if hasattr(self, 'mcp_client') and self.mcp_client:
                # MCP client might have its own cleanup
                if hasattr(self.mcp_client, 'close'):
                    self.mcp_client.close()
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error closing MCP client: {e}")
        
        # Clean up memory resources
        try:
            if hasattr(self, 'memory') and self.memory:
                # ConversationStore might need to close DB connections
                if hasattr(self.memory, 'close'):
                    self.memory.close()
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error closing memory store: {e}")
        
        # Clean up vector store
        try:
            if hasattr(self, 'vector_store') and self.vector_store:
                # VectorStore might need to save index
                if hasattr(self.vector_store, 'close'):
                    self.vector_store.close()
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error closing vector store: {e}")
        
        # Clean up tool manager resources
        try:
            if hasattr(self, 'tool_manager') and self.tool_manager:
                # Tool manager doesn't currently have cleanup, but might in future
                if hasattr(self.tool_manager, 'close'):
                    self.tool_manager.close()
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error closing tool manager: {e}")
        
        # Log successful cleanup
        if self.logger:
            self.logger.debug("LLM context manager cleanup completed")

    def __bool__(self) -> bool:
        """Return True if the LLM instance is properly initialized with an API key."""
        return bool(self.api_key)


