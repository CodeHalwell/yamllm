"""
Main LLM interface for YAMLLM with modular architecture.

This is the refactored version with improved separation of concerns,
thread safety, and async support.
"""

from typing import Optional, Dict, Any, List, Callable
import os
import json
import logging
import asyncio
import time
import re
import dotenv

from yamllm.core.parser import parse_yaml_config, YamlLMConfig
from yamllm.core.config_validator import ConfigValidator
from yamllm.core.exceptions import (
    ConfigurationError, ProviderError
)
from yamllm.core.error_handler import ErrorHandler
from yamllm.core.memory_manager import MemoryManager
from yamllm.core.tool_orchestrator import ToolOrchestrator
from yamllm.core.thinking import ThinkingManager
from yamllm.providers.factory import ProviderFactory
from yamllm.providers.capabilities import get_provider_capabilities
from yamllm.tools.thread_safe_manager import ThreadSafeToolManager
from yamllm.core.utils import mask_string
from openai import OpenAI


# Load environment variables
dotenv.load_dotenv()


def setup_logging(config):
    """Set up logging configuration."""
    # Set logging level for external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Disable propagation to root logger
    logging.getLogger('yamllm').propagate = False
    
    # Get or create yamllm logger
    logger = logging.getLogger('yamllm')
    logger.setLevel(getattr(logging, config.logging.level))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Ensure log directory exists
    log_path = config.logging.file
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    
    # Create handler
    if getattr(config.logging, "rotate", False):
        from logging.handlers import RotatingFileHandler
        max_bytes = getattr(config.logging, "rotate_max_bytes", 1048576)
        backup_count = getattr(config.logging, "rotate_backup_count", 3)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count
        )
    else:
        file_handler = logging.FileHandler(log_path)
    
    # Set formatter
    # Support both new `json_format` and legacy `json` toggle names
    use_json = bool(getattr(config.logging, "json_format", False) or getattr(config.logging, "json", False))
    if use_json:
        import json as _json
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                return _json.dumps({
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                    "exc_info": self.formatException(record.exc_info) if record.exc_info else None
                })
        
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(config.logging.format)
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Optional console handler
    if getattr(config.logging, "console", False):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


class LLM:
    """
    Main LLM interface with modular architecture.
    
    This class provides a clean interface for LLM interactions while
    delegating specific responsibilities to specialized components.
    """
    
    def __init__(self, config_path: str, api_key: str):
        """
        Initialize LLM with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            api_key: API key for the LLM provider
        """
        self.config_path = config_path
        self.api_key = api_key
        # Compatibility: allow tests to patch load_config()
        self.config = self.load_config()
        self.logger = setup_logging(self.config)
        
        # Error handler
        self.error_handler = ErrorHandler(self.logger)
        # Simple per-process embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Extract configuration values
        self._extract_config_values()
        
        # Ensure MCP attribute exists before tool initialization
        self.mcp_client = None
        
        # Initialize components
        self.provider_client = self._initialize_provider()
        self.memory_manager = self._initialize_memory()
        self.tool_orchestrator = self._initialize_tools()
        self.thinking_manager = self._initialize_thinking()
        
        # Async components (lazy initialization)
        self._async_provider = None
        self._async_tool_manager = None
        
        # Embedding client (lazy initialization)
        self.embedding_client = None
        self._embeddings_provider_name = None
        self._embedding_model = "text-embedding-3-small"
        self._setup_embeddings_config()
        
        # Callbacks
        self.stream_callback: Optional[Callable[[str], None]] = None
        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Usage tracking
        self._last_usage: Optional[Dict[str, int]] = None
        self._total_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self._tool_call_count = 0
        
        # MCP client initialization
        self.mcp_client = self._initialize_mcp()
    
    def _load_and_validate_config(self) -> YamlLMConfig:
        """Load and validate configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        config = parse_yaml_config(self.config_path)
        
        # Validate configuration
        try:
            config_dict = config.model_dump()
        except Exception:
            config_dict = config.dict()  # Fallback for older pydantic
        
        errors = ConfigValidator.validate_config(config_dict)
        if errors:
            raise ConfigurationError(
                f"Invalid configuration: {'; '.join(errors)}",
                config_path=self.config_path,
                validation_errors=errors
            )
        
        return config

    # Backward-compatibility for tests that patch LLM.load_config
    def load_config(self) -> YamlLMConfig:  # pragma: no cover - thin wrapper
        return self._load_and_validate_config()
    
    def _extract_config_values(self):
        """Extract configuration values for easier access."""
        # Provider settings
        self.provider_name = self.config.provider.name
        # Preserve subclass overrides (e.g., tests setting self.provider pre-init)
        if not hasattr(self, "provider") or self.provider is None:
            self.provider = self.provider_name  # Compatibility
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
        
        # Memory settings
        self.memory_enabled = self.config.context.memory.enabled
        self.memory_max_messages = self.config.context.memory.max_messages
        self.session_id = getattr(self.config.context.memory, 'session_id', None)
        
        # Output settings
        self.output_format = self.config.output.format
        self.output_stream = self.config.output.stream
        
        # Tools settings
        self.tools_enabled = self.config.tools.enabled
        self.tools = self.config.tools.tools
        self.tools_timeout = self.config.tools.tool_timeout
        
        # Safety settings
        self.content_filtering = self.config.safety.content_filtering
        self.max_requests_per_minute = self.config.safety.max_requests_per_minute
        self.sensitive_keywords = self.config.safety.sensitive_keywords
    
    def _setup_embeddings_config(self):
        """Setup embeddings configuration."""
        try:
            if hasattr(self.config, "embeddings") and self.config.embeddings:
                self._embeddings_provider_name = getattr(self.config.embeddings, "provider", None)
                if getattr(self.config.embeddings, "model", None):
                    self._embedding_model = self.config.embeddings.model
        except Exception:
            pass
    
    def _initialize_provider(self):
        """Initialize the provider client (construction only; network occurs later)."""
        provider_name = (self.provider_name or "").lower()
        try:
            return ProviderFactory.create_provider(
                provider_name,
                api_key=self.api_key,
                base_url=self.base_url,
                **self.extra_settings,
            )
        except Exception as e:
            masked_error = mask_string(str(e))
            self.logger.warning(
                f"Failed to initialize provider '{provider_name}', falling back to OpenAI: {masked_error}"
            )
            return ProviderFactory.create_provider(
                "openai",
                api_key=self.api_key,
                base_url=self.base_url,
                **self.extra_settings,
            )
    
    def _initialize_memory(self) -> MemoryManager:
        """Initialize memory management."""
        memory_config = self.config.context.memory
        
        return MemoryManager(
            enabled=memory_config.enabled,
            max_messages=memory_config.max_messages,
            session_id=getattr(memory_config, 'session_id', None),
            conversation_db_path=getattr(memory_config, 'conversation_db', None),
            vector_index_path=getattr(memory_config.vector_store, 'index_path', None),
            vector_store_top_k=getattr(memory_config.vector_store, 'top_k', 5),
            vector_dim=self._infer_embedding_dimension(),
            logger=self.logger
        )
    
    def _initialize_tools(self) -> ToolOrchestrator:
        """Initialize tool orchestration."""
        tools_config = self.config.tools
        
        # Use thread-safe tool manager
        tool_manager = ThreadSafeToolManager(
            timeout=tools_config.tool_timeout,
            max_concurrent=5,
            logger=self.logger
        )
        
        # Security configuration
        security_config = {
            'allowed_paths': getattr(tools_config, 'allowed_paths', []),
            'safe_mode': getattr(tools_config, 'safe_mode', False),
            'allow_network': getattr(tools_config, 'allow_network', True),
            'allow_filesystem': getattr(tools_config, 'allow_filesystem', True),
            'blocked_domains': getattr(tools_config, 'blocked_domains', [])
        }
        
        orchestrator = ToolOrchestrator(
            enabled=tools_config.enabled,
            tool_list=tools_config.tools,
            tool_packs=getattr(tools_config, 'packs', []),
            tool_timeout=tools_config.tool_timeout,
            security_config=security_config,
            logger=self.logger,
            mcp_client=self.mcp_client,
            include_help_tool=getattr(tools_config, 'include_help_tool', True)
        )
        
        # Replace the tool manager with our thread-safe version
        orchestrator.tool_manager = tool_manager
        orchestrator._register_tools()
        
        return orchestrator

    def _prepare_tools(self, messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Prepare tool definitions based on provider capabilities and prompt content."""
        if not getattr(self, 'tool_orchestrator', None) or not self.tool_orchestrator.enabled:
            return []
        try:
            provider_for_caps = getattr(self, 'provider', None) or self.provider_name
            caps = get_provider_capabilities(provider_for_caps)
            if not caps.supports_tools:
                return []
        except Exception:
            return []

        try:
            tools = self.tool_orchestrator.get_tool_definitions()
            msgs = messages or []
            return self._filter_tools_for_prompt(tools, msgs)
        except Exception:
            try:
                return self.tool_orchestrator.get_tool_definitions()
            except Exception:
                return []
    
    def _initialize_mcp(self):
        """Initialize MCP client if configured."""
        if not self.config.tools.mcp_connectors:
            return None
        
        try:
            from yamllm.mcp.client import MCPClient
            from yamllm.mcp.connector import MCPConnector
            
            mcp_client = MCPClient()
            
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
                
                mcp_client.register_connector(connector)
                self.logger.info(f"Registered MCP connector: {connector_config.name}")
            
            return mcp_client
            
        except Exception as e:
            self.logger.error(f"Error initializing MCP client: {e}")
            return None
    
    def _initialize_thinking(self) -> ThinkingManager:
        """Initialize thinking mode manager."""
        thinking_config = getattr(self.config, 'thinking', None)
        if not thinking_config:
            return ThinkingManager(enabled=False)
        
        tm = ThinkingManager(
            enabled=getattr(thinking_config, 'enabled', False),
            show_tool_reasoning=getattr(thinking_config, 'show_tool_reasoning', True),
            thinking_model=getattr(thinking_config, 'model', None),
            max_thinking_tokens=getattr(thinking_config, 'max_tokens', 2000),
            stream_thinking=getattr(thinking_config, 'stream_thinking', True),
            save_thinking=getattr(thinking_config, 'save_thinking', False),
            temperature=getattr(thinking_config, 'thinking_temperature', 0.7)
        )
        # Apply new hygiene and mode controls
        try:
            tm.mode = str(getattr(thinking_config, 'mode', 'auto') or 'auto').lower()
            tm.redact_logs = bool(getattr(thinking_config, 'redact_logs', True))
        except Exception:
            pass
        return tm
    
    def _infer_embedding_dimension(self) -> int:
        """Infer embedding dimension from model name."""
        model_lower = getattr(self, "_embedding_model", "text-embedding-3-small").lower()
        
        # OpenAI models
        if "text-embedding-3-large" in model_lower:
            return 3072
        elif "text-embedding-3-small" in model_lower:
            return 1536
        elif "ada-002" in model_lower:
            return 1536
        
        # Default
        return 1536
    
    # Main query methods
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a query to the language model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            The model's response
        """
        if not self.api_key:
            raise ValueError("API key is not initialized or invalid.")
        try:
            return self.get_response(prompt, system_prompt)
        except Exception as e:
            # Normalize common SDK errors robustly (without strict import types)
            is_openai_error = False
            try:
                import openai as _openai_mod  # type: ignore
                _err_type = getattr(_openai_mod, "OpenAIError", None)
                if _err_type is not None:
                    try:
                        is_openai_error = isinstance(e, _err_type)
                    except Exception:
                        is_openai_error = False
            except Exception:
                is_openai_error = False
            # Fallback name check
            if not is_openai_error and e.__class__.__name__ == "OpenAIError":
                is_openai_error = True
            if is_openai_error:
                raise Exception(f"OpenAI API error: {e}")
            raise Exception(f"Unexpected error during query: {e}")

    def _make_api_call(self, api_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Make an API call with retry and exponential backoff for transient failures."""
        attempts = 0
        delay = self.retry_initial_delay or 0.1
        max_attempts = max(1, int(self.retry_max_attempts or 1))
        while True:
            try:
                return api_func(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:  # built-ins for tests
                attempts += 1
                if attempts >= max_attempts:
                    raise e
                time.sleep(delay)
                delay *= max(1.0, float(self.retry_backoff_factor or 1.0))
            except Exception as e:
                # Retry on OpenAIError where applicable (use name-based detection)
                is_openai_error = (e.__class__.__name__ == "OpenAIError")
                if is_openai_error:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    time.sleep(delay)
                    delay *= max(1.0, float(self.retry_backoff_factor or 1.0))
                else:
                    raise
    
    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Get response from the model with tool support."""
        # Reset tool execution stack for new request
        self.tool_orchestrator.reset_execution_stack()
        # Reset cancellation flag for a fresh request
        setattr(self, "_cancel_requested", False)
        
        # Prepare messages
        messages = self._prepare_messages(prompt, system_prompt)
        
        # Handle thinking mode
        if self.thinking_manager.enabled:
            self._process_thinking(prompt)
        
        # Get tool definitions
        tools_param = None
        if self.tool_orchestrator.enabled:
            try:
                caps = get_provider_capabilities(self.provider_name)
                if caps.supports_tools:
                    tools_param = self.tool_orchestrator.get_tool_definitions()
                    tools_param = self._filter_tools_for_prompt(tools_param, messages)
                else:
                    self.logger.warning(
                        f"Provider '{self.provider_name}' does not support tools; disabling tool usage"
                    )
            except Exception:
                tools_param = self.tool_orchestrator.get_tool_definitions()
                tools_param = self._filter_tools_for_prompt(tools_param, messages)
        
        # Generate response
        response_text = None
        if self.output_stream:
            if tools_param:
                response_text = self._handle_streaming_with_tools(messages, tools_param)
            else:
                response_text = self._handle_streaming_response(messages)
        else:
            response_text = self._handle_non_streaming_response(messages, tools_param)

        # Store in memory (even if streaming; we have collected text)
        if response_text and self.memory_manager.enabled:
            self._store_memory(prompt, response_text)

        # In streaming mode, return None to align with streaming semantics
        if self.output_stream:
            return None
        return response_text
    
    async def aquery(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Async query to the language model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            The model's response
        """
        if not self._async_provider:
            await self._initialize_async_provider()
        
        messages = self._prepare_messages(prompt, system_prompt)
        
        response = await self._async_provider.get_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences
        )
        
        # Extract text based on provider
        return self._extract_text_from_response(response)
    
    async def _initialize_async_provider(self):
        """Initialize async provider via ProviderFactory when available."""
        provider_name = (self.provider_name or "").lower()
        try:
            # Use factory to build async provider if supported
            if ProviderFactory.supports_async(provider_name):
                self._async_provider = ProviderFactory.create_async_provider(
                    provider_name,
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                # Enter async context if provided
                if hasattr(self._async_provider, "__aenter__"):
                    await self._async_provider.__aenter__()
            else:
                raise ProviderError(provider_name, "Async support not available for this provider")
        except Exception as e:
            masked_error = mask_string(str(e))
            self.logger.error(f"Error initializing async provider {provider_name}: {masked_error}")
            raise
    
    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for the API request."""
        messages = []
        
        # Add system prompt
        if system_prompt or self.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.system_prompt
            })
        
        # Add conversation history
        if self.memory_manager.enabled:
            history = self.memory_manager.get_conversation_history()
            messages.extend(history)
            
            # Add similar past content if available
            try:
                query_embedding = self.create_embedding(prompt)
                similar = self.memory_manager.search_similar(query_embedding, k=2)
                
                filtered = [m for m in similar if m.get('similarity', 0) >= 0.85]
                if filtered and len(prompt) >= 20:
                    context = "\n\n".join([f"{m['role']}: {m['content']}" for m in filtered])
                    messages.append({
                        "role": "system",
                        "content": f"Relevant prior context (for reference only):\n{context}"
                    })
            except Exception as e:
                self.logger.debug(f"Error finding similar messages: {e}")
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _filter_tools_for_prompt(self, tools_param: Optional[List[Dict[str, Any]]], messages: List[Dict[str, Any]]):
        """Filter tools based on prompt content."""
        if not tools_param:
            return tools_param
        
        # Check if gating is enabled
        gate = getattr(self.config.tools, 'gate_web_search', True)
        if not gate:
            return tools_param
        
        # Get latest user message
        prompt_text = ""
        for m in reversed(messages or []):
            if m.get("role") == "user":
                prompt_text = str(m.get("content", ""))
                break
        
        if not prompt_text:
            return tools_param
        
        # Enhanced intent extraction
        wants = self._extract_intent(prompt_text)
        explicit_tool = self._extract_explicit_tool(prompt_text)

        if not any(wants.values()) and not explicit_tool:
            # No strong intent detected; return no tools to minimize token usage
            return []

        # Map tool intents to names
        allow: set[str] = set()
        if wants.get("web"):
            allow.update({"web_search", "web_scraper", "url_metadata", "web_headlines", "weather"})
        if wants.get("calc"):
            allow.update({"calculator"})
        if wants.get("convert"):
            allow.update({"unit_converter"})
        if wants.get("time"):
            allow.update({"timezone", "datetime"})
        if wants.get("files"):
            allow.update({"file_read", "file_search", "csv_preview"})
        if wants.get("url"):
            allow.update({"url_metadata", "web_scraper"})
        if wants.get("csv"):
            allow.update({"csv_preview", "file_read"})
        if wants.get("json"):
            allow.update({"json_tool"})
        if wants.get("regex"):
            allow.update({"regex_extract"})
        if wants.get("hash"):
            allow.update({"hash_text"})
        if wants.get("base64"):
            allow.update({"base64_encode", "base64_decode"})
        if wants.get("uuid"):
            allow.update({"uuid"})
        if explicit_tool:
            allow.add(explicit_tool)

        # Filter by function name where available
        filtered: List[Dict[str, Any]] = []
        for t in tools_param:
            try:
                name = t.get("function", {}).get("name") if t.get("type") == "function" else None
                if not name or name in allow:
                    filtered.append(t)
            except Exception:
                filtered.append(t)

        if not filtered:
            return tools_param
        try:
            self.logger.info(
                "filtered_tools",
                extra={
                    "allow": sorted(allow),
                    "selected": [
                        t.get("function", {}).get("name")
                        for t in filtered
                        if isinstance(t, dict)
                    ],
                },
            )
        except Exception:
            pass
        return filtered

    def _extract_intent(self, prompt_text: str) -> Dict[str, bool]:
        """Extract lightweight intents from prompt to guide tool selection."""
        text = (prompt_text or "").lower()
        import re
        domain_hint = re.search(r"\b(?:[a-z0-9-]+\.)+(?:[a-z]{2,})(?:/[^\s]*)?\b", text)
        explicit_url = re.search(r"https?://[^\s]+", text)
        command_match = re.search(r"(?:use|call|run|invoke)\s+(?:the\s+)?([a-z0-9_\-]+)\s+tool", text)
        wants = {
            "web": any(k in text for k in ("search", "look up", "latest", "today", "current", "news", "headline", "website", "site", "scrape", "scraper", "crawl", "fetch"))
                    or bool(explicit_url) or bool(domain_hint),
            "calc": any(k in text for k in ("calculate", "calc", "sum", "difference", "multiply", "divide", "percent", "product"))
                    or bool(re.search(r"\b\d+\s*([\+\-\*/×÷])\s*\d+", text)),
            "convert": any(k in text for k in ("convert", "units", "unit", "km", "miles", "celsius", "fahrenheit")),
            "time": any(k in text for k in ("time in", "timezone", "utc", "pst", "est")),
            "files": any(k in text for k in ("read file", "open file", "path", "search file", "grep")),
            "url": bool(explicit_url) or bool(domain_hint),
            "csv": "csv" in text,
            "json": "json" in text and any(k in text for k in ("pretty", "minify", "validate")),
            "regex": "regex" in text or bool(re.search(r"\/[a-zA-Z0-9_\-\.\+\*\?\|\(\)\[\]\{\}]+\/", text)),
            "hash": "hash" in text or any(k in text for k in ("md5", "sha256", "sha1")),
            "base64": "base64" in text or any(k in text for k in ("encode", "decode")),
            "uuid": "uuid" in text,
            "direct_tool": bool(command_match),
        }
        # Expand web intent if explicit URL present
        if wants["url"]:
            wants["web"] = True
        return wants

    def _extract_explicit_tool(self, prompt_text: str) -> Optional[str]:
        text = (prompt_text or "").lower()
        matches = re.findall(r"(?:use|call|run|invoke)\s+(?:the\s+)?([a-z0-9_\-]+)\s+tool", text)
        if not matches:
            bare_match = re.search(r"\buse\s+(webscrape|webscraper|web_scraper|webscraping)\b", text)
            if bare_match:
                matches = [bare_match.group(1)]
        if not matches:
            return None
        alias_map = {
            "webscrape": "web_scraper",
            "webscraper": "web_scraper",
            "web_scraper": "web_scraper",
            "webscraping": "web_scraper",
            "scrape": "web_scraper",
            "scraper": "web_scraper",
            "websearch": "web_search",
            "search": "web_search",
            "calc": "calculator",
        }
        requested = matches[-1].replace('-', '_')
        return alias_map.get(requested, requested)
    
    def _process_thinking(self, prompt: str):
        """Process thinking mode if enabled."""
        try:
            # Skip thinking for trivial prompts to reduce tokens
            p = (prompt or "").strip().lower()
            if len(p) < 12 or p in {"hi", "hello", "hey", "thanks", "thank you"}:
                return

            # If the prompt clearly doesn't need tools, optionally emit a compact
            # local thinking note (no model call) and proceed directly.
            try:
                compact_ok = bool(getattr(getattr(self.config, 'thinking', None), 'compact_for_non_tool', True))
            except Exception:
                compact_ok = True
            if not self._intent_requires_tools(prompt) and not self._extract_explicit_tool(prompt):
                if compact_ok and self.thinking_manager.stream_thinking and self.stream_callback:
                    compact = "<thinking>\n=== ANALYSIS ===\nNo tools needed; responding directly.\n" \
                              + ("=" * 50) + "\n</thinking>\n"
                    self.stream_callback(compact)
                    if self.event_callback:
                        self.event_callback({
                            "type": "thinking",
                            "step": "analysis",
                            "content": "No tools needed; responding directly.",
                            "tools": []
                        })
                    if self.thinking_manager.save_thinking and self.memory_manager.enabled:
                        try:
                            self.memory_manager.add_message("system", compact)
                        except Exception:
                            pass
                return

            available_tools = self.tool_orchestrator.tool_manager.list()
            # Respect thinking mode off|on|auto
            if not self.thinking_manager.should_show(prompt, available_tools=available_tools):
                return
            # If streaming is enabled and provider supports streaming, stream thinking deltas per block
            can_stream = bool(self.output_stream and self.stream_callback)
            provider = (self.provider_name or "").lower()
            provider_stream_ok = provider in ("openai", "azure_openai", "openrouter", "deepseek", "mistral", "anthropic", "google")

            # Compact tool planning for clear, simple intents (avoid verbose plans)
            try:
                wants = self._extract_intent(prompt)
            except Exception:
                wants = {}
            if compact_ok and any(wants.values()):
                msg = None
                if wants.get('calc'):
                    msg = "Using calculator to compute the result."
                elif wants.get('convert'):
                    msg = "Using unit conversion to compute the result."
                elif wants.get('time'):
                    msg = "Using timezone/time tool to answer."
                elif wants.get('web'):
                    msg = "Using web search to fetch current information."
                elif wants.get('files'):
                    msg = "Using file tools as requested."
                if msg and self.thinking_manager.stream_thinking and self.stream_callback:
                    compact = f"<thinking>\n=== TOOL_PLAN ===\n{msg}\n" + ("=" * 50) + "\n</thinking>\n"
                    self.stream_callback(compact)
                    if self.event_callback:
                        self.event_callback({
                            "type": "thinking",
                            "step": "tool_planning",
                            "content": msg,
                            "tools": [k for k, v in wants.items() if v]
                        })
                    if self.thinking_manager.save_thinking and self.memory_manager.enabled:
                        try:
                            self.memory_manager.add_message("system", compact)
                        except Exception:
                            pass
                    return

            if self.thinking_manager.stream_thinking and can_stream and provider_stream_ok:
                # Build prompts using ThinkingManager helpers
                try:
                    analysis_prompt = self.thinking_manager._create_analysis_prompt(prompt, available_tools)
                    self._stream_thinking_prompt(analysis_prompt, step="analysis")

                    if available_tools and self.thinking_manager.show_tool_reasoning:
                        # For tool planning, reuse the (so far) streamed content isn't available; request again
                        tool_prompt = self.thinking_manager._create_tool_planning_prompt(prompt, available_tools, "")
                        self._stream_thinking_prompt(tool_prompt, step="tool_planning", tools=available_tools)

                    exec_prompt = self.thinking_manager._create_execution_prompt(prompt, [])
                    self._stream_thinking_prompt(exec_prompt, step="execution_plan")
                except Exception:
                    # Fallback to non-streaming generation if anything fails
                    blocks = self.thinking_manager.generate_thinking(
                        prompt, available_tools, self.provider_client, self.model
                    )
                    formatted = self.thinking_manager.format_thinking_for_display(blocks)
                    if formatted and self.stream_callback:
                        self.stream_callback(formatted)
                    if self.event_callback:
                        for block in blocks:
                            self.event_callback({
                                "type": "thinking",
                                "step": block.step.value,
                                "content": block.content,
                                "tools": block.tools_considered or []
                            })
            else:
                # Non-streaming thinking generation
                blocks = self.thinking_manager.generate_thinking(
                    prompt, available_tools, self.provider_client, self.model
                )
                formatted = self.thinking_manager.format_thinking_for_display(blocks)
                if formatted and self.stream_callback:
                    self.stream_callback(formatted)
                if self.event_callback:
                    for block in blocks:
                        self.event_callback({
                            "type": "thinking",
                            "step": block.step.value,
                            "content": block.content,
                            "tools": block.tools_considered or []
                        })

            # Optionally save thinking
            if self.thinking_manager.save_thinking and self.memory_manager.enabled:
                try:
                    # We don't have captured blocks in the streaming path; request a compact snapshot
                    snapshot_blocks = self.thinking_manager.generate_thinking(
                        prompt, available_tools, self.provider_client, self.model
                    )
                    formatted_all = self.thinking_manager.format_thinking_for_display(snapshot_blocks)
                    if formatted_all:
                        self.memory_manager.add_message("system", formatted_all)
                except Exception:
                    pass
        except Exception as e:
            self.logger.debug(f"Thinking mode error: {e}")

    def _stream_thinking_prompt(self, prompt_text: str, *, step: str, tools: Optional[List[str]] = None) -> None:
        """Stream a single thinking prompt using provider streaming and emit events."""
        try:
            # Header for this block
            header = f"<thinking>\n=== {step.upper()} ===\n"
            if self.stream_callback:
                self.stream_callback(header)

            # Stream from provider
            stream = self.provider_client.get_streaming_completion(
                messages=[{"role": "user", "content": prompt_text}],
                model=self.model,
                temperature=self.thinking_manager.temperature,
                max_tokens=self.thinking_manager.max_thinking_tokens,
                top_p=1.0,
                stop_sequences=None,
            )
            collected = ""
            for chunk in stream:
                delta = self._extract_text_from_chunk(chunk)
                if delta:
                    collected += delta
                    if self.stream_callback:
                        self.stream_callback(delta)

            # Footer separator
            if self.stream_callback:
                self.stream_callback("\n" + ("=" * 50) + "\n</thinking>\n")

            # Emit event callback once per block with collected content
            if self.event_callback:
                self.event_callback({
                    "type": "thinking",
                    "step": step,
                    "content": collected,
                    "tools": tools or [],
                })
        except Exception as e:
            # Non-fatal; just log
            self.logger.debug(f"Thinking streaming error: {e}")

    def _intent_requires_tools(self, prompt_text: str) -> bool:
        """Heuristic to decide whether tools are likely needed for this prompt."""
        text = (prompt_text or "").lower()
        want_web = any(k in text for k in (
            "search", "look up", "latest", "today", "current", "news", "headline", "website", "scrape", "scraper", "crawl", "fetch"
        ))
        if not want_web:
            if re.search(r"https?://", text) or re.search(r"\b(?:[a-z0-9-]+\.)+(?:[a-z]{2,})(?:/[^\s]*)?\b", text):
                want_web = True
        want_calc = any(k in text for k in ("calculate", "calculator", "calc", "sum", "difference", "multiply", "divide", "percent"))
        want_convert = any(k in text for k in ("convert", "units", "unit", "km", "miles", "celsius", "fahrenheit"))
        want_time = any(k in text for k in ("time in", "timezone", "utc", "pst", "est"))
        want_files = any(k in text for k in ("read file", "open file", "path", "csv", "search file", "grep"))
        # Detect arithmetic expressions like 1234*4321, including unicode × ÷
        if not want_calc:
            if re.search(r"\b\d+\s*([\+\-\*/×÷])\s*\d+(\s*([\+\-\*/×÷])\s*\d+)*\b", text):
                want_calc = True
        return any([want_web, want_calc, want_convert, want_time, want_files])

    def _get_last_user_message(self, messages: Optional[List[Dict[str, Any]]]) -> str:
        if not messages:
            return ""
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    def _choose_primary_tool(self, wants: Dict[str, bool], available_names: set[str], explicit: Optional[str]) -> Optional[str]:
        if explicit and explicit in available_names:
            return explicit
        if wants.get("calc") and "calculator" in available_names:
            return "calculator"
        if wants.get("web"):
            if wants.get("url") and "web_scraper" in available_names:
                return "web_scraper"
            if "web_search" in available_names:
                return "web_search"
            if "url_metadata" in available_names:
                return "url_metadata"
            if "web_headlines" in available_names:
                return "web_headlines"
        if wants.get("time") and "timezone" in available_names:
            return "timezone"
        if wants.get("convert") and "unit_converter" in available_names:
            return "unit_converter"
        if wants.get("files"):
            for candidate in ("file_read", "file_search", "csv_preview"):
                if candidate in available_names:
                    return candidate
        if wants.get("direct_tool") and available_names:
            # Fall back to first available tool if directed but alias unknown
            return next(iter(available_names))
        return None

    def _determine_tool_choice(
        self,
        prompt_text: str,
        tools_param: Optional[List[Dict[str, Any]]]
    ) -> Optional[Any]:
        if not tools_param:
            return None
        if not self._intent_requires_tools(prompt_text):
            explicit = self._extract_explicit_tool(prompt_text)
            if not explicit:
                return None
        else:
            explicit = self._extract_explicit_tool(prompt_text)
        try:
            wants = self._extract_intent(prompt_text)
        except Exception:
            wants = {}
        if explicit:
            wants["direct_tool"] = True
        available_names: set[str] = set()
        for t in tools_param:
            if isinstance(t, dict) and t.get("type") == "function":
                name = t.get("function", {}).get("name")
                if name:
                    available_names.add(name)
        primary = self._choose_primary_tool(wants, available_names, explicit)
        if primary:
            return {"type": "function", "function": {"name": primary}}
        return "required"

    def _handle_streaming_with_tools(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> str:
        """Handle streaming with tool support when possible.

        Uses provider-native streaming tool loop if available; otherwise falls back to
        non-streaming tool processing and streams the final text in chunks.
        """
        provider = (self.provider_name or "").lower()
        response_text = ""
        try:
            # Prefer provider-native streaming tool pipeline if present
            if hasattr(self.provider_client, "process_streaming_tool_calls") and provider in (
                "openai", "azure_openai", "openrouter", "deepseek", "mistral"
            ):
                # If strong intent for tools, require a tool call
                tool_choice_kwargs: Dict[str, Any] = {}
                try:
                    last_user = self._get_last_user_message(messages)
                    choice_spec = self._determine_tool_choice(last_user, tools)
                    if choice_spec:
                        self.logger.info(
                            "tool_choice",
                            extra={
                                "choice": choice_spec,
                                "available_tools": [
                                    t.get("function", {}).get("name")
                                    for t in tools
                                    if isinstance(t, dict)
                                ],
                                "prompt": last_user,
                            },
                        )
                        tool_choice_kwargs = {"tool_choice": choice_spec}
                except Exception:
                    tool_choice_kwargs = {}
                # Define a tool executor that uses the orchestrator and returns
                # standardized results for both event streaming and provider formatting
                def tool_executor(formatted_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    results: List[Dict[str, Any]] = []
                    for tc in formatted_tool_calls or []:
                        fname = (tc.get("function") or {}).get("name")
                        fargs_raw = (tc.get("function") or {}).get("arguments")
                        try:
                            fargs = json.loads(fargs_raw) if isinstance(fargs_raw, str) else (fargs_raw or {})
                        except Exception:
                            fargs = {}
                        tool_call_id = tc.get("id")
                        # Execute the tool via orchestrator
                        try:
                            result = self._execute_tool(fname, fargs)
                        except Exception as exec_err:
                            result = {"error": str(exec_err)}
                        # Content for provider formatting
                        if isinstance(result, (dict, list)):
                            content = json.dumps(result)
                        else:
                            content = str(result)
                        results.append({
                            "tool_call_id": tool_call_id,
                            "content": content,
                            # Extra fields for event streaming consumers
                            "tool": fname,
                            "result": result,
                        })
                    return results

                stream = self.provider_client.process_streaming_tool_calls(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    tools=tools,
                    tool_executor=tool_executor,
                    stop_sequences=self.stop_sequences,
                    max_iterations=5,
                    **tool_choice_kwargs
                )
                for item in stream:
                    # Status dicts during tool phases
                    if isinstance(item, dict):
                        typ = item.get("status")
                        if self.event_callback:
                            if typ == "processing":
                                self.event_callback({"type": "tool_request", "iteration": item.get("iteration")})
                            elif typ == "tool_calls":
                                self.event_callback({"type": "tool_request", "tool_calls": item.get("tool_calls", [])})
                            elif typ == "tool_results":
                                # Emit results per tool
                                for tr in item.get("tool_results", []) or []:
                                    self.event_callback({
                                        "type": "tool_result",
                                        "name": tr.get("tool"),
                                        "result": tr.get("result"),
                                    })
                        continue
                    # Streaming chunks for final answer
                    chunk_text = self._extract_text_from_chunk(item)
                    if chunk_text:
                        response_text += chunk_text
                        if self.stream_callback:
                            self.stream_callback(chunk_text)
                    # Cooperative cancellation
                        if getattr(self, "_cancel_requested", False):
                            break
                if response_text:
                    return response_text
                # If streaming yielded no visible output (some providers may suppress final deltas),
                # fall back to non-streaming flow to ensure we produce a reply.
                return self._handle_non_streaming_response(messages, tools)

            # Fallback: non-stream tool loop to produce final text; stream it artificially
            final_text = self._handle_non_streaming_response(messages, tools)
            if self.stream_callback and final_text:
                for i in range(0, len(final_text), 64):
                    delta = final_text[i : i + 64]
                    self.stream_callback(delta)
            return final_text
        except Exception as e:
            error = self.error_handler.handle_provider_error(
                e, self.provider_name, {"model": self.model}
            )
            raise error
    
    def _handle_streaming_response(self, messages: List[Dict[str, str]]) -> str:
        """Handle streaming response from the model."""
        try:
            response = self.provider_client.get_streaming_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop_sequences=self.stop_sequences
            )
            
            response_text = ""
            for chunk in response:
                chunk_text = self._extract_text_from_chunk(chunk)
                if chunk_text:
                    response_text += chunk_text
                    if self.stream_callback:
                        self.stream_callback(chunk_text)
            
            return response_text
            
        except Exception as e:
            error = self.error_handler.handle_provider_error(
                e, self.provider_name, {"model": self.model}
            )
            raise error
    
    def _handle_non_streaming_response(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Handle non-streaming response with optional tool support."""
        try:
            last_user = self._get_last_user_message(messages)
            tool_choice_spec = self._determine_tool_choice(last_user, tools)
            response = self.provider_client.get_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                stop_sequences=self.stop_sequences,
                tools=tools,
                tool_choice=tool_choice_spec if tool_choice_spec else ("auto" if tools else None)
            )
            
            # Track usage
            self._record_usage(response)
            if self.event_callback and self._last_usage:
                self.event_callback({"type": "model_usage", "usage": self._last_usage})
            
            # Check for tool calls
            tool_calls = self._extract_tool_calls(response)
            if tool_calls:
                return self._process_tool_calls(messages, response, tool_calls)
            
            # Extract text response
            return self._extract_text_from_response(response)
            
        except Exception as e:
            error = self.error_handler.handle_provider_error(
                e, self.provider_name, {"model": self.model}
            )
            raise error
    
    def _extract_text_from_chunk(self, chunk) -> str:
        """Extract text from a streaming chunk."""
        if self.provider_name.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek"):
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta'):
                    return chunk.choices[0].delta.content or ""
        elif self.provider_name.lower() == "anthropic":
            if isinstance(chunk, dict):
                if chunk.get("type") == "content_block_delta":
                    return chunk.get("delta", {}).get("text", "")
        elif self.provider_name.lower() == "google":
            return getattr(chunk, "text", "")
        
        return ""
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text from a non-streaming response."""
        if self.provider_name.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek"):
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content or ""
        elif self.provider_name.lower() == "anthropic":
            if isinstance(response, dict):
                content = response.get("content", [])
                if content and isinstance(content[0], dict):
                    return content[0].get("text", "")
        elif self.provider_name.lower() == "google":
            if hasattr(response, 'text'):
                return response.text
        
        return str(response)
    
    def _extract_tool_calls(self, response) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from response if present."""
        if self.provider_name.lower() in ("openai", "azure_openai", "mistral", "openrouter", "deepseek"):
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    return self.provider_client.format_tool_calls(message.tool_calls)
        elif self.provider_name.lower() == "anthropic":
            if isinstance(response, dict) and "content" in response and isinstance(response["content"], list):
                tool_use_parts = [part for part in response["content"] if part.get("type") == "tool_use"]
                if tool_use_parts:
                    return self.provider_client.format_tool_calls(tool_use_parts)
        
        return None
    
    def _process_tool_calls(
        self, messages: List[Dict[str, str]], response: Any, tool_calls: List[Dict[str, Any]],
        max_iterations: int = 5
    ) -> str:
        """Process tool calls and get final response."""
        if max_iterations <= 0:
            return "Maximum tool call iterations reached. Unable to complete the request."
        
        # Log tool call
        self.logger.debug("Tool call requested by model")
        
        # Extract content
        message_content = ""
        if hasattr(response, 'choices'):
            message_content = response.choices[0].message.content or ""
        elif isinstance(response, dict):
            message_content = response.get("content", "")
        
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": message_content,
            "tool_calls": tool_calls
        })
        
        # Execute each tool
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            function_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            tool_call_id = tool_call.get("id")
            
            # Execute tool
            self._tool_call_count += 1
            result = self._execute_tool(function_name, function_args)
            
            if self.event_callback:
                self.event_callback({
                    "type": "tool_call",
                    "name": function_name,
                    "args": function_args
                })
                self.event_callback({
                    "type": "tool_result",
                    "name": function_name,
                    "result": result,
                    "tool_call_id": tool_call_id
                })
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                "name": function_name
            })
        
        # Get next response
        tools_param = self.tool_orchestrator.get_tool_definitions()
        return self._handle_non_streaming_response(messages, tools_param)
    
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Execute a tool."""
        try:
            return self.tool_orchestrator.execute_tool(tool_name, tool_args)
        except Exception as e:
            self.logger.error(f"Error executing tool '{tool_name}': {e}")
            return {"error": str(e)}
    
    def _store_memory(self, prompt: str, response: str):
        """Store interaction in memory."""
        try:
            prompt_embedding = self.create_embedding(prompt)
            response_embedding = self.create_embedding(response)
            
            self.memory_manager.store_interaction(
                prompt, response, prompt_embedding, response_embedding
            )
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text."""
        try:
            if text in self._embedding_cache:
                return self._embedding_cache[text]
            # Try provider embeddings first if supported
            if hasattr(self.provider_client, 'create_embedding') and (
                not self._embeddings_provider_name or 
                self._embeddings_provider_name == self.provider_name
            ):
                try:
                    emb = self.provider_client.create_embedding(text, self._embedding_model)
                    self._embedding_cache[text] = emb
                    if len(self._embedding_cache) > 64:
                        self._embedding_cache.pop(next(iter(self._embedding_cache)))
                    return emb
                except Exception as e:
                    self.logger.debug(f"Provider embeddings failed, falling back to OpenAI: {e}")
            
            # Fallback to OpenAI embeddings
            if self.embedding_client is None:
                self.embedding_client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY") or self.api_key
                )
            
            response = self.embedding_client.embeddings.create(
                input=text,
                model=self._embedding_model
            )
            emb = response.data[0].embedding
            self._embedding_cache[text] = emb
            if len(self._embedding_cache) > 64:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            return emb
            
        except Exception as e:
            masked_error = mask_string(str(e))
            self.logger.error(f"Error creating embedding: {masked_error}")
            raise
    
    def _record_usage(self, response: Any):
        """Record token usage from response."""
        try:
            def _coerce(v):
                try:
                    return int(v or 0)
                except Exception:
                    return 0
            
            usage = None
            if hasattr(response, "usage"):
                usage = response.usage
            elif isinstance(response, dict) and "usage" in response:
                usage = response["usage"]
            
            if usage:
                self._last_usage = {
                    "prompt_tokens": _coerce(getattr(usage, "prompt_tokens", 0)),
                    "completion_tokens": _coerce(getattr(usage, "completion_tokens", 0)),
                    "total_tokens": _coerce(getattr(usage, "total_tokens", 0))
                }
                
                for k, v in self._last_usage.items():
                    self._total_usage[k] += v
        except Exception:
            pass
    
    # Callback methods
    def set_stream_callback(self, callback: Callable[[str], None]):
        """Set streaming callback."""
        self.stream_callback = callback
    
    def clear_stream_callback(self):
        """Clear streaming callback."""
        self.stream_callback = None
    
    def set_event_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set event callback."""
        self.event_callback = callback
    
    def clear_event_callback(self):
        """Clear event callback."""
        self.event_callback = None
    
    # Utility methods
    def update_settings(self, **kwargs):
        """Update instance settings."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def print_settings(self):
        """Print current settings."""
        settings = {
            "Provider Settings": {
                "Provider": self.provider_name,
                "Model": self.model,
                "Base URL": self.base_url
            },
            "Model Settings": {
                "Temperature": self.temperature,
                "Max Tokens": self.max_tokens,
                "Top P": self.top_p,
                "Stop Sequences": self.stop_sequences
            },
            "Memory Settings": {
                "Enabled": self.memory_enabled,
                "Max Messages": self.memory_max_messages,
                "Session ID": self.session_id
            },
            "Tool Settings": {
                "Enabled": self.tools_enabled,
                "Timeout": self.tools_timeout
            },
            "Usage Stats": self._total_usage
        }
        
        print("\nLLM Configuration Settings:")
        print("=" * 50)
        for category, values in settings.items():
            print(f"\n{category}:")
            print("-" * len(category))
            for key, value in values.items():
                print(f"{key:20}: {value}")
    
    # Context manager and cleanup
    def cleanup(self):
        """Clean up all resources."""
        self.__exit__(None, None, None)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with comprehensive cleanup."""
        # Clean up async components
        if self._async_provider:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    # If a loop is running, schedule the close and return
                    loop.create_task(self._async_provider.__aexit__(None, None, None))
                except RuntimeError:
                    # No running loop; create a fresh one
                    asyncio.run(self._async_provider.__aexit__(None, None, None))
            except Exception as e:
                self.logger.debug(f"Error closing async provider: {e}")
        
        # Clean up provider client
        try:
            if hasattr(self.provider_client, 'close'):
                self.provider_client.close()
        except Exception as e:
            self.logger.debug(f"Error closing provider client: {e}")
        
        # Clean up embedding client
        try:
            if self.embedding_client and hasattr(self.embedding_client, 'close'):
                self.embedding_client.close()
        except Exception as e:
            self.logger.debug(f"Error closing embedding client: {e}")
        
        # Clean up MCP client
        try:
            if self.mcp_client and hasattr(self.mcp_client, 'close'):
                self.mcp_client.close()
        except Exception as e:
            self.logger.debug(f"Error closing MCP client: {e}")
        
        # Clean up memory manager
        try:
            if self.memory_manager:
                self.memory_manager.close()
        except Exception as e:
            self.logger.debug(f"Error closing memory manager: {e}")
        
        # Clean up tool orchestrator
        try:
            if self.tool_orchestrator:
                self.tool_orchestrator.close()
        except Exception as e:
            self.logger.debug(f"Error closing tool orchestrator: {e}")
        
        self.logger.debug("LLM cleanup completed")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LLM(provider='{self.provider_name}', model='{self.model}')"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"LLM using {self.provider_name} {self.model}"
    
    def __bool__(self) -> bool:
        """Boolean evaluation."""
        return bool(self.api_key)

    # Convenience wrapper for tests/UX: return thinking + response
    def get_response_with_thinking(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {"thinking": None, "response": None}
        # Generate thinking blocks (non-streaming) for preview
        if getattr(self, "thinking_manager", None) and self.thinking_manager.enabled:
            try:
                available_tools: List[str] = []
                try:
                    available_tools = self.tool_orchestrator.tool_manager.list() if self.tool_orchestrator else []
                except Exception:
                    pass
                blocks = self.thinking_manager.generate_thinking(
                    prompt, available_tools, self.provider_client, self.model
                )
                formatted = self.thinking_manager.format_thinking_for_display(blocks)
                result["thinking"] = formatted
            except Exception as e:
                self.logger.debug(f"Thinking mode error: {e}")
        # Then get the normal response (tools/streaming per config)
        result["response"] = self.get_response(prompt, system_prompt)
        return result
