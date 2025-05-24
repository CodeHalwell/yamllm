from yamllm.core.parser import parse_yaml_config, YamlLMConfig
from yamllm.memory import ConversationStore, VectorStore
from openai import OpenAI, OpenAIError
from typing import Optional, Dict, Any, Callable, TypeVar, cast
import os
import time
from typing import List
import logging
import dotenv
import json
from yamllm.tools.utility_tools import WebSearch, Calculator, TimezoneTool, UnitConverter, WeatherTool, WebScraper
import concurrent.futures
from yamllm.providers.factory import ProviderFactory
from yamllm.providers.base import Message, ToolDefinition


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

    # Create file handler
    file_handler = logging.FileHandler(config.logging.file)
    formatter = logging.Formatter(config.logging.format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class LLM(object):
    """
    Main LLM interface class for YAMLLM.

    This class handles configuration loading and API interactions
    with language models. It serves as a base class for provider-specific
    implementations.

    Args:
        config_path (str): Path to YAML configuration file
        api_key (str): API key for the LLM service

    Examples:
        >>> llm = LLM(config_path = "config.yaml", api_key = "your-api-key")
        >>> response = llm.query("Hello, world!")
    """
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
        self.model = self.config.provider.model
        self.base_url = self.config.provider.base_url

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
        self.session_id = self.config.context.memory.session_id
        self.conversation_db_path = self.config.context.memory.conversation_db
        self.vector_index_path = self.config.context.memory.vector_store.index_path
        self.vector_metadata_path = self.config.context.memory.vector_store.metadata_path
        self.vector_store_top_k = self.config.context.memory.vector_store.top_k

        # Output settings
        self.output_format = self.config.output.format
        self.output_stream = self.config.output.stream

        # Tool settings
        self.tools_enabled = self.config.tools.enabled
        self.tools = self.config.tools.tool_list
        self.tools_timeout = self.config.tools.tool_timeout

        # Safety settings
        self.content_filtering = self.config.safety.content_filtering
        self.max_requests_per_minute = self.config.safety.max_requests_per_minute
        self.sensitive_keywords = self.config.safety.sensitive_keywords

        # Initialize the provider using the factory
        self.provider = ProviderFactory.create_provider(
            provider_name=self.provider_name,
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            logger=self.logger,
            tools=self.tools,
            tools_enabled=self.tools_enabled
        )

        # For backward compatibility
        self.client = self.provider.client if hasattr(self.provider, 'client') else None
        self.embedding_client = self.provider.embedding_client if hasattr(self.provider, 'embedding_client') else None

        # Initialize memory and vector store if enabled
        self.memory = None
        self.vector_store = None
        if self.memory_enabled:
            self._initialize_memory()

    def _initialize_memory(self):
        """Initialize memory and vector store"""
        self.memory = ConversationStore(db_path=self.conversation_db_path)
        self.vector_store = VectorStore(
            store_path=os.path.dirname(self.vector_index_path)
        )
        if not self.memory.db_exists():
            self.memory.create_db()

    def create_embedding(self, text: str) -> bytes:
        """
        Create an embedding for the given text.

        Args:
            text (str): The text to create an embedding for.

        Returns:
            bytes: The embedding as bytes.

        Raises:
            Exception: If there is an error creating the embedding.
        """
        try:
            response = self._make_api_call(
                self.embedding_client.embeddings.create,
                input=text,
                model="text-embedding-3-small"
            )
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
        return parse_yaml_config(self.config_path)

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
        except OpenAIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during query: {str(e)}")
            raise Exception(f"Unexpected error during query: {str(e)}")

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a response from the language model based on the provided prompt and optional system prompt.
        Supports tool calling if tools_enabled is True.

        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (Optional[str]): An optional system prompt for context.

        Returns:
            str: The response from the language model.
        """
        messages = self._prepare_messages(prompt, system_prompt)

        # Prepare tools if enabled
        tools_param = self._prepare_tools() if self.tools_enabled and self.tools else None

        # Generate response based on streaming preference
        if self.output_stream:
            response_text = self._handle_streaming_with_tool_detection(messages, tools_param)
        else:
            response_text = self._handle_non_streaming_response(messages, tools_param)

        # Store conversation in memory
        if response_text:
            self._store_memory(prompt, response_text, self.session_id)

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

            # Find similar messages
            similar_context = ""
            try:
                similar_results = self.find_similar_messages(prompt, k=2)
                if similar_results:
                    similar_context = "\nRelevant context from previous conversations:\n" + \
                        "\n".join([f"{m['role']}: {m['content']}" for m in similar_results])
            except Exception as e:
                self.logger.debug(f"Error finding similar messages: {str(e)}")

            # Add current prompt with context
            messages.append({
                "role": "user", 
                "content": f"{prompt}{similar_context}"
            })
        else:
            # Just add the current prompt if memory is disabled
            messages.append({
                "role": "user",
                "content": str(prompt)
            })

        return messages

    def _prepare_standard_completion_params(self, messages):
        """
        Prepare standard completion parameters that should work across providers.

        Args:
            messages (list): Message objects.

        Returns:
            dict: Parameters for API request.
        """
        # Convert messages to the standardized format
        standardized_messages = [
            Message(role=msg["role"], content=msg["content"], name=msg.get("name"))
            for msg in messages
        ]

        return self.provider.prepare_completion_params(
            messages=standardized_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences
        )

    def _prepare_tools(self):
        """
        Prepare tool definitions for the API call.

        Returns:
            list: List of standardized tool definitions
        """        
        # Map tool names to their respective classes
        tool_classes = {
            "web_search": WebSearch,
            "calculator": Calculator,
            "timezone": TimezoneTool,
            "unit_converter": UnitConverter,
            "weather": WeatherTool,
            "web_scraper": WebScraper
        }

        tool_definitions = []

        for tool_name in self.tools:
            if tool_name not in tool_classes:
                self.logger.warning(f"Tool '{tool_name}' not found in available tools")
                continue

            tool_class = tool_classes[tool_name]

            if tool_name == "weather":
                weather_api_key = os.environ.get("WEATHER_API_KEY")
                if not weather_api_key:
                    raise ValueError("Weather API key not found in environment variable 'WEATHER_API_KEY'")
                tool_instance = tool_class(api_key=weather_api_key)
            else:
                tool_instance = tool_class()

            # Define parameters schema based on the tool
            parameters_schema = self._get_tool_parameters(tool_name)

            # Create a standardized tool definition
            tool_definition = ToolDefinition(
                name=tool_instance.name,
                description=tool_instance.description,
                parameters=parameters_schema
            )

            tool_definitions.append(tool_definition)

        return tool_definitions

    def _get_tool_parameters(self, tool_name):
        """
        Get parameters schema for a specific tool.

        Args:
            tool_name (str): Name of the tool

        Returns:
            dict: Parameters schema for the tool
        """
        # Define parameter schemas for each tool
        tool_parameters = {
            "web_search": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return"
                    }
                },
                "required": ["query"]
            },
            "calculator": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', 'sin(0.5)', 'np.log(100)')"
                    }
                },
                "required": ["expression"]
            },
            "timezone": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "The time to convert (e.g., '2023-04-15 14:30:00')"
                    },
                    "from_tz": {
                        "type": "string",
                        "description": "Source timezone (e.g., 'America/New_York')"
                    },
                    "to_tz": {
                        "type": "string",
                        "description": "Target timezone (e.g., 'Europe/London')"
                    }
                },
                "required": ["time", "from_tz", "to_tz"]
            },
            "unit_converter": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The value to convert"
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "Source unit (e.g., 'kg', 'mile', 'celsius')"
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "Target unit (e.g., 'lb', 'km', 'fahrenheit')"
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            },
            "weather": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location for weather information (e.g., 'New York', 'London')"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date for the weather forecast (e.g., '2023-04-15')"
                    }
                },
                "required": ["location"]
            },
            "web_scraper": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to be scraped and summarised"
                    }
                },
                "required": ["url"]
            }
        }

        return tool_parameters.get(tool_name, {"type": "object", "properties": {}})

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
            # Make a low-token request to see if the model will use tools
            preview_params = self._prepare_standard_completion_params(messages)
            preview_params["max_tokens"] = 10  # Just enough to detect tool usage
            
            if tools_param:
                preview_params["tools"] = tools_param
                preview_params["tool_choice"] = "auto"
                
            preview_response = self._make_api_call(
                self.client.chat.completions.create,
                **preview_params
            )
            
            # Check if model wants to use tools
            if (tools_param and hasattr(preview_response.choices[0].message, "tool_calls") 
                and preview_response.choices[0].message.tool_calls):
                # Model wants to use tools, use non-streaming
                console = Console()
                console.print("\n[yellow]Using tools to answer this question...[/yellow]")
                return self._handle_non_streaming_response(messages, tools_param)
            else:
                # Model doesn't need tools, use streaming
                return self._handle_streaming_response(messages)
              
        except Exception as e:
            self.logger.warning(f"Streaming with tool detection failed: {str(e)}")
            # Fall back to streaming without tool detection
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
            # Convert messages to the standardized format
            standardized_messages = [
                Message(role=msg["role"], content=msg["content"], name=msg.get("name"))
                for msg in messages
            ]

            # Get parameters for the API request
            params = self._prepare_standard_completion_params(messages)

            params["stream"] = True
            
            response = self._make_api_call(
                self.client.chat.completions.create,
                **params
            )
            
            console = Console()
            response_text = ""
            print()
            
            with Live(console=console, refresh_per_second=10) as live:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        md = Markdown(f"\nAI: {response_text}", style="green")
                        live.update(md)

            return response_text

        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            raise Exception(f"Error getting streaming response: {str(e)}")

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
            # Prepare API call parameters
            completion_params = self._prepare_standard_completion_params(messages)
            
            # Add tools if available
            if tools_param:
                completion_params["tools"] = tools_param
                completion_params["tool_choice"] = "auto"
                
            response = self._make_api_call(
                self.client.chat.completions.create,
                **completion_params
            )
                       
            # Check if the model wants to use a tool
            if tools_param and hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                return self._process_tool_calls(messages, response.choices[0].message)
            else:
                response_text = response.choices[0].message.content
                
                # Display the response
                console = Console()
                if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                    md = Markdown("\nAI:" + response_text, style="green")
                    console.print(md)
                else:
                    console.print("\nAI:" + response_text, style="green")
                    
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
        # Convert messages to the standardized format
        standardized_messages = [
            Message(role=msg["role"], content=msg["content"], name=msg.get("name"))
            for msg in messages
        ]

        # Define a function to execute tools that will be passed to the provider
        def execute_tool_func(tool_name, tool_args):
            return self._execute_tool(tool_name, tool_args)

        # Use the provider's implementation
        try:
            # For Google, filter out any messages with null content
            if self.provider == 'google':
                clean_messages = [m for m in messages if m.get("content") is not None]
                for i, m in enumerate(clean_messages):
                    # Ensure content is always a string, never None
                    if m.get("content") is None:
                        clean_messages[i]["content"] = ""
                    
                params = self._prepare_standard_completion_params(clean_messages)
            else:
                params = self._prepare_standard_completion_params(messages)
                
            next_response = self._make_api_call(
                self.client.chat.completions.create,
                **params
            )
            
            next_message = next_response.choices[0].message
            
            # Check if we need more tool calls
            if hasattr(next_message, "tool_calls") and next_message.tool_calls:
                # Recursive call for next round of tool calls
                return self._process_tool_calls(messages, next_message, max_iterations - 1)
            else:
                # We have our final text response
                response_text = next_message.content
                
                # Display the final response
                if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                    md = Markdown("\nAI:" + response_text, style="green")
                    console.print(md)
                else:
                    console.print("\nAI:" + response_text, style="green")
                    
                return response_text
                
        except Exception as e:
            self.logger.error(f"Error processing tool calls: {str(e)}")
            raise Exception(f"Error processing tool calls: {str(e)}")

    def _execute_tool(self, tool_name, tool_args):
        """
        Execute a tool with the provided arguments.

        Args:
            tool_name (str): Name of the tool to execute
            tool_args (dict): Arguments for the tool

        Returns:
            Any: The result of the tool execution
        """
        # Map tool names to their respective classes
        tool_classes = {
            "web_search": WebSearch,
            "calculator": Calculator,
            "timezone": TimezoneTool,
            "unit_converter": UnitConverter,
            "weather": WeatherTool,
            "web_scraper": WebScraper
        }

        if tool_name not in tool_classes:
            return f"Error: Tool '{tool_name}' not found"

        # Initialize the tool
        try:
            if tool_name == "weather":
                weather_api_key = os.environ.get("WEATHER_API_KEY")
                if not weather_api_key:
                    return "Error: WEATHER_API_KEY not set"
                tool = tool_classes[tool_name](api_key=weather_api_key)
            else:
                tool = tool_classes[tool_name]()

            # Execute the tool with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(tool.execute, **tool_args)
                try:
                    result = future.result(timeout=self.tools_timeout)
                    return result
                except concurrent.futures.TimeoutError:
                    return f"Error: Tool execution timed out after {self.tools_timeout} seconds"

        except Exception as e:
            self.logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return f"Error executing tool: {str(e)}"

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

    def __enter__(self):
        """Support context manager interface."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up resources when exiting context manager."""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'embedding_client'):
            self.embedding_client.close()

    def __bool__(self) -> bool:
        """Return True if the LLM instance is properly initialized with an API key."""
        return bool(self.api_key)


class OpenAIGPT(LLM):
    """
    A class to interact with OpenAI's GPT models.

    Attributes:
        provider_name (str): The name of the provider, set to "openai".

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the OpenAIGPT instance with the given configuration path and API key.

    Initializes the OpenAIGPT instance.

    Args:
        config_path (str): The path to the configuration file.
        api_key (str): The API key for accessing OpenAI's services.
    """
    def __init__(self, config_path: str, api_key: str) -> None:
        self.provider = "openai"  # Set provider before super() to ensure correct initialization
        super().__init__(config_path, api_key)

class DeepSeek(LLM):
    """
    DeepSeek is a subclass of LLM that initializes a connection to the DeepSeek provider.

    Attributes:
        provider_name (str): The name of the provider, set to 'deepseek'.

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the DeepSeek instance with the given configuration path and API key.

        Initializes the DeepSeek instance.

        Args:
            config_path (str): The path to the configuration file.
            api_key (str): The API key for authentication.
        """
    def __init__(self, config_path: str, api_key: str) -> None:
        self.provider = 'deepseek'  # Set provider before super() to ensure correct initialization
        super().__init__(config_path, api_key)

class MistralAI(LLM):
    """
    MistralAI class for interacting with the Mistral language model.

    Attributes:
        provider_name (str): The name of the AI provider, set to 'mistral'.

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the MistralAI instance with the given configuration path and API key.
    """
    def __init__(self, config_path: str, api_key: str) -> None:
        self.provider = 'mistral'  # Set provider before super() to ensure correct initialization
        super().__init__(config_path, api_key)

class GoogleGemini(LLM):
    """
    GoogleGemini is a specialized class for interacting with Google's Gemini models
    through their OpenAI-compatible interface.

    Attributes:
        provider_name (str): The name of the AI provider, set to 'google'.

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the GoogleGemini instance with the given configuration path and API key.
    """
    def __init__(self, config_path: str, api_key: str) -> None:
        """Initialize with Google-specific settings"""
        # Set provider before super() to ensure correct initialization
        self.provider = 'google'
        super().__init__(config_path, api_key)
