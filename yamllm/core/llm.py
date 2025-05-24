from yamllm.core.parser import parse_yaml_config, YamlLMConfig
from yamllm.memory import ConversationStore, VectorStore
from openai import OpenAI, OpenAIError
from typing import Optional, Dict, Any
import os
from typing import List
import logging
import dotenv
from rich.live import Live
from rich.markdown import Markdown
from rich.console import Console 
import json
from yamllm.tools.utility_tools import WebSearch, Calculator, TimezoneTool, UnitConverter, WeatherTool, WebScraper
import concurrent.futures

# Import provider interfaces for backwards compatibility
try:
    from yamllm.core.providers import BaseProvider, OpenAIProvider, AnthropicProvider
except ImportError:
    # Provider interfaces not available
    pass


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
        self.provider = self.config.provider.name if not hasattr(self, 'provider') else self.provider
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

        # Initialize OpenAI client for regular requests
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Initialize OpenAI client for embeddings
        self.embedding_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

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
        Create an embedding for the given text using OpenAI's API.

        Args:
            text (str): The text to create an embedding for.

        Returns:
            bytes: The embedding as bytes.

        Raises:
            Exception: If there is an error creating the embedding.
        """
        try:
            response = self.embedding_client.embeddings.create(
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
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop_sequences or None
        }

    def _prepare_tools(self):
        """
        Prepare tool definitions for the API call.
        
        Returns:
            list: List of tool definitions in the format expected by the OpenAI API
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
            
            tool_definition = {
                "type": "function",
                "function": {
                    "name": tool_instance.name,
                    "description": tool_instance.description,
                    "parameters": parameters_schema
                }
            }
            
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
                
            preview_response = self.client.chat.completions.create(**preview_params)
            
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
            self.logger.warning(f"Preview request failed: {str(e)}")
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
            params = self._prepare_standard_completion_params(messages)
            params["stream"] = True
            
            response = self.client.chat.completions.create(**params)
            
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
                
            response = self.client.chat.completions.create(**completion_params)
                       
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
        Special handling for Google Gemini which can't handle null content values.
        
        Args:
            messages (list): List of message objects.
            model_message: The model's message containing tool calls.
            max_iterations (int): Maximum number of tool call iterations.
            
        Returns:
            str: Final response text.
        """
        if max_iterations <= 0:
            return "Maximum tool call iterations reached. Unable to complete the request."
            
        console = Console()
        console.print("\n[bold yellow]Tool Call Requested:[/bold yellow]")
        
        # For Google, ensure content is never null
        message_content = model_message.content or ""
        
        # Add the assistant's message with tool calls to conversation
        messages.append({
            "role": "assistant",
            "content": message_content,  # Ensure content is never null
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in model_message.tool_calls
            ]
        })
        
        # Process each tool call
        for tool_call in model_message.tool_calls:
            function_name = tool_call.function.name
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
                
            console.print(f"[yellow]Function:[/yellow] {function_name}")
            console.print(f"[yellow]Arguments:[/yellow] {tool_call.function.arguments}")
            
            # Execute the tool
            tool_result = self._execute_tool(function_name, function_args)
            
            # Format the result for display (truncated if too long)
            result_str = str(tool_result)
            if len(result_str) > 200:
                display_result = result_str[:200] + "..."
            else:
                display_result = result_str
                
            console.print(f"[yellow]Result:[/yellow] {display_result}")
            
            # Add the tool result to conversation with formatting instructions
            tool_content = json.dumps(tool_result) if isinstance(tool_result, (list, dict)) else str(tool_result)
            
            # Add natural language instruction for the model
            if function_name == "web_search":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content
                })
                
                # Add a user message with formatting instructions
                messages.append({
                    "role": "user",
                    "content": "Please summarize these search results in a natural, conversational way. Highlight the most important points and present them as if you're having a conversation with me."
                })
            else:
                # For other tools, just add the result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content
                })
        
        # Get the next response from the model
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
                
            next_response = self.client.chat.completions.create(**params)
            
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
            
            # Special handling for Google's empty text parameter error
            if self.provider == 'google' and "empty text parameter" in str(e):
                # Attempt to create a direct response based on the tool results
                try:
                    # Collect the tool results we already have
                    tool_results_summary = ""
                    for msg in messages:
                        if msg.get("role") == "tool" and msg.get("content"):
                            tool_name = "unknown tool"
                            # Try to find the related tool call to get its name
                            tool_call_id = msg.get("tool_call_id")
                            for m in messages:
                                if m.get("role") == "assistant" and m.get("tool_calls"):
                                    for tc in m.get("tool_calls", []):
                                        if tc.get("id") == tool_call_id and tc.get("function", {}).get("name"):
                                            tool_name = tc.get("function", {}).get("name")
                                            break
                            
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
                            else:
                                tool_results_summary += f"\n{tool_name} results: {msg.get('content')[:200]}...\n"
                    
                    # Create a helpful response
                    response_text = (
                        f"I found some information for you, but encountered a technical issue when trying to formulate a complete response. "
                        f"Here's what I found:{tool_results_summary}\n\n"
                        f"Could you please ask your question again or try a different approach?"
                    )
                    
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
                    
                except Exception:
                    # If even our fallback fails, return a generic message
                    response_text = "I encountered a technical issue while processing the tool results. Please try your question again or phrase it differently."
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
            else:
                # For other errors, re-raise
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
        provider (str): The name of the provider, set to "openai".

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the OpenAIGPT instance with the given configuration path and API key.

    Initializes the OpenAIGPT instance.

    Args:
        config_path (str): The path to the configuration file.
        api_key (str): The API key for accessing OpenAI's services.
    """
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = "openai"

class DeepSeek(LLM):
    """
    DeepSeek is a subclass of LLM that initializes a connection to the DeepSeek provider.

    Attributes:
        provider (str): The name of the provider, set to 'deepseek'.

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the DeepSeek instance with the given configuration path and API key.

        Initializes the DeepSeek instance.

        Args:
            config_path (str): The path to the configuration file.
            api_key (str): The API key for authentication.
        """
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = 'deepseek'

class MistralAI(LLM):
    """
    MistralAI class for interacting with the Mistral language model.
    
    Attributes:
        provider (str): The name of the AI provider, set to 'mistral'.
    
    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the MistralAI instance with the given configuration path and API key.
        get_response(prompt: str, system_prompt: Optional[str] = None) -> str:
            Generates a response from the Mistral language model based on the given prompt and optional system prompt.
    """
    # Real-time query keywords - similar to GoogleGemini implementation
    real_time_keywords = [
        # Weather and natural phenomena
        "weather", "forecast", "temperature", "humidity", "precipitation", "rain", "snow", "storm", 
        "hurricane", "tornado", "earthquake", "tsunami", "typhoon", "cyclone", "flood", "drought", 
        "wildfire", "air quality", "pollen", "uv index", "sunrise", "sunset", "climate",
        
        # News and current events
        "news", "headline", "latest", "breaking", "current", "recent", "today", "yesterday",
        "this week", "this month", "ongoing", "developing", "situation", "event", "incident", 
        "announcement", "press release", "update", "coverage", "report", "bulletin", "fixture",
        
        # Sports and entertainment
        "score", "game", "match", "tournament", "championship", "playoff", "standings", 
        "leaderboard", "box office", "premiere", "release", "concert", "performance", 
        "episode", "ratings", "award", "nominations", "season", "show", "event",
        
        # Time-specific queries
        "now", "currently", "present", "moment", "tonight", "this morning", "this afternoon", 
        "this evening", "upcoming", "soon", "shortly", "imminent", "expected", "anticipated", 
        "scheduled", "real-time", "live", "happening", "occurring", "next"
    ]
    
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = 'mistral'
    
    def _prepare_standard_completion_params(self, messages):
        """
        Override to prepare parameters compatible with Mistral's API requirements.
        
        Args:
            messages (list): Message objects.
            
        Returns:
            dict: Parameters for API request with Mistral-specific adjustments.
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        # Only add stop parameter if it contains actual stop sequences
        if self.stop_sequences and len(self.stop_sequences) > 0:
            params["stop"] = self.stop_sequences
        return params
    
    def _handle_streaming_with_tool_detection(self, messages, tools_param=None):
        """
        Override to handle streaming with tool detection for Mistral's API.
        
        Args:
            messages (list): List of message objects.
            tools_param (list, optional): Tool definitions.
            
        Returns:
            str: Response text.
        """
        try:
            # Get the last user message and extract a clean query
            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            actual_query = last_user_msg.split("\nRelevant context from previous conversations:")[0].strip()

            is_real_time_query = any(keyword in actual_query.lower() for keyword in self.real_time_keywords) and ("web_search" in self.tools)
            
            if is_real_time_query and tools_param and self.tools_enabled:
                # Use non-streaming path with tools if it's a real-time query
                console = Console()
                console.print("\n[yellow]Using tools to answer this real-time question...[/yellow]")
                return self._handle_non_streaming_response(messages, tools_param)
            else:
                # Otherwise, use streaming response
                return self._handle_streaming_response(messages)
        except Exception as e:
            self.logger.warning(f"Tool detection request failed: {str(e)}")
            return self._handle_streaming_response(messages)
    
    def _handle_non_streaming_response(self, messages, tools_param=None):
        """
        Handle non-streaming response with Mistral-specific tool handling.
        
        For real-time queries, bypass model tool selection by directly executing web search.
        For other messages, include tool definitions in the API request.
        
        Args:
            messages (list): List of message objects.
            tools_param (list, optional): Tool definitions.
            
        Returns:
            str: Response text.
        """
        try:
            # Get the last user message and extract the actual query text
            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            actual_query = last_user_msg.split("\nRelevant context from previous conversations:")[0].strip()

            is_real_time_query = any(keyword in actual_query.lower() for keyword in self.real_time_keywords) and ("web_search" in self.tools)

            if is_real_time_query and tools_param and self.tools_enabled:
                console = Console()
                console.print("\n[yellow]Using tools to answer this real-time question...[/yellow]")
                # Directly execute web search
                web_search = WebSearch()
                web_search_args = {
                    "query": actual_query,
                    "max_results": 5
                }
                
                console.print("\n[bold yellow]Tool Call Requested:[/bold yellow]")
                console.print("[yellow]Function:[/yellow] web_search")
                console.print(f"[yellow]Arguments:[/yellow] {json.dumps(web_search_args)}")
                
                search_results = web_search.execute(**web_search_args)
                result_str = str(search_results)
                display_result = result_str[:200] + "..." if len(result_str) > 200 else result_str
                console.print(f"[yellow]Result:[/yellow] {display_result}")
                
                if isinstance(search_results, dict) and "error" in search_results:
                    error_msg = search_results.get("error", "Unknown search error")
                    console.print(f"[red]Search error: {error_msg}[/red]")
                    response_text = (f"I tried to search for information about '{actual_query}', but encountered a search error. "
                                     f"Could you please try a different query?")
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
                
                # Convert search results to a readable format
                readable_results = ""
                if isinstance(search_results, dict) and "results" in search_results:
                    for i, result in enumerate(search_results.get("results", [])[:3]):
                        readable_results += f"Source {i+1}: {result.get('title', 'No title')}\n"
                        readable_results += f"Summary: {result.get('snippet', 'No information')}\n\n"
                
                # Enhance the user prompt with the search results
                final_user_msg = (f"{actual_query}\n\nHere are some search results I found:\n\n{readable_results}"
                                  f"\n\nPlease summarize this information in a helpful, conversational way.")
                
                new_messages = []
                for m in messages:
                    if m["role"] == "user" and m["content"] == last_user_msg:
                        new_messages.append({"role": "user", "content": final_user_msg})
                    else:
                        new_messages.append(m)
                
                try:
                    params = self._prepare_standard_completion_params(new_messages)
                    final_response = self.client.chat.completions.create(**params)
                    response_text = final_response.choices[0].message.content
                    
                    if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                        md = Markdown("\nAI:" + response_text, style="green")
                        console.print(md)
                    else:
                        console.print("\nAI:" + response_text, style="green")
                    return response_text
                except Exception as e:
                    self.logger.warning(f"Final response generation failed: {str(e)}")
                    if readable_results:
                        response_text = (f"Based on my search for '{actual_query}', I found:\n\n{readable_results}\n\n"
                                         f"I couldn't generate a summary, but these are the relevant search results.")
                    else:
                        response_text = (f"I tried to search for information about '{actual_query}', but couldn't generate a complete response. "
                                         f"Please try rephrasing your question.")
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
            else:
                # For non-real-time queries, include tool definitions if available
                completion_params = self._prepare_standard_completion_params(messages)
                if tools_param and self.tools_enabled:
                    completion_params["tools"] = tools_param
                    completion_params["tool_choice"] = "auto"
                    self.logger.debug(f"Sending tools to Mistral API: {json.dumps(tools_param)}")
                
                console = Console()
                console.print("\n[yellow]Sending request with tools enabled...[/yellow]")
                response = self.client.chat.completions.create(**completion_params)
                
                if tools_param and hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    return self._process_tool_calls(messages, response.choices[0].message)
                else:
                    response_text = response.choices[0].message.content
                    if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                        md = Markdown("\nAI:" + response_text, style="green")
                        console.print(md)
                    else:
                        console.print("\nAI:" + response_text, style="green")
                    return response_text
        except Exception as e:
            self.logger.error(f"Non-streaming error: {str(e)}")
            raise Exception(f"Error getting non-streaming response: {str(e)}")
    
class GoogleGemini(LLM):
    """
    GoogleGemini is a specialized class for interacting with Google's Gemini models
    through their OpenAI-compatible interface.
    
    This class explicitly avoids using parameters that Google's API doesn't support,
    such as frequency_penalty and presence_penalty.
    """

    real_time_keywords = [
    # Weather and natural phenomena
    "weather", "forecast", "temperature", "humidity", "precipitation", "rain", "snow", "storm", 
    "hurricane", "tornado", "earthquake", "tsunami", "typhoon", "cyclone", "flood", "drought", 
    "wildfire", "air quality", "pollen", "uv index", "sunrise", "sunset", "climate",
    
    # News and current events
    "news", "headline", "latest", "breaking", "current", "recent", "today", "yesterday",
    "this week", "this month", "ongoing", "developing", "situation", "event", "incident", 
    "announcement", "press release", "update", "coverage", "report", "bulletin", "fixture"
    
    # Sports and entertainment
    "score", "game", "match", "tournament", "championship", "playoff", "standings", 
    "leaderboard", "box office", "premiere", "release", "concert", "performance", 
    "episode", "ratings", "award", "nominations", "season", "show", "event",
    
    # Time-specific queries
    "now", "currently", "present", "moment", "tonight", "this morning", "this afternoon", 
    "this evening", "upcoming", "soon", "shortly", "imminent", "expected", "anticipated", 
    "scheduled", "real-time", "live", "happening", "occurring", "next"
]

    def __init__(self, config_path: str, api_key: str) -> None:
        """Initialize with Google-specific settings"""
        # Set provider before super() to ensure correct initialization
        self.provider = 'google'
        super().__init__(config_path, api_key)
        
        
        # Re-initialize the client with the correct base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _prepare_standard_completion_params(self, messages):
        """
        Override to prepare parameters compatible with Google's API requirements.
        
        Args:
            messages (list): Message objects.
            
        Returns:
            dict: Parameters for API request with Google-specific adjustments.
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        
        # Only add stop parameter if it contains actual stop sequences
        if self.stop_sequences and len(self.stop_sequences) > 0:
            params["stop"] = self.stop_sequences
            
        return params
        
    def _handle_non_streaming_response(self, messages, tools_param=None):
        """
        Handle non-streaming response from the model with Google-specific tool handling.
        
        Args:
            messages (list): List of message objects.
            tools_param (list, optional): Tool definitions.
            
        Returns:
            str: Response text.
        """
        try:            
            # For Google Gemini, check if this is a real-time query that should use web_search
            is_real_time_query = False
            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            
            # Extract just the user's question without any context annotations
            actual_query = last_user_msg.split("\nRelevant context from previous conversations:")[0].strip()
            
            real_time_keywords = self.real_time_keywords
            if any(keyword in actual_query.lower() for keyword in real_time_keywords) and "web_search" in self.tools:
                is_real_time_query = True
            
            if is_real_time_query and tools_param and self.tools_enabled:
                # Force web search for real-time queries
                console = Console()
                console.print("\n[yellow]Using tools to answer this real-time question...[/yellow]")
                
                # Create a web search directly instead of going through the API
                web_search = WebSearch()
                web_search_args = {
                    "query": actual_query,  # Use the clean query without context annotations
                    "max_results": 5
                }
                
                # Display tool call information
                console.print("\n[bold yellow]Tool Call Requested:[/bold yellow]")
                console.print("[yellow]Function:[/yellow] web_search")
                console.print(f"[yellow]Arguments:[/yellow] {json.dumps(web_search_args)}")
                
                # Execute the search directly
                search_results = web_search.execute(**web_search_args)
                
                # Display a short version of the result
                result_str = str(search_results)
                if len(result_str) > 200:
                    display_result = result_str[:200] + "..."
                else:
                    display_result = result_str
                console.print(f"[yellow]Result:[/yellow] {display_result}")
                
                # Handle search failures gracefully
                if isinstance(search_results, dict) and "error" in search_results:
                    error_msg = search_results.get("error", "Unknown search error")
                    console.print(f"[red]Search error: {error_msg}[/red]")
                    
                    # Give a helpful response about the search failure
                    response_text = (f"I tried to search for information about '{actual_query}', but encountered "
                                    f"a search error. Could you please be more specific?")
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
                
                # Create clean new messages without any null content
                clean_messages = [m for m in messages if m.get("content") is not None]
                
                # Convert search results to a readable format
                readable_results = ""
                if isinstance(search_results, dict) and "results" in search_results:
                    for i, result in enumerate(search_results.get("results", [])[:3]):
                        readable_results += f"Source {i+1}: {result.get('title', 'No title')}\n"
                        readable_results += f"Summary: {result.get('snippet', 'No information')}\n\n"
                
                # Create a new user message with the search results
                final_user_msg = f"{actual_query}\n\nHere are some search results I found:\n\n{readable_results}\n\nPlease summarize this information in a helpful, conversational way."
                
                # Replace the last user message with our enhanced version
                for i in range(len(clean_messages)-1, -1, -1):
                    if clean_messages[i]["role"] == "user":
                        clean_messages[i]["content"] = final_user_msg
                        break
                
                # Send a new request with clean messages
                try:
                    # Use non-streaming for final response
                    params = self._prepare_standard_completion_params(clean_messages)
                    params["tools"] = [] # No tools for the final response
                    
                    final_response = self.client.chat.completions.create(**params)
                    response_text = final_response.choices[0].message.content
                    
                    # Display the final response
                    if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                        md = Markdown("\nAI:" + response_text, style="green")
                        console.print(md)
                    else:
                        console.print("\nAI:" + response_text, style="green")
                    
                    return response_text
                    
                except Exception as e:
                    self.logger.warning(f"Final response generation failed: {str(e)}")
                    # Create a direct response using the search results
                    if readable_results:
                        response_text = f"Based on my search for '{actual_query}', I found:\n\n{readable_results}\n\nI couldn't generate a summary, but these are the relevant search results."
                    else:
                        response_text = f"I tried to search for information about '{actual_query}', but couldn't find relevant results or generate a summary. Could you try rephrasing your question?"
                    
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
            else:
                # Standard non-streaming response for non-real-time queries
                # Prepare API call parameters
                completion_params = self._prepare_standard_completion_params(messages)
                
                # Add tools if available - using Google's expected format
                if tools_param and self.tools_enabled:
                    completion_params["tools"] = tools_param
                    completion_params["tool_choice"] = "auto"
                    
                response = self.client.chat.completions.create(**completion_params)
                        
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
            
    def _handle_streaming_with_tool_detection(self, messages, tools_param=None):
        """
        Override to handle streaming with tool detection for Google's API.
        
        Args:
            messages (list): List of message objects.
            tools_param (list, optional): Tool definitions.
            
        Returns:
            str: Response text.
        """
        try:
            # For Google, directly check if this is a weather/current info query
            is_real_time_query = False
            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            
            # Extract just the user's question without any context annotations
            actual_query = last_user_msg.split("\nRelevant context from previous conversations:")[0].strip()
            
            real_time_keywords = self.real_time_keywords
            if any(keyword in actual_query.lower() for keyword in real_time_keywords) and "web_search" in self.tools:
                is_real_time_query = True
            
            if is_real_time_query and tools_param:
                # Use non-streaming with tools for real-time queries
                console = Console()
                console.print("\n[yellow]Using tools to answer this real-time question...[/yellow]")
                
                # Create a web search directly instead of going through the API
                web_search = WebSearch()
                web_search_args = {
                    "query": actual_query,  # Use the clean query without context annotations
                    "max_results": 5
                }
                
                # Display tool call information
                console.print("\n[bold yellow]Tool Call Requested:[/bold yellow]")
                console.print("[yellow]Function:[/yellow] web_search")
                console.print(f"[yellow]Arguments:[/yellow] {json.dumps(web_search_args)}")
                
                # Execute the search directly
                search_results = web_search.execute(**web_search_args)
                
                # Display a short version of the result
                result_str = str(search_results)
                if len(result_str) > 200:
                    display_result = result_str[:200] + "..."
                else:
                    display_result = result_str
                console.print(f"[yellow]Result:[/yellow] {display_result}")
                
                # Handle search failures gracefully
                if isinstance(search_results, dict) and "error" in search_results:
                    error_msg = search_results.get("error", "Unknown search error")
                    console.print(f"[red]Search error: {error_msg}[/red]")
                    
                    # Give a helpful response about the search failure
                    response_text = (f"I tried to search for information about '{actual_query}', but encountered "
                                    f"a search error. Could you please be more specific about which recent earthquake "
                                    f"you're interested in? For example, mention the location or approximate date.")
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
                
                # Create clean new messages without any null content
                clean_messages = [m for m in messages if m.get("content") is not None]
                
                # Convert search results to a readable format
                readable_results = ""
                if isinstance(search_results, dict) and "results" in search_results:
                    for i, result in enumerate(search_results.get("results", [])[:3]):
                        readable_results += f"Source {i+1}: {result.get('title', 'No title')}\n"
                        readable_results += f"Summary: {result.get('snippet', 'No information')}\n\n"
                
                # Create a new user message with the search results, but maintain conversation context
                final_user_msg = f"{actual_query}\n\nHere are some search results I found:\n\n{readable_results}\n\nPlease summarize this information in a helpful, conversational way."
                
                # Replace the last user message with our enhanced version
                for i in range(len(clean_messages)-1, -1, -1):
                    if clean_messages[i]["role"] == "user":
                        clean_messages[i]["content"] = final_user_msg
                        break
                
                # Send a new request with clean messages
                try:
                    # Use non-streaming for final response
                    params = self._prepare_standard_completion_params(clean_messages)
                    params["tools"] = [] # No tools for the final response
                    
                    final_response = self.client.chat.completions.create(**params)
                    response_text = final_response.choices[0].message.content
                    
                    # Display the final response
                    if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                        md = Markdown("\nAI:" + response_text, style="green")
                        console.print(md)
                    else:
                        console.print("\nAI:" + response_text, style="green")
                    
                    return response_text
                    
                except Exception as e:
                    self.logger.warning(f"Final response generation failed: {str(e)}")
                    # Create a direct response using the search results
                    if readable_results:
                        response_text = f"Based on my search for '{actual_query}', I found:\n\n{readable_results}\n\nI couldn't generate a summary, but these are the relevant search results."
                    else:
                        response_text = f"I tried to search for information about '{actual_query}', but couldn't find relevant results or generate a summary. Could you try rephrasing your question?"
                    
                    console.print("\nAI:" + response_text, style="green")
                    return response_text
            else:
                # For non-real-time queries, use normal streaming
                # Make sure we're using original messages without any null content
                clean_messages = [m for m in messages if m.get("content") is not None]
                return self._handle_streaming_response(clean_messages)
                
        except Exception as e:
            self.logger.warning(f"Tool detection request failed: {str(e)}")
            # Make sure we're using original messages without any null content
            clean_messages = [m for m in messages if m.get("content") is not None]
            return self._handle_streaming_response(clean_messages)


class AnthropicAI(LLM):
    """
    AnthropicAI class for interacting with Anthropic's Claude models.
    
    This is a wrapper that uses the OpenAI-compatible API endpoint for Claude
    while abstracting away the provider-specific details.
    
    Attributes:
        provider (str): The name of the AI provider, set to 'anthropic'.
    
    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the AnthropicAI instance with the given configuration path and API key.
    """
    
    # Real-time query keywords - similar to other models
    real_time_keywords = [
        # Weather and natural phenomena
        "weather", "forecast", "temperature", "humidity", "precipitation", "rain", "snow", "storm", 
        "hurricane", "tornado", "earthquake", "tsunami", "typhoon", "cyclone", "flood", "drought", 
        "wildfire", "air quality", "pollen", "uv index", "sunrise", "sunset", "climate",
        
        # News and current events
        "news", "headline", "latest", "breaking", "current", "recent", "today", "yesterday",
        "this week", "this month", "ongoing", "developing", "situation", "event", "incident", 
        "announcement", "press release", "update", "coverage", "report", "bulletin", "fixture",
        
        # Sports and entertainment
        "score", "game", "match", "tournament", "championship", "playoff", "standings", 
        "leaderboard", "box office", "premiere", "release", "concert", "performance", 
        "episode", "ratings", "award", "nominations", "season", "show", "event",
        
        # Time-specific queries
        "now", "currently", "present", "moment", "tonight", "this morning", "this afternoon", 
        "this evening", "upcoming", "soon", "shortly", "imminent", "expected", "anticipated", 
        "scheduled", "real-time", "live", "happening", "occurring", "next"
    ]
    
    def __init__(self, config_path: str, api_key: str) -> None:
        """
        Initialize the AnthropicAI instance.
        
        Args:
            config_path (str): Path to YAML configuration file
            api_key (str): Anthropic API key
        """
        self.provider = 'anthropic'
        super().__init__(config_path, api_key)
        
        # For backward compatibility, we'll use the OpenAI client since 
        # we're assuming an OpenAI-compatible endpoint for Anthropic
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or "https://api.anthropic.com/v1"
        )
    
    def _prepare_standard_completion_params(self, messages):
        """
        Override to prepare parameters compatible with Anthropic's API requirements.
        
        Args:
            messages (list): Message objects.
            
        Returns:
            dict: Parameters for API request with Anthropic-specific adjustments.
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        
        # Only add stop parameter if it contains actual stop sequences
        if self.stop_sequences and len(self.stop_sequences) > 0:
            params["stop"] = self.stop_sequences
            
        return params