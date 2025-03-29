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
    with language models.

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
        self.provider = self.config.provider.name
        self.model = self.config.provider.model
        self.base_url = self.config.provider.base_url

        # Model settings
        self.temperature = self.config.model_settings.temperature
        self.max_tokens = self.config.model_settings.max_tokens
        self.top_p = self.config.model_settings.top_p
        self.frequency_penalty = self.config.model_settings.frequency_penalty
        self.presence_penalty = self.config.model_settings.presence_penalty
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
            api_key=os.environ["OPENAI_API_KEY"],
        )

        self.memory = None
        self.vector_store = None
        if self.memory_enabled:
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
            raise Exception(f"Error creating embedding: {str(e)}")       
        

    def find_similar_messages(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Find messages similar to the query.

        Args:
            query (str): The text to find similar messages for.
            k (int): Number of similar messages to return. Default is 5.

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
            Exception: For any other unexpected errors.
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
            raise Exception(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during query: {str(e)}")

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a response from the language model based on the provided prompt and optional system prompt.
        Supports tool calling if tools_enabled is True.
        
        Args:
            prompt (str): The user's input prompt to generate a response for.
            system_prompt (Optional[str], optional): An optional system prompt to provide context or instructions to the model. Defaults to None.
        
        Returns:
            str: The generated response from the language model if output_stream is disabled.
            None: If output_stream is enabled, the response is streamed and displayed in real-time.
        
        Raises:
            Exception: If there is an error getting a response from the language model.
        """
        messages = []
        
        # Start with system prompt if provided (must be first)
        if system_prompt or self.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.system_prompt
            })
        
        # Initialize memory if enabled
        if self.memory_enabled:           
            # Get conversation history
            history = self.memory.get_messages(
                session_id=self.session_id, 
                limit=self.memory_max_messages
            )
            
            # Add history messages maintaining strict user-assistant alternation
            messages.extend(history)
            
            # Find and format similar messages as part of the user's prompt
            similar_context = ""
            try:
                similar_results = self.find_similar_messages(prompt, k=2)
                if similar_results:
                    similar_context = "\nRelevant context from previous conversations:\n" + \
                        "\n".join([f"{m['role']}: {m['content']}" for m in similar_results])
            except Exception:
                pass
            
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

        # Prepare tools if enabled
        tools_param = None
        if self.tools_enabled and self.tools:
            tools_param = self._prepare_tools()

        # Process response based on stream setting and tool capabilities
        if self.output_stream and not tools_param:
            # Standard streaming without tools
            response_text = self._handle_streaming_response(messages)
        else:
            # Either non-streaming or streaming with tools
            response_text = self._handle_non_streaming_response(messages, tools_param)

        # Store the conversation in memory
        self._store_memory(prompt, response_text, self.session_id)

        if self.output_stream:
            return None
        else:
            return response_text
    
    def _prepare_tools(self):
        """
        Prepare tool definitions for the API call.
        
        Returns:
            list: List of tool definitions in the format expected by the OpenAI API
        """
        from yamllm.tools.utility_tools import WebSearch, Calculator, TimezoneTool, UnitConverter
        
        # Map tool names to their respective classes
        tool_classes = {
            "web_search": WebSearch,
            "calculator": Calculator,
            "timezone": TimezoneTool,
            "unit_converter": UnitConverter
        }
        
        tool_definitions = []
        
        for tool_name in self.tools:
            if tool_name not in tool_classes:
                self.logger.warning(f"Tool '{tool_name}' not found in available tools")
                continue
                
            tool_class = tool_classes[tool_name]
            
            # Create a dummy instance to get name and description
            # WebSearch doesn't need an API key for DuckDuckGo
            if tool_name == "web_search":
                tool_instance = tool_class()  # No need for API key with DuckDuckGo
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
            }
        }
        
        return tool_parameters.get(tool_name, {"type": "object", "properties": {}})

    def _handle_streaming_response(self, messages):
        """
        Handle streaming response from the model.
        
        Args:
            messages (list): List of message objects
            
        Returns:
            str: Concatenated response text
            
        Raises:
            Exception: If there is an error getting a response from the model
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop_sequences or None,
                stream=True
            )
            console = Console()
            response_text = ""
            print()
            
            with Live(console=console, refresh_per_second=10) as live:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        # Add a newline before the AI response
                        md = Markdown(f"\nAI: {response_text}", style="green")
                        live.update(md)

            return response_text
        except Exception as e:
            raise Exception(f"Error getting streaming response: {str(e)}")

    def _handle_non_streaming_response(self, messages, tools_param=None):
        """
        Handle non-streaming response from the model, with optional tool calling.
        """
        try:            
            # Prepare API call parameters
            completion_params = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stop": self.stop_sequences or None
            }
            
            # Add tools if available
            if tools_param:
                completion_params["tools"] = tools_param
                completion_params["tool_choice"] = "auto"  # Let the model decide when to use tools
                
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
            import traceback
            print(f"Error in _handle_non_streaming_response: {str(e)}")
            print(traceback.format_exc())
            raise Exception(f"Error getting non-streaming response: {str(e)}")

    def _process_tool_calls(self, messages, model_message, max_iterations=5):
        """
        Process tool calls from the model and get the final response.
        """
        if max_iterations <= 0:
            return "Maximum tool call iterations reached. Unable to complete the request."
            
        console = Console()
        console.print("\n[bold yellow]Tool Call Requested:[/bold yellow]")
        
        # Add the assistant's message with tool calls to conversation
        messages.append({
            "role": "assistant",
            "content": model_message.content or "",
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
        
        # Get the next response from the model with explicit instructions for natural language
        try:
            next_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop_sequences or None
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
        from yamllm.tools.utility_tools import WebSearch, Calculator, TimezoneTool, UnitConverter
        
        # Timeout handling for tool execution
        import concurrent.futures
        import threading
        
        # Map tool names to their respective classes
        tool_classes = {
            "web_search": WebSearch,
            "calculator": Calculator,
            "timezone": TimezoneTool,
            "unit_converter": UnitConverter
        }
        
        if tool_name not in tool_classes:
            return f"Error: Tool '{tool_name}' not found"
            
        # Initialize the tool
        try:
            # No special case needed for WebSearch anymore
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
        """Store the conversation in memory.
        
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

        This method iterates over the provided keyword arguments and updates the 
        instance attributes if they exist.

        Args:
            **kwargs (Dict[str, Any]): Keyword arguments where the key is the 
            attribute name and the value is the new value for that attribute.

        Example:
            >>> llm.update_settings(temperature=0.8)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def print_settings(self) -> None:
        """
        Print the current settings of the LLM (Language Model) in an organized format.
        Settings are grouped by category for better readability.
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
                "Frequency Penalty": self.frequency_penalty,
                "Presence Penalty": self.presence_penalty,
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
    """    MistralAI class for interacting with the Mistral language model.
        Attributes:
            provider (str): The name of the AI provider, set to 'mistral'.
        Methods:
            __init__(config_path: str, api_key: str) -> None:
                Initializes the MistralAI instance with the given configuration path and API key.
            get_response(prompt: str, system_prompt: Optional[str] = None) -> str:
                Generates a response from the Mistral language model based on the given prompt and optional system prompt.
                Parameters:
                    prompt (str): The user input prompt to generate a response for.
                    system_prompt (Optional[str]): An optional system prompt to provide context for the response.
                Returns:
                    str: The generated response from the Mistral language model."""
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = 'mistral'

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response based on the given prompt and optional system prompt.
        This method overrides the base `get_response` method to use only Mistral-supported 
        parameters and message ordering. It supports memory initialization, conversation 
        history retrieval, and finding similar messages to provide relevant context.
        
        Parameters:
        - prompt (str): The user prompt to generate a response for.
        - system_prompt (Optional[str]): An optional system prompt to provide context.
        
        Returns:
        - str: The generated response text if `output_stream` is False, otherwise None.
        
        Raises:
        - Exception: If there is an error getting a response from Mistral."""

        messages = []
        
        # Start with system prompt if provided
        if system_prompt or self.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.system_prompt
            })

        # Initialize memory if enabled
        if self.memory_enabled:
            self.memory = ConversationStore()
            self.vector_store = VectorStore()
            if not self.memory.db_exists():
                self.memory.create_db()
                
            # Get conversation history
            history = self.memory.get_messages(
                session_id="session1", 
                limit=self.memory_max_messages
            )
            messages.extend(history)
            
            # Find similar messages and add as context to the prompt
            try:
                similar_results = self.find_similar_messages(prompt, k=2)
                if similar_results:
                    similar_context = "\nRelevant context:\n" + \
                        "\n".join([f"{m['role']}: {m['content']}" for m in similar_results])
                    prompt = f"{prompt}\n{similar_context}"
            except Exception:
                pass

        # Add the current prompt
        messages.append({"role": "user", "content": str(prompt)})

        if self.output_stream:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stream=True
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

                if self.memory_enabled:
                    self._store_memory(prompt, response_text)
                return None

            except Exception as e:
                raise Exception(f"Error getting response from Mistral: {str(e)}")
            
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )         
                response_text = response.choices[0].message.content
                
                console = Console()
                if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                    md = Markdown("\nAI:" + response_text, style="green")
                    console.print(md)
                else:
                    console.print("\nAI:" + response_text, style="green")

            except Exception as e:
                raise Exception(f"Error getting response from Mistral: {str(e)}")
            
        if self.memory_enabled:
            self._store_memory(prompt, response_text)

        return response_text if not self.output_stream else None
    
class GoogleGemini(LLM):
    """GoogleGemini is a subclass of LLM that interacts with Google's language model to generate responses based on given prompts.
        
        Attributes:
        provider (str): The provider of the language model, set to 'google'.
        
        Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the GoogleGemini instance with the given configuration path and API key.
        
        get_response(prompt: str, system_prompt: Optional[str] = None) -> str:
        
        Generates a response from the language model based on the given prompt and optional system prompt.
                str: The response from the language model.
                Exception: If there is an error getting the response from the language model."""
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = 'google'

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Override get_response to use only Google-supported parameters and message ordering.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (Optional[str]): An optional system prompt for context.
            
        Returns:
            str: The response from the Google language model.
            
        Raises:
            Exception: If there is an error getting the response from Google.
        """
        if self.memory_enabled:
            self.memory = ConversationStore()
            self.vector_store = VectorStore()
            if not self.memory.db_exists():
                self.memory.create_db()
                
            similar_messages = []
            try:
                similar_results = self.find_similar_messages(prompt, k=2)
                for result in similar_results:
                    similar_messages.append({
                        "role": result["role"],
                        "content": result["content"]
                    })
            except Exception:
                pass
                
            messages = self.memory.get_messages(
                session_id="session1", 
                limit=self.memory_max_messages
            )
        else:
            self.memory = None
            messages = []
            similar_messages = []
        
        if system_prompt or self.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.system_prompt
            })
        
        if similar_messages:
            context_prompt = {
                "role": "system",
                "content": "Here are some relevant previous conversations:\n" + 
                        "\n".join([f"{m['role']}: {m['content']}" for m in similar_messages])
            }
            messages.append(context_prompt)
        
        messages.append({"role": "user", "content": str(prompt)})

        if self.output_stream:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=1,
                    stream=True
                )
                console = Console()
                response_text = ""
                print()
                
                with Live(console=console, refresh_per_second=10) as live:
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content
                            # Add a newline before the AI response
                            md = Markdown(f"\nAI: {response_text}", style="green")
                            live.update(md)

                if self.memory:
                    self._store_memory(prompt, response_text)
                    
                return None

            except Exception as e:
                raise Exception(f"Error getting response from Google: {str(e)}")
            
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    n=1
                )         
                response_text = response.choices[0].message.content
                
                # Add Rich console formatting
                console = Console()
                
                # Handle markdown formatting if present
                if any(marker in response_text for marker in ['###', '```', '*', '_', '-']):
                    md = Markdown("\nAI:" + response_text, style="green")
                    console.print(md)
                else:
                    console.print("\nAI:" + response_text, style="green")

            except Exception as e:
                raise Exception(f"Error getting response from Google: {str(e)}")
            
        self._store_memory(prompt, response_text)

        if self.output_stream:
            return None
        else:
            return response_text
    