from yamllm.core.parser import parse_yaml_config, YamlLMConfig
from yamllm.memory import ConversationStore, VectorStore
from openai import OpenAI, OpenAIError
from typing import Optional, Dict, Any
import os
from typing import List
import logging
import dotenv 

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
        self.conversation_db_path = self.config.context.memory.conversation_db
        self.vector_index_path = self.config.context.memory.vector_store.index_path
        self.vector_metadata_path = self.config.context.memory.vector_store.metadata_path
        self.vector_store_top_k = self.config.context.memory.vector_store.top_k


        # Output settings
        self.output_format = self.config.output.format
        self.output_stream = self.config.output.stream

        # Tool settings
        self.tools_enabled = self.config.tools.enabled
        self.tools = self.config.tools.tools
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
        Send a query to the language model and get the response.

        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (Optional[str]): An optional system prompt to provide context.

        Returns:
            str: The response from the language model.

        Raises:
            Exception: If there is an error getting the response from OpenAI.
        """        
        # Initialize memory if enabled
        if self.memory_enabled:
            self.memory = ConversationStore()
            self.vector_store = VectorStore()
            if not self.memory.db_exists():
                self.memory.create_db()
                
            # First, find similar messages if we have previous conversations
            similar_messages = []
            try:
                similar_results = self.find_similar_messages(prompt, self.vector_store_top_k)
                for result in similar_results:
                    similar_messages.append({
                        "role": result["role"],
                        "content": result["content"]
                    })
            except Exception:
                # If this is the first message, there won't be any similar messages
                pass
                
            # Get recent conversation history
            messages = self.memory.get_messages(
                session_id="session1", 
                limit=self.memory_max_messages
            )
        else:
            self.memory = None
            messages = []
            similar_messages = []
        
        # Add system prompt if provided
        if system_prompt or self.config.context.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.config.context.system_prompt
            })
        
        # Add context from similar messages if any exist
        if similar_messages:
            context_prompt = {
                "role": "system",
                "content": "Here are some relevant previous conversations:\n" + 
                        "\n".join([f"{m['role']}: {m['content']}" for m in similar_messages])
            }
            messages.append(context_prompt)
        
        # Add user message
        messages.append({"role": "user", "content": str(prompt)})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop_sequences or None
            )         
            response_text = response.choices[0].message.content.strip()

        except Exception as e:
                raise Exception(f"Error getting response from OpenAI: {str(e)}")

        # Handle memory storage if enabled
        if self.memory:
            self._store_memory(prompt, response_text)

        return response_text

    def _store_memory(self, prompt: str, response_text: str) -> None:
        """
        Helper method to store conversation in memory.
        
        Args:
            prompt (str): The user's input prompt
            response_text (str): The model's response text
        """
        # Store user message
        message_id = self.memory.add_message(
            session_id="session1", 
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
            session_id="session1", 
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



    def update_settings(self, **kwargs: Dict[str, Any]) -> None:
        """
        Update the settings of the instance with the provided keyword arguments.

        This method iterates over the provided keyword arguments and updates the 
        instance attributes if they exist.

        Args:
            **kwargs (Dict[str, Any]): Keyword arguments where the key is the 
            attribute name and the value is the new value for that attribute.
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
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = "openai"

class DeepSeek(LLM):
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = 'deepseek'

class MistralAI(LLM):
    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Override get_response to use only Mistral-supported parameters and message ordering.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (Optional[str]): An optional system prompt for context.
            
        Returns:
            str: The response from the Mistral language model.
            
        Raises:
            Exception: If there is an error getting the response from Mistral.
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
            self.memory = ConversationStore()
            self.vector_store = VectorStore()
            if not self.memory.db_exists():
                self.memory.create_db()
            
            # Get conversation history
            history = self.memory.get_messages(
                session_id="session1", 
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

        try:
            # Only use parameters supported by Mistral's API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )         
            response_text = response.choices[0].message.content.strip()

            # Handle memory storage if enabled
            if self.memory_enabled:
                self._store_memory(prompt, response_text)

            return response_text

        except Exception as e:
            raise Exception(f"Error getting response from Mistral: {str(e)}")
    
class GoogleGemini(LLM):
    def __init__(self, config_path: str, api_key: str) -> None:
        super().__init__(config_path, api_key)
        self.provider = 'google'

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Override get_response to use only Google-supported parameters.
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

        try:
            # Only use parameters supported by Google's API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                n=1
            )         
            response_text = response.choices[0].message.content.strip()

        except Exception as e:
             raise Exception(f"Error getting response from Google: {str(e)}")

        # Handle memory storage if enabled
        if self.memory:
            self._store_memory(prompt, response_text)

        return response_text
    