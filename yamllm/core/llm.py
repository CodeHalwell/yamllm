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
        api_key (str, optional): API key for the LLM service

    Examples:
        >>> llm = LLM("config.yaml")
        >>> llm.api_key = "your-api-key"
        >>> response = llm.query("Hello, world!")
    """
    def __init__(self, config_path: str, api_key: Optional[str] = None) -> None:
        """
        Initialize the LLM instance with the given configuration path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config: YamlLMConfig = self.load_config()
        self.logger = setup_logging(self.config)

        self.provider = self.config.provider.name
        self.api_key = api_key       
        self.model = self.config.provider.model
        self.temperature = self.config.model_settings.temperature
        self.max_tokens = self.config.model_settings.max_tokens
        self.top_p = self.config.model_settings.top_p
        self.frequency_penalty = self.config.model_settings.frequency_penalty
        self.presence_penalty = self.config.model_settings.presence_penalty
        self.stop_sequences = self.config.model_settings.stop_sequences
        self.system_prompt = self.config.context.system_prompt
        self.max_context_length = self.config.context.max_context_length
        self.memory_enabled = self.config.context.memory.enabled
        self.memory_max_messages = self.config.context.memory.max_messages
        self.output_format = self.config.output.format
        self.output_stream = self.config.output.stream
        self.tools_enabled = self.config.tools.enabled
        self.tools = self.config.tools.tools
        self.tools_timeout = self.config.tools.tool_timeout
        self.base_url = self.config.provider.base_url

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

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
        
    def find_similar_messages(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find messages similar to the query.

        Args:
            query (str): The text to find similar messages for.
            k (int): Number of similar messages to return. Default is 5.

        Returns:
            List[Dict[str, Any]]: List of similar messages with their metadata and similarity scores.
        """
        query_embedding = self.create_embedding(query)
        similar_messages = self.vector_store.search(query_embedding, k)
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
            # Check if a database is present and create it if not there
            if not self.memory.db_exists():
                self.memory.create_db()
                
            # First, find similar messages if we have previous conversations
            similar_messages = []
            try:
                similar_results = self.find_similar_messages(prompt, k=3)
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

        if self.provider == 'openai':  
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
            
        else: 
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    n=1,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )         
                response_text = response.choices[0].message.content.strip()

            except Exception as e:
                 raise Exception(f"Error getting response from the Google Endpoint: {str(e)}")


        # Create embeddings and store messages if memory is enabled
        if self.memory:
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

        return response_text



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
        Print the current settings of the LLM (Language Model).

        This method outputs the following settings:
        - Model: The model being used.
        - Temperature: The temperature setting for the model.
        - Max Tokens: The maximum number of tokens.
        - Top P: The top-p sampling parameter.
        - Frequency Penalty: The frequency penalty parameter.
        - Presence Penalty: The presence penalty parameter.
        - Stop Sequences: The stop sequences used by the model.
        - System Prompt: The system prompt from the configuration context.
        - Max Context Length: The maximum context length.
        - Memory Enabled: Whether memory is enabled.
        - Memory Max Messages: The maximum number of messages in memory.
        - Output Format: The format of the output.
        - Output Stream: The output stream.
        - Tools Enabled: Whether tools are enabled.
        - Tools: The tools available.
        - Tools Timeout: The timeout setting for tools.
        """
        print("LLM Settings:")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Top P: {self.top_p}")
        print(f"Frequency Penalty: {self.frequency_penalty}")
        print(f"Presence Penalty: {self.presence_penalty}")
        print(f"Stop Sequences: {self.stop_sequences}")
        print(f"System Prompt: {self.config.context.system_prompt}")
        print(f"Max Context Length: {self.max_context_length}")
        print(f"Memory Enabled: {self.memory_enabled}")
        print(f"Memory Max Messages: {self.memory_max_messages}")
        print(f"Output Format: {self.output_format}")
        print(f"Output Stream: {self.output_stream}")
        print(f"Tools Enabled: {self.tools_enabled}")
        print(f"Tools: {self.tools}")
        print(f"Tools Timeout: {self.tools_timeout}")
