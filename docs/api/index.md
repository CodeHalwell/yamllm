[← Back to Index](../index.md)

# API Reference

## Core Classes

### LLM Base Class

The foundational class for all LLM providers:

```python
from yamllm.core.llm import LLM

class LLM:
    def __init__(self, config_path: str, api_key: str):
        """
        Initialize LLM with configuration.

        Args:
            config_path (str): Path to YAML configuration file
            api_key (str): API key for the LLM service
        """

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a query to the language model.

        Args:
            prompt (str): The prompt to send
            system_prompt (Optional[str]): Optional system context

        Returns:
            str: Model response
        """

    def update_settings(self, **kwargs: Dict[str, Any]) -> None:
        """Update configuration settings at runtime."""

    def print_settings(self) -> None:
        """Print the current settings of the LLM in an organized format."""

    def create_embedding(self, text: str) -> bytes:
        """
        Create an embedding for the given text using OpenAI's API.

        Args:
            text (str): The text to create an embedding for.

        Returns:
            bytes: The embedding as bytes.
        """

    def find_similar_messages(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Find messages similar to the query.

        Args:
            query (str): The text to find similar messages for.
            k (int): Number of similar messages to return.

        Returns:
            List[Dict[str, Any]]: List of similar messages with their metadata and similarity scores.
        """

    def load_config(self) -> YamlLMConfig:
        """
        Load configuration from YAML file.

        Returns:
            YamlLMConfig: Parsed configuration.
        """

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates a response from the language model based on the provided prompt and optional system prompt.

        Args:
            prompt (str): The user's input prompt to generate a response for.
            system_prompt (Optional[str], optional): An optional system prompt to provide context or instructions to the model.

        Returns:
            str: The generated response from the language model if output_stream is disabled.
            None: If output_stream is enabled, the response is streamed and displayed in real-time.
        """

    def _store_memory(self, prompt: str, response_text: str) -> None:
        """Store the conversation in memory."""

    def __repr__(self) -> str:
        """Return a detailed string representation of the LLM instance."""

    def __str__(self) -> str:
        """Return a human-readable string representation of the LLM instance."""

    def __enter__(self):
        """Support context manager interface."""

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up resources when exiting context manager."""

    def __bool__(self) -> bool:
        """Return True if the LLM instance is properly initialized with an API key."""
```

### Provider-Specific Classes

#### OpenAIGPT

```python
from yamllm.core.llm import OpenAIGPT

class OpenAIGPT(LLM):
    """OpenAI GPT model implementation."""
    
    def __init__(self, config_path: str, api_key: str):
        """
        Initialize OpenAI GPT client.

        Args:
            config_path (str): Path to config file
            api_key (str): OpenAI API key
        """
```

#### GoogleGemini

```python
from yamllm.core.llm import GoogleGemini

class GoogleGemini(LLM):
    """Google Gemini model implementation."""
    
    def __init__(self, config_path: str, api_key: str):
        """
        Initialize Google Gemini client.

        Args:
            config_path (str): Path to config file
            api_key (str): Google API key
        """
```

#### DeepSeek

```python
from yamllm.core.llm import DeepSeek

class DeepSeek(LLM):
    """DeepSeek model implementation."""
    
    def __init__(self, config_path: str, api_key: str):
        """
        Initialize DeepSeek client.

        Args:
            config_path (str): Path to config file
            api_key (str): DeepSeek API key
        """
```

#### MistralAI

```python
from yamllm.core.llm import MistralAI

class MistralAI(LLM):
    """MistralAI model implementation."""
    
    def __init__(self, config_path: str, api_key: str):
        """
        Initialize MistralAI client.

        Args:
            config_path (str): Path to config file
            api_key (str): Mistral API key
        """
```

## Memory Management

### ConversationStore

```python
from yamllm import ConversationStore

class ConversationStore:
    """SQLite-based conversation history manager."""
    
    def __init__(self, db_path: str = "yamllm/memory/conversation_history.db"):
        """
        Initialize conversation store.

        Args:
            db_path (str): Path to SQLite database
        """
    
    def add_message(self, session_id: str, role: str, content: str) -> int:
        """
        Add a message to history.

        Args:
            session_id (str): Conversation session ID
            role (str): Message role (user/assistant)
            content (str): Message content

        Returns:
            int: Message ID
        """

    def get_messages(self, session_id: str = None, limit: int = None) -> List[Dict[str, str]]:
        """
        Retrieve conversation history.

        Args:
            session_id (str): Optional session filter
            limit (int): Optional message limit

        Returns:
            List[Dict[str, str]]: Message history
        """
```

### VectorStore

```python
from yamllm import VectorStore

class VectorStore:
    """FAISS-based vector storage for semantic search."""
    
    def __init__(self, vector_dim: int = 1536, store_path: str = "yamllm/memory/vector_store"):
        """
        Initialize vector store.

        Args:
            vector_dim (int): Embedding dimension
            store_path (str): Path to store files
        """

    def add_vector(self, vector: List[float], message_id: int, content: str, role: str) -> None:
        """
        Add vector to store.

        Args:
            vector (List[float]): Embedding vector
            message_id (int): Message reference ID
            content (str): Message content
            role (str): Message role
        """

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search similar vectors.

        Args:
            query_vector (List[float]): Search vector
            k (int): Number of results

        Returns:
            List[Dict[str, Any]]: Similar messages
        """
```

## Configuration Schema

Expected YAML configuration structure:

```yaml
provider:
  name: str          # Provider name (openai/google/deepseek/mistral)
  model: str         # Model identifier
  api_key: str       # API key (use env vars)
  base_url: str      # Optional API endpoint

model_settings:
  temperature: float     # Response randomness (0.0-1.0)
  max_tokens: int       # Maximum response length
  top_p: float         # Nucleus sampling parameter
  frequency_penalty: float
  presence_penalty: float
  stop_sequences: list

request:
  timeout: int      # Request timeout seconds
  retry:
    max_attempts: int
    initial_delay: int
    backoff_factor: int

context:
  system_prompt: str    # System context
  max_context_length: int
  memory:
    enabled: bool
    max_messages: int
    session_id: str
    conversation_db: str
    vector_store:
      index_path: str
      metadata_path: str
      top_k: int

output:
  format: str      # text/json/markdown
  stream: bool     # Enable streaming

tools:
  enabled: bool
  tool_timeout: int
  tool_list: list

safety:
  content_filtering: bool
  max_requests_per_minute: int
  sensitive_keywords: list
```

## Usage Examples

See the [Examples](../examples.md) page for detailed usage examples.