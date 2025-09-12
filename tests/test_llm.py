import pytest
from unittest.mock import MagicMock, patch, call
from yamllm.core.llm import LLM
from openai import OpenAIError
import os

@pytest.fixture
def mock_llm():
    """Fixture to create a mock LLM instance."""
    mock_config = MagicMock()
    mock_config.provider.name = "openai"
    mock_config.provider.model = "gpt-3.5-turbo"
    mock_config.provider.base_url = "https://api.openai.com"
    mock_config.model_settings.temperature = 0.7
    mock_config.model_settings.max_tokens = 100
    mock_config.model_settings.top_p = 1.0
    mock_config.model_settings.frequency_penalty = 0.0
    mock_config.model_settings.presence_penalty = 0.0
    mock_config.model_settings.stop_sequences = None
    mock_config.output.stream = False
    
    # Add the missing logging configuration
    mock_config.logging.level = "INFO"
    mock_config.logging.file = "yamllm.log"
    mock_config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    mock_config.logging.rotate = True
    mock_config.logging.rotate_max_bytes = 1024
    mock_config.logging.rotate_backup_count = 3

    with patch("yamllm.core.llm.LLM.load_config", return_value=mock_config):
        llm = LLM(config_path="config.yaml", api_key=os.environ.get("OPENAI_API_KEY"))
        llm.get_response = MagicMock(return_value="Mocked response")
        yield llm

def test_query_success(mock_llm):
    """Test the query method when it succeeds."""
    prompt = "Hello, how are you?"
    response = mock_llm.query(prompt)
    assert response == "Mocked response"
    mock_llm.get_response.assert_called_once_with(prompt, None)

def test_query_with_system_prompt(mock_llm):
    """Test the query method with a system prompt."""
    prompt = "What is the weather today?"
    system_prompt = "You are a helpful assistant."
    response = mock_llm.query(prompt, system_prompt)
    assert response == "Mocked response"
    mock_llm.get_response.assert_called_once_with(prompt, system_prompt)

def test_query_missing_api_key():
    """Test the query method when the API key is missing."""
    mock_config = MagicMock()
    # Add logging configuration
    mock_config.logging.level = "INFO"
    mock_config.logging.file = "yamllm.log"
    mock_config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    mock_config.logging.rotate = True
    mock_config.logging.rotate_max_bytes = 1024
    mock_config.logging.rotate_backup_count = 3
    
    # Add required provider configuration
    mock_config.provider.name = "openai"
    mock_config.provider.model = "gpt-3.5-turbo"
    mock_config.provider.base_url = "https://api.openai.com"
    
    # Add required model settings
    mock_config.model_settings.temperature = 0.7
    mock_config.model_settings.max_tokens = 100
    mock_config.model_settings.top_p = 1.0
    mock_config.model_settings.frequency_penalty = 0.0
    mock_config.model_settings.presence_penalty = 0.0
    mock_config.model_settings.stop_sequences = None
    
    # Add required request settings
    mock_config.request.timeout = 30
    mock_config.request.retry.max_attempts = 3
    mock_config.request.retry.initial_delay = 1
    mock_config.request.retry.backoff_factor = 2
    
    # Add required context settings
    mock_config.context.system_prompt = "You are a helpful assistant."
    mock_config.context.max_context_length = 4000
    mock_config.context.memory.enabled = False
    mock_config.context.memory.max_messages = 10
    mock_config.context.memory.session_id = "test_session"
    mock_config.context.memory.conversation_db = "conversation.db"
    mock_config.context.memory.vector_store.index_path = "vector_store/index.faiss"
    mock_config.context.memory.vector_store.metadata_path = "vector_store/metadata.json"
    mock_config.context.memory.vector_store.top_k = 3
    
    # Add required output settings
    mock_config.output.format = "text"
    mock_config.output.stream = False
    
    # Add required tool settings
    mock_config.tools.enabled = False
    mock_config.tools.tools = []
    mock_config.tools.tool_timeout = 5
    
    # Add required safety settings
    mock_config.safety.content_filtering = True
    mock_config.safety.max_requests_per_minute = 60
    mock_config.safety.sensitive_keywords = []
    
    with patch("yamllm.core.llm.LLM.load_config", return_value=mock_config):
        # Mock the OpenAI client initialization to avoid API calls
        with patch("openai.OpenAI"):
            llm = LLM(config_path="config.yaml", api_key=None)
            with pytest.raises(ValueError, match="API key is not initialized or invalid."):
                llm.query("Test prompt")

def test_query_openai_error(mock_llm):
    """Test the query method when an OpenAIError occurs."""
    mock_llm.get_response.side_effect = OpenAIError("Mocked OpenAI error")
    with pytest.raises(Exception, match="OpenAI API error: Mocked OpenAI error"):
        mock_llm.query("Test prompt")

def test_query_unexpected_error(mock_llm):
    """Test the query method when an unexpected error occurs."""
    mock_llm.get_response.side_effect = Exception("Unexpected error")
    with pytest.raises(Exception, match="Unexpected error during query: Unexpected error"):
        mock_llm.query("Test prompt")

def test_make_api_call_retry_success():
    """Test that _make_api_call retries and succeeds after failures."""
    mock_config = MagicMock()
    # Add logging configuration
    mock_config.logging.level = "INFO"
    mock_config.logging.file = "yamllm.log"
    mock_config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    mock_config.logging.rotate = True
    mock_config.logging.rotate_max_bytes = 1024
    mock_config.logging.rotate_backup_count = 3
    
    # Add required provider configuration
    mock_config.provider.name = "openai"
    mock_config.provider.model = "gpt-3.5-turbo"
    mock_config.provider.base_url = "https://api.openai.com"
    
    # Add required model settings
    mock_config.model_settings.temperature = 0.7
    mock_config.model_settings.max_tokens = 100
    mock_config.model_settings.top_p = 1.0
    mock_config.model_settings.frequency_penalty = 0.0
    mock_config.model_settings.presence_penalty = 0.0
    mock_config.model_settings.stop_sequences = None
    
    # Add required request settings with retry configuration
    mock_config.request.timeout = 30
    mock_config.request.retry.max_attempts = 3
    mock_config.request.retry.initial_delay = 0.01  # Small value for test speed
    mock_config.request.retry.backoff_factor = 2
    
    # Add other required settings
    mock_config.context.system_prompt = "You are a helpful assistant."
    mock_config.context.max_context_length = 4000
    mock_config.context.memory.enabled = False
    mock_config.context.memory.max_messages = 10
    mock_config.context.memory.session_id = "test_session"
    mock_config.context.memory.conversation_db = "conversation.db"
    mock_config.context.memory.vector_store.index_path = "vector_store/index.faiss"
    mock_config.context.memory.vector_store.metadata_path = "vector_store/metadata.json"
    mock_config.context.memory.vector_store.top_k = 3
    mock_config.output.format = "text"
    mock_config.output.stream = False
    mock_config.tools.enabled = False
    mock_config.tools.tools = []
    mock_config.tools.tool_timeout = 5
    mock_config.safety.content_filtering = True
    mock_config.safety.max_requests_per_minute = 60
    mock_config.safety.sensitive_keywords = []
    
    with patch("yamllm.core.llm.LLM.load_config", return_value=mock_config):
        with patch("openai.OpenAI"), patch("yamllm.core.llm.time.sleep") as mock_sleep:
            llm = LLM(config_path="config.yaml", api_key="fake-api-key")
            
            # Create a mock function that fails twice then succeeds
            mock_api_func = MagicMock()
            mock_api_func.side_effect = [
                ConnectionError("Connection failed"),
                TimeoutError("Request timed out"),
                "Success response"
            ]
            
            result = llm._make_api_call(mock_api_func, arg1="value1", arg2="value2")
            
            # Check the function was called exactly 3 times
            assert mock_api_func.call_count == 3
            mock_api_func.assert_has_calls([
                call(arg1="value1", arg2="value2"),
                call(arg1="value1", arg2="value2"),
                call(arg1="value1", arg2="value2")
            ])
            
            # Verify sleep was called with exponential backoff
            mock_sleep.assert_has_calls([
                call(0.01),  # initial_delay * (backoff_factor ^ 0)
                call(0.02),  # initial_delay * (backoff_factor ^ 1)
            ])
            
            # Check we got the successful response
            assert result == "Success response"

def test_make_api_call_max_retries_exceeded():
    """Test that _make_api_call raises an exception when max retries are exceeded."""
    mock_config = MagicMock()
    # Add logging configuration
    mock_config.logging.level = "INFO"
    mock_config.logging.file = "yamllm.log"
    mock_config.logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    mock_config.logging.rotate = True
    mock_config.logging.rotate_max_bytes = 1024
    mock_config.logging.rotate_backup_count = 3
    
    # Add required provider configuration
    mock_config.provider.name = "openai"
    mock_config.provider.model = "gpt-3.5-turbo"
    mock_config.provider.base_url = "https://api.openai.com"
    
    # Add required model settings
    mock_config.model_settings.temperature = 0.7
    mock_config.model_settings.max_tokens = 100
    mock_config.model_settings.top_p = 1.0
    mock_config.model_settings.frequency_penalty = 0.0
    mock_config.model_settings.presence_penalty = 0.0
    mock_config.model_settings.stop_sequences = None
    
    # Add required request settings with retry configuration
    mock_config.request.timeout = 30
    mock_config.request.retry.max_attempts = 2  # Lower for test speed
    mock_config.request.retry.initial_delay = 0.01  # Small value for test speed
    mock_config.request.retry.backoff_factor = 2
    
    # Add other required settings
    mock_config.context.system_prompt = "You are a helpful assistant."
    mock_config.context.max_context_length = 4000
    mock_config.context.memory.enabled = False
    mock_config.context.memory.max_messages = 10
    mock_config.context.memory.session_id = "test_session"
    mock_config.context.memory.conversation_db = "conversation.db"
    mock_config.context.memory.vector_store.index_path = "vector_store/index.faiss"
    mock_config.context.memory.vector_store.metadata_path = "vector_store/metadata.json"
    mock_config.context.memory.vector_store.top_k = 3
    mock_config.output.format = "text"
    mock_config.output.stream = False
    mock_config.tools.enabled = False
    mock_config.tools.tools = []
    mock_config.tools.tool_timeout = 5
    mock_config.safety.content_filtering = True
    mock_config.safety.max_requests_per_minute = 60
    mock_config.safety.sensitive_keywords = []
    
    with patch("yamllm.core.llm.LLM.load_config", return_value=mock_config):
        with patch("openai.OpenAI"), patch("yamllm.core.llm.time.sleep") as mock_sleep:
            llm = LLM(config_path="config.yaml", api_key="fake-api-key")
            
            # Create a mock function that always fails
            error_message = "API rate limit exceeded"
            mock_api_func = MagicMock(side_effect=OpenAIError(error_message))
            
            with pytest.raises(OpenAIError, match=error_message):
                llm._make_api_call(mock_api_func)
            
            # Check the function was called exactly max_attempts (2) times
            assert mock_api_func.call_count == 2
            
            # Verify sleep was called once with correct backoff
            mock_sleep.assert_called_once_with(0.01)  # initial_delay * (backoff_factor ^ 0)
