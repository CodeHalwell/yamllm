import pytest
import logging
from unittest.mock import MagicMock, patch
from yamllm.core.llm import setup_logging, LLM
from yamllm.core.parser import YamlLMConfig

class MockConfig:
    class Logging:
        level = 'DEBUG'
        file = 'test.log'
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging = Logging()

@pytest.fixture
def mock_config():
    return MockConfig()

def test_setup_logging(mock_config):
    logger = setup_logging(mock_config)
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.FileHandler)
    assert logger.handlers[0].formatter._fmt == mock_config.logging.format

@pytest.fixture
def mock_llm_config():
    config = MagicMock(spec=YamlLMConfig)
    config.provider.name = 'openai'
    config.provider.model = 'text-davinci-003'
    config.model_settings.temperature = 0.7
    config.model_settings.max_tokens = 100
    config.model_settings.top_p = 0.9
    config.model_settings.frequency_penalty = 0.5
    config.model_settings.presence_penalty = 0.5
    config.model_settings.stop_sequences = None
    config.context.system_prompt = 'You are a helpful assistant.'
    config.context.max_context_length = 2048
    config.context.memory.enabled = True
    config.context.memory.max_messages = 10
    config.output.format = 'text'
    config.output.stream = False
    config.tools.enabled = False
    config.tools.tools = []
    config.tools.tool_timeout = 5
    config.provider.base_url = 'https://api.openai.com/v1'
    return config

@patch('yamllm.core.llm.parse_yaml_config')
@patch('yamllm.core.llm.OpenAI')
@patch('yamllm.core.llm.setup_logging')
def test_llm_initialization(mock_setup_logging, mock_openai, mock_parse_yaml_config, mock_llm_config):
    mock_parse_yaml_config.return_value = mock_llm_config
    mock_setup_logging.return_value = MagicMock()

    llm = LLM(config_path='config.yaml', api_key='test-api-key')

    assert llm.config_path == 'config.yaml'
    assert llm.api_key == 'test-api-key'
    assert llm.config == mock_llm_config
    assert llm.logger == mock_setup_logging.return_value
    assert llm.provider == 'openai'
    assert llm.model == 'text-davinci-003'
    assert llm.temperature == 0.7
    assert llm.max_tokens == 100
    assert llm.top_p == 0.9
    assert llm.frequency_penalty == 0.5
    assert llm.presence_penalty == 0.5
    assert llm.stop_sequences is None
    assert llm.system_prompt == 'You are a helpful assistant.'
    assert llm.max_context_length == 2048
    assert llm.memory_enabled is True
    assert llm.memory_max_messages == 10
    assert llm.output_format == 'text'
    assert llm.output_stream is False
    assert llm.tools_enabled is False
    assert llm.tools == []
    assert llm.tools_timeout == 5
    assert llm.base_url == 'https://api.openai.com/v1'
    assert llm.client == mock_openai.return_value
    assert llm.embedding_client == mock_openai.return_value

@patch('yamllm.core.llm.OpenAI')
def test_create_embedding(mock_openai, mock_llm_config):
    mock_openai.return_value.embeddings.create.return_value.data = [{'embedding': b'test-embedding'}]
    llm = LLM(config_path='config.yaml', api_key='test-api-key')
    llm.config = mock_llm_config
    llm.embedding_client = mock_openai.return_value

    embedding = llm.create_embedding('test text')
    assert embedding == b'test-embedding'

@patch('yamllm.core.llm.OpenAI')
def test_find_similar_messages(mock_openai, mock_llm_config):
    llm = LLM(config_path='config.yaml', api_key='test-api-key')
    llm.config = mock_llm_config
    llm.vector_store = MagicMock()
    llm.vector_store.search.return_value = [{'role': 'user', 'content': 'test message'}]
    llm.create_embedding = MagicMock(return_value=b'test-embedding')

    similar_messages = llm.find_similar_messages('test query')
    assert len(similar_messages) == 1
    assert similar_messages[0]['role'] == 'user'
    assert similar_messages[0]['content'] == 'test message'

@patch('yamllm.core.llm.OpenAI')
@patch('yamllm.core.llm.ConversationStore')
@patch('yamllm.core.llm.VectorStore')
def test_query(mock_vector_store, mock_conversation_store, mock_openai, mock_llm_config):
    mock_openai.return_value.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content='test response'))]
    llm = LLM(config_path='config.yaml', api_key='test-api-key')
    llm.config = mock_llm_config
    llm.memory = mock_conversation_store.return_value
    llm.vector_store = mock_vector_store.return_value
    llm.memory.get_messages.return_value = []
    llm.find_similar_messages = MagicMock(return_value=[])

    response = llm.query('test prompt')
    assert response == 'test response'