from unittest.mock import MagicMock, patch

from yamllm.core.llm import LLM


def _fake_openai_chunk(delta_text: str):
    choice = MagicMock()
    choice.delta.content = delta_text
    choice.finish_reason = None
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def test_llm_streaming_callback_invoked():
    mock_config = MagicMock()
    # minimal required config fields
    mock_config.provider.name = "openai"
    mock_config.provider.model = "gpt-x"
    mock_config.provider.base_url = "https://api.openai.com"
    mock_config.model_settings.temperature = 0.1
    mock_config.model_settings.max_tokens = 10
    mock_config.model_settings.top_p = 1.0
    mock_config.model_settings.stop_sequences = []
    mock_config.request.timeout = 30
    mock_config.request.retry.max_attempts = 1
    mock_config.request.retry.initial_delay = 0
    mock_config.request.retry.backoff_factor = 1
    mock_config.context.system_prompt = ""
    mock_config.context.max_context_length = 1000
    mock_config.context.memory.enabled = False
    mock_config.context.memory.max_messages = 5
    mock_config.context.memory.session_id = "s"
    mock_config.context.memory.conversation_db = "memory/conversation_history.db"
    mock_config.context.memory.vector_store.index_path = "memory/vector_store/faiss_index.idx"
    mock_config.context.memory.vector_store.metadata_path = "memory/vector_store/metadata.pkl"
    mock_config.context.memory.vector_store.top_k = 2
    mock_config.output.format = "text"
    mock_config.output.stream = True
    mock_config.logging.level = "INFO"
    mock_config.logging.file = "yamllm.log"
    mock_config.logging.format = "%(message)s"
    mock_config.logging.rotate = True
    mock_config.logging.rotate_max_bytes = 1024
    mock_config.logging.rotate_backup_count = 3
    mock_config.logging.console = False
    mock_config.tools.enabled = False
    mock_config.tools.tool_list = []
    mock_config.tools.tool_timeout = 1
    mock_config.safety.content_filtering = False
    mock_config.safety.max_requests_per_minute = 100
    mock_config.safety.sensitive_keywords = []

    with patch("yamllm.core.llm.LLM.load_config", return_value=mock_config):
        llm = LLM(config_path="cfg.yaml", api_key="test")

        # Replace provider client with a stub that yields fake chunks
        class StubProvider:
            def get_streaming_completion(self, **kwargs):
                chunks = [
                    _fake_openai_chunk("Hello"),
                    _fake_openai_chunk(", world"),
                    _fake_openai_chunk("!"),
                ]
                for c in chunks:
                    yield c

        llm.provider_client = StubProvider()

        collected = []
        llm.set_stream_callback(lambda d: collected.append(d))

        result = llm.query("Say hi")

        assert result is None  # streaming mode returns None
        assert "".join(collected) == "Hello, world!"
