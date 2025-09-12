from textwrap import dedent


from yamllm.providers.capabilities import (
    get_provider_capabilities,
    PROVIDER_CAPABILITIES,
)
from yamllm.core.llm import LLM
from yamllm.providers.factory import ProviderFactory


def test_capability_map_contains_known_providers():
    caps_openai = get_provider_capabilities("openai")
    assert caps_openai.supports_tools is True
    assert caps_openai.supports_streaming is True
    assert caps_openai.supports_embeddings is True
    assert caps_openai.tool_calling_format == "openai"

    caps_anthropic = get_provider_capabilities("Anthropic")  # case-insensitive
    assert caps_anthropic.supports_tools is True
    assert caps_anthropic.supports_streaming is True
    assert caps_anthropic.tool_calling_format == "anthropic"

    # Unknown providers return conservative defaults (no tools)
    caps_unknown = get_provider_capabilities("does_not_exist")
    assert caps_unknown.supports_tools is False
    assert caps_unknown.supports_streaming is False


def test_llm_prepare_tools_disabled_if_provider_lacks_tool_support(tmp_path):
    # Write a minimal valid YAML config
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        dedent(
            """
            provider:
              name: openai
              model: gpt-4o-mini
              base_url: https://api.openai.com/v1
            model_settings:
              temperature: 0.0
              max_tokens: 16
              top_p: 1.0
              stop_sequences: []
            request:
              timeout: 5
              retry:
                max_attempts: 1
                initial_delay: 1
                backoff_factor: 1
            context:
              system_prompt: You are a helpful assistant.
              max_context_length: 1024
              memory:
                enabled: false
                max_messages: 0
                session_id: null
                conversation_db: null
                vector_store:
                  index_path: null
                  metadata_path: null
                  top_k: 0
            output:
              format: text
              stream: false
            logging:
              level: ERROR
              file: test.log
              format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
              console: false
              rotate: false
            safety:
              content_filtering: false
              max_requests_per_minute: 100
              sensitive_keywords: []
            tools:
              enabled: true
              tool_timeout: 5
              tools: []
              packs: []
              mcp_connectors: []
            embeddings:
              provider: null
              model: null
            """
        ).strip()
    )

    # Define a subclass that forces an unknown provider name for capability lookup
    class DummyNoToolsLLM(LLM):
        def __init__(self, config_path: str, api_key: str) -> None:
            self.provider = "unknown_no_tools"
            super().__init__(config_path, api_key)

    llm = DummyNoToolsLLM(config_path=str(cfg), api_key="test-key")

    # Even with tools enabled in config, capability gating should disable tools
    tools_param = llm._prepare_tools()
    assert tools_param == []


def test_every_factory_provider_has_capabilities_entry():
    for name in ProviderFactory._MAP.keys():  # type: ignore[attr-defined]
        assert name in PROVIDER_CAPABILITIES, f"Missing capabilities for provider '{name}'"
