from textwrap import dedent

from types import SimpleNamespace

from yamllm.core.llm import LLM


class StubProvider:
    def get_completion(self, **kwargs):
        # Return an OpenAI-like response object
        msg = SimpleNamespace(content="stubbed")
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])


def test_get_response_with_thinking_includes_thinking(tmp_path):
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
            output:
              format: text
              stream: false
            logging:
              level: ERROR
              file: test.log
              format: '%(message)s'
              console: false
              rotate: false
            safety:
              content_filtering: false
              max_requests_per_minute: 100
              sensitive_keywords: []
            tools:
              enabled: false
              tool_timeout: 5
              tools: []
              packs: []
              mcp_connectors: []
            embeddings:
              provider: null
              model: null
            thinking:
              enabled: true
              show_tool_reasoning: true
              model: gpt-4o-mini
              max_tokens: 32
              stream_thinking: false
              save_thinking: false
              thinking_temperature: 0.1
            """
        ).strip()
    )

    llm = LLM(config_path=str(cfg), api_key="test")
    llm.provider_client = StubProvider()

    out = llm.get_response_with_thinking("hello")
    assert isinstance(out, dict)
    assert out.get("thinking") and "<thinking>" in out["thinking"]
    assert isinstance(out.get("response"), (str, type(None)))

