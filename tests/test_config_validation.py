from textwrap import dedent

import pytest

from yamllm.core.config_validator import ConfigValidator
from yamllm.core.exceptions import ConfigurationError
from yamllm.core.llm import LLM


def test_mask_sensitive_data_masks_keys():
    cfg = {
        "provider": {
            "name": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-thisisaverysecretapikeyvalue1234",
        },
        "nested": {
            "token": "tok_abcdef0123456789",
            "password": "hunter2password",
        },
    }

    masked = ConfigValidator.mask_sensitive_data(cfg)
    assert masked["provider"]["api_key"].startswith("sk-t")
    assert masked["provider"]["api_key"].endswith("1234")
    assert "*" in masked["provider"]["api_key"]
    assert masked["nested"]["token"].startswith("tok_") and masked["nested"]["token"].endswith("6789")
    assert "*" in masked["nested"]["token"]


def test_validator_flags_api_key_in_config():
    cfg = {
        "provider": {
            "name": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-thisisnotallowed",
        },
        "model_settings": {"temperature": 0.7, "max_tokens": 64, "top_p": 1.0, "stop_sequences": []},
        "request": {"timeout": 5, "retry": {"max_attempts": 1, "initial_delay": 1, "backoff_factor": 1}},
        "context": {"system_prompt": "x", "max_context_length": 128, "memory": {"enabled": False}},
        "output": {"format": "text", "stream": False},
        "logging": {"level": "ERROR", "file": "test.log", "format": "%(message)s", "console": False, "rotate": False},
        "safety": {"content_filtering": False, "max_requests_per_minute": 10, "sensitive_keywords": []},
        "tools": {"enabled": False, "tool_timeout": 5, "tools": [], "packs": [], "mcp_connectors": []},
        "embeddings": {"provider": None, "model": None},
    }

    errors = ConfigValidator.validate_config(cfg)
    assert any("API keys must not be stored in configuration files" in e for e in errors)


def test_llm_load_config_raises_configuration_error_on_invalid(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        dedent(
            """
            provider:
              name: openai
              model: gpt-4o-mini
              api_key: bad_key
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
            """
        ).strip()
    )

    with pytest.raises(ConfigurationError) as ei:
        # Error is raised during load_config before provider init
        LLM(config_path=str(cfg), api_key="dummy")

    assert "Invalid configuration:" in str(ei.value)
    assert "API keys must not be stored in configuration files" in str(ei.value)

