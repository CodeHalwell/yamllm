import unittest
from yamllm.parser import parse_yaml, extract_items
import tempfile
import os
from yamllm.core.parser import parse_yaml_config, YamlLMConfig
import yaml

class TestParser(unittest.TestCase):

    def test_parse_yaml(self):
        # Test parsing a valid YAML file
        yaml_content = """
        key1: value1
        key2:
          - item1
          - item2
        """
        result = parse_yaml(yaml_content)
        expected = {'key1': 'value1', 'key2': ['item1', 'item2']}
        self.assertEqual(result, expected)

    def test_extract_items(self):
        # Test extracting items from a parsed YAML dictionary
        yaml_dict = {'key1': 'value1', 'key2': ['item1', 'item2']}
        result = extract_items(yaml_dict)
        expected = ['value1', 'item1', 'item2']
        self.assertEqual(result, expected)

class TestParserConfig(unittest.TestCase):

    def setUp(self):
        self.valid_yaml_content = """
        provider:
          name: "provider_name"
          model: "model_name"
          api_key: null
          base_url: "http://example.com"
        model_settings:
          temperature: 0.7
          max_tokens: 1000
          top_p: 1.0
          frequency_penalty: null
          presence_penalty: null
          stop_sequences: []
        request:
          timeout: 30
          retry:
            max_attempts: 3
            initial_delay: 1
            backoff_factor: 2
        context:
          system_prompt: "You are a helpful assistant."
          max_context_length: 4096
          memory:
            enabled: False
            max_messages: 10
        output:
          format: "text"
          stream: False
        logging:
          level: "INFO"
          file: "yamllm.log"
          format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        safety:
          content_filtering: True
          max_requests_per_minute: 60
          sensitive_keywords: []
        tools:
          enabled: False
          tools: []
          tool_timeout: 30
        """

    def test_parse_yaml_config_valid(self):
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_file.write(self.valid_yaml_content)
            temp_file_path = temp_file.name

        try:
            config = parse_yaml_config(temp_file_path)
            self.assertIsInstance(config, YamlLMConfig)
            self.assertEqual(config.provider.name, "provider_name")
            self.assertEqual(config.provider.model, "model_name")
            self.assertEqual(config.provider.base_url, "http://example.com")
            self.assertEqual(config.model_settings.temperature, 0.7)
            self.assertEqual(config.request.timeout, 30)
            self.assertEqual(config.context.system_prompt, "You are a helpful assistant.")
            self.assertEqual(config.output.format, "text")
            self.assertEqual(config.logging.level, "INFO")
            self.assertTrue(config.safety.content_filtering)
            self.assertFalse(config.tools.enabled)
        finally:
            os.remove(temp_file_path)

    def test_parse_yaml_config_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_yaml_config("non_existent_file.yaml")

    def test_parse_yaml_config_yaml_error(self):
        invalid_yaml_content = """
        provider:
          name: "provider_name"
          model: "model_name"
          api_key: null
          base_url: "http://example.com
        """
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_file.write(invalid_yaml_content)
            temp_file_path = temp_file.name

        try:
            with self.assertRaises(yaml.YAMLError):
                parse_yaml_config(temp_file_path)
        finally:
            os.remove(temp_file_path)

    def test_parse_yaml_config_empty_file(self):
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_file.write("")
            temp_file_path = temp_file.name

        try:
            with self.assertRaises(ValueError):
                parse_yaml_config(temp_file_path)
        finally:
            os.remove(temp_file_path)

if __name__ == '__main__':
    unittest.main()
