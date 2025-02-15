from parser import parse_yaml_config, YamlLMConfig
from openai import OpenAI, OpenAIError
from typing import Optional, Dict, Any
import os

class LLM(object):  # Explicitly inherit from object
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config: YamlLMConfig = self.load_config()
        
        # Handle environment variable if api_key starts with $
        self.api_key = self.config.provider.api_key       
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

        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.config.provider.base_url
        )

    def load_config(self) -> YamlLMConfig:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at path: {self.config_path}")
        return parse_yaml_config(self.config_path)

    def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise ValueError("API key is not initialized or invalid.")
        try:
            return self.get_response(prompt, system_prompt)
        except OpenAIError as e:
            raise Exception(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during query: {str(e)}")

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        
        # Add system prompt if provided
        if system_prompt or self.config.context.system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt or self.config.context.system_prompt
            })
        
        # Add user message
        messages.append({"role": "user", "content": prompt})

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
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Error getting response from OpenAI: {str(e)}")

    def update_settings(self, **kwargs: Dict[str, Any]) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


    def print_settings(self) -> None:
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



