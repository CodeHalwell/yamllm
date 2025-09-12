"""
Configuration templates and presets for YAMLLM.

This module provides predefined configuration templates for different use cases
and providers, making it easy for users to get started quickly.
"""

from typing import Dict, Any, List, Optional
import yaml


class ConfigTemplate:
    """Base class for configuration templates."""
    
    def __init__(self, name: str, description: str, provider: str):
        self.name = name
        self.description = description
        self.provider = provider
    
    def generate_config(self, **kwargs) -> Dict[str, Any]:
        """Generate configuration dictionary."""
        raise NotImplementedError
    
    def save_to_file(self, path: str, **kwargs) -> None:
        """Save configuration to a YAML file."""
        config = self.generate_config(**kwargs)
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class OpenAITemplate(ConfigTemplate):
    """OpenAI configuration templates."""
    
    def generate_config(self, 
                       model: str = "gpt-4",
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       tools_enabled: bool = True,
                       memory_enabled: bool = True,
                       streaming: bool = True,
                       preset: str = "casual") -> Dict[str, Any]:
        """Generate OpenAI configuration."""
        
        # Base configuration
        config = {
            "provider": {
                "name": "openai",
                "model": model
            },
            "model_settings": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop_sequences": []
            },
            "request": {
                "timeout": 30,
                "retry": {
                    "max_attempts": 3,
                    "initial_delay": 1,
                    "backoff_factor": 2
                }
            },
            "output": {
                "format": "text",
                "stream": streaming
            },
            "logging": {
                "level": "INFO",
                "file": "yamllm.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "safety": {
                "content_filtering": True,
                "max_requests_per_minute": 60,
                "sensitive_keywords": []
            }
        }
        
        # Preset-specific configurations
        if preset == "casual":
            config["context"] = {
                "system_prompt": "You are a helpful, conversational assistant. Be friendly and engaging while providing accurate information.",
                "max_context_length": 8000,
                "memory": {
                    "enabled": memory_enabled,
                    "max_messages": 10,
                    "session_id": "casual_chat",
                    "conversation_db": "memory/conversation_history.db",
                    "vector_store": {
                        "index_path": "memory/vector_store/faiss_index.idx",
                        "metadata_path": "memory/vector_store/metadata.pkl",
                        "top_k": 3
                    }
                }
            }
            config["tools"] = {
                "enabled": tools_enabled,
                "tool_timeout": 10,
                "packs": ["common", "web"],
                "tools": ["unit_converter"],
                "safe_mode": False,
                "allow_network": True,
                "allow_filesystem": True,
                "allowed_paths": ["./"],
                "blocked_domains": [],
                "gate_web_search": True
            }
        
        elif preset == "coding":
            config["context"] = {
                "system_prompt": "You are an expert programming assistant. Help with code, debugging, architecture decisions, and best practices. Always explain your reasoning.",
                "max_context_length": 16000,
                "memory": {
                    "enabled": memory_enabled,
                    "max_messages": 15,
                    "session_id": "coding_session",
                    "conversation_db": "memory/conversation_history.db",
                    "vector_store": {
                        "index_path": "memory/vector_store/faiss_index.idx",
                        "metadata_path": "memory/vector_store/metadata.pkl",
                        "top_k": 5
                    }
                }
            }
            config["tools"] = {
                "enabled": tools_enabled,
                "tool_timeout": 15,
                "packs": ["common", "web", "files", "text"],
                "tools": [],
                "safe_mode": False,
                "allow_network": True,
                "allow_filesystem": True,
                "allowed_paths": ["./", "~/"],
                "blocked_domains": [],
                "gate_web_search": True
            }
            # Higher max tokens for code generation
            config["model_settings"]["max_tokens"] = 2000
            
        elif preset == "research":
            config["context"] = {
                "system_prompt": "You are a research assistant. Help analyze information, find sources, and provide well-researched answers with citations when possible.",
                "max_context_length": 12000,
                "memory": {
                    "enabled": memory_enabled,
                    "max_messages": 20,
                    "session_id": "research_session",
                    "conversation_db": "memory/conversation_history.db",
                    "vector_store": {
                        "index_path": "memory/vector_store/faiss_index.idx",
                        "metadata_path": "memory/vector_store/metadata.pkl",
                        "top_k": 8
                    }
                }
            }
            config["tools"] = {
                "enabled": tools_enabled,
                "tool_timeout": 20,
                "packs": ["common", "web", "text"],
                "tools": ["web_scraper"],
                "safe_mode": False,
                "allow_network": True,
                "allow_filesystem": False,
                "allowed_paths": [],
                "blocked_domains": [],
                "gate_web_search": False  # Allow unrestricted web search for research
            }
        
        elif preset == "minimal":
            config["context"] = {
                "system_prompt": "You are a helpful assistant.",
                "max_context_length": 4000,
                "memory": {
                    "enabled": False,
                    "max_messages": 5,
                    "session_id": "minimal_session",
                    "conversation_db": "memory/conversation_history.db",
                    "vector_store": {
                        "index_path": "memory/vector_store/faiss_index.idx",
                        "metadata_path": "memory/vector_store/metadata.pkl",
                        "top_k": 1
                    }
                }
            }
            config["tools"] = {
                "enabled": False,
                "tool_timeout": 5,
                "packs": [],
                "tools": [],
                "safe_mode": True,
                "allow_network": False,
                "allow_filesystem": False,
                "allowed_paths": [],
                "blocked_domains": [],
                "gate_web_search": True
            }
            # Lower settings for minimal use
            config["model_settings"]["max_tokens"] = 500
            config["model_settings"]["temperature"] = 0.3
        
        # Add thinking mode and embeddings
        config["thinking"] = {
            "enabled": True if preset in ["coding", "research"] else False,
            "show_tool_reasoning": True,
            "model": None,
            "max_tokens": 512,
            "stream_thinking": True,
            "save_thinking": False,
            "thinking_temperature": 0.7
        }
        
        config["embeddings"] = {
            "provider": None,
            "model": "text-embedding-3-small"
        }
        
        return config


class AnthropicTemplate(ConfigTemplate):
    """Anthropic Claude configuration templates."""
    
    def generate_config(self, 
                       model: str = "claude-3-sonnet-20240229",
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       tools_enabled: bool = True,
                       memory_enabled: bool = True,
                       streaming: bool = True,
                       preset: str = "casual") -> Dict[str, Any]:
        """Generate Anthropic configuration."""
        
        # Get base config from OpenAI template and adapt
        openai_template = OpenAITemplate("anthropic_base", "Base Anthropic config", "anthropic")
        config = openai_template.generate_config(
            model=model, 
            temperature=temperature,
            max_tokens=max_tokens,
            tools_enabled=tools_enabled,
            memory_enabled=memory_enabled,
            streaming=streaming,
            preset=preset
        )
        
        # Update provider
        config["provider"]["name"] = "anthropic"
        config["provider"]["model"] = model
        
        # Claude-specific adjustments
        if preset == "research":
            # Claude excels at long-form analysis
            config["model_settings"]["max_tokens"] = 2000
            config["context"]["max_context_length"] = 20000
        
        return config


class GoogleTemplate(ConfigTemplate):
    """Google Gemini configuration templates."""
    
    def generate_config(self, 
                       model: str = "gemini-pro",
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       tools_enabled: bool = True,
                       memory_enabled: bool = True,
                       streaming: bool = True,
                       preset: str = "casual") -> Dict[str, Any]:
        """Generate Google configuration."""
        
        # Get base config and adapt
        openai_template = OpenAITemplate("google_base", "Base Google config", "google")
        config = openai_template.generate_config(
            model=model, 
            temperature=temperature,
            max_tokens=max_tokens,
            tools_enabled=tools_enabled,
            memory_enabled=memory_enabled,
            streaming=streaming,
            preset=preset
        )
        
        # Update provider
        config["provider"]["name"] = "google"
        config["provider"]["model"] = model
        
        # Google-specific adjustments
        config["safety"]["content_filtering"] = True  # Google has strong safety features
        
        return config


class ConfigTemplateManager:
    """Manages configuration templates and presets."""
    
    def __init__(self):
        self.templates = {
            "openai": OpenAITemplate("openai", "OpenAI GPT models", "openai"),
            "anthropic": AnthropicTemplate("anthropic", "Anthropic Claude models", "anthropic"),
            "google": GoogleTemplate("google", "Google Gemini models", "google")
        }
        
        self.presets = {
            "casual": "Friendly conversational assistant with basic tools",
            "coding": "Programming assistant with development tools",
            "research": "Research assistant with web search and analysis tools",
            "minimal": "Basic assistant with no tools or memory"
        }
        
        self.provider_models = {
            "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "google": ["gemini-pro", "gemini-pro-vision"]
        }
    
    def list_templates(self) -> List[str]:
        """List available template providers."""
        return list(self.templates.keys())
    
    def list_presets(self) -> Dict[str, str]:
        """List available presets."""
        return self.presets.copy()
    
    def list_models(self, provider: str) -> List[str]:
        """List available models for a provider."""
        return self.provider_models.get(provider, [])
    
    def create_config(self, 
                     provider: str,
                     preset: str = "casual",
                     model: Optional[str] = None,
                     output_path: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """Create a configuration from template."""
        
        if provider not in self.templates:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(self.templates.keys())}")
        
        if preset not in self.presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.presets.keys())}")
        
        template = self.templates[provider]
        
        # Use default model if not specified
        if model is None:
            model = self.provider_models[provider][0]
        
        config = template.generate_config(model=model, preset=preset, **kwargs)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return config
    
    def validate_template_params(self, provider: str, preset: str, model: Optional[str] = None) -> List[str]:
        """Validate template parameters and return any warnings."""
        warnings = []
        
        if provider not in self.templates:
            warnings.append(f"Unknown provider: {provider}")
        
        if preset not in self.presets:
            warnings.append(f"Unknown preset: {preset}")
        
        if model and provider in self.provider_models:
            if model not in self.provider_models[provider]:
                warnings.append(f"Model {model} not in recommended list for {provider}")
        
        return warnings


# Global instance for easy access
template_manager = ConfigTemplateManager()