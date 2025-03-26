from typing import Optional, Dict, Any
from pydantic import BaseModel
from yamllm.core.parser import parse_yaml_config

class Config(BaseModel):
    """Configuration class for YAMLLM.
    
    Attributes:
        model (str): The name of the LLM model to use
        temperature (float): Sampling temperature for text generation
        max_tokens (int): Maximum number of tokens to generate
        system_prompt (str): The system prompt to use
        retry_attempts (int): Number of retry attempts for API calls
        timeout (int): Timeout in seconds for API calls
        api_key (Optional[str]): API key for the LLM service
        additional_params (Dict[str, Any]): Additional model parameters
    """
    
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 500
    system_prompt: str = "You are a helpful AI assistant."
    retry_attempts: int = 3
    timeout: int = 30
    api_key: Optional[str] = None
    additional_params: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.
        
        Returns:
            Dict[str, Any]: Configuration as a dictionary
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary
            
        Returns:
            Config: New configuration instance
        """
        return cls(**config_dict)

    def load_config(self, yaml_file_path: str):
        """
        Load configuration from a YAML file.

        Args:
            yaml_file_path (str): Path to the YAML configuration file.

        Returns:
            dict: Parsed configuration data.
        """
        self.config_data = parse_yaml_config(yaml_file_path)
        return self.config_data

    def validate_config(self, config: dict) -> bool:
        """
        Validate the configuration dictionary.

        Args:
            config (dict): Configuration data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        required_keys = ["provider", "model_settings", "request"]
        return all(key in config for key in required_keys)