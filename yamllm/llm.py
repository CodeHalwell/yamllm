from .core.llm import LLM

class OpenAIGPT(LLM):
    """
    A class to interact with OpenAI's GPT models.

    Attributes:
        provider_name (str): The name of the provider, set to "openai".

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the OpenAIGPT instance with the given configuration path and API key.

    Initializes the OpenAIGPT instance.

    Args:
        config_path (str): The path to the configuration file.
        api_key (str): The API key for accessing OpenAI's services.
    """
    def __init__(self, config_path: str, api_key: str) -> None:

        self.provider = "openai"
        super().__init__(config_path, api_key)

class DeepSeek(LLM):
    """
    DeepSeek is a subclass of LLM that initializes a connection to the DeepSeek provider.

    Attributes:
        provider_name (str): The name of the provider, set to 'deepseek'.

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the DeepSeek instance with the given configuration path and API key.

        Initializes the DeepSeek instance.

        Args:
            config_path (str): The path to the configuration file.
            api_key (str): The API key for authentication.
        """
    def __init__(self, config_path: str, api_key: str) -> None:

        self.provider = 'deepseek'
        super().__init__(config_path, api_key)

class MistralAI(LLM):
    """
    MistralAI class for interacting with the Mistral language model.

    Attributes:
        provider_name (str): The name of the AI provider, set to 'mistral'.

    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the MistralAI instance with the given configuration path and API key.
    """

    def __init__(self, config_path: str, api_key: str) -> None:
        self.provider = 'mistral'
        super().__init__(config_path, api_key)
    
class GoogleGemini(LLM):
    """
    GoogleGemini is a specialized class for interacting with Google's Gemini models
    through their OpenAI-compatible interface.
    
    This class uses a provider-based approach to interact with Google's Gemini models.
    """
    def __init__(self, config_path: str, api_key: str) -> None:
        """Initialize with Google-specific settings"""
        # Set provider before super() to ensure correct initialization
        self.provider = 'google'
        super().__init__(config_path, api_key)


class AnthropicAI(LLM):
    """
    AnthropicAI class for interacting with Anthropic's Claude models.
    
    This is a wrapper that uses the OpenAI-compatible API endpoint for Claude
    while abstracting away the provider-specific details.
    
    Attributes:
        provider (str): The name of the AI provider, set to 'anthropic'.
    
    Methods:
        __init__(config_path: str, api_key: str) -> None:
            Initializes the AnthropicAI instance with the given configuration path and API key.
    """
    
    def __init__(self, config_path: str, api_key: str) -> None:
        """
        Initialize the AnthropicAI instance.
        
        Args:
            config_path (str): Path to YAML configuration file
            api_key (str): Anthropic API key
        """
        self.provider = 'anthropic'
        super().__init__(config_path, api_key)
