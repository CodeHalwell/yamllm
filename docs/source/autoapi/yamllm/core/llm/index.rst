yamllm.core.llm
===============

.. py:module:: yamllm.core.llm


Classes
-------

.. autoapisummary::

   yamllm.core.llm.DeepSeek
   yamllm.core.llm.GoogleGemini
   yamllm.core.llm.LLM
   yamllm.core.llm.MistralAI
   yamllm.core.llm.OpenAIGPT


Functions
---------

.. autoapisummary::

   yamllm.core.llm.setup_logging


Module Contents
---------------

.. py:class:: DeepSeek(config_path: str, api_key: str)

   Bases: :py:obj:`LLM`


   DeepSeek is a subclass of LLM that initializes a connection to the DeepSeek provider.

   .. attribute:: provider

      The name of the provider, set to 'deepseek'.

      :type: str

   .. method:: __init__(config_path

      str, api_key: str) -> None:
      Initializes the DeepSeek instance with the given configuration path and API key.
      

   .. method:: Initializes the DeepSeek instance.

      
      

   .. method:: Args

      
      config_path (str): The path to the configuration file.
      api_key (str): The API key for authentication.
      
      

   Initialize the LLM instance with the given configuration path.

   :param config_path: Path to the YAML configuration file
   :type config_path: str
   :param api_key: API key for the LLM service
   :type api_key: str


   .. py:attribute:: provider
      :value: 'deepseek'



.. py:class:: GoogleGemini(config_path: str, api_key: str)

   Bases: :py:obj:`LLM`


   GoogleGemini is a subclass of LLM that interacts with Google's language model to generate responses based on given prompts.

   Attributes:
   provider (str): The provider of the language model, set to 'google'.

   Methods:
   __init__(config_path: str, api_key: str) -> None:
       Initializes the GoogleGemini instance with the given configuration path and API key.

   get_response(prompt: str, system_prompt: Optional[str] = None) -> str:

   Generates a response from the language model based on the given prompt and optional system prompt.
           str: The response from the language model.
           Exception: If there is an error getting the response from the language model.

   Initialize the LLM instance with the given configuration path.

   :param config_path: Path to the YAML configuration file
   :type config_path: str
   :param api_key: API key for the LLM service
   :type api_key: str


   .. py:method:: get_response(prompt: str, system_prompt: Optional[str] = None) -> str

      Override get_response to use only Google-supported parameters and message ordering.

      :param prompt: The prompt to send to the model.
      :type prompt: str
      :param system_prompt: An optional system prompt for context.
      :type system_prompt: Optional[str]

      :returns: The response from the Google language model.
      :rtype: str

      :raises Exception: If there is an error getting the response from Google.



   .. py:attribute:: provider
      :value: 'google'



.. py:class:: LLM(config_path: str, api_key: str)

   Bases: :py:obj:`object`


   Main LLM interface class for YAMLLM.

   This class handles configuration loading and API interactions
   with language models.

   :param config_path: Path to YAML configuration file
   :type config_path: str
   :param api_key: API key for the LLM service
   :type api_key: str

   .. rubric:: Examples

   >>> llm = LLM(config_path = "config.yaml", api_key = "your-api-key")

   >>> response = llm.query("Hello, world!")

   Initialize the LLM instance with the given configuration path.

   :param config_path: Path to the YAML configuration file
   :type config_path: str
   :param api_key: API key for the LLM service
   :type api_key: str


   .. py:method:: create_embedding(text: str) -> bytes

      Create an embedding for the given text using OpenAI's API.

      :param text: The text to create an embedding for.
      :type text: str

      :returns: The embedding as bytes.
      :rtype: bytes

      :raises Exception: If there is an error creating the embedding.



   .. py:method:: find_similar_messages(query: str, k: int) -> List[Dict[str, Any]]

      Find messages similar to the query.

      :param query: The text to find similar messages for.
      :type query: str
      :param k: Number of similar messages to return. Default is 5.
      :type k: int

      :returns: List of similar messages with their metadata and similarity scores.
      :rtype: List[Dict[str, Any]]



   .. py:method:: get_response(prompt: str, system_prompt: Optional[str] = None) -> str

      Generates a response from the language model based on the provided prompt and optional system prompt.

      :param prompt: The user's input prompt to generate a response for.
      :type prompt: str
      :param system_prompt: An optional system prompt to provide context or instructions to the model. Defaults to None.
      :type system_prompt: Optional[str], optional

      :returns: The generated response from the language model if output_stream is disabled.
                None: If output_stream is enabled, the response is streamed and displayed in real-time.
      :rtype: str

      :raises Exception: If there is an error getting a response from the language model.

      Behavior:
          - If a system prompt is provided or exists in the instance, it is added as the first message.
          - If memory is enabled, conversation history is retrieved and added to the messages.
          - Similar messages from previous conversations are found and appended to the user's prompt.
          - The current prompt is added to the messages.
          - If output_stream is enabled, the response is streamed and displayed using Rich's Live and Markdown components.
          - If output_stream is disabled, the response is retrieved in a single call and displayed using Rich's Console and Markdown components.
          - The conversation memory is updated with the new prompt and response.



   .. py:method:: load_config() -> yamllm.core.parser.YamlLMConfig

      Load configuration from YAML file.

      :returns: Parsed configuration.
      :rtype: YamlLMConfig

      :raises FileNotFoundError: If config file is not found.
      :raises ValueError: If config file is empty or could not be parsed.
      :raises Exception: For any other unexpected errors.



   .. py:method:: print_settings() -> None

      Print the current settings of the LLM (Language Model) in an organized format.
      Settings are grouped by category for better readability.



   .. py:method:: query(prompt: str, system_prompt: Optional[str] = None) -> str

      Send a query to the language model.

      :param prompt: The prompt to send to the model.
      :type prompt: str
      :param system_prompt: An optional system prompt to provide context.
      :type system_prompt: Optional[str]

      :returns: The response from the language model.
      :rtype: str

      :raises ValueError: If API key is not initialized or invalid.
      :raises Exception: If there is an error during the query.



   .. py:method:: update_settings(**kwargs: Dict[str, Any]) -> None

      Update the settings of the instance with the provided keyword arguments.

      This method iterates over the provided keyword arguments and updates the
      instance attributes if they exist.

      :param \*\*kwargs: Keyword arguments where the key is the
      :type \*\*kwargs: Dict[str, Any]
      :param attribute name and the value is the new value for that attribute.:

      .. rubric:: Example

      >>> llm.update_settings(temperature=0.8)



   .. py:attribute:: api_key


   .. py:attribute:: base_url


   .. py:attribute:: client


   .. py:attribute:: config
      :type:  yamllm.core.parser.YamlLMConfig


   .. py:attribute:: config_path


   .. py:attribute:: content_filtering


   .. py:attribute:: conversation_db_path


   .. py:attribute:: embedding_client


   .. py:attribute:: frequency_penalty


   .. py:attribute:: logger


   .. py:attribute:: max_context_length


   .. py:attribute:: max_requests_per_minute


   .. py:attribute:: max_tokens


   .. py:attribute:: memory
      :value: None



   .. py:attribute:: memory_enabled


   .. py:attribute:: memory_max_messages


   .. py:attribute:: model


   .. py:attribute:: output_format


   .. py:attribute:: output_stream


   .. py:attribute:: presence_penalty


   .. py:attribute:: provider


   .. py:attribute:: request_timeout


   .. py:attribute:: retry_backoff_factor


   .. py:attribute:: retry_initial_delay


   .. py:attribute:: retry_max_attempts


   .. py:attribute:: sensitive_keywords


   .. py:attribute:: stop_sequences


   .. py:attribute:: system_prompt


   .. py:attribute:: temperature


   .. py:attribute:: tools


   .. py:attribute:: tools_enabled


   .. py:attribute:: tools_timeout


   .. py:attribute:: top_p


   .. py:attribute:: vector_index_path


   .. py:attribute:: vector_metadata_path


   .. py:attribute:: vector_store
      :value: None



   .. py:attribute:: vector_store_top_k


.. py:class:: MistralAI(config_path: str, api_key: str)

   Bases: :py:obj:`LLM`


   MistralAI class for interacting with the Mistral language model.
   .. attribute:: provider

      The name of the AI provider, set to 'mistral'.

      :type: str

   .. method:: __init__(config_path

      str, api_key: str) -> None:
      Initializes the MistralAI instance with the given configuration path and API key.

   .. method:: get_response(prompt

      str, system_prompt: Optional[str] = None) -> str:
      Generates a response from the Mistral language model based on the given prompt and optional system prompt.
      :param prompt: The user input prompt to generate a response for.
      :type prompt: str
      :param system_prompt: An optional system prompt to provide context for the response.
      :type system_prompt: Optional[str]
      
      :returns: The generated response from the Mistral language model.
      :rtype: str
      

   Initialize the LLM instance with the given configuration path.

   :param config_path: Path to the YAML configuration file
   :type config_path: str
   :param api_key: API key for the LLM service
   :type api_key: str


   .. py:method:: get_response(prompt: str, system_prompt: Optional[str] = None) -> str

      Generate a response based on the given prompt and optional system prompt.
      This method overrides the base `get_response` method to use only Mistral-supported
      parameters and message ordering. It supports memory initialization, conversation
      history retrieval, and finding similar messages to provide relevant context.

      Parameters:
      - prompt (str): The user prompt to generate a response for.
      - system_prompt (Optional[str]): An optional system prompt to provide context.

      Returns:
      - str: The generated response text if `output_stream` is False, otherwise None.

      Raises:
      - Exception: If there is an error getting a response from Mistral.



   .. py:attribute:: provider
      :value: 'mistral'



.. py:class:: OpenAIGPT(config_path: str, api_key: str)

   Bases: :py:obj:`LLM`


   A class to interact with OpenAI's GPT models.

   .. attribute:: provider

      The name of the provider, set to "openai".

      :type: str

   .. method:: __init__(config_path

      str, api_key: str) -> None:
      Initializes the OpenAIGPT instance with the given configuration path and API key.
      

   Initializes the OpenAIGPT instance.

   :param config_path: The path to the configuration file.
   :type config_path: str
   :param api_key: The API key for accessing OpenAI's services.
   :type api_key: str

   Initialize the LLM instance with the given configuration path.

   :param config_path: Path to the YAML configuration file
   :type config_path: str
   :param api_key: API key for the LLM service
   :type api_key: str


   .. py:attribute:: provider
      :value: 'openai'



.. py:function:: setup_logging(config)

   Set up logging configuration for the yamllm application.
   This function configures the logging settings based on the provided configuration.
   It sets the logging level for the 'httpx' and 'urllib3' libraries to WARNING to suppress
   INFO messages, disables propagation to the root logger, and configures the 'yamllm' logger
   with the specified logging level, file handler, and formatter.
   :param config: A configuration object that contains logging settings. It should have
                  the following attributes:
                  - logging.level (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
                  - logging.file (str): The file path where log messages should be written.
                  - logging.format (str): The format string for log messages.
   :type config: object

   :returns: The configured logger for the 'yamllm' application.
   :rtype: logging.Logger


