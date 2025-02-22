yamllm.core.config
==================

.. py:module:: yamllm.core.config


Classes
-------

.. autoapisummary::

   yamllm.core.config.Config


Module Contents
---------------

.. py:class:: Config(/, **data: Any)

   Bases: :py:obj:`pydantic.BaseModel`


   Configuration class for YAMLLM.

   .. attribute:: model

      The name of the LLM model to use

      :type: str

   .. attribute:: temperature

      Sampling temperature for text generation

      :type: float

   .. attribute:: max_tokens

      Maximum number of tokens to generate

      :type: int

   .. attribute:: system_prompt

      The system prompt to use

      :type: str

   .. attribute:: retry_attempts

      Number of retry attempts for API calls

      :type: int

   .. attribute:: timeout

      Timeout in seconds for API calls

      :type: int

   .. attribute:: api_key

      API key for the LLM service

      :type: Optional[str]

   .. attribute:: additional_params

      Additional model parameters

      :type: Dict[str, Any]

   Create a new model by parsing and validating input data from keyword arguments.

   Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
   validated to form a valid model.

   `self` is explicitly positional-only to allow `self` as a field name.


   .. py:method:: from_dict(config_dict: Dict[str, Any]) -> Config
      :classmethod:


      Create configuration from dictionary.

      :param config_dict: Configuration dictionary
      :type config_dict: Dict[str, Any]

      :returns: New configuration instance
      :rtype: Config



   .. py:method:: to_dict() -> Dict[str, Any]

      Convert configuration to dictionary format.

      :returns: Configuration as a dictionary
      :rtype: Dict[str, Any]



   .. py:attribute:: additional_params
      :type:  Dict[str, Any]


   .. py:attribute:: api_key
      :type:  Optional[str]
      :value: None



   .. py:attribute:: max_tokens
      :type:  int
      :value: 500



   .. py:attribute:: model
      :type:  str
      :value: 'gpt-4-turbo-preview'



   .. py:attribute:: retry_attempts
      :type:  int
      :value: 3



   .. py:attribute:: system_prompt
      :type:  str
      :value: 'You are a helpful AI assistant.'



   .. py:attribute:: temperature
      :type:  float
      :value: 0.7



   .. py:attribute:: timeout
      :type:  int
      :value: 30



