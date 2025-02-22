yamllm.tools.utility_tools
==========================

.. py:module:: yamllm.tools.utility_tools


Classes
-------

.. autoapisummary::

   yamllm.tools.utility_tools.Calculator
   yamllm.tools.utility_tools.TimezoneTool
   yamllm.tools.utility_tools.UnitConverter
   yamllm.tools.utility_tools.WebSearch


Module Contents
---------------

.. py:class:: Calculator

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(expression: str) -> float


.. py:class:: TimezoneTool

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(time: str, from_tz: str, to_tz: str) -> str


.. py:class:: UnitConverter

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(value: float, from_unit: str, to_unit: str) -> float


.. py:class:: WebSearch(api_key: str)

   Bases: :py:obj:`yamllm.tools.base.Tool`


   .. py:method:: execute(query: str, max_results: int = 5) -> List[Dict]


   .. py:attribute:: api_key


