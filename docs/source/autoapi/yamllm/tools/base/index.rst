yamllm.tools.base
=================

.. py:module:: yamllm.tools.base


Classes
-------

.. autoapisummary::

   yamllm.tools.base.Tool
   yamllm.tools.base.ToolRegistry


Module Contents
---------------

.. py:class:: Tool(name: str, description: str)

   .. py:method:: execute(*args, **kwargs) -> Any
      :abstractmethod:



   .. py:attribute:: description


   .. py:attribute:: name


.. py:class:: ToolRegistry

   .. py:method:: get_tool(name: str) -> Tool

      Get a tool by name



   .. py:method:: list_tools() -> List[str]

      List all registered tool names



   .. py:method:: register(tool: Tool) -> None

      Register a tool in the registry



