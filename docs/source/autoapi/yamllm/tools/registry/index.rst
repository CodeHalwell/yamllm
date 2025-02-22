yamllm.tools.registry
=====================

.. py:module:: yamllm.tools.registry


Classes
-------

.. autoapisummary::

   yamllm.tools.registry.ToolRegistry


Module Contents
---------------

.. py:class:: ToolRegistry

   .. py:method:: get_tool(name: str) -> yamllm.tools.base.Tool

      Retrieve a tool by name



   .. py:method:: list_tools() -> Dict[str, Dict]

      Return all registered tools and their signatures



   .. py:method:: register_tool(tool: yamllm.tools.base.Tool) -> None

      Register a new tool instance



