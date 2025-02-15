from abc import ABC, abstractmethod
from typing import Any, Dict

class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass

    def get_signature(self) -> Dict:
        """Return the tool's signature for LLM context"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters()
        }

    @abstractmethod
    def _get_parameters(self) -> Dict:
        """Return the tool's parameter specifications"""
        pass
