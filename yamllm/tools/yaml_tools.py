from .base import Tool
import yaml

class ParseYAML(Tool):
    def __init__(self):
        super().__init__(
            name="parse_yaml",
            description="Parse YAML string into Python object"
        )

    def execute(self, content: str) -> dict:
        return yaml.safe_load(content)

    def _get_parameters(self) -> dict:
        return {
            "content": {
                "type": "string",
                "description": "YAML content to parse"
            }
        }

class DumpYAML(Tool):
    def __init__(self):
        super().__init__(
            name="dump_yaml",
            description="Convert Python object to YAML string"
        )

    def execute(self, data: dict) -> str:
        return yaml.dump(data, sort_keys=False)

    def _get_parameters(self) -> dict:
        return {
            "data": {
                "type": "object",
                "description": "Python object to convert to YAML"
            }
        }
