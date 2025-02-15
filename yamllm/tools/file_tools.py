from .base import Tool

class ReadFileContent(Tool):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read content from a file"
        )

    def execute(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_parameters(self) -> dict:
        return {
            "filepath": {
                "type": "string",
                "description": "Path to the file to read"
            }
        }

class WriteFileContent(Tool):
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file"
        )

    def execute(self, filepath: str, content: str) -> bool:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    def _get_parameters(self) -> dict:
        return {
            "filepath": {
                "type": "string",
                "description": "Path to the file to write"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        }
