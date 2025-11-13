"""Dynamic tool creation from natural language descriptions."""

import ast
import json
import logging
import inspect
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class ToolSpecification:
    """Specification for a dynamically created tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    code: str
    example_usage: Optional[str] = None
    safety_notes: Optional[str] = None


class ToolValidator:
    """Validates dynamically generated tool code for safety."""

    FORBIDDEN_IMPORTS = {
        "os.system", "subprocess", "eval", "exec", "compile",
        "__import__", "open", "file", "input", "raw_input"
    }

    FORBIDDEN_KEYWORDS = {
        "exec", "eval", "compile", "__import__"
    }

    ALLOWED_IMPORTS = {
        "json", "re", "math", "datetime", "time", "random",
        "collections", "itertools", "functools", "operator",
        "string", "typing"
    }

    @staticmethod
    def validate_code(code: str) -> tuple[bool, Optional[str]]:
        """
        Validate tool code for safety.

        Returns:
            (is_valid, error_message)
        """
        try:
            # Parse the code
            tree = ast.parse(code)

            # Check for forbidden patterns
            for node in ast.walk(tree):
                # Check for forbidden function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ToolValidator.FORBIDDEN_KEYWORDS:
                            return False, f"Forbidden function: {node.func.id}"

                # Check for forbidden imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in ToolValidator.ALLOWED_IMPORTS:
                            return False, f"Forbidden import: {alias.name}"

                if isinstance(node, ast.ImportFrom):
                    if node.module not in ToolValidator.ALLOWED_IMPORTS:
                        return False, f"Forbidden import: {node.module}"

                # Check for attribute access on forbidden modules
                if isinstance(node, ast.Attribute):
                    attr_str = ToolValidator._get_attribute_string(node)
                    for forbidden in ToolValidator.FORBIDDEN_IMPORTS:
                        if attr_str.startswith(forbidden):
                            return False, f"Forbidden operation: {attr_str}"

            return True, None

        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def _get_attribute_string(node: ast.Attribute) -> str:
        """Get full attribute string from AST node."""
        parts = []
        current = node

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return ".".join(reversed(parts))


class DynamicTool:
    """A dynamically created tool from natural language description."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        execute_func: Callable,
        example_usage: Optional[str] = None
    ):
        """
        Initialize dynamic tool.

        Args:
            name: Tool name
            description: Tool description
            parameters: Parameter schema
            execute_func: Execution function
            example_usage: Example usage string
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.execute_func = execute_func
        self.example_usage = example_usage

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        try:
            return self.execute_func(**kwargs)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}


class ToolCreator:
    """Creates tools dynamically from natural language descriptions."""

    def __init__(self, llm, logger: Optional[logging.Logger] = None):
        """
        Initialize tool creator.

        Args:
            llm: LLM instance for generating tool code
            logger: Optional logger
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.created_tools: Dict[str, DynamicTool] = {}

    def create_tool(
        self,
        description: str,
        name: Optional[str] = None,
        validate: bool = True
    ) -> DynamicTool:
        """
        Create a tool from natural language description.

        Args:
            description: Natural language description of desired tool
            name: Optional tool name (auto-generated if not provided)
            validate: Whether to validate generated code (default: True)

        Returns:
            Created DynamicTool

        Raises:
            ValueError: If tool creation fails validation
        """
        self.logger.info(f"Creating tool from description: {description}")

        # Generate tool specification using LLM
        spec = self._generate_tool_spec(description, name)

        # Validate code if requested
        if validate:
            is_valid, error = ToolValidator.validate_code(spec.code)
            if not is_valid:
                raise ValueError(f"Generated code failed validation: {error}")

        # Create executable function
        execute_func = self._compile_tool_function(spec.code, spec.name)

        # Create dynamic tool
        tool = DynamicTool(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            execute_func=execute_func,
            example_usage=spec.example_usage
        )

        # Store tool
        self.created_tools[spec.name] = tool

        self.logger.info(f"Successfully created tool: {spec.name}")
        return tool

    def _generate_tool_spec(
        self,
        description: str,
        name: Optional[str] = None
    ) -> ToolSpecification:
        """Generate tool specification using LLM."""
        prompt = f"""Create a Python tool based on this description: {description}

Generate a JSON response with the following structure:
{{
    "name": "tool_name_in_snake_case",
    "description": "Clear description of what the tool does",
    "parameters": {{
        "type": "object",
        "properties": {{
            "param_name": {{
                "type": "string|number|boolean|array|object",
                "description": "Parameter description"
            }}
        }},
        "required": ["param_name"]
    }},
    "code": "def execute(**kwargs):\\n    # Implementation\\n    return result",
    "example_usage": "Example: tool(param='value')",
    "safety_notes": "Any safety considerations"
}}

Requirements:
1. Use only safe Python operations (no subprocess, eval, exec, file I/O)
2. Use only these imports if needed: json, re, math, datetime, time, random, collections, itertools, functools
3. The code must define an 'execute' function that takes **kwargs
4. Return results as dictionaries or primitive types
5. Handle errors gracefully and return {{"error": "message"}} on failure

Generate the tool specification now:"""

        # Query LLM
        response = self.llm.query(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            spec_dict = json.loads(json_str)

            # Override name if provided
            if name:
                spec_dict["name"] = name

            return ToolSpecification(
                name=spec_dict["name"],
                description=spec_dict["description"],
                parameters=spec_dict["parameters"],
                code=spec_dict["code"],
                example_usage=spec_dict.get("example_usage"),
                safety_notes=spec_dict.get("safety_notes")
            )

        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Failed to parse tool specification: {e}")

    def _compile_tool_function(self, code: str, name: str) -> Callable:
        """Compile tool code into executable function."""
        # Create namespace for execution
        namespace = {
            "json": json,
            "re": __import__("re"),
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "time": __import__("time"),
            "random": __import__("random"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
        }

        try:
            # Execute code in namespace
            exec(code, namespace)

            # Extract execute function
            if "execute" not in namespace:
                raise ValueError("Generated code must define 'execute' function")

            return namespace["execute"]

        except Exception as e:
            self.logger.error(f"Failed to compile tool code: {e}")
            raise ValueError(f"Failed to compile tool function: {e}")

    def list_tools(self) -> List[Dict[str, str]]:
        """List all created tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "example": tool.example_usage or "N/A"
            }
            for tool in self.created_tools.values()
        ]

    def get_tool(self, name: str) -> Optional[DynamicTool]:
        """Get a created tool by name."""
        return self.created_tools.get(name)

    def delete_tool(self, name: str) -> bool:
        """Delete a created tool."""
        if name in self.created_tools:
            del self.created_tools[name]
            return True
        return False

    def export_tool(self, name: str, filepath: str):
        """Export tool specification to file."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        spec = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "code": inspect.getsource(tool.execute_func),
            "example_usage": tool.example_usage
        }

        with open(filepath, 'w') as f:
            json.dump(spec, f, indent=2)

    def import_tool(self, filepath: str) -> DynamicTool:
        """Import tool specification from file."""
        with open(filepath, 'r') as f:
            spec_dict = json.load(f)

        spec = ToolSpecification(
            name=spec_dict["name"],
            description=spec_dict["description"],
            parameters=spec_dict["parameters"],
            code=spec_dict["code"],
            example_usage=spec_dict.get("example_usage")
        )

        # Validate and compile
        is_valid, error = ToolValidator.validate_code(spec.code)
        if not is_valid:
            raise ValueError(f"Imported code failed validation: {error}")

        execute_func = self._compile_tool_function(spec.code, spec.name)

        tool = DynamicTool(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            execute_func=execute_func,
            example_usage=spec.example_usage
        )

        self.created_tools[spec.name] = tool
        return tool
