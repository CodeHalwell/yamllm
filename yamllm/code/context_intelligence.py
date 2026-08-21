"""Code context intelligence system for smart code analysis and understanding."""

import ast
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class SymbolType(Enum):
    """Type of code symbol."""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    CONSTANT = "constant"


@dataclass
class CodeSymbol:
    """Represents a code symbol (class, function, etc.)."""
    name: str
    type: SymbolType
    file_path: str
    line_number: int
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    complexity: int = 0
    is_public: bool = True


@dataclass
class FileContext:
    """Context information for a code file."""
    path: str
    language: str
    symbols: List[CodeSymbol]
    imports: List[str]
    dependencies: Set[str]
    lines_of_code: int
    complexity_score: int
    summary: Optional[str] = None


@dataclass
class ProjectContext:
    """Context information for entire project."""
    root_path: str
    files: Dict[str, FileContext]
    symbol_index: Dict[str, CodeSymbol]
    dependency_graph: Dict[str, Set[str]]
    entry_points: List[str]
    architecture_summary: Optional[str] = None


class PythonAnalyzer:
    """Analyzes Python code for context extraction."""

    @staticmethod
    def analyze_file(filepath: str) -> FileContext:
        """Analyze a Python file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Return basic context if parsing fails
            return FileContext(
                path=filepath,
                language="python",
                symbols=[],
                imports=[],
                dependencies=set(),
                lines_of_code=len(content.splitlines()),
                complexity_score=0
            )

        symbols = []
        imports = []
        dependencies = set()

        # Analyze imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    dependencies.add(alias.name.split('.')[0])

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    dependencies.add(node.module.split('.')[0])

        # Analyze top-level symbols
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                symbols.extend(PythonAnalyzer._analyze_class(node, filepath))

            elif isinstance(node, ast.FunctionDef):
                symbol = PythonAnalyzer._analyze_function(node, filepath)
                symbols.append(symbol)

            elif isinstance(node, ast.Assign):
                # Top-level variables/constants
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        is_constant = target.id.isupper()
                        symbols.append(CodeSymbol(
                            name=target.id,
                            type=SymbolType.CONSTANT if is_constant else SymbolType.VARIABLE,
                            file_path=filepath,
                            line_number=node.lineno,
                            is_public=not target.id.startswith('_')
                        ))

        complexity = sum(s.complexity for s in symbols)

        return FileContext(
            path=filepath,
            language="python",
            symbols=symbols,
            imports=imports,
            dependencies=dependencies,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity
        )

    @staticmethod
    def _analyze_class(node: ast.ClassDef, filepath: str) -> List[CodeSymbol]:
        """Analyze a class definition."""
        symbols = []

        # Class itself
        docstring = ast.get_docstring(node)
        bases = [PythonAnalyzer._get_name(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"

        class_symbol = CodeSymbol(
            name=node.name,
            type=SymbolType.CLASS,
            file_path=filepath,
            line_number=node.lineno,
            end_line=node.end_lineno,
            docstring=docstring,
            signature=signature,
            is_public=not node.name.startswith('_')
        )
        symbols.append(class_symbol)

        # Methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method = PythonAnalyzer._analyze_function(item, filepath, parent=node.name)
                symbols.append(method)

        return symbols

    @staticmethod
    def _analyze_function(
        node: ast.FunctionDef,
        filepath: str,
        parent: Optional[str] = None
    ) -> CodeSymbol:
        """Analyze a function/method definition."""
        docstring = ast.get_docstring(node)

        # Build signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        signature = f"def {node.name}({', '.join(args)})"

        # Calculate cyclomatic complexity (simplified)
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return CodeSymbol(
            name=node.name,
            type=SymbolType.METHOD if parent else SymbolType.FUNCTION,
            file_path=filepath,
            line_number=node.lineno,
            end_line=node.end_lineno,
            docstring=docstring,
            signature=signature,
            parent=parent,
            complexity=complexity,
            is_public=not node.name.startswith('_')
        )

    @staticmethod
    def _get_name(node: ast.expr) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{PythonAnalyzer._get_name(node.value)}.{node.attr}"
        else:
            return str(node)


class CodeContextIntelligence:
    """Intelligent code context analysis and extraction."""

    def __init__(self, llm=None, logger: Optional[logging.Logger] = None):
        """
        Initialize code context intelligence.

        Args:
            llm: Optional LLM for generating summaries
            logger: Optional logger
        """
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.project_context: Optional[ProjectContext] = None

    def analyze_project(self, root_path: str, extensions: Optional[List[str]] = None) -> ProjectContext:
        """
        Analyze entire project for context.

        Args:
            root_path: Root directory of project
            extensions: File extensions to analyze (default: ['.py'])

        Returns:
            ProjectContext with complete analysis
        """
        extensions = extensions or ['.py']
        root = Path(root_path)

        self.logger.info(f"Analyzing project at: {root_path}")

        files: Dict[str, FileContext] = {}
        symbol_index: Dict[str, CodeSymbol] = {}
        dependency_graph: Dict[str, Set[str]] = defaultdict(set)

        # Find and analyze all code files
        for ext in extensions:
            for filepath in root.rglob(f'*{ext}'):
                if self._should_skip(filepath):
                    continue

                try:
                    file_context = self._analyze_file(str(filepath), ext)
                    files[str(filepath)] = file_context

                    # Index symbols
                    for symbol in file_context.symbols:
                        full_name = f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
                        symbol_index[full_name] = symbol

                    # Build dependency graph
                    for dep in file_context.dependencies:
                        dependency_graph[str(filepath)].add(dep)

                except Exception as e:
                    self.logger.warning(f"Failed to analyze {filepath}: {e}")

        # Identify entry points
        entry_points = self._identify_entry_points(files)

        self.project_context = ProjectContext(
            root_path=root_path,
            files=files,
            symbol_index=symbol_index,
            dependency_graph=dependency_graph,
            entry_points=entry_points
        )

        # Generate architecture summary if LLM available
        if self.llm:
            self.project_context.architecture_summary = self._generate_architecture_summary()

        self.logger.info(f"Analyzed {len(files)} files, found {len(symbol_index)} symbols")

        return self.project_context

    def _analyze_file(self, filepath: str, extension: str) -> FileContext:
        """Analyze a single file based on language."""
        if extension == '.py':
            return PythonAnalyzer.analyze_file(filepath)
        else:
            # Basic analysis for other languages
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            return FileContext(
                path=filepath,
                language=extension[1:],
                symbols=[],
                imports=[],
                dependencies=set(),
                lines_of_code=len(content.splitlines()),
                complexity_score=0
            )

    def _should_skip(self, filepath: Path) -> bool:
        """Check if file should be skipped."""
        skip_dirs = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', '.tox', 'build', 'dist'}
        return any(part in skip_dirs for part in filepath.parts)

    def _identify_entry_points(self, files: Dict[str, FileContext]) -> List[str]:
        """Identify project entry points."""
        entry_points = []

        for filepath, context in files.items():
            # Check for main execution
            if any(s.name == '__main__' for s in context.symbols):
                entry_points.append(filepath)

            # Check for setup.py, __init__.py, etc.
            filename = os.path.basename(filepath)
            if filename in ['setup.py', '__main__.py', 'app.py', 'main.py']:
                entry_points.append(filepath)

        return entry_points

    def get_relevant_context(
        self,
        query: str,
        max_symbols: int = 10,
        include_dependencies: bool = True
    ) -> str:
        """
        Get relevant code context for a query.

        Args:
            query: Query or task description
            max_symbols: Maximum symbols to include
            include_dependencies: Whether to include dependency information

        Returns:
            Formatted context string
        """
        if not self.project_context:
            return "No project context available. Run analyze_project() first."

        # Find relevant symbols
        relevant_symbols = self._find_relevant_symbols(query, max_symbols)

        # Build context
        context_parts = []

        context_parts.append("=== RELEVANT CODE CONTEXT ===\n")

        for symbol in relevant_symbols:
            context_parts.append(f"\n{symbol.type.value.upper()}: {symbol.name}")
            context_parts.append(f"Location: {symbol.file_path}:{symbol.line_number}")

            if symbol.signature:
                context_parts.append(f"Signature: {symbol.signature}")

            if symbol.docstring:
                context_parts.append(f"Doc: {symbol.docstring[:200]}...")

            if include_dependencies and symbol.dependencies:
                context_parts.append(f"Uses: {', '.join(symbol.dependencies[:5])}")

            context_parts.append("")

        # Add architecture summary if available
        if self.project_context.architecture_summary:
            context_parts.append("\n=== PROJECT ARCHITECTURE ===")
            context_parts.append(self.project_context.architecture_summary)

        return "\n".join(context_parts)

    def _find_relevant_symbols(self, query: str, max_symbols: int) -> List[CodeSymbol]:
        """Find symbols relevant to query."""
        if not self.project_context:
            return []

        # Simple relevance scoring based on name matching
        scored_symbols = []

        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        for symbol in self.project_context.symbol_index.values():
            score = 0

            # Exact name match
            if symbol.name.lower() in query_lower:
                score += 10

            # Word overlap
            symbol_words = set(re.findall(r'\w+', symbol.name.lower()))
            overlap = len(query_words & symbol_words)
            score += overlap * 3

            # Docstring match
            if symbol.docstring:
                doc_words = set(re.findall(r'\w+', symbol.docstring.lower()))
                doc_overlap = len(query_words & doc_words)
                score += doc_overlap

            # Prefer public symbols
            if symbol.is_public:
                score += 1

            # Prefer classes and functions over variables
            if symbol.type in [SymbolType.CLASS, SymbolType.FUNCTION]:
                score += 2

            if score > 0:
                scored_symbols.append((score, symbol))

        # Sort by score and return top symbols
        scored_symbols.sort(key=lambda x: x[0], reverse=True)
        return [symbol for _, symbol in scored_symbols[:max_symbols]]

    def _generate_architecture_summary(self) -> str:
        """Generate high-level architecture summary using LLM."""
        if not self.llm or not self.project_context:
            return ""

        # Gather project statistics
        stats = {
            "total_files": len(self.project_context.files),
            "total_symbols": len(self.project_context.symbol_index),
            "entry_points": self.project_context.entry_points,
            "top_files": sorted(
                self.project_context.files.items(),
                key=lambda x: x[1].complexity_score,
                reverse=True
            )[:5]
        }

        prompt = f"""Analyze this codebase structure and provide a brief architecture summary (3-4 sentences):

Total Files: {stats['total_files']}
Total Symbols: {stats['total_symbols']}
Entry Points: {', '.join(os.path.basename(p) for p in stats['entry_points'])}

Most Complex Files:
{chr(10).join(f"- {os.path.basename(path)}: {ctx.complexity_score} complexity, {len(ctx.symbols)} symbols"
for path, ctx in stats['top_files'])}

Provide a concise summary of the project architecture and organization:"""

        try:
            summary = self.llm.query(prompt)
            return summary.strip()
        except Exception as e:
            self.logger.warning(f"Failed to generate architecture summary: {e}")
            return ""

    def get_symbol_info(self, symbol_name: str) -> Optional[CodeSymbol]:
        """Get information about a specific symbol."""
        if not self.project_context:
            return None

        return self.project_context.symbol_index.get(symbol_name)

    def get_file_summary(self, filepath: str) -> Optional[str]:
        """Get summary of a file."""
        if not self.project_context:
            return None

        file_context = self.project_context.files.get(filepath)
        if not file_context:
            return None

        parts = [
            f"File: {os.path.basename(filepath)}",
            f"Lines: {file_context.lines_of_code}",
            f"Symbols: {len(file_context.symbols)}",
            f"Complexity: {file_context.complexity_score}",
        ]

        if file_context.imports:
            parts.append(f"Imports: {', '.join(file_context.imports[:5])}")

        return "\n".join(parts)
