from .base import BaseConverter
from pathlib import Path


class DefaultConverter(BaseConverter):
    aliases: list[str] = []  # Fallback converter for all file types
    language_map: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".html": "html",
        ".css": "css",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".json": "json",
        ".xml": "xml",
        ".sh": "bash",
        ".md": "markdown",
        ".lua": "lua",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".xls": "csv",
        ".xlsx": "csv",
        ".pdf": "pdf",
        ".ipynb": "markdown",
    }

    def convert(self, file_path: Path) -> str:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            return f"Error reading {file_path}: {e}"


converter = DefaultConverter()
