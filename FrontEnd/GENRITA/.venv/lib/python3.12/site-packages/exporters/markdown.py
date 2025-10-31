from .base import BaseExporter
from pathlib import Path


class MarkdownExporter(BaseExporter):
    def format(
        self, relative_path: Path, content: str, language: str | None = None
    ) -> str:
        if language:
            return f"## File: {relative_path}\n```{language}\n{content}\n```\n"
        else:
            return f"## File: {relative_path}\n```\n{content}\n```\n"


exporter = MarkdownExporter()

