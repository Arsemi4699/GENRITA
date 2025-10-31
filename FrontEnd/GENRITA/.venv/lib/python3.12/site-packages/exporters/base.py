from abc import ABC, abstractmethod
from pathlib import Path


class BaseExporter(ABC):
    @abstractmethod
    def format(
        self, relative_path: Path, content: str, language: str | None = None
    ) -> str:
        """Format file content for export."""
        pass

