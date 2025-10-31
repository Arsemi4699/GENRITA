from abc import ABC, abstractmethod
from pathlib import Path


class BaseConverter(ABC):
    aliases: list[str] = []  # Supported file extensions (lowercase)
    language_map: dict[str, str] = {}  # Mapping file extensions to languages

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.aliases

    @abstractmethod
    def convert(self, file_path: Path) -> str:
        """Convert file content to a string."""
        pass

    def get_language(self, file_path: Path) -> str | None:
        """Return language for syntax highlighting based on file extension."""
        return self.language_map.get(file_path.suffix.lower())

