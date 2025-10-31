import os
import fnmatch
from pathlib import Path
from pathspec import PathSpec


GLOBAL_IGNORE_FILE = Path(__file__).parent.parent / ".globalignore"
_cached_global_patterns: set[str] | None = None  # Кеш паттернов


# WARN: нахера нужен кеш, если скрипт запускается единожды?
def load_global_ignore() -> set[str]:
    """
    Загружает паттерны из .globalignore и кеширует их.
    """
    global _cached_global_patterns
    if _cached_global_patterns is not None:
        return _cached_global_patterns

    patterns: set[str] = set()
    if GLOBAL_IGNORE_FILE.exists():
        with GLOBAL_IGNORE_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.add(line)

    _cached_global_patterns = patterns
    return patterns


def load_ignore(
    directory: Path, cli_ignore_list: list[str] | None = None
) -> PathSpec | None:
    """
    Загружает .gitignore, .mdignore и .globalignore.
    """
    patterns: set[str] = load_global_ignore()

    for ignore_file in [directory / ".gitignore", directory / ".mdignore"]:
        if ignore_file.exists():
            with ignore_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.add(line)

    if cli_ignore_list:
        patterns.update(cli_ignore_list)

    return PathSpec.from_lines("gitwildmatch", patterns) if patterns else None


def should_ignore(
    path: str,
    ignore_list: list[str],
    git_path: str,
    gitignore_spec: PathSpec | None = None,
) -> bool:
    relative_path = os.path.relpath(path, git_path)

    # Всегда игнорируем .git
    if relative_path == ".git" or relative_path.startswith(".git" + os.sep):
        return True

    if gitignore_spec:
        match_path = relative_path + "/" if os.path.isdir(path) else relative_path
        if gitignore_spec.match_file(match_path):
            return True

    for pattern in ignore_list:
        if fnmatch.fnmatch(relative_path, pattern):
            return True

    return False
