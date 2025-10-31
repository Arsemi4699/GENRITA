from pathlib import Path
from pathspec import PathSpec
from utils.ignore import should_ignore


class TreeNode:
    """Узел дерева, представляющий файл или папку."""

    def __init__(self, name: str, is_dir: bool):
        self.name: str = name
        self.is_dir: bool = is_dir
        self.children: dict[str, "TreeNode"] = {}  # Теперь всегда словарь

    def __repr__(self) -> str:
        return f"TreeNode(name={self.name}, is_dir={self.is_dir}, children={list(self.children.keys())})"


def build_tree(
    directory: Path, base: Path, gitignore_spec: PathSpec | None
) -> TreeNode:
    """
    Рекурсивно обходит папку, создавая дерево `TreeNode`.
    """
    node = TreeNode(directory.name, is_dir=True)

    for item in directory.iterdir():
        if should_ignore(str(item), [], str(base), gitignore_spec):
            continue
        if item.is_dir():
            node.children[item.name] = build_tree(item, base, gitignore_spec)
        else:
            node.children[item.name] = TreeNode(item.name, is_dir=False)

    return node
