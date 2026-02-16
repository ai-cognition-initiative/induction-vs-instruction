from __future__ import annotations

import ast
from pathlib import Path


def extract_symbols(file_path: Path) -> dict[str, list[str]]:
    """Extract classes, functions, and top-level constants from a Python file."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return {"classes": [], "functions": [], "constants": []}

    classes = []
    functions = []
    constants = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if methods:
                classes.append(
                    f"{node.name}({', '.join(methods[:5])}{'...' if len(methods) > 5 else ''})"
                )
            else:
                classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)

    return {"classes": classes, "functions": functions, "constants": constants}


def generate_repomap(root: Path, exclude_dirs: set[str] | None = None) -> str:
    """Generate a repo map string with folder structure."""
    exclude_dirs = exclude_dirs or {
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
    }

    py_files = sorted(
        p for p in root.rglob("*.py") if not any(ex in p.parts for ex in exclude_dirs)
    )

    files_by_folder: dict[str, list[tuple[str, dict]]] = {}
    for py_file in py_files:
        rel_path = py_file.relative_to(root)
        folder = str(rel_path.parent) if rel_path.parent != Path(".") else "."
        filename = rel_path.name
        symbols = extract_symbols(py_file)
        files_by_folder.setdefault(folder, []).append((filename, symbols))

    lines = []
    for folder in sorted(files_by_folder.keys()):
        lines.append(f"+-- {folder}/")
        files = files_by_folder[folder]
        for i, (filename, symbols) in enumerate(files):
            prefix = "|   `--" if i == len(files) - 1 else "|   +--"

            all_symbols = []
            all_symbols.extend(symbols["constants"])
            all_symbols.extend(symbols["classes"])
            all_symbols.extend(symbols["functions"])

            if all_symbols:
                symbols_str = ", ".join(all_symbols)
                lines.append(f"{prefix} {filename}")
                lines.append(
                    f"{'|       ' if i == len(files) - 1 else '|   |   '}{symbols_str}"
                )
            else:
                lines.append(f"{prefix} {filename}")
        lines.append("|")

    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    print(generate_repomap(root))
