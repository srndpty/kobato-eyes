"""Verify layer import boundaries and provide a simple dependency report."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "src" / "core"
TESTS_CORE_DIR = PROJECT_ROOT / "tests" / "core"


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        yield path


def _module_name_from_path(root: Path, path: Path, package_prefix: str) -> str:
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if not parts:
        return package_prefix
    if parts[-1] == "__init__":
        parts[-1] = "__init__"
    return ".".join([package_prefix, *parts])


def _collect_imports(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = ("." * node.level + module).lstrip(".")
            if module:
                imports.add(module)
            for alias in node.names:
                combined = f"{module}.{alias.name}" if module else alias.name
                combined = combined.lstrip(".")
                if combined:
                    imports.add(combined)
    return imports


def _build_import_graph(root: Path, package_prefix: str) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in _iter_python_files(root):
        module_name = _module_name_from_path(root, path, package_prefix)
        graph[module_name] = _collect_imports(path)
    return graph


def test_core_layers_do_not_depend_on_ui() -> None:
    graphs = {
        "core": _build_import_graph(CORE_DIR, "core"),
        "tests.core": _build_import_graph(TESTS_CORE_DIR, "tests.core"),
    }

    offenders: dict[str, set[str]] = defaultdict(set)

    for scope, modules in graphs.items():
        print(f"[import-graph] {scope}")
        for module, deps in sorted(modules.items()):
            if deps:
                formatted = ", ".join(sorted(deps))
                print(f"  {module} -> {formatted}")
            else:
                print(f"  {module} -> (no imports)")

            ui_deps = {dep for dep in deps if dep.split(".")[0] == "ui"}
            if ui_deps:
                offenders[scope].add(f"{module}: {sorted(ui_deps)}")

    assert not offenders, (
        "ui layer must not be imported from core or tests/core: "
        + "; ".join(f"{scope} -> {sorted(dep)}" for scope, dep in offenders.items())
    )
