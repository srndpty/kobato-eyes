# scripts/find_magic_numbers.py
from __future__ import annotations

import ast
import pathlib
import re
import sys

ALLOW_VALUES = {0, 1, -1, 2}
ALLOW_FLOAT_PAT = re.compile(r"^1e-?\d+$", re.I)  # 学習率などの 1e-5 風は許す
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "src")


def is_module_constant(assign: ast.Assign) -> bool:
    # UPPER_CASE = <number> をモジュール直下の定数として許可
    return all(isinstance(t, ast.Name) and t.id.isupper() for t in assign.targets)


def visit_file(path: pathlib.Path) -> None:
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return
    try:
        tree = ast.parse(src, filename=str(path))
    except Exception:
        return

    module_level = True
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and module_level and isinstance(node.value, (ast.Num, ast.UnaryOp)):
            if is_module_constant(node):
                continue

        if isinstance(node, ast.Compare):
            # 比較に現れるリテラル（Ruff PLR2004 に近い）
            to_check = [node.left, *node.comparators]
        else:
            to_check = [node]

        for n in to_check:
            val = None
            if isinstance(n, ast.Num):
                val = n.n
            elif (
                isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)) and isinstance(n.operand, ast.Num)
            ):
                val = -n.operand.n if isinstance(n.op, ast.USub) else n.operand.n

            if val is None:
                continue
            # 許容値のフィルタ
            if isinstance(val, int) and val in ALLOW_VALUES:
                continue
            if isinstance(val, float) and (val in {0.0, 1.0} or ALLOW_FLOAT_PAT.match(str(val))):
                continue

            lineno = getattr(n, "lineno", 0)
            print(f"{path}:{lineno}: magic-number -> {val}")


if __name__ == "__main__":
    for p in ROOT.rglob("*.py"):
        if any(seg in {"venv", ".venv", "__pycache__", "build", "dist", "migrations", "tests"} for seg in p.parts):
            continue
        visit_file(p)
