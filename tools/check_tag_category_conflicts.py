"""Report tags whose PixAI v1 category differs from the category stored in an existing DB.

PixAI v1 導入前の確認用. タグのカテゴリは後勝ちで上書きされる（db.tags 参照）ため、
競合タグは既存画像も含めて新しいカテゴリで検索・表示されるようになる. 変わるタグと影響画像数を事前に確認する.
DB は読み取り専用で開く.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from db.tags import find_category_conflicts
from tagger.base import TagCategory
from tagger.pixai_v1_labels import load_labels_json


def _category_name(value: int) -> str:
    """Old DBs may hold IDs outside the current enum."""
    try:
        return TagCategory(value).name.lower()
    except ValueError:
        return str(value)


def main() -> int:
    """Print conflicts as JSON; exit 1 when any exist so scripts can gate a full retag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50, help="max rows to print")
    args = parser.parse_args()
    labels = load_labels_json(args.labels_json)
    conn = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        conflicts = find_category_conflicts(conn, ((label.name, int(label.category)) for label in labels))
    finally:
        conn.close()
    conflicts.sort(key=lambda row: -row[3])
    report = dict(
        conflicts=len(conflicts),
        affected_assignments=sum(row[3] for row in conflicts),
        rows=[
            dict(name=name, stored=_category_name(stored), v1=_category_name(new), files=files)
            for name, stored, new, files in conflicts[: args.limit]
        ],
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
