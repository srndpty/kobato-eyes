from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from utils.paths import ensure_dirs, get_cache_dir


def get_search_results_root() -> Path:
    r"""
    AppData の kobato-eyes 配下に search_results ルートを作って返す。
    例: %APPDATA%\kobato-eyes\search_results
    """
    ensure_dirs()
    root = get_cache_dir() / "search_results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_for_folder(name: str, max_len: int = 60) -> str:
    """
    Windows フォルダ名に安全な文字だけ残す。空なら 'query'。
    """
    s = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    if not s:
        s = "query"
    return s[:max_len]


def make_export_dir(query: str) -> Path:
    """
    タイムスタンプ＋クエリでユニークな出力フォルダを作成して返す。
    例: 20250103-142233-dog_cat
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = sanitize_for_folder(query)
    dest = get_search_results_root() / f"{ts}-{safe}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest
