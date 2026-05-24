# kobato-eyes 開発ガイド

このドキュメントは、kobato-eyes に貢献する開発者向けの統一的なガイドです。AGENTS.md と組み合わせて参照してください。

---

## プロジェクト概要

**kobato-eyes** は、Windows 向けの統合デスクトップアプリケーションです：
- **目的**: ローカル PC の画像に Danbooru タグを自動付与し、類似画像検出・検索を実現
- **スタック**: PyQt6 (GUI) + SQLite + FTS5 (検索) + ONNX Runtime (推論) + scikit-image (画像処理)
- **規模**: ~15k LOC (102 Python ファイル)、88 型チェック対象、477 自動テスト
- **目標**: 新規貢献者のオンボーディング時間短縮、テスト品質向上、保守性向上

---

## 環境構築

### Python 環境の確認
```powershell
python --version  # Python 3.10.x であることを確認
```

### 仮想環境の構築
```powershell
# Windows PowerShell
git clone https://github.com/srndpty/kobato-eyes.git
cd kobato-eyes
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 依存関係インストール
python -m pip install --upgrade pip
python -m pip install -c requirements-dev.lock -e ".[dev,tagging-cpu]"
# GPU環境の場合: python -m pip install -c requirements-dev.lock -e ".[dev,tagging-gpu]"

# Pre-commit フック インストール
pre-commit install
```

### 環境変数（開発時）

| 変数 | 値 | 用途 |
|-----|-----|------|
| `PYTHONPATH` | `src` | モジュール import 用 (必須) |
| `KOE_HEADLESS` | `1` | GUI 初期化をスキップ（テスト環境）|
| `KOE_LOG_LEVEL` | `DEBUG` or `INFO` | ログレベル指定 |
| `KOE_DATA_DIR` | `./data` | AppData の代わりにローカルパス使用 |

**PowerShell 環境変数設定例**:
```powershell
$env:PYTHONPATH = "src"
$env:KOE_HEADLESS = "1"
$env:KOE_LOG_LEVEL = "DEBUG"
```

---

## コード規約

### 型ヒント
**必須**: すべての関数パラメータと戻り値に型ヒントを付与

```python
# ✓ Good
from __future__ import annotations
from typing import Optional, Sequence

def ensure_signatures(
    conn,
    file_id: int,
    *,
    image: Optional[Image.Image] = None,
    path: Optional[str | Path] = None,
    force: bool = False,
) -> bool:
    """Docstring here."""

# ✗ Bad
def ensure_signatures(conn, file_id, image=None, path=None, force=False):
    pass
```

### Protocol と型安全性
複数の実装を持つ場合、Protocol を使用して構造的部分型を定義

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ITagger(Protocol):
    """推論エンジンの統一インターフェース."""
    def prepare_batch_from_rgb_np(self, images: Sequence[np.ndarray]) -> np.ndarray: ...
    def infer_batch_prepared(...) -> list[TagResult]: ...
```

### Docstring
**原則**: 関数の **WHY**（なぜこの実装か）のみを記述。**WHAT**（何をするか）は関数名と型で明示

```python
# ✓ Good
def ensure_signatures(conn, file_id: int, *, image=None, path=None, force=False) -> bool:
    """
    画像署名を DB に保存。既存署名がある場合は force=True でのみ上書き。
    force=False のとき、既存署名があれば早期リターンで重複計算を避ける。
    """

# ✗ Bad (WHATを繰り返している)
def ensure_signatures(conn, file_id: int) -> bool:
    """Ensure signatures for file_id."""
    pass
```

### コメント
- **日本語 OK**: リスク警告、非自明な制約は日本語でも良い
- **デフォルト日本語**: AGENTS.md に従い、実装と議論は日本語が基本
- **短く**: 多行コメント（docstring 除外）は避ける

```python
# ✓ Good: リスク警告
# WAL モード OFF → 並行 UPDATE でロック競合リスク. 本番環境では WAL=ON 必須

# ✗ Bad: 明白な実装を説明
# file_id を 1 ずつ増やす
file_id += 1
```

---

## モジュール構成ガイド

| モジュール | ファイル数 | 責務 |
|-----------|----------|------|
| **core/** | 8 | パイプライン、ジョブ管理、スキャナー、署名計算 |
| **ui/** | 26 | PyQt6 UI コンポーネント（メインスレッド）、ビューモデル |
| **db/** | 10 | SQLite スキーマ、接続管理、リポジトリ、FTS5、タグ操作 |
| **tagger/** | 9 | ONNX ベース推論エンジン（WD14/PixAI） |
| **services/** | 2 | DB 書き込みサービス、バックグラウンド処理 |
| **dup/** | 4 | 重複検出、pHash、SSIM/ORB 判定 |
| **sig/** | 2 | phash、dhash 画像署名 |
| **utils/** | 7 | 画像 I/O、ファイルシステム、環境判定 |
| **index/** | 1 | ベクトル検索基盤（将来拡張用） |

### 各モジュールの主要ファイル

**core/**
- `jobs.py` — QThread/QRunnable、非同期ジョブ管理
- `pipeline/` — パイプライン本体（scan→tag→write ステージ）
- `config/` — 設定管理（YAML パース、バリデーション）

**ui/**
- `app.py` — メインウィンドウ、タブ管理
- `viewmodels/` — MVVM パターンのビューモデル
- `*_tab.py` — 各タブの UI

**db/**
- `connection.py` — SQLite 接続、quiesce 管理
- `schema.py` — テーブル定義、初期化
- `repository.py` — CRUD 操作
- `fts.py` — FTS5 全文検索

---

## 開発スタイル

### 非同期処理（UI からの長時間実行タスク）

**原則**: PyQt6 メインスレッドで UI 操作、QThread/QRunnable でバックグラウンドタスク

```python
from core.jobs import JobManager, QRunnable

class MyLongTask(QRunnable):
    """バックグラウンドで実行するタスク."""
    
    def __init__(self, param):
        super().__init__()
        self.param = param
    
    def run(self):
        """UI スレッドではない別スレッドで実行される."""
        try:
            result = expensive_computation(self.param)
            # UI 更新は signal を通じて main thread へ
        except Exception as e:
            logger.error(f"Task failed: {e}")

# メインスレッドから起動
job_manager = JobManager()
task = MyLongTask(param_value)
job_manager.submit(task)
```

### DB 並行アクセス保護（quiesce パターン）

**原則**: 複数の書き込みが衝突する操作中は、`quiesced()` で読み込みを一時停止

```python
from db.connection import quiesced, get_conn

# 複数の INSERT/UPDATE を安全に実行
with quiesced():
    conn = get_conn(db_path)
    conn.executemany("INSERT INTO images ...", rows)
    conn.commit()
    # この区間中は他の読み込みが待機

# 書き込み外の通常読み込み
conn = get_conn(db_path)
results = conn.execute("SELECT ...").fetchall()
```

### エラーハンドリング

**パターン 1: 安全な降格（fallback）**
```python
try:
    image = Image.open(image_path)
except (UnidentifiedImageError, OSError) as e:
    logger.warning(f"Failed to load {image_path}: {e}")
    return False  # 無視して続行
```

**パターン 2: 非致命的エラー抑止**
```python
from contextlib import suppress

with suppress(Exception):
    cleanup_temp_files()  # 失敗してもプログラムは続行
```

**パターン 3: ログして再スロー**
```python
try:
    critical_operation()
except ValueError as e:
    logger.error(f"Critical operation failed: {e}", exc_info=True)
    raise
```

### Headless 対応

PyQt6 は表示がない環境で初期化に失敗する。`utils.env.is_headless()` で判定

```python
from utils.env import is_headless

if not is_headless():
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
else:
    # Headless モード: PyQt6 初期化スキップ
    logger.info("Running in headless mode")
```

---

## テスト方針

### スクリプト別の実行チェック

| PowerShell スクリプト | 実行内容 | 実行条件 |
|-----------|----------|---------|
| `.\scripts\check.ps1` | 統合テスト (isort/ruff/mypy/pytest) + カバレッジ | **全コミット前に実行** |
| `.\scripts\check-gui-smoke.ps1` | GUI/smoke テスト | UI モジュール変更時 |
| `.\scripts\check-db-stress.ps1` | DB 並行書き込みストレステスト | db/services 変更時 |
| `.\scripts\check-gpu.ps1` | GPU 推論テスト | tagger/推論 変更時（RTX4090 環境推奨） |
| `.\scripts\check-integration.ps1` | E2E パイプラインテスト | core/pipeline 変更時 |
| `.\scripts\check-package-smoke.ps1` | パッケージ smoke テスト | 依存関係・モジュール構成 変更時 |

### テスト実行例

```powershell
# デフォルト（gui/integration/db_stress を除外）
$env:PYTHONPATH = "src"
$env:KOE_HEADLESS = "1"
pytest -q

# カバレッジ確認（目標 85%）
pytest --cov=src --cov-report=html --cov-report=term-missing

# GUI テストのみ（ディスプレイ必須）
Remove-Item Env:KOE_HEADLESS -ErrorAction SilentlyContinue
$env:PYTHONPATH = "src"
pytest -m "gui" -p no:cov

# 特定モジュール（db/）のテスト
pytest tests/db/ -v
```

### テストマーカー

| マーカー | 説明 | 使用例 |
|---------|------|--------|
| `@pytest.mark.gui` | PyQt6/ディスプレイ必須 | UI 操作、イベント処理 |
| `@pytest.mark.integration` | パイプライン E2E | scan → tag → DB 保存 |
| `@pytest.mark.db_stress` | SQLite WAL/並行書き込み | DB ロック、checkpoint |
| `@pytest.mark.gpu` | CUDA/GPU 推論 | 実 ONNX Runtime テスト |
| `@pytest.mark.smoke` | 軽量起動・パッケージング | 依存関係、import テスト |

---

## PyQt6 特有の注意

以下の import 位置は PyQt5 と異なります：

```python
# ✓ PyQt6 (正)
from PyQt6.QtGui import QFileSystemModel, QAction, QMimeData
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

# ✗ PyQt5 style (PyQt6 では NG)
from PyQt6.QtWidgets import QFileSystemModel  # NOT here
```

### 存在しないクラス・メソッド

| 対象 | 代替案 |
|-----|--------|
| `PyQt6.QtConcurrent` | QThreadPool + QRunnable を使用 |
| `QMediaPlayer.setMuted()` | QAudioOutput を別途管理 |

参照: `src/core/jobs.py` (headless 互換の実装例)

---

## Danbooru タグ仕様に関する前提

このプロジェクトで扱う Danbooru タグについて、以下は存在しないことを前提としています：

- `-` から始まるタグ
- `(` または `)` から始まるタグ
- 空白文字を含むタグ

したがって、クエリ構文やオートコンプリート実装では、これらを「タグ名として解釈すべきケース」として単独で考慮しません。ただし、既存実装の他機能に影響する場合、または外部入力の安全性・例外処理として必要な場合は別途考慮してください。

---

## 既知の制限と設計判断

### モデル出力判定

WD14 / PixAI の判定は `selected_tags.csv` から自動判定。モデルサーバーのダウンデートやモデルの入れ替わりには `model_inspection.py` が対応します。

---

## 参考資料

### ドキュメント
- `AGENTS.md` — エージェント向けの実装原則（完全版）
- `docs/quality-roadmap.md` — フェーズ別の品質改善履歴
- `docs/exception-audit-db-pipeline.md` — 例外処理の分類・監査

### ソースコード例
- **非同期**: `src/core/jobs.py` (JobManager), `src/ui/search_worker.py` (QRunnable)
- **DB 安全性**: `src/db/connection.py` (quiesce), `src/services/db_writing.py` (DBWritingService)
- **型安全性**: `src/tagger/base.py` (Protocol), `src/core/pipeline/types.py` (dataclass)
- **UI MVVM**: `src/ui/viewmodels/tags_view_model.py`

### 外部リソース
- [PyQt6 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [Python typing](https://docs.python.org/3.10/library/typing.html)
- [SQLite WAL](https://www.sqlite.org/wal.html)

---

## よくある質問（FAQ）

**Q: なぜ `KOE_HEADLESS=1` が必要？**  
A: PyQt6 は表示がない環境で `QApplication` 初期化に失敗します。テスト環境では headless 互換の stub を使用します。

**Q: tagger/onnx_backend.py を変更した。何をテストすべき？**  
A: `.\scripts\check-gpu.ps1` で GPU テストを実行（RTX4090 環境）。GPU なし環境なら `tests/tagger/test_dummy.py` で mock テストを追加。

**Q: DB write が遅い。quiesce() はなぜ必要？**  
A: SQLite は並行 UPDATE に弱く、WAL モードでも checkpoint 中のロック競合がある。quiesce() で読み込みを一時停止することで書き込みを優先。

**Q: 新しいテストを追加した。カバレッジはどこまで目標？**  
A: 85% が目標（現在 85.18%）。UI/packaging は除外対象（実装復雑度 vs テスト コスト）。

---

**最後に**: このガイドに矛盾や不足があれば issue 報告または PR お願いします。
