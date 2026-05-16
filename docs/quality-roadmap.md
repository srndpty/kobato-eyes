# kobato-eyes 品質・安定性ロードマップ

## 現状

- 基準チェック: `.\scripts\check.ps1`
- 直近の基準値: `346 passed, 35 deselected`
- 総カバレッジ: `81%`
- mypy対象: 45ファイルへ段階拡大済み
- DBストレスチェック: `.\scripts\check-db-stress.ps1` 成功（`8 passed, 373 deselected`）
- Integrationチェック: `.\scripts\check-integration.ps1` 成功（`7 passed, 374 deselected`）

## 短期: データ整合性と検索安定性

- 目的: タグ書き込み後に検索結果が古いまま残る事故を防ぐ。
- 対象: `core.pipeline.stages.write_stage`, `services.db_writing`, `db.fts_offline`
- 完了条件:
  - DB writer失敗時に成功扱いの結果を返さない。
  - 高速書き込みでFTS更新を省略した後、オフラインFTS再構築が実行される。
  - SQLiteロック解除とFTS再構築がテストで固定されている。
- 検証:
  - `pytest tests/core/pipeline/test_write_stage.py -q`
  - `pytest tests/db/test_fts_offline.py -q`
  - `.\scripts\check-db-stress.ps1`
  - `.\scripts\check-integration.ps1`

## 中期: 型チェックと例外境界

- 目的: DB、pipeline、重複検出、検索UI workerの回帰を早期に検出する。
- 対象: `db/`, `core/pipeline/`, `dup/`, `sig/`, 小型 `ui/` ヘルパー
- 完了条件:
  - mypy対象を45ファイル以上に維持する。
  - 新規に追加するDB/pipeline/serviceモジュールは原則mypy対象に入れる。
  - `except Exception` はログ、ユーザー通知、復旧可否のいずれかを明確にする。
- 検証:
  - `python -m mypy`
  - `python -m ruff check .`

## 長期: UI分割と操作中断の堅牢化

- 目的: `tags_tab.py` と `dup_tab.py` の巨大クラスを縮小し、UIフリーズと状態破損を減らす。
- 対象: `ui.tags_tab`, `ui.dup_tab`, `ui.viewmodels`, `ui.search_worker`
- 完了条件:
  - DBアクセス、ファイル操作、表示モデル更新を小さなサービスまたはヘルパーへ分離する。
  - 検索、重複検出、サムネイル生成のキャンセル時にUI状態が復帰する。
  - GUI smokeテストで主要操作の空結果、例外、キャンセルを固定する。
- 検証:
  - `pytest tests/ui -q`
  - `pytest -m "gui or smoke" -q`

## 運用ルール

- Windows + Python 3.10を主対象にする。
- GPU/ONNX実推論の精度改善は別トラックで扱う。
- カバレッジは全体数値だけでなく、DB、pipeline、検索workerの重要分岐を優先する。
- UI巨大ファイルは一括リライトせず、テストを追加してから小さく切り出す。
