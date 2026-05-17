# kobato-eyes 品質・安定性ロードマップ

## 現状

- 基準チェック: `.\scripts\check.ps1`
- 直近の基準値: `357 passed, 39 deselected`
- 総カバレッジ: `78%`
- mypy対象: `45 source files`
- CI構成: unit, integration, gui-smoke, db-stress, package-smoke を分離済み
- 最大リスク:
  - `ui.tags_tab` と `ui.dup_tab` が巨大で、UI状態・worker・DB操作の境界が読みづらい。
  - `services.db_writing` と `core.pipeline` 周辺に broad `except Exception` が残っている。
  - 標準チェックは GUI / integration / db_stress を除外するため、変更内容に応じた追加確認が必要。

## 短期: DB / pipeline の失敗境界を固定する

- 目的: DB書き込み、FTS再構築、quiesce、checkpoint の失敗時に成功扱いへ流れないようにする。
- 対象: `core.pipeline.manual_refresh`, `core.pipeline.loaders`, `core.pipeline.stages.write_stage`, `services.db_writing`
- 完了条件:
  - 失敗を伝播すべき処理、best effort cleanup、環境依存 fallback の分類がコードまたはテストで明確になっている。
  - `docs/exception-audit-db-pipeline.md` の残タスクを順次解消する。
  - 意図的に握りつぶす cleanup 失敗はログ確認テストまたは近接コメントで理由を残す。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-integration.ps1`
  - `.\scripts\check-db-stress.ps1`

## 中期: UI worker と巨大タブを段階分割する

- 目的: UIフリーズ、キャンセル後の状態破損、worker例外の見落としを減らす。
- 対象: `ui.tags_tab`, `ui.dup_tab`, `ui.search_worker`, `ui.dup_workers`, `ui.thumbnail_tasks`, `ui.file_actions`
- 完了条件:
  - 検索、サムネイル、コピー、重複 scan/refine、trash/export を helper / viewmodel / worker 単位で小さく切り出す。
  - `tags_tab.py` と `dup_tab.py` は一括リライトせず、分割前に既存挙動を smoke / headless テストで固定する。
  - `dup_widgets.py`, `dup_workers.py`, `tag_stats.py`, `spinner_overlay.py`, `thumbnail_tasks.py`, `file_actions.py` の主要な成功系と失敗系をテストする。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`

## 長期: 型チェックとカバレッジの重点を広げる

- 目的: DB、pipeline、service で固めた型チェックを UI worker / viewmodel へ広げる。
- 完了条件:
  - 新規 core / db / service / worker モジュールは原則 mypy 対象へ追加する。
  - 既存の mypy 対象 `45 source files` を下回らない。
  - DB / pipeline / search worker は高水準のカバレッジを維持する。
  - 巨大 UI 本体は総カバレッジ数値より、分割後の境界テストと状態遷移テストを優先する。
- 検証:
  - `python -m mypy`
  - `python -m ruff check .`
  - `.\scripts\check-package-smoke.ps1`（import構造、tools、packaging、モジュール先頭に触れた場合）

## 環境依存チェック

- GPU / ONNX Runtime CUDA / open_clip / 推論backendに触る変更だけ `.\scripts\check-gpu.ps1` を使う。
- GPU精度改善やモデル更新は通常の品質改善とは別トラックで扱う。
- PyQt6実backend、`core.jobs`、QThread / QRunnable、pipeline、DB bootstrap、GUI / integration / smokeに触った場合は、該当する CI 再現コマンドを実行または案内する。

## 運用ルール

- Windows + Python 3.10を主対象にする。
- `PYTHONPATH=src` を前提にする。
- ロードマップ更新時は `.\scripts\check.ps1` の実測値を記録する。
- カバレッジは全体数値だけでなく、DB、pipeline、検索worker、UI worker の重要分岐を優先する。
- UI巨大ファイルは一括リライトせず、テストを追加してから小さく切り出す。
