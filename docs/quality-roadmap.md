# kobato-eyes 品質・安定性ロードマップ

## 現状

- 更新日: 2026-05-17
- 基準チェック: `.\scripts\check.ps1`
- 直近の基準値: `357 passed, 39 deselected`
- 総カバレッジ: `78%`
- mypy対象: `50 source files`
- CI構成: unit, integration, gui-smoke, db-stress, package-smoke を分離済み
- 重点カバレッジ: `core.jobs`, `core.pipeline.retag`, `db.fts`, `ui.search_worker` は合計 `91%`

## 現在の主要リスク

- `ui.tags_tab` は約2000行、`ui.dup_tab` は約1200行で、UI状態、DB、worker制御、表示更新の境界がまだ密である。
- `services.db_writing`, `core.pipeline.manual_refresh`, `core.pipeline.loaders`, `dup.scanner`, UI worker 周辺に broad `except Exception` が残っている。
- 標準チェックは GUI / integration / db_stress を除外するため、変更内容に応じた追加確認を忘れると非同期・DBロック・実Qt backendの回帰を見落としやすい。
- カバレッジが低い実装境界として、`ui.index_tasks`, `ui.dup_workers`, `ui.tag_stats`, `ui.result_delegates`, `ui.widgets.spinner_overlay`, `utils.image_io`, `core.pipeline.watcher` が残っている。
- GPU / ONNX Runtime / open_clip は環境差が大きく、通常チェックとは別トラックで確認する必要がある。

## 短期: DB / pipeline の失敗境界を固定する

- 目的: DB書き込み、FTS再構築、manual refresh、quiesce、checkpoint の失敗時に成功扱いへ流れないようにする。
- 対象:
  - `core.pipeline.manual_refresh`
  - `core.pipeline.loaders`
  - `core.pipeline.stages.write_stage`
  - `services.db_writing`
- 完了条件:
  - 失敗を伝播すべき処理、best effort cleanup、環境依存 fallback の分類がコードまたはテストで明確になっている。
  - `docs/exception-audit-db-pipeline.md` の残タスクを順次解消する。
  - manual refresh の missing cleanup は、キャンセル後の部分処理件数を正しく stats に反映する。
  - `IndexRunnable.signals.error` まで、manual refresh / index pipeline の主要エラーが到達することをテストで固定する。
  - 意図的に握りつぶす cleanup 失敗はログ確認テストまたは近接コメントで理由を残す。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-integration.ps1`
  - `.\scripts\check-db-stress.ps1`

## 中期: UI worker と巨大タブを段階分割する

- 目的: UIフリーズ、キャンセル後の状態破損、worker例外の見落としを減らす。
- 対象:
  - `ui.index_tasks`
  - `ui.search_worker`
  - `ui.thumbnail_tasks`
  - `ui.file_actions`
  - `ui.dup_workers`
  - `ui.tags_tab`
  - `ui.dup_tab`
- 完了条件:
  - index/search/thumbnail/copy/duplicate scan/refine/trash/export を helper / viewmodel / worker 単位で小さく切り出す。
  - `tags_tab.py` と `dup_tab.py` は一括リライトせず、分割前に既存挙動を smoke / headless テストで固定する。
  - worker 内例外が silent failure にならず、UI 側で復帰可能な状態に戻る。
  - `dup_widgets.py`, `dup_workers.py`, `tag_stats.py`, `spinner_overlay.py`, `thumbnail_tasks.py`, `file_actions.py` の主要な成功系と失敗系をテストする。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`

## 中期: 画像 IO / 重複検出の堅牢性を上げる

- 目的: 壊れた画像、巨大画像、optional dependency 不足、キャンセル、並列 refine 失敗で全体処理を壊さない。
- 対象:
  - `utils.image_io`
  - `dup.scanner`
  - `dup.refine`
  - `ui.dup_refine_parallel`
- 完了条件:
  - Pillow / Qt fallback、破損ファイル、画像メタデータ不足、OpenCV不足の挙動をテストで固定する。
  - pHash / CLIP / SSIM / ORB の多段判定で、個別ファイル失敗がクラスタ全体の成功扱いを壊さない。
  - 並列 refine の progress / cancel / 例外集約がユーザーに追跡可能な形で返る。
- 検証:
  - `.\scripts\check.ps1`
  - 重複検出の実データ確認が必要な場合は専用 fixture または手動 smoke を別途実施する。

## 長期: 型チェックとカバレッジの重点を広げる

- 目的: DB、pipeline、service で固めた型チェックを UI worker / viewmodel / IO 境界へ広げる。
- 完了条件:
  - 新規 core / db / service / worker モジュールは原則 mypy 対象へ追加する。
  - 既存の mypy 対象 `50 source files` を下回らない。
  - `src/ui/index_tasks.py`, `src/ui/dup_workers.py`, `src/ui/file_actions.py`, `src/ui/viewmodels/dup_view_model.py`, `src/utils/image_io.py` は mypy 対象として維持する。
  - `src/ui/dup_refine_parallel.py` の未型付け関数に由来する mypy note を減らす。
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
- `docs/exception-audit-db-pipeline.md` は DB / pipeline 例外境界の作業メモとして維持し、完了した候補はロードマップと同期する。

## チェック選択

- 通常の高速確認: `.\scripts\check.ps1`
- PyQt6 / worker / GUI 状態変更: `.\scripts\check-gui-smoke.ps1`
- pipeline / DB bootstrap / end-to-end: `.\scripts\check-integration.ps1`
- WAL / SQLite lock / checkpoint / quiesce: `.\scripts\check-db-stress.ps1`
- import構造 / tools / packaging / module header: `.\scripts\check-package-smoke.ps1`
