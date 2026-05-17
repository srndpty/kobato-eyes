# kobato-eyes 品質・安定性ロードマップ

## 現状

- 更新日: 2026-05-18
- 基準チェック: `.\scripts\check.ps1`
- 直近の基準値: `439 passed, 42 deselected`
- 総カバレッジ: `81%`
- 重点カバレッジ: `core.jobs`, `core.pipeline.retag`, `db.fts`, `ui.search_worker` 合計 `91%`
- mypy対象: `64 source files`
- GUI smoke: `27 passed, 449 deselected`
- db_stress: `8 passed, 441 deselected`
- GPU check: `1 passed, 1 skipped, 476 deselected` を直近の既知基準とする
- integration: `7 passed, 471 deselected` を直近の既知基準とする
- package smoke: compile package smoke OK

## 安定化済みの基盤

- DB / pipeline: shutdown sentinel、checkpoint、restore、progress callback、loader fallback、queue drain、path resolve fallback の主要境界はテスト済み。
- UI helper: index feedback、duplicate status、tags control state、cluster update、thumbnail task は切り出し済み。
- 重複検出: tilehash / pixel refine、SSIM / ORB fallback、破損画像混在、非正方形画像の pixel refine はテスト済み。
- 画像 IO: 壊れた画像、巨大画像、pixel cap 復元、thumbnail cache hit / copy / eviction はテスト済み。
- tagger backend: ONNX provider selection、model / labels file error、label count mismatch は `tagger.onnx_backend` で共通化済み。
- GPU check: CUDA provider smoke を追加し、CUDA provider 不在時は明示 skip、`onnx` package がある環境では dummy ONNX の WD14 実行 smoke を行う。
- GPU tagging: `tools\bench.py tagger` で 1000枚など固定件数の baseline を取得できる。PixAI は RTX4090 / batch 32 / prefetch 4 の実測で約37 images/sec。
- 型チェック: 主要 helper / worker / viewmodel / tagger utility を mypy 対象化済み。
- 型チェック: `tagger.wd14_onnx`, `tagger.pixai_onnx`, `core.pipeline.manual_refresh` を mypy 対象化済み。
- UI orchestration: index / refresh / retag lifecycle と duplicate scan / refine / trash / export の状態判定を pure helper へ切り出し済み。
- 低カバレッジ境界: result delegate、spinner、duplicate widgets、tag stats、watcher、image IO の重要分岐を helper 化し、軽量テストで固定済み。

## 現在の主要リスク

- `ui.tags_tab` と `ui.dup_tab` はまだ大きく、worker 制御、DB 接続復旧、表示更新、状態遷移が密である。
- `services.db_writing`, `core.pipeline.loaders`, `core.pipeline.manual_refresh`, `dup.scanner`, `tagger.wd14_onnx`, `tagger.pixai_onnx` の broad catch は failure policy を近接コメント、helper 名、テスト名、監査文書で分類済み。
- `ui.result_delegates`, `ui.widgets.spinner_overlay`, `ui.dup_widgets`, `ui.tag_stats`, `core.pipeline.watcher`, `utils.image_io` は低カバレッジ境界の重要分岐を一部固定済みだが、描画本体や実 widget 経路には追加余地がある。
- open_clip 依存の実行確認は通常チェックでは保証されない。ONNX Runtime CUDA provider は専用 GPU check で smoke 済み。
- 標準チェックは GUI / integration / db_stress を除外するため、変更内容に応じた追加確認が必要である。
- Settings の Device コンボは現在の tagger provider / ONNX provider 選択に接続されていないため、UI整理または設定スキーマ追加の余地がある。

## フェーズ 6: UI orchestration をさらに薄くする（実装済み）

- 目的: `tags_tab.py` / `dup_tab.py` を widget orchestration に寄せ、状態判定と worker lifecycle を小さくテストできる単位へ移す。
- 対象:
  - `src/ui/tags_tab.py`
  - `src/ui/dup_tab.py`
  - `src/ui/dup_workers.py`
  - `src/ui/viewmodels/*`
- 実装:
  - `src/ui/index_lifecycle.py` に index / refresh / retag の開始、キャンセル、完了、失敗、connection restore retry 判定を切り出した。
  - `src/ui/dup_lifecycle.py` に duplicate scan / refine / trash / export の状態判定と表示テキストを切り出した。
  - `tests/ui/test_index_lifecycle.py` と `tests/ui/test_dup_lifecycle.py` で worker 例外後の button / status / active task / connection state の復帰条件を pure unit test 化した。
- 検証済み:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`
  - `.\scripts\check-package-smoke.ps1`

## フェーズ 7: 例外境界を監査して failure policy を明文化する（実装済み）

- 目的: broad `except Exception` を、best effort、skip、retry、fatal のどれかに分類し、ログとテスト名で意図を固定する。
- 対象:
  - `src/services/db_writing.py`
  - `src/core/pipeline/loaders.py`
  - `src/core/pipeline/manual_refresh.py`
  - `src/dup/scanner.py`
  - `src/tagger/wd14_onnx.py`
  - `src/tagger/pixai_onnx.py`
- 実装:
  - `services.db_writing`, `core.pipeline.loaders`, `core.pipeline.manual_refresh`, `dup.scanner`, `tagger.wd14_onnx`, `tagger.pixai_onnx` の failure policy を must propagate、best effort、environment fallback、input skip、diagnostics only に分類した。
  - manual refresh の progress callback 失敗はログ後に進捗通知だけを無効化し、refresh 本体は継続するよう固定した。
  - duplicate scanner の malformed phash と PixAI metadata parse fallback をテストで固定した。
  - `docs/exception-audit-db-pipeline.md` と同期した。
- 検証済み:
  - `.\scripts\check.ps1`
  - `.\scripts\check-integration.ps1`
  - `.\scripts\check-package-smoke.ps1`
  - `.\scripts\check-gpu.ps1`（環境依存。CUDA provider smoke / dummy ONNX smoke を実行または skip）

## フェーズ 8: 低カバレッジ UI / IO 境界を埋める（実装済み）

- 目的: 描画 delegate、spinner、duplicate widgets、tag stats、watcher、image IO の実事故に近い分岐を軽量に固定する。
- 対象:
  - `src/ui/result_delegates.py`
  - `src/ui/widgets/spinner_overlay.py`
  - `src/ui/dup_widgets.py`
  - `src/ui/tag_stats.py`
  - `src/core/pipeline/watcher.py`
  - `src/utils/image_io.py`
- 実装:
  - `ui.result_delegates` の grid caption / dark background 判定を helper 化した。
  - `ui.widgets.spinner_overlay` の default message / geometry 判定を helper 化した。
  - `ui.dup_widgets` の duplicate tile metadata / panel layout 計算を helper 化した。
  - `ui.tag_stats` の category / score / threshold merge を helper 化した。
  - `core.pipeline.watcher` の enqueue 対象 path 解決を helper 化し、モジュール先頭 docstring を追加した。
  - `utils.image_io.safe_load_image` に EXIF transpose と alpha の白背景合成を追加し、巨大画像 skip / close 漏れと合わせてテストした。
- 検証済み:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`
  - `.\scripts\check-integration.ps1`
  - `.\scripts\check-package-smoke.ps1`

## フェーズ 9: tagger backend と GPU 経路の実テストを作る（実装済み）

- 目的: mocked provider 計画だけでなく、環境がある場合に ONNX Runtime CUDA / CPU fallback / input layout を実行確認できるようにする。
- 対象:
  - `src/tagger/wd14_onnx.py`
  - `src/tagger/pixai_onnx.py`
  - `src/tagger/onnx_backend.py`
  - `scripts/check-gpu.ps1`
- 実装:
  - `tests/tagger/test_gpu_onnx_runtime.py` に `@pytest.mark.gpu` の CUDA provider smoke を追加した。
  - CUDA provider 不在時は provider 一覧付きで skip する。
  - `onnx` package がある環境では小型 dummy ONNX を生成し、`WD14Tagger` の input name / output tensor / label count / inference output を CUDA provider で確認する。
  - `onnx` package がない環境では dummy model smoke のみ skip し、CUDA provider smoke は実行する。
- 検証済み:
  - `.\scripts\check-gpu.ps1`（`1 passed, 1 skipped, 476 deselected`）

## フェーズ 10: 型チェック対象を backend 本体へ広げる（実装済み）

- 目的: helper だけでなく、tagger backend と残る UI controller の public surface を型で保護する。
- 優先候補:
  - `src/tagger/wd14_onnx.py`
  - `src/tagger/pixai_onnx.py`
  - `src/core/pipeline/manual_refresh.py`
  - 新しく切り出す UI controller / helper
- 実装:
  - `src/tagger/wd14_onnx.py` を mypy 対象化した。
  - `src/tagger/pixai_onnx.py` を mypy 対象化した。
  - `src/core/pipeline/manual_refresh.py` を mypy 対象化した。
  - Optional connection / cursor、ONNX provider request、threshold mapping、PIL / ndarray 境界を型で追えるように局所修正した。
- 完了:
  - mypy 対象は `64 source files`。
  - `.\.venv\Scripts\python.exe -m mypy` は `Success: no issues found in 64 source files`。
- 検証済み:
  - `.\scripts\check.ps1`
  - `.\scripts\check-integration.ps1`
  - `.\scripts\check-package-smoke.ps1`
  - `.\scripts\check-gpu.ps1`

## 運用ルール

- Windows + Python 3.10 を主対象にする。
- `PYTHONPATH=src` を前提にする。
- ロードマップ更新時は `.\scripts\check.ps1` の実測値を記録する。
- カバレッジは全体数値より、DB、pipeline、検索 worker、UI worker、画像 IO、重複 refine の重要分岐を優先する。
- UI 巨大ファイルは一括リライトせず、テストを追加してから 1 境界ずつ切り出す。
- GPU / CUDA / open_clip は環境依存なので通常チェックの必須条件に混ぜず、専用チェックに分ける。
- リリース成果物は `.\scripts\package-release.ps1` で PyInstaller build、7z圧縮、SHA256生成をまとめて行う。

## チェック選択

- 通常の高速確認: `.\scripts\check.ps1`
- PyQt6 / worker / GUI 状態変更: `.\scripts\check-gui-smoke.ps1`
- pipeline / DB bootstrap / end-to-end: `.\scripts\check-integration.ps1`
- WAL / SQLite lock / checkpoint / quiesce: `.\scripts\check-db-stress.ps1`
- import構造 / tools / packaging / module header: `.\scripts\check-package-smoke.ps1`
- GPU / ONNX Runtime CUDA / open_clip: `.\scripts\check-gpu.ps1`
