# kobato-eyes 品質・安定性ロードマップ

## 現状

- 更新日: 2026-05-17
- 基準チェック: `.\scripts\check.ps1`
- 直近の基準値: `410 passed, 40 deselected`
- 総カバレッジ: `80%`
- 重点カバレッジ: `core.jobs`, `core.pipeline.retag`, `db.fts`, `ui.search_worker` 合計 `91%`
- mypy対象: `59 source files`
- GUI smoke: `27 passed, 423 deselected`
- db_stress: `8 passed, 441 deselected`
- GPU check: GPU test 未定義時は skip 扱い
- integration: `7 passed, 414 deselected` を直近の既知基準とする
- package smoke: compile package smoke OK

## 安定化済みの基盤

- DB / pipeline: shutdown sentinel、checkpoint、restore、progress callback、loader fallback、queue drain、path resolve fallback の主要境界はテスト済み。
- UI helper: index feedback、duplicate status、tags control state、cluster update、thumbnail task は切り出し済み。
- 重複検出: tilehash / pixel refine、SSIM / ORB fallback、破損画像混在、非正方形画像の pixel refine はテスト済み。
- 画像 IO: 壊れた画像、巨大画像、pixel cap 復元、thumbnail cache hit / copy / eviction はテスト済み。
- tagger backend: ONNX provider selection、model / labels file error、label count mismatch は `tagger.onnx_backend` で共通化済み。
- 型チェック: 主要 helper / worker / viewmodel / tagger utility を mypy 対象化済み。

## 現在の主要リスク

- `ui.tags_tab` と `ui.dup_tab` はまだ大きく、worker 制御、DB 接続復旧、表示更新、状態遷移が密である。
- `services.db_writing`, `core.pipeline.loaders`, `core.pipeline.manual_refresh`, `dup.scanner`, `tagger.wd14_onnx`, `tagger.pixai_onnx` には broad `except Exception` が残る。
- `ui.result_delegates`, `ui.widgets.spinner_overlay`, `ui.dup_widgets`, `ui.tag_stats`, `core.pipeline.watcher`, `utils.image_io` はカバレッジが低い。
- tagger 実 backend、GPU / CUDA provider、open_clip 依存の実行確認は通常チェックでは保証されない。
- 標準チェックは GUI / integration / db_stress を除外するため、変更内容に応じた追加確認が必要である。

## 次フェーズ 6: UI orchestration をさらに薄くする

- 目的: `tags_tab.py` / `dup_tab.py` を widget orchestration に寄せ、状態判定と worker lifecycle を小さくテストできる単位へ移す。
- 対象:
  - `src/ui/tags_tab.py`
  - `src/ui/dup_tab.py`
  - `src/ui/dup_workers.py`
  - `src/ui/viewmodels/*`
- 作業:
  - `tags_tab.py` から index / refresh task lifecycle と connection restore retry を helper または controller へ切り出す。
  - `dup_tab.py` から scan lifecycle、refine dialog lifecycle、trash / export 後の表示更新を段階的に切り出す。
  - 切り出し先は pure unit test を先に追加し、PyQt 実 widget 依存は smoke test に閉じ込める。
- 完了条件:
  - UI worker 例外後の button / status / active task / connection state の復帰条件が単体テストで読める。
  - `tags_tab.py` / `dup_tab.py` に新しい集計ロジックや文言生成を追加しない。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`

## 次フェーズ 7: 例外境界を監査して failure policy を明文化する

- 目的: broad `except Exception` を、best effort、skip、retry、fatal のどれかに分類し、ログとテスト名で意図を固定する。
- 対象:
  - `src/services/db_writing.py`
  - `src/core/pipeline/loaders.py`
  - `src/core/pipeline/manual_refresh.py`
  - `src/dup/scanner.py`
  - `src/tagger/wd14_onnx.py`
  - `src/tagger/pixai_onnx.py`
- 作業:
  - broad catch ごとに failure policy を近接コメントまたは小 helper 名で表す。
  - skip してよい入力不正と呼び出し元へ返すべき実行失敗をテストで分ける。
  - DB / pipeline 変更では `docs/exception-audit-db-pipeline.md` と同期する。
- 完了条件:
  - silent failure が残らず、ユーザーに返る件数、ログ、progress が処理実態と一致する。
  - 新しい broad catch 追加時は対応テストまたは明示的な理由がある。
- 検証:
  - `.\scripts\check.ps1`
  - pipeline / DB に触る場合: `.\scripts\check-integration.ps1`
  - WAL / lock / checkpoint に触る場合: `.\scripts\check-db-stress.ps1`

## 次フェーズ 8: 低カバレッジ UI / IO 境界を埋める

- 目的: 描画 delegate、spinner、duplicate widgets、tag stats、watcher、image IO の実事故に近い分岐を軽量に固定する。
- 対象:
  - `src/ui/result_delegates.py`
  - `src/ui/widgets/spinner_overlay.py`
  - `src/ui/dup_widgets.py`
  - `src/ui/tag_stats.py`
  - `src/core/pipeline/watcher.py`
  - `src/utils/image_io.py`
- 作業:
  - pure helper 化できる計算、表示テキスト、状態遷移を先に抽出して unit test を追加する。
  - PyQt 描画は最小 smoke に留め、ピクセル完全一致に依存しない確認にする。
  - image IO は EXIF transpose、RGBA / palette、巨大画像 skip、close 漏れを追加 fixture で確認する。
- 完了条件:
  - 低カバレッジモジュールの重要分岐が、少なくとも skip / error / success の 3 系統で確認できる。
  - UI 描画変更で `check-gui-smoke.ps1` が必須であることが PR / レビューで判断できる。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`

## 次フェーズ 9: tagger backend と GPU 経路の実テストを作る

- 目的: mocked provider 計画だけでなく、環境がある場合に ONNX Runtime CUDA / CPU fallback / input layout を実行確認できるようにする。
- 対象:
  - `src/tagger/wd14_onnx.py`
  - `src/tagger/pixai_onnx.py`
  - `src/tagger/onnx_backend.py`
  - `scripts/check-gpu.ps1`
- 作業:
  - `@pytest.mark.gpu` の最小 smoke を追加し、CUDA provider がない環境では明示 skip する。
  - 小型 dummy ONNX または session double で input name / output shape / label mismatch を追加確認する。
  - GPU test 未定義時 skip の運用を維持し、実 GPU test 追加後は失敗を正しく検出する。
- 完了条件:
  - RTX 環境で `check-gpu.ps1` が実テストを少なくとも 1 件実行する。
  - GPU 不在環境では skip 理由が明確に表示される。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gpu.ps1`

## 次フェーズ 10: 型チェック対象を backend 本体へ広げる

- 目的: helper だけでなく、tagger backend と残る UI controller の public surface を型で保護する。
- 優先候補:
  - `src/tagger/wd14_onnx.py`
  - `src/tagger/pixai_onnx.py`
  - `src/core/pipeline/manual_refresh.py`
  - 新しく切り出す UI controller / helper
- 作業:
  - まず小 helper を対象へ追加し、型エラーが多い巨大モジュールは分割後に対象化する。
  - Protocol / dataclass / TypedDict を使い、PyQt object や ONNX session の境界は必要最小限に抽象化する。
  - mypy 対象数を減らさず、新規 core / db / service / worker / helper は原則対象に含める。
- 完了条件:
  - mypy 対象が `59 source files` を下回らない。
  - 新規 helper の public 関数は型注釈付きで、未型付け関数 note を増やさない。
- 検証:
  - `.\.venv\Scripts\python.exe -m mypy`
  - `.\scripts\check.ps1`
  - import構造に触った場合: `.\scripts\check-package-smoke.ps1`

## 運用ルール

- Windows + Python 3.10 を主対象にする。
- `PYTHONPATH=src` を前提にする。
- ロードマップ更新時は `.\scripts\check.ps1` の実測値を記録する。
- カバレッジは全体数値より、DB、pipeline、検索 worker、UI worker、画像 IO、重複 refine の重要分岐を優先する。
- UI 巨大ファイルは一括リライトせず、テストを追加してから 1 境界ずつ切り出す。
- GPU / CUDA / open_clip は環境依存なので通常チェックの必須条件に混ぜず、専用チェックに分ける。

## チェック選択

- 通常の高速確認: `.\scripts\check.ps1`
- PyQt6 / worker / GUI 状態変更: `.\scripts\check-gui-smoke.ps1`
- pipeline / DB bootstrap / end-to-end: `.\scripts\check-integration.ps1`
- WAL / SQLite lock / checkpoint / quiesce: `.\scripts\check-db-stress.ps1`
- import構造 / tools / packaging / module header: `.\scripts\check-package-smoke.ps1`
- GPU / ONNX Runtime CUDA / open_clip: `.\scripts\check-gpu.ps1`
