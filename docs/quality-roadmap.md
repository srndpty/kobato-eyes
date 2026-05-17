# kobato-eyes 品質・安定性ロードマップ

## 現状

- 更新日: 2026-05-17
- 基準チェック: `.\scripts\check.ps1`
- 直近の基準値: `392 passed, 40 deselected`
- 総カバレッジ: `80%`
- mypy対象: `53 source files`
- GUI smoke: `27 passed, 405 deselected`
- integration: `7 passed, 414 deselected`
- db_stress: `8 passed, 413 deselected`
- package smoke: compile package smoke OK
- 重点カバレッジ: `core.jobs`, `core.pipeline.retag`, `db.fts`, `ui.search_worker` は合計 `91%`

## ここまでで改善済み

- `manual_refresh` の missing cleanup は、キャンセル後に実際に処理した soft/hard delete 件数を stats に返す。
- フェーズ1: `DBWritingService` の shutdown sentinel、checkpoint、restore、progress callback は best effort 失敗としてログレベル込みで分類済み。
- フェーズ1: `core.pipeline.loaders` は壊れた個別画像をスキップし、producer / IO worker 失敗は iterator で伝播することをテストで固定済み。
- フェーズ1: `ProcessingPipeline` は path resolve 失敗時の fallback と stop / shutdown / finished 後の scheduled queue 解放をテストで固定済み。
- フェーズ2: `tags_tab.py` の control enabled 判定を `ui.tags_control_state` へ切り出し、処理中状態ごとのボタン復帰条件を pure test で固定済み。
- フェーズ2: `dup_tab.py` の trash 後クラスタ再構築を `ui.dup_cluster_update` へ切り出し、keeper 選択・表示順・singleton drop を単体テストで固定済み。
- フェーズ2: `ui.thumbnail_tasks` は cancel と壊れた画像時の空 pixmap emit をテスト済み。重複検出サムネ queue の重複投入防止と `ThumbJob` の壊れた画像 path もテスト済み。
- `IndexRunnable` は pre-run / runner 例外を `signals.error` に流し、失敗時に `signals.finished` を出さないことをテストで固定済み。
- `tags_tab.py` から index / refresh の feedback 生成を `ui.index_feedback` へ切り出し済み。
- `dup_tab.py` から duplicate status 生成を `ui.dup_status` へ切り出し済み。
- `utils.image_io` は巨大画像 skip、壊れた画像、pixel cap 復元、RGB変換時 close をテスト済み。
- `dup.scanner` は BLOB / hex pHash、embedding 不一致時の fallback をテスト済み。
- `DuplicateScanRunnable` は不正 row skip と file_id 型不正時の error signal をテスト済み。
- mypy 対象を `45 source files` から `53 source files` に拡張済み。

## 現在の主要リスク

- `ui.tags_tab` は約1950行、`ui.dup_tab` は約1170行で、まだ UI 状態、worker 制御、DB 接続復旧、表示更新が密である。
- `services.db_writing`, `core.pipeline.loaders`, `core.pipeline.manual_refresh`, `dup.scanner`, `ui.dup_refine_parallel`, tagger backend 周辺に broad `except Exception` が残っている。
- `ui.result_delegates`, `ui.widgets.spinner_overlay`, `ui.dup_widgets`, `ui.tag_stats`, `core.pipeline.watcher`, `utils.image_io` はカバレッジが低く、UI描画・非同期・IO境界の回帰を見落としやすい。
- `tagger.wd14_onnx` と `tagger.pixai_onnx` は大きく、環境依存 fallback と推論 backend の品質確認が通常チェックから外れやすい。
- 標準チェックは GUI / integration / db_stress を除外するため、変更内容に応じた追加確認が必須である。

## 完了フェーズ 1: DB / pipeline の残った失敗境界を閉じる

- 目的: DB writer shutdown、loader fallback、watcher、manual refresh cleanup の失敗分類を最後まで明確にする。
- 対象:
  - `services.db_writing`
  - `core.pipeline.loaders`
  - `core.pipeline.watcher`
  - `core.pipeline.manual_refresh`
- 作業:
  - `DBWritingService.stop`, `_maybe_checkpoint`, `_restore_normal_mode`, `_emit_progress` の best effort failure をログレベル込みで分類する。完了。
  - `core.pipeline.loaders` の optional decoder fallback、壊れた画像、producer thread failure を追加テストする。完了。
  - `core.pipeline.watcher` の start/stop、例外、キャンセル、queue drain をテストする。`ProcessingPipeline` の enqueue / stop / shutdown / finished queue drain と resolve fallback を headless test で固定済み。
  - `docs/exception-audit-db-pipeline.md` は DB / pipeline 例外境界の作業台帳として継続更新する。
- 完了条件:
  - 成功扱いにしてよい cleanup と、呼び出し元へ伝播すべき失敗がテスト名または近接コメントで読める。完了。
  - `check.ps1`, `check-integration.ps1`, 必要に応じて `check-db-stress.ps1` が通る。実行結果はこのファイルの「現状」を更新して記録する。

## 進行中フェーズ 2: 巨大 UI ファイルを状態単位でさらに分割する

- 目的: `tags_tab.py` / `dup_tab.py` の変更リスクを下げ、UI 状態遷移を小さな単位でテストできるようにする。
- 対象:
  - `ui.tags_tab`
  - `ui.dup_tab`
  - `ui.index_feedback`
  - `ui.dup_status`
  - `ui.dup_workers`
  - `ui.thumbnail_tasks`
- 作業:
  - `tags_tab.py` から index task lifecycle、refresh folder selection、connection restore retry を小さな helper / controller に分ける。control enabled 判定は `ui.tags_control_state` へ分離済み。
  - `dup_tab.py` から scan lifecycle、refine dialog lifecycle、trash/export 後の cluster 更新を分ける。trash 後 cluster 更新は `ui.dup_cluster_update` へ分離済み。
  - `thumbnail_tasks` と duplicate thumbnail queue の重複・キャンセル・壊れた画像 path を追加テストする。実装済み。
  - 一括リライトは避け、1回の変更で 1 境界だけ移す。
- 完了条件:
  - UI worker 例外後に button / status / active task / connection state が復帰することを smoke または headless test で確認できる。
  - `tags_tab.py` / `dup_tab.py` の本体は widget orchestration に寄り、文言生成・集計・状態判定は helper 側に寄る。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`

## 次フェーズ 3: 重複検出 refine と画像 IO の実データ耐性を上げる

- 目的: 破損画像、巨大画像、optional dependency 不足、並列 refine の worker 失敗が結果全体を壊さないようにする。
- 対象:
  - `dup.refine`
  - `dup.scanner`
  - `ui.dup_refine_parallel`
  - `ui.dup_workers`
  - `utils.image_io`
- 作業:
  - `ui.dup_refine_parallel` の tilehash / pixel refine で、個別ファイル失敗、cancel、progress、summary logging をテストする。
  - `dup.refine` の SSIM / ORB / pixel fallback の境界を fixture 付きで固定する。
  - `utils.image_io.get_thumbnail` の PyQt 実 backend smoke と cache eviction を追加する。
  - 実データに近い小型 fixture で pHash -> refine までの lightweight integration を作る。
- 完了条件:
  - 破損ファイルや依存不足が cluster 全体の silent failure にならない。
  - ユーザーに返す件数、ログ、progress が実処理と一致する。
- 検証:
  - `.\scripts\check.ps1`
  - `.\scripts\check-gui-smoke.ps1`
  - 実データ fixture を追加した場合は該当 integration test

## 次フェーズ 4: 型チェック対象を UI / tagger へ広げる

- 目的: 非同期 worker、viewmodel、画像 IO で固めた型チェックを、残る UI helper と tagger backend へ広げる。
- 現在維持する対象:
  - `src/ui/index_tasks.py`
  - `src/ui/dup_workers.py`
  - `src/ui/file_actions.py`
  - `src/ui/viewmodels/dup_view_model.py`
  - `src/utils/image_io.py`
- 次の追加候補:
  - `src/ui/index_feedback.py`
  - `src/ui/dup_status.py`
  - `src/ui/thumbnail_tasks.py`
  - `src/ui/tag_rendering.py`
  - `src/ui/viewmodels/settings_view_model.py`
  - `src/tagger/labels_util.py`
- 保留候補:
  - `src/tagger/wd14_onnx.py`
  - `src/tagger/pixai_onnx.py`
  - `src/ui/tags_tab.py`
  - `src/ui/dup_tab.py`
- 完了条件:
- mypy 対象 `53 source files` を下回らない。
  - 新規 core / db / service / worker / helper モジュールは原則 mypy 対象へ追加する。
  - `src/ui/dup_refine_parallel.py` の未型付け関数に由来する mypy note を減らす。
- 検証:
  - `python -m mypy`
  - `python -m ruff check .`
  - import構造や module header に触った場合は `.\scripts\check-package-smoke.ps1`

## 次フェーズ 5: tagger backend / GPU 境界を別トラックで固める

- 目的: ONNX Runtime CUDA / CPU fallback / model input layout / label merge の環境依存回帰を通常品質改善から分離して扱う。
- 対象:
  - `tagger.wd14_onnx`
  - `tagger.pixai_onnx`
  - `tagger.labels_util`
- 作業:
  - ONNX Runtime 不足、CUDA provider 不足、モデルファイル不足、labels CSV 不整合の user-facing error をテストする。
  - CPU fallback と GPU provider 選択のログを確認できるようにする。
  - 大きい backend 本体から provider selection / input layout / label postprocess を小さく切る。
- 検証:
  - 通常: `.\scripts\check.ps1`
  - GPU / CUDA / open_clip に触った場合のみ: `.\scripts\check-gpu.ps1`

## 運用ルール

- Windows + Python 3.10を主対象にする。
- `PYTHONPATH=src` を前提にする。
- ロードマップ更新時は `.\scripts\check.ps1` の実測値を記録する。
- カバレッジは全体数値だけでなく、DB、pipeline、検索worker、UI worker、画像 IO、重複 refine の重要分岐を優先する。
- UI巨大ファイルは一括リライトせず、テストを追加してから小さく切り出す。
- `docs/exception-audit-db-pipeline.md` は DB / pipeline 例外境界の作業メモとして維持し、完了した候補はロードマップと同期する。

## チェック選択

- 通常の高速確認: `.\scripts\check.ps1`
- PyQt6 / worker / GUI 状態変更: `.\scripts\check-gui-smoke.ps1`
- pipeline / DB bootstrap / end-to-end: `.\scripts\check-integration.ps1`
- WAL / SQLite lock / checkpoint / quiesce: `.\scripts\check-db-stress.ps1`
- import構造 / tools / packaging / module header: `.\scripts\check-package-smoke.ps1`
- GPU / ONNX Runtime CUDA / open_clip: `.\scripts\check-gpu.ps1`
