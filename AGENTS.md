# 言語
実装・レビュー含めて日本語で返答してください

# 依頼者からの要望
あなたは経験豊富な Python / PyQt6 / 画像処理 / DB のアーキテクト兼実装エージェントです。
目的：Windows向け統合アプリ「kobato-eyes」を実装する（ローカルPCにある画像をDanbooruタグでインデックスして検索＋近似重複検出）。
原則：
- モジュール分割（ui/, core/, db/, tagger/, dup/, sig/, index/, utils/）。PEP8・型ヒント・docstring必須。
- UIはPyQt6メインスレッドのみ。重い処理（画像IO・推論・DB・索引）はQThread/QRunnableで非同期。
- DBはSQLite（WAL/外部キーON）。全文検索はFTS5。ベクトル近傍はhnswlib。
- 画像読込は最初はPillowでOK（後でpyvipsに差し替え可）。埋め込みはopen_clip（CLIP系）。
- タガーはONNX Runtime(CUDA)優先（後からPyTorch差し替え可）。重複は多段判定（pHash→CLIP→SSIM/ORB）。
- 各タスクは**小さく**出力。巨大単一ファイルは禁止。必ず**作成/変更するファイル一覧**を先頭に出し、続けて内容を出力。
- すべて**実行手順**（pip/pytest/起動方法）を最後に短く付ける。
- モジュール先頭の行順序を厳守：1.module docstring、2. from __future__ import ...、3. 通常の import。これ以外の行を docstring より前に挿入しない。

## Danbooruタグ仕様に関する前提

このプロジェクトで扱う Danbooru タグ名について、以下は存在しないことを前提としてよい。

- `-` から始まるタグ
- `(` または `)` から始まるタグ
- 空白文字を含むタグ

したがって、クエリ構文や autocomplete の実装・レビューでは、これらを「タグ名として解釈すべきケース」として単独で考慮しない。  
ただし、既存実装の他機能に影響する場合、または外部入力の安全性・例外処理として必要な場合は別途考慮する。

## PyQt6 に関する注意
- PyQt6 では QFileSystemModelとQShortcut は PyQt6.QtGui 内に移動されているため留意してください。
  NG: from PyQt6.QtWidgets import QFileSystemModel
  OK: from PyQt6.QtGui import QFileSystemModel
- `PyQt6.QtConcurrent` は存在しません。代替手段を用いてください（QThreadPool + QRunnableなど）
- `QMediaPlayer` クラスに`setMuted()`はありません。PyQt6ではマルチメディア関連のアーキテクチャが変更されました。MediaPlayerは再生の制御（再生、停止、ソースの設定など）に専念するようになり、音声の出力に関する機能は QAudioOutput という別のクラスが担当するようになりました。
- QMimeDataはPyQt6.QtCore、QActionはPyQt6.QtGuiの中です

## 開発環境に関する注意
- `PYTHONPATH=src` を前提にしています

## テスト方針
- 通常の高速確認は `.\scripts\check.ps1` を使う。このスクリプトは `KOE_HEADLESS=1` を設定し、GUI / integration / db_stress を除外するため、開発速度を優先した標準チェックである。
- PyQt6実backend、`core.jobs`、QThread/QRunnable、pipeline、DB bootstrap、GUI/integrationテストに関わる変更では、返答の最後に以下のCI再現コマンドを明記し、可能なら実行する。
  - `.\scripts\check-integration.ps1`
  - 直接実行する場合: `Remove-Item Env:KOE_HEADLESS -ErrorAction SilentlyContinue; $env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m pytest -m "integration and not gpu" -p no:cov`
- SQLiteロック、WAL、並行DB書き込み、checkpoint、quiesceに関わる変更では、必要に応じて `.\scripts\check-db-stress.ps1` を案内または実行する。
- GPU / ONNX Runtime CUDA / open_clip / 推論backendに関わる変更では、RTX4090搭載PCであれば `.\scripts\check-gpu.ps1` を案内または実行する。ただしGPU確認は環境依存なので通常チェックや必須確認には混ぜない。
- 返答では、実際に実行したチェックと、未実行だが変更内容から推奨される追加チェックを分けて書く。
