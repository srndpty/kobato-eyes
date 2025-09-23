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
- 返答は日本語で。

出力形式：①変更ファイル一覧 ②コード ③テスト ④実行手順 ⑤簡易受け入れ基準（チェックリスト）

## PyQt6 に関する注意
- PyQt6 では QFileSystemModelとQShortcut は PyQt6.QtGui 内に移動されているため留意してください。
  NG: from PyQt6.QtWidgets import QFileSystemModel
  OK: from PyQt6.QtGui import QFileSystemModel
- `PyQt6.QtConcurrent` は存在しません。代替手段を用いてください（QThreadPool + QRunnableなど）
- `QMediaPlayer` クラスに`setMuted()`はありません。PyQt6ではマルチメディア関連のアーキテクチャが変更されました。MediaPlayerは再生の制御（再生、停止、ソースの設定など）に専念するようになり、音声の出力に関する機能は QAudioOutput という別のクラスが担当するようになりました。
- QMimeDataはPyQt6.QtCore、QActionはPyQt6.QtGuiの中です