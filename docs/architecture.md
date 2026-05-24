# kobato-eyes システムアーキテクチャ

このドキュメントは、kobato-eyes の全体設計、データフロー、テスト体制を図解しています。

---

## システムアーキテクチャ図

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyQt6 UI Thread (Main)                       │
│                                                                  │
│  ┌─────────┬──────────┬──────────┬───────────┐                 │
│  │ Tags    │ Dups     │ Index    │ Settings  │ (各タブ)        │
│  │ Tab     │ Tab      │ Tab      │ Tab       │                 │
│  └────┬────┴────┬─────┴────┬─────┴────┬──────┘                 │
│       │          │          │          │                        │
│       └──────────┼──────────┼──────────┘                        │
│                  │          │                                   │
│          ┌───────▼──────────▼──────────┐                       │
│          │    ViewModels & State Mgmt   │ (MVVM)              │
│          │  (tags_view_model, etc)      │                     │
│          └───────┬──────────────────────┘                      │
└────────────────┼─────────────────────────────────────────────┘
                 │
                 │ (Signal 経由)
                 │
┌────────────────▼─────────────────────────────────────────────┐
│          JobManager & QThreadPool                            │
│      (バックグラウンド実行スレッド管理)                         │
│                                                              │
│  ┌──────────────┐ ┌────────────┐ ┌──────────────┐          │
│  │  Scan Job    │ │  Tag Job   │ │  Dup Job    │          │
│  │  (QRunnable) │ │(QRunnable) │ │(QRunnable)  │          │
│  └──────────────┘ └────────────┘ └──────────────┘          │
│         │               │               │                  │
└─────────┼───────────────┼───────────────┼──────────────────┘
          │               │               │
          ▼               ▼               ▼
    ┌─────────┬──────────┬──────────┐
    │  Pipeline (core.pipeline)     │
    │                               │
    │  ┌────────┐  ┌────────┐      │
    │  │ Scan   │→ │ Tag    │→...  │
    │  │ Stage  │  │ Stage  │      │
    │  └────────┘  └────────┘      │
    └─────────────┬──────────────┘
                  │
              ┌───┴─────────┐
              │             │
              ▼             ▼
        ┌──────────────┬──────────────┐
        │  Tagger      │  DupScanner  │
        │(ONNX Runtime)│(pHash/SSIM)  │
        │ (WD14/PixAI) │              │
        └──────┬───────┴───────┬──────┘
               │               │
               ▼               ▼
        ┌────────────────────────────┐
        │   DBWritingService         │
        │  (quiesce() で安全保護)    │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │   SQLite3 Database         │
        │  (WAL + FTS5)              │
        │                            │
        │ ┌──────────┐ ┌──────────┐ │
        │ │ files    │ │file_tags │ │
        │ │ table    │ │ table    │ │
        │ └──────────┘ └──────────┘ │
        │ ┌────────────────────────┐│
        │ │ fts_files (FTS5)       ││
        │ └────────────────────────┘│
        └────────────────────────────┘
```

---

## データフロー図

### ユーザー操作からDB保存までの流れ

```
UI イベント (ユーザーが "Index Now" をクリック)
  ↓
UI Thread: MainViewModel.start_indexing()
  ↓
JobManager.submit(IndexJob)
  ↓
Worker Thread: Pipeline.process()
  ├─ Scan Stage: 監視フォルダをスキャン
  │  ├─ ファイル一覧取得
  │  └─ 新規/削除 ファイルを検出
  │
  ├─ Signature Stage: 画像署名計算
  │  ├─ pHash (phash.py)
  │  └─ SSIM/ORB (sig/ module)
  │
  ├─ Tag Stage: 推論実行
  │  ├─ Batch 分割 (max batch size by GPU VRAM)
  │  ├─ ONNX Runtime 実行
  │  │  ├─ PixAI (13000 タグ)
  │  │  └─ WD14 (8000 タグ)
  │  └─ 閾値フィルタリング
  │
  └─ Write Stage: DB 書き込み
     └─ DBWritingService.enqueue()
        ├─ quiesce() で読み込み待機
        ├─ executemany() で INSERT/UPDATE
        └─ commit()
  ↓
Database Updated
  ↓
UI Thread: Signal/Slot で UI 更新 (プログレス表示)
```

### 設定フローと状態管理

```
config.yaml
  ↓
ConfigService (core.config)
  ├─ YAML パース
  ├─ バリデーション
  └─ PipelineSettings/AppSettings
       ↓
SettingsViewModel
  ├─ root paths (監視フォルダ)
  ├─ tagger model path
  ├─ batch size
  ├─ thresholds
  └─ device (CPU/GPU)
       ↓
Pipeline/Tagger/DupScanner の初期化
```

### 検索フロー

```
UI: SearchBox に "character" と入力
  ↓
TagsSearchState.query_changed()
  ↓
Worker Thread: Repository.search_tags()
  │
  ├─ Query 構文解析
  │  ├─ AND (空白区切り)
  │  ├─ OR (|)
  │  └─ NOT (-)
  │
  └─ FTS5 検索実行
     └─ SELECT * FROM fts_files WHERE ... MATCH query
  ↓
結果取得 → UI に表示 (Thumbnail 非同期読み込み)
```

---

## モジュール間の依存関係

```
ui/                      (PyQt6 UI・ViewModel)
  ├─ depends on: core/, db/, dup/, utils/
  └─ signals to: JobManager

core/
  ├─ pipeline/           (scan/tag/write stages)
  │  ├─ depends on: tagger/, db/, sig/, dup/
  │  └─ emits: progress signals
  │
  ├─ jobs.py             (JobManager, QThread/QRunnable)
  │  └─ depends on: utils/
  │
  └─ config/             (設定管理, YAML)
     └─ validates: PipelineSettings

db/                      (SQLite 管理)
  ├─ connection.py       (quiesce 機構)
  ├─ schema.py           (テーブル定義)
  ├─ repository.py       (CRUD)
  └─ fts.py              (FTS5 検索)

tagger/                  (推論エンジン)
  ├─ base.py             (ITagger Protocol)
  ├─ wd14.py             (WD14 実装)
  ├─ pixai.py            (PixAI 実装)
  └─ onnx_backend.py     (ONNX Runtime)

dup/                     (重複検出)
  ├─ scanner.py          (pHash → SSIM/ORB)
  └─ cluster.py          (クラスタリング)

sig/                     (画像署名)
  └─ phash.py            (pHash, dhash)

utils/                   (共通ユーティリティ)
  ├─ image_io.py         (Pillow)
  ├─ env.py              (is_headless など)
  └─ ...
```

---

## テストマップ

各モジュールと対応するテスト：

| モジュール | テスト | マーカー |
|-----------|--------|---------|
| **core/jobs.py** | tests/core/test_jobs.py | (none) |
| **core/pipeline/** | tests/core/pipeline/ | gui, integration |
| **db/connection.py** | tests/db/test_connection_quiesce.py | db_stress |
| **db/repository.py** | tests/db/test_repository*.py | (none) |
| **tagger/wd14.py** | tests/tagger/test_wd14_*.py | gpu |
| **tagger/dummy.py** | tests/tagger/test_dummy.py | (none) |
| **dup/scanner.py** | tests/dup/test_scanner.py | (none) |
| **ui/*.py** | tests/ui/*smoke.py | gui |
| **utils/** | tests/utils/ | (none) |

---

## 実行フロー（メインループ）

```
1. Main Window 起動 (ui/app.py)
   └─ JobManager 初期化
   └─ 各 ViewModel 初期化 (MVVM)
   └─ Database 初期化 (schema apply)

2. ユーザーが "Index Now" をクリック
   └─ UI Signal: tags_tab.index_button.clicked
   └─ ViewModel.start_indexing() call
   └─ JobManager.submit(IndexJob)

3. Job 実行（別スレッド）
   └─ Pipeline.process()
   ├─ config 読み込み
   ├─ scan stage (新規ファイル検出)
   ├─ signature stage (pHash 計算)
   ├─ tag stage (ONNX 推論)
   └─ write stage (DB 保存)
   └─ Signal emit: progress_changed(%)

4. UI スレッド: Signal 受け取り
   └─ progress bar 更新
   └─ status label 更新

5. 完了
   └─ Job finished signal
   └─ status = "Indexing complete"
```

---

## パフォーマンス特性

| 操作 | スケーラビリティ | 制約 |
|-----|----------------|------|
| **スキャン** | O(n) ファイル数 | SSD 必須 |
| **タグ付け** | O(n) with GPU | VRAM ボトルネック |
| **重複検出** | O(n²) クラスタリング | 大規模画像セット遅い |
| **検索** | O(log n) FTS5 | インデックスサイズに依存 |
| **DB 書き込み** | O(n) quiesce 中 | WAL チェックポイント待機 |

---

## 将来の拡張

```
index/ モジュール (現在未使用)
  ├─ CLIP 埋め込み
  ├─ hnswlib ベクトル検索
  └─ 高速近傍検索 (duplicate detection 高速化)

Tier 2: ベクトル検索統合
  └─ pHash → CLIP → SSIM の多段判定
```

---

## デバッグ・トラブルシューティング

### ログを見る
```powershell
$env:KOE_LOG_LEVEL = "DEBUG"
# logs は %APPDATA%\kobato-eyes\logs\ に出力
```

### Headless モードでテスト
```powershell
$env:KOE_HEADLESS = "1"
$env:PYTHONPATH = "src"
pytest tests/core/test_pipeline.py -v
```

### GPU 確認
```python
import onnxruntime as rt
providers = rt.get_available_providers()
print(providers)  # ['CUDAExecutionProvider', 'CPUExecutionProvider'] なら OK
```

---

**参照**: [CLAUDE.md](../CLAUDE.md)、[AGENTS.md](../AGENTS.md)
