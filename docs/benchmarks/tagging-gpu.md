# GPU Tagging Benchmark

このメモは、タグ付け高速化前後の baseline を同じ条件で比較するための記録場所です。

## 実行方法

PowerShell でプロジェクトルートから実行します。

```powershell
Remove-Item Env:KOE_HEADLESS -ErrorAction SilentlyContinue
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe tools\bench.py tagger `
  --tagger pixai `
  --model "C:\path\to\model.onnx" `
  --tags-csv "C:\path\to\selected_tags.csv" `
  --root "D:\images" `
  --limit 1000 `
  --batch-size 32 `
  --warmup-batches 2 `
  --output-json tmp\bench\pixai-baseline.json
```

WD14 を測る場合は `--tagger wd14` と対応するモデル/CSVを指定します。

## 記録項目

`tools\bench.py tagger` はDBへ書き込まず、実アプリに近い `PrefetchLoaderPrepared` 経路で画像ロード、前処理、ONNX推論、後処理を測ります。

- `selected_images`: 固定順で選ばれた画像数
- `processed_images`: 推論結果が返った画像数
- `measured_images`: warmup除外後の集計対象画像数
- `failed_images`: ロード失敗や推論失敗で処理できなかった画像数
- `images_per_second`: warmup除外後の画像/秒
- `wait_batch_seconds_*`: loaderから次バッチを受け取る待ち時間
- `infer_seconds_*`: `infer_batch_prepared()` 呼び出し全体
- `ort_ms_*`: taggerログから取得したONNX Runtime推論時間
- `post_ms_*`: taggerログから取得した後処理時間

## Baseline

| Date | Tagger | Model | Images | Batch | Providers | Images/sec | ORT mean ms | Post mean ms | JSON |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 2026-05-17 | pixai | pixai/model.onnx | 1000 | 32 | CUDAExecutionProvider, CPUExecutionProvider | 17.826 | 843.344 | 615.432 | `tmp\bench\pixai-baseline.json` |
| 2026-05-17 | pixai | pixai/model.onnx | 1000 | 32 | CUDAExecutionProvider, CPUExecutionProvider | 37.043 | 822.331 | 17.440 | `tmp\bench\pixai-after-speedup.json` |

## 比較メモ

高速化の各段階で同じ `--root`、`--limit`、`--batch-size`、`--warmup-batches` を使い、上の表に行を追加します。

2026-05-17 の高速化では、PixAI後処理を全ラベル辞書化からtop-k候補処理へ変更し、前処理をPillow経由からOpenCV/NumPy中心へ変更しました。主な改善は `post_ms` の削減です。
