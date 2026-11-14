# PixAI Tagger v1.0

## 使い方

独自のONNX変換は不要です。検証済みモデルは
[noaione/pixai-tagger-v1.0-onnx](https://huggingface.co/noaione/pixai-tagger-v1.0-onnx)、
revision `68e8f4f02dd56a5f40c1b7474489fa0f599dec34` です。

この実装作業で取得したモデルは `tmp/pixai-v1/noaione/model.onnx` にあります。
`tmp` は一時領域なので、継続利用する場合はモデル一式を保管用フォルダへコピーして使ってください。
新たに取得する場合の例（重みは約1.95GB）：

```powershell
$modelDir = "C:\models\pixai-v1"
$modelBase = "https://huggingface.co/noaione/pixai-tagger-v1.0-onnx/resolve/68e8f4f02dd56a5f40c1b7474489fa0f599dec34"
New-Item -ItemType Directory -Force $modelDir | Out-Null
foreach ($name in @("model.onnx", "model.onnx.data", "tags.json")) {
    Invoke-WebRequest "$modelBase/$name" -OutFile (Join-Path $modelDir $name)
}
```

`model.onnx` と `model.onnx.data` は同じフォルダに置いてください。
アプリのSettingsでTaggerを `pixai-v1-onnx`、Model pathを `model.onnx` に設定します。
Labels JSONは空欄なら隣の `tags.json`、次に `config.json` を探します。
別の場所のJSONを使う場合はLabels JSONへ指定してください。
モデル検証・ラベル補完の読み込みはバックグラウンドで行います。

推論環境と起動：

```powershell
.\.venv\Scripts\python.exe -m pip install -c requirements-dev.lock -e ".[dev,tagging-gpu]"
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m ui.app
```

CPUの場合は `tagging-cpu` extraとDeviceのCPU onlyを使います。
アプリ実行のためにTransformers、timm、ONNX変換用ライブラリを追加する必要はありません。
CUDA onlyでCUDAが使えない場合はエラーを表示します。Autoでは既存方針に従いCPUへフォールバックします。
CUDAのTF32はv1バックエンドで無効にし、FP32比較の再現性を優先します。

## 設定とカテゴリ

YAMLの例：

```yaml
batch_size: 1
prefetch_depth: 1
tagger:
  name: pixai-v1-onnx
  model_path: C:/models/pixai-v1/model.onnx
  labels_json: null
  device: cuda
  thresholds:
    general: 0.17
    character: 0.27
    copyright: 0.24
    style: 0.15
    meta: 0.17
    rating: 0.41
```

6カテゴリの閾値はSettingsで調整できます。初めてv1を選んだときは専用の既定値を適用し、
その後の変更は保存します。既定ではタグ件数を制限しません。
styleは新しいカテゴリID `6` です。artistのID `1` など既存IDは変更しません。
`category:style` による検索、表示色、統計、autocompleteに対応します。
`rating:s` などのタグ名は配布ラベルのまま保存します。

DBは従来どおりタグ名を一意とし、同名タグのカテゴリは最後のタグ登録時に更新します。
したがって、既存artistとv1のstyleに同名タグがある場合、その名前に紐づく既存画像のカテゴリ検索にも影響します。
画像ごとに同名タグのカテゴリを分離するDB変更は、この対応には含みません。

モデル選択・設定保存だけでは全画像の再タグ付けは開始しません。
次のインデックス処理では、モデル・外部重み・ラベルの内容、前処理、実効閾値の署名を比較し、
構成が変わった画像を再タグ対象とします。従来の再タグ付け操作も使用できます。
再タグ付けした画像の予測タグは置き換わるため、旧予測を同時保持する機能はありません。

## 実装上の違い

- `ITagger`のインターフェースは維持し、v1専用バックエンドを追加しています。
- v1は元のRGB画像をPillowで読み込みます。旧ローダーの448px縮小とキャッシュを流用しません。
- 1008pxへのfloat bilinear/antialiasリサイズ、黒の中央パディング、mean/std各0.5の正規化を行います。
  透過は事前に白へ合成します。公式処理でサイズが0になる極端な縦横比では1pxを下限にします。
- 出力logitsにsigmoidを一度だけ適用します。v0.9のcopyright補完は行いません。
- noaione版のONNX入出力は `[1,3,1008,1008]` → `[1,30877]` の固定バッチです。
  アプリの要求バッチが2以上でも内部では1枚ずつ実行します。
  汎用の固定バッチ端数処理と、動的バッチのOOM時半減も実装しています。
- [bdsqlsz版](https://huggingface.co/bdsqlsz/pixai-tagger-v1.0-ONNX)の`config.json`形式も読み込めます。
  ただし同版の実モデル検証は未実施です。今回の採用モデルはnoaione版です。

## 検証と再現

実測結果は `docs/pixai-v1-validation.md`、作業中の詳細JSONは
`tmp/pixai-v1/noaione-validation.json` と `tmp/pixai-v1/edge-validation.json` にあります。

速度・入出力・GPU全体メモリ使用量を確認：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe tools/validate_pixai_v1.py --model C:/models/pixai-v1/model.onnx --images C:/images --output tmp/pixai-v1/validation.json
```

`--labels-json`、`--limit`、`--device cpu`も指定できます。
メモリはnvidia-smiでサンプリングしたGPU全体の使用量で、他プロセスを含み、厳密なピーク値ではありません。

公式FP32と比較する場合は、公式モデルのrevisionを固定して`config.json`、`preprocessor_config.json`、
`tagger_pipeline.py`、`model.safetensors`を同一フォルダへ取得し、`--reference-dir`を追加します。
このオプションは指定したローカルの公式Pythonコードを読み込みます。
参照実行用に検証した追加ライブラリはTransformers 4.57.6、timm 1.0.22、tokenizers 0.22.2です。
Torch/Torchvisionを利用できる別の仮想環境で導入し、通常のアプリ環境と分けてください。
合否は確率の `atol=1e-3, rtol=1e-3` とし、同一入力でのグラフ差・前処理差・閾値判定差を分けて記録します。

通常確認とCI再現：

```powershell
.\scripts\check.ps1
.\scripts\check-gui-smoke.ps1
.\scripts\check-integration.ps1
.\scripts\check-package-smoke.ps1
$env:KE_PIXAI_V1_MODEL = "C:/models/pixai-v1/model.onnx"
.\scripts\check-gpu.ps1
```

GPU確認は通常チェックから分離しています。`KE_PIXAI_V1_MODEL`が未指定ならv1実モデルテストはskipします。
