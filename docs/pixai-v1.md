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
外部重みのファイル名は `model.onnx` 内に記録された参照先（noaione版は `model.onnx.data`）で決まります。
`model.onnx` を改名しても重みファイルは `model.onnx.data` のまま置いてください。
再タグ判定の署名も、ファイル名からの推測ではなくONNXに記録された参照先の内容で計算します。
参照先がモデルフォルダ外を指す、またはONNXを解析できないモデルは署名計算の段階でエラーになります。
アプリのSettingsでTaggerを `pixai-v1-onnx`、Model pathを `model.onnx` に設定します。
Labels JSONは空欄なら隣の `tags.json`、次に `config.json` を探します。
別の場所のJSONを使う場合はLabels JSONへ指定してください。
明示したLabels JSONが存在しない・壊れている場合は、再タグ対象の有無に関係なくインデックス開始前にエラーになります
（隣接ファイルへの自動フォールバックは未指定時のみ）。
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

DBは従来どおりタグ名を一意とし、同名タグのカテゴリは**後勝ち**（最後のタグ登録時のカテゴリ）で更新します。
Danbooru本家でカテゴリが変更されることがあり、基本的に新しいモデルのカテゴリの方が正しいためです。
カテゴリはタグ名単位で共有されるので、更新は再解析していない既存画像の `category:` 検索・表示にも反映されます。

既存DB全体をv1で再タグ付けする前に、DBをバックアップし、カテゴリが変わるタグを確認できます（DBは読み取り専用で開きます）：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe tools/check_tag_category_conflicts.py --db "$env:APPDATA/kobato-eyes/kobato-eyes.db" --labels-json C:/models/pixai-v1/tags.json
```

カテゴリが変わるタグがあれば終了コード1で、該当タグと影響画像数をJSONで出力します。

モデル選択・設定保存だけでは全画像の再タグ付けは開始しません。
次のインデックス処理では、モデル・外部重み・ラベルの内容、前処理、実効閾値の署名を比較し、
構成が変わった画像を再タグ対象とします。従来の再タグ付け操作も使用できます。
再タグ付けした画像の予測タグは置き換わるため、旧予測を同時保持する機能はありません。

## 実装上の違い

- `ITagger`のインターフェースは維持し、v1専用バックエンドを追加しています。
- v1は元のRGB画像をPillowで読み込みます。旧ローダーの448px縮小とキャッシュを流用しません。
  バッチ経路・単一画像経路（`core.tag_job`）とも同じ `load_source_rgb` を使い、
  EXIF回転・4096px縮小は行いません（公式processorと同じく生画素を使用）。
- 原寸保持のため、1億ピクセル超の画像はデコード前に拒否し、その画像だけ失敗扱いにします。
  単一画像経路でも最初に原寸入力を確定し、署名（pHash等）用の縮小読み込みはその後に必要な場合だけ行います。
  入力キャッシュも記録された元サイズで同じ上限を検査し、配列形状と元サイズが一致しないキャッシュは破棄します。
  リサイズはuint8入力から行い、原寸のfloat32中間配列は作りません。
- 前処理バージョンは `pixai-v1-uint8-bilinear-antialias-pad-1008-v2` です。
  旧v1（float変換後にリサイズ）の署名でタグ付けした画像は、次回インデックス時に再タグ対象になり、旧入力キャッシュも使いません。
- 1008pxへのfloat bilinear/antialiasリサイズ、黒の中央パディング、mean/std各0.5の正規化を行います。
  透過は事前に白へ合成します。公式処理でサイズが0になる極端な縦横比では1pxを下限にします。
- 出力logitsにsigmoidを一度だけ適用します。v0.9のcopyright補完は行いません。
  出力テンソル名が `logits` 以外（例：sigmoid済み `probs` を出す配布物）は二重sigmoidを防ぐため拒否します。
- モデル・外部重み・ラベルの内容をセッション生成前後で比較し、ロード中の差し替えを検出したら中止します。
  さらにDBへ署名付きで書く前に、ロード内容と署名に使ったモデル内容が一致することを確認します。
  インデックス処理（`TagStage`）とwatcher経路（`run_tag_job`）の両方が同じ照合（`verify_loaded_identity`）を使います。
  署名にv1のidentityが含まれる場合、identityを持たないtaggerも拒否します（免除はテスト用の明示的overrideのみ）。
  不一致は `TaggerIdentityMismatchError` として扱い、画像単位の失敗とは区別します。
  watcher（`ProcessingPipeline`、現行GUIの主経路ではない互換API）では、ジョブを `completed` ではなく `error` で終了し、
  `reload_required_reason` と `on_reload_required` コールバックで自動タグ付けの停止を通知します。
  `set_tagger()` で再ロードしたtaggerへ差し替えると、未処理の画像を再投入できます。
- noaione版のONNX入出力は `[1,3,1008,1008]` → `[1,30877]` の固定バッチです。
  アプリの要求バッチが2以上でも内部では1枚ずつ実行します。
  汎用の固定バッチ端数処理と、動的バッチのOOM時半減も実装しています。
  動的バッチのモデルは設定の `batch_size` から開始します（実モデルでの動的バッチ動作は未検証）。
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
合否は確率の `atol=1e-3, rtol=1e-3`（`numerical_pass`）と、閾値判定の不一致が0件であること（`decision_pass`）の
両方を満たす場合のみ `passed` とします。グラフ差・前処理差・閾値判定差は分けて記録します。

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
