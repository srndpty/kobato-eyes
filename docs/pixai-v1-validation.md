# PixAI v1.0 検証結果

検証日：2026-11-14。RTX4090 24GB、Windows、Python 3.10.9。

## 採用モデル

- 配布元：[noaione/pixai-tagger-v1.0-onnx](https://huggingface.co/noaione/pixai-tagger-v1.0-onnx)
- revision：`68e8f4f02dd56a5f40c1b7474489fa0f599dec34`
- ONNX：IR 10、opset 20、float32、入力`[1,3,1008,1008]`、出力`[1,30877]`
- 外部重み参照：`model.onnx.data`
- ONNX SHA256：`563f4576c2668560c20f403b957f0ec4a7bd6a2275da2c2aa7f82f898ad34e5c`
- 外部重み SHA256：`4de1c25a38d1f2a2172fbcb0d2485b5f02a058b6e67bedd7bbbcfdf4de9329dc`
- ラベル SHA256：`0d34f2078016798808dc066dc206b18fb6ce7622f64241002ecd24172a4da068`
- ONNX Runtime 1.20.1、CUDAExecutionProviderを実際に使用。CUDAのTF32は無効。

bdsqlsz版はラベル形式への対応のみで、今回のモデル選定では実行していません。

## 公式FP32との数値比較

参照元：[pixai-labs/pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0)、
revision `9fe10addf9326e292da8a85a98ea74cd91b41771`。
PyTorch 2.8.0+cu128、FP32、TF32無効、Transformers 4.57.6、timm 1.0.22。

| 比較対象 | 件数 | 最大入力テンソル差 | 最大確率差 | 閾値判定差 |
| --- | ---: | ---: | ---: | ---: |
| 公開アニメ顔画像 | 32 | 0.0000122 | 0.0000164 | 0 |
| 縦長・横長・正方形・小画像・1px幅・大画像・透過・グレー | 8 | 0.0000158 | 0.00000477 | 0 |

全ケースで確率の `atol=1e-3, rtol=1e-3` を満たしました。
前処理は公式のfloatテンソル処理を基準とし、Pillow uint8リサイズのサンプル実装をそのまま流用していません。

32枚の元画像は [huggan/anime-faces](https://huggingface.co/datasets/huggan/anime-faces) の
`images/1.png`～`images/32.png`です。境界ケース8枚は1枚目からサイズ・モードを変えた検証画像です。
この確認は数値互換性の検証で、未知の大規模コレクション全体のタグ正解率評価ではありません。
詳細は作業フォルダの `tmp/pixai-v1/noaione-validation.json`、`edge-validation.json` にあります。

## 推論速度とメモリ

モデル起動を除外し、各サイズでウォームアップ1回後に測定した参考値です。
前処理込みは画像読込・前処理・推論・後処理の合計です。DB書込は含みません。

| アプリからの要求枚数 | ONNX実行バッチ | 準備済み入力 images/s | 前処理込み images/s |
| --- | ---: | ---: | ---: |
| 1 | 1 | 2.69 | 2.37 |
| 2 | 1 | 4.82 | 4.09 |
| 4 | 1 | 4.79 | 4.08 |
| 8 | 1 | 4.78 | 4.08 |

固定バッチ1なので2枚以上も内部では逐次推論です。初期設定はバッチ1・先読み1です。
小画像中心・短い測定のため、ディスクやCPU、実画像サイズによって速度は変わります。

ONNX単独検証時のGPU全体メモリ使用量の最大サンプルは18,340MiBでした。
他プロセスとCUDAキャッシュを含み、専有VRAMや厳密なピーク値を表しません。
公式PyTorchも同時にロードした比較時は約20,773MiBでした。
この値から低VRAM環境での動作を保証することはできません。
詳細は `tmp/pixai-v1/benchmark.json` にあります。

## 実施したチェック

- `scripts/check.ps1`：ruff、format、mypy、非GUI 586件、同スクリプトのGUI/smoke 52件が通過。カバレッジ83.26%。
- 後から追加した入力契約・署名ケースを含むv1単体テスト：11件通過。
- `scripts/check-gui-smoke.ps1`：GUI/smoke 54件通過。
- `scripts/check-integration.ps1`：8件通過。6カテゴリ保存、style/FTS検索、モデル内容更新時の再タグを含む。
- `scripts/check-package-smoke.ps1`：package treeとcompileallが通過。
- `scripts/check-gpu.ps1`：2件通過、旧WD14用ダミーモデルテスト1件skip（通常環境にonnx未導入）。v1実モデルCUDAテストは実行済み。

追加推奨：利用者の実画像コレクションで閾値と処理速度を確認してください。
DBのロック方式・WAL・スキーマは変更していないため、今回はdb_stressを実行していません。
実行手順は [PixAI v1.0ガイド](pixai-v1.md) を参照してください。
