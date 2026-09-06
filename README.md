<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./docs/logo-light.svg">
  <img src="./docs/logo-light.svg" alt="みんなで翻刻くずし字OCR ローカル版 — honkoku-ocr-py" width="520">
</picture>

**ブラウザで動く「みんなで翻刻OCR」を Python に移植。くずし字の画像から「みんなで翻刻」記法の翻刻をコマンドラインで一括生成する。**

[![MIT](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![models CC BY 4.0](https://img.shields.io/badge/models-CC_BY_4.0-orange)](./NOTICE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![onnxruntime](https://img.shields.io/badge/runs%20on-onnxruntime-5C3EE8)](https://onnxruntime.ai/)
[![CUDA optional](https://img.shields.io/badge/GPU-CUDA_optional-76B900?logo=nvidia&logoColor=white)](#性能)
[![model v18](https://img.shields.io/badge/model-kuzushiji_v18-0b7285)](https://yuta1984.github.io/honkoku-ocr-web/tech.html)
[![tests](https://img.shields.io/badge/tests-25_passing-success?logo=pytest&logoColor=white)](./tests)
[![upstream](https://img.shields.io/badge/upstream-honkoku--ocr--web-8a2f1f)](https://github.com/yuta1984/honkoku-ocr-web)

</div>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/demo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./docs/demo-light.svg">
  <img src="./docs/demo-light.svg" alt="honkoku-ocr がディレクトリの画像を一括翻刻し、行画像と Koji 記法の翻刻を出力する様子" width="760">
</picture>
</p>

[みんなで翻刻OCR](https://yuta1984.github.io/honkoku-ocr-web/)（橋本雄太、CC BY 4.0）の推論パイプラインを
Python と onnxruntime に移した移植版。ブラウザ版と同じモデル・同じ前処理で、くずし字の古典籍画像から
Koji 記法（ふりがな・返り点・送り仮名・割書のタグを含む「みんなで翻刻」の記法）の翻刻テキストを得る。
幾何・正規化・復号の手順は原実装と同じだが、拡大縮小と回転の補間は Pillow のものなので画素値は完全には一致しない。

A Python port of the inference pipeline of みんなで翻刻OCR (honkoku-ocr-web, by Yuta Hashimoto,
CC BY 4.0). Same models, same geometry and normalisation, same output notation; no browser, no UI, and it runs
on a GPU through onnxruntime's CUDA provider. Resampling is Pillow's, so tensors are equivalent rather than bit-identical.

## 構成

| 段階 | 実装 | 由来 |
|------|------|------|
| 行検出 | RTMDet-s、入力 1024×1024 レターボックス、入れ子 box 除去 | NDL古典籍OCR-Lite のモデル、honkoku-ocr-web の前後処理 |
| 読み順 | XY-Cut（縦書きは右の段から左へ） | NDL古典籍OCR-Lite / honkoku-ocr-web |
| 行認識 | ConvNeXt V2 encoder (fp16) + RoBERTa decoder (int8、KV キャッシュ)、greedy、語彙 7,710 | honkoku-ocr-web kuzushiji-v18（v17, v16fs も選択可） |
| 出力 | 特殊トークン列 → Koji 記法 / 素テキスト | honkoku-ocr-web |

モデルの設計と学習・評価については [docs/tech.html](docs/tech.html)（原著作物の技術情報ページの複製）を参照。

## 比較

| | **honkoku-ocr-py** | [みんなで翻刻OCR](https://yuta1984.github.io/honkoku-ocr-web/)（ブラウザ版） | [NDL古典籍OCR-Lite](https://github.com/ndl-lab/ndlkotenocr-lite) |
| :-- | :-: | :-: | :-: |
| 動く場所 | Python / CLI / サーバ | ブラウザ（WebAssembly, Web Worker） | Python / デスクトップアプリ |
| 一括処理 | ✅ ディレクトリ単位、スクリプトから呼べる | ❌ 画像を開いてボタンを押す | ✅ ディレクトリ単位 |
| GPU | ✅ CUDA（encoder） | WebGPU 対応端末のみ | CUDA（ベータ） |
| 行認識モデル | ConvNeXt V2 + RoBERTa（kuzushiji v18） | 同じ | PARSeq |
| 出力 | Koji 記法（ふりがな・返り点・送り仮名・割書のタグ付き）+ JSON | Koji 記法、縦書き表示 | 素テキスト + XML/JSON |
| 行位置の持ち込み | ✅ `run(image, boxes=...)` | 画面上で bbox を編集 | ❌ |
| 行 bbox の編集 UI | ❌ | ✅ | ❌ |
| モデルの検証 | サイズと SHA-256 を照合 | IndexedDB キャッシュ | 同梱 |
| 精度（原著の公表値） | 本文 plain micro CER 0.075（v18） | 同じ | NDL古典籍OCR ver.3 より約 2% 低い |

ブラウザ版の強みは行 bbox の手直しと縦書きの閲覧で、そこはこの移植には無い。自動化と大量処理、他のツールとの接続がこの移植の役割になる。

## なぜ移植したか

ブラウザ版は 1 枚ずつ画像を開いてボタンを押す道具で、数十コマの写本を機械的に処理する用途には向かない。
このリポジトリは同じモデルを、次の使い方ができる形にしたものである。

- **一括処理**: ディレクトリを渡せば全画像を順に翻刻し、画像ごとに Koji 記法の txt と、行位置・読み順・
  認識スコアを持つ JSON を書き出す。シェルスクリプトや cron、CI からそのまま呼べる。
- **他のツールとの接続**: Python から `OCR().run()` を呼ぶだけで行の list が返る。TEI や翻刻プラットフォーム
  への流し込み、別の OCR との突き合わせ、校合ビューアの生成といった後段処理を同じプロセスで書ける。
- **行位置の持ち込み**: `run(image, boxes=...)` で自前の行 bbox を与えられる。たとえば NDL古典籍OCR-Lite の
  行検出結果をそのまま渡せば、二つのエンジンの読みを行ごとに一対一で比べられる。
- **GPU**: encoder を CUDA で動かせる。行認識が支配的なので、GPU があれば 1 コマ数秒で終わる。
- **再現性**: モデルはバージョン名で固定し、取得時にサイズと SHA-256 を照合する。同じ入力からは同じ出力が得られる。
- **ブラウザ不要**: サーバや WSL、ヘッドレス環境で動く。IndexedDB のキャッシュも Web Worker もいらない。

出力はブラウザ版と同じ Koji 記法なので、ブラウザ版で作った翻刻と混ぜて扱える。

## 性能

RTX 5070 Ti（CUDA）と 16 スレッドの CPU で、国書データベースの写本画像（半丁、約 3,200×4,600 px を長辺 3,500 px に縮小）を処理した値。

| 環境 | 行検出 | 行検出＋行認識 | 1 行あたり |
|------|--------|----------------|-----------|
| CUDA (encoder) + CPU (decoder) | 0.3 秒/コマ | 1.3〜1.5 秒/コマ（8〜15 行） | 0.10〜0.16 秒 |
| CPU のみ | 数秒/コマ | 約 60〜100 秒/コマ | 約 7 秒 |

初期化（セッション作成）は約 1 秒。CPU の遅さは fp16 encoder に由来する（int8 版は onnxruntime の CPU プロバイダで
動かせない）。CPU で大量に処理する場合は GPU 版を使うか、行数の少ない画像に限るのが現実的。
認識精度はブラウザ版と同じモデルなので原著作物の [技術情報](docs/tech.html) の評価がそのまま当てはまる。
読み順・Koji 変換の原実装との比較と再現手順は [benchmarks/README.md](benchmarks/README.md) を参照。

## 使い方

```sh
uv sync --extra cpu          # onnxruntime (CPU)
uv sync --extra gpu          # onnxruntime-gpu と CUDA 12 のランタイム (cpu と同時には入れない)

uv run honkoku-ocr --download                    # モデルを取得 (約 250 MB、~/.cache/honkoku-ocr/models)
uv run honkoku-ocr page.jpg -o out               # out/page.txt (Koji 記法、読み順) と out/page.json
uv run honkoku-ocr pages/ -o out --device cuda   # ディレクトリ内の画像を一括処理
uv run honkoku-ocr page.jpg -o out --plain       # タグ無しの素テキスト
```

Python から:

```python
from honkoku_ocr import OCR, Box

ocr = OCR(version="v18", device="cuda")
for line in ocr.run("page.jpg"):
    print(line.reading_order, line.koji)          # line.raw にタグ付きの生文字列、line.plain に素テキスト
```

行の位置を自分で与える場合は `ocr.run(image, boxes=[Box(x, y, w, h, 1.0), ...])`。
画像は EXIF の向きを反映したうえで長辺 3500 px に縮小してから処理され、返される座標は元画像のもの。
`--device cuda` は CUDA プロバイダが無ければエラーで止まる（CPU に黙って落ちない）。
モデルは取得時に SHA-256（v18 と RTMDet）とサイズを照合する。

JSON の各行は `reading_order`, `x`, `y`, `width`, `height`, `confidence`（行検出のスコア）, `raw`, `koji`, `plain` を持つ。

## ブラウザ版との違い

- UI（画像ビューア、bbox の編集、縦書き表示、PDF/TIFF/HEIC の読み込み、LLM 連携）は含まない。
- encoder は fp16 版を使う。int8 版が使う ConvInteger 演算は onnxruntime の CPU/CUDA プロバイダに無い。
  対応する版は fp16 encoder が配布されている v16fs / v17 / v18。
- 行検出は RTMDet のみ（ブラウザ版の 5 クラス YOLO は含まない）。

## 環境変数

- `HONKOKU_OCR_MODELS` … モデルの保存先（既定 `~/.cache/honkoku-ocr/models`）
- `HONKOKU_OCR_MODEL_URL` … モデル配信元（既定は原著作物と同じ公開バケット）

## テスト

```sh
uv run --extra dev pytest
```

## ライセンスと帰属

このリポジトリの Python コードとテストは MIT ライセンス（[LICENSE](LICENSE)）。

原著作物に由来する部分はそれぞれの著作者の CC BY 4.0 のままである（[LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)、
一覧は [NOTICE.md](NOTICE.md)）:
みんなで翻刻OCR（橋本雄太）— モデル、語彙ファイル `honkoku_ocr/config/`、`docs/tech.html`、および移植元となった推論手順。
NDL古典籍OCR-Lite（国立国会図書館）— 行検出モデル、XY-Cut の手続き。
学習データは「みんなで翻刻」の翻刻成果に基づく。

引用する場合は原著作物を挙げること:
橋本雄太「みんなで翻刻OCR — 市民の力で作ったくずし字AI-OCR」 https://yuta1984.github.io/honkoku-ocr-web/
