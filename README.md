# honkoku-ocr-py

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
