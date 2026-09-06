# 帰属表示 / Attribution

## 原著作物

- **みんなで翻刻OCR (honkoku-ocr-web)** — 橋本雄太（国立歴史民俗博物館）
  https://github.com/yuta1984/honkoku-ocr-web ／ https://yuta1984.github.io/honkoku-ocr-web/
  ライセンス CC BY 4.0。行認識モデル（kuzushiji-v16fs / v17 / v18、ConvNeXt V2 encoder + RoBERTa decoder）、
  語彙ファイル、前処理・復号・Koji 記法変換の手続き、技術情報ページ（docs/tech.html）はこの著作物に由来する。
  学習データは「みんなで翻刻」（https://honkoku.org）の翻刻成果に基づく。
- **NDL古典籍OCR-Lite** — 国立国会図書館
  https://github.com/ndl-lab/ndlkotenocr-lite ／ CC BY 4.0。
  行検出モデル rtmdet-s-1280x1280.onnx と XY-Cut 読み順整序の手続きはこの著作物に由来する。

## 本リポジトリでの変更点

- TypeScript（ブラウザ／Web Worker、onnxruntime-web）で書かれた推論パイプラインを Python（onnxruntime）に移植した。
  レイアウト認識（レターボックス・正規化・入れ子 box 除去）、XY-Cut 読み順、行 crop の余白規則、
  傾き補正、to_pixel 前処理、KV キャッシュ付き greedy 復号、反復崩壊の打ち切り、rt2 除去と
  送り仮名・返り点のカタカナ化、Koji 記法への変換を、原実装と同じ定数・手順で実装している。
- 画像 UI（OpenSeadragon ビューア、行 bbox 編集、縦書き表示、PDF/TIFF/HEIC 読み込み、LLM 連携）は含まない。
- encoder には fp16 版 ONNX を用いる（int8 版の ConvInteger 演算は onnxruntime の CPU/CUDA 実行プロバイダに実装がないため）。
  そのため fp16 encoder が配布されている v16fs / v17 / v18 のみに対応する。
- モデルは初回実行時に原著作物と同じ配信元から取得し、ローカルにキャッシュする。
- XY-Cut のメッシュ生成とヒストグラム集計を NumPy に置き換えた。
  原実装との読み順・Koji 変換の比較手順と測定結果は benchmarks/ に記す。
