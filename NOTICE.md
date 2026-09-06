# 帰属表示 / Attribution

このリポジトリのPythonコードとテストはMITライセンス。以下の原著作物に由来する部分（モデル、語彙ファイル、docs/tech.html、
移植元の推論手順）はCC BY 4.0のままで、著作者への帰属表示が必要である。

## 原著作物

- **みんなで翻刻OCR (honkoku-ocr-web)** — 橋本雄太（国立歴史民俗博物館）
  https://github.com/yuta1984/honkoku-ocr-web／https://yuta1984.github.io/honkoku-ocr-web/
  ライセンスCC BY 4.0。行認識モデル（kuzushiji-v16fs / v17 / v18、ConvNeXt V2 encoder + RoBERTa decoder）、
  語彙ファイル、前処理・復号・Koji記法変換の手続き、技術情報ページ（docs/tech.html）はこの著作物に由来する。
  学習データは「みんなで翻刻」（https://honkoku.org）の翻刻成果に基づく。
- **NDL古典籍OCR-Lite** — 国立国会図書館
  https://github.com/ndl-lab/ndlkotenocr-lite／CC BY 4.0。
  行検出モデルrtmdet-s-1280x1280.onnxとXY-Cut読み順整序の手続きはこの著作物に由来する。

## 本リポジトリでの変更点

- TypeScript（ブラウザ／Web Worker、onnxruntime-web）で書かれた推論パイプラインをPython（onnxruntime）に移植した。
  レイアウト認識（レターボックス・正規化・入れ子box除去）、XY-Cut読み順、行cropの余白規則、
  傾き補正、to_pixel前処理、KVキャッシュ付きgreedy復号、反復崩壊の打ち切り、rt2除去と
  送り仮名・返り点のカタカナ化、Koji記法への変換を、原実装と同じ定数・手順で実装している。
- 画像UI（OpenSeadragonビューア、行bbox編集、縦書き表示、PDF/HEIC読み込み、LLM連携）は含まない。
- encoderにはfp16版ONNXを用いる（int8版のConvInteger演算はonnxruntimeのCPU/CUDA実行プロバイダに実装がないため）。
  そのためfp16 encoderが配布されているv16fs / v17 / v18のみに対応する。CPUではfp16版を読み込み時にfp32へ変換した
  ファイルを使う。重みの値は同じで、出力は行によってfp16版とわずかに異なる。
- 行検出の入力への縮小は、ブラウザのcanvas drawImageに合わせて平均化しない双一次補間（Pillowのアフィン変換）で行う。
- 画像の端にかかる行の傾き推定で、画像外の画素を白として扱う。原実装は透明画素（輝度0）を二値化閾値の平均に含める。
- モデルは初回実行時に原著作物と同じ配信元から取得し、ローカルにキャッシュする。
- XY-Cutのメッシュ生成とヒストグラム集計をNumPyに置き換えた。
  原実装との読み順・Koji変換の比較手順と測定結果はbenchmarks/ に記す。
