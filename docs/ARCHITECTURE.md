# 構成と約束事

このリポジトリを変更する人のための短い案内。座標系、画像の所有、モデルの契約、処理の段階、
原実装との一致をどう確かめるかをまとめる。使い方は[README](../README.md)を参照。

## モジュール

| モジュール | 役割 |
|-----------|------|
| `models.py` | モデルの版（`ModelSpec`）、配信ファイルの取得と照合（`ensure`、`EXPECTED`）、CPU向けfp32変換（`encoder_path`）と来歴 |
| `runtime.py` | onnxruntimeの遅延importとセッション生成（デバイス、スレッド数、プロファイル） |
| `layout.py` | RTMDet-sによる行検出（レターボックス、正規化、入れ子box除去） |
| `reading_order.py` | XY-Cutによる読み順 |
| `recognizer.py` | 行画像の前処理（傾き補正、回転、縮小、正規化）、encoder、KVキャッシュ付きgreedy復号、トークン列の後処理 |
| `koji.py` | 特殊トークン列とKoji記法・素テキストの変換 |
| `pipeline.py` | 1ページの処理（`OCR.process`）、複数ページ（`OCR.process_many`）、結果の型 |
| `output.py` | 原子的なファイル書き出し、ページ記録の生成、プレビュー画像、エラー文字列の秘匿 |
| `cli.py` | `honkoku-ocr`コマンド。入力の列挙、指紋、再開、終了コード |

`honkoku_ocr`をimportしてもonnxruntimeは読み込まれない。`raw_to_koji`や`order`はモデル無しで使える。

## 座標系

- 公開APIの座標はすべて**EXIFの向きを反映した元画像**のもの。`PageResult.width`と`height`がその大きさ、
  `settings.coordinate_space`は`exif_oriented_original`。
- 処理は長辺`max_dimension`（既定3,500px）に縮小した画像で行う。`PreparedPage`が縮小画像と縮尺
  （`scale_x`, `scale_y`）を持ち、`scale_box`と`original_box`が両方向の変換を担う。丸めは`js_round`
  （0.5切り上げ）で、JavaScriptの`Math.round`と揃える。
- 呼び出し側が`boxes`を与えたときは、与えた座標がそのまま結果に戻り、与えた順が読み順になる。画像と
  交わらないbox、大きさが正でないbox、有限でない値は`ValueError`。画像からはみ出す部分は行cropを切る
  ときだけ画像内に切り詰める。
- 行cropは上・下・右に`margin`（既定45px）の余白を付け、左には付けない。縦書きでは左隣が次の行だからで、
  学習データも同じ幾何で作られている。画像外は白。

## 画像の所有

- パスや`PIL.Image`を渡された`process`は自分で開いた縮小画像を`finally`で閉じる。呼び出し側の
  `PIL.Image`と`PreparedPage`は閉じない。`PreparedPage`はコンテキストマネージャとして使える。
- 行cropは認識が終わり次第`closing`で閉じる。`_letterbox`の中間画像も同じ。Pillowの`with`は
  ファイルハンドルしか閉じないので、生成した画像は`contextlib.closing`で閉じる。
- `Image.open`は遅延読み込みのため、`PreparedPage.load`が`with Image.open(...)`の中で変換を終える。

## モデルの契約

- 対応する版は`models.SPECS`（v16fs, v17, v18）。すべて入力256×2048、語彙7,710、decoderのKVキャッシュは
  24テンソル（6層×self/cross×K/V）、生成上限192トークン。`ModelSpec`がこれを持ち、`Recognizer`は
  セッション作成時にグラフの入出力名と語彙数を照合して合わなければ`RuntimeError`を出す。
- 配信ファイルは`EXPECTED`のサイズとSHA-256で照合する。取得は一時ファイル経由で、照合に通ったものだけが
  キャッシュに入る。
- encoderはfp16配信。CPUでは`encoder_path`が同じ重みのfp32版を初回に作って隣にキャッシュし、来歴
  `<name>.json`（元のSHA-256、変換の版`FP32_CONVERTER`、生成物のサイズとSHA-256）で再利用の可否を決める。
  変換手順を変えたら`FP32_CONVERTER`を上げる。CUDAではfp16のまま使う。
- decoderはint8で、`--device cuda`でもCPUで動く。
- 行検出のレターボックス縮小は平均化しない双一次補間（`_point_sampled_resize`）。Pillowの`resize`に
  戻すとスコアが変わり、閾値0.3付近の行の有無が変わる。

## 処理の段階と所要時間

`OCR.process`は次の順で進み、`PageResult.timings`に段階ごとの秒数を入れる。

1. `load` 画像を開き、EXIFの向きを反映し、縮小する
2. `detector_setup` / `layout` 行検出（初回のみセッション作成を含む）
3. `reading_order` XY-Cut
4. `recognizer_setup` 初回のみ
5. 行ごとに `preprocess` → `encoder` → `prefill` → `decode`
6. `total`

モデルは最初に必要になった段階で読み込む。`layout()`はencoder/decoderを読まず、`process(boxes=...)`は
行検出モデルを読まない。セッション作成や取得の失敗は`ModelSetupError`で、ページの失敗と区別する。

`OCR(overlap=True)`では行の前処理とencoderを別スレッドで先行させ（先行は最大2行）、主スレッドがdecodeする。
出力は逐次処理と同じ順・同じ内容で、段階ごとの秒数の合計はページの経過時間を上回りうる。`cancelled`
の呼び出しはこのときencoderスレッドからも起きるので、`threading.Event.is_set`のようなスレッド安全なものを渡す。

行認識器を差し替えるには`recognize_result(crop)`を実装する。`overlap`で使うには`encode_crop`と
`decode_encoded`も要る（`StagedRecognizer`）。

## 複数ページと再開

`OCR.process_many`は入力を逐次消費し、ページごとに`PageResult`か`PageFailure`（索引は1始まり、コマは
0始まり、例外の型名と秘匿済みメッセージ）を順に返す。`progress(index, outcome)`は各yieldの直前に呼ぶ。
`cancelled()`は入力を取る前と行の間で見る。途中の1回のモデル呼び出しは中断できない。

CLIの各ページJSONは完了記録で、txt（とpreview）を書いた後に最後に置く。`--resume`は記録の指紋
（画像のSHA-256、コマ、各モデルと語彙のSHA-256、設定、パッケージ・コード・ランタイムの版）が今回と
完全に一致し、成果物のSHA-256も一致するときだけ飛ばす。`overlap`やスレッド数のような、結果を変えない
はずの実行設定も指紋に入れている。出力の同一性を装置やプロバイダをまたいで確かめてはいないので、
実行条件が違えば作り直す方を取る。

## 原実装との一致の確かめ方

- 読み順とKoji変換は`benchmarks/compare_upstream.py`が原実装のTypeScriptをNodeで動かして突き合わせる。
- 行検出は、配信中のブラウザ版に同じ画像を読ませ、Reactの状態から行boxを取り出して比べた
  （行検出はブラウザ版でもWebAssemblyで動くので、WebGPUの無い環境で比べてよい）。
  スコアの差は平均0.01、閾値を越える候補数は51と52。
- 行認識の文字精度は`benchmarks/corpus/manifest.json`の4ページで測る。`benchmarks/fetch_corpus.py`で
  画像を取り、`python -m benchmarks.ocr benchmarks/corpus/manifest.json --output ...`でCERを出す。
  参照翻刻は機械的に行へ割り付けたもので誤りを含み、モデルの学習データと重なりうる。
  設定間の相対比較には使えるが、絶対値をブラウザ版の公表値と並べてはいけない。
- 速度は同じページ、同じ機械、同じ設定で測り、`benchmarks/`のJSONに条件ごと残す。

## 変更するときの手順

```sh
uv sync --extra cpu
uv run pytest            # 90前後のテスト。モデルもネットワークも要らない
uv run ruff check .
HONKOKU_OCR_REAL_MODELS=1 uv run pytest tests/test_real_encoder.py   # キャッシュ済みv18でfp32変換を検算
```

一つの論理的変更につき一つのコミット。座標や丸め、余白、閾値、正規化の定数を変えるときは原実装の
該当箇所（`text-recognizer.ts`, `layout-detector.ts`, `reading-order.ts`, `koji.ts`, `useOCRWorker.ts`）と
突き合わせ、変えた理由をNOTICEに書く。
