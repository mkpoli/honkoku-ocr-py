<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mkpoli/honkoku-ocr-py/main/docs/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mkpoli/honkoku-ocr-py/main/docs/logo-light.svg">
  <img src="https://raw.githubusercontent.com/mkpoli/honkoku-ocr-py/main/docs/logo-light.svg" alt="みんなで翻刻くずし字OCR ローカル版 — honkoku-ocr-py" width="520">
</picture>

[![PyPI](https://img.shields.io/pypi/v/honkoku-ocr-py?color=0b7285)](https://pypi.org/project/honkoku-ocr-py/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![MIT](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![upstream](https://img.shields.io/badge/upstream-honkoku--ocr--web-8a2f1f)](https://github.com/yuta1984/honkoku-ocr-web)

</div>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mkpoli/honkoku-ocr-py/main/docs/demo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mkpoli/honkoku-ocr-py/main/docs/demo-light.svg">
  <img src="https://raw.githubusercontent.com/mkpoli/honkoku-ocr-py/main/docs/demo-light.svg" alt="honkoku-ocrがディレクトリの画像を一括翻刻し、行画像とKoji記法の翻刻を出力する様子" width="760">
</picture>
</p>

[みんなで翻刻OCR](https://yuta1984.github.io/honkoku-ocr-web/)（橋本雄太、CC BY 4.0）の推論パイプラインを
Pythonとonnxruntimeに移した移植版。ブラウザ版と同じ重みを使い、くずし字の古典籍画像から
Koji記法（ふりがな・返り点・送り仮名・割書のタグを含む「みんなで翻刻」の記法）の翻刻テキストを得る。
幾何・正規化・復号の手順は原実装と同じである。拡大縮小と回転の補間はPillowのものなので画素値は一致せず、
encoderもブラウザ版のint8版ではなくfp16版（CPUではそれをfp32に直したもの）を使うため、出力は行によって異なる（[性能](#性能)）。

A Python port of the inference pipeline of みんなで翻刻OCR (honkoku-ocr-web, by Yuta Hashimoto,
CC BY 4.0). Same weights, same geometry, normalisation and decoding, same output notation; no browser and no UI.
Line detection and the encoder can run on CUDA, the decoder runs on the CPU. Resampling is Pillow's and the encoder
is the fp16 export rather than the browser's int8 one, so line texts can differ from the browser version.

## 構成

| 段階 | 実装 | 由来 |
|------|------|------|
| 行検出 | RTMDet-s、入力1024×1024レターボックス、入れ子box除去 | NDL古典籍OCR-Liteのモデル、honkoku-ocr-webの前後処理 |
| 読み順 | XY-Cut（縦書きは右の段から左へ） | NDL古典籍OCR-Lite / honkoku-ocr-web |
| 行認識 | ConvNeXt V2 encoder（fp16、CPUではfp32に変換）+ RoBERTa decoder（int8、KVキャッシュ）、greedy、語彙7,710 | honkoku-ocr-web kuzushiji-v18（v17, v16fsも選択可） |
| 出力 | 特殊トークン列 → Koji記法 / 素テキスト | honkoku-ocr-web |

モデルの設計と学習・評価については[docs/tech.html](docs/tech.html)（原著作物の技術情報ページの複製）を参照。

## 比較

| | **honkoku-ocr-py** | [みんなで翻刻OCR](https://yuta1984.github.io/honkoku-ocr-web/)（ブラウザ版） |
| :-- | :-: | :-: |
| 動く場所 | Python / CLI / サーバ | ブラウザ（WebAssembly, Web Worker） |
| 一括処理 | ディレクトリ単位。スクリプトやcronから呼べ、失敗した画像を飛ばして続行、`--resume`で再開 | タブに開いた複数画像を「全画像OCR実行」でまとめて処理。結果は画面から保存する |
| GPU | CUDA（行検出とencoder） | WebGPU対応端末のみ |
| 行認識モデル | ConvNeXt V2 + RoBERTa（kuzushiji v18） | 同じ重み |
| 出力 | Koji記法 + JSON（行位置、読み順、停止理由、所要時間、モデルと設定の指紋） | Koji記法、縦書き表示 |
| 行位置の持ち込み | `--boxes` / `process(image, boxes=...)` | 画面上でbboxを編集 |
| 行bboxの編集UI | 無し（`--preview`で番号付き画像を出す） | あり |
| モデルの検証 | 全ファイルのサイズとSHA-256を照合 | IndexedDBキャッシュ |
| 精度 | 未測定（同じ重み。encoderの精度と画像補間がブラウザ版と異なる） | 本文plain micro CER 0.075（v18、[技術情報](docs/tech.html)の公表値） |

ブラウザ版の強みは行bboxの手直しと縦書きの閲覧で、そこはこの移植には無い。自動化と大量処理、他のツールとの接続がこの移植の役割になる。

## なぜ移植したか

ブラウザ版はタブに開いた画像をまとめて処理できるが、画像の読み込みも結果の保存も画面の操作で行う。
他のプログラムから呼ぶ、夜間に数百コマを流す、結果をそのまま次の処理に渡す、という使い方のためにこのリポジトリがある。

- **一括処理**: ディレクトリを渡せば全画像を順に翻刻し、画像ごとにKoji記法のtxtと、行位置・読み順・
  行検出スコア・所要時間を持つJSONを書き出す。読めない画像があっても残りを処理し、終了コードで知らせる。
  中断したら`--resume`で続きから再開できる。シェルスクリプトやcron、CIからそのまま呼べる。
- **他のツールとの接続**: Pythonから`OCR().process()`を呼ぶだけでページの結果が返る。TEIや翻刻プラットフォーム
  への流し込み、別のOCRとの突き合わせ、校合ビューアの生成といった後段処理を同じプロセスで書ける。
- **行位置の持ち込み**: `--boxes`や`process(image, boxes=...)`で自前の行bboxを与えられる。たとえばNDL古典籍OCR-Liteの
  行検出結果をそのまま渡せば、二つのエンジンの読みを行ごとに一対一で比べられる。与えた座標はそのまま結果に戻る。
- **GPU**: 行検出とencoderをCUDAで動かせる。行認識が支配的なので、GPUがあれば1コマ数秒で終わる。
- **再現性**: モデルはバージョン名で固定し、取得時にサイズとSHA-256を照合する。JSONには入力画像・各モデル・語彙・
  設定・パッケージとランタイムの版の指紋が入り、どの重みと設定から得た出力かを後から確かめられる。
  onnxruntimeの版やデバイスをまたいで出力が一致するかは確かめていない。
- **ブラウザ不要**: サーバやWSL、ヘッドレス環境で動く。IndexedDBのキャッシュもWeb Workerもいらない。

出力はブラウザ版と同じKoji記法なので、ブラウザ版で作った翻刻と混ぜて扱える。

## 性能

同じ見開き1コマ（6,496×4,613px、長辺3,500pxに縮小、Python版21行・ブラウザ版22行）を、同じ機械（AMD Ryzen 7 9800X3D、16スレッド）で処理した所要時間。
ブラウザ版は配信中の[honkoku-ocr-web](https://yuta1984.github.io/honkoku-ocr-web/)（2026-09-06、モデルv18）を2通りの環境で動かした値。
LinuxのヘッドレスChromiumではWebGPUアダプターを取得できず、encoderはWebAssembly（int8）、認識ワーカーは8本。
WindowsのヘッドレスChromeでは同じRTX 5070 TiのWebGPUアダプターを確認し、fp16 encoder・認識ワーカー2本で別途測定した。
モデルの取得とセッション作成は含まない。

| 実装 | 行検出 | 行認識 | 合計 |
|------|-------:|-------:|-----:|
| ブラウザ版（WebAssembly、8ワーカー） | 2.9秒 | 37.5〜43.2秒（2回の実測） | 約40〜46秒 |
| ブラウザ版（Windows WebGPU、fp16、2ワーカー） | 2.02〜2.03秒 | 3.05〜3.35秒 | 5.07〜5.37秒（warm、2回） |
| honkoku-ocr-py CPU、encoder fp32（既定） | 0.2秒 | 19.5秒（encoder 18.2秒、decoder 0.9秒） | 20.6秒 |
| honkoku-ocr-py CPU、encoder fp16（`--encoder-precision fp16`） | 0.2秒 | 約150秒 | 152秒 |
| honkoku-ocr-py CUDA（RTX 5070 Ti、encoder fp16、`--threads 2 --decoder-threads 2`） | 0.04秒 | 1.1秒（encoder 0.48秒、decoder 0.36秒、前処理 0.21秒） | 1.46秒（warm、3回の中央値） |

表の値はすべて、行検出の入力縮小を平均化しない双一次補間に改める前に測った。改めた後もこの見開きの検出は21行で、
1,024px入力への縮小が変わっただけなので、所要時間は変わらない。

配信されているencoderはfp16で、onnxruntimeのCPUプロバイダではこれをそのまま動かすと1行7.3秒かかる。
同じ重みをfp32に直したファイルは1行0.86秒で、hidden stateの差は最大1e-3程度、この見開きでは21行中1行が
行末の全角空白の有無だけ違った。CPUでは初回に変換してキャッシュし（366MB、数秒）、CUDAではfp16のまま使う。

CUDAの行は同じ見開きを1回の暖機のあと3回処理した中央値（1.508、1.444、1.464秒）で、3回とも同じ文字列を出した。モデルの遅延読み込みを含む最初の1コマは3.51秒、プロセスのピークRSSは約2.0GiB。decoderはCUDAでもCPUで動く。

WebGPU版は初回の行認識が4.65秒、その後の認識のみの再実行が3.32秒と3.22秒。表は続けて行検出からやり直した2回の値。
この1コマではPython CUDA版の合計時間はWebGPU版の約1/3.5〜1/3.7だった。
ブラウザの行検出はWebGPU使用時もCPUのWASMで動く。行認識だけでは3.05〜3.35秒対約1.1秒で、Python版が約2.8〜3.0倍速い。
ブラウザは2ワーカーで認識を並行処理し、Python版は逐次処理する。ブラウザはWindows、PythonはWSL上で動作し、OSによる影響は未測定。検出行数も異なる。ブラウザは画像の準備後からUIの完了までを100ms間隔で測り、Pythonの合計には画像読み込み約0.36秒も含む。
測定条件と各回の値は[benchmarks/webgpu-comparison.json](benchmarks/webgpu-comparison.json)に記録した。

ブラウザ版とこの移植では同じ見開きで検出行数（22行と21行）も行の読みも一部異なる。
どちらが正しいかは正解翻刻との照合が要る。測定に使った見開きは公開許諾を確かめていない手元のスキャンで、
このリポジトリには含めない。[benchmarks/ocr.py](benchmarks/ocr.py)は`attribution`（出典）と`samples`（各要素は`id`、`image`、任意の`frame`と正解`reference`）を持つ
JSON manifestを受け取り、ページ単位の所要時間と、正解があればCERを出す。手元の画像と翻刻で同じ測定ができる。
みんなで翻刻の翻刻文そのものとの突き合わせは[benchmarks/README.md](benchmarks/README.md)の「Whole pages against みんなで翻刻 transcriptions」に
ある。20コマで、文字の取りこぼしは正規化後6.6%、読み順と行の切り方まで含めた文字誤り率は13.7%だった。読み順とKoji変換の原実装との比較は[benchmarks/README.md](benchmarks/README.md)を参照。

## 使い方

PyPIの0.2.0は2026-09-06の版で、`--resume`と`--boxes`までを含む。`--overlap`、`process_many`、参照コーパスは
このリポジトリのmainにあり、次の版で公開する。ここに書く使い方はmainのもの。

PyPIから:

```sh
uvx --from "honkoku-ocr-py[cpu]" honkoku-ocr page.jpg -o out   # 環境を作らずに実行
uv tool install "honkoku-ocr-py[cpu]"                            # honkoku-ocr コマンドを常設
uv tool install "honkoku-ocr-py[gpu]"                            # CUDA 12 / cuDNN 9 の環境向け
pip install "honkoku-ocr-py[cpu]"
```

リポジトリから:

```sh
uv sync --extra cpu          # onnxruntime (CPU)
uv sync --extra gpu          # onnxruntime-gpu と CUDA 12 のランタイム (cpu とは排他)

uv run honkoku-ocr --download                    # モデルを取得して照合 (4 ファイル 289 MB、~/.cache/honkoku-ocr/models)
uv run honkoku-ocr page.jpg -o out               # out/page__<hash>.txt (Koji 記法、読み順) と out/page__<hash>.json
uv run honkoku-ocr pages/ -o out --device cuda   # ディレクトリ内の画像を一括処理
uv run honkoku-ocr pages/ -o out --resume        # 済んだページを飛ばして続きから
uv run honkoku-ocr page.jpg -o out --plain       # タグ無しの素テキスト
uv run honkoku-ocr page.jpg -o out --preview     # 行 bbox と読み順を描いた PNG も書く
uv run honkoku-ocr page.jpg -o out --layout-only # 行検出だけ (行認識モデルを読まない)
uv run honkoku-ocr page.jpg -o out --boxes 'out/page__<hash>.json'   # 行位置を与えて認識だけ
uv run honkoku-ocr scans.tif -o out --frame 3    # 多ページ TIFF の 4 コマ目だけ (省略時は全コマ)
```

主なオプション。全体は`honkoku-ocr --help`。

| オプション | 意味 |
|------------|------|
| `--model {v16fs,v17,v18}` | 行認識モデルの版（既定v18） |
| `--device {cpu,cuda}` | 行検出とencoderのデバイス。decoderは常にCPU |
| `--encoder-precision {auto,fp16,fp32}` | autoはCPUでfp32、CUDAでfp16 |
| `--threads N` / `--decoder-threads N` | onnxruntimeのスレッド数。0で既定 |
| `--overlap` | 行の前処理とencoderを別スレッドで先行させ、decodeと重ねる。既定はoff。1コマの測定では出力が同じまま所要時間が22%短かった（[benchmarks/README.md](benchmarks/README.md)） |
| `--offline` | キャッシュに無いモデルを取りに行かず失敗する |
| `--verify-cache` | キャッシュ済みモデルのSHA-256を照合して終了 |
| `--max-dimension` `--margin` `--confidence-threshold` `--ios-threshold` | 縮小の長辺（3500）、行cropの余白（45）、行検出のスコア閾値（0.3）、入れ子除去の閾値（0.8） |

**出力の名前**。1画像につき`<stem>__<16桁hex>.json`と`.txt`（`--preview`なら`.preview.png`も）。
hexは入力の絶対パスのSHA-256の先頭16桁で、同じstemの画像が別のディレクトリにあっても衝突せず、
別の呼び出しで一部だけ処理しても名前が変わらない。多ページ画像はさらに`__p0001`のようにコマ番号が付く。
入力ディレクトリを移動すると名前が変わる。

**0.1.0からの変更**。0.1.0の出力は`<stem>.txt`と`<stem>.json`で、JSONの行は`confidence`を持ち、`--version`はモデルの版を選ぶ
オプションだった。0.2.0では出力名に上のhexが付き、行のスコアは`detection_confidence`、モデルの版は`--model`で選び、
`--version`はパッケージの版を表示する。出力ファイルと入力画像の対応はJSONの`image`フィールドで取る（stemに`__`を含む
ファイル名もあるので、名前を切って戻さない）。

**再開**。各ページのJSONは完了記録で、txt（とpreview）を書き終えてから最後に置かれる。書き込みは一時ファイル経由なので
途中で止めても壊れたファイルは残らない。`--resume`はJSONの指紋（画像のSHA-256、コマ番号、各モデルと語彙のSHA-256、
設定、パッケージとコードとランタイムの版）が今回と一致し、txt等のSHA-256も記録どおりのときだけそのページを飛ばす。
モデルや設定を変えれば作り直す。

読めない画像や処理中に失敗したページは標準エラーに出して次へ進み、最後に`N completed, N skipped, N failed`を出す。
失敗が1つでもあれば終了コードは1。

## Pythonから

```python
from pathlib import Path

from honkoku_ocr import OCR, Box

ocr = OCR("v18", device="cuda")          # モデルは最初に使う段階で読み込む
for path in sorted(Path("pages").glob("*.jpg")):
    page = ocr.process(path)             # PageResult
    for line in page.lines:
        print(line.reading_order, line.koji)   # line.raw にタグ付きの生文字列、line.plain に素テキスト
    print(page.timings["total"], page.warnings)
```

1つの`OCR`を使い回す。`ocr_image(path)`は1回ごとにモデルを読み直す簡易関数なので、複数ページには向かない。

複数ページは`process_many`が順に返す。失敗したページは`PageFailure`になり、残りは続く。
モデルの取得やセッション作成の失敗は`ModelSetupError`で止まる。

```python
from pathlib import Path
from threading import Event

from honkoku_ocr import OCR, PageFailure

stop = Event()                            # 別スレッドから stop.set() で中断できる
ocr = OCR("v18", device="cuda", overlap=True)
for outcome in ocr.process_many(sorted(Path("pages").glob("*.jpg")), cancelled=stop.is_set,
                                progress=lambda i, o: print(i, type(o).__name__)):
    if isinstance(outcome, PageFailure):
        print("failed:", outcome.index, outcome.error_type, outcome.message)
        continue
    print("\n".join(line.koji for line in outcome.lines))
```

`process_many`の入力にはパスのほか`PageInput(source, boxes=..., frame=...)`も渡せる。

- `ocr.process(image, boxes=None, *, frame=0, layout_only=False)`は`PageResult`を返す。`image`はパス、
  `PIL.Image`、または`ocr.prepare(path)`が返す`PreparedPage`。
  `PreparedPage`は`with ocr.prepare(path) as prepared:`で使うか、使用後に`prepared.close()`で閉じる。
- `ocr.run(image, boxes=None)`は`process(...).lines`、`ocr.layout(image)`は行検出だけを行い`Box`の一覧を返す。
  どちらも必要なモデルしか読まない。
- 座標はすべてEXIFの向きを反映した元画像のもの（`settings.coordinate_space`は`exif_oriented_original`）。
  処理は長辺3,500pxに縮小した画像で行い、結果は元画像の座標に戻す。
- `boxes=[Box(x, y, w, h, confidence), ...]`を与えると行検出を飛ばし、与えた順を読み順、与えた座標をそのまま
  結果の座標とする。幅と高さは正、confidenceは0〜1で、boxが元画像と重なっている必要がある。
  はみ出した部分は認識用の切り出しで除くが、返す座標は変更しない。
- 行検出器や行認識器を差し替えるには`OCR(detector=..., recognizer=...)`。`detect(image, conf_threshold, ios_threshold)`と
  `recognize_result(crop)`を実装したオブジェクトであればよい。

`PageResult`の内容。

| フィールド | 内容 |
|-----------|------|
| `schema_version` | 1 |
| `width`, `height` | EXIFの向きを反映した元画像の大きさ |
| `processed_width`, `processed_height` | 縮小後、実際にモデルへ渡した画像の大きさ |
| `frame` | 多ページ画像のコマ番号（0始まり） |
| `model`, `settings` | 行認識モデルの版と、デバイス・スレッド・精度・縮小・余白・閾値 |
| `lines` | 読み順に並んだ`LineResult` |
| `timings` | 段階ごとの秒数。`load`, `detector_setup`, `layout`, `reading_order`, `recognizer_setup`, `preprocess`, `encoder`, `prefill`, `decode`, `total` |
| `warnings` | 行末まで復号できなかった行など |

`LineResult`は`reading_order`, `x`, `y`, `width`, `height`, `detection_confidence`（行検出のスコア。与えたboxならその値）,
`raw`, `koji`, `plain`, `stop_reason`（`eos`は終端トークンで停止、`repetition`は反復崩壊の打ち切り、`max_tokens`は上限192トークン、
`not_run`は`layout_only`）, `token_count`, `timings`（行ごとの`preprocess`, `encoder`, `prefill`, `decode`）を持つ。

## 出力ファイル

CLIのJSONは`PageResult`に`image`（入力のファイル名）、`fingerprint`、`artifacts`（同時に書いたファイルのSHA-256）を加えたもの。

```json
{
  "schema_version": 1,
  "width": 6496, "height": 4613,
  "processed_width": 3500, "processed_height": 2485,
  "frame": 0,
  "model": "v18",
  "settings": {"device": "cpu", "threads": 0, "decoder_threads": 0, "encoder_precision": "auto",
               "max_dimension": 3500, "margin": 45, "conf_threshold": 0.3, "ios_threshold": 0.8,
               "coordinate_space": "exif_oriented_original"},
  "lines": [
    {"reading_order": 1, "x": 4239, "y": 1221, "width": 334, "height": 422,
     "detection_confidence": 0.6066901683807373,
     "raw": "<ruby>大印<rt>おほしつし</rt></ruby>", "koji": "大印（おほしつし）", "plain": "大印おほしつし",
     "stop_reason": "eos", "token_count": 12,
     "timings": {"preprocess": 0.008, "encoder": 0.672, "prefill": 0.004, "decode": 0.037}}
  ],
  "timings": {"load": 0.269, "detector_setup": 0.075, "layout": 0.173, "reading_order": 0.003,
              "recognizer_setup": 0.551, "preprocess": 0.434, "encoder": 18.227, "prefill": 0.197,
              "decode": 0.672, "total": 20.61},
  "warnings": ["line 7: generation stopped by repetition"],
  "image": "0003.jpg",
  "fingerprint": {
    "source_sha256": "e2ed654248a0…", "frame": 0,
    "models": {"layout": {"file": "rtmdet-s-1280x1280.onnx", "sha256": "f46267754d40…"},
               "encoder": {"file": "kuzushiji-v18-encoder-fp32.onnx", "sha256": "3b3c359bd426…"},
               "prefill": {"file": "kuzushiji-v18-decoder-prefill-int8.onnx", "sha256": "6f3f19011f8d…"},
               "step": {"file": "kuzushiji-v18-decoder-step-int8.onnx", "sha256": "bf0e72a80716…"},
               "vocabulary": "cf0621e68b09…"},
    "settings": {"device": "cpu", "...": "..."},
    "model": "v18", "plain": false, "layout_only": false, "boxes": null,
    "package_version": "0.2.0", "code_sha256": "dfde67eac894…",
    "runtime_versions": {"numpy": "2.5.2", "pillow": "12.3.0", "onnxruntime": "1.29.0", "onnx": "1.22.0"}
  },
  "artifacts": {"0003__5de8f1b5d7a5d622.txt": "ca6949193d5e…"}
}
```

txtは`koji`（`--plain`なら`plain`）を読み順に1行ずつ並べたもの。上の例は構造を示すため、行の一部とハッシュ値を省略している。

## ブラウザ版との違い

- UI（画像ビューア、bboxの編集、縦書き表示、PDF/HEICの読み込み、IIIF、LLM連携）は含まない。多ページTIFFは読める。
- encoderはfp16版を使う。ブラウザ版のWebAssembly経路が使うint8版のConvInteger演算はonnxruntimeのCPU/CUDAプロバイダに無い。
  CPUではfp16版をfp32に変換したファイルを使う。対応する版はfp16 encoderが配布されているv16fs / v17 / v18。
- decoderは`--device cuda`でもCPUで動く。
- 行検出の入力（1024×1024）への縮小はブラウザのcanvasと同じ平均化しない双一次補間で行う。ページの長辺3,500pxへの縮小と
  行画像の縮小・回転はPillow（Lanczos、bicubic）で、ブラウザのcanvasとは補間が異なる。
- 画像の端にかかる行では傾き推定の二値化閾値が異なる。ブラウザ版は画像外の画素を透明（輝度0）のまま平均に入れ、
  この移植は白で埋めてから平均を取る。処理画像で既定の余白45pxが画像外にはみ出す行に影響する。
- 行検出はRTMDetのみ（ブラウザ版の5クラスYOLOは含まない）。

## 環境変数

- `HONKOKU_OCR_MODELS` … モデルの保存先（既定`~/.cache/honkoku-ocr/models`）。fp32変換したencoder（366MB）と来歴ファイル
  `<name>.json`も同じ場所に置く。
- `HONKOKU_OCR_MODEL_URL` … モデル配信元（既定は原著作物と同じ公開バケット）

## トラブルシューティング

- **`CUDAExecutionProvider is not available`** … `uv sync --extra gpu`でonnxruntime-gpuとCUDA 12 / cuDNN 9のランタイムを入れる。
  cpuとgpuのextraは同時に入らない。`--device cpu`に戻せば動く。
- **`SHA-256 mismatch` / `size ... != expected`** … 取得途中で壊れたか配信元が変わった。`~/.cache/honkoku-ocr/models`の
  該当ファイルを消して再実行する。`--verify-cache`でキャッシュ全体を照合できる。
- **`missing from model cache in offline mode`** … `--offline`または`--verify-cache`でキャッシュに無いモデルを求めた。
  ネットワークのある環境で`honkoku-ocr --download --model v18`を先に実行する。
- **CPUで1行に数秒かかる** … `--encoder-precision fp16`を指定しているか、fp32変換が失敗している。JSONの
  `fingerprint.models.encoder.file`が`-fp32.onnx`になっているか確かめる。
- **`DecompressionBombWarning` / `DecompressionBombError`** … Pillowの既定は約8,900万画素で警告、その2倍で停止する。
  それより大きなスキャンは事前に縮小するか、`PIL.Image.MAX_IMAGE_PIXELS`を上げる。
- **`box must overlap the EXIF-oriented image`** … `--boxes`のbboxが画像と重なっていない。座標は回転を反映した
  元画像のもの。少しはみ出したboxは認識時に画像内へ切り詰め、返す座標は元のままにする。
- **メモリ** … CPUのfp32 encoderは約1GB、CUDAのfp16 encoderはVRAM約1GBを使う。ワーカーを並列に立てるなら
  その分だけ増える。

## テスト

```sh
uv sync --extra cpu
uv run pytest
uv run ruff check .
```

CIはPython 3.10〜3.14でlintとテストを走らせ、ビルドしたwheelから語彙ファイルが読めることを確かめる。

## ライセンスと帰属

このリポジトリのPythonコードとテストはMITライセンス（[LICENSE](LICENSE)）。

原著作物に由来する部分はそれぞれの著作者のCC BY 4.0のままである（[LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt)、
一覧は[NOTICE.md](NOTICE.md)）:
みんなで翻刻OCR（橋本雄太）— モデル、語彙ファイル`honkoku_ocr/config/`、`docs/tech.html`、および移植元となった推論手順。
NDL古典籍OCR-Lite（国立国会図書館）— 行検出モデル、XY-Cutの手続き。
学習データは「みんなで翻刻」の翻刻成果に基づく。

引用する場合は原著作物を挙げること:
橋本雄太「みんなで翻刻OCR — 市民の力で作ったくずし字AI-OCR」 https://yuta1984.github.io/honkoku-ocr-web/
