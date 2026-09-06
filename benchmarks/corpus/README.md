# Reference corpus

Four vertically written pages with reference line boxes and transcriptions, for
measuring line detection and character error rate. The images are not stored
here; `python -m benchmarks.fetch_corpus` downloads them from the NDL IIIF
endpoint listed in `manifest.json` and checks their SHA-256.

## Sources and licences

- Page images: 国立国会図書館デジタルコレクション, https://dl.ndl.go.jp/ , public-domain
  works (賀茂社記録 第63冊, https://dl.ndl.go.jp/pid/2540583 ; 新竹斎 5巻,
  https://dl.ndl.go.jp/pid/2557103 ). Each sample records its exact IIIF URL.
- Line boxes and transcriptions: NDL古典籍OCR学習用データセット（みんなで翻刻加工データ）v2,
  2024-02-07, https://github.com/ndl-lab/ndl-minhon-ocrdataset ,
  archive https://lab.ndl.go.jp/dataset/ndlkotensekiocr/ndl-minhon-ocrdataset_20240207.zip ,
  licence CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). It is derived from みんなで翻刻 data,
  https://github.com/yuta1984/honkoku-data , CC BY-SA 4.0. Each sample names the
  dataset JSON it was taken from.

The manifest and this directory are therefore under CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/), unlike the
code in this repository (MIT). Attribution when reusing them: 国立国会図書館 for
the images and the dataset, みんなで翻刻 for the transcriptions.

## Caveats

The dataset matched transcriptions to line images mechanically, so references
contain errors and omissions. The kuzushiji models were trained on みんなで翻刻
data and may have seen these pages. Use the corpus to compare settings of this
port with each other; do not compare its CER with published figures.
