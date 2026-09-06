# みんなで翻刻 page corpus

Twenty pages with their full transcriptions from みんなで翻刻, for measuring the
whole pipeline: line detection, reading order and recognition together, with no
line boxes given. Images are not stored here; `python -m benchmarks.fetch_corpus
benchmarks/corpus-honkoku/manifest.json` downloads them from the NDL IIIF
endpoints listed in the manifest and checks their SHA-256.

## Selection

One page per book, drawn with seed 20260906 from NDL-hosted books in eight
projects (賀茂社記録 ×4, 地震関係 ×3, 草双紙 ×3, 切支丹 ×3, 医療と養生 ×2, 図譜 ×2,
疫病 ×1, Code4Lib JAPAN ×2), restricted to pages whose status in honkoku-data is
`completed` and whose transcription has 8 to 40 lines. Books already used in
`benchmarks/corpus/` were excluded. Manuscripts and printed books are both
present.

## Sources and licences

- Page images: 国立国会図書館デジタルコレクション, https://dl.ndl.go.jp/ ,
  public-domain works; each sample records its IIIF URL and the book's pid.
- Transcriptions: みんなで翻刻データ v3, https://github.com/yuta1984/honkoku-data
  (commit 58ceb1fdc, 2025-11-23), CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). Each sample names the file
  under `v3/` it was copied from. The manifest and this directory are under
  CC BY-SA 4.0, unlike the code in this repository (MIT).

## Caveats

The transcriptions are by volunteers and unreviewed; the project reports about
1.5 errors per 100 characters. They follow みんなで翻刻's conventions rather than
the page's physical lines: items and quantities of a list may share one
transcription line, marginal notes are placed where the transcriber chose, and
editorial remarks appear in 【】. The kuzushiji models were trained on みんなで翻刻
data and may have seen these pages. Read the results as agreement with those
transcriptions under this port's segmentation and reading order.
