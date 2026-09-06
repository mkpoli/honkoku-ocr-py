# Upstream component comparison

This benchmark executes the original TypeScript reading-order and Koji conversion
code from a local checkout of [honkoku-ocr-web](https://github.com/yuta1984/honkoku-ocr-web).
Node strips types without rewriting the algorithms. The Python implementation
receives the same generated inputs. A mismatch makes the command exit nonzero.

```sh
git clone https://github.com/yuta1984/honkoku-ocr-web /tmp/honkoku-ocr-web
git -C /tmp/honkoku-ocr-web checkout f0b0388a2744daaec4c92979e86516e4f8b3f8fd
uv run python -m benchmarks.compare_upstream /tmp/honkoku-ocr-web --output /tmp/comparison.json
```

Requires Node 22.18+ with TypeScript stripping support and the Python package's
base dependencies. No ONNX models, GPU, browser, or npm installation is needed.
Run from the repository root. The JSON records source hashes, upstream commit,
runtime versions, individual timing samples, and mismatching case indices.

## Recorded results

Measured on Linux/WSL, AMD Ryzen 7 9800X3D, Python 3.12.7, Node v22.23.2.
Each process warms up with three complete suites, then measures seven suites.
Times are medians for the entire component suite, excluding process startup,
imports, fixture generation, and JSON serialization. Both implementations run
sequentially on CPU. Background workloads were present; these timings are
indicative and do not establish a stable speedup.

| Component | Cases | Python before | Python after | Upstream after-run baseline | Mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reading order | 69 | 210.79 ms | 174.06 ms | 17.20 ms | 0 |
| Koji + plain conversion | 213 | 0.94 ms | 0.93 ms | 0.50 ms | 0 |

[baseline.json](baseline.json) records the Python reading-order implementation at
`8d47496`. [results.json](results.json) records the NumPy mesh/histogram version.
Text conversion was unchanged. The first post-change trial took 121.62 ms for
reading order; the recorded repeat took 174.06 ms, illustrating host variability.
The original TypeScript remained faster in both trials.

Reading-order cases include vertical columns, horizontal rows, two-tier pages,
random overlapping boxes, duplicates, empty input, and zero-size boxes. Text
cases combine ruby, supplementary-plane kanji, warichu, okurigana, kaeriten,
unknown tags, incomplete tags, and newlines. All generation uses seed 20260906.
These checks establish agreement on these fixtures, not correctness on all inputs.

## Scope

This is a component benchmark. It does not measure image resampling, deskew,
line detection, neural inference, browser worker concurrency, page throughput,
or transcription accuracy. Pillow and browser canvas can produce different
pixels. A full OCR comparison needs identical page images, model versions,
explicit execution providers, warmup rules, and ground-truth transcriptions
for character error rate; matching upstream text alone does not measure accuracy.

# Full-page OCR benchmark

`benchmarks/ocr.py` runs the complete pipeline (load, detection, reading order,
recognition) on the pages listed in a JSON manifest and records per-page timing.
When a page carries a reference transcription it also reports the character
error rate. There is no curated reference corpus in this repository; the manifest
points at the caller's own images, and `cer` is null for pages without a
`reference`.

```sh
uv run python -m benchmarks.ocr pages/manifest.json --output /tmp/ocr-benchmark.json
uv run python -m benchmarks.ocr pages/manifest.json --output /tmp/ocr-benchmark.json --device cpu --encoder-precision fp16
```

Manifest, resolved relative to its own directory:

```json
{
  "attribution": "Where the images come from, who transcribed them, and under what licence",
  "samples": [
    {"id": "p001", "image": "p001.jpg"},
    {"id": "p002", "image": "scans.tif", "frame": 3, "reference": "正解の翻刻テキスト"},
    {"id": "p003", "image": "p003.jpg",
     "boxes": [{"x": 4239, "y": 1221, "width": 334, "height": 422, "confidence": 1.0}]}
  ]
}
```

`attribution` and a nonempty `samples` list are required; `id` values must be
unique strings. `frame` selects a page of a multipage image, `boxes` replaces
line detection with the given boxes in EXIF-oriented source coordinates, and
`reference` is the transcription the prediction is compared with. The comparison
uses the format selected by `--format` (`plain`, the default, or `koji`), so the
reference must be written in the same format.

Options: `--output` (required), `--model`, `--device` (default `cuda`),
`--encoder-precision`, `--threads`, `--decoder-threads`, `--warmups` (default 1),
`--repeats` (default 3), `--format`, `--offline`. The first sample is processed
once before the warmups so that model loading and the fp32 encoder conversion
stay out of the measurement; every sample is then processed `warmups` + `repeats`
times and the median of the repeats is reported as `median_seconds`.

The output JSON records the settings, the model files with their SHA-256, runtime
versions, and per sample the individual run times, stage timings, predictions,
and, when a reference exists, the edit distance and CER of every run. Predictions
and references are copied into the output, so publish it only when the manifest's
attribution allows the transcriptions to be redistributed.

## Overlapping recognition stages

Pass `--overlap` to the full-page benchmark to run preprocessing and encoding
on one background thread while the calling thread decodes the preceding line.
The queue holds at most two futures ahead of the decoder. Stage timings measure
individual calls and can sum to more than the page elapsed time.

[stage-overlap.json](stage-overlap.json) records five warm runs per mode on the
same private spread, using CUDA fp16 with two encoder/layout threads and two
decoder threads. The median fell from 1.525 s serial to 1.187 s with overlap
(22% less elapsed time; 1.28 times the throughput). All ten outputs matched.
Both modes used the earlier antialiased layout resampling and detected 21 lines;
these results precede the canvas-resampling correction. Serial runs preceded
overlap runs, with background CPU work present. This single-page result does not
establish the gain on other documents or hardware. Overlap remains opt-in.

# Reference corpus results

`benchmarks/corpus/manifest.json` lists four vertically written pages from the
NDL みんなで翻刻 dataset with reference line boxes and transcriptions (sources and
licence in `benchmarks/corpus/README.md`). Fetch the images first:

```sh
uv run python -m benchmarks.fetch_corpus
uv run python -m benchmarks.ocr benchmarks/corpus/manifest.json --output corpus-cer.json --device cpu --format plain --warmups 0 --repeats 1
```

## Line detection against the reference boxes

`benchmarks/detection_agreement.py` runs the detector on each page (3,500 px long
side, default thresholds) and matches detected boxes to reference boxes one to
one, greedily by descending IoU, counting pairs at IoU ≥ 0.5. The dataset
matched transcriptions to lines mechanically and its own notes say the matching
is incomplete, so an unmatched detection may be real text that has no reference
line. The columns are therefore agreement with the available references, not
detection accuracy. Result file: `corpus-detection-agreement.json`.

```sh
uv run python -m benchmarks.detection_agreement benchmarks/corpus/manifest.json --output corpus-detection-agreement.json --device cpu
```

| page | reference lines | detected | matched | matched / detected | matched / reference | mean IoU of matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kamosha-799137F6-029 | 24 | 24 | 22 | 0.92 | 0.92 | 0.91 |
| kamosha-799137F6-030 | 20 | 24 | 20 | 0.83 | 1.00 | 0.92 |
| iryotoyojo-DC7F12BC-028 | 20 | 22 | 20 | 0.91 | 1.00 | 0.92 |
| iryotoyojo-DC7F12BC-030 | 21 | 22 | 21 | 0.95 | 1.00 | 0.93 |
| all | 85 | 92 | 83 | 0.90 | 0.98 | |

On page 029 one detection spans two reference lines, so two references stay
unmatched under one-to-one matching. Before the letterbox resize was changed to
point sampling the detected counts on these four pages were the same.

## Character error rate with the reference boxes

Recognition ran on the reference boxes (`boxes` in the manifest), so the numbers
measure line recognition alone. Plain text, newline-joined per page, exact edit
distance. The v18 model was trained on みんなで翻刻 data and may have seen these
pages; use the figures to compare settings of this port with each other, not
against published accuracy. Runs: `corpus-cer-fp32.json`, `corpus-cer-fp16.json`,
both `--device cpu --threads 0 --decoder-threads 0` (onnxruntime's own thread
defaults on a 16-thread CPU), one pass each with no additional warmups; the
harness processes the first sample once for setup before measuring.

| page | lines | CER, encoder fp32 | CER, encoder fp16 | seconds fp32 | seconds fp16 | fp16 / fp32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kamosha-799137F6-029 | 24 | 0.083 | 0.083 | 25.8 | 169.0 | 6.55 |
| kamosha-799137F6-030 | 20 | 0.119 | 0.122 | 28.8 | 135.6 | 4.71 |
| iryotoyojo-DC7F12BC-028 | 20 | 0.282 | 0.282 | 48.3 | 134.9 | 2.79 |
| iryotoyojo-DC7F12BC-030 | 21 | 0.235 | 0.235 | 35.6 | 143.9 | 4.04 |
| all | 85 | 0.1927 | 0.1932 | 138.4 | 583.4 | 4.22 |

Over 1,853 reference characters the fp32 encoder made 357 character errors and
the fp16 encoder 358; 83 of the 85 line predictions are identical. This is one
pass per setting, so there is no estimate of run-to-run variation. Per page the
fp16 encoder took 2.8 to 6.5 times as long, 4.2 times over the four pages. The
two 新竹斎 pages are printed text with dense kana and score worse than the
賀茂社 manuscript pages; four pages are too few to say why.

# Thread settings on one page

`benchmarks/thread_matrix.py` times one manifest page under a matrix of
`--threads` (layout and encoder sessions) and `--decoder-threads` (prefill and
step sessions). `thread-matrix.json` records this run:

```sh
uv run python -m benchmarks.thread_matrix benchmarks/corpus/manifest.json --sample kamosha-799137F6-030 \
  --output thread-matrix.json --device cpu --threads 0 2 4 --decoder-threads 0 2 --warmups 0 --repeats 2
```

CPU only, fp32 encoder, 20 lines with the reference boxes supplied, one setup
pass then two timed passes per cell, medians below. Thread count 0 leaves the
choice to onnxruntime, whose documentation describes the default as one thread
per physical core; this machine exposes 8 physical cores and 16 logical CPUs.

| threads | decoder threads | encoder s | decode s | total s (two runs) |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 22.35 | 1.42 | 24.92 (27.53, 22.32) |
| 0 | 2 | 21.56 | 0.80 | 23.39 (24.64, 22.13) |
| 2 | 0 | 30.43 | 1.14 | 32.70 (32.60, 32.80) |
| 2 | 2 | 30.17 | 0.42 | 31.53 (32.54, 30.52) |
| 4 | 0 | 18.86 | 1.39 | 21.41 (21.36, 21.47) |
| 4 | 2 | 19.08 | 0.47 | 20.39 (20.09, 20.70) |

Within each cell the two repeats produced the same text; texts were not compared
across cells. On this page and machine four encoder threads ran the fp32 encoder
faster than the runtime default, and two decoder threads reduced the decode time
(by 44%, 63% and 66% against the matching 0-decoder-thread cells). The two runs
of the default cell differ by five seconds, so single-page medians of two runs
are indicative only; the defaults stay at 0 and 0 until more pages and machines
are measured.

# Whole pages against みんなで翻刻 transcriptions

`benchmarks/corpus-honkoku/manifest.json` lists twenty pages, one per book,
from NDL-hosted books in eight projects, with the complete page transcription
from honkoku-data v3 as reference and no line boxes (sources and licence in
`benchmarks/corpus-honkoku/README.md`). `benchmarks/page_eval.py` runs the whole
pipeline and compares the page text in Koji notation. Result file:
`corpus-honkoku-page-eval.json`, CPU, fp32 encoder, `--threads 0
--decoder-threads 0`, one pass.

```sh
uv run python -m benchmarks.fetch_corpus benchmarks/corpus-honkoku/manifest.json
uv run python -m benchmarks.page_eval benchmarks/corpus-honkoku/manifest.json --output page-eval.json --device cpu
uv run python -m benchmarks.page_eval benchmarks/corpus-honkoku/manifest.json --output page-eval.json --rescore page-eval.json   # recompute metrics only
```

Four views of the same output, because the transcriptions follow the project's
conventions rather than the page's physical lines:

| metric | what it charges | value over 20 pages |
| --- | --- | ---: |
| raw CER | every character, newline and space, in reading order (10,198 reference characters) | 0.260 |
| squeezed CER | as above with 【】 notes and all whitespace removed (8,394 characters); still charges reading order | 0.162 |
| normalised CER | squeezed, then katakana folded to hiragana, fullwidth forms, and the variant pairs in `page_eval.VARIANTS` (顛/顚, 祷/禱, 畧/略, 国/國 …) folded on both sides | 0.137 |
| bag of characters, missed / extra | reference characters with no predicted counterpart, and the reverse, regardless of order and line splits | 0.090 / 0.079 |
| bag of characters after normalisation, missed / extra | the same on the normalised text | 0.066 / 0.055 |
| paired-line CER | each reference line against its best predicted line; charges split and merged lines, ignores order | 0.143 |

498 lines were detected for 426 transcription lines; 425 of the transcription
lines found a partner. Total processing time 544 s on the CPU.

| page | lines pred./ref. | raw | squeezed | paired | missed | extra | in order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kamosha-E3172F3F-144 | 36/26 | 0.608 | 0.462 | 0.192 | 0.083 | 0.038 | 0.84 |
| kamosha-C4906EB6-016 | 33/25 | 0.300 | 0.275 | 0.374 | 0.217 | 0.106 | 0.96 |
| kamosha-D09E2A0F-081 | 41/19 | 0.508 | 0.451 | 0.391 | 0.174 | 0.209 | 1.00 |
| kamosha-1DB6FDAA-022 | 24/24 | 0.301 | 0.176 | 0.176 | 0.161 | 0.069 | 0.96 |
| zisin-1B4A9AF6-034 | 22/20 | 0.038 | 0.036 | 0.034 | 0.024 | 0.022 | 1.00 |
| zisin-5128812C-001 | 17/16 | 0.179 | 0.150 | 0.086 | 0.033 | 0.116 | 1.00 |
| zisin-A1E1992E-012 | 34/20 | 0.254 | 0.236 | 0.223 | 0.087 | 0.051 | 0.95 |
| kusazoushi-63B24B9C-004 | 26/25 | 0.352 | 0.180 | 0.149 | 0.118 | 0.099 | 0.92 |
| kusazoushi-B5A8C444-003 | 22/21 | 0.259 | 0.056 | 0.023 | 0.023 | 0.051 | 1.00 |
| kusazoushi-5FB68B99-035 | 20/20 | 0.184 | 0.094 | 0.094 | 0.047 | 0.018 | 1.00 |
| kirishitan-2B2C88BC-045 | 21/20 | 0.160 | 0.124 | 0.122 | 0.086 | 0.072 | 0.95 |
| kirishitan-CC187A95-077 | 20/20 | 0.399 | 0.130 | 0.010 | 0.010 | 0.010 | 0.89 |
| kirishitan-4A8E2CD2-005 | 20/20 | 0.169 | 0.178 | 0.178 | 0.178 | 0.178 | 1.00 |
| iryotoyojo-57D59E79-020 | 14/14 | 0.073 | 0.017 | 0.017 | 0.017 | 0.013 | 1.00 |
| iryotoyojo-8409705F-025 | 22/20 | 0.172 | 0.058 | 0.031 | 0.013 | 0.046 | 1.00 |
| zukan-C7EC110D-079 | 11/8 | 0.504 | 0.441 | 0.449 | 0.377 | 0.275 | 1.00 |
| zukan-4D490743-017 | 32/27 | 0.735 | 0.588 | 0.432 | 0.204 | 0.284 | 0.77 |
| epidemic-F03F21CC-003 | 24/21 | 0.227 | 0.172 | 0.140 | 0.093 | 0.151 | 1.00 |
| code4libjp-41AD19BB-017 | 33/34 | 0.079 | 0.024 | 0.024 | 0.024 | 0.012 | 1.00 |
| code4libjp-E2D5D507-044 | 26/26 | 0.165 | 0.087 | 0.096 | 0.087 | 0.025 | 1.00 |

What the gaps between the columns are made of, from reading the paired lines:

- **Segmentation conventions.** In the 賀茂社記録 lists each item and its
  quantity share one transcription line while the detector finds two boxes,
  and 割書 and interlinear notes become separate boxes; page 081 has 41 boxes
  for 19 transcription lines. This is charged by raw, squeezed and paired CER
  and not by the bag of characters.
- **Reading order.** The XY-Cut reads a marginal note at the top of page 144
  first; the transcriber put it last. Squeezed CER 0.46 against bag missed 0.08
  on that page is the size of that effect.
- **Transcription conventions.** In the 切支丹 texts transcribers wrote the
  okurigana in katakana (ニ, ノ, ヲ, ハ) where the model writes hiragana, and kept
  the 字母 of 変体仮名 (多, 連, 里) where the model writes the modern kana; the
  reference uses 顛, 祷, 畧 where the model has 顚, 禱, 略, and sometimes omits
  返り点 that the model outputs. Page kirishitan-005 has every metric at 0.178
  for this reason alone; after normalisation it scores 0.005. Over all pages the
  normalisation removes 2.5 points of squeezed CER (0.162 to 0.137) and 2.4
  points of missed characters (0.090 to 0.066); the folded pairs are listed in
  `page_eval.VARIANTS`, and 変体仮名 written with their 字母 are left alone.
- **Illustrated pages.** The two 図譜 pages are drawings with scattered labels;
  the transcription positions them with leading spaces and the detector finds
  labels the transcription lacks.

Pages of continuous prose (地震 034, 儒医東西評林, 小野湖山翁小伝, 竹斎狂歌物語)
reach squeezed CER 0.02 to 0.06. The published figure for the model, plain CER
0.075 on the authors' test set, was measured on line crops with their own
normalisation and cannot be compared with any column here.
