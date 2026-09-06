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
