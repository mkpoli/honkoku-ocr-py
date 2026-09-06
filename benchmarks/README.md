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
