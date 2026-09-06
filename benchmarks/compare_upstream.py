"""Deterministic component parity/latency comparison; no models or images needed."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from honkoku_ocr.koji import raw_to_koji, raw_to_plain
from honkoku_ocr.reading_order import order


def fixtures():
    rng = random.Random(20260906)
    cases = [[], [(0, 0, 10, 10)]]
    for n in (2, 5, 15, 40, 80):
        cases.append([(x * 60, 0, 30, 900) for x in range(n)])
        cases.append([(0, y * 60, 900, 30) for y in range(n)])
        cases.append([(x * 60, y * 600, 30, 500) for y in range(2) for x in range(n)])
        for _ in range(10):
            cases.append([(rng.randrange(-100, 1500), rng.randrange(-100, 2000),
                           rng.randrange(6, 300), rng.randrange(6, 900)) for _ in range(n)])
    cases += [[(0, 0, 10, 100)] * 5, [(0, 0, 0, 0)] * 2]
    parts = ["", "本文", "𠮷", "<ruby>漢字<rt>かんじ</rt></ruby>",
             "<ruby>親<rt>おや</rt><rt2>よみ</rt2></ruby>",
             "<OKURI>かな</OKURI>", "<KAERI>レ</KAERI>",
             "<WARI>右<WARI_SEP>左</WARI>", "<WARI>注</WARI>",
             "<TATE><BLOCK>", "<unknown>文</unknown>", "<ruby>未完", "\n"]
    texts = parts + ["".join(rng.choices(parts, k=8)) for _ in range(200)]
    return cases, texts


def measure(fn, repeats):
    for _ in range(3):
        fn()
    samples = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {"samples_ms": samples, "output": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Local checkout of honkoku-ocr-web")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cases, texts = fixtures()
    payload = {"cases": cases, "texts": texts, "repeats": args.repeats}
    js = json.loads(subprocess.run(
        ["node", "--experimental-strip-types", str(Path(__file__).with_name("upstream.mjs")),
         str(args.upstream.resolve())], input=json.dumps(payload), text=True,
        capture_output=True, check=True).stdout)
    py = {"order": measure(lambda: [order(boxes) for boxes in cases], args.repeats),
          "text": measure(lambda: [[raw_to_koji(t), raw_to_plain(t)] for t in texts], args.repeats)}
    report = {
        "scope": "Synthetic reading-order and Koji conversion components; excludes OCR inference, image preprocessing and startup",
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_source_sha256": {name: hashlib.sha256((args.upstream / name).read_bytes()).hexdigest()
                                   for name in ("src/ocr/reading-order.ts", "src/lib/koji.ts")},
        "python_source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                 for name in ("honkoku_ocr/reading_order.py", "honkoku_ocr/koji.py")},
        "python": platform.python_version(),
        "node": subprocess.check_output(["node", "--version"], text=True).strip(),
        "platform": platform.system() + " " + platform.machine(),
        "seed": 20260906, "warmups": 3, "repeats": args.repeats,
        "components": {},
    }
    failed = False
    for name, count in (("order", len(cases)), ("text", len(texts))):
        mismatches = [i for i, (a, b) in enumerate(zip(py[name]["output"], js[name]["output"])) if a != b]
        if len(py[name]["output"]) != len(js[name]["output"]):
            raise RuntimeError("upstream returned an unexpected output count")
        failed |= bool(mismatches)
        report["components"][name] = {
            "cases": count, "mismatch_indices": mismatches,
            "python_median_ms": statistics.median(py[name]["samples_ms"]),
            "upstream_median_ms": statistics.median(js[name]["samples_ms"]),
            "python_samples_ms": py[name]["samples_ms"],
            "upstream_samples_ms": js[name]["samples_ms"],
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
