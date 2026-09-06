"""Time one manifest page under a matrix of onnxruntime thread settings.

Each cell runs OCR.process on the same page with the given --threads (layout and
encoder sessions) and --decoder-threads (prefill and step sessions), after one
warmup; the median of the repeats is recorded together with the machine, the
model identity and the exact settings, so the run can be repeated elsewhere.
The default matrix is 15 cells; each cell loads the models afresh and processes
the page warmups + repeats + 1 times, so pick an explicit small matrix for a
quick look (for example --threads 2 4 --decoder-threads 1 2 --repeats 2).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from honkoku_ocr import OCR, Box, models
from honkoku_ocr.output import atomic_write, package_version, safe_error


def run_cell(manifest: Path, sample: dict, *, model: str, device: str, precision: str,
             threads: int, decoder_threads: int, overlap: bool, warmups: int, repeats: int) -> dict:
    ocr = OCR(model, device, quiet=True, threads=threads, decoder_threads=decoder_threads,
              encoder_precision=precision, overlap=overlap)
    boxes = [Box(**box) for box in sample["boxes"]] if "boxes" in sample else None
    image = manifest.parent / sample["image"]
    setup = perf_counter()
    first = ocr.process(image, boxes, frame=sample.get("frame", 0))
    setup = perf_counter() - setup
    for _ in range(warmups):
        ocr.process(image, boxes, frame=sample.get("frame", 0))
    runs = [ocr.process(image, boxes, frame=sample.get("frame", 0)) for _ in range(repeats)]
    texts = {"\n".join(line.plain for line in result.lines) for result in runs}
    stages = {stage: statistics.median(r.timings.get(stage, 0.0) for r in runs)
              for stage in ("load", "layout", "preprocess", "encoder", "prefill", "decode", "total")}
    return {"threads": threads, "decoder_threads": decoder_threads, "overlap": overlap,
            "lines": len(first.lines), "first_page_seconds": setup,
            "total_seconds": [r.timings["total"] for r in runs],
            "median_seconds": statistics.median(r.timings["total"] for r in runs),
            "median_stage_seconds": stages, "identical_text_across_runs": len(texts) == 1,
            "model_identity": ocr.model_identity()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--sample", required=True, help="sample id inside the manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(models.SPECS), default=models.DEFAULT_VERSION)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--encoder-precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--threads", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    parser.add_argument("--decoder-threads", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--overlap", action="store_true", help="also run every cell with overlap")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1 or min(args.threads + args.decoder_threads) < 0:
        parser.error("warmups must be nonnegative, repeats positive and thread counts nonnegative")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample = next((s for s in manifest["samples"] if s["id"] == args.sample), None)
    if sample is None:
        parser.error(f"sample {args.sample!r} is not in the manifest")
    cells = []
    for threads in args.threads:
        for decoder_threads in args.decoder_threads:
            for overlap in ([False, True] if args.overlap else [False]):
                cell = run_cell(args.manifest, sample, model=args.model, device=args.device,
                                precision=args.encoder_precision, threads=threads,
                                decoder_threads=decoder_threads, overlap=overlap,
                                warmups=args.warmups, repeats=args.repeats)
                cells.append(cell)
                print(f"threads={threads:2d} decoder={decoder_threads:2d} overlap={overlap!s:5} "
                      f"median {cell['median_seconds']:.2f}s", file=sys.stderr)
    runtime_versions = {}
    for package in ("numpy", "pillow", "onnxruntime", "onnxruntime-gpu", "onnx"):
        try:
            runtime_versions[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {
        "arguments": {"manifest": safe_error(args.manifest), "sample": args.sample, "model": args.model,
                      "device": args.device, "encoder_precision": args.encoder_precision,
                      "threads": args.threads, "decoder_threads": args.decoder_threads,
                      "overlap": args.overlap, "warmups": args.warmups, "repeats": args.repeats},
        "image": safe_error(sample["image"]), "image_sha256": models._sha256(args.manifest.parent / sample["image"]),
        "boxes_supplied": "boxes" in sample,
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpu_count": os.cpu_count(), "python": platform.python_version(),
                    "package_version": package_version(), "runtime_versions": runtime_versions},
        "note": "One page on one machine; medians of the repeats after warmup. Not a basis for universal defaults.",
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, (json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        sys.exit(1)
