"""Measure full-page OCR on an attributed manifest; optionally compute exact CER."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from honkoku_ocr import OCR, Box, models
from honkoku_ocr.output import atomic_write, package_version, safe_error


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        row = [i]
        for j, right in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = row
    return previous[-1]


def load_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if not samples or not manifest.get("attribution"):
        raise ValueError("manifest requires attribution and a nonempty samples list")
    ids = set()
    for sample in samples:
        if not isinstance(sample["id"], str) or sample["id"] in ids:
            raise ValueError("sample IDs must be unique strings")
        ids.add(sample["id"])
        image = (path.parent / sample["image"]).resolve()
        if not image.is_file():
            raise ValueError(f"sample {sample['id']}: image file is missing")
        if "reference" in sample and not isinstance(sample["reference"], str):
            raise ValueError("reference must be a string")
    return manifest


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if platform.system() == "Darwin" else value * 1024
    except ImportError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(models.SPECS), default=models.DEFAULT_VERSION)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--encoder-precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--decoder-threads", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--format", choices=["plain", "koji"], default="plain")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats positive")
    manifest = load_manifest(args.manifest)
    ocr = OCR(args.model, args.device, offline=args.offline, threads=args.threads,
              decoder_threads=args.decoder_threads, encoder_precision=args.encoder_precision, overlap=args.overlap)
    def run(sample):
        boxes = [Box(**box) for box in sample["boxes"]] if "boxes" in sample else None
        return ocr.process(args.manifest.parent / sample["image"], boxes, frame=sample.get("frame", 0))
    start = perf_counter()
    run(manifest["samples"][0])
    first_page_seconds = perf_counter() - start
    for _ in range(args.warmups):
        for sample in manifest["samples"]:
            run(sample)
    records = []
    total_errors = reference_characters = 0
    for sample in manifest["samples"]:
        runs = [run(sample) for _ in range(args.repeats)]
        predictions = ["\n".join(getattr(line, args.format) for line in result.lines) for result in runs]
        record = {"id": sample["id"],
                  "source_sha256": models._sha256(args.manifest.parent / sample["image"]),
                  "frame": sample.get("frame", 0), "boxes": sample.get("boxes"),
                  "median_seconds": statistics.median(result.timings["total"] for result in runs),
                  "timings_seconds": [result.timings for result in runs],
                  "predictions": predictions,
                  "repeat_outputs_identical": len(set(predictions)) == 1, "cer": None,
                  "lines": [asdict(line) for line in runs[-1].lines]}
        if "reference" in sample:
            reference = sample["reference"]
            errors = [edit_distance(reference, prediction) for prediction in predictions]
            record.update(reference=reference, character_errors=errors,
                          cer=[error / len(reference) for error in errors] if reference else None)
            total_errors += sum(errors)
            reference_characters += len(reference) * len(predictions)
        records.append(record)
    identity = ocr.model_identity()
    report = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "attribution": manifest["attribution"], "package_version": package_version(),
              "python": platform.python_version(), "platform": platform.system(),
              "onnxruntime": __import__("onnxruntime").__version__,
              "model": args.model, "settings": ocr.settings,
              "models": identity,
              "format": args.format, "normalization": "none (Unicode code points; newlines count)",
              "warmups": args.warmups, "repeats": args.repeats,
              "first_page_seconds_including_setup": first_page_seconds,
              "process_peak_rss_bytes": peak_rss_bytes(),
              "reference_characters_across_repeats": reference_characters,
              "character_errors_across_repeats": total_errors,
              "aggregate_cer": total_errors / reference_characters if reference_characters else None,
              "samples": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"samples": len(records), "aggregate_cer": report["aggregate_cer"],
                      "first_page_seconds": first_page_seconds}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        sys.exit(1)
