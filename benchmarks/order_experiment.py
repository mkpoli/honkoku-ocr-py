"""Re-score a page evaluation under alternative reading orders without re-recognising.

Takes the report written by page_eval.py, re-runs line detection only (to get the
boxes in the order the report's predictions were made), re-orders the stored
predictions with a plain column sort (right to left by box centre, top to bottom
within a column, columns split where centres differ by more than a fraction of
the box width) and reports the normalised squeezed CER for the original XY-Cut
order and for each tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.page_eval import cer, normalize, squeeze
from honkoku_ocr import OCR, models
from honkoku_ocr.output import atomic_write, safe_error


def column_sort(boxes, tolerance: float) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: -(boxes[i].x + boxes[i].width / 2))
    columns, current, centre = [], [], None
    for i in order:
        cx = boxes[i].x + boxes[i].width / 2
        if current and abs(cx - centre) > tolerance * boxes[i].width:
            columns.append(current)
            current = []
        if not current:
            centre = cx
        current.append(i)
    columns.append(current)
    return [i for column in columns for i in sorted(column, key=lambda i: boxes[i].y)]


def evaluate(report: dict, manifest: Path, tolerances: list[float], ocr) -> dict:
    samples = {s["id"]: s for s in json.loads(manifest.read_text(encoding="utf-8"))["samples"]}
    pages, errors, characters = [], {"xycut": 0, **{f"column_{t}": 0 for t in tolerances}}, 0
    for page in report["pages"]:
        sample = samples[page["id"]]
        boxes = ocr.layout(manifest.parent / sample["image"])
        if len(boxes) != len(page["predicted"]):
            raise RuntimeError(f"{page['id']}: {len(boxes)} boxes now, {len(page['predicted'])} predictions in the report")
        reference = normalize(squeeze(page["reference"]))
        characters += len(reference)
        row = {"id": page["id"]}
        orders = {"xycut": list(range(len(boxes))), **{f"column_{t}": column_sort(boxes, t) for t in tolerances}}
        for name, order in orders.items():
            hypothesis = normalize(squeeze("\n".join(page["predicted"][i] for i in order)))
            value = cer(reference, hypothesis)
            row[name] = value
            errors[name] += round(value * len(reference)) if value is not None else 0
        pages.append(row)
    return {"manifest": safe_error(manifest), "manifest_sha256": models._sha256(manifest), "tolerances": tolerances,
            "metric": "normalised squeezed CER (page_eval) after re-ordering the stored predictions",
            "totals": {name: e / characters for name, e in errors.items()} | {"characters": characters}, "pages": pages}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path, help="output of page_eval.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    result = evaluate(json.loads(args.report.read_text(encoding="utf-8")), args.manifest, args.tolerance, OCR(models.DEFAULT_VERSION, args.device, quiet=True))
    for name, value in result["totals"].items():
        if name != "characters":
            print(f"{name}: {value:.4f}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, (json.dumps(result, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        sys.exit(1)
