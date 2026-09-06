"""Compare detected line boxes with the reference boxes of a manifest.

Detected and reference boxes are matched one-to-one, greedily by descending IoU,
and a pair counts when its IoU reaches the threshold. Reference boxes come from
a dataset whose line matching is incomplete, so an unmatched detection may be
real text without a reference line; the ratios are agreement, not accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from honkoku_ocr import OCR, models
from honkoku_ocr.output import atomic_write, package_version, safe_error


def iou(a, b) -> float:
    ix = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = ix * iy
    return inter / (a["width"] * a["height"] + b["width"] * b["height"] - inter) if inter else 0.0


def match(detected: list[dict], reference: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    pairs = sorted(((iou(d, r), i, j) for i, d in enumerate(detected) for j, r in enumerate(reference)), reverse=True)
    used_d, used_r, out = set(), set(), []
    for value, i, j in pairs:
        if value < threshold:
            break
        if i in used_d or j in used_r:
            continue
        used_d.add(i)
        used_r.add(j)
        out.append((i, j, value))
    return out


def evaluate(manifest: Path, *, model: str, device: str, threshold: float) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    ocr = OCR(model, device, quiet=True)
    pages, totals = [], {"reference": 0, "detected": 0, "matched": 0}
    for sample in data["samples"]:
        if "boxes" not in sample:
            continue
        source = manifest.parent / sample["image"]
        frame = sample.get("frame", 0)
        result = ocr.process(source, frame=frame, layout_only=True)
        detected = [{"x": line.x, "y": line.y, "width": line.width, "height": line.height,
                     "confidence": line.detection_confidence} for line in result.lines]
        matched = match(detected, sample["boxes"], threshold)
        page = {"id": sample["id"], "frame": frame, "source_sha256": models._sha256(source),
                "reference": len(sample["boxes"]), "detected": len(detected),
                "matched": len(matched), "mean_matched_iou": sum(v for *_, v in matched) / len(matched) if matched else None,
                "unmatched_detected": [detected[i] for i in range(len(detected)) if i not in {i for i, *_ in matched}],
                "unmatched_reference": [j for j in range(len(sample["boxes"])) if j not in {j for _, j, _ in matched}]}
        pages.append(page)
        for key in totals:
            totals[key] += page[key]
    return {"manifest": safe_error(manifest), "manifest_sha256": models._sha256(manifest), "model": model, "device": device,
            "iou_threshold": threshold, "settings": ocr.settings, "models": ocr.model_identity(),
            "package_version": package_version(), "pages": pages, "totals": totals}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(models.SPECS), default=models.DEFAULT_VERSION)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 < args.iou <= 1:
        parser.error("--iou must be in (0, 1]")
    report = evaluate(args.manifest, model=args.model, device=args.device, threshold=args.iou)
    for page in report["pages"]:
        print(f"{page['id']}: reference {page['reference']} detected {page['detected']} matched {page['matched']}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, (json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        sys.exit(1)
