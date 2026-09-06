"""Compare whole-page OCR output with page transcriptions in みんなで翻刻 notation.

Every page runs through the complete pipeline (line detection, reading order,
recognition) and the Koji text of the page is compared with the reference page:

- page CER on the raw text (newlines and spaces count) and on the squeezed text
  (editorial notes in 【】 removed, all whitespace removed), which is sensitive to
  reading order but not to where lines are split;
- bag of characters: how many reference characters have no counterpart among
  the predicted characters (missed) and how many predicted characters have none
  in the reference (extra), regardless of order and segmentation;
- normalised squeezed CER: as squeezed, after folding katakana to hiragana,
  Unicode compatibility forms, and the variant-kanji pairs in VARIANTS, so that
  transcription conventions stop counting and what remains is closer to
  recognition proper;
- line alignment: each reference line is paired greedily with the unused
  predicted line of lowest normalised edit distance, giving a per-line CER that
  ignores reading order, the number of unpaired reference lines (missed) and
  unpaired predicted lines (extra), and the share of paired lines that keep their
  relative order. A transcription line that the detector splits into two boxes
  counts against this metric even when every character is right.

References are volunteer transcriptions; the project reports about 1.5 errors
per 100 characters, so the numbers are agreement with those transcriptions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import unicodedata
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from benchmarks.ocr import edit_distance
from honkoku_ocr import OCR, models
from honkoku_ocr.output import atomic_write, package_version, safe_error

_WHITESPACE = re.compile(r"\s+")
_NOTE = re.compile(r"【[^】]*】")
# Variant pairs seen between みんなで翻刻 transcriptions and model output on the
# honkoku corpus, folded to the first form. 変体仮名 written with their 字母 (多, 連,
# 里, 与) are not folded, since the same characters occur as kanji.
VARIANTS = {"顚": "顛", "禱": "祷", "略": "畧", "幷": "并", "檜": "桧", "澤": "沢", "萬": "万", "國": "国",
            "舩": "船", "○": "〇", "ヽ": "ゝ", "ヾ": "ゞ", "凶": "㐫", "鶏": "雞", "壽": "寿", "與": "与",
            "體": "体", "會": "会", "來": "来", "當": "当", "應": "応", "圖": "図", "廣": "広", "數": "数"}


def normalize(text: str) -> str:
    """Fold script and variant differences that are conventions, not readings."""
    out = []
    for ch in text:
        folded = unicodedata.normalize("NFKC", ch)
        if len(folded) == 1:          # fullwidth digits and letters; ヿ and other ligatures are kept
            ch = folded
        if "ァ" <= ch <= "ヶ":
            ch = chr(ord(ch) - 0x60)
        out.append(VARIANTS.get(ch, ch))
    return "".join(out)


def squeeze(text: str) -> str:
    return _WHITESPACE.sub("", _NOTE.sub("", text))


def bag(reference: str, hypothesis: str) -> dict:
    ref, hyp = Counter(reference), Counter(hypothesis)
    missed = sum(max(0, n - hyp[c]) for c, n in ref.items())
    extra = sum(max(0, n - ref[c]) for c, n in hyp.items())
    return {"missed": missed, "extra": extra,
            "missed_share": missed / len(reference) if reference else None,
            "extra_share": extra / len(reference) if reference else None}


def cer(reference: str, hypothesis: str) -> float | None:
    return edit_distance(reference, hypothesis) / len(reference) if reference else None


def align_lines(reference: list[str], predicted: list[str]) -> dict:
    """Pair reference and predicted lines greedily by lowest normalised distance."""
    scored = []
    for i, ref in enumerate(reference):
        for j, hyp in enumerate(predicted):
            scored.append((edit_distance(ref, hyp) / max(len(ref), len(hyp), 1), i, j))
    scored.sort()
    used_ref, used_hyp, pairs = set(), set(), []
    for score, i, j in scored:
        if i in used_ref or j in used_hyp:
            continue
        used_ref.add(i)
        used_hyp.add(j)
        pairs.append((i, j, score))
    pairs.sort()
    ordered = sum(1 for (a, b, _), (c, d, _) in zip(pairs, pairs[1:], strict=False) if b < d)
    matched_chars = sum(len(reference[i]) for i, _, _ in pairs)
    matched_errors = sum(edit_distance(reference[i], predicted[j]) for i, j, _ in pairs)
    return {"reference_lines": len(reference), "predicted_lines": len(predicted), "paired": len(pairs),
            "missed": len(reference) - len(pairs), "extra": len(predicted) - len(pairs),
            "paired_cer": matched_errors / matched_chars if matched_chars else None,
            "paired_reference_characters": matched_chars, "paired_errors": matched_errors,
            "in_order_share": ordered / (len(pairs) - 1) if len(pairs) > 1 else None,
            "pairs": [{"reference": i, "predicted": j, "distance": round(s, 3)} for i, j, s in pairs]}


def evaluate_page(reference: str, predicted_lines: list[str]) -> dict:
    reference_lines = [line for line in _NOTE.sub("", reference).splitlines() if line.strip()]
    predicted = "\n".join(predicted_lines)
    ref_sq, hyp_sq = squeeze(reference), squeeze(predicted)
    ref_nm, hyp_nm = normalize(ref_sq), normalize(hyp_sq)
    return {"reference_characters": len(reference), "raw_errors": edit_distance(reference, predicted),
            "raw_cer": cer(reference, predicted),
            "squeezed_reference_characters": len(ref_sq), "squeezed_errors": edit_distance(ref_sq, hyp_sq),
            "squeezed_cer": cer(ref_sq, hyp_sq), "bag": bag(ref_sq, hyp_sq),
            "normalized_reference_characters": len(ref_nm), "normalized_errors": edit_distance(ref_nm, hyp_nm),
            "normalized_cer": cer(ref_nm, hyp_nm), "normalized_bag": bag(ref_nm, hyp_nm),
            "lines": align_lines([squeeze(line) for line in reference_lines], [squeeze(line) for line in predicted_lines])}


def run(manifest: Path, *, model: str, device: str, precision: str, threads: int, decoder_threads: int,
        overlap: bool, offline: bool, limit: int | None, log=None) -> dict:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    ocr = OCR(model, device, quiet=True, threads=threads, decoder_threads=decoder_threads,
              encoder_precision=precision, overlap=overlap, offline=offline)
    pages = []
    for sample in data["samples"][:limit]:
        result = ocr.process(manifest.parent / sample["image"], frame=sample.get("frame", 0))
        koji = [line.koji for line in result.lines]
        page = {"id": sample["id"], "frame": sample.get("frame", 0), "image_sha256": models._sha256(manifest.parent / sample["image"]),
                "seconds": result.timings["total"], "warnings": result.warnings,
                "predicted": koji, "reference": sample["reference"], **evaluate_page(sample["reference"], koji)}
        pages.append(page)
        if log:
            print(f"{page['id']}: {page['lines']['predicted_lines']}/{page['lines']['reference_lines']} lines, "
                  f"CER raw {page['raw_cer']:.3f} squeezed {page['squeezed_cer']:.3f} paired {page['lines']['paired_cer'] or 0:.3f} "
                  f"{page['seconds']:.1f}s", file=log)
    return finish(pages, manifest, data, ocr.settings, ocr.model_identity())


def totals_of(pages: list[dict]) -> dict:
    totals = {"pages": len(pages),
              "reference_characters": sum(p["reference_characters"] for p in pages),
              "raw_errors": sum(p["raw_errors"] for p in pages),
              "squeezed_reference_characters": sum(p["squeezed_reference_characters"] for p in pages),
              "squeezed_errors": sum(p["squeezed_errors"] for p in pages),
              "bag_missed": sum(p["bag"]["missed"] for p in pages), "bag_extra": sum(p["bag"]["extra"] for p in pages),
              "normalized_reference_characters": sum(p["normalized_reference_characters"] for p in pages),
              "normalized_errors": sum(p["normalized_errors"] for p in pages),
              "normalized_bag_missed": sum(p["normalized_bag"]["missed"] for p in pages),
              "normalized_bag_extra": sum(p["normalized_bag"]["extra"] for p in pages),
              "reference_lines": sum(p["lines"]["reference_lines"] for p in pages),
              "predicted_lines": sum(p["lines"]["predicted_lines"] for p in pages),
              "paired": sum(p["lines"]["paired"] for p in pages),
              "paired_reference_characters": sum(p["lines"]["paired_reference_characters"] for p in pages),
              "paired_errors": sum(p["lines"]["paired_errors"] for p in pages),
              "seconds": sum(p["seconds"] for p in pages)}
    totals["raw_cer"] = totals["raw_errors"] / totals["reference_characters"] if totals["reference_characters"] else None
    totals["squeezed_cer"] = totals["squeezed_errors"] / totals["squeezed_reference_characters"] if totals["squeezed_reference_characters"] else None
    totals["paired_cer"] = totals["paired_errors"] / totals["paired_reference_characters"] if totals["paired_reference_characters"] else None
    n = totals["squeezed_reference_characters"]
    totals["bag_missed_share"] = totals["bag_missed"] / n if n else None
    totals["bag_extra_share"] = totals["bag_extra"] / n if n else None
    m = totals["normalized_reference_characters"]
    totals["normalized_cer"] = totals["normalized_errors"] / m if m else None
    totals["normalized_bag_missed_share"] = totals["normalized_bag_missed"] / m if m else None
    totals["normalized_bag_extra_share"] = totals["normalized_bag_extra"] / m if m else None
    return totals


def finish(pages: list[dict], manifest: Path, data: dict, settings: dict, identity: dict) -> dict:
    totals = totals_of(pages)
    runtime_versions = {}
    for package in ("numpy", "pillow", "onnxruntime", "onnxruntime-gpu", "onnx"):
        try:
            runtime_versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return {"manifest": safe_error(manifest), "manifest_sha256": models._sha256(manifest),
            "attribution": data.get("attribution"), "settings": settings, "models": identity,
            "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
                        "package_version": package_version(), "runtime_versions": runtime_versions},
            "metric": "Character error rate = Levenshtein distance / reference length, Unicode code points, Koji notation on both sides.",
            "totals": totals, "pages": pages}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(models.SPECS), default=models.DEFAULT_VERSION)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--encoder-precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--decoder-threads", type=int, default=0)
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, help="evaluate only the first N samples")
    parser.add_argument("--rescore", type=Path, help="recompute the metrics from the predictions stored in an earlier report")
    args = parser.parse_args(argv)
    if args.rescore:
        old = json.loads(args.rescore.read_text(encoding="utf-8"))
        pages = [{**{k: p[k] for k in ("id", "frame", "image_sha256", "seconds", "warnings", "predicted", "reference")},
                  **evaluate_page(p["reference"], p["predicted"])} for p in old["pages"]]
        report = finish(pages, args.manifest, json.loads(args.manifest.read_text(encoding="utf-8")), old["settings"], old["models"])
        report["rescored_from"] = safe_error(args.rescore)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(args.output, (json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
        return 0
    if args.threads < 0 or args.decoder_threads < 0 or (args.limit is not None and args.limit < 1):
        parser.error("thread counts must be nonnegative and --limit positive")
    report = run(args.manifest, model=args.model, device=args.device, precision=args.encoder_precision,
                 threads=args.threads, decoder_threads=args.decoder_threads, overlap=args.overlap,
                 offline=args.offline, limit=args.limit, log=sys.stderr)
    t = report["totals"]
    print(f"{t['pages']} pages: CER raw {t['raw_cer']:.4f}, squeezed {t['squeezed_cer']:.4f}, normalized {t['normalized_cer']:.4f}, paired lines {t['paired_cer']:.4f}, "
          f"bag missed {t['bag_missed_share']:.4f} extra {t['bag_extra_share']:.4f}; "
          f"lines {t['predicted_lines']}/{t['reference_lines']} predicted/reference, paired {t['paired']}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, (json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        sys.exit(1)
