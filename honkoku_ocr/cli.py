"""Batch OCR with verified resume records and atomic per-page outputs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import models
from .layout import Box
from .pipeline import OCR, SCHEMA_VERSION

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def package_version() -> str:
    try:
        return version("honkoku-ocr-py")
    except PackageNotFoundError:
        return "unknown"


def safe_error(error) -> str:
    return re.sub(r"/home/[^/\s'\"]+", "~", str(error))


def _unique_names(files: list[Path]) -> dict[Path, str]:
    # The hash makes names independent of batch order or which files are selected.
    return {p: f"{p.stem}__{hashlib.sha256(os.fsencode(p.resolve())).hexdigest()[:16]}"
            for p in files}


def atomic_write(path: Path, data: bytes):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resume_matches(path: Path, fingerprint: dict, expected: set[str]) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return False
        if record["fingerprint"] != fingerprint or record["schema_version"] != SCHEMA_VERSION:
            return False
        artifacts = record["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != expected:
            return False
        return all(models._sha256(path.parent / name) == digest for name, digest in artifacts.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _boxes(path: Path) -> list[Box]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["lines"] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("box input must be a list or a page JSON with lines")
    return [Box(row["x"], row["y"], row["width"], row["height"],
                row.get("detection_confidence", row.get("confidence", 1.0))) for row in rows]


def _preview(source: Path, frame: int, lines) -> bytes:
    with Image.open(source) as opened:
        opened.seek(frame)
        image = ImageOps.exif_transpose(opened).convert("RGB")
    try:
        width, height = image.size
        image.thumbnail((1600, 1600))
        sx, sy = image.width / width, image.height / height
        draw = ImageDraw.Draw(image)
        for line in lines:
            x, y = max(0, line.x) * sx, max(0, line.y) * sy
            right = min(width, line.x + line.width) * sx
            bottom = min(height, line.y + line.height) * sy
            draw.rectangle((x, y, right, bottom),
                           outline="red", width=2)
            draw.text((x + 2, y + 2), str(line.reading_order), fill="red", stroke_width=1,
                      stroke_fill="white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        image.close()


def _model_identity(ocr: OCR, roles: list[str]) -> dict:
    return ocr.model_identity(roles)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="honkoku-ocr", description="Transcribe historical Japanese pages")
    ap.add_argument("inputs", nargs="*", help="image files or directories")
    ap.add_argument("--version", action="version", version=f"%(prog)s {package_version()}")
    ap.add_argument("--model", default=models.DEFAULT_VERSION, choices=sorted(models.SPECS))
    ap.add_argument("-o", "--output", type=Path, default=Path("ocr-out"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="layout and encoder device; decoder uses CPU")
    ap.add_argument("--encoder-precision", choices=["auto", "fp16", "fp32"], default="auto")
    ap.add_argument("--threads", type=int, default=0, help="layout/encoder threads; 0 uses runtime defaults")
    ap.add_argument("--decoder-threads", type=int, default=0)
    ap.add_argument("--plain", action="store_true", help="write plain text instead of Koji")
    ap.add_argument("--download", action="store_true", help="download and validate model files, then exit")
    ap.add_argument("--verify-cache", action="store_true", help="verify cached model hashes without network access")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip matching, complete outputs")
    ap.add_argument("--layout-only", action="store_true")
    ap.add_argument("--boxes", type=Path, help="box list or page JSON for a single image frame")
    ap.add_argument("--preview", action="store_true", help="write a numbered bbox overlay PNG")
    ap.add_argument("--frame", type=int, help="zero-based frame; default processes every frame")
    ap.add_argument("--max-dimension", type=int, default=3500)
    ap.add_argument("--margin", type=int, default=45)
    ap.add_argument("--confidence-threshold", type=float, default=0.3)
    ap.add_argument("--ios-threshold", type=float, default=0.8)
    args = ap.parse_args(argv)
    if args.boxes and (args.download or args.verify_cache):
        ap.error("--boxes cannot be combined with --download or --verify-cache")
    if args.frame is not None and args.frame < 0:
        ap.error("--frame must be nonnegative")
    try:
        ocr = OCR(args.model, args.device, offline=args.offline, threads=args.threads,
                  decoder_threads=args.decoder_threads, encoder_precision=args.encoder_precision,
                  max_dimension=args.max_dimension, margin=args.margin,
                  conf_threshold=args.confidence_threshold, ios_threshold=args.ios_threshold)
    except ValueError as error:
        ap.error(str(error))
    roles = ([] if args.layout_only and args.boxes else
             ["layout"] if args.layout_only else
             ["encoder", "prefill", "step"] if args.boxes else
             ["layout", "encoder", "prefill", "step"])
    if args.download or args.verify_cache:
        try:
            for role, path in models.ensure(args.model, roles=roles, digest=True,
                                            offline=args.offline or args.verify_cache).items():
                print(f"{role}: {path.name} SHA-256 {models._sha256(path)}")
            return 0
        except Exception as error:
            print(safe_error(error), file=sys.stderr)
            return 1
    if not args.inputs:
        ap.error("specify an image or directory")
    failures = 0
    files = []
    for item in args.inputs:
        path = Path(item)
        try:
            candidates = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTS) if path.is_dir() else [path]
            files.extend(candidates)
        except OSError as error:
            failures += 1
            print(f"{path.name}: {safe_error(error)}", file=sys.stderr)
    files = list(dict.fromkeys(p.resolve() for p in files))
    jobs = []
    for path in files:
        try:
            with Image.open(path) as image:
                count = getattr(image, "n_frames", 1)
                frames = range(count) if args.frame is None else [args.frame]
                for frame in frames:
                    if frame >= count:
                        raise ValueError(f"frame {frame} out of range (image has {count} frames)")
                    jobs.append((path, frame, count))
        except Exception as error:
            failures += 1
            print(f"{path.name}: {safe_error(error)}", file=sys.stderr)
    if not jobs:
        print(f"0 completed, 0 skipped, {failures} failed; no readable images", file=sys.stderr)
        return 1
    if args.boxes and len(jobs) != 1:
        ap.error("--boxes requires one image frame; select --frame for a multipage image")
    try:
        supplied = _boxes(args.boxes) if args.boxes else None
        identity = _model_identity(ocr, roles)
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        return 1
    try:
        args.output.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(safe_error(error), file=sys.stderr)
        return 1
    names = _unique_names(files)
    source_hashes = {}
    # Hash installed source too: an editable checkout may change without a version bump.
    code_hash = _digest(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))))
    runtime_versions = {}
    for package in ("numpy", "pillow", "onnxruntime", "onnxruntime-gpu", "onnx"):
        try:
            runtime_versions[package] = version(package)
        except PackageNotFoundError:
            pass
    completed = skipped = 0
    for index, (path, frame, count) in enumerate(jobs, 1):
        label = f"{path.name} [{frame + 1}/{count}]"
        stem = names[path] + (f"__p{frame + 1:04d}" if count > 1 else "")
        output = args.output / f"{stem}.json"
        expected = {f"{stem}.txt"} | ({f"{stem}.preview.png"} if args.preview else set())
        try:
            if path not in source_hashes:
                source_hashes[path] = models._sha256(path)
            fingerprint = {"source_sha256": source_hashes[path], "frame": frame,
                           "models": identity, "settings": ocr.settings, "model": args.model,
                           "plain": args.plain, "layout_only": args.layout_only,
                           "boxes": [asdict(b) for b in supplied] if supplied is not None else None,
                           "package_version": package_version(), "code_sha256": code_hash,
                           "runtime_versions": runtime_versions}
            if args.resume and _resume_matches(output, fingerprint, expected):
                skipped += 1
                print(f"[{index}/{len(jobs)}] {label}: skipped", file=sys.stderr)
                continue
            result = ocr.process(path, supplied, frame=frame, layout_only=args.layout_only)
            text = "\n".join(line.plain if args.plain else line.koji for line in result.lines) + "\n"
            artifacts = {f"{stem}.txt": text.encode("utf-8")}
            if args.preview:
                artifacts[f"{stem}.preview.png"] = _preview(path, frame, result.lines)
            record = {**asdict(result), "image": path.name, "fingerprint": fingerprint,
                      "artifacts": {name: _digest(data) for name, data in artifacts.items()}}
            # The JSON is the completion record; publish it after every artifact.
            for name, data in artifacts.items():
                atomic_write(args.output / name, data)
            atomic_write(output, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
            completed += 1
            print(f"[{index}/{len(jobs)}] {label}: {len(result.lines)} lines, {result.timings['total']:.2f}s", file=sys.stderr)
        except Exception as error:
            failures += 1
            print(f"[{index}/{len(jobs)}] {label}: {safe_error(error)}", file=sys.stderr)
    print(f"{completed} completed, {skipped} skipped, {failures} failed", file=sys.stderr)
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
