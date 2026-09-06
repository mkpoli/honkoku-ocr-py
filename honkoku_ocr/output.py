"""Atomic page artifacts and shared output utilities."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import ImageDraw

from .sources import load_frame

if TYPE_CHECKING:
    from .pipeline import PageResult

def package_version() -> str:
    try:
        return version("honkoku-ocr-py")
    except PackageNotFoundError:
        return "unknown"


def safe_error(error) -> str:
    return re.sub(r"/home/[^/\s'\"]+", "~", str(error))


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


def preview(source: Path, frame: int, lines) -> bytes:
    image = load_frame(source, frame, max_dimension=None)
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


def write_page(path: Path, result: PageResult, *, image: str, fingerprint: dict,
               plain: bool = False, preview_png: bytes | None = None) -> None:
    """Publish text/preview first, then JSON as the completion record.

    The caller creates the output directory and supplies provenance. A failed
    write leaves no valid completion record for changed artifact bytes.
    """
    if path.suffix != ".json":
        raise ValueError("page record path must end in .json")
    text = "\n".join(line.plain if plain else line.koji for line in result.lines) + "\n"
    artifacts = {path.with_suffix(".txt").name: text.encode("utf-8")}
    if preview_png is not None:
        artifacts[path.with_suffix(".preview.png").name] = preview_png
    record = {**asdict(result), "image": image, "fingerprint": fingerprint,
              "artifacts": {name: _digest(data) for name, data in artifacts.items()}}
    for name, data in artifacts.items():
        atomic_write(path.parent / name, data)
    atomic_write(path, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
