"""IIIF Presentation manifest（v2, v3）からページ画像を取り出す。

`canvases(manifest)`がキャンバスごとの画像URLを列挙し、`download(canvas, directory, client)`が
フルサイズの画像を取得して置く。取得済みのファイルはそのまま使う。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .output import atomic_write

RETRY_DELAYS = (1.0, 3.0)
_sleep = time.sleep


@dataclass(frozen=True)
class Canvas:
    index: int              # 1始まり
    label: str
    image_url: str
    width: int | None = None
    height: int | None = None


def _label(value) -> str:
    if isinstance(value, dict):                      # v3: {"none": ["1"]} / {"ja": ["一"]}
        for texts in value.values():
            if texts:
                return str(texts[0])
        return ""
    if isinstance(value, list):
        return _label(value[0]) if value else ""
    return "" if value is None else str(value)


def _service_id(service) -> str | None:
    if isinstance(service, list):
        service = service[0] if service else None
    if isinstance(service, dict):
        return service.get("@id") or service.get("id")
    return None


def canvases(manifest: dict) -> list[Canvas]:
    """マニフェストの各キャンバスについて、フルサイズ画像のURLを返す。"""
    out: list[Canvas] = []
    if "sequences" in manifest:                      # Presentation API 2
        for i, canvas in enumerate(manifest["sequences"][0].get("canvases", []), 1):
            images = canvas.get("images") or []
            resource = (images[0].get("resource") or {}) if images else {}
            service = _service_id(resource.get("service"))
            url = f"{service}/full/full/0/default.jpg" if service else resource.get("@id") or resource.get("id")
            if url:
                out.append(Canvas(i, _label(canvas.get("label")), url, canvas.get("width"), canvas.get("height")))
        return out
    for i, canvas in enumerate(manifest.get("items", []), 1):       # Presentation API 3
        url = None
        for page in canvas.get("items", []):
            for annotation in page.get("items", []):
                body = annotation.get("body")
                if isinstance(body, list):
                    body = body[0] if body else None
                if isinstance(body, dict) and body.get("type") == "Image":
                    service = _service_id(body.get("service"))
                    url = f"{service}/full/max/0/default.jpg" if service else body.get("id")
                    break
            if url:
                break
        if url:
            out.append(Canvas(i, _label(canvas.get("label")), url, canvas.get("width"), canvas.get("height")))
    return out


def new_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=120, headers={"User-Agent": "honkoku-ocr"})


def load_manifest(source: str, client: httpx.Client | None = None) -> dict:
    """URLかローカルファイルからマニフェストを読む。"""
    if re.match(r"^https?://", source):
        if client is None:
            with new_client() as http:
                response = http.get(source)
                response.raise_for_status()
                return response.json()
        response = client.get(source)
        response.raise_for_status()
        return response.json()
    return json.loads(Path(source).read_text(encoding="utf-8"))


def manifest_directory(source: str, root: Path) -> Path:
    """マニフェストごとの画像置き場。URLのSHA-256の先頭16桁で分ける。"""
    return root / hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _safe_stem(label: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z一-鿿぀-ヿ_-]+", "_", label).strip("_")
    return stem[:40]


def image_path(canvas: Canvas, directory: Path) -> Path:
    stem = f"{canvas.index:04d}"
    label = _safe_stem(canvas.label)
    if label and not (label.isdigit() and int(label) == canvas.index):
        stem += "-" + label
    return directory / f"{stem}.jpg"


def download(canvas: Canvas, directory: Path, client: httpx.Client, *, quiet: bool = True) -> Path:
    """キャンバスの画像を取得して置く。既にあれば取得しない。一時的な失敗は取り直す。"""
    target = image_path(canvas, directory)
    if target.exists() and target.stat().st_size > 0:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = client.get(canvas.image_url)
            response.raise_for_status()
            break
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            if not (status is None or status == 429 or status >= 500) or attempt == len(RETRY_DELAYS):
                raise
            _sleep(RETRY_DELAYS[attempt])
    if not response.content:
        raise RuntimeError(f"canvas {canvas.index}: empty image response")
    atomic_write(target, response.content)
    return target


def parse_pages(spec: str | None, count: int) -> list[int]:
    """'3', '3-5', '1,4-6' のような1始まりの指定をキャンバス番号の一覧にする。省略時は全て。"""
    if not spec:
        return list(range(1, count + 1))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        start, end = int(first), int(last or first)
        if start < 1 or end > count or start > end:
            raise ValueError(f"pages {part!r} outside 1-{count}")
        pages.extend(range(start, end + 1))
    return sorted(set(pages))
