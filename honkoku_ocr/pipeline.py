"""画像 1 枚のレイアウト認識と行認識をまとめて行う。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from PIL import Image, ImageOps
from . import models
from .layout import LayoutDetector, Box
from .reading_order import order
from .recognizer import Recognizer, crop_with_margin, js_round
from .koji import raw_to_koji, raw_to_plain

MAX_IMAGE_DIM = 3500
MARGIN = 45

@dataclass
class LineResult:
    reading_order: int
    x: int
    y: int
    width: int
    height: int
    confidence: float
    raw: str
    koji: str
    plain: str

class OCR:
    def __init__(self, version: str = models.DEFAULT_VERSION, device: str = "cpu", quiet: bool = False):
        paths = models.ensure(version, quiet=quiet)
        self.detector = LayoutDetector(paths["layout"], device)
        self.recognizer = Recognizer(paths, version, device)

    @staticmethod
    def _load(image) -> tuple[Image.Image, float, float]:
        """EXIF の向きを反映し、長辺 3500 px に縮小。戻り値は (画像, x 倍率, y 倍率)。"""
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = min(1.0, MAX_IMAGE_DIM / max(img.size))
        if scale < 1:
            nw, nh = js_round(img.width * scale), js_round(img.height * scale)
            sx, sy = nw / img.width, nh / img.height
            img = img.resize((nw, nh), Image.LANCZOS)
            return img, sx, sy
        return img, 1.0, 1.0

    def layout(self, image) -> list[Box]:
        """行 bbox を元画像の座標で返す (読み順にソート済み)。"""
        img, sx, sy = self._load(image)
        boxes = self.detector.detect(img)
        ranks = order([(b.x, b.y, b.width, b.height) for b in boxes])
        ordered = [b for _, b in sorted(zip(ranks, boxes), key=lambda t: t[0])]
        return [_unscale(b, sx, sy) for b in ordered]

    def run(self, image, boxes: list[Box] | None = None) -> list[LineResult]:
        """boxes を与えなければレイアウト認識から行う。boxes は元画像の座標。"""
        img, sx, sy = self._load(image)
        if boxes is None:
            det = self.detector.detect(img)
            ranks = order([(b.x, b.y, b.width, b.height) for b in det])
            work = [(r, b) for r, b in sorted(zip(ranks, det), key=lambda t: t[0])]
        else:
            work = [(i, _scale(b, sx, sy)) for i, b in enumerate(boxes)]
        results = []
        for rank, b in work:
            raw = self.recognizer.recognize(crop_with_margin(img, b.x, b.y, b.width, b.height, MARGIN))
            o = _unscale(b, sx, sy)
            results.append(LineResult(rank + 1, o.x, o.y, o.width, o.height, b.confidence, raw, raw_to_koji(raw), raw_to_plain(raw)))
        return results

def _scale(b: Box, sx: float, sy: float) -> Box:
    x0, y0 = js_round(b.x * sx), js_round(b.y * sy)
    return Box(x0, y0, js_round((b.x + b.width) * sx) - x0, js_round((b.y + b.height) * sy) - y0, b.confidence)

def _unscale(b: Box, sx: float, sy: float) -> Box:
    x0, y0 = js_round(b.x / sx), js_round(b.y / sy)
    return Box(x0, y0, js_round((b.x + b.width) / sx) - x0, js_round((b.y + b.height) / sy) - y0, b.confidence)

def ocr_image(image, version: str = models.DEFAULT_VERSION, device: str = "cpu") -> list[dict]:
    return [asdict(r) for r in OCR(version, device).run(image)]
