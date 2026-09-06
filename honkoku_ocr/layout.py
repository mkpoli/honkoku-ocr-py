"""行検出 (RTMDet-s、NDL古典籍OCR-Lite 附属モデル)。

入力 1024×1024 レターボックス (pad 114)、BGR、mean=[103.53,116.28,123.675] std=[57.375,57.12,58.395]。
出力 dets [1,N,5] (x1,y1,x2,y2,score; NMS 済)。入れ子 box (IoS ≥ 0.8) は大きい方を残す。
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .recognizer import js_round
from .runtime import session

SIZE = 1024
PAD = 114
MEAN_BGR = np.array([103.53, 116.28, 123.675], np.float32)
STD_BGR = np.array([57.375, 57.12, 58.395], np.float32)

@dataclass
class Box:
    x: int
    y: int
    width: int
    height: int
    confidence: float

class LayoutDetector:
    def __init__(self, model_path, device: str = "cpu", *, threads: int = 0):
        self.sess = session(model_path, device, threads=threads)
        self.input = self.sess.get_inputs()[0].name

    def detect(self, image: Image.Image, conf_threshold: float = 0.3, ios_threshold: float = 0.8) -> list[Box]:
        tensor, scale, pad_x, pad_y = self._letterbox(image)
        dets = self.sess.run(["dets"], {self.input: tensor})[0][0]
        boxes: list[Box] = []
        for x1, y1, x2, y2, score in dets:
            if score < conf_threshold:
                continue
            x = max(0, js_round((x1 - pad_x) / scale)); y = max(0, js_round((y1 - pad_y) / scale))
            w = min(image.width, js_round((x2 - pad_x) / scale)) - x
            h = min(image.height, js_round((y2 - pad_y) / scale)) - y
            if w < 6 or h < 6:
                continue
            boxes.append(Box(x, y, w, h, float(score)))
        return _remove_nested(boxes, ios_threshold)

    @staticmethod
    def _letterbox(image: Image.Image):
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        try:
            scale = min(SIZE / rgb.width, SIZE / rgb.height)
            nw, nh = max(1, js_round(rgb.width * scale)), max(1, js_round(rgb.height * scale))
            pad_x, pad_y = (SIZE - nw) // 2, (SIZE - nh) // 2
            with closing(Image.new("RGB", (SIZE, SIZE), (PAD, PAD, PAD))) as canvas, closing(_point_sampled_resize(rgb, nw, nh)) as resized:
                canvas.paste(resized, (pad_x, pad_y))
                a = np.asarray(canvas, np.float32)[..., ::-1]  # BGR
        finally:
            if rgb is not image:
                rgb.close()
        a = (a - MEAN_BGR) / STD_BGR
        return np.ascontiguousarray(a.transpose(2, 0, 1)[None]), scale, pad_x, pad_y

def _point_sampled_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """平均化しない点標本の双一次補間で縮小する。

    Pillowのresizeは縮小時に元画素を平均する（アンチエイリアス）。ブラウザ版のcanvas drawImageは
    既定では平均化せず、Chromium 153のヘッドレス環境で同じ見開きを比べると、平均化する縮小では
    RTMDetのスコアが行あたり平均0.09ずれて0.3の閾値付近の行の有無が変わり、点標本の双一次補間では
    ずれが平均0.01に収まり、閾値を越える候補の数も51と52でほぼ揃った。canvasの標本化はブラウザや設定で変わりうるので
    画素の完全一致は保証しない。
    """
    sx, sy = image.width / width, image.height / height
    return image.transform((width, height), Image.AFFINE, (sx, 0, 0, 0, sy, 0), resample=Image.BILINEAR)


def _remove_nested(boxes: list[Box], ios_threshold: float) -> list[Box]:
    kept: list[Box] = []
    for c in sorted(boxes, key=lambda b: b.width * b.height, reverse=True):
        area = c.width * c.height
        suppressed = False
        for k in kept:
            ix1, iy1 = max(c.x, k.x), max(c.y, k.y)
            ix2, iy2 = min(c.x + c.width, k.x + k.width), min(c.y + c.height, k.y + k.height)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter and inter / area >= ios_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(c)
    return kept
