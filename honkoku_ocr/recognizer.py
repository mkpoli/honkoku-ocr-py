"""行認識 (ConvNeXt V2 encoder + RoBERTa decoder、KV キャッシュ付き greedy)。

前処理 (text-recognizer.ts の to_pixel と同じ):
  1. 投影プロファイル法で行の傾きを推定し、2 度以上なら補正
  2. 縦長なら時計回りに 90 度回転
  3. 高さ 256 にアスペクト比保持で縮小 (幅は最大 2048)
  4. 右側を白で埋めて 256×2048
  5. /255 の後 ImageNet 平均・分散で正規化、NCHW
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from PIL import Image

from . import models
from .runtime import session

CLS, SEP = 2, 3
STRUCT = {0, 1, 2, 3, 4}
REPEAT_WINDOW = 12
SKEW_MAX, SKEW_COARSE, SKEW_MIN_APPLY, SKEW_DOWNSCALE = 12, 3, 2, 120
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
WHITE = (255, 255, 255)

def js_round(v: float) -> int:
    """JavaScript の Math.round (0.5 は切り上げ)。Python の round は偶数丸めで半画素ずれる。"""
    return math.floor(v + 0.5)

def crop_with_margin(page: Image.Image, x: float, y: float, w: float, h: float, margin: int = 45) -> Image.Image:
    """行 bbox に上・下・右の余白を付けて切り出す。左は付けない（縦書きの次行が混入するため）。画像外は白。"""
    return crop_white(page, x, y - margin, w + margin, h + 2 * margin)

def crop_white(page: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    bx, by = js_round(x), js_round(y)
    w, h = max(1, js_round(w)), max(1, js_round(h))
    out = Image.new("RGB", (w, h), WHITE)
    sx, sy = max(0, bx), max(0, by)
    ex, ey = min(page.width, bx + w), min(page.height, by + h)
    if ex > sx and ey > sy:
        out.paste(page.crop((sx, sy, ex, ey)), (sx - bx, sy - by))
    return out

def _luminance(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

def estimate_skew(crop: Image.Image) -> int:
    scale = min(1.0, SKEW_DOWNSCALE / max(crop.width, crop.height))
    sw, sh = max(8, js_round(crop.width * scale)), max(8, js_round(crop.height * scale))
    vertical = crop.height >= crop.width
    small = crop.convert("RGB").resize((sw, sh), Image.BILINEAR)
    thr = _luminance(small).mean() * 0.9
    diag = math.ceil(math.hypot(sw, sh)) + 2
    def score(deg: int) -> float:
        canvas = Image.new("RGB", (diag, diag), WHITE)
        rot = small.rotate(-deg, resample=Image.BILINEAR, expand=True, fillcolor=WHITE)
        canvas.paste(rot, ((diag - rot.width) // 2, (diag - rot.height) // 2))
        acc = (_luminance(canvas) < thr).sum(axis=0 if vertical else 1).astype(np.float64)
        return float((acc * acc).sum())
    best, best_s = 0, -1.0
    for d in range(-SKEW_MAX, SKEW_MAX + 1, SKEW_COARSE):
        s = score(d)
        if s > best_s:
            best_s, best = s, d
    for d in range(best - SKEW_COARSE + 1, best + SKEW_COARSE):
        if d == best:
            continue
        s = score(d)
        if s > best_s:
            best_s, best = s, d
    return best

def to_pixel(crop: Image.Image, img_h: int = 256, img_w: int = 2048) -> np.ndarray:
    work = crop.convert("RGB")
    angle = estimate_skew(work)
    if abs(angle) >= SKEW_MIN_APPLY:
        work = work.rotate(-angle, resample=Image.BICUBIC, expand=True, fillcolor=WHITE)
    if work.height > work.width:
        work = work.rotate(-90, expand=True)
    w, h = work.size
    nw = max(1, min(img_w, js_round(w * img_h / h)))
    final = Image.new("RGB", (img_w, img_h), WHITE)
    final.paste(work.resize((nw, img_h), Image.LANCZOS), (0, 0))
    a = (np.asarray(final, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])

def degenerate_period(seq: list[int]) -> int:
    """直近 REPEAT_WINDOW トークンが周期 1..4 で反復していれば周期を返す (行末の崩壊検出)。"""
    if len(seq) < REPEAT_WINDOW:
        return 0
    start = len(seq) - REPEAT_WINDOW
    for p in range(1, 5):
        if all(seq[i] == seq[i + p] for i in range(start, len(seq) - p)):
            return p
    return 0

_HIRA = lambda s: "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)

def decode_ids(ids: list[int], vocab: list[str]) -> str:
    out = "".join(vocab[i] if 0 <= i < len(vocab) else "" for i in ids if i not in STRUCT)
    out = re.sub(r"<rt2>.*?</rt2>", "", out)
    out = re.sub(r"</?rt2>", "", out)
    out = re.sub(r"<OKURI>(.*?)</OKURI>", lambda m: f"<OKURI>{_HIRA(m.group(1))}</OKURI>", out)
    out = re.sub(r"<KAERI>(.*?)</KAERI>", lambda m: f"<KAERI>{_HIRA(m.group(1))}</KAERI>", out)
    out = re.sub(r"＿([ぁ-ゖ]+)", lambda m: "＿" + _HIRA(m.group(1)), out)
    return out

@dataclass
class RecognitionResult:
    raw: str
    stop_reason: str
    token_count: int
    timings: dict[str, float]


@dataclass
class EncodedLine:
    hidden: np.ndarray
    timings: dict[str, float]


class Recognizer:
    def __init__(self, paths: dict, version: str = models.DEFAULT_VERSION, device: str = "cpu",
                 *, threads: int = 0, decoder_threads: int = 0, encoder_precision: str = "auto", quiet: bool = False, resolved_encoder=None):
        spec = models.specification(version)
        self.version = version
        self.img_h, self.img_w = spec.image_height, spec.image_width
        self.max_tokens = spec.max_tokens
        self.vocab = models.vocab(version)
        self.encoder_path = resolved_encoder if resolved_encoder is not None else models.encoder_path(
            paths["encoder"], device, precision=encoder_precision, quiet=quiet, digest=True)
        self.enc = session(self.encoder_path, device, threads=threads)
        self.pre = session(paths["prefill"], "cpu", threads=decoder_threads)
        self.step = session(paths["step"], "cpu", threads=decoder_threads)
        self.enc_in = self.enc.get_inputs()[0].name
        self.past = [i.name for i in self.step.get_inputs() if i.name.startswith("past_")]
        self.present = [o.name for o in self.step.get_outputs() if o.name.startswith("present_")]
        self.pre_out = [o.name for o in self.pre.get_outputs()]
        if (len(self.past) != spec.cache_tensors or len(self.present) != spec.cache_tensors
                or [p[5:] for p in self.past] != [q[8:] for q in self.present]
                or not {"logits", *self.present}.issubset(self.pre_out)):
            raise RuntimeError("incompatible decoder graph: cache inputs/outputs do not match the model specification")
        for graph in (self.pre, self.step):
            inputs = {i.name for i in graph.get_inputs()}
            if not {"input_ids", "encoder_hidden_states"}.issubset(inputs):
                raise RuntimeError("incompatible decoder graph: missing token or encoder inputs")
            logits = next((o for o in graph.get_outputs() if o.name == "logits"), None)
            if logits is None or (isinstance(logits.shape[-1], int) and logits.shape[-1] != len(self.vocab)):
                raise RuntimeError("incompatible decoder graph: vocabulary size does not match logits")

    def encode_crop(self, crop: Image.Image) -> EncodedLine:
        """Preprocess and encode one crop; decoder state is untouched."""
        timings = {}
        start = perf_counter()
        pixels = to_pixel(crop, self.img_h, self.img_w)
        timings["preprocess"] = perf_counter() - start
        start = perf_counter()
        hidden = self.enc.run(None, {self.enc_in: pixels})[0]
        timings["encoder"] = perf_counter() - start
        return EncodedLine(hidden, timings)

    def _decode(self, encoded: EncodedLine) -> tuple[list[int], str, dict[str, float]]:
        hidden, timings = encoded.hidden, encoded.timings.copy()
        start = perf_counter()
        out = dict(zip(self.pre_out, self.pre.run(None, {
            "input_ids": np.array([[CLS]], np.int64), "encoder_hidden_states": hidden,
        }), strict=True))
        timings["prefill"] = perf_counter() - start
        best = int(out["logits"][0, -1].argmax())
        if best == SEP:
            timings["decode"] = 0.0
            return [], "eos", timings
        gen = [best]
        past = {p: out[q] for p, q in zip(self.past, self.present, strict=True)}
        names = ["logits"] + self.present
        reason = "max_tokens"
        start = perf_counter()
        for _ in range(1, self.max_tokens):
            res = dict(zip(names, self.step.run(names, {
                "input_ids": np.array([[best]], np.int64), "encoder_hidden_states": hidden, **past,
            }), strict=True))
            best = int(res["logits"][0, -1].argmax())
            if best == SEP:
                reason = "eos"
                break
            gen.append(best)
            past = {p: res[q] for p, q in zip(self.past, self.present, strict=True)}
            period = degenerate_period(gen)
            if period > 0:
                del gen[len(gen) - (REPEAT_WINDOW - period):]
                reason = "repetition"
                break
        timings["decode"] = perf_counter() - start
        return gen, reason, timings

    def _generate(self, crop: Image.Image) -> tuple[list[int], str, dict[str, float]]:
        return self._decode(self.encode_crop(crop))

    def decode_encoded(self, encoded: EncodedLine) -> RecognitionResult:
        """Decode an encoded line with line-local KV state."""
        ids, reason, timings = self._decode(encoded)
        return RecognitionResult(decode_ids(ids, self.vocab), reason, len(ids), timings)

    def generate(self, crop: Image.Image) -> list[int]:
        return self._generate(crop)[0]

    def recognize_result(self, crop: Image.Image) -> RecognitionResult:
        ids, reason, timings = self._generate(crop)
        return RecognitionResult(decode_ids(ids, self.vocab), reason, len(ids), timings)

    def recognize(self, crop: Image.Image) -> str:
        """Recognize one line crop, returning text with structural tokens."""
        return self.recognize_result(crop).raw
