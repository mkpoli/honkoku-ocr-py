"""Reusable OCR components and page processing in EXIF-oriented coordinates."""
from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from PIL import Image, ImageOps

from . import models
from .koji import raw_to_koji, raw_to_plain
from .layout import Box, LayoutDetector
from .output import safe_error
from .reading_order import order
from .recognizer import EncodedLine, RecognitionResult, Recognizer, crop_with_margin, js_round
from .sources import load_frame

MAX_IMAGE_DIM = 3500
MARGIN = 45
SCHEMA_VERSION = 1


class Detector(Protocol):
    def detect(self, image: Image.Image, conf_threshold: float = 0.3,
               ios_threshold: float = 0.8) -> list[Box]: ...


class LineRecognizer(Protocol):
    def recognize_result(self, crop: Image.Image) -> RecognitionResult: ...


@runtime_checkable
class StagedRecognizer(Protocol):
    def encode_crop(self, crop: Image.Image) -> EncodedLine: ...
    def decode_encoded(self, encoded: EncodedLine) -> RecognitionResult: ...


@dataclass
class PreparedPage:
    image: Image.Image
    original_width: int
    original_height: int
    scale_x: float
    scale_y: float
    frame: int = 0

    @classmethod
    def load(cls, source, *, frame: int = 0, max_dimension: int = MAX_IMAGE_DIM):
        if max_dimension < 1 or frame < 0:
            raise ValueError("max_dimension must be positive and frame nonnegative")
        if isinstance(source, Image.Image):
            if frame:
                raise ValueError("frame selection requires a file; PIL images use their current frame")
            img = ImageOps.exif_transpose(source).convert("RGB")
        else:
            img = load_frame(source, frame, max_dimension=max_dimension)
        width, height = img.size
        scale = min(1.0, max_dimension / max(width, height))
        nw, nh = max(1, js_round(width * scale)), max(1, js_round(height * scale))
        if img.size != (nw, nh):
            original = img
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
            original.close()
        return cls(img, width, height, nw / width, nh / height, frame)

    def close(self):
        self.image.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def scale_box(self, box: Box) -> Box:
        self.validate_box(box)
        x, y = max(0, box.x), max(0, box.y)
        clipped = Box(x, y, min(self.original_width, box.x + box.width) - x,
                      min(self.original_height, box.y + box.height) - y, box.confidence)
        scaled = _scale(clipped, self.scale_x, self.scale_y)
        return replace(scaled, width=max(1, scaled.width), height=max(1, scaled.height))

    def original_box(self, box: Box) -> Box:
        return _unscale(box, self.scale_x, self.scale_y)

    def validate_box(self, box: Box):
        values = (box.x, box.y, box.width, box.height, box.x + box.width, box.y + box.height, box.confidence)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("box coordinates and confidence must be finite")
        if box.width <= 0 or box.height <= 0 or not 0 <= box.confidence <= 1:
            raise ValueError("box dimensions must be positive and confidence between 0 and 1")
        if (box.x >= self.original_width or box.y >= self.original_height
                or box.x + box.width <= 0 or box.y + box.height <= 0):
            raise ValueError("box must overlap the EXIF-oriented image")


@dataclass
class LineResult:
    reading_order: int
    x: float
    y: float
    width: float
    height: float
    detection_confidence: float
    raw: str
    koji: str
    plain: str
    stop_reason: str = "eos"
    token_count: int = 0
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class PageResult:
    schema_version: int
    width: int
    height: int
    processed_width: int
    processed_height: int
    frame: int
    model: str
    settings: dict
    lines: list[LineResult]
    timings: dict[str, float]
    warnings: list[str]


class ModelSetupError(RuntimeError):
    """Model resolution/session creation failed; retrying each page is inappropriate."""


class ProcessingCancelled(Exception):
    """Cooperative cancellation; no partial page result is returned."""


@dataclass
class PageInput:
    """One batch input; files select a frame, PIL images use their current frame."""

    source: object
    boxes: list[Box] | None = None
    frame: int = 0


@dataclass(frozen=True)
class PageFailure:
    index: int
    frame: int
    error_type: str
    message: str


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise ProcessingCancelled()


class OCR:
    """Reuse one instance across pages. Model sessions load only when needed.

    Injected components implement Detector/LineRecognizer; the caller owns their
    lifecycle. Supplied boxes are validated in EXIF-oriented source coordinates
    and retain both their coordinates and input order in the result.
    """

    def __init__(self, version: str = models.DEFAULT_VERSION, device: str = "cpu",
                 quiet: bool = False, *, detector: Detector | None = None,
                 recognizer: LineRecognizer | None = None, offline: bool = False,
                 threads: int = 0, decoder_threads: int = 0,
                 encoder_precision: str = "auto", max_dimension: int = MAX_IMAGE_DIM,
                 margin: int = MARGIN, conf_threshold: float = 0.3,
                 ios_threshold: float = 0.8, overlap: bool = False):
        models.specification(version)
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if threads < 0 or decoder_threads < 0 or max_dimension < 1 or margin < 0:
            raise ValueError("invalid thread count, maximum dimension or margin")
        if encoder_precision not in {"auto", "fp16", "fp32"}:
            raise ValueError("encoder_precision must be auto, fp16 or fp32")
        if not 0 <= conf_threshold <= 1 or not 0 <= ios_threshold <= 1:
            raise ValueError("thresholds must be between 0 and 1")
        self.version, self.device, self.quiet = version, device, quiet
        self.offline, self.threads, self.decoder_threads = offline, threads, decoder_threads
        self.encoder_precision = encoder_precision
        self.overlap = overlap
        self.max_dimension, self.margin = max_dimension, margin
        self.conf_threshold, self.ios_threshold = conf_threshold, ios_threshold
        self._detector, self._recognizer = detector, recognizer
        self._paths: dict[str, Path] = {}
        self._resolved_encoder: Path | None = None
        self._identity: dict[str, Any] = {}

    def paths(self, roles) -> dict:
        missing = [role for role in roles if role not in self._paths]
        if missing:
            self._paths.update(models.ensure(self.version, roles=missing, quiet=self.quiet,
                                              offline=self.offline, digest=True))
        return {role: self._paths[role] for role in roles}

    def resolved_paths(self, roles) -> dict:
        paths = self.paths(roles)
        if "encoder" in paths:
            if self._resolved_encoder is None:
                self._resolved_encoder = models.encoder_path(paths["encoder"], self.device,
                                                              precision=self.encoder_precision,
                                                              quiet=self.quiet, digest=True)
            paths["encoder"] = self._resolved_encoder
        return paths

    def model_identity(self, roles=None) -> dict:
        """Fingerprint selected roles, or only models already loaded when omitted."""
        if roles is None:
            roles = tuple(self._paths)
        paths = self.resolved_paths(roles)
        for role, path in paths.items():
            if role in self._identity:
                continue
            sidecar = path.with_suffix(path.suffix + ".json")
            if role == "encoder" and path != self._paths[role]:
                # resolved_paths validated the derived bytes against this record.
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                digest = metadata["sha256"]
                self._identity["encoder_conversion"] = models._sha256(sidecar)
            else:
                # paths() verifies the distributed files before caching them.
                digest = models.EXPECTED.get(path.name, (None, None))[1] or models._sha256(path)
            self._identity[role] = {"file": path.name, "sha256": digest}
        if "encoder" in paths and "vocabulary" not in self._identity:
            self._identity["vocabulary"] = models._sha256(models.specification(self.version).vocabulary)
        selected = set(roles) | ({"vocabulary", "encoder_conversion"} if "encoder" in roles else set())
        return {key: value for key, value in self._identity.items() if key in selected}

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            try:
                self._detector = LayoutDetector(self.paths(["layout"])["layout"], self.device,
                                                 threads=self.threads)
            except Exception as error:
                raise ModelSetupError(safe_error(error)) from error
        return self._detector

    @property
    def recognizer(self) -> LineRecognizer:
        if self._recognizer is None:
            try:
                self._recognizer = Recognizer(self.resolved_paths(["encoder", "prefill", "step"]),
                                              self.version, self.device, threads=self.threads,
                                              decoder_threads=self.decoder_threads,
                                              encoder_precision=self.encoder_precision, quiet=self.quiet,
                                              resolved_encoder=self._resolved_encoder)
            except Exception as error:
                raise ModelSetupError(safe_error(error)) from error
        return self._recognizer

    @property
    def settings(self) -> dict:
        return {"device": self.device, "threads": self.threads,
                "decoder_threads": self.decoder_threads, "encoder_precision": self.encoder_precision,
                "max_dimension": self.max_dimension, "margin": self.margin,
                "conf_threshold": self.conf_threshold, "ios_threshold": self.ios_threshold,
                "coordinate_space": "exif_oriented_original", "overlap": self.overlap}

    def prepare(self, image, *, frame: int = 0) -> PreparedPage:
        return PreparedPage.load(image, frame=frame, max_dimension=self.max_dimension)

    def process(self, image, boxes: list[Box] | None = None, *, frame: int = 0,
                layout_only: bool = False, cancelled: Callable[[], bool] | None = None) -> PageResult:
        """Process one page, raising ProcessingCancelled without a partial result.

        With overlap enabled, cancelled is also called on the encoder worker.
        Use a thread-safe predicate such as threading.Event.is_set. Cancellation
        waits for any running preprocessing/model call and worker cleanup.
        """
        _check_cancelled(cancelled)
        total = perf_counter()
        page = image if isinstance(image, PreparedPage) else self.prepare(image, frame=frame)
        timings = {"load": perf_counter() - total}
        try:
            if boxes is not None:
                for box in boxes:
                    page.validate_box(box)
                work = [(page.scale_box(box), replace(box)) for box in boxes]
            else:
                start = perf_counter()
                detector = self.detector
                timings["detector_setup"] = perf_counter() - start
                start = perf_counter()
                detected = detector.detect(page.image, self.conf_threshold, self.ios_threshold)
                timings["layout"] = perf_counter() - start
                start = perf_counter()
                ranks = order([(b.x, b.y, b.width, b.height) for b in detected])
                work = [(b, page.original_box(b)) for _, b in sorted(
                    zip(ranks, detected, strict=True), key=lambda pair: pair[0])]
                timings["reading_order"] = perf_counter() - start
            lines, warnings = [], []
            recognizer = None
            if work and not layout_only:
                start = perf_counter()
                recognizer = self.recognizer
                timings["recognizer_setup"] = perf_counter() - start
            with closing(self._line_results(page, work, recognizer, cancelled)) as results:
                for rank, ((_, original), result) in enumerate(zip(work, results, strict=True), 1):
                    if recognizer is not None and result.stop_reason != "eos":
                        warnings.append(f"line {rank}: generation stopped by {result.stop_reason}")
                    for stage, elapsed in result.timings.items():
                        timings[stage] = timings.get(stage, 0.0) + elapsed
                    lines.append(LineResult(rank, original.x, original.y, original.width,
                                            original.height, original.confidence, result.raw,
                                            raw_to_koji(result.raw), raw_to_plain(result.raw),
                                            result.stop_reason, result.token_count, result.timings))
            _check_cancelled(cancelled)
            timings["total"] = perf_counter() - total
            return PageResult(SCHEMA_VERSION, page.original_width, page.original_height,
                              page.image.width, page.image.height, page.frame, self.version,
                              self.settings, lines, timings, warnings)
        finally:
            if not isinstance(image, PreparedPage):
                page.image.close()

    def _line_results(self, page, work, recognizer, cancelled):
        if recognizer is None or not self.overlap:
            for scaled, _ in work:
                _check_cancelled(cancelled)
                if recognizer is None:
                    yield RecognitionResult("", "not_run", 0, {})
                else:
                    with closing(crop_with_margin(page.image, scaled.x, scaled.y,
                                                  scaled.width, scaled.height, self.margin)) as crop:
                        yield recognizer.recognize_result(crop)
            return
        if not isinstance(recognizer, StagedRecognizer):
            raise TypeError("overlap requires a recognizer with encode_crop and decode_encoded")

        def encode(box):
            _check_cancelled(cancelled)
            with closing(crop_with_margin(page.image, box.x, box.y, box.width, box.height, self.margin)) as crop:
                return recognizer.encode_crop(crop)

        boxes = iter(scaled for scaled, _ in work)
        pending = deque()
        # One encoder producer, one decoder consumer, at most two futures ahead.
        # Joining on every exit protects page/crop ownership on error or cancellation.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-encoder") as executor:
            try:
                for box in boxes:
                    pending.append(executor.submit(encode, box))
                    if len(pending) == 2:
                        break
                while pending:
                    _check_cancelled(cancelled)
                    encoded = pending.popleft().result()
                    _check_cancelled(cancelled)
                    box = next(boxes, None)
                    if box is not None:
                        pending.append(executor.submit(encode, box))
                    yield recognizer.decode_encoded(encoded)
            finally:
                for future in pending:
                    future.cancel()

    def process_many(self, images: Iterable, *, layout_only: bool = False,
                     progress: Callable[[int, PageResult | PageFailure], None] | None = None,
                     cancelled: Callable[[], bool] | None = None) -> Iterator[PageResult | PageFailure]:
        """Yield ordered page results or failures without buffering the inputs.

        Each input is a source accepted by process(), or PageInput for per-page
        boxes/frame selection. Indices are one-based. Progress runs once before
        each yield, including failures; callback, input-iterator and model-setup errors propagate.
        Cancellation is checked before consuming an input and between lines. It
        ends the iterator without yielding an unfinished page. An in-flight model
        call completes before cancellation takes effect. With overlap enabled the
        cancellation predicate also runs on the encoder worker; use a thread-safe
        predicate such as threading.Event.is_set. Caller-owned images and
        PreparedPage objects remain open. One OCR instance is not reentrant.
        """
        sources = iter(images)
        index = 0
        while True:
            if cancelled is not None and cancelled():
                return
            try:
                source = next(sources)
            except StopIteration:
                return
            index += 1
            job = source if isinstance(source, PageInput) else PageInput(source)
            try:
                result: PageResult | PageFailure = self.process(job.source, job.boxes, frame=job.frame,
                                      layout_only=layout_only, cancelled=cancelled)
            except ProcessingCancelled:
                return
            except ModelSetupError:
                raise
            except Exception as error:
                result = PageFailure(index, job.frame, type(error).__name__, safe_error(error))
            if progress is not None:
                progress(index, result)
            yield result

    def layout(self, image) -> list[Box]:
        return [Box(int(line.x), int(line.y), int(line.width), int(line.height), line.detection_confidence)
                for line in self.process(image, layout_only=True).lines]

    def run(self, image, boxes: list[Box] | None = None) -> list[LineResult]:
        return self.process(image, boxes).lines


def _scale(b: Box, sx: float, sy: float) -> Box:
    x0, y0 = js_round(b.x * sx), js_round(b.y * sy)
    return Box(x0, y0, js_round((b.x + b.width) * sx) - x0,
               js_round((b.y + b.height) * sy) - y0, b.confidence)


def _unscale(b: Box, sx: float, sy: float) -> Box:
    x0, y0 = js_round(b.x / sx), js_round(b.y / sy)
    return Box(x0, y0, js_round((b.x + b.width) / sx) - x0,
               js_round((b.y + b.height) / sy) - y0, b.confidence)


def ocr_image(image, version: str = models.DEFAULT_VERSION, device: str = "cpu") -> list[dict]:
    """One-shot convenience; reuse OCR for a collection of pages."""
    return [asdict(result) for result in OCR(version, device).run(image)]
