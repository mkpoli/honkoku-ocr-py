"""みんなで翻刻OCR (honkoku-ocr-web) の Python 移植。

レイアウト認識 (RTMDet-s 行検出 + XY-Cut 読み順) と行認識 (ConvNeXt V2 encoder + RoBERTa decoder) を
onnxruntime で実行し、Koji 記法の翻刻テキストを返す。
"""
from importlib.metadata import PackageNotFoundError, version

from .koji import raw_to_koji, raw_to_plain
from .layout import Box
from .pipeline import OCR, LineResult, PageFailure, PageInput, PageResult, PreparedPage, ProcessingCancelled, ocr_image

__all__ = ["OCR", "Box", "LineResult", "PageResult", "PreparedPage", "PageInput", "PageFailure", "ProcessingCancelled", "ocr_image", "raw_to_koji", "raw_to_plain"]

try:
    __version__ = version("honkoku-ocr-py")
except PackageNotFoundError:
    __version__ = "unknown"
