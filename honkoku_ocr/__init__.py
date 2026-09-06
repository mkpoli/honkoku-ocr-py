"""みんなで翻刻OCR (honkoku-ocr-web) の Python 移植。

レイアウト認識 (RTMDet-s 行検出 + XY-Cut 読み順) と行認識 (ConvNeXt V2 encoder + RoBERTa decoder) を
onnxruntime で実行し、Koji 記法の翻刻テキストを返す。
"""
from .pipeline import OCR, LineResult, ocr_image
from .layout import Box
from .koji import raw_to_koji, raw_to_plain

__all__ = ["OCR", "Box", "LineResult", "ocr_image", "raw_to_koji", "raw_to_plain"]
__version__ = "0.1.0"
