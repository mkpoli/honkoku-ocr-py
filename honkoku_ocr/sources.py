"""入力ファイルの読み込み。画像はPillow、PDFはpypdfium2（extra `pdf`）で開く。

`frame_count(path)`がコマ数を、`load_frame(path, frame, max_dimension)`が1コマをRGBの
`PIL.Image`として返す。画像はEXIFの向きを反映する。PDFの1ページは長辺が`max_dimension`
画素になる倍率で描画する（後段が長辺3,500pxに縮小するので、それ以上の解像度は使われない）。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES
PDF_RENDER_LONG_SIDE = 3500


def is_pdf(path) -> bool:
    return Path(path).suffix.lower() in PDF_SUFFIXES


def _pdfium():
    try:
        import pypdfium2
    except ModuleNotFoundError as exc:
        if exc.name != "pypdfium2":
            raise
        raise RuntimeError("PDF input needs the pdf extra: uv sync --extra cpu --extra pdf") from exc
    return pypdfium2


def frame_count(path) -> int:
    path = Path(path)
    if is_pdf(path):
        document = _pdfium().PdfDocument(str(path))
        try:
            return len(document)
        finally:
            document.close()
    with Image.open(path) as image:
        return getattr(image, "n_frames", 1)


def load_frame(path, frame: int = 0, *, max_dimension: int | None = PDF_RENDER_LONG_SIDE) -> Image.Image:
    """1コマをRGBで返す。呼び出し側が閉じる。PDFは`max_dimension`（省略時3,500px）を長辺として描画する。"""
    path = Path(path)
    if frame < 0:
        raise ValueError("frame must be nonnegative")
    if is_pdf(path):
        document = _pdfium().PdfDocument(str(path))
        try:
            if frame >= len(document):
                raise ValueError(f"frame {frame} out of range (document has {len(document)} pages)")
            page = document[frame]
            try:
                long_side = max(page.get_width(), page.get_height())
                scale = (max_dimension or PDF_RENDER_LONG_SIDE) / long_side
                bitmap = page.render(scale=scale, rotation=0)
                try:
                    return bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()
    with Image.open(path) as opened:
        opened.seek(frame)
        return ImageOps.exif_transpose(opened).convert("RGB")
