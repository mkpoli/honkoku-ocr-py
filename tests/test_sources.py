import pytest
from PIL import Image

from honkoku_ocr import sources


@pytest.fixture
def pdf(tmp_path):
    pages = [Image.new("RGB", (400, 600), (255, 255, 255)), Image.new("RGB", (600, 400), (200, 200, 200))]
    path = tmp_path / "scans.pdf"
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=100.0)
    return path


def test_pdf_pages_are_counted_and_rendered_to_the_requested_long_side(pdf):
    pytest.importorskip("pypdfium2")
    assert sources.frame_count(pdf) == 2
    first = sources.load_frame(pdf, 0, max_dimension=300)
    second = sources.load_frame(pdf, 1, max_dimension=300)
    assert first.mode == "RGB" and max(first.size) == 300 and first.height > first.width
    assert max(second.size) == 300 and second.width > second.height
    with pytest.raises(ValueError, match="out of range"):
        sources.load_frame(pdf, 2)


def test_images_keep_their_size_and_frames(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (50, 30)).save(path)
    assert sources.frame_count(path) == 1
    assert sources.load_frame(path, 0, max_dimension=None).size == (50, 30)
    assert not sources.is_pdf(path) and sources.is_pdf(tmp_path / "x.PDF")


def test_missing_pypdfium2_gives_an_install_hint(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "pypdfium2":
            raise ModuleNotFoundError(name="pypdfium2")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pdf extra"):
        sources.frame_count(tmp_path / "doc.pdf")
