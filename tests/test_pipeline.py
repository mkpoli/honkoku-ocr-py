import math
from dataclasses import asdict

import pytest
from PIL import Image

from honkoku_ocr import OCR, Box, PreparedPage, pipeline
from honkoku_ocr.recognizer import RecognitionResult


class FakeRecognizer:
    def recognize_result(self, crop):
        return RecognitionResult('本文', 'eos', 2, {'encoder': 0.01})


class FakeDetector:
    def detect(self, image, *args):
        return [Box(10, 0, 20, 90, 0.8), Box(70, 0, 20, 90, 0.9)]


def test_constructor_is_lazy_and_supplied_boxes_never_load_detector(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('unexpected model resolution')
    monkeypatch.setattr(pipeline.models, 'ensure', unexpected)
    ocr = OCR(recognizer=FakeRecognizer())
    image = Image.new('RGB', (100, 100))
    assert len(ocr.run(image, [Box(1, 1, 20, 30, 1)])) == 1
    assert ocr.run(image, []) == []
    assert image.getpixel((0, 0)) == (0, 0, 0)  # caller retains ownership


def test_layout_only_never_loads_recognizer(monkeypatch):
    monkeypatch.setattr(pipeline.models, 'ensure', lambda *a, **k: pytest.fail('model load'))
    ocr = OCR(detector=FakeDetector())
    assert [box.x for box in ocr.layout(Image.new('RGB', (100, 100)))] == [70, 10]


def test_supplied_coordinates_survive_downscaling_exactly():
    image = Image.new('RGB', (4001, 4501))
    boxes = [Box(13 + i * 53, 17 + i, 31.5, 303.5, 1.0) for i in range(40)]
    result = OCR(recognizer=FakeRecognizer()).process(image, boxes)
    assert [(l.x, l.y, l.width, l.height) for l in result.lines] == [
        (b.x, b.y, b.width, b.height) for b in boxes]
    assert [l.reading_order for l in result.lines] == list(range(1, 41))
    assert result.processed_height == 3500
    assert result.settings['coordinate_space'] == 'exif_oriented_original'
    assert asdict(result)['lines'][0]['detection_confidence'] == 1.0


def test_exif_orientation_and_owned_files(tmp_path):
    path = tmp_path / 'rotated.jpg'
    image = Image.new('RGB', (40, 80))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    page = PreparedPage.load(path)
    assert (page.original_width, page.original_height) == (80, 40)
    path.unlink()  # prepared pixels survive file closure
    assert page.image.size == (80, 40)
    page.image.close()


@pytest.mark.parametrize('box', [Box(-11, 0, 10, 10, 1), Box(0, 0, 0, 10, 1),
                                 Box(100, 0, 10, 10, 1), Box(math.nan, 0, 1, 1, 1),
                                 Box(0, 0, 1, 1, 1.1)])
def test_invalid_boxes_fail_before_recognition(box):
    with pytest.raises(ValueError):
        OCR().run(Image.new('RGB', (100, 100)), [box])


def test_truncation_is_visible_in_page_result():
    class Truncated:
        def recognize_result(self, crop):
            return RecognitionResult('字', 'max_tokens', 192, {})
    result = OCR(recognizer=Truncated()).process(Image.new('RGB', (100, 100)), [Box(0, 0, 10, 10, 1)])
    assert result.warnings == ['line 1: generation stopped by max_tokens']
    assert result.lines[0].stop_reason == 'max_tokens'


def test_layout_resolves_only_detector_model(monkeypatch):
    calls = []
    def ensure(version, **kwargs):
        calls.append(kwargs['roles'])
        return {'layout': 'layout.onnx'}
    monkeypatch.setattr(pipeline.models, 'ensure', ensure)
    monkeypatch.setattr(pipeline, 'LayoutDetector', lambda *a, **k: FakeDetector())
    ocr = OCR()
    assert calls == []
    ocr.layout(Image.new('RGB', (100, 100)))
    ocr.layout(Image.new('RGB', (100, 100)))
    assert calls == [['layout']]


def test_overhanging_boxes_preserve_coordinates_but_clip_crops():
    box = Box(-1, -2, 102, 103, 1)
    page = PreparedPage.load(Image.new('RGB', (100, 100)))
    scaled = page.scale_box(box)
    assert (scaled.x, scaled.y, scaled.width, scaled.height) == (0, 0, 100, 100)
    result = OCR(recognizer=FakeRecognizer()).process(page, [box])
    line = result.lines[0]
    assert (line.x, line.y, line.width, line.height) == (-1, -2, 102, 103)
    page.close()


def test_identity_and_recognizer_share_verified_encoder(tmp_path, monkeypatch):
    paths = {role: tmp_path / f'{role}.onnx' for role in ('encoder', 'prefill', 'step')}
    for path in paths.values():
        path.write_bytes(b'model')
    resolutions = []
    def encoder_path(path, *args, **kwargs):
        resolutions.append(path)
        return path
    monkeypatch.setattr(pipeline.models, 'ensure', lambda version, **kwargs: {role: paths[role] for role in kwargs['roles']})
    monkeypatch.setattr(pipeline.models, 'encoder_path', encoder_path)
    def recognizer(paths, *args, **kwargs):
        assert kwargs['resolved_encoder'] == paths['encoder']
        return FakeRecognizer()
    monkeypatch.setattr(pipeline, 'Recognizer', recognizer)
    ocr = OCR()
    identity = ocr.model_identity(list(paths))
    assert ocr.recognizer is ocr.recognizer
    assert ocr.model_identity(list(paths)) == identity
    assert resolutions == [paths['encoder']]
