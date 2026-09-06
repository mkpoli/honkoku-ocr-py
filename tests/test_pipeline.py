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


def test_batch_is_lazy_and_continues_after_bad_page(tmp_path):
    from honkoku_ocr import PageFailure, PageInput, PageResult
    image = Image.new('RGB', (100, 100))
    consumed, progress = [], []
    def inputs():
        for item in (PageInput(image, []), tmp_path / 'missing.png', PageInput(image, [])):
            consumed.append(item)
            yield item
    batch = OCR().process_many(inputs(), progress=lambda i, r: progress.append((i, r)))
    assert consumed == []
    assert isinstance(next(batch), PageResult)
    assert len(consumed) == 1
    remaining = list(batch)
    assert isinstance(remaining[0], PageFailure)
    assert remaining[0].index == 2 and remaining[0].error_type == 'FileNotFoundError'
    assert isinstance(remaining[1], PageResult)
    assert [i for i, _ in progress] == [1, 2, 3]
    assert image.getpixel((0, 0)) == (0, 0, 0)


def test_batch_cancellation_does_not_consume_next_input():
    from threading import Event

    from honkoku_ocr import PageInput
    stop = Event()
    def inputs():
        yield PageInput(Image.new('RGB', (10, 10)), [])
        pytest.fail('consumed input after cancellation')
    batch = OCR().process_many(inputs(), cancelled=stop.is_set,
                              progress=lambda i, r: stop.set())
    assert len(list(batch)) == 1


def test_cancellation_between_lines_closes_owned_page(monkeypatch):
    from threading import Event
    stop = Event()
    prepared = PreparedPage.load(Image.new('RGB', (100, 100)))
    class CancellingRecognizer:
        def recognize_result(self, crop):
            stop.set()
            return RecognitionResult('字', 'eos', 1, {})
    ocr = OCR(detector=FakeDetector(), recognizer=CancellingRecognizer())
    monkeypatch.setattr(ocr, 'prepare', lambda *a, **k: prepared)
    assert list(ocr.process_many(['page'], cancelled=stop.is_set)) == []
    with pytest.raises(ValueError):
        prepared.image.getpixel((0, 0))


def test_batch_progress_errors_propagate():
    from honkoku_ocr import PageInput
    def progress(*args):
        raise RuntimeError('callback failed')
    with pytest.raises(RuntimeError, match='callback failed'):
        list(OCR().process_many([PageInput(Image.new('RGB', (10, 10)), [])], progress=progress))


def test_batch_page_inputs_select_frames(tmp_path):
    from honkoku_ocr import PageInput
    path = tmp_path / 'pages.tif'
    Image.new('RGB', (10, 20)).save(path, save_all=True, append_images=[Image.new('RGB', (30, 40))])
    results = list(OCR().process_many([PageInput(path, [], 1), PageInput(path, [], 0)]))
    assert [(r.frame, r.width, r.height) for r in results] == [(1, 30, 40), (0, 10, 20)]


def test_batch_model_setup_failure_aborts_without_retry(monkeypatch):
    from honkoku_ocr import ModelSetupError
    calls = []
    def missing(*args, **kwargs):
        calls.append(1)
        raise FileNotFoundError('offline cache is missing')
    monkeypatch.setattr(pipeline.models, 'ensure', missing)
    def inputs():
        yield Image.new('RGB', (10, 10))
        pytest.fail('setup failure consumed a second page')
    with pytest.raises(ModelSetupError, match='offline cache'):
        list(OCR(offline=True).process_many(inputs()))
    assert calls == [1]


def test_default_model_identity_does_not_load_unused_models(tmp_path, monkeypatch):
    calls = []
    path = tmp_path / 'layout.onnx'
    path.write_bytes(b'layout model')
    def ensure(version, **kwargs):
        calls.append(kwargs['roles'])
        assert kwargs['roles'] == ['layout']
        return {'layout': path}
    monkeypatch.setattr(pipeline.models, 'ensure', ensure)
    monkeypatch.setattr(pipeline, 'LayoutDetector', lambda *a, **k: FakeDetector())
    ocr = OCR()
    assert ocr.model_identity() == {}
    ocr.layout(Image.new('RGB', (100, 100)))
    assert set(ocr.model_identity()) == {'layout'}
    assert calls == [['layout']]
