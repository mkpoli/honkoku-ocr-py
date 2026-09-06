from threading import Event, current_thread
from threading import enumerate as threads

import numpy as np
import pytest
from PIL import Image

from honkoku_ocr import OCR, Box, PageInput
from honkoku_ocr.recognizer import EncodedLine, RecognitionResult


class Staged:
    def __init__(self):
        self.count = 0
        self.next_encoded = Event()
        self.decoded = []

    def encode_crop(self, crop):
        assert current_thread().name.startswith('ocr-encoder')
        index = self.count
        self.count += 1
        if index == 1:
            self.next_encoded.set()
        return EncodedLine(np.array([index]), {'encoder': 0.01})

    def decode_encoded(self, encoded):
        if not self.decoded:
            assert self.next_encoded.wait(2), 'second encoding did not overlap first decoding'
            assert self.count <= 3, 'producer exceeded two futures ahead'
        index = encoded.hidden.item()
        self.decoded.append(index)
        return RecognitionResult(str(index), 'eos', 1, encoded.timings)


def test_overlap_preserves_order_and_bounds_producer():
    rec = Staged()
    boxes = [Box(i * 2, 0, 1, 10, 1) for i in range(10)]
    with Image.new('RGB', (100, 100)) as image:
        result = OCR(recognizer=rec, overlap=True).process(image, boxes)
    assert [line.raw for line in result.lines] == [str(i) for i in range(10)]
    assert rec.decoded == list(range(10))
    assert not any(t.name.startswith('ocr-encoder') for t in threads())


@pytest.mark.parametrize('stage', ['encode', 'decode'])
def test_overlap_failure_joins_worker_and_closes_crops(stage):
    crops = []
    class Failing(Staged):
        def encode_crop(self, crop):
            crops.append(crop)
            if stage == 'encode':
                raise RuntimeError('encoder failure')
            return super().encode_crop(crop)
        def decode_encoded(self, encoded):
            raise RuntimeError('decoder failure')
    with Image.new('RGB', (100, 100)) as image:
        with pytest.raises(RuntimeError, match='failure'):
            OCR(recognizer=Failing(), overlap=True).process(image, [Box(0, 0, 10, 10, 1)] * 5)
    assert crops
    for crop in crops:
        with pytest.raises(ValueError):
            crop.getpixel((0, 0))
    assert not any(t.name.startswith('ocr-encoder') for t in threads())


def test_overlap_cancellation_joins_worker_without_partial_page():
    stop = Event()
    class Cancelling(Staged):
        def decode_encoded(self, encoded):
            stop.set()
            return RecognitionResult('', 'eos', 0, {})
    with Image.new('RGB', (100, 100)) as image:
        inputs = [PageInput(image, [Box(0, 0, 10, 10, 1)] * 10)]
        assert list(OCR(recognizer=Cancelling(), overlap=True).process_many(inputs, cancelled=stop.is_set)) == []
    assert not any(t.name.startswith('ocr-encoder') for t in threads())


def test_overlap_requires_staged_injected_recognizer():
    with Image.new('RGB', (10, 10)) as image:
        with pytest.raises(TypeError, match='encode_crop'):
            OCR(recognizer=object(), overlap=True).process(image, [Box(0, 0, 10, 10, 1)])
