import numpy as np
from PIL import Image

from honkoku_ocr.layout import Box, LayoutDetector, _remove_nested


def test_nested_boxes_keep_largest_even_when_smaller_has_higher_confidence():
    outer = Box(0, 0, 100, 100, 0.3)
    inner = Box(10, 10, 20, 20, 0.99)
    separate = Box(120, 0, 20, 20, 0.8)
    assert _remove_nested([inner, separate, outer], 0.8) == [outer, separate]


def test_letterbox_channel_order_and_padding():
    image = Image.new('RGB', (1024, 512), (255, 0, 0))
    tensor, scale, px, py = LayoutDetector._letterbox(image)
    assert (scale, px, py) == (1, 0, 256)
    np.testing.assert_allclose(tensor[0, :, 256, 0],
                               [(0 - 103.53) / 57.375, (0 - 116.28) / 57.12,
                                (255 - 123.675) / 58.395], rtol=1e-6)
    assert tensor.flags.c_contiguous


def test_detection_clips_edges_and_discards_small_boxes():
    class Session:
        def run(self, names, inputs):
            return [np.array([[[-10, -5, 50, 100, 0.9], [10, 10, 12, 20, 0.9],
                               [20, 20, 40, 40, 0.1]]], dtype=np.float32)]
    detector = object.__new__(LayoutDetector)
    detector.sess, detector.input = Session(), 'image'
    boxes = detector.detect(Image.new('RGB', (1024, 1024)))
    assert [(b.x, b.y, b.width, b.height) for b in boxes] == [(0, 0, 50, 100)]


def test_extremely_thin_images_keep_nonzero_resize_dimensions():
    tensor, scale, _, _ = LayoutDetector._letterbox(Image.new('RGB', (1, 3500)))
    assert tensor.shape == (1, 3, 1024, 1024)
    assert scale > 0
