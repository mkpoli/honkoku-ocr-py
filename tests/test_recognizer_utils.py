import numpy as np
from PIL import Image

from honkoku_ocr.recognizer import crop_with_margin, decode_ids, degenerate_period, to_pixel


def test_degenerate_period_detects_short_cycles():
    assert degenerate_period([5] * 12) == 1
    assert degenerate_period([1, 2] * 6) == 2
    assert degenerate_period(list(range(12))) == 0
    assert degenerate_period([5] * 11) == 0

def test_to_pixel_shape_and_rotation():
    crop = Image.new("RGB", (60, 900), (255, 255, 255))
    px = to_pixel(crop)
    assert px.shape == (1, 3, 256, 2048) and px.dtype == np.float32

def test_crop_with_margin_pads_with_white_outside_image():
    page = Image.new("RGB", (100, 100), (0, 0, 0))
    crop = crop_with_margin(page, 90, 0, 10, 100, margin=45)
    assert crop.size == (55, 190)
    assert crop.getpixel((54, 0)) == (255, 255, 255)   # right margin beyond the image
    assert crop.getpixel((0, 45)) == (0, 0, 0)          # the line itself

def test_decode_ids_strips_struct_and_rt2():
    vocab = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<rt2>", "</rt2>", "<OKURI>", "</OKURI>", "あ", "い"]
    assert decode_ids([2, 9, 5, 10, 6, 7, 9, 8, 3], vocab) == "あ<OKURI>ア</OKURI>"


def test_decode_ids_ignores_out_of_range_ids():
    assert decode_ids([-1, -99, 5, 6, 999], [''] * 5 + ['字']) == '字'


def test_skew_sign_straightens_tilted_vertical_column():
    from PIL import ImageDraw

    from honkoku_ocr.recognizer import estimate_skew
    image = Image.new('RGB', (100, 600), 'white')
    draw = ImageDraw.Draw(image)
    for y in range(50, 550, 60):
        draw.rectangle((40, y, 60, y + 35), fill='black')
    tilted = image.rotate(9, expand=True, fillcolor='white')
    angle = estimate_skew(tilted)
    assert 6 <= angle <= 12
    corrected = tilted.rotate(-angle, expand=True, fillcolor='white')
    assert abs(estimate_skew(corrected)) <= 2
