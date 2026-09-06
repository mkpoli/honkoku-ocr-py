from honkoku_ocr.reading_order import order

def test_vertical_columns_read_right_to_left():
    # three vertical columns, x increasing to the right; reading order must start at the rightmost
    boxes = [(10, 0, 40, 900), (110, 0, 40, 900), (210, 0, 40, 900)]
    assert order(boxes) == [2, 1, 0]

def test_two_pages_side_by_side_each_top_to_bottom():
    # right page (x 600..900) then left page (x 0..300); within a page, columns right to left
    boxes = [(0, 0, 40, 900), (100, 0, 40, 900), (600, 0, 40, 900), (700, 0, 40, 900)]
    assert order(boxes) == [3, 2, 1, 0]

def test_trivial_inputs():
    assert order([]) == []
    assert order([(0, 0, 10, 10)]) == [0]
