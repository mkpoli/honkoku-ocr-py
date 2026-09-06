import json

from benchmarks.order_experiment import column_sort, evaluate
from honkoku_ocr import Box


def test_column_sort_reads_right_to_left_then_top_to_bottom():
    boxes = [Box(0, 50, 10, 100, 1), Box(100, 0, 10, 100, 1), Box(0, 0, 10, 40, 1), Box(102, 120, 10, 50, 1)]
    assert column_sort(boxes, 0.5) == [1, 3, 2, 0]


def test_evaluate_reorders_stored_predictions(tmp_path):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "p.jpg").write_bytes(b"jpeg")
    manifest.write_text(json.dumps({"attribution": "t", "samples": [{"id": "p", "image": "p.jpg"}]}))
    report = {"pages": [{"id": "p", "reference": "あい\nうえ", "predicted": ["うえ", "あい"]}]}
    class FakeOCR:
        def layout(self, source):
            return [Box(0, 0, 10, 100, 1), Box(100, 0, 10, 100, 1)]   # second prediction sits in the right-hand column
    result = evaluate(report, manifest, [0.5], FakeOCR())
    assert result["totals"]["xycut"] == 1.0 and result["totals"]["column_0.5"] == 0.0
