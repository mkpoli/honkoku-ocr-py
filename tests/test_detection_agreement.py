import json

from benchmarks import detection_agreement
from benchmarks.detection_agreement import iou, match


def box(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


def test_matching_is_one_to_one_and_greedy_by_iou():
    reference = [box(0, 0, 10, 100), box(20, 0, 10, 100)]
    detected = [box(1, 0, 10, 100), box(2, 0, 10, 100), box(200, 0, 10, 100)]
    pairs = match(detected, reference, 0.5)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)]        # the second overlapping detection stays unmatched
    assert iou(detected[0], reference[0]) > iou(detected[1], reference[0])


def test_threshold_excludes_weak_overlaps():
    assert match([box(0, 0, 10, 10)], [box(6, 0, 10, 10)], 0.5) == []
    assert len(match([box(0, 0, 10, 10)], [box(1, 0, 10, 10)], 0.5)) == 1


def test_evaluate_forwards_frame_and_records_source(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "scans.tif").write_bytes(b"tiff")
    manifest.write_text(json.dumps({"attribution": "t", "samples": [
        {"id": "p3", "image": "scans.tif", "frame": 3, "boxes": [box(0, 0, 10, 100)]}]}))
    calls = []
    class Line:
        x, y, width, height, detection_confidence = 0, 0, 10, 100, 0.9
    class Result:
        lines = [Line()]
    class FakeOCR:
        settings = {}
        def __init__(self, *a, **kw): pass
        def process(self, source, *, frame, layout_only):
            calls.append((source.name, frame, layout_only)); return Result()
        def model_identity(self): return {}
    monkeypatch.setattr(detection_agreement, "OCR", FakeOCR)
    report = detection_agreement.evaluate(manifest, model="v18", device="cpu", threshold=0.5)
    assert calls == [("scans.tif", 3, True)]
    page = report["pages"][0]
    assert page["frame"] == 3 and page["matched"] == 1 and len(page["source_sha256"]) == 64
