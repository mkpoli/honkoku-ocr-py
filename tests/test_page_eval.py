import json

from benchmarks import page_eval
from benchmarks.page_eval import align_lines, bag, evaluate_page, normalize, squeeze


def test_squeeze_removes_whitespace_and_editorial_notes():
    assert squeeze("相定　ヶ條\n之内 【１行目上部】私云") == "相定ヶ條之内私云"


def test_bag_counts_missing_and_extra_characters_regardless_of_order():
    assert bag("あいう", "ういえ") == {"missed": 1, "extra": 1, "missed_share": 1 / 3, "extra_share": 1 / 3}


def test_alignment_ignores_order_and_counts_missed_and_extra():
    reference = ["相定ヶ條之内", "雖為奉公人", "同前候事"]
    predicted = ["同前候事", "相定ヶ条之内", "雖為奉公人", "余分"]
    lines = align_lines(reference, predicted)
    assert lines["paired"] == 3 and lines["missed"] == 0 and lines["extra"] == 1
    assert [(p["reference"], p["predicted"]) for p in lines["pairs"]] == [(0, 1), (1, 2), (2, 0)]
    assert lines["paired_errors"] == 1 and lines["in_order_share"] == 0.5


def test_page_metrics_distinguish_raw_and_squeezed():
    page = evaluate_page("あい\nうえ", ["あいうえ"])
    assert page["raw_errors"] == 1 and page["squeezed_errors"] == 0
    assert page["lines"]["reference_lines"] == 2 and page["lines"]["predicted_lines"] == 1


def test_run_uses_pipeline_and_records_totals(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "p.jpg").write_bytes(b"jpeg")
    manifest.write_text(json.dumps({"attribution": "t", "samples": [
        {"id": "p", "image": "p.jpg", "reference": "あい\nうえ"}, {"id": "q", "image": "p.jpg", "frame": 2, "reference": "か"}]}))
    calls = []
    class Line:
        def __init__(self, koji): self.koji = koji
    class Result:
        timings = {"total": 1.5}; warnings = []
        def __init__(self, lines): self.lines = [Line(k) for k in lines]
    class FakeOCR:
        settings = {"device": "cpu"}
        def __init__(self, *a, **kw): pass
        def process(self, source, *, frame):
            calls.append((source.name, frame)); return Result(["あい", "うえ"] if frame == 0 else ["き"])
        def model_identity(self): return {}
    monkeypatch.setattr(page_eval, "OCR", FakeOCR)
    report = page_eval.run(manifest, model="v18", device="cpu", precision="auto", threads=0, decoder_threads=0, overlap=False, offline=False, limit=None)
    assert calls == [("p.jpg", 0), ("p.jpg", 2)]
    assert report["totals"]["raw_errors"] == 1 and report["totals"]["squeezed_errors"] == 1 and report["totals"]["pages"] == 2
    assert report["pages"][1]["frame"] == 2 and "/home/" not in json.dumps(report)
    assert report["totals"]["bag_missed"] == 1 and report["totals"]["bag_extra"] == 1


def test_rescore_recomputes_from_stored_predictions(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attribution": "t", "samples": []}))
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"settings": {}, "models": {}, "pages": [
        {"id": "p", "frame": 0, "image_sha256": "0" * 64, "seconds": 1.0, "warnings": [], "predicted": ["あい"], "reference": "あい\n【注】"}]}))
    out = tmp_path / "new.json"
    assert page_eval.main([str(manifest), "--output", str(out), "--rescore", str(old)]) == 0
    report = json.loads(out.read_text())
    assert report["totals"]["squeezed_errors"] == 0 and report["totals"]["raw_errors"] == 4 and report["rescored_from"]


def test_normalize_folds_kana_script_and_variants_only():
    assert normalize("ニ其前キ顚倒ノヿ祷１") == "に其前き顛倒のヿ祷1"
    assert normalize("多連里与") == "多連里与"          # 字母 of 変体仮名 stay as they are
    page = evaluate_page("松樹顛倒ノ由", ["松樹顚倒の由"])
    assert page["squeezed_errors"] == 2 and page["normalized_errors"] == 0
