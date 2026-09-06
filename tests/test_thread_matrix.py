import json

import pytest

from benchmarks import thread_matrix


def test_matrix_report_records_command_and_cells(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attribution": "t", "samples": [{"id": "p1", "image": "p1.jpg", "boxes": []}]}))
    (tmp_path / "p1.jpg").write_bytes(b"jpeg")
    calls = []
    def fake_cell(manifest, sample, **kw):
        calls.append(kw)
        return {"threads": kw["threads"], "decoder_threads": kw["decoder_threads"], "overlap": kw["overlap"], "median_seconds": 1.0}
    monkeypatch.setattr(thread_matrix, "run_cell", fake_cell)
    out = tmp_path / "out.json"
    assert thread_matrix.main([str(manifest), "--sample", "p1", "--output", str(out), "--threads", "0", "4", "--decoder-threads", "2", "--overlap"]) == 0
    report = json.loads(out.read_text())
    assert [(c["threads"], c["decoder_threads"], c["overlap"]) for c in report["cells"]] == [(0, 2, False), (0, 2, True), (4, 2, False), (4, 2, True)]
    assert report["boxes_supplied"] and report["machine"]["cpu_count"] and "runtime_versions" in report["machine"]
    assert report["arguments"]["threads"] == [0, 4] and report["arguments"]["decoder_threads"] == [2]
    assert "/home/" not in json.dumps(report) and len(report["image_sha256"]) == 64
    assert all(kw["warmups"] == 1 and kw["repeats"] == 3 for kw in calls)


def test_matrix_redacts_home_paths_in_image_and_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    image = tmp_path / "scan.jpg"
    image.write_bytes(b"jpeg")
    manifest.write_text(json.dumps({"attribution": "t", "samples": [{"id": "p1", "image": str(image)}]}))
    monkeypatch.setattr(thread_matrix, "run_cell", lambda *a, **kw: {"median_seconds": 1.0})
    monkeypatch.setattr(thread_matrix, "safe_error", lambda value: str(value).replace(str(tmp_path), "~"))
    out = tmp_path / "out.json"
    thread_matrix.main([str(manifest), "--sample", "p1", "--output", str(out), "--threads", "0", "--decoder-threads", "0"])
    report = json.loads(out.read_text())
    assert str(tmp_path) not in report["image"] and str(tmp_path) not in report["arguments"]["manifest"]


def test_matrix_rejects_bad_counts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attribution": "t", "samples": [{"id": "p1", "image": "p1.jpg"}]}))
    for extra in (["--repeats", "0"], ["--warmups", "-1"], ["--threads", "-2"]):
        with pytest.raises(SystemExit):
            thread_matrix.main([str(manifest), "--sample", "p1", "--output", str(tmp_path / "o.json"), *extra])


def test_matrix_rejects_unknown_sample(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attribution": "t", "samples": [{"id": "p1", "image": "p1.jpg"}]}))
    with pytest.raises(SystemExit):
        thread_matrix.main([str(manifest), "--sample", "nope", "--output", str(tmp_path / "o.json")])
