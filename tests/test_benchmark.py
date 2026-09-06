import json

import pytest

from benchmarks.ocr import edit_distance, load_manifest


@pytest.mark.parametrize('reference,hypothesis,distance', [
    ('', '', 0), ('字', '', 1), ('', '字', 1), ('本文', '本女', 1),
    ('𠮷野\n文', '𠮷野文', 1), ('abc', 'yabd', 2),
])
def test_character_distance(reference, hypothesis, distance):
    assert edit_distance(reference, hypothesis) == distance


def test_manifest_requires_attribution(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'samples': []}))
    with pytest.raises(ValueError, match='attribution'):
        load_manifest(path)


def test_manifest_benchmark_writes_timings_and_cer(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from PIL import Image

    from benchmarks import ocr as benchmark
    from honkoku_ocr.pipeline import LineResult, PageResult

    Image.new('RGB', (10, 10)).save(tmp_path / 'page.png')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'attribution': 'synthetic test', 'samples': [
        {'id': 'page', 'image': 'page.png', 'reference': '本文'}]}))
    class FakeOCR:
        settings = {'device': 'cpu'}
        _paths = {}
        def __init__(self, *a, **k):
            pass
        def process(self, *a, **k):
            line = LineResult(1, 0, 0, 10, 10, 1, '本女', '本女', '本女')
            return PageResult(1, 10, 10, 10, 10, 0, 'v18', self.settings, [line], {'total': 0.1}, [])
    monkeypatch.setattr(benchmark, 'OCR', FakeOCR)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(__version__='test'))
    output = tmp_path / 'results.json'
    assert benchmark.main([str(manifest), '--output', str(output), '--device', 'cpu', '--repeats', '2']) == 0
    report = json.loads(output.read_text())
    assert report['aggregate_cer'] == 0.5
    assert report['reference_characters_across_repeats'] == 4
    assert report['character_errors_across_repeats'] == 2
    assert report['samples'][0]['repeat_outputs_identical']
