import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from honkoku_ocr import cli
from honkoku_ocr import output as output_io
from honkoku_ocr.pipeline import OCR
from honkoku_ocr.recognizer import RecognitionResult


@pytest.fixture
def fake_models(monkeypatch):
    seen = []
    class Recognizer:
        def recognize_result(self, crop):
            return RecognitionResult('文', 'eos', 1, {})
    class Detector:
        def detect(self, image, *args):
            return []
    class FakeOCR(OCR):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, detector=Detector(), recognizer=Recognizer(), **kwargs)
        def process(self, image, *args, **kwargs):
            seen.append((Path(image).name, kwargs.get('frame', 0)))
            return super().process(image, *args, **kwargs)
    monkeypatch.setattr(cli, 'OCR', FakeOCR)
    monkeypatch.setattr(cli, '_model_identity', lambda *a: {'test': 'hash'})
    return seen


def test_output_names_stable_across_selections_and_order():
    files = [Path(p) for p in ('a/pages/p.jpg', 'b/pages/p.jpg', 'c/p.png', 'c/p__2.png', 'd/P.jpg')]
    names = cli._unique_names(files)
    assert len({name.casefold() for name in names.values()}) == len(files)
    assert names == cli._unique_names(list(reversed(files)))
    assert names[files[2]] == cli._unique_names([files[2]])[files[2]]


def test_cli_deduplicates_inputs_and_ignores_image_named_directories(tmp_path, fake_models):
    page = tmp_path / 'page.png'
    Image.new('RGB', (30, 30)).save(page)
    (tmp_path / 'directory.jpg').mkdir()
    output = tmp_path / 'out'
    assert cli.main([str(tmp_path), str(page), '-o', str(output)]) == 0
    assert fake_models == [('page.png', 0)]
    assert len(list(output.glob('*.json'))) == 1


def test_bad_input_continues_and_returns_nonzero(tmp_path, fake_models):
    page = tmp_path / 'page.png'
    Image.new('RGB', (30, 30)).save(page)
    output = tmp_path / 'out'
    assert cli.main([str(tmp_path / 'missing.png'), str(page), '-o', str(output)]) == 1
    assert fake_models == [('page.png', 0)]


def test_only_invalid_input_does_not_create_output(tmp_path, fake_models):
    output = tmp_path / 'out'
    assert cli.main([str(tmp_path / 'missing.png'), '-o', str(output)]) == 1
    assert not output.exists()


def test_resume_verifies_input_settings_and_artifacts(tmp_path, fake_models):
    page, output = tmp_path / 'page.png', tmp_path / 'out'
    Image.new('RGB', (30, 30)).save(page)
    args = [str(page), '-o', str(output), '--resume', '--preview']
    assert cli.main(args) == 0
    assert cli.main(args) == 0
    assert len(fake_models) == 1
    next(output.glob('*.txt')).write_text('corruption')
    assert cli.main(args) == 0
    assert len(fake_models) == 2
    assert cli.main(args + ['--plain']) == 0
    assert len(fake_models) == 3
    Image.new('RGB', (30, 30), 'red').save(page)
    assert cli.main(args) == 0
    assert len(fake_models) == 4


def test_failed_output_is_reprocessed_on_resume(tmp_path, fake_models, monkeypatch):
    page, output = tmp_path / 'page.png', tmp_path / 'out'
    Image.new('RGB', (30, 30)).save(page)
    original = output_io.atomic_write
    def fail_json(path, data):
        if path.suffix == '.json':
            raise OSError('simulated interruption')
        original(path, data)
    monkeypatch.setattr(output_io, 'atomic_write', fail_json)
    args = [str(page), '-o', str(output), '--resume']
    assert cli.main(args) == 1
    assert not list(output.glob('*.json'))
    monkeypatch.setattr(output_io, 'atomic_write', original)
    assert cli.main(args) == 0
    assert len(fake_models) == 2


def test_all_tiff_frames_are_processed(tmp_path, fake_models):
    page = tmp_path / 'pages.tiff'
    Image.new('RGB', (30, 30)).save(page, save_all=True, append_images=[Image.new('RGB', (40, 20))])
    output = tmp_path / 'out'
    assert cli.main([str(page), '-o', str(output)]) == 0
    assert fake_models == [('pages.tiff', 0), ('pages.tiff', 1)]
    records = [json.loads(p.read_text()) for p in sorted(output.glob('*.json'))]
    assert [(r['frame'], r['width'], r['height']) for r in records] == [(0, 30, 30), (1, 40, 20)]


def test_box_json_preserves_source_coordinates(tmp_path, fake_models):
    page, boxfile = tmp_path / 'page.png', tmp_path / 'boxes.json'
    Image.new('RGB', (100, 100)).save(page)
    boxfile.write_text(json.dumps([{'x': 3.5, 'y': 4, 'width': 10, 'height': 50}]))
    output = tmp_path / 'out'
    assert cli.main([str(page), '--boxes', str(boxfile), '-o', str(output)]) == 0
    line = json.loads(next(output.glob('*.json')).read_text())['lines'][0]
    assert line['x'] == 3.5 and line['raw'] == '文'


def test_atomic_write_cleans_failed_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / 'result.json'
    target.write_bytes(b'original')
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(output_io.os, 'replace', fail)
    with pytest.raises(OSError):
        output_io.atomic_write(target, b'new')
    assert target.read_bytes() == b'original'
    assert not list(tmp_path.glob('*.part'))


def test_resume_ignores_malformed_completion_records(tmp_path):
    path = tmp_path / 'result.json'
    for record in ([], {'schema_version': 1, 'fingerprint': {}, 'artifacts': []}):
        path.write_text(json.dumps(record))
        assert not cli._resume_matches(path, {}, set())


def test_later_inference_failure_does_not_stop_other_images(tmp_path, fake_models, monkeypatch):
    for name in ('a.png', 'b.png', 'c.png'):
        Image.new('RGB', (30, 30)).save(tmp_path / name)
    original = cli.OCR.process
    def fail_one(self, image, *args, **kwargs):
        if Path(image).name == 'b.png':
            raise RuntimeError('bad image data')
        return original(self, image, *args, **kwargs)
    monkeypatch.setattr(cli.OCR, 'process', fail_one)
    output = tmp_path / 'out'
    assert cli.main([str(tmp_path), '-o', str(output)]) == 1
    assert len(list(output.glob('*.json'))) == 2
    assert fake_models == [('a.png', 0), ('c.png', 0)]


def test_model_setup_failure_stops_cli_batch(tmp_path, fake_models, monkeypatch, capsys):
    from honkoku_ocr import ModelSetupError
    for name in ('a.png', 'b.png'):
        Image.new('RGB', (10, 10)).save(tmp_path / name)
    attempts = []
    def fail_setup(self, image, *args, **kwargs):
        attempts.append(Path(image).name)
        raise ModelSetupError('CUDA unavailable')
    monkeypatch.setattr(cli.OCR, 'process', fail_setup)
    assert cli.main([str(tmp_path), '-o', str(tmp_path / 'out')]) == 1
    assert attempts == ['a.png']
    assert 'batch stopped (1 not attempted)' in capsys.readouterr().err


def test_pdf_pages_become_numbered_outputs(tmp_path, fake_models):
    pytest.importorskip("pypdfium2")
    pages = [Image.new("RGB", (40, 60), (255, 255, 255)), Image.new("RGB", (40, 60), (255, 255, 255))]
    pages[0].save(tmp_path / "scans.pdf", save_all=True, append_images=pages[1:])
    output = tmp_path / "out"
    assert cli.main([str(tmp_path / "scans.pdf"), "-o", str(output)]) == 0
    assert sorted(p.name[-11:] for p in output.glob("*.json")) == ["_p0001.json", "_p0002.json"]


def test_iiif_manifest_pages_are_downloaded_and_processed(tmp_path, fake_models, monkeypatch):
    import json

    from honkoku_ocr import iiif
    page = Image.new("RGB", (40, 60), (255, 255, 255))
    buffer = io.BytesIO(); page.save(buffer, format="PNG")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sequences": [{"canvases": [
        {"label": str(i), "images": [{"resource": {"@id": f"https://example.org/{i}.png"}}]} for i in (1, 2, 3)]}]}))
    fetched = []
    def handler(request):
        fetched.append(request.url.path); return httpx.Response(200, content=buffer.getvalue())
    monkeypatch.setattr(iiif, "new_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    output = tmp_path / "out"
    assert cli.main(["--iiif", str(manifest), "--pages", "2-3", "-o", str(output)]) == 0
    assert fetched == ["/2.png", "/3.png"]
    assert sorted(p.name.split("__")[0] for p in output.glob("*.json")) == ["0002", "0003"]
    assert sorted(p.name for p in (output / "iiif").glob("*/*.jpg")) == ["0002.jpg", "0003.jpg"]
