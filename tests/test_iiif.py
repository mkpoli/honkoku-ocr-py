import httpx
import pytest

from honkoku_ocr import iiif

V2 = {"@context": "http://iiif.io/api/presentation/2/context.json", "sequences": [{"canvases": [
    {"@id": "c1", "label": "1", "width": 100, "height": 50, "images": [{"resource": {"@id": "https://x/1/full/full/0/default.jpg", "service": {"@id": "https://x/R1"}}}]},
    {"@id": "c2", "label": "表紙", "images": [{"resource": {"@id": "https://x/2.jpg"}}]}]}]}
V3 = {"@context": "http://iiif.io/api/presentation/3/context.json", "items": [
    {"id": "c1", "label": {"none": ["p1"]}, "width": 10, "height": 20, "items": [{"items": [{"body": {"type": "Image", "id": "https://y/1.jpg", "service": [{"id": "https://y/svc1"}]}}]}]},
    {"id": "c2", "label": {"ja": ["二"]}, "items": [{"items": [{"body": [{"type": "Image", "id": "https://y/2.jpg"}]}]}]}]}


def test_canvases_v2_and_v3():
    v2 = iiif.canvases(V2)
    assert [(c.index, c.label, c.image_url) for c in v2] == [(1, "1", "https://x/R1/full/full/0/default.jpg"), (2, "表紙", "https://x/2.jpg")]
    assert v2[0].width == 100
    v3 = iiif.canvases(V3)
    assert [(c.index, c.label, c.image_url) for c in v3] == [(1, "p1", "https://y/svc1/full/max/0/default.jpg"), (2, "二", "https://y/2.jpg")]


def test_image_paths_are_numbered_and_labelled(tmp_path):
    cs = iiif.canvases(V2)
    assert iiif.image_path(cs[0], tmp_path).name == "0001.jpg"          # label equal to the number is not repeated
    assert iiif.image_path(cs[1], tmp_path).name == "0002-表紙.jpg"


def test_parse_pages():
    assert iiif.parse_pages(None, 3) == [1, 2, 3]
    assert iiif.parse_pages("3", 5) == [3] and iiif.parse_pages("1,4-5", 5) == [1, 4, 5]
    with pytest.raises(ValueError):
        iiif.parse_pages("0-2", 5)
    with pytest.raises(ValueError):
        iiif.parse_pages("4-9", 5)


def test_download_retries_and_reuses_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(iiif, "_sleep", lambda s: None)
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, content=b"jpeg")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    canvas = iiif.canvases(V2)[1]
    path = iiif.download(canvas, tmp_path / "m", client)
    assert path.read_bytes() == b"jpeg" and len(calls) == 2
    assert iiif.download(canvas, tmp_path / "m", client) == path and len(calls) == 2
    with pytest.raises(httpx.HTTPStatusError):
        iiif.download(iiif.Canvas(3, "", "https://x/404.jpg"), tmp_path / "m", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))))


def test_manifest_directory_is_stable_and_redacted(tmp_path):
    a = iiif.manifest_directory("https://x/manifest.json", tmp_path)
    assert a == iiif.manifest_directory("https://x/manifest.json", tmp_path) and a.parent == tmp_path and len(a.name) == 16


def test_load_manifest_from_url_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(iiif, "new_client", lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=V3))))
    assert len(iiif.canvases(iiif.load_manifest("https://example.org/manifest.json"))) == 2
    path = tmp_path / "m.json"
    path.write_text(__import__("json").dumps(V2), encoding="utf-8")
    assert len(iiif.canvases(iiif.load_manifest(str(path)))) == 2
