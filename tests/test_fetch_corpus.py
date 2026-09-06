import hashlib
import json

import httpx
import pytest

from benchmarks import fetch_corpus


def manifest_with(tmp_path, body):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attribution": "test", "samples": [
        {"id": "p1", "image": "p1.jpg", "image_url": "https://example.com/p1.jpg", "image_sha256": hashlib.sha256(body).hexdigest()},
    ]}), encoding="utf-8")
    return manifest


def client(body):
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))


def test_fetch_writes_verified_image_and_skips_existing(tmp_path):
    manifest = manifest_with(tmp_path, b"image bytes")
    assert [p.name for p in fetch_corpus.fetch(manifest, client=client(b"image bytes"))] == ["p1.jpg"]
    assert (tmp_path / "p1.jpg").read_bytes() == b"image bytes"
    assert fetch_corpus.fetch(manifest, client=client(b"other")) == []


def test_fetch_rejects_changed_source(tmp_path):
    manifest = manifest_with(tmp_path, b"image bytes")
    with pytest.raises(RuntimeError, match="digest"):
        fetch_corpus.fetch(manifest, client=client(b"changed"))
    assert not (tmp_path / "p1.jpg").exists() and not list(tmp_path.glob("*.part"))


def test_repository_manifest_is_well_formed():
    data = json.loads((fetch_corpus.Path(fetch_corpus.__file__).parent / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    assert data["attribution"] and len(data["samples"]) == 4
    for sample in data["samples"]:
        assert len(sample["image_sha256"]) == 64 and "dl.ndl.go.jp/" in sample["image_url"]
        assert len(sample["boxes"]) == len(sample["reference_lines"]) >= 10
        assert sample["reference"] == "\n".join(sample["reference_lines"])
