import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import pytest

from honkoku_ocr import models


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv('HONKOKU_OCR_MODELS', str(tmp_path))
    monkeypatch.setitem(models.EXPECTED, 'test.onnx', (4, hashlib.sha256(b'good').hexdigest()))
    return tmp_path


def install_stream(monkeypatch, body=b'good', status=200, barrier=None):
    @contextmanager
    def stream(*args, **kwargs):
        if barrier:
            barrier.wait(timeout=5)
        yield httpx.Response(status, content=body, request=httpx.Request('GET', 'https://example.com'))
    monkeypatch.setattr(models.httpx, 'stream', stream)


def test_concurrent_downloads_use_independent_temporary_files(cache, monkeypatch):
    install_stream(monkeypatch, barrier=threading.Barrier(2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: models.fetch('test.onnx', quiet=True), range(2)))
    assert paths == [cache / 'test.onnx'] * 2
    assert paths[0].read_bytes() == b'good'
    assert list(cache.glob('*.part')) == []


@pytest.mark.parametrize('body,status,error', [
    (b'bad', 200, RuntimeError), (b'evil', 200, RuntimeError),
    (b'', 503, httpx.HTTPStatusError),
])
def test_failed_download_cleans_temporary_file(cache, monkeypatch, body, status, error):
    install_stream(monkeypatch, body, status)
    with pytest.raises(error):
        models.fetch('test.onnx', quiet=True)
    assert list(cache.iterdir()) == []
    install_stream(monkeypatch)
    assert models.fetch('test.onnx', quiet=True).read_bytes() == b'good'


def test_interrupted_stream_cleans_temporary_file(cache, monkeypatch):
    class BrokenResponse:
        headers = {}
        def raise_for_status(self):
            pass
        def iter_bytes(self, size):
            yield b'go'
            raise httpx.ReadError('interrupted')
    @contextmanager
    def stream(*args, **kwargs):
        yield BrokenResponse()
    monkeypatch.setattr(models.httpx, 'stream', stream)
    with pytest.raises(httpx.ReadError):
        models.fetch('test.onnx', quiet=True)
    assert list(cache.iterdir()) == []


def test_offline_missing_cache_never_contacts_network(cache, monkeypatch):
    monkeypatch.setattr(models.httpx, 'stream', lambda *a, **k: pytest.fail('network request'))
    with pytest.raises(FileNotFoundError, match='offline'):
        models.fetch('test.onnx', offline=True)


def test_role_validation_happens_before_download(monkeypatch):
    monkeypatch.setattr(models, 'fetch', lambda *a, **k: pytest.fail('download'))
    with pytest.raises(ValueError, match='unknown model roles'):
        models.ensure(roles=['bad'])
    assert models.ensure(roles=[]) == {}
