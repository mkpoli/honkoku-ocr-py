import subprocess
import sys
from types import SimpleNamespace
import pytest
from honkoku_ocr import runtime


def test_package_and_cli_help_without_optional_runtime():
    code = '''
import sys
sys.modules['onnxruntime'] = None
import honkoku_ocr
from honkoku_ocr.cli import main
main(['--help'])
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--download' in result.stdout


def test_missing_runtime_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    with pytest.raises(RuntimeError, match='cpu or gpu extra'):
        runtime.providers('cpu')


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match='unsupported device'):
        runtime.providers('cdua')


def test_cuda_session_must_actually_use_cuda(monkeypatch):
    class Options:
        pass
    fake = SimpleNamespace(
        preload_dlls=lambda: None,
        get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        SessionOptions=Options,
        InferenceSession=lambda *args, **kwargs: SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider']),
    )
    monkeypatch.setattr(runtime, '_runtime', lambda: fake)
    with pytest.raises(RuntimeError, match='CUDA session could not be created'):
        runtime.session('model.onnx', 'cuda')
