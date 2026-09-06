"""onnxruntime セッションの生成。"""
from __future__ import annotations


def _runtime():
    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        if exc.name != "onnxruntime":
            raise
        raise RuntimeError("Install the cpu or gpu extra to run inference") from exc
    return onnxruntime


def providers(device: str) -> list[str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device {device!r}; choose cpu or cuda")
    ort = _runtime()
    if device == "cuda":
        try:
            ort.preload_dlls()
        except Exception:
            pass
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available: install the gpu extra (onnxruntime-gpu with CUDA 12 / cuDNN 9) or use --device cpu")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def session(path, device: str = "cpu"):
    selected = providers(device)
    ort = _runtime()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=selected)
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA session could not be created for " + str(path))
    return sess
