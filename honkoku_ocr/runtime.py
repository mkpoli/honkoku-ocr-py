"""onnxruntime セッションの生成。"""
from __future__ import annotations
import onnxruntime as ort

def providers(device: str) -> list[str]:
    if device == "cuda":
        try:
            ort.preload_dlls()
        except Exception:
            pass
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available: install the gpu extra (onnxruntime-gpu with CUDA 12 / cuDNN 9) or use --device cpu")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def session(path, device: str = "cpu") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=providers(device))
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA session could not be created for " + str(path))
    return sess
