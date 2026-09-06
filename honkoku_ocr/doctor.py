"""環境とモデルキャッシュの状態を報告する（`honkoku-ocr --doctor`）。

何も取得せず、何も変えない。onnxruntimeの有無と実行プロバイダ、PDF対応、モデルファイルの
有無とサイズ照合、fp32変換キャッシュとその来歴、パッケージとランタイムの版、実効設定を返す。
"""
from __future__ import annotations

import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import models
from .output import package_version, safe_error


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def physical_cores() -> int | None:
    """Linuxのsysfsから物理コア数を数える。分からなければNone。"""
    try:
        cores = set()
        for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/core_id"):
            package = path.parent / "physical_package_id"
            cores.add((package.read_text().strip() if package.exists() else "0", path.read_text().strip()))
        return len(cores) or None
    except OSError:
        return None


def runtime_report() -> dict[str, Any]:
    report: dict[str, Any] = {"onnxruntime": None, "providers": [], "cuda": False, "cuda_error": None}
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        report["cuda_error"] = "onnxruntime is not installed: uv sync --extra cpu (or --extra gpu)"
        return report
    report["onnxruntime"] = ort.__version__
    if _version("onnxruntime-gpu"):        # the CPU package prints a CUDA warning from preload_dlls, so only the GPU package is preloaded
        try:
            ort.preload_dlls()
        except Exception:
            pass
    report["providers"] = list(ort.get_available_providers())
    report["cuda"] = "CUDAExecutionProvider" in report["providers"]
    if not report["cuda"]:
        report["cuda_error"] = "CUDAExecutionProvider not available (install the gpu extra with CUDA 12 / cuDNN 9)"
    return report


def cache_report(model_version: str) -> dict:
    directory = models.model_dir()
    files = {}
    for role, name in models.specification(model_version).files.items():
        path = directory / name
        entry = {"file": name, "present": path.exists()}
        if path.exists():
            size, _ = models.EXPECTED.get(name, (None, None))
            entry["size"] = path.stat().st_size
            entry["size_ok"] = size is None or path.stat().st_size == size
        files[role] = entry
    fp32 = models._fp32_name(directory / models.specification(model_version).files["encoder"])
    conversion = {"file": fp32.name, "present": fp32.exists()}
    if fp32.exists():
        provenance = models._provenance_path(fp32)
        conversion["provenance"] = provenance.exists()
        if provenance.exists():
            try:
                info = json.loads(provenance.read_text(encoding="utf-8"))
                conversion["converter"] = info.get("converter")
                conversion["current_converter"] = info.get("converter") == models.FP32_CONVERTER
                conversion["size_ok"] = info.get("size") == fp32.stat().st_size
            except (OSError, ValueError):
                conversion["provenance"] = False
    return {"directory": safe_error(directory), "files": files, "fp32_encoder": conversion}


def report(model_version: str = models.DEFAULT_VERSION, settings: dict | None = None) -> dict:
    return {
        "package": {"version": package_version(), "python": platform.python_version(), "platform": platform.platform(),
                    "cpu_count": os.cpu_count(), "physical_cores": physical_cores(),
                    "libraries": {name: _version(name) for name in ("numpy", "pillow", "onnx", "onnxruntime", "onnxruntime-gpu", "pypdfium2", "httpx")}},
        "runtime": runtime_report(),
        "pdf": _version("pypdfium2") is not None,
        "model": model_version,
        "cache": cache_report(model_version),
        "settings": settings or {},
        "environment": {name: safe_error(Path(value)) if name == "HONKOKU_OCR_MODELS" else value
                        for name, value in os.environ.items() if name.startswith("HONKOKU_OCR_")},
    }


def render(data: dict) -> str:
    lines = [f"honkoku-ocr-py {data['package']['version']} on Python {data['package']['python']}, {data['package']['platform']}, {data['package']['cpu_count']} CPUs"]
    libs = ", ".join(f"{k} {v}" for k, v in data["package"]["libraries"].items() if v)
    lines.append(f"libraries: {libs}")
    rt = data["runtime"]
    lines.append(f"onnxruntime: {rt['onnxruntime'] or 'missing'}; providers: {', '.join(rt['providers']) or 'none'}; CUDA: {'yes' if rt['cuda'] else 'no'}")
    if rt["cuda_error"]:
        lines.append(f"  {rt['cuda_error']}")
    lines.append(f"PDF input: {'yes' if data['pdf'] else 'no (uv sync --extra pdf)'}")
    cache = data["cache"]
    lines.append(f"model {data['model']} cache: {cache['directory']}")
    for role, entry in cache["files"].items():
        state = "missing" if not entry["present"] else ("ok" if entry["size_ok"] else f"size mismatch ({entry['size']} bytes)")
        lines.append(f"  {role:8s} {entry['file']}: {state}")
    conv = cache["fp32_encoder"]
    if conv["present"]:
        state = "ok" if conv.get("provenance") and conv.get("current_converter") and conv.get("size_ok") else "stale or unverified; it will be rebuilt"
        lines.append(f"  fp32     {conv['file']}: {state}")
    else:
        lines.append(f"  fp32     {conv['file']}: not built yet (built on first CPU run)")
    if data["settings"]:
        lines.append("settings: " + ", ".join(f"{k}={v}" for k, v in data["settings"].items()))
    cores = data["package"].get("physical_cores")
    if cores and data["package"]["cpu_count"] and cores < data["package"]["cpu_count"] and data["settings"].get("device", "cpu") == "cpu":
        lines.append(f"hint: {cores} physical cores and {data['package']['cpu_count']} logical CPUs; on such CPUs "
                     f"--threads {cores} --decoder-threads 2 ran faster than the runtime default in the recorded sweeps")
    for name, value in data["environment"].items():
        lines.append(f"{name}={value}")
    return "\n".join(lines)
