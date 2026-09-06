"""モデルファイルの取得とキャッシュ。

配信元は honkoku-ocr-web が利用する公開バケット。ファイルは初回のみ取得し
$HONKOKU_OCR_MODELS （既定 ~/.cache/honkoku-ocr/models）に置く。
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import httpx

MODEL_BASE_URL = os.environ.get("HONKOKU_OCR_MODEL_URL", "https://pub-1b00c465f60640a3bf9b7b7d329d06cc.r2.dev")
CONFIG_DIR = Path(__file__).parent / "config"

LAYOUT_FILES = {"rtmdet": "rtmdet-s-1280x1280.onnx"}

# 配信ファイルのサイズと SHA-256。取得後に照合し、キャッシュ読込時はサイズを確かめる。
EXPECTED = {
    "rtmdet-s-1280x1280.onnx": (40188733, "f46267754d406431f6e035f9e20b8552af8ff1ab5ca13bcac8f4b1abbd02090c"),
    "kuzushiji-v18-encoder-fp16.onnx": (183086925, "18425099ce3277d31526e133768b1fc0831d2356887567a62898e47b621375cc"),
    "kuzushiji-v18-decoder-prefill-int8.onnx": (34286083, "6f3f19011f8d08f9d5dc9cf9c2e672d9d68cb512438a90486e5b6d7267a129dc"),
    "kuzushiji-v18-decoder-step-int8.onnx": (31050707, "bf0e72a807168393acb7b6c0cb1b85b7d6107f2d6f4d6bb8fbb6736be2ac4bae"),
    "kuzushiji-v17-encoder-fp16.onnx": (183086925, None),
    "kuzushiji-v17-decoder-prefill-int8.onnx": (34286083, None),
    "kuzushiji-v17-decoder-step-int8.onnx": (31050707, None),
    "kuzushiji-v16fs-encoder-fp16.onnx": (183086925, None),
    "kuzushiji-v16fs-decoder-prefill-int8.onnx": (34286083, None),
    "kuzushiji-v16fs-decoder-step-int8.onnx": (31050707, None),
}

# onnxruntime (CPU/CUDA) は int8 encoder の ConvInteger を実行できないため fp16 encoder を用いる。
# v12/v13 は fp16 encoder が配布されていないので対象外。
OCR_FILES = {
    "v16fs": {"encoder": "kuzushiji-v16fs-encoder-fp16.onnx", "prefill": "kuzushiji-v16fs-decoder-prefill-int8.onnx", "step": "kuzushiji-v16fs-decoder-step-int8.onnx"},
    "v17":   {"encoder": "kuzushiji-v17-encoder-fp16.onnx",   "prefill": "kuzushiji-v17-decoder-prefill-int8.onnx",   "step": "kuzushiji-v17-decoder-step-int8.onnx"},
    "v18":   {"encoder": "kuzushiji-v18-encoder-fp16.onnx",   "prefill": "kuzushiji-v18-decoder-prefill-int8.onnx",   "step": "kuzushiji-v18-decoder-step-int8.onnx"},
}
DEFAULT_VERSION = "v18"
IMG_DIMS = {"v16fs": (256, 2048), "v17": (256, 2048), "v18": (256, 2048)}

def model_dir() -> Path:
    d = Path(os.environ.get("HONKOKU_OCR_MODELS", Path.home() / ".cache/honkoku-ocr/models"))
    d.mkdir(parents=True, exist_ok=True)
    return d

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def verify(path: Path, name: str, digest: bool) -> None:
    size, sha = EXPECTED.get(name, (None, None))
    if size is not None and path.stat().st_size != size:
        raise RuntimeError(f"{name}: size {path.stat().st_size} != expected {size}; delete {path} and retry")
    if digest and sha and _sha256(path) != sha:
        raise RuntimeError(f"{name}: SHA-256 mismatch; delete {path} and retry")

def fetch(name: str, quiet: bool = False) -> Path:
    dst = model_dir() / name
    if dst.exists() and dst.stat().st_size > 0:
        verify(dst, name, digest=False)
        return dst
    url = f"{MODEL_BASE_URL}/{name}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk); done += len(chunk)
                if not quiet and total:
                    print(f"\r{name}: {done * 100 // total:3d}%", end="", file=sys.stderr)
    if not quiet:
        print(file=sys.stderr)
    verify(tmp, name, digest=True)
    tmp.replace(dst)
    return dst

def vocab(version: str) -> list[str]:
    return json.loads((CONFIG_DIR / f"kuzushiji-vocab-{version}.json").read_text(encoding="utf-8"))

def ensure(version: str = DEFAULT_VERSION, layout: str = "rtmdet", quiet: bool = False) -> dict[str, Path]:
    """必要なモデルをすべて取得してパスを返す。"""
    if version not in OCR_FILES:
        raise ValueError(f"unsupported model version {version!r}; choose from {sorted(OCR_FILES)}")
    paths = {k: fetch(v, quiet) for k, v in OCR_FILES[version].items()}
    paths["layout"] = fetch(LAYOUT_FILES[layout], quiet)
    return paths
