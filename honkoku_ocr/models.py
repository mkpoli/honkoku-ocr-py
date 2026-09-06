"""モデルファイルの取得とキャッシュ。

配信元は honkoku-ocr-web が利用する公開バケット。ファイルは初回のみ取得し
$HONKOKU_OCR_MODELS （既定 ~/.cache/honkoku-ocr/models）に置く。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

MODEL_BASE_URL = os.environ.get("HONKOKU_OCR_MODEL_URL", "https://pub-1b00c465f60640a3bf9b7b7d329d06cc.r2.dev")
CONFIG_DIR = Path(__file__).parent / "config"

# 配信ファイルのサイズと SHA-256。取得後に照合し、キャッシュ読込時はサイズを確かめる。
EXPECTED = {
    "rtmdet-s-1280x1280.onnx": (40188733, "f46267754d406431f6e035f9e20b8552af8ff1ab5ca13bcac8f4b1abbd02090c"),
    "kuzushiji-v18-encoder-fp16.onnx": (183086925, "18425099ce3277d31526e133768b1fc0831d2356887567a62898e47b621375cc"),
    "kuzushiji-v18-decoder-prefill-int8.onnx": (34286083, "6f3f19011f8d08f9d5dc9cf9c2e672d9d68cb512438a90486e5b6d7267a129dc"),
    "kuzushiji-v18-decoder-step-int8.onnx": (31050707, "bf0e72a807168393acb7b6c0cb1b85b7d6107f2d6f4d6bb8fbb6736be2ac4bae"),
    "kuzushiji-v17-encoder-fp16.onnx": (183086925, "d23c64c96337e9349e0fcc506425c9a0e3a62a48a94f036281b12912b140e742"),
    "kuzushiji-v17-decoder-prefill-int8.onnx": (34286083, "1ecf5d23b0e1298254e7c11086826df166882cf9806e9bf104fe3a6e2b4dcee9"),
    "kuzushiji-v17-decoder-step-int8.onnx": (31050707, "03056c3a109708dc16368256e9143f111bf1d32a8fa72146e033882dd1364bdc"),
    "kuzushiji-v16fs-encoder-fp16.onnx": (183086925, "8f1b31e170121227a7ab443f4ec168bb181e2a2016805ac9ac240008bfeabba0"),
    "kuzushiji-v16fs-decoder-prefill-int8.onnx": (34286083, "d6ef822579d363fb675bcc8d5bdd774305082a30e66ca18e6950a4e7707df4b6"),
    "kuzushiji-v16fs-decoder-step-int8.onnx": (31050707, "73017d9b3186020cfad1121fc44eb7ba101b7c3b3b9a076165edb6ea501bbe2e"),
}

@dataclass(frozen=True)
class ModelSpec:
    version: str
    image_height: int = 256
    image_width: int = 2048
    cache_tensors: int = 24
    max_tokens: int = 192

    @property
    def files(self) -> dict[str, str]:
        return {
            "layout": "rtmdet-s-1280x1280.onnx",
            "encoder": f"kuzushiji-{self.version}-encoder-fp16.onnx",
            "prefill": f"kuzushiji-{self.version}-decoder-prefill-int8.onnx",
            "step": f"kuzushiji-{self.version}-decoder-step-int8.onnx",
        }

    @property
    def vocabulary(self) -> Path:
        return CONFIG_DIR / f"kuzushiji-vocab-{self.version}.json"


DEFAULT_VERSION = "v18"
SPECS = {version: ModelSpec(version) for version in ("v16fs", "v17", "v18")}


def specification(version: str) -> ModelSpec:
    try:
        return SPECS[version]
    except KeyError:
        raise ValueError(f"unsupported model version {version!r}; choose from {sorted(SPECS)}") from None

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
        raise RuntimeError(f"{name}: size {path.stat().st_size} != expected {size}; remove the cached file and retry")
    if digest and sha and _sha256(path) != sha:
        raise RuntimeError(f"{name}: SHA-256 mismatch; remove the cached file and retry")

def fetch(name: str, quiet: bool = False, *, offline: bool = False, digest: bool = False) -> Path:
    dst = model_dir() / name
    if dst.exists() and dst.stat().st_size > 0:
        verify(dst, name, digest=digest)
        return dst
    if offline:
        raise FileNotFoundError(f"{name}: missing from model cache in offline mode")
    url = f"{MODEL_BASE_URL}/{name}"
    # Each caller owns its temporary file; only verified files enter the cache.
    fd, tmp_name = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=dst.parent)
    tmp = Path(tmp_name)
    try:
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                with open(fd if attempt == 0 else tmp, "wb") as f:
                    _download(url, name, f, quiet)
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                transient = status is None or status == 429 or status >= 500
                if not transient or attempt == len(RETRY_DELAYS):
                    raise
                if not quiet:
                    print(f"{name}: {type(error).__name__}, retrying in {RETRY_DELAYS[attempt]:.0f}s", file=sys.stderr)
                _sleep(RETRY_DELAYS[attempt])
        verify(tmp, name, digest=True)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


# 一時的な失敗（接続断、5xx、429）は間を置いて取り直す。4xxや照合の不一致は取り直さない。
RETRY_DELAYS = (1.0, 3.0)
_sleep = time.sleep


def _download(url: str, name: str, f, quiet: bool) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        for chunk in r.iter_bytes(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if not quiet and total:
                print(f"\r{name}: {done * 100 // total:3d}%", end="", file=sys.stderr)
    if not quiet:
        print(file=sys.stderr)

ENCODER_PRECISIONS = ("auto", "fp16", "fp32")
# fp16 → fp32 変換の版。変換手順を変えたらこの値を上げる（古い変換キャッシュが作り直される）。
FP32_CONVERTER = "fp16-to-fp32/1"


def encoder_path(path, device: str, *, precision: str = "auto", quiet: bool = False, digest: bool = False) -> Path:
    """encoder として読み込むファイルを返す。

    配信されている encoder は fp16。onnxruntime の CPU プロバイダでは、同じ重みを fp32 に
    直したファイルの方が 1 行あたり数倍速い（v18、16 スレッドで 7.3 秒 → 0.86 秒）ので、
    CPU では fp32 版を配信ファイルの隣にキャッシュして使い、CUDA では fp16 のまま使う。
    precision で auto / fp16 / fp32 を明示できる。

    fp32 版には来歴ファイル <name>.json（元ファイルの SHA-256、変換の版、生成物のサイズと SHA-256）が
    付く。再利用時は来歴とサイズを照合し、digest=True なら生成物の SHA-256 も照合する。
    合わなければ作り直す。
    """
    if precision not in ENCODER_PRECISIONS:
        raise ValueError(f"unsupported encoder precision {precision!r}; choose from {ENCODER_PRECISIONS}")
    if precision == "auto":
        precision = "fp32" if device == "cpu" else "fp16"
    path = Path(path)
    if precision == "fp16":
        return path
    dst = _fp32_name(path)
    source_sha = _source_sha256(path)
    if _fp32_cache_valid(dst, source_sha, digest):
        return dst
    _convert_to_fp32(path, dst, source_sha, quiet)
    return dst


def _fp32_name(path: Path) -> Path:
    if "-fp16" in path.name:
        return path.with_name(path.name.replace("-fp16", "-fp32"))
    return path.with_name(f"{path.stem}-fp32{path.suffix}")


def _source_sha256(path: Path) -> str:
    """配信ファイルの実際の内容の SHA-256。同じ大きさの別ファイルや破損を来歴で見分けるため、毎回計算する。"""
    return _sha256(path)


def _provenance_path(dst: Path) -> Path:
    return dst.with_name(dst.name + ".json")


def _fp32_cache_valid(dst: Path, source_sha: str, digest: bool) -> bool:
    prov = _provenance_path(dst)
    if not (dst.exists() and prov.exists()):
        return False
    try:
        info = json.loads(prov.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(info, dict):
        return False
    if info.get("converter") != FP32_CONVERTER or info.get("source_sha256") != source_sha:
        return False
    if info.get("size") != dst.stat().st_size:
        return False
    return not digest or info.get("sha256") == _sha256(dst)


def _convert_to_fp32(src: Path, dst: Path, source_sha: str, quiet: bool) -> None:
    """fp16 の ONNX グラフを fp32 に書き換えて dst に保存し、来歴ファイルを書く。書き込みは一時ファイル経由。"""
    import onnx

    if not quiet:
        print(f"{src.name}: converting to fp32 → {dst.name}", file=sys.stderr)
    model = onnx.load(str(src))
    _graph_to_fp32(model.graph)
    onnx.checker.check_model(model)
    prov = _provenance_path(dst)
    fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".part", dir=dst.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    fd, tmp_prov_name = tempfile.mkstemp(prefix=prov.name + ".", suffix=".part", dir=dst.parent)
    os.close(fd)
    tmp_prov = Path(tmp_prov_name)
    try:
        onnx.save(model, tmp_name)
        info = {
            "source": src.name,
            "source_size": src.stat().st_size,
            "source_sha256": source_sha,
            "converter": FP32_CONVERTER,
            "size": tmp.stat().st_size,
            "sha256": _sha256(tmp),
        }
        tmp_prov.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        # 来歴を先に消してから本体を置き、最後に来歴を置く。途中で止まっても古い来歴が新しい本体を指すことはない。
        prov.unlink(missing_ok=True)
        tmp.replace(dst)
        tmp_prov.replace(prov)
    finally:
        tmp.unlink(missing_ok=True)
        tmp_prov.unlink(missing_ok=True)


def _graph_to_fp32(graph) -> None:
    import numpy as np
    from onnx import AttributeProto, TensorProto, numpy_helper

    def tensor(t) -> None:
        if t.data_type == TensorProto.FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))

    def value(v) -> None:
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = TensorProto.FLOAT

    for t in graph.initializer:
        tensor(t)
    for v in (*graph.input, *graph.output, *graph.value_info):
        value(v)
    for node in graph.node:
        for a in node.attribute:
            if a.type == AttributeProto.TENSOR:
                tensor(a.t)
            elif a.type == AttributeProto.TENSORS:
                for t in a.tensors:
                    tensor(t)
            elif a.type == AttributeProto.GRAPH:
                _graph_to_fp32(a.g)
            elif a.type == AttributeProto.GRAPHS:
                for g in a.graphs:
                    _graph_to_fp32(g)
            elif node.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT


def vocab(version: str) -> list[str]:
    return json.loads(specification(version).vocabulary.read_text(encoding="utf-8"))


def ensure(version: str = DEFAULT_VERSION, *, roles=None, quiet: bool = False,
           offline: bool = False, digest: bool = False) -> dict[str, Path]:
    """Resolve only the requested components; offline mode never downloads."""
    files = specification(version).files
    roles = tuple(files) if roles is None else tuple(roles)
    unknown = set(roles) - files.keys()
    if unknown:
        raise ValueError(f"unknown model roles: {sorted(unknown)}")
    return {role: fetch(files[role], quiet, offline=offline, digest=digest) for role in roles}
