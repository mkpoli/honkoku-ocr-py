import json

import numpy as np
import onnx
import pytest
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from honkoku_ocr import models


def write_fp16_model(path, scale=8):
    """float32 入力を fp16 に Cast して MatMul と Add を行い、float32 に戻す小さなグラフ。"""
    w = numpy_helper.from_array((np.arange(12, dtype=np.float16).reshape(4, 3) / scale), "w")
    bias = numpy_helper.from_array(np.array([0.5, -0.5, 1.0], np.float16), "bias_value")
    nodes = [
        helper.make_node("Cast", ["x"], ["x16"], to=TensorProto.FLOAT16),
        helper.make_node("MatMul", ["x16", "w"], ["h"]),
        helper.make_node("Constant", [], ["bias"], value=bias),
        helper.make_node("Add", ["h", "bias"], ["y16"]),
        helper.make_node("Cast", ["y16"], ["y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[w],
        value_info=[helper.make_tensor_value_info("h", TensorProto.FLOAT16, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def float16_left(model) -> list[str]:
    found = []
    g = model.graph
    found += [t.name for t in g.initializer if t.data_type == TensorProto.FLOAT16]
    found += [v.name for v in (*g.input, *g.output, *g.value_info) if v.type.tensor_type.elem_type == TensorProto.FLOAT16]
    for n in g.node:
        for a in n.attribute:
            if a.type == AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT16:
                found.append(f"{n.name}:{a.name}")
            if n.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                found.append(f"cast:{n.output[0]}")
    return found


@pytest.fixture
def src(tmp_path):
    return write_fp16_model(tmp_path / "kuzushiji-vtest-encoder-fp16.onnx")


def test_conversion_removes_every_float16_and_keeps_outputs(src):
    dst = models.encoder_path(src, "cpu", quiet=True)
    assert dst == src.with_name("kuzushiji-vtest-encoder-fp32.onnx")
    converted = onnx.load(str(dst))
    assert float16_left(onnx.load(str(src)))
    assert float16_left(converted) == []
    onnx.checker.check_model(converted)
    ort = pytest.importorskip("onnxruntime")
    x = np.array([[1.0, -2.0, 0.25, 3.0]], np.float32)
    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    np.testing.assert_allclose(a, b, atol=1e-2)


def test_precision_selection(src):
    assert models.encoder_path(src, "cuda", quiet=True) == src
    assert models.encoder_path(src, "cpu", precision="fp16", quiet=True) == src
    assert models.encoder_path(src, "cuda", precision="fp32", quiet=True).name.endswith("-fp32.onnx")
    with pytest.raises(ValueError):
        models.encoder_path(src, "cpu", precision="int8")


def test_custom_name_never_overwrites_source(tmp_path):
    src = write_fp16_model(tmp_path / "custom.onnx")
    dst = models.encoder_path(src, "cpu", quiet=True)
    assert dst == tmp_path / "custom-fp32.onnx"
    assert src.exists() and dst.exists() and dst != src


def test_provenance_guards_reuse(src):
    dst = models.encoder_path(src, "cpu", quiet=True)
    prov = json.loads(dst.with_name(dst.name + ".json").read_text())
    assert prov["converter"] == models.FP32_CONVERTER
    assert prov["source_sha256"] == models._sha256(src)
    assert prov["size"] == dst.stat().st_size and prov["sha256"] == models._sha256(dst)
    first = dst.stat().st_mtime_ns
    assert models.encoder_path(src, "cpu", digest=True, quiet=True) == dst
    assert dst.stat().st_mtime_ns == first
    # a corrupted derived file is detected by size, or by digest when the size still matches
    dst.write_bytes(dst.read_bytes() + b"x")
    models.encoder_path(src, "cpu", quiet=True)
    assert dst.stat().st_size == prov["size"]
    data = bytearray(dst.read_bytes()); data[-1] ^= 0xFF; dst.write_bytes(bytes(data))
    assert models._sha256(dst) != prov["sha256"]
    models.encoder_path(src, "cpu", digest=True, quiet=True)
    assert models._sha256(dst) == prov["sha256"]
    # a changed source invalidates the cache
    write_fp16_model(src, scale=4)
    models.encoder_path(src, "cpu", quiet=True)
    assert json.loads(dst.with_name(dst.name + ".json").read_text())["source_sha256"] == models._sha256(src)


def test_non_dict_provenance_and_same_size_source_change_reconvert(src):
    dst = models.encoder_path(src, "cpu", quiet=True)
    prov = dst.with_name(dst.name + ".json")
    prov.write_text("[]", encoding="utf-8")
    models.encoder_path(src, "cpu", quiet=True)
    assert json.loads(prov.read_text())["converter"] == models.FP32_CONVERTER
    # a source of the same size and name with different bytes must not reuse the derived file
    size = src.stat().st_size
    write_fp16_model(src, scale=2)
    assert src.stat().st_size == size
    models.encoder_path(src, "cpu", quiet=True)
    info = json.loads(prov.read_text())
    assert info["source_sha256"] == models._sha256(src) and info["sha256"] == models._sha256(dst)
