"""Opt-in checks against the real v18 encoder in the local model cache.

Run with HONKOKU_OCR_REAL_MODELS=1; skipped otherwise. Downloads nothing: the
fp16 encoder must already be cached (honkoku-ocr --download).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from honkoku_ocr import models

pytestmark = pytest.mark.skipif(os.environ.get("HONKOKU_OCR_REAL_MODELS") != "1",
                                reason="set HONKOKU_OCR_REAL_MODELS=1 to run against cached models")


@pytest.fixture(scope="module")
def encoder_fp16() -> Path:
    path = models.model_dir() / models.specification("v18").files["encoder"]
    if not path.exists():
        pytest.skip("v18 encoder is not in the model cache")
    return path


def test_fp32_conversion_of_the_real_encoder_matches_fp16(encoder_fp16, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    # Convert afresh into tmp_path so the current converter code runs, not a cached product.
    source = tmp_path / encoder_fp16.name
    source.symlink_to(encoder_fp16)
    fp32 = models.encoder_path(source, "cpu", precision="fp32", quiet=True, digest=True)
    assert fp32.parent == tmp_path and fp32.stat().st_size > encoder_fp16.stat().st_size * 1.9
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    rng = np.random.default_rng(0)
    pixels = rng.standard_normal((1, 3, 256, 2048), dtype=np.float32)
    a = ort.InferenceSession(str(encoder_fp16), options, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(fp32), options, providers=["CPUExecutionProvider"])
    ha = a.run(None, {a.get_inputs()[0].name: pixels})[0]
    hb = b.run(None, {b.get_inputs()[0].name: pixels})[0]
    assert ha.shape == hb.shape == (1, 512, 512)
    assert float(np.abs(ha.astype(np.float32) - hb).max()) < 0.05
