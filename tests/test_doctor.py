import json

from honkoku_ocr import cli, doctor, models


def test_report_describes_cache_without_touching_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HONKOKU_OCR_MODELS", str(tmp_path))
    spec = models.specification("v18").files
    (tmp_path / spec["layout"]).write_bytes(b"x" * 10)           # wrong size
    (tmp_path / spec["encoder"]).write_bytes(b"")
    fp32 = tmp_path / spec["encoder"].replace("-fp16", "-fp32")
    fp32.write_bytes(b"fp32")
    fp32.with_name(fp32.name + ".json").write_text(json.dumps({"converter": models.FP32_CONVERTER, "size": 4}))
    data = doctor.report("v18", {"device": "cpu"})
    assert data["cache"]["files"]["layout"] == {"file": spec["layout"], "present": True, "size": 10, "size_ok": False}
    assert data["cache"]["files"]["prefill"]["present"] is False
    assert data["cache"]["fp32_encoder"]["current_converter"] and data["cache"]["fp32_encoder"]["size_ok"]
    assert str(tmp_path) not in json.dumps(data) or not str(tmp_path).startswith("/home/")
    text = doctor.render(data)
    assert "size mismatch" in text and "missing" in text and "fp32" in text and "device=cpu" in text
    assert cli.main(["--doctor"]) == 0
    assert "onnxruntime" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([spec["layout"], spec["encoder"], fp32.name, fp32.name + ".json"])


def test_runtime_report_without_onnxruntime(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError(name="onnxruntime")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    data = doctor.runtime_report()
    assert data["onnxruntime"] is None and data["cuda"] is False and "not installed" in data["cuda_error"]
    assert "missing" in doctor.render(doctor.report("v18"))


def test_hint_appears_only_with_hyperthreading(monkeypatch):
    data = doctor.report("v18", {"device": "cpu"})
    data["package"]["physical_cores"], data["package"]["cpu_count"] = 8, 16
    assert "--threads 8 --decoder-threads 2" in doctor.render(data)
    data["package"]["physical_cores"] = 16
    assert "hint:" not in doctor.render(data)
    data["package"]["physical_cores"], data["settings"]["device"] = 8, "cuda"
    assert "hint:" not in doctor.render(data)


def test_runtime_report_preloads_dlls_only_with_the_gpu_package(monkeypatch):
    import onnxruntime
    calls = []
    monkeypatch.setattr(onnxruntime, "preload_dlls", lambda *a, **k: calls.append(1), raising=False)
    monkeypatch.setattr(doctor, "_version", lambda name: None)
    doctor.runtime_report()
    assert calls == []
    monkeypatch.setattr(doctor, "_version", lambda name: "1.29.0" if name == "onnxruntime-gpu" else None)
    doctor.runtime_report()
    assert calls == [1]
