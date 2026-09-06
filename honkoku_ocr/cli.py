"""honkoku-ocr: 画像を翻刻して Koji 記法のテキストと JSON を書き出す。"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
from . import models
from .pipeline import OCR

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def _unique_names(files: list[Path]) -> dict[Path, str]:
    """Allocate unique stems, including collisions with generated names."""
    files = list(dict.fromkeys(files))
    reserved = {f.stem.casefold() for f in files}
    used: set[str] = set()
    names: dict[Path, str] = {}
    for f in files:
        candidate = f.stem
        if candidate.casefold() in used:
            suffix = 2
            while True:
                candidate = f"{f.stem}__{suffix}"
                suffix += 1
                if candidate.casefold() not in used | reserved:
                    break
        used.add(candidate.casefold())
        names[f] = candidate
    return names

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="honkoku-ocr", description="みんなで翻刻OCR (Python 移植) で古典籍画像を翻刻する")
    ap.add_argument("inputs", nargs="*", help="画像ファイルまたはディレクトリ")
    ap.add_argument("-o", "--output", type=Path, default=Path("ocr-out"), help="出力先ディレクトリ (既定 ocr-out)")
    ap.add_argument("--version", default=models.DEFAULT_VERSION, choices=sorted(models.OCR_FILES), help="行認識モデルの版")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="encoder の実行デバイス")
    ap.add_argument("--plain", action="store_true", help="txt をタグ無しの素テキストで書く (既定は Koji 記法)")
    ap.add_argument("--download", action="store_true", help="モデルを取得して終了")
    a = ap.parse_args(argv)
    if a.download:
        for k, p in models.ensure(a.version).items():
            print(f"{k}: {p}")
        return 0
    files: list[Path] = []
    for s in a.inputs:
        p = Path(s)
        files += sorted(q for q in p.iterdir() if q.is_file() and q.suffix.lower() in EXTS) if p.is_dir() else [p]
    files = list(dict.fromkeys(p.resolve() for p in files))
    if not files:
        ap.error("画像を指定してください")
    a.output.mkdir(parents=True, exist_ok=True)
    names = _unique_names(files)
    ocr = OCR(a.version, a.device)
    for f in files:
        t0 = time.perf_counter()
        lines = ocr.run(f)
        stem = names[f]
        (a.output / f"{stem}.json").write_text(json.dumps({"image": f.name, "model": a.version, "lines": [asdict(l) for l in lines]}, ensure_ascii=False, indent=1), encoding="utf-8")
        (a.output / f"{stem}.txt").write_text("\n".join(l.plain if a.plain else l.koji for l in lines) + "\n", encoding="utf-8")
        print(f"{f.name}: {len(lines)} 行 {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
