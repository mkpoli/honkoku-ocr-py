"""Download the reference corpus images named in benchmarks/corpus/manifest.json.

Each sample carries the IIIF URL of its page image and the SHA-256 of the file
the references were aligned with. Existing files whose digest matches are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(manifest: Path, *, client: httpx.Client | None = None) -> list[Path]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    client = client or httpx.Client(follow_redirects=True, timeout=120)
    written = []
    for sample in data["samples"]:
        target = manifest.parent / sample["image"]
        if target.exists() and sha256(target) == sample["image_sha256"]:
            continue
        response = client.get(sample["image_url"])
        response.raise_for_status()
        part = target.with_name(target.name + ".part")
        part.write_bytes(response.content)
        digest = sha256(part)
        if digest != sample["image_sha256"]:
            part.unlink()
            raise RuntimeError(f"{sample['id']}: downloaded image digest {digest[:12]} differs from the manifest; the source image may have changed")
        part.replace(target)
        written.append(target)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the reference corpus images listed in a manifest.")
    parser.add_argument("manifest", nargs="?", type=Path, default=Path(__file__).parent / "corpus" / "manifest.json")
    args = parser.parse_args(argv)
    for path in fetch(args.manifest):
        print(path.name, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
