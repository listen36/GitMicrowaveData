"""Check official clean-file copies against the documented input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify(generation: str, directory: Path, manifest: dict) -> list[str]:
    errors = []
    for item in manifest["clean_input_files"]:
        if item["generation"] != generation:
            continue
        path = directory / item["filename"]
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        if item["size_bytes"] is not None and path.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch: {path}")
        if digest(path) != item["sha256"]:
            errors.append(f"SHA-256 mismatch: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", choices=("gen1", "gen2"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "acquisition_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = verify(args.generation, args.directory, manifest)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print(f"PASS: {args.generation} clean files match documented SHA-256 values")


if __name__ == "__main__":
    main()
