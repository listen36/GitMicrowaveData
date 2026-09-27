"""Create a source-only supplementary ZIP from an explicit allowlist.

This is not a GitHub or Zenodo release. Publish a committed GitHub tag first,
then archive that tag through GitHub Release or Zenodo for an immutable DOI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    ".gitattributes", ".gitignore", "CITATION.cff", "CONTRIBUTING.md",
    "LICENSE", "NOTICE", "README.md", "pyproject.toml",
    "requirements.txt", "requirements-lock.txt",
}
DIRECTORIES = {".github", "configs", "docs", "results", "scripts", "src", "tests"}
BLOCKED_PARTS = {
    ".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv",
    "venv", "verify-venv", "release-verify-venv", "outputs", "build",
    "dist", "data", "node_modules",
}
BLOCKED_SUFFIXES = {".pickle", ".pkl", ".npz", ".pyc", ".log"}


def release_files(root: Path = ROOT) -> list[Path]:
    candidates = [root / name for name in TOP_LEVEL]
    for dirname in DIRECTORIES:
        candidates.extend((root / dirname).rglob("*"))
    files = []
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in BLOCKED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            continue
        files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="ZIP path outside the source tree")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT.resolve()):
        parser.error("The release ZIP must be written outside the source tree")
    files = release_files()
    if not files:
        parser.error("No source files found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Created {output} ({len(files)} files)")


if __name__ == "__main__":
    main()
