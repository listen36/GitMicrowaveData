"""Prevent runtime environments and repository internals entering releases."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_release_allowlist_excludes_runtime_and_git() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "package_release.py"
    spec = importlib.util.spec_from_file_location("package_release", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [path.relative_to(module.ROOT).as_posix() for path in module.release_files()]
    assert "README.md" in names
    assert "scripts/package_release.py" in names
    assert all(not name.startswith((".git/", ".pytest_cache/", "outputs/")) for name in names)
    assert all("verify-venv" not in name for name in names)
    assert all(not name.endswith((".pickle", ".pkl", ".npz")) for name in names)
