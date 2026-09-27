"""Regression check against selecting a model using held-out test scores."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_archived_resnet_selection_is_validation_only() -> None:
    script = ROOT / "scripts" / "khalid_resnet_baseline.py"
    spec = importlib.util.spec_from_file_location("khalid_resnet_baseline", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads((ROOT / "results" / "baselines" / "khalid_resnet_results.json").read_text())
    runs = payload["runs"]
    selected = max(runs, key=module.validation_rank)  # Python max keeps first equal run.
    assert selected["seed"] == payload["selected_by_validation"]["seed"]
    assert selected["config"] == payload["selected_by_validation"]["config"]
    assert selected["test"]["roc_auc"] == payload["selected_by_validation"]["test"]["roc_auc"]
    assert selected["test"]["roc_auc"] == 0.7575
    archived = json.loads(
        (ROOT / "results" / "baselines" / "khalid_resnet_results_archived_20260926.json").read_text()
    )
    assert archived["selected_by_validation"]["test"]["roc_auc"] == 0.8025


def test_test_auc_cannot_break_validation_tie() -> None:
    script = ROOT / "scripts" / "khalid_resnet_baseline.py"
    spec = importlib.util.spec_from_file_location("khalid_resnet_baseline", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = [
        {"validation_auc": 0.92, "test": {"roc_auc": 0.50}},
        {"validation_auc": 0.92, "test": {"roc_auc": 0.99}},
    ]
    assert max(runs, key=module.validation_rank) is runs[0]
