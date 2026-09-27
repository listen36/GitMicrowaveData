"""Regression check for the Gen2 metadata field and numeric ID alignment."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_gen2_builder_reads_tum_diam_and_sorts_ids_numerically(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_gen2_aligned.py"
    ids = ("10", "2", "1")
    metadata = [
        {"id": sample_id, "phant_id": "A1F1", "tum_diam": 2.0 if sample_id == "1" else np.nan,
         "tum_x": 1.0, "tum_y": 2.0, "tum_z": -3.0}
        for sample_id in ids
    ]
    paths = {}
    for port in ("s11", "s21"):
        signal_path = tmp_path / f"{port}_data.pickle"
        metadata_path = tmp_path / f"{port}_metadata.pickle"
        signals = np.zeros((3, 80, 72), dtype=np.complex64)
        for row, sample_id in enumerate(ids):
            signals[row] = int(sample_id) + (100 if port == "s21" else 0)
        signal_path.write_bytes(pickle.dumps(signals))
        metadata_path.write_bytes(pickle.dumps(metadata))
        paths[port] = (signal_path, metadata_path)
    output = tmp_path / "aligned.npz"
    subprocess.run(
        [sys.executable, str(script), "--s11-data", str(paths["s11"][0]),
         "--s11-metadata", str(paths["s11"][1]), "--s21-data", str(paths["s21"][0]),
         "--s21-metadata", str(paths["s21"][1]), "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["sample_ids"].tolist() == ["1", "2", "10"]
        assert archive["y"].tolist() == [1, 0, 0]
        assert archive["X11"][:, 0, 0].real.tolist() == [1, 2, 10]
        assert archive["X21"][:, 0, 0].real.tolist() == [101, 102, 110]
        assert archive["frequency_hz"][0] == 1e9
        assert archive["angle_deg"][-1] == 355

