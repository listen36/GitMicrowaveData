# Release preparation record (not yet a published release)

Package version: `1.0.0` in `pyproject.toml` and `CITATION.cff`.

This source package is **not** a GitHub Release or immutable Zenodo record yet.
No publication date, repository URL, release-tag commit SHA, or software DOI
has been assigned. Do not cite the default branch as an immutable version.

## Verification environment

- Windows x86-64, CPython 3.13.5.
- Exact CPU/test dependency versions: `requirements-lock.txt`.
- PyTorch is optional and not included in this CPU verification lock.
- The original GPU training environment used for the archived ResNet result
  was not captured; this verification does not assert bitwise GPU reproduction.

## Checks completed on 2026-09-27

- A newly created virtual environment successfully installed the package with
  `python -m pip install -e ".[dev]"`.
- `python -m pytest`: **9 passed**.
- `python scripts/smoke_test.py`: **PASS** with synthetic data only.
- Gen2 S11/S21 clean signal and metadata files: sizes and SHA-256 values
  checked against `configs/acquisition_manifest.json`.
- The archived 20-run ResNet record was reselected by validation AUC and the
  first-prespecified-run tie rule; selected test AUC remains 0.8025. No fresh
  GPU training had been performed at this intermediate stage.
- Later on 2026-09-27, a fresh 20-run GPU training was completed on the
  reactivated remote instance using the corrected validation-only rule.
  The selected validation AUC is 0.930 and test AUC is 0.7575. Revised
  Table IV uses this new run. Runtime, data, split, script, and output
  hashes are recorded in `results/baselines/khalid_resnet_rerun_manifest.json`.
  The older 0.8025 result is retained as a labeled archive.

## Pre-publication checklist

1. Run `python -m pip install -e ".[dev]"`, `python -m pytest`, and
   `python scripts/smoke_test.py` in a fresh environment. Record outcomes here.
2. Verify the six clean-input checksums against the exact IEEE DataPort files.
   Gen1 file sizes and upstream DataPort revision are currently unverified.
3. Commit all publication files, tag that commit, and record the passing
   **commit SHA** and tag below. Never invent a SHA from an uncommitted tree.
4. Publish the clean committed tag to GitHub; create a GitHub Release and
   archive that release with Zenodo to obtain a version-specific DOI.
5. Update `CITATION.cff` with the actual release date, version-specific DOI,
   and verified author details. Update the manuscript Code Availability to
   point to that same immutable release and DOI. Re-run release checks.
6. Create the supplementary ZIP using `scripts/package_release.py`; inspect
   its contents before uploading. Do not upload `.git` or any copied venv.

Passing release-tag commit SHA: **pending** (repository has no commit).

GitHub tag / Release URL: **pending**.

Zenodo version DOI: **pending**.

Actual release date and confirmed author metadata: **pending**.
