# Data preparation

## Source data

Obtain UM-BMID from IEEE DataPort: <https://doi.org/10.21227/1Y0Z-8T98>. This repository deliberately excludes raw signals, processed signal archives, copied virtual environments, and dataset-specific caches.

The published reproduction starts from the official `*_emp.pickle` **clean** files. It does not start from raw scanner files. The UM-BMID source archive available to the authors has SHA-256 `76d6499457d11b3027c142f318865a4f4969986888ca476d07b9b6653a6805ca`; its `run/make_clean_files.py` file has SHA-256 `189155ba0d17c5c35dad62109066f4eb4cc28357776ad02f07da6d3fa5a77499`. The upstream routine calls `umbmid.build.import_fd_cal_dataset(cal_type="emp", prune=True, gen=..., sparams=...)`, which subtracts the metadata-linked `emp_ref_id` empty-chamber reference in the complex frequency domain. The provided archive does not identify an immutable upstream Git commit or release tag, and the local copy may differ from the upstream repository; these hashes identify the examined files rather than proving upstream version identity. Prefer the official pre-cleaned files distributed with the dataset. Do not represent the builders below as raw-to-clean scripts.

The preprocessing scripts expect the official cleaned frequency-domain arrays and metadata lists. File names can vary between releases, so paths are supplied explicitly rather than hard-coded.

## Gen2 dual-port archive

Align S11 and S21 by the metadata `id` field:

```bash
python scripts/build_gen2_aligned.py \
  --s11-data /path/to/fd_data_s11_emp.pickle \
  --s11-metadata /path/to/md_list_s11_emp.pickle \
  --s21-data /path/to/fd_data_s21_emp.pickle \
  --s21-metadata /path/to/md_list_s21_emp.pickle \
  --output data/processed/gen2_s11_s21_aligned_dataset.npz \
  --require-published-geometry
```

The builder uses the physical `phant_id` as the leakage-control group, derives the Gen2 lesion label from positive `tum_diam` (Gen1 uses `tum_rad`), rejects mismatched port metadata, and does not invent random splits. It sorts numeric measurement IDs numerically to preserve the reported cohort order.

Expected arrays:

| Key | Meaning |
|---|---|
| `X11` | complex S11, shaped sample × frequency × angle |
| `X21` | complex S21 with the same shape/order as `X11` |
| `y` | binary lesion labels |
| `group_ids` | physical phantom identifiers |
| `sample_ids` | measurement identifiers used for port alignment |
| `tumor_params` | `tum_x`, `tum_y`, `tum_z`, and Gen2 `tum_diam` metadata |
| `frequency_hz`, `angle_deg` | acquisition axes assigned from the UM-BMID equipment/geometry documents (not extracted from scanner headers) |
| `calibration` | empty-chamber subtraction provenance |

For the manuscript cohort, validation should report 200 samples, 1,001 frequency samples, 72 angles, 100 measurements per class, and 20 phantom groups. If it does not, stop: a different dataset release or preprocessing selection is being used.

The geometry, calibration procedure, and SHA-256 hashes of the clean inputs used in this workspace are recorded in `configs/acquisition_manifest.json`. The hashes identify local copies of the official clean files; they do not establish an immutable upstream software release.

Before building the Gen2 archive, verify the local copies against the manifest with `python scripts/verify_clean_inputs.py --generation gen2 --directory /path/to/gen2/clean`. Run the analogous `--generation gen1` check if the historical Gen1 files are available. The Gen1 hashes were recorded previously, but file sizes and the exact DataPort download revision are not recoverable from this workspace.

## Fixed manuscript split

The group identities are stored in `configs/fixed_split_groups.json`. Generate the compact index file locally:

```bash
python scripts/create_fixed_split.py \
  --dataset data/processed/gen2_s11_s21_aligned_dataset.npz \
  --config configs/fixed_split_groups.json \
  --output data/processed/gen2_group_split_indices.npz
```

This yields 140 training samples from 14 groups, 20 validation samples from 2 groups, and 40 held-out test samples from 4 groups. The script refuses missing, extra, or overlapping group assignments.

## Gen1 S11 archive

```bash
python scripts/build_gen1_aligned.py \
  --data /path/to/gen1/fd_data_s11_emp.pickle \
  --metadata /path/to/gen1/md_list_s11_emp.pickle \
  --output data/processed/gen1_s11_aligned.npz \
  --require-published-geometry
```

The reported cross-generation run contained 249 Gen1 measurements from 13 physical phantom groups. Gen1 has no S21 channel in this evaluation.

## Integrity and privacy

UM-BMID is a controlled phantom dataset, not a patient cohort. Nevertheless, do not commit downloaded data unless its distribution terms explicitly permit it. The repository `.gitignore` blocks common raw and processed data formats by default.
