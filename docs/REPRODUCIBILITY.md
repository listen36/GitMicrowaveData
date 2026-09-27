# Reproducibility record

## Canonical experiment identities

- **Fixed Gen2 holdout:** 14 training groups, 2 validation groups, and 4 untouched test groups. Hyperparameters and the operating threshold are selected on the validation partition.
- **Five-fold grouped evaluation:** a separate `GroupKFold` analysis over all 20 Gen2 groups. Every outer fold performs group-aware model selection and threshold calibration inside its development portion. These folds are not repeated measurements of the fixed holdout.
- **Signal-destruction ablations:** the fixed split is retained while the time gate or angular ordering is changed.
- **Permutation test:** labels are shuffled within physical phantom groups. Each repeat re-runs the same regularization search used for the observed-label model.
- **Cross-hardware validation:** S11-only because Gen1 provides S11 in the aligned archive used here. Source-generation grouped out-of-fold predictions select regularization and threshold before untouched target-generation evaluation.

## Determinism

The reported seed is `20260910`. Scikit-learn estimators receive this seed where supported. Bootstrap and permutation random number generators use NumPy's `default_rng` with the same seed. CPU/GPU deep-learning baselines may still vary slightly across platforms.

## Main Gen2 command

```bash
python scripts/run_gen2_supplementary.py \
  --dataset data/processed/gen2_s11_s21_aligned_dataset.npz \
  --split data/processed/gen2_group_split_indices.npz \
  --output outputs/gen2 --seed 20260910 --bootstrap 3000 --permutations 200
```

Use `--quick` only for a smoke test. It reduces search grids and resample counts and must not be used for a reported result.

## S11-only cross-generation command

```bash
python scripts/run_cross_generation_s11.py \
  --gen1 data/processed/gen1_s11_aligned.npz \
  --gen2 data/processed/gen2_s11_s21_aligned_dataset.npz \
  --gen1-band-ghz 1 8 \
  --gen2-band-ghz 1 8 \
  --output outputs/cross_generation_s11.json \
  --seed 20260910
```

The code reads `frequency_hz`, `angle_deg`, and calibration fields from both aligned NPZ archives, checks their shapes, interpolates over the common measured frequency band, and uses Fourier-domain circular resampling by angle index. Frequency-band arguments must reflect the actual acquisition metadata; do not assume `1 8` for another release or acquisition. The native 27.5° Gen1/Gen2 antenna-origin difference was not physically corrected in the reported analysis. Therefore the result supports only an observed acquisition-domain shift, not attribution to a specific hardware cause. The original published run predates the explicit-axis input checks, but for its documented 1–8 GHz, 1,001-point axes the numerical frequency grid is unchanged.

## Threshold warning

The fixed test sensitivity of 1.000 and specificity of 0.350 use a threshold selected on only two validation phantom groups. Across grouped folds, specificity spans 0.05–1.00. These values demonstrate an unresolved calibration problem and must not be presented as a deployment-ready operating point.

## Experimental dual-port transfer script

`scripts/experimental/run_cross_generation_dual_port.py` is retained for code completeness, but it is not the source of the reported cross-hardware results and requires both generations to contain aligned S11 and S21 arrays. The manuscript's reported cross-hardware table comes from `run_cross_generation_s11.py`.

## Residual-network model selection

The four prespecified configurations are evaluated with five seeds each. The selected run maximizes validation ROC AUC; equal values to 12 decimal places retain the first run in the prespecified loop order. Test ROC AUC is never used to break a tie. A fresh 20-run GPU training was completed on 2026-09-27 with the same Gen2 aligned NPZ, fixed group split, seed range, 300-epoch limit, and patience 40. The newly selected run has validation AUC 0.930 and held-out test AUC 0.7575. Its test sensitivity, specificity, balanced accuracy, and F1 are 0.750, 0.250, 0.500, and 0.600. Across all 20 runs, mean test AUC is 0.61375 with population standard deviation 0.06555 and range 0.5175–0.7575. These replace the earlier archived Table IV ResNet values. The older 20-run result is retained under an explicit archival filename for audit, not cited as the revised result. Runtime versions, GPU, and SHA-256 hashes are in `results/baselines/khalid_resnet_rerun_manifest.json`; the full rerun predictions remain in `results/baselines/khalid_resnet_results.json`.
