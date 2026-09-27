# Curated outputs

These files are the final corrected outputs used for the manuscript revision.

- `gen2/supplementary_results.json`: complete Gen2 metrics, group identities, test scores, fold results, ablations, permutation distribution, and physical time-axis metadata.
- `gen2/bootstrap_auc_values.npy`: 3,000 stratified bootstrap AUC values used for Fig. S1.
- `gen2/Table_S1_ablation_performance.csv`: original, late-time, and angular-shuffle performance.
- `gen2/Table_S2_label_permutation_test.csv`: observed versus within-phantom null AUC.
- `cross_generation/cross_generation_s11.json`: bidirectional S11-only transfer.
- `baselines/khalid_resnet_results.json`: authoritative 20-run GPU rerun for revised Table IV, selected by validation AUC only.
- `baselines/khalid_resnet_rerun_manifest.json`: exact script/data/split hashes, runtime versions, and selection audit for that rerun.
- `baselines/khalid_resnet_results_archived_20260926.json`: earlier archived run retained for comparison; **not** the source of revised Table IV.
- `baselines/khalid_resnet_results_rerun_20260927.json`: byte-identical copy of the authoritative rerun as received from the remote machine.
- `baselines/cnn_results.json`: separate exploratory CNN baseline.
- `figures/`: publication figures in raster and vector formats.
- `figures/Fig_1_experimental_design_corrected.jpg`: regenerated Fig. 1 with the L2 readout label and the published group identities.
- `CHECKSUMS.sha256`: integrity hashes for the curated result artifacts.

The old and new ResNet records are explicitly labeled to prevent conflating them. Large signal archives are omitted.
