# Direct Telemetry ML Training in Colab

The notebook:

1. Loads `ml_dataset.parquet`, which already contains one row per
   `scenario_id` + `target_id`.
2. Loads `position_estimates.parquet` separately for conventional baseline
   metrics only.
3. Splits train, validation, and test data by unseen `scenario_id` groups.
4. Dynamically keeps leakage-safe environment, antenna-layout, per-antenna, and
   per-pair observed telemetry feature columns for ML.
5. Benchmarks direct RSSI, TDOA, and DOA estimates.
6. Fits a primary tree-based direct ML model and a secondary MLP comparison
   model.
7. Exports `ml_predictions.parquet` with `scenario_id`, `target_id`,
   `ml_est_x`, and `ml_est_y`.
8. Optionally runs the existing evaluator script from the repo.

## Important Rules

- Full 100-scenario dataset must be use for the real experiment.
- Random-spliting rows across training and testing sets must be avoided.
- Avoid / block data which may squew unrealistic performance of the model.
- Conventional position estimates are baselines only, not ML inputs.

`ml_dataset.parquet` includes observed telemetry features such as
`rssi_antenna_0_signal_dbm`, `doa_antenna_0_bearing_sin`, and
`tdoa_ref_0_cmp_1_observed_ns`, plus explicit `*_present` indicators for
variable antenna layouts.

The notebook explicitly excludes:

- `target_*` input columns from `X`
- `scenario_id`, `target_id`, `target_label`, `seed`
- conventional estimate coordinates such as `rssi_est_x`, `tdoa_est_y`, and
  `doa_est_x`
- all `*_error_m` columns
- `link_*` geometry summaries
- columns containing `distance`, `attenuation`, `blocker`, `loss`, `noise`,
  `sigma`, `ideal`, `arrival_time`, `link_state`, or `path_loss`

## Colab Usage

Open `ProjectMLPipeline.ipynb` in Google Colab, update `DATA_PATH` so it points
to the expected `ml_dataset.parquet` output from the data-generation pipeline,
and keep `position_estimates.parquet` in the same directory for baseline
comparison.

The notebook writes `ml_predictions.parquet` into the same directory as
`ml_dataset.parquet`, so it stays compatible with `Data Generation/evaluate_positioning_performance.py` for plotting and evaluation stages.
