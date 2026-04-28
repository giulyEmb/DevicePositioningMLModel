# ML Fusion Training in Colab

The notebook:

1. Loads `ml_dataset.parquet`, which already contains one row per
   `scenario_id` + `target_id`.
2. Keeps only inference-safe feature columns.
3. Splits train, validation, and test data by unseen `scenario_id` groups.
4. Benchmarks direct RSSI, TDOA, and DOA estimates.
5. Fits a simple linear fusion baseline.
6. Fits a primary tree-based fusion model and a secondary MLP comparison model.
7. Exports `ml_predictions.parquet` with `scenario_id`, `target_id`,
   `ml_est_x`, and `ml_est_y`.
8. Optionally runs the existing evaluator script from the repo.

## Important Rules

- Full 100-scenario dataset must be use for the real experiment.
- Random-spliting rows across training and testing sets must be avoided.
- Avoid / block data which may squew unrealistic performance of the model.

The notebook explicitly excludes:

- `target_*` input columns from `X`
- `scenario_id`, `target_id`, `target_label`, `seed`
- all `*_error_m` columns
- `link_*` geometry summaries
- columns containing `distance`, `attenuation`, `noise_sigma`, or `path_loss`

## Colab Usage

Open `ml_fusion_colab_pipeline.ipynb` in Google Colab, update `DATA_PATH` so it points to the expected `ml_dataset.parquet` (output from data generation pipelin) file, and run the cells from top to bottom.

The notebook writes `ml_predictions.parquet` into the same directory as
`ml_dataset.parquet`, so it stays compatible with `Data Generation/evaluate_positioning_performance.py` for plotting and evaluation stages.
