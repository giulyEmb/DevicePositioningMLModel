from __future__ import annotations
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_MODULE_PATH = PROJECT_ROOT / "Data Generation" / "create_ml_dataset.py"
SPEC = importlib.util.spec_from_file_location("create_ml_dataset", ML_MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {ML_MODULE_PATH}")
CREATE_ML_DATASET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CREATE_ML_DATASET
SPEC.loader.exec_module(CREATE_ML_DATASET)


class LeakageSafeTelemetryFeatureTests(unittest.TestCase):
    def test_rssi_features_include_per_antenna_observed_signal(self) -> None:
        rssi_df = pd.DataFrame(
            [
                {
                    "scenario_id": "s1",
                    "target_id": 1,
                    "antenna_id": 0,
                    "signal_strength_dbm": -40.0,
                    "path_loss_db_with_noise": 70.0,
                },
                {
                    "scenario_id": "s1",
                    "target_id": 1,
                    "antenna_id": 1,
                    "signal_strength_dbm": -50.0,
                    "path_loss_db_with_noise": 80.0,
                },
            ]
        )

        features_df = CREATE_ML_DATASET._rssi_features(rssi_df)
        row = features_df.iloc[0]

        self.assertAlmostEqual(row["rssi_signal_mean_dbm"], -45.0)
        self.assertEqual(row["rssi_antenna_0_present"], 1)
        self.assertEqual(row["rssi_antenna_1_present"], 1)
        self.assertAlmostEqual(row["rssi_antenna_0_signal_dbm"], -40.0)
        self.assertAlmostEqual(row["rssi_antenna_1_signal_dbm"], -50.0)
        self.assertNotIn("rssi_path_loss_mean_db", features_df.columns)

    def test_build_ml_dataset_keeps_wide_observed_telemetry_without_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._write_smoke_tables(data_dir)

            dataset_df = CREATE_ML_DATASET.build_ml_dataset(data_dir)

        expected_columns = {
            "antenna_0_present",
            "antenna_0_x",
            "antenna_0_y",
            "antenna_0_coverage_radius_m",
            "antenna_1_present",
            "rssi_antenna_0_present",
            "rssi_antenna_0_signal_dbm",
            "doa_antenna_1_present",
            "doa_antenna_1_bearing_sin",
            "doa_antenna_1_bearing_cos",
            "doa_antenna_1_doa_sin",
            "doa_antenna_1_doa_cos",
            "tdoa_ref_0_cmp_1_present",
            "tdoa_ref_0_cmp_1_observed_ns",
        }
        self.assertTrue(expected_columns.issubset(dataset_df.columns))

        forbidden_columns = [
            column
            for column in dataset_df.columns
            if CREATE_ML_DATASET._is_forbidden_ml_feature(column)
        ]
        self.assertEqual(forbidden_columns, [])
        self.assertEqual(len(dataset_df), 2)
        self.assertEqual(
            dataset_df.sort_values("target_id")[["target_x", "target_y"]].to_dict("records"),
            [
                {"target_x": 2.0, "target_y": 3.0},
                {"target_x": 8.0, "target_y": 6.0},
            ],
        )
        for estimate_column in [
            "rssi_est_x",
            "rssi_est_y",
            "tdoa_est_x",
            "tdoa_est_y",
            "doa_est_x",
            "doa_est_y",
        ]:
            self.assertNotIn(estimate_column, dataset_df.columns)

    def test_forbidden_ml_feature_detection_catches_leakage_columns(self) -> None:
        forbidden_columns = [
            "rssi_est_x",
            "tdoa_est_y",
            "doa_error_m",
            "rssi_residual_rmse_m",
            "tdoa_anchor_count",
            "reference_distance_m",
            "path_loss_db_with_noise",
            "wall_attenuation_db",
            "tdoa_noise_sigma_ns",
            "ideal_tdoa_ns",
            "true_bearing_rad",
        ]
        for column in forbidden_columns:
            with self.subTest(column=column):
                self.assertTrue(CREATE_ML_DATASET._is_forbidden_ml_feature(column))

        for column in ["target_x", "target_y", "rssi_signal_mean_dbm"]:
            with self.subTest(column=column):
                self.assertFalse(CREATE_ML_DATASET._is_forbidden_ml_feature(column))

    def _write_smoke_tables(self, data_dir: Path) -> None:
        scenario_id = "s1"
        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "target_count": 2,
                    "antenna_count": 2,
                    "area": 100.0,
                    "env_type": "indoor",
                    "width": 10.0,
                    "height": 10.0,
                    "x_domain_min": 0.0,
                    "x_domain_max": 10.0,
                    "y_range_min": 0.0,
                    "y_range_max": 10.0,
                    "human_count": 1,
                    "floor_plan_room_count": 1,
                    "floor_plan_patio_count": 0,
                    "floor_plan_element_count": 3,
                }
            ]
        ).to_parquet(data_dir / "env_summary.parquet", index=False)

        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "antenna_id": 0,
                    "x": 0.0,
                    "y": 0.0,
                    "coverage_radius": 30.0,
                },
                {
                    "scenario_id": scenario_id,
                    "antenna_id": 1,
                    "x": 10.0,
                    "y": 0.0,
                    "coverage_radius": 30.0,
                },
            ]
        ).to_parquet(data_dir / "antennas.parquet", index=False)

        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "target_id": 0,
                    "target_label": "target_00000",
                    "position": [2.0, 3.0],
                },
                {
                    "scenario_id": scenario_id,
                    "target_id": 1,
                    "target_label": "target_00001",
                    "position": [8.0, 6.0],
                },
            ]
        ).to_parquet(data_dir / "targets.parquet", index=False)

        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "target_id": target_id,
                    "target_x": 999.0,
                    "target_y": 999.0,
                    "rssi_est_x": 1.0,
                    "rssi_est_y": 1.0,
                    "tdoa_est_x": 2.0,
                    "tdoa_est_y": 2.0,
                    "doa_est_x": 3.0,
                    "doa_est_y": 3.0,
                    "rssi_error_m": 4.0,
                    "tdoa_anchor_count": 2,
                }
                for target_id in [0, 1]
            ]
        ).to_parquet(data_dir / "position_estimates.parquet", index=False)

        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "target_id": target_id,
                    "antenna_id": antenna_id,
                    "signal_strength_dbm": -45.0 - target_id - antenna_id,
                    "path_loss_db_with_noise": 70.0 + target_id + antenna_id,
                    "obstacle_attenuation_db": 10.0,
                }
                for target_id in [0, 1]
                for antenna_id in [0, 1]
            ]
        ).to_parquet(data_dir / "links_rssi.parquet", index=False)

        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "target_id": 0,
                    "reference_antenna_id": 0,
                    "comparison_antenna_id": 1,
                    "observed_tdoa_ns": 1.2,
                    "reference_distance_m": 3.6,
                    "comparison_distance_m": 8.5,
                    "tdoa_noise_sigma_ns": 0.2,
                    "ideal_tdoa_ns": 1.0,
                },
                {
                    "scenario_id": scenario_id,
                    "target_id": 1,
                    "reference_antenna_id": 0,
                    "comparison_antenna_id": 1,
                    "observed_tdoa_ns": -0.7,
                    "reference_distance_m": 10.0,
                    "comparison_distance_m": 6.3,
                    "tdoa_noise_sigma_ns": 0.2,
                    "ideal_tdoa_ns": -0.5,
                },
            ]
        ).to_parquet(data_dir / "links_tdoa.parquet", index=False)

        pd.DataFrame(
            [
                {
                    "scenario_id": scenario_id,
                    "target_id": target_id,
                    "antenna_id": antenna_id,
                    "observed_bearing_rad": 0.2 + target_id + antenna_id,
                    "observed_doa_rad": 0.1 + target_id + antenna_id,
                    "observed_bearing_deg": math.degrees(0.2 + target_id + antenna_id),
                    "observed_doa_deg": math.degrees(0.1 + target_id + antenna_id),
                    "true_bearing_rad": 0.0,
                    "doa_noise_sigma_deg": 1.0,
                    "is_doa_valid": True,
                }
                for target_id in [0, 1]
                for antenna_id in [0, 1]
            ]
        ).to_parquet(data_dir / "links_doa.parquet", index=False)


if __name__ == "__main__":
    unittest.main()
