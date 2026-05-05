from __future__ import annotations
import importlib.util
import sys
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


def _base_rssi_rows() -> list[dict[str, object]]:
    return [
        {
            "scenario_id": "s1",
            "target_id": 1,
            "signal_strength_dbm": -40.0,
            "path_loss_db_with_noise": 70.0,
            "path_loss_exponent_n": 2.0,
            "shadow_sigma_db": 6.0,
        },
        {
            "scenario_id": "s1",
            "target_id": 1,
            "signal_strength_dbm": -50.0,
            "path_loss_db_with_noise": 80.0,
            "path_loss_exponent_n": 2.0,
            "shadow_sigma_db": 6.0,
        },
    ]


class RSSIFeatureAggregationTests(unittest.TestCase):
    def test_new_material_columns_are_aggregated(self) -> None:
        rows = _base_rssi_rows()
        rows[0].update(
            {
                "obstacle_attenuation_db": 42.0,
                "wall_attenuation_db": 20.0,
                "human_attenuation_db": 0.0,
                "door_attenuation_db": 15.0,
                "window_attenuation_db": 7.0,
                "door_blocker_count": 1,
                "window_blocker_count": 1,
                "wall_loss_db": 20.0,
                "human_loss_db": 31.7075,
                "door_loss_db": 15.0,
                "window_loss_db": 7.0,
            }
        )
        rows[1].update(
            {
                "obstacle_attenuation_db": 0.0,
                "wall_attenuation_db": 0.0,
                "human_attenuation_db": 0.0,
                "door_attenuation_db": 0.0,
                "window_attenuation_db": 0.0,
                "door_blocker_count": 0,
                "window_blocker_count": 0,
                "wall_loss_db": 20.0,
                "human_loss_db": 31.7075,
                "door_loss_db": 15.0,
                "window_loss_db": 7.0,
            }
        )

        features_df = CREATE_ML_DATASET._rssi_features(pd.DataFrame(rows))
        row = features_df.iloc[0]

        self.assertAlmostEqual(row["rssi_door_attenuation_mean_db"], 7.5)
        self.assertAlmostEqual(row["rssi_window_attenuation_mean_db"], 3.5)
        self.assertAlmostEqual(row["rssi_door_blocker_mean"], 0.5)
        self.assertEqual(row["rssi_door_blocker_max"], 1)
        self.assertAlmostEqual(row["rssi_window_blocker_mean"], 0.5)
        self.assertEqual(row["rssi_window_blocker_max"], 1)
        self.assertAlmostEqual(row["rssi_human_loss_mean_db"], 31.7075)
        self.assertAlmostEqual(row["rssi_window_loss_mean_db"], 7.0)

    def test_missing_material_columns_default_to_zero(self) -> None:
        features_df = CREATE_ML_DATASET._rssi_features(pd.DataFrame(_base_rssi_rows()))
        row = features_df.iloc[0]

        self.assertAlmostEqual(row["rssi_door_attenuation_mean_db"], 0.0)
        self.assertAlmostEqual(row["rssi_window_attenuation_mean_db"], 0.0)
        self.assertAlmostEqual(row["rssi_door_blocker_mean"], 0.0)
        self.assertAlmostEqual(row["rssi_window_blocker_mean"], 0.0)
        self.assertAlmostEqual(row["rssi_door_loss_mean_db"], 0.0)
        self.assertAlmostEqual(row["rssi_window_loss_mean_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
