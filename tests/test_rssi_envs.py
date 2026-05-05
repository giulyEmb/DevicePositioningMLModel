from __future__ import annotations
import importlib.util
import math
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RSSI_MODULE_PATH = PROJECT_ROOT / "Data Generation" / "RSSI" / "RSSI_envs.py"
SPEC = importlib.util.spec_from_file_location("rssi_envs", RSSI_MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {RSSI_MODULE_PATH}")
RSSI_ENVS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RSSI_ENVS
SPEC.loader.exec_module(RSSI_ENVS)


def _write_base_tables(
    data_dir: Path,
    *,
    floor_elements: list[dict[str, object]] | None = None,
    wall_count: int = 1,
    human_count: int = 2,
    link_state: str = "NLOS",
) -> None:
    pd.DataFrame([{"scenario_id": "s1", "env_type": "indoor"}]).to_parquet(
        data_dir / "env_summary.parquet",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "scenario_id": "s1",
                "antenna_id": 1,
                "antenna_label": "a1",
                "antenna_x": 0.0,
                "antenna_y": 0.0,
                "target_id": 1,
                "target_label": "t1",
                "target_x": 10.0,
                "target_y": 0.0,
                "target_cell_id": "c1",
                "target_row_idx": 0,
                "target_col_idx": 0,
                "target_space_type": "room",
                "target_room_id": "r1",
                "target_room_type": "office",
                "target_patio_id": None,
                "distance_m": 10.0,
                "is_los": link_state == "LOS",
                "link_state": link_state,
                "wall_blocker_count": wall_count,
                "human_blocker_count": human_count,
                "total_blocker_count": wall_count + human_count,
                "blocking_obstacles_json": "[]",
            }
        ]
    ).to_parquet(data_dir / "links.parquet", index=False)
    if floor_elements is not None:
        pd.DataFrame(floor_elements).to_parquet(
            data_dir / "floor_plan_elements.parquet",
            index=False,
        )


def _door_row(x: float, y: float) -> dict[str, object]:
    return {
        "scenario_id": "s1",
        "element_id": "door_1",
        "element_type": "door",
        "x": x,
        "y": y,
        "orientation_angle": 0.0,
        "doorway_width": 1.0,
        "door_width": 0.8,
        "thickness": 0.2,
    }


def _window_row(x: float, y: float) -> dict[str, object]:
    return {
        "scenario_id": "s1",
        "element_id": "window_1",
        "element_type": "window",
        "x": x,
        "y": y,
        "orientation_angle": 0.0,
        "length": 1.0,
        "overall_thickness": 0.2,
        "single_line_thickness": 0.05,
    }


class RSSIMaterialAttenuationTests(unittest.TestCase):
    def test_human_loss_frequency_projection(self) -> None:
        self.assertAlmostEqual(RSSI_ENVS.human_loss_db_at_freq(2450.0), 17.22)
        self.assertAlmostEqual(RSSI_ENVS.human_loss_db_at_freq(5000.0), 29.3325)
        self.assertAlmostEqual(RSSI_ENVS.human_loss_db_at_freq(5500.0), 31.7075)

    def test_material_counts_and_attenuation_are_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_base_tables(
                data_dir,
                floor_elements=[
                    _door_row(5.0, -0.1),
                    _window_row(7.0, -0.1),
                ],
            )

            rssi_df = RSSI_ENVS.build_rssi_base_table(
                data_dir,
                seed=123,
                freq_mhz=5000.0,
            )
            row = rssi_df.iloc[0]

            self.assertEqual(row["door_blocker_count"], 1)
            self.assertEqual(row["window_blocker_count"], 1)
            self.assertAlmostEqual(row["wall_loss_db"], 20.0)
            self.assertAlmostEqual(row["human_loss_db"], 29.3325)
            self.assertAlmostEqual(row["door_loss_db"], 15.0)
            self.assertAlmostEqual(row["window_loss_db"], 7.0)
            self.assertAlmostEqual(row["wall_attenuation_db"], 20.0)
            self.assertAlmostEqual(row["human_attenuation_db"], 58.665)
            self.assertAlmostEqual(row["door_attenuation_db"], 15.0)
            self.assertAlmostEqual(row["window_attenuation_db"], 7.0)
            self.assertAlmostEqual(row["obstacle_attenuation_db"], 100.665)

            path_loss_increment = (
                10.0
                * row["path_loss_exponent_n"]
                * math.log10(row["distance_m"] / row["reference_distance_m"])
            )
            self.assertAlmostEqual(
                row["path_loss_db_with_noise"],
                path_loss_increment
                + row["obstacle_attenuation_db"]
                + row["shadow_noise_db"],
            )

    def test_non_crossing_door_and_window_do_not_attenuate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_base_tables(
                data_dir,
                floor_elements=[
                    _door_row(5.0, 3.0),
                    _window_row(7.0, 3.0),
                ],
                wall_count=0,
                human_count=0,
                link_state="LOS",
            )

            rssi_df = RSSI_ENVS.build_rssi_base_table(data_dir, seed=123)
            row = rssi_df.iloc[0]

            self.assertEqual(row["door_blocker_count"], 0)
            self.assertEqual(row["window_blocker_count"], 0)
            self.assertAlmostEqual(row["obstacle_attenuation_db"], 0.0)

    def test_missing_floor_plan_elements_keeps_zero_door_window_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_base_tables(
                data_dir,
                floor_elements=None,
                wall_count=0,
                human_count=0,
                link_state="LOS",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rssi_df = RSSI_ENVS.build_rssi_base_table(data_dir, seed=123)

            self.assertTrue(
                any("Door/window RSSI attenuation will be zero" in str(item.message) for item in caught)
            )
            row = rssi_df.iloc[0]
            self.assertEqual(row["door_blocker_count"], 0)
            self.assertEqual(row["window_blocker_count"], 0)
            self.assertAlmostEqual(row["obstacle_attenuation_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
