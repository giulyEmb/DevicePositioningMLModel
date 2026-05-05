from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITION_MODULE_PATH = PROJECT_ROOT / "Data Generation" / "position_estimation.py"
SPEC = importlib.util.spec_from_file_location("position_estimation", POSITION_MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {POSITION_MODULE_PATH}")
POSITION_ESTIMATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POSITION_ESTIMATION
SPEC.loader.exec_module(POSITION_ESTIMATION)


class RSSIRangeEstimationTests(unittest.TestCase):
    def test_obstacle_attenuation_is_not_oracle_compensated(self) -> None:
        group = pd.DataFrame(
            [
                {
                    "reference_distance_m": 1.0,
                    "initial_signal_strength_dbm": -30.0,
                    "signal_strength_dbm": -70.0,
                    "obstacle_attenuation_db": 20.0,
                    "path_loss_exponent_n": 2.0,
                }
            ]
        )

        ranges_m = POSITION_ESTIMATION._rssi_range_estimates_m(group)

        self.assertAlmostEqual(ranges_m[0], 100.0)


if __name__ == "__main__":
    unittest.main()
