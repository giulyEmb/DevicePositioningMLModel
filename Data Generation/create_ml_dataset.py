"""
LABELLED ML DATASET BUILDER
---------------------------

Construct one supervised-learning row per scenario-target pair by merging:
- scenario/environment features
- antenna-layout aggregate statistics (mean, min, max, std)
- aggregate observed telemetry summaries for RSSI, TDOA, and DOA/AOA

NOTE: This dataset uses aggregated features only, not per-antenna wide features.
This keeps the dataset memory-efficient and focuses on environmental constraints
and measurement statistics rather than individual antenna configurations.

Input tables:
- env_summary.parquet
- antennas.parquet
- targets.parquet
- links_rssi.parquet
- links_tdoa.parquet
- links_doa.parquet

Output table:
- ml_dataset.parquet by default

Usage examples:
python3 "Data Generation/create_ml_dataset.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/create_ml_dataset.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --output "Data Generation/generated_network_scenarios/ml_dataset.parquet"
python3 "Data Generation/create_ml_dataset.py" \
  --data-dir "Data Generation/generated_network_scenarios"

python3 "Data Generation/create_ml_dataset.py" --data-dir "Data Generation/generated_network_scenarios_with_plots"

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Iterable, Union
import numpy as np
import pandas as pd


KEY_COLUMNS = ["scenario_id", "target_id"]
LABEL_COLUMNS = ["target_x", "target_y"]
IDENTIFIER_AND_LABEL_COLUMNS = [*KEY_COLUMNS, *LABEL_COLUMNS]
SAFE_ID_PATTERN = re.compile(r"[^0-9A-Za-z]+")


def _require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{table_name} is missing required columns: {missing_list}")


def _read_table(data_dir: Union[str, Path], table_name: str, required: Iterable[str]) -> pd.DataFrame:
    path = Path(data_dir) / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding data-generation stage first."
        )
    df = pd.read_parquet(path)
    _require_columns(df, required, f"{table_name}.parquet")
    return df


def _std_or_zero(series: pd.Series) -> float:
    value = series.std()
    if pd.isna(value):
        return 0.0
    return float(value)


def _feature_id(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Feature IDs cannot be missing.")
    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    feature_id = SAFE_ID_PATTERN.sub("_", str(value).strip()).strip("_")
    if not feature_id:
        raise ValueError("Feature IDs cannot be empty.")
    return feature_id


def _position_component(value: object, index: int) -> float:
    try:
        return float(value[index])  # type: ignore[index]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "targets.parquet position values must contain x/y coordinates."
        ) from exc


def _target_labels(targets_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(targets_df, KEY_COLUMNS, "targets.parquet")
    prepared_df = targets_df.copy()
    if not set(LABEL_COLUMNS).issubset(prepared_df.columns):
        _require_columns(prepared_df, ["position"], "targets.parquet")
        prepared_df["target_x"] = prepared_df["position"].map(
            lambda value: _position_component(value, 0)
        )
        prepared_df["target_y"] = prepared_df["position"].map(
            lambda value: _position_component(value, 1)
        )

    labels_df = prepared_df[IDENTIFIER_AND_LABEL_COLUMNS].copy()
    duplicate_targets = labels_df.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_targets.any():
        duplicate_count = int(duplicate_targets.sum())
        raise ValueError(f"targets.parquet has {duplicate_count} duplicate target-label rows.")
    return labels_df.drop_duplicates().copy()


def _scenario_features(env_summary_df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "scenario_id",
        "area",
        "env_type",
        "width",
        "height",
        "x_domain_min",
        "x_domain_max",
        "y_range_min",
        "y_range_max",
        "antenna_count",
        "human_count",
        "floor_plan_room_count",
        "floor_plan_patio_count",
        "floor_plan_element_count",
    ]
    available_cols = [column for column in preferred_cols if column in env_summary_df.columns]
    scenario_df = env_summary_df[available_cols].drop_duplicates().copy()
    duplicates = scenario_df.duplicated("scenario_id", keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(f"env_summary.parquet has {duplicate_count} duplicate scenario rows.")
    return scenario_df


def _antenna_features(antennas_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        antennas_df,
        ["scenario_id", "antenna_id", "x", "y", "coverage_radius"],
        "antennas.parquet",
    )
    grouped = antennas_df.groupby("scenario_id", sort=True)
    features_df = grouped.agg(
        antenna_layout_count=("antenna_id", "nunique"),
        antenna_x_mean=("x", "mean"),
        antenna_y_mean=("y", "mean"),
        antenna_x_min=("x", "min"),
        antenna_x_max=("x", "max"),
        antenna_y_min=("y", "min"),
        antenna_y_max=("y", "max"),
        antenna_coverage_radius_mean_m=("coverage_radius", "mean"),
        antenna_coverage_radius_min_m=("coverage_radius", "min"),
        antenna_coverage_radius_max_m=("coverage_radius", "max"),
    ).reset_index()
    spread_df = grouped.agg(
        antenna_x_std=("x", _std_or_zero),
        antenna_y_std=("y", _std_or_zero),
    ).reset_index()
    summary_df = features_df.merge(
        spread_df,
        on="scenario_id",
        how="left",
        validate="one_to_one",
    )
    return summary_df











def _rssi_features(rssi_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        rssi_df,
        [
            "scenario_id",
            "target_id",
            "signal_strength_dbm",
        ],
        "links_rssi.parquet",
    )
    grouped = rssi_df.groupby(KEY_COLUMNS, sort=True)
    features_df = grouped.agg(
        rssi_measurement_count=("signal_strength_dbm", "count"),
        rssi_signal_mean_dbm=("signal_strength_dbm", "mean"),
        rssi_signal_min_dbm=("signal_strength_dbm", "min"),
        rssi_signal_max_dbm=("signal_strength_dbm", "max"),
    ).reset_index()
    spread_df = grouped.agg(
        rssi_signal_std_dbm=("signal_strength_dbm", _std_or_zero),
    ).reset_index()
    summary_df = features_df.merge(
        spread_df,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    return summary_df


def _tdoa_features(tdoa_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        tdoa_df,
        [
            "scenario_id",
            "target_id",
            "observed_tdoa_ns",
        ],
        "links_tdoa.parquet",
    )
    grouped = tdoa_df.groupby(KEY_COLUMNS, sort=True)
    features_df = grouped.agg(
        tdoa_measurement_count=("observed_tdoa_ns", "count"),
        tdoa_observed_mean_ns=("observed_tdoa_ns", "mean"),
        tdoa_observed_min_ns=("observed_tdoa_ns", "min"),
        tdoa_observed_max_ns=("observed_tdoa_ns", "max"),
    ).reset_index()
    spread_df = grouped.agg(
        tdoa_observed_std_ns=("observed_tdoa_ns", _std_or_zero),
    ).reset_index()
    summary_df = features_df.merge(
        spread_df,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    return summary_df


def _doa_features(doa_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        doa_df,
        [
            "scenario_id",
            "target_id",
            "observed_bearing_rad",
            "observed_doa_rad",
        ],
        "links_doa.parquet",
    )
    prepared_df = doa_df.copy()
    prepared_df["doa_observed_bearing_sin"] = np.sin(
        prepared_df["observed_bearing_rad"].astype(float)
    )
    prepared_df["doa_observed_bearing_cos"] = np.cos(
        prepared_df["observed_bearing_rad"].astype(float)
    )
    prepared_df["doa_observed_doa_sin"] = np.sin(
        prepared_df["observed_doa_rad"].astype(float)
    )
    prepared_df["doa_observed_doa_cos"] = np.cos(
        prepared_df["observed_doa_rad"].astype(float)
    )

    grouped = prepared_df.groupby(KEY_COLUMNS, sort=True)
    summary_df = grouped.agg(
        doa_measurement_count=("observed_bearing_rad", "count"),
        doa_observed_bearing_sin_mean=("doa_observed_bearing_sin", "mean"),
        doa_observed_bearing_cos_mean=("doa_observed_bearing_cos", "mean"),
        doa_observed_doa_sin_mean=("doa_observed_doa_sin", "mean"),
        doa_observed_doa_cos_mean=("doa_observed_doa_cos", "mean"),
    ).reset_index()
    return summary_df


def _is_forbidden_ml_feature(column: str) -> bool:
    if column in IDENTIFIER_AND_LABEL_COLUMNS:
        return False
    forbidden_exact = {
        "link_count",
        "rssi_anchor_count",
        "tdoa_anchor_count",
        "doa_anchor_count",
    }
    forbidden_tokens = (
        "distance",
        "path_loss",
        "attenuation",
        "blocker",
        "loss",
        "noise",
        "sigma",
        "ideal",
        "arrival_time",
        "link_state",
    )
    forbidden_prefixes = ("link_", "rssi_est", "tdoa_est", "doa_est", "true_")
    forbidden_suffixes = (
        "_est_x",
        "_est_y",
        "_error_m",
        "_residual_rmse_m",
        "_success",
        "_env_type",
    )
    return (
        column in forbidden_exact
        or column.startswith(forbidden_prefixes)
        or column.endswith(forbidden_suffixes)
        or any(token in column for token in forbidden_tokens)
    )


def _validate_one_row_per_target(dataset_df: pd.DataFrame) -> None:
    duplicate_rows = dataset_df.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_rows.any():
        duplicate_count = int(duplicate_rows.sum())
        raise ValueError(f"ML dataset has {duplicate_count} duplicate scenario-target rows.")
    if dataset_df[LABEL_COLUMNS].isna().any().any():
        raise ValueError("ML dataset contains missing target_x or target_y labels.")
    forbidden_columns = sorted(
        column for column in dataset_df.columns if _is_forbidden_ml_feature(column)
    )
    if forbidden_columns:
        forbidden_list = ", ".join(forbidden_columns)
        raise ValueError(f"ML dataset contains leakage or simulator-only columns: {forbidden_list}")


def build_ml_dataset(data_dir: Union[str, Path]) -> pd.DataFrame:
    env_summary_df = _read_table(data_dir, "env_summary", ["scenario_id", "env_type"])
    antennas_df = _read_table(data_dir, "antennas", ["scenario_id", "antenna_id"])
    targets_df = _read_table(data_dir, "targets", ["scenario_id", "target_id"])
    rssi_df = _read_table(data_dir, "links_rssi", ["scenario_id", "target_id"])
    tdoa_df = _read_table(data_dir, "links_tdoa", ["scenario_id", "target_id"])
    doa_df = _read_table(data_dir, "links_doa", ["scenario_id", "target_id"])

    dataset_df = _target_labels(targets_df)
    dataset_df = dataset_df.merge(
        _scenario_features(env_summary_df),
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    dataset_df = dataset_df.merge(
        _antenna_features(antennas_df),
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    dataset_df = dataset_df.merge(
        _rssi_features(rssi_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    dataset_df = dataset_df.merge(
        _tdoa_features(tdoa_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    dataset_df = dataset_df.merge(
        _doa_features(doa_df),
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    _validate_one_row_per_target(dataset_df)
    return dataset_df.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one labelled ML dataset row per scenario-target pair."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing generated parquet tables.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: <data-dir>/ml_dataset.parquet",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "ml_dataset.parquet"
    ml_dataset_df = build_ml_dataset(data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_dataset_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(ml_dataset_df)} labelled ML rows to {output_path}")
    print("Labels: target_x, target_y")


if __name__ == "__main__":
    main()
