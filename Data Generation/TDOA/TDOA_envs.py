"""
TDOA ENVIRONMENT LAYER
----------------------

This module derives TDOA-specific rows from the shared per-link geometry table
produced by `link_factory.py`.

Input tables:
- env_summary.parquet
- links.parquet

Output table:
- links_tdoa.parquet by default

The resulting rows are one row per target-anchor measurement relative to a
fixed scenario reference anchor.

Usage examples:
python3 "Data Generation/TDOA/TDOA_envs.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/TDOA/TDOA_envs.py" --data-dir "generated_network_scenarios" --seed 42
python3 "Data Generation/TDOA/TDOA_envs.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --propagation-speed-m-per-s 299792458.0
python3 "Data Generation/TDOA/TDOA_envs.py" \
  --data-dir "Data Generation/generated_network_scenarios" --seed 7

python3 "Data Generation/TDOA/TDOA_envs.py" --data-dir "Data Generation/generated_network_scenarios_with_plots" --seed 13

@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


DEFAULT_PROPAGATION_SPEED_M_PER_S = 299_792_458.0

TDOA_NOISE_SIGMA_NS_RANGES = {
    "outdoor": (0.05, 0.20),
    "indoor_los": (0.10, 0.30),
    "indoor_nlos": (0.30, 0.50),
}

REQUIRED_LINK_COLUMNS = {
    "scenario_id",
    "target_id",
    "target_label",
    "target_x",
    "target_y",
    "antenna_id",
    "antenna_label",
    "antenna_x",
    "antenna_y",
    "distance_m",
    "link_state",
    "target_space_type",
}

NoiseRangeMap = Dict[str, Tuple[float, float]]


def _normalize_scenario_env_type(env: str) -> str:
    normalized = str(env).strip().lower()
    if normalized not in {"indoor", "outdoor"}:
        raise ValueError(f"Unsupported scenario environment type for TDOA: {env}")
    return normalized


def _normalize_tdoa_env_type(env: str) -> str:
    normalized = str(env).strip().lower()
    if normalized not in TDOA_NOISE_SIGMA_NS_RANGES:
        raise ValueError(f"Unsupported TDOA environment type: {env}")
    return normalized


def _normalize_link_state(link_state: str) -> str:
    normalized = str(link_state).strip().upper()
    if normalized not in {"LOS", "NLOS"}:
        raise ValueError(f"Unsupported TDOA link_state: {link_state}")
    return normalized


def _resolve_tdoa_env_type(target_space_type: str, link_state: str) -> str:
    space = str(target_space_type).strip().lower()
    state = _normalize_link_state(link_state)

    # Patios and exterior free space are modelled as outdoor links regardless
    # of blocker geometry, matching the RSSI environment split.
    if space in {"exterior", "patio", "outdoor"}:
        return "outdoor"

    # Targets inside the floor-map footprint are split by the per-link LOS/NLOS
    # collision check.
    if space in {"room", "building_free", "indoor"}:
        return "indoor_los" if state == "LOS" else "indoor_nlos"

    raise ValueError(
        f"Unsupported target_space_type for TDOA classification: {target_space_type}"
    )


def _validate_noise_range(name: str, lo: float, hi: float) -> None:
    if lo < 0 or hi < 0:
        raise ValueError(f"{name} noise sigma range values must be >= 0.")
    if lo > hi:
        raise ValueError(f"{name} noise sigma range minimum must be <= maximum.")


def _validate_noise_ranges(noise_ranges: NoiseRangeMap) -> None:
    for env in TDOA_NOISE_SIGMA_NS_RANGES:
        if env not in noise_ranges:
            raise ValueError(f"Missing TDOA noise sigma range for environment: {env}")
        lo, hi = noise_ranges[env]
        _validate_noise_range(env, float(lo), float(hi))


def timing_noise_sigma_given_env_type(
    env: str,
    *,
    noise_ranges: NoiseRangeMap | None = None,
    rng: random.Random | None = None,
) -> float:
    normalized = _normalize_tdoa_env_type(env)
    ranges = noise_ranges or TDOA_NOISE_SIGMA_NS_RANGES
    lo, hi = ranges[normalized]
    sampler = rng or random
    return sampler.uniform(float(lo), float(hi))


def extract_tdoa_link_inputs(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Return one row per antenna-target link with the scenario environment joined in.

    Required columns:
    - scenario_id
    - target_id
    - target_label
    - target_x
    - target_y
    - antenna_id
    - antenna_label
    - antenna_x
    - antenna_y
    - distance_m
    - link_state
    - target_space_type
    """
    data_path = Path(data_dir)
    summary_path = data_path / "env_summary.parquet"
    links_path = data_path / "links.parquet"

    summary_env_df = pd.read_parquet(summary_path, columns=["scenario_id", "env_type"])
    links_df = pd.read_parquet(links_path)

    missing_cols = REQUIRED_LINK_COLUMNS.difference(links_df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"links.parquet is missing required columns: {missing_list}")

    if "env_type" in links_df.columns:
        links_df = links_df.drop(columns=["env_type"])

    merged_df = links_df.merge(
        summary_env_df,
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    if merged_df["env_type"].isna().any():
        missing_count = int(merged_df["env_type"].isna().sum())
        raise ValueError(
            f"{missing_count} link rows do not map to a scenario environment."
        )

    merged_df["env_type"] = merged_df["env_type"].map(_normalize_scenario_env_type)
    merged_df["tdoa_env_type"] = [
        _resolve_tdoa_env_type(target_space_type, link_state)
        for target_space_type, link_state in zip(
            merged_df["target_space_type"],
            merged_df["link_state"],
        )
    ]
    return merged_df.copy()


def create_scenario_tdoa_parameters(
    link_inputs_df: pd.DataFrame,
    *,
    propagation_speed_m_per_s: float = DEFAULT_PROPAGATION_SPEED_M_PER_S,
) -> pd.DataFrame:
    """
    Select the scenario reference anchor and attach scenario-level constants.

    The reference anchor is the lowest antenna_id in each scenario.
    """
    if propagation_speed_m_per_s <= 0:
        raise ValueError("propagation_speed_m_per_s must be > 0.")

    rows = []
    for scenario_id, scenario_df in link_inputs_df.groupby("scenario_id", sort=True):
        antenna_ids = sorted(scenario_df["antenna_id"].drop_duplicates().tolist())
        if len(antenna_ids) < 2:
            raise ValueError(
                f"Scenario {scenario_id} must have at least 2 antennas for TDOA. "
                f"Found {len(antenna_ids)}."
            )

        rows.append(
            {
                "scenario_id": scenario_id,
                "reference_antenna_id": antenna_ids[0],
                "propagation_speed_m_per_s": float(propagation_speed_m_per_s),
            }
        )

    return pd.DataFrame(rows)


def create_link_tdoa_parameters(
    link_inputs_df: pd.DataFrame,
    *,
    seed: int | None = None,
    noise_ranges: NoiseRangeMap | None = None,
) -> pd.DataFrame:
    """
    Sample timing-noise sigma and timing noise once per antenna-target link.

    The link-level timing noise is later differenced between the comparison and
    reference links to form each TDOA measurement.
    """
    ranges = noise_ranges or TDOA_NOISE_SIGMA_NS_RANGES
    _validate_noise_ranges(ranges)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    link_params_df = link_inputs_df[
        ["scenario_id", "target_id", "antenna_id", "tdoa_env_type"]
    ].copy()

    link_params_df["timing_noise_sigma_ns"] = link_params_df["tdoa_env_type"].map(
        lambda env: timing_noise_sigma_given_env_type(
            env,
            noise_ranges=ranges,
            rng=rng,
        )
    )
    link_params_df["timing_noise_ns"] = np_rng.normal(
        loc=0.0,
        scale=link_params_df["timing_noise_sigma_ns"].astype(float).to_numpy(),
        size=len(link_params_df),
    )
    return link_params_df


def build_tdoa_base_table(
    data_dir: Union[str, Path],
    *,
    seed: int | None = None,
    propagation_speed_m_per_s: float = DEFAULT_PROPAGATION_SPEED_M_PER_S,
    outdoor_noise_sigma_ns_min: float = 0.05,
    outdoor_noise_sigma_ns_max: float = 0.20,
    indoor_los_noise_sigma_ns_min: float = 0.10,
    indoor_los_noise_sigma_ns_max: float = 0.30,
    indoor_nlos_noise_sigma_ns_min: float = 0.30,
    indoor_nlos_noise_sigma_ns_max: float = 0.50,
    indoor_noise_sigma_ns_min: float | None = None,
    indoor_noise_sigma_ns_max: float | None = None,
) -> pd.DataFrame:
    """
    Return one TDOA row per scenario, target, and non-reference antenna.

    Added columns:
    - reference_antenna_id
    - reference_antenna_label
    - reference_antenna_x
    - reference_antenna_y
    - reference_distance_m
    - reference_link_state
    - reference_tdoa_env_type
    - reference_timing_noise_sigma_ns
    - reference_timing_noise_ns
    - comparison_antenna_id
    - comparison_antenna_label
    - comparison_antenna_x
    - comparison_antenna_y
    - comparison_distance_m
    - comparison_link_state
    - comparison_tdoa_env_type
    - comparison_timing_noise_sigma_ns
    - comparison_timing_noise_ns
    - propagation_speed_m_per_s
    - tdoa_noise_sigma_ns
    - reference_arrival_time_ns
    - comparison_arrival_time_ns
    - reference_observed_arrival_time_ns
    - comparison_observed_arrival_time_ns
    - ideal_tdoa_ns
    - tdoa_noise_ns
    - observed_tdoa_ns
    - delta_distance_m
    """
    if propagation_speed_m_per_s <= 0:
        raise ValueError("propagation_speed_m_per_s must be > 0.")

    if (indoor_noise_sigma_ns_min is None) != (indoor_noise_sigma_ns_max is None):
        raise ValueError(
            "indoor_noise_sigma_ns_min and indoor_noise_sigma_ns_max must be provided together."
        )

    resolved_indoor_los_noise_sigma_ns_min = float(indoor_los_noise_sigma_ns_min)
    resolved_indoor_los_noise_sigma_ns_max = float(indoor_los_noise_sigma_ns_max)
    resolved_indoor_nlos_noise_sigma_ns_min = float(indoor_nlos_noise_sigma_ns_min)
    resolved_indoor_nlos_noise_sigma_ns_max = float(indoor_nlos_noise_sigma_ns_max)

    if indoor_noise_sigma_ns_min is not None and indoor_noise_sigma_ns_max is not None:
        resolved_indoor_los_noise_sigma_ns_min = float(indoor_noise_sigma_ns_min)
        resolved_indoor_los_noise_sigma_ns_max = float(indoor_noise_sigma_ns_max)
        resolved_indoor_nlos_noise_sigma_ns_min = float(indoor_noise_sigma_ns_min)
        resolved_indoor_nlos_noise_sigma_ns_max = float(indoor_noise_sigma_ns_max)

    noise_ranges = {
        "outdoor": (
            float(outdoor_noise_sigma_ns_min),
            float(outdoor_noise_sigma_ns_max),
        ),
        "indoor_los": (
            resolved_indoor_los_noise_sigma_ns_min,
            resolved_indoor_los_noise_sigma_ns_max,
        ),
        "indoor_nlos": (
            resolved_indoor_nlos_noise_sigma_ns_min,
            resolved_indoor_nlos_noise_sigma_ns_max,
        ),
    }
    _validate_noise_ranges(noise_ranges)

    links_df = extract_tdoa_link_inputs(data_dir)

    if (links_df["distance_m"] < 0).any():
        bad_count = int((links_df["distance_m"] < 0).sum())
        raise ValueError(
            f"distance_m must be >= 0 for TDOA calculations. Found {bad_count} invalid rows."
        )
    duplicated_links = links_df.duplicated(
        ["scenario_id", "target_id", "antenna_id"],
        keep=False,
    )
    if duplicated_links.any():
        duplicate_count = int(duplicated_links.sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate links by scenario_id, target_id, and antenna_id."
        )

    scenario_antenna_counts = (
        links_df.groupby("scenario_id")["antenna_id"]
        .nunique()
        .rename("scenario_antenna_count")
        .reset_index()
    )
    target_antenna_counts = (
        links_df.groupby(["scenario_id", "target_id"])["antenna_id"]
        .nunique()
        .rename("target_antenna_count")
        .reset_index()
        .merge(scenario_antenna_counts, on="scenario_id", how="left")
    )
    incomplete_targets = target_antenna_counts[
        target_antenna_counts["target_antenna_count"]
        != target_antenna_counts["scenario_antenna_count"]
    ]
    if not incomplete_targets.empty:
        bad_count = len(incomplete_targets)
        raise ValueError(
            f"{bad_count} scenario-target groups do not have one link per scenario antenna."
        )

    scenario_params_df = create_scenario_tdoa_parameters(
        links_df,
        propagation_speed_m_per_s=propagation_speed_m_per_s,
    )
    link_params_df = create_link_tdoa_parameters(
        links_df,
        seed=seed,
        noise_ranges=noise_ranges,
    )

    links_with_params_df = links_df.merge(
        scenario_params_df,
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    links_with_params_df = links_with_params_df.merge(
        link_params_df,
        on=["scenario_id", "target_id", "antenna_id", "tdoa_env_type"],
        how="left",
        validate="one_to_one",
    )

    reference_mask = (
        links_with_params_df["antenna_id"]
        == links_with_params_df["reference_antenna_id"]
    )
    reference_cols = [
        "scenario_id",
        "target_id",
        "antenna_id",
        "antenna_label",
        "antenna_x",
        "antenna_y",
        "distance_m",
        "link_state",
        "tdoa_env_type",
        "timing_noise_sigma_ns",
        "timing_noise_ns",
    ]
    reference_df = links_with_params_df.loc[reference_mask, reference_cols].rename(
        columns={
            "antenna_id": "reference_antenna_id",
            "antenna_label": "reference_antenna_label",
            "antenna_x": "reference_antenna_x",
            "antenna_y": "reference_antenna_y",
            "distance_m": "reference_distance_m",
            "link_state": "reference_link_state",
            "tdoa_env_type": "reference_tdoa_env_type",
            "timing_noise_sigma_ns": "reference_timing_noise_sigma_ns",
            "timing_noise_ns": "reference_timing_noise_ns",
        }
    )

    duplicated_refs = reference_df.duplicated(["scenario_id", "target_id"], keep=False)
    if duplicated_refs.any():
        duplicate_count = int(duplicated_refs.sum())
        raise ValueError(
            f"Found {duplicate_count} duplicate reference links by scenario_id and target_id."
        )

    comparison_df = links_with_params_df.loc[~reference_mask].copy()
    if comparison_df.empty:
        raise ValueError("No non-reference antenna links are available for TDOA.")

    tdoa_df = comparison_df.merge(
        reference_df,
        on=["scenario_id", "target_id", "reference_antenna_id"],
        how="left",
        validate="many_to_one",
    )

    if tdoa_df["reference_distance_m"].isna().any():
        missing_count = int(tdoa_df["reference_distance_m"].isna().sum())
        raise ValueError(
            f"{missing_count} TDOA comparison rows do not have a reference-anchor link."
        )

    tdoa_df = tdoa_df.rename(
        columns={
            "antenna_id": "comparison_antenna_id",
            "antenna_label": "comparison_antenna_label",
            "antenna_x": "comparison_antenna_x",
            "antenna_y": "comparison_antenna_y",
            "distance_m": "comparison_distance_m",
            "link_state": "comparison_link_state",
            "tdoa_env_type": "comparison_tdoa_env_type",
            "timing_noise_sigma_ns": "comparison_timing_noise_sigma_ns",
            "timing_noise_ns": "comparison_timing_noise_ns",
        }
    )

    tdoa_df["reference_arrival_time_ns"] = (
        tdoa_df["reference_distance_m"].astype(float)
        / tdoa_df["propagation_speed_m_per_s"].astype(float)
        * 1e9
    )
    tdoa_df["comparison_arrival_time_ns"] = (
        tdoa_df["comparison_distance_m"].astype(float)
        / tdoa_df["propagation_speed_m_per_s"].astype(float)
        * 1e9
    )
    tdoa_df["delta_distance_m"] = (
        tdoa_df["comparison_distance_m"].astype(float)
        - tdoa_df["reference_distance_m"].astype(float)
    )
    tdoa_df["ideal_tdoa_ns"] = (
        tdoa_df["delta_distance_m"].astype(float)
        / tdoa_df["propagation_speed_m_per_s"].astype(float)
        * 1e9
    )
    tdoa_df["reference_observed_arrival_time_ns"] = (
        tdoa_df["reference_arrival_time_ns"].astype(float)
        + tdoa_df["reference_timing_noise_ns"].astype(float)
    )
    tdoa_df["comparison_observed_arrival_time_ns"] = (
        tdoa_df["comparison_arrival_time_ns"].astype(float)
        + tdoa_df["comparison_timing_noise_ns"].astype(float)
    )
    tdoa_df["tdoa_noise_sigma_ns"] = np.sqrt(
        np.square(tdoa_df["reference_timing_noise_sigma_ns"].astype(float))
        + np.square(tdoa_df["comparison_timing_noise_sigma_ns"].astype(float))
    )
    tdoa_df["tdoa_noise_ns"] = (
        tdoa_df["comparison_timing_noise_ns"].astype(float)
        - tdoa_df["reference_timing_noise_ns"].astype(float)
    )
    tdoa_df["observed_tdoa_ns"] = (
        tdoa_df["comparison_observed_arrival_time_ns"].astype(float)
        - tdoa_df["reference_observed_arrival_time_ns"].astype(float)
    )

    output_cols = [
        "scenario_id",
        "target_id",
        "target_label",
        "target_x",
        "target_y",
        "reference_antenna_id",
        "reference_antenna_label",
        "reference_antenna_x",
        "reference_antenna_y",
        "reference_distance_m",
        "reference_link_state",
        "reference_tdoa_env_type",
        "reference_timing_noise_sigma_ns",
        "reference_timing_noise_ns",
        "comparison_antenna_id",
        "comparison_antenna_label",
        "comparison_antenna_x",
        "comparison_antenna_y",
        "comparison_distance_m",
        "comparison_link_state",
        "comparison_tdoa_env_type",
        "comparison_timing_noise_sigma_ns",
        "comparison_timing_noise_ns",
        "propagation_speed_m_per_s",
        "tdoa_noise_sigma_ns",
        "reference_arrival_time_ns",
        "comparison_arrival_time_ns",
        "reference_observed_arrival_time_ns",
        "comparison_observed_arrival_time_ns",
        "ideal_tdoa_ns",
        "tdoa_noise_ns",
        "observed_tdoa_ns",
        "delta_distance_m",
    ]
    return tdoa_df[output_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TDOA timing-difference rows from env_summary.parquet and links.parquet."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing env_summary.parquet and links.parquet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible TDOA sigma and timing-noise sampling.",
    )
    parser.add_argument(
        "--propagation-speed-m-per-s",
        type=float,
        default=DEFAULT_PROPAGATION_SPEED_M_PER_S,
        help="Signal propagation speed in meters per second.",
    )
    parser.add_argument(
        "--outdoor-noise-sigma-ns-min",
        type=float,
        default=0.05,
        help="Minimum outdoor link timing-noise sigma in nanoseconds.",
    )
    parser.add_argument(
        "--outdoor-noise-sigma-ns-max",
        type=float,
        default=0.20,
        help="Maximum outdoor link timing-noise sigma in nanoseconds.",
    )
    parser.add_argument(
        "--indoor-los-noise-sigma-ns-min",
        type=float,
        default=0.10,
        help="Minimum indoor LOS link timing-noise sigma in nanoseconds.",
    )
    parser.add_argument(
        "--indoor-los-noise-sigma-ns-max",
        type=float,
        default=0.30,
        help="Maximum indoor LOS link timing-noise sigma in nanoseconds.",
    )
    parser.add_argument(
        "--indoor-nlos-noise-sigma-ns-min",
        type=float,
        default=0.30,
        help="Minimum indoor NLOS link timing-noise sigma in nanoseconds.",
    )
    parser.add_argument(
        "--indoor-nlos-noise-sigma-ns-max",
        type=float,
        default=0.50,
        help="Maximum indoor NLOS link timing-noise sigma in nanoseconds.",
    )
    parser.add_argument(
        "--indoor-noise-sigma-ns-min",
        type=float,
        default=None,
        help="Legacy alias. If set with --indoor-noise-sigma-ns-max, applies the same range to both indoor LOS and indoor NLOS links.",
    )
    parser.add_argument(
        "--indoor-noise-sigma-ns-max",
        type=float,
        default=None,
        help="Legacy alias. If set with --indoor-noise-sigma-ns-min, applies the same range to both indoor LOS and indoor NLOS links.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: <data-dir>/links_tdoa.parquet",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="If set, write output directly to <data-dir>/links.parquet.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.overwrite_links:
        output_path = data_dir / "links.parquet"
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "links_tdoa.parquet"

    tdoa_df = build_tdoa_base_table(
        data_dir=data_dir,
        seed=args.seed,
        propagation_speed_m_per_s=args.propagation_speed_m_per_s,
        outdoor_noise_sigma_ns_min=args.outdoor_noise_sigma_ns_min,
        outdoor_noise_sigma_ns_max=args.outdoor_noise_sigma_ns_max,
        indoor_los_noise_sigma_ns_min=args.indoor_los_noise_sigma_ns_min,
        indoor_los_noise_sigma_ns_max=args.indoor_los_noise_sigma_ns_max,
        indoor_nlos_noise_sigma_ns_min=args.indoor_nlos_noise_sigma_ns_min,
        indoor_nlos_noise_sigma_ns_max=args.indoor_nlos_noise_sigma_ns_max,
        indoor_noise_sigma_ns_min=args.indoor_noise_sigma_ns_min,
        indoor_noise_sigma_ns_max=args.indoor_noise_sigma_ns_max,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tdoa_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(tdoa_df)} TDOA rows to {output_path}")
    if args.seed is None:
        print("Seed: none (sampling varies across runs).")
    else:
        print(f"Seed: {args.seed} (reproducible sampling).")


if __name__ == "__main__":
    main()
