"""
RSSI ENVIRONMENT LAYER
----------------------

This module derives RSSI-specific columns from the shared per-link geometry
table produced by `link_factory.py`.

Input tables:
- env_summary.parquet
- links.parquet
- floor_plan_elements.parquet, optional for RSSI-only door/window attenuation

Output table:
- links_rssi.parquet by default

The resulting rows remain one row per antenna-target link, which is the right
shape for later pivoting into ML feature matrices.

Usage examples:
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios"
python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "generated_network_scenarios" --seed 42
python3 "Data Generation/RSSI/RSSI_envs.py" \
  --data-dir "Data Generation/generated_network_scenarios" \
  --tx-power-dbm 20.0 \
  --tx-gain-dbi 3.0
python3 "Data Generation/RSSI/RSSI_envs.py" \
  --data-dir "Data Generation/generated_network_scenarios" --seed 7

python3 "Data Generation/RSSI/RSSI_envs.py" --data-dir "Data Generation/generated_network_scenarios_with_plots" --seed 13
  
@author: Giuliana Emberson
@date: 7th of May 2026

"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import math
import random
import warnings
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union
import numpy as np
import pandas as pd


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
GEOMETRY_EPSILON = 1e-9

DEFAULT_WALL_LOSS_DB = 20.0
DEFAULT_DOOR_LOSS_DB = 15.0
DEFAULT_WINDOW_LOSS_DB = 7.0
HUMAN_ATTENUATION_REFERENCE_FREQ_GHZ = 2.45
HUMAN_ATTENUATION_REFERENCE_DB = 17.22
HUMAN_ATTENUATION_SLOPE_DB_PER_GHZ = 4.75

PATH_LOSS_EXPONENT_RANGES = {
    "outdoor": (2.7, 3.5),
    "indoor_los": (1.6, 1.8),
    "indoor_nlos": (4.0, 6.0),
}

SHADOW_SIGMA_DB_RANGES = {
    "outdoor": (5.0, 7.0),
    "indoor_los": (8.0, 10.0),
    "indoor_nlos": (11.0, 13.0),
}


@dataclass(frozen=True)
class _MaterialObstacle:
    scenario_id: object | None
    obstacle_type: str
    coordinates: Tuple[Point, ...]
    bbox: BBox


def _normalize_env_type(env: str) -> str:
    normalized = str(env).strip().lower()
    if normalized not in PATH_LOSS_EXPONENT_RANGES:
        raise ValueError(f"Unsupported RSSI environment type: {env}")
    return normalized


def _resolve_rssi_env_type(target_space_type: str, link_state: str) -> str:
    space = str(target_space_type).strip().lower()
    state = str(link_state).strip().upper()

    # Outdoor target areas use the outdoor RSSI class regardless of LOS/NLOS.
    # Patios are treated as open-air spaces for propagation classification.
    if space in {"exterior", "patio", "outdoor"}:
        return "outdoor"

    # Indoor target areas are further split by per-link blocker geometry.
    if space in {"room", "building_free", "indoor"}:
        return "indoor_los" if state == "LOS" else "indoor_nlos"

    raise ValueError(f"Unsupported target_space_type for RSSI classification: {target_space_type}")


def path_loss_exponent_given_env_type(env: str, *, rng: random.Random | None = None) -> float:
    normalized = _normalize_env_type(env)
    lo, hi = PATH_LOSS_EXPONENT_RANGES[normalized]
    sampler = rng or random
    return sampler.uniform(lo, hi)


def shadow_sigma_given_env_type(env: str, *, rng: random.Random | None = None) -> float:
    normalized = _normalize_env_type(env)
    lo, hi = SHADOW_SIGMA_DB_RANGES[normalized]
    sampler = rng or random
    return sampler.uniform(lo, hi)


def human_loss_db_at_freq(freq_mhz: float) -> float:
    if freq_mhz <= 0:
        raise ValueError("freq_mhz must be > 0.")
    freq_ghz = float(freq_mhz) / 1000.0
    return HUMAN_ATTENUATION_REFERENCE_DB + (
        HUMAN_ATTENUATION_SLOPE_DB_PER_GHZ
        * (freq_ghz - HUMAN_ATTENUATION_REFERENCE_FREQ_GHZ)
    )


def _reference_rssi_dbm_from_freq(
    *,
    freq_mhz: Union[float, pd.Series],
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    reference_distance_m: float,
) -> Union[float, pd.Series]:
    speed_light_m_per_s = 299_792_458.0
    freq_hz = pd.Series(freq_mhz, copy=False).astype(float) * 1e6
    pl_d0_db = 20.0 * np.log10((4.0 * math.pi * reference_distance_m * freq_hz) / speed_light_m_per_s)
    return float(tx_power_dbm + tx_gain_dbi + rx_gain_dbi) - pl_d0_db


def _fixed_point_on_segments_mask(
    point: Point,
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
) -> np.ndarray:
    px, py = point
    orientation = ((end_x - start_x) * (py - start_y)) - ((end_y - start_y) * (px - start_x))
    return (
        (np.minimum(start_x, end_x) - GEOMETRY_EPSILON <= px)
        & (px <= np.maximum(start_x, end_x) + GEOMETRY_EPSILON)
        & (np.minimum(start_y, end_y) - GEOMETRY_EPSILON <= py)
        & (py <= np.maximum(start_y, end_y) + GEOMETRY_EPSILON)
        & (np.abs(orientation) <= GEOMETRY_EPSILON)
    )


def _points_on_fixed_segment_mask(
    point_x: np.ndarray,
    point_y: np.ndarray,
    start: Point,
    end: Point,
) -> np.ndarray:
    orientation = ((end[0] - start[0]) * (point_y - start[1])) - (
        (end[1] - start[1]) * (point_x - start[0])
    )
    return (
        (min(start[0], end[0]) - GEOMETRY_EPSILON <= point_x)
        & (point_x <= max(start[0], end[0]) + GEOMETRY_EPSILON)
        & (min(start[1], end[1]) - GEOMETRY_EPSILON <= point_y)
        & (point_y <= max(start[1], end[1]) + GEOMETRY_EPSILON)
        & (np.abs(orientation) <= GEOMETRY_EPSILON)
    )


def _segments_intersect_fixed_edge_mask(
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
    edge_start: Point,
    edge_end: Point,
) -> np.ndarray:
    o1 = ((end_x - start_x) * (edge_start[1] - start_y)) - (
        (end_y - start_y) * (edge_start[0] - start_x)
    )
    o2 = ((end_x - start_x) * (edge_end[1] - start_y)) - (
        (end_y - start_y) * (edge_end[0] - start_x)
    )
    o3 = ((edge_end[0] - edge_start[0]) * (start_y - edge_start[1])) - (
        (edge_end[1] - edge_start[1]) * (start_x - edge_start[0])
    )
    o4 = ((edge_end[0] - edge_start[0]) * (end_y - edge_start[1])) - (
        (edge_end[1] - edge_start[1]) * (end_x - edge_start[0])
    )

    general = (
        (((o1 > GEOMETRY_EPSILON) & (o2 < -GEOMETRY_EPSILON))
        | ((o1 < -GEOMETRY_EPSILON) & (o2 > GEOMETRY_EPSILON)))
        & (((o3 > GEOMETRY_EPSILON) & (o4 < -GEOMETRY_EPSILON))
        | ((o3 < -GEOMETRY_EPSILON) & (o4 > GEOMETRY_EPSILON)))
    )
    special = (
        _fixed_point_on_segments_mask(edge_start, start_x, start_y, end_x, end_y)
        | _fixed_point_on_segments_mask(edge_end, start_x, start_y, end_x, end_y)
        | _points_on_fixed_segment_mask(start_x, start_y, edge_start, edge_end)
        | _points_on_fixed_segment_mask(end_x, end_y, edge_start, edge_end)
    )
    return general | special


def _points_in_convex_polygon_mask(
    point_x: np.ndarray,
    point_y: np.ndarray,
    polygon: Sequence[Point],
) -> np.ndarray:
    all_positive = np.ones(len(point_x), dtype=bool)
    all_negative = np.ones(len(point_x), dtype=bool)
    for idx in range(len(polygon)):
        start = polygon[idx]
        end = polygon[(idx + 1) % len(polygon)]
        orientation = ((end[0] - start[0]) * (point_y - start[1])) - (
            (end[1] - start[1]) * (point_x - start[0])
        )
        all_positive &= orientation >= -GEOMETRY_EPSILON
        all_negative &= orientation <= GEOMETRY_EPSILON
    return all_positive | all_negative


def _segments_intersect_polygon_mask(
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
    polygon: Sequence[Point],
) -> np.ndarray:
    intersects = _points_in_convex_polygon_mask(
        start_x,
        start_y,
        polygon,
    ) | _points_in_convex_polygon_mask(
        end_x,
        end_y,
        polygon,
    )
    for idx in range(len(polygon)):
        edge_start = polygon[idx]
        edge_end = polygon[(idx + 1) % len(polygon)]
        intersects |= _segments_intersect_fixed_edge_mask(
            start_x,
            start_y,
            end_x,
            end_y,
            edge_start,
            edge_end,
        )
    return intersects


def _bbox_from_points(points: Iterable[Point]) -> BBox:
    points_list = list(points)
    xs = [point[0] for point in points_list]
    ys = [point[1] for point in points_list]
    return (min(xs), min(ys), max(xs), max(ys))


def _rotated_rect_corners(
    *,
    anchor_x: float,
    anchor_y: float,
    length: float,
    thickness: float,
    orientation: float,
) -> Tuple[Point, Point, Point, Point]:
    angle = math.radians(orientation % 360.0)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    local_corners = (
        (0.0, 0.0),
        (length, 0.0),
        (length, thickness),
        (0.0, thickness),
    )

    world_corners = []
    for x_local, y_local in local_corners:
        world_x = anchor_x + (x_local * cos_angle) - (y_local * sin_angle)
        world_y = anchor_y + (x_local * sin_angle) + (y_local * cos_angle)
        world_corners.append((world_x, world_y))

    return tuple(world_corners)  # type: ignore[return-value]


def _positive_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    resolved = float(value)
    if resolved <= 0:
        return None
    return resolved


def _material_geometries_from_floor_plan(data_dir: Union[str, Path]) -> list[_MaterialObstacle]:
    floor_plan_path = Path(data_dir) / "floor_plan_elements.parquet"
    if not floor_plan_path.exists():
        warnings.warn(
            f"{floor_plan_path} was not found. Door/window RSSI attenuation will be zero.",
            stacklevel=2,
        )
        return []

    floor_plan_df = pd.read_parquet(floor_plan_path)
    if floor_plan_df.empty:
        return []

    required_cols = {"element_type", "x", "y", "orientation_angle"}
    missing_cols = required_cols.difference(floor_plan_df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(f"floor_plan_elements.parquet is missing required columns: {missing_list}")

    element_type_series = floor_plan_df["element_type"].astype(str).str.lower()
    material_df = floor_plan_df[element_type_series.isin({"door", "window"})]
    geometries: list[_MaterialObstacle] = []

    for row in material_df.to_dict("records"):
        element_type = str(row["element_type"]).strip().lower()
        length: float | None
        thickness: float | None

        if element_type == "door":
            length = _positive_float(row.get("door_width")) or _positive_float(row.get("doorway_width"))
            thickness = _positive_float(row.get("thickness"))
        else:
            length = _positive_float(row.get("length"))
            thickness = (
                _positive_float(row.get("overall_thickness"))
                or _positive_float(row.get("single_line_thickness"))
            )

        if length is None or thickness is None:
            continue

        corners = _rotated_rect_corners(
            anchor_x=float(row["x"]),
            anchor_y=float(row["y"]),
            length=length,
            thickness=thickness,
            orientation=float(row["orientation_angle"]),
        )
        scenario_id = row.get("scenario_id")
        if scenario_id is not None and pd.isna(scenario_id):
            scenario_id = None
        geometries.append(
            _MaterialObstacle(
                scenario_id=scenario_id,
                obstacle_type=element_type,
                coordinates=corners,
                bbox=_bbox_from_points(corners),
            )
        )

    return geometries


def _door_window_blocker_counts(
    data_dir: Union[str, Path],
    links_df: pd.DataFrame,
) -> pd.DataFrame:
    counts_df = pd.DataFrame(
        {
            "door_blocker_count": pd.Series(0, index=links_df.index, dtype="int64"),
            "window_blocker_count": pd.Series(0, index=links_df.index, dtype="int64"),
        }
    )
    material_geometries = _material_geometries_from_floor_plan(data_dir)
    if not material_geometries or links_df.empty:
        return counts_df

    required_link_cols = {"scenario_id", "antenna_x", "antenna_y", "target_x", "target_y"}
    missing_cols = required_link_cols.difference(links_df.columns)
    if missing_cols:
        missing_list = ", ".join(sorted(missing_cols))
        raise ValueError(
            f"links.parquet is missing columns needed for door/window RSSI attenuation: {missing_list}"
        )

    global_obstacles = [obstacle for obstacle in material_geometries if obstacle.scenario_id is None]
    obstacles_by_scenario: dict[object, list[_MaterialObstacle]] = {}
    for obstacle in material_geometries:
        if obstacle.scenario_id is None:
            continue
        obstacles_by_scenario.setdefault(obstacle.scenario_id, []).append(obstacle)

    for scenario_id, scenario_links_df in links_df.groupby("scenario_id", sort=False):
        scenario_obstacles = [
            *global_obstacles,
            *obstacles_by_scenario.get(scenario_id, []),
        ]
        if not scenario_obstacles:
            continue

        antenna_x = scenario_links_df["antenna_x"].astype(float).to_numpy()
        antenna_y = scenario_links_df["antenna_y"].astype(float).to_numpy()
        target_x = scenario_links_df["target_x"].astype(float).to_numpy()
        target_y = scenario_links_df["target_y"].astype(float).to_numpy()
        min_x = np.minimum(antenna_x, target_x)
        max_x = np.maximum(antenna_x, target_x)
        min_y = np.minimum(antenna_y, target_y)
        max_y = np.maximum(antenna_y, target_y)
        door_counts = np.zeros(len(scenario_links_df), dtype=int)
        window_counts = np.zeros(len(scenario_links_df), dtype=int)

        for obstacle in scenario_obstacles:
            bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = obstacle.bbox
            bbox_mask = ~(
                (max_x < bbox_min_x - GEOMETRY_EPSILON)
                | (min_x > bbox_max_x + GEOMETRY_EPSILON)
                | (max_y < bbox_min_y - GEOMETRY_EPSILON)
                | (min_y > bbox_max_y + GEOMETRY_EPSILON)
            )
            candidate_indices = np.flatnonzero(bbox_mask)
            if len(candidate_indices) == 0:
                continue

            intersects = _segments_intersect_polygon_mask(
                antenna_x[candidate_indices],
                antenna_y[candidate_indices],
                target_x[candidate_indices],
                target_y[candidate_indices],
                obstacle.coordinates,
            )
            intersecting_indices = candidate_indices[intersects]
            if obstacle.obstacle_type == "door":
                door_counts[intersecting_indices] += 1
            else:
                window_counts[intersecting_indices] += 1

        counts_df.loc[scenario_links_df.index, "door_blocker_count"] = door_counts
        counts_df.loc[scenario_links_df.index, "window_blocker_count"] = window_counts

    return counts_df


def extract_rssi_link_inputs(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Return one row per antenna-target link with the scenario environment joined in.

    Required columns:
    - scenario_id
    - distance_m
    - link_state
    - target_space_type
    """
    data_path = Path(data_dir)
    summary_path = data_path / "env_summary.parquet"
    links_path = data_path / "links.parquet"

    summary_env_df = pd.read_parquet(summary_path, columns=["scenario_id", "env_type"])
    links_df = pd.read_parquet(links_path)

    required_link_cols = {"scenario_id", "distance_m", "link_state", "target_space_type"}
    missing_cols = required_link_cols.difference(links_df.columns)
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

    merged_df["rssi_env_type"] = [
        _resolve_rssi_env_type(target_space_type, link_state)
        for target_space_type, link_state in zip(
            merged_df["target_space_type"],
            merged_df["link_state"],
        )
    ]
    return merged_df.copy()


def create_scenario_rssi_parameters(
    link_inputs_df: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Sample RSSI parameters once per scenario and per derived RSSI environment
    class. Indoor scenarios can therefore have distinct LOS and NLOS parameter
    draws while preserving scenario-level consistency.
    """
    rng = random.Random(seed)
    scenario_rssi_df = link_inputs_df[["scenario_id", "rssi_env_type"]].drop_duplicates().copy()

    scenario_rssi_df["path_loss_exponent_n"] = scenario_rssi_df["rssi_env_type"].map(
        lambda env: path_loss_exponent_given_env_type(env, rng=rng)
    )
    scenario_rssi_df["shadow_sigma_db"] = scenario_rssi_df["rssi_env_type"].map(
        lambda env: shadow_sigma_given_env_type(env, rng=rng)
    )
    return scenario_rssi_df


def build_rssi_base_table(
    data_dir: Union[str, Path],
    *,
    seed: int | None = None,
    freq_mhz: float = 5500.0,
    tx_power_dbm: float = 0.0,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0,
    reference_distance_m: float = 1.0,
    wall_loss_db: float = DEFAULT_WALL_LOSS_DB,
    human_loss_db: float | None = None,
    door_loss_db: float = DEFAULT_DOOR_LOSS_DB,
    window_loss_db: float = DEFAULT_WINDOW_LOSS_DB,
) -> pd.DataFrame:
    """
    Return the shared link table with RSSI/path-loss columns added.

    Added columns:
    - rssi_env_type
    - path_loss_exponent_n
    - shadow_sigma_db
    - freq_mhz
    - tx_power_dbm
    - tx_gain_dbi
    - rx_gain_dbi
    - reference_distance_m
    - initial_signal_strength_dbm
    - wall_loss_db
    - human_loss_db
    - door_loss_db
    - window_loss_db
    - door_blocker_count
    - window_blocker_count
    - wall_attenuation_db
    - human_attenuation_db
    - door_attenuation_db
    - window_attenuation_db
    - obstacle_attenuation_db
    - shadow_noise_db
    - path_loss_db_with_noise
    - signal_strength_dbm
    """
    if reference_distance_m <= 0:
        raise ValueError("reference_distance_m must be > 0.")
    if freq_mhz <= 0:
        raise ValueError("freq_mhz must be > 0.")
    resolved_wall_loss_db = float(wall_loss_db)
    resolved_human_loss_db = (
        human_loss_db_at_freq(freq_mhz) if human_loss_db is None else float(human_loss_db)
    )
    resolved_door_loss_db = float(door_loss_db)
    resolved_window_loss_db = float(window_loss_db)
    if (
        resolved_wall_loss_db < 0
        or resolved_human_loss_db < 0
        or resolved_door_loss_db < 0
        or resolved_window_loss_db < 0
    ):
        raise ValueError(
            "wall_loss_db, human_loss_db, door_loss_db, and window_loss_db must be >= 0."
        )

    links_df = extract_rssi_link_inputs(data_dir)
    scenario_params_df = create_scenario_rssi_parameters(links_df, seed=seed)

    rssi_df = links_df.merge(
        scenario_params_df,
        on=["scenario_id", "rssi_env_type"],
        how="left",
        validate="many_to_one",
    )

    zero_distance_mask = rssi_df["distance_m"] <= 0
    if zero_distance_mask.any():
        bad_count = int(zero_distance_mask.sum())
        warnings.warn(
            f"Dropping {bad_count} link row(s) where distance_m <= 0 "
            "(target at same position as antenna). RSSI is undefined at zero distance.",
            stacklevel=2,
        )
        rssi_df = rssi_df[~zero_distance_mask].copy()

    np_rng = np.random.default_rng(seed)

    rssi_df["freq_mhz"] = float(freq_mhz)
    rssi_df["tx_power_dbm"] = float(tx_power_dbm)
    rssi_df["tx_gain_dbi"] = float(tx_gain_dbi)
    rssi_df["rx_gain_dbi"] = float(rx_gain_dbi)
    rssi_df["reference_distance_m"] = float(reference_distance_m)
    rssi_df["wall_loss_db"] = resolved_wall_loss_db
    rssi_df["human_loss_db"] = resolved_human_loss_db
    rssi_df["door_loss_db"] = resolved_door_loss_db
    rssi_df["window_loss_db"] = resolved_window_loss_db

    rssi_df["initial_signal_strength_dbm"] = _reference_rssi_dbm_from_freq(
        freq_mhz=rssi_df["freq_mhz"],
        tx_power_dbm=tx_power_dbm,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi,
        reference_distance_m=reference_distance_m,
    )

    wall_counts = (
        rssi_df["wall_blocker_count"].astype(float)
        if "wall_blocker_count" in rssi_df.columns
        else pd.Series(0.0, index=rssi_df.index)
    )
    human_counts = (
        rssi_df["human_blocker_count"].astype(float)
        if "human_blocker_count" in rssi_df.columns
        else pd.Series(0.0, index=rssi_df.index)
    )
    door_window_counts_df = _door_window_blocker_counts(data_dir, rssi_df)
    rssi_df["door_blocker_count"] = door_window_counts_df["door_blocker_count"]
    rssi_df["window_blocker_count"] = door_window_counts_df["window_blocker_count"]

    rssi_df["wall_attenuation_db"] = wall_counts * resolved_wall_loss_db
    rssi_df["human_attenuation_db"] = human_counts * resolved_human_loss_db
    rssi_df["door_attenuation_db"] = (
        rssi_df["door_blocker_count"].astype(float) * resolved_door_loss_db
    )
    rssi_df["window_attenuation_db"] = (
        rssi_df["window_blocker_count"].astype(float) * resolved_window_loss_db
    )
    rssi_df["obstacle_attenuation_db"] = (
        rssi_df["wall_attenuation_db"]
        + rssi_df["human_attenuation_db"]
        + rssi_df["door_attenuation_db"]
        + rssi_df["window_attenuation_db"]
    )

    # Shadowing is modelled as zero-mean Gaussian noise in the log-domain with
    # sigma sampled per scenario and per derived RSSI environment class.
    rssi_df["shadow_noise_db"] = np_rng.normal(
        loc=0.0,
        scale=rssi_df["shadow_sigma_db"].astype(float).to_numpy(),
        size=len(rssi_df),
    )

    path_loss_increment_db = (
        10.0
        * rssi_df["path_loss_exponent_n"].astype(float)
        * np.log10(rssi_df["distance_m"].astype(float) / reference_distance_m)
    )

    rssi_df["path_loss_db_with_noise"] = (
        path_loss_increment_db
        + rssi_df["obstacle_attenuation_db"].astype(float)
        + rssi_df["shadow_noise_db"].astype(float)
    )

    rssi_df["signal_strength_dbm"] = (
        rssi_df["initial_signal_strength_dbm"].astype(float)
        - rssi_df["path_loss_db_with_noise"].astype(float)
    )

    added_cols = [
        "path_loss_exponent_n",
        "shadow_sigma_db",
        "freq_mhz",
        "tx_power_dbm",
        "tx_gain_dbi",
        "rx_gain_dbi",
        "reference_distance_m",
        "initial_signal_strength_dbm",
        "wall_loss_db",
        "human_loss_db",
        "door_loss_db",
        "window_loss_db",
        "door_blocker_count",
        "window_blocker_count",
        "wall_attenuation_db",
        "human_attenuation_db",
        "door_attenuation_db",
        "window_attenuation_db",
        "obstacle_attenuation_db",
        "shadow_noise_db",
        "path_loss_db_with_noise",
        "signal_strength_dbm",
    ]
    ordered_cols = [*links_df.columns, *[column for column in added_cols if column not in links_df.columns]]
    return rssi_df[ordered_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RSSI/path-loss columns from env_summary.parquet and links.parquet."
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
        help="Optional seed for reproducible path-loss, sigma, and shadow-noise sampling.",
    )
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=5500.0,
        help="Carrier frequency in MHz used for the RSSI reference term.",
    )
    parser.add_argument(
        "--tx-power-dbm",
        type=float,
        default=0.0,
        help="Transmit power in dBm used to compute RSSI at the reference distance.",
    )
    parser.add_argument(
        "--tx-gain-dbi",
        type=float,
        default=0.0,
        help="Transmit antenna gain in dBi.",
    )
    parser.add_argument(
        "--rx-gain-dbi",
        type=float,
        default=0.0,
        help="Receive antenna gain in dBi.",
    )
    parser.add_argument(
        "--reference-distance-m",
        type=float,
        default=1.0,
        help="Reference distance d0 in meters used for the RSSI reference term.",
    )
    parser.add_argument(
        "--wall-loss-db",
        type=float,
        default=DEFAULT_WALL_LOSS_DB,
        help=(
            "Attenuation in dB per intersected wall. "
            f"Default: {DEFAULT_WALL_LOSS_DB} dB for ordinary brick; pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--human-loss-db",
        type=float,
        default=None,
        help=(
            "Attenuation in dB per intersected human obstacle. "
            "Default: frequency-projected from 17.22 dB at 2.45 GHz; pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--door-loss-db",
        type=float,
        default=DEFAULT_DOOR_LOSS_DB,
        help=(
            "Attenuation in dB per intersected door. "
            f"Default: {DEFAULT_DOOR_LOSS_DB} dB for solid wood; pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--window-loss-db",
        type=float,
        default=DEFAULT_WINDOW_LOSS_DB,
        help=(
            "Attenuation in dB per intersected window. "
            f"Default: {DEFAULT_WINDOW_LOSS_DB} dB for ordinary glass; pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: <data-dir>/links_rssi.parquet",
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
        output_path = data_dir / "links_rssi.parquet"

    rssi_df = build_rssi_base_table(
        data_dir=data_dir,
        seed=args.seed,
        freq_mhz=args.freq_mhz,
        tx_power_dbm=args.tx_power_dbm,
        tx_gain_dbi=args.tx_gain_dbi,
        rx_gain_dbi=args.rx_gain_dbi,
        reference_distance_m=args.reference_distance_m,
        wall_loss_db=args.wall_loss_db,
        human_loss_db=args.human_loss_db,
        door_loss_db=args.door_loss_db,
        window_loss_db=args.window_loss_db,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rssi_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(rssi_df)} RSSI link rows to {output_path}")
    if args.seed is None:
        print("Seed: none (sampling varies across runs).")
    else:
        print(f"Seed: {args.seed} (reproducible sampling).")


if __name__ == "__main__":
    main()
