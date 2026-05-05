"""
POSITION ESTIMATION DIAGNOSTIC PLOT
-----------------------------------

Render a geometry-only visualisation plot for one scenario/target/method using the
measurement tables and `position_estimates.parquet`.

Inputs:
- env_summary.parquet
- position_estimates.parquet
- links_rssi.parquet for RSSI
- links_tdoa.parquet for TDOA
- links_doa.parquet for DOA/AOA

Run using:
python3 "Data Generation/position_estimation_plot.py" --data-dir "Data Generation/generated_network_scenarios_with_plots" --scenario-id scenario_0001 --target-id 2 --method rssi

Output:
- A PNG plot saved to the requested path or to
  <data-dir>/plots/position_estimation/<scenario>_<target>_<method>.png
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union
import numpy as np
import pandas as pd


DEFAULT_REFERENCE_DISTANCE_M = 1.0
EPSILON = 1e-9
METHOD_LABELS = {
    "rssi": "RSSI",
    "tdoa": "TDOA",
    "doa": "DOA/AOA",
}


def _get_pyplot():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    import matplotlib.pyplot as plt

    return plt


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


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized == "aoa":
        return "doa"
    if normalized not in {"rssi", "tdoa", "doa"}:
        raise ValueError(f"Unsupported method: {method}")
    return normalized


def _parse_target_id(raw_target_id: str) -> object:
    try:
        return int(raw_target_id)
    except ValueError:
        return raw_target_id


def _filter_scenario_target(
    df: pd.DataFrame,
    scenario_id: str,
    target_id: object,
    table_name: str,
) -> pd.DataFrame:
    scenario_mask = df["scenario_id"].astype(str) == str(scenario_id)
    scenario_df = df.loc[scenario_mask].copy()
    if scenario_df.empty:
        raise ValueError(f"{table_name} does not contain scenario_id={scenario_id!r}.")

    target_mask = scenario_df["target_id"] == target_id
    target_df = scenario_df.loc[target_mask].copy()
    if target_df.empty:
        available_targets = ", ".join(map(str, sorted(scenario_df["target_id"].drop_duplicates().tolist())[:10]))
        raise ValueError(
            f"{table_name} does not contain target_id={target_id!r} within scenario_id={scenario_id!r}. "
            f"Available targets include: {available_targets}"
        )
    return target_df


def _scenario_bounds(env_summary_df: pd.DataFrame, scenario_id: str) -> tuple[float, float, float, float]:
    scenario_df = env_summary_df.loc[env_summary_df["scenario_id"].astype(str) == str(scenario_id)].copy()
    if scenario_df.empty:
        raise ValueError(f"env_summary.parquet does not contain scenario_id={scenario_id!r}.")
    first = scenario_df.iloc[0]
    return (
        float(first["x_domain_min"]),
        float(first["x_domain_max"]),
        float(first["y_range_min"]),
        float(first["y_range_max"]),
    )


def _method_columns(method: str) -> tuple[str, str, str, str, str]:
    return (
        f"{method}_est_x",
        f"{method}_est_y",
        f"{method}_error_m",
        f"{method}_residual_rmse_m",
        f"{method}_anchor_count",
    )


def _setup_axis(ax, bounds: tuple[float, float, float, float], title: str) -> None:
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2, linewidth=0.6)


def _scenario_figure_size(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    long_edge = 12.0
    short_edge = max(7.0, long_edge * min(width, height) / max(width, height))
    if width >= height:
        return (long_edge, short_edge)
    return (short_edge, long_edge)


def _finalize_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        ax.legend(
            uniq.values(),
            uniq.keys(),
            loc="upper right",
            fontsize=8,
            framealpha=0.95,
        )


def _annotate_metrics(ax, method: str, estimate_row: pd.Series) -> None:
    _, _, error_col, rmse_col, anchor_count_col = _method_columns(method)
    success_col = f"{method}_success"
    lines = [
        f"Method: {METHOD_LABELS[method]}",
        f"Success: {bool(estimate_row.get(success_col, False))}",
        f"Anchors: {estimate_row.get(anchor_count_col, 'n/a')}",
        f"Error (m): {_fmt_float(estimate_row.get(error_col))}",
        f"Residual RMSE (m): {_fmt_float(estimate_row.get(rmse_col))}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )


def _fmt_float(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.2f}"


def _plot_truth_and_estimate(ax, method: str, estimate_row: pd.Series) -> None:
    est_x_col, est_y_col, _, _, _ = _method_columns(method)
    success = bool(estimate_row.get(f"{method}_success", False))
    target_x = float(estimate_row["target_x"])
    target_y = float(estimate_row["target_y"])
    ax.scatter(
        target_x,
        target_y,
        color="tab:green",
        edgecolors="k",
        s=80,
        marker="o",
        label="ground truth",
        zorder=6,
    )

    est_values = np.array([estimate_row.get(est_x_col), estimate_row.get(est_y_col)], dtype=float)
    if success and np.isfinite(est_values).all():
        ax.scatter(
            est_values[0],
            est_values[1],
            color="tab:orange",
            edgecolors="k",
            s=90,
            marker="X",
            label="estimate",
            zorder=7,
        )
        ax.plot(
            [target_x, est_values[0]],
            [target_y, est_values[1]],
            linestyle="--",
            linewidth=1.2,
            color="0.35",
            label="truth-to-estimate error",
            zorder=5,
        )


def _rssi_range_estimates_m(group: pd.DataFrame) -> np.ndarray:
    reference_distance = (
        group["reference_distance_m"].astype(float).to_numpy()
        if "reference_distance_m" in group.columns
        else np.full(len(group), DEFAULT_REFERENCE_DISTANCE_M, dtype=float)
    )
    path_loss_db = (
        group["initial_signal_strength_dbm"].astype(float).to_numpy()
        - group["signal_strength_dbm"].astype(float).to_numpy()
    )
    exponent_n = group["path_loss_exponent_n"].astype(float).to_numpy()
    exponent = path_loss_db / np.maximum(10.0 * exponent_n, EPSILON)
    exponent = np.clip(exponent, -6.0, 6.0)
    return reference_distance * np.power(10.0, exponent)


def _plot_anchors(ax, anchors: np.ndarray, *, label: str, color: str, marker: str, size: float = 60.0) -> None:
    if len(anchors) == 0:
        return
    ax.scatter(
        anchors[:, 0],
        anchors[:, 1],
        color=color,
        edgecolors="k",
        s=size,
        marker=marker,
        label=label,
        zorder=6,
    )


def _plot_rssi(ax, group: pd.DataFrame) -> None:
    plt = _get_pyplot()
    anchors = group[["antenna_x", "antenna_y"]].astype(float).to_numpy()
    ranges_m = _rssi_range_estimates_m(group)
    _plot_anchors(ax, anchors, label="antenna", color="tab:blue", marker="^")

    for idx, ((x, y), radius) in enumerate(zip(anchors, ranges_m)):
        circle = plt.Circle(
            (float(x), float(y)),
            float(radius),
            facecolor="tab:blue",
            edgecolor="tab:blue",
            alpha=0.08,
            linewidth=1.0,
            label="estimated range" if idx == 0 else None,
            zorder=2,
        )
        ax.add_artist(circle)


def _plot_tdoa(ax, group: pd.DataFrame, bounds: tuple[float, float, float, float]) -> None:
    first = group.iloc[0]
    reference_anchor = np.array(
        [[float(first["reference_antenna_x"]), float(first["reference_antenna_y"])]],
        dtype=float,
    )
    comparison_anchors = group[["comparison_antenna_x", "comparison_antenna_y"]].astype(float).to_numpy()

    _plot_anchors(ax, reference_anchor, label="reference antenna", color="tab:purple", marker="s", size=70.0)
    _plot_anchors(ax, comparison_anchors, label="comparison antenna", color="tab:blue", marker="^")

    xmin, xmax, ymin, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    grid_size = 300
    x_values = np.linspace(xmin, xmax, grid_size)
    y_values = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(x_values, y_values)
    ref_x = float(first["reference_antenna_x"])
    ref_y = float(first["reference_antenna_y"])
    ref_distance = np.hypot(xx - ref_x, yy - ref_y)

    for idx, row in group.reset_index(drop=True).iterrows():
        cmp_x = float(row["comparison_antenna_x"])
        cmp_y = float(row["comparison_antenna_y"])
        delta_distance = (
            float(row["observed_tdoa_ns"]) * float(row["propagation_speed_m_per_s"]) / 1e9
        )
        cmp_distance = np.hypot(xx - cmp_x, yy - cmp_y)
        constraint = cmp_distance - ref_distance - delta_distance
        ax.contour(
            xx,
            yy,
            constraint,
            levels=[0.0],
            colors=["tab:red"],
            linewidths=1.0,
            alpha=0.75,
            linestyles="-",
            zorder=3,
        )
        if idx == 0:
            ax.plot([], [], color="tab:red", linewidth=1.0, label="TDOA hyperbola")

    ax.set_xlim(xmin - 0.02 * width, xmax + 0.02 * width)
    ax.set_ylim(ymin - 0.02 * height, ymax + 0.02 * height)


def _ray_to_bounds(
    origin_x: float,
    origin_y: float,
    angle_rad: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = bounds
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    t_values: list[float] = []

    if abs(dx) > EPSILON:
        t_values.extend([(xmin - origin_x) / dx, (xmax - origin_x) / dx])
    if abs(dy) > EPSILON:
        t_values.extend([(ymin - origin_y) / dy, (ymax - origin_y) / dy])

    valid_points = []
    for t_value in t_values:
        if t_value <= 0:
            continue
        x = origin_x + (dx * t_value)
        y = origin_y + (dy * t_value)
        if xmin - EPSILON <= x <= xmax + EPSILON and ymin - EPSILON <= y <= ymax + EPSILON:
            valid_points.append((t_value, x, y))

    if not valid_points:
        return (origin_x, origin_y)

    _, x_end, y_end = max(valid_points, key=lambda item: item[0])
    return (float(x_end), float(y_end))


def _plot_doa(ax, group: pd.DataFrame, bounds: tuple[float, float, float, float]) -> None:
    anchors = group[["antenna_x", "antenna_y"]].astype(float).to_numpy()
    _plot_anchors(ax, anchors, label="antenna", color="tab:blue", marker="^")

    valid_group = group.copy()
    finite_mask = np.isfinite(valid_group["observed_bearing_rad"].astype(float))
    if "is_doa_valid" in valid_group.columns:
        valid_mask = valid_group["is_doa_valid"].fillna(False).astype(bool) & finite_mask
    else:
        valid_mask = finite_mask

    valid_group = valid_group.loc[valid_mask].reset_index(drop=True)
    if valid_group.empty:
        return

    for idx, row in valid_group.iterrows():
        start_x = float(row["antenna_x"])
        start_y = float(row["antenna_y"])
        end_x, end_y = _ray_to_bounds(
            start_x,
            start_y,
            float(row["observed_bearing_rad"]),
            bounds,
        )
        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            color="tab:red",
            linewidth=1.3,
            alpha=0.8,
            label="bearing ray" if idx == 0 else None,
            zorder=4,
        )


def _default_output_path(data_dir: Union[str, Path], scenario_id: str, target_id: object, method: str) -> Path:
    return (
        Path(data_dir)
        / "plots"
        / "position_estimation"
        / f"{scenario_id}_{target_id}_{method}.png"
    )


def _plot_method(
    data_dir: Union[str, Path],
    scenario_id: str,
    target_id: object,
    method: str,
    output_method_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    env_summary_df = _read_table(
        data_dir,
        "env_summary",
        ["scenario_id", "x_domain_min", "x_domain_max", "y_range_min", "y_range_max"],
    )
    estimates_required = [
        "scenario_id",
        "target_id",
        "target_x",
        "target_y",
        *_method_columns(method),
        f"{method}_success",
    ]
    estimates_df = _read_table(data_dir, "position_estimates", estimates_required)
    estimate_group = _filter_scenario_target(estimates_df, scenario_id, target_id, "position_estimates.parquet")
    estimate_row = estimate_group.iloc[0]
    bounds = _scenario_bounds(env_summary_df, scenario_id)

    if method == "rssi":
        measurement_df = _read_table(
            data_dir,
            "links_rssi",
            [
                "scenario_id",
                "target_id",
                "antenna_x",
                "antenna_y",
                "signal_strength_dbm",
                "initial_signal_strength_dbm",
                "path_loss_exponent_n",
            ],
        )
        measurement_group = _filter_scenario_target(
            measurement_df, scenario_id, target_id, "links_rssi.parquet"
        )
    elif method == "tdoa":
        measurement_df = _read_table(
            data_dir,
            "links_tdoa",
            [
                "scenario_id",
                "target_id",
                "reference_antenna_x",
                "reference_antenna_y",
                "comparison_antenna_x",
                "comparison_antenna_y",
                "observed_tdoa_ns",
                "propagation_speed_m_per_s",
            ],
        )
        measurement_group = _filter_scenario_target(
            measurement_df, scenario_id, target_id, "links_tdoa.parquet"
        )
    else:
        measurement_df = _read_table(
            data_dir,
            "links_doa",
            [
                "scenario_id",
                "target_id",
                "antenna_x",
                "antenna_y",
                "observed_bearing_rad",
            ],
        )
        measurement_group = _filter_scenario_target(
            measurement_df, scenario_id, target_id, "links_doa.parquet"
        )

    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=_scenario_figure_size(bounds))
    success = bool(estimate_row.get(f"{method}_success", False))
    title = (
        f"{METHOD_LABELS[method]} visualisation | scenario={scenario_id} | "
        f"target={target_id} | success={success}"
    )
    _setup_axis(ax, bounds, title)

    if method == "rssi":
        _plot_rssi(ax, measurement_group)
    elif method == "tdoa":
        _plot_tdoa(ax, measurement_group, bounds)
    else:
        _plot_doa(ax, measurement_group, bounds)

    _plot_truth_and_estimate(ax, method, estimate_row)
    _annotate_metrics(ax, method, estimate_row)
    _finalize_legend(ax)

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(
            data_dir,
            scenario_id,
            target_id,
            output_method_name or method,
        )
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(resolved_output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return resolved_output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a position-estimation visualisation plot for one scenario, target, and method."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="generated_network_scenarios",
        help="Directory containing env_summary.parquet, position_estimates.parquet, and method tables.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        required=True,
        help="Scenario identifier to visualise.",
    )
    parser.add_argument(
        "--target-id",
        type=str,
        required=True,
        help="Target identifier to visualise.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["rssi", "tdoa", "doa", "aoa"],
        help="Positioning method to visualise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Default: <data-dir>/plots/position_estimation/<scenario>_<target>_<method>.png",
    )
    args = parser.parse_args()

    method = _normalize_method(args.method)
    target_id = _parse_target_id(args.target_id)
    output_path = _plot_method(
        data_dir=Path(args.data_dir),
        scenario_id=args.scenario_id,
        target_id=target_id,
        method=method,
        output_method_name=str(args.method).strip().lower(),
        output_path=args.output,
    )
    print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
