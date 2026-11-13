"""Systematic magnetometer-disturbance sweep for the MA4822 CubeSat project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fault_analysis import run_scenario
from sensor_parameters import (
    EARTH_FIELD_UT,
    MAG_DISTURBANCE_DIRECTION,
)


def run_magnetic_disturbance_sweep(
    n_samples: int = 1000,
    disturbance_levels_ut: tuple[float, ...] = (
        0.0,
        1.25,
        2.5,
        3.75,
        5.0,
        7.5,
    ),
    seed: int = 42,
) -> list[dict[str, np.ndarray | str | float]]:
    """Sweep magnetic disturbance magnitude along one fixed direction."""
    results: list[dict[str, np.ndarray | str | float]] = []

    for level_ut in disturbance_levels_ut:
        normalized_level = level_ut / EARTH_FIELD_UT
        disturbance = normalized_level * MAG_DISTURBANCE_DIRECTION

        result = run_scenario(
            name=f"Mag dist {level_ut:.2f} uT",
            n_samples=n_samples,
            seed=seed,
            magnetic_disturbance=disturbance,
        )

        result["disturbance_ut"] = float(level_ut)
        results.append(result)

    return results


def print_sweep_summary(
    results: list[dict[str, np.ndarray | str | float]],
) -> None:
    """Print the main metrics for each disturbance level."""
    print("\nMagnetometer disturbance sweep")
    print("-------------------------------")
    print(
        "Disturbance   Mean err   95th err   Mean rel.   Low rel. (<40)"
    )
    print(
        "[uT]          [deg]      [deg]       [%]         [%]"
    )

    for result in results:
        errors = result["attitude_error"]
        reliability = result["reliability"]
        level_ut = float(result["disturbance_ut"])

        print(
            f"{level_ut:10.2f}   "
            f"{np.mean(errors):9.3f}   "
            f"{np.percentile(errors, 95):9.3f}   "
            f"{np.mean(reliability):9.2f}   "
            f"{100.0 * np.mean(reliability < 40.0):12.1f}"
        )


def save_disturbance_sweep_plot(
    results: list[dict[str, np.ndarray | str | float]],
    output_path: Path,
) -> None:
    """Plot attitude error and fuzzy reliability versus magnetic disturbance."""
    levels_ut = np.array([
        float(result["disturbance_ut"])
        for result in results
    ])
    mean_errors = np.array([
        np.mean(result["attitude_error"])
        for result in results
    ])
    p95_errors = np.array([
        np.percentile(result["attitude_error"], 95)
        for result in results
    ])
    mean_reliability = np.array([
        np.mean(result["reliability"])
        for result in results
    ])

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    ax1.plot(
        levels_ut,
        mean_errors,
        marker="o",
        color="steelblue",
        label="Mean attitude error",
    )
    ax1.plot(
        levels_ut,
        p95_errors,
        marker="s",
        color="slateblue",
        label="95th-percentile attitude error",
    )
    ax1.set_xlabel("Magnetic disturbance [microtesla]")
    ax1.set_ylabel("Attitude error [deg]")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(
        levels_ut,
        mean_reliability,
        marker="^",
        linestyle="--",
        color="darkorange",
        label="Mean fuzzy reliability",
    )
    ax2.set_ylabel("Mean fuzzy reliability [%]")
    ax2.set_ylim(0, 100)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")

    plt.title("Effect of magnetometer disturbance magnitude")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_low_reliability_plot(
    results: list[dict[str, np.ndarray | str | float]],
    output_path: Path,
) -> None:
    """Plot the fraction of simulations classified as low reliability."""
    levels_ut = np.array([
        float(result["disturbance_ut"])
        for result in results
    ])
    low_fraction = np.array([
        100.0 * np.mean(result["reliability"] < 40.0)
        for result in results
    ])

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        levels_ut,
        low_fraction,
        marker="o",
        color="crimson",
    )
    plt.xlabel("Magnetic disturbance [microtesla]")
    plt.ylabel("Low-reliability cases [%]")
    plt.title("Fuzzy fault detection versus magnetic disturbance")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    results = run_magnetic_disturbance_sweep()
    print_sweep_summary(results)

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = output_dir / "magnetometer_disturbance_sweep.png"
    detection_path = output_dir / "magnetometer_fault_detection.png"

    save_disturbance_sweep_plot(results, sweep_path)
    save_low_reliability_plot(results, detection_path)

    print("\nSaved figures:")
    print(sweep_path)
    print(detection_path)


if __name__ == "__main__":
    main()
