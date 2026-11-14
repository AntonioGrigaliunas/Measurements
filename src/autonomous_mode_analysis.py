"""Benchmark autonomous NORMAL / RECONFIGURED / SUN_SAFE operation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attitude_estimation import attitude_error_degrees
from autonomous_attitude import autonomous_attitude_solution
from fault_analysis import random_rotation_matrix
from magnetometer import Magnetometer
from sensor_parameters import (
    MAG_BIAS,
    MAG_FAULT_VECTOR,
    MAG_NOISE_STD,
    MAG_SCALE_FACTORS,
    SUN_GAINS,
    SUN_NOISE_STD,
    SUN_OFFSETS,
)
from sun_sensor import six_face_sensor_array, unit_vector


def run_mode_scenario(
    name: str,
    n_samples: int = 1000,
    seed: int = 42,
    failed_sun_sensor: int | None = None,
    magnetic_disturbance: np.ndarray | None = None,
) -> dict[str, object]:
    """Run one autonomous-operating-mode scenario."""
    rng = np.random.default_rng(seed)

    sun_reference = unit_vector(np.array([1.0, 0.4, 0.2]))
    magnetic_reference = unit_vector(np.array([0.2, -0.3, 0.9]))

    sun_array = six_face_sensor_array(
        gains=SUN_GAINS,
        offsets=SUN_OFFSETS,
    )
    magnetometer = Magnetometer(
        scale_factors=MAG_SCALE_FACTORS,
        bias=MAG_BIAS,
    )

    if magnetic_disturbance is None:
        magnetic_disturbance = np.zeros(3)

    modes = Counter()
    full_attitude_errors = []
    unsafe_full_attitude_count = 0

    for _ in range(n_samples):
        true_rotation = random_rotation_matrix(rng)
        true_sun_body = true_rotation @ sun_reference
        true_magnetic_body = true_rotation @ magnetic_reference

        sun_readings = sun_array.measure(
            true_sun_body,
            noise_std=SUN_NOISE_STD,
            rng=rng,
        )
        if failed_sun_sensor is not None:
            sun_readings[failed_sun_sensor] = 0.0

        magnetic_measurement = magnetometer.measure(
            true_magnetic_body,
            noise_std=MAG_NOISE_STD,
            rng=rng,
        )
        magnetic_measurement = magnetic_measurement + magnetic_disturbance

        solution = autonomous_attitude_solution(
            sun_array.sensor_normals,
            sun_readings,
            magnetic_measurement,
            sun_reference,
            magnetic_reference,
            gains=SUN_GAINS,
            offsets=SUN_OFFSETS,
        )

        mode = str(solution["mode"])
        modes[mode] += 1

        if solution["rotation"] is not None:
            error = attitude_error_degrees(
                true_rotation,
                solution["rotation"],
            )
            full_attitude_errors.append(error)
            if error > 10.0:
                unsafe_full_attitude_count += 1

    full_attitude_errors = np.asarray(full_attitude_errors)

    result: dict[str, object] = {
        "name": name,
        "modes": modes,
        "full_attitude_fraction": len(full_attitude_errors) / n_samples,
        "mean_full_attitude_error": (
            float(np.mean(full_attitude_errors))
            if len(full_attitude_errors)
            else np.nan
        ),
        "p95_full_attitude_error": (
            float(np.percentile(full_attitude_errors, 95))
            if len(full_attitude_errors)
            else np.nan
        ),
        "unsafe_full_attitude_fraction": (
            unsafe_full_attitude_count / n_samples
        ),
    }
    return result


def print_summary(results: list[dict[str, object]], n_samples: int) -> None:
    print("\nAutonomous CubeSat attitude-system mode benchmark")
    print("-------------------------------------------------")
    print(
        "Scenario          NORMAL  RECONFIG  SUN_SAFE  SAFE   "
        "Full att.   Mean err   P95 err   Unsafe >10deg"
    )
    print(
        "                    [%]      [%]       [%]     [%]      "
        "[%]       [deg]      [deg]          [%]"
    )

    for result in results:
        modes = result["modes"]
        print(
            f"{str(result['name']):16s} "
            f"{100*modes['NORMAL']/n_samples:7.1f}  "
            f"{100*modes['RECONFIGURED']/n_samples:8.1f}  "
            f"{100*modes['SUN_SAFE']/n_samples:8.1f}  "
            f"{100*modes['SAFE']/n_samples:5.1f}  "
            f"{100*float(result['full_attitude_fraction']):8.1f}  "
            f"{float(result['mean_full_attitude_error']):9.3f}  "
            f"{float(result['p95_full_attitude_error']):8.3f}      "
            f"{100*float(result['unsafe_full_attitude_fraction']):8.2f}"
        )


def save_mode_plot(
    results: list[dict[str, object]],
    n_samples: int,
    output_path: Path,
) -> None:
    names = [str(result["name"]) for result in results]
    mode_names = ["NORMAL", "RECONFIGURED", "SUN_SAFE", "SAFE"]

    x = np.arange(len(names))
    bottom = np.zeros(len(names))

    plt.figure(figsize=(9, 5.5))
    for mode in mode_names:
        values = np.array(
            [
                100.0 * result["modes"][mode] / n_samples
                for result in results
            ]
        )
        plt.bar(x, values, bottom=bottom, label=mode)
        bottom += values

    plt.xticks(x, names, rotation=10)
    plt.ylabel("Fraction of simulations [%]")
    plt.title("Autonomous operating-mode selection")
    plt.ylim(0, 100)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    n_samples = 1000
    seed = 42

    results = [
        run_mode_scenario(
            "Healthy",
            n_samples=n_samples,
            seed=seed,
        ),
        run_mode_scenario(
            "Sun fault",
            n_samples=n_samples,
            seed=seed,
            failed_sun_sensor=0,
        ),
        run_mode_scenario(
            "Mag fault",
            n_samples=n_samples,
            seed=seed,
            magnetic_disturbance=MAG_FAULT_VECTOR,
        ),
        run_mode_scenario(
            "Combined",
            n_samples=n_samples,
            seed=seed,
            failed_sun_sensor=0,
            magnetic_disturbance=MAG_FAULT_VECTOR,
        ),
    ]

    print_summary(results, n_samples)

    print(
        "\nInterpretation: full 3D attitude is intentionally withheld in "
        "SUN_SAFE mode. The system still returns the Sun direction for "
        "survival/power-pointing functions instead of trusting a corrupted "
        "magnetometer."
    )

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "autonomous_operating_modes.png"
    save_mode_plot(results, n_samples, output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
