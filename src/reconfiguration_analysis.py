"""Compare standard and self-reconfigured Sun-sensor attitude estimation.

Stage 1 of the self-reconfiguring CubeSat project:
- the failed Sun-sensor channel is assumed to be known from a channel-status
  diagnostic;
- the standard estimator still uses the corrupted six-channel vector;
- the reconfigured estimator excludes the failed channel and reconstructs the
  missing Sun-vector component when possible.

The next project stage will replace the known failure flag with automatic
per-sensor fault detection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attitude_estimation import (
    attitude_error_degrees,
    estimate_sun_vector,
    estimate_sun_vector_reconfigured,
    triad_attitude,
)
from fault_analysis import random_rotation_matrix
from magnetometer import Magnetometer
from sensor_parameters import (
    MAG_BIAS,
    MAG_NOISE_STD,
    MAG_SCALE_FACTORS,
    SUN_GAINS,
    SUN_NOISE_STD,
    SUN_OFFSETS,
)
from sun_sensor import six_face_sensor_array, unit_vector


SUN_SENSOR_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def compare_one_failed_sensor(
    failed_sensor: int,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, float | str]:
    """Compare standard and reconfigured TRIAD for one failed Sun sensor."""
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

    healthy_mask = np.ones(6, dtype=bool)
    healthy_mask[failed_sensor] = False

    standard_errors = []
    reconfigured_errors = []

    for _ in range(n_samples):
        true_rotation = random_rotation_matrix(rng)
        true_sun_body = true_rotation @ sun_reference
        true_magnetic_body = true_rotation @ magnetic_reference

        sun_readings = sun_array.measure(
            true_sun_body,
            noise_std=SUN_NOISE_STD,
            rng=rng,
        )
        sun_readings[failed_sensor] = 0.0

        magnetic_measurement = magnetometer.measure(
            true_magnetic_body,
            noise_std=MAG_NOISE_STD,
            rng=rng,
        )
        estimated_magnetic_body = unit_vector(magnetic_measurement)

        try:
            standard_sun = estimate_sun_vector(
                sun_array.sensor_normals,
                sun_readings,
            )
            standard_rotation = triad_attitude(
                sun_reference,
                magnetic_reference,
                standard_sun,
                estimated_magnetic_body,
            )
            standard_errors.append(
                attitude_error_degrees(true_rotation, standard_rotation)
            )
        except ValueError:
            pass

        try:
            reconfigured_sun = estimate_sun_vector_reconfigured(
                sun_array.sensor_normals,
                sun_readings,
                healthy_mask,
                gains=SUN_GAINS,
                offsets=SUN_OFFSETS,
            )
            reconfigured_rotation = triad_attitude(
                sun_reference,
                magnetic_reference,
                reconfigured_sun,
                estimated_magnetic_body,
            )
            reconfigured_errors.append(
                attitude_error_degrees(true_rotation, reconfigured_rotation)
            )
        except ValueError:
            pass

    standard_errors = np.asarray(standard_errors)
    reconfigured_errors = np.asarray(reconfigured_errors)

    return {
        "sensor": SUN_SENSOR_NAMES[failed_sensor],
        "standard_mean": float(np.mean(standard_errors)),
        "standard_p95": float(np.percentile(standard_errors, 95)),
        "reconfigured_mean": float(np.mean(reconfigured_errors)),
        "reconfigured_p95": float(np.percentile(reconfigured_errors, 95)),
        "improvement_percent": float(
            100.0
            * (
                np.mean(standard_errors) - np.mean(reconfigured_errors)
            )
            / np.mean(standard_errors)
        ),
    }


def save_comparison_plot(
    results: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    """Save standard-vs-reconfigured mean attitude error comparison."""
    names = [str(result["sensor"]) for result in results]
    standard = [float(result["standard_mean"]) for result in results]
    reconfigured = [float(result["reconfigured_mean"]) for result in results]

    x = np.arange(len(names))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(
        x - width / 2,
        standard,
        width,
        label="Standard estimator",
    )
    plt.bar(
        x + width / 2,
        reconfigured,
        width,
        label="Reconfigured estimator",
    )
    plt.xticks(x, names)
    plt.xlabel("Failed Sun-sensor face")
    plt.ylabel("Mean full-attitude error [deg]")
    plt.title("Benefit of Sun-sensor reconfiguration after a known failure")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    results = [
        compare_one_failed_sensor(index)
        for index in range(len(SUN_SENSOR_NAMES))
    ]

    print("\nKnown Sun-sensor failure: standard vs reconfigured estimator")
    print("-------------------------------------------------------------")
    print(
        "Failed   Standard mean   Standard p95   "
        "Reconfig mean   Reconfig p95   Improvement"
    )
    print(
        "sensor       [deg]           [deg]           "
        "[deg]          [deg]          [%]"
    )

    for result in results:
        print(
            f"{str(result['sensor']):>4s}      "
            f"{float(result['standard_mean']):8.3f}        "
            f"{float(result['standard_p95']):8.3f}        "
            f"{float(result['reconfigured_mean']):8.3f}       "
            f"{float(result['reconfigured_p95']):8.3f}       "
            f"{float(result['improvement_percent']):8.1f}"
        )

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sun_sensor_reconfiguration_comparison.png"
    save_comparison_plot(results, output_path)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
