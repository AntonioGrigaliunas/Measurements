"""Monte Carlo full-attitude sensitivity study for the MA4822 CubeSat project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attitude_estimation import (
    attitude_error_degrees,
    estimate_sun_vector,
    triad_attitude,
)
from magnetometer import Magnetometer
from sun_sensor import six_face_sensor_array, unit_vector


def random_rotation_matrices(
    n_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate uniformly distributed random 3D rotation matrices."""
    rotations: list[np.ndarray] = []

    for _ in range(n_samples):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        w, x, y, z = q

        rotation = np.array(
            [
                [
                    1 - 2 * (y**2 + z**2),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x**2 + z**2),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x**2 + y**2),
                ],
            ]
        )
        rotations.append(rotation)

    return rotations


def run_attitude_noise_sensitivity(
    n_samples: int = 1000,
    sun_noise_levels: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05),
    magnetometer_noise_levels: tuple[float, ...] = (\n        0.0,\n        MAG_NOISE_STD / 2.0,\n        MAG_NOISE_STD,\n        2.0 * MAG_NOISE_STD,\n        4.0 * MAG_NOISE_STD,\n    ),
    seed: int = 42,
) -> dict[tuple[float, float], np.ndarray]:
    """Return full-attitude error samples for each pair of sensor-noise levels."""
    rng = np.random.default_rng(seed)

    # Fixed environmental directions in the reference frame.
    sun_reference = unit_vector(np.array([1.0, 0.4, 0.2]))
    magnetic_reference = unit_vector(np.array([0.2, -0.3, 0.9]))

    # Sun-sensor calibration mismatch.
    sun_gains = np.array([1.03, 0.98, 0.96, 1.02, 1.05, 0.97])
    sun_offsets = np.array([0.010, 0.005, 0.000, 0.008, 0.012, 0.004])

    sun_array = six_face_sensor_array(
        gains=sun_gains,
        offsets=sun_offsets,
    )

    # Magnetometer scale-factor mismatch and bias.
    magnetometer = Magnetometer(
        scale_factors=np.array([1.01, 0.99, 1.02]),
        bias=np.array([0.005, -0.004, 0.003]),
    )

    true_rotations = random_rotation_matrices(n_samples, rng)

    results: dict[tuple[float, float], np.ndarray] = {}

    for sun_noise in sun_noise_levels:
        for mag_noise in magnetometer_noise_levels:
            attitude_errors = []

            for true_rotation in true_rotations:
                true_sun_body = true_rotation @ sun_reference
                true_magnetic_body = true_rotation @ magnetic_reference

                sun_readings = sun_array.measure(
                    true_sun_body,
                    noise_std=sun_noise,
                    rng=rng,
                )

                magnetic_measurement = magnetometer.measure(
                    true_magnetic_body,
                    noise_std=mag_noise,
                    rng=rng,
                )

                try:
                    estimated_sun_body = estimate_sun_vector(
                        sun_array.sensor_normals,
                        sun_readings,
                    )
                    estimated_magnetic_body = unit_vector(
                        magnetic_measurement
                    )

                    estimated_rotation = triad_attitude(
                        sun_reference,
                        magnetic_reference,
                        estimated_sun_body,
                        estimated_magnetic_body,
                    )
                except ValueError:
                    continue

                attitude_errors.append(
                    attitude_error_degrees(
                        true_rotation,
                        estimated_rotation,
                    )
                )

            results[(sun_noise, mag_noise)] = np.asarray(attitude_errors)

    return results


def print_summary(
    results: dict[tuple[float, float], np.ndarray],
) -> None:
    """Print full-attitude error statistics for every sensor-noise pair."""
    print("\nMonte Carlo full-attitude sensitivity results")
    print("--------------------------------------------")
    print(
        "Sun noise   Mag noise   Mean error   Std dev   "
        "95th pct   Max error"
    )
    print(
        "                        [deg]        [deg]     "
        "[deg]      [deg]"
    )

    for (sun_noise, mag_noise), errors in results.items():
        print(
            f"{sun_noise:8.3f}   "
            f"{mag_noise:8.3f}   "
            f"{np.mean(errors):10.4f}   "
            f"{np.std(errors):7.4f}   "
            f"{np.percentile(errors, 95):8.4f}   "
            f"{np.max(errors):9.4f}"
        )


def save_noise_sensitivity_plot(
    results: dict[tuple[float, float], np.ndarray],
    output_path: Path,
) -> None:
    """Plot mean attitude error against Sun-sensor noise for each mag noise."""
    sun_levels = sorted({key[0] for key in results})
    mag_levels = sorted({key[1] for key in results})

    plt.figure(figsize=(7.5, 5.0))

    for mag_noise in mag_levels:
        mean_errors = [
            np.mean(results[(sun_noise, mag_noise)])
            for sun_noise in sun_levels
        ]
        plt.plot(
            sun_levels,
            mean_errors,
            marker="o",
            label=f"Mag noise = {mag_noise:.3f}",
        )

    plt.xlabel("Sun-sensor noise standard deviation")
    plt.ylabel("Mean full-attitude error [deg]")
    plt.title("Attitude sensitivity to Sun-sensor and magnetometer noise")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_attitude_error_histogram(
    errors: np.ndarray,
    sun_noise: float,
    mag_noise: float,
    output_path: Path,
) -> None:
    """Save an attitude-error distribution for one representative case."""
    plt.figure(figsize=(7, 4.5))
    plt.hist(errors, bins=30)
    plt.xlabel("Full-attitude error [deg]")
    plt.ylabel("Number of simulations")
    plt.title(
        "Attitude-error distribution "
        f"(Sun noise={sun_noise:.3f}, Mag noise={mag_noise:.3f})"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    results = run_attitude_noise_sensitivity()

    print_summary(results)

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_path = output_dir / "attitude_error_vs_sensor_noise.png"
    save_noise_sensitivity_plot(
        results,
        sensitivity_path,
    )

    selected_case = (0.02, MAG_NOISE_STD)
    histogram_path = output_dir / "attitude_error_histogram.png"
    save_attitude_error_histogram(
        results[selected_case],
        selected_case[0],
        selected_case[1],
        histogram_path,
    )

    print("\nSaved figures:")
    print(sensitivity_path)
    print(histogram_path)


if __name__ == "__main__":
    main()
