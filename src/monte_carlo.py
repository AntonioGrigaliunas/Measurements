"""Monte Carlo noise-sensitivity study for the CubeSat Sun-sensor project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attitude_estimation import angular_error_degrees, estimate_sun_vector
from sun_sensor import six_face_sensor_array


def random_unit_vectors(
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate uniformly distributed random directions on the unit sphere."""
    vectors = rng.normal(size=(n_samples, 3))
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / norms[:, None]


def run_noise_sensitivity(
    n_samples: int = 1000,
    noise_levels: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05),
    seed: int = 42,
) -> dict[float, np.ndarray]:
    """Return angular-error samples for each measurement-noise level."""
    rng = np.random.default_rng(seed)

    gains = np.array([1.03, 0.98, 0.96, 1.02, 1.05, 0.97])
    offsets = np.array([0.010, 0.005, 0.000, 0.008, 0.012, 0.004])

    sensor_array = six_face_sensor_array(
        gains=gains,
        offsets=offsets,
    )

    sun_vectors = random_unit_vectors(n_samples, rng)

    results: dict[float, np.ndarray] = {}

    for noise_std in noise_levels:
        errors = []

        for true_sun_vector in sun_vectors:
            readings = sensor_array.measure(
                true_sun_vector,
                noise_std=noise_std,
                rng=rng,
            )

            try:
                estimated_sun_vector = estimate_sun_vector(
                    sensor_array.sensor_normals,
                    readings,
                )
            except ValueError:
                continue

            error_deg = angular_error_degrees(
                true_sun_vector,
                estimated_sun_vector,
            )
            errors.append(error_deg)

        results[noise_std] = np.asarray(errors)

    return results


def print_summary(results: dict[float, np.ndarray]) -> None:
    """Print the main error statistics for every noise level."""
    print("\nMonte Carlo noise-sensitivity results")
    print("------------------------------------")
    print("Noise std   Mean error   Std dev   Max error")
    print("            [deg]        [deg]     [deg]")

    for noise_std, errors in results.items():
        print(
            f"{noise_std:8.3f}   "
            f"{np.mean(errors):10.4f}   "
            f"{np.std(errors):7.4f}   "
            f"{np.max(errors):9.4f}"
        )


def save_mean_error_plot(
    results: dict[float, np.ndarray],
    output_path: Path,
) -> None:
    """Save mean angular error as a function of sensor-noise level."""
    noise_levels = np.array(list(results.keys()))
    mean_errors = np.array(
        [np.mean(errors) for errors in results.values()]
    )

    plt.figure(figsize=(7, 4.5))
    plt.plot(noise_levels, mean_errors, marker="o")
    plt.xlabel("Measurement noise standard deviation")
    plt.ylabel("Mean angular error [deg]")
    plt.title("Sun-vector estimation sensitivity to sensor noise")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_error_histogram(
    errors: np.ndarray,
    noise_std: float,
    output_path: Path,
) -> None:
    """Save the angular-error distribution for one selected noise level."""
    plt.figure(figsize=(7, 4.5))
    plt.hist(errors, bins=30)
    plt.xlabel("Angular error [deg]")
    plt.ylabel("Number of simulations")
    plt.title(
        f"Angular-error distribution at noise std = {noise_std:.3f}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    results = run_noise_sensitivity()

    print_summary(results)

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_mean_error_plot(
        results,
        output_dir / "mean_error_vs_noise.png",
    )

    selected_noise = 0.02
    save_error_histogram(
        results[selected_noise],
        selected_noise,
        output_dir / "angular_error_histogram_noise_002.png",
    )

    print("\nSaved figures:")
    print(output_dir / "mean_error_vs_noise.png")
    print(output_dir / "angular_error_histogram_noise_002.png")


if __name__ == "__main__":
    main()
