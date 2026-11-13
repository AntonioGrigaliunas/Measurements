"""Fault analysis with fuzzy reliability assessment for the MA4822 project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from attitude_estimation import (
    angular_error_degrees,
    attitude_error_degrees,
    estimate_sun_vector,
    triad_attitude,
)
from fuzzy_reliability import fuzzy_reliability_score, reliability_label
from magnetometer import Magnetometer
from sun_sensor import six_face_sensor_array, unit_vector


SUN_SENSOR_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate one uniformly distributed random 3D rotation matrix."""
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    return np.array(
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


def sun_reconstruction_residual(
    sensor_array,
    readings: np.ndarray,
    estimated_sun_body: np.ndarray,
) -> float:
    """Return RMS mismatch between measured and reconstructed Sun readings."""
    ideal_prediction = np.maximum(
        0.0,
        sensor_array.sensor_normals @ estimated_sun_body,
    )

    predicted_readings = (
        sensor_array.gains * ideal_prediction
        + sensor_array.offsets
    )
    predicted_readings = np.maximum(0.0, predicted_readings)

    residuals = readings - predicted_readings
    return float(np.sqrt(np.mean(residuals**2)))


def run_scenario(
    name: str,
    n_samples: int,
    seed: int,
    failed_sun_sensor: int | None = None,
    magnetic_disturbance: np.ndarray | None = None,
) -> dict[str, np.ndarray | str]:
    """Simulate one healthy/faulted sensor scenario."""
    rng = np.random.default_rng(seed)

    sun_reference = unit_vector(np.array([1.0, 0.4, 0.2]))
    magnetic_reference = unit_vector(np.array([0.2, -0.3, 0.9]))

    sun_gains = np.array([1.03, 0.98, 0.96, 1.02, 1.05, 0.97])
    sun_offsets = np.array([0.010, 0.005, 0.000, 0.008, 0.012, 0.004])

    sun_array = six_face_sensor_array(
        gains=sun_gains,
        offsets=sun_offsets,
    )

    magnetometer = Magnetometer(
        scale_factors=np.array([1.01, 0.99, 1.02]),
        bias=np.array([0.005, -0.004, 0.003]),
    )

    if magnetic_disturbance is None:
        magnetic_disturbance = np.zeros(3)

    attitude_errors = []
    reliability_scores = []
    consistency_errors = []
    magnetic_norm_errors = []
    sun_residuals = []

    healthy_sun_sensors = 6 if failed_sun_sensor is None else 5
    sun_health_fraction = healthy_sun_sensors / 6.0

    reference_separation = angular_error_degrees(
        sun_reference,
        magnetic_reference,
    )

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

        try:
            estimated_sun_body = estimate_sun_vector(
                sun_array.sensor_normals,
                sun_readings,
            )
            estimated_magnetic_body = unit_vector(magnetic_measurement)

            estimated_rotation = triad_attitude(
                sun_reference,
                magnetic_reference,
                estimated_sun_body,
                estimated_magnetic_body,
            )
        except ValueError:
            continue

        attitude_error = attitude_error_degrees(
            true_rotation,
            estimated_rotation,
        )

        measured_separation = angular_error_degrees(
            estimated_sun_body,
            estimated_magnetic_body,
        )
        consistency_error = abs(
            measured_separation - reference_separation
        )

        magnetic_norm_error = abs(
            np.linalg.norm(magnetic_measurement) - 1.0
        )

        sun_residual = sun_reconstruction_residual(
            sun_array,
            sun_readings,
            estimated_sun_body,
        )

        reliability, _ = fuzzy_reliability_score(
            sun_health_fraction,
            consistency_error,
            magnetic_norm_error,
            sun_residual,
        )

        attitude_errors.append(attitude_error)
        reliability_scores.append(reliability)
        consistency_errors.append(consistency_error)
        magnetic_norm_errors.append(magnetic_norm_error)
        sun_residuals.append(sun_residual)

    return {
        "name": name,
        "attitude_error": np.asarray(attitude_errors),
        "reliability": np.asarray(reliability_scores),
        "consistency_error": np.asarray(consistency_errors),
        "magnetic_norm_error": np.asarray(magnetic_norm_errors),
        "sun_residual": np.asarray(sun_residuals),
    }


def print_summary(results: list[dict[str, np.ndarray | str]]) -> None:
    """Print a compact comparison of fault scenarios."""
    print("\nFault analysis with improved fuzzy reliability")
    print("----------------------------------------------")
    print(
        "Scenario                 Mean err   95th err   Mean rel.   "
        "Low rel.   Sun residual"
    )
    print(
        "                         [deg]      [deg]       [%]         "
        "<40 [%]    RMS"
    )

    for result in results:
        errors = result["attitude_error"]
        reliability = result["reliability"]
        residual = result["sun_residual"]

        print(
            f"{str(result['name']):23s} "
            f"{np.mean(errors):9.3f}   "
            f"{np.percentile(errors, 95):9.3f}   "
            f"{np.mean(reliability):9.2f}   "
            f"{100.0 * np.mean(reliability < 40.0):8.1f}   "
            f"{np.mean(residual):12.4f}"
        )


def save_summary_plot(
    results: list[dict[str, np.ndarray | str]],
    output_path: Path,
) -> None:
    """Plot mean attitude error and fuzzy reliability by scenario."""
    names = [str(result["name"]) for result in results]
    mean_errors = [np.mean(result["attitude_error"]) for result in results]
    mean_reliability = [np.mean(result["reliability"]) for result in results]

    x = np.arange(len(names))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(9, 5))

    bars1 = ax1.bar(
        x - width / 2,
        mean_errors,
        width,
        color="steelblue",
        label="Mean attitude error [deg]",
    )
    ax1.set_ylabel("Mean attitude error [deg]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.grid(True, axis="y")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        mean_reliability,
        width,
        color="darkorange",
        label="Mean reliability [%]",
    )
    ax2.set_ylabel("Mean fuzzy reliability [%]")
    ax2.set_ylim(0, 100)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Fault impact on attitude error and fuzzy reliability")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_reliability_validation_plot(
    results: list[dict[str, np.ndarray | str]],
    output_path: Path,
) -> None:
    """Plot fuzzy reliability against the true simulation attitude error."""
    plt.figure(figsize=(8, 5))

    for result in results:
        plt.scatter(
            result["attitude_error"],
            result["reliability"],
            s=10,
            alpha=0.35,
            label=str(result["name"]),
        )

    plt.xlabel("True attitude error [deg]")
    plt.ylabel("Fuzzy reliability score [%]")
    plt.title("Validation of fuzzy reliability against attitude error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_residual_plot(
    results: list[dict[str, np.ndarray | str]],
    output_path: Path,
) -> None:
    """Compare Sun reconstruction residuals across fault scenarios."""
    names = [str(result["name"]) for result in results]
    mean_residuals = [np.mean(result["sun_residual"]) for result in results]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, mean_residuals, color="mediumpurple")
    plt.ylabel("Mean Sun reconstruction residual (RMS)")
    plt.title("Sun-sensor consistency under fault scenarios")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_all_sun_failures_plot(
    results: list[dict[str, np.ndarray | str]],
    output_path: Path,
) -> None:
    """Compare the effect of failing each individual Sun sensor."""
    names = [str(result["name"]) for result in results]
    mean_errors = [np.mean(result["attitude_error"]) for result in results]
    p95_errors = [
        np.percentile(result["attitude_error"], 95)
        for result in results
    ]
    mean_reliability = [np.mean(result["reliability"]) for result in results]

    x = np.arange(len(names))

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    ax1.plot(
        x,
        mean_errors,
        marker="o",
        label="Mean attitude error",
    )
    ax1.plot(
        x,
        p95_errors,
        marker="s",
        label="95th-percentile attitude error",
    )
    ax1.set_ylabel("Attitude error [deg]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
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
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.title("Effect of individual Sun-sensor failures")
    fig.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    n_samples = 1000
    base_seed = 42

    # Keep the same random attitude/noise sequence for these baseline scenarios
    # so the fault comparisons are fair.
    results = [
        run_scenario(
            "Healthy",
            n_samples,
            base_seed,
        ),
        run_scenario(
            "Sun sensor failed",
            n_samples,
            base_seed,
            failed_sun_sensor=0,
        ),
        run_scenario(
            "Mag disturbance",
            n_samples,
            base_seed,
            magnetic_disturbance=MAG_FAULT_VECTOR,
        ),
        run_scenario(
            "Combined fault",
            n_samples,
            base_seed,
            failed_sun_sensor=0,
            magnetic_disturbance=MAG_FAULT_VECTOR,
        ),
    ]

    print_summary(results)

    print("\nExample fuzzy interpretation:")
    for result in results:
        mean_score = float(np.mean(result["reliability"]))
        print(
            f"{result['name']}: {mean_score:.1f}% "
            f"({reliability_label(mean_score)})"
        )

    print("\nIndividual Sun-sensor failure comparison")
    print("----------------------------------------")
    sun_failure_results = []

    for sensor_index, sensor_name in enumerate(SUN_SENSOR_NAMES):
        result = run_scenario(
            f"{sensor_name} failed",
            n_samples,
            base_seed,
            failed_sun_sensor=sensor_index,
        )
        sun_failure_results.append(result)

    print_summary(sun_failure_results)

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "fault_analysis_summary.png"
    validation_path = output_dir / "fuzzy_reliability_validation.png"
    residual_path = output_dir / "sun_reconstruction_residual.png"
    sun_failure_path = output_dir / "all_sun_sensor_failures.png"

    save_summary_plot(results, summary_path)
    save_reliability_validation_plot(results, validation_path)
    save_residual_plot(results, residual_path)
    save_all_sun_failures_plot(sun_failure_results, sun_failure_path)

    print("\nSaved figures:")
    print(summary_path)
    print(validation_path)
    print(residual_path)
    print(sun_failure_path)


if __name__ == "__main__":
    main()
