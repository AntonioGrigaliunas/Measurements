"""Benchmark automatic Sun-sensor fault detection and reconfiguration.

Three estimators are compared after each injected single Sun-sensor failure:
1. Standard: unaware of the fault.
2. Automatic: detects a failed face from sensor geometry plus magnetic-vector
   consistency and reconfigures only when a fault is detected.
3. Oracle: reconfigures with the true failed-sensor identity supplied directly.

The oracle is not a flight algorithm. It is an upper benchmark showing how
much performance is available if fault identification is perfect.
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
from sun_sensor_fault_detection import (
    DEFAULT_MIN_MISSING_COMPONENT,
    detect_failed_sun_sensor,
)


SUN_SENSOR_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def make_models():
    sun_array = six_face_sensor_array(
        gains=SUN_GAINS,
        offsets=SUN_OFFSETS,
    )
    magnetometer = Magnetometer(
        scale_factors=MAG_SCALE_FACTORS,
        bias=MAG_BIAS,
    )
    return sun_array, magnetometer


def estimate_attitude_error(
    true_rotation: np.ndarray,
    sun_reference: np.ndarray,
    magnetic_reference: np.ndarray,
    estimated_sun_body: np.ndarray,
    estimated_magnetic_body: np.ndarray,
) -> float:
    estimated_rotation = triad_attitude(
        sun_reference,
        magnetic_reference,
        estimated_sun_body,
        estimated_magnetic_body,
    )
    return attitude_error_degrees(
        true_rotation,
        estimated_rotation,
    )


def healthy_false_alarm_rate(
    n_samples: int = 5000,
    seed: int = 123,
) -> float:
    """Return fraction of healthy snapshots incorrectly flagged as failed."""
    rng = np.random.default_rng(seed)
    sun_reference = unit_vector(np.array([1.0, 0.4, 0.2]))
    magnetic_reference = unit_vector(np.array([0.2, -0.3, 0.9]))
    sun_array, magnetometer = make_models()

    false_alarms = 0

    for _ in range(n_samples):
        true_rotation = random_rotation_matrix(rng)
        true_sun_body = true_rotation @ sun_reference
        true_magnetic_body = true_rotation @ magnetic_reference

        readings = sun_array.measure(
            true_sun_body,
            noise_std=SUN_NOISE_STD,
            rng=rng,
        )
        magnetic_measurement = magnetometer.measure(
            true_magnetic_body,
            noise_std=MAG_NOISE_STD,
            rng=rng,
        )

        detection = detect_failed_sun_sensor(
            readings,
            magnetic_measurement,
            sun_reference,
            magnetic_reference,
            SUN_GAINS,
            SUN_OFFSETS,
        )
        false_alarms += detection["failed_sensor"] is not None

    return false_alarms / n_samples


def run_failed_sensor_case(
    failed_sensor: int,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, float | str]:
    """Evaluate one failed face with standard, automatic, and oracle modes."""
    rng = np.random.default_rng(seed)

    sun_reference = unit_vector(np.array([1.0, 0.4, 0.2]))
    magnetic_reference = unit_vector(np.array([0.2, -0.3, 0.9]))
    sun_array, magnetometer = make_models()

    standard_errors = []
    automatic_errors = []
    oracle_errors = []

    detections = 0
    correct_detections = 0
    wrong_detections = 0
    observable_cases = 0
    correct_observable_detections = 0

    true_mask = np.ones(6, dtype=bool)
    true_mask[failed_sensor] = False

    for _ in range(n_samples):
        true_rotation = random_rotation_matrix(rng)
        true_sun_body = true_rotation @ sun_reference
        true_magnetic_body = true_rotation @ magnetic_reference

        ideal_response = np.maximum(
            0.0,
            sun_array.sensor_normals @ true_sun_body,
        )

        readings = sun_array.measure(
            true_sun_body,
            noise_std=SUN_NOISE_STD,
            rng=rng,
        )
        readings[failed_sensor] = 0.0

        magnetic_measurement = magnetometer.measure(
            true_magnetic_body,
            noise_std=MAG_NOISE_STD,
            rng=rng,
        )
        estimated_magnetic_body = unit_vector(magnetic_measurement)

        standard_sun = estimate_sun_vector(
            sun_array.sensor_normals,
            readings,
        )
        standard_errors.append(
            estimate_attitude_error(
                true_rotation,
                sun_reference,
                magnetic_reference,
                standard_sun,
                estimated_magnetic_body,
            )
        )

        oracle_sun = estimate_sun_vector_reconfigured(
            sun_array.sensor_normals,
            readings,
            true_mask,
            gains=SUN_GAINS,
            offsets=SUN_OFFSETS,
        )
        oracle_errors.append(
            estimate_attitude_error(
                true_rotation,
                sun_reference,
                magnetic_reference,
                oracle_sun,
                estimated_magnetic_body,
            )
        )

        detection = detect_failed_sun_sensor(
            readings,
            magnetic_measurement,
            sun_reference,
            magnetic_reference,
            SUN_GAINS,
            SUN_OFFSETS,
        )
        detected_sensor = detection["failed_sensor"]

        if detected_sensor is not None:
            detections += 1
            if detected_sensor == failed_sensor:
                correct_detections += 1
            else:
                wrong_detections += 1

        if ideal_response[failed_sensor] >= DEFAULT_MIN_MISSING_COMPONENT:
            observable_cases += 1
            if detected_sensor == failed_sensor:
                correct_observable_detections += 1

        if detected_sensor is None:
            automatic_sun = standard_sun
        else:
            automatic_mask = np.ones(6, dtype=bool)
            automatic_mask[int(detected_sensor)] = False
            automatic_sun = estimate_sun_vector_reconfigured(
                sun_array.sensor_normals,
                readings,
                automatic_mask,
                gains=SUN_GAINS,
                offsets=SUN_OFFSETS,
            )

        automatic_errors.append(
            estimate_attitude_error(
                true_rotation,
                sun_reference,
                magnetic_reference,
                automatic_sun,
                estimated_magnetic_body,
            )
        )

    standard_errors = np.asarray(standard_errors)
    automatic_errors = np.asarray(automatic_errors)
    oracle_errors = np.asarray(oracle_errors)

    return {
        "sensor": SUN_SENSOR_NAMES[failed_sensor],
        "standard_mean": float(np.mean(standard_errors)),
        "automatic_mean": float(np.mean(automatic_errors)),
        "oracle_mean": float(np.mean(oracle_errors)),
        "automatic_p95": float(np.percentile(automatic_errors, 95)),
        "overall_detection_rate": detections / n_samples,
        "overall_correct_rate": correct_detections / n_samples,
        "wrong_detection_rate": wrong_detections / n_samples,
        "observable_fraction": observable_cases / n_samples,
        "observable_correct_rate": (
            correct_observable_detections / observable_cases
            if observable_cases
            else 0.0
        ),
        "automatic_improvement_percent": float(
            100.0
            * (np.mean(standard_errors) - np.mean(automatic_errors))
            / np.mean(standard_errors)
        ),
    }


def save_plot(
    results: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    names = [str(result["sensor"]) for result in results]
    standard = [float(result["standard_mean"]) for result in results]
    automatic = [float(result["automatic_mean"]) for result in results]
    oracle = [float(result["oracle_mean"]) for result in results]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width, standard, width, label="Standard")
    plt.bar(x, automatic, width, label="Automatic reconfiguration")
    plt.bar(x + width, oracle, width, label="Known-failure benchmark")
    plt.xticks(x, names)
    plt.xlabel("Failed Sun-sensor face")
    plt.ylabel("Mean full-attitude error [deg]")
    plt.title("Automatic fault detection and Sun-sensor reconfiguration")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    false_alarm = healthy_false_alarm_rate()

    results = [
        run_failed_sensor_case(index)
        for index in range(len(SUN_SENSOR_NAMES))
    ]

    print("\nAutomatic Sun-sensor fault detection and reconfiguration")
    print("--------------------------------------------------------")
    print(f"Healthy false-alarm rate: {100.0 * false_alarm:.2f}%")

    print(
        "\nFailed   Standard   Automatic   Oracle    Auto p95   "
        "Observable detect   Wrong detect   Improvement"
    )
    print(
        "sensor    mean[deg]   mean[deg]   mean[deg]   [deg]         "
        "[%]             [%]          [%]"
    )

    for result in results:
        print(
            f"{str(result['sensor']):>4s}      "
            f"{float(result['standard_mean']):8.3f}   "
            f"{float(result['automatic_mean']):8.3f}   "
            f"{float(result['oracle_mean']):8.3f}   "
            f"{float(result['automatic_p95']):8.3f}       "
            f"{100.0 * float(result['observable_correct_rate']):8.1f}       "
            f"{100.0 * float(result['wrong_detection_rate']):8.2f}       "
            f"{float(result['automatic_improvement_percent']):8.1f}"
        )

    print(
        "\nNote: Observable detect is conditioned on the failed face having "
        f"ideal signal >= {DEFAULT_MIN_MISSING_COMPONENT:.2f}. A failed dark "
        "face cannot be identified from one static measurement because zero "
        "is also its expected healthy output."
    )

    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / "automatic_sun_sensor_reconfiguration.png"
    )
    save_plot(results, output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
